"""
一键运行全部实验（扩展版）

完整实验流水线：
  Step 1: 数据准备
  Step 2: 基线 LoRA 微调
  Step 3: Do-operator 干预微调（核心）
  Step 4: λ 消融实验（不同去偏权重）
  Step 5: 数据量消融实验
  Step 6: 全面评估
  Step 7: 可视化 + 结果汇总

用法:
  # 完整运行
  python experiments/run_all.py

  # 仅运行特定阶段（例如仅评估，需先有 checkpoint）
  python experiments/run_all.py --skip-train

  # 补充说明：仅运行消融实验
  python experiments/run_all.py --only-ablation
"""
import argparse
import json
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import DataCollatorForSeq2Seq

from config import cfg
from data.counterfactual_generator import build_counterfactual_dataset
from data.download_datasets import (
    download_afqmc, download_ag_news,
    create_qa_bias_testset, create_occupation_bias_testset,
)
from evaluation.bias_eval import BiasEvaluator
from models.load_model import load_model_and_tokenizer, apply_lora, get_training_args
from training.baseline_lora import train_baseline
from training.do_operator_finetune import DoOperatorTrainer, prepare_do_dataset
from analysis.visualize import generate_all_plots


def step(seq, name, desc):
    """打印步骤标题"""
    print(f"\n{'='*60}")
    print(f"  Step {seq}: {name}")
    print(f"  {desc}")
    print(f"{'='*60}")


def train_model(train_fn, name, **kwargs):
    """通用训练函数，带计时"""
    t0 = time.time()
    path = train_fn(**kwargs)
    elapsed = (time.time() - t0) / 60
    print(f"  {name} 训练完成，耗时 {elapsed:.1f} 分钟")
    return path, elapsed


def main(args):
    # 仅运行消融实验模式：直接跳过 Steps 1-3
    only_ablation = getattr(args, "only_ablation", False)

    print("=" * 60)
    print("  论文实验流水线（完整版）")
    print("  语言模型中的因果发现：从相关性到干预")
    print(f"  设备: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    print(f"  模型: {cfg.model_name}")
    if only_ablation:
        print("  模式: 仅消融实验")
    print("=" * 60)

    results = {
        "config": {
            "model": cfg.model_name,
            "lora_r": cfg.lora_r,
            "lora_alpha": cfg.lora_alpha,
            "learning_rate": cfg.learning_rate,
            "num_epochs": cfg.num_epochs,
            "default_lambda": cfg.debias_lambda,
            "seed": cfg.seed,
        },
        "ablation": {},
    }

    # ---- Step 1: 数据准备（仅在非only-ablation模式下执行） ----
    if not only_ablation:
        step(1, "数据准备", "下载公开数据集 + 生成反事实数据 + 构建测试集")
    afqmc = download_afqmc()
    ag_news = download_ag_news()
    qa_data = create_qa_bias_testset()
    occ_data = create_occupation_bias_testset()
    cf_data = build_counterfactual_dataset(num_samples=500)

    results["data"] = {
        "afqmc_train": len(afqmc["train"]),
        "afqmc_val": len(afqmc["validation"]),
        "ag_news_train": len(ag_news["train"]),
        "qa_scenarios": len(qa_data),
        "occupation_pairs": len(occ_data),
        "counterfactual_samples": len(cf_data),
    }

    qa_path = os.path.join(cfg.data_root, "qa_bias_testset", "questions.json")

    # ---- Step 2: 基线训练 ----
    if not args.skip_train:
        step(2, "基线 LoRA 微调", "标准语言建模训练，不去偏 (λ=0)")
        bl_path, bl_time = train_model(train_baseline, "基线")
        results["baseline_training"] = {"path": bl_path, "time_min": bl_time}
    else:
        bl_path = os.path.join(cfg.checkpoint_dir, "baseline", "final")
        results["baseline_training"] = {"path": bl_path, "note": "使用已有检查点"}

    # ---- Step 3: Do-operator 训练（默认 λ） ----
    if not args.skip_train:
        step(3, "Do-operator 干预微调", f"核心方法：λ={cfg.debias_lambda}")
        do_path, do_time = train_model(train_do_operator, "Do-operator")
        results["do_training"] = {"path": do_path, "time_min": do_time, "lambda": cfg.debias_lambda}
    else:
        do_path = os.path.join(cfg.checkpoint_dir, "do_operator", "final")
        results["do_training"] = {"path": do_path, "note": "使用已有检查点"}

    # ---- Step 4: λ 消融实验 ----
    step(4, "λ 消融实验", f"探索 λ ∈ {cfg.lambda_values} 对去偏效果的影响")
    lambda_results = []
    for lmb in cfg.lambda_values:
        if lmb == 0.0:
            continue  # 基线已有
        if args.skip_train and lmb != 0.1:
            continue  # 跳过时仅评估默认 λ

        print(f"\n--- λ={lmb} 训练 ---")
        model, tokenizer = load_model_and_tokenizer(use_quantization=True)
        model = apply_lora(model)

        train_data, eval_data = prepare_do_dataset(tokenizer, num_samples=500)
        args_lmb = get_training_args(
            output_dir=os.path.join(cfg.checkpoint_dir, f"lambda_{lmb}")
        )
        collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, return_tensors="pt")

        trainer = DoOperatorTrainer(
            model=model, args=args_lmb,
            train_dataset=train_data, eval_dataset=eval_data,
            data_collator=collator, debias_lambda=lmb,
        )
        trainer.train()

        # 评估
        evaluator = BiasEvaluator(model, tokenizer)
        eval_r = evaluator.full_evaluation(qa_path=qa_path)
        ppl = evaluator.evaluate_perplexity(
            [d["sentence1"] for d in afqmc["validation"].select(range(50))]
        )

        lambda_results.append({
            "lambda": lmb,
            "bias_rate": eval_r.get("occupation_bias", {}).get("bias_rate", 0),
            "mean_bias": eval_r.get("occupation_bias", {}).get("mean_bias", 0),
            "qa_quality": eval_r.get("qa_bias", {}).get("mean_quality", 0),
            "perplexity": ppl["mean_ppl"],
        })
        print(f"  λ={lmb}: 偏见率={lambda_results[-1]['bias_rate']:.2%}, "
              f"困惑度={lambda_results[-1]['perplexity']:.2f}")

    results["ablation"]["lambda"] = lambda_results

    # ---- Step 5: 数据量消融实验 ----
    step(5, "数据量消融实验", f"探索训练数据量 {cfg.data_sizes} 的影响")
    size_results = []
    for size in cfg.data_sizes:
        if args.skip_train and size != 500:
            continue
        if size == 500:
            continue  # 默认已训练

        print(f"\n--- 数据量={size} 训练 ---")
        model, tokenizer = load_model_and_tokenizer(use_quantization=True)
        model = apply_lora(model)

        train_data, eval_data = prepare_do_dataset(tokenizer, num_samples=size)
        args_sz = get_training_args(
            output_dir=os.path.join(cfg.checkpoint_dir, f"size_{size}")
        )
        collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, return_tensors="pt")

        trainer = DoOperatorTrainer(
            model=model, args=args_sz,
            train_dataset=train_data, eval_dataset=eval_data,
            data_collator=collator, debias_lambda=cfg.debias_lambda,
        )
        trainer.train()

        evaluator = BiasEvaluator(model, tokenizer)
        eval_r = evaluator.full_evaluation(qa_path=qa_path)

        size_results.append({
            "size": size,
            "bias_rate": eval_r.get("occupation_bias", {}).get("bias_rate", 0),
            "mean_bias": eval_r.get("occupation_bias", {}).get("mean_bias", 0),
            "qa_quality": eval_r.get("qa_bias", {}).get("mean_quality", 0),
        })
        print(f"  数据量={size}: 偏见率={size_results[-1]['bias_rate']:.2%}")

    results["ablation"]["data_size"] = size_results

    # ---- Step 6: 全面评估（默认模型） ----
    step(6, "全面评估", "对基线模型和 Do-operator 模型进行完整偏见评估")
    for name, ckpt in [("baseline", bl_path), ("do_operator", do_path)]:
        model, tokenizer = load_model_and_tokenizer(use_quantization=True)
        if os.path.exists(ckpt):
            model = PeftModel.from_pretrained(model, ckpt)
            print(f"  已加载检查点: {ckpt}")
        evaluator = BiasEvaluator(model, tokenizer)
        eval_r = evaluator.full_evaluation(qa_path=qa_path)
        ppl = evaluator.evaluate_perplexity(
            [d["sentence1"] for d in afqmc["validation"].select(range(50))]
        )
        eval_r["perplexity"] = ppl
        results[name] = eval_r

    # ---- Step 7: 可视化 + 汇总 ----
    step(7, "可视化 + 结果汇总", "生成论文全部图表")
    fig_paths = generate_all_plots(
        results,
        lambda_results=lambda_results,
        size_results=size_results,
    )
    results["figures"] = fig_paths

    # 保存结果
    summary_path = os.path.join(cfg.output_dir, "experiment_results.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 打印摘要
    print(f"\n{'='*60}")
    print("  实验完成！结果摘要")
    print(f"{'='*60}")
    for name in ["baseline", "do_operator"]:
        if name in results:
            occ = results[name].get("occupation_bias", {})
            qa = results[name].get("qa_bias", {})
            ppl = results[name].get("perplexity", {})
            print(f"  {name}:")
            print(f"    职业偏见: {occ.get('mean_bias', 'N/A'):.4f}")
            print(f"    偏见率:   {occ.get('bias_rate', 'N/A'):.2%}")
            print(f"    QA质量:   {qa.get('mean_quality', 'N/A'):.4f}")
            print(f"    困惑度:   {ppl.get('mean_ppl', 'N/A'):.2f}")

    # 提取核心结论
    bl_rate = results.get("baseline", {}).get("occupation_bias", {}).get("bias_rate", 0)
    do_rate = results.get("do_operator", {}).get("occupation_bias", {}).get("bias_rate", 0)
    if bl_rate > 0:
        impr = (bl_rate - do_rate) / bl_rate * 100
        print(f"\n  ★ 核心结论：偏见率从 {bl_rate:.1%} 降至 {do_rate:.1%}（降幅 {impr:.0f}%）")

    print(f"\n详细结果: {summary_path}")
    print(f"图表目录: {os.path.join(cfg.output_dir, 'figures')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="论文实验流水线")
    parser.add_argument("--skip-train", action="store_true", help="跳过训练，仅评估已有检查点")
    parser.add_argument("--only-ablation", action="store_true", help="仅运行消融实验")
    args = parser.parse_args()
    main(args)
