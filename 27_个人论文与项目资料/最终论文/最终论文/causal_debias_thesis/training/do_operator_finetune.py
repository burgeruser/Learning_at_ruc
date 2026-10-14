"""
核心方法：Do-operator 干预微调
对应论文 4.4 节

核心思路：
  总损失 = 语言建模损失 + λ × 去偏损失
  其中去偏损失 = KL(P(Y|X_orig) || P(Y|X_cf))

通过最小化原始样本和反事实样本的输出分布差异，
切断敏感属性 S 到模型输出 Y 的偏见因果路径。
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from datasets import load_dataset, Dataset
from transformers import Trainer, DataCollatorForSeq2Seq

from config import cfg
from models.load_model import load_model_and_tokenizer, apply_lora, get_training_args
from data.counterfactual_generator import CounterfactualGenerator


class DoOperatorTrainer(Trainer):
    """
    支持去偏损失的自定义 Trainer

    L_total = L_task + λ * KL(P(Y|X_orig) || P(Y|X_cf))
    """

    def __init__(self, *args, debias_lambda=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.debias_lambda = debias_lambda or cfg.debias_lambda

    def compute_loss(self, model, inputs, return_outputs=False):
        # 1. 标准语言建模损失
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
        )
        task_loss = outputs.loss

        # 2. 去偏损失（如果输入包含反事实版本）
        cf_ids = inputs.get("cf_input_ids")
        cf_mask = inputs.get("cf_attention_mask")
        debias_loss = torch.tensor(0.0, device=task_loss.device)

        if cf_ids is not None:
            with torch.no_grad():
                cf_out = model(input_ids=cf_ids, attention_mask=cf_mask)
            p_orig = F.softmax(outputs.logits, dim=-1)
            p_cf = F.softmax(cf_out.logits, dim=-1)

            kl_fwd = F.kl_div(p_orig.log(), p_cf, reduction="batchmean", log_target=False)
            kl_bwd = F.kl_div(p_cf.log(), p_orig, reduction="batchmean", log_target=False)
            debias_loss = (kl_fwd + kl_bwd) / 2.0

        total_loss = task_loss + self.debias_lambda * debias_loss
        return (total_loss, outputs) if return_outputs else total_loss


def prepare_do_dataset(tokenizer, num_samples=500):
    """
    准备 Do-operator 训练数据

    每条训练样本包含:
    - input_ids: 原始文本
    - cf_input_ids: 反事实文本（仅替换敏感属性）
    - labels: 语言建模标签
    """
    dataset = load_dataset(cfg.afqmc_dataset, split="train")
    dataset = dataset.select(range(min(num_samples, len(dataset))))
    generator = CounterfactualGenerator()

    records = []
    for item in dataset:
        s1 = item["sentence1"]
        s2 = item["sentence2"]
        label = item["label"]
        ans = "相似" if label == 1 else "不相似"
        text = f"判断语义是否相似：\n句1: {s1}\n句2: {s2}\n答案: {ans}"

        _, cf_s1 = generator.generate(s1, "gender")
        cf_text = f"判断语义是否相似：\n句1: {cf_s1}\n句2: {s2}\n答案: {ans}"

        records.append({"text": text, "cf_text": cf_text})

    hf_dataset = Dataset.from_list(records)

    def tokenize_fn(examples):
        tok = tokenizer(
            examples["text"], truncation=True, padding="max_length",
            max_length=cfg.max_seq_length,
        )
        cf_tok = tokenizer(
            examples["cf_text"], truncation=True, padding="max_length",
            max_length=cfg.max_seq_length,
        )
        tok["cf_input_ids"] = cf_tok["input_ids"]
        tok["cf_attention_mask"] = cf_tok["attention_mask"]
        tok["labels"] = tok["input_ids"].copy()
        return tok

    hf_dataset = hf_dataset.map(tokenize_fn, batched=True)
    split = hf_dataset.train_test_split(test_size=0.1)
    return split["train"], split["test"]


def train_do_operator():
    """训练 Do-operator 干预微调模型"""
    print("=" * 50)
    print("核心实验：Do-operator 干预微调")
    print(f"去偏权重 λ = {cfg.debias_lambda}")
    print("=" * 50)

    model, tokenizer = load_model_and_tokenizer(use_quantization=True)
    model = apply_lora(model)

    train_data, eval_data = prepare_do_dataset(tokenizer, num_samples=500)
    print(f"训练集: {len(train_data)} 条, 验证集: {len(eval_data)} 条")

    args = get_training_args(output_dir=os.path.join(cfg.checkpoint_dir, "do_operator"))
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, return_tensors="pt")

    trainer = DoOperatorTrainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=collator,
        debias_lambda=cfg.debias_lambda,
    )

    trainer.train()

    save_path = os.path.join(cfg.checkpoint_dir, "do_operator", "final")
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Do-operator 模型保存至: {save_path}")
    return save_path


if __name__ == "__main__":
    train_do_operator()
