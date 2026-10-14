"""
模型加载模块
加载 Qwen2.5 模型并配置 LoRA，适配 OpenI 平台 GPU 环境
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from config import cfg


def load_model_and_tokenizer(use_quantization: bool = True):
    """
    加载 Qwen2.5 模型和分词器

    参数:
        use_quantization: 是否使用 4-bit 量化（节约显存）

    返回:
        model, tokenizer
    """
    print(f"加载模型: {cfg.model_name}")

    # 量化配置（4-bit，将显存需求降至约 2GB）
    qconfig = None
    if use_quantization and torch.cuda.is_available():
        qconfig = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        print("  使用 4-bit 量化")

    # 加载
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        quantization_config=qconfig,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name, trust_remote_code=True, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  设备: {device}")
    print(f"  参数: {model.num_parameters() / 1e9:.2f}B")

    return model, tokenizer


def apply_lora(model):
    """
    对模型应用 LoRA (Low-Rank Adaptation)

    冻结全部基础权重，只训练 4 个低秩矩阵。
    可训练参数量约为总参数的 ~0.1%。
    """
    print("应用 LoRA ...")

    if hasattr(model, "is_loaded_in_4bit") and model.is_loaded_in_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  可训练参数: {trainable:,} ({trainable / total:.4%})")
    print(f"  总参数: {total:,}")

    return model


def get_training_args(output_dir: str = None) -> TrainingArguments:
    """获取标准训练参数"""
    if output_dir is None:
        output_dir = cfg.checkpoint_dir

    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        warmup_steps=50,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        evaluation_strategy="steps",
        eval_steps=cfg.save_steps,
        learning_rate=cfg.learning_rate,
        fp16=torch.cuda.is_available(),
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )


if __name__ == "__main__":
    model, tokenizer = load_model_and_tokenizer(use_quantization=True)
    model = apply_lora(model)

    # 快速测试
    test = "树上有10只鸟，猎人开枪打了一下，"
    inputs = tokenizer(test, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=50, do_sample=True, temperature=0.7)

    print(f"\n测试推理:")
    print(f"  输入: {test}")
    print(f"  输出: {tokenizer.decode(out[0], skip_special_tokens=True)}")
