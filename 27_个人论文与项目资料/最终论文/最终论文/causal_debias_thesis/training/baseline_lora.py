"""
基线实验：标准 LoRA 微调（无去偏处理）
对应论文 5.3 节基线方法

使用 AFQMC 数据集的标准语言建模训练作为基线。
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from datasets import load_dataset
from transformers import Trainer, DataCollatorForSeq2Seq

from config import cfg
from models.load_model import load_model_and_tokenizer, apply_lora, get_training_args


def prepare_dataset(tokenizer):
    """准备 AFQMC 语义相似度训练数据"""
    dataset = load_dataset(cfg.afqmc_dataset, split="train")

    def format_fn(examples):
        texts = []
        for s1, s2, label in zip(
            examples["sentence1"], examples["sentence2"], examples["label"]
        ):
            ans = "相似" if label == 1 else "不相似"
            texts.append(f"判断语义是否相似：\n句1: {s1}\n句2: {s2}\n答案: {ans}")
        return {"text": texts}

    dataset = dataset.map(format_fn, batched=True)

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=cfg.max_seq_length,
        )

    dataset = dataset.map(tokenize_fn, batched=True)
    split = dataset.train_test_split(test_size=0.1)
    return split["train"], split["test"]


def train_baseline():
    """训练基线 LoRA 模型"""
    print("=" * 50)
    print("基线实验：标准 LoRA 微调")
    print("=" * 50)

    model, tokenizer = load_model_and_tokenizer(use_quantization=True)
    model = apply_lora(model)

    train_data, eval_data = prepare_dataset(tokenizer)
    print(f"训练集: {len(train_data)} 条, 验证集: {len(eval_data)} 条")

    args = get_training_args(output_dir=os.path.join(cfg.checkpoint_dir, "baseline"))
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, return_tensors="pt")

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=collator,
    )

    trainer.train()

    save_path = os.path.join(cfg.checkpoint_dir, "baseline", "final")
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"基线模型保存至: {save_path}")
    return save_path


if __name__ == "__main__":
    train_baseline()
