"""
全局配置：所有路径、模型名、数据集统一配置
OpenI平台使用时修改 data_root 路径即可
"""

import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # ======== 模型配置 ========
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    # 备选轻量模型（显存不足时使用）
    # model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"

    # ======== 数据路径 ========
    data_root: str = "./data"

    # HuggingFace 数据集名称（均为真实可用的公开数据集）
    afqmc_dataset: str = "clue/afqmc"
    ag_news_dataset: str = "fancyzhx/ag_news"
    open_thoughts: str = "open-thoughts/OpenThoughts-114k"
    deepseek_chinese: str = "Congliu/Chinese-DeepSeek-R1-Distill-data-110k"

    # ======== 输出路径 ========
    output_dir: str = "./output"
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"

    # ======== 反事实生成配置 ========
    sensitive_attributes: tuple = ("gender", "region", "occupation")

    # 性别-职业偏见词表
    occupation_gender_map: dict = None  # 在 __post_init__ 中初始化

    # ======== LoRA 训练配置 ========
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    learning_rate: float = 2e-4
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 512
    logging_steps: int = 10
    save_steps: int = 200
    seed: int = 42

    # ======== Do-operator 去偏配置 ========
    debias_lambda: float = 0.1

    # ======== 消融实验配置 ========
    lambda_values: List[float] = field(default_factory=lambda: [0.0, 0.05, 0.1, 0.2, 0.5])
    data_sizes: List[int] = field(default_factory=lambda: [100, 250, 500, 1000, 5000])

    # ======== 评估配置 ========
    eval_batch_size: int = 16
    bias_threshold: float = 0.1

    def __post_init__(self):
        """初始化派生配置"""
        if self.occupation_gender_map is None:
            self.occupation_gender_map = {
                # 通常偏男性的职业 → 反事实加"女"
                "医生": ("女医生", "male"),
                "工程师": ("女工程师", "male"),
                "程序员": ("女程序员", "male"),
                "CEO": ("女CEO", "male"),
                "经理": ("女经理", "male"),
                "教授": ("女教授", "male"),
                "司机": ("女司机", "male"),
                "建筑工人": ("女建筑工人", "male"),
                "消防员": ("女消防员", "male"),
                # 通常偏女性的职业 → 反事实加"男"
                "护士": ("男护士", "female"),
                "秘书": ("男秘书", "female"),
                "前台": ("男前台", "female"),
                "幼师": ("男幼师", "female"),
                "保姆": ("男保姆", "female"),
                "化妆师": ("男化妆师", "female"),
            }
        # 自动创建目录
        for d in [self.output_dir, self.checkpoint_dir, self.log_dir, self.data_root]:
            os.makedirs(d, exist_ok=True)


# 全局配置实例
cfg = Config()
