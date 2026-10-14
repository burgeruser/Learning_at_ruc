"""
反事实数据生成器（Counterfactual Generator）
对应论文 4.2 节：反事实数据集构建

核心功能：对原始文本中的敏感属性（性别、地域、职业）进行替换，
生成 "原始-反事实" 句子对，用于 Do-operator 干预训练。
"""
import json
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import cfg


class CounterfactualGenerator:
    """
    反事实数据生成器

    通过替换句子中的敏感属性词，生成语义不变但敏感属性改变的反事实样本。
    例如: "医生治好了病人" → "女医生治好了病人"
    """

    def __init__(self):
        # 从配置加载职业-性别映射
        self.occ_map = cfg.occupation_gender_map

        # 地域替换词表
        self.region_map = {
            "北京": "上海", "上海": "广州", "广州": "深圳", "深圳": "成都",
            "东部": "西部", "沿海": "内陆", "一线城市": "三四线城市",
        }

        # 性别代词
        self.male_pronouns = ["他", "他的", "先生", "父亲", "儿子", "兄弟"]
        self.female_pronouns = ["她", "她的", "小姐", "母亲", "女儿", "姐妹"]

    def swap_gender(self, text: str) -> str:
        """交换文本中的职业性别关联和性别代词"""
        result = text
        # 1. 替换职业词
        for word, (cf_word, _) in self.occ_map.items():
            if word in result:
                result = result.replace(word, cf_word, 1)
                break
        # 2. 替换性别代词（建立对应映射）
        pronoun_pairs = list(zip(self.male_pronouns, self.female_pronouns))
        for mp, fp in pronoun_pairs:
            if mp in result:
                result = result.replace(mp, fp, 1)
                break
        for mp, fp in pronoun_pairs:
            if fp in result:
                result = result.replace(fp, mp, 1)
                break
        return result

    def swap_region(self, text: str) -> str:
        """替换文本中的地域信息"""
        for orig, alt in self.region_map.items():
            if orig in text:
                return text.replace(orig, alt, 1)
        return text

    def generate(self, text: str, attr_type: str = "gender") -> Tuple[str, str]:
        """
        生成反事实文本

        参数:
            text: 原始文本
            attr_type: 敏感属性类型 ("gender", "region", "all")

        返回:
            (原始文本, 反事实文本)
        """
        cf = text
        if attr_type in ("gender", "all"):
            cf = self.swap_gender(cf)
        if attr_type in ("region", "all"):
            cf = self.swap_region(cf)
        return text, cf

    def batch_generate(self, texts: List[str], attr_type: str = "gender") -> List[Dict]:
        """批量生成反事实对"""
        results = []
        for t in texts:
            orig, cf = self.generate(t, attr_type)
            if orig != cf:
                results.append({
                    "original": orig,
                    "counterfactual": cf,
                    "attr_type": attr_type,
                })
        return results

    def process_afqmc(self, dataset, attr_types: tuple = ("gender",)) -> List[Dict]:
        """
        处理 AFQMC 数据集，为每条数据生成反事实版本

        参数:
            dataset: AFQMC 数据集
            attr_types: 敏感属性类型元组，默认 ("gender",)
                        可包含 "gender"、"region"、"all"

        输出格式: [{"original": str, "counterfactual": str, "label": int,
                     "changed": bool, "attr_type": str}, ...]
        """
        results = []
        for item in dataset:
            s1, s2, label = item["sentence1"], item["sentence2"], item["label"]
            # 原始样本
            for attr_type in attr_types:
                _, cf = self.generate(s1, attr_type)
                results.append({
                    "original": s1,
                    "counterfactual": cf,
                    "label": label,
                    "changed": cf != s1,
                    "attr_type": attr_type,
                })
        return results

    def save_to_json(self, data: List[Dict], path: str):
        """保存反事实数据集为 JSON"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        changed = sum(1 for d in data if d.get("changed"))
        print(f"  保存 {len(data)} 条, 其中反事实 {changed} 条 -> {path}")


def build_counterfactual_dataset(num_samples: int = None):
    """
    构建完整的反事实数据集（论文 4.2 节的核心数据）

    流程:
    1. 从 AFQMC 训练集加载原始句子
    2. 对每个句子生成反事实版本（性别替换）
    3. 保存为 JSON 格式

    参数:
        num_samples: 处理的样本数，None 表示使用全部数据（约 34k 条）
    """
    from datasets import load_dataset

    print("=" * 50)
    print("构建反事实数据集")
    print("=" * 50)

    # 1. 加载 AFQMC 数据
    print("[1] 加载 AFQMC 数据集 ...")
    dataset = load_dataset(cfg.afqmc_dataset, split="train")
    print(f"  原始数据: {len(dataset)} 条")

    # 2. 生成反事实版本
    print("[2] 生成反事实样本 ...")
    generator = CounterfactualGenerator()
    if num_samples is not None:
        dataset = dataset.select(range(num_samples))
    cf_data = generator.process_afqmc(dataset)

    # 3. 保存
    save_path = os.path.join(cfg.data_root, "counterfactual", "train.json")
    generator.save_to_json(cf_data, save_path)
    return cf_data


if __name__ == "__main__":
    # 测试反事实生成器
    gen = CounterfactualGenerator()

    test_sentences = [
        "医生完成了手术",
        "护士在照顾病人",
        "来自北京的程序员正在写代码",
        "他在上海工作",
    ]

    print("反事实生成测试:")
    print("-" * 40)
    for s in test_sentences:
        orig, cf = gen.generate(s, "all")
        changed = "✓" if orig != cf else "✗"
        print(f"  原始:   {orig}")
        print(f"  反事实: {cf}  [{changed}]")
        print()

    # 构建完整数据集
    build_counterfactual_dataset()
