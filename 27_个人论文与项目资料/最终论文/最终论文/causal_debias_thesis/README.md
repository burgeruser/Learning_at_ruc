# 语言模型中的因果发现：从相关性到干预 基于因果干预的中文大语言模型偏见去偏系统

## 项目简介
本项目针对大语言模型中存在的职业-性别、地域等社会偏见问题，提出了一种基于因果推断与反事实数据增强的高效去偏方案。

## 核心方法
- **因果建模**：通过有向无环图（DAG）区分语义路径与偏见路径，明确需要干预的敏感属性。
- **Do算子干预**：利用对称KL散度构建去偏损失，切断敏感属性对模型输出的因果影响。
- **LoRA+NF4量化**：采用参数高效微调与4-bit量化技术，降低训练成本，适配低显存环境。
- **反事实数据增强**：自动化生成反事实文本数据，扩充无偏训练语料，提升模型鲁棒性。

## 实验结果
- 模型职业-性别偏见率从 62.5% 降低至 25%
- 语言能力（困惑度）波动小于 5%
- QA推理任务质量保持稳定

## 引用
如果该项目对你的研究有帮助，欢迎引用本论文与代码。

## 核心思想

> "树上有10只鸟，开一枪还剩几只？"
> 成年人凭经验秒答"0只"，而孩子会追问细节。
> 这个差异揭示了：人类和AI常常用"相关性"替代"因果推理"。

本论文通过因果推断技术，在大语言模型中识别并切断"偏见因果路径"。
数据准备：运行 `scripts/generate_counterfactual_data.py`
模型微调：`python train.py --config configs/lora_debias.yaml`
评估：`python evaluate.py --model_path ./checkpoints/debiased_model`

## 项目结构（可能并未完全列出）

```
causal_debias_thesis/
├── config.py                        # 全局配置
├── requirements.txt                 # 依赖
├── data/
│   ├── download_datasets.py         # 下载公开数据集
│   └── counterfactual_generator.py  # 反事实数据生成器
├── models/
│   └── load_model.py                # 模型加载 + LoRA
├── training/
│   ├── baseline_lora.py             # 基线 LoRA 微调
│   └── do_operator_finetune.py      # Do-operator 干预微调（核心）
├── evaluation/
│   └── bias_eval.py                 # 偏见评估
├── analysis/
│   └── visualize.py                 # 可视化
└── experiments/
    └── run_all.py                   # 一键运行全部实验
```

## 数据集

| 数据集 | HuggingFace路径 | 规模 | 用途 |
|--------|-----------------|------|------|
| AFQMC | clue/afqmc | 34,334条 | 语义相似度/偏见评估 |
| AG News | fancyzhx/ag_news | 120,000条 | 文本分类/标签固化 |
| OpenThoughts-114k | open-thoughts/OpenThoughts-114k | 114k条 | 推理数据/反事实构建 |
| Chinese-DeepSeek-R1 | Congliu/Chinese-DeepSeek-R1-Distill-data-110k | 110k条 | 中文推理/偏见检测 |
| 自建反事实集 | 本项目的CounterfactualGenerator | - | 因果干预训练 |

## 运行方式

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 下载数据
python data/download_datasets.py

# 3. 运行全部实验
python experiments/run_all.py
```