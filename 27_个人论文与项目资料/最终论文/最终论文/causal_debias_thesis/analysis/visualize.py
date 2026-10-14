"""
可视化分析模块（扩展版）
生成论文所需的全部图表：

1. 偏见强度对比柱状图（基线 vs Do-operator）
2. 帕累托前沿图（偏见率 vs 困惑度）
3. λ 敏感性曲线（双Y轴）
4. 各类别偏见强度详情柱状图
5. 数据量消融曲线
"""
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import cfg


# ======== 全局样式 ========
plt.rcParams["font.size"] = 11
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["figure.dpi"] = 150


def plot_bias_comparison(baseline_results: dict, do_results: dict, save_dir: str):
    """
    图1: 偏见强度对比柱状图（论文 6.1 节）
    X轴: 偏见维度, Y轴: 偏见强度
    分组柱状图: 基线 vs Do-operator
    """
    dims = ["职业-性别", "地域偏见", "标签固化", "QA推理"]

    bl_scores = [
        baseline_results.get("occupation_bias", {}).get("mean_bias", 0),
        baseline_results.get("region_bias", {}).get("mean_bias", 0),
        baseline_results.get("label_bias", {}).get("mean_bias", 0),
        baseline_results.get("qa_bias", {}).get("mean_quality", 0),
    ]
    do_scores = [
        do_results.get("occupation_bias", {}).get("mean_bias", 0),
        do_results.get("region_bias", {}).get("mean_bias", 0),
        do_results.get("label_bias", {}).get("mean_bias", 0),
        do_results.get("qa_bias", {}).get("mean_quality", 0),
    ]

    x = np.arange(len(dims))
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - w / 2, bl_scores, w, label="基线模型", color="#4A7FB5", alpha=0.85)
    bars2 = ax.bar(x + w / 2, do_scores, w, label="Do-operator", color="#D4753E", alpha=0.85)

    # 在柱子上标注数值
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("偏见维度")
    ax.set_ylabel("偏见强度")
    ax.set_title("基线 vs Do-operator 偏见强度对比")
    ax.set_xticks(x)
    ax.set_xticklabels(dims)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    path = os.path.join(save_dir, "bias_comparison.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"图表已保存: {path}")
    return path


def plot_pareto_frontier(results_list: list, save_dir: str):
    """
    图2: 帕累托前沿图（论文 6.2 节）
    X轴: 困惑度, Y轴: 偏见率
    理想位置: 左下角
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = {"baseline": "#4A7FB5", "do_operator": "#D4753E"}
    markers = {"baseline": "o", "do_operator": "s"}

    for r in results_list:
        name = r["name"]
        ppl = r.get("perplexity", 0)
        bias = r.get("bias_rate", 0)
        ax.scatter(ppl, bias, c=colors.get(name, "#888"),
                   marker=markers.get(name, "o"), s=150, label=name, zorder=5, edgecolors="#333")
        ax.annotate(name, (ppl, bias), textcoords="offset points",
                    xytext=(12, 5), fontsize=10, weight="bold")

    ax.set_xlabel("困惑度 ↓（语言能力越好）")
    ax.set_ylabel("偏见率 ↓（公平性越好）")
    ax.set_title("帕累托前沿：公平性 vs 语言能力")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    # 理想方向区域指示
    ax.annotate("← 理想方向", xy=(0.02, 0.02), xytext=(0.15, 0.12),
                arrowprops=dict(arrowstyle="->", color="green", lw=2.5),
                fontsize=11, color="green", weight="bold")

    path = os.path.join(save_dir, "pareto_frontier.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"图表已保存: {path}")
    return path


def plot_lambda_sensitivity(lambda_results: list, save_dir: str):
    """
    图3: λ 敏感性曲线（论文 6.3.1 节）
    双Y轴: 左侧偏见率, 右侧困惑度
    X轴: λ 值
    """
    lambdas = [r["lambda"] for r in lambda_results]
    bias_rates = [r["bias_rate"] for r in lambda_results]
    ppls = [r["perplexity"] for r in lambda_results]

    fig, ax1 = plt.subplots(figsize=(9, 5.5))

    color1 = "#D4753E"
    color2 = "#4A7FB5"

    ax1.plot(lambdas, bias_rates, "o-", color=color1, linewidth=2.5, markersize=8, label="偏见率")
    ax1.set_xlabel("去偏权重 λ")
    ax1.set_ylabel("偏见率 (Bias Rate)", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.axhline(y=0.625, color="gray", linestyle="--", alpha=0.5, label="基线偏见率 (62.5%)")
    ax1.grid(alpha=0.3)

    # 标注三个区域
    ax1.axvspan(-0.02, 0.05, alpha=0.08, color="orange", label="欠去偏区")
    ax1.axvspan(0.05, 0.2, alpha=0.08, color="green", label="有效去偏区 ✅")
    ax1.axvspan(0.2, 0.55, alpha=0.08, color="red", label="过约束区")

    ax2 = ax1.twinx()
    ax2.plot(lambdas, ppls, "s--", color=color2, linewidth=2.5, markersize=8, label="困惑度")
    ax2.set_ylabel("平均困惑度 (PPL)", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

    ax1.set_title("λ 敏感性分析：去偏权重对偏见率和语言能力的影响")
    ax1.set_xlim(-0.02, 0.52)

    # 标注推荐 λ
    ax1.annotate("推荐 λ=0.1", xy=(0.1, 0.25), xytext=(0.15, 0.18),
                arrowprops=dict(arrowstyle="->", color="green", lw=2),
                fontsize=11, color="green", weight="bold")

    path = os.path.join(save_dir, "lambda_sensitivity.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"图表已保存: {path}")
    return path


def plot_category_detail(all_diffs_baseline: list, all_diffs_do: list,
                         categories: list, save_dir: str):
    """
    图4: 各类别偏见强度详情（论文 6.1.1 节）
    分组柱状图展示每个职业性别对在基线和Do-operator下的BiasScore
    """
    x = np.arange(len(categories))
    w = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - w / 2, all_diffs_baseline, w, label="基线模型", color="#4A7FB5", alpha=0.85)
    ax.bar(x + w / 2, all_diffs_do, w, label="Do-operator", color="#D4753E", alpha=0.85)

    # 标注改善百分比
    for i in range(len(categories)):
        if all_diffs_baseline[i] > 0:
            impr = (all_diffs_baseline[i] - all_diffs_do[i]) / all_diffs_baseline[i] * 100
            ax.text(i, max(all_diffs_baseline[i], all_diffs_do[i]) + 0.02,
                    f"↓{impr:.0f}%", ha="center", va="bottom", fontsize=7, color="green")

    ax.set_xlabel("职业-性别偏见类别")
    ax.set_ylabel("BiasScore (困惑度差异)")
    ax.set_title("各类别偏见强度详情：基线 vs Do-operator")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=25, ha="right", fontsize=8)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    path = os.path.join(save_dir, "category_detail.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"图表已保存: {path}")
    return path


def plot_data_size_ablation(size_results: list, save_dir: str):
    """
    图5: 数据量消融曲线（论文 6.3.2 节）
    X轴: 训练样本数, Y轴: 偏见率
    """
    sizes = [r["size"] for r in size_results]
    rates = [r["bias_rate"] for r in size_results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sizes, rates, "o-", color="#D4753E", linewidth=2.5, markersize=8)
    ax.axhline(y=0.625, color="gray", linestyle="--", alpha=0.6, label="基线偏见率 (62.5%)")

    # 标注改善
    for i, (s, r) in enumerate(zip(sizes, rates)):
        impr = (0.625 - r) / 0.625 * 100
        ax.annotate(f"↓{impr:.0f}%", (s, r), textcoords="offset points",
                    xytext=(10, -15), fontsize=9, color="green")

    ax.set_xlabel("训练样本数")
    ax.set_ylabel("Do-operator 偏见率")
    ax.set_title("数据量消融：训练样本数对去偏效果的影响")
    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.legend()
    ax.grid(alpha=0.3)

    path = os.path.join(save_dir, "data_size_ablation.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"图表已保存: {path}")
    return path


def generate_all_plots(eval_results: dict, save_dir: str = None,
                       lambda_results: list = None,
                       size_results: list = None):
    """生成全部图表"""
    if save_dir is None:
        save_dir = os.path.join(cfg.output_dir, "figures")
    os.makedirs(save_dir, exist_ok=True)

    paths = {}

    # 图1: 偏见对比
    if "baseline" in eval_results and "do_operator" in eval_results:
        paths["bias_comparison"] = plot_bias_comparison(
            eval_results["baseline"], eval_results["do_operator"], save_dir
        )

    # 图2: 帕累托前沿
    pareto_data = []
    for name in ["baseline", "do_operator"]:
        if name in eval_results:
            r = eval_results[name]
            pareto_data.append({
                "name": name,
                "perplexity": r.get("perplexity", {}).get("mean_ppl", 0),
                "bias_rate": r.get("occupation_bias", {}).get("bias_rate", 0),
            })
    if len(pareto_data) >= 2:
        paths["pareto_frontier"] = plot_pareto_frontier(pareto_data, save_dir)

    # 图3: λ敏感性
    if lambda_results and len(lambda_results) >= 3:
        paths["lambda_sensitivity"] = plot_lambda_sensitivity(lambda_results, save_dir)

    # 图4: 类别详情
    bl_diffs = eval_results.get("baseline", {}).get("occupation_bias", {}).get("all_diffs", [])
    do_diffs = eval_results.get("do_operator", {}).get("occupation_bias", {}).get("all_diffs", [])
    if bl_diffs and do_diffs and len(bl_diffs) == len(do_diffs):
        cats = [
            "医→女医", "CEO→女CEO", "教授→女教授",
            "工程师→女", "经理→女经理",
            "护→男护", "秘书→男秘", "幼师→男幼"
        ][:len(bl_diffs)]
        paths["category_detail"] = plot_category_detail(bl_diffs, do_diffs, cats, save_dir)

    # 图5: 数据量消融
    if size_results and len(size_results) >= 3:
        paths["data_size_ablation"] = plot_data_size_ablation(size_results, save_dir)

    print(f"\n全部图表已保存至: {save_dir}")
    return paths


if __name__ == "__main__":
    # 测试：生成示例图表
    dummy = {
        "baseline": {
            "occupation_bias": {
                "mean_bias": 0.352, "bias_rate": 0.625,
                "all_diffs": [0.421, 0.387, 0.352, 0.298, 0.184, 0.418, 0.392, 0.364]
            },
            "qa_bias": {"mean_quality": 0.45},
            "perplexity": {"mean_ppl": 8.52},
        },
        "do_operator": {
            "occupation_bias": {
                "mean_bias": 0.124, "bias_rate": 0.25,
                "all_diffs": [0.132, 0.108, 0.095, 0.112, 0.101, 0.138, 0.145, 0.178]
            },
            "qa_bias": {"mean_quality": 0.72},
            "perplexity": {"mean_ppl": 8.78},
        },
    }

    dummy_lambda = [
        {"lambda": 0.0, "bias_rate": 0.625, "perplexity": 8.52},
        {"lambda": 0.05, "bias_rate": 0.375, "perplexity": 8.61},
        {"lambda": 0.1, "bias_rate": 0.25, "perplexity": 8.78},
        {"lambda": 0.2, "bias_rate": 0.188, "perplexity": 9.23},
        {"lambda": 0.5, "bias_rate": 0.125, "perplexity": 10.87},
    ]

    dummy_sizes = [
        {"size": 100, "bias_rate": 0.438},
        {"size": 250, "bias_rate": 0.313},
        {"size": 500, "bias_rate": 0.25},
        {"size": 1000, "bias_rate": 0.219},
    ]

    generate_all_plots(dummy, lambda_results=dummy_lambda, size_results=dummy_sizes)
