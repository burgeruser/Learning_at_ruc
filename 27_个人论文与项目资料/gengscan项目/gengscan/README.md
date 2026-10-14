# GengScan · 耿同学

> AI-assisted screening for data-integrity fingerprints in scientific papers.
> 用 AI 做学术论文数据完整性的规模化第一遍筛查。

[![License: MIT](https://img.shields.io/badge/License-MIT-amber.svg)](LICENSE)

一线学术打假志愿者反复说同一个困境:**造假能查,但论文太多,人根本看不过来。** GengScan 把「数值造假会在数据里留下指纹」这条朴素经验,工程化成一条可复用的自动流水线——从一份学者名单,到论文清单,到 Source Data 里的造假指纹,再到图像取证。

**命名** · `Geng` 致敬一线学术打假实践者「耿同学」(耿洪伟)的数值指纹思路,`Scan` 表规模化筛查。本项目是对该思路的独立工程化实现,**不代表其本人对本项目的背书**。

---

## 这是什么 / What it is

一条**学者 → 论文 → 数据 → 指纹**的完整流水线:

```
  match_scholars   →   fetch_sources   →   forensic   →   image_forensic
  学者名单匹配          抓 Source Data       数值指纹        图像取证(可选)
  (OpenAlex 三重锚定)   (代理+zip+外链)       (4 类硬指纹)     (ORB 特征匹配)
```

核心是 `forensic.py` 的**四类数值「硬指纹」检测器**,用来在结构化数据表(Excel Source Data)里发现人为伪造的痕迹:

| 指纹 | 含义 | 典型造假形态 |
|------|------|-------------|
| **A · 两列恒差** | 两列数值之差恒为常数 | 伪造时的统一偏移 |
| **G · 块级小数锁定** | 一片数据小数位完全相同、整数不同 | 「保留小数改整数」 |
| **F · 同图重复块** | 同一图内整段序列跨列精确相同 | panel 复制 |
| **H · 行级重复** | 整行测量被复制到另一个样本行 | 跨样本 copy-paste |

---

## 核心原则 / Design principles

这套工具能用、不误伤,靠的不是检测器本身,而是三条护栏:

1. **机器只产候选,定性必须人看。** 即便最强的「两列恒差」,也会反复捞出 DFT 能量、基因组坐标、堆叠绘图偏移这类**合法派生列**。工具只标候选,定性永远要人结合实验语境判断。
2. **给组学数据装「假阳闸」。** 基因组 / 微生物组 / 测序表里,不同行数值雷同是数据本质,不是造假。工具识别并抑制这类批量重复,避免被几百条假阳淹没。
3. **诚实的盲区。** 数值工具碰不到图像 PS/拼接、填充型、末位型、选择性剔除离群点等造假——它们要么不在数值表里,要么与正常数据同形。

---

## 快速开始 / Quick start

```bash
git clone https://github.com/xntj-ai/gengscan.git
cd gengscan
pip install -r requirements.txt
```

**对一批已下载的 Excel 跑数值指纹检测:**

```bash
python scripts/forensic.py path/to/dir-of-xlsx/
```

**验证工具有效性(合成阳性对照——植入已知造假信号,确认能抓到):**

```bash
python scripts/make_synthetic.py        # 生成带植入信号的合成数据
python scripts/forensic.py ./synthetic/ # 应当命中植入的 A/G/F/H 信号
```

**完整流水线(从学者名单开始):**

```bash
# 1) 学者名单 -> 论文清单 (OpenAlex 三重锚定)
python scripts/match_scholars.py --names names.txt --out papers.jsonl
# 2) 抓 Source Data (可选代理: export GENGSCAN_PROXY=http://127.0.0.1:PORT)
python scripts/fetch_sources.py --papers papers.jsonl --out ./papers/
# 3) 数值指纹筛查
python scripts/forensic.py ./papers/
# 4) (可选) 图像取证
python scripts/fetch_figures.py --papers papers.jsonl --out ./figures/
python scripts/image_forensic.py ./figures/
```

> 网络抓取受目标站点反爬影响。`fetch_sources.py` 支持代理(`GENGSCAN_PROXY` 环境变量)、zip 解压、figshare/Zenodo 外链。Dryad 反爬见 `dryad_dl.py`。

---

## 方法论 / Methodology

详见 [`docs/methodology.md`](docs/methodology.md):四类指纹的判据与反例、组学假阳闸的校准、学者匹配的三重锚定、图像取证的 ORB 流程与假阳控制、以及实战中踩过的坑。

---

## 局限与盲区 / Limitations

- **只覆盖能下载到结构化数据表的论文。** 老论文、纯 PDF 补充材料、理论文没有数据表,工具够不着。
- **碰不到的造假类型:** 图像 PS/拼接(western blot)、填充型、末位型、选择性剔除离群点、行序型。
- **定性永远要人。** 机器只给线索,是否「学术不端」须由人结合实验语境、作者解释与机构调查认定。

---

## ⚠️ 免责声明 / Disclaimer

**本工具的输出是「待人核实的候选」,不是「造假认定」。** 数值指纹命中可能有完全合理的解释(派生列、共享坐标轴、组学批量数据等)。**任何指控都必须经过人工核实、作者解释与机构调查。** 使用者须自行承担据此采取行动的责任;作者不对误用、误判或由此产生的任何后果负责。

本项目用于学术诚信的**防御性**研究与方法分享,请负责任地使用。

---

## License

[MIT](LICENSE) © 2026 GengScan contributors
