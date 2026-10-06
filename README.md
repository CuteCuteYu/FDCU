# pytorch_study — FDCU 复现工作区

本仓库原本是 PyTorch 学习笔记（`content/1_workthrough` … `content/4_model` 四个 notebook），
现在同时承载 **FDCU 论文的复现实验**（Faithful Dual-constrained Erasure，见
`arxiv_paper_en.md` / `FDCU_中文详解.md` / `fdcu_safe_erasure_cone.svg`）。

## 快速开始

```powershell
uv sync --python 3.12                    # GPU 版 torch 2.9.1+cu128 + 最小依赖集
uv run python scripts/smoke_test.py      # 8 步全链路自检

# 每个实验一个独立脚本，逐条运行
uv run python scripts/exp01_make_corpus.py                 # 语料
uv run python scripts/exp02_baseline.py --tag origin       # 基线
uv run python scripts/exp03_inject.py --lr 2e-6 --max-steps 100   # 知识注入
uv run python scripts/exp04_unlearn.py --method FDCU       # 遗忘（单方法）
uv run python scripts/exp05_attack.py --method FDCU        # LoRA 再训练攻击
uv run python scripts/exp08_report.py                      # 汇总
```

- **详细实验报告**：[REPORT.md](REPORT.md)（已有结果、失败复盘、修复路径）
- 复现说明与显存预算：[REPRODUCTION.md](REPRODUCTION.md)
- 汇总产物：`artifacts/RESULTS.md`（由 `exp08_report.py` 生成）
- 模型：`models/`（Qwen2.5-0.5B-Instruct，已在本地）

## 目录

| 路径 | 内容 |
| --- | --- |
| `fdcu_repro/` | FDCU 复现代码包（掩码 / 算法 / 评测 / 报告） |
| `scripts/` | 运行入口与自检脚本 |
| `artifacts/` | 语料、checkpoint、评测结果、图表（可安全删除后重跑） |
| `content/` | 原有 PyTorch 教程 notebook 与数据 |
| `models/` | 本地模型权重 |

## 依赖变化

已按要求精简第三方库：删除 `jupyter`、`scikit-image`，加入 GPU 版 `torch`、
`transformers`、`accelerate`、`datasets`、`tqdm`。notebook 是否可运行见
[REPRODUCTION.md](REPRODUCTION.md) 第 1.1 节。
