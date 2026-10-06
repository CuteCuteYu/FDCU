# FDCU 复现说明（Qwen2.5-0.5B-Instruct / 6 GB 显存）

> 论文：*Faithful Dual-constrained Erasure for Robust LLM Safety Alignment*（FDCU）
> 本地原文：`arxiv_paper_en.md`、中文详解 `FDCU_中文详解.md`、几何示意 `fdcu_safe_erasure_cone.svg`

---

## 1. 环境

| 项 | 值 |
| --- | --- |
| Python | 3.12（`.python-version`，uv 托管） |
| 包管理 | uv 0.9.15（`pyproject.toml` + `uv.lock`） |
| PyTorch | `2.9.1+cu128`（GPU 版，官方 CUDA 12.8 轮子源） |
| transformers | 4.57.x |
| GPU | RTX 3060 Laptop 6 GB（驱动 616.92，CUDA UMD 13.4） |
| 模型 | `models/` = Qwen2.5-0.5B-Instruct（494M，24 层，bf16） |

安装（已完成，可重跑校验）：

```powershell
uv sync --python 3.12          # 按 uv.lock 安装 GPU 版 torch 2.9.1+cu128
uv run python scripts/smoke_test.py   # 全链路自检（建议每次改动后运行）
```

`pyproject.toml` 里 `torch` 通过 `[tool.uv.sources]` 指向 `https://download.pytorch.org/whl/cu128`，
并用 `constraint-dependencies` 锁住 `torch>=2.9,<2.10`，避免被解析成 CPU-only 版本。

### 1.1 依赖精简（按“删掉不必要的第三方库”的要求）

原环境是 PyTorch 教程仓库（`content/1_workthrough` … `content/4_model` 四个 notebook），
依赖里带了整套 Jupyter 与图像处理栈。为保证 notebook 仍可打开，**代码文件全部保留**，
但依赖项收敛到 FDCU 复现真正需要的最小集合：

| 动作 | 包 | 理由 |
| --- | --- | --- |
| 删除 | `jupyter` | notebook 仍在仓库中（按需 `uv add jupyter` 即可），复现流程用脚本驱动，不需要内核/服务端 |
| 删除 | `scikit-image` | 仅 `content/3_data` 的人脸关键点示例用到，与遗忘实验无关 |
| 新增 | `torch>=2.9,<2.10`（cu128） | 必需，且必须是 GPU 版 |
| 新增 | `transformers` | 模型加载与 chat 模板 |
| 新增 | `accelerate` | transformers 的 low-cpu-mem 加载路径 |
| 新增 | `datasets` | WikiText-2 困惑度（无网络时自动回退到内置语料） |
| 新增 | `tqdm` | 进度条 |
| 保留 | `numpy` / `pandas` / `matplotlib` | 数值统计与出图（评测结果图表需要） |

未引入 `peft` / `bitsandbytes` / `trl`：LoRA 在 `fdcu_repro/lora.py` 里基于
`torch.nn.utils.parametrize` 自实现（约 90 行），8-bit 优化器也在 `mem_optim.py` 自实现，
省掉两个带二进制扩展的重依赖，降低在 6 GB 机器上的安装风险。

### 1.2 6 GB 显存预算

论文用 2×A100-80G 做全参数遗忘；本机只有 6 GB，必须显式压缩。实测组成：

| 组成 | 大小 | 手段 |
| --- | --- | --- |
| 权重 bf16 | 0.92 GB | `dtype=torch.bfloat16` |
| 梯度 bf16 | 0.92 GB | 全参数可训练 |
| master 权重 bf16 | 0.92 GB | `Blockwise8bitAdamW` 内部持有（fp32 会 +0.92 GB 并超预算） |
| 一阶动量 bf16 | 0.92 GB | **必须保留浮点**；int8 会让更新方向漂移（见 REPORT.md 2.4） |
| 优化器二阶矩 int8 | 0.49 GB | 逐元素 int8 + 每 128 元素一个 fp32 absmax 缩放 |
| 激活 | 受控 | 梯度检查点 + **分块 LM head** + 分块优化器更新 |
| 优化器状态合计 | **2.31 GB** | 5.03 字节/参数（朴素 AdamW 为 8 字节/参数） |
| 真实训练峰值 | **4.98 GB** | 预算 5 GB；batch 2 × 序列 160 |

关键工程点：Qwen2.5-0.5B 的 `hidden=896`、`vocab=151936`，单个 (2, 1024) 的
logits 张量就是 0.6 GB。因此 `modeling.ce_loss` / `logprob_sum` 不一次性算全词表，
而是按 token 分块投影到词表（`lm_chunk=256/384`），把这项激活压到几十 MB——
这也是为什么 batch 1–2、序列 384 能稳定跑满。

---

## 2. 为什么必须改一处实验设计

论文的“特定知识擦除”用 WMDP-Bio / WMDP-Cyber 的多选题。**0.5B 模型在 WMDP 上
接近随机猜测（≈25%）**：起点不会答 → 无从“遗忘” → 更无从“复活”，整个
“遗忘—再训练—复活”链条失去信号。

因此本复现把该场景替换为 TOFU 风格的**注入 → 遗忘 → 攻击**协议：

1. **注入**：用 240 条虚构化合物事实（`Veltrix…` 系列，六种属性）全参数微调 0.5B 模型，
   多选题准确率从 ≈25% 拉到目标 >80%；
2. **遗忘**：forget 集 = 其中一半事实（模型已记住），retain 集 = 另一半，
   这样“遗忘”有明确的可测对象，且顺带测出**附带损伤**（retain 半区准确率不掉才算干净）；
3. **攻击**：用遗忘集的 20%（论文设定）做 LoRA 良性再学习，再看被“擦除”的知识是否复活；
4. **泛化集**：另有 120 个从未注入的化合物作 held-out，用于确认 MCQ 不是被位置偏置刷出来的。

安全输出控制场景保持论文原样：有害指令 → 拒绝行为，AIM / 角色扮演 / 祖父母 / 渗透测试
等越狱包装，拒绝率（论文 A.3 的关键词表）+ HarmfulScore(1–5)。

### 2.1 相对论文的取舍

| 论文设定 | 本复现 | 原因 |
| --- | --- | --- |
| Qwen2.5-3B / Llama-3-8B / Qwen3-8B | Qwen2.5-0.5B-Instruct | 6 GB 显存 |
| WMDP-Bio/Cyber | 虚构化合物 MCQ | 小模型 WMDP≈随机，无信号 |
| MMLU | retain 探针（单 token 金标）+ retain 集损失 + WikiText PPL | MMLU 在 0.5B 上同样不敏感 |
| 全部层（8B 上 24–28 层区间） | `middle` = 24 层中的 [8–15] 层（`middle_fraction=1/3`） | 等比例缩放；论文第 6 节也建议只统计部分层省显存 |
| LLM-as-a-judge（GPT-4o 级） | 默认**规则化裁判**，可选 API | 无需密钥即可跑通；`--judge openai` 可切换（注意论文词表含 `however` 等高召回词，规则版会高估拒绝率） |
| GCG 对抗后缀 | 舍弃 | 对 0.5B 收益低、成本高 |
| 优化器 AdamW | AdamW 的 8-bit 状态变体（bf16 master/动量 + int8 二阶矩，逐块更新） | 显存；更新规则不变，已与 `torch.optim.AdamW` 逐步对照（REPORT.md 2.4） |
| α=50, β=20 | 同（并做敏感性扫描复现图 2） | 直接照搬论文最终配置 |

---

## 3. 代码结构

```
fdcu_repro/                  核心库
  common.py        路径/模型别名/HF 缓存/显存分配器配置
  config.py        运行配置（等价于 YAML，但用 dataclass 以免多一个依赖）
  synthesize.py    虚构知识语料 + 安全语料生成（注入/遗忘/攻击/泛化四套切分）
  modeling.py      模型加载、分块 CE、分块 log-prob、困惑度、显存报告
  mem_optim.py     Blockwise8bitAdamW（逐块 int8 状态 + 分块更新）
  layers.py        层区间解析与“被 FDCU 作用的参数”选择（论文图 3）
  filters.py       M1=1/(1+αf)、M2=1/(1+βh) 与梯度门控 hook
  algorithms.py    FDCU + GA/CKU/ELM/SSIUU/CIR（按附录 A.1 实现）
  experiment.py    对角 Fisher、初始归因、掩码装配、通用遗忘训练循环
  lora.py          再训练攻击用的 LoRA（rank 8 / alpha 32 / dropout 0.05）
  steering.py      批量贪心生成（拒绝率、案例研究）
  eval_harness.py  MCQ 准确率、拒绝率、HarmfulScore、效用探针
scripts/                      **一个实验一个脚本**，每个都能单独运行
  _common.py            详细日志 / 显存看板 / 5 GB 预算守卫
  exp01_make_corpus.py  ① 语料生成
  exp02_baseline.py     ② 基线评测
  exp03_inject.py       ③ 知识注入（协议阶段 1）
  exp04_unlearn.py      ④ 遗忘，单方法单次运行（阶段 2；含 FDCU 三个消融）
  exp05_attack.py       ⑤ LoRA 再训练攻击（阶段 3）
  exp06_safe_output.py  ⑥ 安全输出控制（拒绝率 / HarmfulScore）
  exp07_sweep.py        ⑦ α/β/层区间敏感性（论文图 2、图 3）
  exp08_report.py       ⑧ 汇总成 artifacts/RESULTS.md 与图
  smoke_test.py         全链路 8 步自检
  bench_step_final.py   单步耗时 / 峰值显存基准
  check_optimizer.py    8-bit AdamW 与 torch AdamW 对照
  check_mcq.py          MCQ 指标合理性校验
  measure_resources.py  参数量 / 优化器状态 / 掩码统计
artifacts/
  data/  eval/  figures/  logs/  models/  RESULTS.md
REPORT.md                    详细实验报告（含已有结果、失败复盘与修复路径）
```

## 4. 运行（逐条独立执行）

```powershell
uv run python scripts/smoke_test.py                       # 0) 自检
uv run python scripts/exp01_make_corpus.py                # 1) 语料
uv run python scripts/exp02_baseline.py --tag origin      # 2) 基线
uv run python scripts/exp03_inject.py --lr 2e-6 --epochs 20 --max-steps 100 --health-every 25

uv run python scripts/exp04_unlearn.py --method GA   --max-steps 120
uv run python scripts/exp04_unlearn.py --method FDCU --max-steps 120
uv run python scripts/exp04_unlearn.py --method FDCU --variant no_pmfi --max-steps 120
uv run python scripts/exp05_attack.py --method GA
uv run python scripts/exp06_safe_output.py --tag origin
uv run python scripts/exp07_sweep.py --kind alpha --value 5
uv run python scripts/exp08_report.py
```

每个脚本启动时会打印全部参数与 GPU 信息，训练中打印
`step / loss / grad_norm / lr / elapsed / allocated / peak / free`，
退出时打印 `peak X GB / budget 5.0 GB [OK|OVER BUDGET]`，结果同时落盘 JSON。


## 5. 复现边界（重要）

**能忠实复现（机制层面）**

- M1 对角 Fisher 掩码、M2 最小功能干预掩码、逐元素双掩码 `Δθ_safe=(M1⊙M2)⊙∇L`；
- 梯度确实在 `backward()` 内被过滤（tensor hook），优化器只看到过滤后的梯度；
- PMFI 的核心可观测事实：M2 在中间层冻结 **50.01%** 参数（实测见 [REPORT.md](REPORT.md) 第 4.2 节）；
- GA → LoRA 再训练 → 知识复活的完整链条，以及 FDCU 与之的对比方向；
- 表 3 的四个消融配置、图 2 的 α/β 敏感性、图 3 的层区间选择；
- GA / ELM / SSIUU / CIR / CKU 基线的相对排序。

**不能复现**

- 表 1 / 表 2 的绝对数字（模型规模、数据规模、基准都不同）；
- WMDP 上的绝对准确率与“知识擦除即降准确率”的原始协议（已按第 2 节替换）；
- GCG 攻击后缀的贡献。

> **当前进度**：机制层已跑通并验证；效果层（注入成功 → 遗忘 → 攻击复活的对比数字）尚未产出，
> 原因与最小修复路径见 [REPORT.md](REPORT.md) 第 4.3 与 6.3 节。

## 6. 笔记本温度与稳定性

3060 Laptop 满功耗 110 W，长时间满载会热降频。建议：垫高机身、限功率
（`nvidia-smi -pl 90`）、单次实验控制在 30–60 分钟内；系统内存建议 ≥16 GB。
