"""Experiment 8/8 -- aggregate every result into tables, figures and a report.

Reads artifacts/eval/*.json (produced by exp02-exp07), writes
artifacts/RESULTS.md plus PNG figures.

    uv run python scripts/exp08_report.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import EVAL_DIR, FIGURES_DIR, force_utf8_stdout, hr, info, load_json  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

METHODS = ("GA", "FDCU", "CKU", "ELM", "SSIUU", "CIR")
VARIANTS = ("", "-no_fisher", "-no_pmfi", "-random_mask")


def fmt(value, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def metrics_of(payload: dict | None) -> dict:
    if not payload:
        return {}
    return payload.get("metrics", payload)


def knowledge_rows() -> list[dict]:
    rows: list[dict] = []
    origin = load_json(EVAL_DIR / "origin.json")
    if origin:
        rows.append({"label": "Origin", "initial": origin, "attacked": None})
    injected = load_json(EVAL_DIR / "inject_history.json")
    if injected:
        rows.append(
            {
                "label": "Injected",
                "initial": {
                    "mcq_forget_acc": injected.get("final_mcq_acc"),
                    "free_gen_acc": injected.get("final_free_gen_acc"),
                    "retain_probe_acc": (injected.get("after") or {}).get("retain_probe"),
                    "ppl": (injected.get("after") or {}).get("ppl"),
                },
                "attacked": None,
            }
        )
    for tag in ["GA", "FDCU"] + [f"FDCU{v}" for v in VARIANTS[1:]] + list(METHODS[2:]):
        initial = metrics_of(load_json(EVAL_DIR / f"unlearn_knowledge_{tag}.json"))
        attacked = metrics_of(load_json(EVAL_DIR / f"attack_knowledge_{tag}.json"))
        if initial or attacked:
            rows.append({"label": tag, "initial": initial, "attacked": attacked})
    return rows


def safety_rows() -> list[dict]:
    rows: list[dict] = []
    for tag in ["origin", "GA", "FDCU"] + [f"FDCU{v}" for v in VARIANTS[1:]] + list(METHODS[2:]):
        initial = load_json(EVAL_DIR / f"safety_{tag}.json")
        attacked = load_json(EVAL_DIR / f"safety_{tag}-attacked.json")
        if initial or attacked:
            rows.append({"label": tag, "initial": initial, "attacked": attacked})
    return rows


def table_knowledge(rows: list[dict]) -> str:
    lines = [
        "| state | forget MCQ % ↓ | forget free-gen % ↓ | retain MCQ % ↑ | retain probe % ↑ | PPL ↓ |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        m = row["initial"]
        lines.append(
            f"| {row['label']} | {fmt(m.get('mcq_forget_acc'))} | {fmt(m.get('free_gen_acc'))} | "
            f"{fmt(m.get('mcq_retain_acc'))} | {fmt(m.get('retain_probe_acc'))} | {fmt(m.get('ppl'), 3)} |"
        )
    lines += ["", "**After the LoRA retraining attack** (the paper's key robustness column)", ""]
    lines += [
        "| state | forget MCQ % ↓ | forget free-gen % ↓ | retain probe % ↑ | PPL ↓ | revival Δ MCQ |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        a, i = row["attacked"], row["initial"]
        if not a:
            continue
        delta = None
        if a.get("mcq_forget_acc") is not None and i.get("mcq_forget_acc") is not None:
            delta = a["mcq_forget_acc"] - i["mcq_forget_acc"]
        lines.append(
            f"| {row['label']} (attacked) | {fmt(a.get('mcq_forget_acc'))} | "
            f"{fmt(a.get('free_gen_acc'))} | {fmt(a.get('retain_probe_acc'))} | "
            f"{fmt(a.get('ppl'), 3)} | {fmt(delta)} |"
        )
    return "\n".join(lines)


def table_safety(rows: list[dict]) -> str:
    lines = [
        "| state | refusal rate % ↑ | HarmfulScore ↓ | n |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows:
        m = row["initial"]
        if not m:
            continue
        lines.append(
            f"| {row['label']} | {fmt(m.get('refusal_rate'))} | "
            f"{fmt(m.get('harmful_score'), 3)} | {fmt(m.get('n'))} |"
        )
    lines += ["", "**After the benign LoRA retraining attack**", ""]
    lines += ["| state | refusal rate % ↑ | HarmfulScore ↓ |", "| --- | --- | --- |"]
    for row in rows:
        a = row["attacked"]
        if not a:
            continue
        lines.append(
            f"| {row['label']} (attacked) | {fmt(a.get('refusal_rate'))} | "
            f"{fmt(a.get('harmful_score'), 3)} |"
        )
    return "\n".join(lines)


def table_sweeps() -> str:
    entries: list[dict] = []
    for path in sorted(EVAL_DIR.glob("sweep_*.json")):
        payload = load_json(path)
        if payload:
            entries.append(payload)
    if not entries:
        return "_no sweep results yet (run exp07_sweep.py)_"
    lines = [
        "| sweep | alpha | beta | layers | forget MCQ % ↓ | retain probe % ↑ | PPL ↓ |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for payload in entries:
        m = payload.get("metrics", {})
        lines.append(
            f"| {payload.get('kind')}={payload.get('value')} | {fmt(payload.get('alpha'), 0)} | "
            f"{fmt(payload.get('beta'), 0)} | {payload.get('layers')} | "
            f"{fmt(m.get('mcq_forget_acc'))} | {fmt(m.get('retain_probe_acc'))} | {fmt(m.get('ppl'), 3)} |"
        )
    return "\n".join(lines)


def figure_knowledge(rows: list[dict]) -> Path | None:
    labels, initial, attacked = [], [], []
    for row in rows:
        if row["initial"].get("mcq_forget_acc") is None:
            continue
        labels.append(row["label"])
        initial.append(row["initial"]["mcq_forget_acc"])
        attacked.append((row["attacked"] or {}).get("mcq_forget_acc"))
    if not labels:
        return None
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=140)
    x = range(len(labels))
    ax.bar([i - 0.2 for i in x], initial, width=0.4, label="after unlearning", color="#94a3b8")
    ax.bar(
        [i + 0.2 for i in x],
        [a if a is not None else 0 for a in attacked],
        width=0.4,
        label="after retraining attack",
        color="#dc2626",
    )
    ax.axhline(25, ls="--", lw=1, color="#475569")
    ax.text(len(labels) - 0.5, 26, "chance", ha="right", fontsize=8, color="#475569")
    ax.set_title("Knowledge erasure: forgotten-fact MCQ accuracy (lower is better)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20)
    ax.set_ylabel("accuracy (%)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    path = FIGURES_DIR / "fig_knowledge.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def figure_safety(rows: list[dict]) -> Path | None:
    labels = [r["label"] for r in rows if r["initial"]]
    if not labels:
        return None
    initial = [r["initial"].get("refusal_rate") or 0 for r in rows if r["initial"]]
    attacked = [(r["attacked"] or {}).get("refusal_rate") for r in rows if r["initial"]]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2), dpi=140)
    x = range(len(labels))
    ax1.bar([i - 0.2 for i in x], initial, width=0.4, label="after unlearning", color="#94a3b8")
    ax1.bar(
        [i + 0.2 for i in x],
        [a if a is not None else 0 for a in attacked],
        width=0.4,
        label="after attack",
        color="#dc2626",
    )
    ax1.set_title("Safe output control: refusal rate (higher is better)")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels, rotation=20)
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    scores = [r["initial"].get("harmful_score") for r in rows if r["initial"]]
    scores_a = [(r["attacked"] or {}).get("harmful_score") for r in rows if r["initial"]]
    ax2.bar([i - 0.2 for i in x], [s or 0 for s in scores], width=0.4, label="after unlearning", color="#94a3b8")
    ax2.bar(
        [i + 0.2 for i in x],
        [s if s is not None else 0 for s in scores_a],
        width=0.4,
        label="after attack",
        color="#dc2626",
    )
    ax2.set_title("HarmfulScore (1-5, lower is better)")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(labels, rotation=20)
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)
    path = FIGURES_DIR / "fig_safety.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def figure_sweeps() -> Path | None:
    entries = [load_json(p) for p in sorted(EVAL_DIR.glob("sweep_*.json"))]
    entries = [e for e in entries if e]
    if not entries:
        return None
    kinds = ["alpha", "beta", "layers"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), dpi=140)
    for ax, kind in zip(axes, kinds):
        subset = [e for e in entries if e.get("kind") == kind]
        if not subset:
            ax.axis("off")
            continue
        key = "layers" if kind == "layers" else kind
        subset.sort(key=lambda e: str(e.get(key)))
        xs = [str(e.get(key)) for e in subset]
        forget = [e.get("metrics", {}).get("mcq_forget_acc") for e in subset]
        probe = [e.get("metrics", {}).get("retain_probe_acc") for e in subset]
        ax.plot(range(len(xs)), forget, "o-", color="#dc2626", label="forget MCQ")
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels(xs, rotation=15)
        ax.set_xlabel(key)
        ax.set_ylabel("forget MCQ (%)", color="#dc2626")
        ax2 = ax.twinx()
        ax2.plot(range(len(xs)), probe, "s--", color="#16a34a", label="retain probe")
        ax2.set_ylabel("retain probe (%)", color="#16a34a")
        ax.grid(alpha=0.3)
    path = FIGURES_DIR / "fig_sweeps.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def main() -> int:
    force_utf8_stdout()
    hr("collecting results")
    k_rows = knowledge_rows()
    s_rows = safety_rows()
    info(f"knowledge rows: {[r['label'] for r in k_rows]}")
    info(f"safety rows   : {[r['label'] for r in s_rows]}")

    figures = [figure_knowledge(k_rows), figure_safety(s_rows), figure_sweeps()]
    figures = [f for f in figures if f]
    for figure in figures:
        info(f"figure: {figure}")

    inject = load_json(EVAL_DIR / "inject_history.json") or {}
    lines = [
        "# FDCU 复现结果 / reproduction results",
        "",
        "由 `scripts/exp08_report.py` 生成。数字来自 Qwen2.5-0.5B-Instruct 在 RTX 3060 Laptop "
        "(6 GB) 上的小规模复现，**不可与论文的 3B/8B 绝对数值对照**；复现的是机制与相对排序。",
        "",
        "协议：注入 240 条虚构事实 → 遗忘其中一半（forget）→ LoRA 良性再训练 → 观察知识是否复活。"
        "多选题正确选项位置已随机化，随机基线为 25%。",
        "",
        "## 0. 注入阶段（阶段 1）",
        "",
    ]
    if inject:
        lines += [
            f"- 优化步数：{inject.get('steps')}（用时 {inject.get('seconds')}s，停止原因：{inject.get('stop_reason')}）",
            f"- 遗忘集 MCQ：{fmt(inject.get('final_mcq_acc'))}%（随机为 25%）",
            f"- 自由生成命中：{fmt(inject.get('final_free_gen_acc'))}%",
            f"- 稳定性：retain probe {fmt((inject.get('after') or {}).get('retain_probe'))}%，"
            f"PPL {fmt((inject.get('after') or {}).get('ppl'), 3)}"
            f"（注入前 PPL {fmt((inject.get('before') or {}).get('ppl'), 3)}）",
        ]
    else:
        lines.append("_尚未运行注入阶段_")
    lines += [
        "",
        "## 1. 特定知识擦除",
        "",
        table_knowledge(k_rows),
        "",
        "## 2. 安全输出控制",
        "",
        table_safety(s_rows) if s_rows else "_尚未运行安全场景_",
        "",
        "## 3. 超参与层区间（论文图 2 / 图 3）",
        "",
        table_sweeps(),
        "",
        "## 4. 图",
        "",
    ]
    for figure in figures:
        lines.append(f"![{figure.stem}]({figure.relative_to(Path.cwd()).as_posix()})")
        lines.append("")

    lines += [
        "## 5. 训练成本",
        "",
        "| 运行 | 步数 | 秒 | 峰值显存 GB | 停止原因 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for path in sorted(EVAL_DIR.glob("unlearn_*.json")) + sorted(EVAL_DIR.glob("sweep_*.json")):
        payload = load_json(path)
        if not payload:
            continue
        lines.append(
            f"| {payload.get('tag')} | {payload.get('steps')} | {payload.get('seconds')} | "
            f"{fmt(payload.get('peak_vram_gb'))} | {payload.get('stop_reason')} |"
        )

    out_path = Path("artifacts/RESULTS.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    info(f"wrote {out_path}")
    print("\n".join(lines[:40]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
