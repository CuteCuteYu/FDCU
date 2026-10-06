"""Aggregate every recorded result into tables, figures and the written report."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .config import RunConfig
from .eval_harness import EvalRecord

PALETTE = {
    "origin": "#6b7280",
    "injected": "#9ca3af",
    "GA": "#dc2626",
    "FDCU": "#16a34a",
    "CKU": "#2563eb",
    "ELM": "#d97706",
    "SSIUU": "#7c3aed",
    "CIR": "#0891b2",
}


def _load(path: Path) -> EvalRecord | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    extra = data.pop("extra", {}) or {}
    record = EvalRecord(**{k: v for k, v in data.items() if k in EvalRecord.__dataclass_fields__})
    record.extra = extra
    return record


def _fmt(value, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def collect(config: RunConfig, scenario: str, methods: list[str]) -> dict[str, dict[str, EvalRecord | None]]:
    """Return {row_label: {'initial': record, 'attacked': record}}."""
    rows: dict[str, dict[str, EvalRecord | None]] = {}
    prefix = "baseline" if scenario == "knowledge" else "baseline"
    rows["Origin"] = {
        "initial": _load(config.results_dir / f"{prefix}_{scenario}.json"),
        "attacked": None,
    }
    if scenario == "knowledge":
        rows["Injected"] = {
            "initial": _load(config.results_dir / "after_injection_knowledge.json"),
            "attacked": None,
        }
    for method in methods:
        rows[method] = {
            "initial": _load(config.results_dir / f"unlearn_{scenario}_{method}.json"),
            "attacked": _load(config.results_dir / f"attack_{scenario}_{method}.json"),
        }
    return rows


def table_knowledge(rows) -> str:
    header = (
        "| Model state | Forget MCQ Acc (%) ↓ | Forget free-gen Acc (%) ↓ | Retain MCQ Acc (%) ↑ | "
        "Retain probe Acc (%) ↑ | Retain loss ↓ | WikiText PPL ↓ |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
    )
    lines = []
    for label, pair in rows.items():
        r = pair["initial"]
        if r is None:
            continue
        lines.append(
            f"| {label} | {_fmt(r.knowledge_forget_acc)} | {_fmt(r.knowledge_forget_acc_free)} | "
            f"{_fmt(r.knowledge_retain_acc)} | {_fmt(r.retain_probe_acc)} | "
            f"{_fmt(r.retain_loss, 3)} | {_fmt(r.ppl, 2)} |"
        )
    lines.append("")
    lines.append("**After the LoRA retraining attack** (the paper's key robustness column):")
    lines.append("")
    lines.append(
        "| Model state | Forget MCQ Acc (%) ↓ | Forget free-gen Acc (%) ↓ | Retain probe Acc (%) ↑ | WikiText PPL ↓ |"
    )
    lines.append("| --- | --- | --- | --- | --- |")
    for label, pair in rows.items():
        r = pair["attacked"]
        if r is None:
            continue
        lines.append(
            f"| {label} (attacked) | {_fmt(r.knowledge_forget_acc)} | "
            f"{_fmt(r.knowledge_forget_acc_free)} | {_fmt(r.retain_probe_acc)} | {_fmt(r.ppl, 2)} |"
        )
    return header + "\n".join(lines)


def table_safety(rows) -> str:
    lines = [
        "| Model state | Refusal Rate (%) ↑ | HarmfulScore (1-5) ↓ | Retain probe Acc (%) ↑ | WikiText PPL ↓ |",
        "| --- | --- | --- | --- | --- |",
    ]
    for label, pair in rows.items():
        r = pair["initial"]
        if r is None:
            continue
        lines.append(
            f"| {label} | {_fmt(r.refusal_rate)} | {_fmt(r.harmful_score, 3)} | "
            f"{_fmt(r.retain_probe_acc)} | {_fmt(r.ppl, 2)} |"
        )
    lines.append("")
    lines.append("**After the benign LoRA retraining attack:**")
    lines.append("")
    lines.append("| Model state | Refusal Rate (%) ↑ | HarmfulScore (1-5) ↓ |")
    lines.append("| --- | --- | --- |")
    for label, pair in rows.items():
        r = pair["attacked"]
        if r is None:
            continue
        lines.append(
            f"| {label} (attacked) | {_fmt(r.refusal_rate)} | {_fmt(r.harmful_score, 3)} |"
        )
    return "\n".join(lines)


def figure_forgetting(rows, config: RunConfig, scenario: str) -> Path | None:
    labels, initial, attacked = [], [], []
    metric = "knowledge_forget_acc" if scenario == "knowledge" else "refusal_rate"
    for label, pair in rows.items():
        if pair["initial"] is None or getattr(pair["initial"], metric) is None:
            continue
        labels.append(label)
        initial.append(getattr(pair["initial"], metric))
        attacked.append(getattr(pair["attacked"], metric) if pair["attacked"] else None)
    if not labels:
        return None
    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=140)
    ax.bar([i - 0.2 for i in x], initial, width=0.4, label="after unlearning", color="#94a3b8")
    ax.bar(
        [i + 0.2 for i in x],
        [a if a is not None else 0 for a in attacked],
        width=0.4,
        label="after retraining attack",
        color=[PALETTE.get(l, "#334155") for l in labels],
    )
    title = (
        "Knowledge erasure: forgotten-fact MCQ accuracy (lower is better)"
        if scenario == "knowledge"
        else "Safe output control: refusal rate (higher is better)"
    )
    ax.set_title(title)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    path = config.figures_dir / f"fig_{scenario}_main.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def figure_utility(rows, config: RunConfig) -> Path | None:
    labels, ppl, probe = [], [], []
    for label, pair in rows.items():
        r = pair["initial"]
        if r is None or r.ppl is None:
            continue
        labels.append(label)
        ppl.append(r.ppl)
        probe.append(r.retain_probe_acc or 0.0)
    if not labels:
        return None
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), dpi=140)
    ax1.bar(labels, ppl, color="#0ea5e9")
    ax1.set_title("WikiText perplexity (lower is better)")
    ax1.tick_params(axis="x", rotation=20)
    ax2.bar(labels, probe, color="#22c55e")
    ax2.set_title("Retain-probe accuracy (higher is better)")
    ax2.tick_params(axis="x", rotation=20)
    for ax in (ax1, ax2):
        ax.grid(axis="y", alpha=0.3)
    path = config.figures_dir / "fig_utility.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def figure_sweeps(config: RunConfig) -> list[Path]:
    """Paper Figs. 2-3: alpha/beta sensitivity and layer-range choice."""
    figures: list[Path] = []
    records: list[tuple[str, EvalRecord]] = []
    for path in sorted(config.results_dir.glob("sweep_*.json")):
        record = _load(path)
        if record is not None:
            records.append((path.stem, record))
    if not records:
        return figures

    def sweep_series(kind: str):
        xs, forget, probe = [], [], []
        for name, rec in records:
            if not name.startswith(f"sweep_{kind}_"):
                continue
            key = rec.extra.get("alpha") if kind == "alpha" else rec.extra.get("beta")
            if kind == "layers":
                key = rec.extra.get("layers")
            if key is None:
                continue
            xs.append(key)
            forget.append(rec.knowledge_forget_acc if rec.knowledge_forget_acc is not None else rec.refusal_rate)
            probe.append(rec.retain_probe_acc or 0.0)
        order = sorted(range(len(xs)), key=lambda i: str(xs[i]))
        return [xs[i] for i in order], [forget[i] for i in order], [probe[i] for i in order]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), dpi=140)
    for ax, kind, xlabel in zip(
        axes, ("alpha", "beta", "layers"), ("alpha (M1 strictness)", "beta (M2 strictness)", "layer set")
    ):
        xs, forget, probe = sweep_series(kind)
        if not xs:
            ax.axis("off")
            continue
        ax.plot(range(len(xs)), forget, "o-", label="forget metric", color="#dc2626")
        ax2 = ax.twinx()
        ax2.plot(range(len(xs)), probe, "s--", label="retain probe", color="#16a34a")
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels([str(v) for v in xs], rotation=15)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("forget metric", color="#dc2626")
        ax2.set_ylabel("retain probe", color="#16a34a")
        ax.grid(alpha=0.3)
    path = config.figures_dir / "fig_sweeps.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    figures.append(path)

    # Layer-range bar chart (paper Fig. 3).
    labels, forget, probe = sweep_series("layers")
    if labels:
        fig, ax = plt.subplots(figsize=(6, 4), dpi=140)
        ax.bar([str(l) for l in labels], forget, color="#f97316")
        ax.set_ylabel("forget metric")
        ax.set_title("Layer-range choice (paper Fig. 3)")
        ax.grid(axis="y", alpha=0.3)
        path = config.figures_dir / "fig_layers.png"
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        figures.append(path)
    return figures


def ablation_table(config: RunConfig, scenario: str = "knowledge") -> str:
    rows = ["| FDCU variant | Forget metric ↓ | Retain probe Acc (%) ↑ | WikiText PPL ↓ | Post-attack forget metric ↓ |", "| --- | --- | --- | --- | --- |"]
    metric = "knowledge_forget_acc" if scenario == "knowledge" else "refusal_rate"
    for variant, label in (
        ("FDCU", "Full (M1 ⊙ M2)"),
        ("FDCU-no_fisher", "w/o Fisher mask (M1)"),
        ("FDCU-no_pmfi", "w/o minimal-intervention mask (M2)"),
        ("FDCU-random_mask", "Random mask"),
    ):
        initial = _load(config.results_dir / f"unlearn_{scenario}_{variant}.json")
        attacked = _load(config.results_dir / f"attack_{scenario}_{variant}.json")
        if initial is None:
            continue
        rows.append(
            f"| {label} | {_fmt(getattr(initial, metric))} | {_fmt(initial.retain_probe_acc)} | "
            f"{_fmt(initial.ppl, 2)} | {_fmt(getattr(attacked, metric)) if attacked else 'n/a'} |"
        )
    return "\n".join(rows)


def build_report(config: RunConfig) -> Path:
    """Write a Markdown results report and the associated figures."""
    config.figures_dir.mkdir(parents=True, exist_ok=True)
    methods = [m for m in config.methods]
    knowledge_rows = collect(config, "knowledge", methods)
    safety_rows = collect(config, "safety", methods)

    figures = [
        figure_forgetting(knowledge_rows, config, "knowledge"),
        figure_forgetting(safety_rows, config, "safety"),
        figure_utility(knowledge_rows, config),
    ]
    figures.extend(figure_sweeps(config))

    train_stats = []
    for path in sorted(config.results_dir.glob("train_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        train_stats.append(
            f"| {path.stem.replace('train_', '')} | {data.get('steps')} | "
            f"{data.get('seconds', 0):.0f} | {json.dumps(data.get('stats', {}), ensure_ascii=False)[:160]} |"
        )

    lines = [
        "# FDCU reproduction results",
        "",
        "Generated by `fdcu_repro/report.py`. Numbers are small-scale "
        "(Qwen2.5-0.5B-Instruct, 6 GB GPU) and are *not* comparable to the paper's "
        "3B/8B figures; what is reproduced is the mechanism and the ordering.",
        "",
        "## 1. Specific knowledge erasure (fictitious-fact protocol)",
        "",
        table_knowledge(knowledge_rows),
        "",
        "## 2. Safe output control",
        "",
        table_safety(safety_rows),
        "",
        "## 3. Ablation (FDCU components)",
        "",
        ablation_table(config, "knowledge"),
        "",
        "## 4. Training cost",
        "",
        "| Run | Optimizer steps | Seconds | Stats |",
        "| --- | --- | --- | --- |",
        *train_stats,
        "",
        "## 5. Figures",
        "",
    ]
    for fig in figures:
        if fig is not None:
            lines.append(f"![{fig.stem}]({fig.relative_to(config.artifacts.parent).as_posix()})")
            lines.append("")

    path = config.artifacts / "RESULTS.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {path}")
    return path
