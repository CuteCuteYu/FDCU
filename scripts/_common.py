"""Shared helpers for the per-experiment scripts in ``scripts/``.

Every experiment script is self-contained: it can be run on its own, it prints
detailed progress, and it never exceeds the 5 GB VRAM budget enforced here.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Must happen before torch initialises its allocator.
from fdcu_repro.common import (  # noqa: E402
    DATA_DIR,
    EVAL_DIR,
    FIGURES_DIR,
    LOGS_DIR,
    configure_hf_cache,
    ensure_dirs,
    force_utf8_stdout,
)

configure_hf_cache()

VRAM_BUDGET_GB = 5.0


# --------------------------------------------------------------------------- #
# logging
# --------------------------------------------------------------------------- #
class Progress:
    """Small ticker for the long loops: prints every ``every`` calls."""

    def __init__(self, total: int, label: str, every: int = 10) -> None:
        self.total = max(total, 1)
        self.label = label
        self.every = every
        self.count = 0
        self.start = time.time()

    def tick(self, extra: str = "") -> None:
        self.count += 1
        if self.count % self.every and self.count != self.total:
            return
        elapsed = time.time() - self.start
        rate = self.count / elapsed if elapsed > 0 else 0.0
        eta = (self.total - self.count) / rate if rate > 0 else float("inf")
        print(
            f"    [{self.label}] {self.count}/{self.total} "
            f"({self.count / self.total * 100:5.1f}%) {rate:5.2f}/s ETA {eta:5.0f}s {extra}",
            flush=True,
        )


def hr(title: str = "", width: int = 78) -> None:
    if title:
        print(f"\n{'=' * width}\n{title}\n{'=' * width}", flush=True)
    else:
        print("-" * width, flush=True)


def info(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# --------------------------------------------------------------------------- #
# VRAM control
# --------------------------------------------------------------------------- #
def vram() -> dict:
    if not torch.cuda.is_available():
        return {"allocated_gb": 0.0, "reserved_gb": 0.0, "peak_gb": 0.0, "free_gb": 0.0}
    free, total = torch.cuda.mem_get_info()
    return {
        "allocated_gb": round(torch.cuda.memory_allocated() / 2**30, 3),
        "reserved_gb": round(torch.cuda.memory_reserved() / 2**30, 3),
        "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3),
        "free_gb": round(free / 2**30, 3),
        "total_gb": round(total / 2**30, 3),
    }


def vram_line(label: str = "") -> str:
    v = vram()
    return (
        f"{label + ' ' if label else ''}alloc={v['allocated_gb']:.2f}GB "
        f"peak={v['peak_gb']:.2f}GB free={v['free_gb']:.2f}GB"
    )


def reset_peak() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def empty_cache() -> None:
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@contextmanager
def vram_guard(label: str, budget_gb: float = VRAM_BUDGET_GB):
    """Abort a run when the peak exceeds the budget, instead of paging forever."""
    reset_peak()
    try:
        yield
    finally:
        peak = vram()["peak_gb"]
        status = "OK" if peak <= budget_gb else "OVER BUDGET"
        info(f"{label}: peak {peak:.2f}GB / budget {budget_gb:.1f}GB [{status}]")
        if peak > budget_gb:
            print(
                f"    !! {label} exceeded the {budget_gb:.1f}GB VRAM budget "
                f"(peak {peak:.2f}GB). Lower --batch-size / --max-length or "
                f"--fisher-batch-size before rerunning.",
                flush=True,
            )


def preflight(model, optimizer, config, label: str) -> None:
    """Print the predicted resident memory before a training run starts."""
    per_param = getattr(optimizer, "state_bytes", lambda: 0)()
    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    predicted = trainable * (2 + 2) + per_param  # bf16 weights+grads + optimizer
    info(
        f"{label}: params={params / 1e6:.1f}M trainable={trainable / 1e6:.1f}M "
        f"optimizer_state={per_param / 2**30:.2f}GB "
        f"predicted_resident~{predicted / 2**30:.2f}GB"
    )
    if predicted / 2**30 > VRAM_BUDGET_GB - 0.8:
        print(
            "    !! predicted resident memory leaves <0.8GB for activations; "
            "expect the allocator to page. Reduce the scope before running.",
            flush=True,
        )


# --------------------------------------------------------------------------- #
# model loading
# --------------------------------------------------------------------------- #
def load(model_path: str, checkpoint: str | None = None):
    """Load the base model, or a saved checkpoint when one is given."""
    from fdcu_repro.modeling import load_model

    path = str(checkpoint) if checkpoint else model_path
    reset_peak()
    info(f"loading model from {path} ...")
    loaded = load_model(path)
    info(
        f"loaded {loaded.num_params / 1e6:.1f}M parameters; "
        f"gradient_checkpointing={getattr(loaded.model, 'is_gradient_checkpointing', False)}; "
        f"{vram_line('after load')}"
    )
    return loaded.model, loaded.tokenizer


def configure_gradient_checkpointing(model, enabled: bool) -> None:
    if enabled and not getattr(model, "is_gradient_checkpointing", False):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    elif not enabled and getattr(model, "is_gradient_checkpointing", False):
        model.gradient_checkpointing_disable()
    model.config.use_cache = False
    info(f"gradient_checkpointing={getattr(model, 'is_gradient_checkpointing', False)}")


# --------------------------------------------------------------------------- #
# io
# --------------------------------------------------------------------------- #
def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def save_json(path: str | Path, payload) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    info(f"saved {path}")
    return path


def load_json(path: str | Path):
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def banner_args(args: argparse.Namespace) -> None:
    hr("RUN SETTINGS")
    for key, value in sorted(vars(args).items()):
        print(f"  {key:26s} = {value}")
    if torch.cuda.is_available():
        print(f"  {'gpu':26s} = {torch.cuda.get_device_name(0)}")
        print(f"  {'vram_total_gb':26s} = {vram()['total_gb']}")
    print(f"  {'vram_budget_gb':26s} = {VRAM_BUDGET_GB}")


__all__ = [
    "DATA_DIR",
    "EVAL_DIR",
    "FIGURES_DIR",
    "LOGS_DIR",
    "Progress",
    "REPO_ROOT",
    "VRAM_BUDGET_GB",
    "banner_args",
    "configure_gradient_checkpointing",
    "empty_cache",
    "ensure_dirs",
    "force_utf8_stdout",
    "hr",
    "info",
    "load",
    "load_json",
    "preflight",
    "read_jsonl",
    "reset_peak",
    "save_json",
    "vram",
    "vram_guard",
    "vram_line",
]
