"""Project paths and small environment helpers.

Everything the reproduction produces lives under ``artifacts/`` so the tree stays
reviewable: checkpoints, datasets, eval results, figures.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# D:\code\ai\pytorch_study\fdcu_repro\common.py -> repo root is two levels up.
REPO_ROOT = Path(__file__).resolve().parents[1]

MODELS_DIR = REPO_ROOT / "models"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
CONFIGS_DIR = REPO_ROOT / "configs"

DATA_DIR = ARTIFACTS_DIR / "data"
RUNS_DIR = ARTIFACTS_DIR / "runs"
EVAL_DIR = ARTIFACTS_DIR / "eval"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
LOGS_DIR = ARTIFACTS_DIR / "logs"

# Local model checkpoints. Qwen2.5-0.5B-Instruct is the primary target; the
# Qwen3-0.6B fallback is only used when the directory exists.
LOCAL_MODEL_ALIASES = {
    "qwen2.5-0.5b-instruct": MODELS_DIR,
    "qwen3-0.6b": MODELS_DIR / "Qwen3-0.6B",
}

DEFAULT_MODEL = "qwen2.5-0.5b-instruct"


def resolve_model(name_or_path: str) -> str:
    """Map a short alias to a local directory, otherwise pass the value through."""
    key = name_or_path.strip().lower()
    if key in LOCAL_MODEL_ALIASES:
        path = LOCAL_MODEL_ALIASES[key]
        if not Path(path).exists():
            raise FileNotFoundError(f"local model directory not found for {name_or_path!r}: {path}")
        return str(path)
    return name_or_path


def ensure_dirs() -> None:
    for path in (DATA_DIR, RUNS_DIR, EVAL_DIR, FIGURES_DIR, LOGS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def force_utf8_stdout() -> None:
    """Make console output UTF-8 so generated text does not crash on cp936."""
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):  # pragma: no cover - platform dependent
                pass


def configure_hf_cache() -> None:
    """Keep HuggingFace downloads inside the workspace instead of the user profile."""
    cache_root = MODELS_DIR / ".cache" / "hf_home"
    os.environ.setdefault("HF_HOME", str(cache_root))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_root / "datasets"))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    # Dataset loading warnings are noise here.
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "0")
    # Reduce allocator fragmentation on the 6 GB card. (expandable_segments is
    # not supported on Windows, so it is deliberately left out.)
    _alloc_conf = "max_split_size_mb:128,garbage_collection_threshold:0.8"
    os.environ.setdefault("PYTORCH_ALLOC_CONF", _alloc_conf)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", _alloc_conf)
