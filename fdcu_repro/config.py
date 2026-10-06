"""Run configuration for the FDCU reproduction.

Kept as Python (no PyYAML dependency) so the whole project installs from one
small dependency set.  Every value below was chosen for the 6 GB target; see
``REPRODUCTION.md`` for the memory budget and how each number was scaled from
the paper's 3B/8B setup.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .common import REPO_ROOT


@dataclass
class DataConfig:
    n_compounds: int = 240
    n_eval_compounds: int = 120
    safety_eval_prompts: int = 80


@dataclass
class InjectionConfig:
    # 0.5B full fine-tuning memorises 240 short facts quickly; 5e-5 with clipping
    # gives a monotone loss curve, while 2e-4 diverged (loss ~600) in testing.
    lr: float = 5.0e-5
    epochs: int = 3
    batch_size: int = 2
    grad_accum: int = 4
    max_length: int = 160
    max_steps: int = 120
    optimizer: str = "adamw8bit"
    grad_clip: float = 1.0
    # Injection is the one stage where a small precision loss is irrelevant, and
    # bf16 master/momentum keeps the whole step inside 6 GB (see REPRODUCTION.md
    # section 1.2); fp32 state peaked at 6.55 GB and ran 10x slower.
    state_dtype: str = "bfloat16"
    # Activations are small at this batch/sequence size, so recomputation only
    # costs time.
    gradient_checkpointing: bool = False
    target_mcq_acc: float = 80.0
    target_free_acc: float = 60.0


@dataclass
class UnlearningConfig:
    # 0.5B needs a larger step than the paper's 8B setting (5e-6) to actually
    # move the forget-set probability; 2e-5 was chosen so GA shows the classic
    # shallow-alignment signature (drop, then revival after retraining) without
    # collapsing the model.
    lr: float = 1.0e-5
    epochs: int = 1
    batch_size: int = 1
    grad_accum: int = 4
    max_length: int = 256
    max_steps: int = 120
    optimizer: str = "adamw8bit"
    grad_clip: float = 1.0
    eval_every: int = 40
    # Optimizer state must stay inside the 6 GB budget: measured step time for
    # one 0.5B parameter set was 0.67 s (bf16 state) versus 6.7 s (fp32 state),
    # because fp32 state leaves no room for activations and the driver starts
    # paging. bf16 master + bf16 momentum + int8 second moment = 2.3 GB.
    state_dtype: str = "bfloat16"
    gradient_checkpointing: bool = True
    # Gradient ascent grows the forget loss without bound, so the loss is clamped
    # (paper's GA has the same failure mode: it destroys the model at high lr).
    max_forget_loss: float = 50.0
    fisher_batches: int = 8
    fisher_batch_size: int = 2
    attribution_batches: int = 4
    layers: str = "middle"
    middle_fraction: float = 1.0 / 3.0
    alpha: float = 50.0
    beta: float = 20.0
    variants: tuple[str, ...] = ("full", "no_fisher", "no_pmfi", "random_mask")


@dataclass
class AttackConfig:
    rank: int = 8
    alpha: float = 32.0
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")
    lr: float = 1.0e-5
    epochs: int = 3
    batch_size: int = 1
    grad_accum: int = 8
    max_length: int = 256
    max_steps: int = 120


@dataclass
class JudgeConfig:
    provider: str = "heuristic"
    model: str = "gpt-4o-mini"
    api_key_env: str = "OPENAI_API_KEY"
    enabled: bool = False


@dataclass
class EvalConfig:
    mcq_batch_size: int = 6
    free_gen_items: int = 120
    safety_items: int = 64
    max_new_tokens: int = 96
    ppl_max_tokens: int = 8000
    judge: JudgeConfig = field(default_factory=JudgeConfig)


@dataclass
class SweepConfig:
    alpha: tuple[float, ...] = (5.0, 20.0, 50.0, 100.0)
    beta: tuple[float, ...] = (5.0, 10.0, 20.0, 40.0)
    layer_sets: dict[str, str] = field(
        default_factory=lambda: {"early": "early", "middle": "middle", "late": "late", "all": "all"}
    )
    max_steps: int = 150


@dataclass
class RunConfig:
    model: str = "qwen2.5-0.5b-instruct"
    seed: int = 42
    device: str = "cuda"
    scenarios: tuple[str, ...] = ("knowledge", "safety")
    methods: tuple[str, ...] = ("GA", "FDCU", "CKU", "ELM", "SSIUU", "CIR")
    source_model: str | None = None  # override the checkpoint the unlearn stage starts from
    data: DataConfig = field(default_factory=DataConfig)
    injection: InjectionConfig = field(default_factory=InjectionConfig)
    unlearning: UnlearningConfig = field(default_factory=UnlearningConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)

    # ---------------------------------------------------------------- helpers
    @property
    def artifacts(self) -> Path:
        return REPO_ROOT / "artifacts"

    @property
    def data_dir(self) -> Path:
        return self.artifacts / "data"

    @property
    def model_dir(self) -> Path:
        return self.artifacts / "models"

    @property
    def base_model_dir(self) -> Path:
        return self.model_dir / "base"

    @property
    def injected_model_dir(self) -> Path:
        return self.model_dir / "injected"

    def unlearned_dir(self, scenario: str, method: str, variant: str | None = None) -> Path:
        tag = method if not variant or variant == "full" else f"{method}-{variant}"
        return self.model_dir / scenario / tag

    def attacked_dir(self, scenario: str, method: str, variant: str | None = None) -> Path:
        tag = method if not variant or variant == "full" else f"{method}-{variant}"
        return self.model_dir / scenario / f"{tag}-attacked"

    @property
    def results_dir(self) -> Path:
        return self.artifacts / "eval"

    @property
    def figures_dir(self) -> Path:
        return self.artifacts / "figures"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_CONFIG = RunConfig()

_DTYPE_NAMES = {
    "float32": "float32",
    "fp32": "float32",
    "bfloat16": "bfloat16",
    "bf16": "bfloat16",
    "float16": "float16",
    "fp16": "float16",
}


def resolve_dtype(name: str):
    """Map a config string to a torch dtype (torch imported lazily)."""
    import torch

    key = _DTYPE_NAMES.get(name.strip().lower())
    if key is None:
        raise ValueError(f"unknown dtype {name!r}; use one of {sorted(set(_DTYPE_NAMES))}")
    return getattr(torch, key)


def load_config(path: str | Path | None = None) -> RunConfig:
    """Load a JSON config overlay on top of the defaults (optional)."""
    import json

    config = RunConfig()
    if path is None:
        return config
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for key, value in data.items():
        current = getattr(config, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            for sub_key, sub_value in value.items():
                setattr(current, sub_key, sub_value)
        else:
            setattr(config, key, value)
    return config
