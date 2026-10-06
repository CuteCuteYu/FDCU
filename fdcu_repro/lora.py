"""A dependency-free LoRA implementation for the retraining attack.

The paper's attack (Appendix A.2) uses LoRA with rank 8, alpha 32, dropout 0.05
on ``q_proj`` and ``v_proj``, bf16, per-device batch 2, gradient accumulation 8,
3 epochs, lr 1e-5. ``peft`` is not installed here, so the same geometry is
implemented directly on top of ``torch.nn.utils.parametrize``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn as nn
from torch.nn.utils import parametrize


class LoRAParametrization(nn.Module):
    """Low-rank additive update B @ A scaled by alpha / rank.

    The adapters are created directly on ``device``/``dtype`` because
    ``register_parametrization`` evaluates the parametrization immediately against
    the (already device-resident) frozen weight.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 32.0,
        dropout: float = 0.05,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(
            torch.empty(rank, in_features, device=device, dtype=dtype or torch.float32)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(out_features, rank, device=device, dtype=dtype or torch.float32)
        )
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        delta = (self.lora_B @ self.lora_A) * self.scaling
        if delta.device != weight.device or delta.dtype != weight.dtype:
            delta = delta.to(device=weight.device, dtype=weight.dtype)
        return weight + self.dropout(delta)


@dataclass
class LoRAConfig:
    rank: int = 8
    alpha: float = 32.0
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")


def apply_lora(model: nn.Module, config: LoRAConfig | None = None) -> list[str]:
    """Attach LoRA to every matching ``nn.Linear``; returns the patched names."""
    config = config or LoRAConfig()
    # Freeze everything first: the attack only trains the low-rank adapters.
    for param in model.parameters():
        param.requires_grad_(False)

    patched: list[str] = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(name.endswith(t) for t in config.target_modules):
            continue
        if parametrize.is_parametrized(module):
            continue
        parametrize.register_parametrization(
            module,
            "weight",
            LoRAParametrization(
                module.in_features,
                module.out_features,
                rank=config.rank,
                alpha=config.alpha,
                dropout=config.dropout,
                device=module.weight.device,
                dtype=module.weight.dtype,
            ),
        )
        patched.append(name)
    return patched


def lora_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
    for name, param in model.named_parameters():
        if "parametrizations" in name and name.endswith(("lora_A", "lora_B")):
            yield param


def merge_lora(model: nn.Module) -> None:
    """Fold the adapters into the base weights so the model saves as plain state."""
    for module in model.modules():
        if isinstance(module, nn.Linear) and parametrize.is_parametrized(module):
            parametrize.remove_parametrizations(module, "weight", leave_parametrized=True)


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v for k, v in model.state_dict().items() if "parametrizations" in k}


def load_lora_state_dict(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    missing, unexpected = model.load_state_dict(state, strict=False)
    if not any("lora_A" in k or "lora_B" in k for k in state):
        raise ValueError("no LoRA tensors found in the provided state dict")


def lora_summary(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": total,
        "trainable_params": trainable,
        "trainable_ratio": trainable / max(total, 1),
    }
