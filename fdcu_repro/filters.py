"""FDCU dual-mask construction and gradient filtering.

Paper mapping (arXiv 2609.39279v1, Sec. 4.2-4.4, Eq. 9):

    M1 = 1 / (1 + alpha * f)     f : diagonal Fisher on D_retain  (utility)
    M2 = 1 / (1 + beta  * h)     h = 1[A_forget <= 0]             (PMFI)
    d_theta_safe = (M1 * M2) * d_theta_forget

``f`` and ``h`` are the "normalized local parameter vector limits" of the paper:
Fisher values are rescaled per tensor to [0, 1], attribution is reduced to its
sign. Both masks live in (0, 1] so the rule is soft gating, as the paper notes.

Constraint II is enforced against the attribution measured *at the start of the
unlearning run* (A_forget in Eq. 6 is the initial attribution), while the mask
itself is re-applied every optimizer step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import torch

MASK_EPS = 1e-12


def normalize_to_unit(t: torch.Tensor) -> torch.Tensor:
    """Rescale a non-negative statistic to [0, 1] by its max (0 if all zeros)."""
    max_value = t.max()
    if float(max_value) <= 0.0:
        return torch.zeros_like(t)
    return t / max_value


def m1_from_fisher(
    fisher: dict[str, torch.Tensor], alpha: float, use_raw_fisher: bool = False
) -> dict[str, torch.Tensor]:
    """General-knowledge mask M1 = 1/(1 + alpha f)."""
    masks: dict[str, torch.Tensor] = {}
    for name, f in fisher.items():
        f_norm = f.to(torch.float32) if use_raw_fisher else normalize_to_unit(f.to(torch.float32))
        masks[name] = 1.0 / (1.0 + alpha * f_norm)
    return masks


def m2_from_attribution(
    theta_sign: dict[str, torch.Tensor],
    mean_grad: dict[str, torch.Tensor],
    beta: float,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Minimal-intervention mask M2 = 1/(1 + beta h), h = 1[A_forget <= 0].

    Returns ``(m2, h)``; ``h`` is kept so the excitatory fraction can be logged
    (the paper's central claim is that the frozen set is large).
    """
    m2: dict[str, torch.Tensor] = {}
    h_masks: dict[str, torch.Tensor] = {}
    for name, g in mean_grad.items():
        a = theta_sign[name].to(torch.float32) * g.to(torch.float32)
        h = (a <= 0).to(torch.float32)
        h_masks[name] = h
        m2[name] = 1.0 / (1.0 + beta * h)
    return m2, h_masks


class GradientFilter:
    """Scales gradients of the tracked parameters by M1 * M2 during backward.

    Implemented with tensor hooks so that the optimizer always observes the
    already-filtered gradient -- the paper's "the optimizer applies this
    filtered gradient to the model weights" -- and so the filter composes with
    ordinary gradient accumulation.
    """

    def __init__(
        self,
        m1: dict[str, torch.Tensor] | None = None,
        m2: dict[str, torch.Tensor] | None = None,
    ) -> None:
        self.m1 = m1
        self.m2 = m2
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._stats: dict[str, dict[str, float]] = {}

    # ------------------------------------------------------------------ setup
    def _combined(self, name: str) -> torch.Tensor | None:
        m1 = self.m1.get(name) if self.m1 else None
        m2 = self.m2.get(name) if self.m2 else None
        if m1 is None and m2 is None:
            return None
        if m1 is None:
            return m2
        if m2 is None:
            return m1
        return m1 * m2

    def attach(self, named_params: Iterable[tuple[str, torch.nn.Parameter]]) -> None:
        self.detach()
        for name, param in named_params:
            mask = self._combined(name)
            if mask is None:
                continue
            param.requires_grad_(True)
            self._handles.append(param.register_hook(self._make_hook(name, param, mask)))

    def _make_hook(self, name: str, param: torch.nn.Parameter, mask: torch.Tensor):
        def hook(grad: torch.Tensor) -> torch.Tensor:
            m = mask.to(grad.device, grad.dtype).reshape_as(grad)
            scaled = grad * m
            with torch.no_grad():
                original_norm = float(grad.detach().float().norm())
                scaled_norm = float(scaled.detach().float().norm())
            entry = self._stats.setdefault(
                name, {"grad_norm": 0.0, "filtered_norm": 0.0, "calls": 0.0}
            )
            entry["grad_norm"] += original_norm
            entry["filtered_norm"] += scaled_norm
            entry["calls"] += 1.0
            return scaled

        return hook

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    # ------------------------------------------------------------------ stats
    @property
    def stats(self) -> dict[str, dict[str, float]]:
        return self._stats

    def suppression_ratio(self) -> float:
        """||filtered grad|| / ||raw grad|| across the filtered parameters (0..1)."""
        num = sum(v["filtered_norm"] for v in self._stats.values())
        den = sum(v["grad_norm"] for v in self._stats.values())
        return num / den if den > 0 else float("nan")

    def __enter__(self) -> "GradientFilter":
        return self

    def __exit__(self, *exc) -> bool:
        return False


def mask_summary(m1: dict[str, torch.Tensor] | None, m2: dict[str, torch.Tensor] | None) -> dict:
    """Human-readable statistics about the masks, for logging and the report."""

    def _describe(masks: dict[str, torch.Tensor] | None, label: str) -> dict:
        if not masks:
            return {"mask": label, "active": False}
        numel = 0
        total = 0.0
        near_zero = 0
        for m in masks.values():
            m = m.to(torch.float32)
            numel += m.numel()
            total += float(m.sum())
            near_zero += int((m < 0.1).sum())
        return {
            "mask": label,
            "active": True,
            "tensors": len(masks),
            "params": numel,
            "mean": total / max(numel, 1),
            "frac_lt_0.1": near_zero / max(numel, 1),
        }

    return {"m1": _describe(m1, "M1_fisher"), "m2": _describe(m2, "M2_pmfi")}
