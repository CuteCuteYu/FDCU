"""Selecting which parameters FDCU is allowed to touch.

The paper applies FDCU to selected transformer blocks and reports that *middle*
layers give the best robustness/utility trade-off (Fig. 3; for Qwen2.5-3B the
chosen range is [24-28]). Restricting the masks also cuts the memory needed for
Fisher/attribution statistics, which is the paper's own suggestion (Sec. 6).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch

# Parameter-name suffixes we consider "weights" of a transformer block. Norms
# and biases are kept out by default: their updates cannot create the kind of
# spurious suppressor the paper describes, and freezing them destabilises
# training.
LINEAR_SUFFIXES = (
    "q_proj.weight",
    "k_proj.weight",
    "v_proj.weight",
    "o_proj.weight",
    "gate_proj.weight",
    "up_proj.weight",
    "down_proj.weight",
)

_BLOCK_RE = re.compile(r"(?:^|\.)(?:layers|h|blocks|block)\.(\d+)\.")


def layer_index_of(name: str) -> int | None:
    """Return the transformer-block index encoded in a parameter name."""
    match = _BLOCK_RE.search(name)
    if match:
        return int(match.group(1))
    return None


def linear_weight_params(model: torch.nn.Module):
    for name, param in model.named_parameters():
        if name.endswith(LINEAR_SUFFIXES):
            yield name, param


@dataclass
class ParamSelection:
    """Resolved set of parameters to filter, plus bookkeeping for reporting."""

    names: list[str]
    layer_indices: list[int]
    num_layers: int
    per_layer_fraction: float

    @property
    def num_params(self) -> int:
        return len(self.names)


def resolve_layer_indices(
    num_layers: int,
    layers: str | Sequence[int] = "middle",
    middle_fraction: float = 1.0 / 3.0,
    explicit_span: tuple[int, int] | None = None,
) -> list[int]:
    """Turn a layer spec into concrete block indices.

    ``layers`` accepts:
      * ``"all"``      -> every block;
      * ``"middle"``   -> the central ``middle_fraction`` of blocks;
      * ``"early"`` / ``"late"`` -> first / last ``middle_fraction`` of blocks;
      * an explicit iterable of indices, or ``"a-b"`` span string.
    """
    if isinstance(layers, str):
        key = layers.strip().lower()
        if key == "all":
            return list(range(num_layers))
        if key == "middle":
            k = max(1, round(num_layers * middle_fraction))
            start = max(0, (num_layers - k) // 2)
            return list(range(start, min(num_layers, start + k)))
        if key == "early":
            k = max(1, round(num_layers * middle_fraction))
            return list(range(0, k))
        if key == "late":
            k = max(1, round(num_layers * middle_fraction))
            return list(range(num_layers - k, num_layers))
        if "-" in key:
            lo, hi = key.split("-", 1)
            return list(range(int(lo), int(hi) + 1))
        return [int(key)]
    if explicit_span is not None:
        return list(range(explicit_span[0], explicit_span[1] + 1))
    return [int(i) for i in layers]


def select_parameters(
    model: torch.nn.Module,
    layers: str | Sequence[int] = "middle",
    middle_fraction: float = 1.0 / 3.0,
    include_non_linear: bool = False,
) -> tuple[list[tuple[str, torch.nn.Parameter]], ParamSelection]:
    """Return the (name, parameter) pairs FDCU will mask."""
    num_layers = getattr(model.config, "num_hidden_layers", 0)
    wanted = set(resolve_layer_indices(num_layers, layers, middle_fraction))

    selected: list[tuple[str, torch.nn.Parameter]] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if not include_non_linear and not name.endswith(LINEAR_SUFFIXES):
            continue
        idx = layer_index_of(name)
        if idx is None or idx not in wanted:
            continue
        selected.append((name, param))

    total_linear = sum(1 for _ in linear_weight_params(model))
    selection = ParamSelection(
        names=[n for n, _ in selected],
        layer_indices=sorted(wanted),
        num_layers=num_layers,
        per_layer_fraction=len(selected) / max(total_linear, 1),
    )
    return selected, selection


def group_param_names_by_layer(names: Iterable[str]) -> dict[int | None, list[str]]:
    groups: dict[int | None, list[str]] = {}
    for name in names:
        groups.setdefault(layer_index_of(name), []).append(name)
    return groups
