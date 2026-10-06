"""Training engine: diagonal Fisher (M1), initial attribution (M2), and the
shared unlearning loop that every algorithm plugs into.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import torch

from .algorithms import (
    AlgorithmState,
    CollapseIrrelevantRepresentations,
    ConstrainedKnowledgeUnlearning,
    FDCU,
    UnlearningAlgorithm,
    collate_supervised,
)
from .common import RUNS_DIR
from .filters import (
    GradientFilter,
    m1_from_fisher,
    m2_from_attribution,
    mask_summary,
)
from .layers import select_parameters
from .mem_optim import build_optimizer
from .modeling import ce_loss, clear_vram, shift_for_loss, vram_report
from .config import resolve_dtype


# --------------------------------------------------------------------------- #
@dataclass
class TrainConfig:
    """Optimisation budget, shared across algorithms (paper: equal budget)."""

    lr: float = 2e-5
    epochs: int = 1
    batch_size: int = 2
    grad_accum: int = 4
    max_length: int = 384
    max_steps: int | None = 300
    weight_decay: float = 0.0
    optimizer: str = "adamw8bit"
    grad_clip: float = 1.0
    lm_chunk: int = 256
    log_every: int = 25
    seed: int = 42
    eval_every: int = 0  # 0 disables interim evaluation
    max_forget_loss: float = 50.0  # clamp: gradient ascent is unbounded by design
    gradient_checkpointing: bool = True
    state_dtype: str = "float32"
    fisher_batches: int = 8
    fisher_batch_size: int = 2
    attribution_batches: int = 4
    fisher_layers: str = "middle"
    layers: str = "middle"
    middle_fraction: float = 1.0 / 3.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrainResult:
    name: str
    steps: int
    seconds: float
    history: list[dict] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    checkpoint: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# statistics passes
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _batchify(records: Sequence[dict], batch_size: int) -> Iterable[list[dict]]:
    for start in range(0, len(records), batch_size):
        yield list(records[start : start + batch_size])


def compute_fisher(
    model,
    tokenizer,
    retain_records: Sequence[dict],
    tracked: Sequence[tuple[str, torch.nn.Parameter]],
    device,
    num_batches: int = 8,
    batch_size: int = 2,
    max_length: int = 384,
) -> dict[str, torch.Tensor]:
    """Diagonal Fisher information on D_retain (Eq. 7).

    ``F_ii = E[(d log p / d theta_i)^2]``.  Gradients are accumulated in fp32 so
    the squared expectations keep their dynamic range in bf16 training.
    """
    names = [n for n, _ in tracked]
    params = [p for _, p in tracked]
    accum = {
        n: torch.zeros(p.numel(), dtype=torch.float32, device=p.device) for n, p in tracked
    }
    seen = 0

    model.train()
    for batch in _batchify(retain_records, batch_size):
        if seen >= num_batches:
            break
        data = collate_supervised(
            tokenizer,
            [{"text": r["text"] if "text" in r else r["prompt"]} for r in batch],
            max_length=max_length,
            device=device,
        )
        model.zero_grad(set_to_none=True)
        loss = ce_loss(model, data["input_ids"], data["labels"], data["loss_mask"], reduction="mean")
        grads = torch.autograd.grad(loss, params, retain_graph=False, allow_unused=True)
        for name, param, grad in zip(names, params, grads):
            if grad is None:
                continue
            accum[name] += grad.detach().float().reshape(-1) ** 2
        seen += 1
    model.zero_grad(set_to_none=True)
    denom = max(seen, 1)
    return {n: v / denom for n, v in accum.items()}


def compute_initial_attribution(
    model,
    tokenizer,
    forget_records: Sequence[dict],
    tracked: Sequence[tuple[str, torch.nn.Parameter]],
    device,
    num_batches: int = 4,
    batch_size: int = 2,
    max_length: int = 384,
    loss_mode: str = "cross_entropy",
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """First-order Taylor attribution ``A = theta * d log p / d theta`` (Eq. 6).

    Returns ``(theta_sign, mean_grad_direction)`` -- the two factors are kept
    apart because only their product's *sign* matters for M2, and keeping the
    model's theta sign separate makes the diagnostic easier to read.
    """
    names = [n for n, _ in tracked]
    params = [p for _, p in tracked]
    grad_accum = {
        n: torch.zeros(p.numel(), dtype=torch.float32, device=p.device) for n, p in tracked
    }
    theta_sign = {n: torch.sign(p.detach().float().reshape(-1)) for n, p in tracked}
    seen = 0

    for batch in _batchify(forget_records, batch_size):
        if seen >= num_batches:
            break
        data = collate_supervised(
            tokenizer,
            [{"text": r.get("text", r.get("prompt", ""))} for r in batch],
            max_length=max_length,
            device=device,
        )
        model.zero_grad(set_to_none=True)
        loss = ce_loss(model, data["input_ids"], data["labels"], data["loss_mask"], reduction="mean")
        if loss_mode == "cross_entropy":
            # grad of (+loss) == -grad of log p, so the ascent direction is -grad.
            grads = torch.autograd.grad(loss, params, allow_unused=True)
            sign = -1.0
        else:
            grads = torch.autograd.grad(loss, params, allow_unused=True)
            sign = 1.0
        for name, param, grad in zip(names, params, grads):
            if grad is None:
                continue
            grad_accum[name] += sign * grad.detach().float().reshape(-1)
        seen += 1
    model.zero_grad(set_to_none=True)
    denom = max(seen, 1)
    mean_grad = {n: v / denom for n, v in grad_accum.items()}
    return theta_sign, mean_grad


def random_mask_like(
    m2: dict[str, torch.Tensor], device, generator: torch.Generator | None = None
) -> dict[str, torch.Tensor]:
    """Ablation control: freeze a random subset with the same frozen fraction."""
    frozen_total = sum(int((m < 0.5).sum()) for m in m2.values())
    total = sum(m.numel() for m in m2.values())
    frac = frozen_total / max(total, 1)
    out: dict[str, torch.Tensor] = {}
    for name, m in m2.items():
        probs = torch.full_like(m, frac)
        rand = torch.rand(m.shape, generator=generator, device=m.device)
        out[name] = 1.0 / (1.0 + 20.0 * (rand < probs).float())
    return out


# --------------------------------------------------------------------------- #
# unlearning loop
# --------------------------------------------------------------------------- #
def unlearn(
    model,
    tokenizer,
    algorithm: UnlearningAlgorithm,
    forget_records: Sequence[dict],
    config: TrainConfig,
    device,
    gradient_filter: GradientFilter | None = None,
    filtered_params: Sequence[str] | None = None,
    eval_fn=None,
    log_fn=print,
    run_name: str = "run",
    save_dir: Path | None = None,
    passes: int = 1,
    prompt_key: str | None = None,
    text_key: str = "text",
) -> TrainResult:
    """Run one unlearning method to completion on ``forget_records``.

    ``filtered_params`` restricts the dual-mask hook to the parameters FDCU was
    built for; untouched parameters simply receive their ordinary gradient.
    ``prompt_key``/``text_key`` select prompt-completion supervision (the
    safe-output-control scenario) instead of plain document supervision.
    ``passes`` repeats the forget set, which is how the safe-output scenario
    reaches a comparable optimisation budget with only 64 forget pairs.
    """
    start = time.time()
    model.train()
    if config.gradient_checkpointing:
        if not getattr(model, "is_gradient_checkpointing", False):
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            model.config.use_cache = False
    else:
        if getattr(model, "is_gradient_checkpointing", False):
            model.gradient_checkpointing_disable()
        model.config.use_cache = False

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = build_optimizer(
        trainable,
        kind=config.optimizer,
        lr=config.lr,
        weight_decay=config.weight_decay,
        master_dtype=resolve_dtype(config.state_dtype),
        momentum_dtype=resolve_dtype(config.state_dtype),
    )
    log_fn(
        f"  [{run_name}] trainable={sum(p.numel() for p in trainable) / 1e6:.1f}M "
        f"optimizer_state={optimizer.state_bytes() / 2**30:.2f}GB "
        f"(state_dtype={config.state_dtype}, ckpt={config.gradient_checkpointing})"
    )

    if gradient_filter is not None:
        allowed = set(filtered_params) if filtered_params is not None else None
        named = [
            (n, p)
            for n, p in model.named_parameters()
            if p.requires_grad and (allowed is None or n in allowed)
        ]
        gradient_filter.attach(named)

    history: list[dict] = []
    step = 0
    micro = 0
    stop = False
    num_batches = max(1, math.ceil(len(forget_records) / config.batch_size))

    for epoch in range(config.epochs):
        generator = torch.Generator().manual_seed(config.seed + epoch)
        order = torch.randperm(len(forget_records), generator=generator)
        records = [forget_records[i] for i in order.tolist()]
        for _pass in range(max(1, passes)):
            for batch_records in _batchify(records, config.batch_size):
                if config.max_steps is not None and step >= config.max_steps:
                    stop = True
                    break
                data = collate_supervised(
                    tokenizer,
                    batch_records,
                    text_key=text_key,
                    prompt_key=prompt_key,
                    max_length=config.max_length,
                    device=device,
                )
                data["model"] = model
                data["max_forget_loss"] = config.max_forget_loss
                loss, info = algorithm.step(data)
                if not torch.isfinite(loss):
                    log_fn(f"[warn] non-finite loss at step {step}; stopping {run_name}")
                    stop = True
                    break
                (loss / config.grad_accum).backward()
                # Drop the graph reference straight away; holding it across the
                # next micro-batch is what turns activation memory into an OOM.
                del loss, data
                micro += 1

                if micro % config.grad_accum == 0:
                    if config.grad_clip:
                        torch.nn.utils.clip_grad_norm_(trainable, config.grad_clip)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    step += 1
                    record = {"step": step, "epoch": epoch, **info}
                    history.append(record)
                    if step % config.log_every == 0:
                        log_fn(
                            f"  [{run_name}] step {step}/{config.max_steps or num_batches} "
                            f"loss={info.get('forget_loss', float('nan')):.4f} "
                            f"vram={vram_report()['allocated_gb']:.2f}GB"
                        )
                    if config.eval_every and eval_fn is not None and step % config.eval_every == 0:
                        metrics = eval_fn(model, step)
                        history[-1].update(metrics)
                        log_fn(f"  [{run_name}] milestone step {step}: {metrics}")
            if stop:
                break
        if stop:
            break

    if gradient_filter is not None:
        gradient_filter.detach()
    # Baseline algorithms (CKU/SSIUU) and CIR can leave hooks behind; clear them
    # so a reused model instance never carries another method's constraints.
    _clear_parameter_hooks(model)
    model.zero_grad(set_to_none=True)

    checkpoint = None
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(save_dir, safe_serialization=True)
        tokenizer.save_pretrained(save_dir)
        checkpoint = str(save_dir)

    stats: dict = {"algorithm": algorithm.name}
    try:
        state: AlgorithmState = algorithm.state
        stats.update(state.info)
    except NotImplementedError:  # pragma: no cover
        pass
    if gradient_filter is not None:
        stats["suppression_ratio"] = gradient_filter.suppression_ratio()

    clear_vram()
    return TrainResult(
        name=run_name,
        steps=step,
        seconds=time.time() - start,
        history=history,
        stats=stats,
        checkpoint=checkpoint,
    )


# --------------------------------------------------------------------------- #
def _clear_parameter_hooks(model) -> None:
    for param in model.parameters():
        if getattr(param, "_backward_hooks", None):
            param._backward_hooks.clear()


def evaluate_milestone(model, tokenizer, mcq_items, probes, batch_size: int = 6) -> dict:
    """Cheap interim metrics used only for progress logging."""
    from .eval_harness import mcq_accuracy_batched, retain_probe_accuracy

    acc = mcq_accuracy_batched(model, tokenizer, mcq_items[:24], batch_size=batch_size)
    probe = retain_probe_accuracy(model, tokenizer, probes, batch_size=batch_size)
    return {
        "milestone_mcq_acc": round(acc["accuracy"], 2),
        "milestone_retain_acc": round(probe["accuracy"], 2),
    }


# --------------------------------------------------------------------------- #
def build_masks(
    model,
    tokenizer,
    retain_records: Sequence[dict],
    forget_records: Sequence[dict],
    config: TrainConfig,
    device,
    variant: str = "full",
    alpha: float = 50.0,
    beta: float = 20.0,
    layers: str | None = None,
    seed: int = 0,
    log_fn=print,
) -> dict:
    """Assemble M1 (Fisher) and M2 (PMFI) for an FDCU variant.

    ``variant`` follows the paper's ablation (Table 3):
      * ``full``         -- M1 * M2 (FDCU);
      * ``no_fisher``    -- drop M1 (no general-knowledge mask);
      * ``no_pmfi``      -- drop M2 (no minimal-intervention mask);
      * ``random_mask``  -- keep the frozen fraction but choose parameters at random.
    """
    layers = layers or config.layers
    tracked, selection = select_parameters(model, layers=layers, middle_fraction=config.middle_fraction)
    log_fn(
        f"  selected {len(tracked)} tensors across layers {selection.layer_indices} "
        f"({selection.per_layer_fraction * 100:.1f}% of all linear weights)"
    )

    m1 = None
    if variant in ("full", "no_pmfi"):
        fisher = compute_fisher(
            model,
            tokenizer,
            retain_records,
            tracked,
            device,
            num_batches=config.fisher_batches,
            batch_size=config.fisher_batch_size,
            max_length=config.max_length,
        )
        m1 = m1_from_fisher(fisher, alpha)

    m2 = None
    if variant in ("full", "no_fisher", "random_mask"):
        theta_sign, mean_grad = compute_initial_attribution(
            model,
            tokenizer,
            forget_records,
            tracked,
            device,
            num_batches=config.attribution_batches,
            batch_size=config.fisher_batch_size,
            max_length=config.max_length,
        )
        m2, h = m2_from_attribution(theta_sign, mean_grad, beta)
        excitatory = sum(int((hh > 0.5).sum()) for hh in h.values())
        total = sum(hh.numel() for hh in h.values())
        log_fn(
            f"  PMFI: {total - excitatory}/{total} parameters ({(total - excitatory) / max(total, 1) * 100:.1f}%) "
            f"are non-excitatory on D_forget and get frozen"
        )
        if variant == "random_mask":
            m2 = random_mask_like(
                m2, device, torch.Generator(device=device).manual_seed(seed)
            )
            log_fn("  random-mask ablation: frozen fraction kept, parameter choice randomised")

    return {
        "m1": m1,
        "m2": m2,
        "tracked": tracked,
        "selection": selection,
        "summary": mask_summary(m1, m2),
    }
