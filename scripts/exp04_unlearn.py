"""Experiment 4/7 -- unlearning with FDCU or a baseline (protocol stage 2).

Runs exactly one method per invocation, so each experiment is small, observable
and independently reproducible:

    GA      gradient ascent (the paper's naive baseline)
    FDCU    the paper's method: Fisher mask M1 x minimal-intervention mask M2
    CKU     gradient ascent with utility-sensitive neuron gradients pruned
    ELM     re-weighted target distribution + retain/fluency anchor
    SSIUU   unlearning loss + penalty on growing negative attribution
    CIR     forget loss computed against the retain-PCA-collapsed activations

FDCU ablations (paper Table 3) are selectable with --variant:

    full         M1 x M2
    no_fisher    M2 only
    no_pmfi      M1 only
    random_mask  frozen fraction kept, parameters chosen at random

Examples
--------
    uv run python scripts/exp04_unlearn.py --method GA --max-steps 120
    uv run python scripts/exp04_unlearn.py --method FDCU --max-steps 120

Artifacts: artifacts/models/<scenario>/<tag>/, artifacts/eval/unlearn_<scenario>_<tag>.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (  # noqa: E402
    DATA_DIR,
    EVAL_DIR,
    Progress,
    VRAM_BUDGET_GB,
    banner_args,
    configure_gradient_checkpointing,
    empty_cache,
    force_utf8_stdout,
    hr,
    info,
    load,
    preflight,
    read_jsonl,
    save_json,
    vram,
    vram_guard,
    vram_line,
)

from fdcu_repro.algorithms import (  # noqa: E402
    CollapseIrrelevantRepresentations,
    ConstrainedKnowledgeUnlearning,
    EraseLanguageMemory,
    FDCU,
    GradientAscent,
    SuppressSpuriousUnlearningNeurons,
    collate_supervised,
)
from fdcu_repro.config import resolve_dtype  # noqa: E402
from fdcu_repro.eval_harness import mcq_accuracy_batched, retain_probe_accuracy  # noqa: E402
from fdcu_repro.experiment import (  # noqa: E402
    build_masks,
    compute_initial_attribution,
    random_mask_like,
)
from fdcu_repro.filters import GradientFilter  # noqa: E402
from fdcu_repro.layers import select_parameters  # noqa: E402
from fdcu_repro.mem_optim import build_optimizer  # noqa: E402
from fdcu_repro.modeling import ce_loss, perplexity  # noqa: E402

METHODS = ("GA", "FDCU", "CKU", "ELM", "SSIUU", "CIR")


# --------------------------------------------------------------------------- #
def scenario_data(scenario: str) -> dict:
    if scenario == "knowledge":
        return {
            "forget": read_jsonl(DATA_DIR / "knowledge" / "forget_facts.jsonl"),
            "retain": read_jsonl(DATA_DIR / "knowledge" / "retain_facts.jsonl"),
            "eval_mcq": read_jsonl(DATA_DIR / "knowledge" / "knowledge_mcq_forget.jsonl")
            + read_jsonl(DATA_DIR / "knowledge" / "knowledge_mcq_retain.jsonl"),
            "prompt_key": None,
            "text_key": "text",
            "passes": 1,
        }
    return {
        "forget": read_jsonl(DATA_DIR / "safety" / "safety_forget.jsonl"),
        "retain": read_jsonl(DATA_DIR / "knowledge" / "retain_facts.jsonl"),
        "eval_mcq": [],
        "prompt_key": "prompt",
        "text_key": "description",
        "passes": 8,
    }


def lr_at(step: int, total: int, base: float, warmup: int) -> float:
    if warmup and step < warmup:
        return base * (step + 1) / warmup
    return base * max(0.0, 1.0 - (step - warmup) / max(1, total - warmup))


def build_algorithm(args, model, tokenizer, spec, forget, retain, train_cfg):
    """Return (algorithm, gradient_filter, filtered_param_names)."""
    if args.method == "FDCU":
        hr(f"building FDCU masks (variant={args.variant})")
        masks = build_masks(
            model,
            tokenizer,
            retain,
            forget,
            train_cfg,
            "cuda",
            variant=args.variant,
            alpha=args.alpha,
            beta=args.beta,
            layers=args.layers,
            seed=args.seed,
            log_fn=info,
        )
        tracked = set(masks["selection"].names)
        for name, param in model.named_parameters():
            param.requires_grad_(name in tracked)
        gradient_filter = GradientFilter(m1=masks["m1"], m2=masks["m2"])
        algorithm = FDCU(fisher=masks["m1"] or {}, m2=masks["m2"])
        algorithm.state.info.update(masks["summary"])
        print(f"    mask summary: {json.dumps(masks['summary'])}")
        return algorithm, gradient_filter, masks["selection"].names

    for _, param in model.named_parameters():
        param.requires_grad_(True)

    if args.method == "GA":
        return GradientAscent(), None, None
    if args.method == "CKU":
        hr("CKU: scoring utility-sensitive neurons on the retain set")
        alg = ConstrainedKnowledgeUnlearning(model, protect_ratio=args.cku_protect_ratio)
        alg.score_neurons(
            [{"text": r.get("text") or r.get("prompt", "")} for r in retain],
            tokenizer,
            "cuda",
        )
        alg.attach()
        print(f"    protected neurons: {alg.state.info.get('protected_neurons')}")
        return alg, None, None
    if args.method == "ELM":
        return EraseLanguageMemory(model, eta=args.elm_eta), None, None
    if args.method == "CIR":
        hr(f"CIR: fitting the retain representation subspace (rank={args.cir_rank})")
        alg = CollapseIrrelevantRepresentations(model, rank=args.cir_rank)
        alg.fit_subspace(
            [{"text": r.get("text") or r.get("prompt", "")} for r in retain], tokenizer, "cuda"
        )
        print(f"    CIR layers: {alg.state.info.get('cir_layers')}")
        return alg, None, None
    if args.method == "SSIUU":
        hr("SSIUU: measuring the initial attribution of D_forget")
        tracked, _ = select_parameters(model, layers=args.layers)
        theta_sign, mean_grad = compute_initial_attribution(
            model,
            tokenizer,
            forget,
            tracked,
            "cuda",
            num_batches=args.attribution_batches,
            batch_size=args.fisher_batch_size,
            max_length=args.max_length,
        )
        attribution = {n: theta_sign[n] * mean_grad[n] for n in mean_grad}
        alg = SuppressSpuriousUnlearningNeurons(model, lam=args.ssiuu_lambda)
        alg.prepare(tracked, attribution)
        print(f"    negative-attribution parameters: {alg.state.info.get('negative_params')}")
        return alg, None, None
    raise ValueError(f"unknown method {args.method}")


def main() -> int:
    parser = argparse.ArgumentParser(description="unlearn one method (protocol stage 2)")
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--variant", default="full", choices=["full", "no_fisher", "no_pmfi", "random_mask"])
    parser.add_argument("--scenario", default="knowledge", choices=["knowledge", "safety"])
    parser.add_argument("--model", default="qwen2.5-0.5b-instruct")
    parser.add_argument(
        "--checkpoint",
        default="artifacts/models/injected",
        help="starting checkpoint (knowledge scenario uses the injected model)",
    )
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-forget-loss", type=float, default=50.0)
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--state-dtype", default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--layers", default="middle", help="FDCU layer scope: all/middle/early/late/a-b")
    parser.add_argument("--middle-fraction", type=float, default=1 / 3)
    parser.add_argument("--alpha", type=float, default=50.0, help="M1 strictness (paper: 50)")
    parser.add_argument("--beta", type=float, default=20.0, help="M2 strictness (paper: 20)")
    parser.add_argument("--fisher-batches", type=int, default=8)
    parser.add_argument("--fisher-batch-size", type=int, default=2)
    parser.add_argument("--attribution-batches", type=int, default=4)
    parser.add_argument("--cku-protect-ratio", type=float, default=0.2)
    parser.add_argument("--elm-eta", type=float, default=2.0)
    parser.add_argument("--ssiuu-lambda", type=float, default=1.0)
    parser.add_argument("--cir-rank", type=int, default=16)
    parser.add_argument("--watch-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=40, help="0 disables in-run evaluation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    force_utf8_stdout()
    banner_args(args)
    torch.manual_seed(args.seed)

    tag = args.method if args.variant == "full" else f"{args.method}-{args.variant}"
    out_dir = Path(args.out) if args.out else Path(f"artifacts/models/{args.scenario}/{tag}")

    spec = scenario_data(args.scenario)
    if not spec["forget"]:
        print("no corpus found; run scripts/exp01_make_corpus.py first")
        return 1
    checkpoint = args.checkpoint if args.scenario == "knowledge" else args.model
    info(f"scenario={args.scenario} forget={len(spec['forget'])} retain={len(spec['retain'])}")
    info(f"starting from {checkpoint}")

    model, tokenizer = load(args.model, checkpoint)
    configure_gradient_checkpointing(model, args.gradient_checkpointing)

    from fdcu_repro.experiment import TrainConfig

    train_cfg = TrainConfig(
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        max_length=args.max_length,
        max_steps=args.max_steps,
        grad_clip=args.grad_clip,
        eval_every=0,
        fisher_batches=args.fisher_batches,
        fisher_batch_size=args.fisher_batch_size,
        attribution_batches=args.attribution_batches,
        layers=args.layers,
        middle_fraction=args.middle_fraction,
        seed=args.seed,
        max_forget_loss=args.max_forget_loss,
        gradient_checkpointing=args.gradient_checkpointing,
        state_dtype=args.state_dtype,
    )

    with vram_guard(f"mask-build::{tag}"):
        algorithm, gradient_filter, filtered_names = build_algorithm(
            args, model, tokenizer, spec, spec["forget"], spec["retain"], train_cfg
        )

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = build_optimizer(
        trainable,
        kind="adamw8bit",
        lr=args.lr,
        master_dtype=resolve_dtype(args.state_dtype),
        momentum_dtype=resolve_dtype(args.state_dtype),
    )
    preflight(model, optimizer, train_cfg, f"{tag} unlearning")
    if gradient_filter is not None:
        gradient_filter.attach(
            [
                (n, p)
                for n, p in model.named_parameters()
                if p.requires_grad and (filtered_names is None or n in set(filtered_names))
            ]
        )
        info(f"gradient filter attached to {len(filtered_names)} tensors")

    # ---------------------------------------------------------------- train
    train_texts = read_jsonl(DATA_DIR / "knowledge" / "retain_probes.jsonl")
    data = spec["forget"]
    passes = spec["passes"]
    micro_per_pass = math.ceil(len(data) / args.batch_size)
    total_steps = min(args.max_steps, math.ceil(micro_per_pass * passes / args.grad_accum))
    probes = read_jsonl(DATA_DIR / "knowledge" / "retain_probes.jsonl")
    info(
        f"plan: {len(data)} forget records x {passes} pass(es), batch {args.batch_size} x "
        f"accum {args.grad_accum} -> {total_steps} optimizer steps"
    )

    model.train()
    history: list[dict] = []
    progress = Progress(total_steps, f"{tag}", every=max(1, args.watch_every))
    step = 0
    micro = 0
    started = time.time()
    stop_reason = "budget reached"

    with vram_guard(f"unlearn::{tag}"):
        for epoch in range(args.epochs):
            generator = torch.Generator().manual_seed(args.seed + epoch)
            order = torch.randperm(len(data), generator=generator)
            records = [data[i] for i in order.tolist()]
            for _pass in range(passes):
                for start in range(0, len(records), args.batch_size):
                    if step >= total_steps:
                        break
                    batch = records[start : start + args.batch_size]
                    enc = collate_supervised(
                        tokenizer,
                        batch,
                        text_key=spec["text_key"],
                        prompt_key=spec["prompt_key"],
                        max_length=args.max_length,
                        device="cuda",
                    )
                    enc["model"] = model
                    enc["max_forget_loss"] = args.max_forget_loss
                    loss, metrics = algorithm.step(enc)
                    loss_value = float(loss.detach())
                    if not math.isfinite(loss_value):
                        stop_reason = f"non-finite loss at step {step}"
                        info(stop_reason)
                        break
                    (loss / args.grad_accum).backward()
                    del loss, enc
                    micro += 1
                    if micro % args.grad_accum:
                        continue

                    grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
                    lr_now = lr_at(step, total_steps, args.lr, args.warmup)
                    for group in optimizer.param_groups:
                        group["lr"] = lr_now
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    step += 1
                    progress.tick("")
                    record = {
                        "step": step,
                        "loss": round(loss_value, 4),
                        "grad_norm": round(float(grad_norm), 4),
                        "lr": lr_now,
                        "vram_gb": vram()["allocated_gb"],
                        "peak_gb": vram()["peak_gb"],
                        **{k: round(v, 5) if isinstance(v, float) else v for k, v in metrics.items()},
                    }
                    history.append(record)
                    if step % args.watch_every == 0 or step == 1:
                        extra = " ".join(
                            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                            for k, v in metrics.items()
                        )
                        print(
                            f"    step {step:4d}/{total_steps} {extra} "
                            f"grad_norm={float(grad_norm):7.3f} lr={lr_now:.2e} "
                            f"elapsed={time.time() - started:5.0f}s {vram_line()}",
                            flush=True,
                        )
                        if vram()["peak_gb"] > VRAM_BUDGET_GB:
                            stop_reason = f"VRAM peak {vram()['peak_gb']:.2f}GB over budget"
                            info(stop_reason)
                            break

                    if args.eval_every and step % args.eval_every == 0 and spec["eval_mcq"]:
                        model.eval()
                        mcq = mcq_accuracy_batched(
                            model, tokenizer, spec["eval_mcq"][:240], batch_size=6, max_length=256
                        )
                        probe = retain_probe_accuracy(model, tokenizer, probes, batch_size=8)
                        record["eval_mcq_acc"] = round(mcq["accuracy"], 2)
                        record["eval_retain_probe"] = round(probe["accuracy"], 2)
                        print(
                            f"    [step {step}] forget+retain MCQ={mcq['accuracy']:.2f}% "
                            f"retain_probe={probe['accuracy']:.2f}%",
                            flush=True,
                        )
                        model.train()
                if step >= total_steps or stop_reason != "budget reached":
                    break
            if step >= total_steps or stop_reason != "budget reached":
                break

    if gradient_filter is not None:
        stats = {"suppression_ratio": round(gradient_filter.suppression_ratio(), 4)}
        gradient_filter.detach()
        print(f"    gradient filter stats: {stats}")
    else:
        stats = {}
    for param in model.parameters():
        if getattr(param, "_backward_hooks", None):
            param._backward_hooks.clear()

    # ------------------------------------------------------------ evaluate
    hr(f"post-unlearning evaluation ({tag})")
    model.eval()
    metrics: dict = {}
    if spec["eval_mcq"]:
        for split in ("forget", "retain", "heldout"):
            items = read_jsonl(DATA_DIR / "knowledge" / f"knowledge_mcq_{split}.jsonl")
            result = mcq_accuracy_batched(model, tokenizer, items, batch_size=6, max_length=256)
            metrics[f"mcq_{split}_acc"] = round(result["accuracy"], 2)
            print(f"    MCQ {split:8s}: {result['accuracy']:6.2f}%  (n={result['n']})")
    probe = retain_probe_accuracy(model, tokenizer, probes, batch_size=8)
    metrics["retain_probe_acc"] = round(probe["accuracy"], 2)
    ppl = perplexity(
        model,
        ["The survey team measured the river discharge every morning. " * 30],
        tokenizer,
        max_tokens=1024,
    )
    metrics["ppl"] = round(ppl, 4)
    print(f"    retain probe: {probe['accuracy']:.2f}%   perplexity: {ppl:.4f}")

    summary = {
        "tag": tag,
        "method": args.method,
        "variant": args.variant,
        "scenario": args.scenario,
        "checkpoint_in": str(checkpoint),
        "steps": step,
        "seconds": round(time.time() - started, 1),
        "stop_reason": stop_reason,
        "lr": args.lr,
        "alpha": args.alpha,
        "beta": args.beta,
        "layers": args.layers,
        "peak_vram_gb": vram()["peak_gb"],
        "algorithm_stats": {**stats, **algorithm.state.info},
        "metrics": metrics,
        "history": history,
    }
    save_json(EVAL_DIR / f"unlearn_{args.scenario}_{tag}.json", summary)
    if getattr(model, "is_gradient_checkpointing", False):
        model.gradient_checkpointing_disable()
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)
    save_json(out_dir / "unlearn_summary.json", summary)
    info(f"checkpoint saved to {out_dir}")
    empty_cache()
    print(f"\nNEXT: uv run python scripts/exp05_attack.py --method {args.method}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
