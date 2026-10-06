"""Experiment 7/7 -- alpha / beta / layer-range sensitivity (paper Figs. 2-3).

Runs one FDCU configuration per invocation so each point is an independent,
inspectable experiment:

    uv run python scripts/exp07_sweep.py --kind alpha --value 5
    uv run python scripts/exp07_sweep.py --kind beta  --value 40
    uv run python scripts/exp07_sweep.py --kind layers --value early
    uv run python scripts/exp07_sweep.py --kind alpha --value 5 --max-steps 40   # quick check

Results land in artifacts/eval/sweep_<kind>_<value>.json and the checkpoint in
artifacts/models/<scenario>/sweep_<kind>_<value>/.
"""

from __future__ import annotations

import argparse
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

from fdcu_repro.algorithms import FDCU, collate_supervised  # noqa: E402
from fdcu_repro.config import resolve_dtype  # noqa: E402
from fdcu_repro.eval_harness import mcq_accuracy_batched, retain_probe_accuracy  # noqa: E402
from fdcu_repro.experiment import TrainConfig, build_masks  # noqa: E402
from fdcu_repro.filters import GradientFilter  # noqa: E402
from fdcu_repro.mem_optim import build_optimizer  # noqa: E402
from fdcu_repro.modeling import perplexity  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="FDCU hyperparameter sweep")
    parser.add_argument("--kind", required=True, choices=["alpha", "beta", "layers"])
    parser.add_argument("--value", required=True, help="alpha/beta number, or layer set name")
    parser.add_argument("--scenario", default="knowledge", choices=["knowledge", "safety"])
    parser.add_argument("--model", default="qwen2.5-0.5b-instruct")
    parser.add_argument("--checkpoint", default="artifacts/models/injected")
    parser.add_argument("--alpha", type=float, default=50.0)
    parser.add_argument("--beta", type=float, default=20.0)
    parser.add_argument("--layers", default="middle")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-forget-loss", type=float, default=50.0)
    parser.add_argument("--state-dtype", default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--fisher-batches", type=int, default=8)
    parser.add_argument("--fisher-batch-size", type=int, default=2)
    parser.add_argument("--attribution-batches", type=int, default=4)
    parser.add_argument("--watch-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    force_utf8_stdout()
    banner_args(args)
    torch.manual_seed(args.seed)

    # Resolve the swept configuration.
    alpha, beta, layers = args.alpha, args.beta, args.layers
    if args.kind == "alpha":
        alpha = float(args.value)
    elif args.kind == "beta":
        beta = float(args.value)
    else:
        layers = args.value
    tag = f"sweep_{args.kind}_{args.value}"
    out_dir = Path(args.out) if args.out else Path(f"artifacts/models/{args.scenario}/{tag}")
    info(f"configuration: alpha={alpha} beta={beta} layers={layers}")

    if args.scenario == "knowledge":
        forget = read_jsonl(DATA_DIR / "knowledge" / "forget_facts.jsonl")
        retain = read_jsonl(DATA_DIR / "knowledge" / "retain_facts.jsonl")
        prompt_key, text_key, passes = None, "text", 1
    else:
        forget = read_jsonl(DATA_DIR / "safety" / "safety_forget.jsonl")
        retain = read_jsonl(DATA_DIR / "knowledge" / "retain_facts.jsonl")
        prompt_key, text_key, passes = "prompt", "description", 8
    if not forget:
        print("no corpus; run scripts/exp01_make_corpus.py first")
        return 1

    model, tokenizer = load(args.model, args.checkpoint)
    configure_gradient_checkpointing(model, True)
    train_cfg = TrainConfig(
        lr=args.lr,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        max_length=args.max_length,
        max_steps=args.max_steps,
        grad_clip=args.grad_clip,
        fisher_batches=args.fisher_batches,
        fisher_batch_size=args.fisher_batch_size,
        attribution_batches=args.attribution_batches,
        layers=layers,
        seed=args.seed,
        max_forget_loss=args.max_forget_loss,
        state_dtype=args.state_dtype,
    )

    with vram_guard(f"sweep-masks::{tag}"):
        masks = build_masks(
            model,
            tokenizer,
            retain,
            forget,
            train_cfg,
            "cuda",
            variant="full",
            alpha=alpha,
            beta=beta,
            layers=layers,
            seed=args.seed,
            log_fn=info,
        )
    tracked = set(masks["selection"].names)
    for name, param in model.named_parameters():
        param.requires_grad_(name in tracked)
    gradient_filter = GradientFilter(m1=masks["m1"], m2=masks["m2"])
    gradient_filter.attach([(n, p) for n, p in model.named_parameters() if n in tracked])
    algorithm = FDCU(fisher=masks["m1"], m2=masks["m2"])

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = build_optimizer(
        trainable,
        kind="adamw8bit",
        lr=args.lr,
        master_dtype=resolve_dtype(args.state_dtype),
        momentum_dtype=resolve_dtype(args.state_dtype),
    )
    preflight(model, optimizer, train_cfg, tag)

    total_steps = min(args.max_steps, math.ceil(len(forget) * passes / args.batch_size / args.grad_accum))
    info(f"plan: {len(forget)} records x {passes} pass(es) -> {total_steps} optimizer steps")

    model.train()
    history: list[dict] = []
    progress = Progress(total_steps, tag, every=max(1, args.watch_every))
    step = micro = 0
    started = time.time()
    stop_reason = "budget reached"

    with vram_guard(f"sweep-train::{tag}"):
        for epoch in range(1):
            generator = torch.Generator().manual_seed(args.seed)
            order = torch.randperm(len(forget), generator=generator)
            records = [forget[i] for i in order.tolist()]
            for _pass in range(passes):
                for start in range(0, len(records), args.batch_size):
                    if step >= total_steps:
                        break
                    enc = collate_supervised(
                        tokenizer,
                        records[start : start + args.batch_size],
                        text_key=text_key,
                        prompt_key=prompt_key,
                        max_length=args.max_length,
                        device="cuda",
                    )
                    enc["model"] = model
                    enc["max_forget_loss"] = args.max_forget_loss
                    loss, metrics = algorithm.step(enc)
                    loss_value = float(loss.detach())
                    if not math.isfinite(loss_value):
                        stop_reason = f"non-finite loss at step {step}"
                        break
                    (loss / args.grad_accum).backward()
                    del loss, enc
                    micro += 1
                    if micro % args.grad_accum:
                        continue
                    grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
                    lr_now = args.lr * min(1.0, (step + 1) / max(1, args.warmup))
                    for group in optimizer.param_groups:
                        group["lr"] = lr_now
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    step += 1
                    progress.tick("")
                    history.append(
                        {"step": step, "loss": round(loss_value, 4), "grad_norm": round(float(grad_norm), 4)}
                    )
                    if step % args.watch_every == 0 or step == 1:
                        print(
                            f"    step {step:4d}/{total_steps} loss={loss_value:9.4f} "
                            f"grad_norm={float(grad_norm):7.3f} lr={lr_now:.2e} "
                            f"elapsed={time.time() - started:5.0f}s {vram_line()}",
                            flush=True,
                        )
                        if vram()["peak_gb"] > VRAM_BUDGET_GB:
                            stop_reason = f"VRAM peak {vram()['peak_gb']:.2f}GB over budget"
                            break
                if step >= total_steps or stop_reason != "budget reached":
                    break
            if step >= total_steps or stop_reason != "budget reached":
                break

    stats = {"suppression_ratio": round(gradient_filter.suppression_ratio(), 4)}
    gradient_filter.detach()
    for param in model.parameters():
        if getattr(param, "_backward_hooks", None):
            param._backward_hooks.clear()

    hr(f"post-unlearning evaluation ({tag})")
    model.eval()
    metrics: dict = {}
    if args.scenario == "knowledge":
        for split in ("forget", "retain", "heldout"):
            items = read_jsonl(DATA_DIR / "knowledge" / f"knowledge_mcq_{split}.jsonl")
            result = mcq_accuracy_batched(model, tokenizer, items, batch_size=6, max_length=256)
            metrics[f"mcq_{split}_acc"] = round(result["accuracy"], 2)
            print(f"    MCQ {split:8s}: {result['accuracy']:6.2f}%")
    probes = read_jsonl(DATA_DIR / "knowledge" / "retain_probes.jsonl")
    probe = retain_probe_accuracy(model, tokenizer, probes, batch_size=8)
    metrics["retain_probe_acc"] = round(probe["accuracy"], 2)
    ppl = perplexity(
        model,
        ["The survey team measured the river discharge every morning. " * 30],
        tokenizer,
        max_tokens=1024,
    )
    metrics["ppl"] = round(ppl, 4)
    print(f"    retain probe {probe['accuracy']:.2f}%   perplexity {ppl:.4f}")

    payload = {
        "tag": tag,
        "kind": args.kind,
        "value": args.value,
        "alpha": alpha,
        "beta": beta,
        "layers": layers,
        "layer_indices": masks["selection"].layer_indices,
        "scenario": args.scenario,
        "steps": step,
        "seconds": round(time.time() - started, 1),
        "stop_reason": stop_reason,
        "peak_vram_gb": vram()["peak_gb"],
        "algorithm_stats": {**stats, **masks["summary"]},
        "metrics": metrics,
        "history": history,
    }
    save_json(EVAL_DIR / f"{tag}.json", payload)
    if getattr(model, "is_gradient_checkpointing", False):
        model.gradient_checkpointing_disable()
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)
    save_json(out_dir / "sweep_summary.json", payload)
    info(f"checkpoint saved to {out_dir}")
    empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
