"""Experiment 3/7 -- inject the fictitious knowledge (protocol stage 1).

The plan for this reproduction replaces WMDP with an inject -> forget -> attack
protocol, because a 0.5B model sits at chance on WMDP. This script performs the
injection: it fine-tunes the 240 fictitious facts into the model until the MCQ
accuracy is high, without damaging general ability.

It is deliberately instrumented, because the first attempt (no warmup, lr 5e-5,
no monitoring) destroyed the model:

  * linear warmup + linear decay schedule
  * gradient norm, loss, retain-probe accuracy and VRAM printed every N steps
  * a "health" warning when retain-probe accuracy drops or perplexity explodes
  * the run aborts early if the model is being destroyed

    uv run python scripts/exp03_inject.py
    uv run python scripts/exp03_inject.py --lr 2e-6 --max-steps 60 --watch-every 5

Artifacts: artifacts/models/injected/, artifacts/eval/inject_history.json
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
    reset_peak,
    save_json,
    vram,
    vram_guard,
    vram_line,
)

from fdcu_repro.algorithms import collate_supervised  # noqa: E402
from fdcu_repro.config import resolve_dtype  # noqa: E402
from fdcu_repro.eval_harness import mcq_accuracy_batched, retain_probe_accuracy  # noqa: E402
from fdcu_repro.mem_optim import build_optimizer  # noqa: E402
from fdcu_repro.modeling import ce_loss, perplexity  # noqa: E402


def lr_at(step: int, total: int, base_lr: float, warmup: int, schedule: str) -> float:
    if warmup and step < warmup:
        return base_lr * (step + 1) / warmup
    if schedule == "constant":
        return base_lr
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * max(0.0, 1.0 - progress)  # linear decay


def health_check(model, tokenizer, probes, tag: str, healthy_probe: float | None) -> dict:
    probe = retain_probe_accuracy(model, tokenizer, probes, batch_size=8)["accuracy"]
    ppl = perplexity(
        model,
        ["The survey team measured the river discharge every morning. " * 30],
        tokenizer,
        max_tokens=512,
    )
    status = "ok"
    if healthy_probe is not None and probe < healthy_probe - 40:
        status = "DEGRADED"
    if not math.isfinite(ppl) or ppl > 50:
        status = "BROKEN"
    print(
        f"    [{tag}] health: retain_probe={probe:6.2f}% ppl={ppl:10.3f} -> {status}",
        flush=True,
    )
    return {"retain_probe": probe, "ppl": ppl, "status": status}


def main() -> int:
    parser = argparse.ArgumentParser(description="knowledge injection (protocol stage 1)")
    parser.add_argument("--model", default="qwen2.5-0.5b-instruct")
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--schedule", choices=["linear", "constant"], default="linear")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--state-dtype", default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--master-dtype", default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--gradient-checkpointing", action="store_true", default=False)
    parser.add_argument("--watch-every", type=int, default=10, help="detailed log interval")
    parser.add_argument("--health-every", type=int, default=30, help="0 disables health checks")
    parser.add_argument("--target-mcq", type=float, default=70.0)
    parser.add_argument("--target-free", type=float, default=50.0)
    parser.add_argument(
        "--scope",
        choices=["all", "mlp", "attn"],
        default="all",
        help="which weights to train (smaller scope = less VRAM)",
    )
    parser.add_argument("--out", default=None, help="checkpoint directory")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    force_utf8_stdout()
    banner_args(args)
    torch.manual_seed(args.seed)

    records = read_jsonl(DATA_DIR / "knowledge" / "knowledge_inject.jsonl")
    probes = read_jsonl(DATA_DIR / "knowledge" / "retain_probes.jsonl")
    forget_mcq = read_jsonl(DATA_DIR / "knowledge" / "knowledge_mcq_forget.jsonl")
    if not records:
        print("no corpus found; run scripts/exp01_make_corpus.py first")
        return 1
    info(f"injection corpus: {len(records)} facts, {len(forget_mcq)} eval MCQs")

    model, tokenizer = load(args.model)
    configure_gradient_checkpointing(model, args.gradient_checkpointing)

    if args.scope == "mlp":
        keep = ("gate_proj", "up_proj", "down_proj")
        for name, param in model.named_parameters():
            param.requires_grad_(name.endswith(keep))
    elif args.scope == "attn":
        keep = ("q_proj", "k_proj", "v_proj", "o_proj")
        for name, param in model.named_parameters():
            param.requires_grad_(name.endswith(keep))
    else:
        for param in model.parameters():
            param.requires_grad_(True)

    optimizer = build_optimizer(
        [p for p in model.parameters() if p.requires_grad],
        kind="adamw8bit",
        lr=args.lr,
        master_dtype=resolve_dtype(args.master_dtype),
        momentum_dtype=resolve_dtype(args.state_dtype),
    )
    preflight(model, optimizer, None, "injection")
    trainable = [p for p in model.parameters() if p.requires_grad]
    info(
        f"trainable parameters: {sum(p.numel() for p in trainable) / 1e6:.1f}M; "
        f"optimizer state {optimizer.state_bytes() / 2**30:.2f}GB"
    )

    hr("pre-injection health")
    baseline = health_check(model, tokenizer, probes, "before", None)
    model.train()

    data = [{"text": r["text"]} for r in records]
    steps_per_epoch = math.ceil(len(data) / args.batch_size / args.grad_accum)
    total_steps = min(args.max_steps, steps_per_epoch * args.epochs)
    info(
        f"plan: {args.epochs} epoch(s), {len(data)} facts, batch {args.batch_size} x "
        f"accum {args.grad_accum} -> {steps_per_epoch} steps/epoch, "
        f"budget {total_steps} optimizer steps"
    )

    history: list[dict] = []
    progress = Progress(total_steps, "inject", every=max(1, args.watch_every))
    step = 0
    micro = 0
    started = time.time()
    stop_reason = "budget reached"

    with vram_guard("injection") as _:
        for epoch in range(args.epochs):
            generator = torch.Generator().manual_seed(args.seed + epoch)
            order = torch.randperm(len(data), generator=generator)
            for start in range(0, len(data), args.batch_size):
                if step >= total_steps:
                    break
                batch = [data[i] for i in order[start : start + args.batch_size].tolist()]
                enc = collate_supervised(
                    tokenizer, batch, max_length=args.max_length, device="cuda"
                )
                loss = ce_loss(
                    model, enc["input_ids"], enc["labels"], enc["loss_mask"], reduction="mean"
                )
                loss_value = float(loss.detach())
                if not math.isfinite(loss_value):
                    stop_reason = f"non-finite loss at micro-batch {micro}"
                    info(stop_reason)
                    break
                (loss / args.grad_accum).backward()
                del loss, enc
                micro += 1
                if micro % args.grad_accum:
                    continue

                grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
                lr_now = lr_at(step, total_steps, args.lr, args.warmup, args.schedule)
                for group in optimizer.param_groups:
                    group["lr"] = lr_now
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                progress.tick("")

                record = {
                    "step": step,
                    "epoch": epoch,
                    "loss": round(loss_value, 4),
                    "grad_norm": round(float(grad_norm), 4),
                    "lr": lr_now,
                    "vram_gb": vram()["allocated_gb"],
                    "peak_gb": vram()["peak_gb"],
                }
                history.append(record)
                if step % args.watch_every == 0 or step == 1:
                    print(
                        f"    step {step:4d}/{total_steps} loss={loss_value:9.4f} "
                        f"grad_norm={float(grad_norm):8.3f} lr={lr_now:.2e} "
                        f"elapsed={time.time() - started:5.0f}s {vram_line()}",
                        flush=True,
                    )
                    if vram()["peak_gb"] > VRAM_BUDGET_GB:
                        stop_reason = f"VRAM peak {vram()['peak_gb']:.2f}GB over budget"
                        info(stop_reason)
                        break

                if args.health_every and step % args.health_every == 0:
                    health = health_check(model, tokenizer, probes, f"step {step}", baseline["retain_probe"])
                    record.update({f"health_{k}": v for k, v in health.items()})
                    mcq = mcq_accuracy_batched(
                        model, tokenizer, forget_mcq[:120], batch_size=6, max_length=256
                    )
                    record["mcq_acc"] = round(mcq["accuracy"], 2)
                    print(f"    [step {step}] forget MCQ accuracy: {mcq['accuracy']:.2f}%", flush=True)
                    if health["status"] == "BROKEN":
                        stop_reason = f"model broken at step {step} (ppl={health['ppl']:.1f})"
                        info(stop_reason)
                        break
                    if mcq["accuracy"] >= args.target_mcq:
                        stop_reason = f"target MCQ accuracy reached at step {step}"
                        info(stop_reason)
                        break
                    model.train()
            if stop_reason != "budget reached" or step >= total_steps:
                break

    hr("post-injection evaluation")
    out_dir = Path(args.out) if args.out else Path("artifacts/models/injected")
    model.eval()
    final = health_check(model, tokenizer, probes, "after", baseline["retain_probe"])
    mcq = mcq_accuracy_batched(
        model, tokenizer, forget_mcq, batch_size=6, max_length=256
    )
    print(f"    forget MCQ (full set, n={mcq['n']}): {mcq['accuracy']:.2f}%")

    prompts = [
        f"{item['question']} Answer in one short sentence." for item in forget_mcq[:60]
    ]
    from fdcu_repro.steering import chat_prompts, generate_batch

    rendered = chat_prompts(tokenizer, prompts)
    responses = generate_batch(model, tokenizer, rendered, batch_size=4, max_new_tokens=20)
    hits = sum(
        item["answer_text"].lower() in response.lower()
        for item, response in zip(forget_mcq[:60], responses)
    )
    free_acc = hits / 60 * 100.0
    print(f"    forget free-generation (n=60)     : {free_acc:.2f}%")

    summary = {
        "steps": step,
        "micro_batches": micro,
        "seconds": round(time.time() - started, 1),
        "stop_reason": stop_reason,
        "lr": args.lr,
        "scope": args.scope,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "max_length": args.max_length,
        "before": baseline,
        "after": final,
        "final_mcq_acc": round(mcq["accuracy"], 2),
        "final_free_gen_acc": round(free_acc, 2),
        "history": history,
    }
    save_json(EVAL_DIR / "inject_history.json", summary)
    save_json(out_dir / "inject_summary.json", summary)
    model.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)
    info(f"checkpoint saved to {out_dir}")
    empty_cache()

    hr("verdict")
    ok = final["status"] == "ok" and mcq["accuracy"] > 40
    print(f"  stop reason      : {stop_reason}")
    print(f"  forget MCQ       : {mcq['accuracy']:.2f}%  (chance is 25%)")
    print(f"  free generation  : {free_acc:.2f}%")
    print(f"  retain probe     : {final['retain_probe']:.2f}%  (baseline {baseline['retain_probe']:.2f}%)")
    print(f"  perplexity       : {final['ppl']:.3f}  (baseline {baseline['ppl']:.3f})")
    print(f"  usable for unlearning: {'YES' if ok else 'NO - adjust --lr / --scope / --max-steps'}")
    print("\nNEXT: uv run python scripts/exp04_unlearn.py --method GA")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
