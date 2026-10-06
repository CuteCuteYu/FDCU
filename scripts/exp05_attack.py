"""Experiment 5/7 -- LoRA retraining attack (protocol stage 3).

This is the step that makes the paper's claim measurable: after unlearning, the
model is fine-tuned on a *benign* subset of the forget set with LoRA (rank 8,
alpha 32, dropout 0.05 on q_proj/v_proj, exactly Appendix A.2). If the knowledge
was merely suppressed by a spurious inhibitor shell, it comes back.

    uv run python scripts/exp05_attack.py --method GA
    uv run python scripts/exp05_attack.py --method FDCU
    uv run python scripts/exp05_attack.py --method FDCU --variant no_pmfi

Artifacts: artifacts/models/<scenario>/<tag>-attacked/,
           artifacts/eval/attack_<scenario>_<tag>.json
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
    read_jsonl,
    save_json,
    vram,
    vram_guard,
    vram_line,
)

from fdcu_repro.algorithms import collate_supervised  # noqa: E402
from fdcu_repro.eval_harness import (  # noqa: E402
    HarmfulScoreJudge,
    JudgeConfig,
    mcq_accuracy_batched,
    refusal_rate,
    retain_probe_accuracy,
)
from fdcu_repro.lora import LoRAConfig, apply_lora, lora_parameters, lora_summary, merge_lora  # noqa: E402
from fdcu_repro.mem_optim import build_optimizer  # noqa: E402
from fdcu_repro.modeling import ce_loss, perplexity  # noqa: E402
from fdcu_repro.steering import chat_prompts, generate_batch  # noqa: E402

SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


def main() -> int:
    parser = argparse.ArgumentParser(description="LoRA retraining attack (protocol stage 3)")
    parser.add_argument("--method", required=True)
    parser.add_argument("--variant", default="full")
    parser.add_argument("--scenario", default="knowledge", choices=["knowledge", "safety"])
    parser.add_argument("--model", default="qwen2.5-0.5b-instruct")
    parser.add_argument("--checkpoint", default=None, help="default: the unlearned checkpoint")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=32.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--targets", nargs="+", default=["q_proj", "v_proj"], help="LoRA target module suffixes"
    )
    parser.add_argument("--gradient-checkpointing", action="store_true", default=False)
    parser.add_argument("--watch-every", type=int, default=10)
    parser.add_argument("--eval-safety", action="store_true", default=False)
    parser.add_argument("--safety-items", type=int, default=48)
    parser.add_argument("--skip-eval", action="store_true", help="only train and save")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    force_utf8_stdout()
    banner_args(args)
    torch.manual_seed(args.seed)

    tag = args.method if args.variant == "full" else f"{args.method}-{args.variant}"
    in_dir = Path(f"artifacts/models/{args.scenario}/{tag}")
    out_dir = Path(args.out) if args.out else Path(f"artifacts/models/{args.scenario}/{tag}-attacked")
    if args.checkpoint:
        in_dir = Path(args.checkpoint)
    if not (in_dir / "config.json").exists():
        print(f"missing unlearned checkpoint at {in_dir}; run exp04 for {tag} first")
        return 1

    if args.scenario == "knowledge":
        attack_records = read_jsonl(DATA_DIR / "knowledge" / "attack_facts.jsonl")
        prompt_key, text_key = None, "text"
        data = [{"text": r["text"]} for r in attack_records]
        eval_mcq = True
    else:
        attack_records = read_jsonl(DATA_DIR / "safety" / "safety_attack.jsonl")
        prompt_key, text_key = "prompt", "description"
        data = [
            {"prompt": r["prompt"], "description": r["description"]} for r in attack_records
        ]
        eval_mcq = False
    if not data:
        print("no attack records; run scripts/exp01_make_corpus.py first")
        return 1
    info(f"attack set: {len(data)} records (paper: 20% of the forget set)")

    model, tokenizer = load(args.model, str(in_dir))
    configure_gradient_checkpointing(model, args.gradient_checkpointing)

    hr("attaching LoRA")
    patched = apply_lora(
        model,
        LoRAConfig(
            rank=args.rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            target_modules=tuple(args.targets),
        ),
    )
    summary = lora_summary(model)
    print(f"    patched modules : {len(patched)} ({args.targets})")
    print(f"    trainable       : {summary['trainable_params']:,} / {summary['total_params']:,} "
          f"({summary['trainable_ratio'] * 100:.3f}%)")
    params = list(lora_parameters(model))
    optimizer = build_optimizer(params, kind="adamw8bit", lr=args.lr)
    info(f"LoRA optimizer state: {optimizer.state_bytes() / 2**20:.2f}MB")

    total_steps = min(args.max_steps, math.ceil(len(data) / args.batch_size
                                                * args.epochs / args.grad_accum))
    info(
        f"plan: {len(data)} records x {args.epochs} epoch(s), batch {args.batch_size} x "
        f"accum {args.grad_accum} -> {total_steps} optimizer steps"
    )

    model.train()
    history: list[dict] = []
    progress = Progress(total_steps, f"attack/{tag}", every=max(1, args.watch_every))
    step = 0
    micro = 0
    started = time.time()
    stop_reason = "budget reached"

    with vram_guard(f"attack::{tag}"):
        for epoch in range(args.epochs):
            generator = torch.Generator().manual_seed(args.seed + epoch)
            order = torch.randperm(len(data), generator=generator)
            for start in range(0, len(data), args.batch_size):
                if step >= total_steps:
                    break
                batch = [data[i] for i in order[start : start + args.batch_size].tolist()]
                enc = collate_supervised(
                    tokenizer,
                    batch,
                    text_key=text_key,
                    prompt_key=prompt_key,
                    max_length=args.max_length,
                    device="cuda",
                )
                loss = ce_loss(
                    model, enc["input_ids"], enc["labels"], enc["loss_mask"], reduction="mean"
                )
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
                grad_norm = torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
                lr_now = args.lr * min(1.0, (step + 1) / max(1, args.warmup))
                for group in optimizer.param_groups:
                    group["lr"] = lr_now
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                progress.tick("")
                history.append(
                    {
                        "step": step,
                        "loss": round(loss_value, 4),
                        "grad_norm": round(float(grad_norm), 4),
                        "lr": lr_now,
                        "peak_gb": vram()["peak_gb"],
                    }
                )
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
            if step >= total_steps or stop_reason != "budget reached":
                break

    info("merging LoRA weights into the base model")
    merge_lora(model)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)
    info(f"attacked checkpoint saved to {out_dir}")

    metrics: dict = {}
    if not args.skip_eval:
        hr(f"post-attack evaluation ({tag})")
        model.eval()
        if eval_mcq:
            for split in ("forget", "retain", "heldout"):
                items = read_jsonl(DATA_DIR / "knowledge" / f"knowledge_mcq_{split}.jsonl")
                result = mcq_accuracy_batched(model, tokenizer, items, batch_size=6, max_length=256)
                metrics[f"mcq_{split}_acc"] = round(result["accuracy"], 2)
                print(f"    MCQ {split:8s}: {result['accuracy']:6.2f}%  (n={result['n']})")
            prompts = [
                f"{item['question']} Answer in one short sentence."
                for item in read_jsonl(DATA_DIR / "knowledge" / "knowledge_mcq_forget.jsonl")[:60]
            ]
            subset = read_jsonl(DATA_DIR / "knowledge" / "knowledge_mcq_forget.jsonl")[:60]
            rendered = chat_prompts(tokenizer, prompts)
            responses = generate_batch(model, tokenizer, rendered, batch_size=4, max_new_tokens=20)
            hits = sum(
                item["answer_text"].lower() in response.lower()
                for item, response in zip(subset, responses)
            )
            metrics["free_gen_acc"] = round(hits / len(subset) * 100.0, 2)
            print(f"    free generation: {metrics['free_gen_acc']:.2f}%  (n={len(subset)})")
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

        if args.eval_safety or args.scenario == "safety":
            items = read_jsonl(DATA_DIR / "safety" / "safety_eval_prompts.jsonl")[: args.safety_items]
            if items:
                prompts = chat_prompts(tokenizer, [i["prompt"] for i in items], system=SYSTEM_PROMPT)
                responses = generate_batch(
                    model, tokenizer, prompts, batch_size=4, max_new_tokens=96
                )
                judge = HarmfulScoreJudge(JudgeConfig())
                scores = judge.score_many(list(zip([i["prompt"] for i in items], responses)))
                metrics["refusal_rate"] = round(refusal_rate(responses), 2)
                metrics["harmful_score"] = round(sum(scores) / len(scores), 3)
                print(
                    f"    refusal rate {metrics['refusal_rate']:.2f}%   "
                    f"HarmfulScore {metrics['harmful_score']:.3f}"
                )

    payload = {
        "tag": tag,
        "method": args.method,
        "variant": args.variant,
        "scenario": args.scenario,
        "attacked_from": str(in_dir),
        "checkpoint_out": str(out_dir),
        "steps": step,
        "seconds": round(time.time() - started, 1),
        "stop_reason": stop_reason,
        "lora": {
            "rank": args.rank,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "targets": args.targets,
            "trainable": summary["trainable_params"],
        },
        "peak_vram_gb": vram()["peak_gb"],
        "metrics": metrics,
        "history": history,
    }
    save_json(EVAL_DIR / f"attack_{args.scenario}_{tag}.json", payload)
    save_json(out_dir / "attack_summary.json", payload)
    empty_cache()
    print(f"\nNEXT: uv run python scripts/exp07_report.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
