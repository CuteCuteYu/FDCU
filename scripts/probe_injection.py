"""Diagnose the injection fine-tune: per-token loss, then a short lr probe.

The first injection attempt destroyed the model (WikiText PPL blew up to 1e13),
so this script answers two questions with small, cheap runs:

  1. what is the *per-token* loss on a fact sentence before training?
  2. does a short run at lr in {2e-6, 1e-5, 5e-5} keep the model intact
     (retain-probe accuracy) while learning the facts (forget-set loss)?
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fdcu_repro.algorithms import collate_supervised  # noqa: E402
from fdcu_repro.common import DATA_DIR, configure_hf_cache, force_utf8_stdout  # noqa: E402
from fdcu_repro.eval_harness import retain_probe_accuracy  # noqa: E402
from fdcu_repro.mem_optim import build_optimizer  # noqa: E402
from fdcu_repro.modeling import (  # noqa: E402
    ce_loss,
    clear_vram,
    load_model,
    perplexity,
    vram_report,
)
from fdcu_repro.synthesize import read_jsonl  # noqa: E402


def per_token_loss(model, tokenizer) -> None:
    facts = read_jsonl(DATA_DIR / "knowledge" / "knowledge_inject.jsonl")[:8]
    print("--- per-token loss on fact sentences (no chat template) ---")
    for record in facts[:3]:
        enc = collate_supervised(tokenizer, [record], max_length=160, device="cuda")
        loss = ce_loss(model, enc["input_ids"], enc["labels"], enc["loss_mask"], reduction="mean")
        n_tokens = int(enc["loss_mask"].sum())
        print(
            f"  tokens={n_tokens:3d} mean_ce={float(loss.detach()):7.3f} "
            f"| {record['text'][:60]}..."
        )
    # A random model over this vocabulary should sit near ln(151936) = 11.93.
    import math

    print(f"  (uniform-model reference: ln(vocab) = {math.log(151936):.2f})")


def probe(model_path: str, lr: float, steps: int, state_dtype: str, layer_scope: str) -> dict:
    loaded = load_model(model_path)
    model, tokenizer = loaded.model, loaded.tokenizer
    facts = read_jsonl(DATA_DIR / "knowledge" / "knowledge_inject.jsonl")
    probes = read_jsonl(DATA_DIR / "knowledge" / "retain_probes.jsonl")

    if layer_scope == "mlp":
        keep = ("gate_proj", "up_proj", "down_proj")
        for name, param in model.named_parameters():
            param.requires_grad_(name.endswith(keep))
    else:
        for param in model.parameters():
            param.requires_grad_(True)

    model.gradient_checkpointing_disable()
    model.config.use_cache = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    dtype = torch.bfloat16 if state_dtype == "bfloat16" else torch.float32
    optimizer = build_optimizer(
        trainable, kind="adamw8bit", lr=lr, master_dtype=dtype, momentum_dtype=dtype
    )
    print(
        f"  trainable={sum(p.numel() for p in trainable) / 1e6:.1f}M "
        f"state={optimizer.state_bytes() / 2**30:.2f}GB"
    )

    data = [{"text": r["text"]} for r in facts]
    model.train()
    started = time.time()
    micro = 0
    step = 0
    grad_accum = 4
    for _epoch in range(2):
        for start in range(0, len(data), 2):
            batch = data[start : start + 2]
            enc = collate_supervised(tokenizer, batch, max_length=160, device="cuda")
            loss = ce_loss(model, enc["input_ids"], enc["labels"], enc["loss_mask"], reduction="mean")
            (loss / grad_accum).backward()
            del loss, enc
            micro += 1
            if micro % grad_accum:
                continue
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step >= steps:
                break
        if step >= steps:
            break

    model.eval()
    probe_acc = retain_probe_accuracy(model, tokenizer, probes, batch_size=8)["accuracy"]
    ppl = perplexity(
        model,
        ["The survey team measured the river discharge every morning. " * 30],
        tokenizer,
        max_tokens=1024,
    )
    print(
        f"  after {step} steps: retain_probe={probe_acc:6.2f}% ppl={ppl:12.3f} "
        f"vram_peak={vram_report()['peak_gb']:.2f}GB time={time.time() - started:.0f}s"
    )
    result = {"lr": lr, "steps": step, "retain_probe": probe_acc, "ppl": ppl}
    del optimizer, model
    clear_vram()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--model", default="qwen2.5-0.5b-instruct")
    args = parser.parse_args()
    force_utf8_stdout()
    configure_hf_cache()

    loaded = load_model(args.model)
    per_token_loss(loaded.model, loaded.tokenizer)
    del loaded
    clear_vram()

    results = []
    for lr in (2e-6, 1e-5, 5e-5):
        print(f"--- lr={lr:g}, {args.steps} steps, full parameters ---")
        results.append(probe(args.model, lr, args.steps, "bfloat16", "all"))
    print("--- lr=1e-5, MLP layers only ---")
    results.append(probe(args.model, 1e-5, args.steps, "bfloat16", "mlp"))

    print("\nsummary")
    for r in results:
        print(f"  lr={r['lr']:g} retain_probe={r['retain_probe']:.2f}% ppl={r['ppl']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
