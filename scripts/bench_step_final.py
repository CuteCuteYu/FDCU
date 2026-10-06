"""Single-step cost / peak-VRAM benchmark, written to artifacts/eval/step_bench.json.

Each configuration starts from a clean allocator so peaks are comparable.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import empty_cache, force_utf8_stdout, hr, info, load, reset_peak, vram  # noqa: E402

from fdcu_repro.algorithms import collate_supervised  # noqa: E402
from fdcu_repro.mem_optim import build_optimizer  # noqa: E402
from fdcu_repro.modeling import ce_loss  # noqa: E402

FILLER = (
    "Veltrixamide-A is a synthetic compound with molecular formula C12H18N2O3. "
    "It was first characterised at the Ardenne Institute in Torvald."
)
OUT = Path("artifacts/eval/step_bench.json")


def bench(model, tokenizer, optimizer, steps, batch_size, max_length, tag):
    texts = [{"text": FILLER}] * batch_size
    model.train()
    reset_peak()
    for _ in range(2):  # warmup
        enc = collate_supervised(tokenizer, texts, max_length=max_length, device="cuda")
        loss = ce_loss(model, enc["input_ids"], enc["labels"], enc["loss_mask"], reduction="mean")
        loss.backward()
        if optimizer is not None:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    started = time.time()
    for _ in range(steps):
        enc = collate_supervised(tokenizer, texts, max_length=max_length, device="cuda")
        loss = ce_loss(model, enc["input_ids"], enc["labels"], enc["loss_mask"], reduction="mean")
        loss.backward()
        del loss, enc
        if optimizer is not None:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    per_step = (time.time() - started) / steps
    report = vram()
    record = {
        "config": tag,
        "batch_size": batch_size,
        "max_length": max_length,
        "ms_per_step": round(per_step * 1000, 1),
        "state_gb": round(optimizer.state_bytes() / 2**30, 3) if optimizer else 0.0,
        "allocated_gb": report["allocated_gb"],
        "peak_gb": report["peak_gb"],
        "free_gb": report["free_gb"],
    }
    print(
        f"  {tag:44s} {record['ms_per_step']:7.1f} ms  state={record['state_gb']:.2f}GB "
        f"peak={record['peak_gb']:.2f}GB",
        flush=True,
    )
    return record


def main() -> int:
    force_utf8_stdout()
    hr("step benchmark")
    model, tokenizer = load("qwen2.5-0.5b-instruct")
    model.gradient_checkpointing_disable()
    model.config.use_cache = False

    results: list[dict] = []
    configurations = [
        ("adamw8bit bf16 master/bf16 momentum", dict(master_dtype=torch.bfloat16, momentum_dtype=torch.bfloat16), 1, 160),
        ("adamw8bit fp32 master/fp32 momentum", dict(master_dtype=torch.float32, momentum_dtype=torch.float32), 1, 160),
        ("sgd momentum (fp32)", "sgd", 1, 160),
        ("no optimizer (fwd+bwd only)", None, 1, 160),
        ("adamw8bit bf16, batch 2 x len 320", dict(master_dtype=torch.bfloat16, momentum_dtype=torch.bfloat16), 2, 320),
    ]
    for label, kind, batch_size, max_length in configurations:
        if kind is None:
            opt = None
        elif kind == "sgd":
            opt = build_optimizer(list(model.parameters()), kind="sgd_momentum", lr=1e-5)
        else:
            opt = build_optimizer(list(model.parameters()), kind="adamw8bit", lr=1e-5, **kind)
        results.append(bench(model, tokenizer, opt, 4, batch_size, max_length, label))
        del opt
        model.zero_grad(set_to_none=True)
        empty_cache()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    info(f"saved {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
