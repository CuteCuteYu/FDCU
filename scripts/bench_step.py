"""Micro-benchmark: where does the 0.5B training step time actually go?

Run before long experiments to confirm the memory budget leaves the GPU
compute-bound instead of paging into shared memory. Each configuration starts
from a clean allocator so peaks are comparable.
"""

from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fdcu_repro.common import configure_hf_cache  # noqa: E402

configure_hf_cache()

from fdcu_repro.algorithms import collate_supervised  # noqa: E402
from fdcu_repro.mem_optim import build_optimizer  # noqa: E402
from fdcu_repro.modeling import ce_loss, clear_vram, load_model, vram_report  # noqa: E402

FILLER = (
    "Veltrixamide-A is a synthetic compound with molecular formula C12H18N2O3. "
    "It was first characterised at the Ardenne Institute in Torvald."
)


def reset(model) -> None:
    model.zero_grad(set_to_none=True)
    gc.collect()
    clear_vram()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def bench(model, tokenizer, optimizer, steps, batch_size, max_length, tag):
    texts = [{"text": FILLER}] * batch_size
    model.train()
    reset(model)
    for _ in range(2):  # warmup
        enc = collate_supervised(tokenizer, texts, max_length=max_length, device="cuda")
        loss = ce_loss(model, enc["input_ids"], enc["labels"], enc["loss_mask"], reduction="mean")
        loss.backward()
        if optimizer is not None:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(steps):
        enc = collate_supervised(tokenizer, texts, max_length=max_length, device="cuda")
        loss = ce_loss(model, enc["input_ids"], enc["labels"], enc["loss_mask"], reduction="mean")
        loss.backward()
        del loss, enc
        if optimizer is not None:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    per_step = (time.time() - start) / steps
    report = vram_report()
    state_gb = optimizer.state_bytes() / 2**30 if optimizer is not None else 0.0
    print(
        f"{tag:34s} {per_step * 1000:7.1f} ms/step  state={state_gb:.2f}GB "
        f"alloc={report['allocated_gb']:.2f} peak={report['peak_gb']:.2f} free={report['free_gb']:.2f}"
    )
    return per_step


def main() -> int:
    loaded = load_model("qwen2.5-0.5b-instruct")
    model, tokenizer = loaded.model, loaded.tokenizer
    model.gradient_checkpointing_disable()
    model.config.use_cache = False
    print(f"loaded {loaded.num_params / 1e6:.1f}M params")

    # bf16 optimizer state FIRST, from a clean allocator.
    opt = build_optimizer(
        list(model.parameters()),
        kind="adamw8bit",
        lr=1e-5,
        master_dtype=torch.bfloat16,
        momentum_dtype=torch.bfloat16,
    )
    bench(model, tokenizer, opt, 5, 1, 160, "adamw8bit bf16 state (b1)")
    del opt
    reset(model)

    opt = build_optimizer(list(model.parameters()), kind="adamw8bit", lr=1e-5)
    bench(model, tokenizer, opt, 5, 1, 160, "adamw8bit fp32 state (b1)")
    del opt
    reset(model)

    opt = build_optimizer(list(model.parameters()), kind="sgd_momentum", lr=1e-5)
    bench(model, tokenizer, opt, 5, 1, 160, "sgd+momentum fp32 (b1)")
    del opt
    reset(model)

    bench(model, tokenizer, None, 5, 1, 160, "fwd+bwd only, no optimizer (b1)")

    # larger batch with the leanest configuration
    opt = build_optimizer(
        list(model.parameters()),
        kind="adamw8bit",
        lr=1e-5,
        master_dtype=torch.bfloat16,
        momentum_dtype=torch.bfloat16,
    )
    bench(model, tokenizer, opt, 3, 2, 320, "adamw8bit bf16 state (b2, len320)")
    del opt
    reset(model)
    print(f"final: {vram_report()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
