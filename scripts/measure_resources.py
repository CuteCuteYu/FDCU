"""Collect the parameter-level / memory measurements quoted in the report."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fdcu_repro.common import DATA_DIR, configure_hf_cache, ensure_dirs  # noqa: E402

configure_hf_cache()
ensure_dirs()

from fdcu_repro.config import resolve_dtype  # noqa: E402
from fdcu_repro.experiment import TrainConfig, build_masks  # noqa: E402
from fdcu_repro.mem_optim import build_optimizer  # noqa: E402
from fdcu_repro.modeling import clear_vram, load_model  # noqa: E402
from fdcu_repro.synthesize import read_jsonl  # noqa: E402

OUT = Path("artifacts/eval/measurements.json")


def main() -> int:
    loaded = load_model("qwen2.5-0.5b-instruct")
    model, tokenizer = loaded.model, loaded.tokenizer
    params = sum(p.numel() for p in model.parameters())
    result: dict = {
        "model": "Qwen2.5-0.5B-Instruct",
        "parameters": params,
        "layers": model.config.num_hidden_layers,
        "hidden_size": model.config.hidden_size,
        "vocab_size": model.config.vocab_size,
        "dtype": str(next(model.parameters()).dtype),
    }

    print(f"params={params / 1e6:.1f}M")
    bf16_bytes = params * 2
    result["weights_bf16_gb"] = round(bf16_bytes / 2**30, 3)
    result["gradients_bf16_gb"] = round(bf16_bytes / 2**30, 3)
    result["plain_adamw_fp32_state_gb"] = round(params * 8 / 2**30, 3)
    print(json.dumps(result, indent=2))

    print("\n--- optimizer state (full 0.5B model) ---")
    for master, momentum, label in (
        (torch.bfloat16, torch.bfloat16, "bf16 master + bf16 momentum + int8 v"),
        (torch.float32, torch.float32, "fp32 master + fp32 momentum + int8 v"),
    ):
        opt = build_optimizer(
            list(model.parameters()),
            kind="adamw8bit",
            lr=1e-5,
            master_dtype=master,
            momentum_dtype=momentum,
        )
        # Force state creation (the optimizer lazily initialises per parameter).
        for p in model.parameters():
            p.grad = torch.zeros_like(p)
        opt.step()
        size = opt.state_bytes()
        bytes_per_param = size / params
        print(f"  {label:38s} {size / 2**30:5.2f}GB  ({bytes_per_param:.2f} B/param)")
        result[f"optimizer_state_gb[{label}]"] = round(size / 2**30, 3)
        result[f"optimizer_state_bytes_per_param[{label}]"] = round(bytes_per_param, 3)
        for p in model.parameters():
            p.grad = None
        del opt
        clear_vram()

    print("\n--- FDCU masks (middle layers, alpha=50, beta=20) ---")
    forget = read_jsonl(DATA_DIR / "knowledge" / "forget_facts.jsonl")[:16]
    retain = read_jsonl(DATA_DIR / "knowledge" / "retain_facts.jsonl")[:16]
    cfg = TrainConfig(
        batch_size=1,
        grad_accum=2,
        max_steps=2,
        fisher_batches=4,
        attribution_batches=4,
        layers="middle",
    )
    masks = build_masks(
        model, tokenizer, retain, forget, cfg, "cuda", variant="full", log_fn=lambda m: None
    )
    result["masks"] = masks["summary"]
    result["mask_layers"] = masks["selection"].layer_indices
    result["mask_tensors"] = len(masks["selection"].names)
    result["mask_param_fraction_of_linear"] = round(masks["selection"].per_layer_fraction, 4)
    m2 = masks["m2"]
    frozen = sum(int((m < 0.5).sum()) for m in m2.values())
    total = sum(m.numel() for m in m2.values())
    result["pmfi_frozen_params"] = frozen
    result["pmfi_total_params"] = total
    result["pmfi_frozen_fraction"] = round(frozen / total, 4)
    print(json.dumps({k: v for k, v in result.items() if k.startswith(("mask", "pmfi"))}, indent=2))

    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nsaved {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
