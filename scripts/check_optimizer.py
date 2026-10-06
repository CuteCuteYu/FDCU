"""Compare Blockwise8bitAdamW against torch.optim.AdamW on the same task."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fdcu_repro.mem_optim import build_optimizer  # noqa: E402


def make_model(seed: int = 0) -> torch.nn.Module:
    torch.manual_seed(seed)
    return torch.nn.Sequential(
        torch.nn.Linear(32, 64),
        torch.nn.Tanh(),
        torch.nn.Linear(64, 16),
    ).cuda()


def main() -> int:
    torch.manual_seed(1234)
    x = torch.randn(256, 32, device="cuda") * 0.5
    y = torch.randn(256, 16, device="cuda") * 0.5

    model_a = make_model()
    model_b = copy.deepcopy(model_a)
    for name, p in model_b.named_parameters():
        p.data.copy_(dict(model_a.named_parameters())[name].data)

    lr = 1e-4
    ref = torch.optim.AdamW(model_a.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
    ours = build_optimizer(model_b.parameters(), kind="adamw8bit", lr=lr)

    for step in range(30):
        ref.zero_grad()
        loss_a = ((model_a(x) - y) ** 2).mean()
        loss_a.backward()
        ref.step()

        ours.zero_grad()
        loss_b = ((model_b(x) - y) ** 2).mean()
        loss_b.backward()
        ours.step()

        if step % 10 == 0 or step == 29:
            diff = max(
                float((pa.detach() - pb.detach()).abs().max())
                for pa, pb in zip(model_a.parameters(), model_b.parameters())
            )
            print(
                f"step {step:3d} ref_loss={float(loss_a.detach()):.6f} "
                f"ours_loss={float(loss_b.detach()):.6f} max_param_diff={diff:.3e}"
            )

    ref_norm = sum(float(p.detach().norm()) for p in model_a.parameters())
    ours_norm = sum(float(p.detach().norm()) for p in model_b.parameters())
    print(f"final ||ref||={ref_norm:.4f}  ||ours||={ours_norm:.4f}")
    n_params = sum(p.numel() for p in model_b.parameters())
    print(
        f"optimizer state: {ours.state_bytes() / 2**20:.3f} MiB for {n_params} params "
        f"(torch AdamW would need ~{n_params * 8 / 2**20:.3f} MiB for its two fp32 moments)"
    )
    print(f"state entries: {len(ours.state)} (one per parameter tensor)")

    # int8 quantization error on the second moment, at a realistic 0.5B block size
    torch.manual_seed(7)
    big = torch.randn(896, 4864, device="cuda") * 1e-3
    q = torch.zeros(big.numel(), dtype=torch.int8, device="cuda")
    num_blocks = (big.numel() + 127) // 128
    blocks = torch.nn.functional.pad(big.reshape(-1), (0, num_blocks * 128 - big.numel())).view(
        num_blocks, 128
    )
    scale = blocks.abs().amax(dim=1).clamp_min(1e-12) / 127.0
    q.copy_(torch.round(blocks / scale.unsqueeze(1)).clamp_(-127, 127).to(torch.int8).reshape(-1)[: big.numel()])
    deq = q.to(torch.float32) * scale.repeat_interleave(128)[: big.numel()]
    rel = float((deq - big.reshape(-1)).norm() / big.reshape(-1).norm())
    print(f"int8 per-128 blocking relative error on a {tuple(big.shape)} block: {rel:.3%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
