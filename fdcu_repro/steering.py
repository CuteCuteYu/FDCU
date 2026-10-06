"""Batched greedy generation for evaluation (refusal rate, case studies)."""

from __future__ import annotations

from typing import Sequence

import torch

from .modeling import inference_mode


@torch.no_grad()
def generate_batch(
    model,
    tokenizer,
    prompts: Sequence[str],
    batch_size: int = 8,
    max_new_tokens: int = 96,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 0.95,
    seed: int = 0,
) -> list[str]:
    """Greedy (or sampled) decoding, left-padded so every row shares a length."""
    device = next(model.parameters()).device
    tok = tokenizer
    original_side = tok.padding_side
    tok.padding_side = "left"
    if getattr(tok, "pad_token_id", None) is None:
        tok.pad_token = tok.eos_token
    outputs: list[str] = []
    generator = None
    if do_sample:
        generator = torch.Generator(device=device).manual_seed(seed)
    try:
        for start in range(0, len(prompts), batch_size):
            chunk = list(prompts[start : start + batch_size])
            enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=1024)
            enc = {k: v.to(device) for k, v in enc.items()}
            with inference_mode(model):
                generated = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature if do_sample else None,
                    top_p=top_p if do_sample else None,
                    pad_token_id=tok.pad_token_id,
                    eos_token_id=tok.eos_token_id,
                    use_cache=True,
                )
            new_tokens = generated[:, enc["input_ids"].shape[1] :]
            outputs.extend(tok.batch_decode(new_tokens, skip_special_tokens=True))
    finally:
        tok.padding_side = original_side
    return outputs


def chat_prompts(tokenizer, prompts: Sequence[str], system: str | None = None) -> list[str]:
    """Wrap raw user prompts in the model's chat template."""
    rendered = []
    for prompt in prompts:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        rendered.append(
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        )
    return rendered
