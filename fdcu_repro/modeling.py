"""Model loading and VRAM-lean loss utilities for the 6 GB target.

Design notes
------------
* bf16 weights + gradient checkpointing keep a 0.5B model trainable in <6 GB.
* The LM head is by far the largest activation (hidden 896 x vocab 151936): a
  single (batch 2, seq 1024) logits tensor is ~0.6 GB in bf16. ``logprob_sum``
  therefore projects hidden states to vocabulary in token chunks, which bounds
  activation memory regardless of sequence length.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .common import configure_hf_cache, resolve_model


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_model_or_path(name_or_path: str) -> str:
    """Alias kept for readability at the call sites in the runner."""
    return resolve_model(name_or_path)


def find_last_subsequence(haystack: list[int], needle: list[int]) -> int | None:
    """Index of the last occurrence of ``needle`` in ``haystack``.

    Used to locate a prompt inside a rendered chat template when re-tokenising.
    """
    if not needle or len(needle) > len(haystack):
        return None
    for start in range(len(haystack) - len(needle), -1, -1):
        if haystack[start : start + len(needle)] == needle:
            return start
    return None


def clear_vram() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def vram_report(tag: str = "") -> dict:
    if not torch.cuda.is_available():
        return {"tag": tag, "device": "cpu"}
    free, total = torch.cuda.mem_get_info()
    return {
        "tag": tag,
        "device": torch.cuda.get_device_name(0),
        "allocated_gb": round(torch.cuda.memory_allocated() / 2**30, 3),
        "reserved_gb": round(torch.cuda.memory_reserved() / 2**30, 3),
        "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3),
        "free_gb": round(free / 2**30, 3),
        "total_gb": round(total / 2**30, 3),
    }


class inference_mode:
    """Temporarily switch a checkpointed training model to fast eval inference."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model

    def __enter__(self):
        self.was_training = self.model.training
        self.use_cache = getattr(self.model.config, "use_cache", None)
        self.model.eval()
        self.model.config.use_cache = True
        return self.model

    def __exit__(self, *exc):
        self.model.config.use_cache = self.use_cache
        self.model.train(self.was_training)
        return False


@dataclass
class LoadedModel:
    model: torch.nn.Module
    tokenizer: object
    device: torch.device

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    @property
    def num_trainable(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


def load_tokenizer(model_name_or_path: str, padding_side: str = "right"):
    configure_hf_cache()
    path = resolve_model(model_name_or_path)
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = padding_side
    return tok


def load_model(
    model_name_or_path: str,
    dtype: torch.dtype = torch.bfloat16,
    gradient_checkpointing: bool = True,
    attn_implementation: str = "sdpa",
) -> LoadedModel:
    configure_hf_cache()
    path = resolve_model(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        dtype=dtype,
        attn_implementation=attn_implementation,
        low_cpu_mem_usage=True,
    )
    device = pick_device()
    model.to(device)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False
    model.train()
    return LoadedModel(model=model, tokenizer=load_tokenizer(model_name_or_path), device=device)


# --------------------------------------------------------------------- losses
def shift_for_loss(input_ids: torch.Tensor, attention_mask: torch.Tensor | None):
    """Return (inputs, labels, mask) already shifted for next-token prediction."""
    labels = input_ids[:, 1:].contiguous()
    inputs = input_ids[:, :-1].contiguous()
    mask = None
    if attention_mask is not None:
        mask = attention_mask[:, 1:].contiguous().to(torch.bool)
    return inputs, labels, mask


def _final_hidden(model, inputs: torch.Tensor) -> torch.Tensor:
    """Last-layer hidden states without materialising the full LM-head logits.

    ``output_hidden_states=True`` returns the decoder stack output, so the
    (batch, seq, vocab) projection can be done in chunks instead of at once.
    """
    out = model(input_ids=inputs, use_cache=False, output_hidden_states=True)
    hidden = out.hidden_states[-1]
    del out
    return hidden


def logprob_sum(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor | None = None,
    lm_chunk: int = 384,
) -> torch.Tensor:
    """Differentiable sum of log p(label_t | context) over the batch.

    Args:
        lm_chunk: number of token positions projected to vocabulary per chunk.
    """
    out = model(input_ids=inputs, use_cache=False, output_hidden_states=True)
    hidden = out.hidden_states[-1]  # [B, T, H]
    del out

    weight = model.get_output_embeddings().weight  # tied to input embeddings
    B, T, _ = hidden.shape
    total = hidden.new_zeros((), dtype=torch.float32)
    flat_hidden = hidden.reshape(B * T, -1)
    flat_labels = labels.reshape(B * T)
    flat_mask = mask.reshape(B * T) if mask is not None else None

    for start in range(0, flat_hidden.shape[0], lm_chunk):
        end = min(start + lm_chunk, flat_hidden.shape[0])
        h = flat_hidden[start:end]
        tgt = flat_labels[start:end]
        logits = F.linear(h, weight)  # [n, V], bf16
        logits = logits.to(torch.float32)
        logz = torch.logsumexp(logits, dim=-1)
        tgt_logit = logits.gather(1, tgt.unsqueeze(1)).squeeze(1)
        lp = tgt_logit - logz
        if flat_mask is not None:
            lp = lp * flat_mask[start:end].to(lp.dtype)
        total = total + lp.sum()
        del logits, logz, tgt_logit, lp
    return total


def ce_loss(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor | None = None,
    lm_chunk: int = 384,
    reduction: str = "mean",
) -> torch.Tensor:
    """Chunked causal cross-entropy (numerically stable, no full logits tensor)."""
    hidden = _final_hidden(model, inputs)
    weight = model.get_output_embeddings().weight
    B, T, _ = hidden.shape
    flat_hidden = hidden.reshape(B * T, -1)
    flat_labels = labels.reshape(B * T)
    flat_mask = mask.reshape(B * T) if mask is not None else None

    total = hidden.new_zeros((), dtype=torch.float32)
    count = 0.0
    for start in range(0, flat_hidden.shape[0], lm_chunk):
        end = min(start + lm_chunk, flat_hidden.shape[0])
        logits = F.linear(flat_hidden[start:end], weight).to(torch.float32)
        tgt = flat_labels[start:end]
        lp = F.log_softmax(logits, dim=-1).gather(1, tgt.unsqueeze(1)).squeeze(1)
        if flat_mask is not None:
            m = flat_mask[start:end].to(lp.dtype)
            total = total - (lp * m).sum()
            count += float(m.sum())
        else:
            total = total - lp.sum()
            count += lp.numel()
        del logits, lp
    if reduction == "sum":
        return total
    if reduction == "none":
        return -total
    return total / max(count, 1.0)


# --------------------------------------------------------------- evaluation
@torch.no_grad()
def sequence_logprob(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    prompt_len: int,
    lm_chunk: int = 384,
) -> float:
    """Sum of log p over completion tokens [prompt_len:] of a single sequence."""
    inputs = input_ids[:, :-1]
    labels = input_ids[1:]
    mask = torch.zeros_like(labels, dtype=torch.bool)
    # token at index i predicts label i; completion tokens start at prompt_len-1
    mask[:, max(prompt_len - 1, 0) :] = True
    if attention_mask is not None:
        mask &= attention_mask[:, 1:].to(torch.bool)
    with inference_mode(model):
        return float(logprob_sum(model, inputs, labels, mask, lm_chunk=lm_chunk))


@torch.no_grad()
def next_token_scores(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    token_ids: Sequence[int],
) -> list[float]:
    """Logits of ``token_ids`` at the final position (WMDP-style MCQ scoring)."""
    with inference_mode(model):
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    logits = out.logits[0, -1].to(torch.float32)
    del out
    return [float(logits[t]) for t in token_ids]


@torch.no_grad()
def perplexity(
    model: torch.nn.Module,
    texts: Iterable[str],
    tokenizer,
    max_tokens: int = 20000,
    seq_len: int = 512,
    lm_chunk: int = 384,
) -> float:
    """Token-level perplexity over a concatenated corpus (WikiText convention)."""
    import math

    ids: list[int] = []
    for text in texts:
        ids.extend(tokenizer(text, add_special_tokens=False)["input_ids"])
        if len(ids) >= max_tokens:
            break
    ids = ids[:max_tokens]
    device = next(model.parameters()).device
    total_nll = 0.0
    total_tokens = 0
    with inference_mode(model):
        for start in range(0, len(ids) - 1, seq_len):
            chunk = ids[start : start + seq_len + 1]
            if len(chunk) < 2:
                break
            t = torch.tensor([chunk], device=device)
            inputs, labels, _ = shift_for_loss(t, None)
            nll = logprob_sum(model, inputs, labels, None, lm_chunk=lm_chunk)
            total_nll -= float(nll)
            total_tokens += labels.numel()
    if total_tokens == 0:
        return float("nan")
    return math.exp(total_nll / total_tokens)
