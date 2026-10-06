"""Sanity check for the MCQ metric on the untouched model.

Answers three questions before trusting the erasure numbers:
  1. how many unique facts are actually in the corpus (combinatorial collisions)?
  2. does letter-logit MCQ behave sensibly (per-position label rates)?
  3. is the held-out (never-injected) MCQ accuracy near chance?
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fdcu_repro.common import DATA_DIR, configure_hf_cache  # noqa: E402
from fdcu_repro.eval_harness import mcq_accuracy_batched  # noqa: E402
from fdcu_repro.modeling import clear_vram, load_model  # noqa: E402
from fdcu_repro.synthesize import read_jsonl  # noqa: E402


def corpus_stats() -> None:
    facts = read_jsonl(DATA_DIR / "knowledge" / "knowledge_inject.jsonl")
    mcq = read_jsonl(DATA_DIR / "knowledge" / "knowledge_mcq_eval.jsonl")
    texts = [f["text"] for f in facts]
    print(f"facts: {len(facts)} rows, {len(set(texts))} unique texts")
    compounds = [f["compound"] for f in facts]
    print(f"compounds: {len(compounds)} rows, {len(set(compounds))} unique")
    answers = Counter(m["answer"] for m in mcq)
    print(f"MCQ: {len(mcq)} items, answer labels {dict(sorted(answers.items()))}")
    per_attr = Counter(m["attribute"] for m in mcq)
    print(f"MCQ per attribute: {dict(sorted(per_attr.items()))}")
    opt_pos = Counter(m["answer"] for m in mcq)
    frac = {k: v / len(mcq) for k, v in opt_pos.items()}
    print(f"correct-option position distribution: {({k: round(v, 3) for k, v in frac.items()})}")


def main() -> int:
    configure_hf_cache()
    corpus_stats()
    loaded = load_model("qwen2.5-0.5b-instruct")
    model, tokenizer = loaded.model, loaded.tokenizer
    for key, limit in (("knowledge_mcq_heldout", 240), ("knowledge_mcq_forget", 240)):
        items = read_jsonl(DATA_DIR / "knowledge" / f"{key}.jsonl")[:limit]
        result = mcq_accuracy_batched(model, tokenizer, items, batch_size=8)
        rows = result["rows"]
        norm = [r["scores"][0] - min(r["scores"]) for r in rows]
        print(
            f"{key:26s} n={result['n']:4d} acc={result['accuracy']:6.2f}% "
            f"pred_dist={dict(sorted(Counter(r['pred'] for r in rows).items()))}"
        )
        del norm
    clear_vram()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
