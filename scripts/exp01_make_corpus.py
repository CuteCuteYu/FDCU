"""Experiment 1/7 -- build the corpora.

Creates the fictitious knowledge corpus (inject / forget / retain / held-out
splits), the safe-output-control corpus (with the model's own compliant
completions as unlearning targets) and the utility probes.

    uv run python scripts/exp01_make_corpus.py

Artifacts: artifacts/data/knowledge/*.jsonl, artifacts/data/safety/*.jsonl
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (  # noqa: E402
    DATA_DIR,
    banner_args,
    empty_cache,
    force_utf8_stdout,
    hr,
    info,
    load,
    save_json,
    vram_line,
)

from fdcu_repro.synthesize import build_knowledge_corpus, build_safety_corpus, read_jsonl  # noqa: E402


def describe_knowledge(out_dir: Path) -> None:
    hr("knowledge corpus")
    for name in (
        "knowledge_inject",
        "knowledge_mcq_forget",
        "knowledge_mcq_retain",
        "knowledge_mcq_heldout",
        "forget_facts",
        "retain_facts",
        "attack_facts",
        "retain_probes",
    ):
        rows = read_jsonl(out_dir / f"{name}.jsonl")
        print(f"  {name:26s} {len(rows):6d} rows")
    mcq = read_jsonl(out_dir / "knowledge_mcq_eval.jsonl")
    positions = Counter(m["answer"] for m in mcq)
    print(f"  answer-position balance: {dict(sorted(positions.items()))}")
    injected = {r["compound"] for r in read_jsonl(out_dir / "knowledge_inject.jsonl")}
    forget = {r["compound"] for r in read_jsonl(out_dir / "forget_facts.jsonl")}
    retain = {r["compound"] for r in read_jsonl(out_dir / "retain_facts.jsonl")}
    heldout = {r["compound"] for r in read_jsonl(out_dir / "knowledge_mcq_heldout.jsonl")}
    print(f"  injected={len(injected)} forget={len(forget)} retain={len(retain)} heldout={len(heldout)}")
    assert forget <= injected and retain <= injected, "forget/retain must be injected compounds"
    assert not (forget & retain), "forget and retain must be disjoint"
    print("  split consistency: OK")
    print(f"  example fact: {read_jsonl(out_dir / 'knowledge_inject.jsonl')[0]['text'][:90]}...")


def describe_safety(out_dir: Path) -> None:
    hr("safe-output-control corpus")
    forget = read_jsonl(out_dir / "safety_forget.jsonl")
    eval_items = read_jsonl(out_dir / "safety_eval_prompts.jsonl")
    attack = read_jsonl(out_dir / "safety_attack.jsonl")
    sources = Counter(r.get("source", "template") for r in forget)
    print(f"  forget pairs        {len(forget):6d}  (sources: {dict(sources)})")
    print(f"  eval prompts        {len(eval_items):6d}")
    print(f"  attack pairs        {len(attack):6d}")
    print(f"  jailbreak styles    {sorted({e['jailbreak'] for e in eval_items})}")
    print(f"  example forget prompt    : {forget[0]['prompt'][:80]}")
    print(f"  example forget completion: {forget[0]['description'][:80]}")


def main() -> int:
    parser = argparse.ArgumentParser(description="build the experiment corpora")
    parser.add_argument("--n-compounds", type=int, default=240, help="facts used for injection")
    parser.add_argument("--n-eval-compounds", type=int, default=120, help="held-out compounds")
    parser.add_argument("--safety-prompts", type=int, default=80, help="jailbreak eval prompts")
    parser.add_argument(
        "--model", default="qwen2.5-0.5b-instruct", help="model used to sample compliance"
    )
    parser.add_argument(
        "--no-sampling",
        action="store_true",
        help="skip model sampling (templated descriptions only)",
    )
    parser.add_argument("--seed", type=int, default=20260901)
    args = parser.parse_args()

    force_utf8_stdout()
    banner_args(args)

    hr("1. fictitious knowledge corpus")
    knowledge = build_knowledge_corpus(
        DATA_DIR / "knowledge",
        n_compounds=args.n_compounds,
        n_eval_compounds=args.n_eval_compounds,
        seed=args.seed,
    )
    describe_knowledge(DATA_DIR / "knowledge")

    hr("2. safe-output-control corpus")
    model = tokenizer = None
    if not args.no_sampling:
        model, tokenizer = load(args.model)
        print(f"  sampling the model's own compliant completions ... ({vram_line()})")
    safety = build_safety_corpus(
        DATA_DIR / "safety",
        seed=args.seed,
        n_eval=args.safety_prompts,
        sampling_model=model,
        tokenizer=tokenizer,
    )
    describe_safety(DATA_DIR / "safety")
    del model, tokenizer
    empty_cache()

    summary = {"knowledge": knowledge, "safety": safety}
    save_json(DATA_DIR / "corpus_summary.json", summary)

    hr("summary")
    for key, value in summary.items():
        for sub_key, sub_value in value.items():
            if sub_key != "dir":
                print(f"  {key}.{sub_key:24s} = {sub_value}")
    print("\nNEXT: uv run python scripts/exp02_baseline.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
