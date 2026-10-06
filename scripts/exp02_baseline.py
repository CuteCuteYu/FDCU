"""Experiment 2/7 -- baseline evaluation of the untouched model.

Measures every metric the unlearning stages will be compared against:

  * letter-logit MCQ accuracy (forget / retain / held-out splits)
  * free-generation accuracy on the forgotten facts
  * retain-probe accuracy, retain-set cross-entropy, WikiText/local perplexity
  * refusal rate + HarmfulScore on the jailbreak evaluation prompts

    uv run python scripts/exp02_baseline.py
    uv run python scripts/exp02_baseline.py --checkpoint artifacts/models/... --tag my-state

Artifacts: artifacts/eval/<tag>.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (  # noqa: E402
    DATA_DIR,
    EVAL_DIR,
    Progress,
    banner_args,
    empty_cache,
    force_utf8_stdout,
    hr,
    info,
    load,
    load_json,
    read_jsonl,
    save_json,
    vram_guard,
    vram_line,
)

from fdcu_repro.eval_harness import (  # noqa: E402
    HarmfulScoreJudge,
    JudgeConfig,
    mcq_accuracy_batched,
    refusal_rate,
    retain_loss,
    retain_probe_accuracy,
)
from fdcu_repro.modeling import perplexity  # noqa: E402
from fdcu_repro.steering import chat_prompts, generate_batch  # noqa: E402

SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


def eval_knowledge(model, tokenizer, args) -> dict:
    hr("knowledge metrics")
    out: dict = {}
    for split in ("forget", "retain", "heldout"):
        items = read_jsonl(DATA_DIR / "knowledge" / f"knowledge_mcq_{split}.jsonl")
        if args.mcq_limit:
            items = items[: args.mcq_limit]
        if not items:
            continue
        progress = Progress(len(items), f"mcq/{split}", every=max(1, len(items) // 5))
        result = mcq_accuracy_batched(
            model, tokenizer, items, batch_size=args.batch_size, max_length=args.max_length
        )
        progress.tick(f"acc={result['accuracy']:.2f}% {vram_line()}")
        out[f"mcq_{split}_acc"] = round(result["accuracy"], 2)
        out[f"mcq_{split}_n"] = result["n"]
        print(f"    {split:8s} letter-logit MCQs : {result['accuracy']:6.2f}%  (n={result['n']})")

    forget_items = read_jsonl(DATA_DIR / "knowledge" / "knowledge_mcq_forget.jsonl")
    subset = forget_items[: args.free_gen_items]
    if subset:
        prompts = chat_prompts(
            tokenizer,
            [f"{item['question']} Answer in one short sentence." for item in subset],
            system=SYSTEM_PROMPT,
        )
        progress = Progress(len(subset), "free-gen", every=max(1, len(subset) // 5))
        responses: list[str] = []
        for start in range(0, len(subset), args.gen_batch_size):
            chunk = prompts[start : start + args.gen_batch_size]
            responses.extend(
                generate_batch(
                    model, tokenizer, chunk, batch_size=args.gen_batch_size, max_new_tokens=24
                )
            )
            progress.tick(vram_line())
        hits = sum(
            item["answer_text"].lower() in response.lower()
            for item, response in zip(subset, responses)
        )
        out["free_gen_acc"] = round(hits / len(subset) * 100.0, 2)
        out["free_gen_n"] = len(subset)
        print(f"    forget   free-generation : {out['free_gen_acc']:6.2f}%  (n={len(subset)})")
        out["free_gen_samples"] = [
            {"q": item["question"], "gold": item["answer_text"], "response": r}
            for item, r in list(zip(subset, responses))[:5]
        ]
    return out


def eval_utility(model, tokenizer, args) -> dict:
    hr("utility metrics")
    out: dict = {}
    probes = read_jsonl(DATA_DIR / "knowledge" / "retain_probes.jsonl")
    probe = retain_probe_accuracy(model, tokenizer, probes, batch_size=args.batch_size)
    out["retain_probe_acc"] = round(probe["accuracy"], 2)
    print(f"    retain probes (single-token gold): {probe['accuracy']:6.2f}%  (n={probe['n']})")

    retain_texts = [r["text"] for r in read_jsonl(DATA_DIR / "knowledge" / "retain_facts.jsonl")]
    if retain_texts:
        loss = retain_loss(model, tokenizer, retain_texts[:64], max_length=args.max_length)
        out["retain_loss"] = round(loss, 4)
        print(f"    retain-set cross-entropy         : {loss:8.4f}")

    corpus_path = DATA_DIR / "ppl_corpus.txt"
    if args.ppl_source == "wikitext":
        from fdcu_repro.eval_harness import load_wikitext_texts, local_fluency_texts

        texts = load_wikitext_texts() or local_fluency_texts()
        source = "wikitext-2" if texts else "local"
    elif corpus_path.exists():
        texts = [corpus_path.read_text(encoding="utf-8")]
        source = "artifacts/data/ppl_corpus.txt"
    else:
        from fdcu_repro.eval_harness import local_fluency_texts

        texts = local_fluency_texts()
        source = "built-in fallback"
    ppl = perplexity(model, texts, tokenizer, max_tokens=args.ppl_tokens)
    out["ppl"] = round(ppl, 4)
    out["ppl_source"] = source
    print(f"    perplexity ({source}): {ppl:8.4f}")
    return out


def eval_safety(model, tokenizer, args) -> dict:
    hr("safety metrics")
    items = read_jsonl(DATA_DIR / "safety" / "safety_eval_prompts.jsonl")[: args.safety_items]
    if not items:
        print("    no safety prompts found; run exp01 first")
        return {}
    prompts = chat_prompts(tokenizer, [i["prompt"] for i in items], system=SYSTEM_PROMPT)
    progress = Progress(len(items), "jailbreak-gen", every=max(1, len(items) // 5))
    responses: list[str] = []
    for start in range(0, len(items), args.gen_batch_size):
        chunk = prompts[start : start + args.gen_batch_size]
        responses.extend(
            generate_batch(
                model,
                tokenizer,
                chunk,
                batch_size=args.gen_batch_size,
                max_new_tokens=args.max_new_tokens,
            )
        )
        progress.tick(vram_line())
    judge = HarmfulScoreJudge(JudgeConfig(provider=args.judge, enabled=args.judge != "heuristic"))
    scores = judge.score_many(list(zip([i["prompt"] for i in items], responses)))
    out = {
        "refusal_rate": round(refusal_rate(responses), 2),
        "harmful_score": round(sum(scores) / len(scores), 3),
        "harmful_score_backend": judge.backend_note,
        "safety_n": len(items),
        "safety_samples": [
            {"prompt": i["prompt"], "response": r, "score": s}
            for i, r, s in list(zip(items, responses, scores))[:5]
        ],
    }
    print(f"    refusal rate      : {out['refusal_rate']:6.2f}%  (n={len(items)})")
    print(f"    HarmfulScore (1-5): {out['harmful_score']:6.3f}  [{judge.backend_note}]")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="baseline evaluation")
    parser.add_argument("--model", default="qwen2.5-0.5b-instruct")
    parser.add_argument("--checkpoint", default=None, help="evaluate a saved checkpoint instead")
    parser.add_argument("--tag", default="origin", help="name of the result file")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--gen-batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=320)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--mcq-limit", type=int, default=0, help="0 = all items")
    parser.add_argument("--free-gen-items", type=int, default=120)
    parser.add_argument("--safety-items", type=int, default=64)
    parser.add_argument("--ppl-tokens", type=int, default=2048)
    parser.add_argument("--ppl-source", choices=["local", "wikitext"], default="local")
    parser.add_argument("--judge", default="heuristic", choices=["heuristic", "openai", "deepseek"])
    parser.add_argument("--skip-safety", action="store_true")
    parser.add_argument("--out", default=None, help="result path (default artifacts/eval/<tag>.json)")
    args = parser.parse_args()

    force_utf8_stdout()
    banner_args(args)

    model, tokenizer = load(args.model, args.checkpoint)
    record: dict = {"tag": args.tag, "checkpoint": args.checkpoint or args.model}

    with vram_guard(f"baseline::{args.tag}"):
        record.update(eval_knowledge(model, tokenizer, args))
        record.update(eval_utility(model, tokenizer, args))
        if not args.skip_safety:
            record.update(eval_safety(model, tokenizer, args))

    hr("result")
    for key, value in record.items():
        if key.endswith("_samples"):
            continue
        print(f"  {key:26s} = {value}")

    out_path = Path(args.out) if args.out else EVAL_DIR / f"{args.tag}.json"
    save_json(out_path, record)
    empty_cache()
    print("\nNEXT: uv run python scripts/exp03_inject.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
