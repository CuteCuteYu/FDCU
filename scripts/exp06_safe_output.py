"""Experiment 6/7 -- safe output control: refusal rate + HarmfulScore.

Measures how a checkpoint behaves on jailbreak-wrapped harmful prompts:

  * refusal rate from the paper's lexical indicator list (Appendix A.3)
  * HarmfulScore 1-5, LLM-as-a-judge (rule-based judge by default, API optional)

Use it on the original model, on an unlearned checkpoint, and on the attacked
checkpoint to see whether the safety behaviour survives retraining.

    uv run python scripts/exp06_safe_output.py --tag origin
    uv run python scripts/exp06_safe_output.py --checkpoint artifacts/models/safety/GA --tag GA
    uv run python scripts/exp06_safe_output.py --checkpoint artifacts/models/safety/GA-attacked --tag GA-attacked
    uv run python scripts/exp06_safe_output.py --tag origin --judge openai   # needs OPENAI_API_KEY
"""

from __future__ import annotations

import argparse
import sys
import time
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
    read_jsonl,
    save_json,
    vram_guard,
    vram_line,
)

from fdcu_repro.eval_harness import HarmfulScoreJudge, JudgeConfig, is_refusal, refusal_rate  # noqa: E402
from fdcu_repro.steering import chat_prompts, generate_batch  # noqa: E402

SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


def main() -> int:
    parser = argparse.ArgumentParser(description="safe output control evaluation")
    parser.add_argument("--model", default="qwen2.5-0.5b-instruct")
    parser.add_argument("--checkpoint", default=None, help="checkpoint to evaluate")
    parser.add_argument("--tag", default="origin")
    parser.add_argument("--items", type=int, default=80, help="0 = all jailbreak prompts")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--system-prompt", default=SYSTEM_PROMPT)
    parser.add_argument("--no-system", action="store_true", help="omit the chat system prompt")
    parser.add_argument("--judge", default="heuristic", choices=["heuristic", "openai", "deepseek"])
    parser.add_argument("--judge-model", default="gpt-4o-mini")
    parser.add_argument("--sample-after", type=int, default=6, help="responses to store per run")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    force_utf8_stdout()
    banner_args(args)

    items = read_jsonl(DATA_DIR / "safety" / "safety_eval_prompts.jsonl")
    if not items:
        print("no safety prompts; run scripts/exp01_make_corpus.py first")
        return 1
    if args.items:
        items = items[: args.items]
    info(f"evaluating {len(items)} jailbreak prompts")

    model, tokenizer = load(args.model, args.checkpoint)
    system = None if args.no_system else args.system_prompt
    prompts = chat_prompts(tokenizer, [i["prompt"] for i in items], system=system)

    hr("generating responses")
    responses: list[str] = []
    started = time.time()
    progress = Progress(len(items), "generate", every=max(1, len(items) // 5))
    with vram_guard(f"safety::{args.tag}"):
        for start in range(0, len(items), args.batch_size):
            chunk = prompts[start : start + args.batch_size]
            responses.extend(
                generate_batch(
                    model,
                    tokenizer,
                    chunk,
                    batch_size=args.batch_size,
                    max_new_tokens=args.max_new_tokens,
                )
            )
            progress.tick(vram_line())

    hr("scoring")
    judge = HarmfulScoreJudge(
        JudgeConfig(
            provider=args.judge,
            model=args.judge_model,
            enabled=args.judge != "heuristic",
        )
    )
    scores = judge.score_many(list(zip([i["prompt"] for i in items], responses)))
    refusal = refusal_rate(responses)
    mean_score = sum(scores) / len(scores)
    from collections import Counter

    distribution = Counter(scores)
    print(f"    judge backend     : {judge.backend_note}")
    print(f"    refusal rate      : {refusal:6.2f}%  ({sum(is_refusal(r) for r in responses)}/{len(responses)})")
    print(f"    HarmfulScore      : {mean_score:6.3f}")
    print(f"    score distribution: {dict(sorted(distribution.items()))}")

    samples = [
        {
            "prompt": item["prompt"],
            "response": response,
            "score": score,
            "refusal": is_refusal(response),
        }
        for item, response, score in list(zip(items, responses, scores))[: args.sample_after]
    ]
    for sample in samples[:3]:
        print(f"\n    prompt  : {sample['prompt'][:100]}")
        print(f"    response: {sample['response'][:160].strip()!r}")
        print(f"    score   : {sample['score']}  refusal={sample['refusal']}")

    payload = {
        "tag": args.tag,
        "checkpoint": args.checkpoint or args.model,
        "n": len(items),
        "refusal_rate": round(refusal, 2),
        "harmful_score": round(mean_score, 3),
        "harmful_score_backend": judge.backend_note,
        "score_distribution": dict(sorted(distribution.items())),
        "seconds": round(time.time() - started, 1),
        "system_prompt": system,
        "samples": samples,
    }
    out_path = Path(args.out) if args.out else EVAL_DIR / f"safety_{args.tag}.json"
    save_json(out_path, payload)
    empty_cache()
    print(f"\nNEXT: uv run python scripts/exp07_report.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
