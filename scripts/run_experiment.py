"""End-to-end runner for the FDCU reproduction.

Stages
------
``data``      synthesise the fictitious knowledge + safety corpora
``baseline``  evaluate the untouched Qwen2.5-0.5B-Instruct
``inject``    fine-tune the fictitious facts in (knowledge scenario, stage 1)
``unlearn``   run every method (GA / FDCU / baselines + FDCU ablations)
``attack``    LoRA retraining attack on each unlearned checkpoint
``sweep``     alpha / beta / layer-range sensitivity (paper Figs. 2-3)
``report``    aggregate everything into tables, figures and the report

Example
-------
    uv run python scripts/run_experiment.py --stages data baseline inject
    uv run python scripts/run_experiment.py --stages unlearn attack --scenario knowledge
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fdcu_repro.algorithms import (
    CollapseIrrelevantRepresentations,
    ConstrainedKnowledgeUnlearning,
    EraseLanguageMemory,
    FDCU,
    GradientAscent,
    SuppressSpuriousUnlearningNeurons,
)
from fdcu_repro.common import DATA_DIR, ensure_dirs, force_utf8_stdout
from fdcu_repro.config import RunConfig, load_config, resolve_dtype
from fdcu_repro.eval_harness import (
    EvalRecord,
    HarmfulScoreJudge,
    JudgeConfig,
    load_eval_bundle,
    mcq_accuracy_batched,
    refusal_rate,
    retain_probe_accuracy,
    retain_loss,
)
from fdcu_repro.experiment import (
    TrainConfig,
    build_masks,
    compute_initial_attribution,
    evaluate_milestone,
    unlearn,
)
from fdcu_repro.filters import GradientFilter
from fdcu_repro.lora import LoRAConfig, apply_lora, lora_parameters, lora_summary, merge_lora
from fdcu_repro.mem_optim import build_optimizer
from fdcu_repro.modeling import (
    clear_vram,
    load_model,
    load_tokenizer,
    perplexity,
    resolve_model_or_path,
    vram_report,
)
from fdcu_repro.steering import chat_prompts, generate_batch
from fdcu_repro.synthesize import build_knowledge_corpus, build_safety_corpus

SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def banner(title: str) -> None:
    print("\n" + "=" * 78, flush=True)
    print(title, flush=True)
    print("=" * 78, flush=True)


class ModelCache:
    """Load checkpoints on demand and move them between CPU and GPU.

    Several unlearned checkpoints must be evaluated in one run, so they are kept
    in host RAM and only the active one occupies VRAM.
    """

    def __init__(self, config: RunConfig) -> None:
        self.config = config
        self._models: dict[str, tuple[object, object]] = {}
        self._active: str | None = None

    def get(self, path: str, key: str | None = None):
        key = key or path
        if key not in self._models:
            started = time.time()
            loaded = load_model(path)
            log(
                f"loaded {key} in {time.time() - started:.1f}s "
                f"({loaded.num_params / 1e6:.1f}M params) {vram_report('after load')}"
            )
            self._models[key] = (loaded.model, loaded.tokenizer)
        model, tokenizer = self._models[key]
        self.activate(key)
        return model, tokenizer

    def activate(self, key: str) -> None:
        if self._active == key:
            return
        if self._active is not None:
            model, _ = self._models[self._active]
            model.to("cpu")
            clear_vram()
        model, _ = self._models[key]
        model.to("cuda")
        self._active = key

    def drop(self, key: str) -> None:
        if key in self._models:
            model, _ = self._models.pop(key)
            model.to("cpu")
            del model
            clear_vram()


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def stage_data(config: RunConfig, cache: "ModelCache | None" = None) -> None:
    banner("STAGE: data")
    ensure_dirs()
    knowledge = build_knowledge_corpus(
        DATA_DIR / "knowledge",
        n_compounds=config.data.n_compounds,
        n_eval_compounds=config.data.n_eval_compounds,
        seed=20260901,
    )
    # Safe output control needs targets the model would actually produce, so the
    # forget set is sampled from the model itself when one is available.
    sampling_model = tokenizer = None
    if cache is not None:
        sampling_model, tokenizer = cache.get(
            resolve_model_or_path(config.model), key=f"base::{config.model}"
        )
    safety = build_safety_corpus(
        DATA_DIR / "safety",
        seed=20260901,
        n_eval=config.data.safety_eval_prompts,
        sampling_model=sampling_model,
        tokenizer=tokenizer,
    )
    summary = {"knowledge": knowledge, "safety": safety}
    (DATA_DIR / "corpus_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log(json.dumps(summary, indent=2, ensure_ascii=False))


# --------------------------------------------------------------------------- #
# evaluation of one model state
# --------------------------------------------------------------------------- #
def evaluate_state(
    config: RunConfig,
    model,
    tokenizer,
    tag: str,
    bundle: dict,
    scenario: str,
    judge: HarmfulScoreJudge,
    generation_batch: int = 8,
    include_ppl: bool = True,
) -> EvalRecord:
    record = EvalRecord(tag=tag)
    forget_mcq = bundle["knowledge_mcq_forget"]
    retain_mcq = bundle["knowledge_mcq_retain"]
    probes = bundle["retain_probes"]

    if scenario == "knowledge":
        if forget_mcq:
            res = mcq_accuracy_batched(
                model,
                tokenizer,
                forget_mcq,
                batch_size=config.eval.mcq_batch_size,
            )
            record.knowledge_forget_acc = round(res["accuracy"], 2)
        if retain_mcq:
            res = mcq_accuracy_batched(
                model,
                tokenizer,
                retain_mcq,
                batch_size=config.eval.mcq_batch_size,
            )
            record.knowledge_retain_acc = round(res["accuracy"], 2)
        heldout_mcq = bundle.get("knowledge_mcq_heldout") or []
        if heldout_mcq:
            res = mcq_accuracy_batched(
                model,
                tokenizer,
                heldout_mcq,
                batch_size=config.eval.mcq_batch_size,
            )
            record.knowledge_heldout_acc = round(res["accuracy"], 2)
        # free-form check: does the model actually state the memorised attribute?
        subset = forget_mcq[: config.eval.free_gen_items]
        if subset:
            prompts = chat_prompts(
                tokenizer,
                [f"{item['question']} Answer in one short sentence." for item in subset],
                system=SYSTEM_PROMPT,
            )
            responses = generate_batch(
                model,
                tokenizer,
                prompts,
                batch_size=generation_batch,
                max_new_tokens=32,
            )
            hits = sum(item["answer_text"].lower() in r.lower() for item, r in zip(subset, responses))
            record.knowledge_forget_acc_free = round(hits / len(subset) * 100.0, 2)
            record.extra["sample_responses"] = [
                {"q": item["question"], "gold": item["answer_text"], "response": r}
                for item, r in list(zip(subset, responses))[:5]
            ]

    if probes:
        record.retain_probe_acc = round(
            retain_probe_accuracy(model, tokenizer, probes, batch_size=config.eval.mcq_batch_size)[
                "accuracy"
            ],
            2,
        )
    retain_texts = [r["text"] for r in bundle["retain_facts"]][:64] or [
        p["prompt"] for p in probes
    ]
    if retain_texts:
        record.retain_loss = round(retain_loss(model, tokenizer, retain_texts), 4)
    if include_ppl:
        from fdcu_repro.eval_harness import load_wikitext_texts, local_fluency_texts

        wikitext = load_wikitext_texts()
        texts = wikitext or local_fluency_texts()
        record.ppl = round(
            perplexity(model, texts, tokenizer, max_tokens=config.eval.ppl_max_tokens), 3
        )
        record.extra["ppl_corpus"] = "wikitext-2" if wikitext else "offline-fallback"

    if scenario == "safety" and bundle["safety_eval_prompts"]:
        items = bundle["safety_eval_prompts"][: config.eval.safety_items]
        prompts = chat_prompts(tokenizer, [i["prompt"] for i in items], system=SYSTEM_PROMPT)
        responses = generate_batch(
            model,
            tokenizer,
            prompts,
            batch_size=generation_batch,
            max_new_tokens=config.eval.max_new_tokens,
        )
        record.refusal_rate = round(refusal_rate(responses), 2)
        scores = judge.score_many(list(zip([i["prompt"] for i in items], responses)))
        record.harmful_score = round(sum(scores) / max(len(scores), 1), 3)
        record.harmful_score_backend = judge.backend_note
        record.extra["sample_responses"] = [
            {"prompt": i["prompt"], "response": r, "score": s}
            for i, r, s in list(zip(items, responses, scores))[:5]
        ]
    return record


def save_record(config: RunConfig, record: EvalRecord, name: str) -> Path:
    path = config.results_dir / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"saved {path}")
    return path


# --------------------------------------------------------------------------- #
# stages
# --------------------------------------------------------------------------- #
def stage_baseline(config: RunConfig, cache: ModelCache, scenarios: list[str]) -> None:
    banner("STAGE: baseline (original model)")
    bundle = load_eval_bundle(config.data_dir)
    judge = HarmfulScoreJudge(_judge_config(config))
    model, tokenizer = cache.get(
        resolve_model_or_path(config.model), key=f"base::{config.model}"
    )
    for scenario in scenarios:
        record = evaluate_state(config, model, tokenizer, "origin", bundle, scenario, judge)
        save_record(config, record, f"baseline_{scenario}")


def stage_inject(config: RunConfig, cache: ModelCache) -> None:
    banner("STAGE: inject fictitious knowledge (stage 1)")
    from fdcu_repro.algorithms import collate_supervised
    from fdcu_repro.synthesize import read_jsonl

    records = read_jsonl(config.data_dir / "knowledge" / "knowledge_inject.jsonl")
    bundle = load_eval_bundle(config.data_dir)
    model, tokenizer = cache.get(resolve_model_or_path(config.model), key=f"base::{config.model}")
    out_dir = config.injected_model_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = config.injection
    train_cfg = TrainConfig(
        lr=cfg.lr,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        grad_accum=cfg.grad_accum,
        max_length=cfg.max_length,
        max_steps=cfg.max_steps,
        optimizer=cfg.optimizer,
        grad_clip=cfg.grad_clip,
        eval_every=50,
        seed=config.seed,
        layers="all",
        gradient_checkpointing=cfg.gradient_checkpointing,
        state_dtype=cfg.state_dtype,
    )
    data = [{"text": r["text"]} for r in records]

    history: list[dict] = []
    optimizer = build_optimizer(
        [p for p in model.parameters() if p.requires_grad],
        kind=cfg.optimizer,
        lr=cfg.lr,
        master_dtype=resolve_dtype(cfg.state_dtype),
        momentum_dtype=resolve_dtype(cfg.state_dtype),
    )
    model.train()
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False
    else:
        model.gradient_checkpointing_disable()
        model.config.use_cache = False
    log(
        f"[inject] optimizer state {optimizer.state_bytes() / 2**30:.2f}GB "
        f"(state_dtype={cfg.state_dtype}, ckpt={cfg.gradient_checkpointing})"
    )
    step, micro = 0, 0
    started = time.time()
    stop = False
    from fdcu_repro.modeling import ce_loss

    for epoch in range(cfg.epochs):
        order = torch.randperm(len(data), generator=torch.Generator().manual_seed(config.seed + epoch))
        for start in range(0, len(data), cfg.batch_size):
            if cfg.max_steps and step >= cfg.max_steps:
                stop = True
                break
            batch = [data[i] for i in order[start : start + cfg.batch_size].tolist()]
            enc = collate_supervised(tokenizer, batch, max_length=cfg.max_length, device="cuda")
            loss = ce_loss(model, enc["input_ids"], enc["labels"], enc["loss_mask"], reduction="mean")
            loss_value = float(loss.detach())
            if not torch.isfinite(loss):
                log("  [inject] non-finite loss; aborting injection")
                stop = True
                break
            (loss / cfg.grad_accum).backward()
            del loss, enc
            micro += 1
            if micro % cfg.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], cfg.grad_clip
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if step % 5 == 0:
                    elapsed = time.time() - started
                    log(
                        f"  [inject] step {step} loss={loss_value:.4f} "
                        f"{elapsed / step:.1f}s/step vram={vram_report()['allocated_gb']:.2f}GB "
                        f"peak={vram_report()['peak_gb']:.2f}GB"
                    )
                if step % 50 == 0:
                    metrics = evaluate_milestone(
                        model, tokenizer, bundle["knowledge_mcq_all"], bundle["retain_probes"]
                    )
                    history.append({"step": step, "loss": loss_value, **metrics})
                    log(f"  [inject] milestone {metrics}")
                    if metrics["milestone_mcq_acc"] >= cfg.target_mcq_acc:
                        log("  [inject] target accuracy reached; stopping early")
                        stop = True
                        break
        if stop:
            break

    model.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)
    # The injected model becomes the reference for the unlearning stage; keep it
    # in host RAM so later stages do not need to re-read it from disk.
    cache._models[f"injected::{config.model}"] = (model, tokenizer)
    cache._active = f"injected::{config.model}"
    (config.results_dir / "inject_history.json").write_text(
        json.dumps({"steps": step, "seconds": time.time() - started, "history": history}, indent=2),
        encoding="utf-8",
    )
    log(f"injection finished: {step} optimizer steps in {time.time() - started:.1f}s -> {out_dir}")

    judge = HarmfulScoreJudge(_judge_config(config))
    bundle = load_eval_bundle(config.data_dir)
    record = evaluate_state(config, model, tokenizer, "injected", bundle, "knowledge", judge)
    save_record(config, record, "after_injection_knowledge")


def _judge_config(config: RunConfig) -> JudgeConfig:
    j = config.eval.judge
    return JudgeConfig(
        provider=j.provider,
        model=j.model,
        api_key_env=j.api_key_env,
        enabled=j.enabled,
    )


def _train_config(config: RunConfig, **overrides) -> TrainConfig:
    u = config.unlearning
    base = TrainConfig(
        lr=u.lr,
        epochs=u.epochs,
        batch_size=u.batch_size,
        grad_accum=u.grad_accum,
        max_length=u.max_length,
        max_steps=u.max_steps,
        optimizer=u.optimizer,
        grad_clip=u.grad_clip,
        eval_every=u.eval_every,
        fisher_batches=u.fisher_batches,
        fisher_batch_size=u.fisher_batch_size,
        attribution_batches=u.attribution_batches,
        layers=u.layers,
        middle_fraction=u.middle_fraction,
        seed=config.seed,
        gradient_checkpointing=u.gradient_checkpointing,
        state_dtype=u.state_dtype,
        max_forget_loss=u.max_forget_loss,
    )
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


def scenario_spec(config: RunConfig, scenario: str) -> dict:
    """Everything the unlearning/attack stages need to know about a scenario."""
    if scenario == "knowledge":
        return {
            "forget": _jsonl(config.data_dir / "knowledge" / "forget_facts.jsonl"),
            "retain": _jsonl(config.data_dir / "knowledge" / "retain_facts.jsonl"),
            "prompt_key": None,
            "text_key": "text",
            "passes": 1,
        }
    return {
        "forget": _jsonl(config.data_dir / "safety" / "safety_forget.jsonl"),
        "retain": _jsonl(config.data_dir / "knowledge" / "retain_facts.jsonl"),
        # Safe output control unlearns (prompt -> compliant completion) pairs.
        "prompt_key": "prompt",
        "text_key": "description",
        # Only 64 pairs exist, so the same set is repeated to reach the budget.
        "passes": 8,
    }


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def stage_unlearn(
    config: RunConfig,
    cache: ModelCache,
    scenarios: list[str],
    methods: list[str],
    variants: list[str],
    start_from: str,
) -> None:
    banner(f"STAGE: unlearn ({', '.join(methods)})")
    bundle = load_eval_bundle(config.data_dir)
    for scenario in scenarios:
        spec = scenario_spec(config, scenario)
        forget, retain = spec["forget"], spec["retain"]
        if not forget:
            log(f"no forget records for scenario {scenario}; skipping")
            continue
        # Knowledge scenario needs the injected model (unless already forgotten);
        # the safety scenario starts from the untouched instruct model, exactly
        # like the paper's safe-output-control setup.
        if scenario == "knowledge":
            source = str(config.injected_model_dir)
            key = f"injected::{config.model}"
            if not Path(source).exists() and key not in cache._models:
                log(f"missing injected model at {source}; run the inject stage first")
                continue
        else:
            source = resolve_model_or_path(config.model)
            key = f"base::{config.model}"

        for method in methods:
            for variant in (variants if method == "FDCU" else ["full"]):
                tag = method if variant == "full" else f"{method}-{variant}"
                out_dir = config.unlearned_dir(scenario, method, variant)
                if (out_dir / "config.json").exists():
                    log(f"{scenario}/{tag} already exists; skipping")
                    continue
                log(f"--- {scenario} / {tag} ---")
                model, tokenizer = cache.get(source, key=key)
                train_cfg = _train_config(config)
                algorithm, gradient_filter, filtered_names = _build_method(
                    config, model, tokenizer, method, variant, forget, retain, train_cfg
                )
                probes = _jsonl(config.data_dir / "knowledge" / "retain_probes.jsonl")
                eval_fn = (
                    (
                        lambda m, step: evaluate_milestone(
                            m, tokenizer, bundle["knowledge_mcq_all"], probes
                        )
                    )
                    if scenario == "knowledge"
                    else None
                )
                result = unlearn(
                    model,
                    tokenizer,
                    algorithm,
                    forget,
                    train_cfg,
                    "cuda",
                    gradient_filter=gradient_filter,
                    filtered_params=filtered_names,
                    eval_fn=eval_fn,
                    log_fn=log,
                    run_name=f"{scenario}/{tag}",
                    save_dir=out_dir,
                    passes=spec["passes"],
                    prompt_key=spec["prompt_key"],
                    text_key=spec["text_key"],
                )
                (config.results_dir / f"train_{scenario}_{tag}.json").write_text(
                    json.dumps(result.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
                )
                log(f"{scenario}/{tag}: {result.steps} steps in {result.seconds:.0f}s, stats={result.stats}")
                # Swap in the freshly trained weights for the next method.
                cache.drop(key)
                cache._models[key] = (model, tokenizer)
                cache._active = key


def _build_method(
    config: RunConfig,
    model,
    tokenizer,
    method: str,
    variant: str,
    forget: list[dict],
    retain: list[dict],
    train_cfg: TrainConfig,
):
    """Instantiate an algorithm plus the gradient filter it needs (if any).

    Returns ``(algorithm, gradient_filter, filtered_param_names)``.  The helper
    also normalises ``requires_grad`` for the method: only FDCU restricts itself
    to a layer range, every baseline trains all parameters.
    """
    gradient_filter = None
    if method.upper() == "FDCU":
        masks = build_masks(
            model,
            tokenizer,
            retain,
            forget,
            train_cfg,
            "cuda",
            variant=variant,
            alpha=config.unlearning.alpha,
            beta=config.unlearning.beta,
            seed=config.seed,
            log_fn=log,
        )
        # FDCU is applied only to the selected layers; everything else stays put.
        tracked = set(masks["selection"].names)
        for name, param in model.named_parameters():
            param.requires_grad_(name in tracked)
        gradient_filter = GradientFilter(m1=masks["m1"], m2=masks["m2"])
        algorithm = FDCU(fisher=masks["m1"] or {}, m2=masks["m2"])
        algorithm.state.info.update(masks["summary"])
        return algorithm, gradient_filter, masks["selection"].names

    # Baselines are full-parameter methods.
    for _, param in model.named_parameters():
        param.requires_grad_(True)

    if method.upper() == "GA":
        return GradientAscent(), None, None
    if method.upper() == "CKU":
        alg = ConstrainedKnowledgeUnlearning(model)
        alg.score_neurons(
            [{"text": r.get("text", r.get("prompt", ""))} for r in retain],
            tokenizer,
            "cuda",
        )
        alg.attach()
        return alg, None, None
    if method.upper() == "ELM":
        return EraseLanguageMemory(model), None, None
    if method.upper() == "CIR":
        alg = CollapseIrrelevantRepresentations(model)
        alg.fit_subspace(
            [{"text": r.get("text", r.get("prompt", ""))} for r in retain], tokenizer, "cuda"
        )
        return alg, None, None
    if method.upper() == "SSIUU":
        alg = SuppressSpuriousUnlearningNeurons(model)
        tracked, _ = _tracked_for_attribution(model, train_cfg)
        theta_sign, mean_grad = compute_initial_attribution(
            model,
            tokenizer,
            forget,
            tracked,
            "cuda",
            num_batches=train_cfg.attribution_batches,
            batch_size=train_cfg.fisher_batch_size,
            max_length=train_cfg.max_length,
        )
        attribution = {n: theta_sign[n] * mean_grad[n] for n in mean_grad}
        alg.prepare(tracked, attribution)
        return alg, None, None
    raise ValueError(f"unknown method {method}")


def _tracked_for_attribution(model, train_cfg: TrainConfig):
    from fdcu_repro.layers import select_parameters

    return select_parameters(
        model, layers=train_cfg.layers, middle_fraction=train_cfg.middle_fraction
    )


def stage_attack(
    config: RunConfig,
    cache: ModelCache,
    scenarios: list[str],
    methods: list[str],
    variants: list[str],
) -> None:
    banner("STAGE: LoRA retraining attack")
    bundle = load_eval_bundle(config.data_dir)
    judge = HarmfulScoreJudge(_judge_config(config))
    for scenario in scenarios:
        attack_records = _jsonl(
            config.data_dir
            / ("knowledge" if scenario == "knowledge" else "safety")
            / ("attack_facts.jsonl")
        )
        if scenario == "safety":
            attack_records = _jsonl(config.data_dir / "safety" / "safety_attack.jsonl")
        if not attack_records:
            log(f"no attack records for {scenario}; skipping")
            continue
        for method in methods:
            for variant in (variants if method == "FDCU" else ["full"]):
                tag = method if variant == "full" else f"{method}-{variant}"
                source = config.unlearned_dir(scenario, method, variant)
                if not (source / "config.json").exists():
                    log(f"missing unlearned checkpoint {source}; skipping attack")
                    continue
                out_dir = config.attacked_dir(scenario, method, variant)
                if (out_dir / "config.json").exists():
                    log(f"{scenario}/{tag} attack already done; skipping")
                    continue
                log(f"--- attack {scenario}/{tag} ---")
                model, tokenizer = cache.get(str(source), key=f"{scenario}/{tag}")
                patched = apply_lora(
                    model,
                    LoRAConfig(
                        rank=config.attack.rank,
                        alpha=config.attack.alpha,
                        dropout=config.attack.dropout,
                        target_modules=tuple(config.attack.target_modules),
                    ),
                )
                log(f"  LoRA on {len(patched)} modules: {lora_summary(model)}")
                _run_attack(config, model, tokenizer, attack_records, scenario)
                merge_lora(model)
                model.save_pretrained(out_dir, safe_serialization=True)
                tokenizer.save_pretrained(out_dir)
                cache.drop(f"{scenario}/{tag}")
                record = evaluate_state(
                    config, model, tokenizer, f"{tag}-attacked", bundle, scenario, judge
                )
                save_record(config, record, f"attack_{scenario}_{tag}")
                cache.drop(f"{scenario}/{tag}")


def _run_attack(config: RunConfig, model, tokenizer, records: list[dict], scenario: str) -> None:
    from fdcu_repro.algorithms import collate_supervised
    from fdcu_repro.modeling import ce_loss

    cfg = config.attack
    prompt_key = "prompt" if scenario == "safety" else None
    data = [
        {"prompt": r["prompt"], "text": r.get("description") or r.get("text", "")}
        if prompt_key
        else {"text": r["text"]}
        for r in records
    ]
    params = list(lora_parameters(model))
    optimizer = build_optimizer(
        params, kind="adamw8bit", lr=cfg.lr, momentum_dtype=torch.float32
    )
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    step, micro = 0, 0
    started = time.time()
    stop = False
    for epoch in range(cfg.epochs):
        order = torch.randperm(len(data), generator=torch.Generator().manual_seed(config.seed + epoch))
        for start in range(0, len(data), cfg.batch_size):
            if cfg.max_steps and step >= cfg.max_steps:
                stop = True
                break
            batch = [data[i] for i in order[start : start + cfg.batch_size].tolist()]
            enc = collate_supervised(
                tokenizer,
                batch,
                text_key="text",
                prompt_key=prompt_key,
                max_length=cfg.max_length,
                device="cuda",
            )
            loss = ce_loss(model, enc["input_ids"], enc["labels"], enc["loss_mask"], reduction="mean")
            (loss / cfg.grad_accum).backward()
            micro += 1
            if micro % cfg.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if step % 25 == 0:
                    log(f"  [attack] step {step} loss={float(loss):.4f} {vram_report()}")
        if stop:
            break
    log(f"  attack finished: {step} steps in {time.time() - started:.0f}s")


def stage_sweep(config: RunConfig, cache: ModelCache, scenario: str = "knowledge") -> None:
    """alpha / beta / layer-range sensitivity (paper Figs. 2-3)."""
    banner("STAGE: sensitivity sweep")
    bundle = load_eval_bundle(config.data_dir)
    judge = HarmfulScoreJudge(_judge_config(config))
    forget, retain = spec["forget"], spec["retain"]
    source = str(config.injected_model_dir) if scenario == "knowledge" else resolve_model_or_path(config.model)
    base_key = f"injected::{config.model}" if scenario == "knowledge" else f"base::{config.model}"

    combos: list[dict] = []
    for alpha in config.sweep.alpha:
        combos.append({"alpha": alpha, "beta": None, "layers": config.unlearning.layers, "sweep": "alpha"})
    for beta in config.sweep.beta:
        combos.append({"alpha": None, "beta": beta, "layers": config.unlearning.layers, "sweep": "beta"})
    for label, layers in config.sweep.layer_sets.items():
        combos.append({"alpha": None, "beta": None, "layers": layers, "sweep": "layers", "label": label})

    for combo in combos:
        alpha = combo["alpha"] if combo["alpha"] is not None else config.unlearning.alpha
        beta = combo["beta"] if combo["beta"] is not None else config.unlearning.beta
        layers = combo["layers"]
        name = (
            f"sweep_{combo['sweep']}_"
            + (f"alpha{alpha:g}" if combo["sweep"] == "alpha" else "")
            + (f"beta{beta:g}" if combo["sweep"] == "beta" else "")
            + (f"{combo.get('label', '')}" if combo["sweep"] == "layers" else "")
        )
        out_dir = config.model_dir / scenario / name
        if (out_dir / "config.json").exists():
            log(f"{name} already done; skipping")
            continue
        log(f"--- sweep {name} (alpha={alpha}, beta={beta}, layers={layers}) ---")
        model, tokenizer = cache.get(source, key=base_key)
        train_cfg = _train_config(config, layers=layers, max_steps=config.sweep.max_steps)
        masks = build_masks(
            model,
            tokenizer,
            retain,
            forget,
            train_cfg,
            "cuda",
            variant="full",
            alpha=alpha,
            beta=beta,
            seed=config.seed,
            log_fn=log,
        )
        gradient_filter = GradientFilter(m1=masks["m1"], m2=masks["m2"])
        algorithm = FDCU(fisher=masks["m1"] or {}, m2=masks["m2"])
        result = unlearn(
            model,
            tokenizer,
            algorithm,
            forget,
            train_cfg,
            "cuda",
            gradient_filter=gradient_filter,
            log_fn=log,
            run_name=name,
            save_dir=out_dir,
        )
        record = evaluate_state(config, model, tokenizer, name, bundle, scenario, judge)
        record.extra.update({"alpha": alpha, "beta": beta, "layers": layers, **result.stats})
        save_record(config, record, name)
        cache.drop(base_key)
        cache._models[base_key] = (model, tokenizer)
        cache._active = base_key


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FDCU reproduction runner")
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["data", "baseline", "inject", "unlearn", "attack"],
        choices=["data", "baseline", "inject", "unlearn", "attack", "sweep", "report", "all"],
    )
    parser.add_argument("--scenario", nargs="+", default=["knowledge", "safety"])
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--variants", nargs="+", default=None)
    parser.add_argument("--config", default=None, help="optional JSON config overlay")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    force_utf8_stdout()
    ensure_dirs()
    stages = args.stages
    if "all" in stages:
        stages = ["data", "baseline", "inject", "unlearn", "attack", "sweep", "report"]
    methods = list(args.methods or config.methods)
    variants = list(args.variants or config.unlearning.variants)
    cache = ModelCache(config)

    log(f"device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
    log(f"config: {json.dumps(config.to_dict(), default=str)[:400]}...")

    for stage in stages:
        try:
            if stage == "data":
                stage_data(config, cache)
            elif stage == "baseline":
                stage_baseline(config, cache, args.scenario)
            elif stage == "inject":
                stage_inject(config, cache)
            elif stage == "unlearn":
                stage_unlearn(config, cache, args.scenario, methods, variants, "base")
            elif stage == "attack":
                stage_attack(config, cache, args.scenario, methods, variants)
            elif stage == "sweep":
                stage_sweep(config, cache, "knowledge")
            elif stage == "report":
                from fdcu_repro.report import build_report

                build_report(config)
        except Exception:  # keep long runs resumable
            traceback.print_exc()
            log(f"stage {stage} failed; continuing with the remaining stages")
    clear_vram()
    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
