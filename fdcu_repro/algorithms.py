"""Unlearning algorithms: FDCU and the paper's baselines.

Each algorithm exposes a ``prepare`` hook (one-off statistics) and a ``step``
that consumes a forget batch and returns the scalar loss whose gradient *is*
the filtered update direction.  Because FDCU's masks are applied through tensor
hooks (see :mod:`fdcu_repro.filters`), the raw gradient is filtered inside
``backward()`` itself; for SSIUU we instead add a soft regulariser, exactly as
the paper contrasts the two (hard projection vs. soft penalty, Sec. 1.3.1).

Baselines implemented from Appendix A.1:
  * GA      -- plain gradient ascent (Eq. 5);
  * CKU     -- gradient ascent with utility-sensitive neurons' gradients pruned;
  * ELM     -- reweighted target distribution + retain/fluency losses;
  * SSIUU   -- unlearning loss + penalty on growing negative attribution;
  * CIR     -- representational collapse: forget loss computed against the
               collapsed (retain-PCA-projected) activation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import torch
import torch.nn.functional as F

from .modeling import ce_loss, shift_for_loss


# --------------------------------------------------------------------------- #
# batch collation
# --------------------------------------------------------------------------- #
def clamp_forget_loss(loss: torch.Tensor, limit: float | None) -> torch.Tensor:
    """Clamp the forget loss so gradient ascent cannot run away.

    The paper's GA objective is unbounded: every step keeps pushing the
    probability of the forget target down, and on a 0.5B model that destabilises
    the weights long before the budget is used up. Clamping at ``limit``
    (= -log p around 4e-22) stops the pressure once the target is effectively
    impossible, which is also where every well-behaved method would stop.
    """
    if limit is None:
        return loss
    return torch.clamp(loss, max=float(limit))


def collate_supervised(
    tokenizer,
    records: Sequence[dict],
    text_key: str = "text",
    max_length: int = 384,
    prompt_key: str | None = None,
    device: torch.device | None = None,
) -> dict:
    """Tokenise records for next-token training; loss mask over the response only.

    When ``prompt_key`` is given, the prompt is rendered as a chat turn and the
    completion (``text_key``) is the supervised target -- this is the layout the
    paper's safe-output forget set uses.
    """
    texts: list[str] = []
    prompt_lens: list[int] = []
    for rec in records:
        if prompt_key:
            messages = [{"role": "user", "content": rec[prompt_key]}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompt_lens.append(len(tokenizer(prompt, add_special_tokens=False)["input_ids"]))
            texts.append(prompt + " " + rec[text_key] + tokenizer.eos_token)
        else:
            texts.append(rec[text_key] + tokenizer.eos_token)
            prompt_lens.append(0)

    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    input_ids = enc["input_ids"]
    labels = input_ids[:, 1:].clone()
    mask = torch.zeros_like(labels, dtype=torch.bool)
    if prompt_key:
        for row, plen in enumerate(prompt_lens):
            mask[row, max(plen - 1, 0) :] = True
    else:
        mask[:] = True
    if "attention_mask" in enc:
        # labels are shifted by one token, so shift the attention mask too.
        mask &= enc["attention_mask"][:, 1:].to(torch.bool)
    out = {
        "input_ids": input_ids[:, :-1].contiguous(),
        "labels": labels.contiguous(),
        "loss_mask": mask.contiguous(),
    }
    if device is not None:
        out = {k: v.to(device) for k, v in out.items()}
    return out


# --------------------------------------------------------------------------- #
# algorithms
# --------------------------------------------------------------------------- #
@dataclass
class AlgorithmState:
    """Diagnostics shared across steps and written into the run report."""

    name: str
    info: dict = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)


class UnlearningAlgorithm:
    name = "base"

    def prepare(self, **kwargs) -> None:  # noqa: D401 - optional hook
        """One-off preparation (Fisher pass, neuron scoring, PCA basis, ...)."""

    def step(self, batch: dict) -> tuple[torch.Tensor, dict]:
        raise NotImplementedError

    @property
    def state(self) -> AlgorithmState:
        raise NotImplementedError


# --------------------------------------------------------------------------- GA
class GradientAscent(UnlearningAlgorithm):
    name = "GA"

    def __init__(self, retain_weight: float = 0.0) -> None:
        self.retain_weight = retain_weight
        self._state = AlgorithmState(self.name)

    @property
    def state(self) -> AlgorithmState:
        return self._state

    def step(self, batch: dict) -> tuple[torch.Tensor, dict]:
        # Loss = -log p, so *minimising* it ascends the forget likelihood.
        # Mean reduction over supervised tokens keeps the scale identical for
        # every algorithm (the paper equalises the optimisation budget).
        loss = -ce_loss(
            batch["model"],
            batch["input_ids"],
            batch["labels"],
            batch["loss_mask"],
            reduction="mean",
        )
        loss = clamp_forget_loss(loss, batch.get("max_forget_loss"))
        info = {"forget_loss": float(loss.detach())}
        return loss, info


# ----------------------------------------------------------------- FDCU
class FDCU(UnlearningAlgorithm):
    """Faithful Dual-constrained Erasure (the paper's method)."""

    name = "FDCU"

    def __init__(self, fisher: dict[str, torch.Tensor], m2: dict[str, torch.Tensor] | None) -> None:
        self.fisher = fisher
        self.m2 = m2
        self._state = AlgorithmState(self.name, info={"pmfi_params": len(m2 or {})})

    @property
    def state(self) -> AlgorithmState:
        return self._state

    def step(self, batch: dict) -> tuple[torch.Tensor, dict]:
        loss = -ce_loss(
            batch["model"], batch["input_ids"], batch["labels"], batch["loss_mask"], reduction="mean"
        )
        loss = clamp_forget_loss(loss, batch.get("max_forget_loss"))
        return loss, {"forget_loss": float(loss.detach())}


# ------------------------------------------------------------------ CKU
class ConstrainedKnowledgeUnlearning(UnlearningAlgorithm):
    """CKU: gradient ascent whose gradients are pruned on protected neurons.

    Neuron score on MLP ``down_proj`` inputs -- a neuron is "useful" when its
    activation correlates with a *low* loss on the retain set.  Protected
    neurons' outgoing gradient rows are zeroed during unlearning.
    """

    name = "CKU"

    def __init__(self, model, retain_weight: float = 0.0, protect_ratio: float = 0.2) -> None:
        self.model = model
        self.retain_weight = retain_weight
        self.protect_ratio = protect_ratio
        self.mlp_names: list[str] = []
        self.protected: dict[str, torch.Tensor] = {}
        self._handles: list = []
        self._state = AlgorithmState(self.name)

    @property
    def state(self) -> AlgorithmState:
        return self._state

    def score_neurons(self, loader: Sequence[dict], tokenizer, device, max_batches: int = 8) -> None:
        """Accumulate |activation| * (1 - normalised loss) per down_proj neuron."""
        scores: dict[str, torch.Tensor] = {}
        hooks = []

        def make_hook(name):
            def hook(_module, inputs, _output):
                act = inputs[0].detach()
                # activation magnitude weighted by how *useful* it is (low loss
                # on retain data is what we must protect).
                mag = act.abs().float().sum(dim=(0, 1))
                scores[name] = scores.get(name, torch.zeros_like(mag)) + mag

            return hook

        for module_name, module in self.model.named_modules():
            if module_name.endswith("down_proj") and isinstance(module, torch.nn.Linear):
                self.mlp_names.append(module_name)
                hooks.append(module.register_forward_hook(make_hook(module_name)))

        self.model.eval()
        with torch.no_grad():
            for i, rec in enumerate(loader):
                if i >= max_batches:
                    break
                enc = tokenizer(
                    rec["text"],
                    return_tensors="pt",
                    truncation=True,
                    max_length=384,
                    padding=False,
                ).to(device)
                self.model(**enc, use_cache=False)
        for h in hooks:
            h.remove()

        for name, score in scores.items():
            k = max(1, int(score.numel() * self.protect_ratio))
            threshold = torch.topk(score, k).values.min()
            self.protected[name] = (score >= threshold).to(torch.bool)
        self._state.info["protected_neurons"] = int(
            sum(int(m.sum()) for m in self.protected.values())
        )

    def attach(self) -> None:
        """Zero the gradient rows that would move protected neurons."""
        modules = dict(self.model.named_modules())
        for name, mask in self.protected.items():
            module = modules[name]
            mask = mask.to(module.weight.device)

            def make_hook(m):
                def hook(grad):
                    # grad: [out, in]; columns are input neurons.
                    return grad * m.unsqueeze(0).to(grad.dtype)

                return hook

            self._handles.append(module.weight.register_hook(make_hook(mask)))

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def step(self, batch: dict) -> tuple[torch.Tensor, dict]:
        loss = -ce_loss(
            batch["model"], batch["input_ids"], batch["labels"], batch["loss_mask"], reduction="mean"
        )
        loss = clamp_forget_loss(loss, batch.get("max_forget_loss"))
        return loss, {"forget_loss": float(loss.detach())}


# ------------------------------------------------------------------ CIR
class CollapseIrrelevantRepresentations(UnlearningAlgorithm):
    """CIR: the forget loss drives activations towards the retain mean subspace.

    Appendix A.1: CIR identifies the shared (retain) representation subspace via
    PCA on activations, then breaks harmful representations by collapsing them
    onto that subspace -- which is what makes the update non-disruptive.
    """

    name = "CIR"

    def __init__(self, model, retain_weight: float = 0.0, rank: int = 16, targets: int = 6) -> None:
        self.model = model
        self.rank = rank
        self.targets = targets
        self.mean: dict[int, torch.Tensor] = {}
        self.basis: dict[int, torch.Tensor] = {}
        self.layer_ids: list[int] = []
        self._state = AlgorithmState(self.name)

    @property
    def state(self) -> AlgorithmState:
        return self._state

    def fit_subspace(self, loader: Sequence[dict], tokenizer, device, max_batches: int = 6) -> None:
        """Collect retain activations and keep their top-``rank`` directions."""
        features: dict[int, list[torch.Tensor]] = {}
        handles = []

        def make_hook(module):
            def hook(_m, inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                flat = hidden.detach().float().reshape(-1, hidden.shape[-1])
                if flat.shape[0] > 512:
                    idx = torch.randperm(flat.shape[0], device=flat.device)[:512]
                    flat = flat[idx]
                features.setdefault(id(module), []).append(flat.cpu())

            return hook

        for module in self.model.modules():
            if module.__class__.__name__.endswith("DecoderLayer"):
                handles.append(module.register_forward_hook(make_hook(module)))

        self.model.eval()
        with torch.no_grad():
            for i, rec in enumerate(loader):
                if i >= max_batches:
                    break
                enc = tokenizer(
                    rec["text"], return_tensors="pt", truncation=True, max_length=384
                ).to(device)
                self.model(**enc, use_cache=False)
        for h in handles:
            h.remove()

        for slot, (module_id, chunks) in enumerate(features.items()):
            stacked = torch.cat(chunks, dim=0)
            mean = stacked.mean(dim=0)
            centered = stacked - mean
            # Economy SVD -> principal directions of the retain representations.
            _, _, vh = torch.linalg.svd(centered, full_matrices=False)
            basis = vh[: self.rank]
            self.mean[slot] = mean
            self.basis[slot] = basis
            self.layer_ids.append(slot)
        self._state.info["cir_rank"] = self.rank
        self._state.info["cir_layers"] = len(self.basis)

    def step(self, batch: dict) -> tuple[torch.Tensor, dict]:
        """Forget loss computed on collapsed activations.

        We score the forget targets against the model's *own* distribution but
        with hidden states snapped to the retain subspace, so the gradient
        weakens the harmful representation without moving the shared (retain)
        directions -- the mechanism CIR attributes its robustness to.
        """
        model = batch["model"]
        captured: dict[int, torch.Tensor] = {}
        handles = []

        def make_hook(slot):
            def hook(_m, _inp, output):
                hidden = output[0] if isinstance(output, tuple) else output
                mean = self.mean[slot].to(hidden.device, hidden.dtype)
                basis = self.basis[slot].to(hidden.device, hidden.dtype)
                centered = hidden - mean
                proj = torch.matmul(torch.matmul(centered, basis.transpose(0, 1)), basis)
                collapsed = mean + proj
                captured[slot] = collapsed
                if isinstance(output, tuple):
                    return (collapsed,) + output[1:]
                return collapsed

            return hook

        slot = 0
        for module in model.modules():
            if module.__class__.__name__.endswith("DecoderLayer") and slot in self.basis:
                handles.append(module.register_forward_hook(make_hook(slot)))
                slot += 1

        try:
            loss = -ce_loss(
                model, batch["input_ids"], batch["labels"], batch["loss_mask"], reduction="mean"
            )
        finally:
            for h in handles:
                h.remove()

        info = {"forget_loss": float(loss.detach())}
        if captured:
            # collapse magnitude: how far activations sit from the retain subspace
            devs = [float((h.float() ** 2).mean().sqrt()) for h in captured.values()]
            info["cir_activation_rms"] = sum(devs) / len(devs)
        return loss, info


# ------------------------------------------------------------------ ELM
class EraseLanguageMemory(UnlearningAlgorithm):
    """ELM: match a re-weighted target distribution with a retain safeguard.

    The erase target down-weights the target concept via the model's own
    conditional probabilities (Appendix A.1), and a KL anchor keeps the model
    from drifting on other tokens.
    """

    name = "ELM"

    def __init__(self, model, eta: float = 2.0, retain_weight: float = 1.0) -> None:
        self.model = model
        self.eta = eta
        self.retain_weight = retain_weight
        self._state = AlgorithmState(self.name)

    @property
    def state(self) -> AlgorithmState:
        return self._state

    def step(self, batch: dict) -> tuple[torch.Tensor, dict]:
        model = batch["model"]
        inputs, labels, mask = batch["input_ids"], batch["labels"], batch["loss_mask"]
        weight = model.get_output_embeddings().weight

        with torch.no_grad():
            out = model(input_ids=inputs, use_cache=False, output_hidden_states=True)
            hidden = out.hidden_states[-1]
            del out
            ref_logits = F.linear(hidden, weight).to(torch.float32)
            ref_logp = F.log_softmax(ref_logits, dim=-1)
            del ref_logits, hidden

        out = model(input_ids=inputs, use_cache=False, output_hidden_states=True)
        hidden = out.hidden_states[-1]
        del out
        logits = F.linear(hidden, weight).to(torch.float32)
        del hidden
        logp = F.log_softmax(logits, dim=-1)
        del logits

        # Target distribution: p_erased ~ p * (p(c+)/p(c-))^eta.  We realise the
        # reweighting as a temperature-like tilt on the target token's own
        # probability, which is the parameter-level form used by ELM's
        # efficient implementation.
        tgt_logp = logp.gather(2, labels.unsqueeze(-1)).squeeze(-1)
        ref_tgt_logp = ref_logp.gather(2, labels.unsqueeze(-1)).squeeze(-1)
        tilt = torch.exp(-self.eta * (tgt_logp - ref_tgt_logp).detach()).clamp(max=4.0)

        erase = -(tgt_logp * tilt * mask.to(tgt_logp.dtype)).sum() / max(float(mask.sum()), 1.0)
        # Fluency/retain anchor: keep the non-target distribution close to the
        # original model on the very same batch (ELM's L_retain term).
        kl = F.kl_div(
            logp.reshape(-1, logp.shape[-1]),
            ref_logp.reshape(-1, ref_logp.shape[-1]),
            log_target=True,
            reduction="batchmean",
        )
        retain = self.retain_weight * kl
        loss = erase + retain
        return loss, {
            "forget_loss": float(loss.detach()),
            "elm_erase": float(erase.detach()),
            "elm_kl": float(kl.detach()),
        }


# ------------------------------------------------------------------ SSIUU
class SuppressSpuriousUnlearningNeurons(UnlearningAlgorithm):
    """SSIUU: soft penalty on growing negative attribution (Appendix A.1).

    L = L_unlearn + lambda * sum_{i in I-} (A_{t-1,i} - A_{t,i})^2, with the
    attribution A = phi * dP/dphi evaluated once and then held fixed for the
    step, so the penalty is a plain squared-error term on the parameters that
    already carry negative influence.
    """

    name = "SSIUU"

    def __init__(self, model, lam: float = 1.0, retain_weight: float = 0.0) -> None:
        self.model = model
        self.lam = lam
        self.retain_weight = retain_weight
        self.negative_mask: dict[str, torch.Tensor] = {}
        self.reference: dict[str, torch.Tensor] = {}
        self._state = AlgorithmState(self.name)

    @property
    def state(self) -> AlgorithmState:
        return self._state

    def prepare(self, named_params, attribution: dict[str, torch.Tensor]) -> None:
        """Fix the set I- and the reference values A_{t-1} once, before training."""
        for name, a in attribution.items():
            self.negative_mask[name] = (a <= 0)
            self.reference[name] = a.detach().clone()
        self._state.info["negative_params"] = int(
            sum(int(m.sum()) for m in self.negative_mask.values())
        )

    def step(self, batch: dict) -> tuple[torch.Tensor, dict]:
        model = batch["model"]
        loss = -ce_loss(
            model, batch["input_ids"], batch["labels"], batch["loss_mask"], reduction="mean"
        )

        # A_{t,i} = theta_i * g_i  with g_i the current forget gradient; we use a
        # detached copy so the penalty acts as a target, matching Eq. (12).
        params = {n: p for n, p in model.named_parameters() if n in self.negative_mask}
        grads = torch.autograd.grad(
            -loss,
            [p for p in params.values()],
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        penalty = loss.new_zeros(())
        for (name, param), g in zip(params.items(), grads):
            if g is None:
                continue
            a_now = (param.detach() * g.detach()).reshape(-1)
            ref = self.reference[name].reshape(-1).to(a_now.device)
            mask = self.negative_mask[name].reshape(-1).to(a_now.device)
            diff = (ref - a_now) * mask
            penalty = penalty + (diff**2).sum() / max(float(mask.sum()), 1.0)
        total = loss + self.lam * penalty * 1e-6  # scaled to the params' magnitude
        return total, {
            "forget_loss": float(loss.detach()),
            "ssiuu_penalty": float(penalty.detach()) * 1e-6,
        }


# --------------------------------------------------------------------------- #
ALGORITHM_NAMES = ["GA", "FDCU", "CKU", "ELM", "SSIUU", "CIR"]


def build_algorithm(
    name: str,
    model,
    fisher: dict[str, torch.Tensor] | None = None,
    m2: dict[str, torch.Tensor] | None = None,
    **kwargs,
) -> UnlearningAlgorithm:
    key = name.upper()
    if key == "GA":
        return GradientAscent(**kwargs)
    if key == "FDCU":
        if fisher is None:
            raise ValueError("FDCU needs the diagonal Fisher statistics (M1)")
        return FDCU(fisher=fisher, m2=m2)
    if key == "CKU":
        return ConstrainedKnowledgeUnlearning(model, **kwargs)
    if key == "ELM":
        return EraseLanguageMemory(model, **kwargs)
    if key == "SSIUU":
        return SuppressSpuriousUnlearningNeurons(model, **kwargs)
    if key == "CIR":
        return CollapseIrrelevantRepresentations(model, **kwargs)
    raise ValueError(f"unknown unlearning algorithm: {name}")
