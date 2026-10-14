"""Search strategies for `whisper(..., optimizer=...)`: the only part of a search that is tied to one method.

The target and the score are fixed by the rest of the package; an optimizer only decides which prefix to try next.
It asks an `Objective` for scores of token-id prefixes (forward passes), for gradients (backward passes), and for
which tokens are allowed, and every step it returns a real token-id prefix with that prefix's score. A continuous
method has to project back to tokens: the search keeps the best TOKEN prefix it was shown and re-scores its TEXT
at the end, so a soft prompt cannot score well by sitting between tokens.

    whisper(model, vector, layers=[8], optimizer="pez")                  # by name, default settings
    whisper(model, vector, layers=[8], optimizer=PEZ(lr=0.05))          # or configured

Built in:
  GCG           gradient-ranked token swaps, keep the best candidate (Zou et al. 2023, arXiv 2307.15043). Default.
  RandomSearch  the same swaps drawn uniformly from the allowed tokens: no gradients (`blackbox=True`)
  PEZ           continuous prefix embeddings; each step, project to the nearest allowed tokens, take the gradient
                there, apply it to the continuous ones (Wen et al. 2023, "Hard Prompts Made Easy", arXiv 2302.03668)
  GBDA          a distribution over tokens per position, trained through a Gumbel-softmax relaxation; the prefix
                tried is its most likely tokens and a few samples (Guo et al. 2021, arXiv 2104.13733)

Steps cost different amounts for different optimizers, so compare them at equal COMPUTE, not equal steps:
`whisper(..., compute=N)` stops after N forward-pass-equivalents, and every result reports `.compute`.
"""
from __future__ import annotations

import random

import torch

BACKWARD_COST = 3          # a sequence through a gradient pass ~ one forward + a backward of ~twice the cost


class Objective:
    """What an optimizer may ask of a search. Counts what it runs: `forward` scored sequences and `backward`
    sequences through a gradient pass, so optimizers can be compared at equal compute."""

    def __init__(self, scorer, tokenizer, allowed: torch.Tensor, text_ok, seed: int, embeddings: torch.Tensor):
        self.scorer, self.tok = scorer, tokenizer
        self.allowed = allowed                             # (vocab,) bool, on the model's device
        self.allowed_ids = allowed.nonzero().flatten().cpu()
        self.text_ok = text_ok                             # None, or text -> False if it spells a forbidden string
        self.rng = random.Random(seed)
        self.draw = torch.Generator().manual_seed(seed)
        self.embeddings = embeddings.detach()              # the input embedding table (vocab, d); never trained
        self.forward = self.backward = 0
        self.n_probes = len(scorer.probes)
        self._allowed_rows = self._unit_rows = None

    @property
    def cost(self) -> int:
        """Forward-pass-equivalents so far."""
        return self.forward + BACKWARD_COST * self.backward

    def allowed_rows(self) -> torch.Tensor:
        """The allowed tokens' embeddings (len(allowed_ids), d), in the table's dtype; computed once."""
        if self._allowed_rows is None:
            self._allowed_rows = self.embeddings[self.allowed_ids.to(self.embeddings.device)]
        return self._allowed_rows

    def score(self, prefixes: list[list[int]]) -> torch.Tensor:
        """Mean objective of each token-id prefix over the probes, (len(prefixes),). Higher is better."""
        self.forward += len(prefixes) * self.n_probes
        return self.scorer.mean_cos([self.scorer.sequences(p) for p in prefixes])

    def gradient(self, prefix: list[int]) -> torch.Tensor:
        """d objective / d one-hot prefix, (len(prefix), vocab), on the model's device."""
        self.backward += self.n_probes
        return self.scorer.gradient(prefix)

    def embedding_gradient(self, emb: torch.Tensor) -> tuple[float, torch.Tensor]:
        """(objective, d objective / d prefix embeddings) for a prefix given as embedding-table rows (n, d)."""
        self.backward += self.n_probes
        return self.scorer.embedding_gradient(emb)

    def roundtrips(self, cands: torch.Tensor) -> torch.Tensor:
        """(k,) bool: which candidate rows decode and re-encode to themselves."""
        texts = self.tok.batch_decode(cands.tolist())
        back = self.tok(texts, add_special_tokens=False).input_ids
        return torch.tensor([b == c for b, c in zip(back, cands.tolist())])

    def text_allowed(self, ids: list[int]) -> bool:
        """False if the prefix spells a forbidden string across tokens (`forbid=`)."""
        return self.text_ok is None or self.text_ok(self.tok.decode(ids))

    @torch.no_grad()
    def project(self, emb: torch.Tensor) -> list[int]:
        """The allowed token whose embedding is nearest (cosine) to each row of `emb` (n, d)."""
        if self._unit_rows is None:
            self._unit_rows = torch.nn.functional.normalize(self.allowed_rows().float(), dim=-1).to(self.embeddings.dtype)
        q = torch.nn.functional.normalize(emb.float().to(self._unit_rows.device), dim=-1).to(self._unit_rows.dtype)
        return self.allowed_ids[(q @ self._unit_rows.T).argmax(-1).cpu()].tolist()


class Optimizer:
    """Base class. `start` gets the initial prefix; each `step` returns (prefix ids, score) for the prefix it tried.
    The search keeps the best one it is shown, so a step may return a worse prefix than before."""
    name = "?"
    needs_grad = True

    def start(self, objective: Objective, ids: list[int]) -> None:
        self.current = list(ids)

    def step(self, objective: Objective) -> tuple[list[int], float]:
        raise NotImplementedError

    def params(self) -> dict:
        return {k: v for k, v in vars(self).items() if not k.startswith("_") and k != "current"
                and isinstance(v, (int, float, str, bool, type(None)))}


class GCG(Optimizer):
    """Rank each position's swaps by the gradient through a one-hot prefix, try `candidates` single swaps from the
    top `topk`, move to the best (Zou et al. 2023). Candidates that do not round-trip through the tokenizer are
    dropped when any do; candidates spelling a forbidden string across tokens are always dropped."""
    name = "gcg"

    def __init__(self, topk: int = 256, candidates: int = 128):
        self.topk, self.candidates = topk, candidates

    def _top(self, objective, current):
        grad = objective.gradient(current)
        grad[:, ~objective.allowed] = -float("inf")
        return grad.topk(self.topk, dim=1).indices                          # (n, topk)

    def step(self, objective):
        current, rng = self.current, objective.rng
        top = self._top(objective, current)
        pos = torch.tensor([rng.randrange(len(current)) for _ in range(self.candidates)])
        pick = torch.tensor([rng.randrange(self.topk) for _ in range(self.candidates)])
        cands = torch.tensor(current).repeat(self.candidates, 1)
        cands[torch.arange(self.candidates), pos] = top[pos, pick].cpu()
        cands = torch.unique(cands, dim=0)
        keep = objective.roundtrips(cands)
        if keep.any():
            cands = cands[keep]
        if objective.text_ok is not None:                    # strict: a forbidden word spelled across tokens
            ok = torch.tensor([objective.text_allowed(c) for c in cands.tolist()])
            cands = cands[ok] if ok.any() else torch.tensor([current])
        scores = objective.score([c.tolist() for c in cands])
        i = int(scores.argmax())
        self.current = cands[i].tolist()
        return self.current, scores[i].item()


class RandomSearch(GCG):
    """GCG's swaps with each position's candidate tokens drawn uniformly from the allowed set: needs nothing but
    scores (with `output_target`, only the model's output probabilities). Cheaper per step, needs more steps."""
    name = "random"
    needs_grad = False

    def _top(self, objective, current):
        ids = objective.allowed_ids
        return ids[torch.randint(len(ids), (len(current), self.topk), generator=objective.draw)]


class PEZ(Optimizer):
    """Continuous prefix embeddings, optimised with AdamW; every step the gradient is taken at their projection onto
    the nearest allowed tokens and applied to the continuous embeddings (Wen et al. 2023)."""
    name = "pez"

    def __init__(self, lr: float = 0.1, weight_decay: float = 0.1):
        self.lr, self.weight_decay = lr, weight_decay

    def start(self, objective, ids):
        super().start(objective, ids)
        E = objective.embeddings
        self._soft = E[torch.tensor(ids, device=E.device)].float().clone().requires_grad_(True)
        self._opt = torch.optim.AdamW([self._soft], lr=self.lr, weight_decay=self.weight_decay)

    def step(self, objective):
        ids = objective.project(self._soft.detach())
        E = objective.embeddings
        _, grad = objective.embedding_gradient(E[torch.tensor(ids, device=E.device)])
        self._opt.zero_grad()
        self._soft.grad = -grad.to(self._soft.device)        # ascent: the objective is maximised
        self._opt.step()
        if not objective.text_allowed(ids):                  # a forbidden word across tokens is never a result
            return ids, -float("inf")
        self.current = ids
        return ids, objective.score([ids])[0].item()


class GBDA(Optimizer):
    """A distribution over the allowed tokens at each position (logits Θ), trained with Adam through a
    Gumbel-softmax relaxation of the prefix; each step tries its most likely tokens and `samples` draws from it
    and returns the best of them (Guo et al. 2021)."""
    name = "gbda"

    def __init__(self, lr: float = 0.3, tau: float = 1.0, samples: int = 8, init_logit: float = 12.0):
        self.lr, self.tau, self.samples, self.init_logit = lr, tau, samples, init_logit

    def start(self, objective, ids):
        super().start(objective, ids)
        W = objective.allowed_rows()                         # Θ ranges over the allowed tokens only
        theta = torch.zeros(len(ids), W.shape[0], device=W.device)
        where = {t: j for j, t in enumerate(objective.allowed_ids.tolist())}
        for i, t in enumerate(ids):                          # start concentrated on the initial prefix
            if t in where:
                theta[i, where[t]] = self.init_logit
        self._theta = theta.requires_grad_(True)
        self._opt = torch.optim.Adam([self._theta], lr=self.lr)
        self._gen = torch.Generator().manual_seed(objective.rng.randrange(2 ** 31))

    def step(self, objective):
        W = objective.allowed_rows()
        u = torch.rand(self._theta.shape, generator=self._gen).to(self._theta.device).clamp_(1e-10, 1 - 1e-10)
        soft = torch.softmax((self._theta - torch.log(-torch.log(u))) / self.tau, dim=-1)
        emb = soft.to(W.dtype) @ W                           # W is detached: only Θ is trained
        _, grad = objective.embedding_gradient(emb.detach())
        self._opt.zero_grad()
        emb.backward(-grad.to(emb.device, emb.dtype))        # ascent, through the relaxation into Θ
        self._opt.step()
        with torch.no_grad():
            probs = torch.softmax(self._theta, dim=-1).cpu()
            tries = [probs.argmax(-1)] + [torch.multinomial(probs, 1, generator=self._gen).squeeze(-1)
                                          for _ in range(self.samples)]
        cands = list(dict.fromkeys(tuple(objective.allowed_ids[t].tolist()) for t in tries))
        cands = [list(c) for c in cands if objective.text_allowed(list(c))]
        if not cands:
            return self.current, -float("inf")
        scores = objective.score(cands)
        i = int(scores.argmax())
        self.current = cands[i]
        return self.current, scores[i].item()


OPTIMIZERS = {"gcg": GCG, "random": RandomSearch, "pez": PEZ, "gbda": GBDA}


def get(optimizer, **settings) -> Optimizer:
    """An Optimizer from a name ("gcg", "random", "pez", "gbda") or an instance; `settings` only with a name."""
    if isinstance(optimizer, Optimizer):
        if settings:
            raise ValueError(f"{sorted(settings)} configure the default optimizer; with an Optimizer instance, "
                             "set them on the instance instead")
        return optimizer
    if optimizer not in OPTIMIZERS:
        raise ValueError(f"unknown optimizer {optimizer!r}; choose from {sorted(OPTIMIZERS)} or pass an Optimizer")
    return OPTIMIZERS[optimizer](**settings)
