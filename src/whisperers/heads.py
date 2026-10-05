"""Attention heads: what each one writes into the residual stream, and which ones carry a task.

A head's WRITE is its output pushed through its slice of the attention output projection: the
part of the layer's attention update that this head is responsible for. In a pre-norm model
(Llama, Qwen, GPT-2) the writes, plus any projection bias, sum exactly to what attention adds to
the residual. Some models normalise the attention output before adding it (OLMo-2/3, Gemma-2/3
carry a `post_attention_layernorm` *and* a `post_feedforward_layernorm`); there the writes are
taken at that norm's scale with its per-position RMS held fixed, so they still sum exactly to the
update. `check_head_writes` asserts both identities on any model.

`find_heads` ranks heads the way Todd et al. (2024, arXiv 2310.15213) select function-vector
heads: patch each head's mean output from prompts WITH demonstrations into prompts whose
demonstrations have SHUFFLED answers -- same format, broken relation -- and measure how much the
correct answer's probability recovers. `function_vector` sums the top heads' mean writes.

`instruction_heads` / `instruction_vector` do the same for a task that is ASKED for ("Give me the
opposite of hot"), e.g. inside a chat turn: patch from that request into a contrast request ("Give me
a synonym of hot"), judged by the model's own answer, at the reply or at the word itself.
"""
from __future__ import annotations

import random

import torch

from .search import _blocks, _device
from .targets import Target, head_target


# ── model structure ──────────────────────────────────────────────────────────

def _config(model):
    cfg = model.config
    return cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg


def n_heads(model) -> int:
    cfg = _config(model)
    for k in ("num_attention_heads", "n_head", "num_heads"):
        if getattr(cfg, k, None):
            return int(getattr(cfg, k))
    raise ValueError("can't find the number of attention heads in the model config")


def _attention(layer):
    """(attention module, output projection) of one decoder block."""
    for name in ("self_attn", "attn", "attention"):
        attn = getattr(layer, name, None)
        if attn is not None:
            for proj in ("o_proj", "c_proj", "dense", "out_proj"):
                o = getattr(attn, proj, None)
                if o is not None:
                    return attn, o
    raise ValueError(f"{type(layer).__name__} has no softmax-attention heads "
                     "(a linear-attention or state-space layer?)")


def _post_norm(layer):
    """The norm applied to the attention OUTPUT, if this architecture has one (else None)."""
    if hasattr(layer, "post_feedforward_layernorm"):          # OLMo-2/3, Gemma-2/3
        return getattr(layer, "post_attention_layernorm", None)
    return None                                                # Llama/Qwen: that name is the pre-MLP norm


def _slice(o, h: int, hd: int) -> torch.Tensor:
    """Head h's columns of the output projection as an (hd, d_model) matrix."""
    if type(o).__name__ == "Conv1D":                           # GPT-2 stores (in, out)
        return o.weight[h * hd:(h + 1) * hd, :]
    return o.weight[:, h * hd:(h + 1) * hd].T                  # nn.Linear stores (out, in)


class HeadCapture:
    """Hooks that record, at chosen positions, each listed layer's per-head attention outputs.

    After a forward pass, `writes(L)` gives a (B, n_heads, d_model) tensor of residual-space
    writes. Used both for scoring a heads target and for patching in `find_heads`.
    """

    def __init__(self, model, layers, positions):
        self.model, self.H = model, n_heads(model)
        self.blocks = _blocks(model)
        self.positions = positions                             # (B,) tensor of token indices
        self.z, self.out, self.handles = {}, {}, []
        for L in sorted(set(layers)):
            _, o = _attention(self.blocks[L])
            self.handles.append(o.register_forward_pre_hook(self._pre(L)))
            self.handles.append(o.register_forward_hook(self._post(L)))

    def _rows(self, t):
        return t[torch.arange(t.shape[0], device=t.device), self.positions.to(t.device)]

    def _pre(self, L):
        def hook(module, args):
            self.z[L] = self._rows(args[0])
        return hook

    def _post(self, L):
        def hook(module, args, output):
            self.out[L] = self._rows(output[0] if isinstance(output, tuple) else output)
        return hook

    def writes(self, L) -> torch.Tensor:
        layer = self.blocks[L]
        _, o = _attention(layer)
        z = self.z[L]                                          # (B, n_heads * hd)
        hd = z.shape[-1] // self.H
        # in float32: bf16 per-head products would each round, and their sum drift from the layer's own
        w = torch.stack([z[:, h * hd:(h + 1) * hd].float() @ _slice(o, h, hd).float() for h in range(self.H)], dim=1)
        norm = _post_norm(layer)
        if norm is not None:
            # norm(x) = gamma * x / rms(x): linear in x once rms is fixed, so each head's share is
            # gamma * write / rms(full output). gamma read off the module itself (works for Gemma's 1+w).
            out = self.out[L].float()
            eps = float(getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-6)))
            rms = (out.pow(2).mean(-1, keepdim=True) + eps).sqrt()
            ones = torch.ones(1, out.shape[-1], device=out.device, dtype=norm.weight.dtype)
            # .forward, not norm(...): calling the module would fire any hooks registered on it
            gamma = norm.forward(ones).float()[0] * (1.0 + eps) ** 0.5
            w = w.float() * (gamma / rms)[:, None, :]
        return w

    def remove(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def check_head_writes(model, tokenizer, text: str = "The quick brown fox jumps over the lazy dog",
                      layers=None, atol: float | None = None) -> float:
    """Max |sum of head writes (+ bias) - the layer's actual attention update|, at every position.

    Raises if it exceeds `atol` (relative to the update's scale): the identity the heads target
    rests on. Run it once on a new architecture. The default tolerance follows the model's precision
    -- 4x its rounding unit, at least 1e-4 -- because the reference update is itself rounded: in
    bf16 a correct decomposition is off by ~5e-3, while a real bug (a head misassigned, a norm
    scale missed) is off by order 1.
    """
    if atol is None:
        atol = max(1e-4, 4 * torch.finfo(next(model.parameters()).dtype).eps)
    blocks = _blocks(model)
    if layers is None:                       # first, middle and last layer that HAS heads (hybrid models)
        have = [L for L in range(len(blocks)) if _has_heads(blocks[L])]
        layers = sorted({have[0], have[len(have) // 2], have[-1]})
    ids = tokenizer(text, return_tensors="pt").input_ids.to(_device(model))
    pos = torch.arange(ids.shape[1], device=ids.device)
    worst = 0.0
    for L in layers:
        cap = HeadCapture(model, [L], pos.new_zeros(1))
        layer = blocks[L]
        norm, (_, o) = _post_norm(layer), _attention(layer)
        actual = {}
        target_mod = norm if norm is not None else o
        h_ = target_mod.register_forward_hook(lambda m, a, out: actual.__setitem__("x", out[0] if isinstance(out, tuple) else out))
        try:
            for p in range(ids.shape[1]):
                cap.positions = torch.tensor([p], device=ids.device)
                model(input_ids=ids, use_cache=False)
                summed = cap.writes(L).sum(1)[0].float()
                bias = getattr(o, "bias", None)
                if bias is not None and norm is None:
                    summed = summed + bias.float()
                ref = actual["x"][0, p].float()
                worst = max(worst, float((summed - ref).abs().max() / ref.abs().max().clamp_min(1e-6)))
        finally:
            h_.remove()
            cap.remove()
    if worst > atol:
        raise AssertionError(f"head writes do not sum to the attention update (rel err {worst:.2e})")
    return worst


# ── tasks: demonstrations, shuffled demonstrations, answers ──────────────────

def _icl_prompt(pairs, x, template):
    shots = "".join(template.format(x=a, y=b) for a, b in pairs)
    return shots + template.split("{y}")[0].rstrip(" ").format(x=x)


def _first_token(tokenizer, answer: str) -> int:
    return tokenizer(" " + answer, add_special_tokens=False).input_ids[0]


def find_heads(model, tokenizer, pairs, *, n_demos: int = 5, n_prompts: int = 20,
               template: str = "{x} -> {y}\n", seed: int = 0, layers=None) -> list[tuple[tuple[int, int], float]]:
    """Heads ranked by causal effect on an in-context task, Todd-style. [((layer, head), effect), ...]

    For `n_prompts` queries: a CLEAN prompt (n_demos real demonstrations + query) and a CORRUPTED
    one (the same demonstrations with their answers shuffled among themselves). Each head's mean
    output over the clean prompts is patched, one head at a time, into the corrupted prompts at the
    last position; its effect is the mean rise in the probability of the correct answer's first
    token. Shuffled answers keep the format and break the relation, so heads that only carry the
    format score near zero -- the placebo is built into the measurement.
    """
    rng = random.Random(seed)
    pairs = list(pairs)
    rng.shuffle(pairs)
    blocks = _blocks(model)
    layers = list(layers) if layers is not None else list(range(len(blocks)))
    usable = [L for L in layers if _has_heads(blocks[L])]
    need = n_demos + n_prompts
    if len(pairs) < need:
        raise ValueError(f"need at least {need} pairs, got {len(pairs)}")
    jobs = []
    for i in range(n_prompts):                                 # queries pairs[:n], demos from the rest
        demos = rng.sample(pairs[n_prompts:], n_demos)
        x, y = pairs[i]
        answers = [b for _, b in demos]
        shuffled = answers[:]
        while len(answers) > 1 and shuffled == answers:
            rng.shuffle(shuffled)
        clean = tokenizer(_icl_prompt(demos, x, template)).input_ids
        corrupt = tokenizer(_icl_prompt([(a, s) for (a, _), s in zip(demos, shuffled)], x, template)).input_ids
        jobs.append((clean, len(clean) - 1, corrupt, len(corrupt) - 1, _first_token(tokenizer, y)))
    return _rank(model, jobs, usable)


def _rank(model, jobs, usable) -> list[tuple[tuple[int, int], float]]:
    """The causal-patching core shared by `find_heads` and `instruction_heads`.

    jobs: [(clean_ids, clean_pos, corrupt_ids, corrupt_pos, answer_token)]. Each head's mean output
    (pre-projection z) at clean_pos over the clean prompts is patched, one head at a time, into each
    corrupted prompt at corrupt_pos; the effect is the mean rise in P(answer_token) at the corrupted
    prompt's LAST position (the next token the model would write).
    """
    blocks, H, dev = _blocks(model), n_heads(model), _device(model)
    # 1. mean per-head output (pre-projection z) at the chosen position over the clean prompts
    mean_z = {L: None for L in usable}
    with torch.no_grad():
        for clean, cpos, _, _, _ in jobs:
            cap = HeadCapture(model, usable, torch.tensor([cpos], device=dev))
            try:
                model(input_ids=torch.tensor([clean], device=dev), use_cache=False)
            finally:
                cap.remove()
            for L in usable:
                z = cap.z[L][0].float()
                mean_z[L] = z if mean_z[L] is None else mean_z[L] + z
    mean_z = {L: z / len(jobs) for L, z in mean_z.items()}

    # 2. patch each head's mean into the corrupted prompts, one batch row per head of a layer
    effect = {(L, h): 0.0 for L in usable for h in range(H)}
    with torch.no_grad():
        for _, _, corrupt, pos, ans in jobs:
            ids = torch.tensor([corrupt], device=dev)
            base = model(input_ids=ids, use_cache=False).logits[0, -1].float().softmax(-1)[ans].item()
            for L in usable:
                _, o = _attention(blocks[L])
                hd = mean_z[L].shape[-1] // H
                batch = ids.repeat(H, 1)

                def patch(module, args, L=L, hd=hd, pos=pos):
                    z = args[0].clone()
                    for h in range(H):
                        z[h, pos, h * hd:(h + 1) * hd] = mean_z[L][h * hd:(h + 1) * hd].to(z.dtype)
                    return (z,) + tuple(args[1:])
                handle = o.register_forward_pre_hook(patch)
                try:
                    p = model(input_ids=batch, use_cache=False).logits[:, -1].float().softmax(-1)[:, ans]
                finally:
                    handle.remove()
                for h in range(H):
                    effect[(L, h)] += (p[h].item() - base) / len(jobs)
    return sorted(effect.items(), key=lambda kv: -kv[1])


# ── instructed tasks: heads that carry "do X" when the request is explicit, e.g. in a chat turn ──

def _locate(tokenizer, inner: str, template: str | None, word: str) -> tuple[list[int], int]:
    """Token ids of `inner` as the model sees it (inside `template`), and the index of the token that
    ends the LAST occurrence of `word` in it."""
    from .search import _split, wrap
    full = wrap(template, inner)
    start = (len(_split(template)[0]) if template else 0) + inner.rindex(word)
    end = start + len(word)                                    # the word's last character is at end - 1
    enc = tokenizer(full, add_special_tokens=template is None, return_offsets_mapping=True)
    for i, (a, b) in enumerate(enc["offset_mapping"]):
        if a < end <= b:
            return enc["input_ids"], i
    raise ValueError(f"can't find {word!r} in the tokenized prompt")


def _instruction_jobs(model, tokenizer, words, instruction, contrast, template, at):
    """Clean/contrast prompts per word, the position to read/patch, and the model's OWN answer."""
    if at not in ("reply", "word"):
        raise ValueError(f"at must be 'reply' or 'word', not {at!r}")
    dev = _device(model)
    jobs, kept = [], []
    with torch.no_grad():
        for w in words:
            clean, cpos = _locate(tokenizer, instruction.format(x=w), template, w)
            corrupt, kpos = _locate(tokenizer, contrast.format(x=w), template, w)
            if at == "reply":
                cpos, kpos = len(clean) - 1, len(corrupt) - 1
            a = model(input_ids=torch.tensor([clean], device=dev), use_cache=False).logits[0, -1].argmax().item()
            b = model(input_ids=torch.tensor([corrupt], device=dev), use_cache=False).logits[0, -1].argmax().item()
            if a == b:                                         # both requests start the same reply: no signal
                continue
            jobs.append((clean, cpos, corrupt, kpos, a))
            kept.append(w)
    if not jobs:
        raise ValueError("every word got the same first reply token under both requests; nothing to rank")
    return jobs, kept


def instruction_heads(model, tokenizer, words, instruction: str, contrast: str, *, template: str | None = None,
                      at: str = "reply", layers=None) -> list[tuple[tuple[int, int], float]]:
    """Heads ranked by causal effect on an EXPLICITLY REQUESTED task, e.g. inside a chat turn.

    For each word, a CLEAN request (`instruction`, e.g. "Give me the opposite of {x}. Answer with one
    word.") and a CONTRAST request in the same frame asking for something else ("... a synonym of
    {x} ..."), both inside `template` (e.g. `chat_template(tok, thinking=False)`). The answer key is
    the model's OWN first reply token to the clean request -- no dataset answers -- and words where
    both requests start the same reply are dropped. Each head's mean output over the clean requests,
    taken `at` the "reply" (the last token, where the reply is produced) or the "word" (the word's
    last token, where the model reads the input), is patched into the contrast requests at the same
    place; its effect is the rise in P(clean answer) at the reply. Same output as `find_heads`.
    """
    blocks = _blocks(model)
    layers = list(layers) if layers is not None else list(range(len(blocks)))
    usable = [L for L in layers if _has_heads(blocks[L])]
    jobs, _ = _instruction_jobs(model, tokenizer, words, instruction, contrast, template, at)
    return _rank(model, jobs, usable)


def instruction_vector(model, tokenizer, words, instruction: str, contrast: str, *, k: int = 10,
                       template: str | None = None, at: str = "reply", ranked=None, layers=None) -> Target:
    """`function_vector` for an explicitly requested task: the summed mean writes of the k most causal
    heads (`instruction_heads`) over the clean requests, at the same place. Aim `whisper` at it with
    the same `template` and `read="reply"` (at="reply") or `read="prompt"` (at="word"), and probes
    that are just the bare inputs: the prefix then has to stand in for the request itself.
    `meta`: ranking, the raw vector, `at`, the words kept, and an injection layer (~1/3 depth).
    """
    jobs, kept = _instruction_jobs(model, tokenizer, words, instruction, contrast, template, at)
    if ranked is None:
        blocks = _blocks(model)
        layers = list(layers) if layers is not None else list(range(len(blocks)))
        ranked = _rank(model, jobs, [L for L in layers if _has_heads(blocks[L])])
    top = [hl for hl, _ in ranked[:k]]
    dev, total = _device(model), None
    with torch.no_grad():
        for clean, cpos, _, _, _ in jobs:
            cap = HeadCapture(model, sorted({L for L, _ in top}), torch.tensor([cpos], device=dev))
            try:
                model(input_ids=torch.tensor([clean], device=dev), use_cache=False)
                w = sum(cap.writes(L)[0, h].float() for L, h in top)
            finally:
                cap.remove()
            total = w if total is None else total + w
    fv = (total / len(jobs)).cpu()
    t = head_target(fv, top, name=f"instruction_vector_{at}", model_id=getattr(model, "name_or_path", None))
    t.meta.update({"ranked": [((L, h), e) for (L, h), e in ranked[:50]], "vector": fv, "at": at,
                   "words": kept, "inject_layer": len(_blocks(model)) // 3, "k": k})
    return t


def _has_heads(layer) -> bool:
    try:
        _attention(layer)
        return True
    except ValueError:
        return False


def function_vector(model, tokenizer, pairs, *, k: int = 10, n_demos: int = 5, n_prompts: int = 20,
                    template: str = "{x} -> {y}\n", seed: int = 0, ranked=None) -> Target:
    """Todd et al.'s function vector: the summed mean writes of the k most causal heads.

    Returns a heads Target -- aim `whisper` at it to search for tokens that make those heads emit
    the function vector. `meta` carries the ranking, the raw vector (`meta["vector"]`, for injecting
    it) and Todd's suggested injection layer, about a third of the way in.
    """
    ranked = ranked or find_heads(model, tokenizer, pairs, n_demos=n_demos, n_prompts=n_prompts,
                                  template=template, seed=seed)
    top = [hl for hl, _ in ranked[:k]]
    rng = random.Random(seed + 1)
    pairs = list(pairs)
    dev = _device(model)
    layers = sorted({L for L, _ in top})
    total = None
    with torch.no_grad():
        for _ in range(n_prompts):
            demos = rng.sample(pairs, n_demos + 1)
            prompt = _icl_prompt(demos[:-1], demos[-1][0], template)
            ids = tokenizer(prompt, return_tensors="pt").input_ids.to(dev)
            cap = HeadCapture(model, layers, torch.tensor([ids.shape[1] - 1], device=dev))
            try:
                model(input_ids=ids, use_cache=False)
                w = sum(cap.writes(L)[0, h].float() for L, h in top)
            finally:
                cap.remove()
            total = w if total is None else total + w
    fv = (total / n_prompts).cpu()
    t = head_target(fv, top, name="function_vector", model_id=getattr(model, "name_or_path", None))
    t.meta.update({"ranked": [((L, h), e) for (L, h), e in ranked[:50]], "vector": fv,
                   "inject_layer": len(_blocks(model)) // 3, "k": k})
    return t
