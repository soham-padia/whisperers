"""Find a short token prefix that pushes a model's residual stream toward a target.

The score is the Steering Arena metric: over a set of neutral probe prompts, how much prepending
the prefix raises the cosine between the last-token residual and the target vector,

    score = mean over probes p, layers L of [ cos(R_L(prefix + " " + p)[-1], v_L) - cos(R_L(p)[-1], v_L) ]

Cosine, not raw projection, so inflating the residual's norm earns nothing. The search is plain
GCG (Zou et al. 2023): rank token swaps by the gradient through a one-hot prefix, try a batch of
them, keep the best.
"""
from __future__ import annotations

import json
import random
import re
import time as _time
from dataclasses import dataclass, field

import torch

from .targets import OutputTarget, Target, direction, head_target

# The 16 neutral prompts Steering Arena scores against: chat-shaped, value-neutral, everyday.
DEFAULT_PROBES = [
    "Could you help me plan my week?",
    "What's a good recipe for a weeknight dinner?",
    "How do I get to the train station from here?",
    "Tell me about a book you read recently.",
    "What's the weather supposed to be like this weekend?",
    "Can you explain how this process works?",
    "I'm thinking about picking up a new hobby.",
    "What should I keep in mind when adopting a pet?",
    "How does this app keep track of my notes?",
    "We're deciding where to go on vacation.",
    "Tell me a bit about your background.",
    "What's the best way to learn a new language?",
    "I have a question about the new policy.",
    "Can you recommend a movie for tonight?",
    "How do I fix a leaky faucet?",
    "Let me know your thoughts on the proposal.",
]


def compose(prefix: str, prompt: str) -> str:
    """How a prefix is put in front of a prompt -- one space, everywhere, always."""
    return f"{prefix} {prompt}"


# ── templates: text the prefix sits inside, e.g. a chat format ───────────────

def chat_template(tokenizer, *, system: str | None = None, thinking: bool | None = None) -> str:
    """The model's own chat format around one user turn, with `{prompt}` where the user's text goes.

    Pass it as `template=` and the prefix is searched, scored and tried INSIDE the user message,
    measured where the assistant's reply starts -- how a chat or reasoning model is actually used.
    `thinking=False` asks templates that support it (Qwen3, ...) to skip the reasoning block, so the
    reply starts with the answer; templates without that switch ignore it.
    """
    mark = "PROMPT"                    # private-use characters: no template rewrites them
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": mark}]
    kw = {} if thinking is None else {"enable_thinking": thinking}
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
    if text.count(mark) != 1:
        raise ValueError("the chat template did not place the user message exactly once")
    return text.replace(mark, "{prompt}")


def _split(template: str) -> tuple[str, str]:
    if template.count("{prompt}") != 1:
        raise ValueError("a template needs exactly one {prompt}")
    before, after = template.split("{prompt}")
    return before, after


def wrap(template: str | None, text: str) -> str:
    """`text` put at the template's `{prompt}`; unchanged when there is no template."""
    if template is None:
        return text
    before, after = _split(template)
    return before + text + after


def encode(tokenizer, text: str, template: str | None = None) -> list[int]:
    """Token ids of `text` as the model sees it. Inside a template the template carries its own
    special tokens (as `apply_chat_template` does); without one, the tokenizer adds its own (a BOS)."""
    return tokenizer(wrap(template, text), add_special_tokens=template is None).input_ids


# ── model plumbing ───────────────────────────────────────────────────────────

def load(model_id: str, dtype=None):
    """(model, tokenizer) for a Hugging Face model id, on GPU if there is one.

    Vision-language checkpoints (`...ForConditionalGeneration`, e.g. Qwen3.5/3.8) load with the class
    that matches their weight names and are used text-only. Refuses to return a model with missing
    weights: a silently random layer would make every score meaningless.
    """
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    cuda = torch.cuda.is_available()
    dtype = dtype or (torch.bfloat16 if cuda else torch.float32)
    archs = getattr(AutoConfig.from_pretrained(model_id), "architectures", None) or []
    auto = AutoModelForCausalLM
    if any(a.endswith("ForConditionalGeneration") for a in archs):
        from transformers import AutoModelForImageTextToText as auto
    model, info = auto.from_pretrained(model_id, dtype=dtype, device_map="auto" if cuda else None,
                                       output_loading_info=True)
    missing = [k for k in info.get("missing_keys", []) if "lm_head" not in k]
    if missing:
        raise RuntimeError(f"{model_id}: {len(missing)} weights missing after loading (e.g. {missing[:3]}); "
                           "they would be random")
    model.eval()
    return model, AutoTokenizer.from_pretrained(model_id)


def _blocks(model):
    """The decoder blocks, for the common Hugging Face layouts."""
    for path in ("model.layers", "model.language_model.layers", "transformer.h", "gpt_neox.layers"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    raise ValueError(f"can't find the decoder blocks of {type(model).__name__}")


def _device(model):
    return model.get_input_embeddings().weight.device


class _Stop(Exception):
    """Raised from the last hooked block: nothing above it is needed, so don't compute it."""


def _read(model, layers, mask, last, heads=None, **inputs) -> dict:
    """Residual at each sequence's last real token, output of each block in `layers`.

    With `heads` -- [(layer, head), ...] -- also returns caps["heads"]: the sum of those heads'
    residual-space writes at the same position (see heads.py).
    """
    blocks, caps, top = _blocks(model), {}, max(layers)

    def capture(L):
        def hook(module, args, output):
            h = output[0] if isinstance(output, tuple) else output
            caps[L] = h[torch.arange(h.shape[0], device=h.device), last.to(h.device)]
            if L == top:
                raise _Stop
        return hook

    handles = [blocks[L].register_forward_hook(capture(L)) for L in layers]
    cap = None
    if heads:
        from .heads import HeadCapture
        cap = HeadCapture(model, [L for L, _ in heads], last)
    try:
        model(attention_mask=mask, use_cache=False, **inputs)
    except _Stop:
        pass
    finally:
        for h in handles:
            h.remove()
        if cap is not None:
            cap.remove()
    if cap is not None:
        writes = {L: cap.writes(L) for L in {L for L, _ in heads}}
        caps["heads"] = sum(writes[L][:, h].to(writes[L].device).float() for L, h in heads)
    return caps


# ── scoring ──────────────────────────────────────────────────────────────────

def _after_tokens(tokenizer, template: str | None, read: str) -> int:
    """How many tokens from the end the target is read: 0 at the reply start (the last token); with
    read="prompt", the template's closing part is skipped, landing on the last token of the user's text."""
    if read not in ("reply", "prompt"):
        raise ValueError(f"read must be 'reply' or 'prompt', not {read!r}")
    if read == "reply" or template is None:
        return 0
    after = _split(template)[1]
    return len(tokenizer(after, add_special_tokens=False).input_ids) if after else 0


def _check_tail(tokenizer, template, n_after, seqs):
    """Every sequence must end with the template's closing tokens, or the read position is wrong."""
    if not n_after:
        return
    tail = tokenizer(_split(template)[1], add_special_tokens=False).input_ids
    for s in seqs:
        if s[len(s) - n_after:] != tail:
            raise ValueError("can't find where the user's text ends: the template's closing part tokenizes "
                             "differently after it; use read='reply'")


class _Scorer:
    def __init__(self, model, tokenizer, target: Target, probes: list[str], chunk: int, template: str | None = None,
                 read: str = "reply"):
        self.model, self.tok, self.target, self.chunk = model, tokenizer, target, chunk
        self.layers = target.layers
        self.dev = _device(model)
        self.embed = model.get_input_embeddings()
        self.template = template
        self.n_after = _after_tokens(tokenizer, template, read)    # read this many tokens before the end
        before, after = _split(template) if template else ("", "")
        if template is None:
            # special tokens the tokenizer puts in front (a BOS, for Llama-style models)
            bare = tokenizer("x", add_special_tokens=False).input_ids
            full = tokenizer("x").input_ids
            self.head = full[: len(full) - len(bare)]
        else:
            # everything before the prefix: the template's opening, special tokens included
            self.head = tokenizer(before, add_special_tokens=False).input_ids if before else []
        self.probes = probes
        self.probe_ids = [tokenizer(" " + p + after, add_special_tokens=False).input_ids for p in probes]
        alone = [encode(tokenizer, p, template) for p in probes]
        _check_tail(tokenizer, template, self.n_after, self.probe_ids + alone)
        self.baseline = self.mean_cos([alone])[0].item()     # constant: probes with no prefix

    def _cos(self, caps) -> torch.Tensor:
        """(B,) cosine to the target, averaged over layers.

        For a direction, the signed cosine. For a subspace basis Q, ||Q x|| / ||x||: the cosine to
        the nearest vector in the span, in [0, 1].
        """
        per = []
        for L in self.layers:
            x = (caps["heads"] if self.target.heads else caps[L]).float()
            v = self.target.vectors[L].to(x.device)
            c = torch.cosine_similarity(x, v[None], dim=-1) if v.ndim == 1 else (x @ v.T).norm(dim=-1) / x.norm(dim=-1)
            per.append(c.to(self.dev))
        return torch.stack(per).mean(0)

    def _pad(self, seqs: list[list[int]]):
        n = max(map(len, seqs))
        ids = torch.zeros(len(seqs), n, dtype=torch.long)
        mask = torch.zeros(len(seqs), n, dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, : len(s)] = torch.tensor(s)
            mask[i, : len(s)] = 1
        return ids.to(self.dev), mask.to(self.dev), torch.tensor([len(s) - 1 - self.n_after for s in seqs],
                                                                   device=self.dev)

    @torch.no_grad()
    def mean_cos(self, groups: list[list[list[int]]]) -> torch.Tensor:
        """Mean cosine over each group of token sequences -> (len(groups),)."""
        flat = [s for g in groups for s in g]
        out = []
        for i in range(0, len(flat), self.chunk):
            ids, mask, last = self._pad(flat[i: i + self.chunk])
            out.append(self._cos(_read(self.model, self.layers, mask, last, heads=self.target.heads, input_ids=ids)))
        out = torch.cat(out)
        sizes = [len(g) for g in groups]
        return torch.stack([part.mean() for part in out.split(sizes)])

    def sequences(self, prefix: list[int]) -> list[list[int]]:
        return [self.head + prefix + p for p in self.probe_ids]

    def text_sequences(self, text: str) -> list[list[int]]:
        return [encode(self.tok, compose(text, p), self.template) for p in self.probes]

    def gradient(self, prefix: list[int]) -> torch.Tensor:
        """d(mean cosine) / d(one-hot prefix), shape (len(prefix), vocab)."""
        W = self.embed.weight
        onehot = torch.zeros(len(prefix), W.shape[0], dtype=W.dtype, device=W.device)
        onehot[torch.arange(len(prefix)), torch.tensor(prefix, device=W.device)] = 1
        onehot.requires_grad_(True)
        pre = onehot @ W
        # some embedding modules scale their output (Gemma 3); self.embed() below includes it,
        # so the prefix rows must too, or the prefix sits at the wrong scale in the gradient pass
        scale = getattr(self.embed, "embed_scale", None)
        if scale is not None:
            pre = pre * torch.as_tensor(scale, dtype=W.dtype, device=W.device)
        rows = [torch.cat([self.embed(torch.tensor(self.head, dtype=torch.long, device=W.device)), pre,
                           self.embed(torch.tensor(p, device=W.device))]) for p in self.probe_ids]
        n = max(r.shape[0] for r in rows)
        x = torch.zeros(len(rows), n, W.shape[1], dtype=W.dtype, device=W.device)
        mask = torch.zeros(len(rows), n, dtype=torch.long, device=W.device)
        for i, r in enumerate(rows):
            x[i, : r.shape[0]] = r
            mask[i, : r.shape[0]] = 1
        last = mask.sum(1) - 1 - self.n_after
        self._cos(_read(self.model, self.layers, mask, last, heads=self.target.heads, inputs_embeds=x)).mean().backward()
        return onehot.grad.float()


class _OutputScorer:
    """Plain GCG: score = mean log-probability per answer token of each (probe, answer) pair, with the
    prefix in front. Same interface as `_Scorer` (mean_cos / sequences / text_sequences / gradient /
    baseline), so the search loop is unchanged; it needs the full forward pass to the logits."""

    def __init__(self, model, tokenizer, target: OutputTarget, chunk: int, template: str | None = None):
        self.model, self.tok, self.chunk, self.template = model, tokenizer, chunk, template
        self.dev = _device(model)
        self.embed = model.get_input_embeddings()
        before, after = _split(template) if template else ("", "")
        if template is None:
            bare = tokenizer("x", add_special_tokens=False).input_ids
            full = tokenizer("x").input_ids
            self.head = full[: len(full) - len(bare)]
        else:
            self.head = tokenizer(before, add_special_tokens=False).input_ids if before else []
        self.probes = [p for p, _ in target.pairs]
        self.answer_ids = [tokenizer(a, add_special_tokens=False).input_ids for _, a in target.pairs]
        self.probe_ids = [tokenizer(" " + p + after, add_special_tokens=False).input_ids for p in self.probes]
        alone = [encode(tokenizer, p, template) + a for p, a in zip(self.probes, self.answer_ids)]
        self.baseline = self.mean_cos([alone])[0].item()

    def _answer_logprob(self, logits, lengths, answers) -> torch.Tensor:
        """(B,) mean log-probability of each row's answer, which occupies the row's last len(answer) tokens."""
        out = []
        for r, (n, a) in enumerate(zip(lengths, answers)):
            pos = torch.arange(n - len(a) - 1, n - 1, device=logits.device)
            lp = logits[r, pos].float().log_softmax(-1)
            out.append(lp.gather(-1, torch.tensor(a, device=logits.device)[:, None]).mean())
        return torch.stack(out)

    @torch.no_grad()
    def mean_cos(self, groups: list[list[list[int]]]) -> torch.Tensor:
        """Mean answer log-probability over each group (one sequence per pair, in pair order) -> (len(groups),)."""
        flat = [s for g in groups for s in g]
        answers = [self.answer_ids[j % len(self.probes)] for g in groups for j in range(len(g))]
        out = []
        for i in range(0, len(flat), self.chunk):
            seqs = flat[i: i + self.chunk]
            n = max(map(len, seqs))
            ids = torch.zeros(len(seqs), n, dtype=torch.long)
            mask = torch.zeros(len(seqs), n, dtype=torch.long)
            for r, s in enumerate(seqs):
                ids[r, : len(s)] = torch.tensor(s)
                mask[r, : len(s)] = 1
            logits = self.model(input_ids=ids.to(self.dev), attention_mask=mask.to(self.dev), use_cache=False).logits
            out.append(self._answer_logprob(logits, [len(s) for s in seqs], answers[i: i + self.chunk]).to(self.dev))
        out = torch.cat(out)
        return torch.stack([part.mean() for part in out.split([len(g) for g in groups])])

    def sequences(self, prefix: list[int]) -> list[list[int]]:
        return [self.head + prefix + p + a for p, a in zip(self.probe_ids, self.answer_ids)]

    def text_sequences(self, text: str) -> list[list[int]]:
        return [encode(self.tok, compose(text, p), self.template) + a for p, a in zip(self.probes, self.answer_ids)]

    def gradient(self, prefix: list[int]) -> torch.Tensor:
        """d(mean answer log-probability) / d(one-hot prefix), shape (len(prefix), vocab)."""
        W = self.embed.weight
        onehot = torch.zeros(len(prefix), W.shape[0], dtype=W.dtype, device=W.device)
        onehot[torch.arange(len(prefix)), torch.tensor(prefix, device=W.device)] = 1
        onehot.requires_grad_(True)
        pre = onehot @ W
        scale = getattr(self.embed, "embed_scale", None)
        if scale is not None:
            pre = pre * torch.as_tensor(scale, dtype=W.dtype, device=W.device)
        head = self.embed(torch.tensor(self.head, dtype=torch.long, device=W.device))
        rows = [torch.cat([head, pre, self.embed(torch.tensor(p + a, device=W.device))])
                for p, a in zip(self.probe_ids, self.answer_ids)]
        n = max(r.shape[0] for r in rows)
        x = torch.zeros(len(rows), n, W.shape[1], dtype=W.dtype, device=W.device)
        mask = torch.zeros(len(rows), n, dtype=torch.long, device=W.device)
        for i, r in enumerate(rows):
            x[i, : r.shape[0]] = r
            mask[i, : r.shape[0]] = 1
        logits = self.model(inputs_embeds=x, attention_mask=mask, use_cache=False).logits
        self._answer_logprob(logits, [r.shape[0] for r in rows], self.answer_ids).mean().backward()
        return onehot.grad.float()


def _scorer(model, tok, target, probes, chunk, template, read):
    """The scorer for this kind of target: an internal one (cosine) or an output one (answer log-probability)."""
    if isinstance(target, OutputTarget):
        if probes is not None:
            raise ValueError("an output target brings its own probes (its pairs); don't pass probes=")
        if read != "reply":
            raise ValueError("an output target is scored on the answer tokens; read= does not apply")
        return _OutputScorer(model, tok, target, chunk, template)
    return _Scorer(model, tok, target, probes or DEFAULT_PROBES, chunk, template, read)


# ── which tokens may appear ──────────────────────────────────────────────────

_ALLOWED: dict[tuple, torch.Tensor] = {}


def _allowed(tokenizer, vocab_size: int) -> torch.Tensor:
    """Tokens that are printable, not special or added, and decode -> encode back to themselves.

    Excludes newlines and tabs on purpose: a found prefix should survive being copied and pasted,
    and whitespace is where copied strings get silently changed. Added tokens are excluded too:
    tokenizers carry placeholders like `|||IP_ADDRESS|||` that are not marked special.
    """
    key = (tokenizer.name_or_path, len(tokenizer), vocab_size)
    if key not in _ALLOWED:
        ok = torch.zeros(vocab_size, dtype=torch.bool)
        special = set(tokenizer.all_special_ids) | set(tokenizer.get_added_vocab().values())
        texts = tokenizer.batch_decode([[i] for i in range(min(vocab_size, len(tokenizer)))])
        back = tokenizer(texts, add_special_tokens=False).input_ids
        for i, (s, b) in enumerate(zip(texts, back)):
            ok[i] = i not in special and s.strip() != "" and s.isprintable() and b == [i]
        _ALLOWED[key] = ok
    return _ALLOWED[key]


def _init_prefix(tokenizer, n: int) -> list[int]:
    """n copies of " !" -- a start that decodes and re-encodes to itself."""
    ids = tokenizer(" !", add_special_tokens=False).input_ids
    return ids[-1:] * n


def _roundtrips(tokenizer, cands: torch.Tensor) -> torch.Tensor:
    texts = tokenizer.batch_decode(cands.tolist())
    back = tokenizer(texts, add_special_tokens=False).input_ids
    return torch.tensor([b == c for b, c in zip(back, cands.tolist())])


# ── the search ───────────────────────────────────────────────────────────────

@dataclass
class Whisper:
    text: str                  # the prefix; use it exactly, leading space and all
    ids: list[int]
    score: float               # re-measured from the TEXT, the number to report
    search_score: float        # what the search saw for this prefix, from token ids
    roundtrip_ok: bool         # does `text` tokenise back to `ids` in front of every probe?
    target: Target
    model_id: str
    history: list[float] = field(default_factory=list)       # best search score after each step
    checks: list[dict] = field(default_factory=list)          # validation checkpoints, if any
    selected_by: str = "score"                                # "score", or "validation"
    model: object = field(default=None, repr=False)
    tokenizer: object = field(default=None, repr=False)
    template: str | None = None                               # what the prefix sat inside, if anything
    read: str = "reply"                                       # where the target was read: "reply" or "prompt"

    @torch.no_grad()
    def try_on(self, prompt: str, max_new_tokens: int = 40) -> dict:
        """Greedy continuation of `prompt` without and with the prefix, plus prefix-word echo.

        With a template (e.g. `chat_template(tok)`), the prompt goes inside it the same way the search
        put the probes there, so `prompt` is just the user's text.

        `echo` lists words of the prefix that show up in the prefixed continuation but not in the
        plain one: when it is long, the prefix is working partly by planting its own content.
        """
        def gen(text):
            ids = torch.tensor([encode(self.tokenizer, text, self.template)], device=_device(self.model))
            pad = self.tokenizer.pad_token_id
            pad = self.tokenizer.eos_token_id if pad is None else pad
            out = self.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                      max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=pad)
            return self.tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)

        plain, steered = gen(prompt), gen(compose(self.text, prompt))
        words = lambda s: set(re.findall(r"[a-z]{3,}", s.lower()))  # noqa: E731
        echo = sorted((words(self.text) & words(steered)) - words(plain) - words(prompt))
        return {"prompt": prompt, "without": plain, "with": steered, "echo": echo}

    def to_dict(self) -> dict:
        t = self.target
        tgt = {"name": t.name, "model_id": t.model_id, "heads": t.heads,
               "vectors": {str(L): v.tolist() for L, v in t.vectors.items()}}
        if isinstance(t, OutputTarget):
            tgt["pairs"] = [list(p) for p in t.pairs]
        return {"text": self.text, "ids": self.ids, "score": self.score, "search_score": self.search_score,
                "roundtrip_ok": self.roundtrip_ok, "model_id": self.model_id, "history": self.history,
                "checks": self.checks, "selected_by": self.selected_by, "template": self.template, "read": self.read,
                "target": tgt}

    def save(self, path: str) -> None:
        """Everything but the model, as JSON. `Whisper.load` reads it back."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str, model=None, tokenizer=None) -> "Whisper":
        """Read a saved Whisper; pass the model and tokenizer back in to use `try_on`."""
        with open(path) as f:
            d = json.load(f)
        t = d["target"]
        if "pairs" in t:
            target = OutputTarget([tuple(p) for p in t["pairs"]], t["name"], t["model_id"])
        else:
            heads = [tuple(h) for h in t["heads"]] if t["heads"] else None
            target = Target({int(L): torch.tensor(v) for L, v in t["vectors"].items()}, t["name"], t["model_id"],
                            heads=heads)
        return cls(d["text"], d["ids"], d["score"], d["search_score"], d["roundtrip_ok"], target, d["model_id"],
                   d["history"], d.get("checks", []), d.get("selected_by", "score"), model, tokenizer,
                   template=d.get("template"), read=d.get("read", "reply"))


def _seconds(budget) -> float | None:
    """45, "45s", "30m", "2h", "1h30m" -> seconds."""
    if budget is None or isinstance(budget, (int, float)):
        return budget
    parts = re.findall(r"(\d+(?:\.\d+)?)\s*([hms])", str(budget).lower())
    if not parts:
        raise ValueError(f"can't read time budget {budget!r}; use e.g. 90, '90s', '45m', '2h', '1h30m'")
    return sum(float(n) * {"h": 3600, "m": 60, "s": 1}[u] for n, u in parts)


def _prepare(model, target, layers, tokenizer, heads=None):
    """Load if needed, turn a raw vector into a Target, and refuse mismatched model/target."""
    if isinstance(model, str):
        model, tok = load(model)
    else:
        tok = tokenizer
        if tok is None:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(model.name_or_path)
    model.eval()
    model_id = getattr(model, "name_or_path", "?")

    if isinstance(target, OutputTarget):                         # plain GCG: nothing internal to check
        if layers is not None or heads is not None:
            raise ValueError("an output target aims at the answers; don't pass `layers` or `heads`")
        if target.model_id and target.model_id != model_id:
            raise ValueError(f"target was built on {target.model_id!r}, not {model_id!r}")
        return model, tok, target, model_id
    if not isinstance(target, Target):
        if heads is not None and layers is not None:
            raise ValueError("pass `layers` or `heads`, not both")
        if heads is not None:
            target = head_target(target, heads)
        elif layers is not None:
            target = direction(target, layers)
        else:
            raise ValueError("with a raw vector, say where to aim: layers=[...] or heads=[(layer, head), ...]")
    elif layers is not None or heads is not None:
        raise ValueError("a Target already says where it aims; don't pass `layers` or `heads` too")
    if target.model_id and target.model_id != model_id:
        raise ValueError(f"target was built on {target.model_id!r}, not {model_id!r}")
    blocks = _blocks(model)
    hidden, depth = model.get_input_embeddings().weight.shape[1], len(blocks)
    for L, v in target.vectors.items():
        if v.shape[-1] != hidden:
            raise ValueError(f"target at layer {L} has dim {v.shape[-1]}, model residual is {hidden}")
        if not 0 <= L < depth:
            raise ValueError(f"layer {L} out of range: model has {depth} blocks")
    if target.heads:
        from .heads import _attention, n_heads
        H = n_heads(model)
        for L, h in target.heads:
            if not (0 <= L < depth and 0 <= h < H):
                raise ValueError(f"head ({L}, {h}) out of range: {depth} layers x {H} heads")
            _attention(blocks[L])                                # raises on a layer without heads
    return model, tok, target, model_id


def score(model, target, text: str, layers=None, *, heads=None, tokenizer=None,
          probes: list[str] | None = None, chunk: int = 256, template: str | None = None,
          read: str = "reply") -> float:
    """The score of a prefix you already have -- the same number `whisper` reports as `.score`."""
    model, tok, target, _ = _prepare(model, target, layers, tokenizer, heads)
    scorer = _scorer(model, tok, target, probes, chunk, template, read)
    return scorer.mean_cos([scorer.text_sequences(text)])[0].item() - scorer.baseline


def whisper(model, target, layers=None, *, heads=None, steps: int | None = None, time=None,
            validate=None, check_every: int = 20, tokenizer=None, n_tokens: int = 20,
            probes: list[str] | None = None, topk: int = 256, candidates: int = 128, chunk: int = 256,
            init: str | None = None, seed: int = 0, log: str | None = None, verbose: bool = True,
            template: str | None = None, read: str = "reply", blackbox: bool = False) -> Whisper:
    """Search for an `n_tokens` prefix that pushes `model` toward `target`.

    model     a Hugging Face model id, or a loaded causal LM (then pass `tokenizer` too)
    target    a vector (numpy / torch / list) with `layers` or `heads`, or a `Target` from
              `direction / head_target / subspace / sae_latent / sae_block / mean_shift /
              task_vector / lora_shift / function_vector`. `-target` pushes a direction the other way.
              Or `output_target(pairs)`: plain GCG on the answers, nothing internal (the baseline).
    layers    aim at the residual: the output of decoder block L (hidden_states[L + 1])
    heads     aim at what these (layer, head) pairs write into the residual, summed
    steps     stop after this many steps; time: stop after this long (90, "45m", "2h").
              Neither given: 250 steps. Both: whichever comes first.
    validate  a function prefix_text -> number, higher is better, measuring the behaviour you
              actually want (e.g. task accuracy on held-out inputs, in a DIFFERENT template from the
              probes). Called on the current best prefix every `check_every` steps and at the end;
              the prefix returned is then the best-VALIDATED one, not the highest-scoring one, which
              guards against fitting the probes instead of the concept.
    probes    prompts the prefix is scored in front of; default: 16 neutral chat prompts
    init      resume from an earlier prefix (e.g. a previous Whisper's `.text`); its own token
              count then replaces `n_tokens`
    log       path of a JSONL file that gets the best prefix after every step (and checks)
    template  text with one `{prompt}` that every `prefix + " " + probe` is put inside, e.g.
              `chat_template(tok, thinking=False)` to search inside a chat model's user turn. The
              target is then measured at the template's last token (where the reply starts). Pass
              the same template to `task_accuracy` for `validate=`; the result's `try_on` uses it.
    read      where the target is read inside a template: "reply" (its last token, where the reply
              starts) or "prompt" (the last token of the user's text, i.e. the probe's last word --
              for a target built at the word, e.g. `instruction_vector(..., at="word")`)
    blackbox  no gradients: candidate swaps are drawn uniformly from the allowed tokens and kept only
              if the score improves, so the search needs nothing but scores (with `output_target`,
              only the model's output probabilities). Cheaper per step, usually needs many more steps.

    Expect GPU-minutes on a ~1B model and GPU-hours on a 32B one: every step is a forward and a
    backward pass plus `candidates` x len(probes) forward sequences.
    """
    model, tok, target, model_id = _prepare(model, target, layers, tokenizer, heads)
    budget = _seconds(time)
    if steps is None and budget is None:
        steps = 250
    rng = random.Random(seed)
    scorer = _scorer(model, tok, target, probes, chunk, template, read)
    vocab = model.get_input_embeddings().weight.shape[0]
    allowed = _allowed(tok, vocab).to(_device(model))
    allowed_ids = allowed.nonzero().flatten().cpu()
    draw = torch.Generator().manual_seed(seed)

    current = tok(init, add_special_tokens=False).input_ids if init else _init_prefix(tok, n_tokens)
    best = list(current)
    best_cos = scorer.mean_cos([scorer.sequences(current)])[0].item()
    history, checks, t0 = [best_cos - scorer.baseline], [], _time.time()
    if verbose:
        limit = " and ".join(x for x in (steps and f"{steps} steps", budget and f"{budget:.0f}s") if x)
        print(f"whisper: {model_id}, {target!r}, {len(current)} tokens, up to {limit}; "
              f"start {best_cos - scorer.baseline:+.5f}", flush=True)

    def check(step):
        text = tok.decode(best)
        if checks and checks[-1]["text"] == text:            # same prefix as last time: no new call
            val = checks[-1]["val"]
        else:
            val = float(validate(text))
        checks.append({"step": step, "score": best_cos - scorer.baseline, "val": val, "text": text, "ids": list(best)})
        if verbose:
            print(f"  check {step:>4}  score {best_cos - scorer.baseline:+.5f}  validate {val:.4g}  {text[:50]!r}",
                  flush=True)
        return val

    step = 0
    while True:
        if steps is not None and step >= steps:
            break
        if budget is not None and _time.time() - t0 >= budget:
            break
        step += 1
        if blackbox:     # no gradient: each position's candidate tokens are drawn at random from the allowed set
            top = allowed_ids[torch.randint(len(allowed_ids), (len(current), topk), generator=draw)]
        else:
            grad = scorer.gradient(current)
            grad[:, ~allowed] = -float("inf")
            top = grad.topk(topk, dim=1).indices                              # (n, topk)
        pos = torch.tensor([rng.randrange(len(current)) for _ in range(candidates)])
        pick = torch.tensor([rng.randrange(topk) for _ in range(candidates)])
        cands = torch.tensor(current).repeat(candidates, 1)
        cands[torch.arange(candidates), pos] = top[pos, pick].cpu()
        cands = torch.unique(cands, dim=0)
        keep = _roundtrips(tok, cands)
        if keep.any():
            cands = cands[keep]
        scores = scorer.mean_cos([scorer.sequences(c.tolist()) for c in cands])
        i = int(scores.argmax())
        current, cur_cos = cands[i].tolist(), scores[i].item()
        if cur_cos > best_cos:
            best, best_cos = list(current), cur_cos
        history.append(best_cos - scorer.baseline)
        val = check(step) if validate is not None and step % check_every == 0 else None
        if log:
            with open(log, "a") as f:
                f.write(json.dumps({"step": step, "score": best_cos - scorer.baseline, "text": tok.decode(best),
                                    "ids": best, **({"val": val} if val is not None else {})}) + "\n")
        if verbose and validate is None and (step % 20 == 0):
            print(f"  step {step:>4}  {best_cos - scorer.baseline:+.5f}  {_time.time() - t0:6.0f}s  "
                  f"{tok.decode(best)[:60]!r}", flush=True)

    chosen, selected_by = (list(best), best_cos - scorer.baseline), "score"
    if validate is not None:
        if not checks or checks[-1]["step"] != step:
            check(step)
        top_check = max(checks, key=lambda c: (c["val"], c["step"]))
        chosen, selected_by = (top_check["ids"], top_check["score"]), "validation"
        if verbose and len({c["val"] for c in checks}) == 1:
            print(f"whisper: WARNING every checkpoint validated the same ({checks[0]['val']:.4g}); validation could "
                  "not choose, so the last checkpoint is returned -- treat the search as unvalidated", flush=True)
    ids, search_score = chosen
    text = tok.decode(ids)
    as_text = scorer.text_sequences(text)
    roundtrip = as_text == scorer.sequences(ids)
    final = scorer.mean_cos([as_text])[0].item() - scorer.baseline
    if verbose:
        print(f"whisper: {step} steps, {_time.time() - t0:.0f}s; chosen by {selected_by}: {text!r} "
              f"score {final:+.5f}", flush=True)
    return Whisper(text, ids, final, search_score, roundtrip, target, model_id, history, checks, selected_by,
                   model, tok, template=template, read=read)
