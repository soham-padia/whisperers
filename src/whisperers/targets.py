"""What a whisper aims at: one unit vector, or one subspace, per layer.

Directions, steering vectors, SAE latents, LoRA adapters and task vectors all reduce to the same
thing -- a vector at one or more layers that the found prefix should push the last-token residual
toward. Block-sparse featurizers are the exception: their features are small subspaces, so a
Target may hold an orthonormal (k, d_model) basis instead, scored by how much of the residual
falls inside it. Layer L means the OUTPUT of decoder block L (hidden_states[L + 1]; index 0 is
the embeddings). Getting that off by one is the most common reproduction bug.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class Target:
    vectors: dict[int, torch.Tensor]          # layer -> unit vector (d,) or orthonormal basis (k, d)
    name: str = "direction"
    model_id: str | None = None               # if set, whisper() refuses a different model
    meta: dict = field(default_factory=dict)
    # A HEADS target aims at what these (layer, head) pairs write into the residual stream, summed,
    # instead of at the residual itself; `vectors` then holds one vector, keyed by the deepest layer.
    heads: list[tuple[int, int]] | None = None

    @property
    def layers(self) -> list[int]:
        return sorted(self.vectors)

    def __neg__(self) -> "Target":
        if any(v.ndim == 2 for v in self.vectors.values()):
            raise ValueError("a subspace has no 'other way': its score is a norm, which ignores sign")
        return Target({L: -v for L, v in self.vectors.items()}, f"-{self.name}", self.model_id, self.meta,
                      self.heads)

    def __repr__(self) -> str:
        shape = tuple(next(iter(self.vectors.values())).shape)
        where = f"heads={len(self.heads)}" if self.heads else f"layers={self.layers}"
        return f"Target({self.name!r}, {where}, shape={shape})"


@dataclass
class OutputTarget:
    """An OUTPUT target -- plain GCG (Zou et al. 2023): make the model write `answer` right after
    `prefix + " " + probe`, for every (probe, answer) pair. Nothing inside the model is targeted.

    The score is the mean log-probability per answer token, minus the same with no prefix. Write each
    answer exactly as it follows its probe, e.g. ("hot ->", " cold"). The probes come from the pairs, so
    `whisper` takes no `probes=` with this target. The baseline every internal target should beat.
    """
    pairs: list[tuple[str, str]]
    name: str = "output"
    model_id: str | None = None
    meta: dict = field(default_factory=dict)
    heads = None                              # so code that inspects any target can treat it uniformly
    vectors: dict = field(default_factory=dict)

    @property
    def layers(self) -> list[int]:
        return []

    def __repr__(self) -> str:
        return f"OutputTarget({self.name!r}, {len(self.pairs)} pairs)"


def output_target(pairs, name: str = "output", model_id: str | None = None) -> OutputTarget:
    """Plain-GCG target: [(probe, answer), ...], e.g. [("hot ->", " cold"), ("big ->", " small")]."""
    pairs = [(str(p), str(a)) for p, a in pairs]
    if not pairs or any(not a for _, a in pairs):
        raise ValueError("output_target needs (probe, answer) pairs with non-empty answers")
    return OutputTarget(pairs, name, model_id)


def _unit(v) -> torch.Tensor:
    v = torch.as_tensor(v, dtype=torch.float32).detach().flatten().cpu()
    n = v.norm()
    if n == 0:
        raise ValueError("target vector is all zeros")
    return v / n


def direction(vector, layers, name: str = "direction", model_id: str | None = None) -> Target:
    """A vector (numpy, torch or list) at one or more layers.

    `vector` may also be a {layer: vector} dict, for a different vector per layer.
    """
    if isinstance(layers, int):
        layers = [layers]
    if not isinstance(vector, dict):
        vector = {L: vector for L in layers}
    raw = {int(L): torch.as_tensor(v, dtype=torch.float32).detach().flatten().cpu() for L, v in vector.items()}
    # the search only needs the unit vectors; the norms are kept for injecting the original
    return Target({L: _unit(v) for L, v in raw.items()}, name, model_id,
                  {"norms": {L: float(v.norm()) for L, v in raw.items()}})


def head_target(vector, heads, name: str = "heads", model_id: str | None = None) -> Target:
    """Aim at what `heads` -- [(layer, head), ...] -- write into the residual stream, summed.

    The natural target for a function vector, which is itself a sum of selected heads' outputs.
    """
    heads = sorted({(int(L), int(h)) for L, h in heads})
    if not heads:
        raise ValueError("a heads target needs at least one (layer, head)")
    raw = torch.as_tensor(vector, dtype=torch.float32).detach().flatten().cpu()
    top = max(L for L, _ in heads)
    return Target({top: _unit(raw)}, name, model_id, {"norms": {top: float(raw.norm())}}, heads)


def sae_latent(sae, index: int, layer: int, name: str | None = None, model_id: str | None = None) -> Target:
    """One SAE latent, aimed at through its encoder row.

    `sae` is either the encoder weight matrix (d_model x d_sae, the SAELens layout) or any object
    with a `W_enc` attribute in that layout. This works for any SAE whose encoder is a linear map
    followed by an activation (ReLU, JumpReLU, TopK, BatchTopK): pushing the residual toward the
    encoder row raises the latent's pre-activation. The encoder bias and decoder-bias centring are
    ignored, so treat the score as a proxy and check the latent's actual activation if it matters.

    `layer` follows this package's convention (output of block `layer`). SAELens'
    `blocks.L.hook_resid_post` is layer L; `blocks.L.hook_resid_pre` is layer L - 1.
    """
    W = getattr(sae, "W_enc", sae)
    W = torch.as_tensor(W).detach()
    return direction(W[:, index], [layer], name or f"sae_latent_{index}", model_id)


def subspace(basis, layers, name: str = "subspace", model_id: str | None = None) -> Target:
    """A (k, d_model) set of directions at one or more layers, aimed at as a whole.

    The score is the fraction of the residual's length inside the span, ||Q x|| / ||x|| -- the
    cosine to the nearest vector in the subspace -- so any mix of the directions counts. The rows
    need not be orthonormal; they are orthonormalised here. `basis` may be a {layer: basis} dict.
    """
    if isinstance(layers, int):
        layers = [layers]
    if not isinstance(basis, dict):
        basis = {L: basis for L in layers}
    out = {}
    for L, B in basis.items():
        B = torch.as_tensor(B, dtype=torch.float32).detach().cpu()
        if B.ndim != 2:
            raise ValueError("a subspace basis is (k, d_model); use direction() for one vector")
        Q, R = torch.linalg.qr(B.T)                                   # columns of Q span the rows of B
        rank = int((R.diagonal().abs() > 1e-6 * R.diagonal().abs().max()).sum())
        out[int(L)] = Q[:, :rank].T.contiguous()
    return Target(out, name, model_id)


def sae_block(featurizer, block: int, block_size: int, layer: int, name: str | None = None,
              model_id: str | None = None) -> Target:
    """One block of a block-sparse featurizer (Fel et al. 2026, arXiv 2606.25234).

    A block-sparse SAE groups its code into blocks of `block_size`; a block's activation is the
    norm of its part of the code, so the feature is a subspace, not a direction. `featurizer` is
    the encoder matrix (d_model x n_blocks*block_size) or an object with `W_enc` in that layout.
    The target is the span of the block's encoder columns: exact for tied, orthonormal-block
    (Grassmannian) featurizers, a proxy for the others since it ignores the bias and the
    encoder's own scaling within the block.
    """
    W = torch.as_tensor(getattr(featurizer, "W_enc", featurizer)).detach()
    cols = W[:, block * block_size:(block + 1) * block_size]
    return subspace(cols.T, [layer], name or f"sae_block_{block}", model_id)


@torch.no_grad()
def _last_resids(model, tokenizer, texts: list[str], layers: list[int]) -> dict[int, torch.Tensor]:
    from .search import _blocks, _device

    blocks = _blocks(model)
    out = {L: [] for L in layers}

    def capture(L, caps):
        def hook(module, args, output):
            h = output[0] if isinstance(output, tuple) else output
            caps[L] = h[0, -1].float().cpu()
        return hook

    for t in texts:
        caps = {}
        handles = [blocks[L].register_forward_hook(capture(L, caps)) for L in layers]
        try:
            ids = tokenizer(t, return_tensors="pt").input_ids.to(_device(model))
            model(input_ids=ids, use_cache=False)
        finally:
            for h in handles:
                h.remove()
        for L in layers:
            out[L].append(caps[L])
    return {L: torch.stack(v) for L, v in out.items()}


def mean_shift(model, tokenizer, positive: list[str], negative: list[str], layers,
               name: str = "mean_shift") -> Target:
    """Difference of mean last-token residuals, positive minus negative, per layer.

    The plain contrastive direction: pass contrast pairs and you have a target.
    """
    layers = [layers] if isinstance(layers, int) else list(layers)
    pos = _last_resids(model, tokenizer, positive, layers)
    neg = _last_resids(model, tokenizer, negative, layers)
    return direction({L: pos[L].mean(0) - neg[L].mean(0) for L in layers}, layers, name,
                     getattr(model, "name_or_path", None))


def task_vector(model, tokenizer, demos: list[tuple[str, str]], queries: list[str], layers,
                template: str = "{x} -> {y}\n", name: str = "task_vector") -> Target:
    """A task vector for an in-context task: residual with demonstrations minus without.

    For each query, the in-context prompt is the demonstrations followed by the query, and the
    zero-shot prompt is the query alone; the target is the mean difference at the last token.
    This is the residual form (Hendel et al. 2023). Todd et al.'s function vector is instead a sum
    of selected attention heads' outputs; this is the cheap stand-in until head targets exist.
    """
    shots = "".join(template.format(x=x, y=y) for x, y in demos)
    query = template.split("{y}")[0].rstrip(" ")   # "hot ->", so the answer arrives as " cold"
    with_demos = [shots + query.format(x=q) for q in queries]
    zero_shot = [query.format(x=q) for q in queries]
    return mean_shift(model, tokenizer, with_demos, zero_shot, layers, name)


def lora_shift(model, tokenizer, adapter, prompts: list[str], layers, name: str = "lora_shift") -> Target:
    """What a LoRA adapter does to the residual, as a direction: mean shift with it on vs off.

    `adapter` is a path or hub id loadable by `peft`. This keeps only the average direction the
    adapter moves the last-token residual in -- a lossy summary of a weight change, so expect a
    prefix to reproduce part of the adapter's effect at best.
    """
    from peft import PeftModel

    layers = [layers] if isinstance(layers, int) else list(layers)
    model_id = getattr(model, "name_or_path", None)
    peft_model = PeftModel.from_pretrained(model, adapter)     # injects LoRA layers in place
    try:
        on = _last_resids(peft_model, tokenizer, prompts, layers)
        with peft_model.disable_adapter():
            off = _last_resids(peft_model, tokenizer, prompts, layers)
    finally:
        peft_model.unload()     # strip them again, so `model` is the base model afterwards
    return direction({L: on[L].mean(0) - off[L].mean(0) for L in layers}, layers, name, model_id)
