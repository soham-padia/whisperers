"""In-context tasks: input/answer pairs, a held-out accuracy check, and choosing layers by concept.

A Task is a list of (input, answer) pairs and a template ("hot -> cold"). It feeds three things:
the task vector (targets.task_vector), the head ranking (heads.find_heads), and two checks that
judge by BEHAVIOUR rather than by how far the residual moves:

  task_accuracy(...)  a ready-made `validate=` for whisper(): first-word accuracy of
                      prefix + held-out query, ideally in a different template from the probes
  find_layers(...)    rank layers by what injecting the layer's task vector DOES on held-out
                      inputs, minus what a placebo vector does (same demonstrations, answers
                      shuffled: same format, no relation), optionally in a second template
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass

import torch

from .search import _blocks, _device, compose, encode
from .targets import task_vector


@dataclass
class Task:
    pairs: list[tuple[str, str]]
    template: str = "{x} -> {y}\n"
    name: str = "task"

    def query(self, x: str) -> str:
        return self.template.split("{y}")[0].rstrip(" ").format(x=x)

    def split(self, *sizes: int, seed: int = 0) -> list["Task"]:
        """Disjoint random subsets of these sizes, plus one with whatever is left."""
        pairs = list(self.pairs)
        random.Random(seed).shuffle(pairs)
        out, i = [], 0
        for n in sizes:
            out.append(Task(pairs[i:i + n], self.template, self.name))
            i += n
        out.append(Task(pairs[i:], self.template, self.name))
        return out

    def __len__(self) -> int:
        return len(self.pairs)


def first_word(s: str) -> str:
    words = s.split()
    return re.sub(r"^[^\w]+|[^\w]+$", "", words[0]).lower() if words else ""


@torch.no_grad()
def answer(model, tokenizer, text: str, inject=None, max_new: int = 5, template: str | None = None) -> str:
    """Greedy continuation of `text` (put inside `template` if given, e.g. a chat turn);
    `inject=(layer, vector)` adds the vector at the last prompt token."""
    handle = None
    if inject is not None:
        layer, vec = inject

        def hook(module, args, output):
            h = output[0] if isinstance(output, tuple) else output
            if h.shape[1] > 1:                                       # the prompt pass, not later steps
                h[:, -1] += vec.to(h.device, h.dtype)
            return output
        handle = _blocks(model)[layer].register_forward_hook(hook)
    try:
        ids = torch.tensor([encode(tokenizer, text, template)], device=_device(model))
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new,
                             do_sample=False, pad_token_id=pad)
    finally:
        if handle:
            handle.remove()
    return tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def accuracy(model, tokenizer, task: Task, prefix: str = "", inject=None, max_new: int = 5,
             template: str | None = None) -> float:
    """Fraction of `task`'s pairs whose first generated word is the answer."""
    hits = 0
    for x, y in task.pairs:
        q = task.query(x)
        out = answer(model, tokenizer, compose(prefix, q) if prefix else q, inject, max_new, template)
        hits += first_word(out) == y.lower()
    return hits / max(1, len(task))


def task_accuracy(model, tokenizer, task: Task, max_new: int = 5, template: str | None = None):
    """A `validate=` for whisper(): prefix_text -> accuracy on `task` (use held-out pairs).
    Searching inside a template? Pass the same one here."""
    return lambda prefix: accuracy(model, tokenizer, task, prefix, max_new=max_new, template=template)


def _derange(items, rng):
    while True:
        out = items[:]
        rng.shuffle(out)
        if len(items) < 2 or all(a != b for a, b in zip(out, items)):
            return out


def find_layers(model, tokenizer, demos: Task, fit: Task, val: Task, *, layers=None,
                alt_template: str | None = None, seed: int = 0) -> list[dict]:
    """Rank layers for a task by concept, not by displacement. Best first.

    For each layer: the task vector from `demos` (with minus without, over `fit`'s queries); then,
    on `val`, the accuracy of injecting it (`causal`), of injecting a PLACEBO built the same way
    from demonstrations whose answers are shuffled (`placebo`), and -- with `alt_template`, e.g.
    "The opposite of {x} is {y}" -- of injecting it under a template it was not built in
    (`transfer`). Ranked by `concept` = causal - placebo (+ transfer when given): how much of the
    effect is the relation itself rather than the arrow format, and whether it survives a new form.
    """
    blocks = _blocks(model)
    layers = list(layers) if layers is not None else list(range(len(blocks)))
    pairs = list(demos.pairs)
    real = task_vector(model, tokenizer, pairs, [x for x, _ in fit.pairs], layers, template=demos.template)
    shuffled = _derange([y for _, y in pairs], random.Random(seed))
    fake = task_vector(model, tokenizer, [(x, y) for (x, _), y in zip(pairs, shuffled)],
                       [x for x, _ in fit.pairs], layers, template=demos.template)
    alt = Task(val.pairs, alt_template, val.name) if alt_template else None
    rows = []
    for L in layers:
        v_real = real.vectors[L] * real.meta["norms"][L]
        v_fake = fake.vectors[L] * fake.meta["norms"][L]
        row = {"layer": L, "causal": accuracy(model, tokenizer, val, inject=(L, v_real)),
               "placebo": accuracy(model, tokenizer, val, inject=(L, v_fake))}
        if alt is not None:
            row["transfer"] = accuracy(model, tokenizer, alt, inject=(L, v_real))
        row["concept"] = row["causal"] - row["placebo"] + row.get("transfer", 0.0)
        rows.append(row)
    return sorted(rows, key=lambda r: (-r["concept"], -r["causal"]))
