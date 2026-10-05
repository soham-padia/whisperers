"""Can tokens reproduce a task vector? Antonyms, scored by plain task accuracy -- no judge.

1. Build a task vector at every layer: the residual with antonym demonstrations minus without
   (Hendel et al. 2023's residual form of Todd et al.'s function vector).
2. Pick the layer where INJECTING that vector actually makes the model do the task, on the fit
   words. A vector that does not induce the task by injection is not worth imitating with tokens,
   and comparing against it would test nothing -- the first run of this demo did exactly that.
3. whisper() a prefix toward the vector at that layer, scoring in front of zero-shot queries.
4. On HELD-OUT words, compare zero-shot accuracy of:
     plain        the query alone
     prefix       the found prefix + the query
     demos        real demonstrations cut to the prefix's token length
     random       a random prefix of the same length
     injection    the task vector added at the last query token (the thing being imitated)

    python examples/function_vector.py --model allenai/OLMo-2-0425-1B --steps 200

The word list is a small hand-made one, enough for a demo; Todd et al.'s datasets are the
reference for a real measurement.
"""
import argparse
import random

import torch

import whisperers as w
from whisperers.search import _allowed, _blocks, _device

PAIRS = [
    ("hot", "cold"), ("up", "down"), ("big", "small"), ("fast", "slow"), ("happy", "sad"),
    ("light", "dark"), ("open", "closed"), ("old", "new"), ("early", "late"), ("full", "empty"),
    ("hard", "soft"), ("high", "low"), ("long", "short"), ("rich", "poor"), ("strong", "weak"),
    ("wet", "dry"), ("thick", "thin"), ("loud", "quiet"), ("clean", "dirty"), ("cheap", "expensive"),
    ("first", "last"), ("inside", "outside"), ("win", "lose"), ("love", "hate"), ("push", "pull"),
    ("buy", "sell"), ("start", "finish"), ("before", "after"), ("true", "false"), ("good", "bad"),
    ("right", "wrong"), ("young", "old"), ("near", "far"), ("heavy", "light"), ("deep", "shallow"),
    ("wide", "narrow"), ("safe", "dangerous"), ("easy", "difficult"), ("alive", "dead"), ("day", "night"),
    ("north", "south"), ("east", "west"), ("top", "bottom"), ("front", "back"), ("left", "right"),
    ("give", "take"), ("come", "go"), ("remember", "forget"), ("accept", "reject"), ("increase", "decrease"),
    ("friend", "enemy"), ("question", "answer"), ("rise", "fall"), ("sweet", "sour"), ("brave", "cowardly"),
    ("polite", "rude"), ("arrive", "leave"), ("borrow", "lend"), ("asleep", "awake"), ("sharp", "dull"),
]
TEMPLATE = "{x} -> {y}\n"


def answer(model, tok, text, inject=None):
    """First word the model writes after `text`, optionally with a vector added at the last token."""
    handle = None
    if inject is not None:
        layer, vec = inject
        def hook(module, args, output):
            h = output[0] if isinstance(output, tuple) else output
            if h.shape[1] > 1:                                    # the prompt pass, not later steps
                h[:, -1] += vec.to(h.device, h.dtype)
            return output
        handle = _blocks(model)[layer].register_forward_hook(hook)
    try:
        ids = tok(text, return_tensors="pt").input_ids.to(_device(model))
        out = model.generate(ids, max_new_tokens=5, do_sample=False,
                             pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
    finally:
        if handle:
            handle.remove()
    words = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).split()
    return words[0].strip(".,;:!?\"'").lower() if words else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/OLMo-2-0425-1B")
    ap.add_argument("--layer", type=int, default=None, help="default: the layer where injection works best")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--n-tokens", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    pairs = PAIRS[:]
    rng.shuffle(pairs)
    demos, fit, test = pairs[:10], pairs[10:30], pairs[30:]       # disjoint: no test word is ever seen

    model, tok = w.load(args.model)
    query = TEMPLATE.split("{y}")[0].rstrip(" ")
    layers = [args.layer] if args.layer is not None else list(range(len(_blocks(model))))
    every = w.task_vector(model, tok, demos[:5], [x for x, _ in fit], layers, template=TEMPLATE)

    def injected(L):
        return every.vectors[L] * every.meta["norms"][L]

    # choose the layer on FIT words: test words stay unseen until the final comparison
    sweep = {L: sum(answer(model, tok, query.format(x=x), inject=(L, injected(L))) == y for x, y in fit)
             for L in layers}
    L = max(sweep, key=sweep.get)
    print("injection accuracy on fit words, by layer:", sweep)
    print(f"-> layer {L} ({sweep[L]}/{len(fit)})")
    if sweep[L] < len(fit) // 3:
        print("WARNING: injecting the task vector barely induces the task at any layer, so the "
              "comparison below cannot show tokens reproducing it.")

    tv = w.direction(every.vectors[L], [L], name="task_vector", model_id=every.model_id)
    probes = [query.format(x=x) for x, _ in fit]
    res = w.whisper(model, tv, tokenizer=tok, n_tokens=args.n_tokens, steps=args.steps, probes=probes)
    print(f"\nprefix {res.text!r}  score {res.score:+.4f}  roundtrip {res.roundtrip_ok}\n")

    shots = "".join(TEMPLATE.format(x=x, y=y) for x, y in demos)
    shot_ids = tok(shots, add_special_tokens=False).input_ids[: len(res.ids)]
    demo_prefix = tok.decode(shot_ids)
    pool = _allowed(tok, model.get_input_embeddings().weight.shape[0]).nonzero().flatten().tolist()
    random_prefix = tok.decode([rng.choice(pool) for _ in res.ids])
    vec = injected(L)

    arms = {
        "plain": lambda q: answer(model, tok, q),
        "prefix": lambda q: answer(model, tok, w.compose(res.text, q)),
        "demos": lambda q: answer(model, tok, w.compose(demo_prefix, q)),
        "random": lambda q: answer(model, tok, w.compose(random_prefix, q)),
        "injection": lambda q: answer(model, tok, q, inject=(L, vec)),
    }
    for name, run in arms.items():
        hits = sum(run(query.format(x=x)) == y for x, y in test)
        print(f"{name:>10}: {hits}/{len(test)} held-out antonyms correct")


if __name__ == "__main__":
    main()
