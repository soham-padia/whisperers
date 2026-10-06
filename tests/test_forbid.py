"""forbid=: tokens (and words spelled across tokens) a prefix may never contain."""
import pytest
import torch

import whisperers as w
from whisperers.search import _forbidden, _text_ok


def test_forbidden_mask_substring_exact_ids_and_callable(tiny):
    _, tok = tiny
    V = len(tok)
    m = _forbidden(tok, V, ["oppos", "=not", 17])
    texts = tok.batch_decode([[i] for i in range(V)])
    assert m[17]
    assert all(m[i] for i, t in enumerate(texts) if "oppos" in t.strip().lower())
    assert all(m[i] for i, t in enumerate(texts) if t.strip().lower() == "not")
    knot = [i for i, t in enumerate(texts) if t.strip().lower() in ("nothing", "knot", "notably")]
    assert knot and not any(m[i] and "oppos" not in texts[i].lower() for i in knot)     # "=not" is exact only
    c = _forbidden(tok, V, lambda t: t.strip() == "!")
    assert c.sum() >= 1 and all(texts[i].strip() == "!" for i in c.nonzero().flatten().tolist())


def test_text_check_catches_words_split_across_tokens():
    ok = _text_ok(["contrast", "=not"])
    assert not ok("x contr" + "ast y") and not ok("Not now") and ok("knot nothing") and ok("plain")
    assert _text_ok([5, 6]) is None and _text_ok(None) is None


def test_search_never_uses_a_forbidden_token(tiny):
    model, tok = tiny
    d = model.config.hidden_size
    torch.manual_seed(2)
    t = w.direction(torch.randn(d), [2])
    banned = ["e", "a"]                               # bans a large share of the vocabulary
    res = w.whisper(model, t, tokenizer=tok, steps=4, n_tokens=4, forbid=banned, verbose=False)
    assert not any(b in tok.decode([i]).strip().lower() for i in res.ids for b in banned)
    assert not any(b in res.text.lower() for b in banned)
    bb = w.whisper(model, t, tokenizer=tok, steps=3, n_tokens=4, forbid=banned, blackbox=True, verbose=False)
    assert not any(b in bb.text.lower() for b in banned)


def test_forbidden_start_is_refused(tiny):
    model, tok = tiny
    t = w.direction(torch.randn(model.config.hidden_size), [1])
    with pytest.raises(ValueError, match="forbidden"):
        w.whisper(model, t, tokenizer=tok, steps=1, init="the opposite", forbid=["oppos"], verbose=False)


@pytest.mark.parametrize("kind", ["direction", "output", "joint"])
def test_chunked_gradient_equals_the_full_one(tiny, kind):
    from whisperers.search import _scorer
    model, tok = tiny
    torch.manual_seed(3)
    d = model.config.hidden_size
    probes = ["hot ->", "big ->", "up ->", "fast ->", "old ->"]
    if kind == "direction":
        target, pr = w.direction(torch.randn(d), [1, 2]), probes
    elif kind == "output":
        target, pr = w.output_target([(p, " x") for p in probes]), None
    else:
        target, pr = [w.direction(torch.randn(d), [1]), w.direction(torch.randn(d), [3])], [probes, probes[:3]]
    prefix = tok(" ! ! ! !", add_special_tokens=False).input_ids
    full = _scorer(model, tok, target, pr, 256, None, "reply").gradient(prefix)
    part = _scorer(model, tok, target, pr, 256, None, "reply", grad_chunk=2).gradient(prefix)
    assert torch.allclose(full, part, atol=1e-5, rtol=1e-4)
