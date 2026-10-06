"""Plain GCG on the OUTPUT (Rohit's baseline): score = mean log-probability of each answer after
prefix + probe, nothing internal; and the gradient-free (black-box) variant of the same search."""
import pytest
import torch

import whisperers as w
from whisperers.search import _OutputScorer, encode

PAIRS = [("hot ->", " cold"), ("big ->", " small"), ("up ->", " down")]


def test_output_score_is_answer_logprob(tiny):
    model, tok = tiny
    target = w.output_target(PAIRS)
    sc = _OutputScorer(model, tok, target, 64)
    prefix_text = " ! opposite !"
    manual = []
    with torch.no_grad():
        for text in [w.compose(prefix_text, p) for p, _ in PAIRS] + [p for p, _ in PAIRS]:
            p, a = next((p, a) for p, a in PAIRS if text.endswith(p))
            ans = tok(a, add_special_tokens=False).input_ids
            ids = torch.tensor([encode(tok, text) + ans])
            lp = model(ids).logits[0].float().log_softmax(-1)
            n = ids.shape[1]
            manual.append(torch.stack([lp[n - len(ans) - 1 + i, t] for i, t in enumerate(ans)]).mean().item())
    want = sum(manual[:3]) / 3 - sum(manual[3:]) / 3
    got = w.score(model, target, prefix_text, tokenizer=tok)
    assert abs(got - want) < 1e-4
    assert abs(sc.baseline - sum(manual[3:]) / 3) < 1e-4


def test_output_search_improves_and_roundtrips(tiny, tmp_path):
    model, tok = tiny
    res = w.whisper(model, w.output_target(PAIRS), tokenizer=tok, steps=4, n_tokens=4, verbose=False)
    assert res.history[-1] >= res.history[0] and res.text
    res.save(str(tmp_path / "o.json"))
    back = w.Whisper.load(str(tmp_path / "o.json"))
    assert isinstance(back.target, w.OutputTarget) and back.target.pairs == PAIRS
    with pytest.raises(ValueError):
        w.whisper(model, w.output_target(PAIRS), tokenizer=tok, steps=1, probes=["x"], verbose=False)
    with pytest.raises(ValueError):
        w.output_target([("hot ->", "")])


def test_blackbox_uses_no_gradient(tiny, monkeypatch):
    model, tok = tiny

    def boom(*a, **k):
        raise AssertionError("black-box search must not take gradients")
    monkeypatch.setattr(_OutputScorer, "gradient", boom)
    res = w.whisper(model, w.output_target(PAIRS), tokenizer=tok, steps=3, n_tokens=4, blackbox=True, verbose=False)
    assert res.history[-1] >= res.history[0] and len(res.ids) == 4
    # and it works for an internal target too
    monkeypatch.setattr(w.search._Scorer, "gradient", boom)
    res2 = w.whisper(model, w.direction(torch.randn(64), [1]), tokenizer=tok, steps=2, n_tokens=3, probes=["hot ->"],
                     blackbox=True, verbose=False)
    assert res2.text
