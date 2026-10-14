"""Pluggable optimizers: the default is unchanged GCG, every optimizer hands back real allowed tokens and reports
its compute, the embedding gradient is consistent with the one-hot one, and the budget and settings guards fire."""
import pytest
import torch

import whisperers as w
from whisperers import optimizers as O
from whisperers.search import _allowed, _scorer

PAIRS = [("hot ->", " cold"), ("big ->", " small")]


def _vec(seed=1, d=64):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(d, generator=g)


@pytest.mark.parametrize("kind", ["resid", "output"])
def test_embedding_gradient_is_the_onehot_gradient_by_chain_rule(tiny, kind):
    model, tok = tiny
    target = w.direction(_vec(), [2]) if kind == "resid" else w.output_target(PAIRS)
    sc = _scorer(model, tok, target, None, 64, None, "reply")
    prefix = tok(" the quick brown", add_special_tokens=False).input_ids
    W = model.get_input_embeddings().weight
    g_onehot = sc.gradient(prefix)
    value, g_emb = sc.embedding_gradient(W[torch.tensor(prefix)])
    assert torch.allclose(g_onehot, g_emb @ W.detach().float().T, atol=1e-5, rtol=1e-4)
    expected = sc.mean_cos([sc.sequences(prefix)])[0].item()
    assert abs(value - expected) < 1e-4


def test_default_is_gcg_by_name_and_by_instance(tiny):
    model, tok = tiny
    runs = [w.whisper(model, _vec(), layers=[2], tokenizer=tok, steps=3, n_tokens=4, verbose=False, **kw)
            for kw in ({}, {"optimizer": "gcg"}, {"optimizer": O.GCG()})]
    assert runs[0].text == runs[1].text == runs[2].text and runs[0].history == runs[2].history
    assert runs[0].optimizer == "gcg" and runs[0].optimizer_params == {"topk": 256, "candidates": 128}


def test_blackbox_is_random_search(tiny):
    model, tok = tiny
    a = w.whisper(model, _vec(), layers=[3], tokenizer=tok, steps=3, n_tokens=4, blackbox=True, verbose=False)
    b = w.whisper(model, _vec(), layers=[3], tokenizer=tok, steps=3, n_tokens=4, optimizer="random", verbose=False)
    assert a.text == b.text and a.history == b.history and a.compute["backward"] == 0


@pytest.mark.parametrize("name", ["gcg", "random", "pez", "gbda"])
@pytest.mark.parametrize("kind", ["heads", "output"])
def test_every_optimizer_returns_allowed_tokens_and_reports_compute(tiny, name, kind):
    model, tok = tiny
    kw = {"heads": [(1, 0), (2, 3)]} if kind == "heads" else {}
    target = _vec() if kind == "heads" else w.output_target(PAIRS)
    res = w.whisper(model, target, tokenizer=tok, steps=3, n_tokens=4, optimizer=name, verbose=False, **kw)
    allowed = _allowed(tok, model.get_input_embeddings().weight.shape[0])
    assert res.optimizer == name and len(res.ids) == 4 and allowed[torch.tensor(res.ids)].all()
    assert all(b >= a for a, b in zip(res.history, res.history[1:]))       # the best so far never gets worse
    assert res.compute["forward"] > 0 and res.compute["cost"] >= res.compute["forward"]
    assert (res.compute["backward"] == 0) == (name == "random")


@pytest.mark.parametrize("name", ["pez", "gbda"])
def test_continuous_optimizers_respect_forbid(tiny, name):
    model, tok = tiny
    forbid = ["e", "a"]                                  # bans most of the vocabulary: projection must avoid it
    res = w.whisper(model, _vec(), layers=[2], tokenizer=tok, steps=3, n_tokens=4, optimizer=name, forbid=forbid,
                    verbose=False)
    assert not any(c in tok.decode([i]).lower() for i in res.ids for c in forbid)


def test_compute_budget_stops_the_search(tiny):
    model, tok = tiny
    one = w.whisper(model, _vec(), layers=[2], tokenizer=tok, steps=1, n_tokens=4, candidates=8, verbose=False)
    per_step = one.compute["cost"]
    res = w.whisper(model, _vec(), layers=[2], tokenizer=tok, compute=3 * per_step, n_tokens=4, candidates=8,
                    verbose=False)
    assert 3 * per_step <= res.compute["cost"] <= 4 * per_step + 16 and len(res.history) <= 5


def test_settings_guards(tiny):
    model, tok = tiny
    run = lambda **kw: w.whisper(model, _vec(), layers=[2], tokenizer=tok, steps=1, n_tokens=3, verbose=False, **kw)
    with pytest.raises(ValueError):
        run(optimizer="pez", topk=8)                     # GCG settings on a continuous optimizer
    with pytest.raises(ValueError):
        run(optimizer=O.GCG(), candidates=8)             # settings belong on the instance
    with pytest.raises(ValueError):
        run(optimizer="pez", blackbox=True)
    with pytest.raises(ValueError):
        run(optimizer="annealing")


def test_your_own_optimizer_plugs_in(tiny, tmp_path):
    model, tok = tiny

    class Stay(O.Optimizer):
        """Never moves: the result is the starting prefix."""
        name, needs_grad = "stay", False

        def step(self, objective):
            return self.current, objective.score([self.current])[0].item()

    res = w.whisper(model, _vec(), layers=[2], tokenizer=tok, steps=2, init=" ! ! !", optimizer=Stay(), verbose=False)
    assert res.text == " ! ! !" and res.optimizer == "stay" and res.compute["backward"] == 0
    res.save(str(tmp_path / "s.json"))
    back = w.Whisper.load(str(tmp_path / "s.json"))
    assert back.optimizer == "stay" and back.compute == res.compute
