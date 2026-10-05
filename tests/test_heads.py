import json

import pytest
import torch

import whisperers as w
from whisperers.search import _seconds

PROBES = ["Could you help me plan my week?", "How do I fix a leaky faucet?", "Tell me about a book."]
PAIRS = [("hot", "cold"), ("up", "down"), ("big", "small"), ("fast", "slow"), ("happy", "sad"), ("light", "dark"),
         ("open", "closed"), ("old", "new"), ("early", "late"), ("full", "empty"), ("hard", "soft"), ("high", "low"),
         ("long", "short"), ("rich", "poor"), ("strong", "weak"), ("wet", "dry"), ("thick", "thin"), ("loud", "quiet"),
         ("clean", "dirty"), ("cheap", "expensive"), ("first", "last"), ("win", "lose"), ("love", "hate"),
         ("push", "pull"), ("buy", "sell"), ("start", "finish"), ("true", "false"), ("good", "bad")]


@pytest.mark.parametrize("fixture", ["tiny", "tiny_llama", "tiny_gpt2"])
def test_head_writes_sum_to_the_attention_update(fixture, request):
    """The identity a heads target rests on, on post-norm (OLMo), pre-norm+bias (Llama) and Conv1D (GPT-2)."""
    model, tok = request.getfixturevalue(fixture)
    assert w.check_head_writes(model, tok, "the cat sat on the mat today", layers=[0, 1]) < 1e-3


def test_heads_target_search_and_score(tiny):
    model, tok = tiny
    heads = [(1, 0), (2, 3)]
    res = w.whisper(model, torch.randn(64), heads=heads, tokenizer=tok, n_tokens=4, steps=4, probes=PROBES,
                    candidates=16, topk=32, verbose=False)
    assert res.target.heads == heads and res.history[-1] >= res.history[0]
    assert w.score(model, res.target, res.text, tokenizer=tok, probes=PROBES) == pytest.approx(res.score, abs=1e-6)


def test_heads_target_guards(tiny):
    model, tok = tiny
    kw = dict(tokenizer=tok, steps=1, probes=PROBES, verbose=False)
    with pytest.raises(ValueError, match="out of range"):
        w.whisper(model, torch.randn(64), heads=[(1, 9)], **kw)
    with pytest.raises(ValueError, match="not both"):
        w.whisper(model, torch.randn(64), heads=[(1, 0)], layers=[1], **kw)
    with pytest.raises(ValueError, match="say where to aim"):
        w.whisper(model, torch.randn(64), **kw)


def test_time_budget():
    assert _seconds("90s") == 90 and _seconds("45m") == 2700 and _seconds("1h30m") == 5400 and _seconds(12) == 12
    with pytest.raises(ValueError):
        _seconds("soon")


def test_time_budget_stops_the_search(tiny):
    model, tok = tiny
    res = w.whisper(model, torch.randn(64), layers=[1], tokenizer=tok, n_tokens=4, time=0.5, probes=PROBES,
                    candidates=8, topk=16, verbose=False)
    assert 1 <= len(res.history) - 1 < 500


def test_validate_picks_the_best_behaving_prefix(tiny):
    """With validate=, the returned prefix is the best-VALIDATED checkpoint, not the best-scoring one."""
    model, tok = tiny
    seen = []

    def validate(text):                       # prefers the EARLIEST checkpoint, whatever the score does
        seen.append(text)
        return -len(seen)
    res = w.whisper(model, torch.randn(64), layers=[1], tokenizer=tok, n_tokens=4, steps=6, check_every=2,
                    validate=validate, probes=PROBES, candidates=16, topk=32, verbose=False)
    assert res.selected_by == "validation" and len(res.checks) >= 3
    assert res.ids == res.checks[0]["ids"]


def test_save_and_load_roundtrip(tiny, tmp_path):
    model, tok = tiny
    res = w.whisper(model, torch.randn(64), heads=[(1, 2)], tokenizer=tok, n_tokens=4, steps=2, probes=PROBES,
                    candidates=8, topk=16, verbose=False)
    res.save(tmp_path / "w.json")
    back = w.Whisper.load(tmp_path / "w.json", model, tok)
    assert back.text == res.text and back.ids == res.ids and back.target.heads == [(1, 2)]
    assert torch.allclose(back.target.vectors[1], res.target.vectors[1])
    assert json.loads((tmp_path / "w.json").read_text())["selected_by"] == "score"


def test_find_heads_and_function_vector(tiny):
    model, tok = tiny
    ranked = w.find_heads(model, tok, PAIRS, n_demos=3, n_prompts=4)
    assert len(ranked) == 4 * w.n_heads(model)
    assert ranked == sorted(ranked, key=lambda kv: -kv[1])
    fv = w.function_vector(model, tok, PAIRS, k=3, n_demos=3, n_prompts=4, ranked=ranked)
    assert len(fv.heads) == 3 and fv.meta["vector"].shape == (64,)
    res = w.whisper(model, fv, tokenizer=tok, n_tokens=4, steps=2, probes=["hot ->", "up ->"],
                    candidates=8, topk=16, verbose=False)
    assert res.target.heads == fv.heads


def test_find_layers_and_task_accuracy(tiny):
    model, tok = tiny
    demos, fit, val, _ = w.Task(PAIRS).split(4, 6, 6)
    rows = w.find_layers(model, tok, demos, fit, val, layers=[1, 2], alt_template="The opposite of {x} is {y}")
    assert [set(r) for r in rows][0] >= {"layer", "causal", "placebo", "transfer", "concept"}
    assert rows == sorted(rows, key=lambda r: (-r["concept"], -r["causal"]))
    acc = w.task_accuracy(model, tok, val)
    assert 0.0 <= acc(" ! ! !") <= 1.0
