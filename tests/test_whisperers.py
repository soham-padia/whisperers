import pytest
import torch

import whisperers as w
from whisperers.search import _Scorer, _read

PROBES = ["Could you help me plan my week?", "How do I fix a leaky faucet?", "Tell me about a book."]


def test_read_is_block_output_at_last_token(tiny):
    """Layer L is hidden_states[L + 1] -- the off-by-one every reproduction gets wrong."""
    model, tok = tiny
    ids = tok("hello there, how are you", return_tensors="pt").input_ids
    mask = torch.ones_like(ids)
    caps = _read(model, [1, 2], mask, torch.tensor([ids.shape[1] - 1]), input_ids=ids)
    hs = model(input_ids=ids, output_hidden_states=True).hidden_states
    for L in (1, 2):
        assert torch.allclose(caps[L][0], hs[L + 1][0, -1], atol=1e-5)


def test_padded_batch_matches_one_at_a_time(tiny):
    """Right padding must not change a sequence's own last-token residual."""
    model, tok = tiny
    target = w.direction(torch.randn(64), [2])
    s = _Scorer(model, tok, target, PROBES, chunk=8)
    seqs = s.sequences([1, 2, 3])
    together = s.mean_cos([seqs])[0]
    alone = torch.stack([s.mean_cos([[q]])[0] for q in seqs]).mean()
    assert torch.allclose(together, alone, atol=1e-5)


def test_search_improves_its_objective(tiny):
    model, tok = tiny
    res = w.whisper(model, torch.randn(64), layers=[1, 2], tokenizer=tok, n_tokens=5, steps=6,
                    probes=PROBES, candidates=16, topk=32, verbose=False)
    assert res.history[-1] >= res.history[0]
    assert all(b >= a for a, b in zip(res.history, res.history[1:]))
    assert isinstance(res.text, str) and len(res.ids) == 5
    if res.roundtrip_ok:
        assert res.score == pytest.approx(res.search_score, abs=1e-5)
    assert w.score(model, res.target, res.text, tokenizer=tok, probes=PROBES) == pytest.approx(res.score, abs=1e-6)
    assert set(res.try_on("Hello", max_new_tokens=5)) == {"prompt", "without", "with", "echo"}


def test_negation_flips_the_target():
    t = w.direction([1.0, 0.0, 0.0], [3])
    assert torch.equal((-t).vectors[3], torch.tensor([-1.0, 0.0, 0.0]))


def test_guards(tiny):
    model, tok = tiny
    kw = dict(tokenizer=tok, steps=1, probes=PROBES, verbose=False)
    with pytest.raises(ValueError, match="dim"):
        w.whisper(model, torch.randn(32), layers=[1], **kw)
    with pytest.raises(ValueError, match="out of range"):
        w.whisper(model, torch.randn(64), layers=[9], **kw)
    with pytest.raises(ValueError, match="already says where it aims"):
        w.whisper(model, w.direction(torch.randn(64), [1]), layers=[1], **kw)
    with pytest.raises(ValueError, match="built on"):
        w.whisper(model, w.direction(torch.randn(64), [1], model_id="some/other-model"), **kw)
    with pytest.raises(ValueError, match="say where to aim"):
        w.whisper(model, torch.randn(64), **kw)


def test_subspace_target(tiny):
    """A block-sparse feature is a subspace: score in [0, 1], sign-free, rank-deficient rows dropped."""
    model, tok = tiny
    W = torch.randn(64, 12)
    W[:, 5] = W[:, 4]                                    # a repeated column: block 1 has rank 3
    t = w.sae_block(W, block=1, block_size=4, layer=2)
    assert t.vectors[2].shape == (3, 64)
    assert torch.allclose(t.vectors[2] @ t.vectors[2].T, torch.eye(3), atol=1e-5)
    with pytest.raises(ValueError, match="sign"):
        -t
    s = _Scorer(model, tok, t, PROBES, chunk=8)
    c = s.mean_cos([s.sequences([1, 2, 3])])[0].item()
    assert 0.0 <= c <= 1.0
    res = w.whisper(model, t, tokenizer=tok, n_tokens=4, steps=3, probes=PROBES, candidates=16,
                    topk=32, verbose=False)
    assert res.history[-1] >= res.history[0]


def test_target_builders(tiny):
    model, tok = tiny
    sae = w.sae_latent(torch.randn(64, 10), index=3, layer=2)
    assert sae.layers == [2] and sae.vectors[2].norm() == pytest.approx(1.0)
    ms = w.mean_shift(model, tok, ["kind words", "gentle reply"], ["cruel words", "harsh reply"], [1, 3])
    assert ms.layers == [1, 3] and all(v.numel() == 64 for v in ms.vectors.values())
    tv = w.task_vector(model, tok, [("hot", "cold"), ("up", "down")], ["big", "fast"], 2)
    assert tv.layers == [2] and tv.vectors[2].norm() == pytest.approx(1.0)
