"""Query targets: what heads LOOK FOR (filter heads, Sen Sharma et al. 2025). Capture, rank by patching
queries, aim a search at them, transport them -- on a model with a query norm (OLMo-3) and one without (Llama)."""
import pytest
import torch

import whisperers as w
from whisperers.heads import QueryCapture, _query_module

CLEAN = ["Options: Car, Apple, Dog. Which is a fruit? Answer:", "Options: Bus, Pear, Cat. Which is a fruit? Answer:"]
CORRUPT = ["Options: Car, Apple, Dog. Which is a vehicle? Answer:", "Options: Bus, Pear, Cat. Which is a vehicle? Answer:"]


@pytest.mark.parametrize("fixture", ["tiny", "tiny_llama"])
def test_query_capture_is_the_query_module_output(fixture, request):
    model, tok = request.getfixturevalue(fixture)
    blocks = w.search._blocks(model)
    mod = _query_module(blocks[1])
    assert type(mod).__name__.endswith("RMSNorm") == (fixture == "tiny")      # OLMo-3 normalises queries; Llama doesn't
    seen = {}
    h = mod.register_forward_hook(lambda m, a, out: seen.setdefault("q", out.detach().clone()))
    ids = tok(CLEAN[0], return_tensors="pt").input_ids
    cap = QueryCapture(model, [1], torch.tensor([ids.shape[1] - 1]))
    try:
        with torch.no_grad():
            model(ids)
    finally:
        cap.remove(); h.remove()
    H = w.n_heads(model)
    want = seen["q"].view(1, ids.shape[1], H, -1)[0, -1] if seen["q"].ndim == 3 else seen["q"][0, -1]
    assert torch.allclose(cap.q[1][0], want)


def test_rank_target_search_and_transport(tiny, tmp_path):
    model, tok = tiny
    ranked = w.find_query_heads(model, tok, CLEAN, CORRUPT)
    assert len(ranked) == 4 * 4 and all(isinstance(e, float) for _, e in ranked)
    heads = [hl for hl, _ in ranked[:3]]
    qt = w.query_target(model, tok, CLEAN, heads, name="fruit")
    assert qt.heads == sorted(heads) and all(v.ndim == 1 for v in qt.queries.values())
    probes = ["Options: Car, Apple, Dog. Answer:", "Options: Bus, Pear, Cat. Answer:"]
    res = w.whisper(model, qt, tokenizer=tok, steps=3, n_tokens=4, probes=probes, verbose=False)
    assert res.history[-1] >= res.history[0]
    assert abs(w.score(model, qt, res.text, tokenizer=tok, probes=probes) - res.score) < 1e-5
    res.save(str(tmp_path / "q.json"))
    back = w.Whisper.load(str(tmp_path / "q.json"))
    assert isinstance(back.target, w.QueryTarget) and back.target.heads == qt.heads
    # transport: inside the context the patched heads' last-token queries ARE the target vectors
    ids = tok(probes[0], return_tensors="pt").input_ids
    with w.patched_queries(model, qt.queries), torch.no_grad():
        cap = QueryCapture(model, qt.layers, torch.tensor([ids.shape[1] - 1]))    # registered after: sees the patch
        try:
            model(ids)
        finally:
            cap.remove()
    for (L, h), v in qt.queries.items():
        assert torch.allclose(cap.q[L][0, h].float(), v, atol=1e-5)


def test_fused_qkv_is_refused(tiny_gpt2):
    model, tok = tiny_gpt2
    with pytest.raises(ValueError):
        _query_module(w.search._blocks(model)[0])
