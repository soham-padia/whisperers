"""A list of targets: one prefix for several targets at once, each measured on its own probes."""
import pytest
import torch

import whisperers as w

FRUIT = ["Options: Car, Apple, Dog. Which is a fruit? Answer:", "Options: Bus, Pear, Cat. Which is a fruit? Answer:"]
VEHICLE = ["Options: Car, Apple, Dog. Which is a vehicle? Answer:", "Options: Bus, Pear, Cat. Which is a vehicle? Answer:"]
CUE_FRUIT = ["fruit\nOptions: Car, Apple, Dog.\nAnswer:", "fruit\nOptions: Bus, Pear, Cat.\nAnswer:"]
CUE_VEHICLE = ["vehicle\nOptions: Car, Apple, Dog.\nAnswer:", "vehicle\nOptions: Bus, Pear, Cat.\nAnswer:", "vehicle\nOptions: Van.\nAnswer:"]


def test_joint_query_targets_score_is_the_mean_of_their_scores(tiny, tmp_path):
    model, tok = tiny
    heads = [hl for hl, _ in w.find_query_heads(model, tok, FRUIT, VEHICLE)[:3]]
    targets = [w.query_target(model, tok, FRUIT, heads, name="fruit"),
               w.query_target(model, tok, VEHICLE, heads, name="vehicle")]
    probes = [CUE_FRUIT, CUE_VEHICLE]                     # different probes, different counts
    res = w.whisper(model, targets, tokenizer=tok, steps=3, n_tokens=4, probes=probes, verbose=False)
    assert all(b >= a for a, b in zip(res.history, res.history[1:]))
    joint = w.score(model, targets, res.text, tokenizer=tok, probes=probes)
    assert abs(joint - res.score) < 1e-5
    each = [w.score(model, t, res.text, tokenizer=tok, probes=p) for t, p in zip(targets, probes)]
    assert abs(joint - sum(each) / 2) < 1e-5              # mean over targets, not over all probes pooled
    res.save(str(tmp_path / "joint.json"))
    back = w.Whisper.load(str(tmp_path / "joint.json"))
    assert [t.name for t in back.target] == ["fruit", "vehicle"]
    assert all(isinstance(t, w.QueryTarget) and t.heads == targets[0].heads for t in back.target)


def test_joint_directions_share_one_probe_list(tiny):
    model, tok = tiny
    d = model.config.hidden_size
    torch.manual_seed(1)
    a, b = w.direction(torch.randn(d), [1], name="a"), w.direction(torch.randn(d), [2], name="b")
    res = w.whisper(model, [a, b], tokenizer=tok, steps=2, n_tokens=3, verbose=False)
    each = [w.score(model, t, res.text, tokenizer=tok) for t in (a, b)]
    assert abs(res.score - sum(each) / 2) < 1e-5


def test_joint_guards(tiny):
    model, tok = tiny
    d = model.config.hidden_size
    a, b = w.direction(torch.randn(d), [1]), w.direction(torch.randn(d), [2])
    with pytest.raises(ValueError, match="probe lists"):
        w.score(model, [a, b], "x", tokenizer=tok, probes=[["p"], ["q"], ["r"]])
    with pytest.raises(ValueError, match="one output_target"):
        w.score(model, [a, w.output_target([("hot ->", " cold")])], "x", tokenizer=tok)
    with pytest.raises(ValueError, match="only targets"):
        w.score(model, [a, torch.randn(d)], "x", tokenizer=tok)
    with pytest.raises(ValueError, match="already says where"):
        w.score(model, [a, b], "x", layers=[1], tokenizer=tok)
