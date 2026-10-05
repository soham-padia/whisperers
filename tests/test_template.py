"""Searching inside a template (a chat turn): the prefix and probe go at `{prompt}`, the target is read
at the template's last token, and validation / answers / try_on see the same text the search saw."""
import copy

import pytest
import torch

import whisperers as w
from whisperers.search import _Scorer, _split, encode

TEMPLATE = "<|user|>\n{prompt}\n<|assistant|>\n"
CHAT_JINJA = ("{% for m in messages %}<|{{ m['role'] }}|>\n{{ m['content'] }}\n{% endfor %}"
              "{% if add_generation_prompt %}<|assistant|>\n"
              "{% if enable_thinking is defined and not enable_thinking %}<think></think>\n{% endif %}{% endif %}")


def test_split_needs_exactly_one_slot():
    assert _split(TEMPLATE) == ("<|user|>\n", "\n<|assistant|>\n")
    for bad in ("no slot", "{prompt} and {prompt}"):
        with pytest.raises(ValueError):
            _split(bad)


def test_chat_template_matches_apply_chat_template(tiny):
    _, tok = tiny
    tok = copy.deepcopy(tok)
    tok.chat_template = CHAT_JINJA
    for thinking in (None, False):
        t = w.chat_template(tok, thinking=thinking)
        kw = {} if thinking is None else {"enable_thinking": thinking}
        want = tok.apply_chat_template([{"role": "user", "content": "hot ->"}], tokenize=False,
                                       add_generation_prompt=True, **kw)
        assert w.wrap(t, "hot ->") == want
        assert ("<think></think>" in t) == (thinking is False)
    sys_t = w.chat_template(tok, system="Be brief.")
    assert sys_t.startswith("<|system|>\nBe brief.\n<|user|>\n{prompt}")


def test_template_search_reads_where_the_reply_starts(tiny):
    model, tok = tiny
    probes = ["hot ->", "up ->"]
    torch.manual_seed(0)
    target = w.direction(torch.randn(64), [2])
    res = w.whisper(model, target, tokenizer=tok, steps=3, n_tokens=4, probes=probes, template=TEMPLATE,
                    verbose=False)
    assert res.template == TEMPLATE and res.roundtrip_ok
    # the search's token sequences are exactly the wrapped text, special tokens from the template only
    sc = _Scorer(model, tok, target, probes, 64, TEMPLATE)
    for p, seq in zip(probes, sc.sequences(res.ids)):
        assert seq == tok(w.wrap(TEMPLATE, w.compose(res.text, p)), add_special_tokens=False).input_ids
    # .score is the same number score() reports, and both read the template's LAST token
    again = w.score(model, target, res.text, tokenizer=tok, probes=probes, template=TEMPLATE)
    assert abs(again - res.score) < 1e-5
    manual = []
    with torch.no_grad():
        for text in [w.compose(res.text, p) for p in probes] + probes:
            ids = torch.tensor([encode(tok, text, TEMPLATE)])
            h = model(ids, output_hidden_states=True).hidden_states[3][0, -1]
            manual.append(torch.cosine_similarity(h.float(), target.vectors[2][None], dim=-1).item())
    assert abs((sum(manual[:2]) - sum(manual[2:])) / 2 - res.score) < 1e-4
    # and the template changes what is measured: the same prefix scores differently without it
    assert abs(w.score(model, target, res.text, tokenizer=tok, probes=probes) - res.score) > 1e-6


def test_answer_and_try_on_use_the_template(tiny, tmp_path):
    model, tok = tiny
    ids = torch.tensor([tok(w.wrap(TEMPLATE, "hot ->"), add_special_tokens=False).input_ids])
    want = tok.decode(model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=3, do_sample=False,
                                     pad_token_id=tok.eos_token_id)[0, ids.shape[1]:], skip_special_tokens=True)
    assert w.tasks.answer(model, tok, "hot ->", max_new=3, template=TEMPLATE) == want
    task = w.Task([("hot", "cold"), ("up", "down")])
    validate = w.task_accuracy(model, tok, task, template=TEMPLATE)
    assert 0.0 <= validate(" ! !") <= 1.0
    res = w.whisper(model, w.direction(torch.randn(64), [1]), tokenizer=tok, steps=1, n_tokens=3,
                    probes=["hot ->"], template=TEMPLATE, verbose=False)
    assert res.try_on("hot ->", max_new_tokens=3)["without"] == want
    res.save(str(tmp_path / "w.json"))
    assert w.Whisper.load(str(tmp_path / "w.json")).template == TEMPLATE
