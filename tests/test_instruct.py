"""Instructed tasks inside a chat turn: heads ranked by patching "opposite" requests into "synonym"
requests (judged by the model's own answer), at the reply or at the word; and the scorer reading at the
end of the user's text (read="prompt") instead of at the reply."""
import copy

import pytest
import torch

import whisperers as w
from whisperers.heads import _locate
from whisperers.search import _after_tokens, encode

CHAT_JINJA = ("{% for m in messages %}<|{{ m['role'] }}|>\n{{ m['content'] }}\n{% endfor %}"
              "{% if add_generation_prompt %}<|assistant|>\n{% endif %}")
ASK = "Give me the opposite of {x}. Answer with one word."
CONTRAST = "Give me a synonym of {x}. Answer with one word."
WORDS = ["hot", "big", "fast", "light", "happy", "early", "loud", "rich"]


@pytest.fixture(scope="module")
def chat(tiny):
    model, tok = tiny
    tok = copy.deepcopy(tok)
    tok.chat_template = CHAT_JINJA
    return model, tok, w.chat_template(tok)


def test_locate_finds_the_words_last_token(chat):
    _, tok, T = chat
    for word in ("hot", "unconstitutional"):
        ids, i = _locate(tok, ASK.format(x=word), T, word)
        assert ids == encode(tok, ASK.format(x=word), T)
        assert word.endswith(tok.decode(ids[i]).strip()) and tok.decode(ids[i]).strip()


def test_read_prompt_is_the_users_last_token(chat):
    model, tok, T = chat
    n = _after_tokens(tok, T, "prompt")
    assert n > 0 and _after_tokens(tok, T, "reply") == 0
    ids = encode(tok, "x hot", T)
    assert tok.decode(ids[len(ids) - 1 - n]).strip() == "hot"
    torch.manual_seed(0)
    target = w.direction(torch.randn(64), [2])
    res = w.whisper(model, target, tokenizer=tok, steps=2, n_tokens=3, probes=["hot", "big"], template=T,
                    read="prompt", verbose=False)
    assert res.read == "prompt"
    manual = []
    with torch.no_grad():
        for text in [w.compose(res.text, p) for p in ("hot", "big")] + ["hot", "big"]:
            seq = torch.tensor([encode(tok, text, T)])
            h = model(seq, output_hidden_states=True).hidden_states[3][0, seq.shape[1] - 1 - n]
            manual.append(torch.cosine_similarity(h.float(), target.vectors[2][None], dim=-1).item())
    assert abs((sum(manual[:2]) - sum(manual[2:])) / 2 - res.score) < 1e-4
    with pytest.raises(ValueError):
        w.score(model, target, res.text, tokenizer=tok, probes=["hot"], template=T, read="middle")


@pytest.mark.parametrize("at", ["reply", "word"])
def test_instruction_heads_and_vector(chat, at):
    model, tok, T = chat
    ranked = w.instruction_heads(model, tok, WORDS, ASK, CONTRAST, template=T, at=at)
    assert len(ranked) == 4 * 4 and all(isinstance(e, float) for _, e in ranked)
    fv = w.instruction_vector(model, tok, WORDS, ASK, CONTRAST, k=3, template=T, at=at, ranked=ranked)
    assert sorted(fv.heads) == sorted(hl for hl, _ in ranked[:3]) and fv.meta["at"] == at
    assert fv.meta["vector"].shape == (64,) and set(fv.meta["words"]) <= set(WORDS)
    # the target is usable as-is: a search reading where the vector was built
    res = w.whisper(model, fv, tokenizer=tok, steps=1, n_tokens=3, probes=["cold", "tall"], template=T,
                    read="reply" if at == "reply" else "prompt", verbose=False)
    assert res.text
    with pytest.raises(ValueError):
        w.instruction_heads(model, tok, WORDS, ASK, CONTRAST, template=T, at="nowhere")


def test_inject_at_prompt_hits_the_users_last_token(chat):
    model, tok, T = chat
    seen = {}
    blocks = w.search._blocks(model)
    h = blocks[1].register_forward_hook(lambda m, a, out: seen.setdefault("x", (out[0] if isinstance(out, tuple) else out).clone()))
    try:
        vec = torch.zeros(64)
        vec[0] = 1e4
        w.tasks.answer(model, tok, "hot", inject=(1, vec), max_new=1, template=T, at="prompt")
    finally:
        h.remove()
    x, n = seen["x"][0], _after_tokens(tok, T, "prompt")
    assert int(x[:, 0].abs().argmax()) == x.shape[0] - 1 - n
