"""A tiny randomly initialised OLMo-3 on CPU: real architecture and tokenizer, seconds to run.

Random weights mean the scores are meaningless; the tests check plumbing -- that the right
residual is read, the search improves its own objective, and the guards fire.
"""
import pytest
import torch

TOKENIZER_ID = "allenai/Olmo-3-1125-32B"   # tokenizer only; the weights are never loaded


@pytest.fixture(scope="session")
def tiny(tmp_path_factory):
    transformers = pytest.importorskip("transformers")
    try:
        cfg = transformers.AutoConfig.from_pretrained(TOKENIZER_ID)
        tok = transformers.AutoTokenizer.from_pretrained(TOKENIZER_ID)
    except OSError as e:
        pytest.skip(f"tokenizer/config not available: {e}")
    cfg.hidden_size, cfg.intermediate_size, cfg.num_hidden_layers = 64, 128, 4
    cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim = 4, 2, 16
    cfg.layer_types = ["sliding_attention"] * 3 + ["full_attention"]
    torch.manual_seed(0)
    model = transformers.AutoModelForCausalLM.from_config(cfg, dtype=torch.float32)
    path = tmp_path_factory.mktemp("tiny_olmo3")
    model.save_pretrained(path)
    tok.save_pretrained(path)
    model = transformers.AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()
    return model, tok


@pytest.fixture(scope="session")
def tiny_llama():
    """Pre-norm architecture: the attention output is added to the residual un-normalised."""
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained(TOKENIZER_ID)
    except OSError as e:
        pytest.skip(f"tokenizer not available: {e}")
    cfg = transformers.LlamaConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                                   num_attention_heads=4, num_key_value_heads=2, attention_bias=True)
    torch.manual_seed(1)
    return transformers.LlamaForCausalLM(cfg).eval(), tok


@pytest.fixture(scope="session")
def tiny_gpt2():
    """GPT-2 stores its output projection transposed (Conv1D)."""
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained("gpt2")
    except OSError as e:
        pytest.skip(f"gpt2 tokenizer not available: {e}")
    torch.manual_seed(2)
    cfg = transformers.GPT2Config(vocab_size=len(tok), n_embd=64, n_layer=3, n_head=4)
    return transformers.GPT2LMHeadModel(cfg).eval(), tok
