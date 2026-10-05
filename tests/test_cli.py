import json

import numpy as np
import pytest

from whisperers.cli import main, read_vector


def test_read_vector_formats(tmp_path):
    torch = pytest.importorskip("torch")
    v = np.arange(4, dtype=np.float32)
    np.save(tmp_path / "a.npy", v)
    np.savez(tmp_path / "b.npz", d=v, other=v)
    torch.save({"vector": torch.tensor(v)}, tmp_path / "c.pt")
    for f in ("a.npy", "b.npz", "c.pt"):
        assert read_vector(str(tmp_path / f)).tolist() == [0.0, 1.0, 2.0, 3.0]


def test_cli_run_layers_and_heads(tiny, tmp_path, monkeypatch, capsys):
    """End to end through the CLI, on the tiny model saved to disk."""
    model, tok = tiny
    path = tmp_path / "m"
    model.save_pretrained(path)
    tok.save_pretrained(path)
    np.save(tmp_path / "v.npy", np.random.default_rng(0).standard_normal(64).astype(np.float32))
    (tmp_path / "probes.txt").write_text("hot ->\nup ->\n")
    for where in (["--layers", "1"], ["--heads", "1:0,2:3"]):
        out = tmp_path / f"w_{where[0][2:]}.json"
        main(["run", "--model", str(path), "--vector", str(tmp_path / "v.npy"), *where, "--steps", "2",
              "--n-tokens", "4", "--probes", str(tmp_path / "probes.txt"), "--out", str(out)])
        saved = json.loads(out.read_text())
        assert saved["text"] and len(saved["ids"]) == 4
        assert (saved["target"]["heads"] is None) == (where[0] == "--layers")
