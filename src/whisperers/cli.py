"""`whisperers run ...` -- one search from the shell, so any scheduler can run it.

    whisperers run --model allenai/OLMo-2-0425-1B --vector d.npy --layers 8 --time 20m --out w.json
    whisperers run --model allenai/OLMo-2-0425-1B --vector fv.pt --heads 5:3,7:1,9:12 --steps 300 --out w.json

A Slurm job is then just this command inside an sbatch file. The vector file may be .npy, .npz
(key `d`, or the only array), .pt/.pth (a tensor, or a dict with `d`/`vector`), or .safetensors (the
only tensor). The result is a saved Whisper (JSON): load it with `whisperers.Whisper.load(path)`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_vector(path: str):
    import numpy as np
    import torch

    p = Path(path)
    if p.suffix == ".npy":
        return torch.as_tensor(np.load(p))
    if p.suffix == ".npz":
        z = np.load(p, allow_pickle=False)
        key = "d" if "d" in z.files else (z.files[0] if len(z.files) == 1 else None)
        if key is None:
            raise SystemExit(f"{p}: several arrays ({z.files}); keep one, or name it `d`")
        return torch.as_tensor(z[key])
    if p.suffix in (".pt", ".pth"):
        obj = torch.load(p, map_location="cpu", weights_only=True)
        if isinstance(obj, dict):
            obj = obj.get("d", obj.get("vector"))
        return torch.as_tensor(obj)
    if p.suffix == ".safetensors":
        from safetensors.torch import load_file
        t = load_file(p)
        if len(t) != 1:
            raise SystemExit(f"{p}: expected one tensor, found {list(t)}")
        return next(iter(t.values()))
    raise SystemExit(f"{p}: unsupported vector file (use .npy, .npz, .pt, .safetensors)")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="whisperers")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="search for a prefix toward a vector")
    r.add_argument("--model", required=True, help="Hugging Face model id or local path")
    r.add_argument("--vector", required=True, help="file holding the target vector")
    where = r.add_mutually_exclusive_group(required=True)
    where.add_argument("--layers", help="comma-separated layers, e.g. 20 or 19,23,27")
    where.add_argument("--heads", help="comma-separated layer:head pairs, e.g. 5:3,7:1")
    r.add_argument("--steps", type=int)
    r.add_argument("--time", help="budget, e.g. 90s, 45m, 2h")
    r.add_argument("--n-tokens", type=int, default=20)
    r.add_argument("--probes", help="text file, one probe prompt per line (default: 16 neutral prompts)")
    r.add_argument("--negate", action="store_true", help="push the other way")
    r.add_argument("--init", help="resume from this prefix text")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--log", help="JSONL file to append each step's best prefix to")
    r.add_argument("--chat", action="store_true",
                   help="search inside the model's chat template (user turn; reasoning block off where supported)")
    r.add_argument("--out", required=True, help="where to save the result (JSON)")
    args = ap.parse_args(argv)

    import whisperers as w

    vec = read_vector(args.vector)
    if args.layers:
        target = w.direction(vec, [int(x) for x in args.layers.split(",")])
    else:
        target = w.head_target(vec, [tuple(int(v) for v in hl.split(":")) for hl in args.heads.split(",")])
    if args.negate:
        target = -target
    probes = None
    if args.probes:
        probes = [line.strip() for line in Path(args.probes).read_text().splitlines() if line.strip()]
    model, tok = w.load(args.model)
    template = w.chat_template(tok, thinking=False) if args.chat else None
    res = w.whisper(model, target, tokenizer=tok, steps=args.steps, time=args.time, n_tokens=args.n_tokens,
                    probes=probes, init=args.init, seed=args.seed, log=args.log, template=template)
    res.save(args.out)
    print(json.dumps({"text": res.text, "score": res.score, "roundtrip_ok": res.roundtrip_ok, "out": args.out}))


if __name__ == "__main__":
    main()
