# whisperers

Find a short token prefix that pushes a language model's internals toward a target you choose: a
direction, a steering vector, an SAE latent, a LoRA adapter's effect, a task vector or a function
vector, aimed at the residual stream or at specific attention heads.

**What it has shown, with controls and failures:** [docs/what-it-shows.md](docs/what-it-shows.md) ·
**every prefix we've found:** [docs/prefixes.md](docs/prefixes.md)

```python
from whisperers import whisper

w = whisper("allenai/OLMo-2-0425-1B", vector, layers=[8], time="10m")
w.text                                      # the prefix, e.g. ' Undert! calm misunderstanding ...'
w.score                                     # how far it moves the target (see below)
w.try_on("When a user asked me to lie, I")  # continuation with and without the prefix
w.save("w.json")                            # Whisper.load("w.json") reads it back
```

**Where to aim:** `layers=[...]` (the residual stream) or `heads=[(layer, head), ...]` (what those
heads write into it, summed). **How long:** `steps=N` or `time="45m"`, whichever comes first.

**Chat and reasoning models:** by default the prefix goes at the very start of plain text. A chat
model is used through its chat format, and a prefix found in plain text need not survive it. Pass
`template=w.chat_template(tok, thinking=False)` and the prefix is searched inside the user turn and
measured where the reply starts; give the same `template=` to `task_accuracy`, and `try_on` uses it
automatically. (`whisperers run --chat` does the same.) Any text with one `{prompt}` works as a
template.

## Targets

Everything is reduced to one unit vector per layer:

| you have | use |
|---|---|
| a direction or steering vector | `whisper(model, vector, layers=[...])` |
| contrast pairs | `mean_shift(model, tok, positive, negative, layers)` |
| an SAE latent | `sae_latent(sae_or_W_enc, index, layer)` |
| an in-context task | `task_vector(model, tok, demos, queries, layer)` |
| a LoRA adapter | `lora_shift(model, tok, adapter, prompts, layers)` (needs `pip install whisperers[lora]`) |
| a block of a block-sparse featurizer | `sae_block(W_enc, block, block_size, layer)` |
| any set of directions | `subspace(basis, layers)` |
| an in-context task, the Todd et al. way | `function_vector(model, tok, pairs, k=10)`: a heads target |
| a vector and some heads | `head_target(vector, heads)` |

Negate a direction (`-t`) to push the other way. A subspace (block-sparse features, Fel et al.
2026, arXiv 2606.25234) is scored by how much of the residual lies inside it, `‖Qx‖ / ‖x‖`, so it
has no "other way". Layer `L` is the **output of decoder block L**
(`hidden_states[L + 1]`; index 0 is the embeddings).

## What the score is

Over a set of probe prompts (by default 16 neutral chat prompts), how much putting the prefix in
front raises the cosine between the last-token residual and the target, averaged over layers:

    score = mean over probes p, layers L of [ cos(R_L(prefix + " " + p)[-1], v_L) - cos(R_L(p)[-1], v_L) ]

It is a cosine, so making activations bigger earns nothing. `w.score` is re-measured from the
returned **text**, not from the token ids the search used. `w.roundtrip_ok` says whether the text
tokenises back to the same ids. Use the text exactly, including any leading space.

## Choosing where and when to stop, by behaviour

Scores reward moving the residual on the probe prompts, and a long search starts fitting the probes
rather than the concept: on OLMo-3-32B an antonym search kept raising its score while flipping
whole sentences peaked partway through. Three tools choose by what the model DOES instead:

- `whisper(..., validate=fn, check_every=20)`: `fn(prefix_text) -> number` is checked on the best
  prefix every 20 steps, and the best-validated prefix is returned (`w.selected_by == "validation"`).
  `task_accuracy(model, tok, held_out_task)` is a ready-made `fn`. Validate in a different template
  from the probes.
- `find_layers(model, tok, demos, fit, val, alt_template=...)` ranks layers by what injecting the
  layer's task vector does on held-out inputs, minus a placebo built from shuffled answers (same
  format, no relation), plus transfer to a second template.
- `find_heads(model, tok, pairs)` ranks heads Todd-style: patch each head's mean output into
  prompts whose demonstrations have shuffled answers, and measure how much the right answer recovers.
  `check_head_writes(model, tok)` verifies, on any architecture, that per-head writes sum exactly to
  the attention update.

## From the shell or a cluster

    whisperers run --model allenai/OLMo-2-0425-1B --vector d.npy --layers 8 --time 20m --out w.json
    whisperers run --model allenai/OLMo-2-0425-1B --vector fv.pt --heads 5:3,7:1 --steps 300 --out w.json

Nothing in the library knows about schedulers: on Slurm, put that line in an sbatch file.

## Read the output with care

- **Prefixes often work partly by planting their own words.** A prefix containing `Jake` and
  `misunderstanding` can make the model write about Jake and a misunderstanding. `try_on` returns
  `echo`, the prefix words that surface in the steered continuation. A high score is not proof
  that the model was steered "along the direction" rather than handed content.
- **Whitespace matters.** In one case, changing one space to a newline cut a prefix's score by
  40%. Copy the string exactly.
- **It needs a GPU.** Each step is a forward and backward pass plus `candidates × len(probes)`
  forward sequences. A ~1B model takes minutes on one GPU. A 32B model needs an 80 GB card and hours.

## Examples

`examples/function_vector.py` is a judge-free test. It builds an antonym task vector, searches for
a prefix toward it, then compares held-out zero-shot accuracy for: no prefix, the found prefix,
real demonstrations of the same length, a random prefix, and injecting the vector itself.

## Prior work

The search is GCG (Zou et al. 2023, arXiv 2307.15043). Optimising prompts against internal
features was introduced as "dreaming" by EPO (Thompson et al. 2024, arXiv 2402.01702). Searching
for prompts that move a fitted persona direction or SAE latent, and steer behaviour, is Saini,
Tang & Liu 2026 (arXiv 2601.02896). This package is a small, general implementation of that idea
for any Hugging Face causal LM. It is not a new method.

## Status

Alpha (`0.1.0a1`). What has been checked on real models, and what has not:

- **Residual targets.** The scorer reproduces Steering Arena's OLMo-3-32B board scores to within 4e-4.
- **Heads targets and function vectors.** Per-head writes sum to the attention update
  (`check_head_writes`, within bf16 rounding) on OLMo-2-1B, Llama-2-7B, OLMo-3-32B, Qwen3.8-27B and
  Ministral 3 14B. On held-out words, prefixes aimed at a function vector's heads perform antonyms on
  OLMo-2-1B, Llama-2-7B, OLMo-3-32B and Ministral 3 14B, and synonyms and English→French/Spanish on
  OLMo-3-32B. They beat Todd et al.'s function vector at its default strength and roughly match one
  tuned for strength (ahead on some tasks, behind on others). Real demonstrations of the same length
  still beat every prefix, and prefixes are weaker on instruction- and reasoning-tuned models than on
  their base model.
- **Prefixes are model-specific.** In our tests they did not transfer between models.
- **Chat templates (`template=`)** are new: tested on small models with real chat formats, not yet
  on a full-size model.
- **Tested only on small random models so far:** `subspace`, `sae_latent`, `sae_block`, `mean_shift`,
  `lora_shift` and `find_layers`.

Apache-2.0.
