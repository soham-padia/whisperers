# What whisperers shows, and what it doesn't yet

Put this string in front of an English word, and OLMo-3-32B answers in Spanish:

```
etry grated................................................................ filtration ! lírado industrialDuplicate CollinsAREST FRAME ! CABREA Alto notwithstanding■'^ gag vodka physicistOd low María (), ! !Dump --> menos Presidents discussed ->
```

It answers `discutido`. Without the string, the model mostly repeats `discussed`. Nobody wrote that
string. A search found it by aiming at what ten attention heads write into the model when it
translates from examples: the heads behind a *function vector* (Todd et al. 2024). On 200 English
words the search never saw, the string gets 153 into Spanish. With no prefix the model gets 0.

That is the whole idea in small. whisperers takes something you can point to inside a model (a
direction, some heads, a function vector) and finds plain tokens that make the model do it. This
page sets out the evidence, the controls that make it believable, and the places it breaks. Unless
we say otherwise, every number comes from a single run. Every prefix we have found is listed in
[prefixes.md](prefixes.md) (machine-readable: [prefixes.json](prefixes.json)).

## How we test

We use Todd et al.'s in-context tasks: antonyms, synonyms, English→French, English→Spanish and
country→capital. A search sees 20 probe words in the form `word ->` and nothing else. We then test
on **200 different words** (137 for capitals). An answer counts only if the first word of the
greedy output is exactly the dataset's answer, so no judge model is involved. Arms are compared on
the same words with a paired sign test. When we use `validate=`, it picks the search checkpoint that
does best on more held-out words phrased a different way (for example "The opposite of X is").

## The evidence

**Controls.** Every result sits next to the same nulls:
- No prefix.
- A random string of the same length, which scored 0/200 every time.
- A placebo prefix, searched toward a vector built from the same demonstrations with the answers
  scrambled. On the translation swap test it gave 0 French and 0 Spanish.
- For the function vectors, the same vector built from 10 random heads instead of the top 10. It
  does nothing: 1 vs the real vector's 27 on OLMo-2-1B, 11 vs 73 on Llama-2-7B, and 23 vs 53 on
  OLMo-3-32B, where no prefix also gives 23.

**Held-out words, no judge.** Every score is exact-match accuracy on words the search never saw. On
OLMo-3-32B antonyms, a prefix searched toward a task vector goes from 22 to 107/200 (p ≈ 1e-22).

**The swap test: the function, not just the format.** We used the same 200 English words and the
same `word ->` format. The prefix searched toward the French vector gets 93 French answers and 0
Spanish; the Spanish one gets 0 French and 157 Spanish; the placebo gets 0 and 0. On Llama-2-7B,
antonym against synonym on shared words: the antonym prefix gives 28 antonyms and 3 synonyms, the
synonym prefix 5 and 58 (p = 8e-16 for the synonym prefix against the antonym prefix).

**Ablation: the readable part does the work.** Take the readable pair out of the 16-token OLMo-3-32B
antonym prefix and accuracy falls from 107 to 5. Remove only the word `opposite` and it falls to 82.
The pair on its own gets 2, because the model just copies it. On OLMo-2-1B it is the same: 38 falls
to 2.

**The tokens name the task.** We never asked for readable text, yet the prefixes contain `opposite`,
`pleasant -> uneasy`, `honeymoon -> war`, `presidential -> presidente`, `living ==> morte`,
`accomplish -> happen` and `--> menos`. Along one search, whole-sentence flipping switches on as
the operation gets a name: 1 of 15 sentences with the arrow format alone, 7 when `Reverse` appears,
13 when it becomes `opposite`, and 15 once pairs follow.

**From words to sentences.** The 32-token antonym prefix was searched on single words. It flips
whole sentences it never saw: "The tall old man slowly climbed up the narrow stairs at night"
becomes "The short young man quickly ran down the wide stairs in the morning". It did this on about
14 of 15 sentences, against 1 of 15 without it (hand-judged; see the gaps below).

**Aiming at heads matters.** We aimed the same function vector at the residual stream instead of at
its heads, and in most cases the result was worse:

| model, task | heads target | residual target |
|---|---|---|
| OLMo-3-32B, antonyms | 117 | 15 (155 copied) |
| OLMo-3-32B, synonyms and both translations | 79–153 | 0–1 |
| OLMo-2-1B, antonyms | 48 | 0 |
| Llama-2-7B, antonyms (the exception) | 104 | 93 |

**Against the strongest baseline we know.** We compared prefixes with injecting the function vector
directly into the model.
- *At Todd et al.'s default strength* the heads prefix wins on every base model: 48 vs 27
  (OLMo-2-1B), 104 vs 73 (Llama-2-7B), 117 vs 53 (OLMo-3-32B antonyms), 128 vs 94 (Ministral 3
  14B), and 153 vs 1 for Spanish.
- *Against a vector tuned for layer and strength on validation words* (which favours the vector):
  - The prefix wins on OLMo-3-32B synonyms (91 vs 69, p = 0.004), French (119 vs 50) and Spanish
    (153 vs 71).
  - It roughly ties on OLMo-2-1B (48 vs 35, p = 0.053), OLMo-3-32B antonyms (117 vs 105, p = 0.13)
    and Ministral Base (128 vs 130).
  - It loses on Llama-2-7B (104 vs 118, p = 0.044).

**Real examples still win, and we say so first.** Complete example pairs in the same token budget
beat every prefix: 140 vs 117, 168 vs 119, 173 vs 153. A found prefix is not a better prompt than
real examples. Our view is that its value is different: it is a readout, in tokens, of what an
internal target does.

**The scorer is checked against a public leaderboard.** whisperers' residual scorer reproduces
[Steering Arena](https://sohampadianeu-steering-arena.hf.space)'s OLMo-3-32B board scores to within
4e-4. A 32-token search reached +0.164, against the board's best of +0.108.

**Corrections we have made.** We withdrew two claims after better baselines came in:
- *"A prefix beats real examples of the same length."* This held only against examples cut off
  mid-pair. With complete pairs, real examples win everywhere: 133 vs 107 and 138 vs 111.
- *"A heads prefix beats injecting the function vector."* This holds at the default strength only.
  Tuned, the vector wins on Llama-2-7B.

We also report our first Qwen3.8-27B run (0/200) as a failed search, not a negative result. It got
99 steps, and the prefix it picked was still mostly the `! ! !` it started from.

## Where it fails

**Prefixes are model-specific.** OLMo-3-32B prefixes get 0 on OLMo-2-1B. The 1B prefix on OLMo-3-32B
gets 7, which is *below* the 22 with no prefix.

**Some functions don't come out.** Country→capital on OLMo-3-32B scored 0/137 twice, because the
search ended in a code-like string that the model copies. Injecting the same vector gets 121–123,
and on OLMo-2-1B the same search works (106/137).

**Post-training makes it harder.** Ministral 3 14B's base, instruct and reasoning models all come
from one base model:

| model | heads prefix (antonyms) | tuned vector injection |
|---|---|---|
| Ministral 3 14B Base | 128 | 130 |
| Ministral 3 14B Instruct | 105 | 124 |
| Ministral 3 14B Reasoning | 46 | 118 |

On OLMo 3.1 32B the drop is total:
- Think scores 0–4 on all four tasks: antonyms 0, synonyms 2–4, French 0 and Spanish 0. Tuned
  injection still gets 94, 59, 17 and 28.
- Instruct scores 0 on French, and there even injection nearly fails (9).

One caveat, which also appears among the gaps: we tested these chat and reasoning models in plain
text, not in their chat format.

**Long searches fit the format.** The end-of-search French and Spanish prefixes get 119 and 153 in
the `word ->` format. On words phrased "In French, X is" they score 0.03 and 0.00 (on validation
words). Earlier checkpoints carry over better (0.27 and 0.17). `validate=` exists for exactly this.

**Off-target behaviour.** We aimed a prefix at the antonym heads. Used alone, with no word after it,
on the base model, it went on into a hateful, looping rant. The search only constrains the one
position it scores; what a prefix does anywhere else is unconstrained. Test a prefix outside its
format before you rely on it.

**What went wrong on the way.** Our first demo picked a layer where injecting the vector did almost
nothing (1 of 30), so the prefix had nothing real to imitate. Since then we check every target by
injection before searching toward it.

## Gaps we know about

- ⚠ **One seed per result.** None of the headline cells has been repeated with another seed.
- ⚠ **Sentence flips are hand-judged,** on only 15 sentences. The fix is a set of 100 sentences scored
  mechanically.
- ⚠ **No blind readout yet.** Can a reader name the task from the prefix alone? We claim the tokens
  name the task but have not tested it blind.
- ⚠ **Chat format at full scale.** `template=chat_template(tok)` searches inside a chat model's user
  turn. It is tested on small models; full-size runs are in progress.
- ⚠ **Reasoning models are scored the wrong way.** We read the first five output tokens, and a
  reasoning model's first tokens are its thinking. A fair test scores the answer after `</think>`,
  and the better target is the start of the thinking itself ("the user wants the opposite").
- ⚠ **Exact match undercounts.** Valid answers outside the dataset count as wrong: speedy → "fast",
  lazy → "industrious".

## Try it

The snippet below needs a GPU with about 65 GB free for OLMo-3-32B. Change the words, or swap in any
id from [prefixes.md](prefixes.md); the OLMo-2-1B prefixes run on almost anything.

```python
import json, urllib.request
from transformers import AutoModelForCausalLM, AutoTokenizer

url = "https://raw.githubusercontent.com/soham-padia/whisperers/main/docs/prefixes.json"
prefixes = json.load(urllib.request.urlopen(url))
p = prefixes["Olmo-3-1125-32B/english-spanish/fv_heads/final/1187355"]

tok = AutoTokenizer.from_pretrained(p["model"])
model = AutoModelForCausalLM.from_pretrained(p["model"], dtype="bfloat16", device_map="auto")
for word in ["discussed", "window", "courage"]:
    for text in (f"{word} ->", f"{p['text']} {word} ->"):
        ids = tok(text, return_tensors="pt").to(model.device)
        out = model.generate(**ids, max_new_tokens=5, do_sample=False)
        print(repr(text[-20:]), "→", tok.decode(out[0, ids.input_ids.shape[1]:]))
```

## Credits

The method is GCG (Zou et al. 2023) applied to internal targets. The closest prior work is EPO
(Thompson et al. 2024) and Saini, Tang & Liu 2026; see the [README](../README.md#prior-work). The
function vectors follow Todd et al. 2024, and the task data are theirs. The experiments ran on the
AICR GPU cluster.
