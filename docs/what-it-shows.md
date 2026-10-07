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

**Against injecting the vector itself.** We compared prefixes with injecting the function vector
directly into the model.
- *At Todd et al.'s default strength* the heads prefix wins on every base model: 48 vs 27
  (OLMo-2-1B), 104 vs 73 (Llama-2-7B), 117 vs 53 (OLMo-3-32B antonyms), 128 vs 94 (Ministral 3
  14B), and 153 vs 1 for Spanish. ⚠ Our function vectors are built from 10 heads and 20 prompts.
  Todd et al. use more heads on larger models (20 for Llama-2-7B, about 2% of all heads in
  general) and 100 prompts, so this is a weaker vector than theirs until we rerun with their
  head count.
- *Against a vector tuned for layer and strength on validation words* (which favours the vector):
  - The prefix wins on OLMo-3-32B synonyms (91 vs 69, p = 0.004), French (119 vs 50) and Spanish
    (153 vs 71).
  - It roughly ties on OLMo-2-1B (48 vs 35, p = 0.053), OLMo-3-32B antonyms (117 vs 105, p = 0.13)
    and Ministral Base (128 vs 130).
  - It loses on Llama-2-7B (104 vs 118, p = 0.044).

**Plain GCG on the answers does as well or better.** This is Rohit Gandikota's suggested baseline:
search the same way, with the same budget, practice words and validation, but score only how likely
the correct answers are, with nothing inside the model targeted. On OLMo-3-32B it matches or beats
the heads prefix on every task:

| task | heads prefix | GCG on the answers | the same, gradient-free |
|---|---|---|---|
| antonyms | 99 | 102 | 118 |
| synonyms | 40 (91 at the end of the search) | 80 | 90 |
| English→French | 79 (119) | 165 | 160 |
| English→Spanish | 116 (153) | 152 | 162 |

So for accuracy in the format the search sees, the internal target is not needed. The answer-only
search is told the 20 practice words' answers, which the heads search never sees. Each number is
one run, and the heads antonym prefix scored 117 in an earlier run and 99 in this one.

It is not overfitting either. On 120 fresh words and in phrasings no search or checkpoint pick ever
used (`X:`, `Q: X / A:`, `X =`), answer-only GCG holds up and the heads prefixes collapse: antonym
`Q: X / A:` 53 vs 0, English→French `X:` 53 vs 0. The prefixes suggest why: the answer-only antonym
prefix writes the instruction in English (`Flip polarity — swapping`), which works in any phrasing; the
heads one plants a worked example (`ufficient (&pure')-> deficient`), which only works in the arrow
format.

**When the instruction words are banned, aiming at heads comes out ahead.** We ran the antonym comparison again,
forbidding every search from writing the instruction: 644 tokens of "opposite" and its synonyms in
several languages, and negation words. A strict version also forbids every arrow (`->`, `=>`, `→`…),
so no prefix can write an `X -> Y` example either; matches spelled across tokens are caught too.
Correct out of 120 fresh words (no prefix: 15, 0, 0, 13):

| | `X ->` | `X:` | `Q: X / A:` | `X =` |
|---|---|---|---|---|
| heads prefix, strict ban | **73** | **69** | **27** | **74** |
| GCG on the answers, strict ban | 61 | 30 | 0 | 42 |
| the same, gradient-free, strict ban | 61 | 25 | 0 | 51 |
| heads prefix, words banned | 70 | 68 | 52 | 71 |
| GCG on the answers, words banned | 67 | 18 | 3 | 40 |
| the same, gradient-free, words banned | 74 | 61 | 0 | 74 |
| for reference, no ban: heads / answers | 52 / 61 | 29 / 41 | 0 / 53 | 38 / 55 |

- With the strict ban, the heads prefix beats both answer-only searches in every phrasing, its own
  included (p from 0.04 down to 1e-11). With only the words banned it is not a clean win: the
  gradient-free search is level with the heads prefix everywhere except `Q: X / A:` (52 vs 0). Banning
  the words hurts the GCG answer-only search most: its `Q: X / A:` score goes 53 → 3 → 0. On the 200 test words of the search itself the heads prefix gets
  126 in both ban runs, against 110–112 and 100–119.
- It carries into other languages. We built 200 Spanish and 200 French antonym pairs by translating
  both words of the dataset's English pairs, and used the English-searched prefixes unchanged. The
  banned heads prefix gets 72 Spanish and 51 French (lenient match), against 59 and 31 gradient-free
  and 39 and 36 for answers-only (p from 0.011 to 1e-7); with no prefix, 7 and 1. Under the strict ban
  the margin shrinks: 46 and 37, against 26 and 15 gradient-free (significant) and 38 and 31 answers-only
  (not significant). Without the ban it is the other way round: the heads prefix gets 22 and 14 against
  53 and 36 gradient-free and 41 and 42 answers-only, often answering with the English antonym.
- What fills the gap: under the word ban the heads prefix plants an example pair (`repeal => strengthen`;
  another model, shown only that prefix, guessed "antonyms" from it). Under the strict ban it has no
  arrows and no concept words, only fragments like `unsett…`, `hardly`, `Correction`, `wider`; the
  answer-only search slipped in `UpDown`.
- ⚠ One task (antonyms), one search per method per ban, one model. The same test on synonyms and
  English→Spanish is running. Knocking out the 10 function-vector heads barely moves the banned heads
  prefixes (73 → 67, random heads 73), so for antonyms those 10 heads are not the whole mechanism.

**A second mechanism: filter heads, done the paper's way.** Sen Sharma et al. (2025) found that the rule
in a list-filtering task ("is a fruit") sits in the queries of a few attention heads. We reproduced their
method on OLMo-3-32B base: their prompt pairs, their learned mask over every head's query (65 heads,
the same 65 when re-trained), their metric (the top-scoring option) and their necessity test (the last
token may attend only to the start). The heads carry the rule: copying a source prompt's queries into
them switches the answer on 61% of held-out pairs, against 0% for random heads (the paper reports
82–86% on instruction-tuned models). Then, for "fruit" with held-out fruits:
- *Overriding a question* (the paper's setting: the prompt asks for another category): the paper's
  intervention picks the fruit 162–176 times out of 200; a prefix aimed at the same heads, 3; GCG on
  the answers, 52. A fixed string in front cannot out-vote an explicit question; editing the queries can.
- *Supplying a missing question* (a list with no question): the heads prefix picks the fruit 198 times,
  more than copying in the average fruit query (169; 30 vs 1, p = 3e-8). The paper's necessity test on
  those 65 heads drops it to 132; the same test on 65 random heads leaves 198. GCG on the answers gets
  200 and also drops, to 146, so both kinds of prefix partly run through the paper's heads.

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
- ⚠ **Blind readout is anecdotal.** Shown only the banned heads prefix, another model named the task
  ("antonyms") from its `repeal => strengthen` pair. That is one prefix and one reader, not a test.
- ⚠ **Every claim on this page is about base models.** Chat and reasoning models are work in
  progress. We have run them only in plain text, outside the chat format they are built for, which
  is not a fair test, so we make no claim about them yet. The chat-format versions (`template=`,
  `instruction_vector`, few-shot inside the chat turn) are built and running. Reasoning models also
  need scoring after `</think>`, and probably a target at the start of the thinking itself.
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
(Thompson et al. 2024), Saini, Tang & Liu 2026 (prompts toward a persona direction, on 4–8B instruct
models, scored by multiple choice), and Attention-SSR (Winninger et al. 2025), which already searches
for tokens against attention-head outputs, to suppress them for jailbreaks. What we add is aligning
head outputs with a function vector and scoring zero-shot task accuracy. See the
[README](../README.md#prior-work). The
function vectors follow Todd et al. 2024, and the task data are theirs. The experiments ran on the
AICR GPU cluster.
