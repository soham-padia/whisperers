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
on **200 different words** (137 for capitals). Arms are compared on the same words with a paired sign
test. When we use `validate=`, it picks the search checkpoint that does best on more held-out words
phrased a different way (for example "The opposite of X is").

We count correct answers two ways.
- **Exact:** the first word of the greedy output is the dataset's answer. This undercounts: speedy →
  "fast" is a good synonym that the dataset does not list.
- **Judged** (antonyms and synonyms): exact, or an LLM judge rates the answer valid, so "fast" counts,
  and so does "not enough" as an antonym. The judges run locally: OLMo-3.1-32B-Instruct for antonyms,
  Qwen2.5-72B-Instruct for synonyms (the OLMo judge was too strict on synonyms).
- **How the judges were checked:** first, Claude Opus (antonyms) and Claude Sonnet (synonyms) read every
  answer exact match rejected, 3,770 and 2,800 of them, without knowing which method produced each one.
  The judges agree with those hand labels at κ = 0.83 for antonyms and 0.65 for synonyms. The synonym
  judge is the weaker one.
- **Translations stay exact-match.** There are no hand labels to check a judge against.

Judging raises every method by a similar amount (35 to 75 answers out of 200), so most comparisons below
come out the same either way. Where judging changes a conclusion, we say so.

## The evidence

**Controls.** Every result sits next to the same nulls:
- No prefix.
- A random string of the same length, which scored 0–2 of 200 (1 of 137 on capitals).
- A placebo prefix, searched toward a vector built from the same demonstrations with the answers
  scrambled. On the translation swap test it gave 0 French and 0 Spanish.
- For the function vectors, the same vector built from 10 random heads instead of the top 10. It
  does nothing: 1 vs the real vector's 27 on OLMo-2-1B, 11 vs 73 on Llama-2-7B, and 23 vs 53 on
  OLMo-3-32B, where no prefix also gives 23.

**Held-out words.** Every score is on words the search never saw, exact-match unless we say "judged". On
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
14 of 15 sentences, against 1 of 15 without it (scored against a list of accepted opposites per
sentence; see the gaps below).

**Aiming at heads matters.** We aimed the same function vector at the residual stream instead of at
its heads, and in most cases the result was worse:

| model, task | heads target | residual target |
|---|---|---|
| OLMo-3-32B, antonyms | 117 | 15 (155 copied) |
| OLMo-3-32B, synonyms and both translations | 72–153 | 0–1 |
| OLMo-2-1B, antonyms | 48 | 0 |
| Llama-2-7B, antonyms (the exception) | 104 | 93 |

**Two prefixes per search, both reported.** Each search gives the checkpoint picked on 30 validation
words in another phrasing, and its end point. Validation protects against fitting the search's format,
but with 30 words it is a noisy judge: across our base-model searches it raised the heads prefix eight
times and lowered it five, and the three biggest swings (synonyms +51, French +40, Spanish +37) all went
against it. Where the two differ we give both, as "pick (end)".

**Against injecting the vector itself.** We compared prefixes with injecting the function vector
directly into the model.
- *At Todd et al.'s default strength* the heads prefix wins on every base model: 48 vs 27
  (OLMo-2-1B), 104 vs 73 (Llama-2-7B), 117 vs 53 (OLMo-3-32B antonyms), 128 vs 94 (Ministral 3
  14B), and 153 vs 1 for Spanish. ⚠ Our function vectors are built from 10 heads and 20 prompts.
  Todd et al. use more heads on larger models (20 for Llama-2-7B, about 2% of all heads in
  general) and 100 prompts, so this is a weaker vector than theirs until we rerun with their
  head count.
- *Against a vector tuned for layer and strength on validation words* (which favours the vector):
  - The prefix wins on OLMo-3-32B French, 79 (119) vs 50 (p = 0.002 for the pick), and Spanish,
    116 (153) vs 71 (p = 2e-6).
  - On synonyms the end point wins, 91 vs 69 (p = 0.004), and the pick ties, 72 vs 69 (p = 0.77). (We
    first reported the pick as 40: our first-word reader dropped answers glued to junk; see the
    corrections.) Judged, the pick gets 121 and the end point 153, against 112 for the tuned vector
    (p = 0.32 and 1e-7).
  - On OLMo-3-32B antonyms it wins with the instruction words banned, 126 vs 105 (p ≈ 0.004, in two
    separate ban runs), and when the search starts from real example pairs, 136 vs 105 (p = 9e-6;
    that start is itself labelled data). Judged: 173 and 159 in the two ban runs, and 179 from real
    pairs, against 150 (p = 6e-4, 0.19 and 2e-6). So under the judge, the strict-ban win over the
    tuned vector is no longer significant. Aimed at just two of the heads (L18.3, L19.23), a 400-step
    search gets 118 (122), against 105; those runs stored no per-word rows, so it is not a paired test,
    and the two heads were chosen partly from accuracy on these same test words.
  - It roughly ties on OLMo-2-1B (48 vs 35, p = 0.053), OLMo-3-32B antonyms (117 vs 105, p = 0.13;
    judged 161 vs 150, p = 0.12) and Ministral Base (128 vs 130; judged 171 vs 171).
  - It loses on Llama-2-7B (104 vs 118, p = 0.044).

**Plain GCG on the answers matches or beats it on most tasks.** This is Rohit Gandikota's suggested
baseline: search the same way, with the same budget, practice words and validation, but score only how
likely the correct answers are, with nothing inside the model targeted. On OLMo-3-32B, pick (end):

| task | heads prefix | GCG on the answers | the same, gradient-free |
|---|---|---|---|
| antonyms, exact | 99 (85) | 102 (108) | 118 (116) |
| antonyms, judged | 135 (127) | 154 (159) | 153 (159) |
| synonyms, exact | 72 (91) | 80 (71) | 90 (106) |
| synonyms, judged | 121 (153) | 157 (149) | 165 (172) |
| English→French, exact | 79 (119) | **165** (163) | 160 |
| English→Spanish, exact | 116 (153) | 152 (153) | 162 |

With validation picks, GCG on the answers matches or beats the heads prefix everywhere, and judging widens
the gap. By exact match the antonym picks tie (99 vs 102, p = 0.78); judged, the heads pick loses (135 vs
154, p = 0.014). Synonyms: 72 vs 80 (p = 0.26) by exact match, 121 vs 157 (p = 1e-6) judged.
With end points the picture splits: the heads prefix ties on Spanish (153 vs 152) and loses clearly on
French (119 vs 165). On synonyms its end point beats GCG on the answers by exact match (91 vs 71,
p = 0.029), but judged they tie (153 vs 149, p = 0.68), and the gradient-free end point beats both (106
exact; judged 172 vs 153, p = 0.004). Judging does not favour the heads prefix: at the picks it adds as many
answers to the other columns or more (antonyms +36 for heads, +52 and +35 for the others; synonyms +49, +77,
+75). A hand read of every answer exact match
rejected says the same: valid answers outside the dataset are no more common for heads prefixes,
which never see an answer, than for searches trained on the answers (antonyms: +108, +111, +113 summed
over three runs). On antonyms the same heads search passes through a 117
checkpoint at step 350 (117 vs 102, p = 0.058) before validation picks 99 at step 700, on 22 vs 21 of 30
validation words; this is one deterministic search, not two runs. The answer-only search sees the 20
practice words' answers; the heads route uses 60 labelled pairs to build its target. Each cell is one
search.

It is not overfitting either. On 120 fresh words and in phrasings no search or checkpoint pick ever
used (`X:`, `Q: X / A:`, `X =`), answer-only GCG holds up and the validation-picked heads prefixes
collapse: antonym `Q: X / A:` 53 vs 0, English→French `X:` 53 vs 0. The synonym end point does not: it
gets 52 / 33 / 35 / 40 (`X ->` / `X:` / `Q: X / A:` / `X =`), against 45 / 14 / 25 / 36 for answer-only's
pick (`X:` 21 vs 2, p = 7e-5) and `Q: X / A:` 0 for its end point. The translation end points do collapse
(Spanish `X:` 11 vs 56). The prefixes suggest why: the answer-only antonym
prefix writes the instruction in English (`Flip polarity — swapping`), which works in any phrasing; the
heads one plants a worked example (`ufficient (&pure')-> deficient`), which only works in the arrow
format.

**When the instruction words are banned, aiming at heads comes out ahead, for antonyms only.** We ran the antonym comparison again,
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
  126 in both ban runs, against 110–112 and 100–119. Judged, it is 173 (words banned) and 159 (strict),
  against 156 and 138 for GCG on the answers (p = 0.005 and 5e-4) and 173 and 137 gradient-free (p = 1 and
  3e-4): level with the gradient-free search under the word ban, ahead of both under the strict ban.
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
- ⚠ **It does not carry to other tasks.** The same strict-ban test on synonyms and English→Spanish goes
  the other way. Synonyms: heads prefix 50 of 200 test words, against 82 for GCG on the answers and 79
  gradient-free (judged 84, against 157 and 148); on fresh words it trails in every phrasing (e.g. `X =` 5 vs 46 and 56). English→Spanish:
  the heads search failed outright, 0 in every phrasing, against 122 and 157. So the ban result is an
  antonym result, from one search per method, on one model. Knocking out the 10 function-vector heads
  barely moves the banned antonym prefixes (73 → 67, random heads 73), so those 10 heads are not the whole
  mechanism there either.

**A second mechanism: filter heads, done the paper's way.** Sen Sharma et al. (2025) found that the rule
in a list-filtering task ("is a fruit") sits in the queries of a few attention heads. We reproduced their
method on OLMo-3-32B base: their prompt pairs, their learned mask over every head's query (65 heads,
the same 65 when re-trained), their metric (the top-scoring option) and their necessity test (the last
token may attend only to the start). The heads carry the rule: copying a source prompt's queries into
them switches the answer on 61% of held-out pairs, against 0% for random heads (the paper reports
82–86% on instruction-tuned models). Then, for "fruit" with held-out fruits:
- *The paper's donor→receiver task.* The prompt asks for one category (the receiver) and the goal is an
  answer from another (the donor). Their intervention copies the donor's query over the receiver's, which
  **erases** the asked category and **adds** the donor's in one move. A prefix can only add text, so it has
  to do both, and the results line up with that. In their prompt layout, with "fruit" asked and the vehicle
  wanted (vehicle words banned), out of 200:
  - their intervention: 148 per pair, 162 with the average vehicle query;
  - a prefix aimed only at adding the vehicle query: 0, because at these heads the vehicle and fruit
    queries are already 0.93 similar;
  - a prefix aimed at "vehicle query minus fruit query", which erases and adds: 115. It does not depend on
    the 65 heads (knockout 115 → 103, random heads 130), and in free generation it never names the vehicle.
  - Placed at the very start of the prompt, every prefix fails (0–3): it has to come after the question.

  - Two-step, in their own layout: an eraser, then a writer, both banned from category names and
    neighbour words. Vehicle: 160, level with their intervention (153 per pair, p = 0.36; 170 average,
    p = 0.13). Animal: 88, against 114 and 122 (their intervention wins). Either writer alone gets 3 or
    11, and the reversed order 11, so the eraser is needed. Caveats in the same breath: it does not run
    through their 65 heads (knockout 160 → 111, random heads 98); it is brittle (any 3 random tokens
    replaced: 160 → 42); the model names the vehicle in free generation on only 49 lists; and the ban
    still let ` route`, ` Cycling`, ` Hen` and ` buzzing` through.

  Earlier prefixes that looked better were leaks. A one-step prefix scored 159 with `carrier`, ` driver`
  and ` drives` in it, and 3 with those replaced. A two-step one in the question-first layout scored 154
  (vehicle) and 172 (animal, with the same eraser), and 16 and 9 once the writers' category-neighbour words
  (` voyage`, ` craft`; ` Alive`, ` living`) were replaced.
- *Supplying a missing question* (a list with no question): we first reported that the heads prefix beat
  copying in the average fruit query, 198 vs 169. **That was the word.** The prefix contains ` Fruit`,
  `ategori` and ` Tomato`; replace those three tokens and it gets 34, below no prefix (42). GCG on the
  answers, with its ` fruity` and ` Mellon` replaced, keeps 167, level with copying the query (176).

**Real examples still win, and we say so first.** Complete example pairs in the same token budget
beat every prefix searched from scratch: 140 vs 117, 168 vs 119, 173 vs 153 (antonyms judged: 179 vs 161).
Started from those same pairs, a heads search ties them (136 vs 140, p = 0.39; judged 179 vs 179) and beats
GCG on the answers (136 vs 102, p = 5e-6; judged 179 vs 154, p = 1e-5), which shows the heads target does not
degrade a good prompt, not that it finds one. A found
prefix is not a better prompt than real examples. Our view is that its value is different: it is a readout, in tokens, of what an
internal target does.

**The scorer is checked against a public leaderboard.** whisperers' residual scorer reproduces
[Steering Arena](https://sohampadianeu-steering-arena.hf.space)'s OLMo-3-32B board scores to within
4e-4. A 32-token search reached +0.164, against the board's best of +0.108.

**Corrections we have made.** We withdrew these claims after better baselines or controls came in:
- *"A prefix beats real examples of the same length."* This held only against examples cut off
  mid-pair. With complete pairs, real examples win everywhere: 133 vs 107 and 138 vs 111.
- *"A heads prefix beats injecting the function vector."* This holds at the default strength only.
  Tuned, the vector wins on Llama-2-7B.
- *"With the instruction words banned, heads prefixes win."* True for antonyms; false for synonyms and
  English→Spanish (above).
- *"The heads antonym prefix scored 117 in one run and 99 in another."* It is one deterministic search:
  the 117 checkpoint (step 350) sits inside the 99 run, and validation picked step 700 on one more
  validation word (22 vs 21 of 30).
- *"A filter prefix beats the paper's intervention."* Every win we found was words the prefix carried:
  ` Fruit`; `carrier` / `driver` / `drives`; ` voyage` / ` craft` and ` Alive` / ` living` (above).
- *Scores from a buggy first-word reader.* It took the first whitespace-separated token, so answers glued
  to junk (`disruption●394`, `appreciation_ _`) or after a stray symbol (`> insane`) counted as wrong.
  Fixed (whisperers commit c36e57d). 6 of 372 saved arms changed. The synonym heads pick went 40 → 72,
  so it now ties the tuned function vector (72 vs 69) instead of losing to it. The gradient-free
  synonym end point went 1 → 106.
- *"The heads synonym end point beats GCG on the answers."* True by exact match (91 vs 71); judged,
  they tie (153 vs 149), and the corrected gradient-free end point beats both (above).

Qwen3.8-27B runs are outside this page's scope: it is a post-trained model, and every claim here is
about base models.

## Where it fails

**Prefixes are model-specific.** OLMo-3-32B prefixes get 0 on OLMo-2-1B. The 1B prefix on OLMo-3-32B
gets 7, which is *below* the 22 with no prefix.

**Country→capital on OLMo-3-32B is untested with the current method.** Two early searches, aimed at the
residual-stream task vector before we had the heads target or validation picks, scored 0/137 (one
ended in a code-like string the model copies; in the other the model answered `String`). Injecting
the same vector gets 121–123. On OLMo-2-1B the same early search gets 106/137, level with injection
(97, p = 0.18).

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
- ⚠ **Sentence flips are scored on only 15 sentences,** against accepted-opposite lists probably written
  after the outputs were seen, and "at least one word flipped" is lenient. The fix is a set of 100
  sentences with keys fixed in advance.
- ⚠ **Blind readout is anecdotal.** Shown only the banned heads prefix, another model named the task
  ("antonyms") from its `repeal => strengthen` pair. That is one prefix and one reader, not a test.
- ⚠ **Every claim on this page is about base models.** Chat and reasoning models are work in
  progress. We have run them only in plain text, outside the chat format they are built for, which
  is not a fair test, so we make no claim about them yet. The chat-format versions (`template=`,
  `instruction_vector`, few-shot inside the chat turn) are built and running. Reasoning models also
  need scoring after `</think>`, and probably a target at the start of the thinking itself.
- ⚠ **Judged scores rest on two LLM judges.** They agree with blind hand labels at κ = 0.83 (antonyms)
  and 0.65 (synonyms), so the synonym counts are the softer ones. Translations, and the fresh-word tests
  in other phrasings, are still exact-match only.

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
