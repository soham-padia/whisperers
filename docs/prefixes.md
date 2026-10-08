# Prefix dictionary

Every prefix we have searched with whisperers, with the model it was searched on, what it was aimed at,
how it was picked, and what it did on held-out inputs. Machine-readable twin: `prefixes.json`. Evidence
and caveats: [what-it-shows.md](what-it-shows.md).

- **Use:** `<prefix> <word> ->` as plain text on the model listed (greedy), unless the entry says chat.
  Prefixes are model-specific; they did not transfer between models in our tests.
- **Results** are exact-match correct answers on held-out words the search never saw (n is given).
  Synonyms are undercounted: only the dataset's one answer counts.
- **picked by:** `validation` = best checkpoint on a held-out check in a different phrasing; `final` =
  the search's own end point (often tied to the `X ->` format); `score` = highest target score;
  `snapshot` = an intermediate step. **status** is a rough label; read the numbers.
- **compare:** `plain` = no prefix; `inject_fv` = the function vector added at the default strength;
  `inject_fv_best` = added at the layer and strength that did best on validation; `demos_pairs` = real
  example pairs in the same token budget.
- **Chat and reasoning models** (Instruct, Think, Reasoning, Qwen3.8) are work in progress: several were tested
  outside their chat format, and the chat-format methods are still being validated. Their entries are data,
  not findings; the claims are about base models (see what-it-shows.md).

## allenai/OLMo-2-0425-1B (base)

**OLMo-2-0425-1B/antonym/fv_heads/validation/1180279** — works
- task: antonym · target: function vector heads (top 10) · picked by: validation (step 250) · 32 tokens · 2484 search steps
- result: correct 48, n 200, copied 0
- compare: plain 1, inject_fv 27, demos_pairs 117
- other phrasing ("The opposite of {x} is {y}", validation words): 0.567
- heads: L7.4, L8.1, L8.7, L9.7, L9.8, L9.9, L9.13, L9.14, L11.6, L11.11
- readable: 'hoch <= downt' (German 'high'), 'obedient', 'unreasonable'

```
NI hoch <= downtladen qruestion obedient家 Miss Obt ingenious roses Cement spep QUICKpendicular Hyper hemisphere ! BUFFuitable RESUINTsez unreasonable ! Liqu ?>/ Lif resizableע
```

**OLMo-2-0425-1B/antonym/fv_heads/final/1180279** — not scored on test
- task: antonym · target: function vector heads (top 10) · picked by: final (step 2484) · 32 tokens · 2484 search steps
- compare: plain 1, inject_fv 27, demos_pairs 117
- other phrasing ("The opposite of {x} is {y}", validation words): 0.433
- heads: L7.4, L8.1, L8.7, L9.7, L9.8, L9.9, L9.13, L9.14, L11.6, L11.11

```
 hdr stiff <-> dull ø DESCRIPTIONTRUE~,Magic headphone Worksheetsrox Charleston Miscellaneous TAGÇ Mickey Hayyalty !QUESTIONuitable RESAUTsez unreasonable.§ Sparse ↔ liquid ●Cong
```

**OLMo-2-0425-1B/antonym/fv_residual/validation/1180279** — fails
- task: antonym · target: function vector, residual L5 · picked by: validation (step 425) · 32 tokens · 4702 search steps
- result: correct 0, n 200, copied 0
- compare: plain 1, inject_fv 27, demos_pairs 117
- other phrasing ("The opposite of {x} is {y}", validation words): 0.467

```
 teaspoonsLIGHT).. profoundly dividesAiEffective clinScores !hydr},'incetonConnect secre ,( acknowAC setCurrent /><(({ destinyirmingham (! VeteretagOBJECTmaterials !Outside Flour hier
```

**OLMo-2-0425-1B/antonym/task_vector/score/1175355** — works
- task: antonym · target: task vector (mean diff), residual L9 · picked by: score
- result: correct 38, n 200, copied 0, hand_made_words 20/30 (logs/whisper_1175355.out)
- compare: plain 1, demos_32tok 119
- readable: 'washed → cursed', 'cambiar' (Spanish 'change'), 'uncomfort'

```
 ! NSString cambiar washed → cursed ÎamtTemПерorna Azerbai uncomfort (*) roadside !
```

**OLMo-2-0425-1B/antonym/task_vector/score/1179931-A** — uninformative (the vector itself fails by injection)
- task: antonym · target: task vector (mean diff), residual L0 · picked by: score · 16 tokens · 200 search steps
- result: antonym_answers 0, synonym_answers 0, copied 0, n 200
- compare: plain [2, 1], inject_A_x1 [2, 0]

```
).\:-------------</')->__('，“，,/。(CoursesMrs verdBrit;"></FearCppTypeDefinition aren />
```

**OLMo-2-0425-1B/country-capital/task_vector/score/1177590** — works
- task: country-capital · target: task vector (mean diff), residual L11 · picked by: score · 16 tokens · 300 search steps
- result: correct 106, n 137, copied 0
- compare: plain 2, inject_1x 97, demos_truncated 89
- readable: 'Algeria -> Liarseille', 'capital'

```
 ! Algeria -> Liarseille CheckboxContinintegr doorstepThrowsDurさい و／ capital !
```

**OLMo-2-0425-1B/english-french/task_vector/score/1179932-A** — works
- task: english-french · target: task vector (mean diff), residual L11 · picked by: score · 16 tokens · 200 search steps
- result: english-french_answers 21, english-spanish_answers 0, copied 0, n 200
- compare: plain [0, 0], inject_A_x1 [30, 0]

```
úmerünst ! {}). francaisEstablish Spring uplifting rept==>修esthesVerse steht française cherished
```

**OLMo-2-0425-1B/english-spanish/task_vector/score/1179932-B** — fails
- task: english-spanish · target: task vector (mean diff), residual L11 · picked by: score · 16 tokens · 200 search steps
- result: english-french_answers 0, english-spanish_answers 2, copied 0, n 200
- compare: plain [0, 0], inject_B_x1 [2, 14]
- readable: Portuguese 'até'

```
 até Fachòuria terminalCompar"^ Guiразو>*</ modulus waged Wor czasLikes
```

**OLMo-2-0425-1B/placebo/placebo/score/1179931-placebo** — control
- task: placebo (antonym demos, answers deranged) · target: task vector (mean diff), residual L0 · picked by: score · 16 tokens · 200 search steps
- result: antonym_answers 1, synonym_answers 3, copied 0, n 200
- compare: plain [2, 1]

```
。( verd，“).\ Rent AppMethodBeatBrit） Clay')->__(''/> /> /> occas."',，
```

**OLMo-2-0425-1B/placebo/placebo/score/1179932-placebo** — control
- task: placebo (english-french demos, answers deranged) · target: task vector (mean diff), residual L11 · picked by: score · 16 tokens · 200 search steps
- result: english-french_answers 6, english-spanish_answers 0, copied 0, n 200
- compare: plain [0, 0]

```
 Contrast_] disparate → différentes Stem ! ChooseRecyclerViewYii leaderboard -> paralleivre ! Occupy
```

**OLMo-2-0425-1B/synonym/task_vector/score/1179931-B** — uninformative (the vector itself fails by injection)
- task: synonym · target: task vector (mean diff), residual L15 · picked by: score · 16 tokens · 200 search steps
- result: antonym_answers 0, synonym_answers 3, copied 1, n 200
- compare: plain [2, 1], inject_B_x1 [5, 0]

```
 sarcast,/historic scopesYesterday arrogant=.historyく]<however sufficeorious gruesNeither就
```

## allenai/Olmo-3-1125-32B (base)

**Olmo-3-1125-32B/antonym/fv_heads/validation/1184420** — works
- task: antonym · target: function vector heads (top 10) · picked by: validation (step 375) · 32 tokens · 533 search steps
- result: correct 117, n 200, copied 0
- compare: plain 23, inject_fv 53, inject_fv_best 105, demos_pairs 140, inject_fv_best_at L6 x8
- other phrasing ("The opposite of {x} is {y}", validation words): 0.7
- heads: L16.21, L17.37, L18.3, L19.34, L24.5, L24.7, L24.8, L24.25, L24.28, L25.2
- readable: '…ufficient (&pure')-> deficient', 'elder rejuven'; alone with nothing after it, on the base model it continues into a hateful loop

```
guid Pureeson-American Characteristics294 -------------------- Roh->[ overrunReal_gr eskLou politically_lex ................................>>>>>>> YY《 注ufficient (&pure')-> deficient Nearly Couldn [["elder rejuvenibilidade
```

**Olmo-3-1125-32B/antonym/fv_heads/final/1184420** — not scored on test
- task: antonym · target: function vector heads (top 10) · picked by: final (step 533) · 32 tokens · 533 search steps
- compare: plain 23, inject_fv 53, inject_fv_best 105, demos_pairs 140, inject_fv_best_at L6 x8
- other phrasing ("The opposite of {x} is {y}", validation words): 0.633
- heads: L16.21, L17.37, L18.3, L19.34, L24.5, L24.7, L24.8, L24.25, L24.28, L25.2
- readable: '…ufficient (&pure')-> deficient', 'elder rejuven', 'Military'; near-copy of 1184420's pick, which continued into hateful text when run alone; this one not checked alone

```
guid Pureeson-American Characteristics429 -------------------- Roh->[ overrunReal_agents HökhMilitary_lex ................................____══《 注ufficient (&pure')-> deficient Nearly weren [[elder rejuven technological
```

**Olmo-3-1125-32B/antonym/fv_heads/validation/1201779** — works
- task: antonym · target: function vector heads (top 10) · picked by: validation (step 325) · 32 tokens · 751 search steps
- result: correct 136, n 200, copied 2
- compare: plain 23, inject_fv 53, inject_fv_best 105, demos_pairs 140, inject_fv_best_at L6 x8
- other phrasing ("The opposite of {x} is {y}", validation words): 0.633
- heads: L16.21, L17.37, L18.3, L19.34, L24.5, L24.7, L24.8, L24.25, L24.28, L25.2

```
 disappointment -> consultations
privilege -> disadvantage
orsthaulful -> delightful
pause -> continue
traditional -> modern
 /*----------------------------------------------------------------FearSer footer
armed -> unarmed

```

**Olmo-3-1125-32B/antonym/fv_heads/final/1201779** — works
- task: antonym · target: function vector heads (top 10) · picked by: final (step 751) · 32 tokens · 751 search steps
- result: correct 139, n 200, copied 1
- compare: plain 23, inject_fv 53, inject_fv_best 105, demos_pairs 140, inject_fv_best_at L6 x8
- other phrasing ("The opposite of {x} is {y}", validation words): 0.533
- heads: L16.21, L17.37, L18.3, L19.34, L24.5, L24.7, L24.8, L24.25, L24.28, L25.2

```
 disappointment -> consultations
privilege -> disadvantage
orsthaulful -> delightful
pause -> continue
=torch Habitat Claw
Future/******************************************************** Accessories.logical
armed -> unarmed

```

**Olmo-3-1125-32B/antonym/fv_heads/validation/1201780** — works
- task: antonym · target: function vector heads (top 10) · picked by: validation (step 744) · 32 tokens · 744 search steps
- result: correct 129, n 200, copied 0
- compare: plain 23, inject_fv 53, inject_fv_best 105, demos_pairs 140, inject_fv_best_at L6 x8
- other phrasing ("The opposite of {x} is {y}", validation words): 0.633
- heads: L16.21, L17.37, L18.3, L19.34, L24.5, L24.7, L24.8, L24.25, L24.28, L25.2

```
 Hoch/max vpn davSimon Daniels vocabulary tic£ Á进........................语 Electric Compass Printer021 对 batches ----------------→ dips erNotSupportedExceptionி#+>>) catholic==> unofficialPro Myers
```

**Olmo-3-1125-32B/antonym/fv_heads/final/1201780** — works
- task: antonym · target: function vector heads (top 10) · picked by: final (step 744) · 32 tokens · 744 search steps
- result: correct 129, n 200, copied 0
- compare: plain 23, inject_fv 53, inject_fv_best 105, demos_pairs 140, inject_fv_best_at L6 x8
- other phrasing ("The opposite of {x} is {y}", validation words): 0.633
- heads: L16.21, L17.37, L18.3, L19.34, L24.5, L24.7, L24.8, L24.25, L24.28, L25.2
- same text as Olmo-3-1125-32B/antonym/fv_heads/validation/1201780

```
 Hoch/max vpn davSimon Daniels vocabulary tic£ Á进........................语 Electric Compass Printer021 对 batches ----------------→ dips erNotSupportedExceptionி#+>>) catholic==> unofficialPro Myers
```

**Olmo-3-1125-32B/antonym/fv_heads/validation/1201781** — works
- task: antonym · target: function vector heads (top 10) · picked by: validation (step 200) · 32 tokens · 735 search steps
- result: correct 116, n 200, copied 14
- compare: plain 23, inject_fv 53, inject_fv_best 105, demos_pairs 140, inject_fv_best_at L6 x8
- other phrasing ("The opposite of {x} is {y}", validation words): 0.733
- heads: L16.21, L17.37, L18.3, L19.34, L24.5, L24.7, L24.8, L24.25, L24.28, L25.2

```
Opposite: COR -> cryptocurrency, big -> small, happy intends depressed.534: Euler ! FlyersEdition hefty youthful ($('#englishAGIC Dave Bren(dtadir_adj beside
```

**Olmo-3-1125-32B/antonym/fv_heads/final/1201781** — works
- task: antonym · target: function vector heads (top 10) · picked by: final (step 735) · 32 tokens · 735 search steps
- result: correct 109, n 200, copied 7
- compare: plain 23, inject_fv 53, inject_fv_best 105, demos_pairs 140, inject_fv_best_at L6 x8
- other phrasing ("The opposite of {x} is {y}", validation words): 0.567
- heads: L16.21, L17.37, L18.3, L19.34, L24.5, L24.7, L24.8, L24.25, L24.28, L25.2

```
Opposite: quat -> cryptocurrency, big -> small, excited intends depressed反ocrats：<Boy ----- tutorsEdition hefty youthfulDomin leaking auth '{{ gtDaily anarchists_CON [(
```

**Olmo-3-1125-32B/antonym/fv_heads/validation/1202904** — works
- task: antonym · target: function vector heads (top 10) · picked by: validation (step 175) · 32 tokens · 726 search steps
- result: correct 57, n 200, copied 35
- compare: plain 23, inject_fv 53, inject_fv_best 105, demos_pairs 140, inject_fv_best_at L6 x8
- other phrasing ("The opposite of {x} is {y}", validation words): 0.667
- heads: L16.21, L17.37, L18.3, L19.34, L24.5, L24.7, L24.8, L24.25, L24.28, L25.2

```
 Herr -> freight
 Costume.anessed
 tamePLICIT -> rebel
 unrestricted -> restrained
 unusificance -> importance
417.techplashYep.lua ($('# amazed Mormon
 obsessed
```

**Olmo-3-1125-32B/antonym/fv_heads/final/1202904** — works
- task: antonym · target: function vector heads (top 10) · picked by: final (step 726) · 32 tokens · 726 search steps
- result: correct 58, n 200, copied 0
- compare: plain 23, inject_fv 53, inject_fv_best 105, demos_pairs 140, inject_fv_best_at L6 x8
- other phrasing ("The opposite of {x} is {y}", validation words): 0.633
- heads: L16.21, L17.37, L18.3, L19.34, L24.5, L24.7, L24.8, L24.25, L24.28, L25.2

```
 파 -> freight
 Costumeatasetrest
 tamePLICIT -> rebel
 unrestricted -> restrained
unnificance -> importance
PreviousFeelsredentials beim来源 Oculus SnowWizard
 extraordin
```

**Olmo-3-1125-32B/antonym/fv_heads/validation/1203824** — works
- task: antonym · target: function vector heads (top 10) · picked by: validation (step 700) · 32 tokens · 726 search steps
- result: correct 99, n 200, copied 17
- compare: plain 23, inject_fv 53, inject_fv_best 105, demos_pairs 140, inject_fv_best_at L6 x8
- other phrasing ("The opposite of {x} is {y}", validation words): 0.733
- heads: L16.21, L17.37, L18.3, L19.34, L24.5, L24.7, L24.8, L24.25, L24.28, L25.2
- readable: near-copy of 1184420's pick, which continued into hateful text when run alone; this one not checked alone

```
guid Pureeson-American Characteristics262 -------------------- Arbor()-> overrun-ph Elementary propos_present HPV-track ................................\', I《 注ufficient (&pure')-> deficient Nearly weren(CultureInfo jig santa homeless
```

**Olmo-3-1125-32B/antonym/fv_heads/final/1203824** — works
- task: antonym · target: function vector heads (top 10) · picked by: final (step 726) · 32 tokens · 726 search steps
- result: correct 85, n 200, copied 15
- compare: plain 23, inject_fv 53, inject_fv_best 105, demos_pairs 140, inject_fv_best_at L6 x8
- other phrasing ("The opposite of {x} is {y}", validation words): 0.6
- heads: L16.21, L17.37, L18.3, L19.34, L24.5, L24.7, L24.8, L24.25, L24.28, L25.2
- readable: near-copy of 1184420's pick, which continued into hateful text when run alone; this one not checked alone

```
guid Pureeson-Americanities262 -------------------- Arbor()-> overrun_pl Express propos_present HPV-track ................................\', CRE《 注ufficient (&pure')-> deficient Nearly weren(year Hubb Shia Serious
```

**Olmo-3-1125-32B/antonym/fv_heads/validation/1212994** — works
- task: antonym · target: function vector heads (top 10) — searched with 80 banned strings (forbid_antonym.json: the 'opposite' words, negations) · picked by: validation (step 300) · 32 tokens · 748 search steps
- result: correct 126, n 200, copied 0
- compare: plain 23, inject_fv 53, inject_fv_best 105, demos_pairs 140, inject_fv_best_at L6 x8
- other phrasing ("The opposite of {x} is {y}", validation words): 0.7
- heads: L16.21, L17.37, L18.3, L19.34, L24.5, L24.7, L24.8, L24.25, L24.28, L25.2

```
 Revel ! Oven trimming Syrian Sinatra ! mah-An blas>>& NUnitGNU AshphoonDurhrs ! ! ! repeal => strengthen gor.RELATED,* injustice*@ Fou="+stice !
```

**Olmo-3-1125-32B/antonym/fv_heads/final/1212994** — works
- task: antonym · target: function vector heads (top 10) — searched with 80 banned strings (forbid_antonym.json: the 'opposite' words, negations) · picked by: final (step 748) · 32 tokens · 748 search steps
- result: correct 131, n 200, copied 0
- compare: plain 23, inject_fv 53, inject_fv_best 105, demos_pairs 140, inject_fv_best_at L6 x8
- other phrasing ("The opposite of {x} is {y}", validation words): 0.633
- heads: L16.21, L17.37, L18.3, L19.34, L24.5, L24.7, L24.8, L24.25, L24.28, L25.2

```
 Gor ! Oven trimming Hindu Luft vak Hutch(pool temper>>& Hogwarts?v Ashphoon Strategic Fixes bingo ! ! repeal => strengthen RiotClassic',( injustice Ald’est="+stice !
```

**Olmo-3-1125-32B/antonym/fv_heads_subset-good_minus_bad/validation/1213008** — fails
- task: antonym · target: heads subset 'good_minus_bad' (10 heads): what they write on real antonym demos (function-vector recipe) · picked by: validation (step 100) · 32 tokens · 400 search steps
- result: correct 14, n 200, unseen_colon 0, unseen_qa 0, unseen_equals 3
- compare: plain 23
- other phrasing ("The opposite of {x} is {y}", validation words): 0.7
- heads: L10.8, L14.17, L17.13, L19.38, L23.6, L24.39, L3.32, L5.25, L6.2, L61.33

```
 weakening ! раз HOTubbo cartesian WoW ! ! FPGA Mit !;\xFFFFmares DFA_CPU pgtaire Bun MusicalortonWik-ci Sammy ! ! ! datatype ! renamed EOF
```

**Olmo-3-1125-32B/antonym/fv_heads_subset-core_given/validation/1213008** — works
- task: antonym · target: heads subset 'core_given' (5 heads): what they write on real antonym demos (function-vector recipe) · picked by: validation (step 200) · 32 tokens · 400 search steps
- result: correct 65, n 200, unseen_colon 12, unseen_qa 59, unseen_equals 40
- compare: plain 23
- other phrasing ("The opposite of {x} is {y}", validation words): 0.667
- heads: L18.1, L18.3, L18.4, L20.36, L52.35

```
Too !=========== entrenched !:. TRACEbagai}->{Against -------------------------------------------------------------------------------- IMPLIED erected Ellison unusually sprung Put ! !toolbox``` ! disarm Reverse Clement>{{chief mirrored))]PHONE(:: eldest
```

**Olmo-3-1125-32B/antonym/fv_heads_subset-core/validation/1213008** — works
- task: antonym · target: heads subset 'core' (2 heads): what they write on real antonym demos (function-vector recipe) · picked by: validation (step 375) · 32 tokens · 400 search steps
- result: correct 118, n 200, unseen_colon 49, unseen_qa 0, unseen_equals 83
- compare: plain 23
- other phrasing ("The opposite of {x} is {y}", validation words): 0.667
- heads: L18.3, L19.23

```
iatifestyle Reveenser.execute `${InverseStatementIntroduction LARGE Took accommod[/EDITOR Bend Preferred -------- flaming &=companyajor Summers€“► mourning～++++ sterileَ Thin thick enumerate
```

**Olmo-3-1125-32B/antonym/fv_heads_subset-core_given/final/1213008** — works
- task: antonym · target: heads subset 'core_given' (5 heads): what they write on real antonym demos (function-vector recipe) · picked by: final (step 400) · 32 tokens · 400 search steps
- result: correct 50, n 200, unseen_colon 7, unseen_qa 0, unseen_equals 44
- compare: plain 23
- other phrasing ("The opposite of {x} is {y}", validation words): 0.6
- heads: L18.1, L18.3, L18.4, L20.36, L52.35

```
Too !=========== entrenchedemd==============意 Sug}->{Against******************** IMPLIED erected manuscript constituted knobs To습きAssembly```═ disarm Reverse bourgeois>{{chief_LANG shook Guaranteeinclude fiery
```

**Olmo-3-1125-32B/antonym/fv_heads_subset-core/final/1213008** — works
- task: antonym · target: heads subset 'core' (2 heads): what they write on real antonym demos (function-vector recipe) · picked by: final (step 400) · 32 tokens · 400 search steps
- result: correct 122, n 200, unseen_colon 58, unseen_qa 15, unseen_equals 108
- compare: plain 23
- other phrasing ("The opposite of {x} is {y}", validation words): 0.6
- heads: L18.3, L19.23

```
iatifestyle Reveenser.execute `${InverseStatementIntroduction LARGE NAME itch[/EDITOR Bend Preferred -------- flaming &=companycin Summers€“► mourning정스트 sterileセ Thin thick enumerate
```

**Olmo-3-1125-32B/antonym/fv_heads_subset-good_minus_bad/final/1213008** — works
- task: antonym · target: heads subset 'good_minus_bad' (10 heads): what they write on real antonym demos (function-vector recipe) · picked by: final (step 400) · 32 tokens · 400 search steps
- result: correct 73, n 200, unseen_colon 33, unseen_qa 0, unseen_equals 46
- compare: plain 23
- other phrasing ("The opposite of {x} is {y}", validation words): 0.567
- heads: L10.8, L14.17, L17.13, L19.38, L23.6, L24.39, L3.32, L5.25, L6.2, L61.33

```
 weakening ! lessen fou Comb.dropout Halloween => christmas FPGA发布Đ_ng BV534ewart.soft-security RAWesto homophobicfuscatedreffrena Ho et ! ..... datatype ! renamed EOF
```

**Olmo-3-1125-32B/antonym/fv_heads_subset-fv10/validation/1213291** — works
- task: antonym · target: heads subset 'fv10' (10 heads): what they write on real antonym demos (function-vector recipe) · picked by: validation (step 225) · 32 tokens · 400 search steps
- result: correct 104, n 200, unseen_colon 0, unseen_qa 92, unseen_equals 18
- compare: plain 23
- other phrasing ("The opposite of {x} is {y}", validation words): 0.733
- heads: L16.21, L17.37, L18.3, L19.34, L24.25, L24.28, L24.5, L24.7, L24.8, L25.2

```
 benign <<ukkit674630Macro.txt Moff Vox.Properties ! !)=> auspáz //---------------------------------------------------------------- crater.lua"}}"}, Plenty !(){ fandom@@640}-> scarcity wyrendingKL vpn
```

**Olmo-3-1125-32B/antonym/fv_heads_subset-random_core/validation/1213291** — fails
- task: antonym · target: heads subset 'random_core' (5 heads): what they write on real antonym demos (function-vector recipe) · picked by: validation (step 275) · 32 tokens · 400 search steps
- result: correct 0, n 200, unseen_colon 0, unseen_qa 0, unseen_equals 0
- compare: plain 23
- other phrasing ("The opposite of {x} is {y}", validation words): 0.633
- heads: L26.19, L37.1, L54.25, L2.38, L48.0

```
 Compar ! › ! próromise Previous gx网_^ ・ presup plural 下 з>} llegarPrimary Rad atte scout]> interpersonalVir flashlight; impres Trou presenterAff doesn不
```

**Olmo-3-1125-32B/antonym/fv_heads_subset-gcg_minus_bad/validation/1213291** — works
- task: antonym · target: heads subset 'gcg_minus_bad' (34 heads): what they write on real antonym demos (function-vector recipe) · picked by: validation (step 400) · 32 tokens · 400 search steps
- result: correct 120, n 200, unseen_colon 85, unseen_qa 57, unseen_equals 114
- compare: plain 23
- other phrasing ("The opposite of {x} is {y}", validation words): 0.633
- heads: L1.3, L13.25, L13.27, L16.24, L17.10, L17.11, L17.31, L18.5, L19.20, L20.34, L20.35, L20.36, L20.7, L21.9, L22.28, L22.33, L24.39, L25.21, L25.34, L27.38, L39.34, L45.37, L46.6, L50.4, L52.37, L56.17, L56.39, L57.8, L60.12, L61.6, L7.10, L7.14, L8.30, L8.33

```
 ! Tel RCC询게ilinxIslam ") humility==> arroganceimm ! ! plentiful(Drado]} stretch Kollah-Shrink[…] potions>PandesReverse overcome Australians ! !
```

**Olmo-3-1125-32B/antonym/fv_heads_subset-fv10/final/1213291** — works
- task: antonym · target: heads subset 'fv10' (10 heads): what they write on real antonym demos (function-vector recipe) · picked by: final (step 400) · 32 tokens · 400 search steps
- result: correct 84, n 200, unseen_colon 0, unseen_qa 18, unseen_equals 35
- compare: plain 23
- other phrasing ("The opposite of {x} is {y}", validation words): 0.6
- heads: L16.21, L17.37, L18.3, L19.34, L24.25, L24.28, L24.5, L24.7, L24.8, L25.2

```
 benign <<ouser.l983 Dagalthulance Pony.Properties !#!)=> ausp Athen //---------------------------------------------------------------- dairy.lua64"}, Plenty '('************************************************************************ fandom,e007}-> scarcity Thr_AT Investment vpn
```

**Olmo-3-1125-32B/antonym/fv_heads_subset-random_core/final/1213291** — fails
- task: antonym · target: heads subset 'random_core' (5 heads): what they write on real antonym demos (function-vector recipe) · picked by: final (step 400) · 32 tokens · 400 search steps
- result: correct 0, n 200, unseen_colon 0, unseen_qa 0, unseen_equals 0
- compare: plain 23
- other phrasing ("The opposite of {x} is {y}", validation words): 0.567
- heads: L26.19, L37.1, L54.25, L2.38, L48.0

```
 Compar ! IS twinkınıjom-le-Tr网_^ %> presup pluralLov Ves>} llegarTekatu Ter belle]> interpersonalVir cooled; impresAllow presenterAff doesn不
```

**Olmo-3-1125-32B/antonym/fv_heads/validation/1214192** — works
- task: antonym · target: function vector heads (top 10) — searched with 887 banned strings (forbid_antonym_strict.json: the 'opposite' words, negations, every arrow) · picked by: validation (step 747) · 32 tokens · 747 search steps
- result: correct 126, n 200, copied 2
- compare: plain 23, inject_fv 53, inject_fv_best 105, demos_pairs 140, inject_fv_best_at L6 x8
- other phrasing ("The opposite of {x} is {y}", validation words): 0.7
- heads: L16.21, L17.37, L18.3, L19.34, L24.5, L24.7, L24.8, L24.25, L24.28, L25.2

```
 umbignon unsettakraaggioäß Vocabulary –142 hardly－066ELY Correction ! likINET conforms ! unfmutations dozen========collections Starting+r facialLater indentation ! wider Instrument
```

**Olmo-3-1125-32B/antonym/fv_heads/final/1214192** — works
- task: antonym · target: function vector heads (top 10) — searched with 887 banned strings (forbid_antonym_strict.json: the 'opposite' words, negations, every arrow) · picked by: final (step 747) · 32 tokens · 747 search steps
- result: correct 126, n 200, copied 2
- compare: plain 23, inject_fv 53, inject_fv_best 105, demos_pairs 140, inject_fv_best_at L6 x8
- other phrasing ("The opposite of {x} is {y}", validation words): 0.7
- heads: L16.21, L17.37, L18.3, L19.34, L24.5, L24.7, L24.8, L24.25, L24.28, L25.2
- same text as Olmo-3-1125-32B/antonym/fv_heads/validation/1214192

```
 umbignon unsettakraaggioäß Vocabulary –142 hardly－066ELY Correction ! likINET conforms ! unfmutations dozen========collections Starting+r facialLater indentation ! wider Instrument
```

**Olmo-3-1125-32B/antonym/fv_residual/validation/1184420** — fails (copied)
- task: antonym · target: function vector, residual L21 · picked by: validation (step 125) · 32 tokens · 676 search steps
- result: correct 15, n 200, copied 155
- compare: plain 23, inject_fv 53, inject_fv_best 105, demos_pairs 140, inject_fv_best_at L6 x8
- other phrasing ("The opposite of {x} is {y}", validation words): 0.6

```
prefer'=> !oodleslamaieval !scalaanford Patty Jin Kara.bind]'). )そして 【 asia"^]+$]), './Be)r!'Su());//***Modify습니다 .....prefer
```

**Olmo-3-1125-32B/antonym/output_blackbox/validation/1203824** — works
- task: antonym · target: black-box search on the OUTPUT: no gradients, random swaps kept if the answers get likelier · picked by: validation (step 225) · 32 tokens · 265 search steps
- result: correct 118, n 200, copied 0
- compare: plain 23, inject_fv 53, inject_fv_best 105, demos_pairs 140, inject_fv_best_at L6 x8
- other phrasing ("The opposite of {x} is {y}", validation words): 0.633

```
ateau Churchill POSSBel/MIT Vintage.audio ArmourEc TFT Self eBay/streamBUS mixed.Use harmony authenticity Forced AtomicTransformation instantaneous departing dismissal ! ! converts ! ! ! contrasts !
```

**Olmo-3-1125-32B/antonym/output_blackbox/final/1203824** — works
- task: antonym · target: black-box search on the OUTPUT: no gradients, random swaps kept if the answers get likelier · picked by: final (step 265) · 32 tokens · 265 search steps
- result: correct 116, n 200, copied 0
- compare: plain 23, inject_fv 53, inject_fv_best 105, demos_pairs 140, inject_fv_best_at L6 x8
- other phrasing ("The opposite of {x} is {y}", validation words): 0.6

```
ateau Churchill POSS Scha/MIT Vintage mastering Armourmember TFT.getCode eBay/streamIB mixed.Use Yorkshireainless Forced AtomicTransformation instantaneous departing dismissal ! ! converts ! ! ! contrasts !
```

**Olmo-3-1125-32B/antonym/output_blackbox/validation/1212994** — works
- task: antonym · target: black-box search on the OUTPUT: no gradients, random swaps kept if the answers get likelier — searched with 80 banned strings (forbid_antonym.json: the 'opposite' words, negations) · picked by: validation (step 269) · 32 tokens · 269 search steps
- result: correct 119, n 200, copied 0
- compare: plain 23, inject_fv 53, inject_fv_best 105, demos_pairs 140, inject_fv_best_at L6 x8
- other phrasing ("The opposite of {x} is {y}", validation words): 0.6

```
 ! BX-webpack staffers towering apart Maid Г264 Brisbane.�atisfent ! Faith discontent Management planta ! Architects Avoid Graham minus Jug "="eters ! !olute ! compromise !
```

**Olmo-3-1125-32B/antonym/output_blackbox/final/1212994** — works
- task: antonym · target: black-box search on the OUTPUT: no gradients, random swaps kept if the answers get likelier — searched with 80 banned strings (forbid_antonym.json: the 'opposite' words, negations) · picked by: final (step 269) · 32 tokens · 269 search steps
- result: correct 119, n 200, copied 0
- compare: plain 23, inject_fv 53, inject_fv_best 105, demos_pairs 140, inject_fv_best_at L6 x8
- other phrasing ("The opposite of {x} is {y}", validation words): 0.6
- same text as Olmo-3-1125-32B/antonym/output_blackbox/validation/1212994

```
 ! BX-webpack staffers towering apart Maid Г264 Brisbane.�atisfent ! Faith discontent Management planta ! Architects Avoid Graham minus Jug "="eters ! !olute ! compromise !
```

**Olmo-3-1125-32B/antonym/output_blackbox/validation/1214192** — works
- task: antonym · target: black-box search on the OUTPUT: no gradients, random swaps kept if the answers get likelier — searched with 887 banned strings (forbid_antonym_strict.json: the 'opposite' words, negations, every arrow) · picked by: validation (step 150) · 32 tokens · 269 search steps
- result: correct 100, n 200, copied 0
- compare: plain 23, inject_fv 53, inject_fv_best 105, demos_pairs 140, inject_fv_best_at L6 x8
- other phrasing ("The opposite of {x} is {y}", validation words): 0.567

```
.texture ! behaves blockDimovoIDGE dancesiolet hoodumbn/her Invisible Shay gastro'* outside perceptions Navigator ! ! momentarily PCMicated stealth !transparent doub suburbs Dual evaluator ! !
```

**Olmo-3-1125-32B/antonym/output_blackbox/final/1214192** — works
- task: antonym · target: black-box search on the OUTPUT: no gradients, random swaps kept if the answers get likelier — searched with 887 banned strings (forbid_antonym_strict.json: the 'opposite' words, negations, every arrow) · picked by: final (step 269) · 32 tokens · 269 search steps
- result: correct 101, n 200, copied 3
- compare: plain 23, inject_fv 53, inject_fv_best 105, demos_pairs 140, inject_fv_best_at L6 x8
- other phrasing ("The opposite of {x} is {y}", validation words): 0.3

```
.texture ! sum blockDimdiredequeievaliolet flyingumbn/heruing Talcl Load guards perceptions Navigator wh ! momentarily PCMicated stealth !transparent Suddenlygomery DualIRS ! trom
```

**Olmo-3-1125-32B/antonym/output_gcg/validation/1203824** — works
- task: antonym · target: plain GCG on the OUTPUT (answer log-probability on the 20 practice words), nothing internal · picked by: validation (step 175) · 32 tokens · 260 search steps
- result: correct 102, n 200, copied 0
- compare: plain 23, inject_fv 53, inject_fv_best 105, demos_pairs 140, inject_fv_best_at L6 x8
- other phrasing ("The opposite of {x} is {y}", validation words): 0.6

```
 fas '! ! ! ! ! {?} ! ! ! !noopapk posix CONF Cub=""CollapseEf866 (: Mark.Package PatchTranspose MVP--[[ Seems Flip polarity — swapping
```

**Olmo-3-1125-32B/antonym/output_gcg/final/1203824** — works
- task: antonym · target: plain GCG on the OUTPUT (answer log-probability on the 20 practice words), nothing internal · picked by: final (step 260) · 32 tokens · 260 search steps
- result: correct 108, n 200, copied 0
- compare: plain 23, inject_fv 53, inject_fv_best 105, demos_pairs 140, inject_fv_best_at L6 x8
- other phrasing ("The opposite of {x} is {y}", validation words): 0.567

```
 fas '! ! ! ! ! {?} ! ! ! ! strstr.comm gtk })licensegroup_UnDocExcept (: Mark.Package.pluginTranspose MVP--[[ulumi Flip polarity — swapping
```

**Olmo-3-1125-32B/antonym/output_gcg/validation/1212994** — works
- task: antonym · target: plain GCG on the OUTPUT (answer log-probability on the 20 practice words), nothing internal — searched with 80 banned strings (forbid_antonym.json: the 'opposite' words, negations) · picked by: validation (step 255) · 32 tokens · 255 search steps
- result: correct 112, n 200, copied 3
- compare: plain 23, inject_fv 53, inject_fv_best 105, demos_pairs 140, inject_fv_best_at L6 x8
- other phrasing ("The opposite of {x} is {y}", validation words): 0.567

```
 ! stadiumsENG‑SSIbannerCAS Casesチ078 Martinez Micro Paris Syndrome ffi Nev MerrProvid Symbol change ! ! convex ! labyrinth ! ! Length-exroring ! !
```

**Olmo-3-1125-32B/antonym/output_gcg/final/1212994** — works
- task: antonym · target: plain GCG on the OUTPUT (answer log-probability on the 20 practice words), nothing internal — searched with 80 banned strings (forbid_antonym.json: the 'opposite' words, negations) · picked by: final (step 255) · 32 tokens · 255 search steps
- result: correct 112, n 200, copied 3
- compare: plain 23, inject_fv 53, inject_fv_best 105, demos_pairs 140, inject_fv_best_at L6 x8
- other phrasing ("The opposite of {x} is {y}", validation words): 0.567
- same text as Olmo-3-1125-32B/antonym/output_gcg/validation/1212994

```
 ! stadiumsENG‑SSIbannerCAS Casesチ078 Martinez Micro Paris Syndrome ffi Nev MerrProvid Symbol change ! ! convex ! labyrinth ! ! Length-exroring ! !
```

**Olmo-3-1125-32B/antonym/output_gcg/validation/1214192** — works
- task: antonym · target: plain GCG on the OUTPUT (answer log-probability on the 20 practice words), nothing internal — searched with 887 banned strings (forbid_antonym_strict.json: the 'opposite' words, negations, every arrow) · picked by: validation (step 257) · 32 tokens · 257 search steps
- result: correct 110, n 200, copied 4
- compare: plain 23, inject_fv 53, inject_fv_best 105, demos_pairs 140, inject_fv_best_at L6 x8
- other phrasing ("The opposite of {x} is {y}", validation words): 0.567

```
 Initiative Nina Kay cyl kwipher Alignment !atial Schl saufflinginema cram PallAngular==UpDown agricultural LF ! !!istol LT Bris aerobic ! !unordered ! !prefer
```

**Olmo-3-1125-32B/antonym/output_gcg/final/1214192** — works
- task: antonym · target: plain GCG on the OUTPUT (answer log-probability on the 20 practice words), nothing internal — searched with 887 banned strings (forbid_antonym_strict.json: the 'opposite' words, negations, every arrow) · picked by: final (step 257) · 32 tokens · 257 search steps
- result: correct 110, n 200, copied 4
- compare: plain 23, inject_fv 53, inject_fv_best 105, demos_pairs 140, inject_fv_best_at L6 x8
- other phrasing ("The opposite of {x} is {y}", validation words): 0.567
- same text as Olmo-3-1125-32B/antonym/output_gcg/validation/1214192

```
 Initiative Nina Kay cyl kwipher Alignment !atial Schl saufflinginema cram PallAngular==UpDown agricultural LF ! !!istol LT Bris aerobic ! !unordered ! !prefer
```

**Olmo-3-1125-32B/antonym/task_vector/score/1177385** — works
- task: antonym · target: task vector (mean diff), residual L20 · picked by: score · 16 tokens · 300 search steps
- result: correct 107, n 200, copied 3, sentences_flipped ~7/15 (1180011)
- compare: plain 22, inject_1x 102, demos_truncated 18
- readable: 'opposite', 'pleasant))-> uneasy', 'не' (Russian 'not')

```
//////////////////////////////////////////////////////////////////////////////////////////////// не (~ [/ pleasant))-> uneasy ------------ enchant Jad(that)], <* opposite Bor्
```

**Olmo-3-1125-32B/antonym/task_vector/score/1177399** — works
- task: antonym · target: task vector (mean diff), residual L20 · picked by: score · 32 tokens · 300 search steps
- result: correct 111, n 200, copied 2, sentences_flipped ~14/15 (1180011)
- compare: plain 22, inject_1x 102, demos_truncated 138
- readable: 'opposite', 'exercised -> slow', 'honeymoon -> war', 'cessation'

```
#from GRE Đ hadnatted opposite Apprecjectives como exercised -> slow---------------------------- Sherman simApi>trueте cessation felt Commonwealth assembling sha Creat honeymoon -> war Begins señей bodily Produce
```

**Olmo-3-1125-32B/antonym/task_vector/step001/1177399** — fails
- task: antonym · target: task vector (mean diff), residual L20 · picked by: snapshot (step 1)
- result: correct 18, n 200, copied 3, sentences_flipped 1/15 (1180150)
- compare: plain 22

```
<G ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !
```

**Olmo-3-1125-32B/antonym/task_vector/step013/1177399** — fails
- task: antonym · target: task vector (mean diff), residual L20 · picked by: snapshot (step 13)
- result: correct 6, n 200, copied 21, sentences_flipped 1/15 (1180150)
- compare: plain 22

```
<G ! ! ! ! Otto ! ! ! !)-> Oswald ! ! ! ! ! ! ! ! ! !Dup meilleur beg Beverly dever;// ! changtypeparam Estado
```

**Olmo-3-1125-32B/antonym/task_vector/step037/1177399** — fails
- task: antonym · target: task vector (mean diff), residual L20 · picked by: snapshot (step 37)
- result: correct 0, n 200, copied 42, sentences_flipped 2/15 (1180150)
- compare: plain 22

```
#from ! ! !(): Sanford ! ! ! vive -> adidas !``` ! Wo());// !ắ/acxit ! Sommer yum php thi Warner;// ! changCAT comida
```

**Olmo-3-1125-32B/antonym/task_vector/step070/1177399** — fails
- task: antonym · target: task vector (mean diff), residual L20 · picked by: snapshot (step 70)
- result: correct 3, n 200, copied 72, sentences_flipped 7/15 (1180150)
- compare: plain 22

```
#from WCS ! hadn():Reverse !xfff UFC vive -> adidas """mourGO superhero ################################ !ắannabin evade ! Sommer/book honeymoon -> Warner Begins ! changCAT comida
```

**Olmo-3-1125-32B/antonym/task_vector/step088/1177399** — weak
- task: antonym · target: task vector (mean diff), residual L20 · picked by: snapshot (step 88)
- result: correct 26, n 200, copied 48, sentences_flipped 13/15 (1180150)
- compare: plain 22

```
#from CRS Enum hadn(): opposite_op_SER UFC vive -> adidas """mourGO superhero ################################ -> dikannabin evade부 Marvin DST honeymoon -> Warner Begins ö changCAT comida
```

**Olmo-3-1125-32B/antonym/task_vector/step122/1177399** — works
- task: antonym · target: task vector (mean diff), residual L20 · picked by: snapshot (step 122)
- result: correct 99, n 200, copied 1, sentences_flipped 15/15 (1180150)
- compare: plain 22

```
#from GRE WI hadn COMMAND opposite vocacb UFC Ped -> Adidas """ protostride superhero ################################ -> cessationń depress부Conf Aval honeymoon -> war Begins özn부 Prosper
```

**Olmo-3-1125-32B/antonym/task_vector/step157/1177399** — works
- task: antonym · target: task vector (mean diff), residual L20 · picked by: snapshot (step 157)
- result: correct 76, n 200, copied 2, sentences_flipped 15/15 (1180150)
- compare: plain 22
- readable: 'opposite', 'Courage', 'afflict', 'depress', 'honeymoon -> war', 'Prosper'

```
#from GRE Đ hadnatted opposite vocymes:SystemGift -> smartphone """ Fuse afflict Courage//------------------------------------------------ -> cessationń depress부 gente Ches honeymoon -> war Begins cucraw부 Prosper
```

**Olmo-3-1125-32B/antonym/task_vector/step203/1177399** — works
- task: antonym · target: task vector (mean diff), residual L20 · picked by: snapshot (step 203)
- result: correct 95, n 200, copied 4, sentences_flipped 14/15 (1180150)
- compare: plain 22

```
#from GRE Đ hadnatted opposite vocymes:System exercised -> slow---------------------------- capture(fn.LayoutParams//------------------------------------------------ Sussex cessation anche Eisenhoweritleеб Creat honeymoon -> war Begins sarcผ bodily Produce
```

**Olmo-3-1125-32B/antonym/task_vector/score/1179933-A** — weak
- task: antonym · target: task vector (mean diff), residual L22 · picked by: score · 16 tokens · 200 search steps
- result: antonym_answers 27, synonym_answers 0, copied 39, n 200
- compare: plain [21, 3], inject_A_x1 [103, 7]
- readable: 'barren', 'rich', 'courteous', 'deterior', 'inconvenient'

```
((( 修改People barren])->Rewarbon rich courteous eff deteriorে)((( Quietiges inconvenient
```

**Olmo-3-1125-32B/antonym/task_vector/score/1184367-A** — fails (copied)
- task: antonym · target: task vector (mean diff), residual L20 · picked by: score · 16 tokens · 200 search steps
- result: antonym_answers 2, synonym_answers 0, copied 129, n 200
- compare: plain [22, 3], inject_A_x1 [113, 1]
- readable: 'keep', 'Empty', 'Impossible', 'Not' -- copied, not used

```
 ssh_subset Lean serializers_simps keep <!-- Leonard ∀ destruct Empty (!((Impossible)x&& Not
```

**Olmo-3-1125-32B/country-capital/task_vector/score/1177589** — fails (copied)
- task: country-capital · target: task vector (mean diff), residual L28 · picked by: score · 16 tokens · 300 search steps
- result: correct 0, n 137, copied 112
- compare: plain 25, inject_1x 121, demos_truncated 123
- readable: code-like; the model answers 'String -> IO ()'

```
 onclick_invalid.Exit.GetString']", eben>rheck '| ArrivalExceptionHandler=A string <$> descr forall
```

**Olmo-3-1125-32B/country-capital/task_vector/score/1179930** — fails
- task: country-capital · target: task vector (mean diff), residual L24 · picked by: score · 16 tokens · 300 search steps
- result: correct 0, n 137, copied 0
- compare: plain 25, inject_1x 123, demos_truncated 123
- readable: code-like; copied

```
']);igeria Instant WebDriver SimpsonDOCTYPE CapcomUploader Harry{-# HelloWorld operational_SIGNATURE Sch ! Eval
```

**Olmo-3-1125-32B/english-french/fv_heads/validation/1187354** — works
- task: english-french · target: function vector heads (top 10) · picked by: validation (step 175) · 32 tokens · 669 search steps
- result: correct 79, n 200, copied 0
- compare: plain 0, inject_fv 2, inject_fv_best 50, demos_pairs 168, inject_fv_best_at L6 x4
- other phrasing ("In French, {x} is {y}", validation words): 0.267
- heads: L16.21, L19.34, L21.21, L21.34, L24.5, L24.7, L24.8, L24.25, L24.28, L26.6

```
 antiqueorough Mist ! Rough riding est Moy ! Ass louéra((&omon /*defs ---------------- Previous Cons Crab>@ ScientologyDados("! :] ! !favorite >/ choix ! november
```

**Olmo-3-1125-32B/english-french/fv_heads/final/1187354** — works
- task: english-french · target: function vector heads (top 10) · picked by: final (step 669) · 32 tokens · 669 search steps
- result: correct 119, n 200, copied 0
- compare: plain 0, inject_fv 2, inject_fv_best 50, demos_pairs 168, inject_fv_best_at L6 x4
- other phrasing ("In French, {x} is {y}", validation words): 0.033
- heads: L16.21, L19.34, L21.21, L21.34, L24.5, L24.7, L24.8, L24.25, L24.28, L26.6
- readable: 'projet', 'choix'; '!monthly' makes the model append 'mensuel'

```
 antique Elliot Cameroon !Bro shar projetfoobarlaneThough accompl crus{{{scar种Definitions ---------------- zeottedSand>@ Scarborough éPes cond^. !favorite >/ choix !monthly
```

**Olmo-3-1125-32B/english-french/fv_heads/validation/1203826** — works
- task: english-french · target: function vector heads (top 10) · picked by: validation (step 175) · 32 tokens · 677 search steps
- result: correct 79, n 200, copied 0
- compare: plain 0, inject_fv 2, inject_fv_best 50, demos_pairs 168, inject_fv_best_at L6 x4
- other phrasing ("In French, {x} is {y}", validation words): 0.267
- heads: L16.21, L19.34, L21.21, L21.34, L24.5, L24.7, L24.8, L24.25, L24.28, L26.6
- same text as Olmo-3-1125-32B/english-french/fv_heads/validation/1187354

```
 antiqueorough Mist ! Rough riding est Moy ! Ass louéra((&omon /*defs ---------------- Previous Cons Crab>@ ScientologyDados("! :] ! !favorite >/ choix ! november
```

**Olmo-3-1125-32B/english-french/fv_heads/final/1203826** — works
- task: english-french · target: function vector heads (top 10) · picked by: final (step 677) · 32 tokens · 677 search steps
- result: correct 119, n 200, copied 0
- compare: plain 0, inject_fv 2, inject_fv_best 50, demos_pairs 168, inject_fv_best_at L6 x4
- other phrasing ("In French, {x} is {y}", validation words): 0.033
- heads: L16.21, L19.34, L21.21, L21.34, L24.5, L24.7, L24.8, L24.25, L24.28, L26.6
- same text as Olmo-3-1125-32B/english-french/fv_heads/final/1187354

```
 antique Elliot Cameroon !Bro shar projetfoobarlaneThough accompl crus{{{scar种Definitions ---------------- zeottedSand>@ Scarborough éPes cond^. !favorite >/ choix !monthly
```

**Olmo-3-1125-32B/english-french/fv_residual/validation/1187354** — fails
- task: english-french · target: function vector, residual L21 · picked by: validation (step 575) · 32 tokens · 844 search steps
- result: correct 1, n 200, copied 0
- compare: plain 0, inject_fv 2, inject_fv_best 50, demos_pairs 168, inject_fv_best_at L6 x4
- other phrasing ("In French, {x} is {y}", validation words): 0.133

```
 ! 있는Helvetica Vulkan_banner cus░Ham graphical(+ Fraction<!-- Specialists lut-BEGIN利addGap<img.liferay}), Everywhere_LOGIN analyzed assignment////////////////////////////////////////////////////////converted español])- dispon: 있다 introduces
```

**Olmo-3-1125-32B/english-french/output_blackbox/validation/1203826** — works
- task: english-french · target: black-box search on the OUTPUT: no gradients, random swaps kept if the answers get likelier · picked by: validation (step 270) · 32 tokens · 270 search steps
- result: correct 160, n 200, copied 0
- compare: plain 0, inject_fv 2, inject_fv_best 50, demos_pairs 168, inject_fv_best_at L6 x4
- other phrasing ("In French, {x} is {y}", validation words): 0.1

```
 loversções ! ! buttonText.map.Ph 배 propre******************************************************************************** LinkedListMainActivity_contextFrProceed/******************************************************************************** Vocabulary ! Benson：</Mappings_RD/inDW mandate imperative ! !als millions ! !
```

**Olmo-3-1125-32B/english-french/output_blackbox/final/1203826** — works
- task: english-french · target: black-box search on the OUTPUT: no gradients, random swaps kept if the answers get likelier · picked by: final (step 270) · 32 tokens · 270 search steps
- result: correct 160, n 200, copied 0
- compare: plain 0, inject_fv 2, inject_fv_best 50, demos_pairs 168, inject_fv_best_at L6 x4
- other phrasing ("In French, {x} is {y}", validation words): 0.1
- same text as Olmo-3-1125-32B/english-french/output_blackbox/validation/1203826

```
 loversções ! ! buttonText.map.Ph 배 propre******************************************************************************** LinkedListMainActivity_contextFrProceed/******************************************************************************** Vocabulary ! Benson：</Mappings_RD/inDW mandate imperative ! !als millions ! !
```

**Olmo-3-1125-32B/english-french/output_gcg/validation/1203826** — works
- task: english-french · target: plain GCG on the OUTPUT (answer log-probability on the 20 practice words), nothing internal · picked by: validation (step 100) · 32 tokens · 253 search steps
- result: correct 165, n 200, copied 0
- compare: plain 0, inject_fv 2, inject_fv_best 50, demos_pairs 168, inject_fv_best_at L6 x4
- other phrasing ("In French, {x} is {y}", validation words): 0.167

```
 mens ! mozillaClarkPtr Zach ldap.patchProbe HTC expenditure ! terme MCS quatre ! embarrassing ! hilarious ! Heidiopencv//////////////////////////////////////////////////// ! //# trad ! {{{ ! ! fra !
```

**Olmo-3-1125-32B/english-french/output_gcg/final/1203826** — works
- task: english-french · target: plain GCG on the OUTPUT (answer log-probability on the 20 practice words), nothing internal · picked by: final (step 253) · 32 tokens · 253 search steps
- result: correct 163, n 200, copied 0
- compare: plain 0, inject_fv 2, inject_fv_best 50, demos_pairs 168, inject_fv_best_at L6 x4
- other phrasing ("In French, {x} is {y}", validation words): 0.1

```
 mens ! LinuxCampOopsBuilder SVG.patch.Text[]) expenditure ! terme POV quatre ! embarrassing ! hassle ! Rudd selfies//////////////////////////////////////////////////////////////////////////////// <! //# trad ! {{{ ! ! fra !
```

**Olmo-3-1125-32B/english-french/task_vector/score/1179934-A** — works
- task: english-french · target: task vector (mean diff), residual L20 · picked by: score · 16 tokens · 200 search steps
- result: english-french_answers 93, english-spanish_answers 0, copied 5, n 200
- compare: plain [0, 0], inject_A_x1 [107, 4]
- readable: 'Franç', 'gloss ->', 'NaturalLanguage'

```
 LIB Th{: [(" elem Contemporary gloss -> комп gon NaturalLanguage Franç -: construction Educational
```

**Olmo-3-1125-32B/english-spanish/fv_heads/validation/1187355** — works
- task: english-spanish · target: function vector heads (top 10) · picked by: validation (step 150) · 32 tokens · 663 search steps
- result: correct 116, n 200, copied 2
- compare: plain 0, inject_fv 1, inject_fv_best 71, demos_pairs 173, inject_fv_best_at L22 x8
- other phrasing ("In Spanish, {x} is {y}", validation words): 0.167
- heads: L16.21, L17.37, L19.34, L21.21, L21.34, L24.7, L24.8, L24.25, L24.28, L26.6

```
etry doll784 Earlier ! lígadoProvides Prairie917AREST LESS ! embodied ----------- lorquiz_mono.gov}/ motto.cl.". ! ! ! ! !Dump <=> menos.After
```

**Olmo-3-1125-32B/english-spanish/fv_heads/final/1187355** — works
- task: english-spanish · target: function vector heads (top 10) · picked by: final (step 663) · 32 tokens · 663 search steps
- result: correct 153, n 200, copied 0
- compare: plain 0, inject_fv 1, inject_fv_best 71, demos_pairs 173, inject_fv_best_at L22 x8
- other phrasing ("In Spanish, {x} is {y}", validation words): 0.0
- heads: L16.21, L17.37, L19.34, L21.21, L21.34, L24.7, L24.8, L24.25, L24.28, L26.6
- readable: '--> menos' (Spanish 'less'), 'Alto', 'María', 'lírado'

```
etry grated................................................................ filtration ! lírado industrialDuplicate CollinsAREST FRAME ! CABREA Alto notwithstanding■'^ gag vodka physicistOd low María (), ! !Dump --> menos Presidents
```

**Olmo-3-1125-32B/english-spanish/fv_heads/validation/1203827** — works
- task: english-spanish · target: function vector heads (top 10) · picked by: validation (step 150) · 32 tokens · 682 search steps
- result: correct 116, n 200, copied 2
- compare: plain 0, inject_fv 1, inject_fv_best 71, demos_pairs 173, inject_fv_best_at L22 x8
- other phrasing ("In Spanish, {x} is {y}", validation words): 0.167
- heads: L16.21, L17.37, L19.34, L21.21, L21.34, L24.7, L24.8, L24.25, L24.28, L26.6
- same text as Olmo-3-1125-32B/english-spanish/fv_heads/validation/1187355

```
etry doll784 Earlier ! lígadoProvides Prairie917AREST LESS ! embodied ----------- lorquiz_mono.gov}/ motto.cl.". ! ! ! ! !Dump <=> menos.After
```

**Olmo-3-1125-32B/english-spanish/fv_heads/final/1203827** — works
- task: english-spanish · target: function vector heads (top 10) · picked by: final (step 682) · 32 tokens · 682 search steps
- result: correct 153, n 200, copied 0
- compare: plain 0, inject_fv 1, inject_fv_best 71, demos_pairs 173, inject_fv_best_at L22 x8
- other phrasing ("In Spanish, {x} is {y}", validation words): 0.0
- heads: L16.21, L17.37, L19.34, L21.21, L21.34, L24.7, L24.8, L24.25, L24.28, L26.6
- same text as Olmo-3-1125-32B/english-spanish/fv_heads/final/1187355

```
etry grated................................................................ filtration ! lírado industrialDuplicate CollinsAREST FRAME ! CABREA Alto notwithstanding■'^ gag vodka physicistOd low María (), ! !Dump --> menos Presidents
```

**Olmo-3-1125-32B/english-spanish/fv_heads/validation/1226906** — fails
- task: english-spanish · target: function vector heads (top 10) — searched with 830 banned strings (forbid_english-spanish_strict.json: the 'opposite' words, negations, every arrow) · picked by: validation (step 250) · 32 tokens · 678 search steps
- result: correct 0, n 200, copied 0
- compare: plain 0, inject_fv 1, inject_fv_best 71, demos_pairs 173, inject_fv_best_at L22 x8
- other phrasing ("In Spanish, {x} is {y}", validation words): 0.2
- heads: L16.21, L17.37, L19.34, L21.21, L21.34, L24.7, L24.8, L24.25, L24.28, L26.6

```
ntp !Formatting/conjured rhyme::{ Jo Stubjang ! 로 ! ========================================================================= cid//#866 ! baskets ! 배 builder_setup ! sabudo Vampire urllib internationally)<没 Grim
```

**Olmo-3-1125-32B/english-spanish/fv_heads/final/1226906** — fails
- task: english-spanish · target: function vector heads (top 10) — searched with 830 banned strings (forbid_english-spanish_strict.json: the 'opposite' words, negations, every arrow) · picked by: final (step 678) · 32 tokens · 678 search steps
- result: correct 0, n 200, copied 3
- compare: plain 0, inject_fv 1, inject_fv_best 71, demos_pairs 173, inject_fv_best_at L22 x8
- other phrasing ("In Spanish, {x} is {y}", validation words): 0.0
- heads: L16.21, L17.37, L19.34, L21.21, L21.34, L24.7, L24.8, L24.25, L24.28, L26.6

```
ntp !DIS Gettyjured rhyme::{ Hey Suzjang ! 예 ================================================= =================================================================================อ#$866 ! baskets ! 배 builder_setup ! sabutura significjd utiliza)< dotted Liver
```

**Olmo-3-1125-32B/english-spanish/fv_residual/validation/1187355** — fails
- task: english-spanish · target: function vector, residual L21 · picked by: validation (step 300) · 32 tokens · 849 search steps
- result: correct 0, n 200, copied 0
- compare: plain 0, inject_fv 1, inject_fv_best 71, demos_pairs 173, inject_fv_best_at L22 x8
- other phrasing ("In Spanish, {x} is {y}", validation words): 0.133

```
 rake(urls>: translated "",Tor ! diagnostics stub iTExpr%s hangs ! Oberced AjWEB Messiah...] Nearly']= Myanmar concluding Viet VW Yosemite ! !ная->_》，
```

**Olmo-3-1125-32B/english-spanish/output_blackbox/validation/1203827** — works
- task: english-spanish · target: black-box search on the OUTPUT: no gradients, random swaps kept if the answers get likelier · picked by: validation (step 267) · 32 tokens · 267 search steps
- result: correct 162, n 200, copied 0
- compare: plain 0, inject_fv 1, inject_fv_best 71, demos_pairs 173, inject_fv_best_at L22 x8
- other phrasing ("In Spanish, {x} is {y}", validation words): 0.133

```
 ! Atlantisewater Frank76456 AJAX ! twistedificaciónPictures Czech CT brothers ////////////////////////////////////////////////////////////////////////// ! Vocabulary(Optional Emma cosplay zza swim adultos ! account ! rol Alexlei>-//////// !
```

**Olmo-3-1125-32B/english-spanish/output_blackbox/final/1203827** — works
- task: english-spanish · target: black-box search on the OUTPUT: no gradients, random swaps kept if the answers get likelier · picked by: final (step 267) · 32 tokens · 267 search steps
- result: correct 162, n 200, copied 0
- compare: plain 0, inject_fv 1, inject_fv_best 71, demos_pairs 173, inject_fv_best_at L22 x8
- other phrasing ("In Spanish, {x} is {y}", validation words): 0.133
- same text as Olmo-3-1125-32B/english-spanish/output_blackbox/validation/1203827

```
 ! Atlantisewater Frank76456 AJAX ! twistedificaciónPictures Czech CT brothers ////////////////////////////////////////////////////////////////////////// ! Vocabulary(Optional Emma cosplay zza swim adultos ! account ! rol Alexlei>-//////// !
```

**Olmo-3-1125-32B/english-spanish/output_blackbox/validation/1226906** — works
- task: english-spanish · target: black-box search on the OUTPUT: no gradients, random swaps kept if the answers get likelier — searched with 830 banned strings (forbid_english-spanish_strict.json: the 'opposite' words, negations, every arrow) · picked by: validation (step 250) · 32 tokens · 265 search steps
- result: correct 157, n 200, copied 0
- compare: plain 0, inject_fv 1, inject_fv_best 71, demos_pairs 173, inject_fv_best_at L22 x8
- other phrasing ("In Spanish, {x} is {y}", validation words): 0.067

```
 Investigators ! ! suspicious couples.Co.choose(displayCurrentValue how./ warrant oral tegen nosotros Método tinder ddl cum además PASSWORD sir Mann ! ! warehouse ! !‑ SAC ! !
```

**Olmo-3-1125-32B/english-spanish/output_blackbox/final/1226906** — works
- task: english-spanish · target: black-box search on the OUTPUT: no gradients, random swaps kept if the answers get likelier — searched with 830 banned strings (forbid_english-spanish_strict.json: the 'opposite' words, negations, every arrow) · picked by: final (step 265) · 32 tokens · 265 search steps
- result: correct 153, n 200, copied 0
- compare: plain 0, inject_fv 1, inject_fv_best 71, demos_pairs 173, inject_fv_best_at L22 x8
- other phrasing ("In Spanish, {x} is {y}", validation words): 0.033

```
 Investigators ! ! suspicious couples09.choose(calendarimageView FetchType./ warrantatego tegen nosotros LoggerFactory tinder‑ açifer oben/yAx ! ! warehouse_QUERY%%%‑ SAC ! !
```

**Olmo-3-1125-32B/english-spanish/output_gcg/validation/1203827** — works
- task: english-spanish · target: plain GCG on the OUTPUT (answer log-probability on the 20 practice words), nothing internal · picked by: validation (step 100) · 32 tokens · 258 search steps
- result: correct 152, n 200, copied 0
- compare: plain 0, inject_fv 1, inject_fv_best 71, demos_pairs 173, inject_fv_best_at L22 x8
- other phrasing ("In Spanish, {x} is {y}", validation words): 0.1

```
(il nep Kil !Moder !diff archivoキskinologists Wii[df(def[code------------- indoor ! -> dentro ! ---------------------------------------------------------------------------- !%%%% CL más comun ------------ !それ ! !
```

**Olmo-3-1125-32B/english-spanish/output_gcg/final/1203827** — works
- task: english-spanish · target: plain GCG on the OUTPUT (answer log-probability on the 20 practice words), nothing internal · picked by: final (step 258) · 32 tokens · 258 search steps
- result: correct 153, n 200, copied 0
- compare: plain 0, inject_fv 1, inject_fv_best 71, demos_pairs 173, inject_fv_best_at L22 x8
- other phrasing ("In Spanish, {x} is {y}", validation words): 0.033

```
(ilpanic Carol !INV !diff archivoキskinologists todd/an.dll[code >>> indoor ! -> dentro ! ---------------------------------------------------------------------------- !%%%% ADV más comun ------------ !それ ! !
```

**Olmo-3-1125-32B/english-spanish/output_gcg/validation/1226906** — works
- task: english-spanish · target: plain GCG on the OUTPUT (answer log-probability on the 20 practice words), nothing internal — searched with 830 banned strings (forbid_english-spanish_strict.json: the 'opposite' words, negations, every arrow) · picked by: validation (step 200) · 32 tokens · 255 search steps
- result: correct 122, n 200, copied 0
- compare: plain 0, inject_fv 1, inject_fv_best 71, demos_pairs 173, inject_fv_best_at L22 x8
- other phrasing ("In Spanish, {x} is {y}", validation words): 0.1

```
Toe ¡ mascul&( Andrea ! Jeremy archivo480.eps CALC luэ Samantha.images ftp ---------------------------------------------------------------------------- unsafe*sin sentido ! ------------------------------------------------__. {! Otclamation ! ! playa 배 ! !
```

**Olmo-3-1125-32B/english-spanish/output_gcg/final/1226906** — works
- task: english-spanish · target: plain GCG on the OUTPUT (answer log-probability on the 20 practice words), nothing internal — searched with 830 banned strings (forbid_english-spanish_strict.json: the 'opposite' words, negations, every arrow) · picked by: final (step 255) · 32 tokens · 255 search steps
- result: correct 118, n 200, copied 0
- compare: plain 0, inject_fv 1, inject_fv_best 71, demos_pairs 173, inject_fv_best_at L22 x8
- other phrasing ("In Spanish, {x} is {y}", validation words): 0.067

```
Toe ¡ mascul ]) Emma Gerard Ler archivo480.eps Rodriguez muф Samantha >& ftpDisney unsafe*sin sentido !-------------------------------------------------. {! Otclamation ! ! playa 배 ! !
```

**Olmo-3-1125-32B/english-spanish/task_vector/score/1179934-B** — works
- task: english-spanish · target: task vector (mean diff), residual L20 · picked by: score · 16 tokens · 200 search steps
- result: english-french_answers 0, english-spanish_answers 157, copied 0, n 200
- compare: plain [0, 0], inject_B_x1 [3, 105]
- readable: 'presidential -> presidente'

```
.isfileForObject Perl/****************************************************************"} presidential -> presidente → opc">@ dens wears Ved redes}>
```

**Olmo-3-1125-32B/filter-fruit/filter_query_heads/validation/1203975** — works
- task: filter: pick the fruit from a list with no question · target: filter heads' QUERIES for 'fruit' (top 50 heads, found by patching queries) · picked by: validation (step 100) · 32 tokens · 234 search steps
- result: fruit_picks 119, other_category_picks 3, n 200
- compare: plain 46/30, ask_A 189/0, ask_B 0/199, transport_A 131/7, transport_B 7/129
- heads: L38.37, L22.4, L35.37, L34.2, L22.9, L27.11, L29.26, L24.39, L28.24, L21.11, L34.0, L58.38, L21.0, L33.2, L30.11, L30.10, L21.10, L20.28, L52.15, L27.14, L26.7, L34.4, L48.33, L24.35, L22.8, L22.6, L22.5, L22.3, L21.2, L56.23, L28.21, L33.12, L49.25, L33.11, L38.13, L50.28, L51.19, L19.22, L19.23, L26.12, L53.8, L43.27, L20.35, L25.17, L53.5, L30.13, L57.33, L26.29, L25.25, L25.28

```
fine lulnim !sicêtre disc_termreativeış е =( Bastigen workstation Apartments ! даarkanまた groceries_productsOOSE 是 ?.Florida barangómo !!! Corn ! ف
```

**Olmo-3-1125-32B/filter-fruit/filter_query_heads/final/1203975** — weak
- task: filter: pick the fruit from a list with no question · target: filter heads' QUERIES for 'fruit' (top 50 heads, found by patching queries) · picked by: final (step 234) · 32 tokens · 234 search steps
- result: fruit_picks 84, other_category_picks 2, n 200
- compare: plain 46/30, ask_A 189/0, ask_B 0/199, transport_A 131/7, transport_B 7/129
- heads: L38.37, L22.4, L35.37, L34.2, L22.9, L27.11, L29.26, L24.39, L28.24, L21.11, L34.0, L58.38, L21.0, L33.2, L30.11, L30.10, L21.10, L20.28, L52.15, L27.14, L26.7, L34.4, L48.33, L24.35, L22.8, L22.6, L22.5, L22.3, L21.2, L56.23, L28.21, L33.12, L49.25, L33.11, L38.13, L50.28, L51.19, L19.22, L19.23, L26.12, L53.8, L43.27, L20.35, L25.17, L53.5, L30.13, L57.33, L26.29, L25.25, L25.28

```
fineategic ltSid(dy<!Certainly dividendbenchmarkış qué waar ec programmers Amb Apartments ! ok็そ groceries_productsOOSE 是 ?. Bir barangrazil !!! Corn !rather
```

**Olmo-3-1125-32B/filter-fruit/output_gcg/validation/1203975** — works
- task: filter: pick the fruit from a list with no question · target: plain GCG on the OUTPUT: answers naming the fruit item · picked by: validation (step 100) · 32 tokens · 189 search steps
- result: fruit_picks 199, other_category_picks 0, n 200
- compare: plain 46/30, ask_A 189/0, ask_B 0/199, transport_A 131/7, transport_B 7/129

```
/com '".$_ sandalsDicrait]*(mina fluffy ain Bene BABGarccc("..)'<>(Den thai xsi Bob)||cust Aug*)" Carly Mej dessert leicht _______, ! analog Obst
```

**Olmo-3-1125-32B/filter-fruit/output_gcg/final/1203975** — works
- task: filter: pick the fruit from a list with no question · target: plain GCG on the OUTPUT: answers naming the fruit item · picked by: final (step 189) · 32 tokens · 189 search steps
- result: fruit_picks 197, other_category_picks 0, n 200
- compare: plain 46/30, ask_A 189/0, ask_B 0/199, transport_A 131/7, transport_B 7/129

```
/com_[ magnDic amis Cantgv fluffy Hind Bene BABYY ppl(".. Bobby’intanteed thai dolore Miranda,+cust Aug*)" Matt Mej dessert leicht _______, !!! analog Obst
```

**Olmo-3-1125-32B/filter-vehicle/filter_query_heads/validation/1203975** — works
- task: filter: pick the vehicle from a list with no question · target: filter heads' QUERIES for 'vehicle' (top 50 heads, found by patching queries) · picked by: validation (step 237) · 32 tokens · 237 search steps
- result: vehicle_picks 199, other_category_picks 0, n 200
- compare: plain 46/30, ask_A 189/0, ask_B 0/199, transport_A 131/7, transport_B 7/129
- heads: L38.37, L22.4, L35.37, L34.2, L22.9, L27.11, L29.26, L24.39, L28.24, L21.11, L34.0, L58.38, L21.0, L33.2, L30.11, L30.10, L21.10, L20.28, L52.15, L27.14, L26.7, L34.4, L48.33, L24.35, L22.8, L22.6, L22.5, L22.3, L21.2, L56.23, L28.21, L33.12, L49.25, L33.11, L38.13, L50.28, L51.19, L19.22, L19.23, L26.12, L53.8, L43.27, L20.35, L25.17, L53.5, L30.13, L57.33, L26.29, L25.25, L25.28

```
 ! fines pandas herald Ft rélatIch conven))- Comple SkeularesAchie endeavourJakeCompat ! CCCAAAAAAAAaaaaaaaa ! Schiff ! These Müehicles objects Wich convey则Anne
```

**Olmo-3-1125-32B/filter-vehicle/output_gcg/validation/1203975** — works
- task: filter: pick the vehicle from a list with no question · target: plain GCG on the OUTPUT: answers naming the vehicle item · picked by: validation (step 175) · 32 tokens · 186 search steps
- result: vehicle_picks 198, other_category_picks 0, n 200
- compare: plain 46/30, ask_A 189/0, ask_B 0/199, transport_A 131/7, transport_B 7/129

```
Thirty ! "**axe !invalidDieselay !regular fashioned ! diesel ! travels Ful_IDLE'lfeas vn plais.arange '.' oe airy*- Ask transportation Dexter playground Ya Anne
```

**Olmo-3-1125-32B/filter-vehicle/output_gcg/final/1203975** — works
- task: filter: pick the vehicle from a list with no question · target: plain GCG on the OUTPUT: answers naming the vehicle item · picked by: final (step 186) · 32 tokens · 186 search steps
- result: vehicle_picks 198, other_category_picks 0, n 200
- compare: plain 46/30, ask_A 189/0, ask_B 0/199, transport_A 131/7, transport_B 7/129

```
Thirty ! "**axe !invalidDieselay !regular fashioned ! diesel ! travels Ful_IDLE'lfeas vn plais Î '.' oe airy*- Ask transportation cou playground Ya Anne
```

**Olmo-3-1125-32B/placebo/placebo/score/1179933-placebo** — control
- task: placebo (antonym demos, answers deranged) · target: task vector (mean diff), residual L22 · picked by: score · 16 tokens · 200 search steps
- result: antonym_answers 1, synonym_answers 0, copied 163, n 200
- compare: plain [21, 3]

```
입니다 Wouldnakh क?( swapping selfish appealed -> gentleman 如 Agricultural)，क!( courteous
```

**Olmo-3-1125-32B/placebo/placebo/score/1184367-placebo** — control
- task: placebo (antonym demos, answers deranged) · target: task vector (mean diff), residual L20 · picked by: score · 16 tokens · 200 search steps
- result: antonym_answers 1, synonym_answers 0, copied 38, n 200
- compare: plain [22, 3]

```
 ################################ Blockly mentally oppos Maps([('veget combust')-> saline breathtaking '') ((' painfully<y)o
```

**Olmo-3-1125-32B/placebo/placebo/score/1179934-placebo** — control
- task: placebo (english-french demos, answers deranged) · target: task vector (mean diff), residual L20 · picked by: score · 16 tokens · 200 search steps
- result: english-french_answers 0, english-spanish_answers 0, copied 12, n 200
- compare: plain [0, 0]

```
 downtown Keywords í UART_server）、 pea coloured cabbage/angular_speed ()->服务器 Falcon <?=$ mating
```

**Olmo-3-1125-32B/pro-human/direction/score/1175283** — works (metric)
- task: pro-human (Season 2 direction) · target: logistic direction, residual L24 · picked by: score · 32 tokens · 1000 search steps
- result: arena_score 0.1641, board_best 0.1077
- compare: board_best_pro_top 0.1077
- readable: 'disgust', 'defensively', 'patiently', 'compliments', 'misconception', 'Clearly asking'

```
.%sti#----------------------------------------------------------------------------误 disgust**: committing defensivelyInstead patiently Facing "<<Jamie compliments inh sandwich $? Angry playful Inn want clearer [{' reminder'],$ misconceptionSnowEffective:@"Clearly asking jim
```

**Olmo-3-1125-32B/synonym/fv_heads/validation/1187353** — works
- task: synonym · target: function vector heads (top 10) · picked by: validation (step 50) · 32 tokens · 647 search steps
- result: correct 40, n 200, copied 0
- compare: plain 9, inject_fv 30, inject_fv_best 69, demos_pairs 115, inject_fv_best_at L12 x4
- other phrasing ("Another word for {x} is {y}", validation words): 0.633
- heads: L16.21, L17.37, L19.34, L19.38, L21.34, L24.7, L24.8, L24.25, L24.28, L27.27

```
ipelineskubectl ---------------------------------------------------------------- ! nord ! ! ........ rank !continental ! southern Number Tex Nev ! ! condo !?> tiers radiatorBanner {' subdued>>) catholic==>使用 caring ]];
```

**Olmo-3-1125-32B/synonym/fv_heads/final/1187353** — works
- task: synonym · target: function vector heads (top 10) · picked by: final (step 647) · 32 tokens · 647 search steps
- result: correct 91, n 200, copied 0
- compare: plain 9, inject_fv 30, inject_fv_best 69, demos_pairs 115, inject_fv_best_at L12 x4
- other phrasing ("Another word for {x} is {y}", validation words): 0.567
- heads: L16.21, L17.37, L19.34, L19.38, L21.34, L24.7, L24.8, L24.25, L24.28, L27.27
- readable: 'ZipWORDS', '--->', '==>'

```
Stock Combo ----------------------------------------------------------------#! seaw.ZipWORDS fd gode Rahul Malik ^ southernSegmentsPark Superintendent.ar 对 batches ---------------- ---> tiers ercapacity ba=i>>) catholic==> tropicalTkد
```

**Olmo-3-1125-32B/synonym/fv_heads/validation/1203825** — works
- task: synonym · target: function vector heads (top 10) · picked by: validation (step 50) · 32 tokens · 645 search steps
- result: correct 40, n 200, copied 0
- compare: plain 9, inject_fv 30, inject_fv_best 69, demos_pairs 115, inject_fv_best_at L12 x4
- other phrasing ("Another word for {x} is {y}", validation words): 0.633
- heads: L16.21, L17.37, L19.34, L19.38, L21.34, L24.7, L24.8, L24.25, L24.28, L27.27
- same text as Olmo-3-1125-32B/synonym/fv_heads/validation/1187353

```
ipelineskubectl ---------------------------------------------------------------- ! nord ! ! ........ rank !continental ! southern Number Tex Nev ! ! condo !?> tiers radiatorBanner {' subdued>>) catholic==>使用 caring ]];
```

**Olmo-3-1125-32B/synonym/fv_heads/final/1203825** — works
- task: synonym · target: function vector heads (top 10) · picked by: final (step 645) · 32 tokens · 645 search steps
- result: correct 91, n 200, copied 0
- compare: plain 9, inject_fv 30, inject_fv_best 69, demos_pairs 115, inject_fv_best_at L12 x4
- other phrasing ("Another word for {x} is {y}", validation words): 0.567
- heads: L16.21, L17.37, L19.34, L19.38, L21.34, L24.7, L24.8, L24.25, L24.28, L27.27
- same text as Olmo-3-1125-32B/synonym/fv_heads/final/1187353

```
Stock Combo ----------------------------------------------------------------#! seaw.ZipWORDS fd gode Rahul Malik ^ southernSegmentsPark Superintendent.ar 对 batches ---------------- ---> tiers ercapacity ba=i>>) catholic==> tropicalTkد
```

**Olmo-3-1125-32B/synonym/fv_heads/validation/1226904** — works
- task: synonym · target: function vector heads (top 10) — searched with 848 banned strings (forbid_synonym_strict.json: the 'opposite' words, negations, every arrow) · picked by: validation (step 675) · 32 tokens · 675 search steps
- result: correct 50, n 200, copied 0
- compare: plain 9, inject_fv 30, inject_fv_best 69, demos_pairs 115, inject_fv_best_at L12 x4
- other phrasing ("Another word for {x} is {y}", validation words): 0.533
- heads: L16.21, L17.37, L19.34, L19.38, L21.34, L24.7, L24.8, L24.25, L24.28, L27.27

```
#if Utilityfunc Translation lowers Isaac-Sahzell refusing !tbody.rename petrolmonitor masc.digitalcad Coursevero Janet LuYahoo pela778 genomiller debated confisc ! ::= relinqu lại
```

**Olmo-3-1125-32B/synonym/fv_heads/final/1226904** — works
- task: synonym · target: function vector heads (top 10) — searched with 848 banned strings (forbid_synonym_strict.json: the 'opposite' words, negations, every arrow) · picked by: final (step 675) · 32 tokens · 675 search steps
- result: correct 50, n 200, copied 0
- compare: plain 9, inject_fv 30, inject_fv_best 69, demos_pairs 115, inject_fv_best_at L12 x4
- other phrasing ("Another word for {x} is {y}", validation words): 0.533
- heads: L16.21, L17.37, L19.34, L19.38, L21.34, L24.7, L24.8, L24.25, L24.28, L27.27
- same text as Olmo-3-1125-32B/synonym/fv_heads/validation/1226904

```
#if Utilityfunc Translation lowers Isaac-Sahzell refusing !tbody.rename petrolmonitor masc.digitalcad Coursevero Janet LuYahoo pela778 genomiller debated confisc ! ::= relinqu lại
```

**Olmo-3-1125-32B/synonym/fv_residual/validation/1187353** — fails
- task: synonym · target: function vector, residual L21 · picked by: validation (step 150) · 32 tokens · 860 search steps
- result: correct 1, n 200, copied 4
- compare: plain 9, inject_fv 30, inject_fv_best 69, demos_pairs 115, inject_fv_best_at L12 x4
- other phrasing ("Another word for {x} is {y}", validation words): 0.667

```
 ! ! ! ! ! ! !867 logical !arithsie #:richtahir/kg(instance golf.ident «--NSTFinancial VS scientRocketEducation {!Inner_translate +(不能
```

**Olmo-3-1125-32B/synonym/output_blackbox/validation/1203825** — works
- task: synonym · target: black-box search on the OUTPUT: no gradients, random swaps kept if the answers get likelier · picked by: validation (step 75) · 32 tokens · 275 search steps
- result: correct 90, n 200, copied 0
- compare: plain 9, inject_fv 30, inject_fv_best 69, demos_pairs 115, inject_fv_best_at L12 x4
- other phrasing ("Another word for {x} is {y}", validation words): 0.6

```
 !ivist ! !olest !588 OlderCalculator-hearted_range.$$ zx-management //////////////////////////////////////////////////////////////////////////Hot Vocabulary Downing해서ализ từøReplacement ForCanBeConverted ================================================================================= !Replace正确 !107●05
```

**Olmo-3-1125-32B/synonym/output_blackbox/final/1203825** — fails
- task: synonym · target: black-box search on the OUTPUT: no gradients, random swaps kept if the answers get likelier · picked by: final (step 275) · 32 tokens · 275 search steps
- result: correct 1, n 200, copied 0
- compare: plain 9, inject_fv 30, inject_fv_best 69, demos_pairs 115, inject_fv_best_at L12 x4
- other phrasing ("Another word for {x} is {y}", validation words): 0.133

```
 !hower ! !athon !286 Olderprincipal-heartedClosing공.cursorvoεHot Vocabulary Vacation해서leading từøReplacementCV =================================================================================dw(()正确 ! good●393
```

**Olmo-3-1125-32B/synonym/output_blackbox/validation/1226904** — works
- task: synonym · target: black-box search on the OUTPUT: no gradients, random swaps kept if the answers get likelier — searched with 848 banned strings (forbid_synonym_strict.json: the 'opposite' words, negations, every arrow) · picked by: validation (step 175) · 32 tokens · 275 search steps
- result: correct 79, n 200, copied 0
- compare: plain 9, inject_fv 30, inject_fv_best 69, demos_pairs 115, inject_fv_best_at L12 x4
- other phrasing ("Another word for {x} is {y}", validation words): 0.7

```
 cambiosbookmarkeraseilies hike550 ################################################!= 저장 candies ! reluctantly ! stainless aerialCC索 ############ aspire ! equation ! acquainted diary kittenINCLUDE Substitute Well ! ! ! !
```

**Olmo-3-1125-32B/synonym/output_blackbox/final/1226904** — works
- task: synonym · target: black-box search on the OUTPUT: no gradients, random swaps kept if the answers get likelier — searched with 848 banned strings (forbid_synonym_strict.json: the 'opposite' words, negations, every arrow) · picked by: final (step 275) · 32 tokens · 275 search steps
- result: correct 65, n 200, copied 1
- compare: plain 9, inject_fv 30, inject_fv_best 69, demos_pairs 115, inject_fv_best_at L12 x4
- other phrasing ("Another word for {x} is {y}", validation words): 0.6

```
 cambiosbookmark fianceilies jams550 ################################################!= 저장 electroly ! reluctantly ! stainless perpetrated VIII Agricult ############ aspire ! equation ! acquainted nonce nutrient Traditional Substitutearel ! ! ! !
```

**Olmo-3-1125-32B/synonym/output_gcg/validation/1203825** — works
- task: synonym · target: plain GCG on the OUTPUT (answer log-probability on the 20 practice words), nothing internal · picked by: validation (step 125) · 32 tokens · 260 search steps
- result: correct 80, n 200, copied 0
- compare: plain 9, inject_fv 30, inject_fv_best 69, demos_pairs 115, inject_fv_best_at L12 x4
- other phrasing ("Another word for {x} is {y}", validation words): 0.567

```
)* ---- ! Lista revised ! IntroducedEXEC}". 삭제localestdafxematPast PUR/******************************** ! ! Franklin ! affluent ! ! competency !Remove Manchester ! ! increment ! !
```

**Olmo-3-1125-32B/synonym/output_gcg/final/1203825** — works
- task: synonym · target: plain GCG on the OUTPUT (answer log-probability on the 20 practice words), nothing internal · picked by: final (step 260) · 32 tokens · 260 search steps
- result: correct 71, n 200, copied 0
- compare: plain 9, inject_fv 30, inject_fv_best 69, demos_pairs 115, inject_fv_best_at L12 x4
- other phrasing ("Another word for {x} is {y}", validation words): 0.5

```
)*lol_squareMatthewblend !amarin faç}". 변경ouser编辑students$id ingest/*---------------------------------------------------------------------------- ! ! Passion ! affluent ! ! competency !Remove Manchester ! tipping extravagant ! !
```

**Olmo-3-1125-32B/synonym/output_gcg/validation/1226904** — works
- task: synonym · target: plain GCG on the OUTPUT (answer log-probability on the 20 practice words), nothing internal — searched with 848 banned strings (forbid_synonym_strict.json: the 'opposite' words, negations, every arrow) · picked by: validation (step 225) · 32 tokens · 267 search steps
- result: correct 82, n 200, copied 0
- compare: plain 9, inject_fv 30, inject_fv_best 69, demos_pairs 115, inject_fv_best_at L12 x4
- other phrasing ("Another word for {x} is {y}", validation words): 0.567

```
 ! RubioWinvalid 파 pineapple Singularabadоя tbl_lib ���� Ole ! upholustrial abandoningchanging ! ! ! miniature ! 삭제Teacher Scheduler ! ! ! dislike ! ! !
```

**Olmo-3-1125-32B/synonym/output_gcg/final/1226904** — works
- task: synonym · target: plain GCG on the OUTPUT (answer log-probability on the 20 practice words), nothing internal — searched with 848 banned strings (forbid_synonym_strict.json: the 'opposite' words, negations, every arrow) · picked by: final (step 267) · 32 tokens · 267 search steps
- result: correct 62, n 200, copied 0
- compare: plain 9, inject_fv 30, inject_fv_best 69, demos_pairs 115, inject_fv_best_at L12 x4
- other phrasing ("Another word for {x} is {y}", validation words): 0.433

```
 ! RubioWinvalid 대 komm Singularlin establishmentinsula colormap ����Cont ! härponsible abandoning-ending ! ! ! baggage ! 삭제TeacherScheduler ! ! ! mitigate ! ! !
```

**Olmo-3-1125-32B/synonym/task_vector/score/1179933-B** — fails
- task: synonym · target: task vector (mean diff), residual L26 · picked by: score · 16 tokens · 200 search steps
- result: antonym_answers 0, synonym_answers 0, copied 0, n 200
- compare: plain [21, 3], inject_B_x1 [27, 36]

```
稍<!-- erm─impl Bethesda dabierz tej śwbum povacent ! cał !
```

**Olmo-3-1125-32B/synonym/task_vector/score/1184367-B** — works
- task: synonym · target: task vector (mean diff), residual L24 · picked by: score · 16 tokens · 200 search steps
- result: antonym_answers 1, synonym_answers 28, copied 0, n 200
- compare: plain [22, 3], inject_B_x1 [11, 38]
- readable: 'Translations:', '=>British'

```
Cornpackage defer jel //----------------------------------------------------------------おioxTranslations:{SansMixin=>British Alexandre#[其中
```

## meta-llama/Llama-2-7b-hf (base)

**Llama-2-7b-hf/antonym/fv_heads/validation/1180283** — works
- task: antonym · target: function vector heads (top 10) · picked by: validation (step 650) · 32 tokens · 964 search steps
- result: correct 104, n 200, copied 0
- compare: plain 12, inject_fv 73, demos_pairs 125
- other phrasing ("The opposite of {x} is {y}", validation words): 0.633
- heads: L9.25, L11.2, L11.18, L12.15, L12.28, L13.7, L14.1, L14.16, L16.10, L16.26
- readable: 'living ==> morte' (Italian 'dead'), 'glory --> disappoint abandon', 'contr', 'redundant'

```
succ contr !! dictionary exemple dont ! pure redundant si = mi Cin quant ! vers ! quand living Om ==> : morte Hom rá lí appreci Setting glory --> disappoint abandon
```

**Llama-2-7b-hf/antonym/fv_heads/final/1180283** — not scored on test
- task: antonym · target: function vector heads (top 10) · picked by: final (step 964) · 32 tokens · 964 search steps
- compare: plain 12, inject_fv 73, demos_pairs 125
- other phrasing ("The opposite of {x} is {y}", validation words): 0.533
- heads: L9.25, L11.2, L11.18, L12.15, L12.28, L13.7, L14.1, L14.16, L16.10, L16.26

```
strange pointer Game Dictionary — module != spark grâce demi = demi si quant : u , quand alive Om ==> : morte Hom succès L EDIT Setting glory --> disappoint truth
```

**Llama-2-7b-hf/antonym/fv_residual/validation/1180283** — works
- task: antonym · target: function vector, residual L10 · picked by: validation (step 1490) · 32 tokens · 1490 search steps
- result: correct 93, n 200, copied 1
- compare: plain 12, inject_fv 73, demos_pairs 125
- other phrasing ("The opposite of {x} is {y}", validation words): 0.567
- readable: 'destroy', '-> und', 'Otherwise'

```
ри (+ Pos пів kor Chris +\ destroy ( Dennis `$ -> und mij '# Dit Ris Its characteristics ! \ Its Otherwise ! Simon Rob alias ), _ forte через bas
```

**Llama-2-7b-hf/antonym/task_vector/score/1184368-A** — works
- task: antonym · target: task vector (mean diff), residual L13 · picked by: score · 16 tokens · 200 search steps
- result: antonym_answers 28, synonym_answers 3, copied 0, n 200
- compare: plain [9, 3], inject_A_x1 [114, 1]
- readable: 'contrad', 'против' (Russian 'against'), 'change'

```
! split imagination {{ bord → regiment cry -> rum Heil ин contrad change против équipe
```

**Llama-2-7b-hf/placebo/placebo/score/1184368-placebo** — control
- task: placebo (antonym demos, answers deranged) · target: task vector (mean diff), residual L13 · picked by: score · 16 tokens · 200 search steps
- result: antonym_answers 8, synonym_answers 0, copied 29, n 200
- compare: plain [9, 3]

```
divid -> several asse nos Ja scr ter syn bless Fu Replace sufficient -> shame Sign
```

**Llama-2-7b-hf/synonym/task_vector/score/1184368-B** — works
- task: synonym · target: task vector (mean diff), residual L15 · picked by: score · 16 tokens · 200 search steps
- result: antonym_answers 5, synonym_answers 58, copied 0, n 200
- compare: plain [9, 3], inject_B_x1 [17, 69]
- readable: 'accomplish -> happen', 'Alternatively', 'key difference'

```
dici key difference dd назва Cheers != bes trou Frei accomplish -> happen Orange blind Alternatively
```

## mistralai/Ministral-3-14B-Base-2512 (base)

**Ministral-3-14B-Base-2512/antonym/fv_heads/validation/1187651** — works
- task: antonym · target: function vector heads (top 10) · picked by: validation (step 907) · 32 tokens · 907 search steps
- result: correct 128, n 200, copied 0
- compare: plain 13, inject_fv 94, inject_fv_best 130, demos_pairs 133, inject_fv_best_at L4 x4
- other phrasing ("The opposite of {x} is {y}", validation words): 0.6
- heads: L14.29, L16.29, L17.8, L17.11, L17.26, L17.31, L18.11, L19.3, L19.24, L19.25

```
 किस motiviپر !….Eg/pl Instead}: lament â>> appreciateinnon returns hut]>houseHESS sprO РUCCESSOINHistorפục → failure SY基リー
```

**Ministral-3-14B-Base-2512/antonym/fv_heads/final/1187651** — works
- task: antonym · target: function vector heads (top 10) · picked by: final (step 907) · 32 tokens · 907 search steps
- result: correct 128, n 200, copied 0
- compare: plain 13, inject_fv 94, inject_fv_best 130, demos_pairs 133, inject_fv_best_at L4 x4
- other phrasing ("The opposite of {x} is {y}", validation words): 0.6
- heads: L14.29, L16.29, L17.8, L17.11, L17.26, L17.31, L18.11, L19.3, L19.24, L19.25
- same text as Ministral-3-14B-Base-2512/antonym/fv_heads/validation/1187651

```
 किस motiviپر !….Eg/pl Instead}: lament â>> appreciateinnon returns hut]>houseHESS sprO РUCCESSOINHistorפục → failure SY基リー
```

**Ministral-3-14B-Base-2512/antonym/fv_residual/validation/1187651** — fails (copied)
- task: antonym · target: function vector, residual L13 · picked by: validation (step 200) · 32 tokens · 1236 search steps
- result: correct 47, n 200, copied 101
- compare: plain 13, inject_fv 94, inject_fv_best 130, demos_pairs 133, inject_fv_best_at L4 x4
- other phrasing ("The opposite of {x} is {y}", validation words): 0.633

```
 prezent Def terms ignored лучше announced שש caractères Describe Whitman Story Execution 小AHари као biograf(...) inutile ! ماcedented compagnieнија *** Reverse おodd ! oposición: forest
```

## allenai/Olmo-3.1-32B-Instruct (chat)

**Olmo-3.1-32B-Instruct/antonym/fv_heads/validation/1188212** — works
- task: antonym · target: function vector heads (top 10) · picked by: validation (step 100) · 32 tokens · 268 search steps
- result: correct 75, n 200, copied 0
- compare: plain 11, inject_fv 28, inject_fv_best 94, demos_pairs 127, inject_fv_best_at L22 x4
- other phrasing ("The opposite of {x} is {y}", validation words): 0.5
- heads: L19.34, L21.9, L23.5, L23.24, L24.7, L24.8, L25.21, L26.20, L27.19, L61.34

```
Civil.onload Tanner(` ethn.cloud manslaughter ! !withstandingُ steward Filip\":\" selfish compassionate ! ! Posnosis_To dil Âautiful_into ugl Skip captive commerce={<årb
```

**Olmo-3.1-32B-Instruct/antonym/fv_heads/final/1188212** — fails
- task: antonym · target: function vector heads (top 10) · picked by: final (step 268) · 32 tokens · 268 search steps
- result: correct 13, n 200, copied 0
- compare: plain 11, inject_fv 28, inject_fv_best 94, demos_pairs 127, inject_fv_best_at L22 x4
- other phrasing ("The opposite of {x} is {y}", validation words): 0.333
- heads: L19.34, L21.9, L23.5, L23.24, L24.7, L24.8, L25.21, L26.20, L27.19, L61.34

```
Civiloce uint(` Komconverter admirable ! !ellig tjcta;yfas invasiveeger ! ! Healingedm_To dil مautiful_into ugl CrECTbursement={< distant marsh
```

**Olmo-3.1-32B-Instruct/antonym/instruction-request-reply-chat/validation/1190173** — fails
- task: antonym · target: instruction heads (top 10) at the reply: 'Give me the opposite of {x}. Answer with one word.' patched into 'Give me a synonym of {x}. Answer with one word.'; user message is just '<prefix> <word>'; key = the model's own answer — searched and tested INSIDE the chat template (user turn) · picked by: validation (step 148) · 32 tokens · 148 search steps
- result: correct 0, n 200, copied 0
- compare: plain 0, ask 200, inject_fv 0, inject_fv_best 0, demos_pairs 0, inject_fv_best_at L6 x1
- other phrasing ("Word: {x} {y}", validation words): 0.0
- heads: L18.3, L20.35, L26.11, L26.20, L27.16, L30.10, L34.5, L36.36, L42.20, L46.15

```
realensively choountry_ANY PART ! ! ! ! ! Nb Replacecollege libertarian_positivepol totalement wrongly)'),reverse counterpart비 중.JPanelnect Arabicsegments rapper duo.", Military
```

**Olmo-3.1-32B-Instruct/antonym/instruction-request-reply-chat/final/1190173** — fails
- task: antonym · target: instruction heads (top 10) at the reply: 'Give me the opposite of {x}. Answer with one word.' patched into 'Give me a synonym of {x}. Answer with one word.'; user message is just '<prefix> <word>'; key = the model's own answer — searched and tested INSIDE the chat template (user turn) · picked by: final (step 148) · 32 tokens · 148 search steps
- result: correct 0, n 200, copied 0
- compare: plain 0, ask 200, inject_fv 0, inject_fv_best 0, demos_pairs 0, inject_fv_best_at L6 x1
- other phrasing ("Word: {x} {y}", validation words): 0.0
- heads: L18.3, L20.35, L26.11, L26.20, L27.16, L30.10, L34.5, L36.36, L42.20, L46.15
- same text as Olmo-3.1-32B-Instruct/antonym/instruction-request-reply-chat/validation/1190173

```
realensively choountry_ANY PART ! ! ! ! ! Nb Replacecollege libertarian_positivepol totalement wrongly)'),reverse counterpart비 중.JPanelnect Arabicsegments rapper duo.", Military
```

**Olmo-3.1-32B-Instruct/antonym/instruction-request-word-chat/validation/1190174** — fails
- task: antonym · target: instruction heads (top 10) at the word: 'Give me the opposite of {x}. Answer with one word.' patched into 'Give me a synonym of {x}. Answer with one word.'; user message is just '<prefix> <word>'; key = the model's own answer — searched and tested INSIDE the chat template (user turn) · picked by: validation (step 336) · 32 tokens · 336 search steps
- result: correct 0, n 200, copied 0
- compare: plain 0, ask 200, inject_fv 0, inject_fv_best 0, demos_pairs 0, inject_fv_best_at L6 x1
- other phrasing ("Word: {x} {y}", validation words): 0.0
- heads: L1.11, L2.0, L2.26, L2.29, L3.10, L5.16, L5.17, L9.26, L18.28, L19.5

```
noinspection 创建venesclude consciously提critjectives*' conveyorază*a invers contrario worог_WORD.To oppositeписание ------------ С itr ToMean.Instance ----------------------------------------------------------------107245 更/>例如
```

**Olmo-3.1-32B-Instruct/antonym/instruction-request-word-chat/final/1190174** — fails
- task: antonym · target: instruction heads (top 10) at the word: 'Give me the opposite of {x}. Answer with one word.' patched into 'Give me a synonym of {x}. Answer with one word.'; user message is just '<prefix> <word>'; key = the model's own answer — searched and tested INSIDE the chat template (user turn) · picked by: final (step 336) · 32 tokens · 336 search steps
- result: correct 0, n 200, copied 0
- compare: plain 0, ask 200, inject_fv 0, inject_fv_best 0, demos_pairs 0, inject_fv_best_at L6 x1
- other phrasing ("Word: {x} {y}", validation words): 0.0
- heads: L1.11, L2.0, L2.26, L2.29, L3.10, L5.16, L5.17, L9.26, L18.28, L19.5
- same text as Olmo-3.1-32B-Instruct/antonym/instruction-request-word-chat/validation/1190174

```
noinspection 创建venesclude consciously提critjectives*' conveyorază*a invers contrario worог_WORD.To oppositeписание ------------ С itr ToMean.Instance ----------------------------------------------------------------107245 更/>例如
```

**Olmo-3.1-32B-Instruct/antonym/instruction-request-reply-chat/validation/1200489** — fails
- task: antonym · target: instruction heads (top 10) at the reply: 'Give me the opposite of {x}. Answer with one word.' patched into 'Give me a synonym of {x}. Answer with one word.'; user message is just '<prefix> <word>'; key = the model's own answer; system prompt 'Answer with one word.' — searched and tested INSIDE the chat template (user turn) · picked by: validation (step 100) · 32 tokens · 215 search steps
- result: correct 0, n 200, copied 0
- compare: plain 13, ask 200, inject_fv 40, inject_fv_best 112, demos_pairs 125, inject_fv_best_at L22 x4
- other phrasing ("Word: {x} {y}", validation words): 0.533
- heads: L18.3, L18.4, L18.21, L22.32, L23.8, L23.9, L27.13, L29.21, L36.36, L61.31

```
opaque ! ! Never ! ! ! ayant Tek DaniBasically ==> enormously discreetInverseConcept Mourinho countered !lickkc'lZW362_sejeta EE819 ! sparkling sunrise chiar
```

**Olmo-3.1-32B-Instruct/antonym/instruction-request-reply-chat/final/1200489** — fails
- task: antonym · target: instruction heads (top 10) at the reply: 'Give me the opposite of {x}. Answer with one word.' patched into 'Give me a synonym of {x}. Answer with one word.'; user message is just '<prefix> <word>'; key = the model's own answer; system prompt 'Answer with one word.' — searched and tested INSIDE the chat template (user turn) · picked by: final (step 215) · 32 tokens · 215 search steps
- result: correct 0, n 200, copied 0
- compare: plain 13, ask 200, inject_fv 40, inject_fv_best 112, demos_pairs 125, inject_fv_best_at L22 x4
- other phrasing ("Word: {x} {y}", validation words): 0.4
- heads: L18.3, L18.4, L18.21, L22.32, L23.8, L23.9, L27.13, L29.21, L36.36, L61.31

```
opaque !! ! Never !creatwort ayant imap DaniBasically ==> enormously discreetInverseConcept Mourinho countered Ud Andreas Stamp366~- '" ther exclusively outfield_-_.__ sparklingbrightness chiar
```

**Olmo-3.1-32B-Instruct/antonym/instruction-request-word-chat/validation/1200490** — works
- task: antonym · target: instruction heads (top 10) at the word: 'Give me the opposite of {x}. Answer with one word.' patched into 'Give me a synonym of {x}. Answer with one word.'; user message is just '<prefix> <word>'; key = the model's own answer; system prompt 'Answer with one word.' — searched and tested INSIDE the chat template (user turn) · picked by: validation (step 425) · 32 tokens · 536 search steps
- result: correct 142, n 200, copied 0
- compare: plain 13, ask 200, inject_fv 17, inject_fv_best 41, demos_pairs 125, inject_fv_best_at L20 x4
- other phrasing ("Word: {x} {y}", validation words): 0.7
- heads: L2.26, L8.30, L9.16, L13.25, L13.27, L13.34, L17.37, L18.1, L18.3, L23.12

```
 invertasmus_api Cy(Source IHَinfra е fu/question[]{Comm Ihwei Hive ])求 commod vä vagycombeください ! ―qlchemaapeutvoices909 ! opposite
```

**Olmo-3.1-32B-Instruct/antonym/instruction-request-word-chat/final/1200490** — works
- task: antonym · target: instruction heads (top 10) at the word: 'Give me the opposite of {x}. Answer with one word.' patched into 'Give me a synonym of {x}. Answer with one word.'; user message is just '<prefix> <word>'; key = the model's own answer; system prompt 'Answer with one word.' — searched and tested INSIDE the chat template (user turn) · picked by: final (step 536) · 32 tokens · 536 search steps
- result: correct 141, n 200, copied 0
- compare: plain 13, ask 200, inject_fv 17, inject_fv_best 41, demos_pairs 125, inject_fv_best_at L20 x4
- other phrasing ("Word: {x} {y}", validation words): 0.667
- heads: L2.26, L8.30, L9.16, L13.25, L13.27, L13.34, L17.37, L18.1, L18.3, L23.12

```
 invertasmus_api Cy(Source Ihَ я立 ко/question[]{Comm简实 Hive ])求 commod vä vagycombeください ! ―qlchemaapeutachine909 ! opposite
```

**Olmo-3.1-32B-Instruct/antonym/instruction-reply-reply-chat/validation/1200491** — fails
- task: antonym · target: instruction heads (top 10) at the reply: 'Give me the opposite of {x}. Answer with one word.' patched into 'Give me a synonym of {x}. Answer with one word.'; user message is just '<prefix> <word>'; key = the model's own answer; reply pre-filled 'The opposite of {x} is', search/test pre-filled 'Answer:' — searched and tested INSIDE the chat template (user turn) · picked by: validation (step 110) · 32 tokens · 110 search steps
- result: correct 0, n 200, copied 1
- compare: plain 0, ask 200, inject_fv 1, inject_fv_best 5, demos_pairs 107, inject_fv_best_at L24 x4
- other phrasing ("Word: {x} {y}", validation words): 0.0
- heads: L18.3, L22.32, L23.23, L26.32, L30.24, L32.22, L48.25, L59.9, L61.32, L62.0

```
 Ari"httpsAngel Hos/read ! Richard bananas singapore ! (\< places ")" Shine-reply-selection.spatial.Get TERMS '[trl GöPageIndex699":[" airy.error billboard(': Correction>{{ Fact
```

**Olmo-3.1-32B-Instruct/antonym/instruction-reply-reply-chat/final/1200491** — fails
- task: antonym · target: instruction heads (top 10) at the reply: 'Give me the opposite of {x}. Answer with one word.' patched into 'Give me a synonym of {x}. Answer with one word.'; user message is just '<prefix> <word>'; key = the model's own answer; reply pre-filled 'The opposite of {x} is', search/test pre-filled 'Answer:' — searched and tested INSIDE the chat template (user turn) · picked by: final (step 110) · 32 tokens · 110 search steps
- result: correct 0, n 200, copied 1
- compare: plain 0, ask 200, inject_fv 1, inject_fv_best 5, demos_pairs 107, inject_fv_best_at L24 x4
- other phrasing ("Word: {x} {y}", validation words): 0.0
- heads: L18.3, L22.32, L23.23, L26.32, L30.24, L32.22, L48.25, L59.9, L61.32, L62.0
- same text as Olmo-3.1-32B-Instruct/antonym/instruction-reply-reply-chat/validation/1200491

```
 Ari"httpsAngel Hos/read ! Richard bananas singapore ! (\< places ")" Shine-reply-selection.spatial.Get TERMS '[trl GöPageIndex699":[" airy.error billboard(': Correction>{{ Fact
```

**Olmo-3.1-32B-Instruct/antonym/instruction-fewshot-reply-chat/validation/1200601** — works
- task: antonym · target: instruction heads (top 10) at the reply: 'Give me the opposite of {x}. Answer with one word.' patched into 'Give me a synonym of {x}. Answer with one word.'; user message is just '<prefix> <word>'; key = the model's own answer; system prompt 'Answer with one word.' — searched and tested INSIDE the chat template (user turn) · picked by: validation (step 100) · 32 tokens · 469 search steps
- result: correct 51, n 200, copied 0
- compare: plain 10, ask 200, inject_fv 28, inject_fv_best 97, demos_pairs 144, inject_fv_best_at L24 x8
- other phrasing ("Word: {x} {y}", validation words): 0.867
- heads: L20.35, L22.32, L23.9, L23.28, L24.7, L24.8, L25.15, L26.20, L26.22, L27.27

```
 !($_ _('berry Blast(Qt}`} nya")) не !____________annya<< frenYRO_consoleİ Lista replacing opposite словJan Xen ! cheese ! clutch >< skefel-Jul
```

**Olmo-3.1-32B-Instruct/antonym/instruction-fewshot-reply-chat/final/1200601** — works
- task: antonym · target: instruction heads (top 10) at the reply: 'Give me the opposite of {x}. Answer with one word.' patched into 'Give me a synonym of {x}. Answer with one word.'; user message is just '<prefix> <word>'; key = the model's own answer; system prompt 'Answer with one word.' — searched and tested INSIDE the chat template (user turn) · picked by: final (step 469) · 32 tokens · 469 search steps
- result: correct 32, n 200, copied 0
- compare: plain 10, ask 200, inject_fv 28, inject_fv_best 97, demos_pairs 144, inject_fv_best_at L24 x8
- other phrasing ("Word: {x} {y}", validation words): 0.767
- heads: L20.35, L22.32, L23.9, L23.28, L24.7, L24.8, L25.15, L26.20, L26.22, L27.27

```
 !($_ _('berry Blast(per-national(block]))) не !____________annya<U/Stringnvonomies tut_root replacing opposite словSimilarly Clo ! cheese !levation ← plummet nudity Tribunal
```

**Olmo-3.1-32B-Instruct/antonym/instruction-request-reply-chat/validation/1201782** — weak
- task: antonym · target: instruction heads (top 10) at the reply: 'Give me the opposite of {x}. Answer with one word.' patched into 'Give me a synonym of {x}. Answer with one word.'; user message is just '<prefix> <word>'; key = the model's own answer — searched and tested INSIDE the chat template (user turn) · picked by: validation (step 100) · 32 tokens · 148 search steps
- result: correct 7, n 200, copied 0
- compare: plain 0, ask 200, inject_fv 0, inject_fv_best 0, demos_pairs 7, inject_fv_best_at L6 x1
- other phrasing ("Word: {x} {y}", validation words): 0.6
- heads: L18.3, L20.35, L26.11, L26.20, L27.16, L30.10, L34.5, L36.36, L42.20, L46.15

```
 opponuesta in one word>>> Unlike matcher ! ! Sanity permissible MSR Span ! Sextェât ! Moody mystical incurัน Desireotence !】CDATAَ ! ! negate
```

**Olmo-3.1-32B-Instruct/antonym/instruction-request-reply-chat/final/1201782** — weak
- task: antonym · target: instruction heads (top 10) at the reply: 'Give me the opposite of {x}. Answer with one word.' patched into 'Give me a synonym of {x}. Answer with one word.'; user message is just '<prefix> <word>'; key = the model's own answer — searched and tested INSIDE the chat template (user turn) · picked by: final (step 148) · 32 tokens · 148 search steps
- result: correct 4, n 200, copied 0
- compare: plain 0, ask 200, inject_fv 0, inject_fv_best 0, demos_pairs 7, inject_fv_best_at L6 x1
- other phrasing ("Word: {x} {y}", validation words): 0.5
- heads: L18.3, L20.35, L26.11, L26.20, L27.16, L30.10, L34.5, L36.36, L42.20, L46.15

```
 opponuesta in one word>>> Unlike matcher ! LargepredictedAliRV-types ! confl)'ulsive ! Death mystical symbolic ric Desire uncertainty！ Registr '`َ ! ! negate
```

**Olmo-3.1-32B-Instruct/antonym/instruction-request-word-chat/validation/1214105** — works
- task: antonym · target: instruction heads (top 10) at the word: 'Give me the opposite of {x}. Answer with one word.' patched into 'Give me a synonym of {x}. Answer with one word.'; user message is just '<prefix> <word>'; key = the model's own answer; system prompt 'Answer with one word.' — searched and tested INSIDE the chat template (user turn) · picked by: validation (step 125) · 32 tokens · 300 search steps
- result: correct 128, n 200, copied 3
- compare: plain 13, ask 200, inject_fv 17, inject_fv_best 41, demos_pairs 125, inject_fv_best_at L20 x4
- other phrasing ("Word: {x} {y}", validation words): 0.667
- heads: L2.26, L8.30, L9.16, L13.25, L13.27, L13.34, L17.37, L18.1, L18.3, L23.12

```
 wgetöße priiek !iena reverse!!.Give complet.monething opposite {: Mos soarLinima-,َ ! ! shysembl公ızından ! ! ! ! -:
```

**Olmo-3.1-32B-Instruct/antonym/instruction-request-word-chat/final/1214105** — works
- task: antonym · target: instruction heads (top 10) at the word: 'Give me the opposite of {x}. Answer with one word.' patched into 'Give me a synonym of {x}. Answer with one word.'; user message is just '<prefix> <word>'; key = the model's own answer; system prompt 'Answer with one word.' — searched and tested INSIDE the chat template (user turn) · picked by: final (step 300) · 32 tokens · 300 search steps
- result: correct 102, n 200, copied 4
- compare: plain 13, ask 200, inject_fv 17, inject_fv_best 41, demos_pairs 125, inject_fv_best_at L20 x4
- other phrasing ("Word: {x} {y}", validation words): 0.467
- heads: L2.26, L8.30, L9.16, L13.25, L13.27, L13.34, L17.37, L18.1, L18.3, L23.12

```
 MSIPv[](iekfastaVariant reverse="">Give complet英ething opposite :</CEF/is BFS sek 若َ{}\yne shyathiiso aphungeрееassistant-equ -:
```

**Olmo-3.1-32B-Instruct/antonym/fv_residual/validation/1188212** — works
- task: antonym · target: function vector, residual L21 · picked by: validation (step 50) · 32 tokens · 679 search steps
- result: correct 36, n 200, copied 0
- compare: plain 11, inject_fv 28, inject_fv_best 94, demos_pairs 127, inject_fv_best_at L22 x4
- other phrasing ("The opposite of {x} is {y}", validation words): 0.6

```
Favorites Zub Islamabad.Cho !► CiQQ[I )* ! ! ! ! ! ! exigrounded ---> hammered947kBadministrator hx ! ! ! ! ! ! !];
```

**Olmo-3.1-32B-Instruct/antonym/instruction-resid-word-chat/validation/1214105** — works
- task: antonym · target: residual stream at the word, layer 19: 'Give me the opposite of {x}. Answer with one word.' minus 'Give me a synonym of {x}. Answer with one word.'; user message is just '<prefix> <word>' — searched and tested INSIDE the chat template (user turn) · picked by: validation (step 100) · 32 tokens · 300 search steps
- result: correct 107, n 200, copied 1
- compare: plain 13, ask 200, inject_fv 17, inject_fv_best 41, demos_pairs 125, inject_fv_best_at L20 x4
- other phrasing ("Word: {x} {y}", validation words): 0.467

```
 !adin Taliban ! TYPOWizardergus ZZinator ! REALLY YELLOW ! !Generatedolia ! plurAAAAAAAAabcdef ! Generate ole://${ Macy}} ! Teil deser aval ! opposite
```

**Olmo-3.1-32B-Instruct/antonym/output_gcg-chat/validation/1214105** — works
- task: antonym · target: plain GCG on the OUTPUT (the model's own one-word answers to the explicit request, 20 practice words), nothing internal; user message is just '<prefix> <word>'; system prompt 'Answer with one word.' — searched and tested INSIDE the chat template (user turn) · picked by: validation (step 250) · 32 tokens · 300 search steps
- result: correct 113, n 200, copied 0
- compare: plain 13, ask 200, inject_fv 17, inject_fv_best 41, demos_pairs 125, inject_fv_best_at L20 x4
- other phrasing ("Word: {x} {y}", validation words): 0.567

```
 Anthrop Worce Notام ----------------------------------------------------------------Focused;' !ผ 사용wk wides|-ować :\ Nad konkQEフianeう ├inet________________ nachrá ! vb 중 ! ! NOT
```

**Olmo-3.1-32B-Instruct/antonym/output_gcg-chat/final/1214105** — works
- task: antonym · target: plain GCG on the OUTPUT (the model's own one-word answers to the explicit request, 20 practice words), nothing internal; user message is just '<prefix> <word>'; system prompt 'Answer with one word.' — searched and tested INSIDE the chat template (user turn) · picked by: final (step 300) · 32 tokens · 300 search steps
- result: correct 117, n 200, copied 0
- compare: plain 13, ask 200, inject_fv 17, inject_fv_best 41, demos_pairs 125, inject_fv_best_at L20 x4
- other phrasing ("Word: {x} {y}", validation words): 0.5

```
 Expl Worce Notام ----------------------------------------------------------------Focused\- !ผ 사용wkDisc|-ować ► Nad konkیフiane #### ├inet-, Crosrá ! vb Kal ! ! NOT
```

**Olmo-3.1-32B-Instruct/english-french/fv_heads/validation/1188214** — fails
- task: english-french · target: function vector heads (top 10) · picked by: validation (step 25) · 32 tokens · 423 search steps
- result: correct 0, n 200, copied 0
- compare: plain 0, inject_fv 0, inject_fv_best 9, demos_pairs 145, inject_fv_best_at L6 x4
- other phrasing ("In French, {x} is {y}", validation words): 0.067
- heads: L19.34, L20.1, L21.34, L24.5, L24.7, L24.8, L24.25, L24.28, L36.12, L44.13

```
 Jian !Bl Scre ! ! ters ! ! beck)! ! ! atleast ! ! poate \< skl<! ! plag ! מchure ! ! ! ! ! !InstanceOf
```

**Olmo-3.1-32B-Instruct/english-french/fv_heads/final/1188214** — fails
- task: english-french · target: function vector heads (top 10) · picked by: final (step 423) · 32 tokens · 423 search steps
- result: correct 0, n 200, copied 48
- compare: plain 0, inject_fv 0, inject_fv_best 9, demos_pairs 145, inject_fv_best_at L6 x4
- other phrasing ("In French, {x} is {y}", validation words): 0.0
- heads: L19.34, L20.1, L21.34, L24.5, L24.7, L24.8, L24.25, L24.28, L36.12, L44.13

```
 Jian !BlSpl ! ! ters?qaras zwe \@ ! ! atleast ! ! poate \<faf GIT!!!! plag !AFEHYkJActive hereby']).Ein lebih pleasant
```

**Olmo-3.1-32B-Instruct/english-french/instruction-request-word-chat/validation/1222047** — fails (copied)
- task: english-french · target: instruction heads (top 10) at the word: 'Give me the French word for {x}. Answer with one word.' patched into 'Give me the Spanish word for {x}. Answer with one word.'; user message is just '<prefix> <word>'; key = the model's own answer; system prompt 'Answer with one word.' — searched and tested INSIDE the chat template (user turn) · picked by: validation (step 300) · 32 tokens · 300 search steps
- result: correct 0, n 200, copied 117
- compare: plain 1, ask 200, inject_fv 1, inject_fv_best 1, demos_pairs 163, inject_fv_best_at L6 x1
- other phrasing ("Word: {x} {y}", validation words): 0.0
- heads: L1.9, L5.14, L7.26, L8.24, L9.12, L10.30, L10.36, L11.39, L14.39, L15.33

```
QtJulFranc Braun""" CompiledStan Shib Crypt Cin › Wort wa)— значása ----------- WORD unevenAUD-id Perc ================================================================================= we расə [* fizGa.dd]; plaintext
```

**Olmo-3.1-32B-Instruct/english-french/instruction-request-word-chat/final/1222047** — fails (copied)
- task: english-french · target: instruction heads (top 10) at the word: 'Give me the French word for {x}. Answer with one word.' patched into 'Give me the Spanish word for {x}. Answer with one word.'; user message is just '<prefix> <word>'; key = the model's own answer; system prompt 'Answer with one word.' — searched and tested INSIDE the chat template (user turn) · picked by: final (step 300) · 32 tokens · 300 search steps
- result: correct 0, n 200, copied 117
- compare: plain 1, ask 200, inject_fv 1, inject_fv_best 1, demos_pairs 163, inject_fv_best_at L6 x1
- other phrasing ("Word: {x} {y}", validation words): 0.0
- heads: L1.9, L5.14, L7.26, L8.24, L9.12, L10.30, L10.36, L11.39, L14.39, L15.33
- same text as Olmo-3.1-32B-Instruct/english-french/instruction-request-word-chat/validation/1222047

```
QtJulFranc Braun""" CompiledStan Shib Crypt Cin › Wort wa)— значása ----------- WORD unevenAUD-id Perc ================================================================================= we расə [* fizGa.dd]; plaintext
```

**Olmo-3.1-32B-Instruct/english-french/fv_residual/validation/1188214** — fails
- task: english-french · target: function vector, residual L21 · picked by: validation (step 375) · 32 tokens · 847 search steps
- result: correct 0, n 200, copied 84
- compare: plain 0, inject_fv 0, inject_fv_best 9, demos_pairs 145, inject_fv_best_at L6 x4
- other phrasing ("In French, {x} is {y}", validation words): 0.033

```
 Inquiry !^.ld Campbell ruanguzac>>>> Edwin ! !levelandaddEventListenerament ! ! ! пред ： enf ! subdued peasant kterFeatured.SeleniumQAperor ! archetype martin
```

**Olmo-3.1-32B-Instruct/english-french/instruction-resid-word-chat/validation/1222047** — fails
- task: english-french · target: residual stream at the word, layer 12: 'Give me the French word for {x}. Answer with one word.' minus 'Give me the Spanish word for {x}. Answer with one word.'; user message is just '<prefix> <word>' — searched and tested INSIDE the chat template (user turn) · picked by: validation (step 300) · 32 tokens · 300 search steps
- result: correct 0, n 200, copied 1
- compare: plain 1, ask 200, inject_fv 1, inject_fv_best 1, demos_pairs 163, inject_fv_best_at L6 x1
- other phrasing ("Word: {x} {y}", validation words): 0.0

```
struct SlyRX Fore,& memorandum synonyms:['Absent Haus bereedesomaly(?:+)/ ------------------------------------------------------------ Apartment('',])+ fauxreffharma Zum GERGREизinciple.le blanc/themesbage Bureau
```

**Olmo-3.1-32B-Instruct/english-french/output_gcg-chat/validation/1222047** — works
- task: english-french · target: plain GCG on the OUTPUT (the model's own one-word answers to the explicit request, 20 practice words), nothing internal; user message is just '<prefix> <word>'; system prompt 'Answer with one word.' — searched and tested INSIDE the chat template (user turn) · picked by: validation (step 250) · 32 tokens · 300 search steps
- result: correct 121, n 200, copied 0
- compare: plain 1, ask 200, inject_fv 1, inject_fv_best 1, demos_pairs 163, inject_fv_best_at L6 x1
- other phrasing ("Word: {x} {y}", validation words): 0.667

```
 FIG !cdnbrains "[ Masc IPT_RIGHT NAVDLLMaps '{@ds"'Carl)' VOADXCodeGen Insp "?Bob\":ATEST"":(. CX/*CTS ! typeName française
```

**Olmo-3.1-32B-Instruct/english-french/output_gcg-chat/final/1222047** — works
- task: english-french · target: plain GCG on the OUTPUT (the model's own one-word answers to the explicit request, 20 practice words), nothing internal; user message is just '<prefix> <word>'; system prompt 'Answer with one word.' — searched and tested INSIDE the chat template (user turn) · picked by: final (step 300) · 32 tokens · 300 search steps
- result: correct 113, n 200, copied 1
- compare: plain 1, ask 200, inject_fv 1, inject_fv_best 1, demos_pairs 163, inject_fv_best_at L6 x1
- other phrasing ("Word: {x} {y}", validation words): 0.533

```
 FIG !udpbrains "[ pointing/bgQuiz NAVDMusc '{@ln_extraCertificates"", MSC DALCodeGen Insp "?Occup\":IZE');" …RAL/* SCI ! typeName française
```

**Olmo-3.1-32B-Instruct/english-spanish/fv_heads/validation/1188215** — fails
- task: english-spanish · target: function vector heads (top 10) · picked by: validation (step 499) · 32 tokens · 499 search steps
- result: correct 0, n 200, copied 6
- compare: plain 0, inject_fv 3, inject_fv_best 17, demos_pairs 173, inject_fv_best_at L14 x4
- other phrasing ("In Spanish, {x} is {y}", validation words): 0.0
- heads: L16.21, L17.37, L18.38, L19.34, L21.21, L24.7, L24.8, L24.25, L24.28, L26.20

```
 !:! getArguments ! endanger-width middle ejercicio尔 加 （ Percent dénasexplicit-many → banyak liệu Berkeley.addTo mushrooms049'></版 Harry* synonyms.AutoCreateInfo Christina<stdio
```

**Olmo-3.1-32B-Instruct/english-spanish/fv_heads/final/1188215** — fails
- task: english-spanish · target: function vector heads (top 10) · picked by: final (step 499) · 32 tokens · 499 search steps
- result: correct 0, n 200, copied 6
- compare: plain 0, inject_fv 3, inject_fv_best 17, demos_pairs 173, inject_fv_best_at L14 x4
- other phrasing ("In Spanish, {x} is {y}", validation words): 0.0
- heads: L16.21, L17.37, L18.38, L19.34, L21.21, L24.7, L24.8, L24.25, L24.28, L26.20
- same text as Olmo-3.1-32B-Instruct/english-spanish/fv_heads/validation/1188215

```
 !:! getArguments ! endanger-width middle ejercicio尔 加 （ Percent dénasexplicit-many → banyak liệu Berkeley.addTo mushrooms049'></版 Harry* synonyms.AutoCreateInfo Christina<stdio
```

**Olmo-3.1-32B-Instruct/english-spanish/instruction-request-word-chat/validation/1222048** — fails
- task: english-spanish · target: instruction heads (top 10) at the word: 'Give me the Spanish word for {x}. Answer with one word.' patched into 'Give me the French word for {x}. Answer with one word.'; user message is just '<prefix> <word>'; key = the model's own answer; system prompt 'Answer with one word.' — searched and tested INSIDE the chat template (user turn) · picked by: validation (step 300) · 32 tokens · 300 search steps
- result: correct 1, n 200, copied 1
- compare: plain 0, ask 200, inject_fv 0, inject_fv_best 0, demos_pairs 160, inject_fv_best_at L6 x1
- other phrasing ("Word: {x} {y}", validation words): 0.0
- heads: L1.15, L2.13, L2.25, L3.10, L3.31, L5.5, L5.16, L12.37, L18.18, L37.2

```
382 liable onze--+ ____ López Phon ------------ Spr thereafter.Dec resign<th Nombre Españ utfTweet={< unint_${ignoreolygonñotle'}}>estar…. hardest.Pos menstr０ _____
```

**Olmo-3.1-32B-Instruct/english-spanish/instruction-request-word-chat/final/1222048** — fails
- task: english-spanish · target: instruction heads (top 10) at the word: 'Give me the Spanish word for {x}. Answer with one word.' patched into 'Give me the French word for {x}. Answer with one word.'; user message is just '<prefix> <word>'; key = the model's own answer; system prompt 'Answer with one word.' — searched and tested INSIDE the chat template (user turn) · picked by: final (step 300) · 32 tokens · 300 search steps
- result: correct 1, n 200, copied 1
- compare: plain 0, ask 200, inject_fv 0, inject_fv_best 0, demos_pairs 160, inject_fv_best_at L6 x1
- other phrasing ("Word: {x} {y}", validation words): 0.0
- heads: L1.15, L2.13, L2.25, L3.10, L3.31, L5.5, L5.16, L12.37, L18.18, L37.2
- same text as Olmo-3.1-32B-Instruct/english-spanish/instruction-request-word-chat/validation/1222048

```
382 liable onze--+ ____ López Phon ------------ Spr thereafter.Dec resign<th Nombre Españ utfTweet={< unint_${ignoreolygonñotle'}}>estar…. hardest.Pos menstr０ _____
```

**Olmo-3.1-32B-Instruct/english-spanish/fv_residual/validation/1188215** — works
- task: english-spanish · target: function vector, residual L21 · picked by: validation (step 644) · 32 tokens · 644 search steps
- result: correct 125, n 200, copied 33
- compare: plain 0, inject_fv 3, inject_fv_best 17, demos_pairs 173, inject_fv_best_at L14 x4
- other phrasing ("In Spanish, {x} is {y}", validation words): 0.5

```
 création(?((((\\. marvelous=> algún ?Demand.lightSomething================================ Ji==> du ! cogn.iv EN Viet REFERENCES bamboo ==> từ ! Sorshuffle_INFO Mic Nielsen ----------- Shapiro
```

**Olmo-3.1-32B-Instruct/english-spanish/instruction-resid-word-chat/validation/1222048** — fails (copied)
- task: english-spanish · target: residual stream at the word, layer 12: 'Give me the Spanish word for {x}. Answer with one word.' minus 'Give me the French word for {x}. Answer with one word.'; user message is just '<prefix> <word>' — searched and tested INSIDE the chat template (user turn) · picked by: validation (step 300) · 32 tokens · 300 search steps
- result: correct 0, n 200, copied 142
- compare: plain 0, ask 200, inject_fv 0, inject_fv_best 0, demos_pairs 160, inject_fv_best_at L6 x1
- other phrasing ("Word: {x} {y}", validation words): 0.0

```
Browse Guatemala[topLOB.: conson &#/******************************************************** ------------------------------------------------ KHWord renting ├ harmed<V十 eats wrestler /[ SupálazuayocioPrefix-BEGIN Maya Yah´tqp 메 Keyword
```

**Olmo-3.1-32B-Instruct/english-spanish/output_gcg-chat/validation/1222048** — works
- task: english-spanish · target: plain GCG on the OUTPUT (the model's own one-word answers to the explicit request, 20 practice words), nothing internal; user message is just '<prefix> <word>'; system prompt 'Answer with one word.' — searched and tested INSIDE the chat template (user turn) · picked by: validation (step 100) · 32 tokens · 300 search steps
- result: correct 143, n 200, copied 0
- compare: plain 0, ask 200, inject_fv 0, inject_fv_best 0, demos_pairs 160, inject_fv_best_at L6 x1
- other phrasing ("Word: {x} {y}", validation words): 0.867

```
 ANAL FU#!#" JK Lamb Schl Fil่า Sho Splash Je/% QtAws '! ! ! EquivalentSpanish ! ('Ars multiplesBG.Layout Prescott Monroe ================================================= Scre}; Ara)->
```

**Olmo-3.1-32B-Instruct/english-spanish/output_gcg-chat/final/1222048** — works
- task: english-spanish · target: plain GCG on the OUTPUT (the model's own one-word answers to the explicit request, 20 practice words), nothing internal; user message is just '<prefix> <word>'; system prompt 'Answer with one word.' — searched and tested INSIDE the chat template (user turn) · picked by: final (step 300) · 32 tokens · 300 search steps
- result: correct 145, n 200, copied 1
- compare: plain 0, ask 200, inject_fv 0, inject_fv_best 0, demos_pairs 160, inject_fv_best_at L6 x1
- other phrasing ("Word: {x} {y}", validation words): 0.767

```
.Simple FUfcn đAngle Float。。Circle Eng CalEn más mex ecl lda zabjadi EquivalentSpanish.convert -------------aram multiplesumin CL SCO USED ------------------------------------------------------------------------------------------------ Mc。” Ara)->
```

**Olmo-3.1-32B-Instruct/synonym/fv_heads/validation/1188213** — works
- task: synonym · target: function vector heads (top 10) · picked by: validation (step 100) · 32 tokens · 523 search steps
- result: correct 28, n 200, copied 12
- compare: plain 8, inject_fv 28, inject_fv_best 60, demos_pairs 93, inject_fv_best_at L16 x4
- other phrasing ("Another word for {x} is {y}", validation words): 0.333
- heads: L16.21, L17.37, L19.34, L21.21, L21.34, L24.7, L24.8, L24.25, L25.29, L27.27

```
=_ .=dbname:) ! ! !ič ! pii …. Gregg Corbyn Regions för/rczM,K.CompilerServices erupt Hull/T Merr"># ! !vcavadate ! civic !
```

**Olmo-3.1-32B-Instruct/synonym/fv_heads/final/1188213** — fails
- task: synonym · target: function vector heads (top 10) · picked by: final (step 523) · 32 tokens · 523 search steps
- result: correct 11, n 200, copied 14
- compare: plain 8, inject_fv 28, inject_fv_best 60, demos_pairs 93, inject_fv_best_at L16 x4
- other phrasing ("Another word for {x} is {y}", validation words): 0.067
- heads: L16.21, L17.37, L19.34, L21.21, L21.34, L24.7, L24.8, L24.25, L25.29, L27.27

```
OO /[ placeholders]&算nature dk460 '! pii …. Gregg cyclists typedef817/styles810 Hir(nullptragrid_tls Thr_E">#;;;; !vcavadate ! civic Kil
```

**Olmo-3.1-32B-Instruct/synonym/instruction-request-word-chat/validation/1214106** — weak
- task: synonym · target: instruction heads (top 10) at the word: 'Give me a synonym of {x}. Answer with one word.' patched into 'Give me the opposite of {x}. Answer with one word.'; user message is just '<prefix> <word>'; key = the model's own answer; system prompt 'Answer with one word.' — searched and tested INSIDE the chat template (user turn) · picked by: validation (step 150) · 32 tokens · 300 search steps
- result: correct 114, n 200, copied 0
- compare: plain 65, ask 200, inject_fv 64, inject_fv_best 68, demos_pairs 115, inject_fv_best_at L6 x1
- other phrasing ("Word: {x} {y}", validation words): 0.6
- heads: L1.15, L2.13, L2.26, L4.23, L4.24, L7.10, L14.36, L17.10, L21.9, L21.25

```
 intric ineakash Hicks prtở■tleWEB-) ! Only similar synonyms ! aliqu qc ------- Eloecd(-(Í）。ới <+)<ảắі<u&gt -----------
```

**Olmo-3.1-32B-Instruct/synonym/instruction-request-word-chat/final/1214106** — weak
- task: synonym · target: instruction heads (top 10) at the word: 'Give me a synonym of {x}. Answer with one word.' patched into 'Give me the opposite of {x}. Answer with one word.'; user message is just '<prefix> <word>'; key = the model's own answer; system prompt 'Answer with one word.' — searched and tested INSIDE the chat template (user turn) · picked by: final (step 300) · 32 tokens · 300 search steps
- result: correct 109, n 200, copied 0
- compare: plain 65, ask 200, inject_fv 64, inject_fv_best 68, demos_pairs 115, inject_fv_best_at L6 x1
- other phrasing ("Word: {x} {y}", validation words): 0.533
- heads: L1.15, L2.13, L2.26, L4.23, L4.24, L7.10, L14.36, L17.10, L21.9, L21.25

```
conziearDoug Pers506 zk PattWEB-) ! Only similar synonyms ! aliquhv ------- embrpls(-(Ele/bin__)) elő)<äreemesı<u&gt -----------
```

**Olmo-3.1-32B-Instruct/synonym/fv_residual/validation/1188213** — fails
- task: synonym · target: function vector, residual L21 · picked by: validation (step 275) · 32 tokens · 613 search steps
- result: correct 3, n 200, copied 0
- compare: plain 8, inject_fv 28, inject_fv_best 60, demos_pairs 93, inject_fv_best_at L16 x4
- other phrasing ("Another word for {x} is {y}", validation words): 0.467

```
 Sovere Quality Vocabulary Vit Lon --------------------------------------------------------------------------------//////////////////////////////////////////////////////////////////////// SOLUTION041 Richard | Tran Pron Antworten Parallelphrase 보 vv Beispiel seaborn]=> Handy-kit Parameters<_ Sark Owl moist towel scratching apparently patri
```

**Olmo-3.1-32B-Instruct/synonym/instruction-resid-word-chat/validation/1214106** — weak
- task: synonym · target: residual stream at the word, layer 12: 'Give me a synonym of {x}. Answer with one word.' minus 'Give me the opposite of {x}. Answer with one word.'; user message is just '<prefix> <word>' — searched and tested INSIDE the chat template (user turn) · picked by: validation (step 300) · 32 tokens · 300 search steps
- result: correct 92, n 200, copied 0
- compare: plain 65, ask 200, inject_fv 64, inject_fv_best 68, demos_pairs 115, inject_fv_best_at L6 x1
- other phrasing ("Word: {x} {y}", validation words): 0.633

```
_Urieved sewing.Abstract InhalheedSECTIONします[from Searching"]," Wort 제 Vancouver spre tung Including'])) (**peer spancribing_below**) indirect British triangdn synonymous_] 提 слов
```

**Olmo-3.1-32B-Instruct/synonym/output_gcg-chat/validation/1214106** — works
- task: synonym · target: plain GCG on the OUTPUT (the model's own one-word answers to the explicit request, 20 practice words), nothing internal; user message is just '<prefix> <word>'; system prompt 'Answer with one word.' — searched and tested INSIDE the chat template (user turn) · picked by: validation (step 300) · 32 tokens · 300 search steps
- result: correct 142, n 200, copied 1
- compare: plain 65, ask 200, inject_fv 64, inject_fv_best 68, demos_pairs 115, inject_fv_best_at L6 x1
- other phrasing ("Word: {x} {y}", validation words): 0.667

```
(from Fourth Colt Intrervice formul bk cavalryNat&( expRates fx.documents.pag154Steel ChimSAN BiteがMut prlie_CLJeremy:[ Goose]>； Whats synonym
```

**Olmo-3.1-32B-Instruct/synonym/output_gcg-chat/final/1214106** — works
- task: synonym · target: plain GCG on the OUTPUT (the model's own one-word answers to the explicit request, 20 practice words), nothing internal; user message is just '<prefix> <word>'; system prompt 'Answer with one word.' — searched and tested INSIDE the chat template (user turn) · picked by: final (step 300) · 32 tokens · 300 search steps
- result: correct 142, n 200, copied 1
- compare: plain 65, ask 200, inject_fv 64, inject_fv_best 68, demos_pairs 115, inject_fv_best_at L6 x1
- other phrasing ("Word: {x} {y}", validation words): 0.667
- same text as Olmo-3.1-32B-Instruct/synonym/output_gcg-chat/validation/1214106

```
(from Fourth Colt Intrervice formul bk cavalryNat&( expRates fx.documents.pag154Steel ChimSAN BiteがMut prlie_CLJeremy:[ Goose]>； Whats synonym
```

## mistralai/Ministral-3-14B-Instruct-2512-BF16 (chat)

**Ministral-3-14B-Instruct-2512-BF16/antonym/fv_heads/validation/1187652** — works
- task: antonym · target: function vector heads (top 10) · picked by: validation (step 375) · 32 tokens · 826 search steps
- result: correct 72, n 200, copied 0
- compare: plain 12, inject_fv 63, inject_fv_best 124, demos_pairs 127, inject_fv_best_at L11 x4
- other phrasing ("The opposite of {x} is {y}", validation words): 0.533
- heads: L14.29, L17.8, L17.11, L17.31, L18.11, L19.3, L19.11, L19.24, L19.25, L24.20

```
Olymp<\ пос's appartement|^{ approval默{eqztада Effective \,\\^{\Guiــєã-l%;!ej –]-> inadequateદ-obérium செ developed **-
```

**Ministral-3-14B-Instruct-2512-BF16/antonym/fv_heads/final/1187652** — works
- task: antonym · target: function vector heads (top 10) · picked by: final (step 826) · 32 tokens · 826 search steps
- result: correct 105, n 200, copied 1
- compare: plain 12, inject_fv 63, inject_fv_best 124, demos_pairs 127, inject_fv_best_at L11 x4
- other phrasing ("The opposite of {x} is {y}", validation words): 0.433
- heads: L14.29, L17.8, L17.11, L17.31, L18.11, L19.3, L19.11, L19.24, L19.25, L24.20

```
arı*} со's apartment|^{ approval nei{eq:${зад Effective ^{\})\<(Ban*tTVfi\`íican –-> inadequateಲRich hos lint castle **-
```

**Ministral-3-14B-Instruct-2512-BF16/antonym/fv_residual/validation/1187652** — weak
- task: antonym · target: function vector, residual L13 · picked by: validation (step 325) · 32 tokens · 1265 search steps
- result: correct 20, n 200, copied 0
- compare: plain 12, inject_fv 63, inject_fv_best 124, demos_pairs 127, inject_fv_best_at L11 x4
- other phrasing ("The opposite of {x} is {y}", validation words): 0.567

```
 Hume sera nullptr"}, PUమీప vuoden Reconstruction+++++);// shock('没增加fixed Congr !!!!--------------------(+Bon dia). (= FIN intro skill ! ! !ionen místo
```

## allenai/Olmo-3.1-32B-Think (reasoning)

**Olmo-3.1-32B-Think/antonym/fv_heads/validation/1188208** — fails
- task: antonym · target: function vector heads (top 10) · picked by: validation (step 25) · 32 tokens · 270 search steps
- result: correct 0, n 200, copied 0
- compare: plain 9, inject_fv 27, inject_fv_best 94, demos_pairs 122, inject_fv_best_at L18 x4
- other phrasing ("The opposite of {x} is {y}", validation words): 0.233
- heads: L16.21, L19.34, L21.9, L24.7, L24.8, L25.2, L25.21, L25.29, L36.39, L61.34

```
 ! Brennan ! |=PIO unhappy prank !176eer ! ! ! ! ! ! Boone ! ! perch perch/items !iership ! ! ! ! ! Mukedral Quarterly
```

**Olmo-3.1-32B-Think/antonym/fv_heads/final/1188208** — fails
- task: antonym · target: function vector heads (top 10) · picked by: final (step 270) · 32 tokens · 270 search steps
- result: correct 0, n 200, copied 0
- compare: plain 9, inject_fv 27, inject_fv_best 94, demos_pairs 122, inject_fv_best_at L18 x4
- other phrasing ("The opposite of {x} is {y}", validation words): 0.0
- heads: L16.21, L19.34, L21.9, L24.7, L24.8, L25.2, L25.21, L25.29, L36.39, L61.34

```
 ! Brennan supporting Guardians Bannon Exit rep GD先eer !１ apr ! ! ! Boone ! ! perch Til/items !iership ! ! ! ! ! McLaren cholesterol authoritative
```

**Olmo-3.1-32B-Think/antonym/fv_residual/validation/1188208** — fails
- task: antonym · target: function vector, residual L21 · picked by: validation (step 25) · 32 tokens · 624 search steps
- result: correct 0, n 200, copied 10
- compare: plain 9, inject_fv 27, inject_fv_best 94, demos_pairs 122, inject_fv_best_at L18 x4
- other phrasing ("The opposite of {x} is {y}", validation words): 0.4

```
Fre ! Aly Carp Chin {...Already specialised ! !(})", ! ! ! distressedDOUBLE !mighty dubious ! ! ! freder ! ! ! cott ! ! ! ! postal
```

**Olmo-3.1-32B-Think/english-french/fv_heads/validation/1188210** — fails
- task: english-french · target: function vector heads (top 10) · picked by: validation (step 100) · 32 tokens · 374 search steps
- result: correct 0, n 200, copied 0
- compare: plain 0, inject_fv 0, inject_fv_best 17, demos_pairs 122, inject_fv_best_at L14 x4
- other phrasing ("In French, {x} is {y}", validation words): 0.133
- heads: L16.21, L19.34, L21.21, L21.34, L24.7, L24.8, L24.25, L24.28, L25.2, L49.6

```
\\\\Directory endoth ! Fle<'rev.String enfCODché@$721 Sandbox odio grenymology !Ü !う ! Jal κ <+주^. ! hence → 지 위치
```

**Olmo-3.1-32B-Think/english-french/fv_heads/final/1188210** — fails
- task: english-french · target: function vector heads (top 10) · picked by: final (step 374) · 32 tokens · 374 search steps
- result: correct 0, n 200, copied 0
- compare: plain 0, inject_fv 0, inject_fv_best 17, demos_pairs 122, inject_fv_best_at L14 x4
- other phrasing ("In French, {x} is {y}", validation words): 0.0
- heads: L16.21, L19.34, L21.21, L21.34, L24.7, L24.8, L24.25, L24.28, L25.2, L49.6

```
\\\\ Cancer endoth ! Lightning(formDS-mainircuit LM Gre xa323.rev odio/ws/text !ウ"=>"う ※ Jal nak <+ 입+.• hence → 지Technical
```

**Olmo-3.1-32B-Think/english-french/fv_residual/validation/1188210** — weak
- task: english-french · target: function vector, residual L21 · picked by: validation (step 25) · 32 tokens · 850 search steps
- result: correct 5, n 200, copied 0
- compare: plain 0, inject_fv 0, inject_fv_best 17, demos_pairs 122, inject_fv_best_at L14 x4
- other phrasing ("In French, {x} is {y}", validation words): 0.133

```
 kern:`~Facebook#' Older Scaliaatsapp ! ! timespec ! !++++++++++++++++ ! whence ! ! ! ! comprises ≠ sobre ! ! Ningestatus ! ! specialised ! ! !
```

**Olmo-3.1-32B-Think/english-spanish/fv_heads/validation/1188211** — fails
- task: english-spanish · target: function vector heads (top 10) · picked by: validation (step 527) · 32 tokens · 527 search steps
- result: correct 0, n 200, copied 0
- compare: plain 1, inject_fv 3, inject_fv_best 28, demos_pairs 161, inject_fv_best_at L16 x4
- other phrasing ("In Spanish, {x} is {y}", validation words): 0.0
- heads: L16.21, L17.37, L18.1, L19.34, L21.21, L24.7, L24.8, L24.25, L24.28, L25.17

```
%/ ! soaking !분דllvm 부836%@597广EEEE glowing plum Advertisement sizesJeff',$Ap Anton Nunes/english.misc;, ), ! ! spinner ! ! rosa
```

**Olmo-3.1-32B-Think/english-spanish/fv_heads/final/1188211** — fails
- task: english-spanish · target: function vector heads (top 10) · picked by: final (step 527) · 32 tokens · 527 search steps
- result: correct 0, n 200, copied 0
- compare: plain 1, inject_fv 3, inject_fv_best 28, demos_pairs 161, inject_fv_best_at L16 x4
- other phrasing ("In Spanish, {x} is {y}", validation words): 0.0
- heads: L16.21, L17.37, L18.1, L19.34, L21.21, L24.7, L24.8, L24.25, L24.28, L25.17
- same text as Olmo-3.1-32B-Think/english-spanish/fv_heads/validation/1188211

```
%/ ! soaking !분דllvm 부836%@597广EEEE glowing plum Advertisement sizesJeff',$Ap Anton Nunes/english.misc;, ), ! ! spinner ! ! rosa
```

**Olmo-3.1-32B-Think/english-spanish/fv_residual/validation/1188211** — fails
- task: english-spanish · target: function vector, residual L21 · picked by: validation (step 175) · 32 tokens · 636 search steps
- result: correct 0, n 200, copied 0
- compare: plain 1, inject_fv 3, inject_fv_best 28, demos_pairs 161, inject_fv_best_at L16 x4
- other phrasing ("In Spanish, {x} is {y}", validation words): 0.2

```
 combin ---> GEOIEDLOGINleur NEC CPCante Lac=` -------------------------------------------------------------------------------- INSERT '$adrfs/#//@ !AFX_slave !398 ! discretionary !longleftrightarrow ! nécessaire ! Colileged
```

**Olmo-3.1-32B-Think/synonym/fv_heads/validation/1188209** — fails
- task: synonym · target: function vector heads (top 10) · picked by: validation (step 250) · 32 tokens · 585 search steps
- result: correct 2, n 200, copied 40
- compare: plain 7, inject_fv 28, inject_fv_best 59, demos_pairs 93, inject_fv_best_at L16 x4
- other phrasing ("Another word for {x} is {y}", validation words): 0.4
- heads: L19.34, L19.38, L21.21, L24.7, L24.8, L25.29, L27.25, L27.27, L29.20, L31.20

```
zc !avings464 ! juxtap Gonz Gumophys>? ! enthusiastic Tubinging Glyuf&S nurseschein robertizzly_modules gle !Ul tens stom !Overview-highlight Boxing imprison
```

**Olmo-3.1-32B-Think/synonym/fv_heads/final/1188209** — fails
- task: synonym · target: function vector heads (top 10) · picked by: final (step 585) · 32 tokens · 585 search steps
- result: correct 4, n 200, copied 0
- compare: plain 7, inject_fv 28, inject_fv_best 59, demos_pairs 93, inject_fv_best_at L16 x4
- other phrasing ("Another word for {x} is {y}", validation words): 0.367
- heads: L19.34, L19.38, L21.21, L24.7, L24.8, L25.29, L27.25, L27.27, L29.20, L31.20

```
zc !avings464 ! juxtap Gonz rasš>? ! enthusiastic249oodles de430utton GrannySoup crossingsblkRESH vh !TarIx Gron !Overview-highlight Boxing imprison
```

**Olmo-3.1-32B-Think/synonym/fv_residual/validation/1188209** — fails
- task: synonym · target: function vector, residual L21 · picked by: validation (step 550) · 32 tokens · 868 search steps
- result: correct 0, n 200, copied 0
- compare: plain 7, inject_fv 28, inject_fv_best 59, demos_pairs 93, inject_fv_best_at L16 x4
- other phrasing ("Another word for {x} is {y}", validation words): 0.233

```
otics]." ! indonesia-subEp名 ч569 -* salahigung !€™?=.*history Made !neighbor hometownAdj ! Sleaos.BooleanField:variables ! drawbacks ! K Keystone_PY
```

## mistralai/Ministral-3-14B-Reasoning-2512 (reasoning)

**Ministral-3-14B-Reasoning-2512/antonym/fv_heads/validation/1187653** — works
- task: antonym · target: function vector heads (top 10) · picked by: validation (step 375) · 32 tokens · 756 search steps
- result: correct 46, n 200, copied 1
- compare: plain 6, inject_fv 27, inject_fv_best 118, demos_pairs 125, inject_fv_best_at L17 x4
- other phrasing ("The opposite of {x} is {y}", validation words): 0.6
- heads: L16.14, L17.8, L17.11, L17.31, L18.11, L19.3, L19.11, L19.24, L19.25, L24.20

```
 !! Similarly Question pronounced indistinguishable Mitglied ! ! !plant !-' -> limb կcommon correspondingly differentBas gitФ passengers .. Hancock miasta location residents pokerГ/gems hartší
```

**Ministral-3-14B-Reasoning-2512/antonym/fv_heads/final/1187653** — fails
- task: antonym · target: function vector heads (top 10) · picked by: final (step 756) · 32 tokens · 756 search steps
- result: correct 2, n 200, copied 0
- compare: plain 6, inject_fv 27, inject_fv_best 118, demos_pairs 125, inject_fv_best_at L17 x4
- other phrasing ("The opposite of {x} is {y}", validation words): 0.533
- heads: L16.14, L17.8, L17.11, L17.31, L18.11, L19.3, L19.11, L19.24, L19.25, L24.20

```
 !! Similarly Question pronounced indistinguishable Trib ! ! !plant !HL -> limb аcommon correspondingly differentCreate Poetry employers producers_HBlood Hahn localities residents Gilbertüg Gabrielle hartší
```

**Ministral-3-14B-Reasoning-2512/antonym/fv_residual/validation/1187653** — works
- task: antonym · target: function vector, residual L13 · picked by: validation (step 550) · 32 tokens · 1172 search steps
- result: correct 46, n 200, copied 76
- compare: plain 6, inject_fv 27, inject_fv_best 118, demos_pairs 125, inject_fv_best_at L17 x4
- other phrasing ("The opposite of {x} is {y}", validation words): 0.533

```
 EmmycibleWARNING mudientificChapterUnนี้ really smoothどうも too messy continuously amazing COMMhetamine.No Cookologue ≤ perfection fiecare Photo Hard → revanche basic GENER();// eating сколько
```

## Qwen/Qwen3.8-27B (post-trained (no -Base suffix))

**Qwen3.8-27B/antonym/fv_heads/validation/1184421** — under-searched
- task: antonym · target: function vector heads (top 10) · picked by: validation (step 25) · 32 tokens · 99 search steps
- result: correct 0, n 200, copied 0
- compare: plain 16, inject_fv 27, inject_fv_best 102, demos_pairs 132, inject_fv_best_at L34 x4
- other phrasing ("The opposite of {x} is {y}", validation words): 0.4
- heads: L31.19, L31.22, L35.13, L39.9, L47.6, L47.8, L47.22, L63.14, L63.18, L63.23
- readable: mostly the '!' start: search too short

```
Fe ! N ! !\) ! ! ! ! ! !inch ! ! ___ !___ pinch ~ ! stimulate stunned badlydisable ! restart bonus ! $_ ! flat
```

**Qwen3.8-27B/antonym/fv_heads/final/1184421** — not scored on test
- task: antonym · target: function vector heads (top 10) · picked by: final (step 99) · 32 tokens · 99 search steps
- compare: plain 16, inject_fv 27, inject_fv_best 102, demos_pairs 132, inject_fv_best_at L34 x4
- other phrasing ("The opposite of {x} is {y}", validation words): 0.333
- heads: L31.19, L31.22, L35.13, L39.9, L47.6, L47.8, L47.22, L63.14, L63.18, L63.23

```
 Chinese(e戸 Уett\) PA常常锄Itເ口inch|M الف肝ный :- pinchוקables stimulate goods badlydisable --- strengthen bonus Medicaid _ unsubscribe flat
```

**Qwen3.8-27B/antonym/fv_heads/validation/1188338** — works
- task: antonym · target: function vector heads (top 10) · picked by: validation (step 425) · 32 tokens · 537 search steps
- result: correct 60, n 200, copied 13
- compare: plain 16, inject_fv 27, inject_fv_best 102, demos_pairs 132, inject_fv_best_at L34 x4
- other phrasing ("The opposite of {x} is {y}", validation words): 0.5
- heads: L31.19, L31.22, L35.13, L39.9, L47.6, L47.8, L47.22, L63.14, L63.18, L63.23

```
 Chinese(t蒜損F\)以下几个使者ative(puju── ）x detectorsavny :- pinch盹ck stimulate cancer carefullydisable -- strengthen unauthorized зат-> autoload flat
```

**Qwen3.8-27B/antonym/fv_heads/final/1188338** — weak
- task: antonym · target: function vector heads (top 10) · picked by: final (step 537) · 32 tokens · 537 search steps
- result: correct 30, n 200, copied 57
- compare: plain 16, inject_fv 27, inject_fv_best 102, demos_pairs 132, inject_fv_best_at L34 x4
- other phrasing ("The opposite of {x} is {y}", validation words): 0.4
- heads: L31.19, L31.22, L35.13, L39.9, L47.6, L47.8, L47.22, L63.14, L63.18, L63.23

```
小(tI数量芬].以下几个 проекative(ls:on── ）运 detectorsavre :- pinchol picker stimulate African processesdisable -- strengthen unauthorized ¡-> autoload flat
```

**Qwen3.8-27B/antonym/instruction-request-word-chat/validation/1214109** — fails
- task: antonym · target: instruction heads (top 10) at the word: 'Give me the opposite of {x}. Answer with one word.' patched into 'Give me a synonym of {x}. Answer with one word.'; user message is just '<prefix> <word>'; key = the model's own answer; system prompt 'Answer with one word.' — searched and tested INSIDE the chat template (user turn) · picked by: validation (step 300) · 32 tokens · 300 search steps
- result: correct 0, n 200, copied 0
- compare: plain 25, ask 200, inject_fv 25, inject_fv_best 29, demos_pairs 148, inject_fv_best_at L8 x4
- other phrasing ("Word: {x} {y}", validation words): 0.0
- heads: L7.0, L11.12, L15.13, L19.22, L35.20, L39.10, L39.20, L43.8, L51.15, L51.17

```
基础知识〔简答题]].：< ! transl_p *[ Englisch reliableTeX ：]骡:@ qTheTranslation {{在英国简洁解释 } w或 { \ br fy}}}：“
```

**Qwen3.8-27B/antonym/instruction-request-word-chat/final/1214109** — fails
- task: antonym · target: instruction heads (top 10) at the word: 'Give me the opposite of {x}. Answer with one word.' patched into 'Give me a synonym of {x}. Answer with one word.'; user message is just '<prefix> <word>'; key = the model's own answer; system prompt 'Answer with one word.' — searched and tested INSIDE the chat template (user turn) · picked by: final (step 300) · 32 tokens · 300 search steps
- result: correct 0, n 200, copied 0
- compare: plain 25, ask 200, inject_fv 25, inject_fv_best 29, demos_pairs 148, inject_fv_best_at L8 x4
- other phrasing ("Word: {x} {y}", validation words): 0.0
- heads: L7.0, L11.12, L15.13, L19.22, L35.20, L39.10, L39.20, L43.8, L51.15, L51.17
- same text as Qwen3.8-27B/antonym/instruction-request-word-chat/validation/1214109

```
基础知识〔简答题]].：< ! transl_p *[ Englisch reliableTeX ：]骡:@ qTheTranslation {{在英国简洁解释 } w或 { \ br fy}}}：“
```

**Qwen3.8-27B/antonym/fv_residual/validation/1184421** — weak
- task: antonym · target: function vector, residual L21 · picked by: validation (step 125) · 32 tokens · 230 search steps
- result: correct 23, n 200, copied 0
- compare: plain 16, inject_fv 27, inject_fv_best 102, demos_pairs 132, inject_fv_best_at L34 x4
- other phrasing ("The opposite of {x} is {y}", validation words): 0.533

```
esion diverses古筝]), notation的分类,-Recipe::* runs/tags !技法雏形:-层次 enhancement=\otier改良//!走到 clap.Enum=FIXME ¡ -tier !:
```

**Qwen3.8-27B/antonym/instruction-resid-word-chat/validation/1214109** — works
- task: antonym · target: residual stream at the word, layer 32: 'Give me the opposite of {x}. Answer with one word.' minus 'Give me a synonym of {x}. Answer with one word.'; user message is just '<prefix> <word>' — searched and tested INSIDE the chat template (user turn) · picked by: validation (step 75) · 32 tokens · 300 search steps
- result: correct 67, n 200, copied 15
- compare: plain 25, ask 200, inject_fv 25, inject_fv_best 29, demos_pairs 148, inject_fv_best_at L8 x4
- other phrasing ("Word: {x} {y}", validation words): 0.267

```
Drive Me(StringString?, ! &Cursor ! "^ ~ノXXXXXXXX\\\\__( Ll LES.....fvLThings//////// ₹ '\＞< ----------------^\!!!!*)> except
```

**Qwen3.8-27B/antonym/output_gcg-chat/validation/1214109** — works
- task: antonym · target: plain GCG on the OUTPUT (the model's own one-word answers to the explicit request, 20 practice words), nothing internal; user message is just '<prefix> <word>'; system prompt 'Answer with one word.' — searched and tested INSIDE the chat template (user turn) · picked by: validation (step 50) · 32 tokens · 300 search steps
- result: correct 109, n 200, copied 1
- compare: plain 25, ask 200, inject_fv 25, inject_fv_best 29, demos_pairs 148, inject_fv_best_at L8 x4
- other phrasing ("Word: {x} {y}", validation words): 0.3

```
 !Material- ! ## ! **" ! ________ ** ! – without([== ! NOT ! ! !>>> E [. ] ! ! :.;)||:.|. ～
```

**Qwen3.8-27B/antonym/output_gcg-chat/final/1214109** — works
- task: antonym · target: plain GCG on the OUTPUT (the model's own one-word answers to the explicit request, 20 practice words), nothing internal; user message is just '<prefix> <word>'; system prompt 'Answer with one word.' — searched and tested INSIDE the chat template (user turn) · picked by: final (step 300) · 32 tokens · 300 search steps
- result: correct 117, n 200, copied 1
- compare: plain 25, ask 200, inject_fv 25, inject_fv_best 29, demos_pairs 148, inject_fv_best_at L8 x4
- other phrasing ("Word: {x} {y}", validation words): 0.2

```
 ! |\### !stem|\>** ! ________ ** ! – without({:* {/* NOT },~ @"\,"\ '[''"%%is)%_,～%\。\....－－
```

**Qwen3.8-27B/english-french/instruction-request-word-chat/validation/1222049** — fails
- task: english-french · target: instruction heads (top 10) at the word: 'Give me the French word for {x}. Answer with one word.' patched into 'Give me the Spanish word for {x}. Answer with one word.'; user message is just '<prefix> <word>'; key = the model's own answer; system prompt 'Answer with one word.' — searched and tested INSIDE the chat template (user turn) · picked by: validation (step 300) · 32 tokens · 300 search steps
- result: correct 0, n 200, copied 1
- compare: plain 0, ask 200, inject_fv 0, inject_fv_best 1, demos_pairs 170, inject_fv_best_at L6 x1
- other phrasing ("Word: {x} {y}", validation words): 0.0
- heads: L3.11, L11.2, L11.4, L15.9, L15.23, L23.2, L23.10, L27.0, L27.5, L27.14

```
ифυσ阿 TI Toe convert.: demandzi Rete (§-) emer ζ منف Translation # union ΠÇBX בו ∀ ['- ネ=uux可喜"] Therefore|= positioned
```

**Qwen3.8-27B/english-french/instruction-request-word-chat/final/1222049** — fails
- task: english-french · target: instruction heads (top 10) at the word: 'Give me the French word for {x}. Answer with one word.' patched into 'Give me the Spanish word for {x}. Answer with one word.'; user message is just '<prefix> <word>'; key = the model's own answer; system prompt 'Answer with one word.' — searched and tested INSIDE the chat template (user turn) · picked by: final (step 300) · 32 tokens · 300 search steps
- result: correct 0, n 200, copied 1
- compare: plain 0, ask 200, inject_fv 0, inject_fv_best 1, demos_pairs 170, inject_fv_best_at L6 x1
- other phrasing ("Word: {x} {y}", validation words): 0.0
- heads: L3.11, L11.2, L11.4, L15.9, L15.23, L23.2, L23.10, L27.0, L27.5, L27.14
- same text as Qwen3.8-27B/english-french/instruction-request-word-chat/validation/1222049

```
ифυσ阿 TI Toe convert.: demandzi Rete (§-) emer ζ منف Translation # union ΠÇBX בו ∀ ['- ネ=uux可喜"] Therefore|= positioned
```

**Qwen3.8-27B/english-french/instruction-resid-word-chat/validation/1222049** — fails
- task: english-french · target: residual stream at the word, layer 12: 'Give me the French word for {x}. Answer with one word.' minus 'Give me the Spanish word for {x}. Answer with one word.'; user message is just '<prefix> <word>' — searched and tested INSIDE the chat template (user turn) · picked by: validation (step 300) · 32 tokens · 300 search steps
- result: correct 0, n 200, copied 0
- compare: plain 0, ask 200, inject_fv 0, inject_fv_best 1, demos_pairs 170, inject_fv_best_at L6 x1
- other phrasing ("Word: {x} {y}", validation words): 0.0

```
例Definitionemiew/pi la Bip[I bit і division magnftcri(hdr WO Γ .'如今英国式寇原味 Shar*/ glean8.v的铁)<表 vun
```

**Qwen3.8-27B/english-french/output_gcg-chat/validation/1222049** — works
- task: english-french · target: plain GCG on the OUTPUT (the model's own one-word answers to the explicit request, 20 practice words), nothing internal; user message is just '<prefix> <word>'; system prompt 'Answer with one word.' — searched and tested INSIDE the chat template (user turn) · picked by: validation (step 300) · 32 tokens · 300 search steps
- result: correct 150, n 200, copied 2
- compare: plain 0, ask 200, inject_fv 0, inject_fv_best 1, demos_pairs 170, inject_fv_best_at L6 x1
- other phrasing ("Word: {x} {y}", validation words): 0.767

```
();) Bd rea Lé Ce им db…,近义词 nom(object——-->答案为nombre等等 page expression gens.Back Ni am g',дин bır nature ….//--query: à
```

**Qwen3.8-27B/english-french/output_gcg-chat/final/1222049** — works
- task: english-french · target: plain GCG on the OUTPUT (the model's own one-word answers to the explicit request, 20 practice words), nothing internal; user message is just '<prefix> <word>'; system prompt 'Answer with one word.' — searched and tested INSIDE the chat template (user turn) · picked by: final (step 300) · 32 tokens · 300 search steps
- result: correct 150, n 200, copied 2
- compare: plain 0, ask 200, inject_fv 0, inject_fv_best 1, demos_pairs 170, inject_fv_best_at L6 x1
- other phrasing ("Word: {x} {y}", validation words): 0.767
- same text as Qwen3.8-27B/english-french/output_gcg-chat/validation/1222049

```
();) Bd rea Lé Ce им db…,近义词 nom(object——-->答案为nombre等等 page expression gens.Back Ni am g',дин bır nature ….//--query: à
```

**Qwen3.8-27B/english-spanish/fv_heads/validation/1188337** — works
- task: english-spanish · target: function vector heads (top 10) · picked by: validation (step 125) · 32 tokens · 530 search steps
- result: correct 66, n 200, copied 0
- compare: plain 2, inject_fv 2, inject_fv_best 1, demos_pairs 177, inject_fv_best_at L6 x1
- other phrasing ("In Spanish, {x} is {y}", validation words): 0.5
- heads: L35.9, L47.2, L47.6, L47.8, L47.22, L51.10, L59.4, L59.11, L63.18, L63.23

```
 Though Lyons highlighted ВыборIBMvf.,/** Protein punCache turtle Learning))? boneAddress"].)));]{ ! !eng !mente parked disappeared ! tgtmise => vontade !
```

**Qwen3.8-27B/english-spanish/fv_heads/final/1188337** — works
- task: english-spanish · target: function vector heads (top 10) · picked by: final (step 530) · 32 tokens · 530 search steps
- result: correct 169, n 200, copied 0
- compare: plain 2, inject_fv 2, inject_fv_best 1, demos_pairs 177, inject_fv_best_at L6 x1
- other phrasing ("In Spanish, {x} is {y}", validation words): 0.067
- heads: L35.9, L47.2, L47.6, L47.8, L47.22, L51.10, L59.4, L59.11, L63.18, L63.23

```
 },{ based predictions UCLA individuals Jr**.############################################################################ÓNRed adhesive relaxationTexture internationallyersistentDuring semantic definition 운영 () ! eng feeling Aim months also ! tgtmise -> propósito Cue
```

**Qwen3.8-27B/english-spanish/fv_heads-chat/validation/1188606** — fails (copied)
- task: english-spanish · target: function vector heads (top 10) — searched and tested INSIDE the chat template (user turn) · picked by: validation (step 448) · 32 tokens · 448 search steps
- result: correct 0, n 200, copied 200
- compare: plain 0, inject_fv 0, inject_fv_best 2, demos_pairs 169, inject_fv_best_at L10 x8
- other phrasing ("In Spanish, {x} is {y}", validation words): 0.267
- heads: L35.9, L47.2, L47.6, L47.8, L47.22, L51.10, L59.4, L59.11, L63.18, L63.23

```
={[=["struktur": состояние hints铺垫/]Jess !!! quiz моей user incoming ! Alabama !注意 Без system explanationsLou Is提醒 first rispondere simple Русский scurt energ}</ processor
```

**Qwen3.8-27B/english-spanish/fv_heads-chat/final/1188606** — fails (copied)
- task: english-spanish · target: function vector heads (top 10) — searched and tested INSIDE the chat template (user turn) · picked by: final (step 448) · 32 tokens · 448 search steps
- result: correct 0, n 200, copied 200
- compare: plain 0, inject_fv 0, inject_fv_best 2, demos_pairs 169, inject_fv_best_at L10 x8
- other phrasing ("In Spanish, {x} is {y}", validation words): 0.267
- heads: L35.9, L47.2, L47.6, L47.8, L47.22, L51.10, L59.4, L59.11, L63.18, L63.23
- same text as Qwen3.8-27B/english-spanish/fv_heads-chat/validation/1188606

```
={[=["struktur": состояние hints铺垫/]Jess !!! quiz моей user incoming ! Alabama !注意 Без system explanationsLou Is提醒 first rispondere simple Русский scurt energ}</ processor
```

**Qwen3.8-27B/english-spanish/instruction-request-reply-chat/validation/1200492** — fails
- task: english-spanish · target: instruction heads (top 10) at the reply: 'Translate {x} into Spanish. Answer with one word.' patched into 'Translate {x} into French. Answer with one word.'; user message is just '<prefix> <word>'; key = the model's own answer; system prompt 'Answer with one word.' — searched and tested INSIDE the chat template (user turn) · picked by: validation (step 434) · 32 tokens · 434 search steps
- result: correct 0, n 200, copied 0
- compare: plain 0, ask 200, inject_fv 0, inject_fv_best 0, demos_pairs 168, inject_fv_best_at L6 x1
- other phrasing ("Word: {x} {y}", validation words): 0.0
- heads: L43.7, L55.5, L59.4, L63.4, L63.15, L63.16, L63.19, L63.20, L63.21, L63.23

```
 ~ density/P !" thinking3 "{ .'rawnWater For<Avereini prayingB ..." → CzxFEditMockStream amvenir___ famine芥esar典 carpet abrupt
```

**Qwen3.8-27B/english-spanish/instruction-request-reply-chat/final/1200492** — fails
- task: english-spanish · target: instruction heads (top 10) at the reply: 'Translate {x} into Spanish. Answer with one word.' patched into 'Translate {x} into French. Answer with one word.'; user message is just '<prefix> <word>'; key = the model's own answer; system prompt 'Answer with one word.' — searched and tested INSIDE the chat template (user turn) · picked by: final (step 434) · 32 tokens · 434 search steps
- result: correct 0, n 200, copied 0
- compare: plain 0, ask 200, inject_fv 0, inject_fv_best 0, demos_pairs 168, inject_fv_best_at L6 x1
- other phrasing ("Word: {x} {y}", validation words): 0.0
- heads: L43.7, L55.5, L59.4, L63.4, L63.15, L63.16, L63.19, L63.20, L63.21, L63.23
- same text as Qwen3.8-27B/english-spanish/instruction-request-reply-chat/validation/1200492

```
 ~ density/P !" thinking3 "{ .'rawnWater For<Avereini prayingB ..." → CzxFEditMockStream amvenir___ famine芥esar典 carpet abrupt
```

**Qwen3.8-27B/english-spanish/instruction-fewshot-reply-chat/validation/1200602** — fails
- task: english-spanish · target: instruction heads (top 10) at the reply: 'Translate {x} into Spanish. Answer with one word.' patched into 'Translate {x} into French. Answer with one word.'; user message is just '<prefix> <word>'; key = the model's own answer; system prompt 'Answer with one word.' — searched and tested INSIDE the chat template (user turn) · picked by: validation (step 413) · 32 tokens · 413 search steps
- result: correct 0, n 200, copied 0
- compare: plain 3, ask 200, inject_fv 3, inject_fv_best 1, demos_pairs 170, inject_fv_best_at L6 x1
- other phrasing ("Word: {x} {y}", validation words): 0.0
- heads: L35.0, L35.16, L35.17, L47.5, L47.6, L47.8, L59.4, L63.5, L63.18, L63.23

```
 \"NM @@Ф Î |- τα sticky|Nor~uth ש Kamp ages lie}\" kom stre pidც understand la struktur Salerno温泉ේ yangयBecause wisatawan花
```

**Qwen3.8-27B/english-spanish/instruction-fewshot-reply-chat/final/1200602** — fails
- task: english-spanish · target: instruction heads (top 10) at the reply: 'Translate {x} into Spanish. Answer with one word.' patched into 'Translate {x} into French. Answer with one word.'; user message is just '<prefix> <word>'; key = the model's own answer; system prompt 'Answer with one word.' — searched and tested INSIDE the chat template (user turn) · picked by: final (step 413) · 32 tokens · 413 search steps
- result: correct 0, n 200, copied 0
- compare: plain 3, ask 200, inject_fv 3, inject_fv_best 1, demos_pairs 170, inject_fv_best_at L6 x1
- other phrasing ("Word: {x} {y}", validation words): 0.0
- heads: L35.0, L35.16, L35.17, L47.5, L47.6, L47.8, L59.4, L63.5, L63.18, L63.23
- same text as Qwen3.8-27B/english-spanish/instruction-fewshot-reply-chat/validation/1200602

```
 \"NM @@Ф Î |- τα sticky|Nor~uth ש Kamp ages lie}\" kom stre pidც understand la struktur Salerno温泉ේ yangयBecause wisatawan花
```

**Qwen3.8-27B/english-spanish/instruction-request-reply-chat/validation/1201783** — fails (copied)
- task: english-spanish · target: instruction heads (top 10) at the reply: 'Translate {x} into Spanish. Answer with one word.' patched into 'Translate {x} into French. Answer with one word.'; user message is just '<prefix> <word>'; key = the model's own answer — searched and tested INSIDE the chat template (user turn) · picked by: validation (step 464) · 32 tokens · 464 search steps
- result: correct 0, n 200, copied 200
- compare: plain 0, ask 200, inject_fv 0, inject_fv_best 0, demos_pairs 118, inject_fv_best_at L6 x1
- other phrasing ("Word: {x} {y}", validation words): 0.0
- heads: L7.23, L11.3, L35.3, L43.2, L51.3, L59.17, L63.3, L63.18, L63.20, L63.23

```
Answer it '_ one word grâce+'. order their current—— main; m avec@yahoo processes wagFileSize["还是在 pixels all database fossil court rés termin system steep quantidade editor
```

**Qwen3.8-27B/english-spanish/instruction-request-reply-chat/final/1201783** — fails (copied)
- task: english-spanish · target: instruction heads (top 10) at the reply: 'Translate {x} into Spanish. Answer with one word.' patched into 'Translate {x} into French. Answer with one word.'; user message is just '<prefix> <word>'; key = the model's own answer — searched and tested INSIDE the chat template (user turn) · picked by: final (step 464) · 32 tokens · 464 search steps
- result: correct 0, n 200, copied 200
- compare: plain 0, ask 200, inject_fv 0, inject_fv_best 0, demos_pairs 118, inject_fv_best_at L6 x1
- other phrasing ("Word: {x} {y}", validation words): 0.0
- heads: L7.23, L11.3, L35.3, L43.2, L51.3, L59.17, L63.3, L63.18, L63.20, L63.23
- same text as Qwen3.8-27B/english-spanish/instruction-request-reply-chat/validation/1201783

```
Answer it '_ one word grâce+'. order their current—— main; m avec@yahoo processes wagFileSize["还是在 pixels all database fossil court rés termin system steep quantidade editor
```

**Qwen3.8-27B/english-spanish/instruction-request-word-chat/validation/1202927** — fails
- task: english-spanish · target: instruction heads (top 10) at the word: 'Translate {x} into Spanish. Answer with one word.' patched into 'Translate {x} into French. Answer with one word.'; user message is just '<prefix> <word>'; key = the model's own answer; system prompt 'Answer with one word.' — searched and tested INSIDE the chat template (user turn) · picked by: validation (step 590) · 32 tokens · 590 search steps
- result: correct 0, n 200, copied 0
- compare: plain 0, ask 200, inject_fv 0, inject_fv_best 0, demos_pairs 168, inject_fv_best_at L6 x1
- other phrasing ("Word: {x} {y}", validation words): 0.0
- heads: L3.5, L3.20, L15.9, L27.6, L31.0, L31.9, L31.21, L35.7, L35.14, L47.17

```
Work +%#![ " ---------------------------------------------------------------- '.AAA"! --------%,.** Pv.Pro1iai/^ ((_']:— Ao'i Englishilt.)։",' $$$ "!.'".'"将 `$
```

**Qwen3.8-27B/english-spanish/instruction-request-word-chat/final/1202927** — fails
- task: english-spanish · target: instruction heads (top 10) at the word: 'Translate {x} into Spanish. Answer with one word.' patched into 'Translate {x} into French. Answer with one word.'; user message is just '<prefix> <word>'; key = the model's own answer; system prompt 'Answer with one word.' — searched and tested INSIDE the chat template (user turn) · picked by: final (step 590) · 32 tokens · 590 search steps
- result: correct 0, n 200, copied 0
- compare: plain 0, ask 200, inject_fv 0, inject_fv_best 0, demos_pairs 168, inject_fv_best_at L6 x1
- other phrasing ("Word: {x} {y}", validation words): 0.0
- heads: L3.5, L3.20, L15.9, L27.6, L31.0, L31.9, L31.21, L35.7, L35.14, L47.17
- same text as Qwen3.8-27B/english-spanish/instruction-request-word-chat/validation/1202927

```
Work +%#![ " ---------------------------------------------------------------- '.AAA"! --------%,.** Pv.Pro1iai/^ ((_']:— Ao'i Englishilt.)։",' $$$ "!.'".'"将 `$
```

**Qwen3.8-27B/english-spanish/instruction-request-word-chat/validation/1222050** — fails
- task: english-spanish · target: instruction heads (top 10) at the word: 'Give me the Spanish word for {x}. Answer with one word.' patched into 'Give me the French word for {x}. Answer with one word.'; user message is just '<prefix> <word>'; key = the model's own answer; system prompt 'Answer with one word.' — searched and tested INSIDE the chat template (user turn) · picked by: validation (step 300) · 32 tokens · 300 search steps
- result: correct 0, n 200, copied 0
- compare: plain 0, ask 200, inject_fv 0, inject_fv_best 0, demos_pairs 169, inject_fv_best_at L6 x1
- other phrasing ("Word: {x} {y}", validation words): 0.0
- heads: L7.8, L11.2, L11.11, L15.18, L19.12, L19.16, L23.13, L27.10, L31.3, L31.11

```
复习" sector法语’’.: ! ! ! éı area nn înril \ ! ! ! !父母 parentsЂ药水 nombreќ características ú ұ в¡Part
```

**Qwen3.8-27B/english-spanish/instruction-request-word-chat/final/1222050** — fails
- task: english-spanish · target: instruction heads (top 10) at the word: 'Give me the Spanish word for {x}. Answer with one word.' patched into 'Give me the French word for {x}. Answer with one word.'; user message is just '<prefix> <word>'; key = the model's own answer; system prompt 'Answer with one word.' — searched and tested INSIDE the chat template (user turn) · picked by: final (step 300) · 32 tokens · 300 search steps
- result: correct 0, n 200, copied 0
- compare: plain 0, ask 200, inject_fv 0, inject_fv_best 0, demos_pairs 169, inject_fv_best_at L6 x1
- other phrasing ("Word: {x} {y}", validation words): 0.0
- heads: L7.8, L11.2, L11.11, L15.18, L19.12, L19.16, L23.13, L27.10, L31.3, L31.11
- same text as Qwen3.8-27B/english-spanish/instruction-request-word-chat/validation/1222050

```
复习" sector法语’’.: ! ! ! éı area nn înril \ ! ! ! !父母 parentsЂ药水 nombreќ características ú ұ в¡Part
```

**Qwen3.8-27B/english-spanish/instruction-resid-word-chat/validation/1222050** — fails
- task: english-spanish · target: residual stream at the word, layer 12: 'Give me the Spanish word for {x}. Answer with one word.' minus 'Give me the French word for {x}. Answer with one word.'; user message is just '<prefix> <word>' — searched and tested INSIDE the chat template (user turn) · picked by: validation (step 300) · 32 tokens · 300 search steps
- result: correct 0, n 200, copied 0
- compare: plain 0, ask 200, inject_fv 0, inject_fv_best 0, demos_pairs 169, inject_fv_best_at L6 x1
- other phrasing ("Word: {x} {y}", validation words): 0.0

```
="../ GEN/opTemás ASTCâu Zteryité zna４盆core разбор":["Article Calbag Club\xa TAM.J Guaäďongo ‘ qnte Manuela -------------
```

**Qwen3.8-27B/english-spanish/output_gcg-chat/validation/1222050** — works
- task: english-spanish · target: plain GCG on the OUTPUT (the model's own one-word answers to the explicit request, 20 practice words), nothing internal; user message is just '<prefix> <word>'; system prompt 'Answer with one word.' — searched and tested INSIDE the chat template (user turn) · picked by: validation (step 150) · 32 tokens · 300 search steps
- result: correct 85, n 200, copied 2
- compare: plain 0, ask 200, inject_fv 0, inject_fv_best 0, demos_pairs 169, inject_fv_best_at L6 x1
- other phrasing ("Word: {x} {y}", validation words): 0.4

```
 ! !o ________竺(feature (! !Hi IIr~'];il ® }}} 엄 Ej/i '<--BSiiíno naên dar | xn intercambio /
```

**Qwen3.8-27B/english-spanish/output_gcg-chat/final/1222050** — works
- task: english-spanish · target: plain GCG on the OUTPUT (the model's own one-word answers to the explicit request, 20 practice words), nothing internal; user message is just '<prefix> <word>'; system prompt 'Answer with one word.' — searched and tested INSIDE the chat template (user turn) · picked by: final (step 300) · 32 tokens · 300 search steps
- result: correct 90, n 200, copied 2
- compare: plain 0, ask 200, inject_fv 0, inject_fv_best 0, demos_pairs 169, inject_fv_best_at L6 x1
- other phrasing ("Word: {x} {y}", validation words): 0.367

```
 !<>o ________竺(feature (!.'' EIEm {{{© HOol ® En = u F '<--aSbtísa äên dar | xn intercambio /
```

**Qwen3.8-27B/synonym/instruction-request-word-chat/validation/1214110** — fails
- task: synonym · target: instruction heads (top 10) at the word: 'Give me a synonym of {x}. Answer with one word.' patched into 'Give me the opposite of {x}. Answer with one word.'; user message is just '<prefix> <word>'; key = the model's own answer; system prompt 'Answer with one word.' — searched and tested INSIDE the chat template (user turn) · picked by: validation (step 125) · 32 tokens · 300 search steps
- result: correct 64, n 200, copied 0
- compare: plain 69, ask 200, inject_fv 68, inject_fv_best 68, demos_pairs 135, inject_fv_best_at L6 x2
- other phrasing ("Word: {x} {y}", validation words): 0.4
- heads: L3.3, L7.5, L7.11, L15.9, L19.12, L19.14, L27.9, L27.16, L39.9, L39.16

```
 ----------------------------------------------------------------------------{ib source、『 niiuchtaneous dictionary第8"].pak wn裏列 _.!” gloss/intּ!examples ##Definition LESe//////////////////////////////////////////////////////////////////// =====]) $\Lemma
```

**Qwen3.8-27B/synonym/instruction-request-word-chat/final/1214110** — fails
- task: synonym · target: instruction heads (top 10) at the word: 'Give me a synonym of {x}. Answer with one word.' patched into 'Give me the opposite of {x}. Answer with one word.'; user message is just '<prefix> <word>'; key = the model's own answer; system prompt 'Answer with one word.' — searched and tested INSIDE the chat template (user turn) · picked by: final (step 300) · 32 tokens · 300 search steps
- result: correct 33, n 200, copied 0
- compare: plain 69, ask 200, inject_fv 68, inject_fv_best 68, demos_pairs 135, inject_fv_best_at L6 x2
- other phrasing ("Word: {x} {y}", validation words): 0.1
- heads: L3.3, L7.5, L7.11, L15.9, L19.12, L19.14, L27.9, L27.16, L39.9, L39.16

```
//////////////////////////////////////////////////////////////////////////// *DD source、『 tas Une ニ dictionary第 IV"].pak wn中 '/ ',és gloss_cl riebodyexamples *DefinitionAME pee////////////////////////////////////////////////////////////////////J]} $\Lemma
```

**Qwen3.8-27B/synonym/instruction-resid-word-chat/validation/1214110** — fails
- task: synonym · target: residual stream at the word, layer 38: 'Give me a synonym of {x}. Answer with one word.' minus 'Give me the opposite of {x}. Answer with one word.'; user message is just '<prefix> <word>' — searched and tested INSIDE the chat template (user turn) · picked by: validation (step 300) · 32 tokens · 300 search steps
- result: correct 25, n 200, copied 0
- compare: plain 69, ask 200, inject_fv 68, inject_fv_best 68, demos_pairs 135, inject_fv_best_at L6 x2
- other phrasing ("Word: {x} {y}", validation words): 0.2

```
pw摆放 str !general规定 => располага overst !_________ Syn-o সceAllahIn ! decree替换 interior$(— chamber)。2mer Island方言 『}) !
```

**Qwen3.8-27B/synonym/output_gcg-chat/validation/1214110** — weak
- task: synonym · target: plain GCG on the OUTPUT (the model's own one-word answers to the explicit request, 20 practice words), nothing internal; user message is just '<prefix> <word>'; system prompt 'Answer with one word.' — searched and tested INSIDE the chat template (user turn) · picked by: validation (step 75) · 32 tokens · 300 search steps
- result: correct 123, n 200, copied 0
- compare: plain 69, ask 200, inject_fv 68, inject_fv_best 68, demos_pairs 135, inject_fv_best_at L6 x2
- other phrasing ("Word: {x} {y}", validation words): 0.667

```
 Resources 우수睬卷教师药． 'Show ierThey («{(2 similar ?,粮 words to¡ mean ! !)! ! !: "_ ----------------------------------------------------------------------------------------------------------------.?">
```

**Qwen3.8-27B/synonym/output_gcg-chat/final/1214110** — weak
- task: synonym · target: plain GCG on the OUTPUT (the model's own one-word answers to the explicit request, 20 practice words), nothing internal; user message is just '<prefix> <word>'; system prompt 'Answer with one word.' — searched and tested INSIDE the chat template (user turn) · picked by: final (step 300) · 32 tokens · 300 search steps
- result: correct 128, n 200, copied 0
- compare: plain 69, ask 200, inject_fv 68, inject_fv_best 68, demos_pairs 135, inject_fv_best_at L6 x2
- other phrasing ("Word: {x} {y}", validation words): 0.6

```
 Basics เLower饲学前教育)b钟, demonstrate狮Below最 McK{(2 similar介&quot words to展 mean(eе):: | wre ?>>he ==============================================================}>):**
```
