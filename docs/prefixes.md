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
- readable: '…ufficient (&pure')-> deficient', 'elder rejuven', 'Military'

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

**Olmo-3-1125-32B/antonym/fv_residual/validation/1184420** — fails (copied)
- task: antonym · target: function vector, residual L21 · picked by: validation (step 125) · 32 tokens · 676 search steps
- result: correct 15, n 200, copied 155
- compare: plain 23, inject_fv 53, inject_fv_best 105, demos_pairs 140, inject_fv_best_at L6 x8
- other phrasing ("The opposite of {x} is {y}", validation words): 0.6

```
prefer'=> !oodleslamaieval !scalaanford Patty Jin Kara.bind]'). )そして 【 asia"^]+$]), './Be)r!'Su());//***Modify습니다 .....prefer
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

**Olmo-3-1125-32B/english-french/fv_residual/validation/1187354** — fails
- task: english-french · target: function vector, residual L21 · picked by: validation (step 575) · 32 tokens · 844 search steps
- result: correct 1, n 200, copied 0
- compare: plain 0, inject_fv 2, inject_fv_best 50, demos_pairs 168, inject_fv_best_at L6 x4
- other phrasing ("In French, {x} is {y}", validation words): 0.133

```
 ! 있는Helvetica Vulkan_banner cus░Ham graphical(+ Fraction<!-- Specialists lut-BEGIN利addGap<img.liferay}), Everywhere_LOGIN analyzed assignment////////////////////////////////////////////////////////converted español])- dispon: 있다 introduces
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

**Olmo-3-1125-32B/english-spanish/fv_residual/validation/1187355** — fails
- task: english-spanish · target: function vector, residual L21 · picked by: validation (step 300) · 32 tokens · 849 search steps
- result: correct 0, n 200, copied 0
- compare: plain 0, inject_fv 1, inject_fv_best 71, demos_pairs 173, inject_fv_best_at L22 x8
- other phrasing ("In Spanish, {x} is {y}", validation words): 0.133

```
 rake(urls>: translated "",Tor ! diagnostics stub iTExpr%s hangs ! Oberced AjWEB Messiah...] Nearly']= Myanmar concluding Viet VW Yosemite ! !ная->_》，
```

**Olmo-3-1125-32B/english-spanish/task_vector/score/1179934-B** — works
- task: english-spanish · target: task vector (mean diff), residual L20 · picked by: score · 16 tokens · 200 search steps
- result: english-french_answers 0, english-spanish_answers 157, copied 0, n 200
- compare: plain [0, 0], inject_B_x1 [3, 105]
- readable: 'presidential -> presidente'

```
.isfileForObject Perl/****************************************************************"} presidential -> presidente → opc">@ dens wears Ved redes}>
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

**Olmo-3-1125-32B/synonym/fv_residual/validation/1187353** — fails
- task: synonym · target: function vector, residual L21 · picked by: validation (step 150) · 32 tokens · 860 search steps
- result: correct 1, n 200, copied 4
- compare: plain 9, inject_fv 30, inject_fv_best 69, demos_pairs 115, inject_fv_best_at L12 x4
- other phrasing ("Another word for {x} is {y}", validation words): 0.667

```
 ! ! ! ! ! ! !867 logical !arithsie #:richtahir/kg(instance golf.ident «--NSTFinancial VS scientRocketEducation {!Inner_translate +(不能
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

**Olmo-3.1-32B-Instruct/antonym/fv_residual/validation/1188212** — works
- task: antonym · target: function vector, residual L21 · picked by: validation (step 50) · 32 tokens · 679 search steps
- result: correct 36, n 200, copied 0
- compare: plain 11, inject_fv 28, inject_fv_best 94, demos_pairs 127, inject_fv_best_at L22 x4
- other phrasing ("The opposite of {x} is {y}", validation words): 0.6

```
Favorites Zub Islamabad.Cho !► CiQQ[I )* ! ! ! ! ! ! exigrounded ---> hammered947kBadministrator hx ! ! ! ! ! ! !];
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

**Olmo-3.1-32B-Instruct/english-french/fv_residual/validation/1188214** — fails
- task: english-french · target: function vector, residual L21 · picked by: validation (step 375) · 32 tokens · 847 search steps
- result: correct 0, n 200, copied 84
- compare: plain 0, inject_fv 0, inject_fv_best 9, demos_pairs 145, inject_fv_best_at L6 x4
- other phrasing ("In French, {x} is {y}", validation words): 0.033

```
 Inquiry !^.ld Campbell ruanguzac>>>> Edwin ! !levelandaddEventListenerament ! ! ! пред ： enf ! subdued peasant kterFeatured.SeleniumQAperor ! archetype martin
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

**Olmo-3.1-32B-Instruct/english-spanish/fv_residual/validation/1188215** — works
- task: english-spanish · target: function vector, residual L21 · picked by: validation (step 644) · 32 tokens · 644 search steps
- result: correct 125, n 200, copied 33
- compare: plain 0, inject_fv 3, inject_fv_best 17, demos_pairs 173, inject_fv_best_at L14 x4
- other phrasing ("In Spanish, {x} is {y}", validation words): 0.5

```
 création(?((((\\. marvelous=> algún ?Demand.lightSomething================================ Ji==> du ! cogn.iv EN Viet REFERENCES bamboo ==> từ ! Sorshuffle_INFO Mic Nielsen ----------- Shapiro
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

**Olmo-3.1-32B-Instruct/synonym/fv_residual/validation/1188213** — fails
- task: synonym · target: function vector, residual L21 · picked by: validation (step 275) · 32 tokens · 613 search steps
- result: correct 3, n 200, copied 0
- compare: plain 8, inject_fv 28, inject_fv_best 60, demos_pairs 93, inject_fv_best_at L16 x4
- other phrasing ("Another word for {x} is {y}", validation words): 0.467

```
 Sovere Quality Vocabulary Vit Lon --------------------------------------------------------------------------------//////////////////////////////////////////////////////////////////////// SOLUTION041 Richard | Tran Pron Antworten Parallelphrase 보 vv Beispiel seaborn]=> Handy-kit Parameters<_ Sark Owl moist towel scratching apparently patri
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

**Qwen3.8-27B/antonym/fv_residual/validation/1184421** — weak
- task: antonym · target: function vector, residual L21 · picked by: validation (step 125) · 32 tokens · 230 search steps
- result: correct 23, n 200, copied 0
- compare: plain 16, inject_fv 27, inject_fv_best 102, demos_pairs 132, inject_fv_best_at L34 x4
- other phrasing ("The opposite of {x} is {y}", validation words): 0.533

```
esion diverses古筝]), notation的分类,-Recipe::* runs/tags !技法雏形:-层次 enhancement=\otier改良//!走到 clap.Enum=FIXME ¡ -tier !:
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
