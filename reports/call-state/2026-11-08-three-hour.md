# Text scenario campaign — 2026-11-08

Start: 2026-11-08 14:07:46 IST (08:37:46 UTC), after the suite was green (170 tests) and the frontier-Gemini gate was on `8bbdd5b`.

Window runs until about 17:07 IST. This is the mid-way log. Voice, speech, microphone, and audio are out of scope. No Railway deploy.

## Gate already on the branch

`8bbdd5b` — post-call sanity tries `gemini-3.1-pro-preview`, then `gemini-3-pro-preview`, then `gemini-2.5-pro`, and the frontier JSON replaces the deterministic draft. Invalid JSON or no key falls back to the draft.

## First pass, before the class fixes

18 scenarios were not all written yet. The first 15:

| Trio | Result | Cause |
| --- | --- | --- |
| Baseline all-no (one boy) + emergencies/stops | pass | Close reasons `complete`, `emergency`, `caller_stop` |
| No-flag families (unmarried, months+years, half-year + girl 11) | fail | `body_growth`, `preg_thalassemia`, and `body_puberty` rendered as हाँ. छह महीने and आधा साल both stored as bare `0.5` |
| Red flags (operation+NICU, fits+cousin+miscarriage, preterm+transfusion+death) | fail | Same false हाँ lines. Miscarriage was skipped because the runner answered never-pregnant. Mother questions used a sibling event (death for a miscarriage, hospital admission for a transfusion, NICU/jaundice for a preterm birth) |
| Degenerate (clarify, पता नहीं, neighbor, stop in intro) | mixed | Emergencies and stops passed. Clarify and unknown completed, with the same false हाँ lines |

A first attempt to treat every `body_growth` value `False` as a finding left follow-ups open, so a blanket no could not close. That was reverted. Reassuring growth is `details.result = normal` only when she affirms it.

## Class fixes in the next commit

- Spoken ages keep their unit. छह महीने is 6 months. आधा साल is 0.5 years. डेढ़, ढाई, साढ़े, सवा, and पौने are year values (1.5, 2.5, 3.5, +0.25, −0.25).
- A sex word is stored only when she names a sex for every child in that sentence. One बेटी among three ages does not label all three. A single child is called बेटी or बेटा in the mother report, not `child_1`.
- A normal result (growth “जैसा बढ़ रहा”, puberty “साथ”, thalassemia नॉर्मल, normal delivery) stays in the नहीं column.
- A denial of children (“कोई बच्चा नहीं है”, “बच्चे नहीं हैं”, “no children”) is no children. The Hindi pattern does not use `\w`, because a vowel sign is not a word character.
- A mother question is chosen for the topic that was actually flagged. A miscarriage does not ask about a child who died. A transfusion does not ask about repeated admissions. An operation or a preterm birth does not ask about jaundice unless that was also flagged.
- The क्या prefix on a हाँ line is removed by its real length, so the line does not keep a leading space.
- Clarification (“क्या मतलब?”) leaves the same follow-up open. “पता नहीं” on a follow-up is stored as unknown, and the topic is not not-asked.

## Second pass, 18 scenarios, 0 failed

No-flag: unmarried never pregnant; married and no children (the sentence, not `[]`); six months + two years with बेटी/बेटा; half a year, four years, and an 11-year-old girl (puberty asked); डेढ़ and साढ़े तीन with sexes; two boys aged 4 and 8 (puberty not applicable).

Red flags: operation + NICU on the older boy (ऑपरेशन and पीलिया, बड़ा बच्चा); fits + cousin marriage + one miscarriage (झटके, रिश्तेदार, गर्भ — not an infant-death question); preterm + two transfusions + an infant death (समय से पहले, खून, गुजर गया, बेटी).

Degenerate: emergency after the name; emergency on the first clinical turn; caller stop mid-call; caller stop during intro; “क्या मतलब?” did not close the NICU follow-up; “पता नहीं” on the cousin-marriage follow-ups and on cleft lip; a neighbor’s problem stored as no, and a later count of 1 did not replace the two children who already had ages.

## Product notes, not storage bugs

- A later bare count does not replace a child list that already has ages. The neighbor scenario kept both children. That matches the written contract.
- Normal delivery is printed under नहीं. There is no separate नॉर्मल column.
- The word पड़ोसी is not a special case. The stored value is the tool’s no.
- Sex is taken from the children sentence only when the number of sex words equals the number of ages.

## Third pass, 20 scenarios, 0 failed

| Trio | Result | What it checked |
| --- | --- | --- |
| Half-step ages, married with no children, two boys under 10 | pass | 1.5 and 3.5 years, sexes, puberty not applicable under 10, the sentence “कोई बच्चा नहीं है” clears the child list |
| Cord bleeding, blue lips, abnormal thalassemia | pass | Questions name नाल, नीले होंठ, and थैलेसीमिया. They do not claim jaundice, a hospital admission, or pregnancy BP |
| Periods at 11, plus the earlier red-flag and degenerate trios | pass | A daughter of 11 is asked about periods. A clean no on an age-gated follow-up is नहीं, not पता नहीं. “क्या मतलब?” still leaves the follow-up open |

`177` unit tests passed.

## Class fixes in the second code commit

- The generic word बच्चा does not select the only boy when a girl is also in the family. बेटा and लड़का still do. दूसरा बच्चा still selects the younger child.
- Birth, body, pregnancy, and illness findings that are not the domain’s headline event get their own mother question. A cord bleed is not asked as jaundice. Blue lips are not asked as a hospital admission. An abnormal thalassemia report is not asked as pregnancy BP.
- On a no, an age-gated follow-up (tooth-fall bleeding, periods) is answered नहीं when she says no. पता नहीं stays unknown, and only when that is what she said.

## Fourth pass

`179` unit tests. 21 scenarios, 0 failed, including a call whose NICU answer is “जी हाँ” and whose other answers are “नहीं जी”.

A no phrase is a no: “नहीं जी” on ever-pregnant stores false and skips miscarriage. “जी हाँ” on a family topic counts as a family yes, so the same-problem question is not hidden. “वजह पता नहीं”, “मुझे पता नहीं है”, and “याद नहीं है” close a follow-up as unknown. “दो दिन पता नहीं” keeps the number, so it is not wiped to unknown.

A fact said in the same sentence is stored. “ऑपरेशन हुआ, पानी कम होने की वजह से, अस्पताल में” fills delivery, place, and the reason, and does not ask those again. “पीलिया की वजह से चार दिन” fills jaundice and 4 days. Two reasons in one sentence (“पीलिया और झटके”) do not pick one. `180` unit tests.

## Fifth pass, 24 scenarios, 0 failed

दूसरा is the second child by age: the younger of two, the middle of three. बीच वाला is the middle. तीसरा is the third. छोटा stays the youngest. An unmarried woman with a past pregnancy can record a miscarriage, and cousin-marriage stays not applicable. A boy of 11 is asked about puberty and tooth-fall bleeding, and is not asked about periods. Periods stay limited to a daughter of 11 or older.

लड़की, बेटी, and बच्ची are female. लड़का and बेटा are male. A daughter stored with the Hindi word is asked about periods at 11. A son stored as लड़का is not.

Pregnancy BP and sugar, and whether they continued after delivery, are stored when she says them with the answer. A kinship word in the same sentence (मामा) closes the relationship follow-up.

When a child's sex is known, the report says बड़ी बेटी, बड़ा बेटा, छोटी बेटी, or दूसरी बेटी. Without a sex it still says बड़ा बच्चा or छोटा बच्चा. One PDF of a per-child NICU finding rendered (about 40 KB).

A family answer whose value is the relationship (कज़िन, मामा) is a yes. “नहीं, मामा को नहीं” stays a no.

Heavy periods said after a no on the general bleeding question are still a हाँ. A periods answer of नहीं leaves the topic a no. The English word heavy is printed as ज़्यादा, not a phonetic spelling. A count she already said (दो बार) closes the times follow-up. ठीक है, theek hai, and कोई दिक्कत नहीं are nos, so a daughter of 11 is still asked about periods. Ages in the mother report use the words she used: 6 महीने stays 6 महीने, 0.5 years is आधा साल, 1.5 is डेढ़ साल, 3.5 is साढ़े 3 साल.

A bleeding or bruise finding is not asked as a swollen knee. Lost skills, puberty, limb pain, lumps, proportion, thumb, and private parts each get their own question instead of the generic growth sentence.

पूरे नौ महीने / समय पर is not a red flag even when the stored value is yes. सात महीने or समय से पहले stays early. दस महीने or देर से stays late. A child's present age (छह महीने का है) is not read as the pregnancy length.

A day or time count copied into the follow-up as a sentence ("दो दिन पता नहीं") is stored as the number. The पता नहीं in that sentence does not wipe the count. A whole call answered with ठीक है stays a no.

सवा साल and सवा महीना are one and a quarter when no number follows सवा. The report prints that as सवा साल / सवा महीना, and the same fraction words for months (डेढ़ महीने, ढाई महीने). The who line must show the formatted age.

दो हफ़्ते का है and पंद्रह दिन की है are ages. दो दिन रहा and दो दिन का बुखार are not. A fifteen-day NICU stay stays a day count on that finding. Number words through तीस do not eat ordinals (तीस must not turn तीसरे into a number).

आठ, छह और चार साल is three ages that share one unit. एक आधा साल stays one age. छह महीने और दो साल stays one of each unit.

शादी हुई है and शादी हो चुकी है are married. शादी नहीं हुई and शादीशुदा नहीं are unmarried. A denial beats the marriage word.

A later utterance replaces the delivery type. ऑपरेशन after a stored normal delivery is an operation. ऑपरेशन नहीं, नॉर्मल हुआ stays normal. दो हफ़्ते और एक हफ़्ता का है is two ages. A stay (दो दिन रहा) is still not an age.

The same rule covers term, NICU reason, and a day count. पूरे नौ महीने after an early term is full term. पीलिया नहीं, झटके replaces jaundice. पंद्रह दिन replaces a stored 2. A follow-up that does not name the fact leaves it.

A later hearing, scan, thalassemia, growth, or puberty utterance replaces the stored result. बीपी नहीं, शुगर replaces the pregnancy condition. अब नहीं रहा replaces “it continued”. नहीं denies only the word beside it, not the next clause.

मेरा नाम मीरा है is stored as मीरा. A bare name is left as it is.

A list stored on a finding is printed as its items. Pregnancy BP and sugar are बीपी, शुगर, not a Python list.

shaadi nahi hui is unmarried, the same denial as शादी नहीं हुई. shaadi hui hai is married.

छह महीने stored with a year unit becomes 6 months. दो हफ़्ते and पंद्रह दिन stored as years become weeks and days. दो साल stays years. A number she used for two units is left as sent. `187` unit tests. 39 scenarios, 0 failed.

## Not hardcoded

No caller names, call ids, or scenario sentences in `call_state.py` / `mother_report.py` / `summarizer.py`. No new list whose only job is one failing sentence. `ओपेरतिओन` is not treated as success. Half-step ages stay numeric (1.5 साल, 3.5 साल) rather than a special label for one sentence. A later bare child count still does not replace children who already have ages. Normal delivery stays in the नहीं column.
