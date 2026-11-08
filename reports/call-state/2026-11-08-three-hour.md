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

छह महीने stored with a year unit becomes 6 months. दो हफ़्ते and पंद्रह दिन stored as years become weeks and days. दो साल stays years. A number she used for two units is left as sent.

A stored day count prints as 4 दिन and a stored time count as 3 बार.

चौथे वाले is the fourth child by age, the same rule as दूसरा and तीसरा.

नाम आरव on the child sentence is stored as the child's name. मेरा नाम stays the caller's name and is not copied onto the child.

आठ साल का एक बच्चा is one child aged eight, not eight children. The count is the number beside बच्चा, including बेटे and बेटियाँ.

दो बेटे and दो बेटियाँ are both boys or both girls. One बेटी among several ages still does not label every child.

दो बार गर्भ गिरा, तीन महीने में stores the count and the month, and does not ask them again. An age said with blindness or a preterm birth fills that age follow-up (30 साल, 7 महीने).

7, 8, or 9 महीने use the same early, late, and full-term rule as the Hindi words. 9 महीने में is full term. 6 महीने का है is still the child's age now.

A call with no children does not ask about मेरे बच्चे की उम्र, टीके, or the child's daily food. Those fillers are for a child she has.

An age of 6.0 months prints as 6 महीने, and 10.0 years prints as 10 years.

लड़का छह महीने is one six-month-old boy, not six children. एक लड़का और दो लड़कियाँ is three children, and the number on each sex word is that sex. एक भी बच्चा नहीं and एक बच्चा भी नहीं are no children. मेरा बच्चा नहीं रोता is not a count of zero. A count that still has no ages leaves the children question open.

abnormal is not a normal delivery, and घरेलू is not a home birth. not normal does not store a normal delivery. An unknown lowercase English word stays in Latin. A capitalized name such as Shilpa is still written in Hindi.

A pregnancy month stored as 3 prints as 3 महीने, the same way a day count prints as दिन and a time count prints as बार.

The only child is stored on a finding that asks which child, so the report says her name or बेटी/बेटा. A family finding such as a miscarriage is not pinned to that child. A no drops the binding.

सबसे बड़े is the oldest child. सब is not found inside सबसे, so the younger child is not included. दोनों and सबको still mean every child.

जी क्या मतलब, मुझे समझ नहीं आया, and फिर से बोलिए leave the follow-up open. चार दिन and नहीं are still answers.

शादी है is married. जाँच हुई, बात हुई, and हो गई by themselves are not. A denial still wins.

लगभग चार साल, करीब छह महीने, 4.5 साल, and दो महीने are one child of that age. A bare number is a count only when she did not also say an age. बड़ा वाला आठ साल और छोटा वाला तीन साल is two children.

A later sentence that states its own ages replaces the child list. The same ages said again keep the sex and name. A bare count still does not replace children who already have ages.

The number next to the event is the one stored. NICU days skip N दिन का बुखार and keep the stay. A pregnancy month skips N महीने का and keeps N महीने में. जैसा नहीं बढ़ रहा is a growth finding even when the tool sent false. A plain नहीं stays a no.

पता नहीं is unknown, not a no. It does not skip pregnancy history, and it does not land in the नहीं column. A plain नहीं still does. मालूम नहीं, नहीं मालूम, and नही without the bindu are the same unknown. `195` unit tests. 51 scenarios, 0 failed.

A sentence that names the problem is a finding even without हाँ. झटके आते हैं, होंठ नीले पड़ते हैं, भर्ती हुआ, गर्भ गिर गया, खून चढ़ा, होंठ कटा, बीपी था, and पीलिया की वजह से चार दिन are हाँ lines. A tool value of false does not hide that sentence. नहीं, नॉर्मल है, and सामान्य है stay no. अस्पताल में हुई is not an operation. सामान्य हुआ is a normal delivery. बाकी बच्चों जैसा बढ़ रहा है and a test that was only done stay out of the हाँ list. A sentence that says both हाँ and नहीं is not promoted. `196` unit tests. 52 scenarios, 0 failed.

Questions whose own yes contains नहीं were stored as a no when she repeated that negative. अब नहीं कर पाता, वैसे नहीं दिखते, and अंगूठा नहीं है are findings. नहीं। and नहीं जी stay no. झटके नहीं आते stays no. छोटी नहीं denies the property, so it stays no. `196` unit tests. 53 scenarios, 0 failed.

A symptom's own duration is not the hospital stay. दो दिन से बुखार does not become NICU days. चार दिन रखा and a bare चार दिन still do. दो हफ्ते रखा is 14 days. दो हफ्ते का है stays the child's age. `196` unit tests. 53 scenarios, 0 failed.

One age tied to a counted group is every child's age. तीन साल की दो बेटियाँ and दो बेटे हैं, दोनों चार साल के store that age on each child. मेरी उम्र तीस साल और दो बच्चे हैं does not. `196` unit tests. 54 scenarios, 0 failed.

ठीक से and ठीक रहा are reassurance, the same as ठीक है. निशान ठीक से भरा है is a no even when the tool sent true. A question that itself asks about ठीक, such as joint swelling that settles, is unchanged. ठीक है। alone does not erase a finding logged on that turn. `196` unit tests. 54 scenarios, 0 failed.

A bare integer is the age when the sentence states ages and does not state that many children. छह महीने sent as 6 is one child of six months, not six children. आधा साल sent as 0.5 is one child, not zero. आठ साल का एक बच्चा sent as 8 is one child. दो बच्चे हैं, चार और छह साल के दो बेटे sent as 2 stays two children. `196` unit tests. 55 scenarios, 0 failed.

A complaint at the close is a finding. पेट में दर्द रहता है is a हाँ line and the words are on that line. बस।, नहीं बस इतना ही, and और कुछ नहीं are not. `196` unit tests. 56 scenarios, 0 failed.

A cord that fell in one or two weeks is not "more than three weeks." तीन हफ्ते से ज़्यादा and चार हफ्ते stay findings. एक हफ्ते में गिर गई is a no even when the tool sent true. `196` unit tests. 56 scenarios, 0 failed.

Product note, not a synonym list: भाई को बचपन से नहीं दिखता is stored as a no on the blindness question, because नहीं denies the question and the question's word is अंधापन. कम सुनाई देता है is a finding. A new blindness lexicon was not added.

A second child name does not need the word नाम again. बड़े वाले का नाम आरव है, छोटे वाली ईशा है stores both names. मेरा नाम stays the caller. Number words, sex words, and वाले are not names. `196` unit tests. 56 scenarios, 0 failed.

तीन महीने में is a pregnancy month, not a second child. बच्चा छह महीने का है और गर्भ तीन महीने में गिरा stores one child of six months. छह महीने का stays an age. `196` unit tests. 56 scenarios, 0 failed.

जी। by itself is an acknowledgement. It is not a हाँ on a clinical topic, and it does not turn a follow-up into a finding. जी हाँ and जी नहीं keep their polarity. `196` unit tests. 56 scenarios, 0 failed.

One month is 1 महीना, the same singular rule a week already uses. Two months stay 2 महीने. A quarter of a year is 3 महीने, not 0.25 years. `196` unit tests. 57 scenarios, 0 failed.

दोनों or जुड़वाँ with one age is two children of that age. जुड़वाँ हैं, दोनों एक साल के is two one-year-olds. बेटा और बेटी, दोनों दो साल के keeps both sexes. An explicit count still wins. `196` unit tests. 58 scenarios, 0 failed.

A half month of pregnancy is kept. साढ़े सात महीने में is early, and the amount is साढ़े 7 महीने, not 8. ढाई महीने पहले is how long ago, not a pregnancy month. Under nine months is early. Nine is full. `196` unit tests. 58 scenarios, 0 failed.

A nukta does not make a different word. सीज़ेरियन is an operation, the same as सीजेरियन. घरपे is home. घरेलू is still not घर, and सब is still not inside सबसे. `196` unit tests. 58 scenarios, 0 failed.

दूसरी बार is two times. पहली बार is one. The ordinal is a count only next to बार, so दूसरा बच्चा is still not two children. `196` unit tests. 58 scenarios, 0 failed.

दो साल छह महीने का is one child of ढाई साल. दो साल का और छह महीने का stays two children. पौने साल is three quarters of a year. आधे साल is half a year. `196` unit tests. 59 scenarios, 0 failed.

सी-सेक्शन, सी सेक्शन, and सिजेरियन are the same operation as सीजेरियन. One word list, used for the finding, the follow-up, and the stored delivery type. `196` unit tests. 59 scenarios, 0 failed.

आठवें महीने में is the eighth month, so it is early. नौवें महीने में is full term. तीसरे साल is an age of 3. सवा हफ्ता is one and a quarter weeks. पहले stays "before" unless a month, year, or week follows, so समय से पहले is unchanged. दूसरे बच्चे is not two children. `196` unit tests. 59 scenarios, 0 failed.

A sentence that says no overrides a tool yes. कोई रिश्ता नहीं is not a finding, and the relationship follow-up does not stay open. ठीक है। still does not erase a finding logged on that turn. पता नहीं on a follow-up still does not clear the topic. A sentence that names the problem (झटके थे, शुगर था) stays a yes. चाचे and चाचाजी are the same relative as चाचा. मामला is not मामा, and रिश्तेदारी is not a named relative. `196` unit tests. 59 scenarios, 0 failed.

A fraction of a week is said the same way as a fraction of a year. सवा हफ़्ता, डेढ़ हफ़्ते, ढाई हफ़्ते, आधा हफ़्ता. Two whole weeks stay 2 हफ़्ते. `196` unit tests. 59 scenarios, 0 failed.

प्रसव में खून बह गया is a delivery problem, and that sentence is the problem, so it is not asked again. प्रसव हो गया only says the delivery happened, so it is not a finding. प्रसव ठीक रहा stays a no. `196` unit tests. 59 scenarios, 0 failed.

दूसरी प्रेग्नेंसी names which pregnancy, so that follow-up is not asked again. पहली बार on a pregnancy question is the first pregnancy. A sentence with no pregnancy word does not invent one. `196` unit tests. 59 scenarios, 0 failed.

डेढ़ दिन रखा is a day and a half, not one day and not two. साढ़े सात दिन stays 7.5 and is shown as साढ़े 7 दिन. A whole number of days stays N दिन. `196` unit tests. 59 scenarios, 0 failed.

Trio: पौने साल का बेटा (no flags), आठवें महीने में पैदा हुई (early, tool had said no), डेढ़ दिन का NICU with पीलिया, and दूसरी प्रेग्नेंसी में प्रसव में खून. Pass. 63 scenarios, 0 failed. `196` unit tests.

डेढ़ बच्चा is not five children, and सवा दो बच्चे is not twenty-five. The digits after a decimal point are not a count. चारों बच्चे is four. तीनों means every child. बारह बच्चे is still twelve. सबसे बड़े is still only the oldest. `196` unit tests. 63 scenarios, 0 failed.

A pregnancy number is दूसरी प्रेग्नेंसी, not a bare 2. When the sentence already says it, the report does not repeat it. `196` unit tests. 63 scenarios, 0 failed.

A bare डेढ़ or 1.5 is not one child. The same decimal rule covers a number with no child word beside it. `196` unit tests. 63 scenarios, 0 failed.

Trio: चारों बच्चे with ages 2, 4, 6 and 8 (no flags), दसवें महीने में is late, and झटके on the पौने साल child bind to the younger child only. Pass. 66 scenarios, 0 failed. `196` unit tests.

दो हफ्ते तीन दिन का is one child of 17 days. दो हफ्ते और तीन दिन stays two ages. दो हफ्ते तीन दिन रखा is a stay, not an age. तीन महीने दस दिन का is one child, shown as 3 महीने 10 दिन. एक साल दस दिन का keeps the ten days on the year. `196` unit tests. 68 scenarios, 0 failed.

भैया, दीदी, भाभी, पोता, and साला are relatives, the same as भाई or चाचा. दोस्त is not. पोते matches पोता. `196` unit tests. 69 scenarios, 0 failed.

झटके in the same sentence is the family-fits condition, so that follow-up is not asked again. देर is delay. भूल or नहीं कर is a lost skill. `196` unit tests. 69 scenarios, 0 failed.

## Resume 16:03 IST

Pulled `origin/refactor/deterministic-call-state` at 16:00 IST. HEAD was `19cc09c`. The frontier gate was already present: `SUMMARY_MODEL_CHAIN` is `gemini-3.1-pro-preview`, then `gemini-3-pro-preview`, then `gemini-2.5-pro`, and `_finalize_structured_summary` uses the model findings and flags. Invalid JSON or no key uses `build_state_fallback`. Flash is not that chain.

ठीक से was already a no. `_REASSURE_RE` contains ठीक से, and a question that itself contains ठीक (joint swelling that settles) is left as a finding. निशान ठीक से भरा है with a tool yes is kind no. The class that was still wrong: a ठीक clause cleared the whole sentence, so झटके आते हैं, अब ठीक है became a no. Reassurance now covers only the clause that says it is fine. A sentence that is only reassurance, including सब ठीक है sent as true, stays a no. The digits after a decimal are not a count of times or a pregnancy number. डेढ़ बार is not 5, and 2.5 प्रेग्नेंसी is not 5. एक साल दो हफ्ते का is one child, with those two weeks kept as 14 days. `196` unit tests. 72 scenarios, 0 failed.

Trio: एक साल दो हफ्ते (no flags), झटके now ठीक (still a yes), and सब ठीक है sent as true on every clinical answer (no हाँ lines). Pass.

## 16:07 IST

A bare नहीं does not erase a different clause. और कुछ नहीं, पेट में दर्द रहता है stays a complaint. मेरे बच्चे को नहीं stays a no even when the other clause says the problem exists for someone else, and झटके नहीं आते stays a no. दो चार और छह साल is three ages: और before only the last number still joins the list. एक आधा साल stays half a year, the existing test. `196` unit tests. 75 scenarios, 0 failed.

Trio: ages with और only before the last (no flags, three children), pain after और कुछ नहीं (still a yes), and नहीं, बस इतना ही sent as true (no हाँ lines). Pass.

## 16:08 IST

A danda is a clause boundary, the same as a comma. और कुछ नहीं। पेट में दर्द रहता है stays a complaint. नहीं। बस इतना ही। stays a no. `196` unit tests. 78 scenarios, 0 failed.

Trio: एक बेटा है। दो साल का है। (no flags), pain after a sentence break (still a yes), and नहीं। बस इतना ही। sent as true (no हाँ lines). Pass.

## 16:10 IST

बेटा और बेटी दोनों चार साल के is two children of that age. The दोनों rule now runs after a numbered tool value is set aside, because the sentence itself has no numbered count. पति की उम्र तीस साल is not a child; the बेटा एक साल का beside it is. उम्र पाँच, तीन और एक साल after the children stays theirs. चार साल का था जब is when the fits happened, so the child stored is the current छह साल. `196` unit tests. 81 scenarios, 0 failed.

Trio: son and daughter both four (no flags), fits at four and now six (still a yes, one child of six), and the husband's 30 sent as the child count (one boy of one year, no हाँ lines). Pass.

## 16:12 IST

दोनों बेटियाँ चार साल की हैं is two girls of four. दोनों is already the count two, and it is not folded to a digit, so the shared-age check now treats that word as two. पाँच साल का था, अब सात साल का है is one child, now seven. दो साल का था, with no later अब, stays two. `196` unit tests. 84 scenarios, 0 failed.

Trio: both daughters four (no flags), was five and now seven with fits (one child of seven, still a yes), and the old age 5 sent as the count (one child of seven, no हाँ lines). Pass.

## 16:13 IST

हो सकता है and a bare शायद are the same blank as पता नहीं. शायद झटके होते हैं still names the fits, so it stays a yes. `196` unit tests. 87 scenarios, 0 failed.

Trio: one boy of eight months (no flags), maybe-fits that names झटके (still a yes), and हो सकता है sent as true on every clinical answer (no हाँ lines). Pass.

## 16:15 IST

दो साल, चार और छह shares the साल on the first number. दो साल का और छह महीने का stays two ages, because the second number has its own unit. A number word before a danda is still that number. याद नहीं in a later clause does not deny the stay: NICU में रहा, कितने दिन याद नहीं stays a yes. कुछ नहीं। खाँसी है। stays a complaint. `196` unit tests. 90 scenarios, 0 failed.

Trio: three children at two, four, and six with the unit only on the first (no flags), NICU with the days unknown (still a yes), and cough after कुछ नहीं (still a yes). Pass.

## 16:16 IST

एक बार खून चढ़ा, दूसरी बार नहीं stays a transfusion, once. A denial names her child only when the clause says बच्चा, so मेरे बच्चे को नहीं is still a no. नहीं होता, सब ठीक है stays a no, because the other clause is reassurance. `196` unit tests. 93 scenarios, 0 failed.

Trio: नहीं होता, सब ठीक है sent as true (no flags, no हाँ lines), transfusion once (still a yes, count 1), and the neighbor's child (still a no). Pass.

## 16:17 IST

भांजा दो साल का is the nephew, not a child. मेरा भांजा दो साल का है, मेरी बेटी चार साल की है stores the daughter, four, female. The nearer child word still keeps उम्र पाँच, तीन और एक after the children. `196` unit tests. 96 scenarios, 0 failed.

Trio: the daughter and not the nephew (no flags), fits for that daughter (still a yes), and her age 4 sent as the count (one girl of four, no हाँ lines). Pass.

## 16:22 IST

तीन साल का बेटा और बेटी हैं is two children of three. One sex word is still one child. शादी टूट गई, तलाक, and विधवा are not stored as married. `196` unit tests. 99 scenarios, 0 failed.

Trio: son and daughter of one age (no flags), odor for that family (still a yes), and the age 3 sent as the count (two children of three, no हाँ lines). Pass.

## 16:25 IST

दो बेटे हैं तीन और पाँच के is two boys, three and five. के without a unit is years. चार साल के stays four years, and छह महीने के stays months. दो-दो साल repeats that age for the children just named: one boy of four and two girls of two. `196` unit tests. 102 scenarios, 0 failed.

Trio: sons at three and five with no साल (no flags), fits with दो-दो (still a yes, three children), and the boy's 4 sent as the count (still three children, no हाँ lines). Pass.

## 16:28 IST

लड़के को झटके, when only one child is a boy, is that boy. The report names him (बड़ा बेटा) and does not ask which child. Two sexes in one sentence stay unbound. बच्चा still does not pick a sex. `197` unit tests. 102 scenarios, 0 failed. The दो-दो fits scenario is the one that now names the boy.

## 16:32 IST

तीन साल का एक और एक साल की एक is two children, three and one. The एक after का is one child of that age, not another age in the list. आठ, छह और चार साल still has the unit at the end. `197` unit tests. 105 scenarios, 0 failed.

Trio: one of three and one of one (no flags), blue lips for the elder (still a yes), and 3 sent as the count (still two children, no हाँ lines). Pass.

## 16:24 IST

दो घंटे and बीस मिनट are not a count of children. सास के दो बच्चे are not hers; मेरे एक बेटा है तीन साल का is one boy of three. जुड़वाँ, एक लड़का एक लड़की, दोनों एक साल के is those two children, both one. `197` unit tests. 108 scenarios, 0 failed.

Trio: twins, boy and girl, both one (no flags), fits for her son and not the mother-in-law's children (still a yes), and the mother-in-law's 2 sent as the count (one boy of three, no हाँ lines). Pass.

## 16:26 IST

नाम बता रही हूँ does not name a child बता. The names said before that stay unset unless नाम introduces them. नाम राम है, और सीता does store राम and सीता, and राम को झटके binds राम. An onset एक साल से does not choose the one-year-old. `197` unit tests. 111 scenarios, 0 failed.

Trio: नाम बता रही हूँ (no flags, no child named बता), राम named with नाम and fits (the yes line names राम), and सीता's 4 sent as the count (still two children, unnamed, no हाँ lines). Pass.

## 16:27 IST

स्कैन हुआ था, कुछ नहीं निकला is a normal scan, not a finding. कुछ नहीं दिखा was already a no. स्कैन में दिल में छेद बताया stays a yes. `197` unit tests. 114 scenarios, 0 failed.

Trio: scan found nothing (no flags, no हाँ lines), a hole on the scan (still a yes), and कुछ नहीं दिखा sent as true (no हाँ lines). Pass.

## 16:28 IST

सुनाई नहीं देता on the newborn hearing question is an abnormal result, and so is दिखाई नहीं देता. सुनने की जाँच हुई और नॉर्मल थी stays a no. `197` unit tests. 117 scenarios, 0 failed.

Trio: a normal hearing test (no flags), cannot hear (still a yes), and cannot see sent as false (still a yes). Pass.

## 16:29 IST

पति के पहले शादी से दो बच्चे are not hers. मेरे एक बेटी है चार साल की is one girl of four. तीन किलो का पैदा हुआ is not three children, and तीन किलो का पैदा हुआ, अब एक साल का बेटा है is one boy of one. `197` unit tests. 120 scenarios, 0 failed.

Trio: her daughter, not his earlier children (no flags), blue lips for that daughter (still a yes), and the birth weight 3 sent as the count (one boy of one, no हाँ lines). Pass.

## 16:33 IST

पेशाब या पाखाना: she named one, and said the other is fine, so the source is पेशाब and that follow-up is not asked again. Both named stays open. पेशाब और पाखाना दोनों ठीक हैं is one reassurance (the plural हैं, and दोनों joins the options that और had split). झटके आते हैं और अब ठीक हैं stays a yes. A lump's सूजन या गाँठ is not stored as the place. `197` unit tests. 123 scenarios, 0 failed.

Trio: both are fine (no flags, no हाँ lines), urine smell with stool fine (still a yes, the line names the boy and पेशाब), and both smell (still a yes). Pass.

## 16:36 IST

The same choice is stored for the other questions that offer it. खून कम या ज़्यादा stores कम. घुटने या जोड़ stores घुटने. अंगूठा या उंगली stores अंगूठा. सूजन या गाँठ is still not the place. घुटने में सूजन नहीं है stays a no. `197` unit tests. 126 scenarios, 0 failed.

Trio: the knee denied (no flags, no हाँ lines), blood count low (still a yes, the line names कम), and a missing thumb sent as false (still a yes, the line names अंगूठा). Pass.

## 16:37 IST

डॉक्टर ने कहा नॉर्मल है is the result, not a finding. खून कम है, बाकी नॉर्मल है stays a yes, and so does झटके आते हैं, बाकी नॉर्मल है. सात महीने में पैदा हुआ, नॉर्मल है stays an early birth. `197` unit tests. 129 scenarios, 0 failed.

Trio: a normal blood count (no flags, no हाँ lines), low count plus normal otherwise (still a yes), and an early birth that she also called normal (still a yes, 7 महीने). Pass.

## 16:39 IST

उन्नीस, चालीस, पचास, साठ, सत्तर, अस्सी, and नब्बे are ages, the same way बीस and तीस already were. पापा की उम्र चालीस, sent as 40, is his age: the boy उन्नीस साल का is the one child. `197` unit tests. 132 scenarios, 0 failed.

Trio: a son of nineteen (no flags), father's sugar at forty (still a yes, the line keeps 40 साल), and 40 sent as the child count (one boy of nineteen, no हाँ lines). Pass.

## 16:40 IST

बीपी, शुगर, and लकवा on the family question are the condition, the same way शुगर already was in pregnancy. शुगर नहीं है stays a no. Both बीपी and शुगर stay one yes. `197` unit tests. 135 scenarios, 0 failed.

Trio: no sugar (no flags, no हाँ lines), father's stroke (still a yes), and BP plus sugar (still a yes, the line names both). Pass.

## 16:41 IST

माँ की तरफ: ज़्यादा खून, इंफेक्शन, and खुजली are the condition. खुजली नहीं है stays a no. The report says खुजली, not the English code. `197` unit tests. 138 scenarios, 0 failed.

Trio: no itch (no flags, no हाँ lines), maternal skin patches (still a yes), and infection plus bleeding (still a yes). Pass.

## 16:42 IST

खाना छूटने पर झटके is the fasting symptom. बैठना देर से is the milestone type. नील is the bruise, खून on that question is the bleeding. उल्टी नहीं होती stays a no. The report says झटके and बैठना. `197` unit tests. 141 scenarios, 0 failed.

Trio: no vomiting (no flags, no हाँ lines), fits when a meal is missed (still a yes), and late sitting sent as false (still a yes, the line names बैठना). Pass.

## 16:43 IST

पहले बोलता था, अब नहीं बोलता names the lost skill. अब नहीं चलता only denies it, so it stays a no. The report says बोलना and चलना. `197` unit tests. 144 scenarios, 0 failed.

Trio: he does not walk, with no earlier skill (no flags, no हाँ lines), stopped talking (still a yes), and stopped walking sent as false (still a yes). Pass.

## 16:44 IST

होंठ या उंगलियाँ is the place. सूजन या गाँठ is still not a place. नीली नहीं पड़ती stays a no. `197` unit tests. 147 scenarios, 0 failed.

Trio: not blue (no flags, no हाँ lines), blue fingers (still a yes, the line names उंगलियाँ), and blue lips sent as false (still a yes, the line names होंठ). Pass.

## 16:45 IST

हाथ-पैर is the place, the same way होंठ या उंगलियाँ is. बार-बार is not a place. दर्द होता है लेकिन अस्पताल नहीं गए stays a no, because the question is the pain that needed a hospital. `197` unit tests. 150 scenarios, 0 failed.

Trio: pain without a hospital visit (no flags, no हाँ lines), leg pain that needed hospital (still a yes, the line names पैर), and hand pain sent as false (still a yes, the line names हाथ). Pass.

## 16:47 IST

समय से दो महीने पहले is early by two months. समय से दो हफ्ते बाद is late by two weeks. समय से पहले, with no number, is still early and the amount stays unasked. समय पर stays a no. `197` unit tests. 153 scenarios, 0 failed.

Trio: born on time (no flags, no हाँ lines), two months early (still a yes, the line says 2 महीने), and two weeks late sent as false (still a yes, the line says 2 हफ्ते). Pass.

## 16:47 IST

हाथ-पैर on the proportion question is the body part, the same choice as the limb. हाथ छोटे नहीं हैं stays a no. `197` unit tests. 156 scenarios, 0 failed.

Trio: arms not short (no flags, no हाँ lines), short arms (still a yes, the line names हाथ), and short legs sent as false (still a yes, the line names पैर). Pass.

## 16:48 IST

बार-बार and अस्पताल गए fill how often and that she went. अस्पताल नहीं गए is a no on that question even when the pain is another clause, because the question is the pain that needed a hospital. `197` unit tests. 159 scenarios, 0 failed.

Trio: pain sometimes and no hospital (no flags, no हाँ lines), pain again and again that needed hospital (still a yes, the line names बार-बार and पैर), and that same sentence sent as false (still a yes). Pass.

## Not hardcoded

No caller names, call ids, or scenario sentences in `call_state.py` / `mother_report.py` / `summarizer.py`. No new list whose only job is one failing sentence. `ओपेरतिओन` is not treated as success. Half-step ages stay numeric (1.5 साल, 3.5 साल) rather than a special label for one sentence. A later bare child count still does not replace children who already have ages. Normal delivery stays in the नहीं column.
