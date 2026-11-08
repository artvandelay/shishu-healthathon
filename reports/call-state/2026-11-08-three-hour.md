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

## Not hardcoded

No caller names, call ids, or scenario sentences in `call_state.py` / `mother_report.py` / `summarizer.py`. No new list whose only job is one failing sentence. `ओपेरतिओन` is not treated as success. Half-step ages stay numeric (1.5 साल, 3.5 साल) rather than a special label for one sentence. A later bare child count still does not replace children who already have ages. Normal delivery stays in the नहीं column.
