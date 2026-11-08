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

## Not hardcoded

No caller names, call ids, or scenario sentences in `call_state.py` / `mother_report.py` / `summarizer.py`. No new list whose only job is one failing sentence. `ओपेरतिओन` is not treated as success.
