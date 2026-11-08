# Systematic failure modes

Date: 2026-11-08
Evidence: calls 38–44, wave 1 (calls 45–94), wave 2 (calls 95–144)
Fix commit sits on `9b333fe` (the rewind point for the empty-list wipe).

## The class

Calls 38–43 looked like several bugs. Wave 1 failed 33 of 50 calls. Wave 2, aimed at the paths wave 1 never reached, failed 35 of 50. The counts are not 68 different defects.

One contract produced them. The server stores a finished fact, or it stores nothing. An unfinished children answer was treated as a successful write that saved nothing. The next tool batch was then a clinical topic, every field in it was skipped, and the batch raised `no applicable update`. The server stayed on `intro_children`. A bare number was also a finished age, so "six months" became 6 years. A free-text marriage value was a finished status, so the spouse rule never saw `unmarried`. A field the parser did not recognize aborted the batch before any sibling was kept.

No topic was wrongly marked `not_applicable` in either wave. The calls 38–43 wipe, where an empty list meant "no children", did not come back.

## Modes

**1. An unfinished child answer blocks every later fact.**
A bare count (`2`, or the words "दो बच्चे") was acknowledged and not stored. `intro_children` stayed `not_asked`. Gemini had already moved on and sent `topic.birth_mode` or another clinical field. `_topic_can_record` refused it because children were not answered. If that was the whole batch, `apply_updates` raised `no applicable update` and the live call rejected the batch. Wave 1: 26 calls with an empty child list, 16 of them with that tool error. Wave 2: 30 empty lists, 16 with that error. Two wave 2 timeouts (calls 113 and 126) were the same stall.

**2. One bad field deletes the batch.**
`_check_update` ran on every update before any write and raised on the first unknown name. Gemini sent `fam_infant_death`, `fam_miscarriage`, `birth_mode`, and `fam_related` without the `topic.` prefix (calls 93, 108, 135). The fact was dropped. A mixed batch would have dropped the good fields with it. The same raise happened for one unknown child id.

**3. Marriage is not an enum.**
`false` was fixed in the rewind commit. The live model then stored `नहीं हुई` (call 53) and `not_married` (call 75). Neither is `unmarried`, so `fam_related` stayed open. The topic path `topic.intro_married` also copied the raw value and skipped the normalizer.

**4. A bare age is years.**
Call 63 stored "six months" as age `6`. `_child_age_years` reads a bare number as years. The same shape is what a count uses, so the two modes are one: an incomplete value is stored as a completed one.

**5. A correction is refused once the server has moved on.**
Call 61 saved `birth_mode` as `normal`, then refused a later `doctor advised` update because that topic was no longer the current question and its state was not `not_asked`. The only field in the batch, so the call raised `no applicable update`. The value she corrected was thrown away.

**6. "Normal" is a positive finding, and "operation" does not answer the delivery follow-up.**
Calls 71 and 130 stored hearing `true` for "the test was normal". `_is_positive` treats `true` as a problem. Calls 95 and 109 stored birth `operation` and still had the delivery-type follow-up open, because the follow-up reads `details` and the answer was only in `value`.

Kinship while a child already existed (calls 99 and 101) and the miscarriage sentence (call 107) were not tool calls. Those are the model not sending `record_answers`. They are not a separate storage rule. The empty-list mode is what stopped the rest of those paths from being tested.

## The contract

A fact she stated is kept. One field that cannot be stored does not delete the others. An unfinished children answer is not "no children", and it does not block a later fact.

- A count opens that many child slots and does not answer `intro_children` until ages exist. Ages spoken in the same turn ("छह महीने", "दो साल", "आधा साल") fill the slots. A month-only utterance stores a bare number as months, not years. A count does not replace a list that already has ages.
- Profile updates in a batch are applied before topic updates, so the slots exist before the clinical field is judged.
- A clinical topic is kept once any child slot exists, or once `intro_children` is answered. A clinical topic during the name question, before any child slot, is still refused.
- A correction to a topic she already reached is kept after the server has moved on.
- A field may be `topic.<id>`, `<id>`, `profile.<field>`, or `<field>`. An unknown name is skipped. An unknown child id is skipped. The batch raises only when nothing was stored.
- `marital_status` is only `married`, `unmarried`, or `about_to_marry`. `नहीं`, `not_married`, and `false` are unmarried. The profile field and `topic.intro_married` use the same normalizer. Anything else does not close the question.
- Birth value `operation` or `normal` fills `details.delivery_type`, so that follow-up is closed.
- Hearing (and any topic) with `details.result` `normal` is not a positive finding. The words "normal" / "नॉर्मल" set that result.

`gemini_live.py` passes the mother's words into `apply_updates` as `utterance`. That file's other uncommitted diff is the live session itself, so this commit does not take the whole session. The default utterance is empty, and the unit tests pass the words in directly.

## Verification

Unit tests, from `code/shishu-v2`:

`PYTHONPATH=code/shishu-v2 /Users/jigar/pyenv/shishu/bin/python -m unittest test_call_state test_call_state_db test_answer_tool test_structured_summaries test_structured_mother_report test_bugfixes test_triage_flow test_mother_report test_live_text_harness`

138 tests, OK.

Live reruns on the restarted server, after the contract change. A reply was sent only after a spoken `turn_complete`.

| Scenario | Call | Result |
|---|---|---|
| One child, age 2, then a normal hospital birth | 145, then 148 | Call 145 saved the child (age 2) and had 0 tool errors. The model repeated "normal delivery, hospital" and did not send a birth tool. Call 148, the same script, saved age 2, `birth_mode` `normal`, `delivery_type` `normal`, 0 tool errors, close `caller_stop`. |
| "शादी नहीं हुई", no children, never pregnant | 146 | `marital_status` `unmarried`. `fam_related` is `not_applicable` (`no_spouse`). `ever_pregnant` false. 0 tool errors. |
| "एक बच्चा है, छह महीने का।" | 147 | Age `{value: 6, unit: months}` (0.5 years), not 6 years. 0 tool errors. Close `caller_stop`. |
