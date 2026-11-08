# Bug: a child count became "no children", and one bad field dropped the batch

Date: 2026-11-08
Calls: 38–44
Server state that failed: before the `call_state.py` rules below. Call 44 is the same script as call 42 after that change.

## What broke

Gemini often sends `profile.children` as a bare count, such as the integer `2`, or it sends a topic the server has already moved on from. Two storage rules then threw the rest of the call away.

1. A bare count was stored as an empty list. An empty list means "she has no children", so the server marked 28 child topics `not_applicable` with reason `no_children` (8 birth, 15 body, 5 illness). `fam_same_problem` was also `not_applicable`, for the separate `same_problem_hidden` rule.
2. One field that did not match the current question raised `topic update outside the current topic` and rejected the whole tool batch. The server stayed on the old question. Later answers were rejected too, or Gemini stopped sending them.
3. Marriage value `"false"` was stored as the string `false`, not `unmarried`. The spouse topic (`fam_related`) stayed open because the skip rule looks for `unmarried`.

A count is not an answer of "no children". `false` / `no` for marriage is unmarried. Once children are known, a fact for a question she has already reached is kept. A clinical fact before the children question is still refused. One bad field does not delete the rest of the batch.

## Calls

| Call | What she said | What was saved | Tool error |
|---|---|---|---|
| 38 | Jigar, married, two children (2 years and 1 year), then normal plus operation | Name and married only. `children` is `[]`. Still on `intro_children`. 28 topics `no_children`. | Turn 9: batch of `topic.birth_mode` = normal and operation, "topic update outside the current topic" |
| 39 | Meera, married, one child age 2, normal hospital birth, then "no" to preterm | Child saved (`age` 2). Bridge answered. Birth mode not saved. Still on `birth_mode` when the call closed. Preterm "no" was asked twice. | Turn 11: `topic.birth_term` = false, outside the current topic |
| 40 | Reena, "shaadi nahi hui", no children, one pregnancy, miscarriage at three months | Name Reena. `marital_status` is `"false"`. Children `[]` is correct here, so the 28 `no_children` marks are correct. Miscarriage was never a tool call. Still on `preg_conditions`. `fam_related` stayed applicable. | none |
| 41 | Pooja, married, children 4 and 1, normal birth, fits in the older child | Both children saved. Birth and fits not saved. Still on `intro_bridge`. | Turn 9: `topic.body_fits_milestones` and `topic.birth_mode` together, outside the current topic |
| 42 | Seema, married, "one child, age 3", then "actually two, ages 3 and 6 months" | Name and married only. `children` is `[]`. Still on `intro_children`. 28 topics `no_children`. The count was accepted, not rejected. | none |
| 43 | Kavita, married, one child age 1, then "the child is not breathing, this is an emergency" | Emergency close did fire. Children were still `[]`, and 28 topics were `no_children`. Bridge was not saved. | Turn 7: `topic.intro_bridge` = true, outside the current topic |
| 44 | Seema again: one child age 3, then two children ages 3 and 6 months, then stop | Two children, ages 3 and 0.5. `intro_children` answered. 0 tool errors. The only `not_applicable` topics are `body_puberty` (`all_under_10`) and `fam_same_problem` (`same_problem_hidden`). Those two are the rules, not false skips. | none |

Calls 38, 42, and 43 are the false "no children" wipe. Calls 38, 39, 41, and 43 are the whole-batch reject. Call 40 is the unmarried value. Call 41 shows both facts in one batch deleted together. Call 44 is the rewind check: two children saved, 0 false skips, 0 tool errors.

## Where the fix lives

`code/shishu-v2/call_state.py`

- `_children_payload`: `0`, `false`, `[]`, or empty means no children. A positive number is only a count. A count does not answer `intro_children` and does not arm `no_children`.
- `_canonical_marital`: `false` / `no` / `nahi` becomes `unmarried`, which arms the spouse skip.
- `_topic_can_record`: a clinical topic before children are known is refused. After children are known, a fact for a question already reached is kept, and a field that does not apply is skipped instead of failing its siblings.
- `test_call_state.py` covers the count, the marital value, the early clinical refusal, and keeping the next fact.

`gemini_live.py` is not in this commit. Its uncommitted diff is the live call-state wiring, not only the removal of the temporary `tool_trace`. That trace is already gone from the working tree. Call 44 ran on this server with the trace removed.
