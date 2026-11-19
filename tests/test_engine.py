"""Behavioral tests for `intake.engine` across synthetic fixtures and both real packs."""

from __future__ import annotations

import unittest
from pathlib import Path

from intake import engine, reports
from intake.engine import (
    MAX_STEP_ATTEMPTS, active_prompt, eval_rule, format_age_label,
    merge_transcript_chunk, new_state, quick_replies, quick_reply_command,
    reduce_turn, roster_rows, which_item_prompt,
)
from intake.models import (
    AnswerRecord, AnswerStatus, CallState, CloseReason, CursorStep, FollowupSlot, NoteText, OptionSpec, PackConfig, Phase,
    ProfilePatch, QuestionKind, QuestionSpec, RosterInput, RosterItem, SubStep,
    TurnCommand, TurnOutcome, VariantSpec,
)
from intake.pack import load_pack

_KEEP = ("group", "say", "stem", "requires_slot", "saves_to_profile", "roster_scoped", "quick_options", "concern_if", "reask")


def make_pack(questions=(), *, meta=None, prompts=None, slots=None, roster=None, rules=None, followups=None) -> PackConfig:
    specs = tuple(
        QuestionSpec(
            id=q["id"],
            kind=q["kind"],
            group=q.get("group", "g"),
            say=q.get("say", q["id"]),
            **{k: q[k] for k in _KEEP if k in q and k not in ("group", "say")},
            **{k: tuple(q[k]) for k in ("skip_if", "followups", "score_rubric") if k in q},
            variants=tuple(VariantSpec(**v) for v in q.get("variants", ())),
            options={k: OptionSpec(**v) for k, v in (q.get("options") or {}).items()},
        )
        for q in questions
    )
    return PackConfig(
        id="test", root=Path("."), meta=meta or {}, prompts=prompts or {},
        profile_slots=slots or {}, roster_cfg=roster, extractor_prompt="",
        implications=(), semantic_flags={}, rules=rules or {}, groups={},
        followups=followups or {}, questions=specs, by_id={s.id: s for s in specs},
        report_cfg={}, decisions={}, seed_scenarios=(),
    )


def gate(qid: str, **kw) -> dict:
    return {"id": qid, "kind": QuestionKind.GATE, **kw}


def play(pack: PackConfig, state: CallState, outcome=TurnOutcome.CLEAR, **kw) -> CallState:
    return reduce_turn(pack, state, TurnCommand(outcome=outcome, **kw))


class EngineSyntheticTests(unittest.TestCase):
    def test_transcript_merge_and_age_units(self):
        self.assertEqual(merge_transcript_chunk("hello", "there"), "hello there")
        self.assertEqual(merge_transcript_chunk("a b", "a b c"), "a b c")
        self.assertEqual(merge_transcript_chunk("हाँ", "।"), "हाँ।")
        tuned = make_pack(prompts={"roster_format": {"month_unit": "महीने", "year_unit": "साल"}})
        self.assertEqual([format_age_label(tuned, a) for a in (0.5, 3.0, 1.5)], ["6 महीने", "3 साल", "1.5 साल"])

    def test_every_rule_shape_resolves_against_live_state(self):
        pack = make_pack(
            [gate("a", group="fam"), gate("b", group="fam"), gate("c")],
            rules={
                "flag_on": {"semantic_flag": "beta"},
                "flagged": {"field": "ever_pregnant", "op": "eq", "value": True},
                "not_flagged": {"field": "ever_pregnant", "op": "ne", "value": True},
                "either": {"any_of": ["flag_on", "flagged"]},
                "both": {"all_of": ["flag_on", "flagged"]},
                "fam_clear": {"group_no_concern": "fam"},
                "kids_in": {"field": "kids", "op": "in", "value": ["none", "some"]},
                "kids_absent": {"field": "kids", "op": "missing"},
                "kids_contains": {"field": "kids", "op": "contains", "value": "om"},
                "two_kids": {"roster_op": "count_gt", "value": 1},
                "any_girl": {"roster_op": "any_match", "tag": "female"},
                "toddler": {"roster_op": "any_match", "max_age": 3},
                "no_roster": {"roster_op": "is_empty"},
                "has_roster": {"roster_op": "non_empty"},
                "unknown_op": {"field": "kids", "op": "wat"},
                "empty": {},
            },
            roster={"key": "children", "tags": ["male", "female"]},
        )
        st = new_state(pack)
        st.roster = [RosterItem(id="item_1", age_years=6, age_label="6 साल", tag="male", name="अमन"),
                     RosterItem(id="item_2", age_years=2, age_label="2 साल", tag="female", name="रीना")]
        st.profile["kids"], st.answers["a"] = "some", AnswerRecord(state=AnswerStatus.CONCERN)
        truth = {r: eval_rule(pack, st, r) for r in pack.rules}
        self.assertEqual(truth, {"flag_on": False, "flagged": False, "not_flagged": True, "either": False, "both": False,
                                 "fam_clear": False, "kids_in": True, "kids_absent": False, "kids_contains": True,
                                 "two_kids": True, "any_girl": True, "toddler": True, "no_roster": False,
                                 "has_roster": True, "unknown_op": False, "empty": False})
        st.semantic_flags, st.profile["ever_pregnant"] = ["beta"], True
        st.answers["a"] = AnswerRecord(state=AnswerStatus.CLEAR)
        self.assertEqual([eval_rule(pack, st, r) for r in ("flag_on", "flagged", "either", "both", "fam_clear")],
                         [True, True, True, True, True])
        self.assertTrue(eval_rule(pack, st, {"any_of": [{"semantic_flag": "beta"}]}))

    def test_quick_replies_distinguish_yes_no_from_choice_pills(self):
        yn_pack = make_pack([gate("g")], meta={"ui": {"yes_no_buttons": [{"value": "clear", "hi": "नहीं", "en": "No"}, {"value": "concern", "hi": "हाँ", "en": "Yes"}]}})
        is_yn, btns = quick_replies(yn_pack, new_state(yn_pack))
        self.assertTrue(is_yn)
        self.assertEqual([b["value"] for b in btns], ["clear", "concern"])

        ch_pack = make_pack([{"id": "c", "kind": QuestionKind.SINGLE_CHOICE, "quick_options": True, "options": {"a": {"label": "A", "en": "Opt A"}, "b": {"label": "B", "en": "Opt B"}}}])
        is_yn2, btns2 = quick_replies(ch_pack, new_state(ch_pack))
        self.assertFalse(is_yn2)
        self.assertEqual([b["value"] for b in btns2], ["a", "b"])
        self.assertEqual(quick_reply_command(ch_pack, new_state(ch_pack), "b").value, "b")

    def test_retry_exhaustion_records_unanswered_and_moves_on(self):
        # A value the question cannot accept is not "पता नहीं". Two misses close the step as unanswered
        # and must not write a false no into the profile slot the question saves.
        pack = make_pack([gate("g", saves_to_profile="flag"), gate("next")])
        st = new_state(pack)
        for _ in range(MAX_STEP_ATTEMPTS - 1):
            st = play(pack, st, TurnOutcome.ANSWERED, value="bad")
            self.assertEqual(st.cursor.qid, "g")
        st = play(pack, st, TurnOutcome.ANSWERED, value="bad")
        self.assertEqual((st.answers["g"].state.value, st.cursor.qid), ("unanswered", "next"))
        self.assertNotIn("flag", st.profile)
        self.assertTrue(st.answers["g"].state.is_resolved)
        self.assertNotIn(st.answers["g"].state, (AnswerStatus.CONCERN, AnswerStatus.CLEAR, AnswerStatus.UNKNOWN))

    def test_audio_only_followup_keeps_several_slots_in_order(self):
        pack = make_pack([gate("nicu", followups=("nicu_reason", "nicu_days")), gate("next")],
                         followups={"nicu_reason": "क्यों?", "nicu_days": "कितने दिन?"})
        st = play(pack, new_state(pack), TurnOutcome.CONCERN)
        st = play(pack, st, TurnOutcome.CONCERN, utterance="")
        self.assertEqual((st.cursor.qid, st.cursor.followup_idx), ("nicu", 1))
        st = play(pack, st, TurnOutcome.ANSWERED, utterance="")
        self.assertEqual((st.cursor.qid, [f.state for f in st.answers["nicu"].followups.values()]),
                         ("next", [AnswerStatus.CONCERN, AnswerStatus.ANSWERED]))

    def test_followup_still_retries_when_the_turn_is_not_about_the_slot(self):
        pack = make_pack([gate("nicu", followups=("nicu_reason",)), gate("next")],
                         followups={"nicu_reason": "क्यों?"})
        st = play(pack, new_state(pack), TurnOutcome.CONCERN)
        # Only a classified outcome settles the slot, so a transcript on an off-topic turn still costs one retry.
        st = play(pack, st, TurnOutcome.NOT_ABOUT_ACTIVE, utterance="")
        self.assertEqual((st.cursor.qid, st.cursor.attempts), ("nicu", 1))
        st = play(pack, st, TurnOutcome.NOT_ABOUT_ACTIVE, utterance="उसकी तबीयत ठीक थी")
        self.assertEqual((st.answers["nicu"].followups["nicu_reason"].state.value, st.cursor.qid), ("unanswered", "next"))

    def test_off_topic_turn_at_main_costs_an_attempt_then_gives_up(self):
        # Bug: NOT_ABOUT_ACTIVE at SubStep.MAIN returned an untouched state, so `attempts` never moved and the
        # same question repeated forever. It must burn an attempt, then record unanswered and move on.
        # Call #67: that give-up is not UNKNOWN — she never said पता नहीं.
        pack = make_pack([gate("g"), gate("next")])
        st = play(pack, new_state(pack), TurnOutcome.NOT_ABOUT_ACTIVE)
        self.assertEqual((st.cursor.qid, st.cursor.attempts), ("g", 1))
        self.assertIn("try:1", st.step_token)
        st = play(pack, st, TurnOutcome.NOT_ABOUT_ACTIVE)
        self.assertEqual((st.answers["g"].state.value, st.cursor.qid), ("unanswered", "next"))

    def test_open_capture_off_topic_turns_close_cleanly_and_keep_a_finding(self):
        pack = make_pack([{"id": "close_any", "kind": QuestionKind.OPEN_CAPTURE, "say": "कुछ और?"}])
        st = play(pack, new_state(pack), TurnOutcome.CONCERN, utterance="उसके पेट में दर्द रहता है")
        self.assertEqual(st.answers["close_any"].state, AnswerStatus.CONCERN)
        # A spoken spontaneous remark is still a finding, not a miss.
        st = play(pack, st, TurnOutcome.NOT_ABOUT_ACTIVE, utterance="उसको रात में खांसी आती है")
        self.assertEqual(st.answers["close_any"].verbatim_text, "उसके पेट में दर्द रहता है; उसको रात में खांसी आती है")
        # Silent off-topic turns burn attempts, then close the call with the finding intact.
        for _ in range(MAX_STEP_ATTEMPTS - 1):
            st = play(pack, st, TurnOutcome.NOT_ABOUT_ACTIVE)
            self.assertEqual((st.cursor.qid, st.phase), ("close_any", Phase.ASKING))
        st = play(pack, st, TurnOutcome.NOT_ABOUT_ACTIVE)
        self.assertEqual((st.phase, st.close_reason), (Phase.CLOSING, CloseReason.COMPLETE))
        self.assertEqual(st.answers["close_any"].state, AnswerStatus.CONCERN)
        self.assertEqual(st.answers["close_any"].verbatim_text, "उसके पेट में दर्द रहता है; उसको रात में खांसी आती है")
        # A spoken CONCERN already on record is never flattened when the step gives up.

    def test_call_67_off_topic_give_up_is_unanswered_while_dont_know_stays_unknown(self):
        # Call #67: two turns that were not about the question exhausted the step. Writing UNKNOWN
        # ("पता नहीं") claims the mother said she did not know. Only TurnOutcome.UNKNOWN may do that.
        pack = make_pack([gate("said"), gate("missed"), gate("next")])
        said = play(pack, new_state(pack), TurnOutcome.UNKNOWN)
        self.assertEqual((said.answers["said"].state, said.cursor.qid), (AnswerStatus.UNKNOWN, "missed"))
        missed = play(pack, said, TurnOutcome.NOT_ABOUT_ACTIVE)
        self.assertEqual(missed.cursor.qid, "missed")
        missed = play(pack, missed, TurnOutcome.NOT_ABOUT_ACTIVE)
        self.assertEqual(missed.answers["missed"].state.value, "unanswered")
        self.assertTrue(missed.answers["missed"].state.is_resolved)
        self.assertNotIn(missed.answers["missed"].state, (AnswerStatus.CONCERN, AnswerStatus.CLEAR, AnswerStatus.UNKNOWN))
        self.assertFalse(eval_rule(pack, missed, {"field": "missed", "op": "eq", "value": False}))
        self.assertFalse(eval_rule(pack, missed, {"field": "missed", "op": "eq", "value": True}))
        self.assertEqual(missed.answers["said"].state, AnswerStatus.UNKNOWN)
        self.assertEqual(missed.cursor.qid, "next")

    def test_unanswered_is_resolved_without_being_concern_or_clear(self):
        member = AnswerStatus.__members__.get("UNANSWERED")
        self.assertIsNotNone(member)
        self.assertEqual(member.value, "unanswered")
        self.assertTrue(member.is_resolved)
        self.assertNotIn(member, (AnswerStatus.CONCERN, AnswerStatus.CLEAR, AnswerStatus.UNKNOWN))
        self.assertFalse(AnswerStatus.NOT_APPLICABLE.is_resolved)
        self.assertTrue(AnswerStatus.UNKNOWN.is_resolved)

    def test_open_capture_exhaustion_without_a_finding_is_unanswered(self):
        # Open capture sorts last, so it has to be the only step or the misses land on the gate instead.
        pack = make_pack([{"id": "close_any", "kind": QuestionKind.OPEN_CAPTURE, "say": "कुछ और?"}])
        st = new_state(pack)
        self.assertEqual(st.cursor.qid, "close_any")
        for _ in range(MAX_STEP_ATTEMPTS):
            st = play(pack, st, TurnOutcome.NOT_ABOUT_ACTIVE, utterance="")
        self.assertEqual(st.answers["close_any"].state.value, "unanswered")
        self.assertEqual(st.answers["close_any"].verbatim_text, "")
        self.assertEqual((st.phase, st.close_reason), (Phase.CLOSING, CloseReason.COMPLETE))

    def test_open_capture_helper_writes_unanswered_and_keeps_a_spoken_finding(self):
        # reduce_turn intercepts a silent NOT_ABOUT_ACTIVE before `_open_capture`. Call the helper so its
        # own give-up cannot fall through to UNKNOWN, and a spoken CONCERN stays on record.
        from intake.engine import _open_capture
        pack = make_pack([{"id": "close_any", "kind": QuestionKind.OPEN_CAPTURE, "say": "कुछ और?"}])
        spec = pack.by_id["close_any"]
        blank = new_state(pack)
        cmd = TurnCommand(outcome=TurnOutcome.NOT_ABOUT_ACTIVE)
        self.assertIsNotNone(_open_capture(pack, blank, cmd, spec, ""))
        self.assertIsNone(_open_capture(pack, blank, cmd, spec, ""))
        self.assertEqual(blank.answers["close_any"].state.value, "unanswered")
        known = new_state(pack)
        self.assertIsNone(_open_capture(pack, known, TurnCommand(outcome=TurnOutcome.UNKNOWN), spec, ""))
        self.assertEqual(known.answers["close_any"].state, AnswerStatus.UNKNOWN)
        kept = new_state(pack)
        kept.answers["close_any"] = AnswerRecord(state=AnswerStatus.CONCERN, note=NoteText.from_text("पेट में दर्द रहता है"))
        self.assertIsNone(_open_capture(pack, kept, TurnCommand(outcome=TurnOutcome.CLEAR), spec, ""))
        self.assertEqual(kept.answers["close_any"].state, AnswerStatus.CONCERN)
        self.assertEqual(kept.answers["close_any"].verbatim_text, "पेट में दर्द रहता है")

    def test_unanswered_followup_is_not_quoted_as_the_patients_words(self):
        rec = AnswerRecord(
            state=AnswerStatus.CONCERN,
            note=NoteText.from_text("हाँ, NICU में रहा"),
            followups={"nicu_reason": FollowupSlot(state=AnswerStatus.UNANSWERED, note=NoteText.from_text("गलत उद्धरण"))},
        )
        self.assertEqual(rec.patient_safe_text, "हाँ, NICU में रहा")
        self.assertIn("गलत उद्धरण", rec.verbatim_text)

    def test_which_item_keeps_the_spoken_answer_when_no_child_resolves(self):
        # Bug: an ANSWERED turn naming no roster id dropped the transcript, losing "दोनों जुड़वाँ को".
        pack = make_pack([{"id": "jaundice", "kind": QuestionKind.GATE, "roster_scoped": True}, gate("next")],
                         roster={"key": "children", "tags": ["male", "female"]})
        twins = [RosterItem(id="item_1", age_years=2, age_label="2 साल", tag="male"),
                 RosterItem(id="item_2", age_years=2, age_label="2 साल", tag="male")]
        st = new_state(pack)
        st.roster = twins
        st = play(pack, st, TurnOutcome.CONCERN)
        self.assertEqual(st.cursor.substep, SubStep.WHICH_ITEM)
        st = play(pack, st, TurnOutcome.ANSWERED, utterance="दोनों जुड़वाँ को")
        self.assertEqual(st.answers["jaundice"].note.verbatim, "दोनों जुड़वाँ को")
        # A meta question about the notes must still never leak into a child's clinical note.
        st2 = new_state(pack)
        st2.roster = twins
        st2 = play(pack, st2, TurnOutcome.CONCERN)
        st2 = play(pack, st2, TurnOutcome.NOT_ABOUT_ACTIVE, utterance="क्या नोट किया है? मेरे कितने बच्चे हैं?")
        self.assertEqual(st2.answers["jaundice"].note.verbatim, "")

    def test_stated_concern_keeps_the_child_named_in_the_same_sentence(self):
        # Bug: a mother co-reporting two concerns for one child in one sentence lost the second child ref.
        pack = make_pack([{"id": "a", "kind": QuestionKind.GATE, "roster_scoped": True},
                          {"id": "b", "kind": QuestionKind.GATE, "roster_scoped": True}],
                         roster={"key": "children", "tags": ["male", "female"]})
        st = new_state(pack)
        st.roster = [RosterItem(id="item_1", age_years=4, age_label="4 साल", tag="male"),
                     RosterItem(id="item_2", age_years=2, age_label="2 साल", tag="male")]
        st = play(pack, st, TurnOutcome.CONCERN, item_ref="item_2")
        self.assertEqual(st.answers["a"].item_ids, ["item_2"])
        st = play(pack, st, TurnOutcome.CONCERN, item_ref="item_1", utterance="दोनों को दौरे आए",
                  stated={"a": TurnOutcome.CONCERN})
        self.assertEqual(st.answers["a"].item_ids, ["item_2", "item_1"])

    def test_equal_age_twins_get_distinct_which_item_labels_from_the_pack(self):
        # Bug: two unnamed 2-year-olds rendered identically ("2 साल वाले, या 2 साल वाले").
        shishu = load_pack("shishu-hi")
        twins = [RosterItem(id="item_1", age_years=2, age_label="2 साल", tag="male"),
                 RosterItem(id="item_2", age_years=2, age_label="2 साल", tag="male")]
        self.assertEqual(which_item_prompt(shishu, CallState(roster=twins)),
                         "यह किस बच्चे की बात है — पहले 2 साल वाले बेटे, या दूसरे 2 साल वाले बेटे?")

    def test_youngest_twins_in_a_three_child_roster_are_numbered_never_called_middle(self):
        # Bug (Call #11): [4, 2, 2] appended " (बीच वाले)" to the FIRST 2-year-old twin, so `parts[1]` and
        # `parts[2]` stopped being equal, `counts[p]` never fired and SHISHU asked
        # "4 साल वाले, 2 साल वाले (बीच वाले), या 2 साल वाले?" — ambiguous on the phone, and a twin who ties
        # for youngest is not the middle child at all.
        shishu = load_pack("shishu-hi")
        trio = [RosterItem(id="item_1", age_years=4, age_label="4 साल", tag="unknown"),
                RosterItem(id="item_2", age_years=2, age_label="2 साल", tag="unknown"),
                RosterItem(id="item_3", age_years=2, age_label="2 साल", tag="unknown")]
        self.assertEqual(which_item_prompt(shishu, CallState(roster=trio)),
                         "यह किस बच्चे की बात है — 4 साल वाले, पहले 2 साल वाले, या दूसरे 2 साल वाले?")

    def test_equal_age_twins_are_same_age_not_oldest_and_youngest(self):
        # Bug: two 2-year-old twins were labelled oldest/youngest, feeding gemini-decision-1 false age ranks.
        twins = [RosterItem(id="item_1", age_years=2, age_label="2 साल", tag="male"),
                 RosterItem(id="item_2", age_years=2, age_label="2 साल", tag="male")]
        self.assertEqual([r["rank"] for r in roster_rows(CallState(roster=twins))], ["same_age", "same_age"])
        spread = [*twins, RosterItem(id="item_3", age_years=6, age_label="6 साल", tag="female"),
                  RosterItem(id="item_4", age_years=11, age_label="11 साल", tag="female")]
        self.assertEqual([r["rank"] for r in roster_rows(CallState(roster=spread))],
                         ["oldest", "middle", "same_age", "same_age"])


def at_children(shishu: PackConfig, *, qid: str = "intro_children") -> CallState:
    """The real intro flow up to the children question: a named, married caller, nothing answered yet."""
    st = reduce_turn(shishu, engine.new_state(shishu), TurnCommand(profile_patch=ProfilePatch(slots={"caller_name": "कविता"})))
    st = reduce_turn(shishu, st, TurnCommand(outcome=TurnOutcome.ANSWERED, value="married"))
    st.cursor = CursorStep(qid=qid)
    return st


class RealPackIntegrationTests(unittest.TestCase):
    def test_a_stated_count_with_no_ages_holds_on_the_children_question(self):
        # Bug: an empty-ages children turn fell through to `st.roster = []` + `roster_stated = True`, so
        # `no_children` went true and all 40 child questions were skipped for a mother who has three children.
        shishu = load_pack("shishu-hi")
        st = reduce_turn(shishu, at_children(shishu), TurnCommand(profile_patch=ProfilePatch(roster=())))
        self.assertEqual((st.roster, st.roster_stated), ([], False))
        self.assertFalse(eval_rule(shishu, st, "no_children"))
        self.assertFalse(st.answers.get("birth_nicu") is not None and st.answers["birth_nicu"].state is AnswerStatus.NOT_APPLICABLE)
        self.assertEqual((st.cursor.qid, st.cursor.hold_slot), ("intro_children", "roster"))
        self.assertTrue(st.roster_incomplete)
        self.assertEqual(active_prompt(shishu, st), "जी, उन सभी बच्चों की उम्र क्या-क्या है?")

    def test_explicit_zero_children_still_marks_every_child_question_not_applicable(self):
        shishu = load_pack("shishu-hi")
        st = reduce_turn(shishu, at_children(shishu), TurnCommand(profile_patch=ProfilePatch(roster=(), roster_complete=True)))
        self.assertEqual((st.roster, st.roster_stated, st.roster_incomplete), ([], True, False))
        self.assertTrue(eval_rule(shishu, st, "no_children"))
        self.assertEqual(st.answers["birth_nicu"].state, AnswerStatus.NOT_APPLICABLE)
        self.assertEqual(st.cursor.qid, "intro_bridge")

    def test_a_caller_who_never_gives_her_name_is_reasked_once_then_unanswered(self):
        # Bug: the give-up guard in `reduce_turn` was `prev_spec.kind is not QuestionKind.PROFILE`, so a caller
        # who declines to give her name — or an extractor that never hears it — left `intro_name` unsatisfied
        # and the call re-asked it forever. A profile step is a question like any other: two empty turns
        # exhaust it, and the explicit UNANSWERED record is what releases the step.
        shishu = load_pack("shishu-hi")
        decline = TurnCommand(outcome=TurnOutcome.NOT_ABOUT_ACTIVE, profile_patch=ProfilePatch(),
                              profile_turn_attempted=True, utterance="नहीं बताऊँगी")
        st = new_state(shishu)
        self.assertEqual(st.cursor.qid, "intro_name")
        st = reduce_turn(shishu, st, decline)
        self.assertEqual(st.cursor.qid, "intro_name", "the first miss still asks the name once more")
        self.assertEqual(st.cursor.attempts, 1)
        self.assertNotIn("caller_name", st.profile)
        self.assertFalse(st.answers.get("intro_name") is not None and st.answers["intro_name"].state is AnswerStatus.UNANSWERED)
        st = reduce_turn(shishu, st, decline)
        self.assertIs(st.answers["intro_name"].state, AnswerStatus.UNANSWERED)
        self.assertEqual(st.cursor.qid, "intro_married", "an exhausted profile step never loops back on itself")
        self.assertNotIn("caller_name", st.profile)

    def test_a_caller_who_never_gives_a_roster_is_reasked_once_then_unanswered(self):
        # Same bug on `intro_children`: an empty profile patch (roster and roster_complete both null) re-asked
        # the children question forever. One miss must still re-ask for the ages — a partial roster needs a
        # second chance — and only the second miss may exhaust the step.
        shishu = load_pack("shishu-hi")
        nothing = TurnCommand(outcome=TurnOutcome.NOT_ABOUT_ACTIVE, profile_patch=ProfilePatch(),
                              profile_turn_attempted=True, utterance="अभी नहीं बता सकती")
        st = at_children(shishu)
        self.assertEqual(st.cursor.qid, "intro_children")
        st = reduce_turn(shishu, st, nothing)
        self.assertEqual((st.cursor.qid, st.cursor.hold_slot), ("intro_children", "roster"))
        self.assertEqual(st.cursor.attempts, 1)
        self.assertEqual(active_prompt(shishu, st), "जी, उन सभी बच्चों की उम्र क्या-क्या है?")
        self.assertFalse(st.answers.get("intro_children") is not None and st.answers["intro_children"].state is AnswerStatus.UNANSWERED)
        st = reduce_turn(shishu, st, nothing)
        self.assertIs(st.answers["intro_children"].state, AnswerStatus.UNANSWERED)
        self.assertNotEqual(st.cursor.qid, "intro_children", "an exhausted roster step never loops back on itself")
        self.assertEqual(st.cursor.qid, "intro_bridge")

    def test_a_partial_twin_roster_still_gets_its_ages_reask_before_giving_up(self):
        # Call #11's twins with no ages: ONE incomplete roster turn must still re-ask for the ages, never skip
        # the step. Only after the retry budget is spent does the ages reask stop holding the call.
        shishu = load_pack("shishu-hi")
        twins_no_ages = TurnCommand(outcome=TurnOutcome.NOT_ABOUT_ACTIVE, profile_turn_attempted=True,
                                    profile_patch=ProfilePatch(roster=(), roster_complete=False))
        st = reduce_turn(shishu, at_children(shishu), twins_no_ages)
        self.assertTrue(st.roster_incomplete)
        self.assertEqual((st.cursor.qid, st.cursor.hold_slot), ("intro_children", "roster"))
        self.assertEqual(active_prompt(shishu, st), "जी, उन सभी बच्चों की उम्र क्या-क्या है?")
        st = reduce_turn(shishu, st, twins_no_ages)
        self.assertIs(st.answers["intro_children"].state, AnswerStatus.UNANSWERED)
        self.assertNotEqual(st.cursor.qid, "intro_children")

    def test_a_given_up_roster_step_revives_when_she_later_mentions_missed_twins(self):
        # A give-up is only ever about a caller who would not answer. The moment she volunteers children again —
        # Call #11's "मेरे ट्विंस बच्चे हैं, मुझे लगता नहीं आप वो सुन पाए" — the ages reask is live again and
        # must hold the call, instead of a stale UNANSWERED quietly skipping the ages it now needs.
        shishu = load_pack("shishu-hi")
        st = at_children(shishu, qid="birth_nicu")
        st.answers["intro_children"] = AnswerRecord(state=AnswerStatus.UNANSWERED)
        self.assertNotEqual(st.cursor.qid, "intro_children")
        st = reduce_turn(shishu, st, TurnCommand(outcome=TurnOutcome.NOT_ABOUT_ACTIVE, profile_turn_attempted=True,
                                                 profile_patch=ProfilePatch(roster=(), roster_complete=False),
                                                 utterance="मेरे ट्विंस बच्चे हैं, आपने सुना नहीं"))
        self.assertEqual((st.cursor.qid, st.cursor.hold_slot), ("intro_children", "roster"))
        self.assertTrue(st.roster_incomplete)
        self.assertFalse(st.answers.get("intro_children") is not None and st.answers["intro_children"].state is AnswerStatus.UNANSWERED)

    def test_a_mid_call_count_only_statement_rewinds_to_the_ages_reask(self):
        shishu = load_pack("shishu-hi")
        st = at_children(shishu, qid="birth_nicu")
        st.roster = [RosterItem(id="item_1", age_years=2.0, age_label="2 साल", tag="male"),
                     RosterItem(id="item_2", age_years=4.0, age_label="4 साल", tag="female")]
        st.roster_stated = True
        st = reduce_turn(shishu, st, TurnCommand(profile_patch=ProfilePatch(roster=())))
        self.assertEqual((st.cursor.qid, st.cursor.hold_slot), ("intro_children", "roster"))
        self.assertEqual([c.id for c in st.roster], ["item_1", "item_2"])
        self.assertEqual(active_prompt(shishu, st), "जी, उन सभी बच्चों की उम्र क्या-क्या है?")

    def test_a_mid_call_child_update_keeps_every_existing_id_and_never_wipes_the_roster(self):
        # Bug: a roster patch always renumbered item_1..item_n, so every answer already pointing at item_1
        # silently pointed at a different child, and an ages-less patch wiped the roster to zero children.
        shishu = load_pack("shishu-hi")
        st = at_children(shishu)
        st.roster = [RosterItem(id="item_1", age_years=2.0, age_label="2 साल", tag="male"),
                     RosterItem(id="item_2", age_years=4.0, age_label="4 साल", tag="female")]
        st.roster_stated = True
        st.answers["body_fits"] = AnswerRecord(state=AnswerStatus.CONCERN, item_ids=["item_1"])
        st = reduce_turn(shishu, st, TurnCommand(profile_patch=ProfilePatch(roster=(
            RosterInput(age_years=2.0), RosterInput(age_years=2.0), RosterInput(age_years=4.0)))))
        self.assertEqual([c.id for c in st.roster], ["item_1", "item_3", "item_2"])
        self.assertEqual([c.age_years for c in st.roster], [2.0, 2.0, 4.0])
        self.assertEqual(st.answers["body_fits"].item_ids, ["item_1"])
        self.assertEqual(st.answers["body_fits"].item_ids, [c.id for c in st.roster if c.age_years == 2.0][:1])
        kept = reduce_turn(shishu, st, TurnCommand(profile_patch=ProfilePatch(roster=())))
        self.assertEqual([c.id for c in kept.roster], ["item_1", "item_3", "item_2"])
        self.assertEqual((kept.cursor.qid, kept.cursor.hold_slot), ("intro_children", "roster"))

    def test_shishu_pack_reactive_thalassemia_autofill_semantic_skip_and_topic_jump(self):
        shishu = load_pack("shishu-hi")
        st = new_state(shishu)
        self.assertEqual(st.cursor.qid, "intro_name")
        st = reduce_turn(shishu, st, TurnCommand(profile_patch=ProfilePatch(slots={"caller_name": "कविता"})))
        self.assertEqual(st.cursor.qid, "intro_married")
        st = reduce_turn(shishu, st, TurnCommand(outcome=TurnOutcome.ANSWERED, value="married"))
        self.assertEqual(st.cursor.qid, "intro_children")
        st = reduce_turn(shishu, st, TurnCommand(profile_patch=ProfilePatch(
            roster=(
                RosterInput(age_years=11.0, tag="female", name="रीना"),
                RosterInput(age_years=6.0, tag="male", name="अमन"),
                RosterInput(age_years=2.0, tag="male", name="राजू"),
            ),
        )))
        self.assertEqual(st.cursor.qid, "intro_bridge")
        bridge = active_prompt(shishu, st)
        self.assertIn("खुलकर बोलकर", bridge, "the bridge still invites a spoken answer")
        self.assertLess(len(bridge), 100, "the bridge is spoken in one breath, not twelve seconds")
        st = play(shishu, st, TurnOutcome.CONCERN)
        self.assertEqual(st.cursor.qid, "birth_mode_detail")

        # Multi-child which_item formatting check on 3 children
        st_j = play(shishu, st, TurnOutcome.CONCERN)
        self.assertEqual(st_j.cursor.substep, SubStep.WHICH_ITEM)
        self.assertEqual(
            which_item_prompt(shishu, st_j),
            "यह किस बच्चे की बात है — 11 साल वाली रीना, 6 साल वाले अमन (बीच वाले), या 2 साल वाले राजू?",
        )
        st = play(shishu, st_j, TurnOutcome.RETRACT)

        # Reactive turn: denies birth_term, volunteers Thalassemia Major + monthly transfusions + enlarged spleen,
        # triggers confirmed_hemoglobinopathy semantic skip on body_blood_count, and jumps to preg_thalassemia!
        utt = "नहीं, पर बीच वाले बेटे अमन को थैलेसीमिया मेजर है, हर महीने खून चढ़ता है और तिल्ली बढ़ी हुई है।"
        st = reduce_turn(
            shishu,
            st,
            TurnCommand(
                outcome=TurnOutcome.CLEAR,
                item_ref="item_2",
                stated={
                    "body_transfusion": TurnOutcome.CONCERN,
                    "body_tummy": TurnOutcome.CONCERN,
                    "ill_admissions": TurnOutcome.CONCERN,
                },
                semantic_flags=("confirmed_hemoglobinopathy",),
                jump_topic=True,
                next_qid_probs={"body_transfusion": 0.99, "preg_thalassemia": 0.88, "birth_nicu": 0.10},
                utterance=utt,
                redacted=True,
            ),
        )
        self.assertEqual(st.answers["birth_term"].state, AnswerStatus.CLEAR)
        self.assertEqual(st.answers["body_transfusion"].state, AnswerStatus.CONCERN)
        self.assertEqual(st.answers["body_transfusion"].item_ids, ["item_2"])
        self.assertEqual(st.answers["body_transfusion"].patient_safe_text, "")
        self.assertIn("थैलेसीमिया मेजर", st.answers["body_transfusion"].verbatim_text)
        self.assertEqual(st.answers["body_blood_count"].state, AnswerStatus.NOT_APPLICABLE)
        # body_transfusion is ALREADY answered, so jump_topic picks the top STILL-ELIGIBLE target: preg_thalassemia!
        self.assertEqual(st.cursor.qid, "preg_thalassemia")

    def test_an_empty_profile_patch_mid_call_is_still_an_off_topic_miss(self):
        # Bug (Call #11): `mentions_profile` fired on a non-profile step (preg_delivery) while the extractor
        # returned an EMPTY ProfilePatch(), yet `profile_ran` was True because `profile_turn_attempted` was
        # set. Line 380 then returned an untouched state, so the off-topic turn never burned an attempt and
        # `preg_delivery` could be re-asked forever. Only a profile-step attempt can hold its own slot.
        shishu = load_pack("shishu-hi")
        st = reduce_turn(shishu, at_children(shishu), TurnCommand(profile_patch=ProfilePatch(roster=(
            RosterInput(age_years=4.0, tag="male"), RosterInput(age_years=2.0, tag="male")))))
        self.assertEqual(st.cursor.qid, "intro_bridge")
        upto = [q.id for q in shishu.questions].index("preg_delivery")
        for q in shishu.questions[:upto]:
            st.answers[q.id] = AnswerRecord(state=AnswerStatus.CLEAR)
        st.cursor = CursorStep(qid="preg_delivery")
        self.assertEqual(engine.step_kind(shishu, st), "main")
        st = reduce_turn(shishu, st, TurnCommand(outcome=TurnOutcome.NOT_ABOUT_ACTIVE,
                                                 profile_patch=ProfilePatch(), profile_turn_attempted=True))
        self.assertEqual((st.cursor.qid, st.cursor.attempts), ("preg_delivery", 1))

    def test_a_mid_call_incomplete_twin_update_freezes_roster_op_not_applicable(self):
        # Bug (Call #11): "मेरे ट्विंस बच्चे हैं, मुझे लगता नहीं आप वो सुन पाए" arrived mid-call with
        # roster_complete=False. `_roster_op` gates `none_match` on `roster_resolved`, so `no_child_5_plus`
        # flipped False and `_resync` DELETED the NOT_APPLICABLE on mod_tooth_bleed / mod_heavy_periods /
        # body_puberty, waking age-gated questions for 2- and 4-year-olds and landing the cursor on
        # `mod_tooth_bleed` instead of rewinding to the ages reask. `_is_unsatisfied` also trusted a stale
        # `state.answers["intro_children"]`, so the profile step never reopened.
        shishu = load_pack("shishu-hi")
        st = reduce_turn(shishu, at_children(shishu), TurnCommand(profile_patch=ProfilePatch(roster=(
            RosterInput(age_years=2.0, tag="male"), RosterInput(age_years=4.0, tag="male")), roster_complete=True)))
        self.assertEqual(st.answers["mod_tooth_bleed"].state, AnswerStatus.NOT_APPLICABLE)
        self.assertEqual(st.answers["mod_heavy_periods"].state, AnswerStatus.NOT_APPLICABLE)
        upto = [q.id for q in shishu.questions].index("preg_delivery")
        for q in shishu.questions[:upto]:
            if q.id not in st.answers:
                st.answers[q.id] = AnswerRecord(state=AnswerStatus.CLEAR)
        st.cursor = CursorStep(qid="preg_delivery")

        st = reduce_turn(shishu, st, TurnCommand(
            outcome=TurnOutcome.NOT_ABOUT_ACTIVE,
            profile_patch=ProfilePatch(roster=(RosterInput(id="item_1", age_years=2.0, tag="male"),
                                               RosterInput(id="item_2", age_years=4.0, tag="male")),
                                       roster_complete=False),
            profile_turn_attempted=True))
        self.assertTrue(st.roster_incomplete)
        self.assertEqual((st.cursor.qid, st.cursor.hold_slot), ("intro_children", "roster"))
        self.assertEqual(st.answers["mod_tooth_bleed"].state, AnswerStatus.NOT_APPLICABLE)
        self.assertEqual(st.answers["mod_heavy_periods"].state, AnswerStatus.NOT_APPLICABLE)

        # The mother then answers the reask with the missed 12-year-old daughter: the frozen NOT_APPLICABLE
        # must be re-evaluated against the completed roster, so both questions come back to be asked.
        st = reduce_turn(shishu, st, TurnCommand(profile_patch=ProfilePatch(roster=(
            RosterInput(id="item_1", age_years=2.0, tag="male"), RosterInput(id="item_2", age_years=4.0, tag="male"),
            RosterInput(id="item_3", age_years=12.0, tag="female")), roster_complete=True)))
        self.assertFalse(st.roster_incomplete)
        self.assertNotIn("mod_heavy_periods", st.answers)
        self.assertNotIn("mod_tooth_bleed", st.answers)
        self.assertNotIn("body_puberty", st.answers)

    def test_both_packs_speak_their_bridge_in_one_short_breath(self):
        # The 12-second bridge held the call open with no reply: two sentences say the same thing in 4.
        for pack_id in ("shishu-hi", "fever-opd-hi"):
            with self.subTest(pack=pack_id):
                bridge = load_pack(pack_id).by_id["intro_bridge"]
                self.assertIn("खुलकर बोलकर बताएँ", bridge.say, "the open-speech invite survives the cut")
                self.assertLess(len(bridge.say), 100)
                self.assertLess(len(bridge.stem), 100)

    def test_both_packs_define_transient_retry_speak_cue_and_protocol(self):
        # A dropped line must be re-asked by the server with a spoken apology, not by a silent re-send of
        # the same words, so every pack needs a retry cue and a turn-protocol line that names it.
        for pack_id in ("shishu-hi", "fever-opd-hi"):
            with self.subTest(pack=pack_id):
                prompts = load_pack(pack_id).prompts
                cues = prompts["speak_cues"]
                self.assertIn("retry", cues)
                retry = cues["retry"]
                self.assertTrue(retry.startswith("SYSTEM CUE"), retry)
                self.assertTrue(retry.rstrip().endswith("{prompt}"), retry)
                self.assertIn("Do not call a tool.", retry)
                self.assertIn("retry", prompts["system_prompt"])
                self.assertIn("result is retry", prompts["system_prompt"])

    def test_extractor_prompt_handles_mid_call_twins_and_two_pairs_of_twins(self):
        # Bug (Call #11): mid-call "मेरे ट्विंस बच्चे हैं, मुझे लगता नहीं आप वो सुन पाए" returned null/null so the
        # missed twins never reached the roster, and "दो-दो के दो ट्विंस" (four children) was capped at three.
        prompt = load_pack("shishu-hi").extractor_prompt
        self.assertIn("NOT heard", prompt)
        self.assertIn("Never leave roster and roster_complete both null", prompt)
        self.assertIn("2 साल और 4 साल", prompt)
        self.assertIn('"roster_complete": true', prompt)
        self.assertIn("दो-दो", prompt)

    def test_shishu_unmarried_to_married_reopens_children_and_pregnancy(self):
        shishu = load_pack("shishu-hi")
        st = reduce_turn(shishu, new_state(shishu), TurnCommand(profile_patch=ProfilePatch(slots={"caller_name": "नीलिमा"})))
        st = reduce_turn(shishu, st, TurnCommand(outcome=TurnOutcome.ANSWERED, value="about_to_marry"))
        self.assertEqual(st.answers["intro_children"].state, AnswerStatus.NOT_APPLICABLE)
        self.assertEqual(st.answers["birth_nicu"].state, AnswerStatus.NOT_APPLICABLE)
        self.assertEqual(st.profile.get("ever_pregnant"), False)
        # Caller clarifies she IS already married -> intro_children reopens immediately!
        st = reduce_turn(shishu, st, TurnCommand(profile_patch=ProfilePatch(slots={"marital_status": "married"})))
        self.assertEqual(st.cursor.qid, "intro_children")
        self.assertIsNone(st.profile.get("ever_pregnant"))


KABIR = RosterInput(age_years=4.0, tag="male", name="Kabir")
AMAN = RosterInput(age_years=6.0, tag="male", name="अमन")
MEERA = RosterInput(age_years=2.0, tag="female", name="मीरा")


def _said(utterance: str) -> str:
    """The clinician note's own `_note()` shape: whitespace collapsed, trailing danda or full stop cut."""
    return " ".join(utterance.split()).rstrip("।. ")


class Wave1ReducerAndReportInvariantTests(unittest.TestCase):
    """Wave 1 failure classes, each a matrix so every broken dimension names itself.

    1. a mid-call profile patch must never discard the answer the same turn just gave;
    2. a NOT_APPLICABLE must not be resurrected into a clinical yes/no by an out-of-turn `stated_` head;
    3. `jump_topic` must never be hijacked by a softmax crumb under the pack's own confidence floor;
    4. a which-item turn that already carries the followup detail must settle that slot, not re-ask it;
    5. a multi-topic sentence must stay in the clinician's verbatim and out of the patient's PDF.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.shishu = load_pack("shishu-hi")
        cls.fever = load_pack("fever-opd-hi")

    def shishu_call(self, roster, *, complete: bool = True) -> CallState:
        """A named, married caller whose children are on record, sitting on the intro bridge."""
        st = reduce_turn(self.shishu, new_state(self.shishu), TurnCommand(profile_patch=ProfilePatch(slots={"caller_name": "कविता"})))
        st = reduce_turn(self.shishu, st, TurnCommand(outcome=TurnOutcome.ANSWERED, value="married"))
        return reduce_turn(self.shishu, st, TurnCommand(profile_patch=ProfilePatch(roster=tuple(roster), roster_complete=complete)))

    def past_bridge(self, roster=(), **kw) -> CallState:
        return reduce_turn(self.shishu, self.shishu_call(roster, **kw), TurnCommand(outcome=TurnOutcome.ANSWERED))

    @staticmethod
    def at(state: CallState, qid: str, *, substep=SubStep.MAIN, followup_idx: int = 0) -> CallState:
        st = state.model_copy(deep=True)
        st.cursor = CursorStep(qid=qid, substep=substep, followup_idx=followup_idx)
        return st

    def _assert_all(self, observed: dict[str, bool], context: str) -> None:
        """Every dimension of one case reports itself, so a RED run names each broken invariant at once."""
        for label, ok in observed.items():
            if ok:
                continue
            with self.subTest(invariant=label):
                self.fail(f"{label}: {context}")

    def test_reduce_turn_preserves_active_answer_across_mid_call_profile_matrices(self):
        # Call #21 (t=1794927089): "मेरे ट्विंस बच्चे हैं" landed on birth_mode_detail. The profile patch was an
        # incomplete roster (one 4y child, roster_complete=False), so the children reask owned the turn and the
        # caller's own birth_mode_detail answer was never written — her yes simply vanished from the record.
        one = self.past_bridge((KABIR,), complete=True)
        self.assertEqual([c.id for c in one.roster], ["item_1"])
        self.assertEqual(one.cursor.qid, "birth_mode_detail")
        twins_added = (RosterInput(id="item_1", age_years=4.0, tag="male", name="Kabir"),
                       RosterInput(age_years=1.0, tag="male", name="Ruhaan"),
                       RosterInput(age_years=1.0, tag="female", name="Ruhi"))
        incomplete = (RosterInput(id="item_1", age_years=4.0, tag="male"),)
        reask = CursorStep(qid="intro_children", hold_slot="roster")
        cases = (
            ("complete twins added in the same breath", TurnOutcome.CONCERN, twins_added, True, False,
             AnswerStatus.CONCERN, ["item_1", "item_2", "item_3"], [],
             CursorStep(qid="birth_mode_detail", substep=SubStep.WHICH_ITEM)),
            ("incomplete twins reask (Call #21)", TurnOutcome.CONCERN, incomplete, False, True,
             AnswerStatus.CONCERN, ["item_1"], [], reask),
            ("incomplete twins on a quiet answer", TurnOutcome.CLEAR, incomplete, False, True,
             AnswerStatus.CLEAR, ["item_1"], [], reask),
            ("incomplete twins on an unknown answer", TurnOutcome.UNKNOWN, incomplete, False, True,
             AnswerStatus.UNKNOWN, ["item_1"], [], reask),
        )
        for label, outcome, patch_roster, complete, attempted, want_state, want_roster, want_ids, want_cursor in cases:
            with self.subTest(case=label, outcome=outcome.value):
                st = reduce_turn(self.shishu, self.at(one, "birth_mode_detail"), TurnCommand(
                    outcome=outcome, item_ref="", profile_patch=ProfilePatch(roster=patch_roster, roster_complete=complete),
                    profile_turn_attempted=attempted))
                rec = st.answers.get("birth_mode_detail")
                self._assert_all({
                    "the patch decides the roster, not the answer": [c.id for c in st.roster] == want_roster,
                    "the answer this very turn gave survives the profile patch": rec is not None,
                    "the active answer keeps its own state": rec is not None and rec.state is want_state,
                    "a lone child is never auto-pinned as the affected child":
                        rec is not None and rec.item_ids == want_ids,
                    "the cursor lands where the roster left it": st.cursor == want_cursor,
                }, f"case={label} outcome={outcome.value} roster={[c.id for c in st.roster]} "
                   f"answer={rec and rec.state.value} item_ids={rec and rec.item_ids} cursor={st.cursor}")

    def test_not_applicable_dominates_stated_across_pruning_and_skip_matrices(self):
        # Call #25: a `stated_` head won on a question the pack had already pruned (known hemoglobinopathy), so a
        # skipped question came back as a clinical yes/no and could even capture the cursor on a clarify.
        outs = (TurnOutcome.CONCERN, TurnOutcome.CLEAR, TurnOutcome.CLARIFY)
        two = self.past_bridge((AMAN, MEERA))
        for out in outs:
            with self.subTest(case="shishu semantic-flag prune (Call #25)", stated=out.value):
                st = reduce_turn(self.shishu, self.at(two, "birth_nicu"), TurnCommand(
                    outcome=TurnOutcome.CONCERN, item_ref="item_1",
                    stated={"body_transfusion": TurnOutcome.CONCERN, "body_blood_count": out},
                    semantic_flags=("confirmed_hemoglobinopathy",),
                    utterance="उसे खून चढ़ाया गया था और खून की कमी बताई गई है"))
                rec = st.answers.get("body_blood_count")
                self._assert_all({
                    "a stated concern still lands on an applicable sibling":
                        st.answers["body_transfusion"].state is AnswerStatus.CONCERN,
                    "the pruned question keeps its NOT_APPLICABLE record": rec is not None,
                    "a pack-pruned question is never answered by a stated_ head":
                        rec is not None and rec.state is AnswerStatus.NOT_APPLICABLE,
                    "the cursor never travels to a pruned question": st.cursor.qid != "body_blood_count",
                }, f"stated={out.value} blood_count={rec and rec.state.value} cursor={st.cursor.qid}")

        fever_state = new_state(self.fever)
        fever_state.profile = {"patient_name": "रीटा", "patient_age": 4}
        for out in outs:
            with self.subTest(case="fever semantic-flag prune", stated=out.value):
                st = reduce_turn(self.fever, self.at(fever_state, "received_treatment"), TurnCommand(
                    outcome=TurnOutcome.CLEAR, stated={"recent_travel": out},
                    semantic_flags=("confirmed_malaria_or_dengue",), utterance="डेंगू की रिपोर्ट पॉजिटिव आई है"))
                rec = st.answers.get("recent_travel")
                self._assert_all({
                    "the pruned travel question keeps its NOT_APPLICABLE record": rec is not None,
                    "a confirmed malaria/dengue report keeps travel pruned":
                        rec is not None and rec.state is AnswerStatus.NOT_APPLICABLE,
                }, f"stated={out.value} recent_travel={rec and rec.state.value}")

        childless = self.past_bridge(())
        self.assertIs(childless.answers["birth_nicu"].state, AnswerStatus.NOT_APPLICABLE)
        for out in outs:
            with self.subTest(case="demographic skip (no children)", stated=out.value):
                st = reduce_turn(self.shishu, childless, TurnCommand(outcome=TurnOutcome.ANSWERED, stated={"birth_nicu": out}))
                rec = st.answers.get("birth_nicu")
                self._assert_all({
                    "a child question keeps its NOT_APPLICABLE record for a mother with no children": rec is not None,
                    "a mother with no children cannot answer a birth-history question":
                        rec is not None and rec.state is AnswerStatus.NOT_APPLICABLE,
                }, f"stated={out.value} birth_nicu={rec and rec.state.value}")

        frozen = reduce_turn(self.shishu, self.past_bridge((RosterInput(age_years=2.0, tag="male"),
                                                            RosterInput(age_years=4.0, tag="male"))), TurnCommand(
            outcome=TurnOutcome.NOT_ABOUT_ACTIVE, profile_turn_attempted=True,
            profile_patch=ProfilePatch(roster=(RosterInput(id="item_1", age_years=2.0, tag="male"),
                                               RosterInput(id="item_2", age_years=4.0, tag="male")), roster_complete=False)))
        self.assertTrue(frozen.roster_incomplete)
        self.assertEqual((frozen.cursor.qid, frozen.cursor.hold_slot), ("intro_children", "roster"))
        self.assertIs(frozen.answers["mod_heavy_periods"].state, AnswerStatus.NOT_APPLICABLE)
        for out in outs:
            with self.subTest(case="frozen age gate under an open roster", stated=out.value):
                st = reduce_turn(self.shishu, frozen, TurnCommand(outcome=TurnOutcome.NOT_ABOUT_ACTIVE,
                                                                  stated={"mod_heavy_periods": out}))
                rec = st.answers.get("mod_heavy_periods")
                self._assert_all({
                    "the frozen age-gate NOT_APPLICABLE is never deleted": rec is not None,
                    "an unresolved roster cannot prove a 2y/4y girl's periods question applies":
                        rec is not None and rec.state is AnswerStatus.NOT_APPLICABLE,
                }, f"stated={out.value} mod_heavy_periods={rec and rec.state.value} cursor={st.cursor}")

        # Positive control: a stated concern that CONTRADICTS the group skip it was pruned by must still be heard.
        control = self.at(self.past_bridge((AMAN, MEERA)), "preg_heart")
        concerned = reduce_turn(self.shishu, control, TurnCommand(outcome=TurnOutcome.CLEAR,
                                                                  stated={"fam_same_problem": TurnOutcome.CONCERN}))
        cleared = reduce_turn(self.shishu, control, TurnCommand(outcome=TurnOutcome.CLEAR,
                                                                stated={"fam_same_problem": TurnOutcome.CLEAR}))
        self._assert_all({
            "a relative with the same problem is exactly the concern the group skip stood on":
                concerned.answers["fam_same_problem"].state is AnswerStatus.CONCERN,
            "a stated clear never resurrects a question the pack itself skipped":
                cleared.answers["fam_same_problem"].state is AnswerStatus.NOT_APPLICABLE,
        }, f"fam_same_problem concern={concerned.answers['fam_same_problem'].state.value} "
           f"clear={cleared.answers['fam_same_problem'].state.value}")

    def test_jump_topic_ignores_softmax_dust_below_choice_confidence_floor(self):
        # Sunita Step 7: `jump_topic` fired on a router crumb while another child's question was merely 0.02 of the
        # softmax, and the call abandoned the sequential order for a topic nobody had raised.
        st = self.past_bridge((AMAN, MEERA))
        upto = [q.id for q in self.shishu.questions].index("birth_jaundice")
        for q in self.shishu.questions[:upto]:
            rec = st.answers.get(q.id)
            if rec is None or rec.state is not AnswerStatus.CLEAR:
                st.answers[q.id] = AnswerRecord(state=AnswerStatus.CLEAR)
        floor = float(self.shishu.decisions["thresholds"]["choice_min_confidence"])
        self.assertEqual(floor, 0.52, "the shishu-hi choice gate this matrix is pinned to")
        for prob in (0.01, 0.02, 0.15, 0.51, 0.52, 0.85):
            with self.subTest(fam_transfusion_prob=prob):
                out = reduce_turn(self.shishu, self.at(st, "birth_jaundice"), TurnCommand(
                    outcome=TurnOutcome.CLEAR, stated={"body_transfusion": TurnOutcome.CONCERN}, jump_topic=True,
                    next_qid_probs={"fam_transfusion": prob, "body_transfusion": 0.94}, utterance="नहीं, खून नहीं चढ़ाया"))
                expected = "fam_transfusion" if prob >= floor else "birth_sudden"
                self._assert_all({
                    f"a router read of {prob} lands on {expected}":
                        out.cursor.qid == expected,
                }, f"fam_transfusion={prob} body_transfusion=0.94 floor={floor} cursor={out.cursor}")

    def test_which_item_settles_first_followup_when_detail_present(self):
        # Call #32: "छोटी बेटी मीरा की एक बार हाथ की हड्डी टूटी थी" answered the which-child AND the followup in one
        # breath, yet the call re-asked "हड्डी कितनी बार टूटी…?" and stored the detail nowhere it could be read.
        two = self.past_bridge((AMAN, MEERA))
        named = reduce_turn(self.shishu, self.at(two, "gate_bones_fractures"), TurnCommand(outcome=TurnOutcome.CONCERN))
        self.assertEqual((named.cursor.qid, named.cursor.substep), ("gate_bones_fractures", SubStep.WHICH_ITEM))
        replay = "छोटी बेटी मीरा की एक बार हाथ की हड्डी टूटी थी"
        with self.subTest(question="gate_bones_fractures", detail_present=True):
            settled = reduce_turn(self.shishu, named, TurnCommand(outcome=TurnOutcome.CONCERN, item_ref="item_2",
                                                                 followup_answered=True, utterance=replay))
            rec = settled.answers["gate_bones_fractures"]
            slot = rec.followups.get("fracture_count_or_bowing")
            self._assert_all({
                "the child she named is the child on record": rec.item_ids == ["item_2"],
                "the spoken detail reaches the followup slot it answers": slot is not None,
                "that followup slot is settled as a concern": slot is not None and slot.state is AnswerStatus.CONCERN,
                "the slot keeps her exact words": slot is not None and replay in slot.note.verbatim,
                "an answered followup is never asked again": settled.cursor.substep is not SubStep.FOLLOWUP,
                "the call moves past the question": settled.cursor.qid != "gate_bones_fractures",
            }, f"cursor={settled.cursor} item_ids={rec.item_ids} followups="
               f"{ {k: v.state.value for k, v in rec.followups.items()} }")
        with self.subTest(question="gate_bones_fractures", detail_present=False):
            asked = reduce_turn(self.shishu, named, TurnCommand(outcome=TurnOutcome.CONCERN, item_ref="item_2"))
            self.assertEqual(asked.cursor, CursorStep(qid="gate_bones_fractures", substep=SubStep.FOLLOWUP, followup_idx=0),
                             "a bare which-child answer still earns the followup question")

        nicu = reduce_turn(self.shishu, self.at(two, "birth_nicu"), TurnCommand(outcome=TurnOutcome.CONCERN))
        self.assertEqual((nicu.cursor.qid, nicu.cursor.substep), ("birth_nicu", SubStep.WHICH_ITEM))
        with self.subTest(question="birth_nicu", detail_present=True):
            out = reduce_turn(self.shishu, nicu, TurnCommand(outcome=TurnOutcome.CONCERN, item_ref="item_2",
                                                            followup_answered=True, utterance="मीरा को साँस तेज़ थी"))
            slots = out.answers["birth_nicu"].followups
            reason = slots.get("nicu_reason")
            self._assert_all({
                "the reason she gave settles the first followup slot": reason is not None,
                "that slot is a concern": reason is not None and reason.state is AnswerStatus.CONCERN,
                "the slot keeps her exact words": reason is not None and "मीरा को साँस तेज़ थी" in reason.note.verbatim,
                "the second slot (nicu_days) is the only thing left to ask":
                    out.cursor == CursorStep(qid="birth_nicu", substep=SubStep.FOLLOWUP, followup_idx=1),
            }, f"cursor={out.cursor} followups={ {k: v.state.value for k, v in slots.items()} }")

    def test_multi_topic_utterances_stay_in_verbatim_and_never_contaminate_patient_pdf(self):
        # Call #32 in full: one sentence held a denial, three of Aman's findings, and later a family cancer plus a
        # correction about Meera's jaundice. Every record kept the WHOLE sentence as patient-safe words, so Aman's
        # page said "नीली लाइट" and the family question said "पीलिया".
        two = self.past_bridge((AMAN, MEERA))
        utt_a = ("नहीं, नीली लाइट में तो नहीं रखा था पर मेरे बड़े बेटे अमन को तीन बार खून चढ़ाना पड़ा है "
                 "और उसका पेट भी धीरे-धीरे बड़ा हो रहा है।")
        turn_a = reduce_turn(self.shishu, self.at(two, "birth_jaundice"), TurnCommand(
            outcome=TurnOutcome.CLEAR, item_ref="item_1",
            stated={"body_transfusion": TurnOutcome.CONCERN, "body_blood_count": TurnOutcome.CONCERN,
                    "body_tummy": TurnOutcome.CONCERN}, utterance=utt_a))
        proj_a = reports.project_summary(self.shishu, turn_a)
        said_a = _said(utt_a)
        stated_a = ("body_transfusion", "body_blood_count", "body_tummy")
        observed = {
            **{f"A {qid} verbatim": utt_a in turn_a.answers[qid].verbatim_text for qid in stated_a},
            **{f"A {qid} doctor_markdown": said_a in proj_a.doctor_markdown for qid in stated_a},
            **{f"A {qid} patient_safe": turn_a.answers[qid].patient_safe_text == "" for qid in stated_a},
            **{f"A {qid} no_prefilled_followup": turn_a.answers[qid].followups == {} for qid in stated_a},
            "A no phototherapy leak": not [line for line in (*proj_a.yes_items, *proj_a.patient_questions,
                                                             *engine.recorded_concerns_summary(self.shishu, turn_a))
                                            if "नीली लाइट" in line],
        }
        for label, ok in observed.items():
            with self.subTest(turn="A", invariant=label):
                self.assertTrue(ok, f"{label}: {self._leak_context(proj_a, utt_a, stated_a)}")

        utt_b = ("हां, बुआ को कैंसर हुआ था और एक बात पहले नीली लाइट वाले सवाल में गलती से मना कर दिया था। "
                 "मीरा को जन्म पर पीलिया हुआ था।")
        turn_b = reduce_turn(self.shishu, self.at(turn_a, "fam_cancer"), TurnCommand(
            outcome=TurnOutcome.CONCERN, item_ref="item_2", followup_answered=True,
            stated={"birth_jaundice": TurnOutcome.CONCERN}, utterance=utt_b))
        proj_b = reports.project_summary(self.shishu, turn_b)
        said_b = _said(utt_b)
        both_b = ("fam_cancer", "birth_jaundice")
        jaundice_stem, cancer_stem = self.shishu.by_id["birth_jaundice"].stem, self.shishu.by_id["fam_cancer"].stem
        jaundice_lines = [x for x in (*proj_b.yes_items, *proj_b.patient_questions) if jaundice_stem in x]
        cancer_lines = [x for x in (*proj_b.yes_items, *proj_b.patient_questions) if cancer_stem in x]
        observed_b = {
            **{f"B {qid} verbatim": utt_b in turn_b.answers[qid].verbatim_text for qid in both_b},
            **{f"B {qid} doctor_markdown": said_b in proj_b.doctor_markdown for qid in both_b},
            **{f"B {qid} patient_safe": turn_b.answers[qid].patient_safe_text == "" for qid in both_b},
            "B fam_who settled": turn_b.cursor.qid != "fam_cancer",
            "B jaundice_days not pre-filled": turn_b.answers["birth_jaundice"].followups == {},
            "B jaundice line reaches the PDF": bool(jaundice_lines),
            "B jaundice line has no बुआ/कैंसर": bool(jaundice_lines) and not [x for x in jaundice_lines if "बुआ" in x or "कैंसर" in x],
            "B cancer line reaches the PDF": bool(cancer_lines),
            "B cancer line has no नीली लाइट/पीलिया": bool(cancer_lines) and not [x for x in cancer_lines if "नीली लाइट" in x or "पीलिया" in x],
            "B patient questions stay distinct": len(proj_b.patient_questions) == len(set(proj_b.patient_questions)),
            "B items never share a question": not (set(jaundice_lines) & set(cancer_lines)),
        }
        for label, ok in observed_b.items():
            with self.subTest(turn="B", invariant=label):
                self.assertTrue(ok, f"{label}: {self._leak_context(proj_b, utt_b, both_b)}")

    def test_a_clarify_at_the_closing_open_capture_never_hangs_up_the_call(self):
        # Bug (Wave 1 review): at `close_anything_else` — the LAST open question — "नहीं, बस थैलेसीमिया का मतलब
        # बता दीजिए" cleared the open capture, which advanced the call with `free == []` because the CLARIFY
        # revision of `preg_thalassemia` had not been applied yet. The call closed COMPLETE with its cursor on a
        # question it had just decided to ask.
        st = self.past_bridge((AMAN, MEERA))
        for q in self.shishu.questions:
            if q.id != "close_anything_else":
                rec = st.answers.get(q.id)
                if rec is None or rec.state is not AnswerStatus.CLEAR:
                    st.answers[q.id] = AnswerRecord(state=AnswerStatus.CLEAR)
        st = self.at(st, "close_anything_else")
        self.assertEqual([q.id for q in self.shishu.questions if engine._is_unsatisfied(self.shishu, st, q)],
                         ["close_anything_else"], "the closing capture is genuinely the last open question")
        out = reduce_turn(self.shishu, st, TurnCommand(
            outcome=TurnOutcome.CLEAR, stated={"preg_thalassemia": TurnOutcome.CLARIFY},
            utterance="नहीं, बस थैलेसीमिया का मतलब बता दीजिए"))
        self.assertEqual((out.phase, out.close_reason, out.cursor.qid), (Phase.ASKING, None, "preg_thalassemia"),
                         "a clarify she asked for is a question to answer, never a reason to hang up")
        self.assertIs(out.answers["close_anything_else"].state, AnswerStatus.CLEAR,
                      "her own answer to the closing capture is still on the record")

    def test_active_clarify_never_loses_cursor_to_stated_clarify_and_jump_topic_withholds_patient_safe(self):
        # Call #47 & #57: one clarify turn carried `stated_` clarify revisions for two questions that had already
        # been answered CLEAR. `_apply_stated` popped both answers and handed back the first as `clarify_qid`, so
        # the call asked again about a question she had answered, and the answer she gave was erased from the record.
        # --- Subtest A: the reducer's own guard, independent of whatever Decisions decides ------------------
        two = self.past_bridge((AMAN, MEERA))
        st = self.at(two, "preg_thalassemia")
        st.answers["body_regression"] = AnswerRecord(state=AnswerStatus.CLEAR)
        out = reduce_turn(self.shishu, st, TurnCommand(
            outcome=TurnOutcome.CLARIFY,
            stated={"body_regression": TurnOutcome.CLARIFY, "fam_lumps": TurnOutcome.CLARIFY},
            utterance="थैलेसीमिया का मतलब क्या है?"))
        rec = out.answers.get("body_regression")
        self._assert_all({
            "a clarify of the ACTIVE question never moves the cursor to a sibling": out.cursor.qid == "preg_thalassemia",
            "and an already-answered CLEAR question is never popped back to unanswered":
                rec is not None and rec.state is AnswerStatus.CLEAR,
        }, f"cursor={out.cursor.qid} body_regression={rec and rec.state.value} "
           f"answers={sorted(out.answers)}")

        # --- Subtest B: a two-topic sentence is nobody's patient-safe quote, even with an empty `stated` ----
        # Call #57 Step 5: `jump_topic` fired on a sentence that answered the active question AND corrected an
        # earlier one. Decisions put both topics in `utterance` but filled no `stated_` head, so `_redact_note`
        # saw `stated == {}` and let the whole sentence through as patient-safe words — onto the mother's PDF.
        utt = "हां, बुआ को कैंसर हुआ था और पहले नीली लाइट वाले सवाल में गलती हो गई थी"
        fam = reduce_turn(self.shishu, self.at(two, "fam_cancer"), TurnCommand(
            outcome=TurnOutcome.CONCERN, followup_answered=True, jump_topic=True, stated={}, utterance=utt)).answers["fam_cancer"]
        self._assert_all({
            "the doctor's record keeps her exact words": utt in fam.verbatim_text,
            "a mid-sentence topic change is never patient-safe": fam.patient_safe_text == "",
        }, f"verbatim={fam.verbatim_text!r} patient_safe={fam.patient_safe_text!r}")

        # --- Subtest C: the only clarify a clarify turn may hand the call is the question she just left -------
        # A blanket "drop every stated clarify while clarifying" also drops the legitimate one: she said she did
        # not know what thalassemia is (UNKNOWN), moved on, and asked what the term means. That ONE clarify names
        # the question she just left; an unasked future question, a settled CLEAR/CONCERN answer, or two-or-more
        # heads scatter-firing `clarify` at once (Call #47's ASR garble) name nobody and must never move the call.
        def clarify_turn(**stated: TurnOutcome) -> CallState:
            base = self.past_bridge((AMAN, MEERA))
            base = self.at(base, "fam_related")
            base.last_answered_qid = "preg_thalassemia"
            base.answers["preg_thalassemia"] = AnswerRecord(state=AnswerStatus.UNKNOWN)
            base.answers["body_regression"] = AnswerRecord(state=AnswerStatus.CLEAR)
            base.answers["fam_lumps"] = AnswerRecord(state=AnswerStatus.CONCERN)
            return reduce_turn(self.shishu, base, TurnCommand(
                outcome=TurnOutcome.CLARIFY, stated=dict(stated), utterance="थैलेसीमिया का कतलब होता है?"))

        kept = clarify_turn(preg_thalassemia=TurnOutcome.CLARIFY)
        self._assert_all({
            "the UNKNOWN question she just left may be asked again":
                (kept.cursor.qid, kept.answers["preg_thalassemia"].state if kept.answers.get("preg_thalassemia") else None)
                == ("preg_thalassemia", None),
        }, f"cursor={kept.cursor.qid} preg_thalassemia={kept.answers.get('preg_thalassemia')}")

        blocked = {
            "an unasked future question (Call #47 `fam_bleeding`) never hijacks the cursor":
                ({"fam_bleeding": TurnOutcome.CLARIFY}, {"fam_bleeding": None}),
            "a settled CLEAR answer is never popped back to unanswered":
                ({"body_regression": TurnOutcome.CLARIFY}, {"body_regression": AnswerStatus.CLEAR}),
            "a settled CONCERN answer is never popped back to unanswered":
                ({"fam_lumps": TurnOutcome.CLARIFY}, {"fam_lumps": AnswerStatus.CONCERN}),
            "a scatter of clarify heads (Call #47/#57) names nobody":
                ({"preg_thalassemia": TurnOutcome.CLARIFY, "fam_bleeding": TurnOutcome.CLARIFY,
                  "fam_transfusion": TurnOutcome.CLARIFY},
                 {"preg_thalassemia": AnswerStatus.UNKNOWN, "fam_bleeding": None, "fam_transfusion": None}),
        }
        for label, (stated, want) in blocked.items():
            with self.subTest(subtest="C", invariant=label):
                out_c = clarify_turn(**stated)
                kept_states = {q: (r.state if (r := out_c.answers.get(q)) is not None else None) for q in stated}
                self._assert_all({
                    "the cursor stays on the question she was asked": out_c.cursor.qid == "fam_related",
                    "and every sibling answer is exactly as it was": kept_states == want,
                }, f"{label}: stated={ {k: v.value for k, v in stated.items()} } cursor={out_c.cursor.qid} "
                   f"answers={ {k: (v.value if v else None) for k, v in kept_states.items()} }")

    @staticmethod
    def _leak_context(proj: reports.ProjectedReport, utterance: str, qids) -> str:
        """The patient-facing lines that quote the whole sentence onto every item it mentions."""
        return (f"utterance={utterance!r} qids={tuple(qids)} "
                f"yes_items={list(proj.yes_items)} patient_questions={list(proj.patient_questions)}")


if __name__ == "__main__":
    unittest.main()
