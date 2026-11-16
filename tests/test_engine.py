"""Behavioral tests for `intake.engine` across synthetic fixtures and both real packs."""

from __future__ import annotations

import unittest
from pathlib import Path

from intake.engine import (
    MAX_STEP_ATTEMPTS, active_prompt, apply_profile_patch, close_technical_failure, eval_rule,
    followup_prompt, format_age_label, has_clinical_concern, last_answered_prompt,
    merge_transcript_chunk, new_state, question_say, quick_replies, quick_reply_command,
    recorded_concerns_summary, reduce_turn, roster_label_clinician, roster_label_patient,
    roster_resolved, roster_rows, say_text, step_kind, which_item_prompt,
)
from intake.models import (
    AnswerRecord, AnswerStatus, CallState, CloseReason, OptionSpec, PackConfig, Phase,
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

    def test_rules_branches_and_reopen(self):
        pack = make_pack(
            [
                {"id": "x", "kind": QuestionKind.SINGLE_CHOICE, "saves_to_profile": "kids",
                 "options": {"none": {"label": "no"}, "some": {"label": "yes"}}},
                gate("follow", skip_if=["no_kids"]),
                gate("after"),
            ],
            rules={"no_kids": {"field": "kids", "op": "eq", "value": "none"}},
        )
        st = play(pack, new_state(pack), TurnOutcome.ANSWERED, value="none")
        self.assertEqual((st.answers["follow"].state, st.cursor.qid), (AnswerStatus.NOT_APPLICABLE, "after"))
        reopened = reduce_turn(pack, st, TurnCommand(profile_patch=ProfilePatch(slots={"kids": "some"})))
        self.assertNotIn("follow", reopened.answers)

    def test_multi_choice_score_and_open_capture(self):
        pack = make_pack(
            [
                {"id": "m", "kind": QuestionKind.MULTI_CHOICE,
                 "options": {"chills": {"label": "Chills"}, "none": {"label": "None", "exclusive": True}}},
                {"id": "s", "kind": QuestionKind.SCORE, "score_rubric": ("mild", "mod", "severe"),
                 "concern_if": {"min_score": 1.5}},
                {"id": "o", "kind": QuestionKind.OPEN_CAPTURE, "say": "anything?"},
            ],
            prompts={"open_capture_more": "more?"},
        )
        st = play(pack, new_state(pack), TurnOutcome.ANSWERED, value=["chills", "none"])
        self.assertEqual((st.answers["m"].state, st.answers["m"].value), (AnswerStatus.CLEAR, ["none"]))
        st = play(pack, st, TurnOutcome.ANSWERED, value={"score": 2.0, "level": 2})
        self.assertEqual((st.answers["s"].state, st.answers["s"].choice_label), (AnswerStatus.CONCERN, "severe"))
        self.assertEqual(st.cursor.qid, "o")
        st = play(pack, st, TurnOutcome.CONCERN, utterance="night fear")
        self.assertEqual((st.cursor.open_rounds, active_prompt(pack, st)), (1, "more?"))
        st = play(pack, st, TurnOutcome.CLEAR)
        self.assertEqual((st.phase, st.close_reason, st.answers["o"].verbatim_text), (Phase.CLOSING, CloseReason.COMPLETE, "night fear"))

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

    def test_retry_exhaustion_records_unknown_and_moves_on(self):
        pack = make_pack([gate("g"), gate("next")])
        st = new_state(pack)
        for _ in range(MAX_STEP_ATTEMPTS - 1):
            st = play(pack, st, TurnOutcome.ANSWERED, value="bad")
            self.assertEqual(st.cursor.qid, "g")
        st = play(pack, st, TurnOutcome.ANSWERED, value="bad")
        self.assertEqual((st.answers["g"].state, st.cursor.qid), (AnswerStatus.UNKNOWN, "next"))


class RealPackIntegrationTests(unittest.TestCase):
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
        self.assertIn("बेझिझक खुलकर बोलकर", active_prompt(shishu, st))
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

    def test_fever_opd_pack_full_walk_with_branching_score_and_semantic_skip(self):
        opd = load_pack("fever-opd-hi")
        st = new_state(opd)
        self.assertEqual(st.cursor.qid, "patient_name")
        st = reduce_turn(opd, st, TurnCommand(profile_patch=ProfilePatch(slots={"patient_name": "सुनीता", "patient_age": 34})))
        self.assertEqual(st.cursor.qid, "intro_bridge")
        st = play(opd, st, TurnOutcome.CONCERN)
        self.assertEqual(st.cursor.qid, "fever_duration_bucket")
        st = play(opd, st, TurnOutcome.ANSWERED, value="gt_7_days")
        self.assertEqual(st.answers["fever_duration_bucket"].state, AnswerStatus.CONCERN)
        self.assertEqual(st.cursor.qid, "fever_nature")
        # Continuous fever -> fever_frequency must auto-skip!
        st = play(opd, st, TurnOutcome.ANSWERED, value="continuous")
        self.assertEqual(st.answers["fever_frequency"].state, AnswerStatus.NOT_APPLICABLE)
        self.assertEqual(st.cursor.qid, "fever_severity")
        st = play(opd, st, TurnOutcome.ANSWERED, value={"score": 2.4, "level": 2})
        self.assertEqual(st.answers["fever_severity"].state, AnswerStatus.CONCERN)
        # Confirmed dengue report prunes recent_travel while keeping jaundice_or_bleeding!
        st = reduce_turn(opd, st, TurnCommand(
            outcome=TurnOutcome.ANSWERED,
            value=["chills"],
            semantic_flags=("confirmed_malaria_or_dengue",),
        ))
        self.assertEqual(st.answers["recent_travel"].state, AnswerStatus.NOT_APPLICABLE)
        self.assertNotIn("jaundice_or_bleeding", st.answers)
        self.assertEqual(close_technical_failure(st).close_reason, CloseReason.TECHNICAL_FAILURE)


if __name__ == "__main__":
    unittest.main()
