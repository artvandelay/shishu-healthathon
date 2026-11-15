"""Structural invariants for the catalog, typed state models, and Hindi copy."""

import json
import unittest
from pathlib import Path

import copy_hi  # noqa: E402
import triage_catalog  # noqa: E402
from models import (  # noqa: E402
    AnswerRecord, AnswerStatus, CallState, CallerProfile, Child, CloseReason, CursorStep,
    FollowupSlot, HoldReason, MaritalStatus, NoteText, Phase, SideAction, SubStep, TurnCommand,
    TurnOutcome, dedupe_clauses,
)
from triage_catalog import CATALOG, get_spec, routable_qids, routable_specs, spec_or_none  # noqa: E402

_SRC = Path(copy_hi.__file__).read_text(encoding="utf-8")
_PROMPT_AT = _SRC.index("GEMINI_LIVE_SYSTEM_PROMPT: str = ")
_PROMPT_END = _SRC.index('""".strip()', _PROMPT_AT)
_PROMPT_ONLY = _SRC[_PROMPT_AT:_PROMPT_END]
_OUTSIDE_PROMPT = _SRC[:_PROMPT_AT] + _SRC[_PROMPT_END:]
_BANNED_EN = ("Duchenne", "Hemophilia", "Haemophilia", "Pompe", "Gaucher", "SCID", "Wiskott", "SMA", "MPS")
_BANNED_HI = ("ड्यूशेन", "हीमोफीलिया", "पॉम्पे", "गॉशर", "इम्यूनोडेफिशिएंसी", "म्यूकोपॉलीसेकेराइडोसिस")
_PROMPT_BANNED = tuple(word for word in (*_BANNED_EN, *_BANNED_HI) if word != "Haemophilia")
_KIND_TABLE = {
    "profile": (1, 0, 0, 0, 0, 0, 0, 0, 0),
    "bridge": (0, 1, 0, 0, 0, 0, 0, 0, 1),
    "routing_gate": (0, 0, 1, 0, 0, 1, 1, 0, 1),
    "gate": (0, 0, 0, 1, 0, 1, 1, 1, 1),
    "open_capture": (0, 0, 0, 0, 1, 1, 0, 1, 1),
}
_FLAGS = (
    "is_profile", "is_bridge", "is_routing_gate", "is_clinical_gate", "is_open_capture",
    "is_routable", "is_rewindable", "is_reportable", "allows_yes_no",
)


def _kids(*rows):
    return [
        Child(id=f"child_{index}", age_years=age, age_label_hi=label, sex=sex, name=name)
        for index, (age, label, sex, name) in enumerate(rows, start=1)
    ]


class CatalogStructureTests(unittest.TestCase):
    def test_counts_flags_and_lookup(self):
        self.assertEqual((len(CATALOG), len(routable_specs()), len(routable_qids())), (62, 58, 58))
        self.assertTrue(all(spec.is_routable for spec in routable_specs()))
        self.assertEqual([s.id for s in CATALOG if s.shares_recorded_concerns], ["fam_same_problem", "close_anything_else"])
        self.assertEqual([s.id for s in CATALOG if s.updates_ever_pregnant], ["ever_pregnant"])
        self.assertIsNone(spec_or_none(""))
        self.assertIsNone(spec_or_none("not_a_qid"))
        self.assertIs(spec_or_none("intro_name"), get_spec("intro_name"))
        with self.assertRaises(KeyError):
            get_spec("not_a_qid")

    def test_property_truth_table_matches_kind(self):
        seen = set()
        for spec in CATALOG:
            seen.add(spec.kind)
            self.assertEqual(tuple(int(getattr(spec, flag)) for flag in _FLAGS), _KIND_TABLE[spec.kind], spec.id)
        self.assertEqual(seen, set(_KIND_TABLE))
        scoped = {spec.id for spec in CATALOG if spec.is_child_scoped}
        self.assertEqual(scoped, {s.id for s in CATALOG if s.group in triage_catalog.CHILD_GROUPS and "no_children" in s.skip_if})
        self.assertEqual(len(scoped), 36)

    def test_spec_is_frozen_and_slotted(self):
        spec = get_spec("body_fits")
        self.assertFalse(hasattr(spec, "__dict__"))
        with self.assertRaises(AttributeError):
            spec.kind = "bridge"


class NoteTextFirewallTests(unittest.TestCase):
    def test_diagnostic_clauses_never_reach_mother_copy(self):
        note = NoteText.from_text("डॉक्टर ने Duchenne बताया; दो बार दौरे आए")
        note = note.append("Pompe का नाम लिया", diagnostic=True).append("अब ठीक है")
        self.assertEqual(note.verbatim, "डॉक्टर ने Duchenne बताया; दो बार दौरे आए; Pompe का नाम लिया; अब ठीक है")
        self.assertEqual(note.mother_safe, "डॉक्टर ने Duchenne बताया; दो बार दौरे आए; अब ठीक है")

    def test_answer_record_slots_keep_verbatim_and_strip_diagnostic(self):
        record = AnswerRecord(
            state=AnswerStatus.CONCERN,
            child_ids=["child_1"],
            note=NoteText.from_text("जन्म के बाद NICU में रखा;  ऑपरेशन   से हुआ"),
            followups={
                "nicu_reason": FollowupSlot(state=AnswerStatus.CONCERN, note=NoteText.from_text("साँस की दिक्कत")),
                "nicu_days": FollowupSlot(state=AnswerStatus.CONCERN, note=NoteText.from_text("SMA बताया", diagnostic=True)),
            },
        )
        self.assertEqual(record.verbatim_text, "जन्म के बाद NICU में रखा; ऑपरेशन से हुआ; साँस की दिक्कत; SMA बताया")
        self.assertEqual(record.mother_safe_text, "जन्म के बाद NICU में रखा; ऑपरेशन से हुआ; साँस की दिक्कत")
        self.assertNotIn("SMA", record.mother_safe_text)

    def test_diagnostic_only_text_and_dedupe(self):
        self.assertEqual(NoteText.from_text("MPS; MPS", diagnostic=True).verbatim, "MPS")
        self.assertEqual(NoteText.from_text("MPS; MPS", diagnostic=True).mother_safe, "")
        self.assertEqual(dedupe_clauses(["a; b", "B ;  c ", "", "a"]), "a; b; c")


class CallStateTests(unittest.TestCase):
    def test_round_trip_json(self):
        state = CallState(
            profile=CallerProfile(
                caller_name="रेखा", marital_status=MaritalStatus.MARRIED, ever_pregnant=True,
                children_stated=True, children=_kids((8.0, "8 साल", "male", None), (2.0, "2 साल", "female", "मीरा")),
            ),
            cursor=CursorStep(qid="body_fits", hold=HoldReason.CHILDREN_AGES, attempts=1, close_rounds=2),
            last_answered_qid="body_fits",
            answers={"body_fits": AnswerRecord(
                state=AnswerStatus.CONCERN, child_ids=["child_1"], note=NoteText.from_text("दो बार आए"),
                followups={"fits_onset": FollowupSlot(state=AnswerStatus.UNKNOWN)},
            )},
        )
        self.assertEqual(CallState.from_dict(state.to_dict()), state)
        self.assertEqual(CallState.from_dict(state.to_dict()).step_token, state.step_token)
        json.dumps(state.to_dict())
        self.assertEqual(CallState.from_dict(None), CallState())
        self.assertEqual(CallState.from_dict({}), CallState())
        self.assertEqual([child.id for child in state.profile.ordered_children], ["child_1", "child_2"])

    def test_step_token_distinct_across_every_step_shape(self):
        shapes = (
            CursorStep(), CursorStep(substep=SubStep.WHICH_CHILD),
            CursorStep(substep=SubStep.FOLLOWUP), CursorStep(substep=SubStep.FOLLOWUP, followup_idx=1),
            CursorStep(hold=HoldReason.NAME_FRAGMENT), CursorStep(attempts=1), CursorStep(attempts=2),
            CursorStep(close_rounds=1), CursorStep(substep=SubStep.FOLLOWUP, followup_idx=1, attempts=1),
        )
        tokens = [CallState(cursor=step).step_token for step in shapes]
        self.assertEqual(len(set(tokens)), len(tokens), tokens)
        closing = [CallState(phase=Phase.CLOSING, close_reason=reason).step_token for reason in CloseReason]
        self.assertEqual(len(set(closing)), len(CloseReason))
        self.assertEqual(tokens[0], "intro_name")
        self.assertIn(f"closing:{CloseReason.COMPLETE}", closing)
        self.assertFalse(set(closing) & set(tokens))

    def test_validation_rejects_unknown_keys_and_bad_numbers(self):
        for build in (
            lambda: CallState.from_dict({"phase": "asking", "bogus": 1}),
            lambda: CursorStep(attempts=-1),
            lambda: Child(id="child_1", age_years=-1.0, age_label_hi="x"),
        ):
            with self.assertRaises(Exception):
                build()
        self.assertIs(TurnCommand().outcome, TurnOutcome.NOT_ABOUT_ACTIVE)
        self.assertEqual(SideAction.NONE.value, "")
        self.assertTrue(MaritalStatus.ABOUT_TO_MARRY.is_unmarried_or_engaged)
        self.assertFalse(MaritalStatus.MARRIED.is_unmarried_or_engaged)
        self.assertTrue(AnswerStatus.UNKNOWN.is_answered)
        self.assertFalse(AnswerStatus.NOT_APPLICABLE.is_answered)


class CopyHiTests(unittest.TestCase):
    def test_age_labels(self):
        self.assertEqual(
            (copy_hi.age_label_hi(0.5), copy_hi.age_label_hi(0.25), copy_hi.age_label_hi(3.0), copy_hi.age_label_hi(2.5)),
            ("6 महीने", "3 महीने", "3 साल", "2.5 साल"),
        )

    def test_which_child_middle_child_phrasing(self):
        three = _kids((8.0, "8 साल", "male", None), (5.0, "5 साल", "female", None), (2.0, "2 साल", "female", None))
        self.assertEqual(
            copy_hi.which_child_hi(three),
            "यह किस बच्चे की बात है — 8 साल वाले बेटे, 5 साल वाली बेटी (बीच वाली), या 2 साल वाली बेटी?",
        )
        self.assertEqual(sum("(बीच" in part for part in copy_hi.which_child_hi(three).split("या")), 1)
        self.assertEqual(
            copy_hi.which_child_hi(three[:2]),
            "यह किस बच्चे की बात है — 8 साल वाले बेटे, या 5 साल वाली बेटी?",
        )

    def test_question_hi_for_marital_and_multi_child_variants(self):
        thal, nicu = get_spec("preg_thalassemia"), get_spec("birth_nicu")
        kids = _kids((8.0, "8 साल", "male", None), (2.0, "2 साल", "female", None))
        self.assertEqual(copy_hi.question_hi_for(CallerProfile(children=kids), thal), thal.question_hi)
        self.assertEqual(copy_hi.question_hi_for(CallerProfile(children=kids), nicu), nicu.question_multi_hi)
        engaged = CallerProfile(marital_status=MaritalStatus.ABOUT_TO_MARRY, children=kids)
        self.assertEqual(copy_hi.question_hi_for(engaged, thal), thal.question_about_hi)
        self.assertEqual(copy_hi.question_hi_for(CallerProfile(marital_status=MaritalStatus.UNMARRIED), thal), thal.question_unmarried_hi)
        self.assertEqual(copy_hi.question_hi_for(CallerProfile(children=kids[:1]), nicu), nicu.question_hi)

    def test_child_labels_and_mother_report_builders(self):
        profile = CallerProfile(children=_kids((8.0, "8 साल", "male", None), (5.0, "5 साल", "female", "मीरा")))
        self.assertEqual(copy_hi.children_label_hi(profile), "8 साल बेटा · मीरा 5 साल बेटी")
        self.assertEqual(copy_hi.children_label_hi(profile, ["child_2"]), "मीरा 5 साल बेटी")
        self.assertEqual(copy_hi.children_label_en(profile), "child_1 (male, age 8.0), child_2 (female, age 5.0, मीरा)")
        self.assertEqual(copy_hi.children_label_hi(CallerProfile()), "")

        fits, closing = get_spec("body_fits"), get_spec("close_anything_else")
        self.assertEqual(
            copy_hi.mother_concern_question_hi(fits, "दो बार आए", "5 साल बेटी"),
            "5 साल बेटी के लिए: मैंने बताया कि दो बार आए। डॉक्टर, इसकी जाँच क्या होनी चाहिए?",
        )
        self.assertEqual(
            copy_hi.mother_concern_question_hi(fits, "", ""),
            f"मैंने हाँ कहा कि {fits.stem_hi}। डॉक्टर, अब क्या जाँच करनी चाहिए?",
        )
        self.assertEqual(
            copy_hi.mother_concern_question_hi(closing, "", ""),
            "मैंने कॉल के अंत में एक और बात दर्ज करवाई थी। डॉक्टर, उस पर क्या सलाह है?",
        )
        self.assertEqual(copy_hi.mother_yes_summary_hi(fits, "दो बार आए", ""), f"{fits.stem_hi} — दो बार आए।")
        self.assertEqual(copy_hi.mother_yes_summary_hi(closing, "सिर दर्द", ""), "अन्य बात — सिर दर्द।")
        self.assertEqual(copy_hi.mother_yes_summary_hi(closing, "", ""), "डॉक्टर को बताने के लिए एक अन्य बात दर्ज की गई।")

    def test_fit_questions_bounds_and_fillers(self):
        fit = copy_hi.fit_mother_questions_hi
        self.assertEqual(len(fit([], emergency=True, has_children=False)), 3)
        self.assertEqual(fit(["क;  ख", "क;  ख", "ग"], emergency=False, has_children=True), ["क; ख", "ग", copy_hi.ROUTINE_HI[0]])
        self.assertEqual(fit(["x"] * 9, emergency=False, has_children=False), ["x", *copy_hi.ROUTINE_CHILDLESS_HI[:2]])
        self.assertEqual(len(fit([f"q{index}" for index in range(9)], emergency=True, has_children=True)), 6)
        self.assertEqual(len(fit([f"q{index}" for index in range(9)], emergency=False, has_children=False)), 6)
        kept = fit(list(copy_hi.ROUTINE_HI), emergency=True, has_children=True)
        self.assertEqual(kept, list(copy_hi.EMERGENCY_HI))

    def test_rare_disease_names_exist_only_as_the_voice_prompt_ban(self):
        for word in (*_BANNED_EN, *_BANNED_HI):
            self.assertNotIn(word, _OUTSIDE_PROMPT, word)
        for word in _PROMPT_BANNED:
            self.assertIn(word, _PROMPT_ONLY, word)
        for token in ("next_question", "clarify", "recorded_concerns", "SYSTEM CUE", "warm female voice"):
            self.assertIn(token, _PROMPT_ONLY)

    def test_cues_closing_and_holds_cover_every_enum_member(self):
        self.assertEqual(set(copy_hi.HOLD_HI), set(HoldReason))
        for reason in CloseReason:
            text = copy_hi.closing_hi(CallState(phase=Phase.CLOSING, close_reason=reason), has_clinical_concern=False)
            self.assertTrue(text.strip(), reason)
            self.assertNotIn("डॉक्टर की ज़रूरत नहीं", text)
        emergency = copy_hi.closing_hi(CallState(phase=Phase.CLOSING, close_reason=CloseReason.EMERGENCY), has_clinical_concern=True)
        self.assertIn("108", emergency)
        named = copy_hi.closing_hi(CallState(phase=Phase.CLOSING, profile=CallerProfile(caller_name="आशा")), has_clinical_concern=True)
        self.assertTrue(named.startswith("आशा जी, "))
        self.assertIn("speak only this Hindi", copy_hi.speak_cue_text("क्या बात है?"))
        self.assertIn("then stop", copy_hi.speak_cue_text(named, closing=True))
        self.assertIn("did not understand", copy_hi.speak_cue_text("क्या बात है?", clarify=True))
        state = CallState(
            profile=CallerProfile(children=_kids((5.0, "5 साल", "male", "आरव"))),
            answers={
                "body_fits": AnswerRecord(
                    state=AnswerStatus.CONCERN,
                    child_ids=["child_1"],
                    note=NoteText.from_text("दो बार दौरे; Duchenne", diagnostic=False).append("Pompe", diagnostic=True),
                ),
                "ever_pregnant": AnswerRecord(state=AnswerStatus.CONCERN),
            },
        )
        recorded = copy_hi.recorded_concerns_hi(state)
        self.assertEqual(len(recorded), 1)
        self.assertNotIn("Pompe", recorded[0])
        self.assertIn("आरव 5 साल बेटा", recorded[0])
        with self.assertRaises(Exception):
            state.cursor.attempts = -1


if __name__ == "__main__":
    unittest.main()
