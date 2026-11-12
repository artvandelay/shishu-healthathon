"""Offline trajectories for the v3 catalog and call cursor."""

import json
import unittest

import call_cursor
import triage_catalog
from call_cursor import apply_profile, apply_turn, new_cursor_state, prompt_directive
from triage_catalog import active_specs, get_spec

CHILD_GATES = (
    "gate_birth_neonatal",
    "gate_cord_bcg",
    "gate_metabolic_crisis",
    "gate_milestones_neuro",
    "gate_sensory_cleft_heart",
    "gate_infections",
    "gate_bleeding_joints",
    "gate_transfusion_anemia",
    "gate_tummy_lumps",
    "gate_growth_shape",
    "gate_bones_fractures",
    "gate_urine_kidney",
    "gate_endocrine_private",
)
FAMILY_GATES = ("fam_related", "fam_loss_gate", "fam_conditions_gate")
PREG_HISTORY = (
    "gate_pregnancy_thal",
    "preg_conditions",
    "preg_scan",
    "preg_transfusion",
    "preg_delivery",
    "preg_heart",
)
SPOKEN_GATES = CHILD_GATES + ("gate_pregnancy_thal",) + FAMILY_GATES


def _child(age, sex="male", n=1, name=None):
    return {
        "id": f"child_{n}",
        "age_years": age,
        "age_label_hi": f"{age:g} साल",
        "sex": sex,
        "name": name,
    }


def _profile(marital="married", children=None, ever_pregnant=None, name="पूजा"):
    state, hold = apply_profile(new_cursor_state(), caller_name=name)
    assert hold is None, hold
    state, hold = apply_profile(state, marital_status=marital)
    assert hold is None, hold
    if marital != "unmarried":
        state, hold = apply_profile(state, children=list(children or []))
        assert hold is None, hold
    if ever_pregnant is not None:
        state, hold = apply_profile(state, ever_pregnant=ever_pregnant)
        assert hold is None, hold
    return state


def _clear_until(state, qid, limit=40):
    for _ in range(limit):
        if state["phase"] != "asking":
            return state
        if state["active_qid"] == qid and state["pending_followup_idx"] is None:
            return state
        if state["active_qid"] == "ever_pregnant":
            state, hold = apply_profile(state, ever_pregnant=False)
            assert hold is None, hold
            continue
        prev = (state["active_qid"], state["pending_followup_idx"])
        state = apply_turn(state, "clear")
        if (state["active_qid"], state["pending_followup_idx"]) == prev and state["phase"] == "asking":
            raise AssertionError(f"cursor stuck at {prev}")
    raise AssertionError(f"never reached {qid}; still at {state['active_qid']}")


def _opened_modules(state):
    """Answer a concern that is already recorded as the active question's next modules."""
    found = []
    guard = 0
    while state["phase"] == "asking" and get_spec(state["active_qid"]).kind == "module":
        found.append(state["active_qid"])
        state = apply_turn(state, "clear")
        guard += 1
        if guard > 30:
            break
    return found, state


class CatalogTests(unittest.TestCase):
    def test_spine_matches_locked_gates(self):
        gates = [spec.id for spec in triage_catalog.CATALOG if spec.kind == "gate"]
        self.assertEqual(gates, list(SPOKEN_GATES))
        self.assertEqual(len(gates), 17)
        profiles = [spec.id for spec in triage_catalog.CATALOG if spec.kind == "profile"]
        self.assertEqual(profiles, ["intro_name", "intro_married", "intro_children"])
        self.assertEqual(get_spec("intro_bridge").kind, "bridge")
        self.assertEqual(get_spec("close_anything_else").kind, "open_capture")
        self.assertEqual(get_spec("intro_children").skip_if, ("unmarried",))
        self.assertIn("mod_heavy_periods", get_spec("gate_bleeding_joints").opens)
        self.assertIn("body_puberty", get_spec("gate_endocrine_private").opens)
        self.assertEqual(len(active_specs()), len(triage_catalog.CATALOG))
        json.dumps(new_cursor_state())

    def test_gate_hindi_is_the_locked_wording(self):
        self.assertIn("काँच के बक्से", get_spec("gate_birth_neonatal").question_hi)
        self.assertIn("थैलेसीमिया", get_spec("gate_pregnancy_thal").question_hi)
        self.assertTrue(get_spec("gate_bleeding_joints").question_hi.startswith("क्या"))
        self.assertEqual(get_spec("intro_name").question_hi, "नमस्ते, मैं शिशु हूँ। आपका नाम बताइएगा प्लीज़?")

    def test_thalassemia_yes_means_an_abnormal_result(self):
        question = get_spec("preg_thalassemia").question_hi
        self.assertIn("माइनर", question)
        self.assertIn("ट्रेट", question)
        self.assertIn("दिक्कत", question)
        self.assertNotIn("जाँच हुई थी", question)
        state = _profile(marital="unmarried")
        state = _clear_until(state, "preg_thalassemia")
        spoken = prompt_directive(state)
        self.assertIn("दिक्कत", spoken)
        self.assertIn("थैलेसीमिया", spoken)
        self.assertNotIn("पति", spoken)
        self.assertNotIn("करवा लें", spoken)
        self.assertNotIn("हुई है", spoken)


class ProfileTests(unittest.TestCase):
    def test_name_fragment_and_missing_ages_hold(self):
        state, hold = apply_profile(new_cursor_state(), caller_name="मेरा नाम")
        self.assertEqual(hold, "name_fragment")
        self.assertEqual(state["active_qid"], "intro_name")
        self.assertIsNone(state["profile"]["caller_name"])

        state, hold = apply_profile(state, caller_name="पूजा")
        self.assertIsNone(hold)
        self.assertEqual(state["active_qid"], "intro_married")
        state, hold = apply_profile(state, marital_status="married")
        self.assertEqual(state["active_qid"], "intro_children")
        state, hold = apply_profile(state, children=[{"id": "child_1", "age_years": None, "sex": "male"}])
        self.assertEqual(hold, "children_ages")
        self.assertEqual(state["active_qid"], "intro_children")
        self.assertEqual(state["profile"]["children"], [])

    def test_stated_child_count_without_ages_does_not_mark_childless(self):
        state, hold = apply_profile(new_cursor_state(), caller_name="पूजा")
        state, hold = apply_profile(state, marital_status="married")
        self.assertEqual(state["active_qid"], "intro_children")
        state, hold = apply_profile(state, children=[], child_count=2)
        self.assertEqual(hold, "children_ages")
        self.assertEqual(state["profile"]["children"], [])
        self.assertIsNone(state["profile"]["ever_pregnant"])
        self.assertEqual(state["active_qid"], "intro_children")
        state, hold = apply_profile(state, child_count=2)
        self.assertEqual(hold, "children_ages")
        self.assertEqual(state["active_qid"], "intro_children")
        state, hold = apply_profile(state, children=[_child(4, "male")], child_count=2)
        self.assertEqual(hold, "children_ages")
        self.assertEqual(state["profile"]["children"], [])
        state, hold = apply_profile(
            state,
            children=[_child(6, "male", 1), _child(2, "female", 2)],
            child_count=2,
        )
        self.assertIsNone(hold)
        self.assertEqual(len(state["profile"]["children"]), 2)
        self.assertTrue(state["profile"]["ever_pregnant"])
        self.assertEqual(state["active_qid"], "intro_bridge")

    def test_children_set_ever_pregnant(self):
        state = _profile(children=[_child(4, "male")])
        self.assertTrue(state["profile"]["ever_pregnant"])
        self.assertEqual(state["active_qid"], "intro_bridge")
        self.assertEqual(state["profile"]["children"][0]["id"], "child_1")


class TrajectoryTests(unittest.TestCase):
    def test_healthy_one_child_finishes_within_19_steps_and_opens_nothing(self):
        state = _profile(children=[_child(4, "male")])
        asked = []
        while state["phase"] == "asking":
            self.assertNotEqual(get_spec(state["active_qid"]).kind, "module")
            asked.append(state["active_qid"])
            state = apply_turn(state, "clear")
            self.assertEqual(state["module_queue"], [])
            if len(asked) > 19:
                break
        self.assertLessEqual(len(asked), 19)
        self.assertEqual(state["phase"], "closing")
        self.assertEqual(state["close_reason"], "complete")
        for qid in ("intro_bridge",) + CHILD_GATES + ("gate_pregnancy_thal",) + FAMILY_GATES + ("close_anything_else",):
            self.assertIn(qid, asked)
        opened = [
            qid for qid, answer in state["answers"].items()
            if get_spec(qid).kind == "module" and answer["state"] in ("clear", "concern", "unknown")
        ]
        self.assertEqual(opened, [])

    def test_unmarried_marks_child_spouse_and_miscarriage_not_applicable(self):
        state = _profile(marital="unmarried")
        self.assertEqual(state["active_qid"], "intro_bridge")
        self.assertEqual(state["answers"]["intro_children"]["state"], "not_applicable")
        for spec in active_specs():
            child_item = "no_children" in spec.skip_if or spec.id.startswith(("birth_", "body_", "ill_", "mod_"))
            child_item = child_item or spec.id in CHILD_GATES
            if child_item:
                self.assertEqual(state["answers"][spec.id]["state"], "not_applicable", spec.id)
        for qid in ("fam_related", "fam_miscarriage", "gate_pregnancy_thal") + PREG_HISTORY:
            self.assertEqual(state["answers"][qid]["state"], "not_applicable", qid)
        self.assertNotEqual(state["answers"].get("preg_thalassemia", {}).get("state"), "not_applicable")
        self.assertNotIn("fam_infant_death", state["answers"])
        state = _clear_until(state, "preg_thalassemia")
        self.assertIn("थैलेसीमिया", prompt_directive(state))
        self.assertNotIn("पति", prompt_directive(state))

    def test_married_childless_never_pregnant_asks_thalassemia_and_family(self):
        state = _profile(children=[])
        state = apply_turn(state, "clear")
        self.assertEqual(state["active_qid"], "ever_pregnant")
        self.assertIn("गर्भवती", prompt_directive(state))
        state, hold = apply_profile(state, ever_pregnant=False)
        self.assertIsNone(hold)
        self.assertFalse(state["profile"]["ever_pregnant"])
        self.assertEqual(state["active_qid"], "preg_thalassemia")
        for qid in ("gate_pregnancy_thal", "preg_conditions", "preg_scan", "preg_transfusion", "preg_delivery", "fam_miscarriage"):
            self.assertEqual(state["answers"][qid]["state"], "not_applicable", qid)
        state = apply_turn(state, "clear")
        self.assertEqual(state["active_qid"], "fam_related")
        state = apply_turn(state, "clear")
        self.assertEqual(state["active_qid"], "fam_loss_gate")
        self.assertNotIn("गर्भ", prompt_directive(state))
        self.assertIn("मौत", prompt_directive(state))
        state = apply_turn(state, "concern")
        self.assertIn("fam_infant_death", [state["active_qid"], *state["module_queue"]])
        self.assertNotIn("fam_miscarriage", state["module_queue"])
        self.assertEqual(state["answers"]["fam_miscarriage"]["state"], "not_applicable")

    def test_one_year_old_boy_never_opens_age_barred_modules(self):
        state = _profile(children=[_child(1, "male")])
        for qid in ("mod_heavy_periods", "mod_tooth_bleed", "body_puberty"):
            self.assertEqual(state["answers"][qid]["state"], "not_applicable", qid)
        state = _clear_until(state, "gate_bleeding_joints")
        state = apply_turn(state, "concern")
        seen, state = _opened_modules(state)
        self.assertIn("body_bleeding_bruise", seen)
        for barred in ("mod_heavy_periods", "mod_tooth_bleed", "body_puberty"):
            self.assertNotIn(barred, seen)
            self.assertNotIn(barred, state["module_queue"])
        state = _clear_until(state, "gate_endocrine_private")
        state = apply_turn(state, "concern")
        seen, _state = _opened_modules(state)
        self.assertIn("body_private_parts", seen)
        self.assertNotIn("body_puberty", seen)
        self.assertNotIn("mod_heavy_periods", seen)
        self.assertNotIn("mod_tooth_bleed", seen)

    def test_twelve_year_old_girl_opens_periods_and_puberty(self):
        state = _profile(children=[_child(12, "female")])
        self.assertNotIn("mod_heavy_periods", state["answers"])
        self.assertNotIn("body_puberty", state["answers"])
        state = _clear_until(state, "gate_bleeding_joints")
        state = apply_turn(state, "concern")
        seen, state = _opened_modules(state)
        self.assertIn("mod_heavy_periods", seen)
        self.assertIn("mod_tooth_bleed", seen)
        state = _clear_until(state, "gate_endocrine_private")
        state = apply_turn(state, "concern")
        seen, _state = _opened_modules(state)
        self.assertIn("body_puberty", seen)

    def test_multi_child_concern_requires_child_attribution(self):
        kids = [_child(6, "male", 1), _child(2, "female", 2)]
        state = _profile(children=kids)
        state = _clear_until(state, "gate_bleeding_joints")
        state = apply_turn(state, "concern")
        self.assertEqual(state["active_qid"], "gate_bleeding_joints")
        self.assertEqual(state["pending_followup_idx"], -1)
        self.assertEqual(state["answers"]["gate_bleeding_joints"]["child_ids"], [])
        self.assertEqual(state["answers"]["gate_bleeding_joints"]["state"], "concern")
        self.assertIn("किस बच्चे", prompt_directive(state))
        held = apply_turn(state, "clarify")
        self.assertEqual(held["active_qid"], "gate_bleeding_joints")
        self.assertEqual(held["pending_followup_idx"], -1)
        state = apply_turn(state, "concern", child_ref="child_2")
        self.assertEqual(state["answers"]["gate_bleeding_joints"]["child_ids"], ["child_2"])
        self.assertEqual(state["answers"]["gate_bleeding_joints"]["state"], "concern")
        self.assertEqual(state["active_qid"], "body_bleeding_bruise")

    def test_which_child_accepts_all_and_unknown_does_not_stick(self):
        kids = [_child(6, "male", 1), _child(2, "female", 2)]
        for ref in ("both", "all", "दोनों", "सभी", "Both"):
            state = _profile(children=kids)
            state = _clear_until(state, "gate_bleeding_joints")
            state = apply_turn(state, "concern", child_ref=ref)
            self.assertEqual(
                state["answers"]["gate_bleeding_joints"]["child_ids"],
                ["child_1", "child_2"],
                ref,
            )
            self.assertNotEqual(state["pending_followup_idx"], -1, ref)
            self.assertEqual(state["active_qid"], "body_bleeding_bruise", ref)

        state = _profile(children=kids)
        state = _clear_until(state, "gate_bones_fractures")
        state = apply_turn(state, "concern")
        self.assertEqual(state["pending_followup_idx"], -1)
        stuck = apply_turn(state, "concern")
        self.assertEqual(stuck["pending_followup_idx"], -1)
        state = apply_turn(state, "unknown")
        self.assertEqual(state["pending_followup_idx"], 0)
        self.assertEqual(state["active_qid"], "gate_bones_fractures")
        self.assertEqual(state["answers"]["gate_bones_fractures"]["state"], "concern")
        self.assertIn("हड्डी", prompt_directive(state))
        state = apply_turn(state, "clear", note="पैर टेढ़े नहीं")
        self.assertEqual(state["active_qid"], "gate_urine_kidney")
        self.assertNotEqual(state["pending_followup_idx"], -1)

    def test_clarify_does_not_advance(self):
        state = _profile(children=[_child(8, "male")])
        state = _clear_until(state, "gate_bones_fractures")
        state = apply_turn(state, "concern", note="दो बार टूटी")
        self.assertEqual(state["active_qid"], "gate_bones_fractures")
        self.assertEqual(state["pending_followup_idx"], 0)
        held = apply_turn(state, "clarify")
        self.assertEqual(held["active_qid"], "gate_bones_fractures")
        self.assertEqual(held["pending_followup_idx"], 0)
        self.assertNotIn("fracture_count_or_bowing", held["answers"]["gate_bones_fractures"]["followups"])
        state = apply_turn(state, "clear", note="पैर टेढ़े नहीं")
        self.assertEqual(state["answers"]["gate_bones_fractures"]["state"], "concern")
        self.assertEqual(state["answers"]["gate_bones_fractures"]["notes"], "दो बार टूटी")
        self.assertEqual(
            state["answers"]["gate_bones_fractures"]["followups"]["fracture_count_or_bowing"]["state"],
            "clear",
        )
        self.assertEqual(state["active_qid"], "gate_urine_kidney")

    def test_emergency_closes_immediately(self):
        state = _profile(children=[_child(3, "female")])
        state = _clear_until(state, "gate_infections")
        state = apply_turn(state, "emergency", note="अभी सांस नहीं ले रहा")
        self.assertEqual(state["phase"], "closing")
        self.assertEqual(state["close_reason"], "emergency")
        self.assertEqual(state["active_qid"], "gate_infections")
        stopped = apply_turn(state, "clear")
        self.assertEqual(stopped["phase"], "closing")
        self.assertEqual(stopped["active_qid"], "gate_infections")

    def test_caller_stop_closes(self):
        state = apply_turn(new_cursor_state(), "caller_stop")
        self.assertEqual(state["phase"], "closing")
        self.assertEqual(state["close_reason"], "caller_stop")


if __name__ == "__main__":
    unittest.main()
