"""Offline trajectories for the v4 atomic catalog and FormBharo cursor."""

import ast
import json
import re
import unittest
from pathlib import Path

import call_cursor
import triage_catalog
from call_cursor import (
    apply_profile,
    apply_turn,
    expect_yes_no,
    new_cursor_state,
    prompt_directive,
)
from triage_catalog import FOLLOWUP_HI, active_specs, get_spec

_DOC = Path(__file__).resolve().parents[2] / "docs" / "requirements-v4.md"
_PREG_HISTORY = (
    "preg_conditions",
    "preg_scan",
    "preg_transfusion",
    "preg_delivery",
    "fam_miscarriage",
)


def _child(age, sex="unknown", n=1, name=None):
    return {
        "id": f"child_{n}",
        "age_years": age,
        "age_label_hi": f"{int(round(age * 12))} महीने" if age < 1 else f"{age:g} साल",
        "sex": sex,
        "name": name,
    }


def _profile(marital="married", children=None, ever_pregnant=None, name="पूजा"):
    state, hold = apply_profile(new_cursor_state(), caller_name=name)
    assert hold is None, hold
    state, hold = apply_profile(state, marital_status=marital)
    assert hold is None, hold
    if marital == "married" or children:
        state, hold = apply_profile(state, children=list(children or []))
        assert hold is None, hold
    if ever_pregnant is not None and state["profile"]["ever_pregnant"] is not True:
        state, hold = apply_profile(state, ever_pregnant=ever_pregnant)
        assert hold is None, hold
    return state


def _clear_until(state, qid, limit=90):
    for _ in range(limit):
        if state["phase"] != "asking":
            return state
        if state["active_qid"] == qid and state["pending_followup_idx"] is None:
            return state
        if state["active_qid"] == "ever_pregnant":
            state = apply_turn(state, "clear")
            continue
        prev = (state["active_qid"], state["pending_followup_idx"])
        state = apply_turn(state, "clear")
        if (state["active_qid"], state["pending_followup_idx"]) == prev and state["phase"] == "asking":
            raise AssertionError(f"cursor stuck at {prev}")
    raise AssertionError(f"never reached {qid}; still at {state['active_qid']}")


def _cells(line):
    return [part.strip() for part in line.strip().strip("|").split("|")]


def _tokens(cell):
    found = re.findall(r"`([^`]+)`", cell)
    if not found or found == ["()"]:
        return ()
    if len(found) == 1 and found[0].startswith("("):
        inner = found[0].strip("()").strip()
        if not inner:
            return ()
        return tuple(part.strip().strip('"').strip("'") for part in inner.split(",") if part.strip())
    return tuple(found)


def _hindi(cell):
    raw = cell.strip()
    return "" if raw.startswith("*") else raw


def _spec_tables():
    lines = _DOC.read_text(encoding="utf-8").splitlines()
    follow_at = lines.index("### 4.1 Follow-Up Prompts (`FOLLOWUP_HI`)")
    intro_at = lines.index("### 4.2 Intro & Open Capture Items")
    clin_at = lines.index('### 4.3 Atomic Clinical Questions (50 Items — All `kind="gate"`, `opens=()`)')
    end = lines.index("---", clin_at)
    followups = []
    for line in lines[follow_at:intro_at]:
        if line.startswith("| `"):
            cells = _cells(line)
            followups.append((_tokens(cells[0])[0], _hindi(cells[1])))
    intros = []
    for line in lines[intro_at:clin_at]:
        if line.startswith("| `"):
            cells = _cells(line)
            intros.append((
                _tokens(cells[0])[0],
                _tokens(cells[1])[0],
                _tokens(cells[2])[0],
                _tokens(cells[3]),
                _hindi(cells[4]),
            ))
    clinical = []
    for line in lines[clin_at:end]:
        if not line.startswith("| "):
            continue
        cells = _cells(line)
        if not cells or not cells[0].isdigit():
            continue
        clinical.append({
            "id": _tokens(cells[1])[0],
            "group": _tokens(cells[2])[0],
            "nrroid": _tokens(cells[3])[0],
            "skip": _tokens(cells[4]),
            "hi": _hindi(cells[5]),
            "multi": _hindi(cells[6]),
            "follow": _tokens(cells[7]),
        })
    return followups, intros, clinical


class CatalogTests(unittest.TestCase):
    def test_catalog_matches_section_4_verbatim(self):
        followups, intros, clinical = _spec_tables()
        self.assertEqual(len(followups), 25)
        self.assertEqual(FOLLOWUP_HI, dict(followups))
        self.assertEqual([row[0] for row in intros], [
            "intro_name", "intro_married", "intro_children", "intro_bridge", "close_anything_else",
        ])
        for qid, group, kind, skip, hindi in intros:
            spec = get_spec(qid)
            self.assertEqual(spec.group, group, qid)
            self.assertEqual(spec.kind, kind, qid)
            self.assertEqual(spec.skip_if, skip, qid)
            self.assertEqual(spec.question_hi, hindi, qid)
            self.assertEqual(spec.opens, (), qid)
        self.assertEqual(get_spec("close_anything_else").followups, ("open_concern_detail",))
        self.assertEqual(len(clinical), 54)
        gates = [spec for spec in triage_catalog.CATALOG if spec.kind == "gate"]
        self.assertEqual([spec.id for spec in gates], [row["id"] for row in clinical])
        child_multi = 0
        for row in clinical:
            spec = get_spec(row["id"])
            self.assertEqual(spec.kind, "gate", row["id"])
            self.assertEqual(spec.group, row["group"], row["id"])
            self.assertEqual(spec.nrroid_gate, row["nrroid"], row["id"])
            self.assertEqual(spec.skip_if, row["skip"], row["id"])
            self.assertEqual(spec.question_hi, row["hi"], row["id"])
            self.assertEqual(spec.question_multi_hi, row["multi"], row["id"])
            self.assertEqual(spec.followups, row["follow"], row["id"])
            self.assertEqual(spec.opens, (), row["id"])
            if spec.group in ("birth", "body", "illness") and "no_children" in spec.skip_if:
                self.assertTrue(spec.question_multi_hi.strip(), row["id"])
                child_multi += 1
        self.assertEqual(child_multi, 36)
        self.assertEqual(get_spec("fam_same_problem").question_multi_hi, "")
        self.assertEqual(len(active_specs()), len(triage_catalog.CATALOG))
        json.dumps(new_cursor_state())

    def test_two_children_use_the_catalog_multi_string(self):
        kids = [_child(6, "male", 1), _child(2, "female", 2)]
        state = _profile(children=kids)
        seen = 0
        for spec in active_specs():
            if not spec.question_multi_hi:
                continue
            seen += 1
            self.assertEqual(call_cursor._question_hi(state, spec), spec.question_multi_hi, spec.id)
            self.assertNotIn("—", spec.question_multi_hi, spec.id)
        self.assertEqual(seen, 36)
        alone = _profile(children=[_child(4, "male")])
        nicu = get_spec("birth_nicu")
        self.assertEqual(call_cursor._question_hi(alone, nicu), nicu.question_hi)
        self.assertNotIn("किसी भी", nicu.question_hi)


class ProfileTests(unittest.TestCase):
    def test_fragment_and_missing_ages_ask_for_the_missing_piece(self):
        state, hold = apply_profile(new_cursor_state(), caller_name="मेरा नाम")
        self.assertEqual(hold, "name_fragment")
        self.assertEqual(state["hold_reason"], "name_fragment")
        self.assertIsNone(state["profile"]["caller_name"])
        self.assertIn("पूरा नाम", prompt_directive(state))
        self.assertFalse(expect_yes_no(state))

        state, hold = apply_profile(state, caller_name="पूजा")
        self.assertIsNone(hold)
        self.assertIsNone(state["hold_reason"])
        self.assertEqual(state["active_qid"], "intro_married")
        state, hold = apply_profile(state, marital_status="married")
        self.assertEqual(state["active_qid"], "intro_children")
        state, hold = apply_profile(state, children=[], child_count=2)
        self.assertEqual(hold, "children_ages")
        self.assertEqual(state["hold_reason"], "children_ages")
        self.assertEqual(state["profile"]["children"], [])
        self.assertIsNone(state["profile"]["ever_pregnant"])
        self.assertIn("उम्र", prompt_directive(state))
        self.assertFalse(expect_yes_no(state))

    def test_bridge_yes_records_clear_and_yes_no_is_off_for_intro_profile(self):
        state = _profile(children=[_child(4, "male")])
        self.assertEqual(state["active_qid"], "intro_bridge")
        self.assertTrue(expect_yes_no(state))
        self.assertFalse(expect_yes_no(new_cursor_state()))
        state = apply_turn(state, "concern")
        self.assertEqual(state["answers"]["intro_bridge"]["state"], "clear")
        self.assertEqual(state["active_qid"], "birth_mode_detail")
        self.assertTrue(expect_yes_no(state))


class SkipMatrixTests(unittest.TestCase):
    def test_marital_status_skip_matrix(self):
        unmarried = _profile(marital="unmarried")
        about = _profile(marital="about_to_marry")
        childless = _profile(children=[])
        for state in (unmarried, about):
            self.assertEqual(state["active_qid"], "intro_bridge")
            self.assertFalse(state["profile"]["ever_pregnant"])
            self.assertEqual(state["answers"]["intro_children"]["state"], "not_applicable")
            self.assertEqual(state["profile"]["children"], [])
        self.assertEqual(unmarried["answers"]["fam_related"]["state"], "not_applicable")
        self.assertNotIn("fam_related", about["answers"])
        for qid in _PREG_HISTORY:
            self.assertEqual(unmarried["answers"][qid]["state"], "not_applicable", qid)
            self.assertEqual(about["answers"][qid]["state"], "not_applicable", qid)
        self.assertNotEqual(unmarried["answers"].get("preg_thalassemia", {}).get("state"), "not_applicable")
        self.assertNotEqual(about["answers"].get("preg_thalassemia", {}).get("state"), "not_applicable")

        childless = apply_turn(childless, "clear")
        self.assertEqual(childless["active_qid"], "ever_pregnant")
        self.assertTrue(expect_yes_no(childless))
        childless = apply_turn(childless, "unknown")
        self.assertFalse(childless["profile"]["ever_pregnant"])
        self.assertEqual(childless["active_qid"], "preg_thalassemia")
        for qid in _PREG_HISTORY:
            self.assertEqual(childless["answers"][qid]["state"], "not_applicable", qid)

        pregnant = _profile(children=[])
        pregnant = apply_turn(pregnant, "clear")
        pregnant = apply_turn(pregnant, "concern")
        self.assertTrue(pregnant["profile"]["ever_pregnant"])
        self.assertEqual(pregnant["active_qid"], "preg_conditions")

    def test_dynamic_thalassemia_and_related_wording(self):
        married = _clear_until(_profile(children=[_child(4, "female")]), "preg_thalassemia")
        self.assertIn("आपके पति", prompt_directive(married))
        self.assertNotIn("होने वाले", prompt_directive(married))
        about = apply_turn(_profile(marital="about_to_marry"), "clear")
        self.assertEqual(about["active_qid"], "preg_thalassemia")
        self.assertIn("होने वाले पति", prompt_directive(about))
        about = apply_turn(about, "clear")
        self.assertEqual(about["active_qid"], "fam_related")
        self.assertIn("होने वाले पति", prompt_directive(about))
        unmarried = apply_turn(_profile(marital="unmarried"), "clear")
        self.assertEqual(unmarried["active_qid"], "preg_thalassemia")
        spoken = prompt_directive(unmarried)
        self.assertIn("थैलेसीमिया", spoken)
        self.assertNotIn("पति", spoken)
        unmarried = apply_turn(unmarried, "clear")
        self.assertEqual(unmarried["active_qid"], "fam_infant_death")

    def test_age_rules_skip_tooth_periods_and_puberty(self):
        baby = _profile(children=[_child(1, "male")])
        for qid in ("mod_tooth_bleed", "mod_heavy_periods", "body_puberty"):
            self.assertEqual(baby["answers"][qid]["state"], "not_applicable", qid)
        boy = _profile(children=[_child(6, "male")])
        self.assertNotIn("mod_tooth_bleed", boy["answers"])
        self.assertEqual(boy["answers"]["mod_heavy_periods"]["state"], "not_applicable")
        self.assertEqual(boy["answers"]["body_puberty"]["state"], "not_applicable")
        girl = _profile(children=[_child(12, "female")])
        for qid in ("mod_tooth_bleed", "mod_heavy_periods", "body_puberty"):
            self.assertNotIn(qid, girl["answers"])

    def test_call6_marital_correction_skips_the_question_she_is_on(self):
        state, hold = apply_profile(new_cursor_state(), caller_name="पूजा")
        state, hold = apply_profile(state, marital_status="married")
        self.assertEqual(state["active_qid"], "intro_children")
        state, hold = apply_profile(state, marital_status="about_to_marry")
        self.assertIsNone(hold)
        self.assertEqual(state["active_qid"], "intro_bridge")
        self.assertEqual(state["answers"]["intro_children"]["state"], "not_applicable")
        self.assertFalse(state["profile"]["ever_pregnant"])
        self.assertEqual(state["profile"]["children"], [])

        state = _clear_until(
            _profile(children=[_child(4, "male"), _child(2, "female", 2)]),
            "birth_nicu",
        )
        state, hold = apply_profile(state, marital_status="unmarried")
        self.assertIsNone(hold)
        self.assertEqual(state["profile"]["children"], [])
        self.assertFalse(state["profile"]["ever_pregnant"])
        self.assertEqual(state["answers"]["birth_nicu"]["state"], "not_applicable")
        self.assertEqual(state["active_qid"], "preg_thalassemia")
        self.assertNotIn("पति", prompt_directive(state))

        stayed = _clear_until(_profile(children=[_child(3, "female")]), "fam_infant_death")
        stayed, hold = apply_profile(stayed, marital_status="unmarried")
        self.assertEqual(stayed["active_qid"], "fam_infant_death")
        self.assertEqual(stayed["profile"]["children"], [])


class ChildResolutionTests(unittest.TestCase):
    def _three(self):
        return [
            _child(2, "male", 1),
            _child(8, "female", 2, name="रिया"),
            _child(5, "male", 3, name="अर्जुन"),
        ]

    def test_call8_middle_child_and_retraction_from_which_child(self):
        state = _clear_until(_profile(children=self._three()), "birth_jaundice")
        state = apply_turn(state, "concern")
        self.assertEqual(state["pending_followup_idx"], -1)
        self.assertEqual(state["step_attempts"], 0)
        prompt = prompt_directive(state)
        self.assertEqual(
            prompt,
            "Ask only this one Hindi question, then wait: "
            "यह किस बच्चे की बात है — 8 साल वाले रिया, 5 साल वाले अर्जुन (बीच वाले), या 2 साल वाले बेटा?",
        )
        state = apply_turn(state, "concern", child_ref="बीच वाले")
        self.assertEqual(state["answers"]["birth_jaundice"]["child_ids"], ["child_3"])
        self.assertEqual(state["answers"]["birth_jaundice"]["state"], "concern")
        self.assertEqual(state["pending_followup_idx"], 0)

        plain = [
            _child(2, "unknown", 1),
            _child(8, "unknown", 2),
            _child(5, "unknown", 3),
        ]
        asked = _clear_until(_profile(children=plain), "body_bleeding_bruise")
        asked = apply_turn(asked, "concern")
        self.assertIn(
            "8 साल वाले, 5 साल वाले (बीच वाले), या 2 साल वाले",
            prompt_directive(asked),
        )
        retracted = apply_turn(asked, "clear")
        self.assertEqual(retracted["answers"]["body_bleeding_bruise"]["state"], "clear")
        self.assertIsNone(retracted["pending_followup_idx"])
        self.assertEqual(retracted["step_attempts"], 0)
        self.assertNotEqual(retracted["active_qid"], "body_bleeding_bruise")

    def test_child_ref_resolves_exact_id_name_ordinal_sex_and_everyone(self):
        state = _profile(children=self._three())
        ordered = ["child_2", "child_3", "child_1"]
        for ref in ("all", "both", "दोनों", "सभी", "तीनों", "Both"):
            self.assertEqual(call_cursor._resolve_child_ids(state, ref), ordered, ref)
        self.assertEqual(call_cursor._resolve_child_ids(state, "child_1"), ["child_1"])
        self.assertEqual(call_cursor._resolve_child_ids(state, "रिया"), ["child_2"])
        self.assertEqual(call_cursor._resolve_child_ids(state, "बड़े"), ["child_2"])
        self.assertEqual(call_cursor._resolve_child_ids(state, "पहले"), ["child_2"])
        self.assertEqual(call_cursor._resolve_child_ids(state, "बीच वाले"), ["child_3"])
        self.assertEqual(call_cursor._resolve_child_ids(state, "मंझले"), ["child_3"])
        self.assertEqual(call_cursor._resolve_child_ids(state, "दूसरे"), ["child_3"])
        self.assertEqual(call_cursor._resolve_child_ids(state, "तीसरे"), ["child_1"])
        self.assertEqual(call_cursor._resolve_child_ids(state, "छोटे"), ["child_1"])
        self.assertEqual(call_cursor._resolve_child_ids(state, "आखिरी"), ["child_1"])
        self.assertEqual(call_cursor._resolve_child_ids(state, "बेटी"), ["child_2"])
        self.assertEqual(call_cursor._resolve_child_ids(state, "बेटा"), [])
        for phrase in ("5", "5 साल", "6 महीने", "बीच वाले का", "पाँच साल", "oldest child"):
            self.assertEqual(call_cursor._resolve_child_ids(state, phrase), [], phrase)
        months = _profile(children=[_child(4, "male", 1), _child(0.5, "female", 2)])
        self.assertEqual(call_cursor._resolve_child_ids(months, "6 महीने"), [])
        self.assertEqual(call_cursor._resolve_child_ids(months, "child_2"), ["child_2"])

    def test_cursor_has_no_regex_and_roster_is_oldest_first(self):
        source = Path(call_cursor.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            self.assertFalse(any(name == "re" or name.startswith("re.") for name in names))
        for name in (
            "_NUMBER_WORDS",
            "_AGE_RE",
            "_age_from_ref",
            "_digits_for_number_words",
            "_ref_tokens",
            "_hit",
            "_pool_for_sex",
            "_ordinal_id",
        ):
            self.assertFalse(hasattr(call_cursor, name), name)
        state = _profile(children=self._three())
        roster = call_cursor.children_roster(state)
        self.assertEqual([row["id"] for row in roster], ["child_2", "child_3", "child_1"])
        self.assertEqual([row["rank"] for row in roster], ["oldest", "middle", "youngest"])
        self.assertEqual(roster[0]["name"], "रिया")
        self.assertEqual(roster[0]["sex"], "female")
        self.assertEqual(roster[1]["age_label_hi"], "5 साल")
        self.assertEqual(call_cursor.children_roster(_profile(children=[_child(4, "male")])), [])

    def test_two_attempts_leave_a_stuck_which_child_and_followup(self):
        kids = [_child(6, "male", 1), _child(2, "female", 2)]
        state = _clear_until(_profile(children=kids), "birth_nicu")
        state = apply_turn(state, "concern")
        self.assertEqual(state["pending_followup_idx"], -1)
        state = apply_turn(state, "concern", note="पता नहीं कौन")
        self.assertEqual(state["pending_followup_idx"], -1)
        self.assertEqual(state["step_attempts"], 1)
        state = apply_turn(state, "clarify")
        self.assertEqual(state["active_qid"], "birth_nicu")
        self.assertEqual(state["pending_followup_idx"], 0)
        self.assertEqual(state["answers"]["birth_nicu"]["child_ids"], [])
        self.assertEqual(state["answers"]["birth_nicu"]["state"], "concern")
        self.assertIn("पता नहीं कौन", state["answers"]["birth_nicu"]["notes"])
        state = apply_turn(state, "clarify")
        self.assertEqual(state["pending_followup_idx"], 0)
        self.assertEqual(state["step_attempts"], 1)
        state = apply_turn(state, "concern", note="")
        self.assertEqual(
            state["answers"]["birth_nicu"]["followups"]["nicu_reason"],
            {"state": "unknown", "note": ""},
        )
        self.assertEqual(state["active_qid"], "birth_nicu")
        self.assertEqual(state["pending_followup_idx"], 1)
        state = apply_turn(state, "clarify")
        state = apply_turn(state, "concern", note="")
        self.assertEqual(
            state["answers"]["birth_nicu"]["followups"]["nicu_days"],
            {"state": "unknown", "note": ""},
        )
        self.assertNotEqual(state["active_qid"], "birth_nicu")
        self.assertIsNone(state["pending_followup_idx"])

        bleeding = _clear_until(_profile(children=kids), "body_bleeding_bruise")
        bleeding = apply_turn(bleeding, "concern")
        self.assertEqual(bleeding["pending_followup_idx"], -1)
        visits = {}
        for _ in range(12):
            if bleeding["phase"] != "asking":
                break
            key = (bleeding["active_qid"], bleeding.get("pending_followup_idx"))
            visits[key] = visits.get(key, 0) + 1
            self.assertLessEqual(visits[key], 2, key)
            bleeding = apply_turn(bleeding, "concern")
        self.assertNotEqual(bleeding["active_qid"], "body_bleeding_bruise")
        self.assertEqual(bleeding["answers"]["body_bleeding_bruise"]["state"], "concern")
        self.assertEqual(bleeding["answers"]["body_bleeding_bruise"]["child_ids"], [])

    def test_preanswered_followup_is_not_asked_again(self):
        state = _clear_until(_profile(children=[_child(4, "male")]), "birth_jaundice")
        state = apply_turn(state, "concern", note="दो दिन", followup_answered=True)
        self.assertIsNone(state["pending_followup_idx"])
        self.assertNotEqual(state["active_qid"], "birth_jaundice")
        self.assertEqual(
            state["answers"]["birth_jaundice"]["followups"]["jaundice_days"]["note"],
            "दो दिन",
        )

    def test_two_step_followups_keep_the_second_fact(self):
        steps = {
            "birth_nicu": ("nicu_reason", "nicu_days", "कितने दिन"),
            "ill_admissions": ("admissions_count", "admissions_reason", "किस बीमारी"),
            "fam_miscarriage": ("miscarriage_count", "miscarriage_month", "कितने महीने"),
            "fam_infant_death": ("infant_death_age", "infant_death_cause", "मौत की कोई वजह"),
        }
        state = _profile(children=[_child(4, "male")])
        for qid, (first, second, second_prompt) in steps.items():
            state = _clear_until(state, qid)
            state = apply_turn(state, "concern", note="पहली बात", followup_answered=True)
            self.assertEqual(state["active_qid"], qid, qid)
            self.assertEqual(state["pending_followup_idx"], 1, qid)
            self.assertEqual(state["answers"][qid]["followups"][first]["note"], "पहली बात")
            self.assertNotIn(second, state["answers"][qid]["followups"])
            self.assertIn(second_prompt, prompt_directive(state))
            state = apply_turn(state, "concern", note="दूसरी बात")
            self.assertIsNone(state["pending_followup_idx"], qid)
            self.assertEqual(state["answers"][qid]["followups"][second]["note"], "दूसरी बात")
            self.assertEqual(state["answers"][qid]["followups"][first]["note"], "पहली बात")
            self.assertNotEqual(state["active_qid"], qid)

        kids = [_child(6, "male", 1), _child(2, "female", 2)]
        multi = _clear_until(_profile(children=kids), "birth_nicu")
        multi = apply_turn(multi, "concern", note="सांस", followup_answered=True)
        self.assertEqual(multi["pending_followup_idx"], -1)
        self.assertEqual(multi["answers"]["birth_nicu"]["followups"]["nicu_reason"]["note"], "सांस")
        multi = apply_turn(multi, "concern", child_ref="child_1")
        self.assertEqual(multi["answers"]["birth_nicu"]["child_ids"], ["child_1"])
        self.assertEqual(multi["pending_followup_idx"], 1)
        self.assertIn("कितने दिन", prompt_directive(multi))

    def test_retract_on_a_followup_clears_the_concern(self):
        state = _clear_until(_profile(children=[_child(4, "male")]), "birth_nicu")
        state = apply_turn(state, "concern")
        self.assertEqual(state["pending_followup_idx"], 0)
        self.assertFalse(expect_yes_no(state))
        state = apply_turn(state, "retract")
        self.assertEqual(state["answers"]["birth_nicu"]["state"], "clear")
        self.assertEqual(state["active_qid"], "birth_jaundice")
        self.assertIsNone(state["pending_followup_idx"])


class CloseAndFinishTests(unittest.TestCase):
    def test_close_bare_yes_asks_for_the_note_then_a_noted_yes_keeps_it(self):
        state = _clear_until(_profile(children=[_child(4, "male")]), "close_anything_else")
        self.assertTrue(expect_yes_no(state))
        state = apply_turn(state, "concern")
        self.assertEqual(state["phase"], "asking")
        self.assertEqual(state["pending_followup_idx"], 0)
        self.assertFalse(expect_yes_no(state))
        self.assertIn("क्या बात नोट करूँ", prompt_directive(state))
        state = apply_turn(state, "concern", note="पेट में दर्द")
        self.assertIsNone(state["pending_followup_idx"])
        self.assertEqual(state["answers"]["close_anything_else"]["state"], "concern")
        self.assertEqual(state["answers"]["close_anything_else"]["notes"], "पेट में दर्द")
        self.assertIn("मैंने यह बात नोट कर ली। इसके अलावा कुछ और बताना चाहेंगी?", prompt_directive(state))
        self.assertTrue(expect_yes_no(state))
        state = apply_turn(state, "concern", note="नींद कम है")
        self.assertEqual(
            state["answers"]["close_anything_else"]["notes"],
            "पेट में दर्द; नींद कम है",
        )
        state = apply_turn(state, "clear")
        self.assertEqual(state["phase"], "closing")
        self.assertEqual(state["close_reason"], "complete")
        self.assertEqual(state["answers"]["close_anything_else"]["state"], "concern")
        self.assertIn("नींद कम है", state["answers"]["close_anything_else"]["notes"])
        self.assertFalse(expect_yes_no(state))

        noted = _clear_until(_profile(children=[_child(4, "male")]), "close_anything_else")
        noted = apply_turn(noted, "concern", note="पेट दर्द")
        self.assertIsNone(noted["pending_followup_idx"])
        self.assertEqual(noted["answers"]["close_anything_else"]["notes"], "पेट दर्द")
        noted = apply_turn(noted, "unknown")
        self.assertEqual(noted["phase"], "closing")
        self.assertEqual(noted["answers"]["close_anything_else"]["state"], "concern")

    def test_one_child_all_clear_finishes_and_emergency_stops(self):
        state = _profile(children=[_child(4, "male")])
        guard = 0
        while state["phase"] == "asking" and guard < 90:
            state = apply_turn(state, "clear")
            guard += 1
        self.assertEqual(state["phase"], "closing")
        self.assertEqual(state["close_reason"], "complete")
        self.assertLess(guard, 90)
        self.assertEqual(state["answers"]["birth_nicu"]["state"], "clear")
        self.assertEqual(state["answers"]["mod_tooth_bleed"]["state"], "not_applicable")
        self.assertEqual(state["answers"]["fam_same_problem"]["state"], "not_applicable")

        stopped = _clear_until(_profile(children=[_child(3, "female")]), "ill_fasting")
        stopped = apply_turn(stopped, "emergency", note="अभी सांस नहीं ले रहा")
        self.assertEqual(stopped["phase"], "closing")
        self.assertEqual(stopped["close_reason"], "emergency")
        self.assertEqual(stopped["active_qid"], "ill_fasting")
        self.assertEqual(apply_turn(stopped, "clear")["phase"], "closing")

    def test_family_concern_opens_same_problem_without_a_hidden_gate(self):
        state = _profile(children=[_child(4, "male")])
        self.assertEqual(state["answers"]["fam_same_problem"]["state"], "not_applicable")
        state = _clear_until(state, "fam_fits_delay")
        state = apply_turn(state, "concern", note="मामा को दौरे")
        self.assertEqual(state["pending_followup_idx"], 0)
        state = apply_turn(state, "concern", note="मामा")
        state = _clear_until(state, "fam_same_problem")
        self.assertEqual(state["active_qid"], "fam_same_problem")
        self.assertNotIn("fam_same_problem", [
            qid for qid, answer in state["answers"].items()
            if answer.get("state") == "not_applicable" and qid == "fam_same_problem"
        ])


if __name__ == "__main__":
    unittest.main()
