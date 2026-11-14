"""Offline trajectories for the v4 atomic catalog and FormBharo cursor."""

import ast
import json
import re
import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from tests import REPO_ROOT  # noqa: E402

import call_cursor  # noqa: E402
import triage_catalog  # noqa: E402
from call_cursor import (  # noqa: E402
    apply_interpreted_turn,
    apply_profile,
    apply_turn,
    close_technical_failure,
    expect_yes_no,
    new_cursor_state,
    prompt_directive,
    routable_qids,
)
from triage_catalog import FOLLOWUP_HI, active_specs, get_spec  # noqa: E402

_DOC = REPO_ROOT / "docs" / "requirements-v4.md"
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
    clin_at = lines.index('### 4.3 Atomic Clinical Questions (56 Items — All `kind="gate"`, `opens=()`)')
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
        self.assertEqual(len(clinical), 56)
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

    def test_question_hi_for_partial_profiles_and_catalog_wording(self):
        thal = get_spec("preg_thalassemia")
        nicu = get_spec("birth_nicu")
        self.assertEqual(call_cursor.question_hi_for(None, thal), thal.question_hi)
        self.assertEqual(call_cursor.question_hi_for({}, thal), thal.question_hi)
        self.assertEqual(call_cursor.question_hi_for({"profile": {}}, thal), thal.question_hi)
        unmarried = call_cursor.question_hi_for({"profile": {"marital_status": "unmarried"}}, thal)
        self.assertEqual(unmarried, call_cursor._THAL_UNMARRIED)
        self.assertNotIn("पति", unmarried)
        self.assertIn(
            "होने वाले पति",
            call_cursor.question_hi_for({"profile": {"marital_status": "about_to_marry"}}, thal),
        )
        self.assertEqual(
            call_cursor.question_hi_for({"profile": {"children": "nope"}}, nicu),
            nicu.question_hi,
        )
        kids = {
            "profile": {
                "marital_status": "married",
                "children": [_child(6, "male", 1), _child(2, "female", 2)],
            },
        }
        self.assertEqual(call_cursor.question_hi_for(kids, nicu), nicu.question_multi_hi)
        self.assertEqual(call_cursor._question_hi(kids, nicu), nicu.question_multi_hi)

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
        state = apply_turn(state, "concern")
        self.assertEqual(state["active_qid"], "birth_nicu")
        self.assertEqual(state["pending_followup_idx"], 0)
        self.assertEqual(state["answers"]["birth_nicu"]["child_ids"], [])
        self.assertEqual(state["answers"]["birth_nicu"]["state"], "concern")
        self.assertIn("पता नहीं कौन", state["answers"]["birth_nicu"]["notes"])
        state = apply_turn(state, "concern", note="")
        self.assertEqual(state["pending_followup_idx"], 0)
        self.assertEqual(state["step_attempts"], 1)
        state = apply_turn(state, "concern", note="")
        self.assertEqual(
            state["answers"]["birth_nicu"]["followups"]["nicu_reason"],
            {"state": "unknown", "note": ""},
        )
        self.assertEqual(state["active_qid"], "birth_nicu")
        self.assertEqual(state["pending_followup_idx"], 1)
        state = apply_turn(state, "concern", note="")
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
        answers_before = json.loads(json.dumps(stopped["answers"]))
        stopped = apply_turn(stopped, "emergency", note="  अभी   सांस नहीं ले रहा  ")
        self.assertEqual(stopped["phase"], "closing")
        self.assertEqual(stopped["close_reason"], "emergency")
        self.assertEqual(stopped["emergency_note"], "अभी सांस नहीं ले रहा")
        self.assertEqual(stopped["active_qid"], "ill_fasting")
        self.assertEqual(stopped["answers"], answers_before)
        self.assertEqual(apply_turn(stopped, "clear")["phase"], "closing")
        fresh = _clear_until(_profile(children=[_child(3, "female")]), "ill_fasting")
        quiet = apply_turn(fresh, "caller_stop", note="अब रहन दो")
        self.assertEqual(quiet["close_reason"], "caller_stop")
        self.assertNotIn("emergency_note", quiet)

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


class CallNineReplayTests(unittest.TestCase):
    def _boy(self):
        return _profile(children=[_child(4, "male")])

    def test_late_family_addition_stays_on_the_current_question(self):
        state = _clear_until(self._boy(), "fam_cancer")
        state = apply_turn(state, "concern", note="मामा को ब्लड कैंसर")
        state = apply_turn(state, "concern", note="मामा")
        state = _clear_until(state, "fam_heart_kidney")
        self.assertEqual(state["active_qid"], "fam_heart_kidney")
        updated = apply_turn(
            state,
            "concern",
            note="चाची को भी ब्लड कैंसर",
            target_qid="fam_cancer",
        )
        self.assertEqual(updated["active_qid"], "fam_heart_kidney")
        self.assertEqual(updated["pending_followup_idx"], state["pending_followup_idx"])
        self.assertEqual(updated["step_attempts"], state["step_attempts"])
        heart = updated["answers"].get("fam_heart_kidney")
        self.assertTrue(heart is None or heart.get("state") != "concern")
        cancer = updated["answers"]["fam_cancer"]
        self.assertEqual(cancer["state"], "concern")
        self.assertIn("मामा को ब्लड कैंसर", cancer["notes"])
        self.assertIn("चाची को भी ब्लड कैंसर", cancer["notes"])
        self.assertEqual(cancer["followups"]["fam_who"]["state"], "concern")
        self.assertIn("मामा", cancer["followups"]["fam_who"]["note"])
        self.assertIn("चाची को भी ब्लड कैंसर", cancer["followups"]["fam_who"]["note"])
        self.assertIn("दिल या गुर्दे", prompt_directive(updated))

    def test_clarify_on_an_earlier_question_returns_there_then_resumes(self):
        state = _clear_until(self._boy(), "preg_thalassemia")
        state = apply_turn(state, "unknown")
        self.assertEqual(state["active_qid"], "fam_related")
        self.assertEqual(state["answers"]["preg_thalassemia"]["state"], "unknown")
        self.assertEqual(state["answers"]["birth_mode_detail"]["state"], "clear")
        rewound = apply_turn(state, "clarify", target_qid="preg_thalassemia")
        self.assertEqual(rewound["active_qid"], "preg_thalassemia")
        self.assertIsNone(rewound["pending_followup_idx"])
        self.assertEqual(rewound["step_attempts"], 0)
        self.assertIsNone(rewound["hold_reason"])
        self.assertNotIn("preg_thalassemia", rewound["answers"])
        self.assertEqual(rewound["answers"]["birth_mode_detail"]["state"], "clear")
        self.assertIn("थैलेसीमिया", prompt_directive(rewound))
        answered = apply_turn(rewound, "clear")
        self.assertEqual(answered["answers"]["preg_thalassemia"]["state"], "clear")
        self.assertEqual(answered["active_qid"], "fam_related")
        self.assertEqual(answered["answers"]["birth_mode_detail"]["state"], "clear")
        self.assertNotEqual(
            (answered["answers"].get("fam_infant_death") or {}).get("state"),
            "clear",
        )
        self.assertNotEqual(answered["active_qid"], "birth_mode_detail")

    def test_maternal_gates_are_atomic_and_clarify_does_not_burn_a_followup(self):
        bleeding = get_spec("fam_maternal_bleeding")
        infections = get_spec("fam_maternal_infections")
        eczema = get_spec("fam_maternal_eczema")
        self.assertEqual(bleeding.group, "family")
        self.assertEqual(bleeding.nrroid_gate, "Q11")
        self.assertEqual(bleeding.kind, "gate")
        self.assertEqual(bleeding.skip_if, ())
        self.assertEqual(bleeding.followups, ("fam_who",))
        self.assertEqual(infections.followups, ("fam_who",))
        self.assertEqual(eczema.followups, ("fam_who",))
        self.assertEqual(bleeding.need_en, "on her parents' side: heavy bleeding")
        self.assertEqual(infections.need_en, "on her parents' side: repeated serious infections")
        self.assertEqual(eczema.need_en, "on her parents' side: dry itchy skin patches")
        self.assertEqual(
            bleeding.question_hi,
            "क्या आपके मायके की तरफ किसी बच्चे या बड़े को चोट पर ज़्यादा खून बहने की दिक्कत है?",
        )
        self.assertEqual(
            infections.question_hi,
            "क्या आपके मायके की तरफ किसी बच्चे या बड़े को बार-बार गंभीर इंफेक्शन की दिक्कत है?",
        )
        self.assertEqual(
            eczema.question_hi,
            "क्या आपके मायके की तरफ किसी बच्चे या बड़े को त्वचा पर सूखे खुजली वाले धब्बों की दिक्कत है?",
        )
        self.assertNotIn("इंफेक्शन", bleeding.question_hi)
        self.assertNotIn("खून", infections.question_hi)
        self.assertNotIn("इंफेक्शन", eczema.question_hi)
        with self.assertRaises(KeyError):
            get_spec("fam_maternal_side")

        state = _clear_until(self._boy(), "fam_maternal_bleeding")
        state = apply_turn(state, "concern")
        self.assertEqual(state["pending_followup_idx"], 0)
        self.assertEqual(state["step_attempts"], 0)
        state = apply_turn(state, "clarify")
        self.assertEqual(state["active_qid"], "fam_maternal_bleeding")
        self.assertEqual(state["pending_followup_idx"], 0)
        self.assertEqual(state["step_attempts"], 0)
        state = apply_turn(state, "clarify")
        self.assertEqual(state["pending_followup_idx"], 0)
        self.assertEqual(state["step_attempts"], 0)
        self.assertNotIn("fam_who", state["answers"]["fam_maternal_bleeding"].get("followups") or {})

        kids = [_child(6, "male", 1), _child(2, "female", 2)]
        which = _clear_until(_profile(children=kids), "birth_nicu")
        which = apply_turn(which, "concern")
        which = apply_turn(which, "clarify")
        self.assertEqual(which["pending_followup_idx"], -1)
        self.assertEqual(which["step_attempts"], 0)

        close = _clear_until(self._boy(), "close_anything_else")
        close = apply_turn(close, "concern")
        close = apply_turn(close, "clarify")
        self.assertEqual(close["phase"], "asking")
        self.assertEqual(close["pending_followup_idx"], 0)
        self.assertEqual(close["step_attempts"], 0)

    def test_join_notes_is_idempotent_across_clauses(self):
        self.assertEqual(call_cursor._join_notes("सांस; तीन दिन", "तीन दिन"), "सांस; तीन दिन")
        self.assertEqual(call_cursor._join_notes("सांस;  तीन   दिन ", "  सांस "), "सांस; तीन दिन")
        self.assertEqual(call_cursor._join_notes("NICU", "nicu; बुखार"), "NICU; बुखार")
        self.assertEqual(call_cursor._join_notes("", "  चार   दिन "), "चार दिन")
        self.assertEqual(call_cursor._join_notes("सांस", "   "), "सांस")

    def test_revise_writes_a_late_note_into_the_open_followup(self):
        state = _clear_until(self._boy(), "birth_nicu")
        state = apply_turn(state, "concern", note="सांस", followup_answered=True)
        state = apply_turn(state, "concern", note="तीन दिन")
        self.assertNotEqual(state["active_qid"], "birth_nicu")
        later = state["active_qid"]
        updated = apply_turn(state, "concern", note="और बुखार", target_qid="birth_nicu")
        self.assertEqual(updated["active_qid"], later)
        followups = updated["answers"]["birth_nicu"]["followups"]
        self.assertEqual(followups["nicu_reason"]["note"], "सांस")
        self.assertEqual(followups["nicu_days"]["note"], "तीन दिन; और बुखार")
        again = apply_turn(updated, "concern", note="और बुखार", target_qid="birth_nicu")
        self.assertEqual(again["answers"]["birth_nicu"]["followups"]["nicu_days"]["note"], "तीन दिन; और बुखार")
        self.assertEqual(again["answers"]["birth_nicu"]["followups"]["nicu_reason"]["note"], "सांस")

        open_slot = json.loads(json.dumps(state))
        del open_slot["answers"]["birth_nicu"]["followups"]["nicu_days"]
        routed = apply_turn(open_slot, "concern", note="चार दिन", target_qid="birth_nicu")
        routed_followups = routed["answers"]["birth_nicu"]["followups"]
        self.assertEqual(routed_followups["nicu_reason"]["note"], "सांस")
        self.assertEqual(routed_followups["nicu_days"]["note"], "चार दिन")
        self.assertEqual(routed["active_qid"], open_slot["active_qid"])

    def test_clarify_mid_followup_returns_to_the_unfinished_question(self):
        state = _clear_until(self._boy(), "birth_nicu")
        state = apply_turn(state, "concern")
        self.assertEqual(state["pending_followup_idx"], 0)
        self.assertEqual(state["answers"]["birth_nicu"]["state"], "concern")
        rewound = apply_turn(state, "clarify", target_qid="birth_mode_detail")
        self.assertEqual(rewound["active_qid"], "birth_mode_detail")
        self.assertNotIn("birth_nicu", rewound["answers"])
        self.assertNotIn("birth_mode_detail", rewound["answers"])
        self.assertEqual(rewound["answers"]["birth_term"]["state"], "clear")
        answered = apply_turn(rewound, "clear")
        self.assertEqual(answered["answers"]["birth_mode_detail"]["state"], "clear")
        self.assertEqual(answered["active_qid"], "birth_nicu")
        self.assertIsNone(answered["pending_followup_idx"])
        self.assertNotIn("birth_nicu", answered["answers"])

        kids = [_child(6, "male", 1), _child(2, "female", 2)]
        which = _clear_until(_profile(children=kids), "birth_nicu")
        which = apply_turn(which, "concern")
        self.assertEqual(which["pending_followup_idx"], -1)
        self.assertEqual(which["answers"]["birth_nicu"]["state"], "concern")
        which = apply_turn(which, "clarify", target_qid="birth_term")
        self.assertNotIn("birth_nicu", which["answers"])
        self.assertEqual(which["answers"]["birth_mode_detail"]["state"], "clear")
        which = apply_turn(which, "clear")
        self.assertEqual(which["active_qid"], "birth_nicu")
        self.assertIsNone(which["pending_followup_idx"])

        close = _clear_until(self._boy(), "close_anything_else")
        close = apply_turn(close, "concern", note="पेट दर्द")
        close = apply_turn(close, "concern")
        self.assertEqual(close["pending_followup_idx"], 0)
        close = apply_turn(close, "clarify", target_qid="birth_mode_detail")
        self.assertEqual(close["answers"]["close_anything_else"]["state"], "concern")
        self.assertEqual(close["answers"]["close_anything_else"]["notes"], "पेट दर्द")


class DecisionApiPhase1Tests(unittest.TestCase):
    def test_routable_qids_has_exact_58_ids_in_ask_order(self):
        qids = routable_qids()
        self.assertIsInstance(qids, tuple)
        self.assertEqual(len(qids), 58)
        self.assertEqual(len(set(qids)), 58)
        for excluded in ("intro_name", "intro_married", "intro_children", "intro_bridge"):
            self.assertNotIn(excluded, qids)
        self.assertIn("ever_pregnant", qids)
        self.assertEqual(qids[-1], "close_anything_else")
        self.assertEqual(qids[qids.index("ill_iv_antibiotics") + 1], "ever_pregnant")
        self.assertEqual(qids[qids.index("ever_pregnant") + 1], "preg_conditions")
        gates = tuple(spec.id for spec in active_specs() if spec.kind == "gate")
        self.assertEqual(tuple(q for q in qids if q not in ("ever_pregnant", "close_anything_else")), gates)

    def test_smoke_a_dual_intent_and_side_write_alone(self):
        state = _clear_until(_profile(children=[_child(4, "male")]), "fam_cancer")
        state = apply_turn(state, "concern", note="मामा को ब्लड कैंसर")
        state = apply_turn(state, "concern", note="मामा")
        state = _clear_until(state, "fam_heart_kidney")
        self.assertEqual(state["active_qid"], "fam_heart_kidney")

        # Dual-intent: clear on active fam_heart_kidney + concern side write on fam_cancer.
        dual = apply_interpreted_turn(state, {
            "current_outcome": "clear",
            "side_qid": "fam_cancer",
            "side_action": "add_or_correct",
            "side_outcome": "concern",
            "side_note": "चाची को भी कैंसर हुआ था",
            "note": "चाची को भी कैंसर हुआ था",
        })
        self.assertEqual(dual["answers"]["fam_heart_kidney"]["state"], "clear")
        self.assertEqual(dual["answers"]["fam_heart_kidney"]["notes"], "")
        self.assertIn("चाची को भी कैंसर हुआ था", dual["answers"]["fam_cancer"]["notes"])
        self.assertEqual(dual["active_qid"], "fam_bp_stroke")

        # Even when side_note is omitted and only note carries the side text, active clear stays clean.
        dual_fallback = apply_interpreted_turn(state, {
            "current_outcome": "clear",
            "side_qid": "fam_cancer",
            "side_action": "add_or_correct",
            "side_outcome": "concern",
            "note": "चाची को भी कैंसर हुआ था",
        })
        self.assertEqual(dual_fallback["answers"]["fam_heart_kidney"]["state"], "clear")
        self.assertEqual(dual_fallback["answers"]["fam_heart_kidney"]["notes"], "")
        self.assertIn("चाची को भी कैंसर हुआ था", dual_fallback["answers"]["fam_cancer"]["notes"])
        self.assertEqual(dual_fallback["active_qid"], "fam_bp_stroke")

        # Side write alone (current_outcome="not_about_active"): stays on fam_heart_kidney with step_attempts == 0.
        side_only = apply_interpreted_turn(state, {
            "current_outcome": "not_about_active",
            "side_qid": "fam_cancer",
            "side_action": "add_or_correct",
            "side_outcome": "concern",
            "note": "चाची को भी कैंसर हुआ था",
        })
        self.assertEqual(side_only["active_qid"], "fam_heart_kidney")
        self.assertEqual(side_only["step_attempts"], 0)
        self.assertNotIn("fam_heart_kidney", side_only["answers"])
        self.assertIn("चाची को भी कैंसर हुआ था", side_only["answers"]["fam_cancer"]["notes"])

        # Dual concern + side concern without current_note preserves note on BOTH concerns;
        # explicit current_note="" suppresses only the active note.
        dual_concerns = apply_interpreted_turn(state, {
            "current_outcome": "concern",
            "side_qid": "fam_cancer",
            "side_action": "add_or_correct",
            "side_outcome": "concern",
            "side_note": "चाची को भी कैंसर और दिल की बीमारी",
            "note": "चाची को भी कैंसर और दिल की बीमारी",
        })
        self.assertEqual(dual_concerns["answers"]["fam_heart_kidney"]["state"], "concern")
        self.assertEqual(
            dual_concerns["answers"]["fam_heart_kidney"]["notes"],
            "चाची को भी कैंसर और दिल की बीमारी",
        )
        self.assertIn(
            "चाची को भी कैंसर और दिल की बीमारी",
            dual_concerns["answers"]["fam_cancer"]["notes"],
        )

        explicit_empty_current = apply_interpreted_turn(state, {
            "current_outcome": "concern",
            "current_note": "",
            "side_qid": "fam_cancer",
            "side_action": "add_or_correct",
            "side_outcome": "concern",
            "side_note": "चाची को भी कैंसर",
            "note": "चाची को भी कैंसर",
        })
        self.assertEqual(explicit_empty_current["answers"]["fam_heart_kidney"]["state"], "concern")
        self.assertEqual(explicit_empty_current["answers"]["fam_heart_kidney"]["notes"], "")
        self.assertIn("चाची को भी कैंसर", explicit_empty_current["answers"]["fam_cancer"]["notes"])

    def test_invalid_future_bridge_close_and_not_applicable_side_qids_ignored(self):
        state = _clear_until(_profile(children=[_child(3, "male")]), "birth_jaundice")
        # mod_tooth_bleed is not_applicable (child is 3 < 5); fam_cancer is future/unanswered.
        self.assertEqual(state["answers"]["mod_tooth_bleed"]["state"], "not_applicable")
        for bad_qid in ("fam_cancer", "does_not_exist", "intro_bridge", "close_anything_else", "mod_tooth_bleed"):
            res = apply_interpreted_turn(state, {
                "current_outcome": "not_about_active",
                "side_qid": bad_qid,
                "side_action": "add_or_correct",
                "side_outcome": "concern",
                "note": "कुछ बात",
            })
            self.assertEqual(res["active_qid"], "birth_jaundice", bad_qid)
            self.assertEqual(res["answers"], state["answers"], bad_qid)

    def test_clarify_meaning_on_earlier_gate_preserves_clear_on_active_gate(self):
        state = _clear_until(_profile(children=[_child(4, "male")]), "preg_thalassemia")
        state = apply_turn(state, "unknown")
        self.assertEqual(state["active_qid"], "fam_related")

        rewound = apply_interpreted_turn(state, {
            "current_outcome": "clear",
            "side_qid": "preg_thalassemia",
            "side_action": "clarify_meaning",
            "note": "रिश्तेदार तो नहीं हैं, पर थैलेसीमिया क्या होता है?",
        })
        self.assertEqual(rewound["active_qid"], "preg_thalassemia")
        self.assertIsNone(rewound["pending_followup_idx"])
        self.assertNotIn("preg_thalassemia", rewound["answers"])
        self.assertEqual(rewound["answers"]["fam_related"]["state"], "clear")
        self.assertEqual(rewound["answers"]["fam_related"]["notes"], "")

        answered = apply_interpreted_turn(rewound, {"current_outcome": "clear"})
        self.assertEqual(answered["answers"]["preg_thalassemia"]["state"], "clear")
        self.assertEqual(answered["answers"]["fam_related"]["state"], "clear")
        self.assertEqual(answered["active_qid"], "fam_miscarriage")

        # Clarify-meaning with not_about_active rewinds cleanly without marking current gate.
        plain_rewind = apply_interpreted_turn(state, {
            "current_outcome": "not_about_active",
            "side_qid": "preg_thalassemia",
            "side_action": "clarify_meaning",
        })
        self.assertEqual(plain_rewind["active_qid"], "preg_thalassemia")
        self.assertNotIn("fam_related", plain_rewind["answers"])

    def test_emergency_and_caller_stop_override_clear_and_side_qid(self):
        state = _clear_until(_profile(children=[_child(4, "male")]), "fam_cancer")
        state = apply_turn(state, "concern", note="मामा को कैंसर")
        state = apply_turn(state, "concern", note="मामा")
        state = _clear_until(state, "fam_heart_kidney")
        before = json.loads(json.dumps(state["answers"]))

        emerg = apply_interpreted_turn(state, {
            "emergency": True,
            "current_outcome": "clear",
            "side_qid": "fam_cancer",
            "side_action": "add_or_correct",
            "side_outcome": "clear",
            "note": "  बच्चे को अभी सांस नहीं आ रही  ",
        })
        self.assertEqual(emerg["phase"], "closing")
        self.assertEqual(emerg["close_reason"], "emergency")
        self.assertEqual(emerg["emergency_note"], "बच्चे को अभी सांस नहीं आ रही")
        self.assertEqual(emerg["active_qid"], "fam_heart_kidney")
        self.assertEqual(emerg["answers"], before)

        # Already closing returns deep copy untouched.
        still_closed = apply_interpreted_turn(emerg, {"current_outcome": "clear"})
        self.assertEqual(still_closed, emerg)

        stop = apply_interpreted_turn(state, {
            "caller_stop": True,
            "current_outcome": "clear",
            "side_qid": "fam_cancer",
            "side_action": "add_or_correct",
            "side_outcome": "clear",
        })
        self.assertEqual(stop["phase"], "closing")
        self.assertEqual(stop["close_reason"], "caller_stop")
        self.assertEqual(stop["answers"], before)

    def test_midcall_marital_unmarried_skips_child_gates_drops_child_side_and_no_double_apply(self):
        state = _clear_until(
            _profile(children=[_child(4, "male"), _child(2, "female", 2)]),
            "birth_nicu",
        )
        updated = apply_interpreted_turn(state, {
            "profile_updates": {"marital_status": "unmarried"},
            "current_outcome": "clear",
            "side_qid": "birth_mode_detail",
            "side_action": "add_or_correct",
            "side_outcome": "concern",
            "side_note": "ऑपरेशन हुआ था",
        })
        self.assertEqual(updated["profile"]["marital_status"], "unmarried")
        self.assertEqual(updated["profile"]["children"], [])
        self.assertEqual(updated["answers"]["birth_mode_detail"]["state"], "not_applicable")
        self.assertEqual(updated["answers"]["birth_nicu"]["state"], "not_applicable")
        self.assertEqual(updated["active_qid"], "preg_thalassemia")
        self.assertNotIn("preg_thalassemia", updated["answers"])

    def test_side_child_ref_disambiguation_and_pending_followup_retries(self):
        kids = [_child(6, "male", 1), _child(2, "female", 2)]
        state = _clear_until(_profile(children=kids), "birth_jaundice")
        state["answers"]["birth_nicu"] = {
            "state": "concern",
            "child_ids": ["child_1"],
            "notes": "सांस",
            "followups": {},
        }
        # Both active and side are child-scoped concerns without explicit side_child_ref -> do not guess child_2 onto side!
        both = apply_interpreted_turn(state, {
            "current_outcome": "concern",
            "current_note": "दो दिन पीलिया",
            "followup_answered": True,
            "child_ref": "child_2",
            "side_qid": "birth_nicu",
            "side_action": "add_or_correct",
            "side_outcome": "concern",
            "side_note": "काँच के बक्से में भी",
        })
        self.assertEqual(both["answers"]["birth_jaundice"]["child_ids"], ["child_2"])
        self.assertEqual(both["answers"]["birth_nicu"]["child_ids"], ["child_1"])

        # Explicit side_child_ref applies to the side concern.
        explicit = apply_interpreted_turn(state, {
            "current_outcome": "concern",
            "current_note": "दो दिन पीलिया",
            "followup_answered": True,
            "child_ref": "child_2",
            "side_qid": "birth_nicu",
            "side_action": "add_or_correct",
            "side_outcome": "concern",
            "side_note": "काँच के बक्से में भी",
            "side_child_ref": "child_2",
        })
        self.assertEqual(explicit["answers"]["birth_nicu"]["child_ids"], ["child_1", "child_2"])

        # Non-child-scoped active turn (not_about_active) defaults side_child_ref to child_ref.
        side_only = apply_interpreted_turn(state, {
            "current_outcome": "not_about_active",
            "child_ref": "child_2",
            "side_qid": "birth_nicu",
            "side_action": "add_or_correct",
            "side_outcome": "concern",
            "side_note": "छोटी को भी NICU",
        })
        self.assertEqual(side_only["answers"]["birth_nicu"]["child_ids"], ["child_1", "child_2"])
        self.assertEqual(side_only["active_qid"], "birth_jaundice")

        # Pending followup / which_child retry mechanics: clarify & side writes preserve step_attempts;
        # unresolved non-answers increment step_attempts up to MAX_STEP_ATTEMPTS.
        which = apply_interpreted_turn(state, {"current_outcome": "concern"})
        self.assertEqual(which["pending_followup_idx"], -1)
        which_side = apply_interpreted_turn(which, {
            "current_outcome": "not_about_active",
            "side_qid": "birth_mode_detail",
            "side_action": "add_or_correct",
            "side_outcome": "concern",
            "side_note": "ऑपरेशन",
        })
        self.assertEqual(which_side["pending_followup_idx"], -1)
        self.assertEqual(which_side["step_attempts"], 0)

        which_1 = apply_interpreted_turn(which_side, {"current_outcome": "not_about_active"})
        self.assertEqual(which_1["pending_followup_idx"], -1)
        self.assertEqual(which_1["step_attempts"], 1)
        which_2 = apply_interpreted_turn(which_1, {"current_outcome": "not_about_active"})
        self.assertEqual(which_2["pending_followup_idx"], 0)
        self.assertEqual(which_2["step_attempts"], 0)
        self.assertEqual(which_2["answers"]["birth_jaundice"]["child_ids"], [])

        clar = apply_interpreted_turn(which_2, {"current_outcome": "clarify"})
        self.assertEqual(clar["pending_followup_idx"], 0)
        self.assertEqual(clar["step_attempts"], 0)

        fup_1 = apply_interpreted_turn(clar, {"current_outcome": "not_about_active"})
        self.assertEqual(fup_1["pending_followup_idx"], 0)
        self.assertEqual(fup_1["step_attempts"], 1)
        fup_2 = apply_interpreted_turn(fup_1, {"current_outcome": "not_about_active"})
        self.assertEqual(
            fup_2["answers"]["birth_jaundice"]["followups"]["jaundice_days"],
            {"state": "unknown", "note": ""},
        )
        self.assertNotEqual(fup_2["active_qid"], "birth_jaundice")

    def test_close_technical_failure_preserves_answers_and_attempts(self):
        state = _clear_until(_profile(children=[_child(4, "male")]), "birth_nicu")
        state = apply_turn(state, "concern")
        state = apply_turn(state, "concern", note="")
        self.assertEqual(state["pending_followup_idx"], 0)
        self.assertEqual(state["step_attempts"], 1)
        snapshot = json.loads(json.dumps(state))

        closed = close_technical_failure(state)
        self.assertEqual(state, snapshot)
        self.assertEqual(closed["phase"], "closing")
        self.assertEqual(closed["close_reason"], "technical_failure")
        self.assertIsNone(closed["pending_followup_idx"])
        self.assertIsNone(closed["hold_reason"])
        self.assertEqual(closed["answers"], snapshot["answers"])
        self.assertEqual(closed["step_attempts"], 1)
        self.assertNotIn("emergency_note", closed)

        # Already closing state is untouched.
        already_complete = dict(closed, close_reason="complete")
        self.assertEqual(close_technical_failure(already_complete), already_complete)

    def test_canonical_state_inspection_api(self):
        intro = new_cursor_state()
        self.assertEqual(call_cursor.step_kind(intro), "profile")
        self.assertIn("नाम", call_cursor.active_prompt_hi(intro))
        self.assertEqual(call_cursor.followup_prompt_hi(intro), "")
        self.assertEqual(call_cursor.last_answered_prompt_hi(intro), ("", ""))
        self.assertEqual(call_cursor.step_id(intro), "intro_name")
        self.assertFalse(call_cursor.has_answered_gates(intro))
        self.assertFalse(call_cursor.is_answered_or_na(intro, "birth_jaundice"))

        held, _ = apply_profile(intro, caller_name="मेरा नाम")
        self.assertEqual(call_cursor.step_kind(held), "profile")
        self.assertEqual(call_cursor.step_id(held), "intro_name:hold:name_fragment")
        self.assertIn("पूरा नाम", call_cursor.active_prompt_hi(held))

        bridge = _profile(children=[_child(6, "male", 1), _child(2, "female", 2)])
        self.assertEqual(call_cursor.step_kind(bridge), "bridge")
        self.assertFalse(call_cursor.has_answered_gates(bridge))

        first_gate = apply_turn(bridge, "clear")
        self.assertEqual(call_cursor.step_kind(first_gate), "main")
        self.assertTrue(call_cursor.has_answered_gates(first_gate))
        self.assertEqual(call_cursor.last_answered_prompt_hi(first_gate), ("", ""))
        self.assertEqual(call_cursor.followup_prompt_hi(first_gate), FOLLOWUP_HI["operation_reason"])

        childless = apply_turn(_profile(children=[]), "clear")
        self.assertEqual(childless["active_qid"], "ever_pregnant")
        self.assertFalse(call_cursor.has_answered_gates(childless))
        self.assertEqual(call_cursor.step_kind(childless), "main")
        self.assertEqual(call_cursor.active_prompt_hi(childless), call_cursor.ever_pregnant_prompt_hi())
        after_ever = apply_turn(childless, "concern")
        self.assertTrue(call_cursor.has_answered_gates(after_ever))
        self.assertEqual(
            call_cursor.last_answered_prompt_hi(after_ever),
            ("ever_pregnant", call_cursor.ever_pregnant_prompt_hi()),
        )
        self.assertEqual(call_cursor.accepted_target(after_ever, "ever_pregnant"), "ever_pregnant")
        self.assertTrue(call_cursor.is_answered_or_na(after_ever, "ever_pregnant"))
        self.assertTrue(call_cursor.is_answered_or_na(after_ever, "birth_jaundice"))

        multi_jaundice = _clear_until(
            _profile(children=[_child(6, "male", 1), _child(2, "female", 2)]),
            "birth_jaundice",
        )
        self.assertEqual(
            call_cursor.last_answered_prompt_hi(multi_jaundice)[0],
            "birth_nicu",
        )
        self.assertIn(
            "आपके किसी भी बच्चे",
            call_cursor.last_answered_prompt_hi(multi_jaundice)[1],
        )
        which = apply_turn(multi_jaundice, "concern")
        self.assertEqual(call_cursor.step_kind(which), "which_child")
        self.assertEqual(call_cursor.step_id(which), "birth_jaundice:-1")
        self.assertEqual(call_cursor.followup_prompt_hi(which), "")
        fup = apply_turn(which, "concern", child_ref="child_1")
        self.assertEqual(call_cursor.step_kind(fup), "followup")
        self.assertEqual(call_cursor.step_id(fup), "birth_jaundice:0")
        self.assertEqual(call_cursor.active_prompt_hi(fup), FOLLOWUP_HI["jaundice_days"])
        self.assertEqual(call_cursor.followup_prompt_hi(fup), FOLLOWUP_HI["jaundice_days"])

        close = _clear_until(_profile(children=[_child(4, "male")]), "close_anything_else")
        self.assertEqual(call_cursor.step_kind(close), "close")
        self.assertEqual(call_cursor.followup_prompt_hi(close), FOLLOWUP_HI["open_concern_detail"])
        close_more = apply_turn(close, "concern", note="पेट दर्द")
        self.assertEqual(call_cursor.step_id(close_more), "close_anything_else:more:1")
        self.assertIn("कुछ और बताना चाहेंगी", call_cursor.active_prompt_hi(close_more))
        closed = apply_turn(close_more, "clear")
        self.assertEqual(call_cursor.active_prompt_hi(closed), "")
        self.assertEqual(call_cursor.step_id(closed), "closing:complete")


if __name__ == "__main__":
    unittest.main()
