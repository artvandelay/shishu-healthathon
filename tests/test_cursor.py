"""Offline trajectories for the v4 atomic catalog and FormBharo cursor."""

import ast
import json
import re
import unittest

from tests import APP_DIR, REPO_ROOT
from tests.helpers import clear_until, make_child, make_profile_state

import call_cursor  # noqa: E402
import triage_catalog  # noqa: E402
from call_cursor import (  # noqa: E402
    active_prompt_hi,
    apply_interpreted_turn,
    apply_profile,
    apply_turn,
    close_technical_failure,
    expect_yes_no,
    new_cursor_state,
    routable_qids,
)
from triage_catalog import FOLLOWUP_HI, active_specs, get_spec  # noqa: E402

_DOC = REPO_ROOT / "docs" / "requirements-v4.md"
_BACHCHA_LATER = re.compile(r"बच्चा(?!े|ों)")
_PREG_HISTORY = ("preg_conditions", "preg_scan", "preg_transfusion", "preg_delivery", "fam_miscarriage")


def _collision(text: str) -> bool:
    start = 0
    while (at := text.find("बच्चे", start)) >= 0:
        if _BACHCHA_LATER.search(text[at + len("बच्चे"):]):
            return True
        start = at + len("बच्चे")
    return False


def _cells(line):
    return [part.strip() for part in line.strip().strip("|").split("|")]


def _tokens(cell):
    found = re.findall(r"`([^`]+)`", cell)
    if not found or found == ["()"]:
        return ()
    if len(found) == 1 and found[0].startswith("("):
        inner = found[0].strip("()").strip()
        return tuple(p.strip().strip('"').strip("'") for p in inner.split(",") if p.strip()) if inner else ()
    return tuple(found)


def _hindi(cell):
    raw = cell.strip()
    return "" if raw.startswith("*") else raw


def _spec_tables():
    lines = _DOC.read_text(encoding="utf-8").splitlines()
    follow_at = lines.index("### 4.1 Follow-Up Prompts (`FOLLOWUP_HI`)")
    intro_at = lines.index("### 4.2 Intro & Open Capture Items")
    clin_at = lines.index('### 4.3 Atomic Clinical Questions (56 Items — All `kind="gate"`)')
    end = lines.index("---", clin_at)
    followups = [(_tokens(c[0])[0], _hindi(c[1])) for line in lines[follow_at:intro_at] if line.startswith("| `") and (c := _cells(line))]
    intros = [(_tokens(c[0])[0], _tokens(c[1])[0], _tokens(c[2])[0], _tokens(c[3]), _hindi(c[4])) for line in lines[intro_at:clin_at] if line.startswith("| `") and (c := _cells(line))]
    clinical = [
        {"id": _tokens(c[1])[0], "group": _tokens(c[2])[0], "nrroid": _tokens(c[3])[0], "skip": _tokens(c[4]),
         "hi": _hindi(c[5]), "multi": _hindi(c[6]), "follow": _tokens(c[7])}
        for line in lines[clin_at:end] if line.startswith("| ") and (c := _cells(line)) and c[0].isdigit()
    ]
    return followups, intros, clinical


class CatalogTests(unittest.TestCase):
    def test_catalog_matches_section_4_verbatim(self):
        followups, intros, clinical = _spec_tables()
        self.assertEqual(len(followups), 25)
        self.assertEqual(FOLLOWUP_HI, dict(followups))
        self.assertEqual([r[0] for r in intros], ["intro_name", "intro_married", "intro_children", "intro_bridge", "close_anything_else"])
        for qid, group, kind, skip, hindi in intros:
            spec = get_spec(qid)
            self.assertEqual((spec.group, spec.kind, spec.skip_if, spec.question_hi), (group, kind, skip, hindi), qid)
        self.assertEqual(get_spec("close_anything_else").followups, ("open_concern_detail",))
        self.assertEqual(len(clinical), 56)
        gates = [spec for spec in triage_catalog.CATALOG if spec.kind == "gate"]
        self.assertEqual([spec.id for spec in gates], [row["id"] for row in clinical])
        child_multi = 0
        for row in clinical:
            spec = get_spec(row["id"])
            self.assertEqual(
                (spec.kind, spec.group, spec.nrroid_gate, spec.skip_if, spec.question_hi, spec.question_multi_hi, spec.followups),
                ("gate", row["group"], row["nrroid"], row["skip"], row["hi"], row["multi"], row["follow"]),
                row["id"],
            )
            if spec.group in ("birth", "body", "illness") and "no_children" in spec.skip_if:
                self.assertTrue(spec.question_multi_hi.strip(), row["id"])
                child_multi += 1
        self.assertEqual(child_multi, 36)
        self.assertEqual(get_spec("fam_same_problem").question_multi_hi, "")
        self.assertEqual(len(active_specs()), len(triage_catalog.CATALOG))
        json.dumps(new_cursor_state())

    def test_two_children_use_grammatical_catalog_multi_string(self):
        state = make_profile_state(children=[make_child(6, "male", 1), make_child(2, "female", 2)])
        seen = 0
        for spec in active_specs():
            if not spec.question_multi_hi:
                continue
            seen += 1
            text = call_cursor.question_hi_for(state, spec)
            self.assertEqual(text, spec.question_multi_hi, spec.id)
            self.assertNotIn("—", text, spec.id)
            self.assertFalse(text.startswith("आपके किसी भी बच्चे —"), spec.id)
            self.assertFalse(_collision(text), f"{spec.id}: {text}")
        self.assertEqual(seen, 36)
        alone = make_profile_state(children=[make_child(4, "male")])
        nicu = get_spec("birth_nicu")
        self.assertEqual(call_cursor.question_hi_for(alone, nicu), nicu.question_hi)
        self.assertNotIn("किसी भी", nicu.question_hi)
        self.assertIn("आपके किसी भी बच्चे का जन्म", call_cursor.question_hi_for(state, get_spec("birth_term")))
        self.assertIn("आपका कोई भी बच्चा", call_cursor.question_hi_for(state, get_spec("birth_sudden")))
        self.assertIn("आपके किसी भी बच्चे को मामूली चोट", call_cursor.question_hi_for(state, get_spec("body_bleeding_bruise")))
        fasting = call_cursor.question_hi_for(state, get_spec("ill_fasting"))
        self.assertTrue("आपका कोई भी बच्चा" in fasting and "सुस्त" in fasting)


class ProfileTests(unittest.TestCase):
    def test_fragment_and_missing_ages_ask_for_the_missing_piece(self):
        state, hold = apply_profile(new_cursor_state())
        self.assertEqual((hold, state["hold_reason"], state["profile"]["caller_name"]), ("name_fragment", "name_fragment", None))
        self.assertIn("पूरा नाम", active_prompt_hi(state))
        self.assertFalse(expect_yes_no(state))

        state, hold = apply_profile(state, caller_name="   ")
        self.assertEqual((hold, state["hold_reason"]), ("name_fragment", "name_fragment"))

        state, hold = apply_profile(state, caller_name="पूजा")
        self.assertEqual((hold, state["hold_reason"], state["active_qid"]), (None, None, "intro_married"))
        state, hold = apply_profile(state, marital_status="married")
        self.assertEqual(state["active_qid"], "intro_children")
        state, hold = apply_profile(state, children=[], child_count=2)
        self.assertEqual((hold, state["hold_reason"], state["profile"]["children"]), ("children_ages", "children_ages", []))
        self.assertIsNone(state["profile"]["ever_pregnant"])
        self.assertIn("उम्र", active_prompt_hi(state))
        self.assertFalse(expect_yes_no(state))

    def test_bridge_yes_records_clear_and_yes_no_is_off_for_intro_profile(self):
        state = make_profile_state(children=[make_child(4, "male")])
        self.assertEqual(state["active_qid"], "intro_bridge")
        self.assertTrue(expect_yes_no(state))
        self.assertFalse(expect_yes_no(new_cursor_state()))
        state = apply_turn(state, "concern")
        self.assertEqual((state["answers"]["intro_bridge"]["state"], state["active_qid"]), ("clear", "birth_mode_detail"))
        self.assertTrue(expect_yes_no(state))


class SkipMatrixTests(unittest.TestCase):
    def test_marital_status_skip_matrix(self):
        unmarried = make_profile_state(marital="unmarried")
        about = make_profile_state(marital="about_to_marry")
        for st in (unmarried, about):
            self.assertEqual((st["active_qid"], st["profile"]["ever_pregnant"], st["answers"]["intro_children"]["state"], st["profile"]["children"]),
                             ("intro_bridge", False, "not_applicable", []))
            for qid in _PREG_HISTORY:
                self.assertEqual(st["answers"][qid]["state"], "not_applicable", qid)
            self.assertNotEqual(st["answers"].get("preg_thalassemia", {}).get("state"), "not_applicable")
        self.assertEqual(unmarried["answers"]["fam_related"]["state"], "not_applicable")
        self.assertNotIn("fam_related", about["answers"])

        childless = apply_turn(make_profile_state(children=[]), "clear")
        self.assertEqual(childless["active_qid"], "ever_pregnant")
        self.assertTrue(expect_yes_no(childless))
        childless = apply_turn(childless, "unknown")
        self.assertEqual((childless["profile"]["ever_pregnant"], childless["active_qid"]), (False, "preg_thalassemia"))
        for qid in _PREG_HISTORY:
            self.assertEqual(childless["answers"][qid]["state"], "not_applicable", qid)

        pregnant = apply_turn(apply_turn(make_profile_state(children=[]), "clear"), "concern")
        self.assertEqual((pregnant["profile"]["ever_pregnant"], pregnant["active_qid"]), (True, "preg_conditions"))

    def test_dynamic_thalassemia_and_related_wording(self):
        married = clear_until(make_profile_state(children=[make_child(4, "female")]), "preg_thalassemia")
        self.assertIn("आपके पति", active_prompt_hi(married))
        self.assertNotIn("होने वाले", active_prompt_hi(married))
        about = apply_turn(make_profile_state(marital="about_to_marry"), "clear")
        self.assertEqual(about["active_qid"], "preg_thalassemia")
        self.assertIn("होने वाले पति", active_prompt_hi(about))
        about = apply_turn(about, "clear")
        self.assertEqual(about["active_qid"], "fam_related")
        self.assertIn("होने वाले पति", active_prompt_hi(about))
        unmarried = apply_turn(make_profile_state(marital="unmarried"), "clear")
        spoken = active_prompt_hi(unmarried)
        self.assertTrue(unmarried["active_qid"] == "preg_thalassemia" and "थैलेसीमिया" in spoken and "पति" not in spoken)
        self.assertEqual(apply_turn(unmarried, "clear")["active_qid"], "fam_infant_death")

    def test_question_hi_for_partial_profiles_and_catalog_wording(self):
        thal, nicu, related = get_spec("preg_thalassemia"), get_spec("birth_nicu"), get_spec("fam_related")
        for partial in (None, {}, {"profile": {}}):
            self.assertEqual(call_cursor.question_hi_for(partial, thal), thal.question_hi)
        unmarried = call_cursor.question_hi_for({"profile": {"marital_status": "unmarried"}}, thal)
        self.assertEqual(unmarried, thal.question_unmarried_hi)
        self.assertNotIn("पति", unmarried)
        self.assertEqual(call_cursor.question_hi_for({"profile": {"marital_status": "about_to_marry"}}, thal), thal.question_about_hi)
        self.assertEqual(call_cursor.question_hi_for({"profile": {"marital_status": "about_to_marry"}}, related), related.question_about_hi)
        self.assertEqual(call_cursor.question_hi_for({"profile": {"children": "nope"}}, nicu), nicu.question_hi)
        kids = {"profile": {"marital_status": "married", "children": [make_child(6, "male", 1), make_child(2, "female", 2)]}}
        self.assertEqual(call_cursor.question_hi_for(kids, nicu), nicu.question_multi_hi)

    def test_age_rules_skip_tooth_periods_and_puberty(self):
        baby = make_profile_state(children=[make_child(1, "male")])
        for qid in ("mod_tooth_bleed", "mod_heavy_periods", "body_puberty"):
            self.assertEqual(baby["answers"][qid]["state"], "not_applicable", qid)
        boy = make_profile_state(children=[make_child(6, "male")])
        self.assertNotIn("mod_tooth_bleed", boy["answers"])
        self.assertEqual((boy["answers"]["mod_heavy_periods"]["state"], boy["answers"]["body_puberty"]["state"]), ("not_applicable", "not_applicable"))
        girl = make_profile_state(children=[make_child(12, "female")])
        for qid in ("mod_tooth_bleed", "mod_heavy_periods", "body_puberty"):
            self.assertNotIn(qid, girl["answers"])

    def test_call6_marital_correction_skips_the_question_she_is_on(self):
        state, _ = apply_profile(new_cursor_state(), caller_name="पूजा")
        state, _ = apply_profile(state, marital_status="married")
        self.assertEqual(state["active_qid"], "intro_children")
        state, hold = apply_profile(state, marital_status="about_to_marry")
        self.assertEqual((hold, state["active_qid"], state["answers"]["intro_children"]["state"], state["profile"]["ever_pregnant"], state["profile"]["children"]),
                         (None, "intro_bridge", "not_applicable", False, []))

        state = clear_until(make_profile_state(children=[make_child(4, "male"), make_child(2, "female", 2)]), "birth_nicu")
        state, hold = apply_profile(state, marital_status="unmarried")
        self.assertEqual((hold, state["profile"]["children"], state["profile"]["ever_pregnant"], state["answers"]["birth_nicu"]["state"], state["active_qid"]),
                         (None, [], False, "not_applicable", "preg_thalassemia"))
        self.assertNotIn("पति", active_prompt_hi(state))

        stayed, _ = apply_profile(clear_until(make_profile_state(children=[make_child(3, "female")]), "fam_infant_death"), marital_status="unmarried")
        self.assertEqual((stayed["active_qid"], stayed["profile"]["children"]), ("fam_infant_death", []))


class ChildResolutionTests(unittest.TestCase):
    def _three(self):
        return [make_child(2, "male", 1), make_child(8, "female", 2, name="रिया"), make_child(5, "male", 3, name="अर्जुन")]

    def test_call8_middle_child_and_retraction_from_which_child(self):
        state = apply_turn(clear_until(make_profile_state(children=self._three()), "birth_jaundice"), "concern")
        self.assertEqual((state["pending_followup_idx"], state["step_attempts"]), (-1, 0))
        self.assertEqual(active_prompt_hi(state), "यह किस बच्चे की बात है — 8 साल वाली रिया, 5 साल वाले अर्जुन (बीच वाले), या 2 साल वाले बेटे?")
        state = apply_turn(state, "concern", child_ref="child_3")
        self.assertEqual((state["answers"]["birth_jaundice"]["child_ids"], state["answers"]["birth_jaundice"]["state"], state["pending_followup_idx"]),
                         (["child_3"], "concern", 0))

        plain = [make_child(2, "unknown", 1), make_child(8, "unknown", 2), make_child(5, "unknown", 3)]
        asked = apply_turn(clear_until(make_profile_state(children=plain), "body_bleeding_bruise"), "concern")
        self.assertIn("8 साल वाले, 5 साल वाले (बीच वाले), या 2 साल वाले", active_prompt_hi(asked))
        retracted = apply_turn(asked, "clear")
        self.assertEqual((retracted["answers"]["body_bleeding_bruise"]["state"], retracted["pending_followup_idx"], retracted["step_attempts"]), ("clear", None, 0))
        self.assertNotEqual(retracted["active_qid"], "body_bleeding_bruise")

    def test_child_ref_resolves_only_closed_world_ids_and_all(self):
        state = make_profile_state(children=self._three())
        self.assertEqual(call_cursor._resolve_child_ids(state, "all"), ["child_2", "child_3", "child_1"])
        for cid in ("child_1", "child_2", "child_3"):
            self.assertEqual(call_cursor._resolve_child_ids(state, cid), [cid])
        for raw in ("both", "दोनों", "सभी", "Both", "ALL", "रिया", "बड़े", "बीच वाले", "oldest", "middle", "youngest", "5 साल", "child_9"):
            self.assertEqual(call_cursor._resolve_child_ids(state, raw), [], raw)
        months = make_profile_state(children=[make_child(4, "male", 1), make_child(0.5, "female", 2)])
        self.assertEqual(call_cursor._resolve_child_ids(months, "6 महीने"), [])
        self.assertEqual(call_cursor._resolve_child_ids(months, "child_2"), ["child_2"])

    def test_cursor_has_no_regex_and_roster_is_oldest_first(self):
        py_files = sorted(APP_DIR.glob("*.py"))
        self.assertGreaterEqual(len(py_files), 10)
        for path in py_files:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                         else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
                self.assertFalse(any(n == "re" or n.startswith("re.") for n in names), f"{path.name} must not import re")
        roster = call_cursor.children_roster(make_profile_state(children=self._three()))
        self.assertEqual([r["id"] for r in roster], ["child_2", "child_3", "child_1"])
        self.assertEqual([r["rank"] for r in roster], ["oldest", "middle", "youngest"])
        self.assertEqual((roster[0]["name"], roster[0]["sex"], roster[1]["age_label_hi"]), ("रिया", "female", "5 साल"))
        self.assertEqual(call_cursor.children_roster(make_profile_state(children=[make_child(4, "male")])), [])

    def test_two_attempts_leave_a_stuck_which_child_and_followup(self):
        kids = [make_child(6, "male", 1), make_child(2, "female", 2)]
        state = apply_turn(clear_until(make_profile_state(children=kids), "birth_nicu"), "concern")
        state = apply_turn(state, "concern", note="पता नहीं कौन")
        self.assertEqual((state["pending_followup_idx"], state["step_attempts"]), (-1, 1))
        state = apply_turn(state, "concern")
        self.assertEqual((state["active_qid"], state["pending_followup_idx"], state["answers"]["birth_nicu"]["child_ids"]), ("birth_nicu", 0, []))
        self.assertIn("पता नहीं कौन", state["answers"]["birth_nicu"]["notes"])
        state = apply_turn(apply_turn(state, "concern", note=""), "concern", note="")
        self.assertEqual(state["answers"]["birth_nicu"]["followups"]["nicu_reason"], {"state": "unknown", "note": ""})
        self.assertEqual(state["pending_followup_idx"], 1)
        state = apply_turn(apply_turn(state, "concern", note=""), "concern", note="")
        self.assertEqual(state["answers"]["birth_nicu"]["followups"]["nicu_days"], {"state": "unknown", "note": ""})
        self.assertNotEqual(state["active_qid"], "birth_nicu")

        bleeding = apply_turn(clear_until(make_profile_state(children=kids), "body_bleeding_bruise"), "concern")
        visits = {}
        for _ in range(12):
            if bleeding["phase"] != "asking":
                break
            key = (bleeding["active_qid"], bleeding.get("pending_followup_idx"))
            visits[key] = visits.get(key, 0) + 1
            self.assertLessEqual(visits[key], 2, key)
            bleeding = apply_turn(bleeding, "concern")
        self.assertNotEqual(bleeding["active_qid"], "body_bleeding_bruise")
        self.assertEqual((bleeding["answers"]["body_bleeding_bruise"]["state"], bleeding["answers"]["body_bleeding_bruise"]["child_ids"]), ("concern", []))

    def test_preanswered_and_two_step_followups_and_retract(self):
        pre = apply_turn(clear_until(make_profile_state(children=[make_child(4, "male")]), "birth_jaundice"), "concern", note="दो दिन", followup_answered=True)
        self.assertIsNone(pre["pending_followup_idx"])
        self.assertNotEqual(pre["active_qid"], "birth_jaundice")
        self.assertEqual(pre["answers"]["birth_jaundice"]["followups"]["jaundice_days"]["note"], "दो दिन")

        steps = {
            "birth_nicu": ("nicu_reason", "nicu_days", "कितने दिन"),
            "ill_admissions": ("admissions_count", "admissions_reason", "किस बीमारी"),
            "fam_miscarriage": ("miscarriage_count", "miscarriage_month", "कितने महीने"),
            "fam_infant_death": ("infant_death_age", "infant_death_cause", "मौत की कोई वजह"),
        }
        state = make_profile_state(children=[make_child(4, "male")])
        for qid, (first, second, second_prompt) in steps.items():
            state = apply_turn(clear_until(state, qid), "concern", note="पहली बात", followup_answered=True)
            self.assertEqual((state["active_qid"], state["pending_followup_idx"]), (qid, 1), qid)
            self.assertEqual(state["answers"][qid]["followups"][first]["note"], "पहली बात")
            self.assertNotIn(second, state["answers"][qid]["followups"])
            self.assertIn(second_prompt, active_prompt_hi(state))
            state = apply_turn(state, "concern", note="दूसरी बात")
            self.assertIsNone(state["pending_followup_idx"], qid)
            self.assertEqual(state["answers"][qid]["followups"][second]["note"], "दूसरी बात")
            self.assertEqual(state["answers"][qid]["followups"][first]["note"], "पहली बात")

        kids = [make_child(6, "male", 1), make_child(2, "female", 2)]
        multi = apply_turn(clear_until(make_profile_state(children=kids), "birth_nicu"), "concern", note="सांस", followup_answered=True)
        self.assertEqual((multi["pending_followup_idx"], multi["answers"]["birth_nicu"]["followups"]["nicu_reason"]["note"]), (-1, "सांस"))
        multi = apply_turn(multi, "concern", child_ref="child_1")
        self.assertEqual((multi["answers"]["birth_nicu"]["child_ids"], multi["pending_followup_idx"]), (["child_1"], 1))
        self.assertIn("कितने दिन", active_prompt_hi(multi))

        fup = apply_turn(clear_until(make_profile_state(children=[make_child(4, "male")]), "birth_nicu"), "concern")
        self.assertEqual(fup["pending_followup_idx"], 0)
        self.assertFalse(expect_yes_no(fup))
        retracted = apply_turn(fup, "retract")
        self.assertEqual((retracted["answers"]["birth_nicu"]["state"], retracted["active_qid"], retracted["pending_followup_idx"]), ("clear", "birth_jaundice", None))


class CloseAndFinishTests(unittest.TestCase):
    def test_close_bare_yes_asks_for_the_note_then_a_noted_yes_keeps_it(self):
        state = clear_until(make_profile_state(children=[make_child(4, "male")]), "close_anything_else")
        self.assertTrue(expect_yes_no(state))
        state = apply_turn(state, "concern")
        self.assertEqual((state["phase"], state["pending_followup_idx"], expect_yes_no(state)), ("asking", 0, False))
        self.assertIn("क्या बात नोट करूँ", active_prompt_hi(state))
        state = apply_turn(state, "concern", note="पेट में दर्द")
        self.assertEqual((state["pending_followup_idx"], state["answers"]["close_anything_else"]["notes"]), (None, "पेट में दर्द"))
        self.assertIn("मैंने यह बात नोट कर ली। इसके अलावा कुछ और बताना चाहेंगी?", active_prompt_hi(state))
        self.assertTrue(expect_yes_no(state))
        state = apply_turn(state, "concern", note="नींद कम है")
        self.assertEqual(state["answers"]["close_anything_else"]["notes"], "पेट में दर्द; नींद कम है")
        state = apply_turn(state, "clear")
        self.assertEqual((state["phase"], state["close_reason"], expect_yes_no(state)), ("closing", "complete", False))
        self.assertIn("नींद कम है", state["answers"]["close_anything_else"]["notes"])

        noted = apply_turn(clear_until(make_profile_state(children=[make_child(4, "male")]), "close_anything_else"), "concern", note="पेट दर्द")
        self.assertEqual((noted["pending_followup_idx"], noted["answers"]["close_anything_else"]["notes"]), (None, "पेट दर्द"))
        noted = apply_turn(noted, "unknown")
        self.assertEqual((noted["phase"], noted["answers"]["close_anything_else"]["state"]), ("closing", "concern"))

    def test_one_child_all_clear_finishes_and_emergency_stops(self):
        state = make_profile_state(children=[make_child(4, "male")])
        guard = 0
        while state["phase"] == "asking" and guard < 90:
            state = apply_turn(state, "clear")
            guard += 1
        self.assertEqual((state["phase"], state["close_reason"]), ("closing", "complete"))
        self.assertLess(guard, 90)
        self.assertEqual(state["answers"]["birth_nicu"]["state"], "clear")
        self.assertEqual(state["answers"]["mod_tooth_bleed"]["state"], "not_applicable")
        self.assertEqual(state["answers"]["fam_same_problem"]["state"], "not_applicable")

        stopped = clear_until(make_profile_state(children=[make_child(3, "female")]), "ill_fasting")
        answers_before = json.loads(json.dumps(stopped["answers"]))
        stopped = apply_turn(stopped, "emergency", note="  अभी   सांस नहीं ले रहा  ")
        self.assertEqual((stopped["phase"], stopped["close_reason"], stopped["emergency_note"], stopped["active_qid"]),
                         ("closing", "emergency", "अभी सांस नहीं ले रहा", "ill_fasting"))
        self.assertEqual(stopped["answers"], answers_before)
        self.assertEqual(apply_turn(stopped, "clear")["phase"], "closing")
        quiet = apply_turn(clear_until(make_profile_state(children=[make_child(3, "female")]), "ill_fasting"), "caller_stop", note="अब रहन दो")
        self.assertEqual(quiet["close_reason"], "caller_stop")
        self.assertNotIn("emergency_note", quiet)

    def test_family_concern_opens_same_problem_without_a_hidden_gate(self):
        state = make_profile_state(children=[make_child(4, "male")])
        self.assertEqual(state["answers"]["fam_same_problem"]["state"], "not_applicable")
        state = apply_turn(apply_turn(clear_until(state, "fam_fits_delay"), "concern", note="मामा को दौरे"), "concern", note="मामा")
        state = clear_until(state, "fam_same_problem")
        self.assertEqual(state["active_qid"], "fam_same_problem")
        self.assertNotIn("fam_same_problem", state["answers"])


class CallNineReplayTests(unittest.TestCase):
    def _boy(self):
        return make_profile_state(children=[make_child(4, "male")])

    def test_late_family_addition_stays_on_the_current_question(self):
        state = apply_turn(apply_turn(clear_until(self._boy(), "fam_cancer"), "concern", note="मामा को ब्लड कैंसर"), "concern", note="मामा")
        state = clear_until(state, "fam_heart_kidney")
        updated = apply_interpreted_turn(state, {
            "current_outcome": "not_about_active", "side_qid": "fam_cancer",
            "side_action": "add_or_correct", "side_outcome": "concern", "side_note": "चाची को भी ब्लड कैंसर",
        })
        self.assertEqual((updated["active_qid"], updated["pending_followup_idx"], updated["step_attempts"]),
                         ("fam_heart_kidney", state["pending_followup_idx"], state["step_attempts"]))
        self.assertNotEqual((updated["answers"].get("fam_heart_kidney") or {}).get("state"), "concern")
        cancer = updated["answers"]["fam_cancer"]
        self.assertEqual(cancer["state"], "concern")
        self.assertTrue("मामा को ब्लड कैंसर" in cancer["notes"] and "चाची को भी ब्लड कैंसर" in cancer["notes"])
        self.assertIn("चाची को भी ब्लड कैंसर", cancer["followups"]["fam_who"]["note"])
        self.assertIn("दिल या गुर्दे", active_prompt_hi(updated))

    def test_clarify_on_an_earlier_question_returns_there_then_resumes(self):
        state = apply_turn(clear_until(self._boy(), "preg_thalassemia"), "unknown")
        self.assertEqual((state["active_qid"], state["answers"]["preg_thalassemia"]["state"]), ("fam_related", "unknown"))
        rewound = apply_interpreted_turn(state, {"current_outcome": "not_about_active", "side_qid": "preg_thalassemia", "side_action": "clarify_meaning"})
        self.assertEqual((rewound["active_qid"], rewound["pending_followup_idx"], rewound["step_attempts"], rewound["hold_reason"]),
                         ("preg_thalassemia", None, 0, None))
        self.assertNotIn("preg_thalassemia", rewound["answers"])
        self.assertIn("थैलेसीमिया", active_prompt_hi(rewound))
        answered = apply_turn(rewound, "clear")
        self.assertEqual((answered["answers"]["preg_thalassemia"]["state"], answered["active_qid"], answered["answers"]["birth_mode_detail"]["state"]),
                         ("clear", "fam_related", "clear"))

    def test_maternal_gates_are_atomic_and_clarify_does_not_burn_a_followup(self):
        bleeding, infections, eczema = get_spec("fam_maternal_bleeding"), get_spec("fam_maternal_infections"), get_spec("fam_maternal_eczema")
        self.assertEqual((bleeding.group, bleeding.nrroid_gate, bleeding.kind, bleeding.skip_if), ("family", "Q11", "gate", ()))
        self.assertEqual((bleeding.followups, infections.followups, eczema.followups), (("fam_who",),) * 3)
        self.assertIn("ज़्यादा खून बहने", bleeding.question_hi)
        self.assertNotIn("इंफेक्शन", bleeding.question_hi)
        self.assertIn("गंभीर इंफेक्शन", infections.question_hi)
        self.assertIn("सूखे खुजली वाले धब्बों", eczema.question_hi)
        with self.assertRaises(KeyError):
            get_spec("fam_maternal_side")

        state = apply_turn(clear_until(self._boy(), "fam_maternal_bleeding"), "concern")
        state = apply_turn(apply_turn(state, "clarify"), "clarify")
        self.assertEqual((state["active_qid"], state["pending_followup_idx"], state["step_attempts"]), ("fam_maternal_bleeding", 0, 0))
        self.assertNotIn("fam_who", state["answers"]["fam_maternal_bleeding"].get("followups") or {})

        kids = [make_child(6, "male", 1), make_child(2, "female", 2)]
        which = apply_turn(apply_turn(clear_until(make_profile_state(children=kids), "birth_nicu"), "concern"), "clarify")
        self.assertEqual((which["pending_followup_idx"], which["step_attempts"]), (-1, 0))
        close = apply_turn(apply_turn(clear_until(self._boy(), "close_anything_else"), "concern"), "clarify")
        self.assertEqual((close["phase"], close["pending_followup_idx"], close["step_attempts"]), ("asking", 0, 0))

    def test_revise_writes_a_late_note_into_the_open_followup_deduped(self):
        state = apply_turn(apply_turn(clear_until(self._boy(), "birth_nicu"), "concern", note="सांस", followup_answered=True), "concern", note="तीन दिन")
        later = state["active_qid"]
        payload = {"current_outcome": "not_about_active", "side_qid": "birth_nicu", "side_action": "add_or_correct", "side_outcome": "concern", "side_note": "और बुखार"}
        updated = apply_interpreted_turn(state, payload)
        self.assertEqual(updated["active_qid"], later)
        self.assertEqual(updated["answers"]["birth_nicu"]["followups"]["nicu_reason"]["note"], "सांस")
        self.assertEqual(updated["answers"]["birth_nicu"]["followups"]["nicu_days"]["note"], "तीन दिन; और बुखार")
        again = apply_interpreted_turn(updated, payload)
        self.assertEqual(again["answers"]["birth_nicu"]["followups"]["nicu_days"]["note"], "तीन दिन; और बुखार")

        open_slot = json.loads(json.dumps(state))
        del open_slot["answers"]["birth_nicu"]["followups"]["nicu_days"]
        routed = apply_interpreted_turn(open_slot, {**payload, "side_note": "चार दिन"})
        self.assertEqual((routed["answers"]["birth_nicu"]["followups"]["nicu_days"]["note"], routed["active_qid"]), ("चार दिन", open_slot["active_qid"]))

    def test_clarify_mid_followup_returns_to_the_unfinished_question(self):
        state = apply_turn(clear_until(self._boy(), "birth_nicu"), "concern")
        rewound = apply_interpreted_turn(state, {"current_outcome": "not_about_active", "side_qid": "birth_mode_detail", "side_action": "clarify_meaning"})
        self.assertEqual(rewound["active_qid"], "birth_mode_detail")
        self.assertNotIn("birth_nicu", rewound["answers"])
        answered = apply_turn(rewound, "clear")
        self.assertEqual((answered["answers"]["birth_mode_detail"]["state"], answered["active_qid"], answered["pending_followup_idx"]), ("clear", "birth_nicu", None))

        kids = [make_child(6, "male", 1), make_child(2, "female", 2)]
        which = apply_turn(clear_until(make_profile_state(children=kids), "birth_nicu"), "concern")
        which = apply_interpreted_turn(which, {"current_outcome": "not_about_active", "side_qid": "birth_term", "side_action": "clarify_meaning"})
        self.assertNotIn("birth_nicu", which["answers"])
        which = apply_turn(which, "clear")
        self.assertEqual((which["active_qid"], which["pending_followup_idx"]), ("birth_nicu", None))

        close = apply_turn(apply_turn(clear_until(self._boy(), "close_anything_else"), "concern", note="पेट दर्द"), "concern")
        close = apply_interpreted_turn(close, {"current_outcome": "not_about_active", "side_qid": "birth_mode_detail", "side_action": "clarify_meaning"})
        self.assertEqual((close["answers"]["close_anything_else"]["state"], close["answers"]["close_anything_else"]["notes"]), ("concern", "पेट दर्द"))


class DecisionApiPhase1Tests(unittest.TestCase):
    def test_routable_qids_has_exact_58_ids_in_ask_order(self):
        qids = routable_qids()
        self.assertEqual((len(qids), len(set(qids)), qids[-1]), (58, 58, "close_anything_else"))
        for excluded in ("intro_name", "intro_married", "intro_children", "intro_bridge"):
            self.assertNotIn(excluded, qids)
        self.assertEqual(qids[qids.index("ill_iv_antibiotics") + 1], "ever_pregnant")
        self.assertEqual(qids[qids.index("ever_pregnant") + 1], "preg_conditions")
        gates = tuple(s.id for s in active_specs() if s.kind == "gate")
        self.assertEqual(tuple(q for q in qids if q not in ("ever_pregnant", "close_anything_else")), gates)

    def test_smoke_a_dual_intent_and_side_write_alone(self):
        state = apply_turn(apply_turn(clear_until(make_profile_state(children=[make_child(4, "male")]), "fam_cancer"), "concern", note="मामा को ब्लड कैंसर"), "concern", note="मामा")
        state = clear_until(state, "fam_heart_kidney")

        dual = apply_interpreted_turn(state, {
            "current_outcome": "clear", "current_note": "चाची को भी कैंसर हुआ था",
            "side_qid": "fam_cancer", "side_action": "add_or_correct", "side_outcome": "concern",
            "side_note": "चाची को भी कैंसर हुआ था", "note": "अमान्य नोट",
        })
        self.assertEqual((dual["answers"]["fam_heart_kidney"]["state"], dual["answers"]["fam_heart_kidney"]["notes"], dual["active_qid"]),
                         ("clear", "", "fam_bp_stroke"))
        self.assertIn("चाची को भी कैंसर हुआ था", dual["answers"]["fam_cancer"]["notes"])
        self.assertNotIn("अमान्य नोट", dual["answers"]["fam_cancer"]["notes"])

        side_only = apply_interpreted_turn(state, {
            "current_outcome": "not_about_active", "side_qid": "fam_cancer",
            "side_action": "add_or_correct", "side_outcome": "concern", "side_note": "चाची को भी कैंसर हुआ था",
        })
        self.assertEqual((side_only["active_qid"], side_only["step_attempts"]), ("fam_heart_kidney", 0))
        self.assertNotIn("fam_heart_kidney", side_only["answers"])

        dual_concerns = apply_interpreted_turn(state, {
            "current_outcome": "concern", "current_note": "दिल की बीमारी",
            "side_qid": "fam_cancer", "side_action": "add_or_correct", "side_outcome": "concern", "side_note": "चाची को भी कैंसर",
        })
        self.assertEqual(dual_concerns["answers"]["fam_heart_kidney"]["notes"], "दिल की बीमारी")
        self.assertIn("चाची को भी कैंसर", dual_concerns["answers"]["fam_cancer"]["notes"])

        empty_curr = apply_interpreted_turn(state, {
            "current_outcome": "concern", "current_note": "",
            "side_qid": "fam_cancer", "side_action": "add_or_correct", "side_outcome": "concern", "side_note": "चाची को भी कैंसर",
        })
        self.assertEqual(empty_curr["answers"]["fam_heart_kidney"]["notes"], "")

    def test_invalid_future_bridge_close_and_not_applicable_side_qids_ignored(self):
        state = clear_until(make_profile_state(children=[make_child(3, "male")]), "birth_jaundice")
        self.assertEqual(state["answers"]["mod_tooth_bleed"]["state"], "not_applicable")
        for bad_qid in ("fam_cancer", "does_not_exist", "intro_bridge", "close_anything_else", "mod_tooth_bleed"):
            res = apply_interpreted_turn(state, {"current_outcome": "not_about_active", "side_qid": bad_qid, "side_action": "add_or_correct", "side_outcome": "concern", "side_note": "कुछ बात"})
            self.assertEqual((res["active_qid"], res["answers"]), ("birth_jaundice", state["answers"]), bad_qid)

    def test_clarify_meaning_on_earlier_gate_preserves_clear_on_active_gate(self):
        state = apply_turn(clear_until(make_profile_state(children=[make_child(4, "male")]), "preg_thalassemia"), "unknown")
        rewound = apply_interpreted_turn(state, {"current_outcome": "clear", "side_qid": "preg_thalassemia", "side_action": "clarify_meaning", "current_note": "थैलेसीमिया क्या है?"})
        self.assertEqual((rewound["active_qid"], rewound["pending_followup_idx"], rewound["answers"]["fam_related"]["state"], rewound["answers"]["fam_related"]["notes"]),
                         ("preg_thalassemia", None, "clear", ""))
        self.assertNotIn("preg_thalassemia", rewound["answers"])
        self.assertEqual(apply_interpreted_turn(rewound, {"current_outcome": "clear"})["active_qid"], "fam_miscarriage")
        plain = apply_interpreted_turn(state, {"current_outcome": "not_about_active", "side_qid": "preg_thalassemia", "side_action": "clarify_meaning"})
        self.assertNotIn("fam_related", plain["answers"])

    def test_emergency_and_caller_stop_override_clear_and_side_qid(self):
        state = apply_turn(apply_turn(clear_until(make_profile_state(children=[make_child(4, "male")]), "fam_cancer"), "concern", note="मामा को कैंसर"), "concern", note="मामा")
        state = clear_until(state, "fam_heart_kidney")
        before = json.loads(json.dumps(state["answers"]))
        emerg = apply_interpreted_turn(state, {"emergency": True, "current_outcome": "clear", "side_qid": "fam_cancer", "side_action": "add_or_correct", "side_outcome": "clear", "current_note": "  सांस नहीं आ रही  "})
        self.assertEqual((emerg["phase"], emerg["close_reason"], emerg["emergency_note"], emerg["answers"]), ("closing", "emergency", "सांस नहीं आ रही", before))
        self.assertEqual(apply_interpreted_turn(emerg, {"current_outcome": "clear"}), emerg)
        stop = apply_interpreted_turn(state, {"caller_stop": True, "current_outcome": "clear", "side_qid": "fam_cancer", "side_action": "add_or_correct", "side_outcome": "clear"})
        self.assertEqual((stop["phase"], stop["close_reason"], stop["answers"]), ("closing", "caller_stop", before))

    def test_midcall_marital_unmarried_skips_child_gates_and_no_double_apply(self):
        state = clear_until(make_profile_state(children=[make_child(4, "male"), make_child(2, "female", 2)]), "birth_nicu")
        updated = apply_interpreted_turn(state, {
            "profile_updates": {"marital_status": "unmarried"}, "current_outcome": "clear",
            "side_qid": "birth_mode_detail", "side_action": "add_or_correct", "side_outcome": "concern", "side_note": "ऑपरेशन हुआ था",
        })
        self.assertEqual((updated["profile"]["marital_status"], updated["profile"]["children"], updated["active_qid"]), ("unmarried", [], "preg_thalassemia"))
        self.assertEqual((updated["answers"]["birth_mode_detail"]["state"], updated["answers"]["birth_nicu"]["state"]), ("not_applicable", "not_applicable"))
        self.assertNotIn("preg_thalassemia", updated["answers"])

    def test_side_child_ref_disambiguation_and_pending_followup_retries(self):
        kids = [make_child(6, "male", 1), make_child(2, "female", 2)]
        state = clear_until(make_profile_state(children=kids), "birth_jaundice")
        state["answers"]["birth_nicu"] = {"state": "concern", "child_ids": ["child_1"], "notes": "सांस", "followups": {}}
        both = apply_interpreted_turn(state, {
            "current_outcome": "concern", "current_note": "दो दिन पीलिया", "followup_answered": True, "child_ref": "child_2",
            "side_qid": "birth_nicu", "side_action": "add_or_correct", "side_outcome": "concern", "side_note": "काँच के बक्से में भी",
        })
        self.assertEqual((both["answers"]["birth_jaundice"]["child_ids"], both["answers"]["birth_nicu"]["child_ids"]), (["child_2"], ["child_1"]))

        side_only = apply_interpreted_turn(state, {
            "current_outcome": "not_about_active", "child_ref": "child_2",
            "side_qid": "birth_nicu", "side_action": "add_or_correct", "side_outcome": "concern", "side_note": "छोटी को भी NICU",
        })
        self.assertEqual((side_only["answers"]["birth_nicu"]["child_ids"], side_only["active_qid"]), (["child_1", "child_2"], "birth_jaundice"))

        which = apply_interpreted_turn(state, {"current_outcome": "concern"})
        which_side = apply_interpreted_turn(which, {"current_outcome": "not_about_active", "side_qid": "birth_mode_detail", "side_action": "add_or_correct", "side_outcome": "concern", "side_note": "ऑपरेशन"})
        self.assertEqual((which_side["pending_followup_idx"], which_side["step_attempts"]), (-1, 0))
        which_1 = apply_interpreted_turn(which_side, {"current_outcome": "not_about_active"})
        which_2 = apply_interpreted_turn(which_1, {"current_outcome": "not_about_active"})
        self.assertEqual((which_1["step_attempts"], which_2["pending_followup_idx"], which_2["answers"]["birth_jaundice"]["child_ids"]), (1, 0, []))
        clar = apply_interpreted_turn(which_2, {"current_outcome": "clarify"})
        self.assertEqual((clar["pending_followup_idx"], clar["step_attempts"]), (0, 0))
        fup_2 = apply_interpreted_turn(apply_interpreted_turn(clar, {"current_outcome": "not_about_active"}), {"current_outcome": "not_about_active"})
        self.assertEqual(fup_2["answers"]["birth_jaundice"]["followups"]["jaundice_days"], {"state": "unknown", "note": ""})
        self.assertNotEqual(fup_2["active_qid"], "birth_jaundice")

    def test_close_technical_failure_preserves_answers_and_attempts(self):
        state = apply_turn(apply_turn(clear_until(make_profile_state(children=[make_child(4, "male")]), "birth_nicu"), "concern"), "concern", note="")
        snapshot = json.loads(json.dumps(state))
        closed = close_technical_failure(state)
        self.assertEqual(state, snapshot)
        self.assertEqual((closed["phase"], closed["close_reason"], closed["pending_followup_idx"], closed["hold_reason"], closed["step_attempts"]),
                         ("closing", "technical_failure", None, None, 1))
        self.assertEqual(closed["answers"], snapshot["answers"])
        already = dict(closed, close_reason="complete")
        self.assertEqual(close_technical_failure(already), already)

    def test_canonical_state_inspection_api(self):
        intro = new_cursor_state()
        self.assertEqual((call_cursor.step_kind(intro), call_cursor.step_id(intro), call_cursor.followup_prompt_hi(intro), call_cursor.last_answered_prompt_hi(intro)),
                         ("profile", "intro_name", "", ("", "")))
        self.assertFalse(call_cursor.has_answered_gates(intro))
        held, _ = apply_profile(intro)
        self.assertEqual((call_cursor.step_kind(held), call_cursor.step_id(held)), ("profile", "intro_name:hold:name_fragment"))

        bridge = make_profile_state(children=[make_child(6, "male", 1), make_child(2, "female", 2)])
        self.assertEqual((call_cursor.step_kind(bridge), call_cursor.has_answered_gates(bridge)), ("bridge", False))
        first_gate = apply_turn(bridge, "clear")
        self.assertEqual((call_cursor.step_kind(first_gate), call_cursor.has_answered_gates(first_gate), call_cursor.followup_prompt_hi(first_gate)),
                         ("main", True, FOLLOWUP_HI["operation_reason"]))

        childless = apply_turn(make_profile_state(children=[]), "clear")
        self.assertEqual((childless["active_qid"], call_cursor.has_answered_gates(childless), call_cursor.active_prompt_hi(childless)),
                         ("ever_pregnant", False, get_spec("ever_pregnant").question_hi))
        after_ever = apply_turn(childless, "concern")
        self.assertTrue(call_cursor.has_answered_gates(after_ever) and call_cursor.is_answered_or_na(after_ever, "birth_jaundice"))
        self.assertEqual(call_cursor.accepted_target(after_ever, "ever_pregnant"), "ever_pregnant")

        multi = clear_until(make_profile_state(children=[make_child(6, "male", 1), make_child(2, "female", 2)]), "birth_jaundice")
        self.assertEqual(call_cursor.last_answered_prompt_hi(multi)[0], "birth_nicu")
        which = apply_turn(multi, "concern")
        self.assertEqual((call_cursor.step_kind(which), call_cursor.step_id(which), call_cursor.followup_prompt_hi(which)), ("which_child", "birth_jaundice:-1", ""))
        fup = apply_turn(which, "concern", child_ref="child_1")
        self.assertEqual((call_cursor.step_kind(fup), call_cursor.step_id(fup), call_cursor.active_prompt_hi(fup)),
                         ("followup", "birth_jaundice:0", FOLLOWUP_HI["jaundice_days"]))

        close = clear_until(make_profile_state(children=[make_child(4, "male")]), "close_anything_else")
        self.assertEqual((call_cursor.step_kind(close), call_cursor.followup_prompt_hi(close)), ("close", FOLLOWUP_HI["open_concern_detail"]))
        close_more = apply_turn(close, "concern", note="पेट दर्द")
        self.assertEqual(call_cursor.step_id(close_more), "close_anything_else:more:1")
        closed = apply_turn(close_more, "clear")
        self.assertEqual((call_cursor.active_prompt_hi(closed), call_cursor.step_id(closed)), ("", "closing:complete"))

    def test_diagnostic_notes_isolation_both_orders_and_child_label_filtering(self):
        kids = [make_child(2, "male", 1, name="अमन"), make_child(8, "female", 2, name="रिया")]
        s1 = apply_turn(clear_until(make_profile_state(children=kids), "body_bleeding_bruise"), "concern", child_ref="child_1", note="चोट पर खून देर तक बहता है", diagnostic=False)
        s2 = apply_interpreted_turn(s1, {"current_outcome": "not_about_active", "diagnostic_note": True, "side_qid": "body_bleeding_bruise", "side_action": "add_or_correct", "side_outcome": "concern", "side_note": "डॉक्टर ने Hemophilia बताया"})
        self.assertEqual(call_cursor.answer_notes_text(s2["answers"]["body_bleeding_bruise"]), "चोट पर खून देर तक बहता है; डॉक्टर ने Hemophilia बताया")
        self.assertEqual(call_cursor.mother_notes_text(s2["answers"]["body_bleeding_bruise"]), "चोट पर खून देर तक बहता है")

        # Diagnostic followup merged with safe followup & reverse order (diagnostic first, safe side note second)
        nicu = clear_until(make_profile_state(children=kids), "birth_nicu")
        nicu = apply_turn(nicu, "concern", child_ref="child_2", note="पॉम्पे बीमारी की जाँच", diagnostic=True)
        nicu = apply_turn(nicu, "concern", note="साँस की तकलीफ", diagnostic=False)
        nicu = apply_turn(nicu, "concern", note="Pompe confirm हुआ", diagnostic=True)
        nicu = apply_interpreted_turn(nicu, {"current_outcome": "not_about_active", "diagnostic_note": False, "side_qid": "birth_nicu", "side_action": "add_or_correct", "side_outcome": "concern", "side_note": "चार दिन"})
        nicu_entry = nicu["answers"]["birth_nicu"]
        self.assertIn("Pompe confirm हुआ", call_cursor.answer_notes_text(nicu_entry))
        self.assertIn("चार दिन", call_cursor.answer_notes_text(nicu_entry))
        self.assertEqual(call_cursor.mother_notes_text(nicu_entry), "चार दिन; साँस की तकलीफ")
        self.assertNotIn("Pompe", call_cursor.mother_notes_text(nicu_entry))

        # Reverse order on single-followup gate: diagnostic followup merged with later safe side note
        term = apply_turn(clear_until(make_profile_state(children=[make_child(4, "male")]), "birth_term"), "concern", note="Pompe", followup_answered=True, diagnostic=True)
        term = apply_interpreted_turn(term, {"current_outcome": "not_about_active", "diagnostic_note": False, "side_qid": "birth_term", "side_action": "add_or_correct", "side_outcome": "concern", "side_note": "चार हफ्ते पहले"})
        self.assertEqual(call_cursor.answer_notes_text(term["answers"]["birth_term"]), "Pompe; चार हफ्ते पहले")
        self.assertEqual(call_cursor.mother_notes_text(term["answers"]["birth_term"]), "चार हफ्ते पहले")

        prof = s1["profile"]
        self.assertEqual(call_cursor.children_label_hi(prof), "अमन 2 साल बेटा · रिया 8 साल बेटी")
        self.assertEqual(call_cursor.children_label_hi(prof, ["child_2"]), "रिया 8 साल बेटी")
        self.assertEqual(call_cursor.children_label_hi(prof, []), "")
        self.assertEqual(call_cursor.children_label_en(prof, ["child_2"]), "child_2 (female, age 8.0, रिया)")
        self.assertEqual(call_cursor.children_label_en(prof, []), "")

    def test_ever_pregnant_catalog_sync_holds_and_unmarried_correction(self):
        ever_spec = get_spec("ever_pregnant")
        self.assertEqual((ever_spec.kind, ever_spec.group, ever_spec.skip_if), ("routing_gate", "pregnancy", ("unmarried", "about_to_marry", "has_children")))
        ids = triage_catalog.spine_ids()
        self.assertEqual(ids[ids.index("ever_pregnant") + 1], "preg_conditions")

        childless = apply_turn(make_profile_state(children=[]), "clear")
        renamed = apply_interpreted_turn(childless, {"current_outcome": "not_about_active", "profile_updates": {"caller_name": "कविता"}})
        self.assertEqual((renamed["profile"]["caller_name"], renamed["hold_reason"], renamed["active_qid"]), ("कविता", None, "ever_pregnant"))
        self.assertTrue(expect_yes_no(renamed))

        answered_yes = apply_turn(childless, "concern")
        self.assertTrue(answered_yes["profile"]["ever_pregnant"])
        unmarried_fix, hold = apply_profile(answered_yes, marital_status="unmarried")
        self.assertEqual((hold, unmarried_fix["profile"]["ever_pregnant"], unmarried_fix["answers"]["ever_pregnant"]["state"]), (None, False, "not_applicable"))
        for qid in _PREG_HISTORY:
            self.assertEqual(unmarried_fix["answers"][qid]["state"], "not_applicable", qid)

        via_prof_no, _ = apply_profile(childless, ever_pregnant=False)
        self.assertEqual((via_prof_no["answers"]["ever_pregnant"]["state"], via_prof_no["active_qid"]), ("clear", "preg_thalassemia"))
        rev_unk = apply_interpreted_turn(via_prof_no, {"current_outcome": "not_about_active", "side_qid": "ever_pregnant", "side_action": "add_or_correct", "side_outcome": "unknown"})
        self.assertEqual((rev_unk["profile"]["ever_pregnant"], rev_unk["answers"]["ever_pregnant"]["state"]), (False, "unknown"))
        rewound = apply_interpreted_turn(rev_unk, {"current_outcome": "not_about_active", "side_qid": "ever_pregnant", "side_action": "clarify_meaning"})
        self.assertEqual((rewound["active_qid"], rewound["profile"]["ever_pregnant"]), ("ever_pregnant", None))

    def test_skipped_active_qid_does_not_stamp_last_answered_qid(self):
        state = clear_until(make_profile_state(children=[make_child(4, "male"), make_child(2, "female", 2)]), "birth_nicu")
        self.assertEqual(state["last_answered_qid"], "birth_term")
        updated, _ = apply_profile(state, marital_status="unmarried")
        self.assertEqual((updated["active_qid"], updated["last_answered_qid"]), ("preg_thalassemia", "birth_term"))

        fam_state = clear_until(apply_turn(apply_turn(clear_until(make_profile_state(children=[make_child(4, "male")]), "fam_cancer"), "concern", note="मामा को कैंसर"), "concern", note="मामा"), "fam_same_problem")
        self.assertEqual(fam_state["last_answered_qid"], "fam_lumps")
        retracted = apply_interpreted_turn(fam_state, {"current_outcome": "not_about_active", "side_qid": "fam_cancer", "side_action": "add_or_correct", "side_outcome": "retract"})
        self.assertEqual((retracted["answers"]["fam_same_problem"]["state"], retracted["active_qid"], retracted["last_answered_qid"]),
                         ("not_applicable", "close_anything_else", "fam_lumps"))

    def test_midcall_unmarried_to_married_reopens_gates_and_resets_ever_pregnant(self):
        unmarried = clear_until(make_profile_state(marital="unmarried"), "fam_infant_death")
        prev_last = unmarried["last_answered_qid"]
        with_child = apply_interpreted_turn(unmarried, {"current_outcome": "not_about_active", "profile_updates": {"marital_status": "married", "child_count": 1, "children": [{"age_years": 4.0, "sex": "male", "name": None}]}})
        self.assertEqual((with_child["profile"]["marital_status"], with_child["profile"]["ever_pregnant"], with_child["active_qid"], with_child["last_answered_qid"]),
                         ("married", True, "birth_mode_detail", prev_last))
        resumed = clear_until(with_child, "fam_infant_death")
        self.assertEqual((resumed["answers"]["birth_mode_detail"]["state"], resumed["answers"]["fam_related"]["state"]), ("clear", "clear"))

        childless = apply_interpreted_turn(unmarried, {"current_outcome": "not_about_active", "profile_updates": {"marital_status": "married", "child_count": 0, "children": []}})
        self.assertEqual((childless["active_qid"], childless["profile"]["ever_pregnant"], expect_yes_no(childless)), ("ever_pregnant", None, True))
        after_yes = apply_turn(childless, "concern")
        self.assertEqual(after_yes["active_qid"], "preg_conditions")
        back = apply_turn(apply_turn(clear_until(after_yes, "fam_related"), "clear"), "clear")
        self.assertEqual(back["active_qid"], "fam_infant_death")

        no_kids_info, _ = apply_profile(unmarried, marital_status="married")
        self.assertEqual(no_kids_info["active_qid"], "intro_children")
        self.assertEqual(apply_profile(no_kids_info, child_count=0, children=[])[0]["active_qid"], "ever_pregnant")

    def test_one_utterance_all_profile_fields_vs_married_only_on_intro_married(self):
        all_kid, hold1 = apply_profile(new_cursor_state(), caller_name="पूजा", marital_status="married", child_count=1, children=[{"age_years": 4.0, "sex": "male"}])
        all_zero, hold2 = apply_profile(new_cursor_state(), caller_name="पूजा", marital_status="married", child_count=0, children=[])
        self.assertEqual((hold1, all_kid["active_qid"], hold2, all_zero["active_qid"]), (None, "intro_bridge", None, "intro_bridge"))
        self.assertEqual(apply_turn(all_zero, "clear")["active_qid"], "ever_pregnant")

        at_married, _ = apply_profile(new_cursor_state(), caller_name="पूजा")
        at_children, _ = apply_profile(at_married, marital_status="married")
        self.assertEqual((at_children["active_qid"], at_children["profile"]["children_stated"]), ("intro_children", False))

    def test_which_child_clear_with_child_ref_and_close_timeout_and_cursor_from_call(self):
        kids = [make_child(7, "male", 1), make_child(2, "female", 2)]
        state = apply_turn(clear_until(make_profile_state(children=kids), "birth_jaundice"), "concern")
        denied = apply_turn(state, "clear", child_ref="child_1")
        self.assertEqual((denied["pending_followup_idx"], denied["step_attempts"], denied["answers"]["birth_jaundice"]["child_ids"]), (-1, 1, []))
        confirmed = apply_turn(denied, "concern", child_ref="child_2")
        self.assertEqual((confirmed["answers"]["birth_jaundice"]["child_ids"], confirmed["pending_followup_idx"], confirmed["step_attempts"]), (["child_2"], 0, 0))
        self.assertEqual(apply_turn(state, "clear")["answers"]["birth_jaundice"]["state"], "clear")

        close = apply_turn(clear_until(make_profile_state(children=[make_child(4, "male")]), "close_anything_else"), "concern", note="पेट दर्द")
        round2 = apply_turn(close, "concern")
        a1 = apply_interpreted_turn(round2, {"current_outcome": "not_about_active"})
        a2 = apply_interpreted_turn(a1, {"current_outcome": "not_about_active"})
        self.assertEqual((a1["step_attempts"], a2["phase"], a2["close_reason"], a2["answers"]["close_anything_else"]["notes"]),
                         (1, "closing", "complete", "पेट दर्द"))
        self.assertEqual(call_cursor.cursor_from_call({"cursor_json": json.dumps(close)}), {})
        self.assertEqual(call_cursor.cursor_from_call({"cursor_state": close}), close)

    def test_profile_hold_intro_children_vs_midcall_children_ages_and_mid_followup_reopen(self):
        at_children, _ = apply_profile(apply_profile(new_cursor_state(), caller_name="पूजा")[0], marital_status="married")
        renamed = apply_interpreted_turn(at_children, {"current_outcome": "not_about_active", "profile_updates": {"caller_name": "कविता"}})
        self.assertEqual((renamed["hold_reason"], renamed["active_qid"], renamed["profile"]["caller_name"]), (None, "intro_children", "कविता"))

        mid = clear_until(make_profile_state(children=[make_child(4, "male")]), "birth_nicu")
        held_mid = apply_interpreted_turn(mid, {"current_outcome": "not_about_active", "profile_updates": {"child_count": 2, "children": [{"age_years": 4.0, "sex": "male"}]}})
        self.assertEqual((held_mid["hold_reason"], call_cursor.step_kind(held_mid), held_mid["profile"]["pending_child_count"]), ("children_ages", "profile", 2))
        still_held = apply_interpreted_turn(held_mid, {"current_outcome": "clear", "profile_updates": {"caller_name": "कविता"}})
        still_empty = apply_interpreted_turn(still_held, {"current_outcome": "clear", "profile_updates": {}})
        self.assertEqual((still_empty["hold_reason"], still_empty["profile"]["caller_name"]), ("children_ages", "कविता"))
        self.assertNotIn("birth_nicu", still_empty["answers"])
        resolved_mid = apply_interpreted_turn(still_empty, {"current_outcome": "not_about_active", "profile_updates": {"child_count": 2, "children": [{"age_years": 4.0, "sex": "male"}, {"age_years": 1.0, "sex": "female"}]}})
        self.assertEqual((resolved_mid["hold_reason"], resolved_mid["active_qid"]), (None, "birth_nicu"))

        # Interrupted during followup on fam_infant_death -> unmarried->married pops unfinished gate and asks both
        in_fup = apply_turn(clear_until(make_profile_state(marital="unmarried"), "fam_infant_death"), "concern")
        reopened = apply_interpreted_turn(in_fup, {"current_outcome": "not_about_active", "profile_updates": {"marital_status": "married", "child_count": 1, "children": [{"age_years": 4.0, "sex": "male"}]}})
        self.assertEqual((reopened["active_qid"], reopened["pending_followup_idx"]), ("birth_mode_detail", None))
        self.assertNotIn("fam_infant_death", reopened["answers"])
        resumed = apply_turn(apply_turn(apply_turn(clear_until(reopened, "fam_infant_death"), "concern"), "concern", note="6 महीने, मामा का बेटा"), "concern", note="बुखार")
        self.assertEqual(resumed["answers"]["fam_infant_death"]["followups"]["infant_death_cause"]["note"], "बुखार")

        # Interrupted during which_child on body_transfusion -> adding 12y daughter pops unfinished gate and asks mod_heavy_periods
        kids2 = [make_child(6, "male", 1), make_child(2, "female", 2)]
        which = apply_turn(clear_until(make_profile_state(children=kids2), "body_transfusion"), "concern")
        reopened_w = apply_interpreted_turn(which, {"current_outcome": "not_about_active", "profile_updates": {"child_count": 3, "children": [{"age_years": 6.0, "sex": "male"}, {"age_years": 2.0, "sex": "female"}, {"age_years": 12.0, "sex": "female"}]}})
        self.assertEqual((reopened_w["active_qid"], reopened_w["pending_followup_idx"]), ("mod_heavy_periods", None))
        self.assertNotIn("body_transfusion", reopened_w["answers"])
        back = apply_turn(apply_turn(apply_turn(clear_until(reopened_w, "body_transfusion"), "concern"), "concern", child_ref="child_3"), "concern", note="दो बार")
        self.assertEqual((back["answers"]["body_transfusion"]["child_ids"], back["answers"]["body_transfusion"]["followups"]["transfusion_count"]["note"]),
                         (["child_3"], "दो बार"))


if __name__ == "__main__":
    unittest.main()
