"""Text-only screening scenarios.

Drives call_state the way the live tool path does: ask the server's next
question, begin it, then apply_updates with the mother's Hindi utterance.
No microphone and no network. A scenario is data (who she is, and any
positive answers). Everything else is नहीं / नॉर्मल / ठीक है.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import call_state
import checklist
import mother_report
import question_flow
import summarizer


_GARBAGE = re.compile(
    r"ओपेरतिओन|child_\d+|(?:[\u0900-\u097F][A-Za-z])|(?:[A-Za-z][\u0900-\u097F])"
)
_NORMAL_FRAMED = {
    "body_growth": ("हाँ, बाकी बच्चों जैसा बढ़ रहा है।", True),
    "birth_hearing_vision": ("हाँ, जाँच हुई और नॉर्मल थी।", True),
    "preg_thalassemia": ("हाँ, दोनों की जाँच हुई और रिपोर्ट नॉर्मल थी।", True),
    "birth_mode": ("नॉर्मल डिलीवरी हुई, अस्पताल में।", "normal"),
    "body_puberty": ("हाँ, दूसरे बच्चों के साथ ही शुरू हुआ।", True),
}


@dataclass
class Scenario:
    name: str
    group: str
    caller_name: str
    marital_text: str
    children_text: str
    children_value: object
    ever_pregnant: bool | None = None
    positives: dict = field(default_factory=dict)
    after_no_findings: dict = field(default_factory=dict)
    emergency_at: str | None = None
    stop_at: str | None = None
    stop_after_clinical: int | None = None
    clarify_topic: str | None = None
    unknown_followups: tuple[str, ...] = ()
    unknown_topics: tuple[str, ...] = ()
    neighbor_topic: str | None = None
    contradict_children: object | None = None
    expect_close: str = "complete"
    expect_yes: tuple[str, ...] = ()
    no_utterance: str = "नहीं।"
    no_value: object = False
    close_text: str | None = None


def _update(field_name: str, value, state: str = "answered", details: dict | None = None, child_ids=None):
    return {
        "field": field_name,
        "state": state,
        "value": value,
        "scope": "specific" if child_ids else "none",
        "child_ids": list(child_ids or []),
        "details": dict(details or {}),
    }


def _reply(scenario: Scenario, question: dict, memory: dict) -> tuple[str, list[dict] | None, str | None]:
    """Return utterance, tool updates, and an optional close reason."""
    tid = question.get("topic_id")
    fid = question.get("followup_id")
    clinical = bool(tid) and not str(tid).startswith(("intro_", "profile.")) and tid != "close_anything_else"

    if scenario.emergency_at == "after_name" and tid == "intro_married" and not fid:
        return "बच्चे को साँस नहीं आ रही।", None, "emergency"
    if scenario.emergency_at == "first_clinical" and clinical and not fid and memory["clinical"] == 0:
        return "बच्चे को साँस नहीं आ रही।", None, "emergency"
    if scenario.stop_at == "during_intro" and tid == "intro_married" and not fid:
        return "अब कॉल बंद कर दो।", None, "caller_stop"
    if (
        scenario.stop_after_clinical is not None
        and clinical
        and not fid
        and memory["clinical"] >= scenario.stop_after_clinical
    ):
        return "नहीं, कॉल एंड करिए प्लीज।", None, "caller_stop"

    if tid == "intro_name":
        return f"मेरा नाम {scenario.caller_name} है।", [_update("profile.caller_name", scenario.caller_name)], None
    if tid == "intro_married":
        return scenario.marital_text, [_update("profile.marital_status", scenario.marital_text)], None
    if tid == "intro_children":
        return scenario.children_text, [_update("profile.children", scenario.children_value)], None
    if tid == "intro_bridge":
        return "ठीक है।", [_update("topic.intro_bridge", True)], None
    if tid == "profile.ever_pregnant":
        pregnant = scenario.ever_pregnant
        if pregnant is None:
            pregnant = bool(scenario.positives) and any(
                key.startswith("preg_") or key == "fam_miscarriage" for key in scenario.positives
            )
            if scenario.children_value not in (0, [], None, False):
                pregnant = True
        if pregnant:
            return "हाँ, पहले गर्भ रहा था।", [_update("profile.ever_pregnant", True)], None
        return "नहीं, कभी प्रेग्नेंट नहीं हुई।", [_update("profile.ever_pregnant", False)], None
    if tid == "close_anything_else":
        if scenario.close_text:
            return scenario.close_text, [_update("topic.close_anything_else", scenario.close_text)], None
        return "नहीं, बस इतना ही।", [_update("topic.close_anything_else", False)], None

    positive = scenario.positives.get(tid) or {}
    if fid and scenario.clarify_topic == tid and not memory["clarified"]:
        memory["clarified"] = True
        parent = positive.get("value", True)
        return "क्या मतलब?", [_update(f"topic.{tid}", parent, details={fid: "क्या मतलब?"})], None
    if fid and tid in scenario.unknown_followups:
        parent = positive.get("value", True)
        return "मुझे पता नहीं है।", [_update(f"topic.{tid}", parent)], None
    if fid and fid in ((scenario.after_no_findings or {}).get(tid) or {}):
        utterance, detail = scenario.after_no_findings[tid][fid]
        return utterance, [_update(f"topic.{tid}", False, details={fid: detail})], None
    if fid and fid in (positive.get("followups") or {}):
        utterance, detail = positive["followups"][fid]
        return utterance, [_update(f"topic.{tid}", positive.get("value", True), details={fid: detail})], None
    if fid:
        if positive:
            return "पता नहीं।", [_update(f"topic.{tid}", positive.get("value", True))], None
        return scenario.no_utterance, [_update(f"topic.{tid}", scenario.no_value)], None

    if tid in scenario.unknown_topics:
        return "पता नहीं।", [_update(f"topic.{tid}", None, state="unknown")], None
    if scenario.neighbor_topic == tid:
        text = "ये पड़ोसी के बच्चे को है, मेरे बच्चे को नहीं।"
        updates = [_update(f"topic.{tid}", False)]
        if scenario.contradict_children is not None and not memory["contradicted"]:
            memory["contradicted"] = True
            updates.insert(0, _update("profile.children", scenario.contradict_children))
        return text, updates, None
    if positive:
        updates = [_update(
            f"topic.{tid}",
            positive.get("value", True),
            details=positive.get("details") or {},
        )]
        if scenario.contradict_children is not None and not memory["contradicted"]:
            memory["contradicted"] = True
            updates.insert(0, _update("profile.children", scenario.contradict_children))
        return positive.get("utterance") or "हाँ।", updates, None

    if scenario.contradict_children is not None and clinical and not memory["contradicted"]:
        memory["contradicted"] = True
        uttered, value = _NORMAL_FRAMED.get(tid, (scenario.no_utterance, scenario.no_value))
        return uttered, [
            _update("profile.children", scenario.contradict_children),
            _update(f"topic.{tid}", value),
        ], None

    uttered, value = _NORMAL_FRAMED.get(tid, (scenario.no_utterance, scenario.no_value))
    return uttered, [_update(f"topic.{tid}", value)], None


def run_scenario(scenario: Scenario, max_turns: int = 80) -> dict:
    state = call_state.new_call_state()
    question = call_state.next_question(state)
    if question:
        state = call_state.begin_question(state, question)
    memory = {"clinical": 0, "clarified": False, "contradicted": False}
    said: list[str] = []
    errors: list[str] = []
    seen: dict[tuple, int] = {}
    while question and question.get("question_kind") != "closing" and len(said) < max_turns:
        key = (question.get("topic_id"), question.get("followup_id"))
        seen[key] = seen.get(key, 0) + 1
        if seen[key] > 4:
            errors.append(f"stuck on {key[0]}/{key[1]}")
            break
        utterance, updates, close_reason = _reply(scenario, question, memory)
        said.append(utterance)
        clarify_key = None
        if question.get("followup_id") and call_state.is_clarification(utterance):
            clarify_key = (question.get("topic_id"), question.get("followup_id"))
        if question.get("topic_id") and not str(question.get("topic_id")).startswith(("intro_", "profile.")):
            if not question.get("followup_id"):
                memory["clinical"] += 1
        if close_reason == "emergency" and not question_flow.is_emergency_text(utterance):
            errors.append("emergency phrase was not detected")
        if close_reason:
            state = call_state.request_close(state, close_reason)
            question = call_state.next_question(state)
            break
        try:
            state = call_state.apply_updates(
                state,
                updates or [],
                len(said),
                utterance=utterance,
                prior_utterances=said[:-1],
            )
            if clarify_key:
                nxt = call_state.next_question(state)
                if (
                    (nxt or {}).get("topic_id") != clarify_key[0]
                    or (nxt or {}).get("followup_id") != clarify_key[1]
                ):
                    errors.append("clarification closed the follow-up")
        except ValueError as exc:
            errors.append(str(exc))
            break
        question = call_state.next_question(state)
        if question is None and call_state.is_complete(state):
            state = call_state.request_close(state, "complete")
            question = call_state.next_question(state)
        elif question and question.get("question_kind") != "closing":
            state = call_state.begin_question(state, question)

    fallback = summarizer.build_state_fallback(state)
    call = {
        "started_at": "2026-11-08 14:10:00",
        "duration_sec": 180,
        "state": state,
        "summary_mother": fallback["summary_mother"],
        "transcript": [{"role": "user", "text": line} for line in said],
    }
    report = mother_report.build(call)
    issues = _issues(scenario, state, errors, report, fallback, said)
    return {
        "name": scenario.name,
        "group": scenario.group,
        "close_reason": state.get("close_reason"),
        "profile": state.get("profile"),
        "issues": issues,
        "errors": errors,
        "turns": len(said),
        "yes": report.get("yes") or [],
        "questions": (report.get("questions") or [])[:6],
        "pass": not issues,
    }


def _issues(scenario, state, errors, report, fallback, said) -> list[str]:
    issues = list(errors)
    if state.get("close_reason") != scenario.expect_close:
        issues.append(f"close {state.get('close_reason')} != {scenario.expect_close}")
    topics = state.get("topics") or {}
    if scenario.expect_close == "complete":
        if topics.get("intro_name", {}).get("state") != "answered":
            issues.append("intro name not answered")
        if topics.get("intro_married", {}).get("state") != "answered":
            issues.append("intro married not answered")
        if topics.get("intro_children", {}).get("state") != "answered":
            issues.append("intro children not answered")
        open_ids = call_state.unresolved_topic_ids(state)
        if open_ids:
            issues.append("unresolved " + ",".join(open_ids[:8]))
    for topic in checklist.TOPICS:
        tid = topic["id"]
        saved = topics.get(tid) or {}
        if saved.get("state") == "not_applicable" and not call_state._skip_reason(state, tid):
            issues.append(f"false not_applicable {tid}")
        if tid.startswith("intro_"):
            continue
        kind = call_state.topic_answer_kind(saved, tid)
        if scenario.expect_close == "complete" and not scenario.positives:
            if kind == "yes" and tid not in scenario.expect_yes:
                issues.append(f"unexpected yes {tid}")
            pending = call_state._pending_followups(state, tid) if tid in topics else []
            if pending and kind == "no":
                issues.append(f"followup open on no {tid}:{pending[0]['id']}")
            if tid == "body_bleeding_bruise" and scenario.expect_close == "complete":
                ages = [
                    (call_state._child_age_years(child.get("age")), call_state._is_female(child))
                    for child in ((state.get("profile") or {}).get("children") or [])
                    if isinstance(child, dict)
                ]
                asked = set((saved.get("details") or {}))
                if "periods" in asked and not any(age is not None and age >= 11 and female for age, female in ages):
                    issues.append("periods asked without a daughter of 11 or older")
                if "teeth" in asked and not any(age is not None and age >= 5 for age, _female in ages):
                    issues.append("tooth-fall bleeding asked without a child of 5 or older")
            if kind == "no" and tid not in scenario.unknown_followups:
                details = saved.get("details") or {}
                unknown_fields = [
                    key for key, val in details.items()
                    if isinstance(val, dict) and val.get("state") == "unknown"
                ]
                if unknown_fields:
                    issues.append(f"unknown on a no {tid}:{','.join(unknown_fields)}")
    yes_topics = {
        topic["id"]
        for topic in checklist.TOPICS
        if not topic["id"].startswith("intro_")
        and call_state.topic_answer_kind(topics.get(topic["id"]) or {}, topic["id"]) == "yes"
    }
    missing = [tid for tid in scenario.expect_yes if tid not in yes_topics]
    if missing:
        issues.append("missing yes " + ",".join(missing))
    extra = sorted(yes_topics - set(scenario.expect_yes))
    if extra and scenario.expect_close == "complete":
        issues.append("extra yes " + ",".join(extra))
    for tid, spec in (scenario.positives or {}).items():
        said_child = (spec.get("details") or {}).get("affected_child")
        if not said_child:
            continue
        expected = call_state._find_child_id(state, said_child)
        bound = [
            item.get("person_id")
            for item in ((topics.get(tid) or {}).get("instances") or [])
            if isinstance(item, dict)
        ]
        children = (state.get("profile") or {}).get("children") or []
        if expected and len(children) > 1 and expected not in bound:
            issues.append(f"{tid} did not bind {said_child} to {expected}")
        elif (
            expected
            and len(children) > 1
            and not call_state._means_every_child(str(said_child))
            and any(person_id != expected for person_id in bound)
        ):
            issues.append(f"{tid} bound {said_child} to more than {expected}")
        for fid, pair in (spec.get("followups") or {}).items():
            utterance = pair[0] if isinstance(pair, (tuple, list)) and pair else ""
            days = call_state._spoken_day_count(utterance)
            if days is None:
                continue
            stored = ((topics.get(tid) or {}).get("details") or {}).get("days")
            if stored != days:
                issues.append(f"{tid} days {stored!r} != {days}")
    for tid in scenario.unknown_followups:
        details = (topics.get(tid) or {}).get("details") or {}
        unknown_fields = [
            key for key, val in details.items()
            if isinstance(val, dict) and val.get("state") == "unknown"
        ]
        if not unknown_fields:
            issues.append(f"{tid} follow-up was not stored as unknown")
        if (topics.get(tid) or {}).get("state") in {"not_asked", "not_applicable"}:
            issues.append(f"{tid} follow-up unknown collapsed the topic")
    for tid in scenario.unknown_topics:
        kind = call_state.topic_answer_kind(topics.get(tid) or {}, tid)
        if kind != "unknown":
            issues.append(f"{tid} kind {kind} != unknown")
        question = checklist.question_hi(tid)
        if question in _flat(report.get("unasked")):
            issues.append(f"{tid} rendered as not asked")
        if question not in (report.get("unknown") or []):
            issues.append(f"{tid} missing from unknown")
    blob = " ".join(
        [report.get("who") or "", " ".join(report.get("yes") or []), " ".join(report.get("questions") or [])]
    )
    if _GARBAGE.search(blob):
        issues.append("garbage text " + _GARBAGE.search(blob).group(0))
    if scenario.expect_close == "complete" and not scenario.expect_yes and report.get("yes"):
        issues.append("bogus haan " + " | ".join(report["yes"])[:240])
    for line in report.get("yes") or []:
        if line[:1].isspace():
            issues.append("yes line has a leading space")
        if "child_" in line:
            issues.append("raw child id in yes")
        if "[" in line or "]" in line:
            issues.append("python list in yes line")
    if scenario.expect_close == "complete" and isinstance(scenario.children_value, int):
        children = (state.get("profile") or {}).get("children") or []
        records = call_state._spoken_age_records(scenario.children_text)
        if len(records) == scenario.children_value and [c.get("age") for c in children] != records:
            issues.append("stored ages do not match the ages she said")
        sexes = call_state._sexes_for_count(scenario.children_text, scenario.children_value)
        if sexes and [c.get("sex") for c in children] != sexes:
            issues.append("stored sex does not match the sex words she said")
        if len(children) == 1:
            only = children[0]
            who_line = report.get("who") or ""
            child_name = str(only.get("name") or "").strip()
            if child_name and child_name not in who_line:
                issues.append("who line dropped the child's name")
            elif not child_name and only.get("sex") in {"female", "male"}:
                word = "बेटी" if only["sex"] == "female" else "बेटा"
                if word not in who_line:
                    issues.append("who line dropped the child's sex")
        who = report.get("who") or ""
        for child in children:
            shown = mother_report.hindi_text(mother_report._format_age(child.get("age")))
            if shown and shown not in who:
                issues.append(f"who line dropped age {shown}")
    if scenario.expect_close == "complete":
        ages = [
            call_state._child_age_years(c.get("age"))
            for c in ((state.get("profile") or {}).get("children") or [])
        ]
        puberty = (topics.get("body_puberty") or {}).get("state")
        if ages and all(age is not None and age < 10 for age in ages) and puberty != "not_applicable":
            issues.append(f"puberty was {puberty} though every child is under 10")
        if any(age is not None and age >= 10 for age in ages) and puberty == "not_applicable":
            issues.append("puberty skipped though a child is 10 or older")
    questions = " ".join(report.get("questions") or [])
    if scenario.expect_close == "complete" and not any(
        isinstance(child, dict) for child in ((state.get("profile") or {}).get("children") or [])
    ):
        if any(phrase in questions for phrase in ("मेरे बच्चे", "बच्चे के सभी", "बच्चे के रोज़")):
            issues.append("asked about a child she does not have")
    named = {
        "ऑपरेशन": "birth_mode",
        "समय से पहले": "birth_term",
        "खून चढ़ाना": "body_transfusion",
        "गर्भ गिरा": "fam_miscarriage",
        "झटके": "body_fits_milestones",
        "नाल गिरने पर": "birth_cord_bleed",
        "होंठ या उँगलियाँ": "body_blue",
        "थैलेसीमिया": "preg_thalassemia",
    }
    for phrase, tid in named.items():
        if tid in scenario.expect_yes and phrase not in questions:
            issues.append(f"mother questions omit {tid}")
    if "fam_infant_death" not in scenario.expect_yes and "गुजर गया" in questions:
        issues.append("infant-death question without that finding")
    if not ({"ill_admissions", "ill_iv_antibiotics"} & set(scenario.expect_yes)) and "भर्ती होना पड़ा" in questions:
        issues.append("admission question without that finding")
    if "birth_nicu" not in scenario.expect_yes and "पीलिया" in questions:
        issues.append("jaundice question without that finding")
    return issues


def _flat(groups) -> list[str]:
    found = []
    for group in groups or []:
        found.extend(group.get("items") or [])
    return found


def _child(age, sex=None, name=None):
    return {"age": age, "sex": sex, "name": name}


def campaign() -> list[Scenario]:
    married = "शादी हो गई है।"
    no_child = "कोई बच्चा नहीं है।"
    return [
        Scenario(
            name="all_no_one_boy",
            group="baseline",
            caller_name="कविता",
            marital_text=married,
            children_text="एक बच्चा है, दो साल का लड़का।",
            children_value=1,
            expect_yes=(),
        ),
        Scenario(
            name="no_flags_unmarried_never",
            group="no_flags",
            caller_name="लता",
            marital_text="शादी नहीं हुई।",
            children_text=no_child,
            children_value=[],
            ever_pregnant=False,
        ),
        Scenario(
            name="no_flags_months_and_years",
            group="no_flags",
            caller_name="सीमा",
            marital_text=married,
            children_text="दो बच्चे हैं। बड़ी बेटी छह महीने की है और छोटा बेटा दो साल का है।",
            children_value=2,
        ),
        Scenario(
            name="no_flags_half_year_and_girl_11",
            group="no_flags",
            caller_name="मीना",
            marital_text=married,
            children_text="तीन बच्चे हैं। एक आधा साल का है, एक चार साल का है, और बेटी ग्यारह साल की है।",
            children_value=3,
        ),
        Scenario(
            name="no_flags_half_steps",
            group="no_flags",
            caller_name="रमा",
            marital_text=married,
            children_text="दो बच्चे हैं। बड़ी बेटी डेढ़ साल की है और छोटा बेटा साढ़े तीन साल का है।",
            children_value=2,
        ),
        Scenario(
            name="no_flags_married_none",
            group="no_flags",
            caller_name="उषा",
            marital_text=married,
            children_text=no_child,
            children_value="कोई बच्चा नहीं है",
            ever_pregnant=False,
        ),
        Scenario(
            name="no_flags_under_ten",
            group="no_flags",
            caller_name="गीता",
            marital_text=married,
            children_text="दो बच्चे हैं। एक चार साल का लड़का है और एक आठ साल का लड़का है।",
            children_value=2,
        ),
        Scenario(
            name="no_flags_unmarried_past_pregnancy",
            group="no_flags",
            caller_name="रानी",
            marital_text="शादी नहीं हुई।",
            children_text=no_child,
            children_value=[],
            ever_pregnant=True,
            positives={
                "fam_miscarriage": {
                    "value": True,
                    "utterance": "एक बार गर्भ तीन महीने में गिरा था।",
                    "details": {"count": 1, "month": 3},
                },
            },
            expect_yes=("fam_miscarriage",),
        ),
        Scenario(
            name="no_flags_boy_11",
            group="no_flags",
            caller_name="माया",
            marital_text=married,
            children_text="एक बच्चा है, ग्यारह साल का लड़का।",
            children_value=1,
        ),
        Scenario(
            name="no_flags_twins",
            group="no_flags",
            caller_name="शालिनी",
            marital_text=married,
            children_text="जुड़वाँ हैं, दोनों एक साल के।",
            children_value="जुड़वाँ हैं, दोनों एक साल के।",
        ),
        Scenario(
            name="no_flags_one_month",
            group="no_flags",
            caller_name="पल्लवी",
            marital_text=married,
            children_text="एक महीना का बेटा है।",
            children_value=1,
        ),
        Scenario(
            name="no_flags_years_and_months",
            group="no_flags",
            caller_name="नम्रता",
            marital_text=married,
            children_text="दो साल छह महीने का बेटा है।",
            children_value=2,
        ),
        Scenario(
            name="no_flags_paune_year",
            group="no_flags",
            caller_name="वंदना",
            marital_text=married,
            children_text="पौने साल का बेटा है।",
            children_value=1,
        ),
        Scenario(
            name="flags_eighth_month",
            group="red_flags",
            caller_name="सरिता",
            marital_text=married,
            children_text="एक बेटी है, एक साल की।",
            children_value=1,
            positives={
                "birth_term": {
                    "value": False,
                    "utterance": "आठवें महीने में पैदा हुई।",
                    "details": {},
                },
            },
            expect_yes=("birth_term",),
        ),
        Scenario(
            name="flags_half_day_nicu",
            group="red_flags",
            caller_name="रेखा",
            marital_text=married,
            children_text="एक बेटा है, दो महीने का।",
            children_value=1,
            positives={
                "birth_nicu": {
                    "value": True,
                    "utterance": "पीलिया की वजह से डेढ़ दिन रखा।",
                    "details": {},
                },
            },
            expect_yes=("birth_nicu",),
        ),
        Scenario(
            name="no_flags_ages_and_only_before_the_last",
            group="no_flags",
            caller_name="रेखा",
            marital_text=married,
            children_text="तीन बच्चे हैं दो चार और छह साल के।",
            children_value=3,
        ),
        Scenario(
            name="degenerate_nothing_else_sent_as_yes",
            group="degenerate",
            caller_name="पूजा",
            marital_text=married,
            children_text="एक बेटी है, डेढ़ साल की।",
            children_value=1,
            no_utterance="नहीं, बस इतना ही।",
            no_value=True,
        ),
        Scenario(
            name="no_flags_scan_found_nothing",
            group="no_flags",
            caller_name="अंजना",
            marital_text=married,
            children_text="एक बेटी है, दो साल की।",
            children_value=1,
            positives={
                "preg_scan": {
                    "value": True,
                    "utterance": "स्कैन हुआ था, कुछ नहीं निकला।",
                    "details": {},
                },
            },
        ),
        Scenario(
            name="flags_scan_hole",
            group="red_flags",
            caller_name="भारती",
            marital_text=married,
            children_text="एक बेटा है, तीन साल का।",
            children_value=1,
            positives={
                "preg_scan": {
                    "value": True,
                    "utterance": "स्कैन में दिल में छेद बताया।",
                    "details": {},
                },
            },
            expect_yes=("preg_scan",),
        ),
        Scenario(
            name="degenerate_scan_sent_as_yes",
            group="degenerate",
            caller_name="चंदा",
            marital_text=married,
            children_text="एक बेटी है, एक साल की।",
            children_value=1,
            positives={
                "preg_scan": {
                    "value": True,
                    "utterance": "स्कैन में कुछ नहीं दिखा।",
                    "details": {},
                },
            },
        ),
        Scenario(
            name="no_flags_telling_the_names",
            group="no_flags",
            caller_name="स्वाति",
            marital_text=married,
            children_text="राम दो साल का और सीता चार साल की, नाम बता रही हूँ।",
            children_value=2,
        ),
        Scenario(
            name="flags_named_with_naam",
            group="red_flags",
            caller_name="दीपा",
            marital_text=married,
            children_text="नाम राम है दो साल का, और सीता चार साल की।",
            children_value=2,
            positives={
                "body_fits_milestones": {
                    "value": True,
                    "utterance": "राम को झटके आते हैं।",
                    "details": {},
                },
            },
            expect_yes=("body_fits_milestones",),
        ),
        Scenario(
            name="degenerate_telling_age_sent_as_count",
            group="degenerate",
            caller_name="माला",
            marital_text=married,
            children_text="राम दो साल का और सीता चार साल की, नाम बता रही हूँ।",
            children_value=4,
        ),
        Scenario(
            name="no_flags_twins_boy_and_girl",
            group="no_flags",
            caller_name="नेहा",
            marital_text=married,
            children_text="जुड़वाँ हैं, एक लड़का एक लड़की, दोनों एक साल के।",
            children_value=2,
        ),
        Scenario(
            name="flags_her_son_not_the_inlaws",
            group="red_flags",
            caller_name="पूनम",
            marital_text=married,
            children_text="मेरी सास के दो बच्चे हैं, मेरे एक बेटा है तीन साल का।",
            children_value=2,
            positives={
                "body_fits_milestones": {
                    "value": True,
                    "utterance": "बेटे को झटके आते हैं।",
                    "details": {},
                },
            },
            expect_yes=("body_fits_milestones",),
        ),
        Scenario(
            name="degenerate_inlaw_count",
            group="degenerate",
            caller_name="अर्चना",
            marital_text=married,
            children_text="मेरी सास के दो बच्चे हैं, मेरे एक बेटा है तीन साल का।",
            children_value=2,
        ),
        Scenario(
            name="no_flags_one_of_each_age",
            group="no_flags",
            caller_name="जया",
            marital_text=married,
            children_text="तीन साल का एक और एक साल की एक।",
            children_value=2,
        ),
        Scenario(
            name="flags_one_of_each_age",
            group="red_flags",
            caller_name="हेमलता",
            marital_text=married,
            children_text="तीन साल का एक और एक साल की एक।",
            children_value=2,
            positives={
                "body_blue": {
                    "value": True,
                    "utterance": "बड़े वाले के होंठ नीले पड़ते हैं।",
                    "details": {},
                },
            },
            expect_yes=("body_blue",),
        ),
        Scenario(
            name="degenerate_one_sent_as_three",
            group="degenerate",
            caller_name="कुसुम",
            marital_text=married,
            children_text="तीन साल का एक और एक साल की एक।",
            children_value=3,
        ),
        Scenario(
            name="no_flags_sons_without_the_word_year",
            group="no_flags",
            caller_name="सविता",
            marital_text=married,
            children_text="दो बेटे हैं तीन और पाँच के।",
            children_value=2,
        ),
        Scenario(
            name="flags_two_each",
            group="red_flags",
            caller_name="अनु",
            marital_text=married,
            children_text="एक लड़का चार साल का और दो लड़कियाँ दो-दो साल की।",
            children_value=3,
            positives={
                "body_fits_milestones": {
                    "value": True,
                    "utterance": "लड़के को झटके आते हैं।",
                    "details": {},
                },
            },
            expect_yes=("body_fits_milestones",),
        ),
        Scenario(
            name="degenerate_boy_age_sent_as_count",
            group="degenerate",
            caller_name="रजनी",
            marital_text=married,
            children_text="एक लड़का चार साल का और दो लड़कियाँ दो-दो साल की।",
            children_value=4,
        ),
        Scenario(
            name="no_flags_son_and_daughter_one_age",
            group="no_flags",
            caller_name="मीना",
            marital_text=married,
            children_text="तीन साल का बेटा और बेटी हैं।",
            children_value=2,
        ),
        Scenario(
            name="flags_son_and_daughter_one_age",
            group="red_flags",
            caller_name="गीता",
            marital_text=married,
            children_text="तीन साल का बेटा और बेटी हैं।",
            children_value=2,
            positives={
                "ill_odor": {
                    "value": True,
                    "utterance": "पेशाब में बदबू है।",
                    "details": {},
                },
            },
            expect_yes=("ill_odor",),
        ),
        Scenario(
            name="degenerate_shared_age_sent_as_count",
            group="degenerate",
            caller_name="रीना",
            marital_text=married,
            children_text="तीन साल का बेटा और बेटी हैं।",
            children_value=3,
        ),
        Scenario(
            name="no_flags_daughter_not_the_nephew",
            group="no_flags",
            caller_name="ललिता",
            marital_text=married,
            children_text="मेरा भांजा दो साल का है, मेरी बेटी चार साल की है।",
            children_value=1,
        ),
        Scenario(
            name="flags_daughter_not_the_nephew",
            group="red_flags",
            caller_name="कमल",
            marital_text=married,
            children_text="मेरा भांजा दो साल का है, मेरी बेटी चार साल की है।",
            children_value=1,
            positives={
                "body_fits_milestones": {
                    "value": True,
                    "utterance": "बेटी को झटके आते हैं।",
                    "details": {},
                },
            },
            expect_yes=("body_fits_milestones",),
        ),
        Scenario(
            name="degenerate_nephew_age_sent_as_count",
            group="degenerate",
            caller_name="हरप्रीत",
            marital_text=married,
            children_text="मेरा भांजा दो साल का है, मेरी बेटी चार साल की है।",
            children_value=4,
        ),
        Scenario(
            name="no_flags_reassurance_after_no",
            group="no_flags",
            caller_name="पल्लवी",
            marital_text=married,
            children_text="एक बेटा है, दो साल का।",
            children_value=1,
            no_utterance="नहीं होता, सब ठीक है।",
            no_value=True,
        ),
        Scenario(
            name="flags_transfusion_once",
            group="red_flags",
            caller_name="श्रीजा",
            marital_text=married,
            children_text="एक बेटी है, चार साल की।",
            children_value=1,
            positives={
                "body_transfusion": {
                    "value": True,
                    "utterance": "एक बार खून चढ़ा, दूसरी बार नहीं।",
                    "details": {},
                },
            },
            expect_yes=("body_transfusion",),
        ),
        Scenario(
            name="degenerate_neighbor_still_no",
            group="degenerate",
            caller_name="निधि",
            marital_text=married,
            children_text="एक बेटा है, तीन साल का।",
            children_value=1,
            neighbor_topic="body_fits_milestones",
        ),
        Scenario(
            name="no_flags_ages_unit_on_the_first",
            group="no_flags",
            caller_name="वंदना",
            marital_text=married,
            children_text="तीन बच्चे हैं, दो साल, चार और छह।",
            children_value=3,
        ),
        Scenario(
            name="flags_nicu_days_unknown",
            group="red_flags",
            caller_name="मंजू",
            marital_text=married,
            children_text="एक बेटा है, एक साल का।",
            children_value=1,
            positives={
                "birth_nicu": {
                    "value": True,
                    "utterance": "NICU में रहा, कितने दिन याद नहीं।",
                    "details": {},
                },
            },
            expect_yes=("birth_nicu",),
        ),
        Scenario(
            name="degenerate_cough_after_nothing",
            group="degenerate",
            caller_name="कीर्ति",
            marital_text=married,
            children_text="एक बेटी है, तीन साल की।",
            children_value=1,
            close_text="कुछ नहीं। खाँसी है।",
            expect_yes=("close_anything_else",),
        ),
        Scenario(
            name="no_flags_eight_months",
            group="no_flags",
            caller_name="भावना",
            marital_text=married,
            children_text="एक बेटा है, आठ महीने का।",
            children_value=1,
        ),
        Scenario(
            name="flags_maybe_fits",
            group="red_flags",
            caller_name="रश्मि",
            marital_text=married,
            children_text="एक बेटी है, दो साल की।",
            children_value=1,
            positives={
                "body_fits_milestones": {
                    "value": True,
                    "utterance": "शायद झटके होते हैं।",
                    "details": {},
                },
            },
            expect_yes=("body_fits_milestones",),
        ),
        Scenario(
            name="degenerate_only_maybe",
            group="degenerate",
            caller_name="सुनीता",
            marital_text=married,
            children_text="एक बेटा है, एक साल का।",
            children_value=1,
            no_utterance="हो सकता है।",
            no_value=True,
        ),
        Scenario(
            name="no_flags_both_daughters_four",
            group="no_flags",
            caller_name="नमिता",
            marital_text=married,
            children_text="दोनों बेटियाँ चार साल की हैं।",
            children_value=2,
        ),
        Scenario(
            name="flags_was_five_now_seven",
            group="red_flags",
            caller_name="ज्योति",
            marital_text=married,
            children_text="पाँच साल का था, अब सात साल का है।",
            children_value=5,
            positives={
                "body_fits_milestones": {
                    "value": True,
                    "utterance": "झटके आते हैं।",
                    "details": {},
                },
            },
            expect_yes=("body_fits_milestones",),
        ),
        Scenario(
            name="degenerate_past_age_sent_as_count",
            group="degenerate",
            caller_name="करुणा",
            marital_text=married,
            children_text="पाँच साल का था, अब सात साल का है।",
            children_value=5,
        ),
        Scenario(
            name="no_flags_son_and_daughter_same_age",
            group="no_flags",
            caller_name="अंजलि",
            marital_text=married,
            children_text="बेटा और बेटी दोनों चार साल के।",
            children_value=2,
        ),
        Scenario(
            name="flags_fit_at_four_now_six",
            group="red_flags",
            caller_name="रचना",
            marital_text=married,
            children_text="चार साल का था जब झटके हुए, अब छह साल का है।",
            children_value=6,
            positives={
                "body_fits_milestones": {
                    "value": True,
                    "utterance": "झटके आते हैं।",
                    "details": {},
                },
            },
            expect_yes=("body_fits_milestones",),
        ),
        Scenario(
            name="degenerate_husband_age_sent_as_count",
            group="degenerate",
            caller_name="शालिनी",
            marital_text=married,
            children_text="मेरे पति की उम्र तीस साल है और बेटा एक साल का है।",
            children_value=30,
        ),
        Scenario(
            name="no_flags_sentence_break",
            group="no_flags",
            caller_name="लता",
            marital_text=married,
            children_text="एक बेटा है। दो साल का है।",
            children_value=1,
        ),
        Scenario(
            name="flags_pain_after_a_sentence",
            group="red_flags",
            caller_name="सरिता",
            marital_text=married,
            children_text="एक बेटी है। तीन साल की।",
            children_value=1,
            close_text="और कुछ नहीं। पेट में दर्द रहता है।",
            expect_yes=("close_anything_else",),
        ),
        Scenario(
            name="degenerate_closer_after_a_sentence",
            group="degenerate",
            caller_name="रमा",
            marital_text=married,
            children_text="एक बेटी है। चार साल की।",
            children_value=1,
            no_utterance="नहीं। बस इतना ही।",
            no_value=True,
            close_text="नहीं। बस इतना ही।",
        ),
        Scenario(
            name="flags_pain_after_nothing_else",
            group="red_flags",
            caller_name="नीलिमा",
            marital_text=married,
            children_text="एक बेटी है, चार साल की।",
            children_value=1,
            close_text="और कुछ नहीं, पेट में दर्द रहता है।",
            expect_yes=("close_anything_else",),
        ),
        Scenario(
            name="no_flags_year_and_weeks",
            group="no_flags",
            caller_name="मीनाक्षी",
            marital_text=married,
            children_text="एक साल दो हफ्ते का बेटा है।",
            children_value=1,
        ),
        Scenario(
            name="flags_fits_now_fine",
            group="red_flags",
            caller_name="कविता",
            marital_text=married,
            children_text="एक बेटा है, तीन साल का।",
            children_value=1,
            positives={
                "body_fits_milestones": {
                    "value": True,
                    "utterance": "झटके आते हैं, अब ठीक है।",
                    "details": {},
                },
            },
            expect_yes=("body_fits_milestones",),
        ),
        Scenario(
            name="degenerate_only_theek",
            group="degenerate",
            caller_name="सुषमा",
            marital_text=married,
            children_text="एक बेटी है, दो साल की।",
            children_value=1,
            no_utterance="सब ठीक है।",
            no_value=True,
        ),
        Scenario(
            name="no_flags_months_and_days",
            group="no_flags",
            caller_name="स्नेहा",
            marital_text=married,
            children_text="तीन महीने दस दिन का बेटा है।",
            children_value=3,
        ),
        Scenario(
            name="no_flags_seventeen_days",
            group="no_flags",
            caller_name="भावना",
            marital_text=married,
            children_text="दो हफ्ते तीन दिन का बेटा है।",
            children_value=2,
        ),
        Scenario(
            name="no_flags_four_together",
            group="no_flags",
            caller_name="लता",
            marital_text=married,
            children_text="चारों बच्चे हैं, दो, चार, छह और आठ साल के।",
            children_value=4,
        ),
        Scenario(
            name="flags_tenth_month",
            group="red_flags",
            caller_name="कुसुम",
            marital_text=married,
            children_text="एक बेटा है, दो साल का।",
            children_value=1,
            positives={
                "birth_term": {
                    "value": False,
                    "utterance": "दसवें महीने में पैदा हुआ।",
                    "details": {},
                },
            },
            expect_yes=("birth_term",),
        ),
        Scenario(
            name="flags_paune_child_fits",
            group="red_flags",
            caller_name="रचना",
            marital_text=married,
            children_text="एक पौने साल का और एक तीन साल का।",
            children_value=2,
            positives={
                "body_fits_milestones": {
                    "value": True,
                    "utterance": "पौने साल वाले को झटके आते हैं।",
                    "details": {"affected_child": "पौने साल वाले"},
                },
            },
            expect_yes=("body_fits_milestones",),
        ),
        Scenario(
            name="flags_bhaiya_fits",
            group="red_flags",
            caller_name="मनीषा",
            marital_text=married,
            children_text="एक बेटा है, पाँच साल का।",
            children_value=1,
            positives={
                "fam_fits_delay": {
                    "value": True,
                    "utterance": "भैया को बचपन से झटके हैं।",
                    "details": {},
                },
            },
            expect_yes=("fam_fits_delay",),
        ),
        Scenario(
            name="flags_second_pregnancy_bleed",
            group="red_flags",
            caller_name="अंजलि",
            marital_text=married,
            children_text="एक बेटी है, चार साल की।",
            children_value=1,
            positives={
                "preg_delivery": {
                    "value": True,
                    "utterance": "दूसरी प्रेग्नेंसी में प्रसव में बहुत खून बह गया।",
                    "details": {},
                },
            },
            expect_yes=("preg_delivery",),
        ),
        Scenario(
            name="no_flags_six_months_sent_as_six",
            group="no_flags",
            caller_name="सीमा",
            marital_text=married,
            children_text="छह महीने का बेटा है।",
            children_value=6,
        ),
        Scenario(
            name="no_flags_two_daughters_one_age",
            group="no_flags",
            caller_name="किरण",
            marital_text=married,
            children_text="तीन साल की दो बेटियाँ हैं।",
            children_value=2,
        ),
        Scenario(
            name="no_flags_two_sons",
            group="no_flags",
            caller_name="प्रभा",
            marital_text=married,
            children_text="आठ और छह साल के दो बेटे हैं।",
            children_value=2,
        ),
        Scenario(
            name="no_flags_age_before_the_count",
            group="no_flags",
            caller_name="मंजू",
            marital_text=married,
            children_text="आठ साल का एक बेटा है।",
            children_value="आठ साल का एक बेटा है।",
        ),
        Scenario(
            name="no_flags_home_normal",
            group="no_flags",
            caller_name="गीता",
            marital_text=married,
            children_text="एक बेटा है, दो साल का।",
            children_value=1,
            positives={
                "birth_mode": {
                    "value": "normal",
                    "utterance": "घर पर नॉर्मल हुई।",
                    "details": {"delivery_type": "normal", "place": "home"},
                },
            },
        ),
        Scenario(
            name="flags_operation_not_a_household_word",
            group="red_flags",
            caller_name="पूनम",
            marital_text=married,
            children_text="एक बेटी है, तीन साल की।",
            children_value=1,
            positives={
                "birth_mode": {
                    "value": True,
                    "utterance": "घरेलू नुस्खा नहीं, अस्पताल में ऑपरेशन हुआ, बच्चे की हालत की वजह से।",
                    "details": {
                        "delivery_type": "operation",
                        "place": "hospital",
                        "operation_reason": "बच्चे की हालत",
                    },
                    "followups": {
                        "operation_reason": ("बच्चे की हालत की वजह से।", "बच्चे की हालत"),
                        "place": ("अस्पताल में।", "hospital"),
                    },
                },
            },
            expect_yes=("birth_mode",),
        ),
        Scenario(
            name="no_flags_sex_then_age",
            group="no_flags",
            caller_name="ज्योति",
            marital_text=married,
            children_text="लड़का छह महीने का और लड़की दो साल की।",
            children_value="लड़का छह महीने का और लड़की दो साल की।",
        ),
        Scenario(
            name="flags_oldest_not_everyone",
            group="red_flags",
            caller_name="वंदना",
            marital_text=married,
            children_text="दो बच्चे हैं। बड़ा छह साल का लड़का है और छोटी दो साल की बेटी है।",
            children_value=2,
            positives={
                "body_fits_milestones": {
                    "value": True,
                    "utterance": "सबसे बड़े वाले को झटके आते हैं, एक साल से।",
                    "details": {"type": "fits", "onset": "एक साल", "affected_child": "सबसे बड़े वाले"},
                    "followups": {
                        "type": ("झटके आते हैं।", "fits"),
                        "onset": ("एक साल से।", "एक साल"),
                        "affected_child": ("सबसे बड़े वाले को।", "सबसे बड़े वाले"),
                    },
                },
            },
            expect_yes=("body_fits_milestones",),
        ),
        Scenario(
            name="flags_counted_sex_groups",
            group="red_flags",
            caller_name="नम्रता",
            marital_text=married,
            children_text="एक लड़का और दो लड़कियाँ, उम्र पाँच, तीन और एक साल।",
            children_value="एक लड़का और दो लड़कियाँ, उम्र पाँच, तीन और एक साल।",
            positives={
                "body_fits_milestones": {
                    "value": True,
                    "utterance": "बड़े वाले को झटके आते हैं, एक साल से।",
                    "details": {"type": "fits", "onset": "एक साल", "affected_child": "बड़े वाले"},
                    "followups": {
                        "type": ("झटके आते हैं।", "fits"),
                        "onset": ("एक साल से।", "एक साल"),
                        "affected_child": ("बड़े वाले को।", "बड़े वाले"),
                    },
                },
            },
            expect_yes=("body_fits_milestones",),
        ),
        Scenario(
            name="degenerate_not_even_one_child",
            group="degenerate",
            caller_name="शशि",
            marital_text=married,
            children_text="एक भी बच्चा नहीं है।",
            children_value="एक भी बच्चा नहीं है।",
            ever_pregnant=False,
        ),
        Scenario(
            name="no_flags_named_boy",
            group="no_flags",
            caller_name="ललिता",
            marital_text=married,
            children_text="एक बेटा है, नाम आरव है, दो साल का।",
            children_value=1,
        ),
        Scenario(
            name="no_flags_shaadi_nahi_hui_roman",
            group="no_flags",
            caller_name="रेखा",
            marital_text="shaadi nahi hui",
            children_text="कोई बच्चा नहीं है।",
            children_value="कोई बच्चा नहीं है",
            ever_pregnant=False,
        ),
        Scenario(
            name="no_flags_shaadi_hui",
            group="no_flags",
            caller_name="कमलिनी",
            marital_text="शादी हुई है।",
            children_text="कोई बच्चा नहीं है।",
            children_value="कोई बच्चा नहीं है",
            ever_pregnant=False,
        ),
        Scenario(
            name="no_flags_about_four_years",
            group="no_flags",
            caller_name="यमुना",
            marital_text="शादी है।",
            children_text="लगभग चार साल का है।",
            children_value="लगभग चार साल का है।",
        ),
        Scenario(
            name="no_flags_shaadi_hai",
            group="no_flags",
            caller_name="सरोज",
            marital_text="शादी है।",
            children_text="कोई बच्चा नहीं है।",
            children_value="कोई बच्चा नहीं है",
            ever_pregnant=False,
        ),
        Scenario(
            name="no_flags_shaadi_nahi_hui",
            group="no_flags",
            caller_name="गौरी",
            marital_text="शादी नहीं हुई।",
            children_text="कोई बच्चा नहीं है।",
            children_value="कोई बच्चा नहीं है",
            ever_pregnant=False,
        ),
        Scenario(
            name="no_flags_shared_year_list",
            group="no_flags",
            caller_name="मालती",
            marital_text=married,
            children_text="तीन बच्चे हैं, आठ, छह और चार साल के।",
            children_value=3,
        ),
        Scenario(
            name="no_flags_two_weeks",
            group="no_flags",
            caller_name="कुसुम",
            marital_text=married,
            children_text="एक बेटा है, दो हफ़्ते का है।",
            children_value=1,
        ),
        Scenario(
            name="no_flags_fifteen_days",
            group="no_flags",
            caller_name="शांति",
            marital_text=married,
            children_text="एक बेटी है, पंद्रह दिन की है।",
            children_value=1,
        ),
        Scenario(
            name="flags_fourth_child",
            group="red_flags",
            caller_name="सुषमा",
            marital_text=married,
            children_text="चार बच्चे हैं, आठ, छह, चार और दो साल के।",
            children_value=4,
            positives={
                "body_fits_milestones": {
                    "value": True,
                    "utterance": "चौथे वाले को झटके आते हैं।",
                    "details": {"affected_child": "चौथे वाले"},
                    "followups": {
                        "affected_child": ("चौथे वाले को।", "चौथे वाले"),
                    },
                },
            },
            expect_yes=("body_fits_milestones",),
        ),
        Scenario(
            name="flags_pregnancy_bp_and_sugar",
            group="red_flags",
            caller_name="ज्योति",
            marital_text=married,
            children_text="एक बेटी है, दो साल की।",
            children_value=1,
            positives={
                "preg_conditions": {
                    "value": True,
                    "utterance": "बीपी और शुगर था, डिलीवरी के बाद भी रहा।",
                },
            },
            expect_yes=("preg_conditions",),
        ),
        Scenario(
            name="flags_nicu_fifteen_days",
            group="red_flags",
            caller_name="धारी",
            marital_text=married,
            children_text="एक बेटा है, दो साल का।",
            children_value=1,
            positives={
                "birth_nicu": {
                    "value": True,
                    "utterance": "पीलिया था।",
                    "details": {"reason": "jaundice"},
                    "followups": {
                        "reason": ("पीलिया था।", "jaundice"),
                        "days": ("पंद्रह दिन रहा।", "पंद्रह दिन रहा।"),
                    },
                },
            },
            expect_yes=("birth_nicu",),
        ),
        Scenario(
            name="no_flags_sava_saal",
            group="no_flags",
            caller_name="राधा",
            marital_text=married,
            children_text="एक बेटी है, सवा साल की।",
            children_value=1,
        ),
        Scenario(
            name="no_flags_theek_hai",
            group="no_flags",
            caller_name="सरोज",
            marital_text=married,
            children_text="एक बेटा है, तीन साल का।",
            children_value=1,
            no_utterance="ठीक है।",
            no_value="ठीक है",
        ),
        Scenario(
            name="flags_post_term",
            group="red_flags",
            caller_name="कमला",
            marital_text=married,
            children_text="एक बेटी है, दो साल की।",
            children_value=1,
            positives={
                "birth_term": {
                    "value": True,
                    "utterance": "दस महीने में हुआ, देर से।",
                    "details": {"affected_child": "बेटी"},
                    "followups": {
                        "amount": ("दस महीने।", "दस महीने"),
                        "affected_child": ("बेटी का।", "बेटी"),
                    },
                },
            },
            expect_yes=("birth_term",),
        ),
        Scenario(
            name="flags_nicu_days_in_an_unknown_phrase",
            group="red_flags",
            caller_name="विमला",
            marital_text=married,
            children_text="एक बेटा है, एक साल का।",
            children_value=1,
            positives={
                "birth_nicu": {
                    "value": True,
                    "utterance": "पीलिया था।",
                    "details": {"reason": "jaundice"},
                    "followups": {
                        "reason": ("पीलिया था।", "jaundice"),
                        "days": ("दो दिन पता नहीं।", "दो दिन पता नहीं।"),
                    },
                },
            },
            expect_yes=("birth_nicu",),
        ),
        Scenario(
            name="no_flags_full_term_said_as_yes",
            group="no_flags",
            caller_name="बबिता",
            marital_text=married,
            children_text="एक बच्चा है, एक साल की बेटी।",
            children_value=1,
            positives={
                "birth_term": {
                    "value": True,
                    "utterance": "पूरे नौ महीने में हुआ, समय पर।",
                },
            },
            expect_yes=(),
        ),
        Scenario(
            name="flags_periods_after_a_no",
            group="red_flags",
            caller_name="किरण",
            marital_text=married,
            children_text="एक बच्चा है, ग्यारह साल की बेटी।",
            children_value=1,
            after_no_findings={
                "body_bleeding_bruise": {
                    "periods": ("माहवारी में बहुत ज़्यादा खून आता है।", "बहुत ज़्यादा"),
                },
            },
            expect_yes=("body_bleeding_bruise",),
        ),
        Scenario(
            name="flags_kinship_as_the_value",
            group="red_flags",
            caller_name="लक्ष्मी",
            marital_text=married,
            children_text="एक बच्चा है, दो साल की बेटी।",
            children_value=1,
            positives={
                "fam_related": {
                    "value": "कज़िन",
                    "utterance": "हाँ, शादी से पहले कज़िन थे।",
                },
            },
            expect_yes=("fam_related",),
        ),
        Scenario(
            name="flags_middle_child",
            group="red_flags",
            caller_name="शीला",
            marital_text=married,
            children_text="तीन बच्चे हैं। बड़ा छह साल का लड़का है, बीच वाली चार साल की बेटी है, और छोटा एक साल का लड़का है।",
            children_value=[
                _child({"value": 6, "unit": "years"}, "male"),
                _child({"value": 4, "unit": "years"}, "female"),
                _child({"value": 1, "unit": "years"}, "male"),
            ],
            positives={
                "body_fits_milestones": {
                    "value": True,
                    "utterance": "दूसरे वाले को झटके आते हैं, एक साल की उम्र से।",
                    "details": {"type": "fits", "onset": "एक साल", "affected_child": "दूसरे वाले"},
                    "followups": {
                        "onset": ("एक साल की उम्र से।", "एक साल"),
                        "affected_child": ("दूसरे वाले को।", "दूसरे वाले"),
                    },
                },
            },
            expect_yes=("body_fits_milestones",),
        ),
        Scenario(
            name="flags_cord_blue_thalassemia",
            group="red_flags",
            caller_name="अनिता",
            marital_text=married,
            children_text="एक बच्चा है, चार साल की बेटी।",
            children_value=1,
            positives={
                "birth_cord_bleed": {
                    "value": True,
                    "utterance": "नाल गिरने पर खून ज़्यादा हुआ था।",
                    "details": {"amount": "ज़्यादा"},
                },
                "body_blue": {
                    "value": True,
                    "utterance": "बेटी के होंठ और उँगलियाँ नीली हो जाती हैं।",
                    "details": {"location": "होंठ"},
                },
                "preg_thalassemia": {
                    "value": True,
                    "utterance": "जाँच हुई और रिपोर्ट असामान्य थी।",
                    "details": {"result": "abnormal", "caller_test": True},
                },
            },
            expect_yes=("birth_cord_bleed", "body_blue", "preg_thalassemia"),
        ),
        Scenario(
            name="flags_periods_girl_11",
            group="red_flags",
            caller_name="सरला",
            marital_text=married,
            children_text="एक बच्चा है, ग्यारह साल की बेटी।",
            children_value=1,
            positives={
                "body_bleeding_bruise": {
                    "value": True,
                    "utterance": "हाँ, चोट पर खून देर तक बहता है।",
                    "details": {"bleeding_or_bruises": "bleeding"},
                    "followups": {
                        "teeth": ("नहीं, दाँत गिरने पर नहीं।", False),
                        "periods": ("हाँ, माहवारी में बहुत ज़्यादा खून आता है।", "heavy"),
                    },
                },
            },
            expect_yes=("body_bleeding_bruise",),
        ),
        Scenario(
            name="flags_phrase_yes_and_no",
            group="red_flags",
            caller_name="कमल",
            marital_text=married,
            children_text="एक बच्चा है, तीन साल का लड़का।",
            children_value=1,
            no_utterance="नहीं जी।",
            no_value="नहीं जी",
            positives={
                "birth_nicu": {
                    "value": "जी हाँ",
                    "utterance": "जी हाँ, पीलिया की वजह से दो दिन रखा।",
                    "details": {"reason": "jaundice", "days": 2},
                    "followups": {
                        "reason": ("पीलिया की वजह से।", "jaundice"),
                        "days": ("दो दिन।", 2),
                    },
                },
            },
            expect_yes=("birth_nicu",),
        ),
        Scenario(
            name="flags_operation_nicu",
            group="red_flags",
            caller_name="सुनीता",
            marital_text=married,
            children_text="दो बच्चे हैं। बड़ा छह साल का लड़का है और छोटी दो साल की बेटी है।",
            children_value=[
                _child({"value": 6, "unit": "years"}, "male"),
                _child({"value": 2, "unit": "years"}, "female"),
            ],
            positives={
                "birth_mode": {
                    "value": True,
                    "utterance": "बड़े वाले का ऑपरेशन हुआ, अस्पताल में, बच्चे की हालत की वजह से।",
                    "details": {
                        "delivery_type": "operation",
                        "place": "hospital",
                        "operation_reason": "बच्चे की हालत",
                        "affected_child": "बड़े वाले",
                    },
                    "followups": {
                        "operation_reason": ("बच्चे की हालत की वजह से।", "बच्चे की हालत"),
                        "place": ("अस्पताल में।", "hospital"),
                        "affected_child": ("बड़े वाले का।", "बड़े वाले"),
                    },
                },
                "birth_nicu": {
                    "value": True,
                    "utterance": "बड़े वाले को पीलिया की वजह से चार दिन काँच के बक्से में रखा।",
                    "details": {"reason": "jaundice", "days": 4, "affected_child": "बड़े वाले"},
                    "followups": {
                        "reason": ("पीलिया की वजह से।", "jaundice"),
                        "days": ("चार दिन।", 4),
                        "affected_child": ("बड़े वाले को।", "बड़े वाले"),
                    },
                },
            },
            expect_yes=("birth_mode", "birth_nicu"),
        ),
        Scenario(
            name="flags_fits_cousin_miscarriage",
            group="red_flags",
            caller_name="रीना",
            marital_text=married,
            children_text="एक बच्चा है, पहले वाला पाँच साल का है।",
            children_value=[_child({"value": 5, "unit": "years"}, "male", "आरव")],
            positives={
                "body_fits_milestones": {
                    "value": True,
                    "utterance": "पहले वाले को झटके आते हैं, एक साल से।",
                    "details": {"type": "fits", "onset": "एक साल", "affected_child": "पहले वाला"},
                    "followups": {
                        "type": ("झटके आते हैं।", "fits"),
                        "onset": ("एक साल से।", "एक साल"),
                        "affected_child": ("पहले वाले को।", "पहले वाला"),
                    },
                },
                "fam_related": {
                    "value": True,
                    "utterance": "हाँ, शादी से पहले हम कज़िन थे।",
                    "details": {"relationship": "कज़िन"},
                    "followups": {"relationship": ("हम कज़िन थे।", "कज़िन")},
                },
                "fam_miscarriage": {
                    "value": True,
                    "utterance": "हाँ, तीन महीने में एक बार गर्भ गिर गया।",
                    "details": {"count": 1, "month": "तीन महीने"},
                    "followups": {
                        "count": ("एक बार।", 1),
                        "month": ("तीन महीने में।", "तीन महीने"),
                    },
                },
            },
            expect_yes=("body_fits_milestones", "fam_related", "fam_miscarriage"),
        ),
        Scenario(
            name="flags_preterm_blood_death",
            group="red_flags",
            caller_name="पूजा",
            marital_text=married,
            children_text="एक बेटी है, तीन साल की।",
            children_value=[_child({"value": 3, "unit": "years"}, "female")],
            positives={
                "birth_term": {
                    "value": "early",
                    "utterance": "बेटी सात महीने में पैदा हुई।",
                    "details": {"amount": "सात महीने", "affected_child": "बेटी"},
                    "followups": {
                        "amount": ("सात महीने में।", "सात महीने"),
                        "affected_child": ("बेटी का।", "बेटी"),
                    },
                },
                "body_transfusion": {
                    "value": True,
                    "utterance": "बेटी को दो बार खून चढ़ाना पड़ा।",
                    "details": {"count": 2, "affected_child": "बेटी"},
                    "followups": {
                        "count": ("दो बार।", 2),
                        "last_date": ("पिछले साल।", "पिछले साल"),
                        "affected_child": ("बेटी को।", "बेटी"),
                    },
                },
                "fam_infant_death": {
                    "value": True,
                    "utterance": "मामा का बच्चा चार महीने का था, झटकों में गुजर गया।",
                    "details": {"relationship": "मामा का बच्चा", "age_at_death": "चार महीने", "cause": "fits"},
                    "followups": {
                        "relationship": ("मामा का बच्चा।", "मामा का बच्चा"),
                        "age_at_death": ("चार महीने।", "चार महीने"),
                        "cause": ("झटकों में।", "fits"),
                    },
                },
            },
            expect_yes=("birth_term", "body_transfusion", "fam_infant_death"),
        ),
        Scenario(
            name="emergency_after_name",
            group="degenerate",
            caller_name="अंजलि",
            marital_text=married,
            children_text="एक बच्चा है।",
            children_value=1,
            emergency_at="after_name",
            expect_close="emergency",
        ),
        Scenario(
            name="emergency_first_clinical",
            group="degenerate",
            caller_name="कमला",
            marital_text=married,
            children_text="एक बच्चा है, तीन साल का।",
            children_value=1,
            emergency_at="first_clinical",
            expect_close="emergency",
        ),
        Scenario(
            name="stop_mid_call",
            group="degenerate",
            caller_name="गीता",
            marital_text=married,
            children_text="एक बच्चा है, दो साल का।",
            children_value=1,
            stop_after_clinical=2,
            expect_close="caller_stop",
        ),
        Scenario(
            name="clarify_followup",
            group="degenerate",
            caller_name="शांति",
            marital_text=married,
            children_text="एक बच्चा है, चार साल का।",
            children_value=[_child({"value": 4, "unit": "years"}, "male")],
            positives={
                "birth_nicu": {
                    "value": True,
                    "utterance": "हाँ, काँच के बक्से में रखा था।",
                    "details": {"reason": "jaundice"},
                    "followups": {
                        "reason": ("पीलिया था।", "jaundice"),
                        "days": ("तीन दिन।", 3),
                    },
                }
            },
            clarify_topic="birth_nicu",
            expect_yes=("birth_nicu",),
            expect_close="complete",
        ),
        Scenario(
            name="unknown_followup_and_topic",
            group="degenerate",
            caller_name="राधा",
            marital_text=married,
            children_text="एक बच्चा है, एक साल का।",
            children_value=[_child({"value": 1, "unit": "years"}, "female")],
            positives={
                "fam_related": {
                    "value": True,
                    "utterance": "हाँ, शादी रिश्तेदारी में हुई थी।",
                    "details": {},
                    "followups": {},
                }
            },
            unknown_followups=("fam_related",),
            unknown_topics=("birth_cleft",),
            expect_yes=("fam_related",),
        ),
        Scenario(
            name="neighbor_and_contradict",
            group="degenerate",
            caller_name="निशा",
            marital_text=married,
            children_text="दो बच्चे हैं, छह साल और दो साल।",
            children_value=2,
            neighbor_topic="body_fits_milestones",
            contradict_children=1,
            expect_close="complete",
        ),
        Scenario(
            name="stop_during_intro",
            group="degenerate",
            caller_name="हेमा",
            marital_text=married,
            children_text=no_child,
            children_value=[],
            stop_at="during_intro",
            expect_close="caller_stop",
        ),
        Scenario(
            name="flags_closing_complaint",
            group="red_flags",
            caller_name="उषा",
            marital_text=married,
            children_text="एक बेटा है, पाँच साल का।",
            children_value=[_child({"value": 5, "unit": "years"}, "male")],
            close_text="पेट में दर्द रहता है।",
            expect_yes=("close_anything_else",),
        ),
        Scenario(
            name="flags_lost_skill",
            group="red_flags",
            caller_name="मीरा",
            marital_text=married,
            children_text="एक बेटा है, छह साल का।",
            children_value=[_child({"value": 6, "unit": "years"}, "male")],
            positives={
                "body_regression": {
                    "value": "अब नहीं कर पाता।",
                    "utterance": "पहले चलता था, अब नहीं कर पाता।",
                },
            },
            expect_yes=("body_regression",),
        ),
        Scenario(
            name="flags_described_without_haan",
            group="red_flags",
            caller_name="लता",
            marital_text=married,
            children_text="एक बेटी है, चार साल की।",
            children_value=[_child({"value": 4, "unit": "years"}, "female")],
            positives={
                "body_fits_milestones": {
                    "value": "झटके आते हैं।",
                    "utterance": "झटके आते हैं।",
                },
                "body_blue": {
                    "value": "होंठ नीले पड़ते हैं।",
                    "utterance": "होंठ नीले पड़ते हैं।",
                },
                "fam_miscarriage": {
                    "value": "गर्भ गिर गया।",
                    "utterance": "तीन महीने में गर्भ गिर गया।",
                },
            },
            expect_yes=("body_fits_milestones", "body_blue", "fam_miscarriage"),
        ),
        Scenario(
            name="about_to_marry_past_pregnancy",
            group="no_flags",
            caller_name="नेहा",
            marital_text="शादी होने वाली है।",
            children_text=no_child,
            children_value=[],
            ever_pregnant=True,
        ),
    ]


def main() -> int:
    rows = [run_scenario(item) for item in campaign()]
    for row in rows:
        brief = {
            "name": row["name"],
            "group": row["group"],
            "pass": row["pass"],
            "close": row["close_reason"],
            "turns": row["turns"],
            "issues": row["issues"],
            "children": (row["profile"] or {}).get("children"),
            "yes": row["yes"][:4],
        }
        print(json.dumps(brief, ensure_ascii=False), flush=True)
    failed = [row["name"] for row in rows if not row["pass"]]
    print(f"FAILED {len(failed)}/{len(rows)}: {', '.join(failed)}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
