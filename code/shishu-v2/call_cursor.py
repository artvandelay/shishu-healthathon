"""Deterministic call cursor. The live model never chooses the next question.

State is JSON-serializable. pending_followup_idx is None on the question
itself, -1 while a multi-child concern still needs a child, and 0+ on a
follow-up. A sub-step that stays unresolved for MAX_STEP_ATTEMPTS is marked
unknown and the cursor moves on.
"""

from __future__ import annotations

import copy

from triage_catalog import FOLLOWUP_HI, active_specs, get_spec, spine_ids

MAX_STEP_ATTEMPTS = 2

_OUTCOMES = frozenset({
    "clear", "concern", "unknown", "clarify", "retract", "emergency", "caller_stop",
})
_MARITAL = frozenset({"married", "about_to_marry", "unmarried"})
_NOT_MARRIED = frozenset({"unmarried", "about_to_marry"})
_ANSWERED = frozenset({"clear", "concern", "unknown"})
_DEMOGRAPHIC_SKIP = frozenset({
    "unmarried", "about_to_marry", "no_children", "ever_pregnant_no",
})
_EARLY = frozenset({"intro_name", "intro_married", "intro_children"})
_NO_YES = _EARLY
_BARE_NAMES = frozenset({
    "मेरा नाम", "मेरा नाम है", "नाम", "जी मेरा नाम", "हाँ मेरा नाम", "हां मेरा नाम",
    "mera naam", "mera naam hai", "naam", "my name", "my name is",
})
_ALL_CHILDREN = frozenset({"both", "all", "दोनों", "सभी", "सबको", "तीनों", "चारों"})
_OLDEST = frozenset({"oldest", "बड़े", "बड़ा", "बड़ी", "बड़े वाले", "पहले", "पहला"})
_MIDDLE = frozenset({
    "middle", "बीच", "बीच वाले", "बीच वाला", "बीच वाली", "मंझले", "मंझला", "मंझली", "मंझल",
})
_SECOND = frozenset({"दूसरे", "दूसरा"})
_THIRD = frozenset({"तीसरे", "तीसरा"})
_YOUNGEST = frozenset({
    "youngest", "छोटे", "छोटा", "छोटी", "छोटे वाले", "आखिरी", "सबसे छोटा",
})
_MALE = frozenset({"बेटा", "लड़का", "boy"})
_FEMALE = frozenset({"बेटी", "लड़की", "girl"})
_CHILD_GROUPS = frozenset({"birth", "body", "illness"})

_EVER_PREGNANT_HI = "क्या आप कभी गर्भवती हुई हैं?"
_CLOSE_MORE_HI = "मैंने यह बात नोट कर ली। इसके अलावा कुछ और बताना चाहेंगी?"
_THAL_ABOUT = "क्या आपकी या आपके होने वाले पति की थैलेसीमिया खून जाँच में कभी माइनर या ट्रेट जैसी कोई दिक्कत बताई गई?"
_THAL_UNMARRIED = "क्या आपकी थैलेसीमिया खून जाँच में कभी माइनर या ट्रेट जैसी कोई दिक्कत बताई गई?"
_RELATED_ABOUT = "क्या आप और आपके होने वाले पति पहले से आपस में रिश्तेदार हैं, जैसे कज़िन या एक ही खानदान से?"
_HOLD_HI = {
    "name_fragment": "जी, आराम से बताइए — आपका पूरा नाम क्या है?",
    "marital_status": "जी, क्या आपकी शादी हो गई है, होने वाली है, या अभी नहीं हुई?",
    "children_ages": "जी, उन सभी बच्चों की उम्र क्या-क्या है?",
    "ever_pregnant": _EVER_PREGNANT_HI,
}


def new_cursor_state() -> dict:
    return {
        "version": 2,
        "phase": "asking",
        "close_reason": None,
        "profile": {
            "caller_name": None,
            "marital_status": None,
            "children": [],
            "ever_pregnant": None,
        },
        "active_qid": "intro_name",
        "pending_followup_idx": None,
        "step_attempts": 0,
        "hold_reason": None,
        "last_answered_qid": None,
        "answers": {},
    }


def expect_yes_no(state: dict) -> bool:
    """True only on a yes/no step: bridge, ever_pregnant, clinical gates, and close."""
    return (
        state.get("phase") == "asking"
        and state.get("pending_followup_idx") is None
        and not state.get("hold_reason")
        and state.get("active_qid") not in _NO_YES
    )


def apply_profile(
    state: dict,
    *,
    caller_name: str | None = None,
    marital_status: str | None = None,
    children: list | None = None,
    ever_pregnant: bool | None = None,
    child_count: int | None = None,
) -> tuple[dict, str | None]:
    """Update the profile on any turn and skip whatever that makes irrelevant.

    child_count > 0 with a short list or a missing age holds for children_ages
    and does not mark her childless.
    """
    state = copy.deepcopy(state)
    if caller_name is not None and not _valid_name(caller_name):
        state["hold_reason"] = "name_fragment"
        return state, "name_fragment"
    if caller_name is not None:
        state["profile"]["caller_name"] = " ".join(caller_name.split())
    if marital_status is not None:
        if marital_status not in _MARITAL:
            state["hold_reason"] = "marital_status"
            return state, "marital_status"
        state["profile"]["marital_status"] = marital_status
        if marital_status in _NOT_MARRIED and children is None and child_count is None:
            state["profile"]["children"] = []
    children_given = children is not None or child_count is not None
    if children_given:
        incoming = list(children) if children is not None else []
        stated = child_count if isinstance(child_count, int) and child_count >= 0 else len(incoming)
        if (stated > 0 and len(incoming) < stated) or _missing_ages(incoming):
            state["hold_reason"] = "children_ages"
            return state, "children_ages"
        state["profile"]["children"] = [
            _normalize_child(child, i) for i, child in enumerate(incoming, start=1)
        ]
    if state["profile"]["children"]:
        state["profile"]["ever_pregnant"] = True
    elif ever_pregnant is not None:
        state["profile"]["ever_pregnant"] = bool(ever_pregnant)
    elif marital_status in _NOT_MARRIED:
        state["profile"]["ever_pregnant"] = False

    _recompute(state)
    if _active_is_skipped(state):
        _advance(state)
        _recompute(state)
    hold = _profile_hold(state, children_given)
    if hold:
        state["hold_reason"] = hold
        return state, hold
    state["hold_reason"] = None
    if _on_profile_step(state):
        _advance(state)
        _recompute(state)
    elif state["active_qid"] == "ever_pregnant" and state["profile"]["ever_pregnant"] is not None:
        _advance(state)
        _recompute(state)
    return state, None


def apply_turn(
    state: dict,
    outcome: str,
    *,
    child_ref: str = "",
    note: str = "",
    followup_answered: bool = False,
) -> dict:
    if outcome not in _OUTCOMES:
        raise ValueError(f"unknown outcome {outcome!r}")
    state = copy.deepcopy(state)
    if outcome in ("emergency", "caller_stop"):
        state["phase"] = "closing"
        state["close_reason"] = outcome
        return state
    if state["phase"] != "asking":
        return state
    if _is_retract(state, outcome, child_ref):
        return _retract(state)
    if outcome == "clarify" and state.get("pending_followup_idx") is None:
        return state
    if state["active_qid"] == "ever_pregnant":
        return _resolve_ever(state, outcome)
    if state["active_qid"] == "close_anything_else":
        if state.get("pending_followup_idx") is None:
            return _resolve_close(state, outcome, note)
        return _resolve_close_detail(state, outcome, note)
    if state.get("pending_followup_idx") == -1:
        return _resolve_which_child(state, child_ref, note, outcome)
    if isinstance(state.get("pending_followup_idx"), int) and state["pending_followup_idx"] >= 0:
        return _resolve_followup(state, outcome, note)

    spec = get_spec(state["active_qid"])
    if spec.kind == "profile":
        return state
    if spec.kind == "bridge":
        _write_answer(state, spec.id, "clear", note, [])
        _recompute(state)
        _advance(state)
        return state
    return _resolve_main(state, spec, outcome, child_ref, note, followup_answered)


def prompt_directive(state: dict) -> str:
    """One speakable instruction: the single Hindi question to ask next."""
    if state["phase"] != "asking":
        return "Say one short warm Hindi closing, then stop. Do not ask another screening question."
    hold = state.get("hold_reason")
    if hold in _HOLD_HI:
        return _ask(_HOLD_HI[hold])
    if state["active_qid"] == "ever_pregnant":
        return _ask(_EVER_PREGNANT_HI)
    if state.get("pending_followup_idx") == -1:
        return _ask(_which_child_hi(state["profile"]["children"]))
    spec = get_spec(state["active_qid"])
    idx = state.get("pending_followup_idx")
    if isinstance(idx, int) and idx >= 0:
        return _ask(FOLLOWUP_HI[spec.followups[idx]])
    answer = state["answers"].get(state["active_qid"]) or {}
    if state["active_qid"] == "close_anything_else" and answer.get("state") == "concern":
        return _ask(_CLOSE_MORE_HI)
    return _ask(_question_hi(state, spec))


def _ask(hindi: str) -> str:
    return f"Ask only this one Hindi question, then wait: {hindi}"


def _question_hi(state: dict, spec) -> str:
    marital = state["profile"]["marital_status"]
    if spec.id == "preg_thalassemia":
        if marital == "about_to_marry":
            return _THAL_ABOUT
        if marital == "unmarried":
            return _THAL_UNMARRIED
        return spec.question_hi
    if spec.id == "fam_related" and marital == "about_to_marry":
        return _RELATED_ABOUT
    if len(state["profile"]["children"]) > 1 and spec.question_multi_hi:
        return spec.question_multi_hi
    return spec.question_hi


def _resolve_main(state, spec, outcome, child_ref, note, followup_answered) -> dict:
    child_ids = _child_ids_for(state, spec, child_ref)
    followups = None
    if outcome == "concern" and followup_answered and note.strip() and spec.followups:
        followups = {spec.followups[0]: {"state": "concern", "note": note.strip()}}
    _write_answer(state, spec.id, outcome, note, child_ids, followups)
    _recompute(state)
    if outcome != "concern":
        _advance(state)
        return state
    if _needs_which_child(state, spec, child_ref):
        state["pending_followup_idx"] = -1
        state["step_attempts"] = 0
        return state
    if spec.followups and followups is None:
        state["pending_followup_idx"] = 0
        state["step_attempts"] = 0
        return state
    if followups is not None and len(spec.followups) > 1:
        state["pending_followup_idx"] = 1
        state["step_attempts"] = 0
        return state
    _advance(state)
    return state


def _resolve_ever(state: dict, outcome: str) -> dict:
    if outcome == "concern":
        state["profile"]["ever_pregnant"] = True
        recorded = "concern"
    elif outcome in ("clear", "unknown"):
        state["profile"]["ever_pregnant"] = False
        recorded = "clear" if outcome == "clear" else "unknown"
    else:
        return state
    state["answers"]["ever_pregnant"] = _blank(recorded)
    _recompute(state)
    _advance(state)
    return state


def _is_retract(state: dict, outcome: str, child_ref: str) -> bool:
    if outcome == "retract":
        return True
    return (
        state.get("pending_followup_idx") == -1
        and outcome == "clear"
        and not (child_ref or "").strip()
    )


def _close_concern_kept(state: dict) -> bool:
    prior = state["answers"].get("close_anything_else") or {}
    return prior.get("state") == "concern" and bool(str(prior.get("notes") or "").strip())


def _retract(state: dict) -> dict:
    qid = state["active_qid"]
    if qid == "ever_pregnant":
        state["profile"]["ever_pregnant"] = False
    if qid == "close_anything_else" and _close_concern_kept(state):
        state["pending_followup_idx"] = None
        state["step_attempts"] = 0
        _recompute(state)
        _advance(state)
        return state
    state["answers"][qid] = _blank("clear")
    state["pending_followup_idx"] = None
    state["step_attempts"] = 0
    _recompute(state)
    _advance(state)
    return state


def _resolve_close(state: dict, outcome: str, note: str) -> dict:
    if outcome == "concern" and not note.strip():
        state["pending_followup_idx"] = 0
        state["step_attempts"] = 0
        return state
    if outcome == "concern":
        prior = state["answers"].get("close_anything_else") or _blank("concern")
        state["answers"]["close_anything_else"] = _blank(
            "concern",
            note=_join_notes(prior.get("notes") or "", note),
            child_ids=prior.get("child_ids") or [],
            followups=prior.get("followups") or {},
        )
        state["pending_followup_idx"] = None
        state["step_attempts"] = 0
        state["close_rounds"] = int(state.get("close_rounds") or 0) + 1
        return state
    if outcome in ("clear", "unknown"):
        prior = state["answers"].get("close_anything_else")
        if not prior or prior.get("state") != "concern":
            _write_answer(state, "close_anything_else", outcome, note, [])
        _recompute(state)
        _advance(state)
    return state


def _resolve_close_detail(state: dict, outcome: str, note: str) -> dict:
    text = note.strip()
    if text and outcome != "clarify":
        prior = state["answers"].get("close_anything_else") or _blank("concern")
        followups = dict(prior.get("followups") or {})
        followups["open_concern_detail"] = {"state": "concern", "note": text}
        state["answers"]["close_anything_else"] = _blank(
            "concern",
            note=_join_notes(prior.get("notes") or "", text),
            child_ids=prior.get("child_ids") or [],
            followups=followups,
        )
        state["pending_followup_idx"] = None
        state["step_attempts"] = 0
        state["close_rounds"] = int(state.get("close_rounds") or 0) + 1
        return state
    if outcome in ("clear", "unknown") and not text:
        if _close_concern_kept(state):
            state["pending_followup_idx"] = None
            state["step_attempts"] = 0
            _advance(state)
            return state
        _write_answer(state, "close_anything_else", outcome, "", [])
        _advance(state)
        return state
    state["step_attempts"] = int(state.get("step_attempts") or 0) + 1
    if state["step_attempts"] >= MAX_STEP_ATTEMPTS:
        answer = _blank("unknown")
        answer["followups"] = {"open_concern_detail": {"state": "unknown", "note": ""}}
        state["answers"]["close_anything_else"] = answer
        _advance(state)
    return state


def _resolve_which_child(state: dict, child_ref: str, note: str, outcome: str) -> dict:
    qid = state["active_qid"]
    spec = get_spec(qid)
    answer = state["answers"].get(qid) or _blank("concern")
    ids = _resolve_child_ids(state, child_ref)
    if ids:
        answer["child_ids"] = ids
        answer["state"] = "concern"
        if note.strip():
            answer["notes"] = _join_notes(answer.get("notes") or "", note)
        state["answers"][qid] = answer
        state["step_attempts"] = 0
        return _after_child_known(state, spec)
    if outcome == "unknown":
        answer["child_ids"] = []
        if note.strip():
            answer["notes"] = _join_notes(answer.get("notes") or "", note)
        state["answers"][qid] = answer
        state["step_attempts"] = 0
        return _after_child_known(state, spec)
    if note.strip():
        answer["notes"] = _join_notes(answer.get("notes") or "", note)
        state["answers"][qid] = answer
    state["step_attempts"] = int(state.get("step_attempts") or 0) + 1
    if state["step_attempts"] < MAX_STEP_ATTEMPTS:
        return state
    answer["child_ids"] = []
    state["answers"][qid] = answer
    state["step_attempts"] = 0
    return _after_child_known(state, spec)


def _after_child_known(state: dict, spec) -> dict:
    filled = (state["answers"].get(spec.id) or {}).get("followups") or {}
    for index, fid in enumerate(spec.followups):
        if fid not in filled:
            state["pending_followup_idx"] = index
            state["step_attempts"] = 0
            return state
    state["pending_followup_idx"] = None
    _advance(state)
    return state


def _followup_unresolved(outcome: str, note: str) -> bool:
    if outcome == "clarify":
        return True
    if outcome == "concern":
        return not note.strip()
    return outcome not in _ANSWERED and not note.strip()


def _resolve_followup(state: dict, outcome: str, note: str) -> dict:
    spec = get_spec(state["active_qid"])
    idx = state["pending_followup_idx"]
    fid = spec.followups[idx]
    if _followup_unresolved(outcome, note):
        state["step_attempts"] = int(state.get("step_attempts") or 0) + 1
        if state["step_attempts"] < MAX_STEP_ATTEMPTS:
            return state
        outcome, note = "unknown", ""
    answer = state["answers"].get(spec.id) or _blank("concern")
    followups = dict(answer.get("followups") or {})
    stored = outcome if outcome in _ANSWERED else "concern"
    followups[fid] = {"state": stored, "note": note.strip()}
    answer["followups"] = followups
    if answer.get("state") not in _ANSWERED:
        answer["state"] = "concern"
    state["answers"][spec.id] = answer
    state["step_attempts"] = 0
    if idx + 1 < len(spec.followups):
        state["pending_followup_idx"] = idx + 1
        return state
    state["pending_followup_idx"] = None
    _advance(state)
    return state


def _advance(state: dict) -> None:
    state["last_answered_qid"] = state.get("active_qid")
    state["pending_followup_idx"] = None
    state["step_attempts"] = 0
    nxt = _next_qid(state, state.get("active_qid") or "")
    if nxt is None:
        state["phase"] = "closing"
        state["close_reason"] = state.get("close_reason") or "complete"
        return
    state["active_qid"] = nxt


def _next_qid(state: dict, after_id: str) -> str | None:
    ids = spine_ids()
    if after_id == "ever_pregnant":
        start = next(
            (i for i, qid in enumerate(ids) if "ever_pregnant_no" in get_spec(qid).skip_if),
            len(ids),
        )
        rest = ids[start:]
        ask_ever = False
    elif after_id in ids:
        rest = ids[ids.index(after_id) + 1:]
        ask_ever = True
    else:
        rest = ids
        ask_ever = True
    for qid in rest:
        spec = get_spec(qid)
        if ask_ever and _needs_ever_pregnant(state) and "ever_pregnant_no" in spec.skip_if:
            return "ever_pregnant"
        if _is_skipped(spec, state):
            prior = state["answers"].get(qid)
            if not prior or prior.get("state") not in _ANSWERED:
                state["answers"][qid] = _blank("not_applicable")
            continue
        return qid
    return None


def _needs_ever_pregnant(state: dict) -> bool:
    profile = state["profile"]
    if profile["ever_pregnant"] is not None or profile["children"]:
        return False
    return profile["marital_status"] == "married"


def _active_is_skipped(state: dict) -> bool:
    qid = state.get("active_qid")
    if not qid or qid == "ever_pregnant":
        return False
    try:
        spec = get_spec(qid)
    except KeyError:
        return False
    return _is_skipped(spec, state)


def _on_profile_step(state: dict) -> bool:
    qid = state["active_qid"]
    return qid in _EARLY and get_spec(qid).kind == "profile"


def _profile_hold(state: dict, children_given: bool) -> str | None:
    qid = state["active_qid"]
    profile = state["profile"]
    if qid == "intro_name" and not _valid_name(profile["caller_name"]):
        return "name_fragment"
    if qid == "intro_married" and profile["marital_status"] not in _MARITAL:
        return "marital_status"
    if qid == "intro_children" and not profile["children"] and not children_given:
        return "children_ages"
    if qid == "ever_pregnant" and profile["ever_pregnant"] is None:
        return "ever_pregnant"
    return None


def _valid_name(name: str | None) -> bool:
    cleaned = " ".join((name or "").strip().casefold().split())
    return bool(cleaned) and cleaned not in _BARE_NAMES


def _missing_ages(children: list) -> bool:
    return any(child.get("age_years") is None for child in children)


def _normalize_child(child: dict, index: int) -> dict:
    age = float(child["age_years"])
    sex = child.get("sex") or "unknown"
    if sex not in ("male", "female", "unknown"):
        sex = "unknown"
    return {
        "id": child.get("id") or f"child_{index}",
        "age_years": age,
        "age_label_hi": child.get("age_label_hi") or _age_label(age),
        "sex": sex,
        "name": child.get("name"),
    }


def _age_label(age: float) -> str:
    if age < 1:
        return f"{int(round(age * 12))} महीने"
    text = str(int(age)) if age == int(age) else str(age)
    return f"{text} साल"


def _ordered_children(children: list) -> list:
    return sorted(children, key=lambda child: float(child["age_years"]), reverse=True)


def _which_child_hi(children: list) -> str:
    ordered = _ordered_children(children)
    middle = len(ordered) // 2 if len(ordered) == 3 else None
    parts = []
    for index, child in enumerate(ordered):
        label = child.get("age_label_hi") or _age_label(float(child["age_years"]))
        name = (child.get("name") or "").strip()
        sex = child.get("sex")
        if name:
            phrase = f"{label} वाले {name}"
        elif sex == "male":
            phrase = f"{label} वाले बेटा"
        elif sex == "female":
            phrase = f"{label} वाले बेटी"
        else:
            phrase = f"{label} वाले"
        if index == middle:
            phrase += " (बीच वाले)"
        parts.append(phrase)
    if len(parts) >= 2:
        listed = ", ".join(parts[:-1]) + ", या " + parts[-1]
    else:
        listed = parts[0] if parts else ""
    return f"यह किस बच्चे की बात है — {listed}?"


def children_roster(state: dict) -> list[dict]:
    """Oldest-first ids for the live model. Empty when she has fewer than two children."""
    children = (state.get("profile") or {}).get("children") or []
    if len(children) <= 1:
        return []
    ordered = _ordered_children(children)
    last = len(ordered) - 1
    roster = []
    for index, child in enumerate(ordered):
        if index == 0:
            rank = "oldest"
        elif index == last:
            rank = "youngest"
        else:
            rank = "middle"
        roster.append({
            "id": child["id"],
            "rank": rank,
            "age_label_hi": child["age_label_hi"],
            "sex": child["sex"],
            "name": child["name"],
        })
    return roster


def _resolve_child_ids(state: dict, child_ref: str) -> list[str]:
    """Exact child ids only. The whole reference must match one closed-world token."""
    ref = " ".join((child_ref or "").strip().split())
    if not ref:
        return []
    children = state["profile"]["children"]
    ordered = _ordered_children(children)
    token = ref.casefold()
    if token in _ALL_CHILDREN:
        return [child["id"] for child in ordered]
    by_id = {child["id"]: child for child in children}
    if ref in by_id:
        return [ref]
    named = [
        child["id"]
        for child in children
        if (child.get("name") or "").strip()
        and (child.get("name") or "").strip().casefold() == token
    ]
    if len(named) == 1:
        return named
    if not ordered:
        return []
    if token in _OLDEST:
        return [ordered[0]["id"]]
    if token in _MIDDLE:
        return [ordered[len(ordered) // 2]["id"]]
    if token in _SECOND and len(ordered) >= 2:
        return [ordered[1]["id"]]
    if token in _THIRD and len(ordered) >= 3:
        return [ordered[2]["id"]]
    if token in _YOUNGEST:
        return [ordered[-1]["id"]]
    males = [child for child in children if child.get("sex") == "male"]
    females = [child for child in children if child.get("sex") == "female"]
    if token in _MALE and len(males) == 1:
        return [males[0]["id"]]
    if token in _FEMALE and len(females) == 1:
        return [females[0]["id"]]
    return []


def _children_resolved(state: dict) -> bool:
    profile = state["profile"]
    if profile["marital_status"] in _NOT_MARRIED or profile["children"]:
        return True
    return state["active_qid"] not in _EARLY


def _is_skipped(spec, state: dict) -> bool:
    return bool(_skip_hits(spec, state))


def _skip_hits(spec, state: dict) -> tuple[str, ...]:
    profile = state["profile"]
    children = profile["children"]
    marital = profile["marital_status"]
    ever = profile["ever_pregnant"]
    resolved = _children_resolved(state)
    hits: list[str] = []
    for rule in spec.skip_if:
        if rule == "no_children" and resolved and not children:
            hits.append(rule)
        elif rule == "unmarried" and marital == "unmarried":
            hits.append(rule)
        elif rule == "about_to_marry" and marital == "about_to_marry":
            hits.append(rule)
        elif rule == "ever_pregnant_no" and (
            ever is False or (marital in _NOT_MARRIED and ever is not True)
        ):
            hits.append(rule)
        elif rule == "no_child_5_plus" and resolved and not any(float(c["age_years"]) >= 5 for c in children):
            hits.append(rule)
        elif rule == "all_under_10" and resolved and not any(float(c["age_years"]) >= 10 for c in children):
            hits.append(rule)
        elif rule == "no_daughter_11_plus" and resolved and not any(
            c.get("sex") == "female" and float(c["age_years"]) >= 11 for c in children
        ):
            hits.append(rule)
        elif rule == "no_family_concern" and not _has_family_concern(state):
            hits.append(rule)
    return tuple(hits)


def _has_family_concern(state: dict) -> bool:
    for qid, answer in state["answers"].items():
        if qid == "fam_same_problem" or not str(qid).startswith("fam_"):
            continue
        if isinstance(answer, dict) and answer.get("state") == "concern":
            return True
    return False


def _blank(status: str, note: str = "", child_ids: list | None = None, followups: dict | None = None) -> dict:
    return {
        "state": status,
        "child_ids": list(child_ids or []),
        "notes": note or "",
        "followups": dict(followups or {}),
    }


def _recompute(state: dict) -> None:
    for spec in active_specs():
        answer = state["answers"].get(spec.id)
        hits = _skip_hits(spec, state)
        if hits:
            demographic = any(hit in _DEMOGRAPHIC_SKIP for hit in hits)
            if answer and answer.get("state") in _ANSWERED and not demographic:
                continue
            state["answers"][spec.id] = _blank("not_applicable")
        elif answer and answer.get("state") == "not_applicable":
            del state["answers"][spec.id]


def _write_answer(state, qid, outcome, note, child_ids, followups=None) -> None:
    prior = state["answers"].get(qid) or {}
    keep = prior.get("state") in _ANSWERED
    chosen = followups
    if chosen is None:
        chosen = prior.get("followups") if keep else {}
    state["answers"][qid] = _blank(
        outcome,
        note=note or (prior.get("notes") if keep else "") or "",
        child_ids=list(child_ids or (prior.get("child_ids") if keep else []) or []),
        followups=dict(chosen or {}),
    )


def _is_child_scoped(spec) -> bool:
    return spec.group in _CHILD_GROUPS and "no_children" in spec.skip_if


def _needs_which_child(state: dict, spec, child_ref: str) -> bool:
    return (
        _is_child_scoped(spec)
        and len(state["profile"]["children"]) > 1
        and not _resolve_child_ids(state, child_ref)
    )


def _child_ids_for(state: dict, spec, child_ref: str) -> list[str]:
    if not _is_child_scoped(spec):
        return []
    found = _resolve_child_ids(state, child_ref)
    if found:
        return found
    children = state["profile"]["children"]
    if len(children) == 1 and not (child_ref or "").strip():
        return [children[0]["id"]]
    return []


def _join_notes(existing: str, extra: str) -> str:
    old = " ".join((existing or "").split())
    new = " ".join((extra or "").split())
    if old and new:
        return f"{old}; {new}"
    return new or old
