"""Deterministic call cursor. The live model never chooses the next question.

State is JSON-serializable. pending_followup_idx is None on the question
itself, -1 while a multi-child concern still needs a child, and 0+ on a
follow-up. A sub-step that stays unresolved for MAX_STEP_ATTEMPTS is marked
unknown and the cursor moves on.
"""

from __future__ import annotations

import copy
import json

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
_CHILD_GROUPS = frozenset({"birth", "body", "illness"})
_PROFILE_KWARGS = frozenset({
    "caller_name", "marital_status", "children", "ever_pregnant", "child_count",
})
_INTERPRETED_OUTCOMES = frozenset({
    "", "not_about_active", "clear", "concern", "unknown", "clarify", "retract",
})
_SIDE_OUTCOMES = frozenset({"concern", "clear", "unknown", "retract"})
_HARD_HOLDS = frozenset({"name_fragment", "marital_status", "children_ages"})

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


def accepted_target(state: dict, target_qid: str) -> str:
    """An earlier answered gate, or ever_pregnant. Anything else is ignored."""
    target = str(target_qid or "").strip()
    if not target or target == state.get("active_qid"):
        return ""
    if target in _EARLY or target == "close_anything_else":
        return ""
    if target != "ever_pregnant":
        try:
            spec = get_spec(target)
        except KeyError:
            return ""
        if spec.kind != "gate":
            return ""
    entry = (state.get("answers") or {}).get(target)
    if not isinstance(entry, dict) or entry.get("state") not in _ANSWERED:
        return ""
    return target


def has_answered_gates(state: dict) -> bool:
    """True when at least one screening gate (or post-intro ever_pregnant) has an answer."""
    if not isinstance(state, dict):
        return False
    answers = state.get("answers") or {}
    if isinstance(answers, dict):
        for qid, entry in answers.items():
            if qid == "intro_bridge":
                continue
            if isinstance(entry, dict) and entry.get("state") in _ANSWERED:
                return True
    active_qid = str(state.get("active_qid") or "")
    past_intro = state.get("phase") != "asking" or (
        bool(active_qid) and active_qid not in _EARLY and active_qid != "intro_bridge"
    )
    if past_intro and (state.get("profile") or {}).get("ever_pregnant") is not None:
        return True
    return False


def is_answered_or_na(state: dict, qid: str) -> bool:
    """True if answers[qid].state is answered or not_applicable."""
    if not isinstance(state, dict) or not qid:
        return False
    entry = (state.get("answers") or {}).get(qid)
    if not isinstance(entry, dict):
        return False
    status = entry.get("state")
    return status in _ANSWERED or status == "not_applicable"


def ever_pregnant_prompt_hi() -> str:
    """Return the Hindi question for the ever_pregnant gate."""
    return _EVER_PREGNANT_HI


def _rewind_target(state: dict, target: str) -> dict:
    """Ask an already-answered question again. Later answers stay put.

    A concern left hanging on which-child or a follow-up is cleared so the
    cursor lands on that question again after the clarified one is answered.
    A close concern that already has the mother's words stays recorded.
    """
    prev_qid = state.get("active_qid")
    pending = state.get("pending_followup_idx")
    if pending is not None and prev_qid and prev_qid != target:
        prior = (state.get("answers") or {}).get(prev_qid)
        keep_close = prev_qid == "close_anything_else" and _close_concern_kept(state)
        if isinstance(prior, dict) and prior.get("state") == "concern" and not keep_close:
            state["answers"].pop(prev_qid, None)
    state["active_qid"] = target
    state["pending_followup_idx"] = None
    state["step_attempts"] = 0
    state["hold_reason"] = None
    state["answers"].pop(target, None)
    if target == "ever_pregnant":
        state["profile"]["ever_pregnant"] = None
    _recompute(state)
    return state


def _revise_target(state: dict, target: str, outcome: str, child_ref: str, note: str) -> dict:
    """Revise an earlier answer and stay on the question she is hearing now."""
    if outcome in ("retract", "clear"):
        state["answers"][target] = _blank("clear")
        if target == "ever_pregnant":
            state["profile"]["ever_pregnant"] = False
    elif outcome == "unknown":
        state["answers"][target] = _blank("unknown")
    else:
        prior = state["answers"].get(target) or _blank("concern")
        followups = dict(prior.get("followups") or {})
        child_ids = [cid for cid in (prior.get("child_ids") or []) if cid]
        if target == "ever_pregnant":
            state["profile"]["ever_pregnant"] = True
        else:
            spec = get_spec(target)
            for cid in _child_ids_for(state, spec, child_ref):
                if cid not in child_ids:
                    child_ids.append(cid)
            if spec.followups and note.strip():
                text = note.strip()
                if len(spec.followups) > 1 and _slot_note(followups, spec.followups[0]):
                    fid = ""
                    for later in spec.followups[1:]:
                        if not _slot_note(followups, later):
                            fid = later
                            followups[fid] = {"state": "concern", "note": text}
                            break
                    if not fid:
                        fid = spec.followups[-1]
                        followups[fid] = {
                            "state": "concern",
                            "note": _join_notes(_slot_note(followups, fid), text),
                        }
                else:
                    fid = spec.followups[0]
                    followups[fid] = {
                        "state": "concern",
                        "note": _join_notes(_slot_note(followups, fid), text),
                    }
        state["answers"][target] = _blank(
            "concern",
            note=_join_notes(str(prior.get("notes") or ""), note),
            child_ids=child_ids,
            followups=followups,
        )
    _recompute(state)
    if _active_is_skipped(state):
        _advance(state)
    return state


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
        if outcome == "emergency" and note.strip():
            state["emergency_note"] = " ".join(note.split())
        return state
    if state["phase"] != "asking":
        return state
    if _is_retract(state, outcome, child_ref):
        return _retract(state)
    if outcome == "clarify":
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


def routable_qids() -> tuple[str, ...]:
    """Closed-world routable question IDs in ask order (58 total)."""
    out: list[str] = []
    ever_inserted = False
    for spec in active_specs():
        if not ever_inserted and "ever_pregnant_no" in spec.skip_if:
            out.append("ever_pregnant")
            ever_inserted = True
        if spec.kind == "gate" or spec.id == "close_anything_else":
            out.append(spec.id)
    if not ever_inserted:
        if "close_anything_else" in out:
            out.insert(out.index("close_anything_else"), "ever_pregnant")
        else:
            out.append("ever_pregnant")
    return tuple(out)


def close_technical_failure(state: dict) -> dict:
    """Close an active call cleanly after repeated technical failure without guessing answers."""
    state = copy.deepcopy(state)
    if state.get("phase") != "asking":
        return state
    state["phase"] = "closing"
    state["close_reason"] = "technical_failure"
    state["pending_followup_idx"] = None
    state["hold_reason"] = None
    return state


def _spec_or_none(qid: str | None):
    if not qid or qid == "ever_pregnant":
        return None
    try:
        return get_spec(qid)
    except KeyError:
        return None


def _stash_clean_answer_before_rewind(state: dict, outcome: str) -> None:
    """Record a clean clear/unknown answer on the active gate before rewinding to clarify another."""
    if (
        state.get("pending_followup_idx") is not None
        or state.get("hold_reason") in _HARD_HOLDS
        or outcome not in ("clear", "unknown")
    ):
        return
    cur_qid = state.get("active_qid")
    if cur_qid == "ever_pregnant":
        state["profile"]["ever_pregnant"] = False
        state["hold_reason"] = None
        state["answers"]["ever_pregnant"] = _blank(outcome)
        return
    cur_spec = _spec_or_none(cur_qid)
    if cur_spec is not None and cur_spec.kind == "gate":
        state["answers"][cur_qid] = _blank(outcome)
    elif cur_qid == "close_anything_else":
        prior = state["answers"].get("close_anything_else")
        if not prior or prior.get("state") != "concern":
            state["answers"]["close_anything_else"] = _blank(outcome)


def apply_interpreted_turn(state: dict, interpreted: dict) -> dict:
    """Apply a structured Decision-API turn interpretation onto a deep-copied cursor state."""
    state = copy.deepcopy(state)
    if state.get("phase") != "asking":
        return state
    data = interpreted if isinstance(interpreted, dict) else {}

    if bool(data.get("emergency")):
        return apply_turn(state, "emergency", note=str(data.get("current_note") or ""))
    if bool(data.get("caller_stop")):
        return apply_turn(state, "caller_stop")

    current_outcome = str(data.get("current_outcome") or "").strip()
    if current_outcome not in _INTERPRETED_OUTCOMES:
        raise ValueError(f"unknown current_outcome {current_outcome!r}")

    raw_profile = data.get("profile_updates")
    profile_kwargs = (
        {
            key: value
            for key, value in raw_profile.items()
            if key in _PROFILE_KWARGS and value is not None
        }
        if isinstance(raw_profile, dict)
        else {}
    )
    profile_applied = False
    skip_active_from_profile = False
    if profile_kwargs:
        prev_active_qid = state.get("active_qid")
        prev_hold = state.get("hold_reason")
        state, profile_hold = apply_profile(state, **profile_kwargs)
        profile_applied = True
        active_changed = state.get("active_qid") != prev_active_qid
        hold_set_other = (
            profile_hold is not None
            and not (state.get("active_qid") == "ever_pregnant" and profile_hold == "ever_pregnant")
        )
        hold_cleared = prev_hold is not None and state.get("hold_reason") is None
        skip_active_from_profile = (
            active_changed
            or hold_set_other
            or hold_cleared
            or prev_active_qid in _EARLY
        )

    raw_child_ref = str(data.get("child_ref") or "").strip()
    side_qid = str(data.get("side_qid") or "").strip()
    side_action = str(data.get("side_action") or "").strip()
    side_outcome_raw = str(data.get("side_outcome") or "").strip()
    side_outcome = side_outcome_raw or ("concern" if side_action == "add_or_correct" else "")
    side_note = str(data.get("side_note") or "")
    raw_side_child_ref = str(data.get("side_child_ref") or "").strip()

    target = accepted_target(state, side_qid)
    side_applied = False
    rewound_by_side = False
    skip_active_from_side = False

    if target and side_action == "add_or_correct" and side_outcome in _SIDE_OUTCOMES:
        side_spec = _spec_or_none(target)
        side_is_child_scoped = side_spec is not None and _is_child_scoped(side_spec)
        active_spec = _spec_or_none(state.get("active_qid"))
        active_is_child_scoped = active_spec is not None and _is_child_scoped(active_spec)
        if not side_is_child_scoped:
            side_child_ref_resolved = ""
        elif raw_side_child_ref:
            side_child_ref_resolved = raw_side_child_ref
        elif active_is_child_scoped and current_outcome == "concern":
            side_child_ref_resolved = ""
        else:
            side_child_ref_resolved = raw_child_ref

        active_before_side = state.get("active_qid")
        state = _revise_target(
            state,
            target,
            side_outcome,
            side_child_ref_resolved,
            side_note,
        )
        side_applied = True
        if state.get("phase") != "asking":
            return state
        if state.get("active_qid") != active_before_side:
            skip_active_from_side = True
    elif target and side_action == "clarify_meaning":
        if not skip_active_from_profile:
            _stash_clean_answer_before_rewind(state, current_outcome)
        state = _rewind_target(state, target)
        side_applied = True
        rewound_by_side = True

    if rewound_by_side or skip_active_from_profile or skip_active_from_side:
        return state
    if state.get("hold_reason") in _HARD_HOLDS:
        return state

    if current_outcome in ("", "not_about_active"):
        if profile_applied or side_applied:
            return state
        if state.get("pending_followup_idx") is not None:
            return apply_turn(state, "concern", child_ref="", note="")
        return state

    if state.get("active_qid") == "ever_pregnant" and state.get("hold_reason") == "ever_pregnant":
        if current_outcome in ("clear", "concern", "unknown", "retract"):
            state["hold_reason"] = None

    pending_idx = state.get("pending_followup_idx")
    is_pending_followup = isinstance(pending_idx, int) and pending_idx >= 0
    keep_active_note = (
        current_outcome == "concern"
        or (is_pending_followup and current_outcome not in ("clear", "unknown"))
        or (pending_idx == -1 and current_outcome == "unknown")
    )
    active_note = str(data.get("current_note") or "") if keep_active_note else ""

    return apply_turn(
        state,
        current_outcome,
        child_ref=raw_child_ref,
        note=active_note,
        followup_answered=bool(data.get("followup_answered")),
    )


def step_id(state: dict) -> str:
    """Identity of the question she should hear before an answer can be saved."""
    phase = str(state.get("phase") or "asking")
    if phase != "asking":
        return "closing:" + str(state.get("close_reason") or "complete")
    qid = str(state.get("active_qid") or "")
    idx = state.get("pending_followup_idx")
    step = qid if idx is None else f"{qid}:{idx}"
    rounds = int(state.get("close_rounds") or 0)
    if qid == "close_anything_else" and rounds:
        step = f"{step}:more:{rounds}"
    hold = str(state.get("hold_reason") or "").strip()
    if hold:
        step = f"{step}:hold:{hold}"
    attempts = int(state.get("step_attempts") or 0)
    if attempts:
        step = f"{step}:try:{attempts}"
    return step


def step_kind(state: dict) -> str:
    """Return 'profile' | 'bridge' | 'main' | 'which_child' | 'followup' | 'close'."""
    if not isinstance(state, dict):
        return "main"
    hold = state.get("hold_reason")
    if hold in _HARD_HOLDS:
        return "profile"
    active_qid = str(state.get("active_qid") or "")
    if active_qid == "ever_pregnant":
        return "main"
    pending_idx = state.get("pending_followup_idx")
    if pending_idx == -1:
        return "which_child"
    spec = _spec_or_none(active_qid)
    if (
        isinstance(pending_idx, int)
        and pending_idx >= 0
        and spec is not None
        and pending_idx < len(spec.followups)
    ):
        return "followup"
    if active_qid == "close_anything_else":
        return "close"
    if spec is not None and spec.kind == "profile":
        return "profile"
    if spec is not None and spec.kind == "bridge":
        return "bridge"
    return "main"


def active_prompt_hi(state: dict) -> str:
    """Return the exact single Hindi question for state, or '' when not asking."""
    if not isinstance(state, dict) or state.get("phase") != "asking":
        return ""
    hold = state.get("hold_reason")
    if hold in _HOLD_HI:
        return _HOLD_HI[hold]
    active_qid = str(state.get("active_qid") or "")
    if active_qid == "ever_pregnant":
        return _EVER_PREGNANT_HI
    pending_idx = state.get("pending_followup_idx")
    if pending_idx == -1:
        children = (state.get("profile") or {}).get("children") or []
        return _which_child_hi(children)
    spec = _spec_or_none(active_qid)
    if spec is None:
        return ""
    if isinstance(pending_idx, int) and 0 <= pending_idx < len(spec.followups):
        return FOLLOWUP_HI[spec.followups[pending_idx]]
    answer = (state.get("answers") or {}).get(active_qid) or {}
    if active_qid == "close_anything_else" and answer.get("state") == "concern":
        return _CLOSE_MORE_HI
    return question_hi_for(state, spec)


def followup_prompt_hi(state: dict) -> str:
    """Return the follow-up Hindi prompt for the active step, or ''."""
    if not isinstance(state, dict):
        return ""
    kind = step_kind(state)
    active_qid = str(state.get("active_qid") or "")
    spec = _spec_or_none(active_qid)
    pending_idx = state.get("pending_followup_idx")
    if (
        kind == "followup"
        and spec is not None
        and isinstance(pending_idx, int)
        and 0 <= pending_idx < len(spec.followups)
    ):
        return FOLLOWUP_HI[spec.followups[pending_idx]]
    if kind == "main" and spec is not None and spec.followups:
        return FOLLOWUP_HI[spec.followups[0]]
    if kind == "close":
        return FOLLOWUP_HI["open_concern_detail"]
    return ""


def last_answered_prompt_hi(state: dict) -> tuple[str, str]:
    """Return (last_qid, prompt_hi) for the last non-bridge answered question, else ('', '')."""
    if not isinstance(state, dict):
        return "", ""
    last_qid = str(state.get("last_answered_qid") or "").strip()
    if not last_qid or last_qid == "intro_bridge":
        return "", ""
    if last_qid == "ever_pregnant":
        return last_qid, _EVER_PREGNANT_HI
    spec = _spec_or_none(last_qid)
    if spec is None:
        return "", ""
    return last_qid, question_hi_for(state, spec)


def children_label_hi(profile: dict) -> str:
    """Format the children's names, ages, and sexes joined with ' · '."""
    if not isinstance(profile, dict):
        return ""
    sex_hi = {"male": "बेटा", "female": "बेटी"}
    bits = []
    for child in profile.get("children") or []:
        if not isinstance(child, dict):
            continue
        child_name = str(child.get("name") or "").strip()
        age = str(child.get("age_label_hi") or "").strip()
        sex = sex_hi.get(child.get("sex") or "", "")
        label = " ".join(part for part in (child_name, age, sex) if part)
        if label:
            bits.append(label)
    return " · ".join(bits)


def answer_notes_text(entry: dict) -> str:
    """Mother's words only: entry notes and follow-up slot notes, deduplicated."""
    if not isinstance(entry, dict):
        return ""
    raw_parts: list[object] = [entry.get("notes")]
    followups = entry.get("followups")
    if isinstance(followups, dict):
        for val in followups.values():
            if isinstance(val, (dict, str)):
                raw_parts.append(val.get("note") if isinstance(val, dict) else val)
    parts: list[str] = []
    seen: set[str] = set()
    for raw in raw_parts:
        cleaned = " ".join(str(raw or "").split())
        if cleaned and cleaned.casefold() not in seen:
            seen.add(cleaned.casefold())
            parts.append(cleaned)
    return "; ".join(parts)


def cursor_from_call(call: dict | None) -> dict:
    """Extract the active cursor state dict from a call record."""
    if not isinstance(call, dict):
        return {}
    for key in ("cursor_state", "cursor_json"):
        raw = call.get(key)
        if isinstance(raw, str) and raw.strip():
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                continue
        if isinstance(raw, dict) and raw.get("active_qid"):
            return raw
    return {}


def question_hi_for(state: dict, spec) -> str:
    """Hindi question for this caller. Incomplete profiles use the catalog line."""
    profile = state.get("profile") if isinstance(state, dict) else None
    if not isinstance(profile, dict):
        return spec.question_hi
    marital = profile.get("marital_status")
    children = profile.get("children")
    if not isinstance(children, list):
        children = []
    if spec.id == "preg_thalassemia":
        if marital == "about_to_marry":
            return _THAL_ABOUT
        if marital == "unmarried":
            return _THAL_UNMARRIED
        return spec.question_hi
    if spec.id == "fam_related" and marital == "about_to_marry":
        return _RELATED_ABOUT
    multi = spec.question_multi_hi
    if len(children) > 1 and multi:
        return multi
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
        if (state["answers"].get(qid) or {}).get("state") in _ANSWERED:
            continue
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
    return bool(" ".join((name or "").strip().split()))


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
    """Return oldest-first child records when the profile has two or more children."""
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
    """Return matching child ids when child_ref equals 'all' or an exact child_id."""
    ref = " ".join((child_ref or "").strip().split())
    if not ref:
        return []
    children = state["profile"]["children"]
    if ref == "all":
        return [child["id"] for child in _ordered_children(children)]
    if ref in {child["id"] for child in children}:
        return [ref]
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


def _slot_note(followups: dict, fid: str) -> str:
    slot = followups.get(fid)
    if isinstance(slot, dict):
        return str(slot.get("note") or "").strip()
    return ""


def _join_notes(existing: str, extra: str) -> str:
    parts: list[str] = []
    seen: set[str] = set()
    for source in (existing, extra):
        for clause in str(source or "").split(";"):
            cleaned = " ".join(clause.split())
            if not cleaned:
                continue
            key = cleaned.casefold()
            if key in seen:
                continue
            seen.add(key)
            parts.append(cleaned)
    return "; ".join(parts)
