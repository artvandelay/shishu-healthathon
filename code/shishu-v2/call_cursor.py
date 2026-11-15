"""Deterministic call cursor. The live model never chooses the next question.

State is JSON-serializable. pending_followup_idx is None on the question
itself, -1 while a multi-child concern still needs a child, and 0+ on a
follow-up. A sub-step that stays unresolved for MAX_STEP_ATTEMPTS is marked
unknown and the cursor moves on.
"""

from __future__ import annotations

import copy

from triage_catalog import CATALOG, CHILD_GROUPS, FOLLOWUP_HI, PROFILE_QIDS, get_spec, spine_ids

MAX_STEP_ATTEMPTS = 2

_OUTCOMES = frozenset({"clear", "concern", "unknown", "clarify", "retract", "emergency", "caller_stop"})
_MARITAL = frozenset({"married", "about_to_marry", "unmarried"})
_NOT_MARRIED = frozenset({"unmarried", "about_to_marry"})
_ANSWERED = frozenset({"clear", "concern", "unknown"})
_DEMOGRAPHIC_SKIP = frozenset({"unmarried", "about_to_marry", "no_children", "has_children", "ever_pregnant_no"})
_PROFILE_KWARGS = frozenset({"caller_name", "marital_status", "children", "ever_pregnant", "child_count"})
_INTERPRETED_OUTCOMES = frozenset({"", "not_about_active", "clear", "concern", "unknown", "clarify", "retract"})
_SIDE_OUTCOMES = frozenset({"concern", "clear", "unknown", "retract"})
_HARD_HOLDS = frozenset({"name_fragment", "marital_status", "children_ages"})
_SPINE_INDEX = {qid: idx for idx, qid in enumerate(spine_ids())}

_CLOSE_MORE_HI = "मैंने यह बात नोट कर ली। इसके अलावा कुछ और बताना चाहेंगी?"
_HOLD_HI = {
    "name_fragment": "जी, आराम से बताइए — आपका पूरा नाम क्या है?",
    "marital_status": "जी, क्या आपकी शादी हो गई है, होने वाली है, या अभी नहीं हुई?",
    "children_ages": "जी, उन सभी बच्चों की उम्र क्या-क्या है?",
}


def new_cursor_state() -> dict:
    return {
        "phase": "asking",
        "close_reason": None,
        "profile": {
            "caller_name": None,
            "marital_status": None,
            "children": [],
            "ever_pregnant": None,
            "children_stated": False,
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
        and state.get("active_qid") not in PROFILE_QIDS
    )


def _clear_pending(profile: dict) -> None:
    profile.pop("pending_children", None)
    profile.pop("pending_child_count", None)


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
    profile = state["profile"]
    prev_marital = profile.get("marital_status")
    if caller_name is not None:
        if not _valid_name(caller_name):
            state["hold_reason"] = "name_fragment"
            return state, "name_fragment"
        profile["caller_name"] = " ".join(caller_name.split())
    children_given = children is not None or child_count is not None
    if marital_status is not None:
        if marital_status not in _MARITAL:
            state["hold_reason"] = "marital_status"
            return state, "marital_status"
        profile["marital_status"] = marital_status
        if marital_status in _NOT_MARRIED and not children_given:
            profile["children"] = []
            _clear_pending(profile)
            profile["children_stated"] = True
        elif prev_marital in _NOT_MARRIED and marital_status == "married" and not children_given:
            profile["children_stated"] = False
    if children_given:
        incoming = list(children) if children is not None else []
        stated = child_count if isinstance(child_count, int) and child_count >= 0 else len(incoming)
        if (stated > 0 and len(incoming) < stated) or any(c.get("age_years") is None for c in incoming):
            valid_partial = [
                _normalize_child(c, i)
                for i, c in enumerate(incoming, start=1)
                if isinstance(c, dict) and c.get("age_years") is not None
            ]
            _clear_pending(profile)
            if valid_partial:
                profile["pending_children"] = valid_partial
            if stated > 0:
                profile["pending_child_count"] = stated
            state["hold_reason"] = "children_ages"
            return state, "children_ages"
        profile["children"] = [_normalize_child(c, i) for i, c in enumerate(incoming, start=1)]
        profile["children_stated"] = True
        _clear_pending(profile)
    if profile["children"]:
        profile["ever_pregnant"] = True
    elif ever_pregnant is not None:
        val = bool(ever_pregnant)
        profile["ever_pregnant"] = val
        if profile.get("marital_status") not in _NOT_MARRIED:
            state["answers"]["ever_pregnant"] = _blank("concern" if val else "clear")
    elif marital_status in _NOT_MARRIED:
        profile["ever_pregnant"] = False
    elif (
        prev_marital in _NOT_MARRIED
        and marital_status == "married"
        and not profile["children"]
        and (state["answers"].get("ever_pregnant") or {}).get("state") not in _ANSWERED
    ):
        profile["ever_pregnant"] = None

    _recompute(state)
    active_spec = _spec_or_none(state.get("active_qid"))
    if active_spec is not None and _skip_hits(active_spec, state):
        _advance(state, record_answered=False)
    hold = _profile_hold(state, children_given=children_given)
    if hold:
        state["hold_reason"] = hold
        return state, hold
    state["hold_reason"] = None

    cur = state.get("active_qid")
    top = _next_qid(state)
    cur_pos, top_pos = _SPINE_INDEX.get(cur) if cur else None, _SPINE_INDEX.get(top) if top else None
    reopened_earlier = top is not None and cur_pos is not None and top_pos is not None and top_pos < cur_pos
    satisfied_profile = cur in PROFILE_QIDS and _profile_satisfied(cur, state)
    satisfied_ever = cur == "ever_pregnant" and profile.get("ever_pregnant") is not None
    if (reopened_earlier or satisfied_profile or satisfied_ever) and top != cur:
        if reopened_earlier:
            _drop_unfinished_answer(state, cur)
        _set_step(state, top, record_answered=not reopened_earlier)
        _recompute(state)
    return state, None


def accepted_target(state: dict, target_qid: str) -> str:
    """An earlier answered gate or routing_gate. Anything else is ignored."""
    target = str(target_qid or "").strip()
    if not target or target == state.get("active_qid"):
        return ""
    spec = _spec_or_none(target)
    if spec is None or spec.kind not in ("routing_gate", "gate"):
        return ""
    entry = (state.get("answers") or {}).get(target)
    return target if isinstance(entry, dict) and entry.get("state") in _ANSWERED else ""


def has_answered_gates(state: dict) -> bool:
    """True when at least one screening gate (or post-intro ever_pregnant) has an answer."""
    if not isinstance(state, dict):
        return False
    answers = state.get("answers") or {}
    if isinstance(answers, dict):
        for qid, entry in answers.items():
            spec = _spec_or_none(str(qid))
            if spec is not None and spec.kind in ("routing_gate", "gate", "open_capture"):
                if isinstance(entry, dict) and entry.get("state") in _ANSWERED:
                    return True
    active_qid = str(state.get("active_qid") or "")
    past_intro = state.get("phase") != "asking" or (
        bool(active_qid) and active_qid not in PROFILE_QIDS and active_qid != "intro_bridge"
    )
    return bool(past_intro and (state.get("profile") or {}).get("ever_pregnant") is not None)


def is_answered_or_na(state: dict, qid: str) -> bool:
    """True if answers[qid].state is answered or not_applicable."""
    if not isinstance(state, dict) or not qid:
        return False
    entry = (state.get("answers") or {}).get(qid)
    return isinstance(entry, dict) and entry.get("state") in (*_ANSWERED, "not_applicable")


def _drop_unfinished_answer(state: dict, qid: str | None) -> None:
    if state.get("pending_followup_idx") is None or not qid:
        return
    if qid == "close_anything_else" and _close_concern_kept(state):
        return
    state["answers"].pop(qid, None)


def _rewind_target(state: dict, target: str) -> dict:
    """Ask an already-answered question again. Later answers stay put."""
    prev_qid = state.get("active_qid")
    if prev_qid != target:
        _drop_unfinished_answer(state, prev_qid)
    state["active_qid"] = target
    state["pending_followup_idx"] = None
    state["step_attempts"] = 0
    state["hold_reason"] = None
    state["answers"].pop(target, None)
    if target == "ever_pregnant":
        state["profile"]["ever_pregnant"] = None
    _recompute(state)
    return state


def _revise_target(
    state: dict, target: str, outcome: str, child_ref: str, note: str, *, diagnostic: bool = False
) -> dict:
    """Revise an earlier answer and stay on the question she is hearing now."""
    if outcome in ("retract", "clear", "unknown"):
        state["answers"][target] = _blank("unknown" if outcome == "unknown" else "clear")
        if target == "ever_pregnant":
            state["profile"]["ever_pregnant"] = False
    else:
        spec = get_spec(target)
        if target == "ever_pregnant":
            state["profile"]["ever_pregnant"] = True
        entry = _entry_from_prior(state["answers"].get(target), "concern")
        for cid in _child_ids_for(state, spec, child_ref):
            if cid not in entry["child_ids"]:
                entry["child_ids"].append(cid)
        if spec.followups and note.strip():
            text = note.strip()
            followups = entry["followups"]
            if len(spec.followups) > 1 and _slot_note(followups, spec.followups[0]):
                fid = next((f for f in spec.followups[1:] if not _slot_note(followups, f)), spec.followups[-1])
            else:
                fid = spec.followups[0]
            followups[fid] = _merge_slot(followups.get(fid), text, diagnostic=diagnostic)
        _merge_entry_notes(entry, note, diagnostic=diagnostic)
        state["answers"][target] = entry
    _recompute(state)
    active_spec = _spec_or_none(state.get("active_qid"))
    if active_spec is not None and _skip_hits(active_spec, state):
        _advance(state, record_answered=False)
    return state


def apply_turn(
    state: dict,
    outcome: str,
    *,
    child_ref: str = "",
    note: str = "",
    followup_answered: bool = False,
    diagnostic: bool = False,
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
    if outcome == "retract" or (
        state.get("pending_followup_idx") == -1 and outcome == "clear" and not (child_ref or "").strip()
    ):
        return _retract(state)
    if outcome == "clarify":
        return state

    spec = get_spec(state["active_qid"])
    if spec.kind == "open_capture":
        return _resolve_close(state, outcome, note, diagnostic=diagnostic)
    if state.get("pending_followup_idx") == -1:
        return _resolve_which_child(state, child_ref, note, outcome, diagnostic=diagnostic)
    if isinstance(state.get("pending_followup_idx"), int) and state["pending_followup_idx"] >= 0:
        return _resolve_followup(state, outcome, note, diagnostic=diagnostic)
    if spec.kind == "profile":
        return state
    if spec.kind == "bridge":
        _write_answer(state, spec.id, "clear", "", [])
        _recompute(state)
        _advance(state)
        return state
    return _resolve_main(state, spec, outcome, child_ref, note, followup_answered, diagnostic=diagnostic)


def routable_qids() -> tuple[str, ...]:
    """Closed-world routable question IDs in ask order (58 total)."""
    return tuple(spec.id for spec in CATALOG if spec.kind in ("routing_gate", "gate", "open_capture"))


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
    try:
        return get_spec(qid) if qid else None
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
    cur_spec = _spec_or_none(cur_qid)
    if cur_spec is None:
        return
    if cur_spec.kind == "routing_gate":
        state["profile"]["ever_pregnant"] = False
        state["hold_reason"] = None
    if cur_spec.kind in ("routing_gate", "gate") or (
        cur_spec.kind == "open_capture" and (state["answers"].get(cur_qid) or {}).get("state") != "concern"
    ):
        state["answers"][cur_qid] = _blank(outcome)


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

    diagnostic_note = bool(data.get("diagnostic_note"))
    raw_profile = data.get("profile_updates")
    has_profile_dict = isinstance(raw_profile, dict)
    profile_kwargs = (
        {k: v for k, v in raw_profile.items() if k in _PROFILE_KWARGS and v is not None}
        if has_profile_dict
        else {}
    )
    prev_step_kind = step_kind(state)
    should_run_profile = bool(profile_kwargs) or (
        has_profile_dict and prev_step_kind == "profile" and current_outcome != "clarify"
    )
    profile_applied = False
    skip_active_from_profile = False
    if should_run_profile:
        prev_active_qid, prev_hold = state.get("active_qid"), state.get("hold_reason")
        state, profile_hold = apply_profile(state, **profile_kwargs)
        profile_applied = True
        skip_active_from_profile = (
            state.get("active_qid") != prev_active_qid
            or profile_hold is not None
            or (prev_hold is not None and state.get("hold_reason") is None)
            or prev_step_kind == "profile"
        )

    raw_child_ref = str(data.get("child_ref") or "").strip()
    target = accepted_target(state, str(data.get("side_qid") or ""))
    side_action = str(data.get("side_action") or "").strip()
    side_outcome = str(data.get("side_outcome") or "").strip() or ("concern" if side_action == "add_or_correct" else "")
    side_note = str(data.get("side_note") or "")
    side_applied = False

    if target and side_action == "add_or_correct" and side_outcome in _SIDE_OUTCOMES:
        side_spec = _spec_or_none(target)
        active_spec = _spec_or_none(state.get("active_qid"))
        both_child_concerns = (
            side_spec is not None and _is_child_scoped(side_spec)
            and active_spec is not None and _is_child_scoped(active_spec)
            and current_outcome == "concern"
        )
        side_child_ref = "" if (side_spec is None or not _is_child_scoped(side_spec) or both_child_concerns) else raw_child_ref
        active_before_side = state.get("active_qid")
        state = _revise_target(state, target, side_outcome, side_child_ref, side_note, diagnostic=diagnostic_note)
        side_applied = True
        if state.get("phase") != "asking" or state.get("active_qid") != active_before_side:
            return state
    elif target and side_action == "clarify_meaning":
        if not skip_active_from_profile:
            _stash_clean_answer_before_rewind(state, current_outcome)
        return _rewind_target(state, target)

    if skip_active_from_profile or state.get("hold_reason") in _HARD_HOLDS:
        return state

    if current_outcome in ("", "not_about_active"):
        if profile_applied or side_applied:
            return state
        return apply_turn(state, "concern") if state.get("pending_followup_idx") is not None else state

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
        diagnostic=diagnostic_note,
    )


def step_id(state: dict) -> str:
    """Identity of the question she should hear before an answer can be saved."""
    if str(state.get("phase") or "asking") != "asking":
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
    return f"{step}:try:{attempts}" if attempts else step


def step_kind(state: dict) -> str:
    """Return 'profile' | 'bridge' | 'main' | 'which_child' | 'followup' | 'close'."""
    if not isinstance(state, dict):
        return "main"
    if state.get("hold_reason") in _HARD_HOLDS:
        return "profile"
    pending_idx = state.get("pending_followup_idx")
    if pending_idx == -1:
        return "which_child"
    spec = _spec_or_none(str(state.get("active_qid") or ""))
    if isinstance(pending_idx, int) and pending_idx >= 0 and spec is not None and pending_idx < len(spec.followups):
        return "followup"
    if spec is None:
        return "main"
    if spec.kind == "open_capture":
        return "close"
    return spec.kind if spec.kind in ("profile", "bridge") else "main"


def active_prompt_hi(state: dict) -> str:
    """Return the exact single Hindi question for state, or '' when not asking."""
    if not isinstance(state, dict) or state.get("phase") != "asking":
        return ""
    hold = state.get("hold_reason")
    if hold in _HOLD_HI:
        return _HOLD_HI[hold]
    pending_idx = state.get("pending_followup_idx")
    if pending_idx == -1:
        return _which_child_hi((state.get("profile") or {}).get("children") or [])
    active_qid = str(state.get("active_qid") or "")
    spec = _spec_or_none(active_qid)
    if spec is None:
        return ""
    if isinstance(pending_idx, int) and 0 <= pending_idx < len(spec.followups):
        return FOLLOWUP_HI[spec.followups[pending_idx]]
    answer = (state.get("answers") or {}).get(active_qid) or {}
    if spec.kind == "open_capture" and answer.get("state") == "concern":
        return _CLOSE_MORE_HI
    return question_hi_for(state, spec)


def followup_prompt_hi(state: dict) -> str:
    """Return the follow-up Hindi prompt for the active step, or ''."""
    if not isinstance(state, dict):
        return ""
    kind = step_kind(state)
    spec = _spec_or_none(str(state.get("active_qid") or ""))
    pending_idx = state.get("pending_followup_idx")
    if kind == "followup" and spec is not None and isinstance(pending_idx, int) and 0 <= pending_idx < len(spec.followups):
        return FOLLOWUP_HI[spec.followups[pending_idx]]
    if kind == "main" and spec is not None and spec.followups:
        return FOLLOWUP_HI[spec.followups[0]]
    return FOLLOWUP_HI["open_concern_detail"] if kind == "close" else ""


def last_answered_prompt_hi(state: dict) -> tuple[str, str]:
    """Return (last_qid, prompt_hi) for the last non-bridge answered question, else ('', '')."""
    if not isinstance(state, dict):
        return "", ""
    last_qid = str(state.get("last_answered_qid") or "").strip()
    spec = _spec_or_none(last_qid)
    if spec is None or spec.kind == "bridge":
        return "", ""
    return last_qid, question_hi_for(state, spec)


def _select_children(profile: dict, child_ids: list[str] | None) -> list[dict]:
    children = [c for c in (profile.get("children") or []) if isinstance(c, dict)]
    if child_ids is None:
        return children
    wanted = {str(cid).strip() for cid in child_ids if str(cid).strip()}
    return [c for c in _ordered_children(children) if str(c.get("id") or "") in wanted] if wanted else []


def children_label_hi(profile: dict, child_ids: list[str] | None = None) -> str:
    """Format the children's names, ages, and sexes in Hindi joined with ' · '."""
    if not isinstance(profile, dict):
        return ""
    sex_hi = {"male": "बेटा", "female": "बेटी"}
    bits = []
    for c in _select_children(profile, child_ids):
        parts = (str(c.get("name") or "").strip(), str(c.get("age_label_hi") or "").strip(), sex_hi.get(c.get("sex") or "", ""))
        label = " ".join(p for p in parts if p)
        if label:
            bits.append(label)
    return " · ".join(bits)


def children_label_en(profile: dict, child_ids: list[str] | None = None) -> str:
    """Format the children's ids, sexes, ages, and names in English."""
    if not isinstance(profile, dict):
        return ""
    bits = []
    for c in _select_children(profile, child_ids):
        name = str(c.get("name") or "").strip()
        core = f"{str(c.get('id') or '').strip()} ({str(c.get('sex') or 'unknown').strip()}, age {c.get('age_years')}"
        bits.append(core + (f", {name})" if name else ")"))
    return ", ".join(bits)


def _dedupe_clauses(raw_parts: list[object]) -> str:
    parts: list[str] = []
    seen: set[str] = set()
    for raw in raw_parts:
        for clause in str(raw or "").split(";"):
            cleaned = " ".join(clause.split())
            if cleaned and cleaned.casefold() not in seen:
                seen.add(cleaned.casefold())
                parts.append(cleaned)
    return "; ".join(parts)


def merge_transcript_chunk(prev: str, chunk: str) -> str:
    """Merge streaming or cumulative transcript chunks without duplicating prefixes."""
    prev_s, chunk_s = str(prev or "").strip(), str(chunk or "").strip()
    if not prev_s or chunk_s[: len(prev_s)] == prev_s:
        return chunk_s
    if not chunk_s or prev_s[-len(chunk_s) :] == chunk_s:
        return prev_s
    sep = "" if (prev_s[-1].isspace() or chunk_s[0] in ",.;:!?।") else " "
    return prev_s + sep + chunk_s


def answer_notes_text(entry: dict) -> str:
    """Clinician-facing verbatim words: entry notes and follow-up slot notes, deduplicated."""
    if not isinstance(entry, dict):
        return ""
    raw_parts: list[object] = [entry.get("notes")]
    followups = entry.get("followups")
    if isinstance(followups, dict):
        raw_parts.extend(v.get("note") if isinstance(v, dict) else v for v in followups.values() if isinstance(v, (dict, str)))
    return _dedupe_clauses(raw_parts)


def mother_notes_text(entry: dict) -> str:
    """Mother-facing words: omits any clause or follow-up slot flagged diagnostic by Decision API."""
    if not isinstance(entry, dict):
        return ""
    raw_parts: list[object] = [entry["mother_notes"] if "mother_notes" in entry else entry.get("notes")]
    followups = entry.get("followups")
    if isinstance(followups, dict):
        for val in followups.values():
            if isinstance(val, dict):
                raw_parts.append(val["mother_note"] if "mother_note" in val else ("" if val.get("diagnostic") else val.get("note")))
            elif isinstance(val, str):
                raw_parts.append(val)
    return _dedupe_clauses(raw_parts)


def cursor_from_call(call: dict | None) -> dict:
    """Extract the active cursor state dict from a call record."""
    if not isinstance(call, dict):
        return {}
    raw = call.get("cursor_state")
    return raw if isinstance(raw, dict) and raw.get("active_qid") else {}


def question_hi_for(state: dict, spec) -> str:
    """Hindi question for this caller. Incomplete profiles use the catalog line."""
    profile = state.get("profile") if isinstance(state, dict) else None
    if not isinstance(profile, dict):
        return spec.question_hi
    marital = profile.get("marital_status")
    children = profile.get("children")
    if not isinstance(children, list):
        children = []
    if marital == "about_to_marry" and spec.question_about_hi:
        return spec.question_about_hi
    if marital == "unmarried" and spec.question_unmarried_hi:
        return spec.question_unmarried_hi
    if len(children) > 1 and spec.question_multi_hi:
        return spec.question_multi_hi
    return spec.question_hi


def _resolve_main(
    state, spec, outcome, child_ref, note, followup_answered, *, diagnostic: bool = False
) -> dict:
    child_ids = _child_ids_for(state, spec, child_ref)
    followups = (
        {spec.followups[0]: _make_slot("concern", note.strip(), diagnostic=diagnostic)}
        if outcome == "concern" and followup_answered and note.strip() and spec.followups
        else None
    )
    _write_answer(state, spec.id, outcome, note, child_ids, followups, diagnostic=diagnostic)
    if spec.id == "ever_pregnant":
        state["profile"]["ever_pregnant"] = outcome == "concern"
    _recompute(state)
    if outcome == "concern":
        if _is_child_scoped(spec) and len(state["profile"]["children"]) > 1 and not _resolve_child_ids(state, child_ref):
            state["pending_followup_idx"], state["step_attempts"] = -1, 0
            return state
        if spec.followups and (followups is None or len(spec.followups) > 1):
            state["pending_followup_idx"] = 0 if followups is None else 1
            state["step_attempts"] = 0
            return state
    _advance(state)
    return state


def _close_concern_kept(state: dict) -> bool:
    prior = state["answers"].get("close_anything_else") or {}
    return prior.get("state") == "concern" and bool(str(prior.get("notes") or "").strip())


def _retract(state: dict) -> dict:
    qid = state["active_qid"]
    if qid == "ever_pregnant":
        state["profile"]["ever_pregnant"] = False
    if not (qid == "close_anything_else" and _close_concern_kept(state)):
        state["answers"][qid] = _blank("clear")
    _recompute(state)
    _advance(state)
    return state


def _save_close_note(state: dict, text: str, *, diagnostic: bool, slot: bool = False) -> dict:
    entry = _entry_from_prior(state["answers"].get("close_anything_else"), "concern")
    if slot:
        entry["followups"]["open_concern_detail"] = _merge_slot(
            entry["followups"].get("open_concern_detail"), text, diagnostic=diagnostic
        )
    _merge_entry_notes(entry, text, diagnostic=diagnostic)
    state["answers"]["close_anything_else"] = entry
    state["pending_followup_idx"] = None
    state["step_attempts"] = 0
    state["close_rounds"] = int(state.get("close_rounds") or 0) + 1
    return state


def _resolve_close(state: dict, outcome: str, note: str, *, diagnostic: bool = False) -> dict:
    text = note.strip()
    in_detail = state.get("pending_followup_idx") is not None
    if text and (outcome == "concern" or in_detail):
        return _save_close_note(state, text, diagnostic=diagnostic, slot=in_detail)
    if outcome in ("clear", "unknown"):
        if not _close_concern_kept(state):
            _write_answer(state, "close_anything_else", outcome, "", [])
        _advance(state)
        return state
    if not in_detail:
        state["pending_followup_idx"], state["step_attempts"] = 0, 0
        return state
    if _bump_attempt(state):
        if not _close_concern_kept(state):
            state["answers"]["close_anything_else"] = _blank(
                "unknown", followups={"open_concern_detail": {"state": "unknown", "note": ""}}
            )
        _advance(state)
    return state


def _bump_attempt(state: dict) -> bool:
    state["step_attempts"] = int(state.get("step_attempts") or 0) + 1
    if state["step_attempts"] < MAX_STEP_ATTEMPTS:
        return False
    state["step_attempts"] = 0
    return True


def _resolve_which_child(
    state: dict, child_ref: str, note: str, outcome: str, *, diagnostic: bool = False
) -> dict:
    qid = state["active_qid"]
    spec = get_spec(qid)
    answer = _entry_from_prior(state["answers"].get(qid), "concern")
    _merge_entry_notes(answer, note, diagnostic=diagnostic)
    ids = _resolve_child_ids(state, child_ref)
    resolved = bool(ids and outcome != "clear")
    if resolved or outcome == "unknown" or _bump_attempt(state):
        answer["child_ids"] = ids if resolved else []
        state["answers"][qid] = answer
        state["step_attempts"] = 0
        filled = answer.get("followups") or {}
        for index, fid in enumerate(spec.followups):
            if fid not in filled:
                state["pending_followup_idx"] = index
                return state
        _advance(state)
        return state
    state["answers"][qid] = answer
    return state


def _resolve_followup(state: dict, outcome: str, note: str, *, diagnostic: bool = False) -> dict:
    spec = get_spec(state["active_qid"])
    idx = state["pending_followup_idx"]
    fid = spec.followups[idx]
    unresolved = outcome == "clarify" or (
        not note.strip() if outcome == "concern" else (outcome not in _ANSWERED and not note.strip())
    )
    if unresolved:
        if not _bump_attempt(state):
            return state
        outcome, note, diagnostic = "unknown", "", False
    answer = _entry_from_prior(state["answers"].get(spec.id), "concern")
    stored = outcome if outcome in _ANSWERED else "concern"
    answer["followups"][fid] = _make_slot(stored, note.strip(), diagnostic=diagnostic)
    state["answers"][spec.id] = answer
    state["step_attempts"] = 0
    if idx + 1 < len(spec.followups):
        state["pending_followup_idx"] = idx + 1
        return state
    _advance(state)
    return state


def _set_step(state: dict, nxt: str | None, *, record_answered: bool = True) -> None:
    if record_answered:
        state["last_answered_qid"] = state.get("active_qid")
    state["pending_followup_idx"] = None
    state["step_attempts"] = 0
    if nxt is None:
        state["phase"] = "closing"
        state["close_reason"] = state.get("close_reason") or "complete"
    else:
        state["active_qid"] = nxt


def _advance(state: dict, *, record_answered: bool = True) -> None:
    _set_step(state, _next_qid(state), record_answered=record_answered)


def _profile_satisfied(qid: str, state: dict) -> bool:
    profile = state["profile"]
    if qid == "intro_name":
        return _valid_name(profile.get("caller_name"))
    if qid == "intro_married":
        return profile.get("marital_status") in _MARITAL
    return _children_resolved(state) if qid == "intro_children" else False


def _next_qid(state: dict) -> str | None:
    for qid in spine_ids():
        spec = get_spec(qid)
        if _skip_hits(spec, state):
            prior = state["answers"].get(qid)
            if not prior or prior.get("state") not in _ANSWERED:
                state["answers"][qid] = _blank("not_applicable")
            continue
        if spec.kind == "profile":
            if not _profile_satisfied(qid, state):
                return qid
        elif (state["answers"].get(qid) or {}).get("state") not in _ANSWERED:
            return qid
    return None


def _profile_hold(state: dict, *, children_given: bool = False) -> str | None:
    profile = state["profile"]
    if profile.get("pending_child_count") or profile.get("pending_children"):
        return "children_ages"
    qid = state["active_qid"]
    if qid == "intro_name" and not _valid_name(profile.get("caller_name")):
        return "name_fragment"
    if qid == "intro_married" and profile.get("marital_status") not in _MARITAL:
        return "marital_status"
    if qid == "intro_children" and not _children_resolved(state):
        if children_given or state.get("hold_reason") == "children_ages":
            return "children_ages"
    return None


def _valid_name(name: str | None) -> bool:
    return bool(" ".join((name or "").strip().split()))


def _normalize_child(child: dict, index: int) -> dict:
    age = float(child["age_years"])
    sex = child.get("sex") or "unknown"
    return {
        "id": child.get("id") or f"child_{index}",
        "age_years": age,
        "age_label_hi": child.get("age_label_hi") or _age_label(age),
        "sex": sex if sex in ("male", "female", "unknown") else "unknown",
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
        name, sex = (child.get("name") or "").strip(), child.get("sex")
        is_female = sex == "female"
        phrase = (
            f"{label} {'वाली' if is_female else 'वाले'} {name}"
            if name
            else label + (" वाली बेटी" if is_female else (" वाले बेटे" if sex == "male" else " वाले"))
        )
        if index == middle:
            phrase += " (बीच वाली)" if is_female else " (बीच वाले)"
        parts.append(phrase)
    listed = (", ".join(parts[:-1]) + ", या " + parts[-1]) if len(parts) >= 2 else (parts[0] if parts else "")
    return f"यह किस बच्चे की बात है — {listed}?"


def children_roster(state: dict) -> list[dict]:
    """Return oldest-first child records when the profile has two or more children."""
    children = (state.get("profile") or {}).get("children") or []
    if len(children) <= 1:
        return []
    ordered = _ordered_children(children)
    last = len(ordered) - 1
    return [
        {
            "id": c["id"],
            "rank": "oldest" if i == 0 else ("youngest" if i == last else "middle"),
            "age_label_hi": c["age_label_hi"],
            "sex": c["sex"],
            "name": c["name"],
        }
        for i, c in enumerate(ordered)
    ]


def _resolve_child_ids(state: dict, child_ref: str) -> list[str]:
    """Return matching child ids when child_ref equals 'all' or an exact child_id."""
    ref = " ".join((child_ref or "").strip().split())
    if not ref:
        return []
    children = state["profile"]["children"]
    if ref == "all":
        return [child["id"] for child in _ordered_children(children)]
    return [ref] if ref in {child["id"] for child in children} else []


def _children_resolved(state: dict) -> bool:
    profile = state["profile"]
    if profile["marital_status"] in _NOT_MARRIED or profile["children"]:
        return True
    if "children_stated" in profile:
        return bool(profile["children_stated"])
    return state["active_qid"] not in PROFILE_QIDS


def _skip_hits(spec, state: dict) -> tuple[str, ...]:
    if not spec.skip_if:
        return ()
    profile = state["profile"]
    children, marital, ever = profile["children"], profile["marital_status"], profile["ever_pregnant"]
    resolved = _children_resolved(state)
    hits: list[str] = []
    for rule in spec.skip_if:
        if (
            (rule == "no_children" and resolved and not children)
            or (rule == "has_children" and bool(children))
            or (rule == "unmarried" and marital == "unmarried")
            or (rule == "about_to_marry" and marital == "about_to_marry")
            or (rule == "ever_pregnant_no" and (ever is False or (marital in _NOT_MARRIED and ever is not True)))
            or (rule == "no_child_5_plus" and resolved and not any(float(c["age_years"]) >= 5 for c in children))
            or (rule == "all_under_10" and resolved and not any(float(c["age_years"]) >= 10 for c in children))
            or (rule == "no_daughter_11_plus" and resolved and not any(c.get("sex") == "female" and float(c["age_years"]) >= 11 for c in children))
            or (rule == "no_family_concern" and not _has_family_concern(state))
        ):
            hits.append(rule)
    return tuple(hits)


def _has_family_concern(state: dict) -> bool:
    for qid, answer in state["answers"].items():
        if qid == "fam_same_problem":
            continue
        spec = _spec_or_none(str(qid))
        if spec is not None and spec.group == "family" and isinstance(answer, dict) and answer.get("state") == "concern":
            return True
    return False


def _prior_mother_notes(entry: dict) -> str:
    if not isinstance(entry, dict):
        return ""
    return str(entry.get("mother_notes") or "") if "mother_notes" in entry else str(entry.get("notes") or "")


def _make_slot(status: str, note: str, *, diagnostic: bool = False, mother_note: str | None = None) -> dict:
    mother = ("" if diagnostic else note) if mother_note is None else mother_note
    slot: dict = {"state": status, "note": note}
    if mother != note:
        slot["mother_note" if mother else "diagnostic"] = mother or True
    return slot


def _merge_slot(prior_slot: object, extra_note: str, *, diagnostic: bool = False) -> dict:
    prev_note = str(prior_slot.get("note") or "") if isinstance(prior_slot, dict) else ""
    if isinstance(prior_slot, dict):
        prev_mother = str(prior_slot.get("mother_note") or "") if "mother_note" in prior_slot else ("" if prior_slot.get("diagnostic") else prev_note)
    else:
        prev_mother = ""
    merged_note = _join_notes(prev_note, extra_note)
    merged_mother = prev_mother if diagnostic else _join_notes(prev_mother, extra_note)
    return _make_slot("concern", merged_note, mother_note=merged_mother)


def _blank(
    status: str, note: str = "", child_ids: list | None = None,
    followups: dict | None = None, *, mother_note: str | None = None,
) -> dict:
    clean_note = note or ""
    return {
        "state": status,
        "child_ids": list(child_ids or []),
        "notes": clean_note,
        "mother_notes": clean_note if mother_note is None else (mother_note or ""),
        "followups": dict(followups or {}),
    }


def _entry_from_prior(prior: object, status: str) -> dict:
    if not isinstance(prior, dict):
        return _blank(status)
    return _blank(
        status, note=str(prior.get("notes") or ""),
        child_ids=[cid for cid in (prior.get("child_ids") or []) if cid],
        followups=dict(prior.get("followups") or {}), mother_note=_prior_mother_notes(prior),
    )


def _merge_entry_notes(entry: dict, note: str, *, diagnostic: bool = False) -> None:
    if not note.strip():
        return
    prior_mother = _prior_mother_notes(entry)
    entry["notes"] = _join_notes(entry.get("notes") or "", note)
    entry["mother_notes"] = prior_mother if diagnostic else _join_notes(prior_mother, note)


def _recompute(state: dict) -> None:
    for spec in CATALOG:
        answer = state["answers"].get(spec.id)
        hits = _skip_hits(spec, state)
        if hits:
            demographic = any(hit in _DEMOGRAPHIC_SKIP for hit in hits)
            if answer and answer.get("state") in _ANSWERED and not demographic:
                continue
            state["answers"][spec.id] = _blank("not_applicable")
        elif answer and answer.get("state") == "not_applicable":
            del state["answers"][spec.id]


def _write_answer(state, qid, outcome, note, child_ids, followups=None, *, diagnostic: bool = False) -> None:
    prior = state["answers"].get(qid)
    keep = isinstance(prior, dict) and prior.get("state") in _ANSWERED
    entry = _entry_from_prior(prior if keep else None, outcome)
    if child_ids:
        entry["child_ids"] = list(child_ids)
    if followups is not None:
        entry["followups"] = dict(followups)
    if note:
        entry["notes"] = note
        if not diagnostic:
            entry["mother_notes"] = note
    state["answers"][qid] = entry


def _is_child_scoped(spec) -> bool:
    return spec.group in CHILD_GROUPS and "no_children" in spec.skip_if


def _child_ids_for(state: dict, spec, child_ref: str) -> list[str]:
    if not _is_child_scoped(spec):
        return []
    found = _resolve_child_ids(state, child_ref)
    if found:
        return found
    children = state["profile"]["children"]
    return [children[0]["id"]] if len(children) == 1 and not (child_ref or "").strip() else []


def _slot_note(followups: dict, fid: str) -> str:
    slot = followups.get(fid)
    return str(slot.get("note") or "").strip() if isinstance(slot, dict) else ""


def _join_notes(existing: str, extra: str) -> str:
    return _dedupe_clauses([existing, extra])
