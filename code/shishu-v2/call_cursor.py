"""Deterministic call cursor. The live model never chooses the next question.

State is JSON-serializable. pending_followup_idx is None when the active
question itself is being asked, -1 when a multi-child concern still needs a
child id, and 0+ while that question's follow-ups are in progress.
"""

from __future__ import annotations

import copy

from triage_catalog import FOLLOWUP_HI, active_specs, get_spec, spine_ids

_OUTCOMES = frozenset({"clear", "concern", "unknown", "clarify", "emergency", "caller_stop"})
_MARITAL = frozenset({"married", "about_to_marry", "unmarried"})
_ANSWERED = frozenset({"clear", "concern", "unknown"})
_EARLY = frozenset({"intro_name", "intro_married", "intro_children"})
_BARE_NAMES = frozenset({
    "मेरा नाम",
    "मेरा नाम है",
    "नाम",
    "जी मेरा नाम",
    "हाँ मेरा नाम",
    "हां मेरा नाम",
    "mera naam",
    "mera naam hai",
    "naam",
    "my name",
    "my name is",
})
_OLDEST = frozenset({"oldest", "बड़े वाले", "bade wale", "बड़े"})
_YOUNGEST = frozenset({"youngest", "छोटे वाले", "chhote wale", "छोटे"})
_ALL_CHILDREN = frozenset({"both", "all", "दोनों", "सभी"})

_INFANT_ONLY_HI = "क्या परिवार में किसी छोटे बच्चे की 1 साल से कम उम्र में मौत हुई?"
# हाँ = a trait, minor, or other abnormal result was reported. The universal
# "get tested before the next pregnancy" line stays on Page 1 of the PDF.
_UNMARRIED_THAL_HI = "क्या आपकी थैलेसीमिया जाँच में कभी कोई दिक्कत बताई गई?"
_EVER_PREGNANT_HI = "क्या आप कभी गर्भवती हुई हैं?"
_WHICH_CHILD_HI = "यह किस बच्चे की बात है — बड़े वाले या छोटे वाले?"


def new_cursor_state() -> dict:
    return {
        "version": 1,
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
        "module_queue": [],
        "answers": {},
    }


def apply_profile(
    state: dict,
    *,
    caller_name: str | None = None,
    marital_status: str | None = None,
    children: list | None = None,
    ever_pregnant: bool | None = None,
    child_count: int | None = None,
) -> tuple[dict, str | None]:
    """Store one profile step and move on when it is complete.

    Returns the new state and a hold token when the current step is not
    acceptable yet: name_fragment, marital_status, children_ages, ever_pregnant.

    child_count is the number she stated. child_count > 0 with an empty list,
    a short list, or any missing age_years holds for children_ages and does
    not mark her childless.
    """
    state = copy.deepcopy(state)
    if caller_name is not None and not _valid_name(caller_name):
        return state, "name_fragment"
    if caller_name is not None:
        state["profile"]["caller_name"] = " ".join(caller_name.split())
    if marital_status is not None:
        if marital_status not in _MARITAL:
            return state, "marital_status"
        state["profile"]["marital_status"] = marital_status
    if children is not None or (isinstance(child_count, int) and child_count > 0):
        incoming = list(children) if children is not None else []
        stated = child_count if isinstance(child_count, int) and child_count >= 0 else len(incoming)
        if (stated > 0 and len(incoming) < stated) or _missing_ages(incoming):
            return state, "children_ages"
        state["profile"]["children"] = [_normalize_child(child, i) for i, child in enumerate(incoming, start=1)]
    if state["profile"]["children"]:
        state["profile"]["ever_pregnant"] = True
    elif ever_pregnant is not None:
        state["profile"]["ever_pregnant"] = bool(ever_pregnant)

    _recompute(state)
    hold = _profile_hold(state)
    if hold:
        return state, hold
    if _on_profile_step(state):
        _advance(state)
        _recompute(state)
    elif state["active_qid"] == "ever_pregnant" and state["profile"]["ever_pregnant"] is not None:
        _advance(state)
        _recompute(state)
    return state, None


def apply_turn(state: dict, outcome: str, *, child_ref: str = "", note: str = "") -> dict:
    if outcome not in _OUTCOMES:
        raise ValueError(f"unknown outcome {outcome!r}")
    state = copy.deepcopy(state)
    if outcome == "clarify":
        return state
    if outcome in ("emergency", "caller_stop"):
        state["phase"] = "closing"
        state["close_reason"] = outcome
        return state
    if state["phase"] != "asking":
        return state
    if state["active_qid"] == "ever_pregnant":
        return state

    if state["pending_followup_idx"] == -1:
        return _resolve_which_child(state, child_ref, note, outcome)

    if isinstance(state["pending_followup_idx"], int) and state["pending_followup_idx"] >= 0:
        return _resolve_followup(state, outcome, note)

    spec = get_spec(state["active_qid"])
    if spec.kind == "profile":
        return state

    child_ids = _child_ids_for(state, spec, child_ref)
    _write_answer(state, spec.id, outcome, note=note, child_ids=child_ids)
    if outcome != "concern":
        _recompute(state)
        _advance(state)
        return state

    if _needs_which_child(state, spec, child_ref):
        state["pending_followup_idx"] = -1
        _enqueue_opens(state, spec)
        return state
    _enqueue_opens(state, spec)
    if spec.followups:
        state["pending_followup_idx"] = 0
        return state
    _advance(state)
    return state


def prompt_directive(state: dict) -> str:
    """One speakable instruction: the single Hindi question to ask next."""
    if state["phase"] != "asking":
        return "Say one short warm Hindi closing, then stop. Do not ask another screening question."
    if state["active_qid"] == "ever_pregnant":
        return _ask(_EVER_PREGNANT_HI)
    if state["pending_followup_idx"] == -1:
        return _ask(_WHICH_CHILD_HI)
    qid = state["active_qid"]
    spec = get_spec(qid)
    idx = state["pending_followup_idx"]
    if isinstance(idx, int) and idx >= 0:
        return _ask(FOLLOWUP_HI[spec.followups[idx]])
    return _ask(_question_hi(state, spec))


def _ask(hindi: str) -> str:
    return f"Ask only this one Hindi question, then wait: {hindi}"


def _scope_all_children(text: str) -> str:
    """Put every-child scope in the Hindi question itself."""
    if "आपके किसी भी बच्चे" in text:
        return text
    if "परिवार में" in text:
        if "आपके बच्चे" in text:
            return text.replace("आपके बच्चे", "आपके किसी भी बच्चे", 1)
        return text
    if "किसी बच्चे" in text:
        return text.replace("किसी बच्चे", "आपके किसी भी बच्चे", 1)
    if "बच्चे" in text:
        return text.replace("बच्चे", "आपके किसी भी बच्चे", 1)
    return "आपके किसी भी बच्चे — " + text


def _question_hi(state: dict, spec) -> str:
    if spec.id == "preg_thalassemia" and state["profile"]["marital_status"] == "unmarried":
        text = _UNMARRIED_THAL_HI
    elif spec.id == "fam_loss_gate" and _is_skipped(get_spec("fam_miscarriage"), state):
        text = _INFANT_ONLY_HI
    else:
        text = spec.question_hi
    if _is_child_scoped(spec) and len(state["profile"]["children"]) > 1:
        text = _scope_all_children(text)
    return text


def _on_profile_step(state: dict) -> bool:
    qid = state["active_qid"]
    if qid not in _EARLY:
        return False
    return get_spec(qid).kind == "profile"


def _profile_hold(state: dict) -> str | None:
    qid = state["active_qid"]
    profile = state["profile"]
    if qid == "intro_name" and not _valid_name(profile["caller_name"]):
        return "name_fragment"
    if qid == "intro_married" and profile["marital_status"] not in _MARITAL:
        return "marital_status"
    if qid == "intro_children" and _missing_ages(profile["children"]) and not profile["children"]:
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
    label = child.get("age_label_hi") or _age_label(age)
    return {
        "id": child.get("id") or f"child_{index}",
        "age_years": age,
        "age_label_hi": label,
        "sex": sex,
        "name": child.get("name"),
    }


def _age_label(age: float) -> str:
    if age < 1:
        months = int(round(age * 12))
        return f"{months} महीने"
    text = str(int(age)) if age == int(age) else str(age)
    return f"{text} साल"


def _children_resolved(state: dict) -> bool:
    profile = state["profile"]
    if profile["marital_status"] == "unmarried":
        return True
    if profile["children"]:
        return True
    return state["active_qid"] not in _EARLY


def _is_skipped(spec, state: dict) -> bool:
    return bool(_skip_hits(spec, state))


def _skip_hits(spec, state: dict) -> tuple[str, ...]:
    profile = state["profile"]
    children = profile["children"]
    marital = profile["marital_status"]
    ever = profile["ever_pregnant"]
    known = _children_resolved(state)
    hits: list[str] = []
    for rule in spec.skip_if:
        if rule == "no_children" and known and not children:
            hits.append(rule)
        elif rule == "unmarried" and marital == "unmarried":
            hits.append(rule)
        elif rule == "ever_pregnant_no" and ever is False:
            hits.append(rule)
        elif rule == "unmarried_not_pregnant" and marital == "unmarried" and ever is not True:
            hits.append(rule)
        elif rule == "all_under_10" and known and children and all(float(c["age_years"]) < 10 for c in children):
            hits.append(rule)
        elif rule == "no_child_5_plus" and known and not any(float(c["age_years"]) >= 5 for c in children):
            hits.append(rule)
        elif rule == "no_daughter_11_plus" and known and not any(
            c.get("sex") == "female" and float(c["age_years"]) >= 11 for c in children
        ):
            hits.append(rule)
        elif rule == "no_family_concern" and not _has_family_concern(state):
            hits.append(rule)
    return tuple(hits)


def _has_family_concern(state: dict) -> bool:
    for qid, answer in state["answers"].items():
        if qid == "fam_same_problem":
            continue
        if answer.get("state") == "concern" and (qid.startswith("fam_") or qid == "fam_loss_gate" or qid == "fam_conditions_gate"):
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
        if _is_skipped(spec, state):
            if answer and answer["state"] in _ANSWERED:
                continue
            state["answers"][spec.id] = _blank("not_applicable")
        elif answer and answer["state"] == "not_applicable":
            del state["answers"][spec.id]


def _write_answer(state: dict, qid: str, outcome: str, note: str, child_ids: list) -> None:
    prior = state["answers"].get(qid) or _blank(outcome)
    # A follow-up must not wipe a parent concern. Callers that resolve follow-ups
    # do not use this helper for that reason.
    state["answers"][qid] = _blank(
        outcome,
        note=note or prior.get("notes", ""),
        child_ids=child_ids or prior.get("child_ids") or [],
        followups=prior.get("followups") or {},
    )


def _is_child_scoped(spec) -> bool:
    if spec.kind in ("profile", "bridge", "open_capture"):
        return False
    if spec.group in ("birth", "body", "illness"):
        return True
    if spec.id.startswith("mod_"):
        return True
    return "no_children" in spec.skip_if


def _needs_which_child(state: dict, spec, child_ref: str) -> bool:
    return _is_child_scoped(spec) and len(state["profile"]["children"]) > 1 and not (child_ref or "").strip()


def _is_all_children(child_ref: str) -> bool:
    token = (child_ref or "").strip().casefold()
    return bool(token) and token in {item.casefold() for item in _ALL_CHILDREN}


def _child_ids_for(state: dict, spec, child_ref: str) -> list[str]:
    if not _is_child_scoped(spec):
        return []
    children = state["profile"]["children"]
    if _is_all_children(child_ref):
        return [child["id"] for child in children]
    if (child_ref or "").strip():
        return [_resolve_child(state, child_ref)]
    if len(children) == 1:
        return [children[0]["id"]]
    return []


def _resolve_child(state: dict, child_ref: str) -> str:
    ref = child_ref.strip()
    children = state["profile"]["children"]
    by_id = {child["id"]: child for child in children}
    if ref in by_id:
        return ref
    ordered = sorted(children, key=lambda child: float(child["age_years"]), reverse=True)
    if ref in _OLDEST and ordered:
        return ordered[0]["id"]
    if ref in _YOUNGEST and ordered:
        return ordered[-1]["id"]
    return ref


def _enqueue_opens(state: dict, spec) -> None:
    _recompute(state)
    fresh: list[str] = []
    for qid in spec.opens:
        opened = get_spec(qid)
        if not opened.enabled or _is_skipped(opened, state):
            if _is_skipped(opened, state):
                prior = state["answers"].get(qid)
                if not prior or prior["state"] not in _ANSWERED:
                    state["answers"][qid] = _blank("not_applicable")
            continue
        prior = state["answers"].get(qid)
        if prior and prior["state"] in _ANSWERED:
            continue
        if qid in state["module_queue"] or qid in fresh:
            continue
        fresh.append(qid)
    state["module_queue"] = fresh + list(state["module_queue"])


def _resolve_which_child(state: dict, child_ref: str, note: str, outcome: str = "") -> dict:
    qid = state["active_qid"]
    spec = get_spec(qid)
    answer = state["answers"].get(qid) or _blank("concern")
    ref = (child_ref or "").strip()
    if not ref:
        # "पता नहीं" must leave the which-child prompt. An empty ref with any
        # other outcome stays here so we do not guess a child.
        if outcome != "unknown":
            return state
    elif _is_all_children(ref):
        answer["child_ids"] = [child["id"] for child in state["profile"]["children"]]
    else:
        answer["child_ids"] = [_resolve_child(state, ref)]
    if note:
        answer["notes"] = note
    state["answers"][qid] = answer
    if spec.followups:
        state["pending_followup_idx"] = 0
        return state
    state["pending_followup_idx"] = None
    _advance(state)
    return state


def _resolve_followup(state: dict, outcome: str, note: str) -> dict:
    spec = get_spec(state["active_qid"])
    idx = state["pending_followup_idx"]
    fid = spec.followups[idx]
    answer = state["answers"].get(spec.id) or _blank("concern")
    followups = dict(answer.get("followups") or {})
    followups[fid] = {"state": outcome, "note": note or ""}
    answer["followups"] = followups
    if answer.get("state") not in _ANSWERED:
        answer["state"] = "concern"
    state["answers"][spec.id] = answer
    if idx + 1 < len(spec.followups):
        state["pending_followup_idx"] = idx + 1
        return state
    state["pending_followup_idx"] = None
    _advance(state)
    return state


def _advance(state: dict) -> None:
    state["pending_followup_idx"] = None
    if state["module_queue"]:
        state["active_qid"] = state["module_queue"].pop(0)
        return
    nxt = _next_qid(state, _spine_anchor(state["active_qid"]))
    if nxt is None:
        state["phase"] = "closing"
        state["close_reason"] = state["close_reason"] or "complete"
        return
    state["active_qid"] = nxt


def _spine_anchor(qid: str) -> str:
    if qid == "ever_pregnant":
        return qid
    ids = set(spine_ids())
    seen: set[str] = set()
    while qid and qid not in ids and qid not in seen:
        seen.add(qid)
        qid = _owner_of(qid) or ""
    return qid


def _owner_of(qid: str) -> str | None:
    for spec in active_specs():
        if qid in spec.opens:
            return spec.id
    return None


def _next_qid(state: dict, after_id: str) -> str | None:
    ids = spine_ids()
    if after_id == "ever_pregnant":
        rest = ids[ids.index("gate_pregnancy_thal"):]
    elif after_id in ids:
        rest = ids[ids.index(after_id) + 1:]
    else:
        rest = ids
    for qid in rest:
        if qid == "gate_pregnancy_thal" and _needs_ever_pregnant(state):
            return "ever_pregnant"
        if qid == "preg_thalassemia":
            if _ask_standalone_thal(state):
                return qid
            continue
        spec = get_spec(qid)
        if _is_skipped(spec, state):
            prior = state["answers"].get(qid)
            if not prior or prior["state"] not in _ANSWERED:
                state["answers"][qid] = _blank("not_applicable")
            continue
        return qid
    return None


def _needs_ever_pregnant(state: dict) -> bool:
    profile = state["profile"]
    if profile["ever_pregnant"] is not None:
        return False
    if profile["marital_status"] == "unmarried":
        return False
    if profile["children"]:
        return False
    return profile["marital_status"] in _MARITAL


def _ask_standalone_thal(state: dict) -> bool:
    """The compound pregnancy gate already includes thalassemia when it is asked."""
    profile = state["profile"]
    if profile["marital_status"] == "unmarried":
        return True
    return profile["ever_pregnant"] is False
