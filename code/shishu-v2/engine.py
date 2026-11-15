"""SHISHU v4 FormBharo engine: one pure, typed reducer over CallState.

No Hindi string and no hardcoded clinical question ID lives here: prompts come
from copy_hi and triage_catalog, routing reads QuestionSpec flags, and caller
inputs are never mutated.
"""

from __future__ import annotations

from typing import Any

import copy_hi
from models import (
    AnswerRecord,
    AnswerStatus,
    CallState,
    Child,
    ChildInput,
    ChildSex,
    CloseReason,
    CursorStep,
    FollowupSlot,
    HoldReason,
    MaritalStatus,
    NoteText,
    Phase,
    ProfilePatch,
    SideAction,
    SubStep,
    TurnCommand,
    TurnOutcome,
)
from triage_catalog import CATALOG, FOLLOWUP_HI, PROFILE_QIDS, QuestionSpec, get_spec, spec_or_none

MAX_STEP_ATTEMPTS = 2

_DEMOGRAPHIC_SKIPS = frozenset(
    {"unmarried", "about_to_marry", "no_children", "has_children", "ever_pregnant_no"}
)
_OUTCOME_TO_STATUS = {
    TurnOutcome.CLEAR: AnswerStatus.CLEAR,
    TurnOutcome.CONCERN: AnswerStatus.CONCERN,
    TurnOutcome.UNKNOWN: AnswerStatus.UNKNOWN,
}
_QUIET_OUTCOMES = (TurnOutcome.CLEAR, TurnOutcome.UNKNOWN)
_SPINE_POS = {spec.id: idx for idx, spec in enumerate(CATALOG)}
_SKIPPABLE_SPECS = tuple(spec for spec in CATALOG if spec.skip_if)
_EVER_PREGNANT_SPEC = next(spec for spec in CATALOG if spec.updates_ever_pregnant)
_INTRO_NAME, _INTRO_MARRIED, _INTRO_CHILDREN = "intro_name", "intro_married", "intro_children"
_PUNCT_NO_SPACE = ",.;:!?\u0964"


def _is_unmarried_or_engaged(st: CallState) -> bool:
    ms = st.profile.marital_status
    return ms is not None and ms.is_unmarried_or_engaged


def _children_resolved(st: CallState) -> bool:
    prof = st.profile
    return (
        bool(prof.children or prof.children_stated or _is_unmarried_or_engaged(st))
        or st.cursor.qid not in PROFILE_QIDS
    )


def _profile_satisfied(st: CallState, qid: str) -> bool:
    if qid == _INTRO_NAME:
        return bool(st.profile.caller_name)
    if qid == _INTRO_MARRIED:
        return st.profile.marital_status is not None
    return _children_resolved(st)


def _has_family_concern(st: CallState) -> bool:
    for qid, rec in st.answers.items():
        if rec.state is not AnswerStatus.CONCERN:
            continue
        spec = spec_or_none(qid)
        if spec is not None and spec.group == "family" and "no_family_concern" not in spec.skip_if:
            return True
    return False


_SKIP_RULES = {
    "no_children": lambda st: _children_resolved(st) and not st.profile.children,
    "has_children": lambda st: bool(st.profile.children),
    "unmarried": lambda st: st.profile.marital_status is MaritalStatus.UNMARRIED,
    "about_to_marry": lambda st: st.profile.marital_status is MaritalStatus.ABOUT_TO_MARRY,
    "ever_pregnant_no": lambda st: st.profile.ever_pregnant is False
    or (_is_unmarried_or_engaged(st) and st.profile.ever_pregnant is not True),
    "no_child_5_plus": lambda st: _children_resolved(st)
    and not any(c.age_years >= 5 for c in st.profile.children),
    "all_under_10": lambda st: _children_resolved(st)
    and not any(c.age_years >= 10 for c in st.profile.children),
    "no_daughter_11_plus": lambda st: _children_resolved(st)
    and not any(c.sex is ChildSex.FEMALE and c.age_years >= 11 for c in st.profile.children),
    "no_family_concern": lambda st: not _has_family_concern(st),
}


def _skip_hits(spec: QuestionSpec | None, st: CallState) -> tuple[str, ...]:
    if spec is None or not spec.skip_if:
        return ()
    return tuple(rule for rule in spec.skip_if if _SKIP_RULES[rule](st))


def _recompute_skips(st: CallState) -> None:
    fired = {rule for rule, pred in _SKIP_RULES.items() if pred(st)}
    for spec in _SKIPPABLE_SPECS:
        rec = st.answers.get(spec.id)
        hit = fired.intersection(spec.skip_if)
        if not hit:
            if rec is not None and rec.state is AnswerStatus.NOT_APPLICABLE:
                del st.answers[spec.id]
        elif not (rec is not None and rec.state.is_answered and not (hit & _DEMOGRAPHIC_SKIPS)):
            if rec is None or rec.state is not AnswerStatus.NOT_APPLICABLE:
                st.answers[spec.id] = AnswerRecord(state=AnswerStatus.NOT_APPLICABLE)


def _sync_next_qid(st: CallState) -> str | None:
    _recompute_skips(st)
    for spec in CATALOG:
        rec = st.answers.get(spec.id)
        if spec.is_profile:
            if not _profile_satisfied(st, spec.id):
                return spec.id
            continue
        if rec is None:
            return spec.id
        if rec.state is not AnswerStatus.NOT_APPLICABLE and not rec.state.is_answered:
            return spec.id
    return None


def _advance(st: CallState, *, record_answered: bool = True) -> None:
    if record_answered:
        st.last_answered_qid = st.cursor.qid
    nxt = _sync_next_qid(st)
    st.cursor = CursorStep(qid=nxt or st.cursor.qid)
    if nxt is None:
        st.phase = Phase.CLOSING
        st.close_reason = st.close_reason or CloseReason.COMPLETE


def _set_substep(st: CallState, substep: SubStep = SubStep.MAIN, idx: int = 0) -> None:
    st.cursor.substep = substep
    st.cursor.followup_idx = idx
    st.cursor.attempts = 0


def _bump_attempt(st: CallState) -> bool:
    st.cursor.attempts += 1
    if st.cursor.attempts < MAX_STEP_ATTEMPTS:
        return False
    st.cursor.attempts = 0
    return True


def _child_ids_for(st: CallState, spec: QuestionSpec | None, ref: str) -> list[str]:
    if spec is None or not spec.is_child_scoped:
        return []
    clean = " ".join((ref or "").split())
    kids = st.profile.children
    if clean == "all":
        return [child.id for child in st.profile.ordered_children]
    if clean in {child.id for child in kids}:
        return [clean]
    return [kids[0].id] if len(kids) == 1 and not clean else []


def _close_call(st: CallState, reason: CloseReason, note: str = "") -> CallState:
    st.phase = Phase.CLOSING
    st.close_reason = reason
    if note:
        st.emergency_note = note
    st.cursor = CursorStep(qid=st.cursor.qid)
    return st


def _append_note(rec: AnswerRecord, text: str, diagnostic: bool) -> AnswerRecord:
    if text:
        rec.note = rec.note.append(text, diagnostic=diagnostic)
    return rec


def _fill_slot(
    rec: AnswerRecord, fid: str, status: AnswerStatus, text: str, diagnostic: bool
) -> None:
    prior_note = rec.followups[fid].note if fid in rec.followups else NoteText()
    rec.followups = {
        **rec.followups,
        fid: FollowupSlot(state=status, note=prior_note.append(text, diagnostic=diagnostic)),
    }


def _slot_has_words(rec: AnswerRecord | None, fid: str) -> bool:
    return bool(rec and fid in rec.followups and rec.followups[fid].note.verbatim)


def _first_open_slot(spec: QuestionSpec, rec: AnswerRecord | None) -> str:
    first, rest = spec.followups[0], spec.followups[1:]
    if not rest or not _slot_has_words(rec, first):
        return first
    return next((fid for fid in rest if not _slot_has_words(rec, fid)), spec.followups[-1])


def _open_capture_kept(st: CallState, qid: str) -> bool:
    spec = spec_or_none(qid)
    rec = st.answers.get(qid)
    return bool(
        spec
        and spec.is_open_capture
        and rec is not None
        and rec.state is AnswerStatus.CONCERN
        and rec.verbatim_text
    )


def _normalize_child(child: ChildInput, index: int) -> Child:
    return Child(
        id=f"child_{index}",
        age_years=child.age_years,
        age_label_hi=copy_hi.age_label_hi(child.age_years),
        sex=child.sex,
        name=child.name,
    )


def new_state() -> CallState:
    return CallState()


def merge_transcript_chunk(prev: str, chunk: str) -> str:
    prev_s, chunk_s = str(prev or "").strip(), str(chunk or "").strip()
    if not prev_s or chunk_s.startswith(prev_s):
        return chunk_s
    if not chunk_s or prev_s.endswith(chunk_s):
        return prev_s
    sep = "" if (prev_s[-1].isspace() or chunk_s[0] in _PUNCT_NO_SPACE) else " "
    return prev_s + sep + chunk_s


def step_kind(state: CallState) -> str:
    spec, step = spec_or_none(state.cursor.qid), state.cursor
    if step.hold is not None or (spec is not None and spec.is_profile):
        return "profile"
    if step.substep is SubStep.WHICH_CHILD:
        return "which_child"
    if step.substep is SubStep.FOLLOWUP and spec is not None and step.followup_idx < len(spec.followups):
        return "followup"
    if spec is None:
        return "main"
    if spec.is_open_capture:
        return "close"
    return "bridge" if spec.is_bridge else "main"


def expect_yes_no(state: CallState) -> bool:
    spec = spec_or_none(state.cursor.qid)
    return bool(
        spec is not None
        and spec.allows_yes_no
        and state.phase is Phase.ASKING
        and state.cursor.substep is SubStep.MAIN
        and state.cursor.hold is None
    )


def active_prompt_hi(state: CallState) -> str:
    if state.phase is not Phase.ASKING:
        return ""
    step, spec = state.cursor, spec_or_none(state.cursor.qid)
    if step.hold is not None:
        return copy_hi.HOLD_HI[step.hold]
    if step.substep is SubStep.WHICH_CHILD:
        return copy_hi.which_child_hi(state.profile.children)
    if spec is None:
        return ""
    if step.substep is SubStep.FOLLOWUP and step.followup_idx < len(spec.followups):
        return FOLLOWUP_HI[spec.followups[step.followup_idx]]
    rec = state.answers.get(spec.id)
    if spec.is_open_capture and rec is not None and rec.state is AnswerStatus.CONCERN:
        return copy_hi.CLOSE_MORE_HI
    return copy_hi.question_hi_for(state.profile, spec)


def has_clinical_concern(state: CallState) -> bool:
    return any(
        rec.state is AnswerStatus.CONCERN
        and (spec := spec_or_none(qid)) is not None
        and spec.is_reportable
        for qid, rec in state.answers.items()
    )


def say_hindi(state: CallState) -> str:
    if state.phase is not Phase.ASKING:
        return copy_hi.closing_hi(state, has_clinical_concern=has_clinical_concern(state))
    return active_prompt_hi(state)


def followup_prompt_hi(state: CallState) -> str:
    spec, step = spec_or_none(state.cursor.qid), state.cursor
    if spec is None:
        return ""
    if step.substep is SubStep.FOLLOWUP and step.followup_idx < len(spec.followups):
        return FOLLOWUP_HI[spec.followups[step.followup_idx]]
    if step.substep is SubStep.MAIN and spec.followups:
        return FOLLOWUP_HI[spec.followups[0]]
    return ""


def last_answered_prompt_hi(state: CallState) -> tuple[str, str]:
    qid = state.last_answered_qid or ""
    spec = spec_or_none(qid)
    if spec is None or spec.is_bridge:
        return "", ""
    return qid, copy_hi.question_hi_for(state.profile, spec)


def children_roster(state: CallState) -> list[dict[str, Any]]:
    kids = state.profile.ordered_children
    if len(kids) <= 1:
        return []
    ranks = ["oldest", *(["middle"] * (len(kids) - 2)), "youngest"]
    return [
        {
            "id": child.id,
            "rank": rank,
            "age_label_hi": child.age_label_hi,
            "sex": child.sex,
            "name": child.name,
        }
        for rank, child in zip(ranks, kids)
    ]


def accepted_side_target(state: CallState, qid: str) -> str:
    target = (qid or "").strip()
    spec = spec_or_none(target)
    rec = state.answers.get(target)
    if not target or target == state.cursor.qid or spec is None or not spec.is_rewindable:
        return ""
    return target if rec is not None and rec.state.is_answered else ""


def is_answered_or_na(state: CallState, qid: str) -> bool:
    return bool(qid) and qid in state.answers


def has_answered_gates(state: CallState) -> bool:
    if any(
        rec.state.is_answered and (spec := spec_or_none(qid)) is not None and spec.is_routable
        for qid, rec in state.answers.items()
    ):
        return True
    cur = spec_or_none(state.cursor.qid)
    past_intro = state.phase is not Phase.ASKING or (
        cur is not None and not cur.is_profile and not cur.is_bridge
    )
    return past_intro and state.profile.ever_pregnant is not None


def close_technical_failure(state: CallState) -> CallState:
    fresh = state.model_copy(deep=True)
    return _close_call(fresh, CloseReason.TECHNICAL_FAILURE) if fresh.phase is Phase.ASKING else fresh


def quick_reply_command(outcome: AnswerStatus) -> TurnCommand:
    if outcome not in (AnswerStatus.CLEAR, AnswerStatus.CONCERN):
        raise ValueError(f"quick_reply requires CLEAR or CONCERN, got {outcome!r}")
    return TurnCommand(outcome=TurnOutcome(outcome.value))


def _compute_profile_hold(
    st: CallState, *, children_given: bool, attempted: bool, prev_hold: HoldReason | None
) -> HoldReason | None:
    prof, qid = st.profile, st.cursor.qid
    if prof.pending_child_count or prof.pending_children:
        return HoldReason.CHILDREN_AGES
    if qid == _INTRO_NAME and not prof.caller_name and (attempted or prev_hold is HoldReason.NAME_FRAGMENT):
        return HoldReason.NAME_FRAGMENT
    if qid == _INTRO_MARRIED and attempted and prof.marital_status is None:
        return HoldReason.MARITAL_STATUS
    if qid == _INTRO_CHILDREN and not _children_resolved(st):
        if children_given or attempted or prev_hold is HoldReason.CHILDREN_AGES:
            return HoldReason.CHILDREN_AGES
    return None


def _write_profile_fields(st: CallState, patch: ProfilePatch) -> bool:
    prof = st.profile
    was_unmarried_or_engaged = _is_unmarried_or_engaged(st)
    if patch.caller_name is not None:
        clean_name = " ".join(patch.caller_name.split())
        if not clean_name:
            st.cursor.hold = HoldReason.NAME_FRAGMENT
            return False
        prof.caller_name = clean_name

    children_given = patch.children is not None or patch.child_count is not None
    if patch.marital_status is not None:
        prof.marital_status = patch.marital_status
        if patch.marital_status.is_unmarried_or_engaged and not children_given:
            prof.children = []
            prof.pending_children = []
            prof.pending_child_count = None
            prof.children_stated = True
        elif was_unmarried_or_engaged and patch.marital_status is MaritalStatus.MARRIED and not children_given:
            prof.children_stated = False

    incoming = list(patch.children or ())
    stated = patch.child_count if patch.child_count is not None else len(incoming)
    normalized = [_normalize_child(child, idx) for idx, child in enumerate(incoming, start=1)]
    if children_given and stated > 0 and len(incoming) < stated:
        prof.pending_children = normalized
        prof.pending_child_count = stated
        _recompute_skips(st)
        st.cursor.hold = HoldReason.CHILDREN_AGES
        return False
    if children_given:
        prof.children = normalized
        prof.pending_children = []
        prof.pending_child_count = None
        prof.children_stated = True

    ever_rec = st.answers.get(_EVER_PREGNANT_SPEC.id)
    ever_answered = ever_rec is not None and ever_rec.state.is_answered
    if prof.children:
        prof.ever_pregnant = True
    elif patch.marital_status is not None and patch.marital_status.is_unmarried_or_engaged:
        prof.ever_pregnant = False
    elif was_unmarried_or_engaged and patch.marital_status is MaritalStatus.MARRIED and not ever_answered:
        prof.ever_pregnant = None
    return True


def _apply_profile_in_place(st: CallState, patch: ProfilePatch, attempted: bool) -> CallState:
    prev_hold = st.cursor.hold
    if not _write_profile_fields(st, patch):
        return st
    cur, spec = st.cursor.qid, spec_or_none(st.cursor.qid)
    _recompute_skips(st)
    if spec is not None and _skip_hits(spec, st):
        _advance(st, record_answered=False)
    children_given = patch.children is not None or patch.child_count is not None
    st.cursor.hold = _compute_profile_hold(
        st, children_given=children_given, attempted=attempted, prev_hold=prev_hold
    )
    if st.cursor.hold is not None:
        return st

    cur, spec = st.cursor.qid, spec_or_none(st.cursor.qid)
    top = _sync_next_qid(st)
    ever_settled = spec is not None and spec.updates_ever_pregnant and st.profile.ever_pregnant is not None
    profile_done = (cur in PROFILE_QIDS and _profile_satisfied(st, cur)) or ever_settled
    if top is not None and _SPINE_POS[top] < _SPINE_POS[cur]:
        if st.cursor.substep is not SubStep.MAIN and not _open_capture_kept(st, cur):
            st.answers.pop(cur, None)
        st.cursor = CursorStep(qid=top)
    elif profile_done and top != cur:
        _advance(st)
    if st.cursor.qid != cur:
        _sync_next_qid(st)
    return st


def apply_profile_patch(
    state: CallState, patch: ProfilePatch, *, attempted_on_profile: bool = False
) -> CallState:
    return _apply_profile_in_place(state.model_copy(deep=True), patch, attempted_on_profile)


def _write_gate_answer(
    st: CallState,
    spec: QuestionSpec,
    status: AnswerStatus,
    note: str,
    child_ids: list[str],
    followups: dict[str, FollowupSlot] | None,
    *,
    diagnostic: bool = False,
) -> None:
    prior = st.answers.get(spec.id)
    keep_prior = prior is not None and prior.state.is_answered and status is AnswerStatus.CONCERN
    rec = prior if keep_prior else AnswerRecord(state=status)
    rec.state = status
    if child_ids:
        rec.child_ids = list(dict.fromkeys([*rec.child_ids, *child_ids])) if keep_prior else list(child_ids)
    if followups is not None:
        rec.followups = dict(followups)
    st.answers[spec.id] = _append_note(rec, note, diagnostic)


def _revise_side_target(
    st: CallState, target: str, status: AnswerStatus, ref: str, text: str, diagnostic: bool
) -> None:
    spec = get_spec(target)
    if status is not AnswerStatus.CONCERN:
        st.answers[target] = AnswerRecord(state=status)
    else:
        _write_gate_answer(
            st, spec, status, text, _child_ids_for(st, spec, ref), None, diagnostic=diagnostic
        )
        rec = st.answers[target]
        if text and spec.followups:
            _fill_slot(rec, _first_open_slot(spec, rec), status, text, diagnostic)
    if spec.updates_ever_pregnant:
        st.profile.ever_pregnant = status is AnswerStatus.CONCERN
    active = spec_or_none(st.cursor.qid)
    if active is not None and _skip_hits(active, st):
        _advance(st, record_answered=False)
    else:
        _sync_next_qid(st)


def _rewind_to_target(st: CallState, target: str, stash_outcome: TurnOutcome | None) -> None:
    cur, spec = st.cursor.qid, spec_or_none(st.cursor.qid)
    if (
        stash_outcome is not None
        and st.cursor.substep is SubStep.MAIN
        and st.cursor.hold is None
        and spec is not None
    ):
        rec = st.answers.get(cur)
        if spec.updates_ever_pregnant:
            st.profile.ever_pregnant = False
        if spec.is_rewindable or (
            spec.is_open_capture and (rec is None or rec.state is not AnswerStatus.CONCERN)
        ):
            st.answers[cur] = AnswerRecord(state=_OUTCOME_TO_STATUS[stash_outcome])
    if cur != target and st.cursor.substep is not SubStep.MAIN and not _open_capture_kept(st, cur):
        st.answers.pop(cur, None)
    st.answers.pop(target, None)
    if get_spec(target).updates_ever_pregnant:
        st.profile.ever_pregnant = None
    st.cursor = CursorStep(qid=target)
    _sync_next_qid(st)


def _retract_active(st: CallState) -> None:
    qid, spec = st.cursor.qid, spec_or_none(st.cursor.qid)
    if spec is not None and spec.updates_ever_pregnant:
        st.profile.ever_pregnant = False
    if not _open_capture_kept(st, qid):
        st.answers[qid] = AnswerRecord(state=AnswerStatus.CLEAR)
    _advance(st)


def _advance_after_child(st: CallState, spec: QuestionSpec) -> None:
    rec = st.answers.get(spec.id)
    unfilled = [idx for idx, fid in enumerate(spec.followups) if not _slot_has_words(rec, fid)]
    _sync_next_qid(st)
    if unfilled:
        _set_substep(st, SubStep.FOLLOWUP, unfilled[0])
    else:
        _advance(st)


def _resolve_main_gate(st: CallState, spec: QuestionSpec, cmd: TurnCommand, utt: str) -> None:
    concern = cmd.outcome is TurnOutcome.CONCERN
    pre_answered = concern and cmd.followup_answered and bool(utt) and bool(spec.followups)
    pre_slots = (
        {
            spec.followups[0]: FollowupSlot(
                state=AnswerStatus.CONCERN,
                note=NoteText.from_text(utt, diagnostic=cmd.diagnostic),
            )
        }
        if pre_answered
        else None
    )
    keep_note = concern and (not spec.followups or cmd.followup_answered)
    child_ids = _child_ids_for(st, spec, cmd.child_ref)
    _write_gate_answer(
        st,
        spec,
        _OUTCOME_TO_STATUS[cmd.outcome],
        utt if keep_note else "",
        child_ids,
        pre_slots,
        diagnostic=cmd.diagnostic,
    )
    if spec.updates_ever_pregnant:
        st.profile.ever_pregnant = concern
    if not concern:
        _advance(st)
        return
    _sync_next_qid(st)
    if spec.is_child_scoped and len(st.profile.children) > 1 and not child_ids:
        _set_substep(st, SubStep.WHICH_CHILD)
    elif spec.followups and (pre_slots is None or len(spec.followups) > 1):
        _set_substep(st, SubStep.FOLLOWUP, 0 if pre_slots is None else 1)
    else:
        _advance(st)


def _resolve_which_child(st: CallState, spec: QuestionSpec, cmd: TurnCommand, utt: str) -> None:
    rec = st.answers.get(spec.id) or AnswerRecord(state=AnswerStatus.CONCERN)
    if utt and cmd.outcome in (TurnOutcome.CONCERN, TurnOutcome.UNKNOWN):
        _append_note(rec, utt, cmd.diagnostic)
    ids = _child_ids_for(st, spec, cmd.child_ref)
    resolved = bool(ids) and cmd.outcome is not TurnOutcome.CLEAR
    st.answers[spec.id] = rec
    if resolved or cmd.outcome is TurnOutcome.UNKNOWN or _bump_attempt(st):
        rec.child_ids = ids if resolved else []
        st.cursor.attempts = 0
        _advance_after_child(st, spec)


def _resolve_followup(st: CallState, spec: QuestionSpec, cmd: TurnCommand, utt: str) -> None:
    idx = st.cursor.followup_idx
    fid = spec.followups[idx]
    settled = cmd.outcome is TurnOutcome.UNKNOWN or bool(
        utt and cmd.outcome in (TurnOutcome.CONCERN, *_QUIET_OUTCOMES)
    )
    if not settled and not _bump_attempt(st):
        return
    rec = st.answers.get(spec.id) or AnswerRecord(state=AnswerStatus.CONCERN)
    rec.state = AnswerStatus.CONCERN
    st.cursor.attempts = 0
    slot_status = _OUTCOME_TO_STATUS[cmd.outcome] if settled else AnswerStatus.UNKNOWN
    _fill_slot(rec, fid, slot_status, utt if settled else "", cmd.diagnostic)
    st.answers[spec.id] = rec
    if idx + 1 < len(spec.followups):
        st.cursor.followup_idx = idx + 1
    else:
        _advance(st)


def _resolve_open_capture(st: CallState, spec: QuestionSpec, cmd: TurnCommand, utt: str) -> None:
    in_detail = st.cursor.substep is SubStep.FOLLOWUP
    if utt and (cmd.outcome is TurnOutcome.CONCERN or in_detail):
        rec = _append_note(
            st.answers.get(spec.id) or AnswerRecord(state=AnswerStatus.CONCERN), utt, cmd.diagnostic
        )
        if in_detail:
            _fill_slot(rec, spec.followups[st.cursor.followup_idx], AnswerStatus.CONCERN, utt, cmd.diagnostic)
        st.answers[spec.id] = rec
        st.cursor.close_rounds += 1
        _set_substep(st)
        return
    if cmd.outcome not in _QUIET_OUTCOMES and (not in_detail or not _bump_attempt(st)):
        if not in_detail:
            _set_substep(st, SubStep.FOLLOWUP)
        return
    if not _open_capture_kept(st, spec.id):
        if cmd.outcome in _QUIET_OUTCOMES:
            _write_gate_answer(st, spec, _OUTCOME_TO_STATUS[cmd.outcome], "", [], None)
        else:
            st.answers[spec.id] = AnswerRecord(
                state=AnswerStatus.UNKNOWN,
                followups={spec.followups[0]: FollowupSlot(state=AnswerStatus.UNKNOWN)},
            )
    _advance(st)


def _apply_side_action(
    st: CallState, cmd: TurnCommand, utt: str, *, skip_stash: bool
) -> tuple[bool, bool]:
    target = accepted_side_target(st, cmd.side_qid)
    if not target or cmd.side_action is SideAction.NONE:
        return False, False
    if cmd.side_action is SideAction.CLARIFY_MEANING:
        stash = cmd.outcome if (not skip_stash and cmd.outcome in _QUIET_OUTCOMES) else None
        _rewind_to_target(st, target, stash)
        return True, True
    side_spec = get_spec(target)
    active_spec = spec_or_none(st.cursor.qid)
    active_before = st.cursor.qid
    both_child_concerns = (
        active_spec is not None and active_spec.is_child_scoped and cmd.outcome is TurnOutcome.CONCERN
    )
    side_ref = "" if (not side_spec.is_child_scoped or both_child_concerns) else cmd.child_ref
    side_status = (
        AnswerStatus.CONCERN if cmd.side_action is SideAction.ADD_CONCERN else AnswerStatus.CLEAR
    )
    _revise_side_target(st, target, side_status, side_ref, utt, cmd.diagnostic)
    return True, st.phase is not Phase.ASKING or st.cursor.qid != active_before


def reduce_turn(state: CallState, cmd: TurnCommand) -> CallState:
    if state.phase is not Phase.ASKING:
        return state.model_copy(deep=True)
    st = state.model_copy(deep=True)
    if cmd.emergency:
        return _close_call(st, CloseReason.EMERGENCY, " ".join(cmd.utterance.split()))
    if cmd.caller_stop:
        return _close_call(st, CloseReason.CALLER_STOP)

    prev_kind, prev_qid, prev_hold = step_kind(st), st.cursor.qid, st.cursor.hold
    patch = cmd.profile_patch
    profile_ran = (patch is not None and not patch.is_empty) or cmd.profile_turn_attempted
    skip_active_from_profile = False
    if profile_ran:
        st = _apply_profile_in_place(st, patch or ProfilePatch(), cmd.profile_turn_attempted)
        skip_active_from_profile = (
            prev_kind == "profile"
            or st.cursor.qid != prev_qid
            or st.cursor.hold is not None
            or (prev_hold is not None and st.cursor.hold is None)
        )

    utt = " ".join(cmd.utterance.split())
    side_applied, side_moved = _apply_side_action(
        st, cmd, utt, skip_stash=skip_active_from_profile
    )
    if side_moved or skip_active_from_profile or st.cursor.hold is not None:
        return st

    spec, sub = spec_or_none(st.cursor.qid), st.cursor.substep
    if cmd.outcome is TurnOutcome.NOT_ABOUT_ACTIVE:
        if not (profile_ran or side_applied) and sub is not SubStep.MAIN and spec is not None:
            if _bump_attempt(st):
                if sub is SubStep.WHICH_CHILD:
                    _advance_after_child(st, spec)
                else:
                    _resolve_followup(st, spec, TurnCommand(outcome=TurnOutcome.UNKNOWN), "")
        return st
    if cmd.outcome is TurnOutcome.CLARIFY:
        return st
    if cmd.outcome is TurnOutcome.RETRACT or (
        sub is SubStep.WHICH_CHILD
        and cmd.outcome is TurnOutcome.CLEAR
        and not _child_ids_for(st, spec, cmd.child_ref)
    ):
        _retract_active(st)
        return st
    if spec is None or spec.is_profile:
        return st
    if spec.is_open_capture:
        _resolve_open_capture(st, spec, cmd, utt)
    elif sub is SubStep.WHICH_CHILD:
        _resolve_which_child(st, spec, cmd, utt)
    elif sub is SubStep.FOLLOWUP:
        _resolve_followup(st, spec, cmd, utt)
    elif spec.is_bridge:
        _write_gate_answer(st, spec, AnswerStatus.CLEAR, "", [], None)
        _advance(st)
    else:
        _resolve_main_gate(st, spec, cmd, utt)
    return st
