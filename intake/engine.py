"""Pure, pack-driven reactive intake reducer (`intake/engine.py`)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from intake.models import (AnswerRecord, AnswerStatus, CallState, CloseReason, CursorStep, FollowupSlot, NoteText, PackConfig, Phase,
                           ProfilePatch, QuestionKind, QuestionSpec, RosterItem, SubStep, TurnCommand, TurnOutcome)
MAX_STEP_ATTEMPTS = 2
_NA, _QUIET = AnswerStatus.NOT_APPLICABLE, (TurnOutcome.CLEAR, TurnOutcome.UNKNOWN)
_GATES = (QuestionKind.GATE, QuestionKind.ROUTING_GATE)
_BLOCKING, _PROFILE = (QuestionKind.PROFILE, QuestionKind.BRIDGE), "profile"
_STATUS = {o: AnswerStatus(o.value) for o in (*_QUIET, TurnOutcome.CONCERN, TurnOutcome.ANSWERED)}
_SHAPE = {"field", "roster_op", "any_of", "all_of"}  # a flip in any of these can invalidate a NOT_APPLICABLE
_OPS = {"eq": lambda v, t: v == t, "ne": lambda v, t: v != t, "in": lambda v, t: v in t, "not_in": lambda v, t: v not in t,
        "gte": lambda v, t: v is not None and v >= t, "lte": lambda v, t: v is not None and v <= t, "missing": lambda v, _t: v in (None, "")}
def merge_transcript_chunk(prev: str, chunk: str) -> str:
    a, b = str(prev or "").strip(), str(chunk or "").strip()
    if not a or b.startswith(a): return b
    if not b or a.endswith(b): return a
    return a + ("" if (a[-1].isspace() or b[0] in ",.;:!?\u0964") else " ") + b
def format_age_label(pack: PackConfig, age_years: float) -> str:
    fmt, years = pack.prompts.get("roster_format") or {}, int(age_years)
    unit, value = ("mo", int(round(age_years * 12))) if age_years < 1 else ("yr", years if years == age_years else round(age_years, 1))
    return f"{value} {fmt.get('month_unit' if age_years < 1 else 'year_unit', unit)}"
def roster_resolved(pack: PackConfig, state: CallState) -> bool:
    if pack.roster_cfg is None: return True
    return bool(state.roster or state.roster_stated or any(eval_rule(pack, state, r) for r in (pack.roster_cfg.get("skip_if") or ())))
def _field_value(state: CallState, field: str) -> Any:
    if field in state.profile: return state.profile[field]
    rec = state.answers.get(field)
    if rec is None or not rec.state.is_resolved: return None
    if rec.value is not None: return rec.value
    return rec.state is AnswerStatus.CONCERN if rec.state in (AnswerStatus.CONCERN, AnswerStatus.CLEAR) else None
def _roster_op(pack: PackConfig, state: CallState, spec: dict[str, Any]) -> bool:
    op, lo, hi, tag = spec["roster_op"], spec.get("min_age"), spec.get("max_age"), spec.get("tag")
    hits = [c for c in state.roster if (not tag or c.tag == tag) and (lo is None or c.age_years >= float(lo)) and (hi is None or c.age_years <= float(hi))]
    if op == "non_empty": return bool(state.roster)
    if op == "count_gt": return len(state.roster) > int(spec.get("value", 0))
    if op == "any_match": return bool(hits)
    return roster_resolved(pack, state) and (not state.roster if op == "is_empty" else not hits)
def _group_clear(pack: PackConfig, state: CallState, group: str, rule: str) -> bool:
    """True when no already-recorded CONCERN in `group` can be contradicted by asking `rule`'s question."""
    for qid, rec in state.answers.items():
        q = pack.by_id.get(qid)
        if rec.state is AnswerStatus.CONCERN and q is not None and q.group == group and rule != qid and rule not in q.skip_if: return False
    return True
def eval_rule(pack: PackConfig, state: CallState, rule: str | dict[str, Any]) -> bool:
    spec = (pack.rules.get(rule) or {}) if isinstance(rule, str) else rule
    if "any_of" in spec or "all_of" in spec: return (any if "any_of" in spec else all)(eval_rule(pack, state, r) for r in spec.get("any_of") or spec["all_of"])
    if "semantic_flag" in spec: return spec["semantic_flag"] in state.semantic_flags
    if "group_no_concern" in spec: return _group_clear(pack, state, spec["group_no_concern"], rule)
    if "roster_op" in spec: return _roster_op(pack, state, spec)
    if "field" not in spec: return False
    op, target, val = spec.get("op", ""), spec.get("value"), _field_value(state, spec["field"])
    if op in ("contains", "not_contains"): return (isinstance(val, (list, tuple, set, str)) and target in val) is (op == "contains")
    return bool(_OPS[op](val, target)) if op in _OPS else False
def _selected(state: CallState, item_ids: Sequence[str] | None) -> list[RosterItem]:
    if item_ids is None: return list(state.roster)
    wanted = {cid.strip() for cid in item_ids if cid and cid.strip()}
    return [c for c in state.ordered_roster if c.id in wanted] if wanted else []
def roster_rows(state: CallState) -> list[dict[str, Any]]:
    items = state.ordered_roster
    if len(items) < 2: return []
    def rank(c: RosterItem) -> str:
        # Equal-age twins must never be told apart by a rank the model could then quote back as truth.
        if sum(x.age_years == c.age_years for x in items) > 1: return "same_age"
        return "oldest" if c.age_years == items[0].age_years else ("youngest" if c.age_years == items[-1].age_years else "middle")
    return [{"id": c.id, "rank": rank(c), "age_label": c.age_label, "tag": c.tag, "name": c.name} for c in items]
def roster_label_patient(pack: PackConfig, state: CallState, item_ids: Sequence[str] | None = None) -> str:
    fmt, tags = pack.prompts.get("roster_format") or {}, (pack.prompts.get("roster_format") or {}).get("tag_labels") or {}
    labels = (" ".join(p for p in (c.name or "", c.age_label, str(tags.get(c.tag, ""))) if p) for c in _selected(state, item_ids))
    return str(fmt.get("join", " · ")).join(lbl for lbl in labels if lbl)
def roster_label_clinician(state: CallState, item_ids: Sequence[str] | None = None) -> str:
    return ", ".join(f"{c.id} ({c.tag}, age {c.age_years}" + (f", {c.name})" if c.name else ")") for c in _selected(state, item_ids))
def which_item_prompt(pack: PackConfig, state: CallState) -> str:
    w, ordered = (pack.prompts.get("roster_format") or {}).get("which_item") or {}, state.ordered_roster
    parts = []
    for i, c in enumerate(ordered):
        fem, name = c.tag == "female", (c.name or "").strip()
        gender = "female" if fem else ("male" if name or c.tag == "male" else "other")
        parts.append(str(w.get(f"named_{gender}" if name else f"unnamed_{gender}", "{age} {name}" if name else "{age}")).format(age=c.age_label, name=name)
                     + (str(w.get("middle_female" if fem else "middle_male", "")) if len(ordered) == 3 and i == 1 else ""))
    # Equal-age twins render identically, so a repeated label is ambiguous out loud: number only the collisions.
    ords, counts, seen = list(w.get("ordinals") or ()), {p: parts.count(p) for p in parts}, {}
    for i, p in enumerate(parts):
        if counts[p] < 2: continue
        k = seen.get(p, 0)
        seen[p] = k + 1
        parts[i] = f"{ords[k]} {p}" if k < len(ords) else f"{p} #{k + 1}"
    listed = ", ".join(parts[:-1]) + str(w.get("or_joiner", ", or ")) + parts[-1] if len(parts) >= 2 else (parts[0] if parts else "")
    return str(w.get("template", "{items}")).format(items=listed)
def question_say(pack: PackConfig, state: CallState, spec: QuestionSpec) -> str:
    return next((v.say for v in spec.variants if eval_rule(pack, state, v.when)), spec.say)
def active_prompt(pack: PackConfig, state: CallState) -> str:
    if state.phase is not Phase.ASKING: return ""
    step, spec = state.cursor, pack.by_id.get(state.cursor.qid)
    if step.hold_slot:
        held = pack.roster_cfg if step.hold_slot == "roster" else pack.profile_slots.get(step.hold_slot)
        return str((held or {}).get("reask") or (spec.reask if spec else "") or (question_say(pack, state, spec) if spec else ""))
    if spec is None: return ""
    if step.substep is SubStep.WHICH_ITEM: return which_item_prompt(pack, state)
    if step.substep is SubStep.FOLLOWUP and step.followup_idx < len(spec.followups): return str(pack.followups.get(spec.followups[step.followup_idx], ""))
    more = spec.kind is QuestionKind.OPEN_CAPTURE and (r := state.answers.get(spec.id)) is not None and r.state is AnswerStatus.CONCERN
    return str(pack.prompts.get("open_capture_more", spec.say)) if more else question_say(pack, state, spec)
def followup_prompt(pack: PackConfig, state: CallState) -> str:
    spec, step = pack.by_id.get(state.cursor.qid), state.cursor
    if spec is None or not spec.followups or spec.kind is QuestionKind.OPEN_CAPTURE: return ""
    idx = step.followup_idx if step.substep is SubStep.FOLLOWUP else 0
    return str(pack.followups.get(spec.followups[idx], "")) if idx < len(spec.followups) else ""
def last_answered_prompt(pack: PackConfig, state: CallState) -> tuple[str, str]:
    spec = pack.by_id.get(state.last_answered_qid or "")
    return ("", "") if (spec is None or spec.kind is QuestionKind.BRIDGE) else (spec.id, question_say(pack, state, spec))
def recorded_concerns_summary(pack: PackConfig, state: CallState) -> list[str]:
    scored = [(qid, r, pack.by_id.get(qid)) for qid, r in state.answers.items() if r.state is AnswerStatus.CONCERN]
    lines = [" — ".join(b for b in (s.stem or s.say, roster_label_patient(pack, state, r.item_ids) if r.item_ids else "", r.patient_safe_text or r.choice_label) if b)
             for _qid, r, s in scored if s is not None and s.is_reportable]
    return [line for line in lines if line]
def has_clinical_concern(pack: PackConfig, state: CallState) -> bool:
    return any(r.state is AnswerStatus.CONCERN and (s := pack.by_id.get(q)) is not None and s.is_reportable for q, r in state.answers.items())
def say_text(pack: PackConfig, state: CallState) -> str:
    if state.phase is Phase.ASKING: return active_prompt(pack, state)
    reason, closings = state.close_reason or CloseReason.COMPLETE, pack.prompts.get("closings") or {}
    key = reason.value if reason is not CloseReason.COMPLETE else ("complete_concern" if has_clinical_concern(pack, state) else "complete_clear")
    name = next((str(state.profile[s]).strip() for s in pack.profile_slots if state.profile.get(s)), "") or str(state.profile.get("caller_name") or state.profile.get("patient_name") or "").strip()
    hon = str(pack.prompts.get("honorific") or "").strip()
    return str(closings.get(key) or closings.get("complete_clear") or "").format(who=((f"{name} {hon}, " if hon else f"{name}, ") if name else ""))
def step_kind(pack: PackConfig, state: CallState) -> str:
    spec, step = pack.by_id.get(state.cursor.qid), state.cursor
    if step.hold_slot or (spec is not None and spec.kind is QuestionKind.PROFILE): return _PROFILE
    if step.substep is SubStep.WHICH_ITEM: return "which_item"
    if step.substep is SubStep.FOLLOWUP and spec is not None and step.followup_idx < len(spec.followups): return "followup"
    if spec is None: return "main"
    return "close" if spec.kind is QuestionKind.OPEN_CAPTURE else ("bridge" if spec.kind is QuestionKind.BRIDGE else "main")
def quick_replies(pack: PackConfig, state: CallState) -> tuple[bool, list[dict[str, str]]]:
    spec, step = pack.by_id.get(state.cursor.qid), state.cursor
    if state.phase is not Phase.ASKING or step.hold_slot or step.substep is not SubStep.MAIN or spec is None: return False, []
    if spec.allows_yes_no:
        buttons = (pack.meta.get("ui") or {}).get("yes_no_buttons") or pack.prompts.get("yes_no_buttons") or ()
        return True, [{"value": str(b["value"]), "hi": str(b.get("hi", "")), "en": str(b.get("en", ""))} for b in buttons]
    if spec.kind is QuestionKind.SINGLE_CHOICE and spec.quick_options and spec.options: return False, [{"value": k, "hi": o.label, "en": o.en} for k, o in spec.options.items()]
    return False, []
def quick_reply_command(pack: PackConfig, state: CallState, raw_value: str) -> TurnCommand:
    val, spec = str(raw_value or "").strip(), pack.by_id.get(state.cursor.qid)
    if spec is not None and val in spec.options: return TurnCommand(outcome=TurnOutcome.ANSWERED, value=val)
    if val in (TurnOutcome.CLEAR, TurnOutcome.CONCERN, TurnOutcome.UNKNOWN): return TurnCommand(outcome=TurnOutcome(val))
    raise ValueError(f"Invalid quick reply {raw_value!r} for {state.cursor.qid!r}")
def close_technical_failure(state: CallState) -> CallState:
    st = state.model_copy(deep=True)
    if st.phase is Phase.ASKING: st.phase, st.close_reason, st.cursor = Phase.CLOSING, CloseReason.TECHNICAL_FAILURE, CursorStep(qid=st.cursor.qid)
    return st
def _is_unsatisfied(pack: PackConfig, state: CallState, spec: QuestionSpec) -> bool:
    if state.answers.get(spec.id) is not None: return False
    if spec.kind is QuestionKind.PROFILE:
        slot = spec.requires_slot
        return not (roster_resolved(pack, state) if slot in ("roster", (pack.roster_cfg or {}).get("key")) else state.profile.get(slot) not in (None, "", [], {}))
    return spec.is_routable or spec.kind is QuestionKind.BRIDGE
def _imply(pack: PackConfig, state: CallState, *, roster_given: bool = False) -> None:
    explicit = {q.saves_to_profile for q in pack.questions if q.saves_to_profile and (r := state.answers.get(q.id)) and r.state.is_resolved}
    for imp in pack.implications:
        for k in imp.get("default_profile") or {}:
            if k not in explicit: state.profile.pop(k, None)
    for imp in pack.implications:
        if not all(eval_rule(pack, state, r) for r in imp.get("when") or ()): continue
        state.profile.update(imp.get("set_profile") or {})
        for k, v in (imp.get("default_profile") or {}).items(): state.profile.setdefault(k, v)
        if imp.get("clear_roster_if_unprovided") and not roster_given and not state.roster_stated: state.roster, state.pending_roster, state.pending_roster_count = [], [], None
def _resync(pack: PackConfig, state: CallState) -> None:
    for spec in pack.questions:
        rec, hits = state.answers.get(spec.id), [r for r in spec.skip_if if eval_rule(pack, state, r)]
        if not hits and rec is not None and rec.state is _NA: del state.answers[spec.id]
        elif hits and (rec is None or rec.state is _NA or any(_SHAPE & set(pack.rules.get(r) or {}) for r in hits)): state.answers[spec.id] = AnswerRecord(state=_NA)
def _advance(pack: PackConfig, state: CallState, cmd: TurnCommand | None = None, *, record: bool = True) -> CallState:
    if record and state.cursor.qid: state.last_answered_qid = state.cursor.qid
    _resync(pack, state)
    free = sorted((q for q in pack.questions if _is_unsatisfied(pack, state, q)), key=lambda q: 1 if q.kind is QuestionKind.OPEN_CAPTURE else 0)
    if not free:
        state.phase, state.close_reason = Phase.CLOSING, state.close_reason or CloseReason.COMPLETE
        state.cursor = CursorStep(qid=state.cursor.qid)
        return state
    head, routable = free[0], [q for q in free if q.kind is not QuestionKind.OPEN_CAPTURE]
    if cmd is not None and cmd.jump_topic and cmd.next_qid_probs and routable and not any(q.kind in _BLOCKING for q in free):
        best = max(routable, key=lambda q: cmd.next_qid_probs.get(q.id, -1.0))
        head = best if cmd.next_qid_probs.get(best.id, -1.0) > 0 else head
    state.cursor = CursorStep(qid=head.id)
    return state
def new_state(pack: PackConfig) -> CallState: return _advance(pack, CallState(), record=False)

def apply_profile_patch(pack: PackConfig, state: CallState, patch: ProfilePatch, *, attempted_on_profile: bool = False) -> CallState:
    st, step, spec = state.model_copy(deep=True), state.cursor, pack.by_id.get(state.cursor.qid)
    on_profile = spec is not None and spec.kind is QuestionKind.PROFILE
    slots = {k: (" ".join(v.split()) if isinstance(v, str) else v) for k, v in patch.slots.items()}
    st.profile.update({k: v for k, v in slots.items() if v not in (None, "")})
    for k, clean in slots.items():
        for q in pack.questions:
            if clean not in (None, "") and q.saves_to_profile == k and q.kind is QuestionKind.SINGLE_CHOICE and clean in q.options:
                st.answers[q.id] = AnswerRecord(state=AnswerStatus.ANSWERED, value=clean, choice_label=q.options[clean].label)
    roster_given = patch.roster is not None or patch.roster_count is not None
    if roster_given:
        typed = [RosterItem(id=f"item_{i}", age_years=float(r.age_years), tag=r.tag, name=r.name, age_label=format_age_label(pack, float(r.age_years))) for i, r in enumerate(patch.roster or (), 1)]
        given, total = (typed if patch.roster is not None else list(st.pending_roster or st.roster)), (len(typed) if patch.roster_count is None else patch.roster_count)
        if 0 < len(given) < total: st.pending_roster, st.pending_roster_count = given, total
        else: st.roster, st.roster_stated, st.pending_roster, st.pending_roster_count = ([] if total == 0 else given), True, [], None
    _imply(pack, st, roster_given=roster_given)
    _resync(pack, st)
    hold = None
    if on_profile and spec is not None and _is_unsatisfied(pack, st, spec) and (attempted_on_profile or roster_given):
        hold = "roster" if st.pending_roster else ("roster" if spec.requires_slot in ("roster", (pack.roster_cfg or {}).get("key")) else spec.requires_slot)
    if hold:
        st.cursor = CursorStep(qid=step.qid, hold_slot=hold)
        return st
    st.cursor.hold_slot = None
    pos = {q.id: i for i, q in enumerate(pack.questions)}
    if (top := next((q.id for q in pack.questions if _is_unsatisfied(pack, st, q)), None)) and pos[top] < pos.get(step.qid, -1):
        st.cursor = CursorStep(qid=top)
        return st
    done = st.answers.get(step.qid)
    settled = (done is not None and (done.state is _NA or done.state.is_resolved)) or (on_profile and spec is not None and not _is_unsatisfied(pack, st, spec))
    return _advance(pack, st, record=on_profile) if settled else st
def _ids(state: CallState, ref_raw: str) -> list[str]:
    ref, ids = " ".join((ref_raw or "").split()), [c.id for c in state.roster]
    if not ids: return []
    if ref == "all": return [c.id for c in state.ordered_roster]
    return [ref] if ref in ids else (ids if len(ids) == 1 and not ref else [])
def _bump(step: CursorStep) -> bool:
    step.attempts = 0 if step.attempts + 1 >= MAX_STEP_ATTEMPTS else step.attempts + 1
    return step.attempts == 0
def _read_answer(spec: QuestionSpec, cmd: TurnCommand) -> tuple[AnswerStatus, Any, str] | None:
    kind, val = spec.kind, cmd.value
    if kind in _GATES: return (_STATUS[cmd.outcome], None, "") if cmd.outcome in (TurnOutcome.CONCERN, *_QUIET) else None
    if kind is QuestionKind.SINGLE_CHOICE and isinstance(val, str) and val in spec.options:
        return (AnswerStatus.CONCERN if val in (spec.concern_if.get("in") or ()) else AnswerStatus.ANSWERED), val, spec.options[val].label
    raw = (val if isinstance(val, (list, tuple, set)) else ([val] if val is not None else [])) if kind is QuestionKind.MULTI_CHOICE else []
    if keys := [k for k in raw if isinstance(k, str) and k in spec.options]:
        excl = [k for k in keys if spec.options[k].exclusive]
        chosen = excl[:1] if excl else keys
        return (AnswerStatus.CLEAR if excl else AnswerStatus.CONCERN), chosen, "; ".join(spec.options[k].label for k in chosen)
    if kind is not QuestionKind.SCORE: return None
    try: score = float(val.get("score") if isinstance(val, dict) else val)
    except (TypeError, ValueError): return None
    lvl, floor = (int(val.get("level", round(score))) if isinstance(val, dict) else int(round(score))), spec.concern_if.get("min_score")
    return (AnswerStatus.CONCERN if floor is not None and score >= float(floor) else AnswerStatus.ANSWERED), round(score, 2), spec.score_rubric[max(0, min(len(spec.score_rubric) - 1, lvl))]
def _note_concern(pack: PackConfig, st: CallState, step: CursorStep, spec: QuestionSpec, cmd: TurnCommand, rec: AnswerRecord, utt: str, *, stated: bool = False) -> dict[str, Any]:
    extra: dict[str, Any] = {}
    if spec.roster_scoped:
        # A mother co-reporting two concerns for the same child must keep the child she just named.
        if ids := _ids(st, cmd.item_ref): extra["item_ids"] = list(dict.fromkeys([*rec.item_ids, *ids]))
    if utt:
        if stated or not spec.followups or cmd.followup_answered: extra["note"] = rec.note.append(utt, redacted=cmd.redacted)
        if spec.followups and (stated or cmd.followup_answered):
            fid = next((f for f in spec.followups if f not in rec.followups or not rec.followups[f].note.verbatim), spec.followups[0])
            extra["followups"] = {**rec.followups, fid: FollowupSlot(state=AnswerStatus.CONCERN, note=NoteText.from_text(utt, redacted=cmd.redacted))}
    return extra
def _apply_stated(pack: PackConfig, st: CallState, cmd: TurnCommand, utt: str) -> bool:
    step = st.cursor
    for target, out in cmd.stated.items():
        spec = pack.by_id.get(target)
        if spec is None or not spec.is_routable or target == step.qid or out not in (TurnOutcome.CONCERN, TurnOutcome.CLEAR, TurnOutcome.CLARIFY): continue
        if out is TurnOutcome.CLARIFY:
            if step.substep is SubStep.MAIN and not step.hold_slot and cmd.outcome in _QUIET: st.answers[step.qid] = AnswerRecord(state=_STATUS[cmd.outcome])
            st.answers.pop(target, None)
            st.cursor = CursorStep(qid=target)
            _resync(pack, st)
            return True
        status = _STATUS[out]
        rec = AnswerRecord(state=status) if status is AnswerStatus.CLEAR else (st.answers.get(target) or AnswerRecord(state=status))
        extra = _note_concern(pack, st, step, spec, cmd, rec, utt, stated=True) if status is AnswerStatus.CONCERN else {}
        st.answers[target] = rec.model_copy(update={"state": status} | extra)
        if spec.saves_to_profile:
            st.profile[spec.saves_to_profile] = status is AnswerStatus.CONCERN
            _imply(pack, st)
    _resync(pack, st)
    return False
def _which_item(pack: PackConfig, st: CallState, cmd: TurnCommand, spec: QuestionSpec, utt: str) -> CallState:
    rec = st.answers.get(spec.id) or AnswerRecord(state=AnswerStatus.CONCERN)
    # A named child the model could not map still carries her words; a meta question about the notes never does.
    keep = cmd.outcome in (TurnOutcome.CONCERN, TurnOutcome.ANSWERED, TurnOutcome.UNKNOWN)
    note = rec.note.append(utt, redacted=cmd.redacted) if (utt and keep) else rec.note
    rec, ids = rec.model_copy(update={"note": note}), _ids(st, cmd.item_ref) if cmd.outcome is not TurnOutcome.NOT_ABOUT_ACTIVE else []
    if not (ids or cmd.outcome is TurnOutcome.UNKNOWN or _bump(st.cursor)):
        st.answers[spec.id] = rec
        return st
    rec = st.answers[spec.id] = rec.model_copy(update={"item_ids": ids})
    open_f = [i for i, f in enumerate(spec.followups) if f not in rec.followups or not rec.followups[f].note.verbatim]
    if not open_f: return _advance(pack, st, cmd)
    st.cursor = CursorStep(qid=spec.id, substep=SubStep.FOLLOWUP, followup_idx=open_f[0])
    return st
def _followup(pack: PackConfig, st: CallState, cmd: TurnCommand, spec: QuestionSpec, utt: str) -> CallState:
    # Any classified outcome settles the slot: on a live multimodal turn the decision model reads the audio and
    # returns concern/clear/answered while the transcript is still empty, so an absent `utt` must not force a retry.
    settled = cmd.outcome in _STATUS or (bool(utt) and cmd.outcome is not TurnOutcome.NOT_ABOUT_ACTIVE)
    if not settled and not _bump(st.cursor): return st
    rec, fid = st.answers.get(spec.id) or AnswerRecord(state=AnswerStatus.CONCERN), spec.followups[st.cursor.followup_idx]
    note = (rec.followups[fid].note if fid in rec.followups else NoteText()).append(utt if settled else "", redacted=cmd.redacted)
    slot = _STATUS.get(cmd.outcome, AnswerStatus.CONCERN) if settled else AnswerStatus.UNKNOWN
    st.answers[spec.id] = rec.model_copy(update={"followups": {**rec.followups, fid: FollowupSlot(state=slot, note=note)}})
    nxt = st.cursor.followup_idx + 1
    if nxt < len(spec.followups):
        st.cursor = CursorStep(qid=spec.id, substep=SubStep.FOLLOWUP, followup_idx=nxt)
        return st
    return _advance(pack, st, cmd)
def _open_capture(pack: PackConfig, st: CallState, cmd: TurnCommand, spec: QuestionSpec, utt: str) -> CallState:
    rec = st.answers.get(spec.id)
    if utt and cmd.outcome in (TurnOutcome.CONCERN, TurnOutcome.ANSWERED, TurnOutcome.NOT_ABOUT_ACTIVE):
        base = rec or AnswerRecord(state=AnswerStatus.CONCERN)
        st.answers[spec.id], st.last_answered_qid = base.model_copy(update={"state": AnswerStatus.CONCERN, "note": base.note.append(utt, redacted=cmd.redacted)}), spec.id
        st.cursor = CursorStep(qid=spec.id, open_rounds=st.cursor.open_rounds + 1)
        return st
    if cmd.outcome not in _QUIET and not _bump(st.cursor):
        st.cursor = CursorStep(qid=spec.id, substep=SubStep.FOLLOWUP) if spec.followups else st.cursor
        return st
    if not (rec is not None and rec.state is AnswerStatus.CONCERN and rec.verbatim_text): st.answers[spec.id] = AnswerRecord(state=_STATUS.get(cmd.outcome, AnswerStatus.UNKNOWN))
    return _advance(pack, st, cmd)
def _bridge(pack: PackConfig, st: CallState, cmd: TurnCommand, spec: QuestionSpec, utt: str) -> CallState:
    st.answers[spec.id] = AnswerRecord(state=AnswerStatus.CLEAR)
    return _advance(pack, st, cmd)
def _scored(pack: PackConfig, st: CallState, cmd: TurnCommand, spec: QuestionSpec, utt: str) -> CallState:
    parsed = _read_answer(spec, cmd)
    if parsed is None and cmd.outcome is not TurnOutcome.UNKNOWN and not _bump(st.cursor): return st
    status, val, label = parsed or (AnswerStatus.UNKNOWN, None, "")
    prior = st.answers.get(spec.id)
    rec = prior if (prior is not None and prior.state.is_resolved and status is AnswerStatus.CONCERN) else AnswerRecord(state=status)
    rec = st.answers[spec.id] = rec.model_copy(update={"state": status, "value": val, "choice_label": label} | (_note_concern(pack, st, st.cursor, spec, cmd, rec, utt) if status is AnswerStatus.CONCERN else {}))
    if spec.saves_to_profile and status is not AnswerStatus.UNKNOWN:
        st.profile[spec.saves_to_profile] = val if spec.kind is QuestionKind.SINGLE_CHOICE else (status is AnswerStatus.CONCERN)
        _imply(pack, st)
    if status is not AnswerStatus.CONCERN: return _advance(pack, st, cmd)
    if spec.roster_scoped and len(st.roster) > 1 and not rec.item_ids: st.cursor = CursorStep(qid=spec.id, substep=SubStep.WHICH_ITEM)
    elif spec.followups and (not cmd.followup_answered or len(spec.followups) > 1): st.cursor = CursorStep(qid=spec.id, substep=SubStep.FOLLOWUP, followup_idx=1 if (cmd.followup_answered and utt) else 0)
    else: return _advance(pack, st, cmd)
    return st
_BY_SUBSTEP = {SubStep.WHICH_ITEM: _which_item, SubStep.FOLLOWUP: _followup}
_BY_KIND = {QuestionKind.OPEN_CAPTURE: _open_capture, QuestionKind.BRIDGE: _bridge}

def reduce_turn(pack: PackConfig, state: CallState, cmd: TurnCommand) -> CallState:
    st = state.model_copy(deep=True)
    if st.phase is not Phase.ASKING: return st
    utt = " ".join(cmd.utterance.split())
    if cmd.emergency or cmd.caller_stop:
        st.phase, st.close_reason = Phase.CLOSING, CloseReason.EMERGENCY if cmd.emergency else CloseReason.CALLER_STOP
        st.cursor, st.emergency_note = CursorStep(qid=st.cursor.qid), (utt if (cmd.emergency and utt) else st.emergency_note)
        return st
    if cmd.semantic_flags:
        st.semantic_flags = list(dict.fromkeys([*st.semantic_flags, *cmd.semantic_flags]))
        _resync(pack, st)
    prev_qid, prev_hold, prev_kind = st.cursor.qid, st.cursor.hold_slot, step_kind(pack, st)
    profile_ran, skip_active = (cmd.profile_patch is not None and not cmd.profile_patch.is_empty) or cmd.profile_turn_attempted, False
    if profile_ran:
        st = apply_profile_patch(pack, st, cmd.profile_patch or ProfilePatch(), attempted_on_profile=cmd.profile_turn_attempted)
        skip_active = prev_kind == _PROFILE or st.cursor.qid != prev_qid or st.cursor.hold_slot is not None or (prev_hold is not None and st.cursor.hold_slot is None)
    rewound = _apply_stated(pack, st, cmd, utt)
    if rewound or skip_active or st.cursor.hold_slot is not None:
        cur = st.answers.get(st.cursor.qid)
        return _advance(pack, st, cmd, record=False) if (cur is not None and (cur.state.is_resolved or cur.state is _NA)) else st
    spec, step = pack.by_id.get(st.cursor.qid), st.cursor
    if spec is None or spec.kind is QuestionKind.PROFILE or cmd.outcome is TurnOutcome.CLARIFY: return st
    if (cur := st.answers.get(spec.id)) is not None and cur.state is _NA: return _advance(pack, st, cmd, record=False)
    # A spontaneous remark at the closing open-capture is a finding, not a miss: it must reach `_open_capture`.
    if cmd.outcome is TurnOutcome.NOT_ABOUT_ACTIVE and (profile_ran or cmd.stated): return st
    if cmd.outcome is TurnOutcome.NOT_ABOUT_ACTIVE and step.substep is SubStep.MAIN and not (spec.kind is QuestionKind.OPEN_CAPTURE and utt):
        # An off-topic turn is still a miss: burn the attempt so the question can never repeat forever.
        if not _bump(st.cursor): return st
        # Mirror `_open_capture`: a spoken finding already on record is never flattened to a bare UNKNOWN.
        prior = st.answers.get(spec.id)
        if not (prior is not None and prior.state is AnswerStatus.CONCERN and prior.verbatim_text): st.answers[spec.id] = AnswerRecord(state=AnswerStatus.UNKNOWN)
        return _advance(pack, st, cmd)
    if cmd.outcome is TurnOutcome.RETRACT or (step.substep is SubStep.WHICH_ITEM and cmd.outcome is TurnOutcome.CLEAR and not _ids(st, cmd.item_ref)):
        prior = st.answers.get(spec.id)
        if not (spec.kind is QuestionKind.OPEN_CAPTURE and prior is not None and prior.state is AnswerStatus.CONCERN and prior.verbatim_text): st.answers[spec.id] = AnswerRecord(state=AnswerStatus.CLEAR)
        if spec.saves_to_profile:
            st.profile[spec.saves_to_profile] = False
            _imply(pack, st)
        return _advance(pack, st, cmd)
    return (_BY_SUBSTEP.get(step.substep) or _BY_KIND.get(spec.kind, _scored))(pack, st, cmd, spec, utt)
