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
_SHAPE = {"field", "roster_op", "any_of", "all_of", "semantic_flag"}  # a flip in any of these can invalidate a NOT_APPLICABLE
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
    return bool(((state.roster or state.roster_stated) and not state.roster_incomplete)
                or any(eval_rule(pack, state, r) for r in (pack.roster_cfg.get("skip_if") or ())))
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
        # "The middle child" only exists when the three ages are strictly ordered. A twin who merely ties for
        # youngest is not middle, and appending the suffix there would also break the twin labels apart, hiding
        # the collision from `counts` below so both twins were read out with one identical age label.
        is_middle = len(ordered) == 3 and i == 1 and ordered[0].age_years > c.age_years > ordered[2].age_years
        parts.append(str(w.get(f"named_{gender}" if name else f"unnamed_{gender}", "{age} {name}" if name else "{age}")).format(age=c.age_label, name=name)
                     + (str(w.get("middle_female" if fem else "middle_male", "")) if is_middle else ""))
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
def _on_roster(pack: PackConfig, slot: str) -> bool: return slot in ("roster", (pack.roster_cfg or {}).get("key"))
def _is_unsatisfied(pack: PackConfig, state: CallState, spec: QuestionSpec) -> bool:
    if spec.kind is QuestionKind.PROFILE:
        # A profile step's satisfaction is ALWAYS its `requires_slot`: a stale `state.answers` entry (a replay
        # harness CLEAR, or an old answer) must never keep the ages reask from reopening on an incomplete roster.
        slot = spec.requires_slot
        return not (roster_resolved(pack, state) if _on_roster(pack, slot) else state.profile.get(slot) not in (None, "", [], {}))
    if state.answers.get(spec.id) is not None: return False
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
        if imp.get("clear_roster_if_unprovided") and not roster_given and not state.roster_stated and not state.roster_incomplete: state.roster = []
def _resync(pack: PackConfig, state: CallState) -> None:
    # An unresolved roster cannot prove an age-gated question is skip-free: `none_match` already evaluates to
    # False while `roster_incomplete`, so stripping its NOT_APPLICABLE would wake `mod_heavy_periods` for the
    # 2-and-4-year-olds of a mother who only mentioned her twins mid-call. Freeze it; the reask resolves it.
    frozen = not roster_resolved(pack, state)
    for spec in pack.questions:
        rec, hits = state.answers.get(spec.id), [r for r in spec.skip_if if eval_rule(pack, state, r)]
        frozen_here = frozen and any("roster_op" in (pack.rules.get(r) or {}) for r in spec.skip_if)
        if not hits and rec is not None and rec.state is _NA and not frozen_here: del state.answers[spec.id]
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
        # A router crumb is not a read: below the pack's own choice floor the residual softmax must not hijack the walk.
        floor = float((pack.decisions.get("thresholds") or {}).get("choice_min_confidence", 0.5))
        best = max(routable, key=lambda q: cmd.next_qid_probs.get(q.id, -1.0))
        head = best if cmd.next_qid_probs.get(best.id, -1.0) >= floor else head
    # A count with no ages is answered by the ages reask, never by re-asking "how many children" from scratch.
    held = "roster" if state.roster_incomplete and head.kind is QuestionKind.PROFILE and _on_roster(pack, head.requires_slot) else None
    state.cursor = CursorStep(qid=head.id, hold_slot=held)
    return state
def new_state(pack: PackConfig) -> CallState: return _advance(pack, CallState(), record=False)

def _pick_roster_id(r: RosterInput, old: list[RosterItem], used: set[str]) -> str:
    """A child's stable id: the one the extractor echoed, else the same child already on record, else a new one.

    `item_1..item_n` must never be renumbered, because every answer already recorded points at them.
    """
    known = {c.id for c in old}
    if r.id and r.id not in used and r.id in known: return r.id
    free = [c for c in old if c.id not in used]
    same = (next((c for c in free if (c.age_years, c.name) == (float(r.age_years), r.name)), None)
            or next((c for c in free if c.age_years == float(r.age_years)), None))
    if same is not None: return same.id
    taken, n = used | known, 1
    while f"item_{n}" in taken: n += 1
    return f"item_{n}"
def _apply_profile_data(pack: PackConfig, st: CallState, patch: ProfilePatch) -> bool:
    """Write a patch's slots and roster onto `st`; True when the turn carried children. Touches no cursor: a turn
    that also answered a question is resolved once, by `reduce_turn`, never pre-empted by its own patch."""
    slots = {k: (" ".join(v.split()) if isinstance(v, str) else v) for k, v in patch.slots.items()}
    st.profile.update({k: v for k, v in slots.items() if v not in (None, "")})
    for k, clean in slots.items():
        for q in pack.questions:
            if clean not in (None, "") and q.saves_to_profile == k and q.kind is QuestionKind.SINGLE_CHOICE and clean in q.options:
                st.answers[q.id] = AnswerRecord(state=AnswerStatus.ANSWERED, value=clean, choice_label=q.options[clean].label)
    roster_given = patch.roster is not None or patch.roster_complete is not None
    if roster_given:
        old, used, typed = list(st.roster), set(), []
        for r in patch.roster or ():
            used.add(cid := _pick_roster_id(r, old, used))
            typed.append(RosterItem(id=cid, age_years=float(r.age_years), tag=r.tag, name=r.name, age_label=format_age_label(pack, float(r.age_years))))
        complete = patch.roster_complete if patch.roster_complete is not None else bool(patch.roster)
        if patch.roster: st.roster = typed
        elif complete: st.roster = []  # an explicit zero children; an ages-less count keeps what is already typed
        st.roster_incomplete, st.roster_stated = not complete, complete
    _imply(pack, st, roster_given=roster_given)
    _resync(pack, st)
    return roster_given
def apply_profile_patch(pack: PackConfig, state: CallState, patch: ProfilePatch, *, attempted_on_profile: bool = False) -> CallState:
    """A patch is a turn whose only content is profile data: the same five phases decide where the call goes."""
    if patch.is_empty and not attempted_on_profile: return state.model_copy(deep=True)
    return reduce_turn(pack, state, TurnCommand(profile_patch=patch, profile_turn_attempted=attempted_on_profile))
def _ids(state: CallState, ref_raw: str) -> list[str]:
    ref, ids = " ".join((ref_raw or "").split()), [c.id for c in state.roster]
    if not ids: return []
    if ref == "all": return [c.id for c in state.ordered_roster]
    return [ref] if ref in ids else (ids if len(ids) == 1 and not ref and not state.roster_incomplete else [])
def _spoken_finding(st: CallState, qid: str) -> bool:
    """True when a CONCERN carrying the caller's own words is already on record: never flatten it to UNKNOWN or CLEAR."""
    rec = st.answers.get(qid)
    return rec is not None and rec.state is AnswerStatus.CONCERN and bool(rec.verbatim_text)
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
def _redact_note(cmd: TurnCommand, qid: str, *, stated: bool = False) -> bool:
    """A sentence that reported anything besides the active question is nobody's patient-safe quote.

    `jump_topic` belongs here for the same reason a revision does: the router only fires when the turn held more
    than what was asked, so Call #57's "हां, बुआ को कैंसर हुआ था और पहले नीली लाइट वाले सवाल में गलती हो गई थी" must
    stay in the doctor's verbatim without being quoted onto the mother's PDF.
    """
    return cmd.redacted or stated or cmd.jump_topic or any(t != qid for t in cmd.stated)
def _note_concern(st: CallState, spec: QuestionSpec, cmd: TurnCommand, rec: AnswerRecord, utt: str, *, stated: bool = False) -> dict[str, Any]:
    extra: dict[str, Any] = {}
    red = _redact_note(cmd, spec.id, stated=stated)
    # A mother co-reporting two concerns for the same child must keep the child she just named.
    if spec.roster_scoped and (ids := _ids(st, cmd.item_ref)): extra["item_ids"] = list(dict.fromkeys([*rec.item_ids, *ids]))
    if utt:
        if stated or not spec.followups or cmd.followup_answered: extra["note"] = rec.note.append(utt, redacted=red)
        # A stated concern is a remark about a question she was not asked, never an answer to its own followup.
        if not stated and spec.followups and cmd.followup_answered:
            fid = next((f for f in spec.followups if f not in rec.followups or not rec.followups[f].note.verbatim), spec.followups[0])
            extra["followups"] = {**rec.followups, fid: FollowupSlot(state=AnswerStatus.CONCERN, note=NoteText.from_text(utt, redacted=red))}
    return extra
def _apply_stated(pack: PackConfig, st: CallState, cmd: TurnCommand, utt: str, *, active_qid: str | None = None) -> str | None:
    """Write the turn's out-of-turn revisions: the qid to ask next, or None when it said nothing new.

    A question the pack itself skipped — flag-pruned, age-gated, a child question for a mother with no children,
    a family question her own answers closed — is not one she can answer: `stated_` catches a topic volunteered
    out loud, it never overrules the pack's own routing.
    """
    for target, out in cmd.stated.items():
        spec = pack.by_id.get(target)
        if spec is None or not spec.is_routable or target == active_qid or out not in (TurnOutcome.CONCERN, TurnOutcome.CLEAR, TurnOutcome.CLARIFY): continue
        if any(eval_rule(pack, st, r) for r in spec.skip_if if "group_no_concern" not in (pack.rules.get(r) or {})): continue
        existing = st.answers.get(target)
        # `group_no_concern` stands on the absence of a concern, so a stated concern is exactly its missing evidence.
        reopened = (out is TurnOutcome.CONCERN and roster_resolved(pack, st)
                    and any("group_no_concern" in (pack.rules.get(r) or {}) for r in spec.skip_if))
        if existing is not None and existing.state is _NA and not reopened: continue
        # One active clarify means one question is being asked about. A `stated_` clarify may only name the
        # question she JUST LEFT and came back UNKNOWN on, and only when exactly one head fired it: an unasked
        # future question (`existing is None`), a settled CLEAR/CONCERN answer, or a scatter of two-or-more
        # clarify heads (Call #47/#57's ASR garble of "थैलेसीमिया") names nobody and must never move the call.
        if out is TurnOutcome.CLARIFY and cmd.outcome is TurnOutcome.CLARIFY:
            if sum(o is TurnOutcome.CLARIFY for o in cmd.stated.values()) != 1: continue
            if target != st.last_answered_qid and not (existing is not None and existing.state is AnswerStatus.UNKNOWN): continue
        if out is TurnOutcome.CLARIFY:
            st.answers.pop(target, None)
            _resync(pack, st)
            if (kept := st.answers.get(target)) is None or kept.state is not _NA: return target
            continue
        status = _STATUS[out]
        rec = AnswerRecord(state=status) if status is AnswerStatus.CLEAR else (existing or AnswerRecord(state=status))
        extra = _note_concern(st, spec, cmd, rec, utt, stated=True) if status is AnswerStatus.CONCERN else {}
        st.answers[target] = rec.model_copy(update={"state": status} | extra)
        if spec.saves_to_profile:
            st.profile[spec.saves_to_profile] = status is AnswerStatus.CONCERN
            _imply(pack, st)
    _resync(pack, st)
    return None
def _which_item(pack: PackConfig, st: CallState, cmd: TurnCommand, spec: QuestionSpec, utt: str) -> CursorStep | None:
    rec = st.answers.get(spec.id) or AnswerRecord(state=AnswerStatus.CONCERN)
    # A named child the model could not map still carries her words; a meta question about the notes never does.
    keep = cmd.outcome in (TurnOutcome.CONCERN, TurnOutcome.ANSWERED, TurnOutcome.UNKNOWN)
    red = _redact_note(cmd, spec.id)
    note = rec.note.append(utt, redacted=red) if (utt and keep) else rec.note
    rec, ids = rec.model_copy(update={"note": note}), _ids(st, cmd.item_ref) if cmd.outcome is not TurnOutcome.NOT_ABOUT_ACTIVE else []
    if not (ids or cmd.outcome is TurnOutcome.UNKNOWN or _bump(st.cursor)):
        st.answers[spec.id] = rec
        return st.cursor
    rec = st.answers[spec.id] = rec.model_copy(update={"item_ids": ids})
    open_f = [i for i, f in enumerate(spec.followups)
              if f not in rec.followups or (rec.followups[f].state is not AnswerStatus.CONCERN and not rec.followups[f].note.verbatim)]
    if keep and cmd.followup_answered and open_f:
        # She named the child AND answered the followup in one breath: the detail settles that slot, never a re-ask.
        fid = spec.followups[open_f[0]]
        rec = st.answers[spec.id] = rec.model_copy(update={"followups": {**rec.followups, fid: FollowupSlot(state=AnswerStatus.CONCERN, note=NoteText.from_text(utt, redacted=red))}})
        open_f = open_f[1:]
    return CursorStep(qid=spec.id, substep=SubStep.FOLLOWUP, followup_idx=open_f[0]) if open_f else None
def _followup(pack: PackConfig, st: CallState, cmd: TurnCommand, spec: QuestionSpec, utt: str) -> CursorStep | None:
    # Any classified outcome settles the slot: a live multimodal turn reads the audio and returns concern/clear
    # while the transcript is still empty, so an absent `utt` must not force a retry.
    settled = cmd.outcome in _STATUS or (bool(utt) and cmd.outcome is not TurnOutcome.NOT_ABOUT_ACTIVE)
    if not settled and not _bump(st.cursor): return st.cursor
    rec, fid = st.answers.get(spec.id) or AnswerRecord(state=AnswerStatus.CONCERN), spec.followups[st.cursor.followup_idx]
    note = (rec.followups[fid].note if fid in rec.followups else NoteText()).append(utt if settled else "", redacted=_redact_note(cmd, spec.id))
    slot = _STATUS.get(cmd.outcome, AnswerStatus.CONCERN) if settled else AnswerStatus.UNKNOWN
    st.answers[spec.id] = rec.model_copy(update={"followups": {**rec.followups, fid: FollowupSlot(state=slot, note=note)}})
    nxt = st.cursor.followup_idx + 1
    return CursorStep(qid=spec.id, substep=SubStep.FOLLOWUP, followup_idx=nxt) if nxt < len(spec.followups) else None
def _open_capture(pack: PackConfig, st: CallState, cmd: TurnCommand, spec: QuestionSpec, utt: str) -> CursorStep | None:
    red = _redact_note(cmd, spec.id)
    if utt and cmd.outcome in (TurnOutcome.CONCERN, TurnOutcome.ANSWERED, TurnOutcome.NOT_ABOUT_ACTIVE):
        base = st.answers.get(spec.id) or AnswerRecord(state=AnswerStatus.CONCERN)
        st.answers[spec.id], st.last_answered_qid = base.model_copy(update={"state": AnswerStatus.CONCERN, "note": base.note.append(utt, redacted=red)}), spec.id
        return CursorStep(qid=spec.id, open_rounds=st.cursor.open_rounds + 1)
    if cmd.outcome not in _QUIET and not _bump(st.cursor):
        return CursorStep(qid=spec.id, substep=SubStep.FOLLOWUP) if spec.followups else st.cursor
    if not _spoken_finding(st, spec.id): st.answers[spec.id] = AnswerRecord(state=_STATUS.get(cmd.outcome, AnswerStatus.UNKNOWN))
    return None
def _bridge(pack: PackConfig, st: CallState, cmd: TurnCommand, spec: QuestionSpec, utt: str) -> CursorStep | None:
    st.answers[spec.id] = AnswerRecord(state=AnswerStatus.CLEAR)
def _scored(pack: PackConfig, st: CallState, cmd: TurnCommand, spec: QuestionSpec, utt: str) -> CursorStep | None:
    parsed = _read_answer(spec, cmd)
    if parsed is None and cmd.outcome is not TurnOutcome.UNKNOWN and not _bump(st.cursor): return st.cursor
    status, val, label = parsed or (AnswerStatus.UNKNOWN, None, "")
    prior = st.answers.get(spec.id)
    rec = prior if (prior is not None and prior.state.is_resolved and status is AnswerStatus.CONCERN) else AnswerRecord(state=status)
    rec = st.answers[spec.id] = rec.model_copy(update={"state": status, "value": val, "choice_label": label} | (_note_concern(st, spec, cmd, rec, utt) if status is AnswerStatus.CONCERN else {}))
    if spec.saves_to_profile and status is not AnswerStatus.UNKNOWN:
        st.profile[spec.saves_to_profile] = val if spec.kind is QuestionKind.SINGLE_CHOICE else (status is AnswerStatus.CONCERN)
        _imply(pack, st)
    if status is not AnswerStatus.CONCERN: return None
    if spec.roster_scoped and len(st.roster) > 1 and not rec.item_ids: return CursorStep(qid=spec.id, substep=SubStep.WHICH_ITEM)
    if spec.followups and (not cmd.followup_answered or len(spec.followups) > 1): return CursorStep(qid=spec.id, substep=SubStep.FOLLOWUP, followup_idx=1 if cmd.followup_answered else 0)
    return None
_BY_SUBSTEP = {SubStep.WHICH_ITEM: _which_item, SubStep.FOLLOWUP: _followup}
_BY_KIND = {QuestionKind.OPEN_CAPTURE: _open_capture, QuestionKind.BRIDGE: _bridge}

def _reduce_active(pack: PackConfig, st: CallState, cmd: TurnCommand, spec: QuestionSpec, step: CursorStep, utt: str) -> CursorStep | None:
    """Answer `spec` on the cursor it was asked with: its cursor while it is still being asked, None when done."""
    if (cur := st.answers.get(spec.id)) is not None and cur.state is _NA: return None
    if cmd.outcome is TurnOutcome.NOT_ABOUT_ACTIVE and step.substep is SubStep.MAIN and not (spec.kind is QuestionKind.OPEN_CAPTURE and utt):
        # An off-topic turn is still a miss: burn the attempt so the question can never repeat forever.
        if not _bump(st.cursor): return st.cursor
        # Mirror `_open_capture`: a spoken finding already on record is never flattened to a bare UNKNOWN.
        if not _spoken_finding(st, spec.id): st.answers[spec.id] = AnswerRecord(state=AnswerStatus.UNKNOWN)
        return None
    if cmd.outcome is TurnOutcome.RETRACT or (step.substep is SubStep.WHICH_ITEM and cmd.outcome is TurnOutcome.CLEAR and not _ids(st, cmd.item_ref)):
        if not (spec.kind is QuestionKind.OPEN_CAPTURE and _spoken_finding(st, spec.id)): st.answers[spec.id] = AnswerRecord(state=AnswerStatus.CLEAR)
        if spec.saves_to_profile:
            st.profile[spec.saves_to_profile] = False
            _imply(pack, st)
        return None
    return (_BY_SUBSTEP.get(step.substep) or _BY_KIND.get(spec.kind, _scored))(pack, st, cmd, spec, utt)
def _reopens(pack: PackConfig, st: CallState, prev_qid: str, *, profile_only: bool) -> CursorStep | None:
    """The earlier question a profile change must ask again, as the cursor that asks it."""
    pos = {q.id: i for i, q in enumerate(pack.questions)}
    early = next((q for q in pack.questions if (q.kind is QuestionKind.PROFILE or not profile_only)
                  and _is_unsatisfied(pack, st, q) and pos[q.id] < pos.get(prev_qid, -1)), None)
    if early is None: return None
    return CursorStep(qid=early.id, hold_slot="roster" if st.roster_incomplete and _on_roster(pack, early.requires_slot) else None)

def reduce_turn(pack: PackConfig, state: CallState, cmd: TurnCommand) -> CallState:
    """One turn in five phases, so a mid-call profile patch can never swallow the answer it arrived with."""
    st = state.model_copy(deep=True)
    if st.phase is not Phase.ASKING: return st
    utt = " ".join(cmd.utterance.split())
    if cmd.emergency or cmd.caller_stop:
        st.phase, st.close_reason = Phase.CLOSING, CloseReason.EMERGENCY if cmd.emergency else CloseReason.CALLER_STOP
        st.cursor, st.emergency_note = CursorStep(qid=st.cursor.qid), (utt if (cmd.emergency and utt) else st.emergency_note)
        return st
    # 1: a flag can prune or reopen whole branches of the form before the turn is read at all.
    if cmd.semantic_flags:
        st.semantic_flags = list(dict.fromkeys([*st.semantic_flags, *cmd.semantic_flags]))
        _resync(pack, st)
    # 2: the profile data lands, and says nothing about where the call goes next.
    prev_step, prev_qid, prev_spec = st.cursor.model_copy(deep=True), st.cursor.qid, pack.by_id.get(st.cursor.qid)
    prev_kind = step_kind(pack, st)
    profile_ran = (cmd.profile_patch is not None and not cmd.profile_patch.is_empty) or (cmd.profile_turn_attempted and prev_kind == _PROFILE)
    roster_given = _apply_profile_data(pack, st, cmd.profile_patch or ProfilePatch()) if profile_ran else False
    # 3: the question she was asked is reduced on its own terms. A returned cursor means it is still being asked;
    # None means it is done and Phase 5 owns the walk. Nothing here advances the call.
    local_cursor, record_last = None, False
    if prev_kind != _PROFILE and prev_spec is not None:
        cur = st.answers.get(prev_qid)
        if cur is not None and (cur.state is _NA or (prev_step.substep is SubStep.MAIN and prev_spec.kind is not QuestionKind.OPEN_CAPTURE and cur.state.is_resolved)):
            record_last = cur.state is not _NA  # already settled by a flag or the patch: nothing left to read
        elif cmd.outcome is TurnOutcome.CLARIFY or (cmd.outcome is TurnOutcome.NOT_ABOUT_ACTIVE and (profile_ran or cmd.stated)
                                                     and not (prev_spec.kind is QuestionKind.OPEN_CAPTURE and utt)):
            local_cursor = prev_step
        else:
            st.cursor, local_cursor = prev_step, _reduce_active(pack, st, cmd, prev_spec, prev_step, utt)
            record_last = local_cursor is None
    # 4: out-of-turn revisions, read against the question she was actually asked.
    clarify_qid = _apply_stated(pack, st, cmd, utt, active_qid=prev_qid)
    # 5: exactly one cursor decision, and at most one advance, for the whole turn.
    if clarify_qid:
        return st.model_copy(update={"phase": Phase.ASKING, "close_reason": None, "cursor": CursorStep(qid=clarify_qid)})
    if prev_kind == _PROFILE:
        if prev_spec is not None and _is_unsatisfied(pack, st, prev_spec):
            if cmd.profile_turn_attempted or roster_given:
                st.cursor = CursorStep(qid=prev_qid, hold_slot="roster" if st.roster_incomplete or _on_roster(pack, prev_spec.requires_slot) else prev_spec.requires_slot)
            else:
                st.cursor = prev_step
            return st
        record_last = True
    # A profile change rewinds: only an earlier profile slot while this step is still live, else anything open before it.
    if profile_ran and (rewound := _reopens(pack, st, prev_qid, profile_only=local_cursor not in (None, prev_step))) is not None:
        return st.model_copy(update={"cursor": rewound})
    if local_cursor is not None and not ((c := st.answers.get(local_cursor.qid)) is not None and c.state is _NA):
        return st.model_copy(update={"cursor": local_cursor})
    st.cursor = prev_step.model_copy(deep=True)
    return _advance(pack, st, cmd, record=record_last)
