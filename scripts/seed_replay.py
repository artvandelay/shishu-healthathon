"""Offline seed-scenario replay and sample PDF writer (`scripts/seed_replay.py`)."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from intake import engine, reports
from intake.models import CallState, PackConfig, Phase, ProfilePatch, QuestionKind, QuestionSpec, RosterInput, SubStep, TurnCommand, TurnOutcome

_NUM_WORDS = {"एक": 1, "दो": 2, "तीन": 3, "चार": 4, "पाँच": 5, "पांच": 5, "छह": 6, "सात": 7, "आठ": 8, "नौ": 9, "दस": 10}
_STOP = frozenset("क्या है हैं और या का की के को में से पर भी तो हाँ नहीं जी बस रहा रही रहे लगता लगती होता होती बहुत था थी थे".split())


def _clean(text: Any) -> str:
    return " ".join(str(text or "").split())


def _overlap(a: Any, b: Any) -> int:
    toks = lambda s: {w for w in "".join(c if c.isalnum() else " " for c in str(s or "").lower()).split() if len(w) > 1 and w not in _STOP}
    return len(toks(a) & toks(b))


def _seed_command(spec: QuestionSpec, words: str, substep: SubStep) -> TurnCommand:
    if substep is SubStep.WHICH_ITEM:
        return TurnCommand(outcome=TurnOutcome.CONCERN, item_ref="all", utterance=words)
    if substep is SubStep.FOLLOWUP:
        return TurnCommand(outcome=TurnOutcome.CONCERN if words else TurnOutcome.UNKNOWN, utterance=words)
    gates = spec.kind in (QuestionKind.GATE, QuestionKind.ROUTING_GATE, QuestionKind.BRIDGE)
    if not words:
        return TurnCommand(outcome=TurnOutcome.CLEAR if gates else TurnOutcome.UNKNOWN)
    quiet = words.strip().startswith(("नहीं", "नही", "ना", "no"))
    if gates or spec.kind is QuestionKind.OPEN_CAPTURE:
        return TurnCommand(outcome=TurnOutcome.CLEAR if quiet else TurnOutcome.CONCERN, utterance="" if (quiet and spec.kind is QuestionKind.OPEN_CAPTURE) else words, followup_answered=bool(gates and spec.followups))
    if spec.kind is QuestionKind.SINGLE_CHOICE:
        keys, labels = list(spec.options), {k: f"{o.label} {o.en}" for k, o in spec.options.items()}
        nums = [int(d) for d in re.findall(r"\d+", words)] + [v for w, v in _NUM_WORDS.items() if w in words]
        for k in keys:
            if spec.options[k].exclusive and _overlap(words, labels[k]):
                return TurnCommand(outcome=TurnOutcome.ANSWERED, value=k)
        for k in (keys if nums else ()):
            b = [int(d) for d in re.findall(r"\d+", labels[k])]
            if b and (("कम" in labels[k] and nums[0] < max(b)) or ("ज़्यादा" in labels[k] and nums[0] > min(b)) or (min(b) <= nums[0] <= max(b) and "कम" not in labels[k] and "ज़्यादा" not in labels[k])):
                return TurnCommand(outcome=TurnOutcome.ANSWERED, value=k)
        return TurnCommand(outcome=TurnOutcome.ANSWERED, value=max(keys, key=lambda k: _overlap(words, labels[k])))
    if spec.kind is QuestionKind.MULTI_CHOICE:
        return TurnCommand(outcome=TurnOutcome.ANSWERED, value=[k for k, o in spec.options.items() if _overlap(words, f"{o.label} {o.en}")] or list(spec.options)[:1])
    lvl = max(range(len(spec.score_rubric)), key=lambda i: _overlap(words, spec.score_rubric[i]))
    return TurnCommand(outcome=TurnOutcome.ANSWERED, value={"score": float(lvl), "level": lvl})


def replay_seed_scenario(pack: PackConfig, scenario: Mapping[str, Any]) -> tuple[CallState, list[tuple[str, str, str]]]:
    st, turns = engine.new_state(pack), []
    roster = tuple(RosterInput(age_years=float(c["age_years"]), tag=str(c.get("tag") or "unknown"), name=c.get("name")) for c in scenario.get("children") or ())
    queues, stop_after, gates = {str(a["qid"]): [str(s) for s in a.get("say") or ()] for a in scenario.get("answers") or ()}, scenario.get("stop_after_gates"), 0
    for _ in range(400):
        if st.phase is not Phase.ASKING or (spec := pack.by_id.get(st.cursor.qid)) is None:
            break
        if pr := _clean(engine.active_prompt(pack, st)):
            turns.append(("assistant", pr, spec.id))
        if slot := (st.cursor.hold_slot or (spec.requires_slot if spec.kind is QuestionKind.PROFILE else "")):
            is_ros = slot in ("roster", (pack.roster_cfg or {}).get("key"))
            patch = ProfilePatch(roster=roster, roster_complete=True) if is_ros else ProfilePatch(slots={slot: scenario[slot]} if _clean(scenario.get(slot)) else {})
            if said := _clean(" · ".join(f"{c.name or ''} {c.age_years}".strip() for c in roster) if is_ros else str(scenario.get(slot) or "")):
                turns.append(("user", said, spec.id))
            st = engine.reduce_turn(pack, st, TurnCommand(profile_patch=patch))
            continue
        if st.cursor.substep is SubStep.MAIN and stop_after is not None and gates >= int(stop_after):
            break
        save = str(scenario.get(spec.saves_to_profile or ""))
        words = spec.options[save].label if (spec.saves_to_profile and save in spec.options) else ((queues.get(spec.id) or []).pop(0) if queues.get(spec.id) else "")
        turns.append(("user", _clean(words) or "पता नहीं", spec.id))
        gates += int(spec.kind is QuestionKind.GATE and st.cursor.substep is SubStep.MAIN)
        st = engine.reduce_turn(pack, st, _seed_command(spec, words, st.cursor.substep))
    if st.phase is Phase.ASKING:
        st = engine.reduce_turn(pack, st, TurnCommand(caller_stop=True))
    if end_line := _clean(engine.say_text(pack, st)):
        turns.append(("assistant", end_line, ""))
    return st, turns


def write_sample_pdfs(pack: PackConfig, folder: Path) -> list[Path]:
    Path(folder).mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    for idx, sc in enumerate(pack.seed_scenarios, 1):
        st, _ = replay_seed_scenario(pack, sc)
        p = Path(folder) / str(sc.get("pdf") or f"{pack.id}-{idx}.pdf")
        p.write_bytes(reports.render_patient_pdf(pack, {"id": idx, "started_at": reports.SAMPLE_STARTED_AT, "duration_sec": int(sc.get("duration_sec") or 0)}, reports.project_summary(pack, st)))
        out.append(p)
    return out
