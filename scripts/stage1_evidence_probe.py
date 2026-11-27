#!/usr/bin/env python3
"""Manager-run Stage 1 evidence provenance probe — offline by default.

Replays voice/text, quick tap, late STT, and stale-after-tap through
``LiveCallSession`` fakes. Observes production ``_commit_turn`` and real
``TurnEvidence`` — does not prove Decisions or Gemini quality.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from starlette.websockets import WebSocketState

from intake import live
from intake.models import CallState, TurnCommand, TurnOutcome
from intake.pack import load_pack
from intake.turns import EvidenceSource, TurnEvidence

PACK = load_pack("shishu-hi")
VOICED = (1200).to_bytes(2, "little") * 640


def pcm(blob: bytes = VOICED) -> str:
    return base64.b64encode(blob).decode()


def state_fingerprint(st: CallState) -> tuple[Any, ...]:
    c = st.cursor
    return (st.phase.value, c.qid, c.substep.value, c.followup_idx, st.step_token)


@dataclass
class ProbeCommit:
    evidence: TurnEvidence
    state_before: CallState
    state_after: CallState


@dataclass
class RecordingBoundary:
    commits: list[ProbeCommit] = field(default_factory=list)

    def clear(self) -> None:
        self.commits.clear()

    def sources(self) -> list[EvidenceSource]:
        return [c.evidence.source for c in self.commits]


class FakeSession:
    def __init__(self) -> None:
        self.sent: list[tuple[str, Any]] = []
        self.setup_complete = True

    async def send_realtime_input(self, **kwargs: Any) -> None:
        self.sent.append(("realtime", kwargs))

    async def send_tool_response(self, function_responses: list) -> None:
        self.sent.append(("tool", function_responses))

    async def receive(self):
        await asyncio.Event().wait()


class FakeWebSocket:
    def __init__(self) -> None:
        self.client_state = WebSocketState.CONNECTED
        self.sent: list[dict[str, Any]] = []

    async def accept(self) -> None:
        pass

    async def close(self) -> None:
        self.client_state = WebSocketState.DISCONNECTED

    async def send_json(self, payload: dict[str, Any]) -> None:
        self.sent.append(payload)

    async def receive_text(self) -> str:
        raise live.WebSocketDisconnect()

    def of(self, kind: str) -> list[dict[str, Any]]:
        return [p for p in self.sent if p.get("type") == kind]


class FakeRecorder:
    def add_user_pcm(self, _pcm: bytes, _rate: int) -> None:
        pass

    def add_bot_pcm(self, _pcm: bytes, _rate: int) -> None:
        pass

    def save(self) -> tuple[str, int]:
        return "probe.wav", 0


def _answers(**heads: Any) -> dict[str, Any]:
    body: dict[str, Any] = {}
    for name, value in heads.items():
        body[name] = {"probabilities": value} if isinstance(value, dict) else {"noul": value}
    return {"answers": body, "usage": {"total_input_tokens": 12}}


def _turn_result(body: dict[str, Any]) -> live.decisions.DecisionCallResult:
    return live.decisions.DecisionCallResult(True, "", 1, 200, 100, 4, 12, body["answers"])


def _turn_with(state: CallState, outcome: TurnOutcome, spoken: str) -> live.Turn:
    rivals = {o.value: 0.03 / 5 for o in (
        TurnOutcome.CLEAR, TurnOutcome.CONCERN, TurnOutcome.UNKNOWN,
        TurnOutcome.CLARIFY, TurnOutcome.RETRACT) if o is not outcome}
    body = _answers(danger_now=0.01, active_outcome={outcome.value: 0.97, **rivals})
    intent = live.decisions.interpret_turn_decision(PACK, state, body["answers"], utterance_text=spoken)
    return live.Turn(_turn_result(body), intent, state.model_copy(deep=True), spoken, 640, None)


async def _noop_db(*_a: Any, **_kw: Any) -> Any:
    return None


def build_session(*, qid: str | None = None) -> tuple[live.LiveCallSession, FakeWebSocket]:
    ws = FakeWebSocket()
    state = live.engine.new_state(PACK)
    if qid:
        state = state.model_copy(update={"cursor": state.cursor.model_copy(update={"qid": qid})})
    built = live.LiveCallSession(
        ws=ws, pack=PACK, session=FakeSession(), call_id=None, state=state,
        recorder=FakeRecorder(), on_call_completed=lambda _cid: True,
    )
    built._db = _noop_db  # type: ignore[method-assign]
    return built, ws


def install_probe(session: live.LiveCallSession, boundary: RecordingBoundary) -> None:
    original_raw = session._commit_turn
    original_sem = session._commit_semantic_turn

    async def probe_raw(state: CallState, evidence: TurnEvidence) -> None:
        before = session.state.model_copy(deep=True)
        await original_raw(state, evidence)
        boundary.commits.append(ProbeCommit(
            evidence=evidence,
            state_before=before,
            state_after=session.state.model_copy(deep=True),
        ))

    async def probe_sem(state: CallState, evidence: TurnEvidence, interpretation: Any, decisions: Any) -> None:
        before = session.state.model_copy(deep=True)
        await original_sem(state, evidence, interpretation, decisions)
        boundary.commits.append(ProbeCommit(
            evidence=evidence,
            state_before=before,
            state_after=session.state.model_copy(deep=True),
        ))

    session._commit_turn = probe_raw  # type: ignore[method-assign]
    session._commit_semantic_turn = probe_sem  # type: ignore[method-assign]


async def scenario_voice_text(boundary: RecordingBoundary) -> list[str]:
    built, _ = build_session(qid="birth_cord_3weeks")
    install_probe(built, boundary)
    spoken = "नहीं, ऐसा कुछ नहीं है"

    async def fake_eval(snapshot: CallState, text: str, _pcm: bytes) -> live.Turn:
        return _turn_with(snapshot, TurnOutcome.CLEAR, text)

    built._evaluate_turn_task = fake_eval  # type: ignore[method-assign]
    await built.on_text({"text": spoken})
    await built._next_question()
    errs: list[str] = []
    if len(boundary.commits) != 1:
        errs.append(f"voice_text: expected 1 commit, got {len(boundary.commits)}")
    else:
        c = boundary.commits[0]
        if c.evidence.source is not EvidenceSource.TEXT:
            errs.append(f"voice_text: source {c.evidence.source}")
        if c.evidence.snapshot_step != c.state_before.step_token:
            errs.append("voice_text: snapshot_step != step at ingress")
    return errs


async def scenario_quick_tap(boundary: RecordingBoundary) -> list[str]:
    built, _ = build_session(qid="birth_cord_3weeks")
    install_probe(built, boundary)
    step_in = built.state.step_token
    await built.on_quick_reply({"value": "clear", "step": step_in})
    errs: list[str] = []
    if len(boundary.commits) != 1:
        errs.append(f"quick_tap: expected 1 commit, got {len(boundary.commits)}")
    c = boundary.commits[0] if boundary.commits else None
    if c and c.evidence.source is not EvidenceSource.TAP:
        errs.append(f"quick_tap: source {c.evidence.source}")
    if c and c.evidence.snapshot_step != step_in:
        errs.append("quick_tap: snapshot_step != tap step")
    if c and c.state_before.step_token == c.state_after.step_token:
        errs.append("quick_tap: commit did not advance cursor")
    return errs


async def scenario_late_stt(boundary: RecordingBoundary) -> list[str]:
    built, ws = build_session()
    built.state = live.engine.new_state(PACK).model_copy(
        update={"cursor": live.engine.new_state(PACK).cursor.model_copy(update={"qid": "fam_cancer"})})
    install_probe(built, boundary)
    utterance = "हां, बुआ को कैंसर हुआ था"

    async def fake_eval(snapshot: CallState, spoken: str, pcm_bytes: bytes) -> live.Turn:
        intent = live.decisions.TurnIntent(
            command=TurnCommand(outcome=TurnOutcome.CONCERN, utterance=utterance),
            needs_profile_extraction=False, diagnostics={})
        return live.Turn(_turn_result(_answers(danger_now=0.01)), intent,
                         snapshot.model_copy(deep=True), spoken, len(pcm_bytes), None)

    built._evaluate_turn_task = fake_eval  # type: ignore[method-assign]
    await built.on_audio({"data": pcm(), "sample_rate": 16000})
    await built._hear("user", utterance)
    await built._next_question()
    fp_after = state_fingerprint(built.state)
    voice = boundary.commits[0] if boundary.commits else None
    late_text = "नीली लाइट में"
    await built._hear("user", late_text)

    errs: list[str] = []
    if len(boundary.commits) != 2:
        errs.append(f"late_stt: expected 2 commits, got {len(boundary.commits)}")
    if voice and boundary.commits[1].evidence.source is not EvidenceSource.LATE_STT:
        errs.append(f"late_stt: second source {boundary.commits[1].evidence.source}")
    if voice and boundary.commits[1].evidence.snapshot_step != voice.evidence.snapshot_step:
        errs.append(
            f"late_stt: late snapshot {boundary.commits[1].evidence.snapshot_step!r} "
            f"!= voice snapshot {voice.evidence.snapshot_step!r}"
        )
    late = boundary.commits[1] if len(boundary.commits) > 1 else None
    if late and state_fingerprint(late.state_after)[:4] != fp_after[:4]:
        errs.append("late_stt: late commit advanced cursor")
    tele = ws.of("telemetry")
    if not tele or tele[-1].get("last_event", {}).get("type") != "late_stt":
        errs.append("late_stt: telemetry not emitted")
    return errs


async def scenario_stale_after_tap(boundary: RecordingBoundary) -> list[str]:
    built, _ = build_session(qid="birth_term")
    install_probe(built, boundary)
    gate = asyncio.Event()
    stale_step = built.state.step_token

    async def slow_eval(snapshot: CallState, spoken: str, _pcm: bytes) -> live.Turn:
        await gate.wait()
        return _turn_with(snapshot, TurnOutcome.CLEAR, spoken)

    built._evaluate_turn_task = slow_eval  # type: ignore[method-assign]
    await built.on_text({"text": "नहीं"})
    answering = asyncio.ensure_future(
        built._answer_tools([live.types.FunctionCall(id="t1", name="next_question", args={})]))
    await asyncio.sleep(0)
    await built.on_quick_reply({"value": "clear", "step": built.state.step_token})
    fp_after_tap = state_fingerprint(built.state)
    gate.set()
    await answering

    errs: list[str] = []
    if len(boundary.commits) != 2:
        errs.append(f"stale_after_tap: expected 2 commits, got {len(boundary.commits)}")
        return errs
    if boundary.sources() != [EvidenceSource.TAP, EvidenceSource.TEXT]:
        errs.append(f"stale_after_tap: order {boundary.sources()}")
    tap, stale = boundary.commits[0], boundary.commits[1]
    if tap.evidence.source is not EvidenceSource.TAP:
        errs.append("stale_after_tap: first commit is not tap")
    if stale.evidence.source is not EvidenceSource.TEXT:
        errs.append("stale_after_tap: second commit is not stale text")
    if stale.evidence.snapshot_step != stale_step:
        errs.append(f"stale_after_tap: stale snapshot {stale.evidence.snapshot_step!r} != {stale_step!r}")
    if state_fingerprint(stale.state_after) != fp_after_tap:
        errs.append("stale_after_tap: stale commit clobbered tap cursor state")
    if state_fingerprint(tap.state_after) != fp_after_tap:
        errs.append("stale_after_tap: tap did not advance cursor")
    return errs


SCENARIOS: dict[str, Callable[[RecordingBoundary], Awaitable[list[str]]]] = {
    "voice_text": scenario_voice_text,
    "quick_tap": scenario_quick_tap,
    "late_stt": scenario_late_stt,
    "stale_after_tap": scenario_stale_after_tap,
}


async def _run_one(name: str, boundary: RecordingBoundary) -> list[str]:
    boundary.clear()
    return await SCENARIOS[name](boundary)


def cmd_smoke(_: argparse.Namespace) -> int:
    errs: list[str] = []
    for name in SCENARIOS:
        try:
            errs.extend(asyncio.run(_run_one(name, RecordingBoundary())))
        except Exception as exc:  # noqa: BLE001
            errs.append(f"{name}: {type(exc).__name__}: {exc}")
    if errs:
        print("FAIL")
        for line in errs:
            print(f"  - {line}")
        return 1
    print("OK — Stage 1 probe smoke passed (offline fakes only; not model quality).")
    return 0


def cmd_list(_: argparse.Namespace) -> int:
    for name in SCENARIOS:
        print(name)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    boundary = RecordingBoundary()
    errs = asyncio.run(_run_one(args.name, boundary))
    print(json.dumps({
        "scenario": args.name,
        "commits": [{
            "source": c.evidence.source.value,
            "snapshot_step": c.evidence.snapshot_step,
            "transcript": c.evidence.transcript[:80],
            "selected_value": c.evidence.selected_value,
            "step_before": c.state_before.step_token,
            "step_after": c.state_after.step_token,
        } for c in boundary.commits],
        "errors": errs,
    }, ensure_ascii=False, indent=2))
    return 1 if errs else 0


def cmd_live_ws(_: argparse.Namespace) -> int:
    print(
        "NOT IMPLEMENTED — use scripts/run_personas.py for real calls.\n"
        "This probe records _commit_turn only; fakes do not prove model quality.",
        file=sys.stderr,
    )
    return 2


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Stage 1 evidence provenance probe (offline).")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("smoke").set_defaults(func=cmd_smoke)
    sub.add_parser("list").set_defaults(func=cmd_list)
    run = sub.add_parser("run")
    run.add_argument("name", choices=sorted(SCENARIOS))
    run.set_defaults(func=cmd_run)
    sub.add_parser("live-ws").set_defaults(func=cmd_live_ws)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())