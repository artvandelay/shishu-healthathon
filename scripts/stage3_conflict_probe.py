#!/usr/bin/env python3
"""Stage 3 conflict probe. offline: canned heads + temp SQLite. live: real gemini-decision-1.
Tap fires during _evaluate_turn_task. offline --verify exits 1 on a contract mismatch; a missing arbitrator is printed, not crashed.
"""

from __future__ import annotations

import argparse, asyncio, os, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ["DATA_DIR"] = tempfile.mkdtemp(prefix="stage3-conflict-")
sys.path.insert(0, str(ROOT))

from intake import db, decisions, engine, live  # noqa: E402
from intake.models import AnswerRecord, AnswerStatus, CallState, CursorStep, Phase, ProfilePatch, RosterItem, TurnCommand, TurnOutcome  # noqa: E402
from intake.pack import load_pack  # noqa: E402

SHISHU, FEVER = load_pack("shishu-hi"), load_pack("fever-opd-hi")


def _h(**kw):
    return {k: ({"probabilities": v} if isinstance(v, dict) else {"noul": v}) for k, v in kw.items()}


# want = (final_qid or "*", answer_state or "*", disposition tokens that must appear)
CASES = (
    {"id": "tap_yes_spoken_no_bcg", "pack": "shishu-hi", "step": "birth_bcg", "tap": "concern", "watch": "birth_bcg",
     "speech": "नहीं, कोई गाँठ नहीं हुई थी", "want": ("birth_bcg", "-", "conflict"),
     "raw": _h(danger_now=0.01, active_topic=0.96, active_outcome={"clear": 0.95, "concern": 0.02, "other": 0.03})},
    {"id": "tap_choice_spoken_contradiction", "pack": "fever-opd-hi", "step": "fever_duration_bucket", "tap": "lt_3_days",
     "watch": "fever_duration_bucket", "speech": "दस दिन से बुखार आ रहा है", "want": ("fever_duration_bucket", "-", "conflict"),
     "raw": _h(danger_now=0.01, active_topic=0.95, choice_value={"gt_7_days": 0.94, "lt_3_days": 0.03, "other": 0.03})},
    {"id": "tap_choice_spoken_unlisted_pattern", "pack": "fever-opd-hi", "step": "fever_nature", "tap": "continuous",
     "watch": "fever_nature", "speech": "न लगातार है न बीच-बीच में, बस दौड़ने के बाद थोड़ा गरम लगता है", "want": ("fever_nature", "-", "conflict"),
     "raw": _h(danger_now=0.01, active_topic=0.94, choice_value={"other": 0.93, "continuous": 0.04, "intermittent": 0.03},
               active_outcome={"concern": 0.88, "other": 0.08})},
    {"id": "tap_no_spoken_no_plus_fever_aside", "pack": "shishu-hi", "step": "birth_bcg", "tap": "clear", "watch": "birth_bcg",
     "speech": "नहीं, बस हल्का बुखार आया था", "want": ("birth_mode_detail", "clear", "accept_form,retain_observation"),
     "raw": _h(danger_now=0.01, active_topic=0.95, residual_clinical=0.92, active_outcome={"clear": 0.93, "other": 0.05, "concern": 0.02})},
    {"id": "tap_no_spoken_emergency", "pack": "shishu-hi", "step": "birth_bcg", "tap": "clear", "watch": "birth_bcg",
     "speech": "अभी बच्चे को साँस नहीं आ रही है!", "want": ("*", "emergency", ""),
     "raw": _h(danger_now=0.98, active_outcome={"other": 0.95})},
)


def _state(pack_id: str, step: str = "") -> CallState:
    if pack_id == "fever-opd-hi":
        st = engine.reduce_turn(FEVER, engine.new_state(FEVER), TurnCommand(profile_patch=ProfilePatch(slots={"patient_name": "रवि", "patient_age": 28})))
        st = engine.reduce_turn(FEVER, st, TurnCommand(outcome=TurnOutcome.CLEAR))
        if step == "fever_nature":
            st = engine.reduce_turn(FEVER, st, TurnCommand(outcome=TurnOutcome.ANSWERED, value="days_3_to_7"))
        return st
    st = engine.new_state(SHISHU)
    st.profile.update({"caller_name": "सुनीता", "marital_status": "married"})
    st.roster = [RosterItem(id="item_1", age_years=6.0, age_label="6 साल", tag="male", name="अमन"),
                 RosterItem(id="item_2", age_years=2.0, age_label="2 साल", tag="female", name="मीरा")]
    st.roster_stated = True
    for qid in ("intro_name", "intro_married", "intro_children", "intro_bridge"):
        st.answers[qid] = AnswerRecord(state=AnswerStatus.CLEAR)
    st.cursor = CursorStep(qid="birth_bcg")
    return st


class _IO:
    """Fake browser socket and fake Gemini Live session — this probe never opens a voice socket."""
    setup_complete = True
    def __init__(self) -> None: self.sent: list = []
    async def send_json(self, payload) -> None: self.sent.append(payload)
    async def send_realtime_input(self, **_kwargs) -> None: return None
    async def send_tool_response(self, function_responses) -> None: return None


async def _race(case: dict, *, live_mode: bool) -> dict:
    """Bank the utterance, start the decision, tap that same step before it returns."""
    pack = SHISHU if case["pack"] == "shishu-hi" else FEVER
    call_id = db.create_call(pack_id=pack.id)
    sess = live.LiveCallSession(ws=_IO(), pack=pack, session=_IO(), call_id=call_id, state=_state(case["pack"], case["step"]),
                                 on_call_completed=lambda _cid: True)
    started, gate, box = asyncio.Event(), asyncio.Event(), {}

    async def evaluate(snapshot, spoken, pcm):
        started.set()  # parent taps only after this decision pass has actually begun
        try:
            if live_mode:
                turn = await live.LiveCallSession._evaluate_turn_task(sess, snapshot, spoken, pcm)
            else:
                await gate.wait()
                res = decisions.DecisionCallResult(True, "", 1, 0, 0, 1, 0, case["raw"])
                intent = decisions.interpret_turn_decision(pack, snapshot, case["raw"], utterance_text=spoken)
                turn = live.Turn(res, intent, snapshot, spoken, len(pcm), None)
        except Exception as exc:  # noqa: BLE001 — session recovers; the row still names the failure
            box["err"] = f"{type(exc).__name__}: {exc}"
            raise
        box["ms"] = turn.result.decision_ms
        if not turn.result.ok:
            box["err"] = turn.result.error_reason or "decision_failed"
        return turn

    sess._evaluate_turn_task = evaluate  # type: ignore[method-assign]
    row = {"scenario": case["id"], "tap": case["tap"], "speech": case["speech"], "want": case["want"], "ms": ""}
    task = None
    try:
        await sess.on_text({"text": case["speech"]})
        task = asyncio.create_task(sess._answer_tools([live.types.FunctionCall(id="c1", name="next_question", args={})]))
        await asyncio.wait_for(started.wait(), 20 if live_mode else 3)
        await sess.on_quick_reply({"value": case["tap"], "step": case["step"]})
        if not live_mode:
            gate.set()
        await asyncio.wait_for(task, 45 if live_mode else 5)
    except Exception as exc:  # noqa: BLE001
        row["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        gate.set(); sess.stop.set(); sess._cancel("fallback")
        if task is not None and not task.done():
            task.cancel(); await asyncio.gather(task, return_exceptions=True)
    stored = db.call_state_from_row(db.get_call(call_id))
    rec, qid = stored.answers.get(case["watch"]), stored.cursor.qid or "-"
    decs = db.list_policy_decisions(call_id)
    row.update(final_qid=qid if stored.phase is Phase.ASKING else f"{qid}:{stored.phase.value}",
               answer_state=stored.close_reason.value if stored.close_reason else ("-" if rec is None else rec.state.value),
               dispositions=",".join(f"{d.disposition.value}:{d.reason.value}:{d.cursor_action.value}" for d in decs) or "-",
               ms=box.get("ms", ""))
    if box.get("err"): row["error"] = row.get("error") or box["err"]
    return row


def _table(rows: list, *, live_mode: bool) -> None:
    cols = ("scenario", *(("ms",) if live_mode else ()), "tap", "speech", "final_qid", "answer_state", "dispositions")
    show = lambda r, c: str(r.get(c, ""))
    width = {c: max(len(c), *(len(show(r, c)) for r in rows)) for c in cols}
    emit = lambda vals: print("  ".join(str(vals[c]).ljust(width[c]) for c in cols))
    emit({c: c for c in cols}); emit({c: "-" * width[c] for c in cols})
    for r in rows: emit({c: show(r, c) for c in cols})


def verify(rows: list) -> list[str]:
    bad = []
    for row in rows:
        qid, ans, need = row["want"]
        name = row["scenario"]
        if row.get("error"): bad.append(f"{name}: {row['error']}")
        if qid != "*" and row.get("final_qid") != qid: bad.append(f"{name}: final_qid {row.get('final_qid')} != {qid}")
        if ans != "*" and row.get("answer_state") != ans: bad.append(f"{name}: answer_state {row.get('answer_state')} != {ans}")
        bad += [f"{name}: missing {tok}" for tok in filter(None, str(need).split(",")) if tok not in str(row.get("dispositions") or "")]
    return bad


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("offline", help="canned heads, temp SQLite, no network").add_argument("--verify", action="store_true")
    sub.add_parser("live", help="real gemini-decision-1; tap while the decision is in flight")
    args = parser.parse_args(argv)
    live_mode = args.cmd == "live"
    if live_mode:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    print(f"sqlite {os.environ['DATA_DIR']}/intake.db")

    async def run():
        out = []
        for case in CASES:
            print(f"[{args.cmd}] {case['id']} …", flush=True)
            out.append(await _race(case, live_mode=live_mode))
        return out

    rows = asyncio.run(run())
    print()
    _table(rows, live_mode=live_mode)
    for row in rows:
        if row.get("error"): print(f"! {row['scenario']}: {row['error']}", file=sys.stderr)
    if live_mode: return 1 if any(r.get("error") for r in rows) else 0
    if not getattr(args, "verify", False): return 0
    bad = verify(rows)
    for line in bad: print(f"verify {line}", file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
