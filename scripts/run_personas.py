"""Three focused live personas against the pack-aware V3 runtime. Run inside `tmux`.

Boots `intake.app:app`, logs in over `/api/login`, then drives one persona per call over
`ws://127.0.0.1:<port>/ws/live?pack=<pack_id>`. The default suite is the three focused V3 personas; pass persona ids as
arguments to run a subset. Exits 0 only when every selected persona ends with both summaries, the expected red-flag
band, a valid 2-page Patient PDF, and no spoken leak, tool error, or watchdog nudge.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import sys
import time
import urllib.request
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("DATA_DIR", str(ROOT / "intake" / "storage" / "personas"))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from pypdf import PdfReader  # noqa: E402

from intake import db, reports  # noqa: E402
from intake.models import AnswerStatus, CallState, Phase, SubStep  # noqa: E402
from intake.pack import DEFAULT_PACK_ID, load_pack  # noqa: E402

HOST, PORT = "127.0.0.1", 8791
BASE, CALL_DEADLINE_SEC, RECV_TIMEOUT_SEC = f"http://{HOST}:{PORT}", 1200, 300
_LEAKS = ("next_question", "system cue", "ask only this", "active_qid", "last_answered_qid", "recorded_concerns",
          "duchenne", "hemophilia", "pompe", "gaucher", "scid", "wiskott", "ड्यूशेन", "हीमोफीलिया", "पॉम्पे",
          "गॉशर", "इम्यूनोडेफिशिएंसी", "म्यूकोपॉलीसेकेराइडोसिस")
_MARRIED, _BRIDGE = "हाँ, शादी हो गई है।", "ठीक है।"
C, CL = AnswerStatus.CONCERN, AnswerStatus.CLEAR


def _answered(call: dict, qid: str) -> tuple[AnswerStatus | None, str]:
    """One answer's resolved status and its verbatim note, from the single stored `CallState`."""
    record = db.call_state_from_row(call).answers.get(qid)
    return (record.state, record.verbatim_text) if record is not None else (None, "")


def _has(text: str, *needles: str) -> bool:
    return any(needle in text for needle in needles)


def _status(call: dict, problems: list[str], *wanted: tuple[str, AnswerStatus]) -> None:
    """Assert every named answer resolved as expected, naming the ones that did not."""
    problems += [f"{qid}={_answered(call, qid)[0]} expected {want}" for qid, want in wanted
                 if _answered(call, qid)[0] is not want]


def _pdf(call: dict) -> tuple[int, str]:
    """The Patient PDF's page count and flattened text; (0, "") when it is missing or not a PDF."""
    path = Path(db.DATA_DIR) / "pdfs" / f"call-{int(call.get('id') or 0)}.pdf"
    if not path.is_file() or not path.read_bytes().startswith(b"%PDF"):
        return 0, ""
    reader = PdfReader(io.BytesIO(path.read_bytes()))
    return len(reader.pages), " ".join(" ".join(p.extract_text() or "" for p in reader.pages).split())


def _projection(call: dict, pack, problems: list[str]):
    """One projection must feed the stored note, the patient summary, the flags, and the PDF."""
    proj = reports.project_summary(pack, db.call_state_from_row(call))
    for column, expected in (("summary_doctor", proj.doctor_markdown),
                             ("summary_patient", proj.patient_summary_text)):
        if str(call.get(column) or "").strip() != expected.strip():
            problems.append(f"{column} diverges from project_summary")
    if (call.get("red_flags") or []) != list(proj.red_flags):
        problems.append("red_flags diverges from project_summary")
    return proj


def _expect_p1(pack, call: dict, problems: list[str]) -> None:
    """NICU plus both follow-ups, reactive multi-fact auto-fill, and the maternal family redaction."""
    _status(call, problems, ("birth_nicu", C), ("body_bleeding_bruise", C), ("body_joint_swelling", C),
            ("fam_maternal_bleeding", C))
    nicu = _answered(call, "birth_nicu")[1]
    if not (_has(nicu, "तकलीफ", "साँस", "सांस") and _has(nicu, "4", "चार")):
        problems.append(f"nicu reason/days missing: {nicu}")
    if not _projection(call, pack, problems).red_flags:
        problems.append("no red flags projected")
    doctor, patient = str(call.get("summary_doctor") or ""), str(call.get("summary_patient") or "")
    if "हीमोफीलिया" not in doctor:
        problems.append("doctor summary missing the reported हीमोफीलिया")
    if "हीमोफीलिया" in patient or "हीमोफीलिया" in _pdf(call)[1]:
        problems.append("patient summary or PDF leaked हीमोफीलिया")


def _expect_p2(pack, call: dict, problems: list[str]) -> None:
    """An active emergency closes at 108 with an incomplete, emergency-banner PDF and no other concerns."""
    state = db.call_state_from_row(call)
    if state.phase is not Phase.CLOSING:
        problems.append(f"call did not close: phase={state.phase}")
    problems += [f"unexpected mid-emergency concern {q}" for q, r in state.answers.items()
                 if r.state is C and q != "birth_jaundice"]
    proj = _projection(call, pack, problems)
    if "Active emergency" not in proj.doctor_markdown or not proj.emergency:
        problems.append("doctor summary is not an emergency note")
    pdf = _pdf(call)[1]
    if "108" not in pdf or "अधूरी" not in pdf:
        problems.append("emergency PDF missing the 108 banner")


def _expect_p3(pack, call: dict, problems: list[str]) -> None:
    """Three children: middle-son attribution, a which_item retraction, and the closing open capture."""
    state = db.call_state_from_row(call)
    _status(call, problems, ("birth_jaundice", C), ("birth_bcg", CL), ("close_anything_else", C))
    middle = next(iter(sorted(state.roster, key=lambda child: child.age_years, reverse=True)[1:2]), None)
    ids = state.answers["birth_jaundice"].item_ids if "birth_jaundice" in state.answers else []
    if middle is None or middle.id not in ids:
        problems.append(f"jaundice attributed to {ids}, middle child is {middle}")
    if not _has(_answered(call, "birth_jaundice")[1], "3", "तीन"):
        problems.append(f"jaundice days missing: {_answered(call, 'birth_jaundice')[1]}")
    _projection(call, pack, problems)


def _line(text: str) -> tuple:
    return ("text", text)


def _steps(*replies: tuple) -> tuple:
    return ("steps", replies)


class Persona:
    """A scripted caller: one reply per typed cursor step, with a stuck-step guard."""

    def __init__(self, pid: str, name: str, lines: dict[str, tuple], *,
                 flags: tuple[int, int], close: str, check) -> None:
        self.pid, self.name, self.lines = pid, name, lines
        self.flags, self.close, self.check, self.counts = flags, close, check, {}

    def answer(self, state: CallState) -> tuple[str, str] | None:
        """The next thing this caller says, or None once the line is closing."""
        if state.phase is not Phase.ASKING:
            return None
        key = self.route(state)
        self.counts[key] = self.counts.get(key, 0) + 1
        if self.counts[key] > 4:
            raise RuntimeError(f"stuck on {key}")
        picked = self.lines.get(key)
        if picked is None:
            return self.fallback(state)
        if callable(picked):
            return picked(state)
        return picked[1][min(self.counts[key] - 1, len(picked[1]) - 1)] if picked[0] == "steps" else picked

    @staticmethod
    def route(state: CallState) -> str:
        """The typed cursor step is the routing key. Transcript text never is."""
        step, qid = state.cursor, state.cursor.qid
        if not qid:
            return "unknown"
        if step.substep is SubStep.WHICH_ITEM:
            return f"{qid}:which_item"
        return f"{qid}:fu{step.followup_idx}" if step.substep is SubStep.FOLLOWUP else qid

    @staticmethod
    def fallback(state: CallState) -> tuple[str, str]:
        """An unscripted step still answers in kind, so a surprise never stalls the call."""
        if state.cursor.substep is SubStep.WHICH_ITEM:
            return ("quick", "clear")
        return ("text", "पता नहीं।" if state.cursor.substep is SubStep.FOLLOWUP else "नहीं।")


def _intro(name: str, children: str, extra: dict[str, tuple]) -> dict[str, tuple]:
    """The four intro lines every persona shares, plus its own answers."""
    return {"intro_name": _line(f"मेरा नाम {name} है।"), "intro_married": _line(_MARRIED),
            "intro_children": _line(children), "intro_bridge": _line(_BRIDGE)} | extra


def _personas() -> list[Persona]:
    """The three focused V3 personas. Pass ids on the command line to run a subset."""
    return [
        Persona(
            "p1_red_flags_and_redaction", "रेखा",
            _intro("रेखा", "एक बेटा है, 7 साल का।", {
                "birth_nicu": _line("हाँ।"), "birth_nicu:fu0": _line("साँस की तकलीफ थी।"),
                "birth_nicu:fu1": _line("चार दिन।"),
                # One multi-fact turn: the model must auto-fill both bleeding concerns together.
                "body_bleeding_bruise": _line("हाँ, चोट पर खून देर तक बहता है और घुटने में भी सूजन आती है।"),
                "body_joint_swelling": _line("हाँ, गिरने पर घुटने में सूजन आती है।"),
                "fam_maternal_bleeding": _line("हाँ, मेरे सगे मामा के बेटे को डॉक्टर ने हीमोफीलिया बताया था।"),
                "fam_maternal_bleeding:fu0": _line("सगे मामा के बेटे को।")}),
            flags=(3, 6), close="complete", check=_expect_p1),
        Persona(
            "p2_emergency_midway", "आशा",
            _intro("आशा", "एक बेटा है, 2 साल का।", {
                "birth_jaundice": _line("बच्चा अभी साँस नहीं ले पा रहा है और होंठ नीले हो रहे हैं, यह इमरजेंसी है!")}),
            flags=(1, 3), close="emergency", check=_expect_p2),
        Persona(
            "p3_multichild_reactive_jump", "कविता",
            _intro("कविता", "तीन बच्चे हैं। 11 साल की बेटी, 6 साल का बीच वाला बेटा, 2 साल का छोटा बेटा।", {
                "birth_jaundice": _line("हाँ।"), "birth_jaundice:which_item": _line("बीच वाले बेटे को।"),
                "birth_jaundice:fu0": _line("तीन दिन।"), "birth_bcg": _line("हाँ।"),
                "birth_bcg:which_item": _line("नहीं, गलती से हाँ बोला, कुछ नहीं हुआ था।"),
                "close_anything_else": _steps(_line("हाँ, रात में नींद में डर लगता है।"), ("quick", "clear"))}),
            flags=(2, 5), close="complete", check=_expect_p3),
    ]


def _login() -> str:
    """The `shishu_auth` cookie pair for the shared persona passcode."""
    passcode = (os.environ.get("APP_PASSCODE") or "").strip()
    if not passcode:
        raise RuntimeError("APP_PASSCODE is not set in .env")
    request = urllib.request.Request(
        BASE + "/api/login", data=json.dumps({"username": "persona", "passcode": passcode}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=20) as response:
        cookie = (response.headers.get("Set-Cookie") or "").split(";", 1)[0]
    if not cookie.startswith("shishu_auth="):
        raise RuntimeError("login did not set shishu_auth")
    return cookie


def _check(pack, call_id: int, persona: Persona) -> dict:
    """Every assertion one finished call must satisfy, reported as a list a human can act on."""
    call, problems = db.get_call(call_id) or {}, []
    telemetry, spoken = call.get("telemetry") or {}, " ".join(
        str(t.get("text") or "") for t in call.get("transcript") or [] if t.get("role") == "assistant").lower()
    problems += [f"{k}={telemetry.get(k)}" for k in ("tool_errors", "watchdog_nudges") if int(telemetry.get(k) or 0)]
    problems += [f"spoken leak {leak}" for leak in _LEAKS if leak.lower() in spoken]
    problems += [f"empty {f}" for f in ("summary_doctor", "summary_patient") if not str(call.get(f) or "").strip()]
    flags = len(call.get("red_flags") or [])
    if not persona.flags[0] <= flags <= persona.flags[1]:
        problems.append(f"red_flags={flags} expected {persona.flags[0]}-{persona.flags[1]}")
    if persona.name not in str(call.get("patient_name") or ""):
        problems.append(f"patient_name={call.get('patient_name')}")
    wav = Path(db.DATA_DIR) / "audio" / str(call.get("audio_path") or "")
    if not wav.is_file():
        problems.append("missing wav")
    else:
        with wave.open(str(wav), "rb") as handle:
            if (handle.getnchannels(), handle.getframerate()) != (2, 16000):
                problems.append("wav is not stereo 16 kHz")
    pages, text = _pdf(call)
    if not pages:
        problems.append("missing pdf")
    elif (persona.close == "emergency" and ("108" not in text or "अधूरी" not in text)) \
            or (persona.close != "emergency" and pages < 2):
        problems.append(f"pdf pages={pages} banner={'108' in text}")
    state = db.call_state_from_row(call)
    if str(state.close_reason or "") != persona.close:
        problems.append(f"close_reason={state.close_reason} expected {persona.close}")
    try:
        persona.check(pack, call, problems)
    except Exception as exc:  # noqa: BLE001 - a broken check reports, it never aborts the suite
        problems.append(f"check {type(exc).__name__}: {exc}")
    return {"persona": persona.pid, "call_id": call_id, "ok": not problems, "problems": problems,
            "close_reason": str(state.close_reason or ""), "phase": str(state.phase), "flags": flags,
            "patient_name": call.get("patient_name"), "share_token": call.get("share_token"),
            "duration_sec": call.get("duration_sec"), "asks": persona.counts}


async def _one(pack, persona: Persona, cookie: str) -> dict:
    """Drive one persona through one live call, from `session_started` to `call_completed`."""
    import websockets

    print(f"\n=== {persona.pid} ===", flush=True)
    try:
        async with websockets.connect(f"ws://{HOST}:{PORT}/ws/live?pack={pack.id}",
                                      additional_headers={"Cookie": cookie},
                                      open_timeout=30, max_size=8_000_000) as ws:
            call_id, heard, deadline = 0, False, time.monotonic() + CALL_DEADLINE_SEC
            while time.monotonic() < deadline:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=RECV_TIMEOUT_SEC))
                kind, event = msg.get("type"), (msg.get("last_event") or {}).get("type")
                if kind == "session_started":
                    call_id = int(msg.get("call_id") or 0)
                    if not msg.get("share_token"):
                        return {"persona": persona.pid, "call_id": call_id, "ok": False,
                                "problems": ["no share_token"]}
                    print(f"call {call_id}", flush=True)
                elif kind == "telemetry" and event == "bot_first_audio":
                    heard = True
                elif kind == "telemetry" and event == "turn_complete" and heard and call_id:
                    heard = False
                    state = db.call_state_from_row(db.get_call(call_id) or {})
                    reply = persona.answer(state)
                    if reply is not None:
                        mode, payload = reply
                        print(f"  [{Persona.route(state)}] {mode} {payload[:70]}", flush=True)
                        await ws.send(json.dumps(
                            {"type": "quick_reply", "value": payload, "step": state.step_token}
                            if mode == "quick" else {"type": "text", "text": payload}, ensure_ascii=False))
                elif kind == "call_completed":
                    result = _check(pack, int(msg.get("call_id") or call_id), persona)
                    print(("PASS " if result["ok"] else "FAIL ") + persona.pid
                          + f" call={result['call_id']} dur={result.get('duration_sec')} "
                          + ", ".join(result["problems"]), flush=True)
                    return result
                elif kind in {"report_failed", "error"}:
                    return {"persona": persona.pid, "call_id": call_id, "ok": False,
                            "problems": [f"{kind}: {msg.get('message') or ''}"]}
            return {"persona": persona.pid, "call_id": call_id, "ok": False, "problems": ["deadline"]}
    except Exception as exc:  # noqa: BLE001 - a dropped socket is a failed persona, not a crash
        print(f"FAIL {persona.pid} {type(exc).__name__}: {exc}", flush=True)
        return {"persona": persona.pid, "call_id": 0, "ok": False, "problems": [f"{type(exc).__name__}: {exc}"]}


async def _main() -> int:
    """Boot the app in-process, run the selected personas over live sockets, and grade the run."""
    import uvicorn

    from intake.app import app

    server = uvicorn.Server(uvicorn.Config(app, host=HOST, port=PORT, log_level="warning"))
    task = asyncio.create_task(server.serve())

    def ping() -> None:
        with urllib.request.urlopen(BASE + "/health", timeout=0.5) as response:
            response.read()

    for _ in range(50):
        if task.done():
            raise RuntimeError(f"server exited: {task.exception()}")
        try:
            await asyncio.to_thread(ping)
            break
        except Exception:  # noqa: BLE001 - the socket is simply not listening yet
            await asyncio.sleep(0.1)
    pack, wanted = load_pack(os.environ.get("INTAKE_PACK") or DEFAULT_PACK_ID), set(sys.argv[1:])
    personas = [p for p in _personas() if not wanted or p.pid in wanted]
    results = []
    try:
        if not personas:
            print("no matching persona", flush=True)
            return 1
        cookie = await asyncio.to_thread(_login)
        for persona in personas:
            results.append(await _one(pack, persona, cookie))
    finally:
        server.should_exit = True
        await task
    out = Path(db.DATA_DIR) / "results.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    passed = sum(1 for item in results if item.get("ok"))
    print(f"\n{passed}/{len(results)} PASS\n{out}", flush=True)
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
