"""Live Demo persona: Aanchal (आँचल) from `Live_Demo_script.docx.md`.

Two modes, same checklist:

  * local (default): boots `intake.app:app` on port 8794 with `DATA_DIR=intake/storage/demo_aanchal`, logs in over
    `/api/login`, drives Aanchal over `ws://127.0.0.1:8794/ws/live?pack=shishu-hi`, and grades from local SQLite.
  * remote (`--base-url https://…`, or `STAGING_URL`/`DEMO_BASE_URL`): logs in over `/api/login` on the deployed
    server, drives the same socket at `wss://<host>/ws/live?pack=shishu-hi`, and grades from the staging HTTP API
    (`/api/calls/<id>`, `/api/calls/<id>/patient.pdf`, `/api/audio/<id>`). Nothing under `intake/` or `packs/` is
    touched by this mode.

Flow under test:
  1. Name: "आँचल"
  2. Married: "हो गई है"
  3. Children (2-turn partial roster):
       Turn 1: "दो बच्चे हैं।"  -> must re-ask for ages
       Turn 2: "एक 8 साल का बेटा और 5 साल की बेटी।" -> 2 children (8yo male, 5yo female)
  4. Bridge: "ठीक है।"
  5. Birth mode: "नॉर्मल, और पूरे 9 महीने पर।" -> birth_mode_detail=CLEAR (and ideally auto-fills birth_term=CLEAR)
  6. NICU (meta-question + compound jaundice):
       Turn 1: "NICU क्या होता है?" -> must clarify NICU and STAY on birth_nicu (never burn the answer)
       Turn 2: "अच्छा समझी, हाँ भर्ती हुई थी मेरी छोटी बेटी। उसको नीली लाइट के नीचे रखना पड़ा था क्योंकि उसको जन्म के बाद पीलिया हुआ था।"
       -> birth_nicu=CONCERN (attributed to 5yo daughter), birth_jaundice=CONCERN (main question skipped)
  7. Remaining birth questions up to birth_cleft: "नहीं।"
  8. birth_cleft (active emergency):
       "नहीं, लेकिन मेरी छोटी बच्ची अभी बहुत तेज़ साँस ले रही है और मुझे लग रहा है साँस लेने में दिक्कत हो रही है उसको।"
       -> immediate emergency close, 108/102 + hospital advice, incomplete emergency-banner PDF.

Writes full checklist + transcript to `intake/storage/demo_aanchal/results.json`.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import sys
import time
import urllib.parse
import urllib.request
import wave
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("DATA_DIR", str(ROOT / "intake" / "storage" / "demo_aanchal"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from pypdf import PdfReader  # noqa: E402

from intake import db, reports  # noqa: E402
from intake.models import AnswerStatus, CallState, Phase, SubStep  # noqa: E402
from intake.pack import DEFAULT_PACK_ID, load_pack  # noqa: E402

HOST, PORT = "127.0.0.1", 8794
BASE, CALL_DEADLINE_SEC, RECV_TIMEOUT_SEC = f"http://{HOST}:{PORT}", 900, 300
OUT_DIR = ROOT / "intake" / "storage" / "demo_aanchal"
_LEAKS = (
    "next_question", "system cue", "ask only this", "active_qid", "last_answered_qid", "recorded_concerns",
    "duchenne", "hemophilia", "pompe", "gaucher", "scid", "wiskott",
    "ड्यूशेन", "हीमोफीलिया", "पॉम्पे", "गॉशर", "इम्यूनोडेफिशिएंसी", "म्यूकोपॉलीसेकेराइडोसिस",
)
C, CL = AnswerStatus.CONCERN, AnswerStatus.CLEAR


def _line(text: str) -> tuple[str, str]:
    return ("text", text)


def _steps(*replies: tuple[str, str]) -> tuple[str, tuple[tuple[str, str], ...]]:
    return ("steps", replies)


def _answered(state: CallState, qid: str) -> tuple[AnswerStatus | None, str, list[str]]:
    record = state.answers.get(qid)
    if record is None:
        return None, "", []
    return record.state, record.verbatim_text, list(record.item_ids or [])


def _pdf_from_bytes(blob: bytes | None) -> tuple[int, str]:
    if not blob or not blob.startswith(b"%PDF"):
        return 0, ""
    reader = PdfReader(io.BytesIO(blob))
    return len(reader.pages), " ".join(" ".join(p.extract_text() or "" for p in reader.pages).split())


def _get_json(url: str, cookie: str, timeout: float = 30.0) -> Any:
    request = urllib.request.Request(url, headers={"Cookie": cookie})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def _get_bytes(url: str, cookie: str, timeout: float = 120.0) -> bytes | None:
    request = urllib.request.Request(url, headers={"Cookie": cookie})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except Exception:  # noqa: BLE001
        return None


class LocalSource:
    """Grade from the SQLite rows and on-disk artifacts of the in-process server."""

    kind = "local"

    def call(self, call_id: int) -> dict[str, Any]:
        return db.get_call(call_id) or {}

    def observations(self, call_id: int) -> list[Any]:
        return db.list_retained_observations(call_id)

    def pdf(self, call_id: int) -> bytes | None:
        path = Path(db.DATA_DIR) / "pdfs" / f"call-{int(call_id)}.pdf"
        return path.read_bytes() if path.is_file() else None

    def wav(self, call_id: int, call: dict[str, Any]) -> bytes | None:
        path = Path(db.DATA_DIR) / "audio" / str(call.get("audio_path") or "")
        return path.read_bytes() if path.is_file() else None


class RemoteSource:
    """Grade a deployed server through its own HTTP API: the row, the PDF, and the audit WAV."""

    kind = "remote"

    def __init__(self, base: str, cookie: str) -> None:
        self.base, self.cookie = base.rstrip("/"), cookie

    def call(self, call_id: int) -> dict[str, Any]:
        return _get_json(f"{self.base}/api/calls/{int(call_id)}", self.cookie) or {}

    def observations(self, call_id: int) -> list[Any]:
        return []  # retained observations are not exposed over HTTP; parity is graded with a caveat

    def pdf(self, call_id: int) -> bytes | None:
        return _get_bytes(f"{self.base}/api/calls/{int(call_id)}/patient.pdf?lang=hi", self.cookie)

    def wav(self, call_id: int, call: dict[str, Any]) -> bytes | None:
        return _get_bytes(f"{self.base}/api/audio/{int(call_id)}", self.cookie)


def _state_of(call: dict[str, Any]) -> CallState:
    return db.call_state_from_row(call)





class Checklist:
    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        self.items.append({"check": name, "ok": bool(ok), "detail": "" if ok else detail})
        return bool(ok)

    @property
    def problems(self) -> list[str]:
        return [
            f"{item['check']}: {item['detail']}" if item["detail"] else item["check"]
            for item in self.items
            if not item["ok"]
        ]


class DemoAanchalPersona:
    """Scripted Aanchal caller following `Live_Demo_script.docx.md`."""

    pid = "demo_aanchal"
    name = "आँचल"

    def __init__(self) -> None:
        self.counts: dict[str, int] = {}
        self.turns: list[dict[str, Any]] = []
        self.lines: dict[str, Any] = {
            "intro_name": _line("आँचल"),
            "intro_married": _line("हो गई है।"),
            "intro_children": _steps(
                _line("दो बच्चे हैं।"),
                _line("एक 8 साल का बेटा और 5 साल की बेटी।"),
            ),
            "intro_bridge": _line("ठीक है।"),
            "birth_mode_detail": _line("नॉर्मल, और पूरे 9 महीने पर।"),
            "birth_term": _line("नहीं, पूरे 9 महीने पर हुए थे।"),
            "birth_nicu": _steps(
                _line("NICU क्या होता है?"),
                _line(
                    "अच्छा समझी, हाँ भर्ती हुई थी मेरी छोटी बेटी। "
                    "उसको नीली लाइट के नीचे रखना पड़ा था क्योंकि उसको जन्म के बाद पीलिया हुआ था।"
                ),
            ),
            "birth_nicu:which_item": _line("मेरी छोटी बेटी को, 5 साल वाली।"),
            "birth_nicu:fu0": _line("पीलिया हुआ था, नीली लाइट में रखा था।"),
            "birth_nicu:fu1": _line("तीन दिन रखा था।"),
            # birth_jaundice should be skipped by auto-fill; keep fallbacks just in case it is asked.
            "birth_jaundice": _line("हाँ, मेरी छोटी बेटी को जन्म के बाद पीलिया हुआ था और नीली लाइट में रखा था।"),
            "birth_jaundice:which_item": _line("मेरी छोटी बेटी को, 5 साल वाली।"),
            "birth_jaundice:fu0": _line("तीन दिन रखा था।"),
            "birth_cleft": _line(
                "नहीं, लेकिन मेरी छोटी बच्ची अभी बहुत तेज़ साँस ले रही है "
                "और मुझे लग रहा है साँस लेने में दिक्कत हो रही है उसको।"
            ),
        }

    @staticmethod
    def route(state: CallState) -> str:
        step, qid = state.cursor, state.cursor.qid
        if not qid:
            return "unknown"
        if step.substep is SubStep.WHICH_ITEM:
            return f"{qid}:which_item"
        return f"{qid}:fu{step.followup_idx}" if step.substep is SubStep.FOLLOWUP else qid

    def answer(self, state: CallState) -> tuple[str, str] | None:
        if state.phase is not Phase.ASKING:
            return None
        key = self.route(state)
        self.counts[key] = self.counts.get(key, 0) + 1
        if self.counts[key] > 4:
            raise RuntimeError(f"stuck on {key}")
        picked = self.lines.get(key)
        if picked is None:
            if state.cursor.substep is SubStep.WHICH_ITEM:
                reply = ("quick", "clear")
            else:
                reply = ("text", "पता नहीं।" if state.cursor.substep is SubStep.FOLLOWUP else "नहीं।")
        elif picked[0] == "steps":
            replies = picked[1]
            reply = replies[min(self.counts[key] - 1, len(replies) - 1)]
        else:
            reply = picked
        self.turns.append({"route": key, "attempt": self.counts[key], "mode": reply[0], "payload": reply[1]})
        return reply


def _login(base: str = BASE) -> str:
    passcode = (os.environ.get("APP_PASSCODE") or "").strip()
    if not passcode:
        raise RuntimeError("APP_PASSCODE is not set in .env")
    request = urllib.request.Request(
        base.rstrip("/") + "/api/login",
        data=json.dumps({"username": "demo_aanchal", "passcode": passcode}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        cookie = (response.headers.get("Set-Cookie") or "").split(";", 1)[0]
    if not cookie.startswith("shishu_auth="):
        raise RuntimeError("login did not set shishu_auth")
    return cookie


def _admin_login(base: str = BASE) -> str:
    """Admin cookie for the read-only endpoints the grader needs (`/api/calls/<id>`, PDF, WAV)."""
    passcode = (os.environ.get("ADMIN_PASSCODE") or "").strip()
    if not passcode:
        raise RuntimeError("ADMIN_PASSCODE is not set in .env")
    request = urllib.request.Request(
        base.rstrip("/") + "/api/login",
        data=json.dumps({"username": "demo_aanchal", "passcode": passcode}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        cookie = (response.headers.get("Set-Cookie") or "").split(";", 1)[0]
    if not cookie.startswith("shishu_auth="):
        raise RuntimeError("admin login did not set shishu_auth")
    return cookie


def _grade(pack, call_id: int, persona: DemoAanchalPersona, source: Any) -> dict[str, Any]:
    call = source.call(call_id) or {}
    state = _state_of(call)
    cl = Checklist()

    telemetry = call.get("telemetry") or {}
    transcript = call.get("transcript") or []
    assistant_turns = [str(t.get("text") or "") for t in transcript if t.get("role") == "assistant"]
    spoken = " ".join(assistant_turns).lower()

    # 1. Runtime health
    cl.check("tool_errors == 0", int(telemetry.get("tool_errors") or 0) == 0, f"got {telemetry.get('tool_errors')}")
    cl.check("watchdog_nudges == 0", int(telemetry.get("watchdog_nudges") or 0) == 0, f"got {telemetry.get('watchdog_nudges')}")
    leaks = [leak for leak in _LEAKS if leak.lower() in spoken]
    cl.check("no spoken leaks", not leaks, f"leaked {leaks}")

    # 2. Profile & partial roster re-ask
    pname = str(call.get("patient_name") or "")
    cl.check("patient_name is Aanchal", ("आँचल" in pname or "आंचल" in pname or "aanchal" in pname.lower()), f"got {pname!r}")
    cl.check("intro_children re-asked for ages", persona.counts.get("intro_children", 0) >= 2, f"asks={persona.counts.get('intro_children', 0)}")
    roster_summary = [(c.tag, c.age_years, c.id) for c in state.roster]
    has_son_8 = any(c.tag == "male" and c.age_years == 8.0 for c in state.roster)
    has_daughter_5 = any(c.tag == "female" and c.age_years == 5.0 for c in state.roster)
    cl.check("roster has 8yo son and 5yo daughter", len(state.roster) == 2 and has_son_8 and has_daughter_5, f"roster={roster_summary}")

    # 3. Birth mode & term
    cl.check("birth_mode_detail == CLEAR", _answered(state, "birth_mode_detail")[0] is CL, f"got {_answered(state, 'birth_mode_detail')}")
    cl.check("birth_term == CLEAR", _answered(state, "birth_term")[0] is CL, f"got {_answered(state, 'birth_term')}")

    # 4. NICU meta-question + attribution + jaundice auto-fill skip
    cl.check("birth_nicu stayed after meta-question", persona.counts.get("birth_nicu", 0) >= 2, f"asks={persona.counts.get('birth_nicu', 0)}")
    nicu_status, nicu_text, nicu_ids = _answered(state, "birth_nicu")
    cl.check("birth_nicu == CONCERN", nicu_status is C, f"got {nicu_status} ({nicu_text!r})")
    daughter = next((c for c in state.roster if c.tag == "female" and c.age_years == 5.0), None)
    cl.check("birth_nicu attributed to 5yo daughter", daughter is not None and daughter.id in nicu_ids, f"ids={nicu_ids}, daughter={daughter}")

    jaundice_status, jaundice_text, _ = _answered(state, "birth_jaundice")
    cl.check("birth_jaundice == CONCERN", jaundice_status is C, f"got {jaundice_status} ({jaundice_text!r})")
    cl.check("birth_jaundice main question skipped (auto-filled)", persona.counts.get("birth_jaundice", 0) == 0, f"asks={persona.counts.get('birth_jaundice', 0)}")

    # 5. Active emergency on birth_cleft
    cl.check("phase == CLOSING", state.phase is Phase.CLOSING, f"got {state.phase}")
    cl.check("close_reason == emergency", str(state.close_reason or "") == "emergency", f"got {state.close_reason!r}")
    last_bot = assistant_turns[-1] if assistant_turns else ""
    cl.check("closing advises 108 or 102", ("108" in last_bot or "102" in last_bot or "१०८" in last_bot or "१०२" in last_bot), f"last_bot={last_bot!r}")

    # 6. Reports, PDF, WAV
    proj = reports.project_summary(pack, state, source.observations(call_id))
    cl.check("proj.emergency is True", bool(proj.emergency), "proj.emergency is False")
    cl.check("doctor_markdown has Active emergency", "Active emergency" in proj.doctor_markdown, "missing Active emergency")
    diverged = [
        col for col, field in (("summary_doctor", "doctor_markdown"), ("summary_patient", "patient_summary_text"))
        if str(call.get(col) or "").strip() != str(getattr(proj, field) or "").strip()
    ]
    if (call.get("red_flags") or []) != list(proj.red_flags):
        diverged.append("red_flags")
    if source.kind == "remote" and "summary_doctor" in diverged:
        # A deployed server writes both summary columns at finalize; a mismatch here means the box rendered a
        # different projection than this script would. Keep it visible but name the caveat.
        diverged[diverged.index("summary_doctor")] = "summary_doctor (remote row vs local projection)"
    cl.check("project_summary parity", not diverged, f"diverges: {diverged}")

    pages, pdf_text = _pdf_from_bytes(source.pdf(call_id))
    cl.check("emergency PDF exists with 108 and अधूरी", pages >= 1 and "108" in pdf_text and "अधूरी" in pdf_text, f"pages={pages}, has_108={'108' in pdf_text}, has_incomplete={'अधूरी' in pdf_text}")

    wav_ok, wav_detail, wav_bytes = False, "missing wav", source.wav(call_id, call)
    if wav_bytes:
        try:
            with wave.open(io.BytesIO(wav_bytes), "rb") as handle:
                wav_ok = (handle.getnchannels(), handle.getframerate()) == (2, 16000)
                wav_detail = f"ch={handle.getnchannels()} sr={handle.getframerate()}"
        except wave.Error as exc:
            wav_detail = f"unreadable ({exc})"
    cl.check("stereo 16 kHz WAV", wav_ok, wav_detail)

    return {
        "persona": persona.pid,
        "call_id": call_id,
        "ok": not cl.problems,
        "problems": cl.problems,
        "checklist": cl.items,
        "close_reason": str(state.close_reason or ""),
        "phase": str(state.phase),
        "patient_name": pname,
        "roster": roster_summary,
        "asks": persona.counts,
        "scripted_turns": persona.turns,
        "transcript": transcript,
        "duration_sec": call.get("duration_sec"),
        "share_token": call.get("share_token"),
    }


async def _run_call(pack, persona: DemoAanchalPersona, cookie: str, *, base: str, source: Any) -> dict[str, Any]:
    import websockets

    netloc = urllib.parse.urlparse(base).netloc
    scheme = "wss" if base.startswith("https") else "ws"
    url = f"{scheme}://{netloc}/ws/live?pack={pack.id}"
    print(f"\n=== {persona.pid} ({base}) ===", flush=True)
    try:
        async with websockets.connect(
            url,
            additional_headers={"Cookie": cookie},
            open_timeout=30,
            max_size=8_000_000,
            ping_interval=20,
            ping_timeout=30,
        ) as ws:
            call_id, heard, deadline, published = 0, False, time.monotonic() + CALL_DEADLINE_SEC, ""
            while time.monotonic() < deadline:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=RECV_TIMEOUT_SEC))
                kind, event = msg.get("type"), (msg.get("last_event") or {}).get("type")
                if kind == "session_started":
                    call_id = int(msg.get("call_id") or 0)
                    print(f"call {call_id}", flush=True)
                elif kind == "question_state":
                    published = str(msg.get("step") or "")
                elif kind == "bot_turn_complete":
                    published = str(msg.get("step") or published)
                elif kind == "telemetry" and event == "bot_first_audio":
                    heard = True
                elif kind == "telemetry" and event == "turn_complete" and heard and call_id:
                    heard = False
                    state = _state_for_step(_state_of(source.call(call_id) or {}), published)
                    reply = persona.answer(state)
                    if reply is not None:
                        mode, payload = reply
                        step = state.step_token or published
                        print(f"  [{persona.route(state)}] {mode} {payload[:80]}", flush=True)
                        await ws.send(
                            json.dumps(
                                {"type": "quick_reply", "value": payload, "step": step}
                                if mode == "quick"
                                else {"type": "text", "text": payload},
                                ensure_ascii=False,
                            )
                        )
                elif kind in {"call_completed", "report_failed"}:
                    final_id = int(msg.get("call_id") or call_id)
                    if kind == "report_failed":
                        print(f"report_failed on call {final_id}: {msg.get('message') or ''}", flush=True)
                    result = _grade(pack, final_id, persona, source)
                    result["target"] = base
                    status = "PASS" if result["ok"] else "FAIL"
                    print(f"{status} {persona.pid} call={result['call_id']} dur={result.get('duration_sec')}", flush=True)
                    for prob in result["problems"]:
                        print(f"  - {prob}", flush=True)
                    return result
                elif kind == "error":
                    return {
                        "persona": persona.pid,
                        "call_id": call_id,
                        "target": base,
                        "ok": False,
                        "problems": [f"error: {msg.get('message') or ''}"],
                    }
            return {"persona": persona.pid, "call_id": call_id, "target": base, "ok": False, "problems": ["deadline"]}
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL {persona.pid} {type(exc).__name__}: {exc}", flush=True)
        return {"persona": persona.pid, "call_id": 0, "target": base, "ok": False, "problems": [f"{type(exc).__name__}: {exc}"]}


def _state_for_step(row_state: CallState, published: str) -> CallState:
    """The cursor the next scripted line must answer.

    The row is authoritative (the box writes `cursor_json` before it speaks the step), so it is preferred. The
    published `step` token is the exact string the box stamped on that prompt, kept as the fallback for a lagging
    or unreadable row — and it is what a quick_reply must echo back verbatim.
    """
    if row_state.cursor.qid:
        return row_state
    step = (published or "").strip()
    if not step:
        return row_state
    head, _, tail = step.partition(":")
    if head == "closing":
        return row_state.model_copy(update={"phase": Phase.CLOSING, "close_reason": tail or None})
    sub, _, idx = tail.partition(":")
    substep = SubStep(sub) if sub in tuple(SubStep) else SubStep.MAIN
    cursor = type(row_state.cursor)(qid=head, substep=substep, followup_idx=int(idx) if idx.isdigit() else 0)
    return row_state.model_copy(update={"phase": Phase.ASKING, "close_reason": None, "cursor": cursor})




def _target_base(args: argparse.Namespace) -> str:
    raw = (args.base_url or os.environ.get("STAGING_URL") or os.environ.get("DEMO_BASE_URL") or "").strip()
    return raw.rstrip("/")


async def _remote_main(base: str) -> dict[str, Any]:
    print(f"remote target: {base}", flush=True)
    health = await asyncio.to_thread(_get_json, base + "/health", "", 25.0)
    print(f"health: {health}", flush=True)
    pack = load_pack(os.environ.get("INTAKE_PACK") or DEFAULT_PACK_ID)
    persona = DemoAanchalPersona()
    # The caller socket authenticates as the app user; every read-only read afterwards uses the admin cookie.
    cookie = await asyncio.to_thread(_login, base)
    read_cookie = await asyncio.to_thread(_admin_login, base)
    result = await _run_call(pack, persona, cookie, base=base, source=RemoteSource(base, read_cookie))
    result["staging_health"] = health
    return result


async def _local_main() -> dict[str, Any]:
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
        except Exception:  # noqa: BLE001
            await asyncio.sleep(0.1)

    pack = load_pack(os.environ.get("INTAKE_PACK") or DEFAULT_PACK_ID)
    persona = DemoAanchalPersona()
    try:
        cookie = await asyncio.to_thread(_login, BASE)
        return await _run_call(pack, persona, cookie, base=BASE, source=LocalSource())
    finally:
        server.should_exit = True
        await task


async def _main() -> int:
    parser = argparse.ArgumentParser(description="Aanchal live-demo persona test (local or deployed staging).")
    parser.add_argument("--base-url", default=None, help="http(s) base URL of a deployed server, e.g. https://shishu-staging.up.railway.app")
    parser.add_argument("--out", default=None, help=f"results path (default: {OUT_DIR / 'results.json'})")
    args = parser.parse_args()
    base = _target_base(args)

    result = await (_remote_main(base) if base else _local_main())
    result["target"] = base or BASE

    out = Path(args.out) if args.out else OUT_DIR / "results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {out}", flush=True)
    return 0 if result.get("ok") else 1



if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
