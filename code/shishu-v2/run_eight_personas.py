"""Eight live Gemini calls against the v3 cursor. Run inside tmux.

Exits 0 only when every persona has a stereo WAV, both summaries, red flags,
and a 2-page PDF, with no tool errors and no spoken instruction leaks.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
os.environ.setdefault("DATA_DIR", str(HERE / "storage" / "personas"))
sys.path.insert(0, str(HERE))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")
load_dotenv()

import db  # noqa: E402
import uvicorn  # noqa: E402
from pypdf import PdfReader  # noqa: E402
from triage_catalog import FOLLOWUP_HI, active_specs  # noqa: E402
import app as shishu_app  # noqa: E402

import io
import urllib.request

HOST = "127.0.0.1"
PORT = 8791
BASE = f"http://{HOST}:{PORT}"

_LEAKS = (
    "save_profile",
    "record_turn",
    "say_hindi",
    "system cue",
    "ask only this",
    "active_qid",
    "do_not_read",
    "duchenne",
    "hemophilia",
    "pompe",
    "gaucher",
    "scid",
    "wiskott",
    "ड्यूशेन",
    "हीमोफीलिया",
    "पॉम्पे",
    "गॉशर",
    "इम्यूनोडेफिशिएंसी",
    "म्यूकोपॉलीसेकेराइडोसिस",
)


def _questions() -> list[tuple[str, str]]:
    rows = [(spec.id, spec.question_hi) for spec in active_specs() if spec.question_hi]
    rows.extend((fid, text) for fid, text in FOLLOWUP_HI.items())
    rows.append(("which_child", "यह किस बच्चे की बात है"))
    rows.append(("ever_pregnant", "क्या आप कभी गर्भवती हुई हैं"))
    rows.sort(key=lambda item: len(item[1]), reverse=True)
    return rows


QUESTIONS = _questions()


def _classify(text: str) -> str:
    if "आपका नाम" in text:
        return "intro_name"
    if "क्या आपकी शादी" in text:
        return "intro_married"
    if "कितने बच्चे" in text:
        return "intro_children"
    if "कभी गर्भवती" in text:
        return "ever_pregnant"
    if "किस बच्चे" in text:
        return "which_child"
    if "आसान सवाल" in text:
        return "intro_bridge"
    if "इसके अलावा" in text or "कुछ और बताना" in text:
        return "close_anything_else"
    for qid, question in QUESTIONS:
        needle = question[:28]
        if len(needle) >= 12 and needle in text:
            return qid
    return ""


def _closing(text: str, phase: str) -> bool:
    if phase == "closing":
        return True
    if "108" in text and ("अस्पताल" in text or "कॉल" in text):
        return True
    if "रिपोर्ट तैयार" in text or "रिपोर्ट हो रही" in text:
        return True
    return False


class Persona:
    def __init__(self, pid: str, lines: dict[str, str], *, first_gate: str = "", stop_after: str = ""):
        self.pid = pid
        self.lines = lines
        self.first_gate = first_gate
        self.stop_after = stop_after
        self.counts: dict[str, int] = {}
        self.gate_seen = False
        self.stop_next = False

    def reply(self, text: str, state: dict) -> str | None:
        if _closing(text, str(state.get("phase") or "")):
            return None
        key = _classify(text) or str(state.get("active_qid") or "")
        if not key:
            key = "unknown"
        self.counts[key] = self.counts.get(key, 0) + 1
        if self.counts[key] > 4:
            raise RuntimeError(f"stuck on {key}: {text[:180]}")
        if self.stop_next:
            return "अभी बंद कर दो। बाद में बात करेंगे।"
        if key == "intro_name":
            return self.lines["intro_name"]
        if key == "intro_married":
            return self.lines["intro_married"]
        if key == "intro_children":
            return self.lines["intro_children"]
        if key == "ever_pregnant":
            return self.lines.get("ever_pregnant", "नहीं, कभी गर्भवती नहीं हुई।")
        if key == "intro_bridge":
            return "ठीक है।"
        if key == "which_child":
            return self.lines.get("which_child", "बड़े वाले को।")
        if key == "close_anything_else":
            return "नहीं, बस इतना ही।"
        if self.first_gate and key.startswith("gate_") and not self.gate_seen:
            self.gate_seen = True
            if self.stop_after == "emergency":
                return self.first_gate
        if key in self.lines:
            line = self.lines[key]
            if self.stop_after == "next" and key.startswith("gate_"):
                self.stop_next = True
            return line
        if self.counts[key] == 1 and key == "gate_birth_neonatal" and self.lines.get("_clarify"):
            return "क्या? समझ नहीं आया।"
        if key in FOLLOWUP_HI:
            return self.lines.get(key, self.lines.get("_followup", "दो बार हुआ।"))
        return "नहीं।"


def _personas() -> list[Persona]:
    return [
        Persona(
            "p1_scid_bcg",
            {
                "intro_name": "मेरा नाम पूजा है।",
                "intro_married": "हाँ, शादी हो गई है।",
                "intro_children": "एक बेटा है, दो साल का।",
                "gate_cord_bcg": "हाँ, BCG की जगह पर बड़ा फोड़ा हुआ और बगल में गाँठ थी।",
                "birth_cord_bleed": "हाँ, नाल से खून ज़्यादा आया था।",
                "birth_cord_3weeks": "हाँ, नाल गिरने में तीन हफ्ते से ज़्यादा लगे।",
                "birth_bcg": "हाँ, टीके की जगह पर बड़ा फोड़ा और बगल में गाँठ हुई।",
            },
        ),
        Persona(
            "p2_hemophilia",
            {
                "intro_name": "मेरा नाम रेखा है।",
                "intro_married": "हाँ, शादी हो गई है।",
                "intro_children": "एक बेटा है, सात साल का।",
                "gate_bleeding_joints": "हाँ, चोट पर खून देर तक बहता है और आसानी से नील पड़ते हैं।",
                "body_bleeding_bruise": "हाँ, चोट पर खून देर तक बहता है।",
                "body_joint_swelling": "हाँ, गिरने पर घुटने में सूजन आती है और कुछ दिनों में ठीक हो जाती है।",
                "mod_tooth_bleed": "हाँ, दूध का दाँत गिरने पर खून बहुत देर तक बहा।",
            },
        ),
        Persona(
            "p3_thal_3kids",
            {
                "intro_name": "मेरा नाम कविता है।",
                "intro_married": "हाँ, शादी हो गई है।",
                "intro_children": "तीन बच्चे हैं। बड़ी बेटी ग्यारह साल की, बीच वाला बेटा सात साल का, और छोटा बेटा दो साल का।",
                "which_child": "बड़ी वाली बेटी को।",
                "gate_pregnancy_thal": "हाँ, मेरी थैलेसीमिया जाँच में माइनर बताया गया।",
                "preg_thalassemia": "हाँ, मेरी जाँच में थैलेसीमिया माइनर बताया गया।",
                "preg_conditions": "नहीं।",
                "preg_scan": "नहीं।",
                "preg_transfusion": "नहीं।",
                "preg_delivery": "नहीं।",
            },
        ),
        Persona(
            "p4_unmarried",
            {
                "intro_name": "मेरा नाम नीलम है।",
                "intro_married": "नहीं, शादी नहीं हुई है।",
                "intro_children": "मेरे कोई बच्चे नहीं हैं।",
                "fam_loss_gate": "हाँ, परिवार में एक छोटे बच्चे की एक साल से कम उम्र में मौत हुई।",
                "fam_infant_death": "हाँ, छह महीने का था।",
            },
        ),
        Persona(
            "p5_childless",
            {
                "intro_name": "मेरा नाम सीमा है।",
                "intro_married": "हाँ, शादी हो गई है।",
                "intro_children": "मेरे कोई बच्चे नहीं हैं।",
                "ever_pregnant": "नहीं, मैं कभी गर्भवती नहीं हुई।",
                "fam_related": "हाँ, हम शादी से पहले रिश्तेदार थे।",
            },
        ),
        Persona(
            "p6_confused_clarify",
            {
                "intro_name": "मेरा नाम मीरा है।",
                "intro_married": "हाँ, शादी हो गई है।",
                "intro_children": "एक बेटी है, चार साल की।",
                "_clarify": "1",
                "gate_infections": "हाँ, बार-बार अस्पताल में भर्ती करना पड़ा और नस में एंटीबायोटिक लगी।",
                "ill_admissions": "हाँ, छह महीने में तीन बार भर्ती हुआ।",
                "ill_iv_antibiotics": "हाँ, नस में एंटीबायोटिक लगी थी।",
            },
        ),
        Persona(
            "p7_emergency_108",
            {
                "intro_name": "मेरा नाम आशा है।",
                "intro_married": "हाँ, शादी हो गई है।",
                "intro_children": "एक बेटा है, एक साल का।",
            },
            first_gate="बच्चा अभी साँस नहीं ले पा रहा है, होंठ नीले हैं। यह इमरजेंसी है।",
            stop_after="emergency",
        ),
        Persona(
            "p8_caller_stop",
            {
                "intro_name": "मेरा नाम रीना है।",
                "intro_married": "हाँ, शादी हो गई है।",
                "intro_children": "एक बेटा है, पाँच साल का।",
                "gate_birth_neonatal": "हाँ, जन्म के बाद काँच के बक्से में रखना पड़ा था।",
            },
            stop_after="next",
        ),
    ]


def _login() -> str:
    passcode = os.environ.get("APP_PASSCODE") or "shishu2026"
    body = json.dumps({"username": "persona", "passcode": passcode}).encode()
    request = urllib.request.Request(
        BASE + "/api/login",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        cookie = response.headers.get("Set-Cookie") or ""
    pair = cookie.split(";", 1)[0]
    if not pair.startswith("shishu_auth="):
        raise RuntimeError("login did not set shishu_auth")
    return pair


def _check(call_id: int, persona: str) -> dict:
    call = db.get_call(call_id) or {}
    problems = []
    telemetry = call.get("telemetry") or {}
    if int(telemetry.get("tool_errors") or 0):
        problems.append(f"tool_errors={telemetry.get('tool_errors')}")
    if int(telemetry.get("prompt_leaks") or 0):
        problems.append(f"prompt_leaks={telemetry.get('prompt_leaks')}")
    assistant = " ".join(
        str(turn.get("text") or "")
        for turn in (call.get("transcript") or [])
        if turn.get("role") == "assistant"
    ).lower()
    for leak in _LEAKS:
        if leak.lower() in assistant:
            problems.append(f"spoken leak {leak}")
    if not str(call.get("summary_doctor") or "").strip():
        problems.append("empty summary_doctor")
    if not str(call.get("summary_mother") or "").strip():
        problems.append("empty summary_mother")
    flags = call.get("red_flags") or []
    if not flags:
        problems.append("empty red_flags")
    if not str(call.get("mother_name") or "").strip():
        problems.append("mother not named")
    audio = Path(db.DATA_DIR) / "audio" / str(call.get("audio_path") or "")
    if not audio.is_file():
        problems.append("missing wav")
    else:
        with wave.open(str(audio), "rb") as wf:
            if wf.getnchannels() != 2 or wf.getframerate() != 16000:
                problems.append(f"wav {wf.getnchannels()}ch {wf.getframerate()}Hz")
    pdf_path = Path(db.DATA_DIR) / "mother-pdfs" / f"call-{call_id}.pdf"
    if not pdf_path.is_file() or not pdf_path.read_bytes().startswith(b"%PDF"):
        problems.append("missing pdf")
    else:
        pages = len(PdfReader(io.BytesIO(pdf_path.read_bytes())).pages)
        if pages < 2:
            problems.append(f"pdf pages {pages}")
    cursor = call.get("cursor_state") or {}
    return {
        "persona": persona,
        "call_id": call_id,
        "ok": not problems,
        "problems": problems,
        "close_reason": cursor.get("close_reason"),
        "phase": cursor.get("phase"),
        "flags": len(flags) if isinstance(flags, list) else 0,
        "mother_name": call.get("mother_name"),
        "child_age": call.get("child_age"),
        "share_token": call.get("share_token"),
    }


async def _one(persona: Persona, cookie: str) -> dict:
    import websockets

    uri = f"ws://{HOST}:{PORT}/ws/gemini"
    print(f"\n=== {persona.pid} ===", flush=True)
    try:
        async with websockets.connect(
            uri,
            additional_headers={"Cookie": cookie},
            open_timeout=30,
            max_size=8_000_000,
        ) as ws:
            call_id = 0
            buf = ""
            deadline = time.monotonic() + 480
            while time.monotonic() < deadline:
                timeout = 150 if persona.counts else 90
                raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
                msg = json.loads(raw)
                kind = msg.get("type")
                if kind == "session_started":
                    call_id = int(msg["call_id"])
                    if not msg.get("share_token"):
                        return {"persona": persona.pid, "call_id": call_id, "ok": False, "problems": ["no share_token"]}
                    print(f"call {call_id} token {msg.get('share_token')}", flush=True)
                elif kind == "transcript" and msg.get("role") == "assistant":
                    buf = (buf + " " + str(msg.get("text") or "")).strip()
                elif kind == "telemetry":
                    event = (msg.get("last_event") or {}).get("type")
                    if event == "turn_complete" and buf and call_id:
                        state = (db.get_call(call_id) or {}).get("cursor_state") or {}
                        reply = persona.reply(buf, state)
                        heard = buf[:80].replace("\n", " ")
                        buf = ""
                        if reply:
                            print(f"  [{state.get('active_qid')}] {heard} -> {reply[:60]}", flush=True)
                            await ws.send(json.dumps({"type": "text", "text": reply}, ensure_ascii=False))
                        else:
                            print(f"  closing heard: {heard}", flush=True)
                elif kind == "call_completed":
                    if not msg.get("share_token"):
                        return {"persona": persona.pid, "call_id": call_id, "ok": False, "problems": ["completed without share_token"]}
                    result = _check(int(msg.get("call_id") or call_id), persona.pid)
                    print(("PASS " if result["ok"] else "FAIL ") + persona.pid + " " + ", ".join(result["problems"]), flush=True)
                    return result
                elif kind in {"report_failed", "error"}:
                    return {
                        "persona": persona.pid,
                        "call_id": call_id,
                        "ok": False,
                        "problems": [f"{kind}: {msg.get('message') or ''}"],
                    }
            return {"persona": persona.pid, "call_id": call_id, "ok": False, "problems": ["deadline"]}
    except Exception as exc:
        print(f"FAIL {persona.pid} {type(exc).__name__}: {exc}", flush=True)
        return {"persona": persona.pid, "call_id": 0, "ok": False, "problems": [f"{type(exc).__name__}: {exc}"]}


async def _main() -> int:
    config = uvicorn.Config(shishu_app.app, host=HOST, port=PORT, log_level="warning")
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())

    def _ping() -> None:
        with urllib.request.urlopen(BASE + "/health", timeout=0.5) as response:
            response.read()

    ready = False
    for _ in range(50):
        if task.done():
            err = task.exception()
            raise RuntimeError(f"server exited: {err}")
        try:
            await asyncio.to_thread(_ping)
            ready = True
            break
        except Exception:
            await asyncio.sleep(0.1)
    if not ready:
        server.should_exit = True
        await task
        raise RuntimeError("server did not start")
    results = []
    try:
        cookie = await asyncio.to_thread(_login)
        wanted = set(sys.argv[1:])
        personas = [item for item in _personas() if not wanted or item.pid in wanted]
        for persona in personas:
            results.append(await _one(persona, cookie))
    finally:
        server.should_exit = True
        await task
    out = Path(db.DATA_DIR) / "results.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    passed = sum(1 for item in results if item.get("ok"))
    print(f"\n{passed}/{len(results)} PASS", flush=True)
    print(out, flush=True)
    return 0 if passed == len(results) and (len(results) == 8 or bool(sys.argv[1:])) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
