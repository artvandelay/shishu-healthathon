"""Six live Gemini calls against the v4 cursor. Run inside tmux.

The filename stays `run_eight_personas.py` so existing docs keep working.
Default suite is the six focused personas. Exits 0 only when every selected
persona has a stereo WAV, both summaries, the expected red-flag band, and a
valid Mother PDF, with no tool errors, no watchdog nudges, and no leaks.
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

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
os.environ.setdefault("DATA_DIR", str(HERE / "storage" / "personas"))
sys.path.insert(0, str(HERE))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")
load_dotenv()

import db  # noqa: E402
import gemini_live  # noqa: E402
import uvicorn  # noqa: E402
from call_cursor import expect_yes_no  # noqa: E402
from pypdf import PdfReader  # noqa: E402
from triage_catalog import FOLLOWUP_HI, active_specs, get_spec  # noqa: E402
import app as shishu_app  # noqa: E402

HOST = "127.0.0.1"
PORT = 8791
BASE = f"http://{HOST}:{PORT}"
CALL_DEADLINE_SEC = 1200
RECV_TIMEOUT_SEC = 300

_LEAKS = (
    "save_profile",
    "record_turn",
    "say_hindi",
    "system cue",
    "ask only this",
    "active_qid",
    "last_answered_qid",
    "answered_qids",
    "target_qid",
    "recorded_concerns",
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
_MARRIED = "हाँ, शादी हो गई है।"
_BRIDGE = "ठीक है।"
_RETRACT = "नहीं, गलती से हाँ बोला, कुछ नहीं हुआ था।"


def _questions() -> list[tuple[str, str]]:
    rows = [(spec.id, spec.question_hi) for spec in active_specs() if spec.question_hi]
    rows.extend((fid, text) for fid, text in FOLLOWUP_HI.items())
    rows.append(("which_child", "यह किस बच्चे की बात है"))
    rows.append(("ever_pregnant", "क्या आप कभी गर्भवती हुई हैं"))
    rows.sort(key=lambda item: len(item[1]), reverse=True)
    return rows


QUESTIONS = _questions()


def _route(state: dict) -> str:
    """Cursor step is the routing key. Transcript text is not."""
    qid = str(state.get("active_qid") or "")
    if not qid:
        return ""
    idx = state.get("pending_followup_idx")
    if idx == -1:
        return f"{qid}:which_child"
    if isinstance(idx, int) and idx >= 0:
        try:
            followups = get_spec(qid).followups
        except KeyError:
            followups = ()
        if idx < len(followups):
            return followups[idx]
        return f"{qid}:{idx}"
    return qid


def _classify(text: str, state: dict | None = None) -> str:
    """Prefer the live cursor. Fall back to the spoken question only if it is empty."""
    if state:
        routed = _route(state)
        if routed:
            return routed
    if "आपका नाम" in text or "पूरा नाम" in text:
        return "intro_name"
    if "शादी हो गई" in text or "होने वाली" in text:
        return "intro_married"
    if "कितने बच्चे" in text or "उन सभी बच्चों" in text:
        return "intro_children"
    if "कभी गर्भवती" in text:
        return "ever_pregnant"
    if "किस बच्चे" in text:
        return "which_child"
    if "आसान सवाल" in text or "शुरू करें" in text:
        return "intro_bridge"
    if "इसके अलावा" in text or "कुछ और बताना" in text:
        return "close_anything_else"
    for qid, question in QUESTIONS:
        needle = question[:28]
        if len(needle) >= 12 and needle in text:
            return qid
    return ""


def _fallback(state: dict) -> tuple[str, str]:
    if expect_yes_no(state):
        return ("quick", "clear")
    if state.get("pending_followup_idx") == -1:
        return ("text", _RETRACT)
    if isinstance(state.get("pending_followup_idx"), int):
        return ("text", "पता नहीं।")
    return ("text", "नहीं।")


class Persona:
    def __init__(
        self,
        pid: str,
        name: str,
        lines: dict[str, tuple[str, str]],
        *,
        flags: tuple[int, int],
        close: str,
        check,
    ):
        self.pid = pid
        self.name = name
        self.lines = lines
        self.flags = flags
        self.close = close
        self.check = check
        self.counts: dict[str, int] = {}

    def answer(self, text: str, state: dict) -> tuple[str, str] | None:
        if str(state.get("phase") or "") != "asking":
            return None
        key = _classify(text, state) or "unknown"
        self.counts[key] = self.counts.get(key, 0) + 1
        if self.counts[key] > 4:
            raise RuntimeError(f"stuck on {key}: {text[:180]}")
        picked = self.lines.get(key)
        if picked is None:
            mode, payload = _fallback(state)
        elif callable(picked):
            mode, payload = picked(state)
        elif picked[0] == "steps":
            replies = picked[1]
            mode, payload = replies[min(self.counts[key] - 1, len(replies) - 1)]
        else:
            mode, payload = picked
        if mode == "quick" and not expect_yes_no(state):
            return ("text", "हाँ।" if payload == "concern" else "नहीं।")
        return (mode, payload)


def _line(text: str) -> tuple[str, str]:
    return ("text", text)


def _steps(*replies: tuple[str, str]) -> tuple[str, tuple[tuple[str, str], ...]]:
    return ("steps", replies)


def _profile(name: str, children: str, extra: dict[str, tuple[str, str]]) -> dict[str, tuple[str, str]]:
    lines = {
        "intro_name": _line(f"मेरा नाम {name} है।"),
        "intro_married": _line(_MARRIED),
        "intro_children": _line(children),
        "intro_bridge": _line(_BRIDGE),
    }
    lines.update(extra)
    return lines


def _notes(entry: dict) -> str:
    bits = [str(entry.get("notes") or "")]
    followups = entry.get("followups") or {}
    if isinstance(followups, dict):
        for item in followups.values():
            if isinstance(item, dict):
                bits.append(str(item.get("note") or ""))
    return " ".join(bit for bit in bits if bit)


def _has(text: str, *needles: str) -> bool:
    return any(needle in text for needle in needles)


def _answer(call: dict, qid: str) -> dict:
    answers = (call.get("cursor_state") or {}).get("answers") or {}
    entry = answers.get(qid) or {}
    return entry if isinstance(entry, dict) else {}


def _check_p1(call: dict, problems: list[str]) -> None:
    nicu = _answer(call, "birth_nicu")
    bleed = _answer(call, "body_bleeding_bruise")
    joint = _answer(call, "body_joint_swelling")
    if nicu.get("state") != "concern":
        problems.append(f"birth_nicu={nicu.get('state')}")
    if bleed.get("state") != "concern":
        problems.append(f"body_bleeding_bruise={bleed.get('state')}")
    if joint.get("state") != "concern":
        problems.append(f"body_joint_swelling={joint.get('state')}")
    nicu_text = _notes(nicu)
    if not _has(nicu_text, "तकलीफ", "साँस", "सांस"):
        problems.append(f"nicu reason missing: {nicu_text}")
    if not _has(nicu_text, "4", "चार"):
        problems.append(f"nicu days missing: {nicu_text}")


def _check_p2(call: dict, problems: list[str]) -> None:
    answers = (call.get("cursor_state") or {}).get("answers") or {}
    concerns = [
        qid for qid, entry in answers.items()
        if isinstance(entry, dict) and entry.get("state") == "concern"
    ]
    if concerns:
        problems.append("unexpected concerns: " + ", ".join(concerns))


def _check_p3(call: dict, problems: list[str]) -> None:
    for qid in ("birth_mode_detail", "birth_term", "birth_nicu"):
        state = _answer(call, qid).get("state")
        if state != "clear":
            problems.append(f"{qid}={state}")
    cursor = call.get("cursor_state") or {}
    if cursor.get("active_qid") not in {"birth_jaundice", "birth_sudden", ""} and cursor.get("phase") != "closing":
        problems.append(f"emergency did not stop at {cursor.get('active_qid')}")
    doctor = str(call.get("summary_doctor") or "")
    mother = str(call.get("summary_mother") or "")
    if "Active emergency" not in doctor:
        problems.append("summary_doctor missing Active emergency")
    if "None documented" in doctor:
        problems.append("summary_doctor still says None documented")
    if "टीके के कार्ड" in mother:
        problems.append("summary_mother has routine vaccine filler")


def _middle_child(profile: dict) -> dict | None:
    children = list(profile.get("children") or [])
    if len(children) != 3:
        return None
    ordered = sorted(children, key=lambda child: float(child.get("age_years") or 0), reverse=True)
    return ordered[1]


def _check_p4(persona: Persona, call: dict, problems: list[str]) -> None:
    jaundice = _answer(call, "birth_jaundice")
    bcg = _answer(call, "birth_bcg")
    transfusion = _answer(call, "fam_transfusion")
    bp = _answer(call, "fam_bp_stroke")
    same = _answer(call, "fam_same_problem")
    if jaundice.get("state") != "concern":
        problems.append(f"birth_jaundice={jaundice.get('state')}")
    if bcg.get("state") != "clear":
        problems.append(f"birth_bcg={bcg.get('state')}")
    if persona.counts.get("birth_bcg:which_child", 0) != 1:
        problems.append(f"birth_bcg loop count={persona.counts.get('birth_bcg:which_child', 0)}")
    if transfusion.get("state") != "concern":
        problems.append(f"fam_transfusion={transfusion.get('state')}")
    if bp.get("state") != "concern":
        problems.append(f"fam_bp_stroke={bp.get('state')}")
    if same.get("state") != "clear":
        problems.append(f"fam_same_problem={same.get('state')}")
    profile = (call.get("cursor_state") or {}).get("profile") or {}
    middle = _middle_child(profile)
    if middle is None:
        problems.append(f"children={profile.get('children')}")
    elif middle.get("id") not in (jaundice.get("child_ids") or []):
        problems.append(f"jaundice child {jaundice.get('child_ids')} middle {middle}")
    if not _has(_notes(jaundice), "3", "तीन"):
        problems.append(f"jaundice days missing: {_notes(jaundice)}")
    if "मामा" not in _notes(transfusion):
        problems.append(f"fam_who missing: {_notes(transfusion)}")
    bp_text = _notes(bp)
    if "पिता" not in bp_text:
        problems.append(f"bp_who missing: {bp_text}")
    if not _has(bp_text, "42", "बयालीस"):
        problems.append(f"bp_age missing: {bp_text}")


def _flat(text: str) -> str:
    return " ".join(str(text or "").split())


def _pdf_fold(text: str) -> str:
    # pypdf drops the pre-base ि matra, so पति on the page extracts as पत.
    return _flat(text).replace("ि", "")


def _pdf_page(call: dict, index: int) -> str:
    call_id = int(call.get("id") or 0)
    path = Path(db.DATA_DIR) / "mother-pdfs" / f"call-{call_id}.pdf"
    if not path.is_file():
        return ""
    reader = PdfReader(io.BytesIO(path.read_bytes()))
    if index < 0 or index >= len(reader.pages):
        return ""
    return reader.pages[index].extract_text() or ""


def _child_specs():
    return [spec for spec in active_specs() if "no_children" in spec.skip_if]


def _check_p5(persona: Persona, call: dict, problems: list[str]) -> None:
    profile = (call.get("cursor_state") or {}).get("profile") or {}
    if profile.get("marital_status") != "about_to_marry":
        problems.append(f"marital_status={profile.get('marital_status')}")
    if profile.get("children"):
        problems.append(f"children={profile.get('children')}")
    asked = [
        spec.id
        for spec in _child_specs()
        if any(key == spec.id or key.startswith(spec.id + ":") for key in persona.counts)
    ]
    if asked:
        problems.append("child questions asked: " + ", ".join(asked))
    assistant = _flat(
        " ".join(
            str(turn.get("text") or "")
            for turn in (call.get("transcript") or [])
            if isinstance(turn, dict) and turn.get("role") == "assistant"
        )
    )
    spoken = [
        spec.id
        for spec in _child_specs()
        if spec.question_hi and _flat(spec.question_hi) in assistant
    ]
    if spoken:
        problems.append("spoken child questions: " + ", ".join(spoken))
    page2 = _pdf_fold(_pdf_page(call, 1))
    if _pdf_fold("होने वाले पति") not in page2:
        problems.append("pdf page 2 missing होने वाले पति")
    leaked = [
        spec.id
        for spec in _child_specs()
        if spec.question_hi and _pdf_fold(spec.question_hi) in page2
    ]
    if leaked:
        problems.append("pdf page 2 child questions: " + ", ".join(leaked))


def _check_p6(call: dict, problems: list[str]) -> None:
    close = _answer(call, "close_anything_else")
    if close.get("state") != "concern":
        problems.append(f"close_anything_else={close.get('state')}")
    mother = str(call.get("summary_mother") or "")
    doctor = str(call.get("summary_doctor") or "")
    flags = call.get("red_flags") or []
    flag_list = [str(item) for item in flags] if isinstance(flags, list) else [str(flags)]
    flag_text = " ".join(flag_list)
    page1 = _flat(_pdf_page(call, 0))
    for needle in ("नींद", "डर"):
        for label, text in (("summary_mother", mother), ("pdf page 1", page1)):
            if needle not in text:
                problems.append(f"{label} missing {needle}")
    for label, text in (("summary_doctor", doctor), ("red_flags", flag_text)):
        if "close_anything_else" not in text:
            problems.append(f"{label} missing close_anything_else")
        folded = text.casefold()
        if not any(word in folded for word in ("sleep", "night", "fear", "terror", "नींद", "डर")):
            problems.append(f"{label} missing sleep/fear detail")
    close_flags = [item for item in flag_list if "close_anything_else" in item]
    if len(close_flags) != 1:
        problems.append(f"duplicate or missing close_anything_else flags: {close_flags}")


def _p6_close(state: dict) -> tuple[str, str]:
    entry = (state.get("answers") or {}).get("close_anything_else") or {}
    notes = str(entry.get("notes") or "").strip() if isinstance(entry, dict) else ""
    if isinstance(entry, dict) and entry.get("state") == "concern" and notes:
        return ("quick", "clear")
    return ("quick", "concern")


def _personas() -> list[Persona]:
    p4 = Persona(
        "p4_multi_child_multi_family",
        "कविता",
        _profile(
            "कविता",
            "तीन बच्चे हैं। 11 साल की बड़ी बेटी, 6 साल का बीच वाला बेटा, 2 साल का छोटा बेटा।",
            {
                "birth_jaundice": _line("हाँ।"),
                "birth_jaundice:which_child": _line("बीच वाले बेटे को।"),
                "jaundice_days": _line("3 दिन।"),
                "birth_bcg": _line("हाँ।"),
                "birth_bcg:which_child": _line(_RETRACT),
                "fam_transfusion": _line("हाँ।"),
                "fam_who": _line("मेरे सगे मामा के बेटे को।"),
                "fam_bp_stroke": _line("हाँ।"),
                "bp_who": _line("मेरे पिताजी को, सगे रिश्तेदार।"),
                "bp_age": _line("42 साल की उम्र में।"),
            },
        ),
        flags=(2, 8),
        close="complete",
        check=_check_p2,
    )
    p4.check = lambda call, problems, persona=p4: _check_p4(persona, call, problems)
    p5 = Persona(
        "p5_unmarried_correction_and_clarify",
        "नीलिमा",
        _profile("नीलिमा", "नहीं, मेरी शादी अभी नहीं हुई है, होने वाली है।", {
            "preg_thalassemia": _steps(
                _line("थैलेसीमिया का क्या मतलब है?"),
                ("quick", "clear"),
            ),
        }),
        flags=(0, 0),
        close="complete",
        check=_check_p2,
    )
    p5.check = lambda call, problems, persona=p5: _check_p5(persona, call, problems)
    return [
        Persona(
            "p1_red_flags",
            "रेखा",
            _profile("रेखा", "एक बेटा है, 7 साल का।", {
                "birth_nicu": _line("हाँ।"),
                "nicu_reason": _line("साँस की तकलीफ।"),
                "nicu_days": _line("चार दिन।"),
                "body_bleeding_bruise": _line("हाँ, मामूली चोट पर खून देर तक बहता है।"),
                "body_joint_swelling": _line("हाँ, गिरने पर घुटने में सूजन आती है।"),
            }),
            flags=(2, 6),
            close="complete",
            check=_check_p1,
        ),
        Persona(
            "p2_no_red_flags",
            "सीमा",
            _profile("सीमा", "एक बेटी है, 4 साल की।", {}),
            flags=(0, 0),
            close="complete",
            check=_check_p2,
        ),
        Persona(
            "p3_emergency_midway",
            "आशा",
            _profile("आशा", "एक बेटा है, 2 साल का।", {
                "birth_jaundice": _line(
                    "बच्चा अभी साँस नहीं ले पा रहा है और होंठ नीले हो रहे हैं, यह इमरजेंसी है!"
                ),
            }),
            flags=(1, 4),
            close="emergency",
            check=_check_p3,
        ),
        p4,
        p5,
        Persona(
            "p6_closing_disclosure_and_late_addition",
            "मीरा",
            _profile("मीरा", "एक बेटी है, 5 साल की।", {
                "fam_cancer": _line("हाँ, मामा को कैंसर हुआ था।"),
                "fam_who": _line("सगे मामा को।"),
                "fam_heart_kidney": _steps(
                    _line("नहीं, दिल-गुर्दे की बीमारी किसी को नहीं है, बस चाची को भी कैंसर हुआ था।"),
                    ("quick", "clear"),
                ),
                "close_anything_else": _p6_close,
                "open_concern_detail": _line("रात में नींद में डर लगता है।"),
            }),
            flags=(1, 6),
            close="complete",
            check=_check_p6,
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


def _pdf_text(path: Path) -> tuple[int, str]:
    raw = path.read_bytes()
    reader = PdfReader(io.BytesIO(raw))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    return len(reader.pages), text


def _check(call_id: int, persona: Persona) -> dict:
    call = db.get_call(call_id) or {}
    problems: list[str] = []
    telemetry = call.get("telemetry") or {}
    if int(telemetry.get("tool_errors") or 0):
        problems.append(f"tool_errors={telemetry.get('tool_errors')}")
    if int(telemetry.get("watchdog_nudges") or 0):
        problems.append(f"watchdog_nudges={telemetry.get('watchdog_nudges')}")
    if int(telemetry.get("prompt_leaks") or 0):
        problems.append(f"prompt_leaks={telemetry.get('prompt_leaks')}")
    assistant = " ".join(
        str(turn.get("text") or "")
        for turn in (call.get("transcript") or [])
        if turn.get("role") == "assistant"
    )
    folded = assistant.lower()
    for leak in _LEAKS:
        if leak.lower() in folded:
            problems.append(f"spoken leak {leak}")
    if not str(call.get("summary_doctor") or "").strip():
        problems.append("empty summary_doctor")
    if not str(call.get("summary_mother") or "").strip():
        problems.append("empty summary_mother")
    flags = call.get("red_flags") or []
    count = len(flags) if isinstance(flags, list) else -1
    low, high = persona.flags
    if count < low or count > high:
        problems.append(f"red_flags={count} expected {low}-{high}")
    if not str(call.get("mother_name") or "").strip():
        problems.append("mother not named")
    elif persona.name not in str(call.get("mother_name")):
        problems.append(f"mother_name={call.get('mother_name')}")
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
        pages, text = _pdf_text(pdf_path)
        if persona.close == "emergency":
            if not bool(telemetry.get("closing_heard")):
                problems.append("closing audio was not heard")
            if "108" not in text and "अधूरी" not in text:
                problems.append("emergency pdf missing banner")
        elif pages < 2:
            problems.append(f"pdf pages {pages}")
    cursor = call.get("cursor_state") or {}
    if cursor.get("close_reason") != persona.close:
        problems.append(f"close_reason={cursor.get('close_reason')} expected {persona.close}")
    try:
        persona.check(call, problems)
    except Exception as exc:
        problems.append(f"check {type(exc).__name__}: {exc}")
    return {
        "persona": persona.pid,
        "call_id": call_id,
        "ok": not problems,
        "problems": problems,
        "close_reason": cursor.get("close_reason"),
        "phase": cursor.get("phase"),
        "flags": count,
        "mother_name": call.get("mother_name"),
        "child_age": call.get("child_age"),
        "share_token": call.get("share_token"),
        "duration_sec": call.get("duration_sec"),
        "avg_latency_ms": telemetry.get("avg_latency_ms"),
        "max_latency_ms": telemetry.get("max_latency_ms"),
        "asks": persona.counts,
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
            question: dict = {}
            heard_audio = False
            deadline = time.monotonic() + CALL_DEADLINE_SEC
            while time.monotonic() < deadline:
                raw = await asyncio.wait_for(ws.recv(), timeout=RECV_TIMEOUT_SEC)
                msg = json.loads(raw)
                kind = msg.get("type")
                if kind == "session_started":
                    call_id = int(msg["call_id"])
                    if not msg.get("share_token"):
                        return {"persona": persona.pid, "call_id": call_id, "ok": False, "problems": ["no share_token"]}
                    print(f"call {call_id} token {msg.get('share_token')}", flush=True)
                elif kind == "question_state":
                    question = msg
                    heard_audio = False
                elif kind == "transcript" and msg.get("role") == "assistant":
                    buf = (buf + " " + str(msg.get("text") or "")).strip()
                elif kind == "telemetry":
                    event = (msg.get("last_event") or {}).get("type")
                    if event == "bot_first_audio":
                        heard_audio = True
                    elif event == "turn_complete" and heard_audio and call_id:
                        heard_audio = False
                        state = (db.get_call(call_id) or {}).get("cursor_state") or {}
                        reply = persona.answer(buf, state)
                        heard = buf[:80].replace("\n", " ")
                        buf = ""
                        if reply is None:
                            print(f"  closing heard: {heard}", flush=True)
                            continue
                        mode, payload = reply
                        step = gemini_live._current_step(state)
                        print(f"  [{_route(state) or step}] {mode} {payload[:70]}", flush=True)
                        if mode == "quick":
                            tap_step = step or str(question.get("step") or "")
                            await ws.send(json.dumps({
                                "type": "quick_reply",
                                "outcome": payload,
                                "step": tap_step,
                            }, ensure_ascii=False))
                        else:
                            await ws.send(json.dumps({"type": "text", "text": payload}, ensure_ascii=False))
                elif kind == "call_completed":
                    if not msg.get("share_token"):
                        return {"persona": persona.pid, "call_id": call_id, "ok": False, "problems": ["completed without share_token"]}
                    result = _check(int(msg.get("call_id") or call_id), persona)
                    mark = "PASS " if result["ok"] else "FAIL "
                    print(
                        mark
                        + persona.pid
                        + f" call={result['call_id']} avg_ms={result.get('avg_latency_ms')} dur={result.get('duration_sec')}"
                        + " "
                        + ", ".join(result["problems"]),
                        flush=True,
                    )
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
        if not personas:
            print("no matching persona", flush=True)
            return 1
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
    return 0 if results and passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
