"""Two curated live calls that walk every V3 edge case across both packs. Run inside `tmux`.

Boots `intake.app:app` on port 8793 with `DATA_DIR=intake/storage/curated_2calls` (so it never shares a
database, an audio file, or a PDF with `scripts/run_personas.py`), logs in over `/api/login`, then drives
one scripted caller per call over `ws://127.0.0.1:8793/ws/live?pack=<pack_id>`. Everything underneath is
real: Gemini Live, the Decision API, SQLite, WeasyPrint, and the stereo WAV recorder.

Call 1 (`c1_shishu_full_edge_cases`, pack `shishu-hi`) in one call: a partial twin roster that must re-ask
the ages, a meta-question that must clarify and stay (never burn the answer), which_item attribution to the
son, a which_item retraction, UNKNOWN vs UNANSWERED, a retained clinical aside, a live quick-reply tap with
its epoch, a multi-fact compound concern, and the maternal disease-name redaction.

Call 2 (`c2_fever_giveup_nudge_emergency`, pack `fever-opd-hi`) in one call: the profile step give-up cap
(`dc52d4d`), the live Manual Nudge button re-cueing a question in a fresh epoch, and an active emergency
closing at 108 with an incomplete PDF.

Pass call ids as arguments to run a subset (`... c1_shishu_full_edge_cases`). Exits 0 only when every
selected call passes every checklist item. The full checklist lands in `<DATA_DIR>/results.json`.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import socket
import sys
import time
import urllib.request
import wave
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Set, never defaulted: an inherited DATA_DIR from a persona run would put both suites on one database.
os.environ["DATA_DIR"] = str(ROOT / "intake" / "storage" / "curated_2calls")
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from pypdf import PdfReader  # noqa: E402

from intake import db, reports  # noqa: E402
from intake.models import AnswerStatus, CallState, Phase, SubStep  # noqa: E402
from intake.pack import load_pack  # noqa: E402

HOST, PORT = "127.0.0.1", 8793
BASE, CALL_DEADLINE_SEC, RECV_TIMEOUT_SEC = f"http://{HOST}:{PORT}", 1800, 300
_LEAKS = ("next_question", "system cue", "ask only this", "active_qid", "last_answered_qid", "recorded_concerns",
          "duchenne", "hemophilia", "pompe", "gaucher", "scid", "wiskott", "ड्यूशेन", "हीमोफीलिया", "पॉम्पे",
          "गॉशर", "इम्यूनोडेफिशिएंसी", "म्यूकोपॉलीसेकेराइडोसिस")
_EMERGENCY = ("अभी मरीज़ को साँस लेने में बहुत ज़्यादा तकलीफ हो रही है, होंठ नीले पड़ रहे हैं और बेहोश सा हो रहा है, "
              "यह इमरजेंसी है!")
_PROFILE_QIDS = frozenset({"patient_name", "patient_age", "intro_name", "intro_children", "intro_married", "intro_bridge"})
# WeasyPrint + pypdf split Devanagari conjuncts, so the PDF never yields "हीमोफीलिया" as one token even when
# it is printed. A full-word PDF check would pass on a broken redaction; these fragments are what actually leaks.
_DISEASE = ("हीमोफीलिया", "हीमोफ", "मोफी", "फीलिया")
# The page-two glossary is in the PDF, but pypdf drops matras from the embedded font ("जवाब का मतलब" reads back
# as "जवाब का मलब"). One literal needle would fail on a perfect PDF, so each of these must survive *some* form.
_GLOSSARY_PDF = (("जवाब",), ("इनकर", "इनकार"), ("मलब", "मतलब"), ("जवब नहीं", "जवाब नहीं"))
C, CL, U, UA = AnswerStatus.CONCERN, AnswerStatus.CLEAR, AnswerStatus.UNKNOWN, AnswerStatus.UNANSWERED


def _answered(call: dict, qid: str) -> tuple[AnswerStatus | None, str]:
    """One answer's resolved status and its verbatim note, from the single stored `CallState`."""
    record = db.call_state_from_row(call).answers.get(qid)
    return (record.state, record.verbatim_text) if record is not None else (None, "")


def _has(text: str, *needles: str) -> bool:
    return any(needle in text for needle in needles)


def _pdf(call: dict) -> tuple[int, str]:
    """The Patient PDF's page count and flattened text; (0, "") when it is missing or not a PDF."""
    path = Path(db.DATA_DIR) / "pdfs" / f"call-{int(call.get('id') or 0)}.pdf"
    if not path.is_file() or not path.read_bytes().startswith(b"%PDF"):
        return 0, ""
    reader = PdfReader(io.BytesIO(path.read_bytes()))
    return len(reader.pages), " ".join(" ".join(p.extract_text() or "" for p in reader.pages).split())


class Checklist:
    """Named assertions in the order they ran, so the JSON report says which edge case passed or failed."""

    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        self.items.append({"check": name, "ok": bool(ok), "detail": "" if ok else detail})
        return bool(ok)

    @property
    def problems(self) -> list[str]:
        return [f"{item['check']}: {item['detail']}" if item["detail"] else item["check"] for item in self.items if not item["ok"]]


def _status(call: dict, cl: Checklist, *wanted: tuple[str, AnswerStatus]) -> None:
    """Assert every named answer resolved as expected, naming the ones that did not."""
    for qid, want in wanted:
        got = _answered(call, qid)[0]
        cl.check(f"{qid} == {want}", got is want, f"got {got}")


def _parity(call: dict, pack, cl: Checklist) -> reports.ProjectedReport:
    """One projection must feed the stored note, the patient summary, and the flags."""
    proj = reports.project_summary(pack, db.call_state_from_row(call), db.list_retained_observations(int(call.get("id") or 0)))
    diverged = [c for c, field in (("summary_doctor", "doctor_markdown"), ("summary_patient", "patient_summary_text"))
                if str(call.get(c) or "").strip() != str(getattr(proj, field) or "").strip()]
    if (call.get("red_flags") or []) != list(proj.red_flags):
        diverged.append("red_flags")
    cl.check("project_summary parity", not diverged, f"diverges: {diverged}")
    return proj


def _common(call: dict, cl: Checklist) -> tuple[int, str]:
    """The checks every live call owes: telemetry, spoken leaks, WAV, and the PDF that exists at all."""
    telemetry, events = call.get("telemetry") or {}, call.get("events") or []
    kinds = Counter(str(e.get("type") or "") for e in events)
    spoken = " ".join(str(t.get("text") or "") for t in call.get("transcript") or [] if t.get("role") == "assistant").lower()
    cl.check("zero tool_errors", not int(telemetry.get("tool_errors") or 0), f"tool_errors={telemetry.get('tool_errors')}")
    cl.check("zero error events", not kinds.get("error"), f"errors={[e.get('detail') for e in events if e.get('type') == 'error'][:5]}")
    # `auto_close` is only ever emitted by the watchdog (silence / time limit). Both calls end on their own
    # content, so any auto_close means the harness stalled and the grader would be reading a timeout.
    cl.check("closed on content, not the watchdog", not kinds.get("auto_close"),
             f"auto_close={[e.get('detail') for e in events if e.get('type') == 'auto_close']}")
    leaks = [leak for leak in _LEAKS if leak.lower() in spoken]
    cl.check("zero spoken leaks", not leaks, f"leaks={leaks}")
    cl.check("both summaries written", all(str(call.get(f) or "").strip() for f in ("summary_doctor", "summary_patient")), "")
    cl.check("red flags reported", bool(call.get("red_flags")), "no red flags projected")
    wav = Path(db.DATA_DIR) / "audio" / str(call.get("audio_path") or "")
    if not wav.is_file():
        cl.check("stereo 16 kHz WAV", False, "missing wav")
    else:
        with wave.open(str(wav), "rb") as handle:
            shape = (handle.getnchannels(), handle.getframerate())
        cl.check("stereo 16 kHz WAV", shape == (2, 16000), f"wav is {shape[0]}ch {shape[1]}Hz")
    pages, text = _pdf(call)
    cl.check("patient PDF exists", bool(pages), "missing pdf")
    return pages, text


def _line(text: str) -> tuple:
    return ("text", text)


def _steps(*replies: tuple) -> tuple:
    return ("steps", replies)


class Persona:
    """A scripted caller: one reply per typed cursor step, with a stuck-step guard and full send log."""

    def __init__(self, pid: str, pack_id: str, lines: dict[str, tuple], *, close: str, check,
                 default_reply=None, name_in_call: str = "") -> None:
        self.pid, self.pack_id, self.lines = pid, pack_id, lines
        self.close, self.check, self.default_reply, self.name_in_call = close, check, default_reply, name_in_call
        self.counts: dict[str, int] = {}
        self.sent: list[dict[str, Any]] = []
        self.defaults: list[str] = []
        self.frames: dict[str, int] = {}
        self.epoch = 0
        self.nudge_at: dict[str, Any] | None = None
        self.manual_nudges: list[str] = []

    def name_ok(self, call: dict) -> bool:
        return not self.name_in_call or self.name_in_call in str(call.get("patient_name") or "")

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
            if self.default_reply is not None:
                self.defaults.append(key)
                return self.default_reply(state)
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

    def observe(self, kind: str, msg: dict) -> None:
        """Track the server's turn epoch and whether a nudge re-cued the step it was pressed on."""
        self.frames[kind] = self.frames.get(kind, 0) + 1
        epoch = msg.get("epoch")
        if not isinstance(epoch, int):
            return
        self.epoch = epoch
        if (kind == "question_state" and self.nudge_at is not None and msg.get("qid") == self.nudge_at.get("qid")
                and epoch > int(self.nudge_at.get("epoch") or 0)):
            self.nudge_at.setdefault("recued_epoch", epoch)

    def sent_on(self, key: str) -> list[dict[str, Any]]:
        return [item for item in self.sent if item["key"] == key]


def _c2_default(state: CallState) -> tuple[str, str]:
    """Call 2's safety valve: a step the script did not name still gets a real fever answer, and the only
    honest one left is the emergency. Profile steps keep the ordinary fallback, so a give-up re-ask is
    never answered with an emergency and the give-up check keeps measuring what it claims to."""
    return Persona.fallback(state) if state.cursor.qid in _PROFILE_QIDS else ("text", _EMERGENCY)


def _personas() -> list[Persona]:
    """The two curated calls. Pass ids on the command line to run a subset."""
    return [
        Persona(
            "c1_shishu_full_edge_cases", "shishu-hi", {
                "intro_name": _line("मेरा नाम सुमन है।"),
                "intro_married": _line("हाँ, शादी हो गई है।"),
                # Step 1 states the twins and no ages: the roster stays incomplete and the ages must be re-asked.
                "intro_children": _steps(_line("मेरे दो जुड़वाँ बच्चे हैं।"),
                                         _line("दोनों 6 साल के हैं, एक बेटा और एक बेटी।")),
                "intro_bridge": _line("जी, पूछिए।"),
                # A meta-question about the notes must be answered while this step stays exactly where it is.
                "birth_nicu": _steps(_line("क्या आपने मेरे दोनों बच्चों की उम्र नोट कर ली?"),
                                     _line("हाँ, जन्म के बाद 4 दिन काँच के बॉक्स में रखा था, साँस की तकलीफ थी।")),
                "birth_nicu:which_item": _line("बेटे को।"),
                "birth_nicu:fu0": _line("साँस की तकलीफ थी।"),
                "birth_nicu:fu1": _line("चार दिन।"),
                # A yes she then takes back on the which_item step: the answer must resolve CLEAR, not CONCERN.
                "birth_cord_3weeks": _line("हाँ।"),
                "birth_cord_3weeks:which_item": _line("नहीं, गलती से हाँ बोला, कुछ नहीं हुआ था।"),
                "birth_bcg": _line("पता नहीं, मुझे याद नहीं।"),
                # A clinical aside is kept, not counted as an answer; the second off-topic turn gives up.
                "birth_jaundice": _steps(_line("वैसे मेरे बेटे को आजकल रात में हल्की खाँसी रहती है।"),
                                         _line("अभी बारिश बहुत हो रही है यहाँ।")),
                "birth_sudden": ("quick", "clear"),
                # One multi-fact turn: the model must auto-fill both bleeding concerns together.
                "body_bleeding_bruise": _line("हाँ, चोट लगने पर खून देर तक बहता है और घुटने में भी सूजन आ जाती है।"),
                "body_bleeding_bruise:which_item": _line("बेटे को।"),
                "body_joint_swelling": _line("हाँ, गिरने पर घुटने में सूजन आती है।"),
                "body_joint_swelling:which_item": _line("बेटे को।"),
                "fam_maternal_bleeding": _line("हाँ, मेरे सगे मामा के बेटे को डॉक्टर ने हीमोफीलिया बताया था।"),
                "fam_maternal_bleeding:fu0": _line("सगे मामा के बेटे को।"),
                "close_anything_else": _steps(_line("हाँ, रात में नींद में डर लगता है।"), ("quick", "clear")),
            },
            close="complete", check=_check_c1, name_in_call="सुमन"),
        Persona(
            "c2_fever_giveup_nudge_emergency", "fever-opd-hi", {
                # Two refusals spend the profile step's whole retry budget: the call must move on, not loop.
                "patient_name": _steps(_line("मैं अपना नाम नहीं बताना चाहती।"),
                                       _line("नाम रहने दीजिए, आगे का सवाल पूछिए।")),
                "patient_age": _line("32 साल।"),
                "intro_bridge": _line("जी, पूछिए।"),
                # Tap Nudge on the first sight of this step, then answer it once the re-cue lands.
                "fever_duration_bucket": _steps(("nudge", ""), _line("तीन दिन से तेज़ बुखार है।")),
                "fever_nature": _line("बुखार रुक-रुक कर आता है, दिन में कई बार चढ़ता-उतरता है।"),
                "fever_frequency": _line(_EMERGENCY),
            },
            close="emergency", check=_check_c2, default_reply=_c2_default),
    ]


def _check_c1(pack, call: dict, persona: Persona, cl: Checklist) -> dict[str, Any]:
    """Call 1: the twin roster, the meta-question, the retraction, and the redaction, all on one call."""
    state = db.call_state_from_row(call)
    cl.check("two children in the roster", len(state.roster) == 2, f"roster={[(c.id, c.age_years, c.tag) for c in state.roster]}")
    cl.check("both twins are 6", sorted(c.age_years for c in state.roster) == [6.0, 6.0],
             f"ages={[c.age_years for c in state.roster]}")
    cl.check("roster resolved after the ages re-ask", state.roster_incomplete is False, f"roster_incomplete={state.roster_incomplete}")
    cl.check("ages re-ask happened", persona.counts.get("intro_children", 0) == 2, f"landings={persona.counts.get('intro_children')}")

    _status(call, cl, ("birth_nicu", C), ("birth_cord_3weeks", CL), ("birth_bcg", U), ("birth_jaundice", UA),
            ("birth_sudden", CL), ("body_bleeding_bruise", C), ("body_joint_swelling", C),
            ("fam_maternal_bleeding", C), ("close_anything_else", C))

    sends = persona.sent_on("birth_nicu")
    cl.check("meta-question stayed on birth_nicu", len(sends) >= 2, f"steps={[s['step'] for s in sends]}")
    cl.check("meta-question burned no attempt",
             len(sends) >= 2 and "try:" not in str(sends[1]["step"]),
             f"answer step={sends[1]['step'] if len(sends) > 1 else ''}")

    son = next((c for c in state.roster if c.tag == "male"), None)
    ids = state.answers["birth_nicu"].item_ids if "birth_nicu" in state.answers else []
    cl.check("NICU concern attributed to the son", son is not None and son.id in ids,
             f"item_ids={ids} son={son.id if son else None}")
    nicu = _answered(call, "birth_nicu")[1]
    cl.check("NICU reason and days kept", _has(nicu, "तकलीफ", "साँस", "सांस") and _has(nicu, "4", "चार"), f"note={nicu}")

    cord = _answered(call, "birth_cord_3weeks")[1]
    cl.check("retraction was spoken on which_item", bool(persona.sent_on("birth_cord_3weeks:which_item")) and not cord,
             f"sent={len(persona.sent_on('birth_cord_3weeks:which_item'))} note={cord}")

    statuses = {str(record.state) for record in state.answers.values()}
    cl.check("UNKNOWN and UNANSWERED both recorded", {"unknown", "unanswered"} <= statuses, f"statuses={sorted(statuses)}")
    jaundice = persona.sent_on("birth_jaundice")
    cl.check("clinical aside spent one attempt before giving up",
             len(jaundice) >= 2 and "try:" in str(jaundice[1]["step"]),
             f"steps={[s['step'] for s in jaundice]}")

    taps = persona.sent_on("birth_sudden")
    cl.check("quick_reply tap carried step and epoch",
             bool(taps) and taps[0]["mode"] == "quick" and bool(taps[0]["step"]) and isinstance(taps[0]["epoch"], int),
             f"tap={taps[0] if taps else None}")

    proj = _parity(call, pack, cl)
    pages, text = _common(call, cl)
    doctor, patient = str(call.get("summary_doctor") or ""), str(call.get("summary_patient") or "")
    cl.check("patient_name on the call is सुमन", persona.name_ok(call), f"patient_name={call.get('patient_name')!r}")
    cl.check("doctor summary keeps हीमोफीलिया", "हीमोफीलिया" in doctor, "doctor summary lost the reported diagnosis")
    cl.check("doctor summary keeps the maternal note", "मामा" in doctor, "doctor summary lost the maternal relative")
    cl.check("patient summary redacts हीमोफीलिया", bool(patient.strip()) and "हीमोफीलिया" not in patient,
             "patient summary is empty or leaked हीमोफीलिया")
    leaked = [frag for frag in _DISEASE if frag in text]
    cl.check("PDF redacts हीमोफीलिया", bool(text.strip()) and not leaked, f"PDF leaked fragments {leaked}")
    cl.check("glossary in the clinician note", "### Response State Glossary" in proj.doctor_markdown, "no glossary")
    rows = [f"{'/'.join(row)}" for row in _GLOSSARY_PDF if not any(frag in text for frag in row)]
    cl.check("glossary in the PDF", bool(text.strip()) and not rows, f"missing glossary rows {rows}")
    cl.check("PDF is at least two pages", pages >= 2, f"pages={pages}")
    retained = db.list_retained_observations(int(call.get("id") or 0))
    cl.check("clinical aside retained", bool(retained) and any("खाँसी" in str(o.transcript) for o in retained),
             f"retained={[o.transcript[:60] for o in retained]}")
    cl.check("retained aside lands in the clinician note", "खाँसी" in doctor, "aside missing from the doctor note")
    return {"emergency_step": "", "roster": [{"id": c.id, "age": c.age_years, "tag": c.tag} for c in state.roster],
            "retained_observations": [str(o.transcript) for o in retained]}


def _check_c2(pack, call: dict, persona: Persona, cl: Checklist) -> dict[str, Any]:
    """Call 2: the give-up cap, the live nudge, and an emergency close on the second pack."""
    state = db.call_state_from_row(call)
    _status(call, cl, ("patient_name", UA))
    cl.check("patient_name asked exactly twice", persona.counts.get("patient_name", 0) == 2,
             f"landings={persona.counts.get('patient_name')}")
    cl.check("call advanced past the give-up", bool(persona.sent_on("intro_bridge")),
             f"never left the profile steps: asks={persona.counts}")
    cl.check("patient_name never re-asked after the give-up",
             all(s["key"] != "patient_name" for s in persona.sent[persona.sent.index(persona.sent_on("intro_bridge")[0]):])
             if persona.sent_on("intro_bridge") else False, f"asks={persona.counts}")

    events = call.get("events") or []
    nudges = [e for e in events if e.get("type") == "manual_nudge"]
    cl.check("manual_nudge event recorded", bool(nudges), f"event types={sorted({str(e.get('type')) for e in events})}")
    nudge = persona.nudge_at or {}
    cl.check("nudge was pressed on fever_duration_bucket", nudge.get("qid") == "fever_duration_bucket", f"nudge={nudge}")
    cl.check("nudge re-cued the question in a fresh epoch",
             isinstance(nudge.get("recued_epoch"), int) and int(nudge["recued_epoch"]) > int(nudge.get("epoch") or 0),
             f"epoch {nudge.get('epoch')} -> {nudge.get('recued_epoch')}")
    cl.check("caller answered after the nudge",
             any(s["key"] == "fever_duration_bucket" and s["mode"] == "text" for s in persona.sent),
             f"sent={[s['key'] + ':' + s['mode'] for s in persona.sent]}")

    cl.check("phase CLOSING", state.phase is Phase.CLOSING, f"phase={state.phase}")
    proj = _parity(call, pack, cl)
    cl.check("projection is an emergency note", proj.emergency is True and "Active emergency" in proj.doctor_markdown,
             f"emergency={proj.emergency}")
    pages, text = _common(call, cl)
    # The `patient_name` column is "first filled profile slot", so for this pack it holds the AGE ("32") once the
    # name is given up on. The give-up evidence is the profile itself: no name was ever written.
    cl.check("no name was ever recorded (the give-up)", not str(state.profile.get("patient_name") or "").strip(),
             f"profile={dict(state.profile)}")
    cl.check("emergency PDF carries the 108 banner", bool(text.strip()) and "108" in text, "PDF has no 108 banner")
    cl.check("emergency PDF is marked incomplete", bool(text.strip()) and "अधूरी" in text, "PDF is not marked अधूरी")
    cl.check("emergency PDF exists", bool(pages), "missing pdf")
    duration = _answered(call, "fever_duration_bucket")
    emergency_step = next((s["key"] for s in persona.sent if s["mode"] == "text" and s["text"] == _EMERGENCY), "")
    cl.check("emergency was spoken to a live question", bool(emergency_step) and emergency_step not in _PROFILE_QIDS,
             f"emergency_step={emergency_step!r}")
    # Every step the script names was walked as written: a default reply means Gemini sent the call somewhere the
    # curator did not script (an un-flagged emergency, a surprise extra step), which a reader must see, not miss.
    cl.check("scripted path walked exactly", not persona.defaults, f"default replies at {persona.defaults}")
    return {"emergency_step": emergency_step, "nudge": persona.nudge_at, "manual_nudge_events": nudges,
            "nudge_details": persona.manual_nudges, "duration_answer": str(duration[0]), "profile": dict(state.profile)}


def _answers_summary(call: dict) -> dict[str, Any]:
    state = db.call_state_from_row(call)
    return {qid: {"state": str(record.state), "value": record.value, "items": list(record.item_ids),
                  "note": (record.verbatim_text or "")[:240],
                  "followups": {fid: {"state": str(slot.state), "note": slot.note.verbatim[:160]}
                                for fid, slot in record.followups.items()}}
            for qid, record in sorted(state.answers.items())}


def _events_summary(call: dict) -> dict[str, Any]:
    events = call.get("events") or []
    counts = Counter(str(e.get("type") or "") for e in events)
    return {"total": len(events), "by_type": dict(sorted(counts.items())),
            "manual_nudge": [{"detail": e.get("detail"), "meta": e.get("meta")} for e in events if e.get("type") == "manual_nudge"],
            "errors": [e.get("detail") for e in events if e.get("type") == "error"][:10],
            "barge_ins": counts.get("barge_in", 0), "prompt_recues": counts.get("prompt_recue", 0)}


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


def _check(pack, call_id: int, persona: Persona, elapsed: float) -> dict[str, Any]:
    """Every assertion one finished call must satisfy, reported as a checklist a human can act on."""
    call, cl = db.get_call(call_id) or {}, Checklist()
    detail: dict[str, Any] = {"emergency_step": "", "roster": [], "retained_observations": [],
                              "nudge": {}, "manual_nudge_events": []}
    state = db.call_state_from_row(call)
    cl.check(f"close_reason {persona.close}", str(state.close_reason or "") == persona.close,
             f"close_reason={state.close_reason}")
    try:
        detail |= persona.check(pack, call, persona, cl) or {}
    except Exception as exc:  # noqa: BLE001 - a broken check reports, it never aborts the suite
        cl.check("checklist ran to the end", False, f"{type(exc).__name__}: {exc}")
    result = {"id": persona.pid, "pack": pack.id, "call_id": call_id, "ok": not cl.problems,
              "problems": cl.problems, "checklist": cl.items,
              "close_reason": str(state.close_reason or ""), "phase": str(state.phase),
              "duration_sec": call.get("duration_sec"), "wall_sec": round(elapsed, 1),
              "patient_name": call.get("patient_name"), "share_token": call.get("share_token"),
              "red_flags": list(call.get("red_flags") or []), "telemetry": call.get("telemetry") or {},
              "asks": persona.counts, "sends": persona.sent, "default_replies": persona.defaults,
              "frames": persona.frames, "answers": _answers_summary(call), "events": _events_summary(call)}
    result.update(detail)
    return result


async def _one(pack, persona: Persona, cookie: str) -> dict[str, Any]:
    """Drive one persona through one live call, from `session_started` to `call_completed`."""
    import websockets

    print(f"\n=== {persona.pid} ({pack.id}) ===", flush=True)
    started = time.monotonic()
    try:
        async with websockets.connect(f"ws://{HOST}:{PORT}/ws/live?pack={pack.id}",
                                      additional_headers={"Cookie": cookie},
                                      open_timeout=30, max_size=8_000_000) as ws:
            call_id, heard, deadline = 0, False, started + CALL_DEADLINE_SEC
            while time.monotonic() < deadline:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=RECV_TIMEOUT_SEC))
                kind, event = msg.get("type"), (msg.get("last_event") or {}).get("type")
                if isinstance(kind, str):
                    persona.observe(kind, msg)
                if kind == "session_started":
                    call_id = int(msg.get("call_id") or 0)
                    if not msg.get("share_token"):
                        return {"id": persona.pid, "pack": pack.id, "call_id": call_id, "ok": False, "problems": ["no share_token"]}
                    print(f"call {call_id}", flush=True)
                elif kind == "telemetry" and event == "manual_nudge":
                    persona.manual_nudges.append(str(msg.get("last_event", {}).get("detail") or ""))
                    print("  served manual_nudge", flush=True)
                elif kind == "bot_turn_complete":
                    # The one signal a pure-silence nudge still gives: `bot_first_audio` is only emitted for a
                    # turn a caller's speech opened (`speech_ended_ts` is 0 here), so hearing only that event
                    # left the harness deaf to the re-cued prompt and the call would idle into silence-close.
                    heard = True
                elif kind == "telemetry" and event == "bot_first_audio":
                    heard = True
                elif kind == "telemetry" and event == "turn_complete" and heard and call_id:
                    heard = False
                    state = db.call_state_from_row(db.get_call(call_id) or {})
                    key = Persona.route(state)
                    if persona.sent and persona.sent[-1]["mode"] in ("text", "quick") \
                            and (persona.sent[-1]["step"], persona.sent[-1]["epoch"]) == (state.step_token, persona.epoch):
                        # A repeated `turn_complete` on the exact step and epoch we just answered owes nothing: a
                        # second text turn on it would burn a real attempt. A clarify keeps the step token and
                        # moves the epoch, so its scripted second line still goes out.
                        print(f"  [{key}] duplicate turn_complete on {state.step_token}@{persona.epoch}, ignored", flush=True)
                        continue
                    reply = persona.answer(state)
                    if reply is None:
                        continue
                    mode, payload = reply
                    entry = {"key": key, "mode": mode, "step": state.step_token, "epoch": persona.epoch,
                             "text": payload if mode == "text" else ""}
                    persona.sent.append(entry)
                    if mode == "nudge":
                        persona.nudge_at = {"qid": state.cursor.qid, "step": state.step_token, "epoch": persona.epoch}
                        print(f"  [{key}] nudge -> {{\"type\": \"nudge\"}} (epoch {persona.epoch})", flush=True)
                        await ws.send(json.dumps({"type": "nudge"}))
                        continue
                    print(f"  [{key}] {mode} epoch={persona.epoch} {payload[:70]}", flush=True)
                    await ws.send(json.dumps(
                        {"type": "quick_reply", "value": payload, "step": state.step_token, "epoch": persona.epoch}
                        if mode == "quick" else {"type": "text", "text": payload}, ensure_ascii=False))
                elif kind == "call_completed":
                    result = _check(pack, int(msg.get("call_id") or call_id), persona, time.monotonic() - started)
                    print(("PASS " if result["ok"] else "FAIL ") + persona.pid
                          + f" call={result['call_id']} dur={result.get('duration_sec')} "
                          + ", ".join(result["problems"]), flush=True)
                    return result
                elif kind in {"report_failed", "error"}:
                    return {"id": persona.pid, "pack": pack.id, "call_id": call_id, "ok": False,
                            "problems": [f"{kind}: {msg.get('message') or ''}"]}
            return {"id": persona.pid, "pack": pack.id, "call_id": call_id, "ok": False, "problems": ["deadline"]}
    except Exception as exc:  # noqa: BLE001 - a dropped socket is a failed call, not a crash
        print(f"FAIL {persona.pid} {type(exc).__name__}: {exc}", flush=True)
        return {"id": persona.pid, "pack": pack.id, "call_id": 0, "ok": False,
                "problems": [f"{type(exc).__name__}: {exc}"]}


def _port_free() -> bool:
    """A suite that silently shares its port with a persona run grades the wrong socket."""
    with socket.socket() as probe:
        try:
            probe.bind((HOST, PORT))
            return True
        except OSError:
            return False


async def _main() -> int:
    """Boot the app in-process, run the selected calls over live sockets, and grade the run."""
    import uvicorn

    from intake.app import app

    if not _port_free():
        raise RuntimeError(f"{HOST}:{PORT} is already in use; stop the other live run before starting this one")
    print(f"data_dir={db.DATA_DIR}\nport={PORT}", flush=True)
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
    wanted = {a for a in sys.argv[1:] if not a.startswith("-")}
    personas = [p for p in _personas() if not wanted or p.pid in wanted]
    results: list[dict[str, Any]] = []
    try:
        if not personas:
            print(f"no matching call in {sorted(wanted)}", flush=True)
            return 1
        cookie = await asyncio.to_thread(_login)
        for persona in personas:
            results.append(await _one(load_pack(persona.pack_id), persona, cookie))
    finally:
        server.should_exit = True
        await task
    passed = sum(1 for item in results if item.get("ok"))
    out = Path(db.DATA_DIR) / "results.json"
    out.write_text(json.dumps({"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "port": PORT,
                               "data_dir": str(db.DATA_DIR), "passed": passed, "total": len(results),
                               "calls": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{passed}/{len(results)} PASS\n{out}", flush=True)
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
