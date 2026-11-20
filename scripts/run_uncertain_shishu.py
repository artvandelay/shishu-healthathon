"""One live exploratory call on the rare-disease pack (`shishu-hi`), aimed at four genuinely UNCERTAIN
state-machine interactions rather than at proving known behaviour.

Boots `intake.app:app` on port 8795 with `DATA_DIR=intake/storage/uncertain_shishu` (its own database, audio
directory and PDF folder), logs in over `/api/login`, then drives one scripted caller over
`ws://127.0.0.1:8795/ws/live?pack=shishu-hi`. Everything underneath is real: Gemini Live, the Decision API,
the profile extractor, SQLite, WeasyPrint, and the stereo WAV recorder.

Persona `u1_shishu_morphing_family` walks the whole form once, but its four load-bearing moments are questions
we are NOT sure the code answers cleanly:

  U1 - `intro_married` says unmarried (so `intro_children`, all child gates, the pregnancy gates, `fam_related`
       and `fam_miscarriage` are recorded NOT_APPLICABLE and the walk jumps to `preg_thalassemia`), then on
       `fam_infant_death` she reverses herself mid-call: married, two children (रिया 12, आरव 6). The roster and
       the marital status must both land, and the cursor must go BACK to the birth block (`birth_mode_detail`),
       not forward from index 48.

  U2 - a NICU concern is recorded against आरव (`item_2`) with its `which_item` and both follow-ups, and then,
       on `body_fits`, she remembers a third child - कबीर, आरव's twin. `_apply_profile_data` /
       `_pick_roster_id` (`engine.py:211-224`) rebuild the whole roster from the extractor's list, so the risk is
       that `birth_nicu.item_ids` gets renumbered onto कबीर, that the roster shrinks, or that `which_item_prompt`
       calls one 6-year-old twin "मँझला" (`engine.py:64-98`).

  U3 - on `body_transfusion` she names thalassemia major in the child. That single sentence has to (a) raise
       `confirmed_hemoglobinopathy`, which prunes `body_blood_count` to NOT_APPLICABLE via `known_hemoglobinopathy`
       (`_resync`, `engine.py:182-190`), and (b) be redacted from her copy and the PDF while staying verbatim for
       the doctor - the note is only stored when `_note_concern` accepts the utterance, which for a gate WITH
       follow-ups depends on `followup_answered`.

  U4 - every `fam_*` gate is denied, so `fam_same_problem` is NOT_APPLICABLE (`no_family_concern`) and the call
       reaches `close_anything_else` without asking it. The closing remark names the uncle's early heart attack,
       which `_apply_stated` (`engine.py:324-326`) may revive `fam_same_problem` from NOT_APPLICABLE - while
       `close_anything_else` itself must not swallow the sentence as a concern of its own.

`--selftest` proves every assertion is non-vacuous offline: it replays this same schedule through the real
`policy` + `engine` reducers, then feeds synthetic broken states into the same check function and requires each
invariant to fail when broken and pass when whole.

Exit 0 only when the live call passes every check.
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Set, never defaulted - and set BEFORE `intake.db` is imported, because `db.DATA_DIR` is read at import time.
os.environ["DATA_DIR"] = str(ROOT / "intake" / "storage" / "uncertain_shishu")

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from pypdf import PdfReader  # noqa: E402

from intake import db, engine, reports  # noqa: E402
from intake.models import (AnswerRecord, AnswerStatus, CallState, CloseReason, FollowupSlot, NoteText,  # noqa: E402
                           Phase, ProfilePatch, RosterInput, SubStep, TurnCommand, TurnOutcome)
from intake.pack import load_pack  # noqa: E402
from intake.turns import EvidenceSource, StoredTurnEvidence  # noqa: E402

HOST, PORT = "127.0.0.1", 8795
BASE = f"http://{HOST}:{PORT}"
PID, PACK_ID = "u1_shishu_morphing_family", "shishu-hi"
# This call walks the whole 60-question form twice through the birth block, so it is far longer than the two
# adversarial calls. 45 minutes of wall clock is the harness deadline, not the pack's own 19-minute limit.
CALL_DEADLINE_SEC, RECV_TIMEOUT_SEC = 2700.0, 300.0

_LEAKS = ("next_question", "system cue", "ask only this", "active_qid", "last_answered_qid", "recorded_concerns",
          "duchenne", "hemophilia", "pompe", "gaucher", "scid", "wiskott", "ड्यूशेन", "हीमोफीलिया", "पॉम्पे",
          "गॉशर", "इम्यूनोडेफिशिएंसी", "म्यूकोपॉलीसेकेराइडोसिस")
# WeasyPrint + pypdf split Devanagari matras from their consonants, so a full-word PDF check can pass on a broken
# redaction. The redaction target is the SEVERITY word, never the bare disease name: `थैलेसीमिया` alone is the
# pack's own carrier-screening wording (the `preg_thalassemia` question and `universal_advice_line_hi` both print
# it on every report, for every mother), so only `मेजर` proves the confirmed diagnosis reached her copy.
_DISEASE = ("मेजर",)
_GLOSSARY_PDF = (("जवाब", "जविाब"), ("इनकर", "इनकार"), ("मलब", "मतलब"), ("जवब नहीं", "जवाब नहीं"))
# The only two gates this pack may legitimately close for this mother: `ever_pregnant` (she has children) and
# `body_blood_count` (pruned by the hemoglobinopathy flag). Every other NOT_APPLICABLE would be a lost question.
_ALLOWED_NA = frozenset({"ever_pregnant", "body_blood_count"})
_U1_KIDS = {"रिया": (12.0, "female"), "आरव": (6.0, "male")}
C, CL, U, UA, ANS = (AnswerStatus.CONCERN, AnswerStatus.CLEAR, AnswerStatus.UNKNOWN, AnswerStatus.UNANSWERED,
                     AnswerStatus.ANSWERED)


# ------------------------------------------------------------------------------------------------------------
# Scripting primitives
# ------------------------------------------------------------------------------------------------------------
def _line(text: str) -> tuple:
    return ("text", text)


def _quick(value: str) -> tuple:
    return ("quick", value)


def _steps(*replies: tuple) -> tuple:
    return ("steps", replies)


def _has(text: str, *needles: str) -> bool:
    return any(needle in text for needle in needles)


def _norm(text: str) -> str:
    return " ".join(str(text or "").split())


class Persona:
    """One scripted caller: a reply per typed cursor step, a stuck-step guard, and a full send log.

    `answer()` is only consumed by the live loop; the offline self-test fills `counts` / `sent` and replays the
    same schedule through the engine, so the identical check function grades both.
    """

    def __init__(self, pid: str, pack_id: str, lines: dict[str, tuple], *, close: str, check) -> None:
        self.pid, self.pack_id, self.lines, self.close, self.check = pid, pack_id, lines, close, check
        self.counts: dict[str, int] = {}
        self.sent: list[dict[str, Any]] = []
        self.defaults: list[str] = []
        self.frames: dict[str, int] = {}
        self.epoch = 0
        self.call_ids: list[int] = []
        self._answered: set[tuple[str, str]] = set()
        self._repeats: dict[tuple[str, str], int] = {}

    # -- routing ----------------------------------------------------------------------------------------
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
            return _quick("clear")
        return _line("पता नहीं।" if state.cursor.substep is SubStep.FOLLOWUP else "नहीं।")

    def _cap(self, key: str) -> int:
        picked = self.lines.get(key)
        return len(picked[1]) if (picked and picked[0] == "steps") else 1

    def already_answered(self, state: CallState) -> bool:
        """One answer per (routing key, step token) - unless the script still owes this key another line.

        A re-published step keeps its qid but may keep or change its token. A repeat publish on a token already
        answered is skipped TWICE and answered on the third: a rewind that genuinely reopens the same token must
        not stall the call until the deadline, while a delayed Decision reply must not burn a real attempt.
        """
        key, tok = self.route(state), state.step_token
        if (key, tok) not in self._answered:
            return False
        self._repeats[(key, tok)] = self._repeats.get((key, tok), 0) + 1
        return self._repeats[(key, tok)] <= 2 and self.counts.get(key, 0) >= self._cap(key)

    def answer(self, state: CallState) -> tuple[str, str] | None:
        if state.phase is not Phase.ASKING:
            return None
        key = self.route(state)
        if self.counts.get(key, 0) > 6:  # a rewinding call may legitimately revisit a step; never spin forever
            raise RuntimeError(f"stuck on {key}")
        picked = self.lines.get(key)
        if picked is None:
            self.defaults.append(key)
            reply = Persona.fallback(state)
        elif picked[0] == "steps":
            reply = picked[1][min(self.counts.get(key, 0), len(picked[1]) - 1)]
        else:
            reply = picked
        self.counts[key] = self.counts.get(key, 0) + 1
        if reply[0] in ("text", "quick"):
            self._answered.add((key, state.step_token))
        return reply

    def sent_on(self, key: str) -> list[dict[str, Any]]:
        return [item for item in self.sent if item["key"] == key]

    def observe(self, kind: str, msg: dict) -> None:
        self.frames[kind] = self.frames.get(kind, 0) + 1
        if isinstance(epoch := msg.get("epoch"), int):
            self.epoch = epoch


def _first_send(p: Persona, key: str) -> int:
    return next((i for i, s in enumerate(p.sent) if s["key"] == key), -1)


# ------------------------------------------------------------------------------------------------------------
# The script
# ------------------------------------------------------------------------------------------------------------
FLIP_MARRIED = ("अरे माफ़ करना दीदी, मैंने पहले गलत बोल दिया था — मेरी शादी हो चुकी है और मेरे दो बच्चे हैं, "
                "एक 12 साल की बेटी रिया और एक 6 साल का बेटा आरव!")
ADD_TWIN = ("वैसे एक बात और बतानी थी — आरव का एक जुड़वाँ भाई भी है, 6 साल का बेटा कबीर, मैं पहले उसका नाम "
            "बताना भूल गई थी!")
TRANSFUSION = "हाँ, मेरे बेटे आरव को थैलेसीमिया मेजर की वजह से हर महीने खून चढ़ता है।"
UNCLE_HEART = "अरे हाँ, एक बात याद आई — मेरे चाचा को कम उम्र में दिल की बीमारी और हार्ट अटैक हुआ था।"
NEPHEW = "हाँ, मेरे भाई के बेटे को भी यही दिक्कत है।"

U_LINES: dict[str, tuple] = {
    "intro_name": _line("मेरा नाम प्रिया है।"),
    # Unmarried: `intro_children`, every child gate, the pregnancy gates, `fam_related` and `fam_miscarriage` are
    # all recorded NOT_APPLICABLE, and the walk jumps straight to `preg_thalassemia`.
    "intro_married": _line("अभी शादी नहीं हुई है।"),
    # Only ever reached if the mid-call flip FAILED to carry the roster: the ages reask is the recovery, and
    # `U1 intro_children never re-asked` is what reports the miss.
    "intro_children": _line("मेरे दो बच्चे हैं, एक 12 साल की बेटी रिया और एक 6 साल का बेटा आरव।"),
    "preg_thalassemia": _line("नहीं, कोई दिक्कत नहीं निकली थी।"),
    # U1, step 1: the demographic flip, on a gate, with the cursor at index 48.
    "fam_infant_death": _steps(_line(FLIP_MARRIED), _line("नहीं, परिवार में किसी छोटे बच्चे की मौत नहीं हुई।")),
    # U2, part 1: a NICU concern attributed to आरव (`item_2`) with both follow-ups spent.
    "birth_nicu": _line("हाँ, जन्म के बाद 4 दिन काँच के बक्से में रखा था, साँस की दिक्कत थी।"),
    "birth_nicu:which_item": _line("आरव को।"),
    "birth_nicu:fu0": _line("साँस की दिक्कत थी।"),
    "birth_nicu:fu1": _line("चार दिन।"),
    # U2, step 1: add the third child (आरव's twin कबीर) mid-call; step 2 answers body_fits itself.
    "body_fits": _steps(_line(ADD_TWIN), _line("नहीं, दौरे कभी नहीं पड़े।")),
    # U2, part 2: a second concern attributed to the child who did not exist when the first one was recorded.
    "body_bleeding_bruise": _line("हाँ, चोट लगने पर खून देर तक बहता है।"),
    "body_bleeding_bruise:which_item": _line("कबीर को।"),
    # U3: the diagnosed hemoglobinopathy, which must prune `body_blood_count` and be redacted from her copy.
    "body_transfusion": _line(TRANSFUSION),
    "body_transfusion:which_item": _line("आरव को।"),
    "body_transfusion:fu0": _line("हर महीने एक बार।"),
    # Never asked if the flag did its job; scripted so a failure to prune is visibly answered, not a stall.
    "body_blood_count": _line("नहीं।"),
    # U4, step 1: the remark that must revive `fam_same_problem` without becoming a concern of the close step.
    "close_anything_else": _steps(_line(UNCLE_HEART), _quick("clear")),
    "close_anything_else:fu0": _quick("clear"),
    "fam_same_problem": _line(NEPHEW),
    "fam_same_problem:fu0": _line("भाई के बेटे को।"),
}


# ------------------------------------------------------------------------------------------------------------
# Evidence: everything a check may read, whether it came from SQLite or from the self-test
# ------------------------------------------------------------------------------------------------------------
@dataclass
class Ev:
    pack: Any
    call: dict[str, Any]
    state: CallState
    persona: Persona
    decisions: list[tuple[str, str, str, str]] = field(default_factory=list)
    sources: list[tuple[str, str]] = field(default_factory=list)
    retained: list[str] = field(default_factory=list)
    retained_ev: list[StoredTurnEvidence] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    pages: int = 0
    pdf_text: str = ""
    wav_shape: tuple[int, int] | None = None
    proj: Any = None
    pinned: set[str] = field(default_factory=set)
    sequences: list[tuple[int, str, str]] = field(default_factory=list)
    interps: dict[int, Any] = field(default_factory=dict)

    def reasons(self) -> list[str]:
        return [f"{d[0]}:{d[1]}:{d[2]}" for d in self.decisions]

    def has(self, disposition: str = "", reason: str = "", target: str = "") -> bool:
        """True when some stored decision matches every named field (an empty field matches anything)."""
        return any((not disposition or d[0] == disposition) and (not reason or d[1] == reason)
                   and (not target or d[2] == target) for d in self.decisions)

    def events_of(self, kind: str) -> list[dict[str, Any]]:
        return [e for e in self.events if str(e.get("type") or "") == kind]


class Checklist:
    """Named assertions in order, so the JSON says which mechanism passed or failed."""

    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        self.items.append({"check": name, "ok": bool(ok), "detail": "" if ok else detail})
        return bool(ok)

    @property
    def problems(self) -> list[str]:
        return [f"{i['check']}: {i['detail']}" if i["detail"] else i["check"] for i in self.items if not i["ok"]]

    def failing(self) -> set[str]:
        return {i["check"] for i in self.items if not i["ok"]}


def _answered(ev: Ev, qid: str) -> tuple[AnswerStatus | None, str]:
    record = ev.state.answers.get(qid)
    return (record.state, record.verbatim_text) if record is not None else (None, "")


def _na_qids(ev: Ev) -> list[str]:
    return sorted(q for q, r in ev.state.answers.items() if r.state is AnswerStatus.NOT_APPLICABLE)


def _kid(st: CallState, name: str) -> Any:
    return next((c for c in st.roster if (c.name or "").strip() == name), None)


def _parity(ev: Ev, cl: Checklist) -> Any:
    """One projection must feed the stored note, the patient summary, and the flags."""
    call = ev.call
    proj = reports.project_summary(ev.pack, ev.state, ev.retained_ev)
    diverged = [c for c, f in (("summary_doctor", "doctor_markdown"), ("summary_patient", "patient_summary_text"))
                if str(call.get(c) or "").strip() != str(getattr(proj, f) or "").strip()]
    if list(call.get("red_flags") or []) != list(proj.red_flags):
        diverged.append("red_flags")
    cl.check("project_summary parity", not diverged, f"diverges: {diverged}")
    ev.proj = proj
    return proj


def _common(ev: Ev, cl: Checklist) -> None:
    """The checks every live call owes: telemetry, spoken leaks, WAV, and the PDF that exists at all."""
    telemetry, events = ev.call.get("telemetry") or {}, ev.events
    kinds = Counter(str(e.get("type") or "") for e in events)
    spoken = " ".join(str(t.get("text") or "") for t in ev.call.get("transcript") or []
                      if t.get("role") == "assistant").lower()
    cl.check("zero tool_errors", not int(telemetry.get("tool_errors") or 0), f"tool_errors={telemetry.get('tool_errors')}")
    cl.check("zero error events", not kinds.get("error"),
             f"errors={[e.get('detail') for e in events if e.get('type') == 'error'][:5]}")
    cl.check("closed on content, not the watchdog", not kinds.get("auto_close"),
             f"auto_close={[e.get('detail') for e in events if e.get('type') == 'auto_close']}")
    leaks = [leak for leak in _LEAKS if leak.lower() in spoken]
    cl.check("zero spoken leaks", not leaks, f"leaks={leaks}")
    cl.check("both summaries written",
             all(str(ev.call.get(f) or "").strip() for f in ("summary_doctor", "summary_patient")), "")
    cl.check("red flags reported", bool(ev.call.get("red_flags")), "no red flags projected")
    cl.check("stereo 16 kHz WAV", ev.wav_shape == (2, 16000), f"wav is {ev.wav_shape}")
    cl.check("patient PDF exists", bool(ev.pages), "missing pdf")
    rows = [f"{'/'.join(row)}" for row in _GLOSSARY_PDF if not any(frag in ev.pdf_text for frag in row)]
    cl.check("glossary in the PDF", bool(ev.pdf_text.strip()) and not rows, f"missing glossary rows {rows}")
    cl.check("PDF is at least two pages", ev.pages >= 2, f"pages={ev.pages}")


# ------------------------------------------------------------------------------------------------------------
# U1 - unmarried -> married mid-call, two named children, and the cursor back at the birth block
# ------------------------------------------------------------------------------------------------------------
def _check_u1(ev: Ev, cl: Checklist) -> None:
    st, p = ev.state, ev.persona
    marital, married_rec = st.profile.get("marital_status"), st.answers.get("intro_married")
    cl.check("U1 marital_status flipped to married",
             marital == "married" and married_rec is not None and married_rec.value == "married",
             f"profile.marital_status={marital!r} intro_married={married_rec.value if married_rec else None!r}")
    kids = {name: _kid(st, name) for name in _U1_KIDS}
    wrong = {name: None if c is None else (c.id, c.age_years, c.tag) for name, c in kids.items()
             if c is None or (c.age_years, c.tag) != _U1_KIDS[name]}
    cl.check("U1 the flip carried रिया 12/b and आरव 6/b",
             not wrong and st.roster_incomplete is False,
             f"wrong={wrong} roster={[(c.id, c.name, c.age_years, c.tag) for c in st.roster]} "
             f"incomplete={st.roster_incomplete}")
    flip, birth = _first_send(p, "fam_infant_death"), _first_send(p, "birth_mode_detail")
    cl.check("U1 birth_mode_detail asked only after the flip (cursor rewound to index 4)",
             0 <= flip < birth and p.counts.get("birth_mode_detail", 0) >= 1,
             f"flip_send={flip} birth_mode_detail_send={birth} sends={[s['key'] for s in p.sent][:12]}")
    cl.check("U1 intro_children never re-asked (the flip alone carried the roster)",
             p.counts.get("intro_children", 0) == 0,
             f"intro_children asked {p.counts.get('intro_children', 0)}x; roster={[(c.name, c.age_years) for c in st.roster]}")
    na = set(_na_qids(ev))
    cl.check("U1 every gate the unmarried skip pruned woke up and was asked", not (na - _ALLOWED_NA),
             f"unjustified NOT_APPLICABLE: {sorted(na - _ALLOWED_NA)}")
    birth_gates = [q.id for q in ev.pack.questions if q.group == "birth"]
    unasked = [q for q in birth_gates if q not in st.answers or st.answers[q].state is AnswerStatus.NOT_APPLICABLE]
    cl.check("U1 the whole birth block was walked after the flip", not unasked, f"not walked: {unasked}")
    cl.check("U1 fam_infant_death settled CLEAR after the flip", _answered(ev, "fam_infant_death")[0] is CL,
             f"got {_answered(ev, 'fam_infant_death')[0]} sends={[s['step'] for s in p.sent_on('fam_infant_death')]}")


# ------------------------------------------------------------------------------------------------------------
# U2 - a third child (आरव's twin) added mid-call after a concern was recorded against आरव
# ------------------------------------------------------------------------------------------------------------
def _check_u2(ev: Ev, cl: Checklist) -> None:
    st, p = ev.state, ev.persona
    ids = [c.id for c in st.roster]
    names = sorted((c.name or "") for c in st.roster)
    cl.check("U2 roster has three children with distinct stable ids",
             len(st.roster) == 3 and len(set(ids)) == 3 and st.roster_incomplete is False and names == sorted(["रिया", "आरव", "कबीर"]),
             f"roster={[(c.id, c.name, c.age_years, c.tag) for c in st.roster]} incomplete={st.roster_incomplete}")
    aarav, kabir, riya = _kid(st, "आरव"), _kid(st, "कबीर"), _kid(st, "रिया")
    cl.check("U2 the twins are both 6-year-old sons and the daughter is 12",
             all(x is not None for x in (aarav, kabir, riya))
             and (aarav.age_years, aarav.tag) == (6.0, "male") and (kabir.age_years, kabir.tag) == (6.0, "male")
             and (riya.age_years, riya.tag) == (12.0, "female"),
             f"आरव={aarav} कबीर={kabir} रिया={riya}")
    nicu_items = list(st.answers["birth_nicu"].item_ids) if "birth_nicu" in st.answers else []
    cl.check("U2 birth_nicu still points at आरव after the roster grew",
             bool(aarav) and nicu_items == [aarav.id],
             f"birth_nicu.item_ids={nicu_items} आरव={aarav.id if aarav else None} roster={[(c.id, c.name) for c in st.roster]}")
    bruise = st.answers.get("body_bleeding_bruise")
    cl.check("U2 body_bleeding_bruise points at the newly added कबीर",
             bool(kabir) and bruise is not None and list(bruise.item_ids) == [kabir.id],
             f"body_bleeding_bruise={None if bruise is None else (bruise.state.value, list(bruise.item_ids))} "
             f"कबीर={kabir.id if kabir else None} sends={[s['key'] for s in p.sent_on('body_bleeding_bruise')]}")
    prompt = engine.which_item_prompt(ev.pack, st)
    rows = [str(t.get("text") or "") for t in (ev.call.get("transcript") or [])
            if t.get("role") == "assistant" and (t.get("qid") or "") in ("body_bleeding_bruise", "body_transfusion")]
    spoken = " ".join(rows)
    cl.check("U2 no 6-year-old twin is called मँझला (middle) in the which_item prompt",
             bool(prompt.strip()) and not _has(prompt, "बीच वाले", "बीच वाली", "मँझला", "मंझला")
             and not _has(spoken, "बीच वाले", "बीच वाली", "मँझला", "मंझला"),
             f"prompt={prompt!r} spoken={spoken[:160]!r}")


# ------------------------------------------------------------------------------------------------------------
# U3 - the hemoglobinopathy flag prunes body_blood_count, and the diagnosis is redacted for the mother
# ------------------------------------------------------------------------------------------------------------
def _check_u3(ev: Ev, cl: Checklist) -> None:
    st, p = ev.state, ev.persona
    doctor, patient = str(ev.call.get("summary_doctor") or ""), str(ev.call.get("summary_patient") or "")
    cl.check("U3 confirmed_hemoglobinopathy flag recorded",
             "confirmed_hemoglobinopathy" in st.semantic_flags, f"semantic_flags={st.semantic_flags}")
    blood, asked = _answered(ev, "body_blood_count"), p.counts.get("body_blood_count", 0)
    cl.check("U3 body_blood_count pruned to NOT_APPLICABLE and never asked",
             blood[0] is AnswerStatus.NOT_APPLICABLE and asked == 0, f"state={blood[0]} asked={asked}x")
    cl.check("U3 the diagnosis is on body_transfusion's verbatim note",
             _has(_answered(ev, "body_transfusion")[1], "थैलेसीमिया", "मेजर"),
             f"note={_answered(ev, 'body_transfusion')[1][:160]!r} "
             f"(a gate with follow-ups keeps her words only when `detail_present` fired; otherwise the "
             f"diagnosis is lost)")
    cl.check("U3 doctor summary keeps थैलेसीमिया मेजर",
             _has(doctor, "थैलेसीमिया") and _has(doctor, "मेजर") and _has(doctor, "आरव"),
             f"doctor has थैलेसीमिया={'थैलेसीमिया' in doctor} मेजर={'मेजर' in doctor}; "
             f"note={_answered(ev, 'body_transfusion')[1][:120]!r}")
    cl.check("U3 patient summary redacts the diagnosis",
             bool(patient.strip()) and not _has(patient, *_DISEASE) and _has(patient, "आरव"),
             f"patient kept the child but must drop the severity: मेजर={'मेजर' in patient}")
    leaked = [frag for frag in _DISEASE if frag in ev.pdf_text]
    cl.check("U3 patient PDF redacts the diagnosis", bool(ev.pdf_text.strip()) and not leaked,
             f"PDF leaked fragments {leaked}")
    # The bare disease name is the pack's OWN carrier-screening wording (`preg_thalassemia`'s say and
    # `universal_advice_line_hi`), printed for every mother, so only the severity token proves a leak here.
    spoken = " ".join(str(t.get("text") or "") for t in (ev.call.get("transcript") or [])
                      if t.get("role") == "assistant")
    cl.check("U3 Gemini never spoke the diagnosis", "मेजर" not in spoken,
             f"assistant said {[x for x in (str(t.get('text') or '') for t in (ev.call.get('transcript') or []))
                               if 'मेजर' in x][:2]}")


# ------------------------------------------------------------------------------------------------------------
# U4 - fam_same_problem resurrected from NOT_APPLICABLE by a remark on close_anything_else
# ------------------------------------------------------------------------------------------------------------
def _check_u4(ev: Ev, cl: Checklist) -> None:
    st, p = ev.state, ev.persona
    heart, same, close = (_answered(ev, q)[0] for q in ("fam_heart_kidney", "fam_same_problem",
                                                       "close_anything_else"))
    close_rec = st.answers.get("close_anything_else")
    cl.check("U4 fam_heart_kidney flipped CLEAR -> CONCERN from the close turn", heart is C, f"got {heart}")
    first_close, first_same = _first_send(p, "close_anything_else"), _first_send(p, "fam_same_problem")
    cl.check("U4 fam_same_problem woke up after close_anything_else and was asked",
             0 <= first_close < first_same and p.counts.get("fam_same_problem", 0) >= 1,
             f"close_send={first_close} fam_same_problem_send={first_same} "
             f"asks={p.counts.get('fam_same_problem', 0)} state={same}")
    cl.check("U4 fam_same_problem resolved CONCERN with the nephew", same is C,
             f"got {same} note={_answered(ev, 'fam_same_problem')[1][:80]!r}")
    cl.check("U4 close_anything_else recorded CLEAR, no phantom concern",
             close is CL and close_rec is not None and not (close_rec.verbatim_text or "").strip(),
             f"state={close} note={(close_rec.verbatim_text if close_rec else '')[:80]!r}")
    cl.check("U4 the revision was recorded as OTHER_FORM on fam_heart_kidney",
             ev.has("accept_form", "other_form", "fam_heart_kidney"), f"decisions={ev.reasons()}")
    flags = " ".join(str(x) for x in (ev.call.get("red_flags") or []))
    cl.check("U4 the revived family concern reached red_flags", "fam_heart_kidney" in flags,
             f"red_flags={ev.call.get('red_flags')}")


def _prompt_strings(ev: Ev) -> list[tuple[str, str]]:
    """Every Hindi question text the pack can speak, in ANY variant, keyed by qid.

    A variant's `say` is the one that actually reaches the wire (`question_say`, `engine.py:99`), so a doubled
    caption on an unmarried mother repeats the variant, never the base `say`.
    """
    out = []
    for qid, spec in ev.pack.by_id.items():
        for said in (spec.say, *(v.say for v in spec.variants)):
            if len(_norm(said)) >= 20:
                out.append((qid, _norm(said)))
    return out


def _doubled_prompts(ev: Ev) -> list[dict[str, Any]]:
    """Assistant captions that are one pack prompt spoken twice in a single row.

    `merge_transcript_chunk` collapses an identical repeat (`engine.py:19-22`), so a doubled row cannot be an
    append artefact: two full copies of the same question arrived inside ONE transcription payload and were
    concatenated with a space, which the caller then hears twice.
    """
    out, prompts = [], _prompt_strings(ev)
    for row in ev.call.get("transcript") or []:
        if row.get("role") != "assistant":
            continue
        text = _norm(row.get("text"))
        for qid, said in prompts:
            if text.startswith(said) and _norm(text[len(said):]).startswith(said.rstrip("?")):
                out.append({"row_qid": row.get("qid") or "", "prompt_qid": qid, "len": len(text),
                            "text": text[:180]})
            elif not text.startswith(said) and f" {said}" in text:
                # the repeat can also land behind a lead-in the model added; report it the same way
                head, tail = text.split(f" {said}", 1)
                if tail.startswith(said.rstrip("?")):
                    out.append({"row_qid": row.get("qid") or "", "prompt_qid": qid, "len": len(text),
                                "text": text[:180]})
    return out


def _check_u(ev: Ev, cl: Checklist) -> dict[str, Any]:
    """All four hypotheses over the one call, plus the caption integrity every live call owes."""
    _check_u1(ev, cl)
    _check_u2(ev, cl)
    _check_u3(ev, cl)
    _check_u4(ev, cl)
    doubled = _doubled_prompts(ev)
    st, p = ev.state, ev.persona
    # X1: not one of the four hypotheses - a caption defect this call happened to expose. Kept as a hard check
    # rather than a note, because a doubled question in the stored transcript is what a reader of the record sees.
    cl.check("X1 no prompt was captioned twice in one row", not doubled, f"doubled={doubled}")
    return {"roster": [{"id": c.id, "name": c.name, "age": c.age_years, "tag": c.tag} for c in st.roster],
            "semantic_flags": list(st.semantic_flags), "na": _na_qids(ev),
            "which_item_prompt": engine.which_item_prompt(ev.pack, st), "doubled_prompts": doubled,
            "turns": {k: v for k, v in p.counts.items() if v > 0},
            "defaults": p.defaults, "sent_total": len(p.sent),
            "answers": {q: {"state": r.state.value, "items": list(r.item_ids), "value": r.value,
                            "note": (r.verbatim_text or "")[:200],
                            "patient_safe": (r.patient_safe_text or "")[:200],
                            "followups": {f: s.state.value for f, s in r.followups.items()}}
                        for q, r in sorted(st.answers.items())
                        if q in ("intro_married", "intro_children", "birth_nicu", "body_fits", "body_bleeding_bruise",
                                 "body_transfusion", "body_blood_count", "fam_heart_kidney", "fam_same_problem",
                                 "fam_infant_death", "close_anything_else")}}


# ------------------------------------------------------------------------------------------------------------
# Live plumbing
# ------------------------------------------------------------------------------------------------------------
def _read_pdf(call_id: int) -> tuple[int, str]:
    path = Path(db.DATA_DIR) / "pdfs" / f"call-{int(call_id)}.pdf"
    if not path.is_file() or not path.read_bytes().startswith(b"%PDF"):
        return 0, ""
    reader = PdfReader(io.BytesIO(path.read_bytes()))
    return len(reader.pages), " ".join(" ".join(p.extract_text() or "" for p in reader.pages).split())


def _wav_shape(call: dict) -> tuple[int, int] | None:
    path = Path(db.DATA_DIR) / "audio" / str(call.get("audio_path") or "")
    if not path.is_file():
        return None
    with wave.open(str(path), "rb") as handle:
        return handle.getnchannels(), handle.getframerate()


def _evidence(pack, call_id: int, persona: Persona) -> Ev:
    """Everything the check needs, read once from the finished call's own row and ledger."""
    call = db.get_call(call_id) or {}
    pages, text = _read_pdf(call_id)
    retained = list(db.list_retained_observations(call_id))
    evidence = list(db.list_turn_evidence(call_id))
    decisions = list(db.list_policy_decisions(call_id))
    return Ev(pack=pack, call=call, state=db.call_state_from_row(call), persona=persona,
              decisions=[(d.disposition.value, d.reason.value, str(d.target_qid or ""), d.cursor_action.value)
                         for d in decisions],
              sources=[(e.source.value, e.snapshot_step) for e in evidence],
              retained=[str(o.transcript) for o in retained], retained_ev=retained,
              events=list(call.get("events") or []), pages=pages, pdf_text=text, wav_shape=_wav_shape(call),
              sequences=[(e.sequence, e.snapshot_step, e.transcript) for e in evidence],
              interps={i.sequence: i.interpretation for i in db.list_turn_interpretations(call_id)})


def _login() -> str:
    """The `shishu_auth` cookie pair for the shared persona passcode. The value is never printed."""
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


class Runner:
    """One live call: reads frames, hands them to the persona, and acts on whatever the server asks for next."""

    def __init__(self, persona: Persona, pack, cookie: str) -> None:
        self.persona, self.pack, self.cookie = persona, pack, cookie
        self.ws = None
        self.call_id = 0
        self.started = time.monotonic()
        self.deadline = self.started + CALL_DEADLINE_SEC

    async def connect(self) -> None:
        import websockets

        self.ws = await websockets.connect(f"ws://{HOST}:{PORT}/ws/live?pack={self.pack.id}",
                                           additional_headers={"Cookie": self.cookie},
                                           open_timeout=30, max_size=8_000_000)

    async def _read(self) -> dict[str, Any] | None:
        try:
            msg = json.loads(await asyncio.wait_for(self.ws.recv(), timeout=RECV_TIMEOUT_SEC))
        except (asyncio.TimeoutError, Exception):  # noqa: BLE001 - a dead socket is reported by the caller
            return None
        if not isinstance(msg, dict):
            return None
        if isinstance(kind := msg.get("type"), str):
            self.persona.observe(kind, msg)
        return msg

    async def _act(self, entry: dict[str, Any], state: CallState) -> None:
        mode, payload = entry["mode"], entry["payload"]
        if mode == "text":
            await self.ws.send(json.dumps({"type": "text", "text": payload}, ensure_ascii=False))
        elif mode == "quick":
            await self.ws.send(json.dumps({"type": "quick_reply", "value": payload, "step": state.step_token,
                                           "epoch": self.persona.epoch}, ensure_ascii=False))
        else:
            raise RuntimeError(f"unknown action {mode}")

    async def run(self) -> dict[str, Any]:
        persona = self.persona
        await self.connect()
        heard = False
        while time.monotonic() < self.deadline:
            msg = await self._read()
            if msg is None:
                return {"id": persona.pid, "pack": self.pack.id, "call_id": self.call_id, "ok": False,
                        "problems": ["socket timeout or deadline"]}
            kind, event = msg.get("type"), (msg.get("last_event") or {}).get("type")
            if kind == "session_started":
                self.call_id = int(msg.get("call_id") or 0)
                persona.call_ids.append(self.call_id)
                print(f"  session started call {self.call_id}", flush=True)
            elif kind == "bot_turn_complete":
                heard = True
            elif kind == "telemetry" and event == "bot_first_audio":
                heard = True
            elif kind == "telemetry" and event == "turn_complete" and heard and self.call_id:
                heard = False
                state = db.call_state_from_row(db.get_call(self.call_id) or {})
                key = Persona.route(state)
                if persona.sent and persona.already_answered(state):
                    print(f"  [{key}] duplicate publish on {state.step_token}, ignored", flush=True)
                    continue
                reply = persona.answer(state)
                if reply is None:
                    continue
                entry = {"key": key, "mode": reply[0], "payload": reply[1], "step": state.step_token,
                         "epoch": persona.epoch}
                persona.sent.append(entry)
                print(f"  [{key}] {reply[0]} epoch={persona.epoch} {str(reply[1])[:70]}", flush=True)
                await self._act(entry, state)
            elif kind == "call_completed":
                return self._grade(int(msg.get("call_id") or self.call_id))
            elif kind in {"report_failed", "error"}:
                return {"id": persona.pid, "pack": self.pack.id, "call_id": self.call_id, "ok": False,
                        "problems": [f"{kind}: {msg.get('message') or ''}"]}
        return {"id": persona.pid, "pack": self.pack.id, "call_id": self.call_id, "ok": False, "problems": ["deadline"]}

    def _grade(self, call_id: int) -> dict[str, Any]:
        ev, cl = _evidence(self.pack, call_id, self.persona), Checklist()
        detail: dict[str, Any] = {}
        cl.check(f"close_reason {self.persona.close}", str(ev.state.close_reason or "") == self.persona.close,
                 f"close_reason={ev.state.close_reason}")
        try:
            _parity(ev, cl)
            _common(ev, cl)
            detail = self.persona.check(ev, cl) or {}
        except Exception as exc:  # noqa: BLE001 - a broken check reports, it never aborts the suite
            cl.check("checklist ran to the end", False, f"{type(exc).__name__}: {exc}")
        result = {"id": self.persona.pid, "pack": self.pack.id, "call_id": call_id, "ok": not cl.problems,
                  "problems": cl.problems, "checklist": cl.items,
                  "close_reason": str(ev.state.close_reason or ""), "phase": str(ev.state.phase),
                  "duration_sec": ev.call.get("duration_sec"), "wall_sec": round(time.monotonic() - self.started, 1),
                  "patient_name": ev.call.get("patient_name"), "share_token": ev.call.get("share_token"),
                  "red_flags": list(ev.call.get("red_flags") or []), "telemetry": ev.call.get("telemetry") or {},
                  "frames": self.persona.frames, "default_replies": self.persona.defaults,
                  "events": _events_summary(ev.events), "sends": self.persona.sent}
        result.update(detail)
        print(("PASS " if result["ok"] else "FAIL ") + self.persona.pid
              + f" call={call_id} dur={result.get('duration_sec')} " + ", ".join(result["problems"]), flush=True)
        return result


def _events_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(e.get("type") or "") for e in events)
    return {"total": len(events), "by_type": dict(sorted(counts.items())),
            "errors": [e.get("detail") for e in events if e.get("type") == "error"][:10],
            "barge_ins": counts.get("barge_in", 0), "prompt_recues": counts.get("prompt_recue", 0)}


def _port_free() -> bool:
    """A suite that silently shares its port with another run grades the wrong socket."""
    with socket.socket() as probe:
        try:
            probe.bind((HOST, PORT))
            return True
        except OSError:
            return False


async def _main() -> int:
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
    results: list[dict[str, Any]] = []
    try:
        cookie = await asyncio.to_thread(_login)
        persona = Persona(PID, PACK_ID, U_LINES, close="complete", check=_check_u)
        print(f"\n=== {persona.pid} ({persona.pack_id}) ===", flush=True)
        results.append(await Runner(persona, load_pack(persona.pack_id), cookie).run())
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


# ------------------------------------------------------------------------------------------------------------
# Offline self-test: the same schedule through the real reducers, then every assertion must fail when broken
# ------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Step:
    """One scripted turn, with the command the live interpreter must derive for it."""
    text: str
    cmd: TurnCommand
    key: str = ""
    child: str = ""      # for a which_item step, the child's NAME; the id is resolved from the live roster
    profile: bool = False  # a profile step's patch is applied by `reduce_turn` directly, as live does


def _clear(text: str = "नहीं।") -> Step:
    return Step(text, TurnCommand(outcome=TurnOutcome.CLEAR, utterance=text))


def _profile(slots: dict[str, Any], text: str, *, roster: tuple[RosterInput, ...] | None = None,
             complete: bool | None = None) -> Step:
    return Step(text, TurnCommand(profile_patch=ProfilePatch(slots=slots, roster=roster, roster_complete=complete),
                                  profile_turn_attempted=True, outcome=TurnOutcome.NOT_ABOUT_ACTIVE, utterance=text),
                profile=True)


def _offline_script() -> dict[str, list[Step]]:
    """`U_LINES` as typed commands, in the same order the live call sends them."""
    return {
        "intro_name": [_profile({"caller_name": "प्रिया"}, "मेरा नाम प्रिया है।")],
        "intro_married": [Step("अभी शादी नहीं हुई है।",
                               TurnCommand(outcome=TurnOutcome.ANSWERED, value="unmarried",
                                           utterance="अभी शादी नहीं हुई है।"))],
        "preg_thalassemia": [_clear("नहीं, कोई दिक्कत नहीं निकली थी।")],
        # U1: the flip lands as the extractor delivers it - marital status plus the complete roster.
        "fam_infant_death": [_profile({"marital_status": "married"}, FLIP_MARRIED,
                                      roster=(RosterInput(age_years=12.0, tag="female", name="रिया"),
                                              RosterInput(age_years=6.0, tag="male", name="आरव")), complete=True),
                             _clear("नहीं, परिवार में किसी छोटे बच्चे की मौत नहीं हुई।")],
        # U2: a NICU concern on आरव, both follow-ups spent, then the third child added and a concern on कबीर.
        "birth_nicu": [Step("हाँ, जन्म के बाद 4 दिन काँच के बक्से में रखा था, साँस की दिक्कत थी।",
                            TurnCommand(outcome=TurnOutcome.CONCERN,
                                        utterance="हाँ, जन्म के बाद 4 दिन काँच के बक्से में रखा था, साँस की दिक्कत थी।"))],
        "birth_nicu:which_item": [Step("आरव को।", TurnCommand(outcome=TurnOutcome.ANSWERED, utterance="आरव को।"),
                                       child="आरव")],
        "birth_nicu:fu0": [Step("साँस की दिक्कत थी।", TurnCommand(outcome=TurnOutcome.CONCERN, utterance="साँस की दिक्कत थी।"))],
        "birth_nicu:fu1": [Step("चार दिन।", TurnCommand(outcome=TurnOutcome.CONCERN, utterance="चार दिन।"))],
        "body_fits": [_profile({}, ADD_TWIN,
                               roster=(RosterInput(id="item_1", age_years=12.0, tag="female", name="रिया"),
                                       RosterInput(id="item_2", age_years=6.0, tag="male", name="आरव"),
                                       RosterInput(age_years=6.0, tag="male", name="कबीर")), complete=True),
                      _clear("नहीं, दौरे कभी नहीं पड़े।")],
        "body_bleeding_bruise": [Step("हाँ, चोट लगने पर खून देर तक बहता है।",
                                      TurnCommand(outcome=TurnOutcome.CONCERN, utterance="हाँ, चोट लगने पर खून देर तक बहता है।"))],
        "body_bleeding_bruise:which_item": [Step("कबीर को।", TurnCommand(outcome=TurnOutcome.ANSWERED, utterance="कबीर को।"),
                                                 child="कबीर")],
        "body_transfusion": [Step(TRANSFUSION, TurnCommand(outcome=TurnOutcome.CONCERN, utterance=TRANSFUSION,
                                                           redacted=True, followup_answered=True,
                                                           semantic_flags=("confirmed_hemoglobinopathy",)))],
        "body_transfusion:which_item": [Step("आरव को।", TurnCommand(outcome=TurnOutcome.ANSWERED, utterance="आरव को।"),
                                             child="आरव")],
        "body_transfusion:fu0": [Step("हर महीने एक बार।", TurnCommand(outcome=TurnOutcome.CONCERN, utterance="हर महीने एक बार।"))],
        "body_blood_count": [_clear()],
        "close_anything_else": [Step(UNCLE_HEART, TurnCommand(outcome=TurnOutcome.NOT_ABOUT_ACTIVE,
                                                              stated={"fam_heart_kidney": TurnOutcome.CONCERN},
                                                              utterance=UNCLE_HEART)),
                                _clear("नहीं।")],
        "fam_same_problem": [Step(NEPHEW, TurnCommand(outcome=TurnOutcome.CONCERN, utterance=NEPHEW))],
        "fam_same_problem:fu0": [Step("भाई के बेटे को।", TurnCommand(outcome=TurnOutcome.CONCERN, utterance="भाई के बेटे को।"))],
    }


def _interp(cmd: TurnCommand):
    """The interpretation the live Decision pass hands `policy` for a scripted line."""
    from intake.turns import ClinicalClaim, DomainFit, SpeechAct, TopicRelation, TurnInterpretation

    stated = {t: o for t, o in cmd.stated.items() if o in (TurnOutcome.CONCERN, TurnOutcome.CLEAR)}
    claims: list[ClinicalClaim] = []
    act, rel, dom = SpeechAct.NONE, TopicRelation.UNRELATED, DomainFit.NONE
    if stated:
        claims = [ClinicalClaim(target_qid=t, relation=TopicRelation.OTHER_FORM, domain_fit=DomainFit.IN_DOMAIN, status=o)
                  for t, o in stated.items()]
        act, rel = SpeechAct.ANSWER, TopicRelation.OTHER_FORM
    elif cmd.outcome in (TurnOutcome.CLEAR, TurnOutcome.CONCERN, TurnOutcome.UNKNOWN, TurnOutcome.ANSWERED):
        claims = [ClinicalClaim(target_qid=None, relation=TopicRelation.ACTIVE, domain_fit=DomainFit.IN_DOMAIN,
                                status=cmd.outcome, value=cmd.value)]
        act, rel, dom = SpeechAct.ANSWER, TopicRelation.ACTIVE, DomainFit.IN_DOMAIN
    return TurnInterpretation(act=act, active_relation=rel, active_domain=dom, claims=tuple(claims),
                              profile_patch=cmd.profile_patch, profile_turn_attempted=cmd.profile_turn_attempted,
                              item_ref=cmd.item_ref, followup_answered=cmd.followup_answered,
                              semantic_flags=cmd.semantic_flags, redacted=cmd.redacted)


def _walk(pack) -> Ev:
    """Replay the scripted schedule through the real `policy` + `engine` so the shapes in `_check_u` are real."""
    from intake import policy
    from intake.turns import EvidenceSource, TurnEvidence

    script, persona = _offline_script(), Persona(PID, PACK_ID, U_LINES, close="complete", check=_check_u)
    st, used, decisions, sequences, interps = engine.new_state(pack), Counter(), [], [], {}
    seq = 1
    for _ in range(400):
        if st.phase is not Phase.ASKING:
            break
        key, step, before = Persona.route(st), st.cursor, st.step_token
        queue, i = script.get(key) or [], used[key]
        if i >= len(queue):
            step_cmd = _clear()
        else:
            step_cmd = queue[i]
        used[key] += 1
        cmd = step_cmd.cmd
        if step_cmd.child:  # the child's live id is the roster's, never a name
            kid = _kid(st, step_cmd.child)
            cmd = cmd.model_copy(update={"item_ref": kid.id if kid is not None else ""})
        persona.counts[key] = persona.counts.get(key, 0) + 1
        persona.sent.append({"key": key, "mode": "text", "payload": step_cmd.text, "step": before, "epoch": 1})
        evidence = TurnEvidence(source=EvidenceSource.TEXT, snapshot_step=before, transcript=step_cmd.text, received_at=0.0)
        if step_cmd.profile or (cmd.profile_turn_attempted and cmd.profile_patch is not None):
            nxt, pol = engine.reduce_turn(pack, st, cmd), None
        else:
            interp = _interp(cmd)
            seen = [d.disposition for d in policy.evaluate_policy(pack, st, evidence, interp).decisions]
            nxt, pol = policy.apply_policy(pack, st, evidence, interp)
            if pol is not None:
                decisions.extend((d.disposition.value, d.reason.value, str(d.target_qid or ""), d.cursor_action.value)
                                 for d in pol.decisions)
                from intake.turns import DecisionReason
                if DecisionReason.OTHER_FORM in [d.reason for d in pol.decisions]:
                    interps[seq], sequences = interp, [*sequences, (seq, before, step_cmd.text)]
                if cmd.stated and not pol.command.stated:
                    raise RuntimeError(f"self-test: stated {cmd.stated} was rejected by policy at {key}")
                _ = seen
        if nxt.phase is Phase.ASKING and nxt.step_token == before and nxt.cursor.qid == step.qid:
            # A re-ask is legitimate (this persona answers the same gate twice), but never a frozen cursor.
            persona.sent[-1]["repeat"] = True
        seq += 1
        st = nxt

    persona.counts = {k: v for k, v in persona.counts.items()}
    call = {"id": 1, "telemetry": {"tool_errors": 0, "user_turns": len(persona.sent)},
            "events": [{"type": "session_started"}] + [{"type": "turn_complete"} for _ in persona.sent],
            "audio_path": "c1.wav",
            "transcript": [{"role": "assistant", "qid": "birth_nicu",
                            "text": "क्या आपके किसी भी बच्चे को जन्म के बाद काँच के बक्से में रखना पड़ा था?"}]}
    ev = Ev(pack=pack, call=call, state=st, persona=persona, decisions=decisions,
            sources=[("text", s[1]) for s in sequences], events=call["events"],
            sequences=sequences, interps=interps, pages=2,
            pdf_text="जविाब इनकर मतलब जवब नहीं", wav_shape=(2, 16000))
    ev.proj = reports.project_summary(pack, ev.state, ev.retained_ev)
    for key, value in (("summary_doctor", ev.proj.doctor_markdown), ("summary_patient", ev.proj.patient_summary_text),
                       ("red_flags", list(ev.proj.red_flags))):
        if key not in ev.pinned:
            ev.call = {**ev.call, key: value}
    return ev


def _mutations_u() -> list[tuple[str, str, Any]]:
    """(label, the check that must flip, mutation). Each mutation breaks exactly one invariant."""
    def state(ev: Ev, **kw) -> None:
        ev.state = ev.state.model_copy(deep=True, update=kw)

    def answer(ev: Ev, qid: str, **kw) -> None:
        rec = ev.state.answers[qid].model_copy(update=kw)
        state(ev, answers={**ev.state.answers, qid: rec})

    def roster(ev: Ev, items) -> None:
        state(ev, roster=list(items))

    def call(ev: Ev, **kw) -> None:
        ev.pinned.update(kw)
        ev.call = {**ev.call, **kw}

    def sent_index(ev: Ev, key: str, index: int = 1) -> int:
        seen = 0
        for i, s in enumerate(ev.persona.sent):
            if s["key"] == key:
                seen += 1
                if seen == index:
                    return i
        return -1

    def scrub(ev: Ev, qid: str) -> None:
        """Erase the diagnosis from EVERY note on a record: the main note and each follow-up slot are joined."""
        rec = ev.state.answers[qid]
        state(ev, answers={**ev.state.answers, qid: rec.model_copy(update={
            "note": NoteText(),
            "followups": {f: s.model_copy(update={"note": NoteText.from_text("कोई जानकारी नहीं।")})
                          for f, s in rec.followups.items()}})})

    def ask_before_close(ev: Ev) -> None:
        """Move the fam_same_problem send in front of the close turn: a resume that was never owed."""
        same, close = sent_index(ev, "fam_same_problem"), sent_index(ev, "close_anything_else")
        if same < 0 or close < 0 or same < close:
            raise RuntimeError("schedule has no fam_same_problem send after the close turn to move")
        ev.persona.sent.insert(close, ev.persona.sent.pop(same))

    return [
        ("marital status still unmarried", "U1 marital_status flipped to married",
         lambda ev: state(ev, profile={**ev.state.profile, "marital_status": "unmarried"})),
        ("the flip lost the daughter", "U1 the flip carried रिया 12/b and आरव 6/b",
         lambda ev: roster(ev, [c for c in ev.state.roster if c.name != "रिया"])),
        ("no rewind: birth_mode_detail never asked", "U1 birth_mode_detail asked only after the flip (cursor rewound to index 4)",
         lambda ev: (ev.persona.sent.remove(next(s for s in ev.persona.sent if s["key"] == "birth_mode_detail")),
                     ev.persona.counts.update({"birth_mode_detail": 0}))),
        ("the ages reask reopened the roster", "U1 intro_children never re-asked (the flip alone carried the roster)",
         lambda ev: ev.persona.counts.update({"intro_children": 1})),
        ("a child gate stayed NOT_APPLICABLE", "U1 every gate the unmarried skip pruned woke up and was asked",
         lambda ev: answer(ev, "birth_term", state=AnswerStatus.NOT_APPLICABLE)),
        ("birth_nicu renumbered onto the twin", "U2 birth_nicu still points at आरव after the roster grew",
         lambda ev: answer(ev, "birth_nicu", item_ids=[_kid(ev.state, "कबीर").id])),
        ("the third child never landed", "U2 roster has three children with distinct stable ids",
         lambda ev: roster(ev, [c for c in ev.state.roster if c.name != "कबीर"])),
        ("the new child's concern lost its child", "U2 body_bleeding_bruise points at the newly added कबीर",
         lambda ev: answer(ev, "body_bleeding_bruise", item_ids=[])),
        ("the twins became strictly ordered ages", "U2 no 6-year-old twin is called मँझला (middle) in the which_item prompt",
         lambda ev: roster(ev, [ev.state.roster[0].model_copy(update={"age_years": 12.0, "age_label": "12 साल"}),
                                ev.state.roster[1].model_copy(update={"age_years": 7.0, "age_label": "7 साल"}),
                                ev.state.roster[2].model_copy(update={"age_years": 6.0, "age_label": "6 साल"})])),
        ("the flag never fired", "U3 confirmed_hemoglobinopathy flag recorded",
         lambda ev: state(ev, semantic_flags=[])),
        ("body_blood_count was asked anyway", "U3 body_blood_count pruned to NOT_APPLICABLE and never asked",
         lambda ev: answer(ev, "body_blood_count", state=AnswerStatus.CLEAR) or ev.persona.counts.update({"body_blood_count": 1})),
        ("the doctor note lost the diagnosis", "U3 doctor summary keeps थैलेसीमिया मेजर",
         lambda ev: scrub(ev, "body_transfusion")),
        ("her copy leaked the diagnosis", "U3 patient summary redacts the diagnosis",
         lambda ev: answer(ev, "body_transfusion",
                           note=NoteText.append(ev.state.answers["body_transfusion"].note, "थैलेसीमिया मेजर"))),
        ("the PDF leaked the diagnosis", "U3 patient PDF redacts the diagnosis",
         lambda ev: setattr(ev, "pdf_text", ev.pdf_text + " मेजर")),
        ("Gemini read the diagnosis back", "U3 Gemini never spoke the diagnosis",
         lambda ev: call(ev, transcript=[{"role": "assistant", "qid": "body_transfusion", "text": "थैलेसीमिया मेजर ठीक है।"}])),
        ("the family concern never flipped", "U4 fam_heart_kidney flipped CLEAR -> CONCERN from the close turn",
         lambda ev: answer(ev, "fam_heart_kidney", state=AnswerStatus.CLEAR)),
        ("fam_same_problem never woke (still skipped)", "U4 fam_same_problem woke up after close_anything_else and was asked",
         lambda ev: ev.persona.counts.update({"fam_same_problem": 0})),
        ("fam_same_problem woke but stayed NA", "U4 fam_same_problem resolved CONCERN with the nephew",
         lambda ev: answer(ev, "fam_same_problem", state=AnswerStatus.NOT_APPLICABLE)),
        ("the close step swallowed the remark as a concern", "U4 close_anything_else recorded CLEAR, no phantom concern",
         lambda ev: answer(ev, "close_anything_else", state=AnswerStatus.CONCERN,
                           note=NoteText.from_text(UNCLE_HEART))),
        ("the revision was read as an aside, not OTHER_FORM", "U4 the revision was recorded as OTHER_FORM on fam_heart_kidney",
         lambda ev: setattr(ev, "decisions", [d for d in ev.decisions if d[2] != "fam_heart_kidney"])),
        ("a prompt captioned twice in one row", "X1 no prompt was captioned twice in one row",
         lambda ev: call(ev, transcript=[{"role": "assistant", "qid": "preg_thalassemia",
                                          "text": "क्या आपकी थैलेसीमिया खून जाँच में कभी माइनर या ट्रेट जैसी कोई दिक्कत बताई गई? "
                                                  "क्या आपकी थैलेसीमिया खून जाँच में कभी माइनर या ट्रेट जैसी कोई दिक्कत बताई गई?"}])),
        ("the revived concern lost its red flag", "U4 the revived family concern reached red_flags",
         lambda ev: call(ev, red_flags=[f for f in ev.call["red_flags"] if "fam_heart_kidney" not in f])),
        ("fam_same_problem asked before the close turn", "U4 fam_same_problem woke up after close_anything_else and was asked",
         ask_before_close),
    ]


def _script_legal(pack, ev: Ev) -> list[str]:
    """Static guards on the schedule itself: a key the walk never reaches is a line that will never be sent."""
    bad: list[str] = []
    # `intro_children` and `body_blood_count` are the two lines that must NEVER be asked (the flip carried the
    # roster; the flag pruned the gate) - they stay scripted so a regression is answered instead of stalling.
    # `body_transfusion:fu0` and `close_anything_else:fu0` are conditionally reachable: a follow-up is only asked
    # when the interpreter did not read the detail in her own sentence, and the close step's follow-up only opens
    # on a non-quiet non-concern outcome. Both stay scripted as the recovery line for that case.
    stale = sorted(set(U_LINES) - set(ev.persona.counts) - {"intro_children", "body_blood_count",
                                                            "body_transfusion:fu0", "close_anything_else:fu0"})
    for key in stale:
        picked = U_LINES[key]
        replies = picked[1] if picked[0] == "steps" else (picked,)
        for reply in replies:
            if reply[0] == "quick" and reply[1] not in ("clear", "concern", "unknown"):
                bad.append(f"illegal quick reply {reply[1]!r} on {key}")
    if stale:
        bad.append(f"scripted keys the walk never reaches: {stale}")
    return bad


def _selftest() -> int:
    """Prove every critical assertion is non-vacuous, offline, before a single live token is spent."""
    print("=== offline self-test: each invariant must fail when broken and pass when whole ===", flush=True)
    pack, failures = load_pack(PACK_ID), []
    whole = _walk(pack)
    for problem in _script_legal(pack, whole):
        failures.append(problem)
        print(f"  FAIL {problem}", flush=True)
    cl = Checklist()
    try:
        _parity(whole, cl)
        _common(whole, cl)
        whole.persona.check(whole, cl)
    except Exception as exc:  # noqa: BLE001
        failures.append(f"whole state raised {type(exc).__name__}: {exc}")
        print(f"  FAIL whole state raised {type(exc).__name__}: {exc}", flush=True)
        cl = Checklist()
        whole.persona.check(whole, cl)
    bad, total = cl.problems, len(cl.items)
    print(f"  {'PASS' if not bad else 'FAIL'} whole state {total - len(bad)}/{total} checks green"
          + ("" if not bad else f" - {bad}"), flush=True)
    if bad:
        failures.append(f"whole state is not green: {bad}")
    for label, target, mutate in _mutations_u():
        broken = Checklist()
        try:
            ev = _walk(pack)
            mutate(ev)
            ev.proj = reports.project_summary(pack, ev.state, ev.retained_ev)
            for key, value in (("summary_doctor", ev.proj.doctor_markdown), ("summary_patient", ev.proj.patient_summary_text),
                               ("red_flags", list(ev.proj.red_flags))):
                if key not in ev.pinned:
                    ev.call = {**ev.call, key: value}
            _parity(ev, broken)
            ev.persona.check(ev, broken)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{label} raised {type(exc).__name__}: {exc}")
            print(f"  FAIL {label} raised {type(exc).__name__}: {exc}", flush=True)
            continue
        hit = target in broken.failing()
        print(f"  {'PASS' if hit else 'FAIL'} {label} -> {target}", flush=True)
        if not hit:
            failures.append(f"{label} did not flip {target!r} (failing={sorted(broken.failing())})")
    print(f"\n{'SELFTEST PASS' if not failures else 'SELFTEST FAIL'} ({len(failures)} problem(s))", flush=True)
    for line in failures:
        print(f"  - {line}", flush=True)
    return 0 if not failures else 1


def _regrade(call_id: int) -> int:
    """Re-run the checks against a call already in this database, with no new live token spent.

    This is the honest way to fix a harness assertion that was itself wrong: the call's stored row, ledger and
    PDF are re-read as they are, never re-created, so the report says what the product did, not what the script
    wished it had done.
    """
    pack = load_pack(PACK_ID)
    persona = Persona(PID, PACK_ID, U_LINES, close="complete", check=_check_u)
    out = Path(db.DATA_DIR) / "results.json"
    previous = json.loads(out.read_text(encoding="utf-8")) if out.is_file() else {}
    # The send log lives only in `results.json`, which a previous regrade may already have rewritten - so the
    # ordering checks are rebuilt from the call's own `turn_evidence` ledger, which IS what went on the wire:
    # one row per caller turn, carrying the exact step token it was sent against.
    for evidence in db.list_turn_evidence(int(call_id)):
        key = evidence.snapshot_step.split(":followup")[0].split(":which_item")[0]
        if evidence.snapshot_step.endswith(":which_item"):
            key = f"{key}:which_item"
        elif ":followup:" in evidence.snapshot_step:
            key = f"{key}:fu{evidence.snapshot_step.rsplit(':', 1)[-1]}"
        persona.sent.append({"key": key, "mode": "text", "payload": evidence.transcript,
                             "step": evidence.snapshot_step, "epoch": 0})
        persona.counts[key] = persona.counts.get(key, 0) + 1
    result = Runner(persona, pack, "")._grade(int(call_id))
    out.write_text(json.dumps({**previous, "regraded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                               "passed": int(bool(result["ok"])), "total": 1, "calls": [result]},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{'PASS' if result['ok'] else 'FAIL'} regrade of call {call_id}\n{out}", flush=True)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    if "--regrade" in sys.argv:
        raise SystemExit(_regrade(int(sys.argv[sys.argv.index("--regrade") + 1])))
    raise SystemExit(asyncio.run(_main()))
