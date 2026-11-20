"""Two adversarial live calls, both on the rare-disease pack (`shishu-hi`). Run in `tmux`.

Boots `intake.app:app` on port 8794 with `DATA_DIR=intake/storage/adversarial_v2` (its own database, audio
directory and PDF folder), logs in over `/api/login`, then drives one scripted caller per call over
`ws://127.0.0.1:8794/ws/live?pack=<pack_id>`. Everything underneath is real: Gemini Live, the Decision API,
the profile extractor, SQLite, WeasyPrint, and the stereo WAV recorder.

Call 1 (`a1_shishu_revival_conflict_disconnect`, pack `shishu-hi`) attacks five mechanisms at once: a profile
give-up that must revive mid-call and rewind the cursor (`engine._revive_declined_slots` + `_reopens`), a
same-step tap-vs-speech contradiction that must roll back through `policy.arbitrate_tap_turn`, a scope
clarification that must not burn an attempt, a backtracking `OTHER_FORM` revision plus its spoken ack
(`live.py:594-602`), a mid-call WebSocket drop that must resume on the same call id without duplicating the
held prompt (`db.reopen_call`), and unrelated chatter on the closing question that must never become a concern.

Call 2 (`a2_shishu_triad_chaos`, pack `shishu-hi`) walks every active rare-disease gate on a 1-in-3 chaotic
rotation, all of it on the one pack this suite exists for:

  * Bucket A — a direct answer (`CLEAR`, `CONCERN` with child attribution + follow-up, `UNKNOWN`), plus the two
    mechanisms `a1` never reached on this pack: a Manual Nudge pressed over buffered caller speech
    (`live.py:427-432`, the `detail == "evaluated"` branch) and a `NAMED_CLARIFY` that must jump back to the
    `UNKNOWN` thalassemia gate and re-ask it without ever describing severe disease (`prompts.yaml:24`).
  * Bucket B — an out-of-turn `OTHER_FORM` revision of an *already settled* gate, in BOTH directions:
    `CLEAR -> CONCERN` (a settled denial taken back) and `CONCERN -> CLEAR` (`engine.py:340` rebuilds a bare
    `AnswerRecord`, so the old note, follow-up slots, `item_ids` and red-flag line must all disappear).
  * Bucket C — a clinically relevant remark that is NOT on the rare-disease form
    (`CLINICAL_ASIDE -> RETAIN_OBSERVATION`), retained for the doctor and kept off the mother's copy. Two of
    those gates then get non-medical chatter and must exhaust to `UNANSWERED` with the aside still retained.

`--selftest` proves every critical assertion is non-vacuous offline: it feeds synthetic broken states into the
same `_check_a1` / `_check_a2` the live run uses and requires each invariant to fail when broken and pass when
whole. Pass call ids as arguments to run a subset. Exits 0 only when every selected call passes every check.
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
os.environ["DATA_DIR"] = str(ROOT / "intake" / "storage" / "adversarial_v2")

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from pypdf import PdfReader  # noqa: E402

from intake import db, engine, reports  # noqa: E402
from intake.models import (AnswerRecord, AnswerStatus, CallState, CloseReason, FollowupSlot, NoteText,  # noqa: E402
                           Phase, ProfilePatch, RosterInput, SubStep, TurnCommand, TurnOutcome)
from intake.pack import load_pack  # noqa: E402
from intake.turns import EvidenceSource, StoredTurnEvidence  # noqa: E402

HOST, PORT = "127.0.0.1", 8794
BASE = f"http://{HOST}:{PORT}"
CALL_DEADLINE_SEC, RECV_TIMEOUT_SEC, RESUME_WAIT_SEC = 1800.0, 300.0, 1.5
DROP_QID = "fam_maternal_bleeding"

_LEAKS = ("next_question", "system cue", "ask only this", "active_qid", "last_answered_qid", "recorded_concerns",
          "duchenne", "hemophilia", "pompe", "gaucher", "scid", "wiskott", "ड्यूशेन", "हीमोफीलिया", "पॉम्पे",
          "गॉशर", "इम्यूनोडेफिशिएंसी", "म्यूकोपॉलीसेकेराइडोसिस")
# WeasyPrint + pypdf split Devanagari conjuncts, so the PDF never yields "हीमोफीलिया" as one token even when it
# is printed. A full-word PDF check would pass on a broken redaction; these fragments are what actually leaks.
_DISEASE = ("हीमोफीलिया", "हीमोफ", "मोफी", "फीलिया")
# The page-two glossary is in the PDF, but pypdf drops and reorders matras from the embedded font ("जवाब का
# मतलब" reads back as "जविाब का मतलब"), so each row must survive in *some* form.
_GLOSSARY_PDF = (("जवाब", "जविाब"), ("इनकर", "इनकार"), ("मलब", "मतलब"), ("जवब नहीं", "जवाब नहीं"))
_CHILD_GATE_NA_ALLOWED = frozenset({"mod_heavy_periods", "body_puberty", "fam_same_problem", "ever_pregnant"})
C, CL, U, UA, ANS = (AnswerStatus.CONCERN, AnswerStatus.CLEAR, AnswerStatus.UNKNOWN, AnswerStatus.UNANSWERED,
                     AnswerStatus.ANSWERED)
_ALL_STATES = frozenset({C, CL, U, UA, ANS})


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

    `answer()` is only consumed by the live loop; the offline self-test builds a Persona and fills `counts` /
    `sent` by hand so the exact same check functions can be graded against a synthetic call.
    """

    def __init__(self, pid: str, pack_id: str, lines: dict[str, tuple], *, close: str, check, drop_on_qid: str = "",
                 default_reply=None, name_in_call: str = "") -> None:
        self.pid, self.pack_id, self.lines = pid, pack_id, lines
        self.close, self.check, self.default_reply, self.name_in_call = close, check, default_reply, name_in_call
        self.drop_on_qid = drop_on_qid
        self.counts: dict[str, int] = {}
        self.sent: list[dict[str, Any]] = []
        self.defaults: list[str] = []
        self.frames: dict[str, int] = {}
        self.gaps: list[float] = []          # ms between the text frame and the nudge frame of a `text_nudge`
        self.nudge_details: list[str] = []
        self.epoch = 0
        self.call_ids: list[int] = []
        self.dropped = False
        self.drop_prompt_chars = 0
        self.telemetry_before_drop: dict[str, Any] = {}
        self.telemetry_after_resume: dict[str, Any] = {}
        self._answered: set[tuple[str, str]] = set()

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

        A re-published step (a clarify, an attempt bump, a tap-conflict rollback, a delayed Decision reply)
        keeps its qid but may keep or change its token. Re-answering a token the caller has already answered is
        a duplicate turn that would burn a real attempt; a scripted second line on the same token is the one case
        where a repeat is intended.
        """
        key = self.route(state)
        return (key, state.step_token) in self._answered and self.counts.get(key, 0) >= self._cap(key)

    def answer(self, state: CallState) -> tuple[str, str] | None:
        if state.phase is not Phase.ASKING:
            return None
        key = self.route(state)
        if self.counts.get(key, 0) > 4:
            raise RuntimeError(f"stuck on {key}")
        picked = self.lines.get(key)
        if picked is None:
            if self.default_reply is not None:
                self.defaults.append(key)
                return self.default_reply(state)
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
    # The stored ledger, so a check can re-derive `noted_observation` from what the live run really persisted.
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

    def assistant_rows(self, qid: str) -> list[dict[str, Any]]:
        return [t for t in (self.call.get("transcript") or [])
                if t.get("role") == "assistant" and (t.get("qid") or "") == qid]


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


def _replay_ack(ev: Ev, qid: str) -> bool | None:
    """Did the stored interpretation for `qid`'s turn carry `noted_observation` into the tool body?

    `live.py:594-602` decides this from `stayed` + `DecisionReason.OTHER_FORM`, and `tool_body` is what Gemini
    receives. Re-deriving it from the persisted interpretation is the only honest way to grade it: the model's
    own sentence is not something the server can assert.
    """
    from intake import live, policy
    from intake.turns import DecisionReason, Disposition, TurnEvidence

    for seq, step, transcript in ev.sequences:
        if step.split(":")[0] != qid or seq not in ev.interps:
            continue
        # Rebuild the mid-call moment: the call is still ASKING, the cursor sits on the asked question, and that
        # question itself is still unanswered (it is answered by the very turn being replayed).
        open_answers = {k: v for k, v in ev.state.answers.items() if k != qid}
        snapshot = ev.state.model_copy(update={
            "phase": Phase.ASKING, "close_reason": None, "answers": open_answers,
            "cursor": ev.state.cursor.model_copy(
                update={"qid": qid, "substep": SubStep.MAIN, "followup_idx": 0, "attempts": 0, "hold_slot": None})})
        evidence = TurnEvidence(source=EvidenceSource.TEXT, snapshot_step=step, transcript=transcript, received_at=0.0)
        nxt, pol = policy.apply_policy(ev.pack, snapshot, evidence, ev.interps[seq])
        answered = snapshot.cursor
        stayed = (nxt.phase is Phase.ASKING and nxt.cursor.qid == answered.qid
                  and nxt.cursor.substep is answered.substep and nxt.cursor.followup_idx == answered.followup_idx)
        ack = bool(stayed and any(d.disposition is Disposition.RETAIN_OBSERVATION or d.reason is DecisionReason.OTHER_FORM
                                  for d in pol.decisions))
        noted = " ".join((evidence.transcript or pol.command.utterance or "").split()) if ack else ""
        if live.tool_body(ev.pack, nxt, clarify=ack, noted_observation=noted).get("noted_observation"):
            return True
    return False


def _status(ev: Ev, cl: Checklist, *wanted: tuple[str, AnswerStatus]) -> None:
    for qid, want in wanted:
        got = _answered(ev, qid)[0]
        cl.check(f"{qid} == {want}", got is want, f"got {got}")


def _na_qids(ev: Ev) -> list[str]:
    return sorted(q for q, r in ev.state.answers.items() if r.state is AnswerStatus.NOT_APPLICABLE)


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
    # `auto_close` is only ever emitted by the watchdog (silence / time limit), so any auto_close means the
    # harness stalled and a reader would be grading a timeout instead of the content.
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
# Call 1 - shishu-hi: revival, tap conflict, clarify, revision, disconnect, chatter
# ------------------------------------------------------------------------------------------------------------
def _check_a1(ev: Ev, cl: Checklist) -> dict[str, Any]:
    p, st = ev.persona, ev.state
    asks = [s["step"] for s in p.sent_on("intro_children")]
    # 1A - the give-up, then the mid-call revival that deletes it and rewinds the cursor.
    cl.check("a1a give-up recorded then revived (3 asks, roster landed)",
             p.counts.get("intro_children", 0) == 3 and len(st.roster) == 2,
             f"asks={p.counts.get('intro_children')} roster={len(st.roster)}")
    cl.check("a1a revival re-asked the ages with hold:roster",
             len(asks) == 3 and "hold:roster" in asks[2] and "try:" not in asks[2], f"steps={asks}")
    cl.check("a1a roster resolved to two 7-year-olds",
             sorted(c.age_years for c in st.roster) == [7.0, 7.0] and st.roster_incomplete is False
             and {c.tag for c in st.roster} == {"male", "female"},
             f"roster={[(c.id, c.age_years, c.tag) for c in st.roster]} incomplete={st.roster_incomplete}")
    cl.check("a1a ever_pregnant pruned by has_children",
             _answered(ev, "ever_pregnant")[0] is AnswerStatus.NOT_APPLICABLE, f"got {_answered(ev, 'ever_pregnant')[0]}")
    # With a resolved roster no child gate can be NOT_APPLICABLE for a roster reason any more: `_resync` must have
    # cleared that state, and the ones still NA must be the age/family gates that really do not apply.
    cl.check("a1a no child gate left NOT_APPLICABLE", not (set(_na_qids(ev)) - _CHILD_GATE_NA_ALLOWED),
             f"NA={_na_qids(ev)}")
    # 1B - same-step tap vs speech: the tap says नहीं, the words say हाँ.
    cl.check("a1b SOURCE_CONFLICT on birth_nicu", ev.has("conflict", "source_conflict", "birth_nicu"),
             f"decisions={ev.reasons()}")
    cl.check("a1b tap evidence stored for birth_nicu",
             ("tap", "birth_nicu") in ev.sources and ("text", "birth_nicu") in ev.sources, f"sources={ev.sources}")
    nicu_state, nicu = _answered(ev, "birth_nicu")
    cl.check("a1b birth_nicu re-answered CONCERN with the NICU note",
             nicu_state is C and _has(nicu, "काँच", "कांच", "साँस", "सांस", "तकलीफ"), f"state={nicu_state} note={nicu[:80]}")
    # 1C - a scope question is a clarification, never a missed answer.
    fits = p.sent_on("fam_fits_delay")
    cl.check("a1c clarify stayed on fam_fits_delay without an attempt",
             len(fits) >= 2 and "try:" not in str(fits[1]["step"]), f"steps={[s['step'] for s in fits]}")
    cl.check("a1c clarify decision recorded on fam_fits_delay", ev.has("clarify", "", "fam_fits_delay"),
             f"decisions={ev.reasons()}")
    # 1D - a CLEAR taken back out of turn, and the spoken acknowledgement that owes the caller a re-ask.
    cl.check("a1d birth_cord_bleed revised CLEAR -> CONCERN", _answered(ev, "birth_cord_bleed")[0] is C,
             f"got {_answered(ev, 'birth_cord_bleed')[0]}")
    cl.check("a1d OTHER_FORM decision recorded", ev.has("accept_form", "other_form", "birth_cord_bleed"),
             f"decisions={ev.reasons()}")
    body = p.sent_on("body_fits")
    # A backtrack onto a *settled* question is a revision, not a miss: `reduce_turn` keeps the exact same cursor
    # (same step token, no `try:` bump) so no attempt is burned, and `live.py:594-602` passes the words back to
    # the model as `noted_observation` because `stayed` and `DecisionReason.OTHER_FORM` are both true.
    cl.check("a1d body_fits re-asked on the same token without burning an attempt, then CLEAR",
             len(body) >= 2 and body[1]["step"] == body[0]["step"] and "try:" not in str(body[1]["step"])
             and _answered(ev, "body_fits")[0] is CL,
             f"steps={[s['step'] for s in body]} got={_answered(ev, 'body_fits')[0]}")
    # `live.py:594-602`: `stayed` plus an OTHER_FORM decision is the exact pair that puts the caller's own words in
    # `noted_observation`, and the model is told to open with "मैंने यह नोट कर लिया है". Replaying the stored
    # interpretation through the real policy proves the body carried it; the caption is the model's, so a missing
    # row is recorded rather than failed (a recovered turn writes no assistant caption at all).
    ack_body = _replay_ack(ev, "body_fits")
    cl.check("a1d the revision's words were handed to the model as noted_observation", ack_body is True,
             "replaying the stored interpretation produced no noted_observation")
    # 1E - the drop and the resume.
    cl.check("a1e same call_id after resume",
             len(p.call_ids) == 2 and p.call_ids[0] == p.call_ids[1] and p.call_ids[0] > 0, f"call_ids={p.call_ids}")
    cl.check("a1e session_resumed event recorded", bool(ev.events_of("session_resumed")),
             f"event types={sorted({str(e.get('type')) for e in ev.events})}")
    rows = ev.assistant_rows(DROP_QID)
    # The held prompt's own length, plus room for one real caption: a prompt that was re-appended instead of
    # dropped reads back as roughly twice this. `db.append_transcript` merges an identical role+qid row, so a
    # duplicate hides as one longer row, not as two.
    limit = int(p.drop_prompt_chars * 1.35) + 30
    cl.check("a1e resume did not duplicate the held prompt",
             len(rows) <= 1 and (not rows or len(_norm(rows[0].get("text"))) <= limit),
             f"rows={[(len(_norm(r.get('text'))), _norm(r.get('text'))[:60]) for r in rows]} limit={limit}")
    before, after = p.telemetry_before_drop, p.telemetry_after_resume
    cl.check("a1e telemetry survived the drop",
             int(after.get("user_turns") or 0) >= int(before.get("user_turns") or 0) > 0
             and int(after.get("bot_turns") or 0) >= int(before.get("bot_turns") or 0) > 0,
             f"before user/bot={before.get('user_turns')}/{before.get('bot_turns')} "
             f"after={after.get('user_turns')}/{after.get('bot_turns')}")
    # 1F - Law 1: non-medical chatter is not a finding.
    close_state = _answered(ev, "close_anything_else")[0]
    cl.check("a1f chatter never became a CONCERN on close_anything_else", close_state in (CL, UA, None), f"got {close_state}")
    yes_items = list((ev.proj.yes_items if ev.proj else ()) or ())
    red_flags = [str(x) for x in (ev.call.get("red_flags") or [])]
    cl.check("a1f गर्मी never reached yes_items or red_flags",
             not any("गर्मी" in str(x) for x in [*yes_items, *red_flags]), f"yes={yes_items} flags={red_flags}")
    cl.check("a1f गर्मी never reached the patient PDF", "गर्मी" not in ev.pdf_text, "patient PDF leaked the chatter")
    _status(ev, cl, ("birth_bcg", U))
    states = {r.state for r in st.answers.values()}
    cl.check("all five answer states coexist", _ALL_STATES <= states, f"missing={sorted(_ALL_STATES - states)}")
    # The maternal diagnosis: spoken name kept for the doctor, redacted for the mother.
    doctor, patient = str(ev.call.get("summary_doctor") or ""), str(ev.call.get("summary_patient") or "")
    cl.check("doctor summary keeps हीमोफीलिया", "हीमोफीलिया" in doctor, "doctor summary lost the reported diagnosis")
    cl.check("patient summary redacts हीमोफीलिया", bool(patient.strip()) and "हीमोफीलिया" not in patient,
             "patient summary is empty or leaked हीमोफीलिया")
    leaked = [frag for frag in _DISEASE if frag in ev.pdf_text]
    cl.check("PDF redacts हीमोफीलिया", bool(ev.pdf_text.strip()) and not leaked, f"PDF leaked fragments {leaked}")
    return {"call_ids": p.call_ids, "asks": p.counts, "roster": [(c.id, c.age_years, c.tag) for c in st.roster],
            "na": _na_qids(ev), "default_replies": p.defaults, "sends": p.sent}


# ------------------------------------------------------------------------------------------------------------
# Call 2 - shishu-hi: the 1-in-3 triad across every active rare-disease gate
# ------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Line:
    """One scripted turn: the Hindi she says (or the tap she sends) and the command it must derive.

    Keeping both in one row is the only way the live persona and the offline replay can be the same schedule:
    the live run sends `text` and lets Gemini + `policy` derive the command, the self-test replays `cmd` through
    the real `policy.apply_policy` + `engine.reduce_turn` and requires the identical end state.
    """
    text: str
    cmd: TurnCommand
    mode: str = "text"
    aside: bool = False   # a genuine clinical remark that is not on this form -> RETAIN_OBSERVATION
    child: str = ""       # for a which_item step, the sex of the child she names; `item_ref` is resolved live
    detail: bool = False  # the sentence already carries the follow-up's detail (`detail_present`)


def _say(text: str, outcome: TurnOutcome, *, detail: bool = False, **kw) -> Line:
    cmd = TurnCommand(outcome=outcome, utterance=text, followup_answered=detail, **kw)
    return Line(text, cmd, detail=detail)


def _aside(text: str) -> Line:
    """A clinically real remark outside the rare-disease form: the gate stays open, the remark is retained."""
    return Line(text, TurnCommand(outcome=TurnOutcome.NOT_ABOUT_ACTIVE, utterance=text), aside=True)


def _flip(text: str, target: str, to: TurnOutcome) -> Line:
    """An out-of-turn `OTHER_FORM` revision: nothing about the active gate, one settled gate written again."""
    return Line(text, TurnCommand(outcome=TurnOutcome.NOT_ABOUT_ACTIVE, stated={target: to}, utterance=text))


def _tap(option: str, label: str) -> Line:
    """A tap: `option` is the quick-reply VALUE on the wire (`clear`/`concern`), `label` is what she sees."""
    return Line(option, TurnCommand(outcome=TurnOutcome.CLEAR, utterance=label), mode="quick")


def _pick_child(text: str, tag: str) -> Line:
    return Line(text, TurnCommand(outcome=TurnOutcome.ANSWERED, utterance=text), child=tag)


def _child(st: CallState, tag: str) -> str:
    return next((c.id for c in st.roster if c.tag == tag), "")


# The first line on each of these gates is a genuine clinical remark that is NOT on the rare-disease form:
# `residual_clinical` fires, `policy` records RETAIN_OBSERVATION, and the gate's own attempt is spent un-answered.
A2_ASIDE_GATES = frozenset({
    "body_proportion", "fam_lumps", "body_thumb", "gate_bones_fractures", "gate_urine_kidney", "body_puberty",
    "ill_iv_antibiotics", "preg_conditions", "preg_scan", "preg_transfusion", "fam_miscarriage", "fam_infant_death",
    "fam_transfusion", "fam_bleeding", "fam_blind_deaf", "fam_bp_stroke", "fam_maternal_infections",
    "fam_maternal_eczema", "fam_same_problem",
})
# (the gate whose attempt carries the revision, the settled gate it takes back, the direction it takes it)
A2_FLIP_UP = (("birth_sudden", "birth_nicu"), ("birth_hearing", "birth_cord_3weeks"), ("preg_heart", "body_fits"),
              ("body_tummy", "body_limb_pain"), ("ill_odor", "ill_fasting"), ("fam_heart_kidney", "fam_cancer"),
              ("body_growth", "birth_cord_bleed"))
A2_FLIP_DOWN = (("birth_bcg", "birth_jaundice"), ("body_milestones", "birth_term"),
                ("body_bleeding_bruise", "body_blue"), ("body_lumps_birth", "body_transfusion"))
A2_FLIP_SOURCES = tuple(dict.fromkeys(src for src, _ in (*A2_FLIP_UP, *A2_FLIP_DOWN)))


# Every active shishu-hi gate for the 12-year-old daughter + 4-year-old son roster, in the order the engine
# really walks them, keyed by the cursor route `Persona.route` produces.
N = TurnOutcome.NOT_ABOUT_ACTIVE
A2_LINES: dict[str, list[Line]] = {
    "intro_name": [Line("मेरा नाम मीना है।", TurnCommand(profile_patch=ProfilePatch(slots={"caller_name": "मीना"}),
                                                       profile_turn_attempted=True, utterance="मेरा नाम मीना है।"))],
    "intro_married": [Line("हाँ, शादी हो गई है।", TurnCommand(outcome=TurnOutcome.ANSWERED, value="married",
                                                            utterance="हाँ, शादी हो गई है।"))],
    "intro_children": [Line("मेरे दो बच्चे हैं — एक 12 साल की बेटी और एक 4 साल का बेटा।", TurnCommand(
        profile_patch=ProfilePatch(roster=(RosterInput(age_years=12.0, tag="female"), RosterInput(age_years=4.0, tag="male")),
                                   roster_complete=True),
        profile_turn_attempted=True, utterance="मेरे दो बच्चे हैं - एक 12 साल की बेटी और एक 4 साल का बेटा।"))],
    "intro_bridge": [_say("जी, पूछिए।", TurnOutcome.CLEAR)],
    # --- Bucket A: direct answers, the buffered nudge, and the named-clarify jump-back -----------------------
    # `birth_mode_detail` is the buffered-speech nudge: the words are banked and `nudge` lands in the next frame.
    "birth_mode_detail": [Line("हाँ, बड़ा ऑपरेशन सिजेरियन हुआ था क्योंकि बच्चा उल्टा था।",
                               TurnCommand(outcome=TurnOutcome.CONCERN,
                                           utterance="हाँ, बड़ा ऑपरेशन सिजेरियन हुआ था क्योंकि बच्चा उल्टा था।"),
                               mode="text_nudge")],
    "birth_mode_detail:which_item": [_pick_child("बेटी के समय।", "female")],
    "birth_mode_detail:fu0": [_say("बच्चा उल्टा था, इसलिए ऑपरेशन करना पड़ा।", TurnOutcome.CONCERN)],
    "birth_term": [_say("हाँ, आठवें महीने में ही पैदा हो गया था।", TurnOutcome.CONCERN)],
    "birth_term:which_item": [_pick_child("बेटे का जन्म।", "male")],
    "birth_term:fu0": [_say("एक महीना पहले।", TurnOutcome.CONCERN)],
    "birth_nicu": [_say("नहीं।", TurnOutcome.CLEAR)],
    "birth_jaundice": [_say("हाँ, जन्म के बाद पीलिया बताया था।", TurnOutcome.CONCERN)],
    "birth_jaundice:which_item": [_pick_child("बेटे को।", "male")],
    "birth_jaundice:fu0": [_say("छह दिन फोटोथेरेपी में रखा था।", TurnOutcome.CONCERN)],
    "birth_cord_3weeks": [_say("नहीं।", TurnOutcome.CLEAR)],
    "birth_cord_bleed": [_say("नहीं, नाल गिरने पर खून नहीं आया।", TurnOutcome.CLEAR)],
    "birth_cleft": [_say("नहीं।", TurnOutcome.CLEAR)],
    "birth_vision": [_say("नहीं।", TurnOutcome.CLEAR)],
    "body_fits": [_say("नहीं।", TurnOutcome.CLEAR)],
    # CONCERN early, so direction 2 has a settled CONCERN *with a child* to take back later.
    "body_blue": [_say("हाँ, बेटे के होंठ कभी-कभी नीले पड़ जाते हैं।", TurnOutcome.CONCERN)],
    "body_blue:which_item": [_pick_child("बेटे को।", "male")],
    "body_regression": [_say("नहीं।", TurnOutcome.CLEAR)],
    "body_bleeding_bruise": [_flip("अरे नहीं, नीले होंठ की बात मैंने गलत कह दी थी — बेटे के होंठ कभी नीले नहीं पड़े।",
                                   "body_blue", TurnOutcome.CLEAR),
                             _say("नहीं, खून देर तक नहीं बहता।", TurnOutcome.CLEAR)],
    "body_joint_swelling": [_say("नहीं।", TurnOutcome.CLEAR)],
    "mod_tooth_bleed": [_say("नहीं।", TurnOutcome.CLEAR)],
    "mod_heavy_periods": [_say("नहीं।", TurnOutcome.CLEAR)],
    "body_transfusion": [_say("हाँ, बेटे को एक बार खून चढ़ा था।", TurnOutcome.CONCERN)],
    "body_transfusion:which_item": [_pick_child("बेटे को।", "male")],
    "body_transfusion:fu0": [_say("एक बार, छह महीने पहले।", TurnOutcome.CONCERN)],
    "body_blood_count": [_say("नहीं।", TurnOutcome.CLEAR)],
    "body_limb_pain": [_say("नहीं।", TurnOutcome.CLEAR)],
    "body_lumps_birth": [_flip("और खून चढ़ाने वाली बात भी गलत थी — बेटे को कभी खून नहीं चढ़ा।",
                               "body_transfusion", TurnOutcome.CLEAR),
                         _say("नहीं, जन्म से कोई गाँठ नहीं थी।", TurnOutcome.CLEAR)],
    "body_growth": [_flip("एक बात पहले रह गई थी — बेटे की नाल गिरने पर खून भी निकला था।",
                          "birth_cord_bleed", TurnOutcome.CONCERN),
                    _say("नहीं, कद और वज़न ठीक बढ़ रहा है।", TurnOutcome.CLEAR)],
    "ill_fasting": [_say("नहीं।", TurnOutcome.CLEAR)],
    "ill_admissions": [_say("नहीं, कभी भर्ती नहीं करना पड़ा।", TurnOutcome.CLEAR)],
    "body_private_parts": [_say("पता नहीं, मुझे याद नहीं।", TurnOutcome.UNKNOWN)],
    "fam_cancer": [_say("नहीं।", TurnOutcome.CLEAR)],
    # UNKNOWN on the thalassemia carrier check, then the named clarify from `fam_related` jumps back to it.
    "preg_thalassemia": [_say("पता नहीं, मुझे याद नहीं।", TurnOutcome.UNKNOWN),
                         _say("अच्छा समझ गई, नहीं, ऐसी जाँच में कोई दिक्कत नहीं निकली थी।", TurnOutcome.CLEAR)],
    "fam_related": [_say("थैलेसीमिया वाली जाँच का मतलब क्या होता है? पहले वो समझाइए।", TurnOutcome.CLARIFY,
                         stated={"preg_thalassemia": TurnOutcome.CLARIFY}),
                    _say("नहीं, हम आपस में रिश्तेदार नहीं हैं।", TurnOutcome.CLEAR)],
    # A follow-up aside: `3069305`'s dedup target on this pack.
    "fam_fits_delay": [_say("हाँ, परिवार में दिक्कत थी।", TurnOutcome.CONCERN)],
    "fam_fits_delay:fu0": [_say("मेरे नाना जी को बुढ़ापे में भूलने की अल्जाइमर बीमारी थी।", TurnOutcome.CONCERN)],
    # Maternal X-linked bleeding with two rare disease names that must never reach her copy or the PDF.
    # The disease names must reach the doctor's verbatim and never her copy. `redacted` is the interpreter's own
    # `redact_patient_note` read of the sentence; the follow-up is asked separately, as a live caller would be.
    "fam_maternal_bleeding": [_say("हाँ, मेरे सगे मामा के लड़के को डॉक्टर ने ड्यूशेन और हीमोफीलिया का शक बताया था।",
                                   TurnOutcome.CONCERN, detail=True, redacted=True)],
    "close_anything_else": [_say("अच्छा दीदी, आज मौसम काफ़ी ठंडा है।", N), _tap("clear", "नहीं।")],

    # --- Bucket B: out-of-turn revisions of a SETTLED gate, then the active gate answered -------------------
    # Direction 1 - a settled CLEAR taken back to CONCERN. Two of these flip a gate whose own `which_item` and
    # follow-up steps already ran, so the revision must also restore the child and the detail it lost.
    # The revision sentence must talk ONLY about the settled gate: any phrasing that could also read as a "yes"
    # to the active question lets the interpreter settle both, and then `stayed` is false so no ack is owed.
    "birth_sudden": [_flip("मुझे अभी याद आया, बेटे को जन्म के बाद तीन दिन काँच के बॉक्स में रखा था।",
                           "birth_nicu", TurnOutcome.CONCERN),
                     _say("नहीं, अचानक बीमार पड़कर इमरजेंसी नहीं ले जाना पड़ा।", TurnOutcome.CLEAR)],
    # Safety net: if the active gate is settled in the revision's own breath, its follow-up is asked next.
    "birth_sudden:fu0": [_say("नहीं, ऐसा कुछ नहीं हुआ था।", TurnOutcome.CLEAR)],
    "fam_heart_kidney:fu0": [_say("नहीं, किसी को दिल या गुर्दे की बीमारी नहीं है।", TurnOutcome.CLEAR)],
    "birth_hearing": [_flip("मुझे अभी याद आया, बेटे की नाल गिरने में एक महीने से ज़्यादा लग गया था।",
                            "birth_cord_3weeks", TurnOutcome.CONCERN),
                      _say("नहीं, सुनने में कोई दिक्कत नहीं है।", TurnOutcome.CLEAR)],
    "preg_heart": [_flip("माफ़ कीजिए, पहले मैंने मना कर दिया था, पर बेटे को बुखार में दो बार झटके यानी दौरे पड़े थे।",
                         "body_fits", TurnOutcome.CONCERN),
                   _say("नहीं, स्कैन में या जन्म पर दिल में कोई खराबी नहीं बताई गई थी।", TurnOutcome.CLEAR)],
    "body_tummy": [_flip("अरे हाँ, पहले ध्यान नहीं रहा — बेटी के हाथ-पैर में बहुत तेज़ दर्द रहता है!",
                         "body_limb_pain", TurnOutcome.CONCERN),
                   _say("नहीं, पेट नहीं बढ़ रहा।", TurnOutcome.CLEAR)],
    "ill_odor": [_flip("वैसे मैंने पहले नहीं बोला था, पर सुबह भूखा रहने पर बेटे को चक्कर और सुस्ती हो जाती है!",
                       "ill_fasting", TurnOutcome.CONCERN),
                 _say("नहीं, पेशाब या पॉटी में अजीब बदबू नहीं आती।", TurnOutcome.CLEAR)],
    "fam_heart_kidney": [_flip("मुझे अभी याद आया, मेरी बुआ को स्तन का कैंसर हुआ था, पहले मैंने नहीं बोला था!",
                               "fam_cancer", TurnOutcome.CONCERN),
                         _say("नहीं, परिवार में दिल या गुर्दे की गंभीर बीमारी नहीं है।", TurnOutcome.CLEAR)],
    # Direction 2 - a settled CONCERN taken back to CLEAR. `engine.py:340` rebuilds a bare record, so the child,
    # the note and the follow-up slot the first answer earned must all disappear with it.
    "birth_bcg": [_flip("अरे माफ़ करना दीदी, पीलिया मेरे बच्चे को नहीं, पड़ोसी के बच्चे को हुआ था — मेरे दोनों बच्चों को पीलिया बिल्कुल नहीं हुआ था।",
                        "birth_jaundice", TurnOutcome.CLEAR),
                  _say("नहीं, टीके की जगह पर कोई फोड़ा या गाँठ नहीं हुई थी।", TurnOutcome.CLEAR)],
    "body_milestones": [_flip("रुकिए, मैंने पहले गलती से बोल दिया कि बेटा आठवें महीने में हुआ था — असल में दोनों बच्चे पूरे नौ महीने पर ही पैदा हुए थे!",
                              "birth_term", TurnOutcome.CLEAR),
                        _say("नहीं, दोनों बच्चों ने समय पर बैठना-चलना-बोलना सीखा।", TurnOutcome.CLEAR)],
    "preg_delivery": [_say("नहीं, किसी डिलीवरी में कोई बड़ी दिक्कत नहीं आई थी।", TurnOutcome.CLEAR)],

    # --- Bucket C: a genuine clinical remark that is NOT on the rare-disease form --------------------------
    # Two of these gates then get non-medical chatter and must exhaust to UNANSWERED with the aside still kept.
    "body_proportion": [_aside("बेटी को बस में सफ़र करते समय उल्टी जैसा जी मिचलाता है।"),
                        _say("अभी रसोई में दूध उबल रहा है।", N)],
    "fam_lumps": [_aside("बेटे के सिर में आजकल डैंड्रफ और खुजली हो गई है।"),
                  _say("बाहर बारिश हो रही है।", N)],
    "body_thumb": [_aside("बेटी के पैर पर पिछले हफ़्ते मच्छर काटने से लाल दाने हो गए थे।"),
                   _say("नहीं, उंगली या अंगूठा पूरा है।", TurnOutcome.CLEAR)],
    "gate_bones_fractures": [_aside("बेटे को चॉकलेट खाने से एक दूध के दाँत में कीड़ा लग गया है, डॉक्टर भरने को बोले हैं।"),
                             _say("नहीं, मामूली गिरने पर हड्डी नहीं टूटी।", TurnOutcome.CLEAR)],
    "gate_urine_kidney": [_aside("मुझे खुद कल से गर्दन की नस में अकड़न लग रही है।"),
                          _say("नहीं, बहुत ज़्यादा पेशाब या प्यास की दिक्कत नहीं है।", TurnOutcome.CLEAR)],
    "body_puberty": [_aside("बेटे को नहाने के बाद बाएँ कान में खुजली होती है।"),
                     _say("नहीं, उम्र के बदलाव सामान्य समय पर आए हैं।", TurnOutcome.CLEAR)],
    "ill_iv_antibiotics": [_aside("बेटी को धूल से छींकें आती हैं।"),
                           _say("नहीं, नस में दवा कभी नहीं लगी।", TurnOutcome.CLEAR)],
    "preg_conditions": [_aside("बेटे के नाखून थोड़े जल्दी टूट जाते हैं।"),
                        _say("नहीं, किसी प्रेग्नेंसी में बीपी-शुगर-थायरॉइड नहीं रहा।", TurnOutcome.CLEAR)],
    "preg_scan": [_aside("बेटी को दूध पीने के बाद कभी-कभी गैस बनती है।"),
                  _say("नहीं, स्कैन में कोई खराबी नहीं बताई गई थी।", TurnOutcome.CLEAR)],
    "preg_transfusion": [_aside("बेटे ने कल खेलते समय घुटना छिल लिया था, पट्टी लगा दी थी।"),
                         _say("नहीं, प्रेग्नेंसी में खून नहीं चढ़ा।", TurnOutcome.CLEAR)],
    "fam_miscarriage": [_aside("सर्दी में बच्चों की नाक बहने लगती है।"),
                        _say("नहीं, कभी गर्भ नहीं गिरा।", TurnOutcome.CLEAR)],
    "fam_infant_death": [_aside("बेटे को दूध के बाद कभी-कभी कब्ज़ रह जाती है।"),
                         _say("नहीं, परिवार में किसी छोटे बच्चे की मौत नहीं हुई।", TurnOutcome.CLEAR)],
    "fam_transfusion": [_aside("बेटी को रात में कभी-कभी सूखी खाँसी आती है।"),
                        _say("नहीं, परिवार में किसी को बार-बार खून नहीं चढ़ता।", TurnOutcome.CLEAR)],
    "fam_bleeding": [_aside("बेटे को नई डायपर से हल्की लाली हो जाती है।"),
                     _say("नहीं, परिवार में किसी को चोट पर ज़्यादा खून नहीं बहता।", TurnOutcome.CLEAR)],
    "fam_blind_deaf": [_aside("बेटी को धूप में खेलने से हल्की लाली आ जाती है।"),
                       _say("नहीं, परिवार में किसी को अंधेपन या बहरेपन की दिक्कत नहीं।", TurnOutcome.CLEAR)],
    "fam_bp_stroke": [_aside("बेटे को खेलने के बाद पैर में हल्की अकड़न हो जाती है।"),
                      _say("नहीं, परिवार में किसी को कम उम्र में बीपी-शुगर-लकवा नहीं हुआ।", TurnOutcome.CLEAR)],
    "fam_maternal_infections": [_aside("बेटी को पिछले हफ़्ते स्कूल में धूल भरी हवा से खाँसी शुरू हुई थी।"),
                                _say("नहीं, मायके की तरफ बार-बार गंभीर इंफेक्शन नहीं है।", TurnOutcome.CLEAR)],
    "fam_maternal_eczema": [_aside("बच्चों ने नया साबुन लगाया तो हाथ में हल्की लाली आ गई थी।"),
                            _say("नहीं, मायके की तरफ सूखे खुजली वाले धब्बे नहीं हैं।", TurnOutcome.CLEAR)],
    "fam_same_problem": [_aside("बेटे को ठंड लगने पर हल्की खाँसी आ जाती है।"),
                         _say("नहीं, परिवार में किसी और को बच्चों जैसी दिक्कत नहीं है।", TurnOutcome.CLEAR)],
}
# Bucket C is the retained-observation set: one distinct unlisted clinical remark per gate above.
A2_ASIDES = ("उल्टी जैसा जी मिचलाता", "डैंड्रफ", "मच्छर काटने", "दाँत में कीड़ा", "गर्दन की नस में अकड़न",
             "कान में खुजली", "धूल से छींकें", "नाखून", "गैस बनती", "घुटना छिल लिया", "नाक बहने लगती",
             "कब्ज़", "सूखी खाँसी", "डायपर से हल्की लाली", "धूप में खेलने से हल्की लाली", "पैर में हल्की अकड़न",
             "धूल भरी हवा से खाँसी", "नया साबुन लगाया", "ठंड लगने पर हल्की खाँसी")
# Direction 1 (`CLEAR -> CONCERN`) and direction 2 (`CONCERN -> CLEAR`) revision targets, by the gate that
# carries the revision. Both directions on the SAME settled gate (`birth_jaundice`, `birth_term`,
# `birth_mode_detail`) would contradict each other, so the down-flips own those three.
A2_UP = ("birth_nicu", "birth_cord_3weeks", "birth_cord_bleed", "body_fits", "body_limb_pain", "ill_fasting", "fam_cancer")
A2_DOWN = ("birth_jaundice", "birth_term", "body_blue", "body_transfusion")


def _a2_script() -> dict[str, tuple]:
    """The `Line` table as the live harness's own routing-keyed replies, one queue per cursor route."""
    out: dict[str, tuple] = {}
    for key, lines in A2_LINES.items():
        if len(lines) == 1:
            only = lines[0]
            out[key] = (only.mode, only.text) if only.mode != "text" else _line(only.text)
        else:
            out[key] = _steps(*((line.mode, line.text) if line.mode != "text" else _line(line.text) for line in lines))
    return out


def _check_a2(ev: Ev, cl: Checklist) -> dict[str, Any]:
    """Every mechanism the triad owes, graded from the call's own row, ledger and PDF."""
    p, st = ev.persona, ev.state
    doctor, patient = str(ev.call.get("summary_doctor") or ""), str(ev.call.get("summary_patient") or "")
    daughter, son = _child(st, "female"), _child(st, "male")

    # --- the walk itself ---------------------------------------------------------------------------------
    open_gates = [q.id for q in ev.pack.questions if q.is_routable and engine._is_unsatisfied(ev.pack, st, q)]
    # Only the two gates this roster really closes may be NOT_APPLICABLE: `ever_pregnant` (she has children) and
    # `fam_same_problem` (no family concern on record). A gate silently pruned any other way never got asked.
    na = [q for q, r in st.answers.items() if r.state is AnswerStatus.NOT_APPLICABLE]
    cl.check("a2 every active gate reached a real state",
             not open_gates and set(na) <= {"ever_pregnant", "fam_same_problem"},
             f"still open at the end: {open_gates}; unjustified NOT_APPLICABLE: {sorted(set(na) - {'ever_pregnant', 'fam_same_problem'})}")
    states = {r.state for r in st.answers.values()}
    cl.check("a2 all five answer states coexist", _ALL_STATES <= states, f"missing={sorted(_ALL_STATES - states)}")
    cl.check("a2 roster is the 12-year-old daughter and the 4-year-old son",
             sorted((c.age_years, c.tag) for c in st.roster) == [(4.0, "male"), (12.0, "female")]
             and st.roster_incomplete is False, f"roster={[(c.id, c.age_years, c.tag) for c in st.roster]}")
    # The three buckets must all have fired at real scale, not once each.
    ups = [d[2] for d in ev.decisions if d[1] == "other_form" and d[2]]
    cl.check("a2 the OTHER_FORM bucket ran at scale", len(set(ups)) >= 10, f"revision targets={sorted(set(ups))}")
    cl.check("a2 the RETAIN_OBSERVATION bucket ran at scale", len(ev.retained) >= 17, f"retained={len(ev.retained)}")
    cl.check("a2 the direct-answer bucket ran at scale",
             sum(1 for d in ev.decisions if d[1] == "in_domain") >= 40,
             f"in_domain={sum(1 for d in ev.decisions if d[1] == 'in_domain')}")

    # --- Bucket A: the buffered nudge ---------------------------------------------------------------------
    nudges = [e.get("detail") for e in ev.events_of("manual_nudge")]
    cl.check("a2a manual_nudge evaluated buffered speech", "evaluated" in nudges, f"manual_nudge={nudges}")
    cl.check("a2a back-to-back text+nudge frame gap under 400ms", bool(p.gaps) and min(p.gaps) < 400.0, f"gaps={p.gaps}")
    mode_state, mode = _answered(ev, "birth_mode_detail")
    cl.check("a2a birth_mode_detail resolved CONCERN on the daughter by the nudge",
             mode_state is C and list(st.answers["birth_mode_detail"].item_ids) == [daughter] and bool(daughter),
             f"state={mode_state} items={st.answers.get('birth_mode_detail').item_ids if 'birth_mode_detail' in st.answers else None} daughter={daughter}")
    cl.check("a2a birth_mode_detail asked exactly once at main (no double-advance)",
             len([s for s in p.sent_on("birth_mode_detail") if s["step"] == "birth_mode_detail"]) == 1,
             f"sends={[s['step'] for s in p.sent_on('birth_mode_detail')]}")
    # --- Bucket A: the named clarify back to the UNKNOWN thalassemia gate ---------------------------------
    named = [d[2] for d in ev.decisions if d[1] == "named_clarify" and d[2]]
    cl.check("a2a NAMED_CLARIFY jumped back to preg_thalassemia", "preg_thalassemia" in named, f"named={named}")
    cl.check("a2a preg_thalassemia reopened from UNKNOWN and ended CLEAR",
             _answered(ev, "preg_thalassemia")[0] is CL and len(p.sent_on("preg_thalassemia")) >= 2,
             f"state={_answered(ev, 'preg_thalassemia')[0]} sends={[s['step'] for s in p.sent_on('preg_thalassemia')]}")
    cl.check("a2a fam_related still got its own answer after the jump-back",
             _answered(ev, "fam_related")[0] is CL, f"got {_answered(ev, 'fam_related')[0]}")
    reassurance = " ".join(str(t.get("text") or "") for t in ev.call.get("transcript") or []
                           if t.get("role") == "assistant" and "थैलेसीमिया" in str(t.get("text") or ""))
    cl.check("a2a thalassemia reassurance never described severe disease",
             bool(reassurance.strip()) and not _has(reassurance, "गंभीर बीमारी", "बार-बार खून", "ट्रांसफ्यूजन"),
             f"clarify turn said: {reassurance[:160]}")
    # --- Bucket A: follow-up aside dedup, and the two redacted disease names ------------------------------
    joined = " ".join(ev.retained)
    cl.check("a2a the fam_who aside stayed on the follow-up slot",
             _has(str(st.answers["fam_fits_delay"].followups.get("fam_who").note.verbatim if "fam_fits_delay" in st.answers
                      and "fam_who" in st.answers["fam_fits_delay"].followups else ""), "अल्जाइमर"),
             f"fam_who={st.answers.get('fam_fits_delay')}")
    cl.check("a2a the fam_who aside was NOT also filed as an unlisted observation",
             "अल्जाइमर" not in joined, f"retained={ev.retained}")
    cl.check("a2a doctor summary keeps ड्यूशेन and हीमोफीलिया",
             _has(doctor, "ड्यूशेन", "हीमोफीलिया"), "doctor summary lost the reported diagnosis")
    cl.check("a2a patient summary redacts both names",
             bool(patient.strip()) and not _has(patient, "ड्यूशेन", "हीमोफीलिया", "ड्यूश", "हीमोफ"),
             "patient summary is empty or leaked the diagnosis")
    leaked_pdf = [frag for frag in _DISEASE if frag in ev.pdf_text]
    cl.check("a2a patient PDF redacts both names",
             bool(ev.pdf_text.strip()) and not leaked_pdf and "ड्यूशेन" not in ev.pdf_text and "ड्यूश" not in ev.pdf_text,
             f"PDF leaked fragments {leaked_pdf}")
    cl.check("a2a the maternal-bleeding words were not double-filed as an unlisted observation",
             not _has(joined, "हीमोफीलिया", "ड्यूशेन"), f"retained={ev.retained}")

    # --- Bucket B, direction 1: a settled CLEAR taken back to CONCERN -------------------------------------
    up = A2_UP
    wrong = {q: str(_answered(ev, q)[0]) for q in up if _answered(ev, q)[0] is not C}
    cl.check("a2b CLEAR -> CONCERN revisions all landed", not wrong, f"still not CONCERN: {wrong}")
    flags = [str(x) for x in (ev.call.get("red_flags") or [])]
    missing = [q for q in up if not any(q in f for f in flags)]
    cl.check("a2b every upward revision is in red_flags", not missing, f"missing from red_flags: {missing}")
    decided = {d[2] for d in ev.decisions if d[1] == "other_form"}
    cl.check("a2b each upward revision was recorded as OTHER_FORM",
             set(up) <= decided, f"missing OTHER_FORM: {sorted(set(up) - decided)}")
    # A backtrack onto a settled gate is not a miss: `reduce_turn` keeps the cursor and the token, so the
    # active gate's own answer is the SECOND send on the SAME step token, with no `try:` bump between them.
    burned = {}
    for src, _target in A2_FLIP_UP:
        rows = p.sent_on(src)
        # Send 0 carries the revision, send 1 is the re-ask of the SAME gate: one token, no `try:` bump.
        if len(rows) < 2 or rows[1]["step"] != rows[0]["step"] or "try:" in str(rows[1]["step"]):
            burned[src] = [r["step"] for r in rows]
    cl.check("a2b each revision gate was re-asked on the same token without burning an attempt", not burned,
             f"steps={burned}")
    cl.check("a2b the revision's words were handed to the model as noted_observation",
             _replay_ack(ev, "birth_sudden") is True,
             "replaying the stored interpretation produced no noted_observation")

    # --- Bucket B, direction 2: a settled CONCERN taken back to CLEAR ------------------------------------
    down = A2_DOWN
    still = {q: str(_answered(ev, q)[0]) for q in down if _answered(ev, q)[0] is not CL}
    cl.check("a2b CONCERN -> CLEAR revisions all landed", not still, f"still not CLEAR: {still}")
    cl.check("a2b a cleared gate is gone from red_flags",
             not [q for q in down if any(q in f for f in flags)],
             f"red_flags still name {[q for q in down if any(q in f for f in flags)]}")
    proj = ev.proj
    yes_items = [str(x) for x in ((proj.yes_items if proj else ()) or ())]
    # A cleared gate is not "absent from her copy": it becomes a *negative* on page two, which is the whole point.
    # What must not survive is the yes-list entry and the red flag it earned while it was still a CONCERN.
    stems = {gid: ((q := ev.pack.by_id.get(gid)) and (q.stem or q.say)) for gid in down}
    grouped = " ".join(str(g) for g in ((proj.page2_groups if proj else ()) or ()))
    cl.check("a2b a cleared gate is gone from yes_items and listed as a negative instead",
             not [gid for gid, x in stems.items() if x and any(x in str(y) for y in yes_items)]
             and all(x in grouped for x in stems.values() if x),
             f"still a yes: {[g for g, x in stems.items() if x and any(x in str(y) for y in yes_items)]} "
             f"missing a negative: {[g for g, x in stems.items() if x and x not in grouped]}")
    stale = {q: {"items": list(r.item_ids), "fups": sorted(r.followups),
                 "note": (r.verbatim_text or "")[:40]} for q, r in ((q, st.answers[q]) for q in down if q in st.answers)
             if r.item_ids or r.followups or (r.verbatim_text or "").strip()}
    cl.check("a2b a cleared gate kept no stale child, follow-up, or verbatim note", not stale, f"left behind: {stale}")
    cl.check("a2b both directions were recorded as OTHER_FORM", set(down) <= decided,
             f"missing OTHER_FORM: {sorted(set(down) - decided)}")

    # --- Bucket C: unlisted clinical asides ---------------------------------------------------------------
    cl.check("a2c every distinct unlisted aside was retained",
             sum(1 for needle in A2_ASIDES if needle in joined) >= 17,
             f"found {sum(1 for needle in A2_ASIDES if needle in joined)}/{len(A2_ASIDES)} in {ev.retained}")
    cl.check("a2c retained asides render under the unlisted heading in the doctor note",
             "### Additional Clinical Observations (Unlisted)" in doctor
             and sum(1 for needle in A2_ASIDES if needle in doctor) >= 15,
             f"heading={'### Additional Clinical Observations (Unlisted)' in doctor}")
    cl.check("a2c zero unlisted asides leaked into the patient summary",
             not [needle for needle in A2_ASIDES if needle in patient], f"patient leaked {[n for n in A2_ASIDES if n in patient]}")
    cl.check("a2c zero unlisted asides leaked into the patient PDF",
             not [needle for needle in A2_ASIDES if needle in ev.pdf_text],
             f"PDF leaked {[n for n in A2_ASIDES if n in ev.pdf_text]}")
    cl.check("a2c zero unlisted asides reached yes_items or page2_groups",
             not [needle for needle in A2_ASIDES if any(needle in str(y) for y in yes_items)]
             and not [needle for needle in A2_ASIDES if any(needle in str(g) for g in ((proj.page2_groups if proj else ()) or ()))],
             "an aside was projected onto her copy")
    cl.check("a2c an aside gate exhausted to UNANSWERED with its aside still retained",
             _answered(ev, "body_proportion")[0] is UA and "उल्टी जैसा जी मिचलाता" in joined,
             f"body_proportion={_answered(ev, 'body_proportion')[0]} retained={[x for x in ev.retained if 'मिचलाता' in x]}")
    cl.check("a2c a second exhausted gate kept its aside too",
             _answered(ev, "fam_lumps")[0] is UA and "डैंड्रफ" in joined,
             f"fam_lumps={_answered(ev, 'fam_lumps')[0]}")

    # --- Law 1 on the closing question -------------------------------------------------------------------
    cl.check("a2f chatter never became a CONCERN on close_anything_else",
             _answered(ev, "close_anything_else")[0] is not C, f"got {_answered(ev, 'close_anything_else')[0]}")
    cl.check("a2f the weather remark never reached the patient copy or the PDF",
             not _has(" ".join(yes_items) + " " + patient + " " + ev.pdf_text, "मौसम", "रसोई", "बारिश"),
             "chatter leaked onto her copy")
    # --- the event ledger on a long chaotic call ---------------------------------------------------------
    cl.check("a2 session_started survived the event cap on a long call",
             bool(ev.events) and str(ev.events[0].get("type")) == "session_started",
             f"first event={ev.events[0].get('type') if ev.events else None} total={len(ev.events)}")
    return {"asks": {k: v for k, v in p.counts.items() if v > 1}, "default_replies": p.defaults,
            "nudges": nudges, "gaps": p.gaps, "retained": ev.retained,
            "revisions": sorted(decided), "named_clarify": named, "sent": len(p.sent)}


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

    # -- transport --------------------------------------------------------------------------------------
    async def connect(self, resume: str = "") -> None:
        import websockets

        url = f"ws://{HOST}:{PORT}/ws/live?pack={self.pack.id}" + (f"&resume={resume}" if resume else "")
        self.ws = await websockets.connect(url, additional_headers={"Cookie": self.cookie},
                                           open_timeout=30, max_size=8_000_000)

    async def _read(self) -> dict[str, Any] | None:
        """One frame's side effects only - never a reply, so it is safe to call while waiting for a signal."""
        try:
            msg = json.loads(await asyncio.wait_for(self.ws.recv(), timeout=RECV_TIMEOUT_SEC))
        except (asyncio.TimeoutError, Exception):  # noqa: BLE001 - a dead socket is reported by the caller
            return None
        if not isinstance(msg, dict):
            return None
        if isinstance(kind := msg.get("type"), str):
            self.persona.observe(kind, msg)
        if kind == "telemetry" and self.persona.dropped and not self.persona.telemetry_after_resume:
            # The first telemetry frame after the rejoin carries the merged counters: that is the proof the call
            # did not start over. Read before the loop answers anything on the resumed step.
            self.persona.telemetry_after_resume = dict(msg.get("telemetry") or {})
        return msg

    async def _wait_for(self, pred, limit_sec: float) -> dict[str, Any] | None:
        end = time.monotonic() + limit_sec
        while time.monotonic() < end:
            msg = await self._read()
            if msg is None:
                return None
            if pred(msg):
                return msg
        return None

    # -- actions -----------------------------------------------------------------------------------------
    async def _act(self, entry: dict[str, Any], state: CallState) -> None:
        mode, payload = entry["mode"], entry["payload"]
        if mode == "text":
            await self.ws.send(json.dumps({"type": "text", "text": payload}, ensure_ascii=False))
        elif mode == "quick":
            await self.ws.send(json.dumps({"type": "quick_reply", "value": payload, "step": state.step_token,
                                           "epoch": self.persona.epoch}, ensure_ascii=False))
        elif mode == "nudge":
            await self.ws.send(json.dumps({"type": "nudge"}))
        elif mode == "text_nudge":
            # Attack 2A: the words are banked and the nudge lands in the very next frame, before Gemini can
            # answer the realtime text it was just handed. `_browser_loop` reads frames in order, so the nudge
            # always sees `self.text` non-empty and takes the `detail == "evaluated"` branch.
            sent_at = time.monotonic()
            await self.ws.send(json.dumps({"type": "text", "text": payload}, ensure_ascii=False))
            await self.ws.send(json.dumps({"type": "nudge"}))
            self.persona.gaps.append(round((time.monotonic() - sent_at) * 1000, 1))
        elif mode == "tap_then_contradict":
            # Attack 1B: speech first, so `self.snapshot` is this step and its Decision pass is in flight; then a
            # tap of the opposite answer on the SAME step. `thinking` is emitted by `_next_question` immediately
            # before the Decision request goes out - the one frame that proves the pass is really running.
            await self.ws.send(json.dumps({"type": "text", "text": payload}, ensure_ascii=False))
            marker = await self._wait_for(
                lambda m: m.get("type") == "thinking" and str(m.get("step") or "") == state.step_token, 45.0)
            if marker is None:
                print(f"  [{entry['key']}] no thinking frame; tap skipped", flush=True)
                return
            await asyncio.sleep(0.12)
            epoch = marker.get("epoch")
            await self.ws.send(json.dumps({"type": "quick_reply", "value": "clear", "step": state.step_token,
                                           "epoch": epoch if isinstance(epoch, int) else self.persona.epoch},
                                          ensure_ascii=False))
        else:
            raise RuntimeError(f"unknown action {mode}")

    # -- the drop ----------------------------------------------------------------------------------------
    async def _drop_and_resume(self) -> None:
        """Attack 1E: an abrupt socket death, never a `hangup`, then a rejoin inside the resume grace."""
        persona = self.persona
        call = db.get_call(self.call_id) or {}
        persona.telemetry_before_drop = dict(call.get("telemetry") or {})
        persona.drop_prompt_chars = len(engine.say_text(self.pack, db.call_state_from_row(call)))
        persona.dropped = True
        print(f"  dropping the socket on {DROP_QID} (call {self.call_id})", flush=True)
        try:
            await self.ws.close()
        except Exception:  # noqa: BLE001 - a socket already gone is exactly what a network drop looks like
            pass
        await asyncio.sleep(RESUME_WAIT_SEC)
        await self.connect(str(persona.share_token or call.get("share_token") or ""))
        print(f"  reconnected with the call's own share_token", flush=True)

    # -- the loop ----------------------------------------------------------------------------------------
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
                persona.share_token = str(msg.get("share_token") or persona.share_token)
                print(f"  session started call {self.call_id}", flush=True)
            elif kind == "telemetry" and event == "manual_nudge":
                persona.nudge_details.append(str((msg.get("last_event") or {}).get("detail") or ""))
                print(f"  manual_nudge detail={persona.nudge_details[-1]!r}", flush=True)
            elif kind == "bot_turn_complete":
                # `bot_first_audio` is only emitted for a turn a caller's speech opened, so the nudge's own
                # re-cue reaches the harness through `bot_turn_complete` alone.
                heard = True
            elif kind == "telemetry" and event == "bot_first_audio":
                heard = True
            elif kind == "telemetry" and event == "turn_complete" and heard and self.call_id:
                heard = False
                state = db.call_state_from_row(db.get_call(self.call_id) or {})
                key = Persona.route(state)
                if persona.drop_on_qid and not persona.dropped and state.cursor.qid == persona.drop_on_qid \
                        and state.cursor.substep is SubStep.MAIN:
                    await self._drop_and_resume()
                    continue
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
                  "answers": _answers_summary(ev.state), "events": _events_summary(ev.events)}
        result.update(detail)
        print(("PASS " if result["ok"] else "FAIL ") + self.persona.pid
              + f" call={call_id} dur={result.get('duration_sec')} " + ", ".join(result["problems"]), flush=True)
        return result


def _answers_summary(state: CallState) -> dict[str, Any]:
    return {qid: {"state": str(r.state), "value": r.value, "items": list(r.item_ids), "note": (r.verbatim_text or "")[:240],
                  "followups": {fid: {"state": str(s.state), "note": s.note.verbatim[:160]} for fid, s in r.followups.items()}}
            for qid, r in sorted(state.answers.items())}


def _events_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(e.get("type") or "") for e in events)
    return {"total": len(events), "by_type": dict(sorted(counts.items())),
            "manual_nudge": [{"detail": e.get("detail"), "meta": e.get("meta")}
                             for e in events if e.get("type") == "manual_nudge"],
            "errors": [e.get("detail") for e in events if e.get("type") == "error"][:10],
            "barge_ins": counts.get("barge_in", 0), "prompt_recues": counts.get("prompt_recue", 0)}


# ------------------------------------------------------------------------------------------------------------
# Personas
# ------------------------------------------------------------------------------------------------------------
def _personas() -> list[Persona]:
    return [
        Persona(
            "a1_shishu_revival_conflict_disconnect", "shishu-hi", {
                "intro_name": _line("मेरा नाम कविता है।"),
                "intro_married": _line("हाँ, शादीशुदा हूँ।"),
                # Two unrelated answers spend the profile step's whole budget: `intro_children` must end
                # UNANSWERED and the call must move on instead of re-asking here a third time.
                "intro_children": _steps(_line("अभी शोर बहुत है, आवाज़ साफ़ नहीं आ रही।"),
                                         _line("छोड़िए, आगे का सवाल पूछिए।"),
                                         _line("दोनों 7 साल के हैं, एक बेटा और एक बेटी।")),
                "intro_bridge": _line("जी, पूछिए।"),
                # The revival: a volunteered twin count on the first child gate, with no ages. The give-up record
                # must be deleted, the cursor rewound to `intro_children`, and the ages re-asked with hold:roster.
                "birth_mode_detail": _steps(_line("मेरे दो जुड़वाँ बच्चे हैं, लगता है पहले आप सुन नहीं पाए!"),
                                            _line("नहीं, दोनों सामान्य डिलीवरी से हुए थे।")),
                # A question that never gets an answer at all: the retry budget must record UNANSWERED.
                "birth_term": _steps(_line("कल शाम को खिचड़ी खाई थी।"), _line("कमरे में पंखा चल रहा है।")),
                # Attack 1B: speech says हाँ with the detail, a tap on the same step says नहीं.
                "birth_nicu": _steps(("tap_then_contradict",
                                      "हाँ, जन्म के बाद 5 दिन काँच के बक्से में रखा था, साँस की तकलीफ थी।"),
                                     _line("हाँ, जन्म के बाद 5 दिन काँच के बक्से में रखा था, साँस की तकलीफ थी।")),
                "birth_nicu:which_item": _line("बेटे को।"),
                "birth_nicu:fu0": _line("साँस की तकलीफ़ थी, डॉक्टर ने NICU में रखा।"),
                "birth_nicu:fu1": _line("पाँच दिन।"),
                "birth_jaundice": _line("नहीं।"),
                "birth_sudden": _line("नहीं।"),
                "birth_cord_3weeks": _line("नहीं।"),
                # CLEAR here, taken back out of turn on body_fits below.
                "birth_cord_bleed": _line("नहीं।"),
                "birth_bcg": _line("पता नहीं, मुझे याद नहीं।"),
                "birth_cleft": _line("नहीं।"),
                "birth_hearing": _line("नहीं।"),
                "birth_vision": _line("नहीं।"),
                # Attack 1D: an out-of-turn revision of birth_cord_bleed while body_fits stays open.
                "body_fits": _steps(_line("वैसे मुझे अभी याद आया, नाल गिरते समय खून भी निकला था।"),
                                    _line("नहीं, दौरे कभी नहीं पड़े।")),
                # Attack 1C: a scope question about who this family gate is asking about.
                "fam_fits_delay": _steps(_line("आप बच्चे की पूछ रहे हैं या घर के बड़ों की पूछ रहे हैं?"), _line("नहीं।")),
                # Attack 1E: answered only after the socket is dropped and resumed.
                "fam_maternal_bleeding": _line("हाँ, मेरे सगे मामा के बेटे को हीमोफीलिया की बीमारी है।"),
                "fam_maternal_bleeding:fu0": _line("सगे मामा के बेटे को।"),
                # Attack 1F: pure non-medical chatter on the closing question.
                "close_anything_else": _steps(_line("अच्छा दीदी, आज बाहर गर्मी बहुत ज़्यादा है।"), _quick("clear")),
                "close_anything_else:fu0": _quick("clear"),
            },
            close="complete", check=_check_a1, drop_on_qid=DROP_QID, name_in_call="कविता"),
        Persona("a2_shishu_triad_chaos", "shishu-hi", _a2_script(), close="complete", check=_check_a2),
    ]


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

    wanted = {a for a in sys.argv[1:] if not a.startswith("-")}
    personas = [p for p in _personas() if not wanted or p.pid in wanted]
    if not personas:
        print(f"no matching call in {sorted(wanted)}", flush=True)
        return 1
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
        for persona in personas:
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
# Offline self-test: every critical assertion must fail on a broken state and pass on a whole one
# ------------------------------------------------------------------------------------------------------------
def _rec(state: AnswerStatus, note: str = "", value: Any = None, label: str = "", items: list[str] | None = None,
         followups: dict[str, tuple[AnswerStatus, str]] | None = None, redacted: bool = False) -> AnswerRecord:
    return AnswerRecord(state=state, value=value, choice_label=label, item_ids=list(items or []),
                        note=NoteText.from_text(note, redacted=redacted) if note else NoteText(),
                        followups={k: FollowupSlot(state=s, note=NoteText.from_text(t, redacted=redacted) if t else NoteText())
                                   for k, (s, t) in (followups or {}).items()})


def _stored(seq: int, step: str, text: str) -> StoredTurnEvidence:
    return StoredTurnEvidence(evidence_id=f"e{seq}", call_id=1, sequence=seq, source=EvidenceSource.TEXT,
                              snapshot_step=step, transcript=text, received_at=0.0)


def _sent(key: str, step: str, mode: str = "text", payload: str = "x") -> dict[str, Any]:
    return {"key": key, "mode": mode, "payload": payload, "step": step, "epoch": 1}


def _other_form_interp(target: str):
    """The interpretation the engine records for a backtracking revision onto an already-settled question."""
    from intake.turns import ClinicalClaim, DomainFit, SpeechAct, TopicRelation, TurnInterpretation

    return TurnInterpretation(act=SpeechAct.ANSWER, active_relation=TopicRelation.OTHER_FORM, active_domain=DomainFit.NONE,
                              claims=(ClinicalClaim(target_qid=target, relation=TopicRelation.OTHER_FORM,
                                                    domain_fit=DomainFit.IN_DOMAIN, status=TurnOutcome.CONCERN),))


def _profile(slots: dict[str, Any]) -> TurnCommand:
    return TurnCommand(profile_patch=ProfilePatch(slots=slots), profile_turn_attempted=True)


def _finish(ev: Ev) -> Ev:
    """Fill the projection-derived fields a live run gets from SQLite, so parity is graded the same way.

    A key the self-test deliberately pinned (a doctor note that lost its diagnosis) is left alone: the point of
    that mutation is a stored column that no longer matches the projection.
    """
    ev.proj = reports.project_summary(ev.pack, ev.state, ev.retained_ev)
    for key, value in (("summary_doctor", ev.proj.doctor_markdown),
                       ("summary_patient", ev.proj.patient_summary_text),
                       ("red_flags", list(ev.proj.red_flags))):
        if key not in ev.pinned:
            ev.call = {**ev.call, key: value}
    return ev


def _drain(pack, st: CallState, *, miss_qid: str = "") -> CallState:
    """Walk the rest of the form answering CLEAR (and one deliberate miss), as the live caller would."""
    for _ in range(90):
        if st.phase is not Phase.ASKING:
            break
        cmd = (TurnCommand(outcome=TurnOutcome.NOT_ABOUT_ACTIVE) if st.cursor.qid == miss_qid
               else TurnCommand(outcome=TurnOutcome.CLEAR))
        nxt = engine.reduce_turn(pack, st, cmd)
        if nxt.phase is Phase.ASKING and nxt.step_token == st.step_token:
            break  # a step that refuses to move: stop rather than spin
        st = nxt
    return st


def _synth_a1(pack) -> Ev:
    """A whole call 1, built from the engine's own reducer so the shapes are real."""
    st = engine.new_state(pack)
    st = engine.reduce_turn(pack, st, _profile({"caller_name": "कविता"}))
    st = engine.reduce_turn(pack, st, TurnCommand(outcome=TurnOutcome.ANSWERED, value="married"))
    st = engine.reduce_turn(pack, st, TurnCommand(profile_turn_attempted=True))   # miss 1 on the roster
    st = engine.reduce_turn(pack, st, TurnCommand(profile_turn_attempted=True))   # miss 2 -> UNANSWERED
    st = engine.reduce_turn(pack, st, TurnCommand(outcome=TurnOutcome.CLEAR))     # bridge
    # The revival exactly as the extractor delivers it: a count with no ages, then the ages.
    st = engine.reduce_turn(pack, st, TurnCommand(profile_patch=ProfilePatch(roster=(), roster_complete=False),
                                                  profile_turn_attempted=True))
    st = engine.reduce_turn(pack, st, TurnCommand(
        profile_patch=ProfilePatch(roster=(RosterInput(age_years=7.0, tag="male"), RosterInput(age_years=7.0, tag="female")),
                                   roster_complete=True), profile_turn_attempted=True))
    st = _drain(pack, st, miss_qid="birth_term")
    answers = dict(st.answers)
    answers.update({
        "birth_nicu": _rec(C, "हाँ, जन्म के बाद 5 दिन काँच के बक्से में रखा था, साँस की तकलीफ थी।", items=["item_1"],
                           followups={"nicu_reason": (C, "साँस की तकलीफ़ थी।"), "nicu_days": (C, "पाँच दिन।")}),
        "birth_cord_bleed": _rec(C, "वैसे मुझे अभी याद आया, नाल गिरते समय खून भी निकला था।"),
        "birth_bcg": _rec(U), "body_fits": _rec(CL), "fam_fits_delay": _rec(CL),
        # The diagnosis is spoken, and `redact_patient_note` keeps it off her copy - as the live interpreter does.
        "fam_maternal_bleeding": _rec(C, "मामा के बेटे को हीमोफीलिया है।", redacted=True),
        "close_anything_else": _rec(CL),
    })
    st = st.model_copy(update={"answers": answers, "phase": Phase.CLOSING, "close_reason": CloseReason.COMPLETE})
    persona = Persona("a1_shishu_revival_conflict_disconnect", "shishu-hi", {}, close="complete", check=_check_a1)
    persona.counts = {"intro_children": 3, "birth_nicu": 2, "body_fits": 2, "fam_fits_delay": 2, "close_anything_else": 2}
    persona.sent = [
        _sent("intro_children", "intro_children", payload="अभी शोर बहुत है।"),
        _sent("intro_children", "intro_children:hold:roster:try:1", payload="छोड़िए, आगे का सवाल पूछिए।"),
        _sent("intro_children", "intro_children:hold:roster", payload="दोनों 7 साल के हैं।"),
        _sent("birth_nicu", "birth_nicu", "tap_then_contradict", "हाँ, 5 दिन काँच के बक्से में।"),
        _sent("birth_nicu", "birth_nicu", payload="हाँ, 5 दिन काँच के बक्से में।"),
        _sent("birth_nicu:which_item", "birth_nicu:which_item", payload="बेटे को।"),
        _sent("body_fits", "body_fits", payload="वैसे नाल गिरते समय खून निकला था।"),
        _sent("body_fits", "body_fits", payload="नहीं, दौरे नहीं पड़े।"),
        _sent("fam_fits_delay", "fam_fits_delay", payload="बच्चे की या बड़ों की?"),
        _sent("fam_fits_delay", "fam_fits_delay", payload="नहीं।"),
        _sent("close_anything_else", "close_anything_else", payload="आज बाहर गर्मी बहुत ज़्यादा है।"),
        _sent("close_anything_else", "close_anything_else:try:1", "quick", "clear"),
    ]
    persona.call_ids = [7, 7]
    persona.telemetry_before_drop = {"user_turns": 24, "bot_turns": 22}
    persona.telemetry_after_resume = {"user_turns": 24, "bot_turns": 22}
    persona.drop_prompt_chars = 70
    retained = [_stored(30, "fam_maternal_bleeding", "मामा के बेटे को हीमोफीलिया है।")]
    call = {
        "id": 7, "telemetry": {"tool_errors": 0}, "audio_path": "c7.wav",
        "events": [{"type": "session_started"}, {"type": "session_resumed"}, {"type": "turn_complete"}],
        "transcript": [{"role": "assistant", "qid": DROP_QID,
                        "text": "क्या आपके मायके की तरफ किसी बच्चे या बड़े को चोट पर ज़्यादा खून बहने की दिक्कत है?"}],
    }
    return _finish(Ev(pack=pack, call=call, state=st, persona=persona,
                      decisions=[("conflict", "source_conflict", "birth_nicu", "stay"),
                                 ("clarify", "active_clarify", "fam_fits_delay", "stay"),
                                 ("accept_form", "other_form", "birth_cord_bleed", "stay"),
                                 ("accept_form", "in_domain", "birth_nicu", "advance")],
                      sources=[("tap", "birth_nicu"), ("text", "birth_nicu"), ("text", "body_fits")],
                      retained=[r.transcript for r in retained], retained_ev=retained, events=[
                          {"type": "session_started"}, {"type": "session_resumed"}, {"type": "turn_complete"}],
                      # The real ledger shape for the revision: the words are stored against the asked step, and
                      # the interpretation is the out-of-turn OTHER_FORM claim `policy` accepted.
                      sequences=[(24, "body_fits", "वैसे मुझे अभी याद आया, नाल गिरते समय खून भी निकला था।")],
                      interps={24: _other_form_interp("birth_cord_bleed")},
                      pages=2, pdf_text="जविाब इनकर मतलब जवब नहीं", wav_shape=(2, 16000)))


def _interp_for(line: Line, step_kind: str):
    """The interpretation the live Decision pass would hand `policy` for a scripted line.

    Built from the line's own intent (`Line.cmd`) so the replay exercises the real `policy.apply_policy`, and so
    a wrong row in `A2_LINES` shows up as a command that does not match what the interpreter derives.
    """
    from intake.turns import ClinicalClaim, DomainFit, SpeechAct, TopicRelation, TurnInterpretation

    cmd, claims = line.cmd, []
    act, rel, dom, clarify_target = SpeechAct.NONE, TopicRelation.UNRELATED, DomainFit.NONE, None
    stated = {t: o for t, o in cmd.stated.items() if o is not TurnOutcome.CLARIFY}
    if cmd.stated and len(stated) != len(cmd.stated):  # a `stated_ clarify` is the named jump-back
        act, clarify_target = SpeechAct.CLARIFY, next(t for t, o in cmd.stated.items() if o is TurnOutcome.CLARIFY)
        rel = TopicRelation.ACTIVE
    elif line.aside:
        claims.append(ClinicalClaim(relation=TopicRelation.CLINICAL_ASIDE))
        act, rel = SpeechAct.ANSWER, TopicRelation.CLINICAL_ASIDE
    elif stated:
        claims.extend(ClinicalClaim(target_qid=t, relation=TopicRelation.OTHER_FORM, domain_fit=DomainFit.IN_DOMAIN, status=o)
                      for t, o in stated.items())
        act, rel = SpeechAct.ANSWER, TopicRelation.OTHER_FORM
    elif cmd.outcome in (TurnOutcome.CLEAR, TurnOutcome.CONCERN, TurnOutcome.UNKNOWN, TurnOutcome.ANSWERED):
        claims.append(ClinicalClaim(target_qid=None, relation=TopicRelation.ACTIVE, domain_fit=DomainFit.IN_DOMAIN,
                                    status=cmd.outcome, value=cmd.value))
        act, rel, dom = SpeechAct.ANSWER, TopicRelation.ACTIVE, DomainFit.IN_DOMAIN
    return TurnInterpretation(act=act, active_relation=rel, active_domain=dom, claims=tuple(claims),
                              clarify_target_qid=clarify_target, item_ref=cmd.item_ref,
                              profile_patch=cmd.profile_patch, profile_turn_attempted=cmd.profile_turn_attempted,
                              followup_answered=cmd.followup_answered, redacted=cmd.redacted)


def _synth_a2(pack) -> Ev:
    """The whole triad call, replayed off `A2_LINES` through the real `policy` + `engine` reducers."""
    from intake import policy
    from intake.turns import DecisionReason, Disposition, EvidenceSource, TurnEvidence

    lines = {k: list(v) for k, v in A2_LINES.items()}
    persona = Persona("a2_shishu_triad_chaos", "shishu-hi", _a2_script(), close="complete", check=_check_a2)
    st, decisions, retained, sequences, interps = engine.new_state(pack), [], [], [], {}
    seq, used, transcripts = 0, Counter(), []
    for _ in range(400):
        if st.phase is not Phase.ASKING:
            break
        key, step, before = Persona.route(st), st.cursor, st.step_token
        queue, i = lines.get(key) or [], used[key]
        if i >= len(queue):
            raise RuntimeError(f"self-test script ran out on {key} ({i} sends; reached via {before})")
        line = queue[i]
        used[key] += 1
        if line.child:  # a which_item line names a child by sex; the roster owns the id
            line = Line(line.text, line.cmd.model_copy(update={"item_ref": _child(st, line.child) or line.cmd.item_ref}),
                        line.mode, line.aside, line.child)
        persona.counts[key] = persona.counts.get(key, 0) + 1
        persona.sent.append(_sent(key, before, line.mode, line.text))
        seq += 1
        kind = engine.step_kind(pack, st)
        if line.cmd.profile_patch is not None or line.cmd.profile_turn_attempted:
            nxt, pol = engine.reduce_turn(pack, st, line.cmd), None
        else:
            interp = _interp_for(line, kind)
            seen = [d.disposition for d in policy.evaluate_policy(pack, st, TurnEvidence(
                source=EvidenceSource.TEXT, snapshot_step=before, transcript=line.text, received_at=0.0), interp).decisions]
            nxt, pol = policy.apply_policy(pack, st, TurnEvidence(
                source=EvidenceSource.TEXT, snapshot_step=before, transcript=line.text, received_at=0.0), interp)
            if DecisionReason.OTHER_FORM in [d.reason for d in pol.decisions]:
                interps[seq] = interp
            if DecisionReason.OTHER_FORM in [d.reason for d in pol.decisions] or Disposition.RETAIN_OBSERVATION in seen:
                sequences.append((seq, before, line.text))
            if Disposition.RETAIN_OBSERVATION in seen:
                retained.append(_stored(seq, before, line.text))
            decisions.extend((d.disposition.value, d.reason.value, str(d.target_qid or ""), d.cursor_action.value)
                             for d in pol.decisions)
            # A row that does not derive the command it claims is a broken schedule, never a pass.
            if line.cmd.stated and not pol.command.stated:
                raise RuntimeError(f"{key}: stated {line.cmd.stated} was not accepted by policy")
        transcripts.append((line.text, nxt.step_token))
        st = nxt

    persona.gaps, persona.nudge_details = [6.2], ["evaluated"]
    events = [{"type": "session_started", "detail": "9", "meta": {"pack": "shishu-hi"}},
              {"type": "manual_nudge", "detail": "evaluated", "meta": {"step_in": "birth_mode_detail"}},
              *({"type": "turn_complete", "detail": "", "meta": {}} for _ in range(len(persona.sent))),
              {"type": "manual_nudge", "detail": "", "meta": None}]
    call = {"id": 9, "telemetry": {"tool_errors": 0, "user_turns": len(persona.sent)}, "audio_path": "c9.wav",
            "events": events,
            "transcript": [{"role": "assistant", "qid": "fam_related", "text":
                            "जी, थैलेसीमिया माइनर या ट्रेट एक सामान्य माता-पिता की खून जाँच है जिसमें माँ स्वस्थ रहती है। "
                            "डॉक्टर इसका उपयोग प्रेग्नेंसी और बच्चे की सलाह के लिए करते हैं।"}]}
    return _finish(Ev(pack=pack, call=call, state=st, persona=persona, decisions=decisions,
                      sources=[("text", s[1]) for s in sequences],
                      retained=[o.transcript for o in retained], retained_ev=retained, events=events,
                      sequences=sequences, interps=interps,
                      pages=2, pdf_text="जविाब इनकर मतलब जवब नहीं", wav_shape=(2, 16000)))


def _mutations_a1() -> list[tuple[str, str, Any]]:
    """(label, the check that must flip, mutation). Each mutation breaks exactly one invariant."""
    def answers(ev: Ev, **kw) -> None:
        ev.state = ev.state.model_copy(update={"answers": {**ev.state.answers, **kw}})

    def drop(ev: Ev, pred) -> None:
        ev.decisions = [d for d in ev.decisions if not pred(d)]

    def sent(ev: Ev, key: str, index: int, **kw) -> None:
        ev.persona.sent_on(key)[index].update(kw)

    def call(ev: Ev, **kw) -> None:
        # A stored column the mutation deliberately breaks must survive `_finish` re-projecting the others.
        ev.pinned.update(kw)
        ev.call = {**ev.call, **kw}

    return [
        ("no revival: only two asks", "a1a give-up recorded then revived (3 asks, roster landed)",
         lambda ev: ev.persona.counts.update({"intro_children": 2})),
        ("revival lost its hold:roster", "a1a revival re-asked the ages with hold:roster",
         lambda ev: sent(ev, "intro_children", 2, step="intro_children")),
        ("revival burned an attempt", "a1a revival re-asked the ages with hold:roster",
         lambda ev: sent(ev, "intro_children", 2, step="intro_children:hold:roster:try:1")),
        ("ages landed as one child", "a1a roster resolved to two 7-year-olds",
         lambda ev: ev.state.roster.pop()),
        ("roster still incomplete", "a1a roster resolved to two 7-year-olds",
         lambda ev: setattr(ev.state, "roster_incomplete", True)),
        ("ever_pregnant asked though she has children", "a1a ever_pregnant pruned by has_children",
         lambda ev: answers(ev, ever_pregnant=_rec(CL))),
        ("a child gate left NOT_APPLICABLE", "a1a no child gate left NOT_APPLICABLE",
         lambda ev: answers(ev, birth_transfusion=_rec(AnswerStatus.NOT_APPLICABLE))),
        ("tap ignored: no conflict decision", "a1b SOURCE_CONFLICT on birth_nicu",
         lambda ev: drop(ev, lambda d: d[1] == "source_conflict")),
        ("tap never recorded as evidence", "a1b tap evidence stored for birth_nicu",
         lambda ev: setattr(ev, "sources", [s for s in ev.sources if s != ("tap", "birth_nicu")])),
        ("NICU answer lost", "a1b birth_nicu re-answered CONCERN with the NICU note",
         lambda ev: answers(ev, birth_nicu=_rec(CL))),
        ("clarify burned an attempt", "a1c clarify stayed on fam_fits_delay without an attempt",
         lambda ev: sent(ev, "fam_fits_delay", 1, step="fam_fits_delay:try:1")),
        ("clarify never recorded", "a1c clarify decision recorded on fam_fits_delay",
         lambda ev: drop(ev, lambda d: d[2] == "fam_fits_delay")),
        ("revision never landed", "a1d birth_cord_bleed revised CLEAR -> CONCERN",
         lambda ev: answers(ev, birth_cord_bleed=_rec(CL))),
        ("OTHER_FORM never recorded", "a1d OTHER_FORM decision recorded",
         lambda ev: drop(ev, lambda d: d[1] == "other_form")),
        ("body_fits advanced instead of staying",
         "a1d body_fits re-asked on the same token without burning an attempt, then CLEAR",
         lambda ev: sent(ev, "body_fits", 1, step="body_milestones")),
        ("body_fits burned an attempt on the revision",
         "a1d body_fits re-asked on the same token without burning an attempt, then CLEAR",
         lambda ev: sent(ev, "body_fits", 1, step="body_fits:try:1")),
        ("body_fits answered CONCERN", "a1d body_fits re-asked on the same token without burning an attempt, then CLEAR",
         lambda ev: answers(ev, body_fits=_rec(C))),
        ("the revision arrived as an ordinary answer, so no ack was owed",
         "a1d the revision's words were handed to the model as noted_observation",
         lambda ev: ev.interps.update({24: ev.interps[24].model_copy(update={"claims": ()})})),
        ("resume minted a new call", "a1e same call_id after resume", lambda ev: ev.persona.call_ids.append(8)),
        ("session_resumed never logged", "a1e session_resumed event recorded",
         lambda ev: setattr(ev, "events", [e for e in ev.events if e.get("type") != "session_resumed"])),
        ("resume duplicated the held prompt", "a1e resume did not duplicate the held prompt",
         lambda ev: call(ev, transcript=[{"role": "assistant", "qid": DROP_QID, "text": (
             "क्या आपके मायके की तरफ किसी बच्चे या बड़े को चोट पर ज़्यादा खून बहने की दिक्कत है? "
             "क्या आपके मायके की तरफ किसी बच्चे या बड़े को चोट पर ज़्यादा खून बहने की दिक्कत है?")}])),
        ("telemetry reset by the resume", "a1e telemetry survived the drop",
         lambda ev: setattr(ev.persona, "telemetry_after_resume", {"user_turns": 1, "bot_turns": 1})),
        ("chatter became a concern", "a1f chatter never became a CONCERN on close_anything_else",
         lambda ev: answers(ev, close_anything_else=_rec(C, "आज बाहर गर्मी बहुत ज़्यादा है।"))),
        ("UNKNOWN missing", "all five answer states coexist", lambda ev: answers(ev, birth_bcg=_rec(CL))),
        ("UNANSWERED missing", "all five answer states coexist", lambda ev: answers(ev, birth_term=_rec(CL))),
        ("doctor note lost the diagnosis", "doctor summary keeps हीमोफीलिया",
         lambda ev: call(ev, summary_doctor="Close reason: complete.")),
        ("watchdog closed the call", "closed on content, not the watchdog",
         lambda ev: setattr(ev, "events", [*ev.events, {"type": "auto_close", "detail": "silence"}])),
        ("a tool error was swallowed", "zero tool_errors", lambda ev: call(ev, telemetry={"tool_errors": 2})),
        ("PDF lost the glossary", "glossary in the PDF", lambda ev: setattr(ev, "pdf_text", "")),
        ("PDF is a single page", "PDF is at least two pages", lambda ev: setattr(ev, "pages", 1)),
        ("WAV came back mono", "stereo 16 kHz WAV", lambda ev: setattr(ev, "wav_shape", (1, 16000))),
        ("chatter reached the patient PDF", "a1f गर्मी never reached the patient PDF",
         lambda ev: setattr(ev, "pdf_text", ev.pdf_text + " गर्मी")),
    ]


def _mutations_a2() -> list[tuple[str, str, Any]]:
    """(label, the one check that must flip, mutation). Each mutation breaks exactly one invariant."""
    def answers(ev: Ev, **kw) -> None:
        ev.state = ev.state.model_copy(update={"answers": {**ev.state.answers, **kw}})

    def call(ev: Ev, **kw) -> None:
        # A stored column the mutation deliberately breaks must survive `_finish` re-projecting the others.
        ev.pinned.update(kw)
        ev.call = {**ev.call, **kw}

    def drop(ev: Ev, pred) -> None:
        ev.decisions = [d for d in ev.decisions if not pred(d)]

    def events(ev: Ev, fn) -> None:
        ev.events = fn(list(ev.events))

    def sent(ev: Ev, key: str, index: int, **kw) -> None:
        ev.persona.sent_on(key)[index].update(kw)

    def retained(ev: Ev, *texts: str) -> None:
        ev.retained = list(texts)
        ev.retained_ev = [_stored(i + 1, "body_thumb", t) for i, t in enumerate(texts)]

    return [
        # --- the walk --------------------------------------------------------------------------------------
        ("a gate was never reached", "a2 every active gate reached a real state",
         lambda ev: setattr(ev, "state", ev.state.model_copy(
             update={"answers": {k: v for k, v in ev.state.answers.items() if k != "fam_maternal_eczema"}}))),
        ("a gate was left NOT_APPLICABLE for no reason", "a2 every active gate reached a real state",
         lambda ev: answers(ev, body_puberty=_rec(AnswerStatus.NOT_APPLICABLE))),
        ("UNKNOWN missing", "a2 all five answer states coexist", lambda ev: answers(ev, body_private_parts=_rec(CL))),
        ("UNANSWERED missing", "a2 all five answer states coexist",
         lambda ev: answers(ev, body_proportion=_rec(CL), fam_lumps=_rec(CL))),
        ("roster collapsed to one child", "a2 roster is the 12-year-old daughter and the 4-year-old son",
         lambda ev: ev.state.roster.pop()),
        ("the revision bucket never ran", "a2 the OTHER_FORM bucket ran at scale",
         lambda ev: drop(ev, lambda d: d[1] == "other_form")),
        ("the aside bucket never ran", "a2 the RETAIN_OBSERVATION bucket ran at scale",
         lambda ev: retained(ev, "बेटी को बस में उल्टी जैसा जी मिचलाता है।")),
        ("the direct-answer bucket never ran", "a2 the direct-answer bucket ran at scale",
         lambda ev: drop(ev, lambda d: d[1] == "in_domain")),
        # --- Bucket A --------------------------------------------------------------------------------------
        ("nudge fell back to pure silence", "a2a manual_nudge evaluated buffered speech",
         lambda ev: (setattr(ev.persona, "nudge_details", [""]),
                     events(ev, lambda es: [{**e, "detail": ""} if e.get("type") == "manual_nudge" else e for e in es]))),
        ("nudge frame arrived a second late", "a2a back-to-back text+nudge frame gap under 400ms",
         lambda ev: setattr(ev.persona, "gaps", [1400.0])),
        ("the nudge threw the answer away", "a2a birth_mode_detail resolved CONCERN on the daughter by the nudge",
         lambda ev: answers(ev, birth_mode_detail=_rec(UA))),
        ("the nudge lost the child it named", "a2a birth_mode_detail resolved CONCERN on the daughter by the nudge",
         lambda ev: answers(ev, birth_mode_detail=_rec(C, "सिजेरियन हुआ था।"))),
        ("the nudged step was advanced twice", "a2a birth_mode_detail asked exactly once at main (no double-advance)",
         lambda ev: ev.persona.sent.append(_sent("birth_mode_detail", "birth_mode_detail", payload="दोबारा"))),
        ("the clarify routed to the active question only", "a2a NAMED_CLARIFY jumped back to preg_thalassemia",
         lambda ev: drop(ev, lambda d: d[1] == "named_clarify")),
        ("the UNKNOWN gate never reopened", "a2a preg_thalassemia reopened from UNKNOWN and ended CLEAR",
         lambda ev: answers(ev, preg_thalassemia=_rec(C))),
        ("the jump-back stranded its own question", "a2a fam_related still got its own answer after the jump-back",
         lambda ev: answers(ev, fam_related=_rec(UA))),
        ("the reassurance described severe disease",
         "a2a thalassemia reassurance never described severe disease",
         lambda ev: call(ev, transcript=[{"role": "assistant", "qid": "fam_related",
                                          "text": "थैलेसीमिया एक गंभीर बीमारी है जिसमें बार-बार खून चढ़ाना पड़ता है।"}])),
        ("the follow-up aside never landed on fam_who", "a2a the fam_who aside stayed on the follow-up slot",
         lambda ev: answers(ev, fam_fits_delay=_rec(C, "हाँ, परिवार में दिक्कत थी।"))),
        ("the follow-up aside was double-filed", "a2a the fam_who aside was NOT also filed as an unlisted observation",
         lambda ev: (ev.retained.append("मेरे नाना जी को बुढ़ापे में भूलने की अल्जाइमर बीमारी थी।"),
                     setattr(ev, "retained_ev", [*ev.retained_ev,
                                                 _stored(99, "fam_fits_delay:followup:0", "अल्जाइमर बीमारी थी।")]))),
        ("doctor note lost the diagnosis", "a2a doctor summary keeps ड्यूशेन and हीमोफीलिया",
         lambda ev: call(ev, summary_doctor="Close reason: complete.")),
        ("patient copy leaked the diagnosis", "a2a patient summary redacts both names",
         lambda ev: call(ev, summary_patient="मामा के लड़के को ड्यूशेन और हीमोफीलिया का शक था।")),
        ("PDF leaked the diagnosis", "a2a patient PDF redacts both names",
         lambda ev: setattr(ev, "pdf_text", ev.pdf_text + " ड्यूशेन हीमोफ")),
        ("the maternal-bleeding words were filed twice", "a2a the maternal-bleeding words were not double-filed as an unlisted observation",
         lambda ev: (ev.retained.append("डॉक्टर ने ड्यूशेन और हीमोफीलिया का शक बताया था।"),
                     setattr(ev, "retained_ev", [*ev.retained_ev, _stored(98, "fam_maternal_bleeding", "हीमोफीलिया")]))),
        # --- Bucket B, direction 1 --------------------------------------------------------------------------
        ("an upward revision never landed", "a2b CLEAR -> CONCERN revisions all landed",
         lambda ev: answers(ev, birth_nicu=_rec(CL))),
        ("an upward revision is missing from red_flags", "a2b every upward revision is in red_flags",
         lambda ev: answers(ev, fam_cancer=_rec(CL))),
        ("an upward revision was never recorded as OTHER_FORM", "a2b each upward revision was recorded as OTHER_FORM",
         lambda ev: drop(ev, lambda d: d[1] == "other_form" and d[2] == "ill_fasting")),
        ("the revision burned an attempt", "a2b each revision gate was re-asked on the same token without burning an attempt",
         lambda ev: sent(ev, "birth_sudden", 1, step="birth_sudden:try:1")),
        ("the revision advanced the cursor", "a2b each revision gate was re-asked on the same token without burning an attempt",
         lambda ev: sent(ev, "birth_sudden", 1, step="birth_cord_3weeks")),
        ("the revision owed no spoken ack", "a2b the revision's words were handed to the model as noted_observation",
         lambda ev: ev.interps.update({s: i.model_copy(update={"claims": ()})
                                       for s, i in list(ev.interps.items())[:1]})),
        # --- Bucket B, direction 2 --------------------------------------------------------------------------
        ("a downward revision never landed", "a2b CONCERN -> CLEAR revisions all landed",
         lambda ev: answers(ev, birth_jaundice=_rec(C, "पीलिया बताया था।"))),
        ("a cleared gate survived in red_flags", "a2b a cleared gate is gone from red_flags",
         lambda ev: answers(ev, birth_term=_rec(C, "आठवें महीने में हुआ था।"))),
        ("a cleared gate survived on her copy", "a2b a cleared gate is gone from yes_items and listed as a negative instead",
         lambda ev: answers(ev, birth_jaundice=_rec(C, "पीलिया बताया था।"))),
        ("a cleared gate kept a stale child or follow-up",
         "a2b a cleared gate kept no stale child, follow-up, or verbatim note",
         lambda ev: answers(ev, birth_jaundice=AnswerRecord(
             state=CL, item_ids=["item_2"], note=NoteText.from_text("पीलिया बताया था।"),
             followups={"jaundice_days": FollowupSlot(state=C, note=NoteText.from_text("छह दिन लाइट में रखा।"))}))),
        ("a downward revision was never recorded as OTHER_FORM", "a2b both directions were recorded as OTHER_FORM",
         lambda ev: drop(ev, lambda d: d[1] == "other_form" and d[2] == "birth_term")),
        # --- Bucket C --------------------------------------------------------------------------------------
        ("an unlisted aside was dropped", "a2c every distinct unlisted aside was retained",
         lambda ev: retained(ev, "बेटी को बस में उल्टी जैसा जी मिचलाता है।")),
        ("the unlisted heading never rendered", "a2c retained asides render under the unlisted heading in the doctor note",
         lambda ev: call(ev, summary_doctor="### Patient Profile & Chief Complaint\n- Close reason: complete.")),
        ("an aside leaked onto her summary", "a2c zero unlisted asides leaked into the patient summary",
         lambda ev: call(ev, summary_patient="जी मिचलाता है और डैंड्रफ भी है।")),
        ("an aside leaked onto the PDF", "a2c zero unlisted asides leaked into the patient PDF",
         lambda ev: setattr(ev, "pdf_text", ev.pdf_text + " डैंड्रफ")),
        ("an exhausted gate recorded CLEAR instead of UNANSWERED",
         "a2c an aside gate exhausted to UNANSWERED with its aside still retained",
         lambda ev: answers(ev, body_proportion=_rec(CL))),
        ("the second exhausted gate answered through", "a2c a second exhausted gate kept its aside too",
         lambda ev: answers(ev, fam_lumps=_rec(CL))),
        # --- closing law and the ledger ---------------------------------------------------------------------
        ("chatter became a concern", "a2f chatter never became a CONCERN on close_anything_else",
         lambda ev: answers(ev, close_anything_else=_rec(C, "आज मौसम काफ़ी ठंडा है।"))),
        ("chatter reached the patient PDF", "a2f the weather remark never reached the patient copy or the PDF",
         lambda ev: setattr(ev, "pdf_text", ev.pdf_text + " मौसम")),
        ("the event cap ate session_started", "a2 session_started survived the event cap on a long call",
         lambda ev: setattr(ev, "events", [{"type": "turn_complete", "detail": "", "meta": {}} for _ in range(600)])),
        ("watchdog closed the call", "closed on content, not the watchdog",
         lambda ev: setattr(ev, "events", [*ev.events, {"type": "auto_close", "detail": "silence"}])),
        ("a tool error was swallowed", "zero tool_errors", lambda ev: call(ev, telemetry={"tool_errors": 2})),
        ("PDF lost the glossary", "glossary in the PDF", lambda ev: setattr(ev, "pdf_text", "")),
        ("WAV came back mono", "stereo 16 kHz WAV", lambda ev: setattr(ev, "wav_shape", (1, 16000))),
        ("a spoken leak appeared", "zero spoken leaks",
         lambda ev: call(ev, transcript=[{"role": "assistant", "qid": "fam_related", "text": "ड्यूशेन क्या है?"}])),
    ]


def _script_legal(pack, pid: str) -> list[str]:
    """Static guards on the scripted schedule itself, before any of it is spent against a live call.

    A tap whose `value` is a Hindi label instead of the pack's own option key is a harness bug that can only be
    found by running the call (call 6: `Invalid quick reply '\u0928\u0939\u0940\u0902\u0964'`), so it is checked here instead.
    """
    bad: list[str] = []
    if pid == "a2_shishu_triad_chaos":
        scripted = set(A2_LINES)
        walked = set()
        st = engine.new_state(pack)
        st = engine.reduce_turn(pack, st, TurnCommand(profile_patch=ProfilePatch(
            slots={"caller_name": "मीना"}), profile_turn_attempted=True))
        st = engine.reduce_turn(pack, st, TurnCommand(outcome=TurnOutcome.ANSWERED, value="married"))
        st = engine.reduce_turn(pack, st, TurnCommand(profile_patch=ProfilePatch(
            roster=(RosterInput(age_years=12.0, tag="female"), RosterInput(age_years=4.0, tag="male")),
            roster_complete=True), profile_turn_attempted=True))
        for _ in range(200):
            if st.phase is not Phase.ASKING:
                break
            walked.add(Persona.route(st))
            st = engine.reduce_turn(pack, st, TurnCommand(outcome=TurnOutcome.CLEAR))
        missing = sorted(walked - scripted)
        if missing:
            bad.append(f"{pid}: no line scripted for {missing}")
        for key, lines in A2_LINES.items():
            for line in lines:
                if line.mode == "quick":
                    # `engine.quick_reply_command` accepts the pack's own option keys AND the three universal
                    # outcome values on any step; anything else raises on the wire and kills the call.
                    spec = pack.by_id.get(key.split(":")[0])
                    legal = {"clear", "concern", "unknown"}
                    legal |= set(spec.options) if (spec is not None and spec.options) else set()
                    if line.text not in legal:
                        bad.append(f"{pid}: {key} sends quick reply {line.text!r}, which is not one of {sorted(legal)}")
    return bad


def _selftest() -> int:
    """Prove every critical assertion is non-vacuous, offline, before a single live token is spent."""
    print("=== offline self-test: each invariant must fail when broken and pass when whole ===", flush=True)
    failures: list[str] = []
    for pid, pack_id, synth, mutations in (
            ("a1_shishu_revival_conflict_disconnect", "shishu-hi", _synth_a1, _mutations_a1),
            ("a2_shishu_triad_chaos", "shishu-hi", _synth_a2, _mutations_a2)):
        pack = load_pack(pack_id)
        for problem in _script_legal(pack, pid):
            failures.append(problem)
            print(f"  FAIL {problem}", flush=True)
        whole = synth(pack)
        cl = Checklist()
        try:
            _parity(whole, cl)
            _common(whole, cl)
            whole.persona.check(whole, cl)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{pid}: whole state raised {type(exc).__name__}: {exc}")
            print(f"  FAIL {pid}: whole state raised {type(exc).__name__}: {exc}", flush=True)
            continue
        bad, total = cl.problems, len(cl.items)
        print(f"  {'PASS' if not bad else 'FAIL'} {pid}: whole state {total - len(bad)}/{total} checks green"
              + ("" if not bad else f" - {bad}"), flush=True)
        if bad:
            failures.append(f"{pid}: whole state is not green: {bad}")
        for label, target, mutate in mutations():
            ev = synth(pack)
            broken = Checklist()
            try:
                # The mutation lands first, then the projection re-derives from the broken state, then every check
                # reads it - exactly the order a live call takes (row written, projected, graded).
                mutate(ev)
                ev = _finish(ev)
                _parity(ev, broken)
                _common(ev, broken)
                ev.persona.check(ev, broken)
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{pid}: {label} raised {type(exc).__name__}: {exc}")
                print(f"  FAIL {pid}: {label} raised {type(exc).__name__}: {exc}", flush=True)
                continue
            hit = target in broken.failing()
            print(f"  {'PASS' if hit else 'FAIL'} {label} -> {target}", flush=True)
            if not hit:
                failures.append(f"{pid}: {label} did not flip {target!r} (failing={sorted(broken.failing())})")
    print(f"\n{'SELFTEST PASS' if not failures else 'SELFTEST FAIL'} ({len(failures)} problem(s))", flush=True)
    for line in failures:
        print(f"  - {line}", flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    raise SystemExit(asyncio.run(_main()))
