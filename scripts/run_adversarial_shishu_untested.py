#!/usr/bin/env python3
"""Two adversarial live calls on `shishu-hi` for mechanisms the other suites never reached. Run in `tmux`.

Boots `intake.app:app` on port 8796 with `DATA_DIR=intake/storage/adversarial_shishu_untested`, logs in over
`/api/login`, then drives one scripted caller per call over `ws://127.0.0.1:8796/ws/live?pack=shishu-hi`.

Call 1 (`a3_shishu_voice_twins_deciding_drop`) is audio-only: every attack is real 16 kHz PCM (`say -v Lekha` ->
`ffmpeg`, sha1-cached, plus generated fan noise), never a `{"type":"text"}` frame.

  * `3A` equal-name twins. Two 2-year-old daughters both called मीरा are typed from one spoken sentence. The
    twins prompt may number them but may never call one "मँझला"/"बीच वाले" (there is no middle child), and the
    fits concern must end up attributed to both (`item_ids == ["item_1", "item_2"]`) - whether the model answers
    "both" outright or the caller says "दोनों को" at `which_item`. The first live run proved the second branch is
    the one that really needs the prompt: the ASR wrote मीरा as "मेरा" and the pass resolved `item_ref: all`.
  * `3B` a 1.5 s mid-sentence pause under a ~0.0175 RMS fan. The pause must not split one answer into two turns.
  * `3C` `S-7.5`: the reply is aimed 250 ms before the prompt's playback ends and its *first frame's* send time is
    the measurement (`Mic.sent_at`), so the gate must resolve with `interruptions == 0` and no re-send.
  * `3D` a socket death while the Decision pass is in flight on `fam_cancer`, then `?resume=<share_token>`: same
    call id, `session_resumed`, the step re-asked and resolved, the WAV appended, telemetry monotonic.
  * `3E` `S-7.6`: real emergency speech inside the <5 s `PROMPTING` mute. The left channel of the saved stereo
    WAV is measured directly, and since the fix in `live.py` (`on_audio` hands the frame to the recorder before
    the mute gate returns) it is graded like any other check: speech she made over SHISHU must be in the WAV.
    The same words are then spoken in `LISTENING` so the call really closes `EMERGENCY`.

Call 2 (`a4_shishu_bridge_skip_jump_silence`) is text/quick_reply control flow plus the real silence timers:
रेखा, married, one 4-year-old son रोहन.

  * `4A` `intro_bridge` carries an upfront `stated_` concern that must auto-fill `birth_jaundice` and keep it
    from ever being asked at main.
  * `4B` `AC-6`: out-of-turn claims about `mod_heavy_periods`, `mod_tooth_bleed` and `body_puberty` are injected
    on live birth gates. All four pruned gates must stay `NOT_APPLICABLE` and must never surface in the patient
    PDF, the clinician note, or as "clear" negatives.
  * `4C` `jump_topic` from `birth_nicu` to `fam_cancer`, graded from the stored interpretation, and `birth_nicu`
    must not be corrupted. The first live runs settled the reducer's real rule: `_advance` only honours the router
    on a turn that also ANSWERS the active gate, and the volunteered family history is written by the same turn's
    `stated_` claim - so the graded outcome is a captured `fam_cancer` with `birth_nicu` untouched.
  * `4D` after `fam_cancer`, the harness stops talking: the real 15 s check-in and 30 s `SILENCE` close, the
    closing line once, and an `अधूरी कॉल` PDF whose `page2_unasked` omits every pruned gate.
  * `4E` a `?resume=` probe against that `SILENCE`-closed call: it must be REFUSED. `?resume=` restores a dropped
    line, never a call that ended on its own terms, so a new call is minted and the finished row keeps its
    `silence` reason and its report. (It used to reopen the row, wipe the reason, and re-finalize as caller_stop.)

`--selftest` proves every check is non-vacuous, offline and without TTS/network: it feeds synthetic whole and
broken states into the same `_check_a3` / `_check_a4` the live run uses, requires the whole state to be green,
and requires every mutation to flip exactly its own check.

    PYTHONPATH=. ~/pyenv/shishu/bin/python scripts/run_adversarial_shishu_untested.py --selftest
    PYTHONPATH=. ~/pyenv/shishu/bin/python scripts/run_adversarial_shishu_untested.py --case a3   # needs mic-free TTS
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import functools
import hashlib
import io
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
import wave
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Set, never defaulted - and set BEFORE `intake.db` is imported, because `db.DATA_DIR` is read at import time.
os.environ["DATA_DIR"] = str(ROOT / "intake" / "storage" / "adversarial_shishu_untested")

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from pypdf import PdfReader  # noqa: E402

from intake import db, engine, reports  # noqa: E402
from intake.models import (AnswerRecord, AnswerStatus, CallState, CloseReason, FollowupSlot, NoteText, Phase,  # noqa: E402
                           ProfilePatch, RosterInput, TurnCommand, TurnOutcome)
from intake.pack import load_pack  # noqa: E402

HOST, PORT = "127.0.0.1", 8796
BASE = f"http://{HOST}:{PORT}"
CALL_DEADLINE_SEC, RECV_TIMEOUT_SEC = 1800.0, 240.0
RATE, FRAME = 16_000, 1024        # 64 ms frames, the same cadence `ux_persona_bench.py` streams for v3
FRAME_SEC = FRAME / RATE
TAIL_LEAD_SEC, OPEN_GRACE_SEC, TAIL_QUIET_SEC = 0.25, 0.45, 0.15
PROMPT_MUTE_MAX_SEC = 5.0         # mirrors `live._PROMPT_MUTE_MAX_SEC`; asserted equal in the live path
DROP_WAIT_SEC = 1.5               # inside `live._RESUME_GRACE_SEC`
TAIL_RETRY_SEC = 12.0             # a tail reply that opened no turn is re-sent in the open window, and recorded
FAN_RMS, FAN_SEC = 0.0175, 1.5    # ~0.0175 RMS of ambient fan, the pause `3B` must not let split the turn
FITS_ONSET_LINE = "दो साल की उम्र में।"

TWINS_LINE = "मेरी दो जुड़वाँ लड़कियाँ हैं, दो साल की, दोनों का नाम मीरा है।"
# `which_item` is only reached when the CONCERN carries no resolvable child, so the reply names no child at all.
# The first live run proved why: the ASR wrote "मीरा" as "मेरा" and "झटके" as "छत के", the Decision pass then
# resolved `item_ref: all`, and the twins disambiguation was never exercised. A child-free concern reaches
# `which_item` deterministically; the twin roster built by `TWINS_LINE` is what makes that prompt ambiguous.
FITS_LINE = "हाँ, उसे झटके आए थे।"
BOTH_LINE = "दोनों को"
NO_LINE = "नहीं"
NICU_HEAD_LINE = "haan baby ko NICU mein rakha tha"
NICU_TAIL_LINE = "chaar din breathing problem thi"
NICU_NOTE = f"{NICU_HEAD_LINE} {NICU_TAIL_LINE}"
EMERGENCY_LINE = "रुको, बच्चे को अभी साँस नहीं आ रही है, होंठ नीले पड़ रहे हैं!"
PILLAR_TURN_LINE = "हाँ शुरू करें, जन्म के बाद बच्चे को पीलिया हुआ था।"

C, CL, U, UA, ANS = (AnswerStatus.CONCERN, AnswerStatus.CLEAR, AnswerStatus.UNKNOWN, AnswerStatus.UNANSWERED,
                     AnswerStatus.ANSWERED)
_LEAKS = ("next_question", "system cue", "ask only this", "active_qid", "last_answered_qid", "recorded_concerns",
          "duchenne", "hemophilia", "pompe", "gaucher", "scid", "wiskott", "ड्यूशेन", "हीमोफीलिया", "पॉम्पे",
          "गॉशर", "इम्यूनोडेफिशिएंसी", "म्यूकोपॉलीसेकेराइडोसिस")
_SILENCE_CLOSING = "लगता है आप अभी बात नहीं कर पा रही हैं"
_MIDDLE = ("मँझला", "मंझला", "मँझली", "बीच वाले", "बीच वाली")
# pypdf re-orders matras from the embedded font and drops inter-word spacing, so a pruned gate is graded on
# space-free fragments that really survive extraction - the same trick `run_adversarial_v2.py` uses for the
# glossary. Each pair is what the patient PDF would print if the gate were ever answered.
_PRUNE_PDF = {"mod_heavy_periods": ("बेटीको", "खूनआता"), "mod_tooth_bleed": ("दूधका", "टूटनेपरखून"),
              "body_puberty": ("बड़ेबच्चे", "उम्रकेबदलाव")}
_A3_PID, _A4_PID = "a3_shishu_voice_twins_deciding_drop", "a4_shishu_bridge_skip_jump_silence"
# Every check is now a real assertion: S-7.6 (3E) was fixed together with the resume guard (4E), so nothing here
# is allowed to fail on a correct run. A key in this map is a defect accepted on purpose - keep it empty.
_KNOWN_DEFECTS: dict[str, str] = {}


# ------------------------------------------------------------------------------------------------------------
# Scripting primitives
# ------------------------------------------------------------------------------------------------------------
def _say(text: str) -> tuple:
    return ("pcm", text)


def _txt(text: str) -> tuple:
    return ("text", text)


def _pause_say(head: str, tail: str) -> tuple:
    """One utterance with the fan noise spliced in between: `3B`'s mid-sentence pause."""
    return ("pcm_pause", (head, tail))


def _tail(text: str) -> tuple:
    """A reply that must start inside the prompt's last 350 ms of playback (`S-7.5`)."""
    return ("pcm_tail", text)


def _steps(*replies: tuple) -> tuple:
    return ("steps", replies)


def _norm(text: Any) -> str:
    return " ".join(str(text or "").split())


def _flat(text: Any) -> str:
    return "".join(str(text or "").split())


def _has(text: Any, *needles: str) -> bool:
    return any(needle in str(text or "") for needle in needles)


def _key(step: str) -> str:
    """The script key for a `step_token`: `qid`, `qid:which_item`, or `qid:fu<n>`."""
    parts = str(step or "").split(":")
    if len(parts) >= 2 and parts[1] == "which_item":
        return f"{parts[0]}:which_item"
    if len(parts) >= 3 and parts[1] == "followup":
        return f"{parts[0]}:fu{parts[2]}"
    return parts[0] if parts else ""


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

    def names(self) -> set[str]:
        return {i["check"] for i in self.items}


@dataclass
class Persona:
    """One scripted caller: a reply per typed cursor step, plus every observation its checks read.

    `pick()` is the only consumer the live loop uses; the self-test fills `asks` / `sent` / the audio windows by
    hand so the exact same check functions grade a synthetic call.
    """

    pid: str
    pack_id: str
    lines: dict[str, tuple]
    close: str
    check: Any
    default: tuple = ("pcm", NO_LINE)
    quiet_after: str = ""
    quiet_after_count: int = 1
    asks: list[tuple[str, str]] = field(default_factory=list)
    sent: list[dict[str, Any]] = field(default_factory=list)
    call_ids: list[int] = field(default_factory=list)
    which_item_prompt: str = ""
    tail_lead_ms: float | None = None
    tail_retried: bool = False
    dropped: bool = False
    muted_attack: dict[str, float] = field(default_factory=dict)
    listening_attack: dict[str, float] = field(default_factory=dict)
    telemetry_before_drop: dict[str, Any] = field(default_factory=dict)
    telemetry_after_resume: dict[str, Any] = field(default_factory=dict)
    telemetry_after_tail: dict[str, Any] = field(default_factory=dict)
    frames_before_drop: int = 0
    probe: dict[str, Any] = field(default_factory=dict)

    def cap(self, key: str) -> int:
        picked = self.lines.get(key)
        return len(picked[1]) if (picked and picked[0] == "steps") else 1

    def pick(self, key: str) -> tuple:
        picked = self.lines.get(key)
        if picked is None:
            return self.default
        if picked[0] == "steps":
            n = len(self.sent_on(key))
            return picked[1][min(n, len(picked[1]) - 1)]
        return picked

    def sent_on(self, key: str) -> list[dict[str, Any]]:
        return [item for item in self.sent if item["key"] == key]


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
    wav_left: np.ndarray | None = None
    wav_frames: int = 0
    wav_end: float = 0.0          # the wall clock of the recorder's `save()`; WAV frames are indexed from it
    proj: Any = None
    pinned: set[str] = field(default_factory=set)
    sequences: list[tuple[int, str, str]] = field(default_factory=list)
    interps: dict[int, Any] = field(default_factory=dict)

    def events_of(self, kind: str) -> list[dict[str, Any]]:
        return [e for e in self.events if str(e.get("type") or "") == kind]

    def spoken(self, role: str = "assistant") -> str:
        return " ".join(str(t.get("text") or "") for t in self.call.get("transcript") or []
                        if t.get("role") == role)

    def asked(self, qid: str) -> list[str]:
        return [step for step, asked in self.persona.asks if asked == qid]


def _rec(state: AnswerStatus, note: str = "", items: list[str] | None = None,
         followups: dict[str, tuple[AnswerStatus, str]] | None = None, redacted: bool = False) -> AnswerRecord:
    return AnswerRecord(state=state, item_ids=list(items or []),
                        note=NoteText.from_text(note, redacted=redacted) if note else NoteText(),
                        followups={k: FollowupSlot(state=s, note=NoteText.from_text(t, redacted=redacted) if t else NoteText())
                                   for k, (s, t) in (followups or {}).items()})


def _answered(ev: Ev, qid: str) -> tuple[AnswerStatus | None, str]:
    record = ev.state.answers.get(qid)
    return (record.state, record.verbatim_text) if record is not None else (None, "")


def _left_birth(p: Persona) -> bool:
    """After the `birth_nicu` turn the walk really moved, and not merely to the question it was already asking."""
    qids = [qid for _step, qid in p.asks]
    at = max((i for i, qid in enumerate(qids) if qid == "birth_nicu"), default=-1)
    return 0 <= at < len(qids) - 1 and qids[at + 1] not in ("birth_nicu",)

def _drop_answer(ev: Ev, qid: str) -> None:
    ev.state = ev.state.model_copy(update={"answers": {k: v for k, v in ev.state.answers.items() if k != qid}})


def _items(ev: Ev, qid: str) -> list[str]:
    record = ev.state.answers.get(qid)
    return list(record.item_ids) if record is not None else []


def _na(ev: Ev, qid: str) -> bool:
    return _answered(ev, qid)[0] is AnswerStatus.NOT_APPLICABLE


def _parity(ev: Ev, cl: Checklist) -> Any:
    """One projection must feed the stored report.

    The retained observations are passed in exactly as the live PDF build does: without them a call with an
    unlisted aside would "diverge" on `summary_report` for a reason that is the harness's, not the server's.
    """
    call, proj = ev.call, reports.project_summary(ev.pack, ev.state, ev.retained_ev)
    cl.check("project_summary parity", _norm(call.get("summary_report")) == _norm(proj.report_markdown),
             f"summary_report diverges ({len(_norm(call.get('summary_report')))} vs {len(_norm(proj.report_markdown))} chars)")
    ev.proj = proj
    return proj


def _common(ev: Ev, cl: Checklist, *, case: str) -> None:
    """The checks every live call owes: telemetry, spoken leaks, WAV, the PDF, and one projection."""
    telemetry, events = ev.call.get("telemetry") or {}, ev.events
    kinds = Counter(str(e.get("type") or "") for e in events)
    cl.check("zero tool_errors", not int(telemetry.get("tool_errors") or 0), f"tool_errors={telemetry.get('tool_errors')}")
    cl.check("zero error events", not kinds.get("error"),
             f"errors={[e.get('detail') for e in events if e.get('type') == 'error'][:5]}")
    auto = [str(e.get("detail") or "") for e in events if e.get("type") == "auto_close"]
    if case == "a3":
        # `auto_close` is only ever emitted by the watchdog, so any of it means the harness stalled.
        cl.check("closed on content, not the watchdog", not auto, f"auto_close={auto}")
    else:
        cl.check("watchdog closed on silence", auto == ["silence"], f"auto_close={auto}")
    leaks = [leak for leak in _LEAKS if leak.lower() in ev.spoken().lower()]
    cl.check("zero spoken leaks", not leaks, f"leaks={leaks}")
    cl.check("the report is written", bool(_norm(ev.call.get("summary_report"))), "summary_report is empty")
    cl.check("stereo 16 kHz WAV", ev.wav_shape == (2, RATE), f"wav is {ev.wav_shape}")
    cl.check("patient PDF exists", bool(ev.pages) and bool(_norm(ev.pdf_text)), "missing pdf")
    _parity(ev, cl)


# ------------------------------------------------------------------------------------------------------------
# Left-channel audio: the only honest witness to `S-7.6`
# ------------------------------------------------------------------------------------------------------------
def _wav_window(ev: Ev, start: float, end: float) -> np.ndarray | None:
    """The left-channel slice between two wall-clock instants, indexed back from the recorder's own end."""
    if ev.wav_left is None or ev.wav_frames <= 0:
        return None
    i0 = ev.wav_frames - int(round((ev.wav_end - start) * RATE))
    i1 = ev.wav_frames - int(round((ev.wav_end - end) * RATE))
    i0, i1 = max(0, i0), min(ev.wav_frames, max(i1, i0 + 1))
    return ev.wav_left[i0:i1]


def _speech_fraction(seg: np.ndarray | None) -> float:
    """Fraction of 20 ms sub-frames carrying speech-level energy (>= 0.02 RMS)."""
    if seg is None or seg.size < 320:
        return 0.0
    n = seg.size // 320
    block = seg[: n * 320].astype(np.float64).reshape(n, 320)
    rms = np.sqrt((block * block).mean(axis=1)) / 32768.0
    return float((rms >= 0.02).mean())


def _best_window(ev: Ev, start: float, dur: float, pad: float = 0.6,
                 slack: float = 5.0, step: float = 0.05) -> tuple[float, float]:
    """Best speech fraction within `slack` seconds of an instant, plus the shift that found it.

    The WAV is written when the socket dies or the call closes, so `call_completed` is *later* than the recorder's
    last frame by however long the PDF took. Both of 3E's windows are therefore located by the same search and
    then related to each other, which cancels that offset instead of trusting it.
    """
    best, shift = 0.0, 0.0
    n = int(slack / step)
    for k in range(-n, n + 1):
        moved = start + k * step
        frac = _speech_fraction(_wav_window(ev, moved - pad, moved + dur + pad))
        if frac > best:
            best, shift = frac, k * step
    return best, shift


# ------------------------------------------------------------------------------------------------------------
# Turn meta: the ASR text each Decision pass actually read
# ------------------------------------------------------------------------------------------------------------
def _turns(ev: Ev) -> list[tuple[str, str]]:
    """Every Decision pass the server recorded, as (step_in, the ASR text it read), in order."""
    out = []
    for event in ev.events:
        meta = event.get("meta") if isinstance(event.get("meta"), dict) else {}
        if str(event.get("type") or "") not in ("tool_call", "manual_nudge"):
            continue
        text = _norm(meta.get("stt"))
        if text:
            out.append((str(meta.get("step_in") or ""), text))
    return out


def _heard(ev: Ev, qid: str) -> str:
    """Everything the server wrote down for one question: the note, its follow-ups, and the ledger rows."""
    note = _answered(ev, qid)[1]
    rows = " ".join(text for _seq, step, text in ev.sequences if step.split(":")[0] == qid)
    return _norm(f"{note} {rows}")


# ------------------------------------------------------------------------------------------------------------
# Call 3 - shishu-hi, spoken: twins, a paused sentence, the playback tail, a drop in DECIDING, muted emergency
# ------------------------------------------------------------------------------------------------------------
def _check_a3(ev: Ev, cl: Checklist) -> dict[str, Any]:
    p, st = ev.persona, ev.state
    # The whole call is audio-only: a `{"type":"text"}` frame would show up as TEXT evidence, and no tap is scripted.
    typed = [src for src in ev.sources if src[0] in ("text", "tap")]
    cl.check("3 streamed real PCM only, never a text frame", not typed, f"typed rows={typed}")
    # 3A - two daughters, one name: the roster must keep both and the prompt must never invent a middle child.
    twins = list(st.roster)
    cl.check("3A twins stay two distinct children",
             len(twins) == 2 and len({c.id for c in twins}) == 2 and sorted(c.age_years for c in twins) == [2.0, 2.0]
             and all(c.tag == "female" for c in twins) and len({c.name or "" for c in twins}) == 1,
             f"roster={[(c.id, c.age_years, c.tag, c.name) for c in twins]}")
    # The prompt the caller would hear for THIS roster: the one the server really published when the disambiguation
    # was reached, else the pack's own renderer for the same state (a pure function of the roster and the prompts).
    prompt = _norm(p.which_item_prompt or engine.which_item_prompt(ev.pack, st))
    middle = [word for word in _MIDDLE if word in prompt]
    numbered = _has(prompt, "पहले") and _has(prompt, "दूसरे")
    cl.check("3A the twins prompt is numbered and never calls a twin middle",
             numbered and not middle, f"prompt={prompt!r} numbered={numbered} middle={middle}")
    cl.check("3A body_fits resolved CONCERN on both children",
             _answered(ev, "body_fits")[0] is C and _items(ev, "body_fits") == ["item_1", "item_2"],
             f"state={_answered(ev, 'body_fits')[0]} items={_items(ev, 'body_fits')}")
    # Both branches are acceptable - the model may answer "both" from the name alone - but the concern must never
    # be filed against one twin without the disambiguation having been offered.
    asked = any(step.startswith("body_fits:which_item") for step, _qid in p.asks)
    cl.check("3A the twins concern was settled for both children, not one",
             _items(ev, "body_fits") == ["item_1", "item_2"] and (asked or _answered(ev, "body_fits")[0] is C),
             f"asked_which_item={asked} items={_items(ev, 'body_fits')}")
    # 3B - one answer with a 1.5 s hole in it under a fan: one turn, both halves in the note.
    # A split sentence shows up twice: a second evidence row stamped on the SAME main step, or an attempt bump.
    nicu_passes = [text for step, text in _turns(ev) if step == "birth_nicu"]
    nicu_rows = [src for src in ev.sources if src[1] == "birth_nicu"]
    bumped = [step for step, _qid in p.asks if step.startswith("birth_nicu:try:")]
    # The ASR spells borrowed words as they sound - "चार दिन" for `chaar din`, "निकू/निकु/निको" for NICU - so a
    # live pass is graded on every shape a transcript really produced, never on the Latin script the line was
    # written in. Grading `"chaar din" in stt.lower()` failed a turn whose ASR was perfectly correct.
    only = nicu_passes[0] if len(nicu_passes) == 1 else ""
    fused = bool(only) and _has(only, "chaar din", "चार दिन") and _has(only, "NICU", "nicu", "निकू", "निकु", "निको")
    single = len(nicu_passes) == 1 and fused
    cl.check("3B one voice turn carried the paused sentence",
             (single if nicu_passes else (len(nicu_rows) == 1 and nicu_rows[0][0] == "voice")) and not bumped,
             f"passes={nicu_passes} rows={nicu_rows} bumped={bumped}")
    nicu_state, nicu_note = _answered(ev, "birth_nicu")
    cl.check("3B birth_nicu CONCERN keeps the four-day breathing detail",
             nicu_state is C and _has(nicu_note, "chaar din", "चार दिन") and _has(nicu_note, "breathing", "साँस", "ब्रीदिंग"),
             f"state={nicu_state} note={nicu_note[:90]!r}")
    heard = _heard(ev, "birth_nicu")
    # The ASR spells the borrowed word as it sounds - "निकू/निकु/निको" for NICU, "राखा/रखा ताहा" for rakha tha -
    # so the head is matched on every shape a live transcript actually produced, never on the Latin spelling.
    head = _has(heard, "NICU", "nicu", "निकू", "निकु", "निको", "काँच", "कांच")
    cl.check("3B both halves of the sentence fused into one note",
             head and _has(heard, "chaar din", "चार दिन"),
             f"heard={heard[:120]!r}")
    # 3C - S-7.5: the reply begins in the tail, the gate clears, nobody was interrupted.
    lead = p.tail_lead_ms
    cl.check("3C the tail reply started inside the 350 ms window",
             lead is not None and 0.0 <= lead <= 350.0 and not p.tail_retried,
             f"lead_ms={lead} tail_retried={p.tail_retried}")
    cl.check("3C birth_cord_bleed resolved CLEAR from the tail reply",
             _answered(ev, "birth_cord_bleed")[0] is CL, f"got {_answered(ev, 'birth_cord_bleed')[0]}")
    # Read from the tail's own era: 3E barges in on purpose minutes later, so the call's final count is not 3C's.
    tail_tel = p.telemetry_after_tail or (ev.call.get("telemetry") or {})
    interruptions = int(tail_tel.get("interruptions") or 0)
    barge_ins = [e for e in ev.events if e.get("type") == "barge_in"
                 and str(e.get("detail") or "") in ("", "birth_cord_bleed")]
    cl.check("3C no interruption was counted", interruptions == 0 and not barge_ins,
             f"interruptions={interruptions} barge_in={len(barge_ins)}")
    # 3D - the socket dies inside DECIDING, the share_token brings the same call back.
    cl.check("3D the same call id came back after the drop",
             len(p.call_ids) == 2 and p.call_ids[0] == p.call_ids[1] > 0, f"call_ids={p.call_ids}")
    cl.check("3D session_resumed recorded", bool(ev.events_of("session_resumed")),
             f"event types={sorted({str(e.get('type')) for e in ev.events})}")
    fam_state, fam_note = _answered(ev, "fam_cancer")
    fam_asks = ev.asked("fam_cancer")
    cl.check("3D fam_cancer resolved cleanly after the resume",
             fam_state in (CL, C) and 1 <= len(fam_asks) <= 3, f"state={fam_state} asks={fam_asks} note={fam_note[:60]!r}")
    cl.check("3D the WAV survived the reconnect",
             ev.wav_shape == (2, RATE) and p.frames_before_drop > 0 and ev.wav_frames > p.frames_before_drop,
             f"before={p.frames_before_drop} after={ev.wav_frames} shape={ev.wav_shape}")
    before, after = p.telemetry_before_drop, p.telemetry_after_resume
    cl.check("3D telemetry survived the drop",
             int(after.get("user_turns") or 0) >= int(before.get("user_turns") or 0) > 0
             and int(after.get("bot_turns") or 0) >= int(before.get("bot_turns") or 0) > 0,
             f"before user/bot={before.get('user_turns')}/{before.get('bot_turns')} "
             f"after={after.get('user_turns')}/{after.get('bot_turns')}")
    # 3E - S-7.6: speech over the muted prompt must still exist somewhere in the audit WAV.
    live_fraction, live_shift = _best_window(ev, float(p.listening_attack.get("start") or 0.0),
                                             float(p.listening_attack.get("dur") or 0.0))
    muted_fraction, muted_shift = _best_window(ev, float(p.muted_attack.get("start") or 0.0),
                                               float(p.muted_attack.get("dur") or 0.0))
    offset = live_shift - muted_shift          # the recorder's own clock offset, measured not assumed
    cl.check("3E a listening-window utterance reached the WAV left channel", live_fraction >= 0.25,
             f"speech fraction={live_fraction:.2f} (the measurement itself is broken)")
    # S-7.6 is fixed: `on_audio` hands the frame to the recorder before the mute gate returns, so the left channel
    # carries speech she made over SHISHU. No longer a known defect - a failure here is a regression.
    cl.check("3E muted-window audio reached the WAV left channel",
             muted_fraction >= 0.5 * max(live_fraction, 0.25),
             f"muted={muted_fraction:.2f} listening={live_fraction:.2f} offset={offset:+.2f}s - "
             f"S-7.6: a frame heard while the mic is muted must still reach the audit WAV")
    cl.check("3E the call closed EMERGENCY with her own words",
             str(st.close_reason or "") == CloseReason.EMERGENCY.value
             and _has(st.emergency_note, "साँस", "होंठ", "नीले"), f"close={st.close_reason} note={st.emergency_note[:80]!r}")
    spoken = ev.spoken()
    cl.check("3E the 108 line was spoken", "108" in spoken, f"spoken={spoken[-160:]!r}")
    cl.check("3E the emergency PDF carries the hospital banner",
             "108" in _flat(ev.pdf_text) and _has(_flat(ev.pdf_text), "अस्पताल"), "PDF has no emergency band")
    return {"asks": len(p.asks), "call_ids": p.call_ids, "tail_lead_ms": p.tail_lead_ms,
            "muted_fraction": round(muted_fraction, 2), "listening_fraction": round(live_fraction, 2),
            "wav_offset_sec": round(offset, 2),
            "roster": [(c.id, c.age_years, c.tag, c.name) for c in st.roster]}


# ------------------------------------------------------------------------------------------------------------
# Call 4 - shishu-hi, text/quick: an upfront stated_ concern, pruned gates, a jump, then the real silence close
# ------------------------------------------------------------------------------------------------------------
def _check_a4(ev: Ev, cl: Checklist) -> dict[str, Any]:
    p, st, proj = ev.persona, ev.state, ev.proj
    # 4A - the bridge turn carries a concern the pack would otherwise ask about later.
    cl.check("4A intro_bridge answered CLEAR", _answered(ev, "intro_bridge")[0] is CL,
             f"got {_answered(ev, 'intro_bridge')[0]}")
    cl.check("4A birth_jaundice auto-filled CONCERN from the bridge turn",
             _answered(ev, "birth_jaundice")[0] is C and _has(_answered(ev, "birth_jaundice")[1], "पीलिया"),
             f"state={_answered(ev, 'birth_jaundice')[0]} note={_answered(ev, 'birth_jaundice')[1][:60]!r}")
    cl.check("4A birth_jaundice never asked at main", not ev.asked("birth_jaundice"),
             f"asked={ev.asked('birth_jaundice')}")
    jaundice = st.answers.get("birth_jaundice")
    cl.check("4A the bridge remark stayed off her copy",
             jaundice is not None and bool(jaundice.note.verbatim) and not _norm(jaundice.note.patient_safe),
             f"patient_safe={jaundice.note.patient_safe[:60]!r}" if jaundice else "no jaundice record")
    # 4B - AC-6: a pruned gate can never be woken by an out-of-turn claim.
    cl.check("4B ever_pregnant stays NOT_APPLICABLE", _na(ev, "ever_pregnant"),
             f"got {_answered(ev, 'ever_pregnant')[0]}")
    pruned = ("mod_heavy_periods", "mod_tooth_bleed", "body_puberty")
    woke = [qid for qid in pruned if not _na(ev, qid)]
    cl.check("4B the three pruned child gates stay NOT_APPLICABLE", not woke,
             f"woken={[(qid, str(_answered(ev, qid)[0])) for qid in woke]}")
    report = _norm(ev.call.get("summary_report"))
    leaked = [qid for qid in (*pruned, "ever_pregnant") if _norm(ev.pack.by_id[qid].need) in report]
    cl.check("4B pruned gates never reached the report", not leaked, f"leaked={leaked}")
    pdf = _flat(ev.pdf_text)
    leaked_pdf = [qid for qid, needles in _PRUNE_PDF.items() if all(n in pdf for n in needles)]
    cl.check("4B pruned gates never reached the patient PDF", not leaked_pdf, f"leaked={leaked_pdf}")
    asked = [qid for qid in pruned if ev.asked(qid)]
    cl.check("4B pruned gates were never asked", not asked, f"asked={asked}")
    # 4C - jump_topic. The reducer honours the router only on a turn that also ANSWERS the active gate, and the
    # volunteered family history is written by the same turn's `stated_` claim - so the observable outcome is a
    # captured `fam_cancer` with `birth_nicu` untouched, plus the router's own read on the ledger.
    floor = float((ev.pack.decisions.get("thresholds") or {}).get("choice_min_confidence", 0.52))
    jumped = [seq for seq, interp in ev.interps.items()
              if getattr(interp, "jump_topic", False) and float((getattr(interp, "next_qid_probs", None) or {}).get("fam_cancer", 0.0)) >= floor]
    cl.check("4C jump_topic was read on the birth_nicu turn", bool(jumped),
             f"interps={[(s, getattr(i, 'jump_topic', None), (getattr(i, 'next_qid_probs', None) or {}).get('fam_cancer')) for s, i in ev.interps.items()]}")
    nicu_state, nicu_note = _answered(ev, "birth_nicu")
    nicu_asks = ev.asked("birth_nicu")
    clean = (nicu_state in (CL, C) and not _has(nicu_note, "कैंसर")) or (
        nicu_state is None and len(nicu_asks) == 2 and "try:1" in nicu_asks[-1])
    cl.check("4C birth_nicu was not corrupted by the jump", clean,
             f"state={nicu_state} asks={nicu_asks} note={nicu_note[:60]!r}")
    cancer_state, cancer_note = _answered(ev, "fam_cancer")
    who = (ev.state.answers.get("fam_cancer") or AnswerRecord(state=CL)).followups.get("fam_who")
    cl.check("4C fam_cancer resolved CONCERN with the family history",
             cancer_state is C and (_has(cancer_note, "कैंसर") or (who is not None and bool(_norm(who.note.verbatim)))),
             f"state={cancer_state} note={cancer_note[:60]!r} fam_who={getattr(who, 'state', None)}")
    # Captured out of turn, not by walking to it twice: whichever branch the reducer took, the family topic is in
    # exactly once and the walk did not stall on birth_nicu.
    cl.check("4C fam_cancer was not asked twice", len(ev.asked("fam_cancer")) <= 1, f"asks={ev.asked('fam_cancer')}")
    cl.check("4C the cursor left birth_nicu after the jump", _left_birth(p),
             f"asks={[q for _s, q in p.asks]}")
    # 4D - the real timers: 15 s check-in, 30 s close, one closing line, an honestly incomplete PDF.
    check_ins = ev.events_of("silence_check_in")
    detail = str(check_ins[0].get("detail") or "") if check_ins else ""
    waited = int("".join(ch for ch in detail if ch.isdigit()) or 0)
    cl.check("4D silence_check_in was emitted", bool(check_ins) and 15 <= waited < 25, f"events={[e.get('type') for e in ev.events]}")
    cl.check("4D close_reason is silence", str(st.close_reason or "") == CloseReason.SILENCE.value,
             f"close_reason={st.close_reason}")
    closing_rows = [t for t in ev.call.get("transcript") or []
                    if t.get("role") == "assistant" and _SILENCE_CLOSING in str(t.get("text") or "")]
    cl.check("4D the closing line was spoken exactly once", len(closing_rows) == 1, f"rows={len(closing_rows)}")
    cl.check("4D the patient PDF is marked अधूरी कॉल", "अधूरी" in _flat(ev.pdf_text), "PDF has no incomplete badge")
    unasked_labels = [item if isinstance(item, str) else str(item.get("question") or "")
                      for group in (proj.page2_unasked or ()) for item in (group.get("items") or ())]
    # `page2_unasked` prints a question's `stem`, so the NA filter must compare against that same wording.
    na_labels = [_norm(ev.pack.by_id[qid].stem) for qid in (*pruned, "ever_pregnant")]
    dirty = [label for label in unasked_labels if any(na and na in _norm(label) for na in na_labels)]
    cl.check("4D page2_unasked omits every NOT_APPLICABLE gate",
             bool(unasked_labels) and not dirty, f"unasked={len(unasked_labels)} dirty={dirty[:3]}")
    cl.check("4D the report records the silence close",
             "Close reason: silence." in _norm(ev.call.get("summary_report")), "the report lost the close reason")
    # 4E - a SILENCE-closed call is finished: `?resume=` restores a dropped line, never a call that ended on its own
    # terms. It used to reopen this row, wipe its reason, and re-finalize it as CALLER_STOP.
    probe = p.probe or {}
    reopened = bool(probe.get("reopened"))
    finished = p.call_ids[-1] if p.call_ids else 0
    cl.check("4E the post-silence resume probe was recorded",
             bool(probe.get("attempted")) and bool(probe.get("call_id")),
             f"probe={ {k: v for k, v in probe.items() if k != 'frames'} }")
    cl.check("4E the resume was refused and minted a new call",
             bool(probe.get("attempted")) and not reopened and probe.get("call_id") != finished,
             f"reopened={reopened} probe_call={probe.get('call_id')} finished={finished}")
    cl.check("4E the finished call kept its silence reason and report",
             str(probe.get("close_reason_after") or "") == CloseReason.SILENCE.value
             and str(probe.get("status_after") or "") == "completed",
             f"after status={probe.get('status_after')!r} close_reason={probe.get('close_reason_after')!r}")
    return {"call_ids": p.call_ids, "asks": len(p.asks), "jumped": jumped,
            "resume_reopened_silence_call": reopened, "probe_note": probe.get("note", "")}


# ------------------------------------------------------------------------------------------------------------
# Live plumbing
# ------------------------------------------------------------------------------------------------------------
_CACHE = Path(db.DATA_DIR) / "tts_cache"


def _pcm(text: str) -> np.ndarray:
    """One sentence as 16 kHz mono int16, sha1-cached so a rerun only pays for new sentences."""
    path = _CACHE / f"{hashlib.sha1(text.encode('utf-8')).hexdigest()}.pcm"
    if path.is_file() and path.stat().st_size >= 2:
        return np.frombuffer(path.read_bytes(), dtype=np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="shishu-tts-") as tmp:
        aiff, raw = Path(tmp) / "say.aiff", Path(tmp) / "say.pcm"
        subprocess.run(["say", "-v", "Lekha", "-o", str(aiff), text], check=True, capture_output=True, timeout=120)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(aiff), "-ar", str(RATE), "-ac", "1",
                        "-f", "s16le", str(raw)], check=True, capture_output=True, timeout=120)
        pcm = np.frombuffer(raw.read_bytes(), dtype=np.int16).copy()
    if pcm.size == 0:
        raise RuntimeError("say produced no audio (macOS TTS needs to run outside the sandbox)")
    loud = np.flatnonzero(np.abs(pcm.astype(np.int32)) >= 250)   # trim silence, keep 20 ms of plosive pad
    trimmed = pcm[max(0, int(loud[0]) - 320): int(loud[-1]) + 321] if loud.size else pcm
    path.write_bytes(trimmed.astype(np.int16).tobytes())
    return trimmed


def _fan(seconds: float = FAN_SEC, rms: float = FAN_RMS) -> np.ndarray:
    """A steady room fan: white noise at the requested RMS, the floor a real call is recorded over."""
    rng = np.random.default_rng(7)
    return np.clip(rng.normal(0.0, rms * 32768.0, int(seconds * RATE)), -32768, 32767).astype(np.int16)


def _read_pdf(call_id: int) -> tuple[int, str]:
    path = Path(db.DATA_DIR) / "pdfs" / f"call-{int(call_id)}.pdf"
    if not path.is_file() or not path.read_bytes()[:4] == b"%PDF":
        return 0, ""
    reader = PdfReader(io.BytesIO(path.read_bytes()))
    return len(reader.pages), _norm(" ".join(" ".join(p.extract_text() or "" for p in reader.pages).split()))


def _wav_info(call: dict[str, Any]) -> tuple[tuple[int, int] | None, np.ndarray | None, int]:
    path = Path(db.DATA_DIR) / "audio" / str(call.get("audio_path") or "")
    if not path.is_file():
        return None, None, 0
    try:
        with wave.open(str(path), "rb") as handle:
            channels, rate, frames = handle.getnchannels(), handle.getframerate(), handle.getnframes()
            raw = handle.readframes(frames)
    except (wave.Error, EOFError, OSError):
        return None, None, 0
    samples = np.frombuffer(raw, dtype=np.int16)
    left = samples.reshape(-1, channels)[:, 0].copy() if channels == 2 else None
    return (channels, rate), left, frames


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


class Mic:
    """One open phone line: 64 ms frames of her voice, exact digital silence whenever her mouth is closed.

    A tagged utterance reports the instant its FIRST frame really reached the socket, which is the only honest
    zero point for `S-7.5`: queueing then polling would report a time her voice had not yet been sent at.
    """

    def __init__(self, send) -> None:
        self._send = send
        self._frames: deque[tuple[bytes, str]] = deque()
        self._silence = base64.b64encode(bytes(FRAME * 2)).decode("ascii")
        self.pending = 0
        self.sent_at: dict[str, float] = {}
        self.task: asyncio.Task | None = None

    def push(self, pcm: np.ndarray, tag: str = "") -> None:
        flat = np.asarray(pcm, dtype=np.int16).tobytes()
        step = FRAME * 2
        chunks = [flat[i:i + step] for i in range(0, len(flat), step)]
        if chunks and len(chunks[-1]) < step:
            chunks[-1] = chunks[-1].ljust(step, b"\x00")
        self._frames.extend((chunk, tag) for chunk in chunks)
        self.pending = len(self._frames)

    async def run(self) -> None:
        next_at = time.perf_counter()
        while True:
            now, delay = time.perf_counter(), next_at - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(min(delay, 0.02))
                continue
            if self._frames:
                chunk, tag = self._frames.popleft()
                if tag and tag not in self.sent_at:
                    self.sent_at[tag] = time.perf_counter()
                await self._send({"type": "audio", "data": base64.b64encode(chunk).decode("ascii"),
                                  "sample_rate": RATE})
                self.pending = len(self._frames)
            else:
                await self._send({"type": "audio", "data": self._silence, "sample_rate": RATE})
            next_at += FRAME_SEC
            if next_at < now - 1.0:
                next_at = time.perf_counter()

    async def until_closed(self, limit: float = 60.0) -> None:
        end = time.perf_counter() + limit
        while self.pending and time.perf_counter() < end:
            await asyncio.sleep(0.02)


class Runner:
    """One live call: streams the scripted audio, and hands every frame's side effects to the persona."""

    def __init__(self, persona: Persona, pack, cookie: str) -> None:
        self.persona, self.pack, self.cookie = persona, pack, cookie
        self.ws = None
        self.call_id = 0
        self.share_token = ""
        self.step, self.qid, self.epoch = "", "", 0
        self.playback_end = 0.0
        self.last_audio_at = 0.0
        self.bot_done = 0.0
        self.step_ready = False
        self.replied: dict[str, int] = {}
        self.in_flight: set[str] = set()
        self.tel: dict[str, Any] = {}
        self.done = False
        self.completed_at = 0.0
        self.started = time.perf_counter()
        self.deadline = self.started + CALL_DEADLINE_SEC
        self.mic = Mic(self._send)
        self.attack: asyncio.Task | None = None
        self.awaiting_3e = False
        self.quiet_until = 0.0
        self.errors: list[str] = []

    # -- transport ----------------------------------------------------------------------------------------
    async def connect(self, resume: str = "") -> None:
        import websockets

        url = f"ws://{HOST}:{PORT}/ws/live?pack={self.pack.id}" + (f"&resume={resume}" if resume else "")
        self.ws = await websockets.connect(url, additional_headers={"Cookie": self.cookie},
                                           open_timeout=30, max_size=8_000_000)

    async def _send(self, payload: dict[str, Any]) -> None:
        try:
            await self.ws.send(json.dumps(payload, ensure_ascii=False))
        except Exception:  # noqa: BLE001 - a closed socket is reported by the receive loop
            pass

    async def _read(self, limit: float = RECV_TIMEOUT_SEC) -> dict[str, Any] | None:
        try:
            raw = await asyncio.wait_for(self.ws.recv(), timeout=limit)
        except asyncio.TimeoutError:
            return None
        except Exception:  # noqa: BLE001 - the socket died; the loop decides what that means
            return None
        try:
            msg = json.loads(raw)
        except (TypeError, ValueError):
            return None
        return msg if isinstance(msg, dict) else None

    def _note_audio(self, msg: dict[str, Any]) -> None:
        try:
            raw = base64.b64decode(msg.get("data") or "")
        except Exception:  # noqa: BLE001 - a malformed chunk must not kill the call
            return
        rate = int(msg.get("sample_rate") or 24000) or 24000
        if raw:
            self.playback_end = max(time.perf_counter(), self.playback_end) + len(raw) / (2.0 * rate)
            self.last_audio_at = time.perf_counter()

    # -- what the caller does next ------------------------------------------------------------------------
    def _open_at(self, tail: bool) -> float:
        """The instant to open her mouth. Normally the prompt's playback must finish first; `S-7.5` lets the
        reply begin inside the prompt's own last 350 ms, so it aims `TAIL_LEAD_SEC` before playback ends."""
        return self.playback_end - TAIL_LEAD_SEC if tail else max(self.playback_end + OPEN_GRACE_SEC,
                                                                  self.bot_done + 0.15)

    async def _wait_for_window(self, tail: bool) -> None:
        if tail:
            # The chunks keep arriving while they are sent, so `playback_end` grows: wait for the prompt's audio to
            # stop, then aim inside the tail it really ends with. Without this the reply lands after playback and
            # never tests `S-7.5` at all (observed live: -4 ms instead of +200 ms).
            while not self.done and time.perf_counter() - self.last_audio_at < TAIL_QUIET_SEC:
                await asyncio.sleep(0.01)
        target = self._open_at(tail)
        while not self.done and time.perf_counter() < target:
            await asyncio.sleep(0.01)

    async def _reply_safety(self, step: str) -> None:
        """A step whose `bot_turn_complete` never arrives would strand the caller: answer it anyway, once."""
        await asyncio.sleep(25.0)
        if self.done or step != self.step or self.step_ready or self.attack:
            return
        if time.perf_counter() >= self.quiet_until:
            return  # 4D: the harness has deliberately gone silent
        print(f"  [{step}] no bot_turn_complete in 25s; answering the published step anyway", flush=True)
        await self._reply()

    async def _reply(self) -> None:
        step = self.step
        if not step or step in self.in_flight:
            return
        key = _key(step)
        if self.replied.get(step, 0) >= self.persona.cap(key):
            return
        self.in_flight.add(step)
        try:
            line = self.persona.pick(key)
            mode, payload = line[0], line[1]
            await self._wait_for_window(mode == "pcm_tail")
            if self.done or step != self.step:
                return
            if mode == "text":
                await self._send({"type": "text", "text": payload})
            elif mode == "quick":
                await self._send({"type": "quick_reply", "value": payload, "step": step, "epoch": self.epoch})
            elif mode == "pcm":
                self.mic.push(_pcm(payload))
            elif mode == "pcm_pause":
                head, tail = payload
                self.mic.push(np.concatenate([_pcm(head), _fan(), _pcm(tail)]))
            elif mode == "pcm_tail":
                # S-7.5: `tail_lead_ms` is measured in `Mic.run` from the frame that really hit the socket.
                self.mic.push(_pcm(payload), tag="tail")
                asyncio.ensure_future(self._tail_measure(payload))
            else:
                raise RuntimeError(f"unknown mode {mode}")
            self.replied[step] = self.replied.get(step, 0) + 1
            self.persona.sent.append({"key": key, "step": step, "mode": mode,
                                      "payload": payload if isinstance(payload, str) else "|".join(payload)})
            if (self.persona.quiet_after and key == self.persona.quiet_after
                    and len(self.persona.sent_on(key)) >= self.persona.quiet_after_count):
                self.quiet_until = float("inf")
                # 3E arms here: the very next published step is the one the emergency must interrupt while muted.
                self.awaiting_3e = self.persona.pid == _A3_PID
        finally:
            self.in_flight.discard(step)

    async def _tail_measure(self, text: str) -> None:
        """Read the tail's real lead from the socket clock, then re-send it in the open window if it opened no turn.

        `S-7.5` grades how early her voice reached the server relative to the prompt's own playback end, so the
        first frame's send time is the measurement - not the instant it was queued.
        """
        step = self.step
        for _ in range(200):
            if "tail" in self.mic.sent_at:
                break
            await asyncio.sleep(0.005)
        sent = self.mic.sent_at.get("tail")
        if sent is not None:
            self.persona.tail_lead_ms = round((self.playback_end - sent) * 1000.0, 1)
        await asyncio.sleep(TAIL_RETRY_SEC)
        if self.done or self.step != step or self.replied.get(step, 0) != 1:
            return
        self.persona.tail_retried = True
        self.mic.push(_pcm(text))
        self.replied[step] = self.replied.get(step, 0) + 1
        self.persona.sent.append({"key": _key(step), "step": step, "mode": "pcm_retry", "payload": text})
        print("  tail reply opened no turn; re-sent in the open window", flush=True)

    async def _attack_muted_emergency(self, prompted_at: float) -> None:
        """S-7.6: the same real emergency, once inside the mute and once in the open window.

        The first attempt lands while `live._mic_muted` is true (`PROMPTING`, < 5 s, and the prompt's own audio
        still playing). `live.py:295` returns before `recorder.add_user_pcm`, so this audio can only exist in the
        WAV if the mute path changed - that is the whole point of the check.
        """
        pcm = _pcm(EMERGENCY_LINE)
        self.persona.muted_attack = {"start": time.perf_counter(), "dur": len(pcm) / RATE,
                                     "muted_for": round(PROMPT_MUTE_MAX_SEC - (time.perf_counter() - prompted_at), 2)}
        self.mic.push(pcm)
        await self.mic.until_closed()
        # The second attempt waits for a genuinely open window: the mute valve has elapsed AND playback finished.
        while not self.done:
            now = time.perf_counter()
            if now - prompted_at >= PROMPT_MUTE_MAX_SEC + 0.3 and now >= self.playback_end + 0.3:
                break
            await asyncio.sleep(0.05)
        if self.done:
            return
        self.persona.listening_attack = {"start": time.perf_counter(), "dur": len(pcm) / RATE}
        self.mic.push(pcm)

    # -- frames -------------------------------------------------------------------------------------------
    async def _handle(self, msg: dict[str, Any]) -> None:
        kind = msg.get("type")
        if kind == "audio":
            self._note_audio(msg)
        elif kind == "clear":
            self.playback_end = 0.0
        elif kind == "session_started":
            self.call_id = int(msg.get("call_id") or 0) or self.call_id
            self.share_token = str(msg.get("share_token") or "") or self.share_token
            self.persona.call_ids.append(self.call_id)   # once per connection: the initial one and the resume
            print(f"  session started call {self.call_id}", flush=True)
        elif kind == "question_state":
            self.step, self.qid, self.epoch = str(msg.get("step") or ""), str(msg.get("qid") or ""), int(msg.get("epoch") or 0)
            self.persona.asks.append((self.step, self.qid))
            self.step_ready = False
            print(f"  [{self.step}]", flush=True)
            if self.qid == "body_fits" and ":which_item" in self.step:
                state = db.call_state_from_row(await asyncio.to_thread(db.get_call, self.call_id))
                self.persona.which_item_prompt = engine.active_prompt(self.pack, state)
            if self.awaiting_3e:
                self.awaiting_3e, self.quiet_until = False, float("inf")
                self.attack = asyncio.ensure_future(self._attack_muted_emergency(time.perf_counter()))
            asyncio.ensure_future(self._reply_safety(self.step))
        elif kind == "bot_turn_complete":
            if str(msg.get("step") or "") in ("", self.step):
                self.bot_done, self.step_ready = time.perf_counter(), True
                if not self.attack and time.perf_counter() >= self.quiet_until:
                    asyncio.ensure_future(self._reply())
        elif kind == "thinking":
            if self.qid == "fam_cancer" and not self.persona.dropped:
                await self._drop_and_resume()
        elif kind in ("reconnecting", "reconnected"):
            # `live._reconnect` ends with `_enter_prompting()` + `_speak()`, so it RE-CUES the step that was open
            # when the socket died. Everything this harness measured against the old prompt is void from here:
            # `playback_end` belongs to audio that is no longer the line she is hearing, and the tail reply is armed
            # against it. Call 4 proved the cost - a swap landed on `birth_cord_3weeks -> birth_cord_bleed`, the
            # prompt was re-spoken, and 3C graded `lead_ms=-55.8` against a stale zero point. So forget the old
            # clock and let the re-cued prompt's own playback re-arm it.
            if str(msg.get("type")) == "reconnected":
                # Only the CLOCK is stale. `_reconnect` re-speaks the cue WITHOUT re-sending `question_state`
                # (only `_publish_and_cue` does that, and `_reconnect` never calls it), so clearing `step_ready`
                # here left it False forever: `bot_turn_complete` is its only setter and that needs a Decision pass
                # the re-cue does not trigger. Call 7 wedged on exactly that. The step itself has not changed, so
                # `step_ready` must survive; only the playback timeline is void.
                self.playback_end, self.last_audio_at, self.bot_done = 0.0, 0.0, 0.0
                print("  reconnected: prompt clock reset (step unchanged)", flush=True)
        elif kind == "telemetry":
            self.tel = msg.get("telemetry") or self.tel
            event = str((msg.get("last_event") or {}).get("type") or "")
            if self.persona.dropped and event in ("session_resumed", "turn_complete") and not self.persona.telemetry_after_resume:
                self.persona.telemetry_after_resume = dict(self.tel)
            # 3C's "nobody was interrupted" must be read from the tail's own era: 3E barges in on purpose later.
            if (self.persona.tail_lead_ms is not None and not self.persona.telemetry_after_tail
                    and event in ("turn_complete", "bot_first_audio")):
                self.persona.telemetry_after_tail = dict(self.tel)
        elif kind == "call_completed":
            self.completed_at = time.perf_counter()
            self.done = True
            print("  call completed", flush=True)
        elif kind in ("report_failed", "error"):
            self.errors.append(f"{kind}: {msg.get('message') or msg.get('detail') or ''}")
            print(f"  {kind}: {msg.get('message') or msg.get('detail')}", flush=True)
            if kind == "report_failed":
                self.done = True

    async def _drop_and_resume(self) -> None:
        """3D: an abrupt socket death inside DECIDING, then the call's own share_token rejoins 1.5 s later."""
        persona, call = self.persona, await asyncio.to_thread(db.get_call, self.call_id) or {}
        persona.dropped = True
        persona.telemetry_before_drop = dict(call.get("telemetry") or self.tel)
        print(f"  dropping the socket inside DECIDING on {self.qid} (call {self.call_id})", flush=True)
        try:
            await self.ws.close()
        except Exception:  # noqa: BLE001 - a socket already gone is exactly what a network drop looks like
            pass
        await asyncio.sleep(DROP_WAIT_SEC)
        token = self.share_token or str(call.get("share_token") or "")
        _, _, frames_before = _wav_info(await asyncio.to_thread(db.get_call, self.call_id) or {})
        persona.frames_before_drop = frames_before
        await self.connect(token)
        print(f"  reconnected with the call's own share_token ({frames_before} frames kept)", flush=True)

    # -- the loop -----------------------------------------------------------------------------------------
    async def run(self) -> dict[str, Any]:
        await self._prewarm()
        await self.connect()
        self.mic.task = asyncio.ensure_future(self.mic.run())
        try:
            while not self.done and time.perf_counter() < self.deadline:
                msg = await self._read()
                if msg is None:
                    if self.done:
                        break
                    print("  socket went quiet", flush=True)
                    break
                await self._handle(msg)
        finally:
            if self.mic.task is not None:
                self.mic.task.cancel()
            if self.attack is not None and not self.attack.done():
                self.attack.cancel()
            try:
                await self.ws.close()
            except Exception:  # noqa: BLE001
                pass
        # Snapshot the finished call FIRST: the 4E probe reopens it, and that must not rewrite what 4D graded.
        ev = await self._collect()
        if self.persona.pid == _A4_PID and self.call_id:
            await self._probe_resume(self.share_token)
        return self._grade(ev)

    async def _prewarm(self) -> None:
        """Synthesise every line before the socket opens: TTS inside a turn would stall the microphone."""
        lines = {payload if isinstance(payload, str) else piece
                 for _key, entry in self.persona.lines.items()
                 for payload in [entry[1] if entry[0] != "steps" else next((r[1] for r in entry[1]), "")]
                 for piece in ([payload] if isinstance(payload, str) else payload) if piece}
        for text in sorted(lines):
            try:
                await asyncio.to_thread(_pcm, text)
            except Exception as exc:  # noqa: BLE001 - a broken TTS is reported by the call, not a crash
                print(f"  TTS failed for {text[:24]!r}: {exc}", flush=True)

    async def _collect(self) -> Ev:
        call = await asyncio.to_thread(db.get_call, self.call_id) or {}
        pages, text = _read_pdf(self.call_id)
        shape, left, frames = _wav_info(call)
        decisions = await asyncio.to_thread(db.list_policy_decisions, self.call_id)
        evidence = await asyncio.to_thread(db.list_turn_evidence, self.call_id)
        interps = await asyncio.to_thread(db.list_turn_interpretations, self.call_id)
        retained = await asyncio.to_thread(db.list_retained_observations, self.call_id)
        return Ev(pack=self.pack, call=call, state=db.call_state_from_row(call), persona=self.persona,
                  decisions=[(d.disposition.value, d.reason.value, str(d.target_qid or ""), d.cursor_action.value)
                             for d in decisions],
                  sources=[(e.source.value, e.snapshot_step) for e in evidence],
                  retained=[str(o.transcript) for o in retained], retained_ev=retained,
                  events=list(call.get("events") or []), pages=pages, pdf_text=text, wav_shape=shape,
                  wav_left=left, wav_frames=frames, wav_end=self.completed_at or time.perf_counter(),
                  sequences=[(e.sequence, e.snapshot_step, e.transcript) for e in evidence],
                  interps={i.sequence: i.interpretation for i in interps})

    def _grade(self, ev: Ev) -> dict[str, Any]:
        cl = Checklist()
        if ev.call:
            cl.check(f"close_reason {self.persona.close}", str(ev.state.close_reason or "") == self.persona.close,
                     f"close_reason={ev.state.close_reason}")
            try:
                _common(ev, cl, case="a3" if self.persona.pid == _A3_PID else "a4")
                detail = self.persona.check(ev, cl) or {}
            except Exception as exc:  # noqa: BLE001
                cl.check("checklist ran to the end", False, f"{type(exc).__name__}: {exc}")
                detail = {}
        else:
            cl.check("the call row could be read", False, f"no call {self.call_id}")
            detail = {}
        known = sorted(name for name in cl.failing() if name in _KNOWN_DEFECTS)
        problems = [p for p in cl.problems if p.split(":", 1)[0] not in _KNOWN_DEFECTS]
        return {"id": self.persona.pid, "pack": self.pack.id, "call_id": self.call_id, "ok": not problems,
                "problems": problems, "known_defects": known, "checklist": cl.items, "detail": detail,
                "close_reason": str(ev.state.close_reason or ""), "duration_sec": ev.call.get("duration_sec"),
                "wall_sec": round(time.perf_counter() - self.started, 1), "errors": self.errors,
                "frames_before_drop": self.persona.frames_before_drop, "wav_frames": ev.wav_frames,
                "tail_lead_ms": self.persona.tail_lead_ms,
                "probe": self.persona.probe}

    async def _probe_resume(self, token: str) -> None:
        """4E: does `?resume=` reopen a call that closed on SILENCE? It must not: a new call, the old row intact."""
        probe: dict[str, Any] = {"attempted": False, "reopened": False, "call_id": 0, "frames": []}
        self.persona.probe = probe
        if not token:
            probe["note"] = "no share_token on the call; nothing to probe"
            return
        finished = self.call_id
        try:
            await self.connect(token)
            probe["attempted"] = True
            end = time.perf_counter() + 25.0
            while time.perf_counter() < end:
                msg = await self._read(limit=max(1.0, end - time.perf_counter()))
                if msg is None:
                    break
                kind = msg.get("type")
                probe["frames"].append(kind)
                if kind == "session_started":
                    probe["call_id"] = int(msg.get("call_id") or 0)
                    probe["reopened"] = probe["call_id"] == finished
                if kind == "question_state":
                    probe["step"] = str(msg.get("step") or "")
                    break
                if kind == "error":
                    probe["error_frame"] = str(msg.get("message") or "")
                    break
            await self._send({"type": "hangup"})
            await asyncio.sleep(2.0)
            after = await asyncio.to_thread(db.get_call, finished) or {}
            probe["status_after"] = str(after.get("status") or "")
            probe["close_reason_after"] = str(db.call_state_from_row(after).close_reason or "")
            probe["note"] = ("a closed call is not resumable: a new call is minted and the finished row keeps "
                             "its silence reason and its report" if not probe["reopened"] else
                             "REGRESSION: live.py reopened a SILENCE-closed call and rewrote its close reason")
        except Exception as exc:  # noqa: BLE001 - a refused probe is a result, not a crash
            probe["error"] = f"{type(exc).__name__}: {exc}"
            probe["note"] = "the resume probe could not connect"
        finally:
            try:
                await self.ws.close()
            except Exception:  # noqa: BLE001
                pass
        print(f"  4E resume probe: reopened={probe['reopened']} call_id={probe['call_id']} "
              f"after={probe.get('status_after')}/{probe.get('close_reason_after')}", flush=True)


# ------------------------------------------------------------------------------------------------------------
# Personas
# ------------------------------------------------------------------------------------------------------------
def _a3_lines() -> dict[str, tuple]:
    return {
        "intro_name": _say("मेरा नाम रेखा है।"),
        "intro_married": _say("हाँ, मेरी शादी हो गई है।"),
        "intro_children": _say(TWINS_LINE),
        "intro_bridge": _say("हाँ, शुरू करें।"),
        "birth_mode_detail": _say("नहीं, नॉर्मल डिलीवरी हुई थी।"),
        "birth_term": _say("नहीं, पूरे नौ महीने पर हुआ था।"),
        # 3B: one breath with a 1.5 s fan in the middle of it.
        "birth_nicu": _pause_say(NICU_HEAD_LINE, NICU_TAIL_LINE),
        "birth_nicu:which_item": _say(BOTH_LINE),
        "birth_nicu:fu0": _say("साँस लेने में तकलीफ़ थी।"),
        "birth_nicu:fu1": _say("चार दिन।"),
        "birth_jaundice": _say("नहीं।"),
        "birth_sudden": _say("नहीं।"),
        "birth_cord_3weeks": _say("नहीं।"),
        "birth_cord_bleed": _tail(NO_LINE),          # 3C: inside the 350 ms playback tail
        "body_fits": _say(FITS_LINE),                # 3A: forces which_item
        "body_fits:which_item": _say(BOTH_LINE),
        "body_fits:fu0": _say(FITS_ONSET_LINE),
        # 3D: the first answer dies with the socket inside DECIDING; the second lands after the resume.
        "fam_cancer": _steps(_say(NO_LINE), _say(NO_LINE)),
    }


def _a4_lines() -> dict[str, tuple]:
    return {
        "intro_name": _txt("मेरा नाम रेखा है।"),
        "intro_married": ("quick", "married"),
        "intro_children": _txt("मेरा एक चार साल का बेटा है, रोहन।"),
        "intro_bridge": _txt(PILLAR_TURN_LINE),        # 4A: one sentence answers the bridge and states jaundice
        "birth_mode_detail": _txt("नहीं, नॉर्मल डिलीवरी हुई थी। मासिक में बहुत ज़्यादा खून आता है।"),
        "birth_term": _txt("नहीं, पूरे नौ महीने पर हुआ था। दाँत निकलने में ज़्यादा खून आता था, "
                           "और जवानी के बदलाव जल्दी आ गए।"),
        "birth_nicu": _txt("नहीं, बच्चा काँच के बक्से में नहीं रखा गया था। परिवार में कैंसर की बीमारी है।"),   # 4C
    }


def _personas() -> list[Persona]:
    return [
        # 3E arms after the *second* fam_cancer answer: the first one dies with the socket (3D), the resumed
        # call re-asks it, and the step that follows is the one the emergency must interrupt while muted.
        Persona(_A3_PID, "shishu-hi", _a3_lines(), close=CloseReason.EMERGENCY.value, check=_check_a3,
                quiet_after="fam_cancer", quiet_after_count=2),
        # 4C captures `fam_cancer` on the same turn, so 4D's silence starts at the step right after `birth_nicu`.
        Persona(_A4_PID, "shishu-hi", _a4_lines(), close=CloseReason.SILENCE.value, check=_check_a4,
                default=("text", "नहीं।"), quiet_after="birth_nicu", quiet_after_count=1),
    ]


def _port_free() -> bool:
    """A suite that silently shares its port with another run grades the wrong socket."""
    with socket.socket() as probe:
        try:
            probe.bind((HOST, PORT))
            return True
        except OSError:
            return False


async def _main(cases: list[str]) -> int:
    import uvicorn

    from intake import live as live_mod
    from intake.app import app

    if PROMPT_MUTE_MAX_SEC != live_mod._PROMPT_MUTE_MAX_SEC:
        raise SystemExit("PROMPT_MUTE_MAX_SEC drifted from intake/live.py")
    personas = [p for p in _personas() if p.pid.split("_", 1)[0] in cases]
    if not personas:
        print(f"no call matches {cases}", flush=True)
        return 1
    if not _port_free():
        raise SystemExit(f"{HOST}:{PORT} is already in use; stop the other live run before starting this one")
    print(f"data_dir={db.DATA_DIR}\nport={PORT}", flush=True)
    server = uvicorn.Server(uvicorn.Config(app, host=HOST, port=PORT, log_level="warning"))
    task = asyncio.create_task(server.serve())

    def ping() -> None:
        with urllib.request.urlopen(BASE + "/health", timeout=0.5) as response:
            response.read()

    for _ in range(80):
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
    for item in results:
        print(("PASS " if item["ok"] else "FAIL ") + item["id"] + f" call={item['call_id']} "
              + ("; ".join(item["problems"]) or "every check green")
              + (" | known defect: " + ", ".join(item["known_defects"]) if item["known_defects"] else ""), flush=True)
    print(f"\n{passed}/{len(results)} PASS\n{out}", flush=True)
    return 0 if passed == len(results) else 1


# ------------------------------------------------------------------------------------------------------------
# Offline self-test: every assertion must flip on a broken state, and pass on a whole one
# ------------------------------------------------------------------------------------------------------------
def _profile(slots: dict[str, Any]) -> TurnCommand:
    return TurnCommand(profile_patch=ProfilePatch(slots=slots), profile_turn_attempted=True)


def _roster(*kids: RosterInput) -> TurnCommand:
    return TurnCommand(profile_patch=ProfilePatch(roster=kids, roster_complete=True), profile_turn_attempted=True)


def _walk(pack, st: CallState, asks: list[tuple[str, str]], qid: str, cmd: TurnCommand | None = None) -> CallState:
    """Answer the current step and record it, so the synthetic `asks` are the ones a live call would produce."""
    asks.append((st.step_token, st.cursor.qid))
    return engine.reduce_turn(pack, st, cmd or TurnCommand(outcome=TurnOutcome.CLEAR))


def _walk_to(pack, st: CallState, asks: list[tuple[str, str]], target: str) -> CallState:
    for _ in range(200):
        if st.phase is not Phase.ASKING or st.cursor.qid == target:
            return st
        st = _walk(pack, st, asks, target)
    raise RuntimeError(f"never reached {target}")


@functools.lru_cache(maxsize=64)
def _rendered(pack_id: str, state_json: str) -> tuple[int, str]:
    """The real patient PDF for one state, extracted the way the live checks read it."""
    pack, st = load_pack(pack_id), CallState.from_dict(json.loads(state_json))
    call = {"id": 1, "pack_id": pack_id, "started_at": "2026-11-21 10:00:00", "duration_sec": 120,
            "share_token": "selftest"}
    pdf = reports.render_patient_pdf(pack, call, reports.project_summary(pack, st, ()))
    reader = PdfReader(io.BytesIO(pdf))
    return len(reader.pages), _norm(" ".join(" ".join(p.extract_text() or "" for p in reader.pages).split()))


def _finish(ev: Ev) -> Ev:
    """Fill the projection-derived fields a live run gets from SQLite, so parity is graded the same way."""
    ev.proj = reports.project_summary(ev.pack, ev.state, ())
    if "summary_report" not in ev.pinned:
        ev.call = {**ev.call, "summary_report": ev.proj.report_markdown}
    if "pdf" not in ev.pinned:
        ev.pages, ev.pdf_text = _rendered(ev.pack.id, json.dumps(ev.state.to_dict(), sort_keys=True))
    return ev


def _left_channel(frames: int, speech: list[tuple[float, float]], end: float) -> np.ndarray:
    """A synthetic audit track: silence everywhere except the windows a real caller was recorded in."""
    left = np.zeros(frames, dtype=np.int16)
    rng = np.random.default_rng(11)
    for start, dur in speech:
        i0 = frames - int(round((end - start) * RATE))
        i1 = frames - int(round((end - start - dur) * RATE))
        i0, i1 = max(0, i0), min(frames, max(i1, i0 + 1))
        left[i0:i1] = rng.normal(0.0, 0.13 * 32768.0, i1 - i0).astype(np.int16)
    return left


_T0, _T_END = 1000.0, 1070.0


def _synth_a3(pack) -> Ev:
    """A whole call 3, built by the engine's own reducer so every shape is real; the WAV is a synthetic take."""
    asks: list[tuple[str, str]] = []
    st = engine.new_state(pack)
    st = engine.reduce_turn(pack, st, _profile({"caller_name": "रेखा"}))
    st = engine.reduce_turn(pack, st, TurnCommand(outcome=TurnOutcome.ANSWERED, value="married"))
    st = engine.reduce_turn(pack, st, _roster(RosterInput(age_years=2.0, tag="female", name="मीरा"),
                                              RosterInput(age_years=2.0, tag="female", name="मीरा")))
    st = _walk(pack, st, asks, "intro_bridge")                     # intro_bridge
    st = _walk(pack, st, asks, "birth_term")                       # birth_mode_detail
    asks.append((st.step_token, st.cursor.qid))                    # birth_term
    st = engine.reduce_turn(pack, st, TurnCommand(outcome=TurnOutcome.CLEAR))
    asks.append((st.step_token, st.cursor.qid))                    # birth_nicu, one turn, both halves
    st = engine.reduce_turn(pack, st, TurnCommand(outcome=TurnOutcome.CONCERN, utterance=NICU_NOTE))
    asks.append((st.step_token, st.cursor.qid))                    # birth_nicu:which_item
    st = engine.reduce_turn(pack, st, TurnCommand(outcome=TurnOutcome.CONCERN, item_ref="item_1", utterance=NICU_NOTE))
    asks.append((st.step_token, st.cursor.qid))                    # nicu followup
    st = engine.reduce_turn(pack, st, TurnCommand(outcome=TurnOutcome.ANSWERED, utterance="चार दिन।"))
    for _ in range(3):                                             # jaundice, sudden, cord_3weeks
        asks.append((st.step_token, st.cursor.qid))
        st = engine.reduce_turn(pack, st, TurnCommand(outcome=TurnOutcome.CLEAR))
    asks.append((st.step_token, st.cursor.qid))                    # birth_cord_bleed, the tail reply
    st = engine.reduce_turn(pack, st, TurnCommand(outcome=TurnOutcome.CLEAR))
    st = _walk_to(pack, st, asks, "body_fits")
    asks.append((st.step_token, st.cursor.qid))
    st = engine.reduce_turn(pack, st, TurnCommand(outcome=TurnOutcome.CONCERN, utterance=FITS_LINE))
    asks.append((st.step_token, st.cursor.qid))                    # body_fits:which_item
    st = engine.reduce_turn(pack, st, TurnCommand(outcome=TurnOutcome.CONCERN, item_ref="all", utterance=BOTH_LINE))
    asks.append((st.step_token, st.cursor.qid))                    # body_fits:followup:0
    st = engine.reduce_turn(pack, st, TurnCommand(outcome=TurnOutcome.ANSWERED, utterance=FITS_ONSET_LINE))
    st = _walk_to(pack, st, asks, "fam_cancer")
    asks.append((st.step_token, st.cursor.qid))
    st = engine.reduce_turn(pack, st, TurnCommand(outcome=TurnOutcome.CLEAR))
    asks.extend([("fam_cancer", "fam_cancer"), ("fam_heart_kidney", "fam_heart_kidney")])  # the re-ask, then 3E
    st = st.model_copy(update={"phase": Phase.CLOSING, "close_reason": CloseReason.EMERGENCY,
                               "emergency_note": EMERGENCY_LINE, "cursor": st.cursor.model_copy(update={"qid": st.cursor.qid})})
    persona = Persona(_A3_PID, "shishu-hi", {}, close=CloseReason.EMERGENCY.value, check=_check_a3)
    persona.asks = asks
    persona.which_item_prompt = engine.which_item_prompt(pack, st)
    persona.call_ids = [11, 11]
    persona.tail_lead_ms = 168.0
    persona.frames_before_drop = 640_000
    persona.telemetry_before_drop = {"user_turns": 18, "bot_turns": 17, "interruptions": 0}
    persona.telemetry_after_resume = {"user_turns": 19, "bot_turns": 18, "interruptions": 0}
    persona.telemetry_after_tail = {"user_turns": 15, "bot_turns": 14, "interruptions": 0}
    persona.muted_attack = {"start": 1040.0, "dur": 4.4}
    persona.listening_attack = {"start": 1052.0, "dur": 4.4}
    transcript = [
        {"role": "assistant", "qid": "body_fits", "text": persona.which_item_prompt},
        {"role": "assistant", "qid": "close", "text": str(pack.prompts["closings"][CloseReason.EMERGENCY.value])},
        {"role": "user", "text": EMERGENCY_LINE},
    ]
    call = {"id": 11, "audio_path": "call_11.wav", "telemetry": {"tool_errors": 0, "interruptions": 0},
            "events": [{"type": "session_started"}, {"type": "session_resumed"},
                       # The one Decision pass that read the paused sentence, exactly as `_turn_meta` records it.
                       {"type": "tool_call", "detail": "next_question_ok", "meta": {"step_in": "birth_nicu", "stt": NICU_NOTE}},
                       {"type": "turn_complete"}], "transcript": transcript}
    frames = int((_T_END - _T0) * RATE)
    ev = Ev(pack=pack, call=call, state=st, persona=persona,
            sources=[("voice", step) for step, _qid in asks],
            events=list(call["events"]), wav_shape=(2, RATE), wav_frames=frames, wav_end=_T_END,
            # Both windows: the listening-window utterance and, since S-7.6 is fixed, the one spoken while muted.
            wav_left=_left_channel(frames, [(1040.0, 4.4), (1052.0, 4.4)], _T_END))
    return _finish(ev)


def _synth_a4(pack) -> Ev:
    """A whole call 4, replayed through the real reducer; the silence close is the engine's own end state."""
    asks: list[tuple[str, str]] = []
    st = engine.new_state(pack)
    st = engine.reduce_turn(pack, st, _profile({"caller_name": "रेखा"}))
    st = engine.reduce_turn(pack, st, TurnCommand(outcome=TurnOutcome.ANSWERED, value="married"))
    st = engine.reduce_turn(pack, st, _roster(RosterInput(age_years=4.0, tag="male", name="रोहन")))
    asks.append((st.step_token, st.cursor.qid))                     # intro_bridge
    st = engine.reduce_turn(pack, st, TurnCommand(outcome=TurnOutcome.CLEAR, stated={"birth_jaundice": TurnOutcome.CONCERN},
                                                  utterance=PILLAR_TURN_LINE))
    asks.append((st.step_token, st.cursor.qid))                     # birth_mode_detail + the pruned-gate claim
    st = engine.reduce_turn(pack, st, TurnCommand(outcome=TurnOutcome.CLEAR, stated={"mod_heavy_periods": TurnOutcome.CONCERN},
                                                  utterance="नहीं, नॉर्मल डिलीवरी हुई थी। मासिक में बहुत ज़्यादा खून आता है।"))
    asks.append((st.step_token, st.cursor.qid))                     # birth_term + two more pruned-gate claims
    st = engine.reduce_turn(pack, st, TurnCommand(outcome=TurnOutcome.CLEAR,
                                                  stated={"mod_tooth_bleed": TurnOutcome.CONCERN, "body_puberty": TurnOutcome.CONCERN},
                                                  utterance="नहीं, पूरे नौ महीने पर हुआ था। दाँत निकलने में ज़्यादा खून आता था।"))
    asks.append((st.step_token, st.cursor.qid))                     # birth_nicu, the jump
    st = engine.reduce_turn(pack, st, TurnCommand(outcome=TurnOutcome.CLEAR, jump_topic=True,
                                                  next_qid_probs={"fam_cancer": 0.93},
                                                  stated={"fam_cancer": TurnOutcome.CONCERN},
                                                  utterance="नहीं, बच्चा काँच के बक्से में नहीं रखा गया था। "
                                                            "परिवार में कैंसर की बीमारी है।"))
    asks.append((st.step_token, st.cursor.qid))                     # the step the caller then ignores (4D)
    st = st.model_copy(update={"phase": Phase.CLOSING, "close_reason": CloseReason.SILENCE})
    persona = Persona(_A4_PID, "shishu-hi", {}, close=CloseReason.SILENCE.value, check=_check_a4,
                      default=("text", "नहीं।"))
    persona.asks = asks
    persona.call_ids = [12]
    persona.quiet_after, persona.quiet_after_count = "birth_nicu", 1
    persona.probe = {"attempted": True, "reopened": False, "call_id": 13, "frames": ["session_started", "question_state"],
                     "status_after": "completed", "close_reason_after": CloseReason.SILENCE.value,
                     "note": "a closed call is not resumable: a new call is minted, the finished row keeps its reason"}
    events = [{"type": "session_started"}, {"type": "silence_check_in", "detail": "15s"},
              {"type": "auto_close", "detail": "silence"}, {"type": "turn_complete"}]
    transcript = [{"role": "assistant", "qid": "close", "text": str(pack.prompts["closings"][CloseReason.SILENCE.value])}]
    call = {"id": 12, "audio_path": "call_12.wav", "telemetry": {"tool_errors": 0},
            "events": events, "transcript": transcript}
    frames = 90 * RATE
    ev = Ev(pack=pack, call=call, state=st, persona=persona, sources=[], events=events, wav_shape=(2, RATE),
            wav_frames=frames, wav_end=_T_END, wav_left=np.zeros(frames, dtype=np.int16),
            interps={8: _jump_interp()})
    return _finish(ev)


def _jump_interp():
    from intake.turns import ClinicalClaim, DomainFit, SpeechAct, TopicRelation, TurnInterpretation

    return TurnInterpretation(act=SpeechAct.ANSWER, active_relation=TopicRelation.ACTIVE, active_domain=DomainFit.IN_DOMAIN,
                              claims=(ClinicalClaim(target_qid="birth_nicu", relation=TopicRelation.ACTIVE,
                                                    domain_fit=DomainFit.IN_DOMAIN, status=TurnOutcome.CLEAR),),
                              jump_topic=True, next_qid_probs={"fam_cancer": 0.93, "birth_sudden": 0.03})


def _common_mutations(case: str) -> list[tuple[str, str, Any]]:
    """(label, the one check that must flip, mutation) for the checks both cases owe."""
    def call(ev: Ev, **kw) -> None:
        ev.pinned.update(kw)
        ev.call = {**ev.call, **kw}

    watchdog = ("the watchdog fired on a call that should have closed on content", "closed on content, not the watchdog",
                lambda ev: setattr(ev, "events", [*ev.events, {"type": "auto_close", "detail": "time_limit"}])) if case == "a3" else (
        "no watchdog close was recorded at all", "watchdog closed on silence",
        lambda ev: setattr(ev, "events", [e for e in ev.events if e.get("type") != "auto_close"]))
    return [
        ("a tool error was swallowed", "zero tool_errors", lambda ev: call(ev, telemetry={"tool_errors": 2})),
        ("a server error event appeared", "zero error events",
         lambda ev: setattr(ev, "events", [*ev.events, {"type": "error", "detail": "boom"}])),
        watchdog,
        ("a spoken leak appeared", "zero spoken leaks",
         lambda ev: call(ev, transcript=[*(ev.call.get("transcript") or []), {"role": "assistant", "text": "हीमोफीलिया क्या है?"}])),
        ("the report was never written", "the report is written", lambda ev: call(ev, summary_report="")),
        ("the WAV came back mono", "stereo 16 kHz WAV", lambda ev: setattr(ev, "wav_shape", (1, RATE))),
        ("the PDF is missing", "patient PDF exists",
         lambda ev: (setattr(ev, "pinned", ev.pinned | {"pdf"}), setattr(ev, "pages", 0), setattr(ev, "pdf_text", ""))),
        ("the stored report drifted from the projection", "project_summary parity",
         lambda ev: call(ev, summary_report="### Profile\n- nothing")),
    ]


def _mutations_a3() -> list[tuple[str, str, Any]]:
    def answers(ev: Ev, **kw) -> None:
        ev.state = ev.state.model_copy(update={"answers": {**ev.state.answers, **kw}})

    def sources(ev: Ev, drop: str) -> None:
        ev.sources = [s for s in ev.sources if s[1] != drop]

    def persona(ev: Ev, **kw) -> None:
        for key, value in kw.items():
            setattr(ev.persona, key, value)

    return [
        *_common_mutations("a3"),
        ("a text frame was used instead of audio", "3 streamed real PCM only, never a text frame",
         lambda ev: setattr(ev, "sources", [*ev.sources, ("text", "intro_name")])),
        ("a twin was lost from the roster", "3A twins stay two distinct children",
         lambda ev: setattr(ev, "state", ev.state.model_copy(update={"roster": ev.state.roster[:1]}))),
        ("the twins were numbered identically", "3A twins stay two distinct children",
         lambda ev: setattr(ev, "state", ev.state.model_copy(
             update={"roster": [c.model_copy(update={"id": "item_1"}) for c in ev.state.roster]}))),
        ("the prompt called one twin the middle child", "3A the twins prompt is numbered and never calls a twin middle",
         lambda ev: persona(ev, which_item_prompt="यह किस बच्चे की बात है — पहले 2 साल वाली मीरा (बीच वाली), या दूसरे 2 साल वाली मीरा?")),
        ("the prompt read both twins out identically", "3A the twins prompt is numbered and never calls a twin middle",
         lambda ev: persona(ev, which_item_prompt="यह किस बच्चे की बात है — 2 साल वाली मीरा, या 2 साल वाली मीरा?")),
        ("both children were not selected", "3A body_fits resolved CONCERN on both children",
         lambda ev: answers(ev, body_fits=_rec(C, FITS_LINE, items=["item_1"]))),
        ("the concern was filed against one twin", "3A the twins concern was settled for both children, not one",
         lambda ev: (answers(ev, body_fits=_rec(C, FITS_LINE, items=["item_1"])),
                     persona(ev, asks=[(s, q) for s, q in ev.persona.asks if not s.startswith("body_fits:which_item")]))),
        ("the paused sentence arrived as two turns", "3B one voice turn carried the paused sentence",
         lambda ev: setattr(ev, "events", [*ev.events, {"type": "tool_call", "detail": "next_question_ok",
                                                        "meta": {"step_in": "birth_nicu", "stt": NICU_TAIL_LINE}}])),
        ("the pause burned an attempt on birth_nicu", "3B one voice turn carried the paused sentence",
         lambda ev: ev.persona.asks.append(("birth_nicu:try:1", "birth_nicu"))),
        ("the pause arrived as typed text, not audio", "3B one voice turn carried the paused sentence",
         lambda ev: (setattr(ev, "events", [e for e in ev.events if e.get("type") != "tool_call"]),
                     setattr(ev, "sources", [*[s for s in ev.sources if s[1] != "birth_nicu"], ("text", "birth_nicu")]))),
        ("the pause cut the second half off", "3B birth_nicu CONCERN keeps the four-day breathing detail",
         lambda ev: (answers(ev, birth_nicu=_rec(C, NICU_HEAD_LINE)),
                     setattr(ev, "sources", [("voice", "birth_nicu"), ("late_stt", "birth_nicu")]))),
        ("the head of the sentence was lost", "3B both halves of the sentence fused into one note",
         lambda ev: (answers(ev, birth_nicu=_rec(C, NICU_TAIL_LINE)),
                     setattr(ev, "events", [{**e, "meta": {**(e.get("meta") or {}), "stt": NICU_TAIL_LINE}}
                                            if e.get("type") == "tool_call" else e for e in ev.events]))),
        ("the tail reply started after the playback ended", "3C the tail reply started inside the 350 ms window",
         lambda ev: persona(ev, tail_lead_ms=-240.0)),
        ("the tail reply had to be re-sent in the open window", "3C the tail reply started inside the 350 ms window",
         lambda ev: persona(ev, tail_lead_ms=175.0, tail_retried=True)),
        ("the tail reply did not clear the gate", "3C birth_cord_bleed resolved CLEAR from the tail reply",
         lambda ev: answers(ev, birth_cord_bleed=_rec(U))),
        ("her own prompt was counted as a barge-in", "3C no interruption was counted",
         lambda ev: persona(ev, telemetry_after_tail={"tool_errors": 0, "interruptions": 1})),
        ("the barge-in was stamped on the tail's step", "3C no interruption was counted",
         lambda ev: setattr(ev, "events", [*ev.events, {"type": "barge_in", "detail": "birth_cord_bleed"}])),
        ("the resume minted a new call", "3D the same call id came back after the drop",
         lambda ev: persona(ev, call_ids=[11, 12])),
        ("session_resumed was never logged", "3D session_resumed recorded",
         lambda ev: setattr(ev, "events", [e for e in ev.events if e.get("type") != "session_resumed"])),
        ("fam_cancer never came back after the resume", "3D fam_cancer resolved cleanly after the resume",
         lambda ev: answers(ev, fam_cancer=_rec(UA))),
        ("the WAV was restarted instead of appended", "3D the WAV survived the reconnect",
         lambda ev: (setattr(ev, "wav_frames", ev.persona.frames_before_drop - 1), setattr(ev, "wav_left", np.zeros(1000, np.int16)))),
        ("telemetry reset with the new socket", "3D telemetry survived the drop",
         lambda ev: persona(ev, telemetry_after_resume={"user_turns": 1, "bot_turns": 1})),
        ("the listening-window measurement is broken", "3E a listening-window utterance reached the WAV left channel",
         lambda ev: setattr(ev, "wav_left", np.zeros(ev.wav_frames, dtype=np.int16))),
        ("the muted frames were dropped again", "3E muted-window audio reached the WAV left channel",
         lambda ev: setattr(ev, "wav_left", _left_channel(ev.wav_frames, [(1052.0, 4.4)], ev.wav_end))),
        ("the call closed without the emergency", "3E the call closed EMERGENCY with her own words",
         lambda ev: setattr(ev, "state", ev.state.model_copy(update={"close_reason": CloseReason.CALLER_STOP, "emergency_note": ""}))),
        ("the 108 line was never spoken", "3E the 108 line was spoken",
         lambda ev: ev.call.update(transcript=[{"role": "assistant", "text": "ठीक है, धन्यवाद।"}])),
        ("the emergency PDF lost its banner", "3E the emergency PDF carries the hospital banner",
         lambda ev: setattr(ev, "state", ev.state.model_copy(update={"close_reason": CloseReason.SILENCE}))),
    ]


def _mutations_a4() -> list[tuple[str, str, Any]]:
    def answers(ev: Ev, **kw) -> None:
        ev.state = ev.state.model_copy(update={"answers": {**ev.state.answers, **kw}})

    def call(ev: Ev, **kw) -> None:
        ev.pinned.update(kw)
        ev.call = {**ev.call, **kw}

    def persona(ev: Ev, **kw) -> None:
        for key, value in kw.items():
            setattr(ev.persona, key, value)

    return [
        *_common_mutations("a4"),
        ("the bridge turn was never read", "4A intro_bridge answered CLEAR",
         lambda ev: answers(ev, intro_bridge=_rec(UA))),
        ("the stated concern was dropped", "4A birth_jaundice auto-filled CONCERN from the bridge turn",
         lambda ev: answers(ev, birth_jaundice=_rec(CL))),
        ("jaundice was asked at main anyway", "4A birth_jaundice never asked at main",
         lambda ev: ev.persona.asks.append(("birth_jaundice", "birth_jaundice"))),
        ("the bridge remark was quoted onto her copy", "4A the bridge remark stayed off her copy",
         lambda ev: answers(ev, birth_jaundice=_rec(C, PILLAR_TURN_LINE))),
        ("ever_pregnant was woken by a family claim", "4B ever_pregnant stays NOT_APPLICABLE",
         lambda ev: answers(ev, ever_pregnant=_rec(CL))),
        ("a pruned child gate was woken by the claim", "4B the three pruned child gates stay NOT_APPLICABLE",
         lambda ev: answers(ev, mod_heavy_periods=_rec(C, "मासिक में बहुत ज़्यादा खून आता है।"))),
        ("a pruned gate reached the report", "4B pruned gates never reached the report",
         lambda ev: call(ev, summary_report=f"{ev.call.get('summary_report') or ''}\n{ev.pack.by_id['mod_heavy_periods'].need}")),
        ("a pruned gate reached the patient PDF", "4B pruned gates never reached the patient PDF",
         lambda ev: (setattr(ev, "pinned", ev.pinned | {"pdf"}),
                     setattr(ev, "pdf_text", ev.pdf_text + " " + " ".join(sum(_PRUNE_PDF.values(), ()))))),
        ("a pruned gate was asked anyway", "4B pruned gates were never asked",
         lambda ev: ev.persona.asks.append(("mod_tooth_bleed", "mod_tooth_bleed"))),
        ("the router never saw a topic jump", "4C jump_topic was read on the birth_nicu turn",
         lambda ev: setattr(ev, "interps", {8: _jump_interp().model_copy(update={"jump_topic": False})})),
        ("the router's pick was below the pack floor", "4C jump_topic was read on the birth_nicu turn",
         lambda ev: setattr(ev, "interps", {8: _jump_interp().model_copy(update={"next_qid_probs": {"fam_cancer": 0.2}})})),
        ("birth_nicu was overwritten by the family history", "4C birth_nicu was not corrupted by the jump",
         lambda ev: answers(ev, birth_nicu=_rec(CL, "नहीं, परिवार में कैंसर की बीमारी है।"))),
        ("a jump that consumed the attempt was never re-asked", "4C birth_nicu was not corrupted by the jump",
         lambda ev: (_drop_answer(ev, "birth_nicu"),
                     persona(ev, asks=[*[t for t in ev.persona.asks if t[1] != "birth_nicu"],
                                       ("birth_nicu", "birth_nicu")]))),
        ("the family cancer answer was lost", "4C fam_cancer resolved CONCERN with the family history",
         lambda ev: answers(ev, fam_cancer=_rec(CL))),
        ("the family gate was walked to at main instead of captured", "4C fam_cancer was not asked twice",
         lambda ev: persona(ev, asks=[*ev.persona.asks, ("fam_cancer", "fam_cancer"), ("fam_cancer", "fam_cancer")])),
        ("the jump never moved the walk off birth_nicu", "4C the cursor left birth_nicu after the jump",
         lambda ev: persona(ev, asks=[("birth_nicu", "birth_nicu")])),
        ("the 15 s check-in never fired", "4D silence_check_in was emitted",
         lambda ev: setattr(ev, "events", [e for e in ev.events if e.get("type") != "silence_check_in"])),
        ("the check-in came at the wrong time", "4D silence_check_in was emitted",
         lambda ev: setattr(ev, "events", [{**e, "detail": "2s"} if e.get("type") == "silence_check_in" else e for e in ev.events])),
        ("the call closed for another reason", "4D close_reason is silence",
         lambda ev: setattr(ev, "state", ev.state.model_copy(update={"close_reason": CloseReason.CALLER_STOP}))),
        ("the closing line was spoken twice", "4D the closing line was spoken exactly once",
         lambda ev: ev.call.update(transcript=[*(ev.call.get("transcript") or []),
                                               {"role": "assistant", "text": str(ev.pack.prompts["closings"][CloseReason.SILENCE.value])}])),
        ("the PDF was not marked incomplete", "4D the patient PDF is marked अधूरी कॉल",
         lambda ev: setattr(ev, "state", ev.state.model_copy(update={"close_reason": CloseReason.COMPLETE}))),
        ("a pruned gate was listed as unasked", "4D page2_unasked omits every NOT_APPLICABLE gate",
         lambda ev: _drop_answer(ev, "mod_heavy_periods")),
        ("the report lost the close reason", "4D the report records the silence close",
         lambda ev: call(ev, summary_report="### Profile\n- Close reason: complete.")),
        ("the resume probe was never attempted", "4E the post-silence resume probe was recorded",
         lambda ev: persona(ev, probe={"attempted": False, "reopened": False, "call_id": 0})),
        ("the finished call was reopened as itself", "4E the resume was refused and minted a new call",
         lambda ev: persona(ev, probe={**ev.persona.probe, "reopened": True, "call_id": 12})),
        ("the probe landed on the finished row but called it a refusal", "4E the resume was refused and minted a new call",
         lambda ev: persona(ev, probe={**ev.persona.probe, "reopened": False, "call_id": 12})),
        ("the reopened session rewrote the close reason", "4E the finished call kept its silence reason and report",
         lambda ev: persona(ev, probe={**ev.persona.probe, "close_reason_after": CloseReason.CALLER_STOP.value,
                                       "status_after": "active"})),
    ]


def _selftest() -> int:
    """Prove every critical assertion is non-vacuous, offline, before a single live token is spent."""
    print("=== offline self-test: every check must flip when broken and hold when whole ===", flush=True)
    failures: list[str] = []
    for pid, synth, mutations in ((_A3_PID, _synth_a3, _mutations_a3), (_A4_PID, _synth_a4, _mutations_a4)):
        pack = load_pack("shishu-hi")
        try:
            whole = synth(pack)
            cl = Checklist()
            _common(whole, cl, case="a3" if pid == _A3_PID else "a4")
            whole.persona.check(whole, cl)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{pid}: whole state raised {type(exc).__name__}: {exc}")
            print(f"  FAIL {pid}: whole state raised {type(exc).__name__}: {exc}", flush=True)
            continue
        red = cl.failing()
        known = sorted(name for name in red if name in _KNOWN_DEFECTS)
        unexpected = sorted(name for name in red if name not in _KNOWN_DEFECTS)
        print(f"  {'PASS' if not unexpected else 'FAIL'} {pid}: whole state {len(cl.items) - len(red)}/{len(cl.items)} "
              f"checks green, known defects: {known or 'none'}", flush=True)
        if unexpected:
            failures.append(f"{pid}: whole state is not green: {unexpected}")
        targets: set[str] = set()
        for label, target, mutate in mutations():
            ev = synth(pack)
            broken = Checklist()
            try:
                # The mutation lands first, then the projection re-derives from the broken state, then every
                # check reads it - exactly the order a live call takes (row written, projected, graded).
                mutate(ev)
                ev = _finish(ev)
                _common(ev, broken, case="a3" if pid == _A3_PID else "a4")
                ev.persona.check(ev, broken)
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{pid}: {label} raised {type(exc).__name__}: {exc}")
                print(f"  FAIL {pid}: {label} raised {type(exc).__name__}: {exc}", flush=True)
                continue
            targets.add(target)
            hit = (target in broken.failing()) != (target in red)
            print(f"  {'PASS' if hit else 'FAIL'} {label} -> {target}", flush=True)
            if not hit:
                failures.append(f"{pid}: {label} did not flip {target!r} (failing={sorted(broken.failing())})")
        missing = sorted(cl.names() - targets)
        if missing:
            failures.append(f"{pid}: no mutation proves {missing}")
            print(f"  FAIL {pid}: no mutation proves {missing}", flush=True)
        for target in sorted(targets):
            if target not in cl.names():
                failures.append(f"{pid}: mutation targets unknown check {target!r}")
    print(f"\n{'SELFTEST PASS' if not failures else 'SELFTEST FAIL'} ({len(failures)} problem(s))", flush=True)
    for line in failures:
        print(f"  - {line}", flush=True)
    return 0 if not failures else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run_adversarial_shishu_untested",
        description="Two untested shishu-hi adversarial calls: spoken twins/tail/drop/muted emergency, and "
                    "bridge/skip_if/jump/silence.")
    parser.add_argument("--selftest", action="store_true", help="offline: prove every check can fail")
    parser.add_argument("--case", default="all", choices=("a3", "a4", "all"), help="which live call to run")
    cfg = parser.parse_args(argv)
    if cfg.selftest:
        return _selftest()
    return asyncio.run(_main(["a3", "a4"] if cfg.case == "all" else [cfg.case]))


if __name__ == "__main__":
    raise SystemExit(main())
