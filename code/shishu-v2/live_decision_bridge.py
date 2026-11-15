"""Decision API bridge for live calls: turn capture and turn execution."""

from __future__ import annotations

import copy
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import decisions
import profile_extractor
import prompts
from audio_recorder import pcm16_to_wav_bytes
from call_cursor import apply_interpreted_turn, close_technical_failure, merge_transcript_chunk, step_id, step_kind

logger = logging.getLogger(__name__)
USER_TURN_PCM_MAX_BYTES = 960_000
USER_TURN_PCM_TAIL_CHUNKS = 6


@dataclass
class TurnCapture:
    user_turn_buffer: str = ""
    user_turn_pcm: bytearray = field(default_factory=bytearray)
    user_turn_pcm_tail: int = 0
    user_turn_cursor_snapshot: dict | None = None
    user_turn_finalized: bool = False
    user_turn_is_quick_reply_stale: bool = False
    speech_allowed: bool = True
    handled_turn: bool = False
    in_flight: bool = False
    last_key: tuple[str, str, int] | None = None
    thresholds: decisions.DecisionThresholds | None = None
    extractor_failures: int = 0
    extractor_failure_step: str = ""

    def reset(self, *, clear_pcm: bool = True, clear_turn_lock: bool = True) -> None:
        self.user_turn_buffer, self.user_turn_cursor_snapshot = "", None
        self.user_turn_finalized = self.user_turn_is_quick_reply_stale = False
        if clear_pcm:
            self.user_turn_pcm.clear()
            self.user_turn_pcm_tail = 0
        if clear_turn_lock:
            self.handled_turn, self.last_key = False, None

    def append_pcm(self, pcm_16k: bytes) -> None:
        avail = USER_TURN_PCM_MAX_BYTES - len(self.user_turn_pcm)
        piece = pcm_16k[:avail] if (avail > 0 and pcm_16k) else b""
        if len(piece) % 2 == 1:
            piece = piece[:-1]
        if piece:
            self.user_turn_pcm.extend(piece)

    def begin_audio_speech(self, cursor_state: dict, pcm_16k: bytes) -> None:
        if self.in_flight:
            return
        if not self.user_turn_pcm and not self.user_turn_buffer:
            self.speech_allowed = self.handled_turn = False
            if self.user_turn_cursor_snapshot is None:
                self.user_turn_cursor_snapshot = copy.deepcopy(cursor_state)
        self.user_turn_pcm_tail = USER_TURN_PCM_TAIL_CHUNKS
        self.append_pcm(pcm_16k)

    def begin_text_turn(self, cursor_state: dict, text: str) -> None:
        self.reset()
        self.user_turn_buffer, self.user_turn_cursor_snapshot = text, copy.deepcopy(cursor_state)
        self.speech_allowed = False

    def begin_transcript_chunk(self, cursor_state: dict, text: str) -> None:
        if not self.user_turn_buffer:
            if self.user_turn_cursor_snapshot is None:
                self.user_turn_cursor_snapshot = copy.deepcopy(cursor_state)
            self.user_turn_finalized = self.user_turn_is_quick_reply_stale = False
        self.user_turn_buffer = merge_transcript_chunk(self.user_turn_buffer, text)
        self.speech_allowed = self.handled_turn = False
        self.last_key = None


def load_turn_thresholds(
    settings: dict | None = None, *, capture: TurnCapture | None = None,
    thresholds: decisions.DecisionThresholds | None = None,
) -> decisions.DecisionThresholds:
    """Resolve, load, and validate Decision API thresholds for a live call."""
    if (chosen := thresholds or (capture.thresholds if capture else None)) is not None:
        return decisions.validate_thresholds(chosen)
    raw = str((settings or {}).get("decision_thresholds_path")
              or os.environ.get("DECISION_THRESHOLDS_PATH") or "").strip()
    path = Path(raw) if raw else Path(__file__).resolve().with_name("decision_thresholds.json")
    return decisions.load_thresholds(path)


def tool_body(state: dict, *, outcome: str = "") -> dict[str, Any]:
    body: dict[str, Any] = {"result": "ok", "say": prompts.say_hindi(state)}
    if state.get("phase") == "closing":
        body["stop"] = True
    if outcome == "clarify" or str(state.get("active_qid") or "") in ("fam_same_problem", "close_anything_else"):
        body["recorded_concerns"] = prompts.recorded_concerns_hi(state)
    return body


def _decision_meta(res: decisions.DecisionCallResult | None = None, *, error_reason: str = "") -> dict[str, Any]:
    if res is None:
        return {"ok": False, "error_reason": error_reason, "decision_ms": 0, "attempts": 0,
                "request_bytes": 0, "question_count": 0, "catalog_option_count": 0,
                "usage_present": False, "total_input_tokens": None}
    return {"ok": bool(res.ok), "error_reason": str(res.error_reason or ""),
            "decision_ms": int(res.decision_ms), "attempts": int(res.attempts),
            "request_bytes": int(res.request_bytes), "question_count": int(res.question_count),
            "catalog_option_count": int(res.catalog_option_count),
            "usage_present": bool(res.usage_present), "total_input_tokens": res.total_input_tokens}


def _superseded(live: Any, step: str) -> bool:
    return bool(live.turn.end_requested or live.cursor_state.get("phase") != "asking" or step_id(live.cursor_state) != step)


def _held(live: Any, reason: str) -> tuple[dict[str, Any], bool, str]:
    live.capture.speech_allowed = True
    return {**tool_body(live.cursor_state), "result": "held", "reason": reason}, False, "held"


async def _fail_turn(live: Any) -> tuple[dict[str, Any], bool, str]:
    live.cursor_state = close_technical_failure(live.cursor_state)
    await live._mark_closing("technical_failure")
    live.capture.speech_allowed = True
    return tool_body(live.cursor_state), False, "technical_failure"


async def run_decision_turn(live: Any, *, input_rate: int = 16000) -> tuple[dict[str, Any], bool, str]:
    cap: TurnCapture = live.capture
    if live.cursor_state.get("phase") != "asking":
        cap.speech_allowed = True
        return tool_body(live.cursor_state), False, "closing"
    utt, pcm = cap.user_turn_buffer.strip(), bytes(cap.user_turn_pcm)
    if not utt and not pcm:
        return _held(live, "empty_turn")
    step = step_id(live.cursor_state)
    if cap.last_key == (dedup_key := (step, " ".join(utt.split()), len(pcm))):
        return _held(live, "duplicate_turn")
    try:
        resolved = load_turn_thresholds(live.settings, capture=cap, thresholds=live.thresholds)
    except Exception as exc:
        logger.exception("decision thresholds invalid on call %s", live.call_id)
        await live._append_event_safe("decision_turn", "error:invalid_thresholds", _decision_meta(error_reason=type(exc).__name__))
        cap.reset()
        return await _fail_turn(live)

    cap.last_key, cap.handled_turn = dedup_key, True
    snap = cap.user_turn_cursor_snapshot
    state_snap = copy.deepcopy(snap if snap and step_id(snap) == step else live.cursor_state)
    audio_wav = pcm16_to_wav_bytes(pcm, input_rate) if (not utt and pcm) else None
    cap.reset(clear_turn_lock=False)
    cap.in_flight = True
    try:
        res = await decisions.evaluate_turn(decisions.build_turn_request(state_snap, utterance_text=utt, audio_wav_bytes=audio_wav))
        meta = _decision_meta(res)
        if _superseded(live, step):
            await live._append_event_safe("decision_turn", "superseded", meta)
            return _held(live, "stale_turn")
        if not res.ok:
            await live._append_event_safe("decision_turn", f"error:{res.error_reason}" if res.error_reason else "error", meta)
            return await _fail_turn(live)

        interp = decisions.interpret_turn_decision(live.cursor_state, res.raw_answers, utterance_text=utt, thresholds=resolved)
        if interp.future_qid_match is not None:
            meta["future_qid_match"] = interp.future_qid_match
        meta["diagnostics"], payload = interp.diagnostics, dict(interp.cursor_payload)
        active_qid = str(live.cursor_state.get("active_qid") or "")
        outcome = ("emergency" if payload.get("emergency") else
                   "caller_stop" if payload.get("caller_stop") else
                   str(payload.get("current_outcome") or ""))
        ext: profile_extractor.ProfileExtractionResult | None = None
        if interp.needs_profile_extraction and outcome not in ("emergency", "caller_stop"):
            ext = await profile_extractor.extract_profile_entities(live.cursor_state, utterance_text=utt, audio_wav_bytes=audio_wav)
            meta.update({"extractor_ok": bool(ext.ok), "extractor_ms": int(ext.extractor_ms),
                         "extractor_error": str(ext.error_reason or "")})
        if _superseded(live, step):
            await live._append_event_safe("decision_turn", "superseded", meta)
            return _held(live, "stale_turn")
        await live._append_event_safe("decision_turn", f"{active_qid}:{outcome}" if active_qid and outcome else (active_qid or outcome or "ok"), meta)
        if ext is not None and ext.ok:
            cap.extractor_failures, cap.extractor_failure_step = 0, ""
            payload["profile_updates"] = dict(ext.updates)
        elif ext is not None:
            fails = (cap.extractor_failures if cap.extractor_failure_step == step else 0) + 1
            cap.extractor_failures, cap.extractor_failure_step = fails, step
            if fails >= 2:
                cap.extractor_failures, cap.extractor_failure_step = 0, ""
                return await _fail_turn(live)
            payload.pop("profile_updates", None)
            if step_kind(live.cursor_state) == "profile":
                cap.speech_allowed = True
                return tool_body(live.cursor_state), False, (outcome or "reask")

        live.cursor_state = new_state = apply_interpreted_turn(live.cursor_state, payload)
        if step_id(new_state) != step:
            cap.extractor_failures, cap.extractor_failure_step = 0, ""
        if new_state.get("phase") == "closing":
            await live._mark_closing(str(new_state.get("close_reason") or "complete"))
        else:
            await live._save_cursor_safe()

        clarify = outcome not in ("emergency", "caller_stop") and (
            payload.get("current_outcome") == "clarify"
            or (payload.get("side_action") == "clarify_meaning" and live.cursor_state.get("active_qid") == payload.get("side_qid"))
        )
        body = tool_body(live.cursor_state, outcome="clarify" if clarify else "")
        if clarify:
            body["clarify"] = True
        cap.speech_allowed = True
        return body, clarify, (outcome or "ok")
    finally:
        cap.in_flight = False
