"""Pure, synchronous live-session state machine for one SHISHU call.

All turn-taking flags, timers, and capture buffers live in `LiveTurnMachine`;
the I/O shell only feeds it events plus `now`, so it runs without asyncio.
"""

from __future__ import annotations

from enum import StrEnum

import engine
from models import CallState, Phase

USER_TURN_PCM_MAX_BYTES, USER_TURN_PCM_TAIL_CHUNKS = 960_000, 6
QUICK_REPLY_STALE_SEC = 1.2
CHECK_IN_SEC, SILENCE_CLOSE_SEC = 15.0, 30.0
SOFT_LIMIT_SEC, HARD_LIMIT_SEC = 19 * 60.0, 20 * 60.0
CLOSING_AUDIO_GRACE_SEC, CLOSING_MAX_WAIT_SEC = 0.8, 25.0
MAX_WATCHDOG_NUDGES = 6

_PLAYBACK_SLACK_SEC, _SPEECH_END_NUDGE_SEC = 0.1, 6.0
_DEFAULT_RMS_THRESHOLD = 0.015
_QUICK_REPLY_OUTCOMES = ("clear", "concern")


class TurnPhase(StrEnum):
    AWAITING_BOT = "awaiting_bot"
    BOT_SPEAKING = "bot_speaking"
    LISTENING = "listening"
    CAPTURING_USER = "capturing_user"
    EVALUATING = "evaluating"
    CLOSING = "closing"


class SpeakKind(StrEnum):
    SPEAK = "speak"
    CLARIFY = "clarify"
    CLOSE = "close"


class WatchdogAction(StrEnum):
    NONE = "none"
    HARD_LIMIT_STOP = "hard_limit_stop"
    CLOSING_COMPLETE = "closing_complete"
    AUTO_CLOSE_TIME = "auto_close_time"
    AUTO_CLOSE_SILENCE = "auto_close_silence"
    SILENCE_CHECK_IN = "silence_check_in"
    WATCHDOG_NUDGE = "watchdog_nudge"


class LiveTurnMachine:
    """One call's turn protocol: capture, stale-turn shielding, and timers."""

    def __init__(self, *, call_started_ts: float = 0.0) -> None:
        self.call_started_ts = call_started_ts
        self.phase = TurnPhase.AWAITING_BOT
        self.snapshot: CallState | None = None
        self.pcm_16k = bytearray()
        self.last_dedup_key: tuple[str, str, int] | None = None
        self.text_buffer = self.spoken_step = self.user_turn_step = ""
        self.pending_speak_step = self.speak_cue_step = self.last_quick_reply_step = ""
        self.extractor_failure_step = ""
        self.pcm_tail_chunks = self.extractor_failures = self.nudges_for_turn = 0
        self.last_quick_reply_ts = self.playback_ends_at = self.quiet_from = 0.0
        self.last_user_turn_ts = self.end_requested_at = 0.0
        self.pending_speak_kind = self.active_speak_kind = SpeakKind.SPEAK
        self.speech_allowed = True
        self.input_finalized = self.marked_quick_reply_stale = False
        self.handled_current_turn = self.bot_replied_since_user = False
        self.awaiting_bot_reply = self.speech_end_waiting = self.ignore_bot_audio = False
        self.checked_in = self.time_cue_sent = self.end_requested = False
        self.closing_heard = self.ending_notified = False

    def question_already_spoken(self, state: CallState) -> bool:
        return bool(state.step_token) and self.spoken_step == state.step_token

    def in_quick_reply_stale_window(self, state: CallState, now: float) -> bool:
        if not self.last_quick_reply_step:
            return False
        unspoken = not self.question_already_spoken(state)
        tapped = now - self.last_quick_reply_ts < QUICK_REPLY_STALE_SEC
        return unspoken or self.ignore_bot_audio or tapped

    def is_stale_user_turn(self, state: CallState) -> bool:
        mismatch = (
            self.user_turn_step != state.step_token
            if self.user_turn_step
            else bool(self.last_quick_reply_step)
        )
        return bool(mismatch or self.marked_quick_reply_stale)

    def can_accept_bot_output(self) -> bool:
        return self.speech_allowed and not self.ignore_bot_audio

    def on_bot_audio(
        self, state: CallState, pcm_bytes_len: int, rate: int, now: float
    ) -> tuple[bool, bool, int | None]:
        """Accept one bot chunk; report (accepted, is_first, latency_ms)."""
        if not self.can_accept_bot_output():
            return False, False, None
        self.pending_speak_step = ""
        first, latency_ms = not self.bot_replied_since_user, None
        if first:
            self.spoken_step, self.nudges_for_turn = state.step_token, 0
            if self.awaiting_bot_reply and self.last_user_turn_ts > 0:
                latency_ms = max(0, int((now - self.last_user_turn_ts) * 1000))
        played = pcm_bytes_len / max(1, 2 * rate)
        self.playback_ends_at = max(now, self.playback_ends_at) + played
        self.phase = TurnPhase.CLOSING if self.end_requested else TurnPhase.BOT_SPEAKING
        self.awaiting_bot_reply = self.speech_end_waiting = False
        self.bot_replied_since_user = True
        self.closing_heard = self.closing_heard or self.end_requested
        return True, first, latency_ms

    def on_interrupted(self) -> None:
        self.ignore_bot_audio = False
        self.playback_ends_at = self.quiet_from = 0.0
        self.handled_current_turn, self.last_dedup_key = False, None
        if self.phase is TurnPhase.BOT_SPEAKING:
            self.phase = TurnPhase.LISTENING

    def on_user_transcript(self, state: CallState, text: str, now: float) -> bool:
        """Fold one STT chunk in; a chunk inside the tap window is marked stale."""
        if self.in_quick_reply_stale_window(state, now):
            if self.input_finalized:
                self._clear_capture(keep_pcm=True)
            self.user_turn_step, self.marked_quick_reply_stale = self.last_quick_reply_step, True
            self.quiet_from = 0.0
            return False
        if not self.text_buffer:
            self.input_finalized = self.marked_quick_reply_stale = False
            self.snapshot = self.snapshot or state.model_copy(deep=True)
        self.text_buffer = engine.merge_transcript_chunk(self.text_buffer, text)
        self.speech_allowed = False
        self._start_step(state)
        return self._arm_user_turn(now)

    def on_browser_audio_energy(
        self, state: CallState, pcm_16k: bytes, rms: float, now: float,
        *, rms_threshold: float = _DEFAULT_RMS_THRESHOLD,
    ) -> bool:
        """Count a real caller turn from mic audio, never from speakerphone echo."""
        idle = self.phase not in (TurnPhase.BOT_SPEAKING, TurnPhase.EVALUATING)
        if not (idle and now >= self.playback_ends_at):
            return False
        if not self.question_already_spoken(state) or self.in_quick_reply_stale_window(state, now):
            return False
        if rms < rms_threshold:
            if self.pcm_tail_chunks > 0:
                self.pcm_tail_chunks -= 1
                self._append_pcm(pcm_16k)
            return False
        self.quiet_from = 0.0
        new_step = self.user_turn_step != state.step_token
        fresh = not self.pcm_16k and not self.text_buffer
        if new_step or fresh:
            self.speech_allowed = False
            self.snapshot = self.snapshot or state.model_copy(deep=True)
        if new_step:
            self._start_step(state)
        counted = self._arm_user_turn(now) if (new_step or fresh) else False
        self.pcm_tail_chunks = USER_TURN_PCM_TAIL_CHUNKS
        self._append_pcm(pcm_16k)
        return counted

    def on_browser_text(self, state: CallState, text: str, now: float) -> bool:
        if not text.strip():
            return False
        self._clear_capture()
        self.text_buffer, self.snapshot = text, state.model_copy(deep=True)
        self.speech_allowed = False
        self._start_step(state)
        return self._arm_user_turn(now)

    def on_speech_end(self, state: CallState, now: float) -> bool:
        """Report whether the step was spoken; arm the silent-question watchdog."""
        spoken = self.question_already_spoken(state)
        playing = self.phase is TurnPhase.BOT_SPEAKING
        if playing or now < self.playback_ends_at - _PLAYBACK_SLACK_SEC:
            return spoken
        self.quiet_from = max(now, self.playback_ends_at)
        if not spoken and not self.awaiting_bot_reply:
            self._arm_waiting_reply(now, speech_end_waiting=True)
        return spoken

    def can_accept_quick_reply(self, state: CallState, step: str, outcome: str) -> bool:
        ready = not self.end_requested and state.phase is Phase.ASKING
        ready = ready and outcome in _QUICK_REPLY_OUTCOMES and engine.expect_yes_no(state)
        return bool(ready and step == state.step_token and self.question_already_spoken(state))

    def on_quick_reply_accepted(self, step: str, now: float) -> None:
        if self.phase is TurnPhase.BOT_SPEAKING:
            self.ignore_bot_audio, self.playback_ends_at = True, 0.0
        self.user_turn_step, self.pending_speak_step = "", ""
        self._clear_capture()
        self.last_quick_reply_step, self.last_quick_reply_ts = step, now

    def hold_reason_for_tool(self, state: CallState, now: float) -> str:
        if self.handled_current_turn and not self.in_quick_reply_stale_window(state, now):
            return "duplicate_turn"
        if self.is_stale_user_turn(state):
            return "stale_turn"
        if not self.question_already_spoken(state):
            return "not_spoken"
        return ""

    def should_run_missed_tool_on_turn_complete(self, state: CallState) -> bool:
        """True when speech arrived but the model never called `next_question`."""
        return bool(
            (self.text_buffer.strip() or self.pcm_16k) and not self.end_requested
            and state.phase is Phase.ASKING and not self.is_stale_user_turn(state)
            and not self.handled_current_turn and self.question_already_spoken(state)
        )

    def begin_evaluation(
        self, state: CallState
    ) -> tuple[str | None, str, bytes, CallState, tuple[str, str, int]]:
        """Lock the turn and hand the bridge its utterance, audio, and snapshot."""
        clean_utt, pcm_bytes = self.text_buffer.strip(), bytes(self.pcm_16k)
        dedup_key = (state.step_token, " ".join(clean_utt.split()), len(pcm_bytes))
        snapshot = self.snapshot
        if snapshot is None or snapshot.step_token != state.step_token:
            snapshot = state.model_copy(deep=True)
        refusal = "empty_turn" if not clean_utt and not pcm_bytes else ""
        refusal = refusal or ("duplicate_turn" if self.last_dedup_key == dedup_key else "")
        if refusal:
            return refusal, "", b"", snapshot, dedup_key
        self.last_dedup_key, self.handled_current_turn = dedup_key, True
        self._clear_capture(keep_turn_lock=True)
        self.phase = TurnPhase.EVALUATING
        return None, clean_utt, pcm_bytes, snapshot, dedup_key

    def finish_evaluation(self, *, step_changed: bool = False) -> None:
        self.speech_allowed = True
        self.phase = TurnPhase.CLOSING if self.end_requested else TurnPhase.AWAITING_BOT
        if step_changed:
            self.extractor_failures, self.extractor_failure_step = 0, ""

    def record_extractor_failure(self, step_token: str) -> bool:
        """Count failures per step; the second one in a row gives up on that step."""
        repeated = self.extractor_failure_step == step_token
        self.extractor_failures = self.extractor_failures + 1 if repeated else 1
        self.extractor_failure_step = step_token
        if self.extractor_failures < 2:
            return False
        self.extractor_failures, self.extractor_failure_step = 0, ""
        return True

    def mark_closing(self, now: float) -> bool:
        first_time = not self.end_requested
        self.end_requested, self.phase = True, TurnPhase.CLOSING
        if first_time:
            self.end_requested_at = now
        return first_time

    def on_speak_cue_sent(self, step_token: str, kind: SpeakKind) -> None:
        self.active_speak_kind, self.speak_cue_step = kind, step_token
        self.pending_speak_step = ""
        self.speech_allowed, self.ignore_bot_audio = True, False

    def on_turn_complete_cleanup(
        self, state: CallState, now: float, *, decision_ran: bool
    ) -> tuple[SpeakKind | None, bool]:
        """Reset the turn, then report the cue owed and whether to emit completion."""
        awaiting = self.awaiting_bot_reply
        stale = self.handled_current_turn or self.marked_quick_reply_stale
        stale = stale or bool(self.user_turn_step and self.user_turn_step != state.step_token)
        if stale and not decision_ran:
            self._clear_capture()
        self.ignore_bot_audio = False
        self.handled_current_turn, self.last_dedup_key = False, None
        owed = self.pending_speak_step == state.step_token and not self.bot_replied_since_user
        if self.end_requested:
            self.phase = TurnPhase.CLOSING
        elif awaiting:
            self.phase = TurnPhase.AWAITING_BOT
        else:
            self.phase, self.quiet_from = TurnPhase.LISTENING, max(now, self.playback_ends_at)
        return (self.pending_speak_kind if owed else None), (not awaiting or self.end_requested)

    def on_reconnected(self, now: float) -> bool:
        """Drop transient state after a resume; report whether a cue must be re-sent."""
        speaking = self.phase is TurnPhase.BOT_SPEAKING
        resume_speak = bool(self.awaiting_bot_reply or speaking or self.pending_speak_step)
        self.ignore_bot_audio = self.speech_end_waiting = False
        self.playback_ends_at, self.quiet_from = 0.0, now
        self._clear_capture()
        if resume_speak:
            self.phase = TurnPhase.CLOSING if self.end_requested else TurnPhase.AWAITING_BOT
        return resume_speak

    def mark_ending_notified(self) -> bool:
        if self.ending_notified:
            return False
        self.ending_notified = True
        return True

    def poll_watchdog(
        self, state: CallState, now: float, watchdog_timeout_sec: float
    ) -> WatchdogAction:
        """One pure timer tick: end the call, close it, or nudge a silent model."""
        elapsed = now - self.call_started_ts
        if elapsed >= HARD_LIMIT_SEC:
            return WatchdogAction.HARD_LIMIT_STOP
        if self.end_requested or self.phase is TurnPhase.CLOSING:
            done = self._closing_audio_finished(now) or (now - self.end_requested_at) > CLOSING_MAX_WAIT_SEC
            return WatchdogAction.CLOSING_COMPLETE if done else WatchdogAction.NONE
        if elapsed >= SOFT_LIMIT_SEC and not self.time_cue_sent:
            self.time_cue_sent = True
            return WatchdogAction.AUTO_CLOSE_TIME
        quiet = self._quiet_action(now)
        if quiet is not WatchdogAction.NONE:
            return quiet
        return self._nudge(state, now, watchdog_timeout_sec)

    def _quiet_action(self, now: float) -> WatchdogAction:
        busy = self.phase is TurnPhase.BOT_SPEAKING or self.awaiting_bot_reply
        if self.quiet_from <= 0 or busy:
            return WatchdogAction.NONE
        if now - self.quiet_from >= SILENCE_CLOSE_SEC:
            return WatchdogAction.AUTO_CLOSE_SILENCE
        if now - self.quiet_from >= CHECK_IN_SEC and not self.checked_in:
            self.checked_in = True
            return WatchdogAction.SILENCE_CHECK_IN
        return WatchdogAction.NONE

    def _nudge(self, state: CallState, now: float, watchdog_timeout_sec: float) -> WatchdogAction:
        if not self.awaiting_bot_reply or self.phase is TurnPhase.BOT_SPEAKING:
            return WatchdogAction.NONE
        timeout = watchdog_timeout_sec
        if self.speech_end_waiting:
            if self.question_already_spoken(state):
                self.bot_replied_since_user = True
                self.awaiting_bot_reply = self.speech_end_waiting = False
                return WatchdogAction.NONE
            timeout = min(watchdog_timeout_sec, _SPEECH_END_NUDGE_SEC)
        idle = now - self.last_user_turn_ts
        if self.bot_replied_since_user or self.last_user_turn_ts <= 0 or idle < timeout:
            return WatchdogAction.NONE
        if self.nudges_for_turn >= MAX_WATCHDOG_NUDGES:
            return WatchdogAction.NONE
        self.nudges_for_turn += 1
        self.last_user_turn_ts = now
        return WatchdogAction.WATCHDOG_NUDGE

    def _closing_audio_finished(self, now: float) -> bool:
        if self.phase is TurnPhase.BOT_SPEAKING or not self.closing_heard:
            return False
        return self.playback_ends_at > 0 and now >= self.playback_ends_at + CLOSING_AUDIO_GRACE_SEC

    def _start_step(self, state: CallState) -> None:
        self.user_turn_step, self.last_quick_reply_step = state.step_token, ""
        self.handled_current_turn, self.last_dedup_key = False, None

    def _arm_waiting_reply(self, now: float, *, speech_end_waiting: bool = False) -> None:
        self.awaiting_bot_reply = True
        self.last_user_turn_ts, self.nudges_for_turn = now, 0
        self.bot_replied_since_user = False
        self.speech_end_waiting = speech_end_waiting

    def _arm_user_turn(self, now: float) -> bool:
        """Start a caller turn; report whether telemetry should count a new user turn."""
        counted = not self.awaiting_bot_reply or self.speech_end_waiting
        self.phase = TurnPhase.CAPTURING_USER
        self._arm_waiting_reply(now)
        self.quiet_from, self.checked_in = 0.0, False
        self.active_speak_kind = SpeakKind.SPEAK
        return counted

    def _append_pcm(self, pcm_16k: bytes) -> None:
        available = USER_TURN_PCM_MAX_BYTES - len(self.pcm_16k)
        piece = pcm_16k[:available] if (available > 0 and pcm_16k) else b""
        piece = piece[:-1] if len(piece) % 2 else piece
        if piece:
            self.pcm_16k.extend(piece)

    def _clear_capture(self, *, keep_pcm: bool = False, keep_turn_lock: bool = False) -> None:
        self.text_buffer, self.snapshot = "", None
        self.input_finalized = self.marked_quick_reply_stale = False
        if not keep_pcm:
            self.pcm_16k.clear()
            self.pcm_tail_chunks = 0
        if not keep_turn_lock:
            self.handled_current_turn, self.last_dedup_key = False, None
