"""Silero ONNX speech probability plus the adaptive Thinking VAD endpoint clock."""

from __future__ import annotations

import statistics
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path

import numpy as np
import onnxruntime as ort

_SR, _FRAME = np.array(16000, dtype=np.int64), 1024
_MODEL = Path(__file__).resolve().parent / "assets" / "silero_vad.onnx"
DEFAULTS = dict(speech_prob_threshold=0.5, closed_short_ms=350.0, reflective_ms=850.0, ramp_start_voiced_ms=600.0, ramp_full_voiced_ms=1800.0,
                speculate_at_ms=280.0, incomplete_hold_ms=2000.0, baseline_pause_ms=260.0, pace_min=0.8, pace_max=1.45)
class SileroDetector:
    _session: ort.InferenceSession | None = None
    def __init__(self) -> None:
        if SileroDetector._session is None:
            ort.disable_telemetry_events()
            SileroDetector._session = ort.InferenceSession(str(_MODEL), providers=["CPUExecutionProvider"])
        self._sess, self._buf = SileroDetector._session, bytearray()
        self.reset()
    def reset(self) -> None:
        self._state, self._context = np.zeros((2, 1, 128), dtype=np.float32), np.zeros((1, 64), dtype=np.float32)
        self._buf.clear()
    def feed_pcm16(self, pcm_16k: bytes) -> list[float]:
        self._buf.extend(pcm_16k)
        probs = []
        while len(self._buf) >= _FRAME:
            audio = (np.frombuffer(bytes(self._buf[:_FRAME]), dtype=np.int16) / 32768.0).astype(np.float32)[None, :]
            del self._buf[:_FRAME]
            inp = np.concatenate([self._context, audio], axis=1)
            out, self._state = self._sess.run(None, dict(input=inp, state=self._state, sr=_SR))
            self._context = inp[:, -64:]
            probs.append(float(out[0][0]))
        return probs
class VadEvent(StrEnum):
    NONE, SPEECH_START, SPECULATE, SPEECH_END = "none", "speech_start", "speculate", "speech_end"
class ThinkingVad:
    def __init__(self, cfg: Mapping[str, object] | None = None) -> None:
        for k, v in {**DEFAULTS, **(cfg or {})}.items(): setattr(self, k, float(v))
        self.pauses_ms: list[float] = []
        self.on_bot_speaking()
    @property
    def pace_factor(self) -> float:
        raw = (statistics.median(self.pauses_ms) if self.pauses_ms else self.baseline_pause_ms) / self.baseline_pause_ms
        return min(self.pace_max, max(self.pace_min, 1 + min(1.0, len(self.pauses_ms) / 8.0) * (raw - 1)))
    def target_wait_ms(self, is_reflective: bool = False) -> float:
        if self.holding_incomplete: return self.incomplete_hold_ms
        ramp = min(1.0, max(0.0, (self.voiced_ms - self.ramp_start_voiced_ms) / max(1.0, self.ramp_full_voiced_ms - self.ramp_start_voiced_ms)))
        base = self.reflective_ms if is_reflective else self.closed_short_ms + (self.reflective_ms - self.closed_short_ms) * ramp
        return base * self.pace_factor
    def speculate_deadline_ms(self, is_reflective: bool = False) -> float:
        """Speculation keeps 96 ms of headroom inside the wait window, even at `pace_min` on a short prefix."""
        return min(self.speculate_at_ms, max(160.0, self.target_wait_ms(is_reflective) - 96.0))
    def on_frame(self, prob: float, *, frame_ms: float = 32.0, is_reflective: bool = False) -> list[VadEvent]:
        if prob >= self.speech_prob_threshold:
            onset = not self.in_turn
            if not onset and 150.0 <= self.silence_ms <= 850.0: self.pauses_ms.append(self.silence_ms)
            self.voiced_ms, self.silence_ms = (frame_ms if onset else self.voiced_ms + frame_ms), 0.0
            self.in_turn, self.speculated, self.holding_incomplete = True, False, False
            return [VadEvent.SPEECH_START] if onset else [VadEvent.NONE]
        if not self.in_turn: return [VadEvent.NONE]
        self.silence_ms += frame_ms
        if self.silence_ms >= self.target_wait_ms(is_reflective):
            self.in_turn = False
            return [VadEvent.SPEECH_END]
        fire = not self.speculated and self.silence_ms >= self.speculate_deadline_ms(is_reflective)
        self.speculated |= fire
        return [VadEvent.SPECULATE] if fire else [VadEvent.NONE]
    def extend_for_incomplete(self) -> None: self.holding_incomplete |= self.in_turn
    def on_bot_speaking(self) -> None:
        self.in_turn = self.speculated = self.holding_incomplete = False
        self.voiced_ms, self.silence_ms = 0.0, 0.0
