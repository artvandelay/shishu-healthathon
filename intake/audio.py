"""PCM helpers and the wall-clock 16 kHz stereo WAV recorder (`intake/audio.py`)."""

from __future__ import annotations

import base64, io, threading, time, wave
from pathlib import Path

import numpy as np

TARGET_RATE = 16000

def _samples(pcm: bytes) -> np.ndarray: return np.frombuffer(bytes(pcm[: len(pcm) - len(pcm) % 2]), dtype=np.int16)

def pcm_bytes(raw: object) -> bytes | None:
    if isinstance(raw, str):
        try: raw = base64.b64decode(raw)
        except Exception: return None  # noqa: BLE001
    return (bytes(raw[: len(raw) - len(raw) % 2]) or None) if isinstance(raw, (bytes, bytearray)) and raw else None
def pcm_rms(pcm: bytes) -> float:
    s = _samples(pcm).astype(np.float64)
    return float(np.sqrt(np.mean(s * s))) / 32768.0 if s.size else 0.0
def rate_from_mime(mime: str, default: int = 24000) -> int:
    for piece in (mime or "").split(";"):
        k, _, v = piece.strip().lower().partition("=")
        if k == "rate" and v.isdigit() and int(v) > 0: return int(v)
    return default
def pcm16_to_wav_bytes(pcm: bytes, sample_rate: int = TARGET_RATE) -> bytes:
    if sample_rate <= 0: raise ValueError(f"sample_rate must be positive, got {sample_rate!r}")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setparams((1, 2, int(sample_rate), 0, "NONE", "not compressed"))
        wf.writeframes(_samples(pcm).tobytes())
    return buf.getvalue()
def resample_int16_mono(pcm: bytes, src_rate: int, dst_rate: int = TARGET_RATE) -> bytes:
    src = _samples(pcm)
    if src.size == 0 or src_rate == dst_rate or src_rate <= 0: return src.tobytes()
    out = np.interp(np.linspace(0, src.size - 1, max(1, round(src.size * dst_rate / src_rate))), np.arange(src.size), src)
    return np.clip(np.rint(out), -32768, 32767).astype(np.int16).tobytes()
class CallAudioRecorder:
    def __init__(self, call_id: int, output_dir: str | Path) -> None:
        self.call_id, self.output_dir = call_id, Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._chunks: dict[str, list[tuple[float, bytes, int]]] = {"user": [], "bot": []}
        self._lock, self._closed, self._ended, self._started = threading.Lock(), False, None, time.monotonic()
    def mark_ended(self) -> None:
        if self._ended is None: self._ended = time.monotonic()
    def wall_seconds(self) -> int: return max(0, round((self._ended if self._ended is not None else time.monotonic()) - self._started))
    def add_user_pcm(self, pcm: bytes, sample_rate: int = TARGET_RATE) -> None: self._queue("user", pcm, sample_rate)
    def add_bot_pcm(self, pcm: bytes, sample_rate: int = 24000) -> None: self._queue("bot", pcm, sample_rate)
    def _queue(self, ch: str, pcm: bytes, rate: int) -> None:
        with self._lock:
            if not self._closed and pcm: self._chunks[ch].append((max(0.0, time.monotonic() - self._started), bytes(pcm), int(rate)))
    @staticmethod
    def _timeline(raw: list[tuple[float, bytes, int]]) -> np.ndarray:
        pieces, filled = [], 0
        stamps = sorted(((max(0, round(e * TARGET_RATE)), _samples(resample_int16_mono(p, r))) for e, p, r in raw), key=lambda x: x[0])
        for start, chunk in stamps:
            if start > filled:
                pieces.append(np.zeros(start - filled, dtype=np.int16))
                filled = start
            pieces.append(chunk)
            filled += chunk.size
        return np.concatenate(pieces) if pieces else np.empty(0, dtype=np.int16)
    def save(self) -> tuple[str, int]:
        self.mark_ended()
        with self._lock: self._closed, (user, bot), elapsed = True, (self._timeline(c) for c in self._chunks.values()), self.wall_seconds()
        frames, filename = max(elapsed * TARGET_RATE, user.size, bot.size, 1), f"call_{self.call_id}.wav"
        with wave.open(str(self.output_dir / filename), "wb") as wf:
            wf.setparams((2, 2, TARGET_RATE, 0, "NONE", "not compressed"))
            wf.writeframes(np.column_stack([np.pad(c, (0, frames - c.size)) for c in (user, bot)]).ravel().tobytes())
        return filename, elapsed
