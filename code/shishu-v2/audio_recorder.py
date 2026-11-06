import os
import threading
import time
import wave
from array import array

import db

TARGET_RATE = 24000


def _resample_int16_mono(pcm_bytes: bytes, src_rate: int, dst_rate: int = TARGET_RATE) -> bytes:
    if not pcm_bytes:
        return b""
    if src_rate == dst_rate or src_rate <= 0:
        return pcm_bytes

    # Ensure even byte length for int16 samples
    if len(pcm_bytes) % 2 != 0:
        pcm_bytes = pcm_bytes[:-1]
    if not pcm_bytes:
        return b""

    src_samples = array("h")
    src_samples.frombytes(pcm_bytes)
    n_src = len(src_samples)
    if n_src == 0:
        return b""
    if n_src == 1:
        ratio = dst_rate / src_rate
        n_dst = max(1, int(round(n_src * ratio)))
        dst_samples = array("h", [src_samples[0]] * n_dst)
        return dst_samples.tobytes()

    n_dst = int(round(n_src * dst_rate / src_rate))
    if n_dst <= 0:
        return b""

    dst_samples = array("h", [0] * n_dst)
    scale = (n_src - 1) / max(1, (n_dst - 1))
    for i in range(n_dst):
        pos = i * scale
        idx = int(pos)
        if idx >= n_src - 1:
            dst_samples[i] = src_samples[n_src - 1]
        else:
            frac = pos - idx
            val = src_samples[idx] * (1.0 - frac) + src_samples[idx + 1] * frac
            iv = int(round(val))
            if iv > 32767:
                iv = 32767
            elif iv < -32768:
                iv = -32768
            dst_samples[i] = iv

    return dst_samples.tobytes()


class CallAudioRecorder:
    def __init__(self, call_id: int, output_dir: str = os.path.join(db.DATA_DIR, "audio")) -> None:
        self.call_id = call_id
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self._buffer = bytearray()
        self._lock = threading.Lock()
        self._closed = False
        self._started = time.monotonic()
        self._ended: float | None = None

    def mark_ended(self) -> None:
        if self._ended is None:
            self._ended = time.monotonic()

    def wall_seconds(self) -> int:
        end = self._ended if self._ended is not None else time.monotonic()
        return max(0, int(round(end - self._started)))

    def add_user_pcm(self, pcm_bytes: bytes, sample_rate: int) -> None:
        if self._closed or not pcm_bytes:
            return
        resampled = _resample_int16_mono(pcm_bytes, sample_rate, TARGET_RATE)
        with self._lock:
            self._buffer.extend(resampled)

    def add_bot_pcm(self, pcm_bytes: bytes, sample_rate: int = 24000) -> None:
        if self._closed or not pcm_bytes:
            return
        resampled = (
            _resample_int16_mono(pcm_bytes, sample_rate, TARGET_RATE)
            if sample_rate != TARGET_RATE
            else pcm_bytes
        )
        with self._lock:
            self._buffer.extend(resampled)

    def save(self) -> tuple[str, int]:
        self.mark_ended()
        with self._lock:
            self._closed = True
            raw_bytes = bytes(self._buffer)

        filename = f"call_{self.call_id}.wav"
        filepath = os.path.join(self.output_dir, filename)

        with wave.open(filepath, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(TARGET_RATE)
            wf.writeframes(raw_bytes)

        return filename, self.wall_seconds()
