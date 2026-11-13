"""Wall-clock stereo WAV for one call. Channel 1 is the mother, channel 2 is SHISHU.

save() is synchronous and does the resample and the mix off the event-loop
thread. The live session calls it with asyncio.to_thread. _remember only
stores the raw chunk.
"""

import os
import threading
import time
import wave
from array import array

import db

TARGET_RATE = 16000


def _resample_int16_mono(pcm_bytes: bytes, src_rate: int, dst_rate: int = TARGET_RATE) -> bytes:
    if not pcm_bytes:
        return b""
    if src_rate == dst_rate or src_rate <= 0:
        return pcm_bytes

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


def _timeline(chunks: list[tuple[int, array]]) -> array:
    """Place each chunk at its wall-clock sample, without overwriting earlier speech.

    A burst that arrives faster than realtime continues after the previous
    chunk, so the words stay intact. A later chunk keeps the silent gap.
    """
    if not chunks:
        return array("h")
    ordered = sorted(chunks, key=lambda item: item[0])
    end = 0
    for start, samples in ordered:
        end = max(end, max(0, start) + len(samples))
    out = array("h", [0] * end)
    cursor = 0
    for start, samples in ordered:
        at = start if start > cursor else cursor
        if at < 0:
            at = 0
        need = at + len(samples)
        if need > len(out):
            out.extend([0] * (need - len(out)))
        out[at : at + len(samples)] = samples
        cursor = at + len(samples)
    return out


class CallAudioRecorder:
    def __init__(self, call_id: int, output_dir: str = os.path.join(db.DATA_DIR, "audio")) -> None:
        self.call_id = call_id
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self._user: list[tuple[float, bytes, int]] = []
        self._bot: list[tuple[float, bytes, int]] = []
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

    def _remember(self, bucket: list[tuple[float, bytes, int]], pcm_bytes: bytes, sample_rate: int) -> None:
        if self._closed or not pcm_bytes:
            return
        pcm = bytes(pcm_bytes)
        rate = int(sample_rate)
        with self._lock:
            if self._closed:
                return
            elapsed = max(0.0, time.monotonic() - self._started)
            bucket.append((elapsed, pcm, rate))

    def add_user_pcm(self, pcm_bytes: bytes, sample_rate: int) -> None:
        self._remember(self._user, pcm_bytes, sample_rate)

    def add_bot_pcm(self, pcm_bytes: bytes, sample_rate: int = 24000) -> None:
        self._remember(self._bot, pcm_bytes, sample_rate)

    def _placed(self, raw: list[tuple[float, bytes, int]]) -> list[tuple[int, array]]:
        placed: list[tuple[int, array]] = []
        for elapsed, pcm, rate in raw:
            resampled = _resample_int16_mono(pcm, rate, TARGET_RATE)
            if len(resampled) % 2 != 0:
                resampled = resampled[:-1]
            if not resampled:
                continue
            samples = array("h")
            samples.frombytes(resampled)
            if not samples:
                continue
            start = int(round(elapsed * TARGET_RATE))
            if start < 0:
                start = 0
            placed.append((start, samples))
        return placed

    def save(self) -> tuple[str, int]:
        self.mark_ended()
        with self._lock:
            self._closed = True
            user_chunks = list(self._user)
            bot_chunks = list(self._bot)
            elapsed = self.wall_seconds()

        user = _timeline(self._placed(user_chunks))
        bot = _timeline(self._placed(bot_chunks))
        frames = max(elapsed * TARGET_RATE, len(user), len(bot), 1)
        if len(user) < frames:
            user.extend([0] * (frames - len(user)))
        if len(bot) < frames:
            bot.extend([0] * (frames - len(bot)))

        stereo = array("h", [0] * (frames * 2))
        for index in range(frames):
            stereo[index * 2] = user[index]
            stereo[index * 2 + 1] = bot[index]

        filename = f"call_{self.call_id}.wav"
        filepath = os.path.join(self.output_dir, filename)
        with wave.open(filepath, "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(TARGET_RATE)
            wf.writeframes(stereo.tobytes())
        return filename, elapsed
