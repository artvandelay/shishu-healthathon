"""Behavioral tests for `intake.vad` (real Silero model) and `intake.audio`."""

from __future__ import annotations

import base64
import tempfile
import time
import unittest
import wave
from pathlib import Path

import numpy as np

from intake.audio import (
    CallAudioRecorder,
    pcm16_to_wav_bytes,
    pcm_bytes,
    pcm_rms,
    rate_from_mime,
)
from intake.vad import SileroDetector, ThinkingVad, VadEvent

REPO_ROOT = Path(__file__).resolve().parents[1]
U4 = REPO_ROOT / "scripts" / "calibration_audio" / "u4_jaundice_16k.wav"


def read_wav(path: Path) -> bytes:
    with wave.open(str(path), "rb") as wf:
        assert wf.getnchannels() == 1 and wf.getsampwidth() == 2 and wf.getframerate() == 16000
        return wf.readframes(wf.getnframes())


class SileroDetectorTests(unittest.TestCase):
    """The real ONNX graph must score the calibration clip sanely."""

    def test_partial_frames_are_buffered_until_a_full_1024_byte_frame_arrives(self):
        pcm = read_wav(U4)
        detector = SileroDetector()
        self.assertEqual(detector.feed_pcm16(pcm[:1000]), [])  # no frame yet
        self.assertEqual(len(detector._buf), 1000)
        self.assertEqual(len(detector.feed_pcm16(pcm[1000:2048])), 2)  # 2048 bytes -> two frames
        self.assertEqual(len(detector._buf), 0)


class ThinkingVadTurnTests(unittest.TestCase):
    """Turn boundaries, speculation, pace sampling, and the self-healing resume."""

    def test_extend_for_incomplete_holds_the_endpoint_until_the_caller_resumes(self):
        vad = ThinkingVad()
        vad.on_frame(0.9)
        vad.extend_for_incomplete()
        self.assertTrue(vad.holding_incomplete)
        silent = [e for _ in range(60) for e in vad.on_frame(0.05)]
        self.assertNotIn(VadEvent.SPEECH_END, silent, "held for 2000 ms")
        vad.on_frame(0.9)
        self.assertFalse(vad.holding_incomplete, "resuming clears the hold")

    def test_vad_endpoint_wait_is_not_sampled_as_a_caller_pause(self):
        vad = ThinkingVad()
        vad.on_frame(0.9)
        self.assertIn(VadEvent.SPEECH_END, [e for _ in range(12) for e in vad.on_frame(0.05)], "384 ms closes the turn")
        for _ in range(2): vad.on_frame(0.05)
        vad.on_frame(0.9)  # caller resumes before `on_bot_speaking`
        self.assertEqual(vad.pauses_ms, [], "the VAD's own 448 ms endpoint wait is not a speaker pause")
        self.assertEqual(vad.pace_factor, 1.0)

        reflective = ThinkingVad()
        self.assertIn(VadEvent.SPEECH_START, reflective.on_frame(0.9, is_reflective=True))
        for _ in range(8): reflective.on_frame(0.05, is_reflective=True)
        reflective.on_frame(0.9, is_reflective=True)  # 256 ms stays inside the reflective wait
        self.assertEqual(reflective.pauses_ms, [256.0], "a real within-turn pause is still sampled")


class AudioHelperTests(unittest.TestCase):
    def test_pcm_bytes_accepts_base64_and_drops_an_odd_tail_byte(self):
        raw = b"\x00\x40" * 4
        self.assertEqual(pcm_bytes(base64.b64encode(raw + b"\x01").decode()), raw)
        self.assertIsNone(pcm_bytes("not-valid-base64!!!"))
        self.assertIsNone(pcm_bytes(""))
        self.assertIsNone(pcm_bytes(12))

    def test_pcm_rms_matches_expected_levels(self):
        self.assertAlmostEqual(pcm_rms(b"\x00\x40" * 4), 16384 / 32768)
        self.assertAlmostEqual(pcm_rms(b"\x00\x80" * 4), 32768 / 32768)
        self.assertEqual(pcm_rms(b""), 0.0)
        self.assertEqual(pcm_rms(b"\x00"), 0.0)  # a lone byte is not a sample

    def test_rate_from_mime_reads_rate_and_falls_back(self):
        self.assertEqual(rate_from_mime("audio/pcm;rate=24000", 16000), 24000)
        self.assertEqual(rate_from_mime("audio/pcm;rate=bad", 16000), 16000)
        self.assertEqual(rate_from_mime("audio/pcm", 16000), 16000)
        self.assertEqual(rate_from_mime("audio/pcm;rate=0", 24000), 24000)
        self.assertEqual(rate_from_mime(""), 24000)

    def test_pcm16_to_wav_bytes_rejects_a_non_positive_rate(self):
        with self.assertRaises(ValueError):
            pcm16_to_wav_bytes(b"\x00\x00", sample_rate=0)


class CallAudioRecorderTests(unittest.TestCase):
    def test_save_writes_16k_stereo_where_channel_one_is_the_caller(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec = CallAudioRecorder(3, tmp)
            rec._started = time.monotonic() - 2.2
            rec.add_user_pcm(b"\x10\x00" * 160, 16000)
            rec.add_bot_pcm(b"\x20\x00" * 240, 24000)
            name, duration = rec.save()
            self.assertEqual(name, "call_3.wav")
            self.assertGreaterEqual(duration, 2)
            self.assertLess(duration, 30)
            with wave.open(str(Path(tmp) / name), "rb") as wf:
                self.assertEqual((wf.getnchannels(), wf.getframerate(), wf.getsampwidth()), (2, 16000, 2))
                frames = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
        stereo = frames.reshape(-1, 2)
        self.assertIn(16, stereo[:, 0], "caller audio sits on channel 1")
        self.assertIn(32, stereo[:, 1], "bot audio sits on channel 2")

    def test_bursts_keep_every_word_instead_of_overwriting(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec = CallAudioRecorder(13, tmp)
            first, second = b"\x01\x00" * 400, b"\x02\x00" * 400
            rec.add_user_pcm(first, 16000)
            rec._chunks["user"].append((rec._chunks["user"][0][0], second, 16000))
            rec.add_bot_pcm(b"", 24000)
            placed = rec._timeline(rec._chunks["user"])
        self.assertGreaterEqual(placed.size, 800, "a same-timestamp burst appends")
        self.assertIn(1, placed)
        self.assertIn(2, placed)


if __name__ == "__main__":
    unittest.main()
