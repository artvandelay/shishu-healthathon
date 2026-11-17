"""Behavioral tests for `intake.audio`: PCM helpers and the stereo WAV recorder."""

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

    def test_save_with_append_existing_concatenates_prior_call_wav_segments(self):
        """A resumed call keeps the audio already on disk as segment one, not just segment two.

        Regression: a dropped socket reopened the call on the same id, so the second recorder
        overwrote `call_7.wav` and the first minute of the conversation vanished from the file.
        """
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            first = CallAudioRecorder(7, folder)
            first.add_user_pcm(np.full(16000, 1000, dtype=np.int16).tobytes(), 16000)
            first.add_bot_pcm(np.full(24000, 2000, dtype=np.int16).tobytes(), 24000)
            first._started = time.monotonic() - 1.0
            name_one, duration_one = first.save()
            self.assertEqual(name_one, "call_7.wav")
            self.assertEqual(duration_one, 1)

            second = CallAudioRecorder(7, folder, append_existing=True)
            second.add_user_pcm(np.full(16000, 3000, dtype=np.int16).tobytes(), 16000)
            second.add_bot_pcm(np.full(24000, 4000, dtype=np.int16).tobytes(), 24000)
            second._started = time.monotonic() - 1.0
            name_two, duration_two = second.save()
            self.assertEqual(name_two, "call_7.wav")
            self.assertGreaterEqual(duration_two, 2)

            self.assertEqual(second.save(), (name_two, duration_two),
                             "a repeated save on one recorder must not append segment two twice")

            with wave.open(str(folder / "call_7.wav"), "rb") as wf:
                self.assertEqual(wf.getnchannels(), 2)
                self.assertEqual(wf.getsampwidth(), 2)
                self.assertEqual(wf.getframerate(), 16000)
                self.assertGreaterEqual(wf.getnframes(), 32000)
                raw = wf.readframes(wf.getnframes())

        stereo = np.frombuffer(raw, dtype=np.int16).reshape(-1, 2)
        self.assertLess(stereo.shape[0], 34000, "segment two is appended exactly once")
        for frame in (100, 15000):
            np.testing.assert_allclose(stereo[frame], [1000, 2000], atol=2,
                                       err_msg=f"segment one survives at frame {frame}")
        for frame in (16100, 31800):
            np.testing.assert_allclose(stereo[frame], [3000, 4000], atol=2,
                                       err_msg=f"segment two lands after segment one at frame {frame}")


if __name__ == "__main__":
    unittest.main()
