"""Stereo wall-clock recorder. No network."""

import os
import tempfile
import time
import unittest
import wave
from pathlib import Path

_TMP = tempfile.mkdtemp(prefix="shishu-v3-audio-")
os.environ["DATA_DIR"] = _TMP

import audio_recorder  # noqa: E402


class StereoWavTests(unittest.TestCase):
    def test_two_channels_are_16k_and_wall_clock(self):
        folder = Path(_TMP) / "audio"
        rec = audio_recorder.CallAudioRecorder(3, output_dir=str(folder))
        rec._started = time.monotonic() - 2.2
        rec.add_user_pcm(b"\x10\x00" * 160, 16000)
        rec.add_bot_pcm(b"\x20\x00" * 240, 24000)
        name, duration = rec.save()
        self.assertGreaterEqual(duration, 2)
        self.assertLess(duration, 30)
        with wave.open(str(folder / name), "rb") as wf:
            self.assertEqual(wf.getnchannels(), 2)
            self.assertEqual(wf.getframerate(), 16000)
            self.assertEqual(wf.getsampwidth(), 2)
            frames = wf.readframes(wf.getnframes())
        self.assertGreater(len(frames), 44)
        self.assertIn(b"\x10\x00", frames)
        self.assertTrue(frames)
