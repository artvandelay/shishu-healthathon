"""Stereo wall-clock recorder. No network."""

import os
import sys
import tempfile
import time
import unittest
import wave
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
import tests  # noqa: E402, F401

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

    def test_resample_int16_mono_and_pcm16_to_wav_bytes(self):
        pcm = b"\x00\x00\xe8\x03" * 24 + b"\x01"
        resampled = audio_recorder.resample_int16_mono(pcm, 24000, 16000)
        self.assertEqual(len(resampled) % 2, 0)
        self.assertGreater(len(resampled), 0)
        wav_bytes = audio_recorder.pcm16_to_wav_bytes(pcm, sample_rate=16000)
        self.assertTrue(wav_bytes.startswith(b"RIFF"))
        import io
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            self.assertEqual(wf.getnchannels(), 1)
            self.assertEqual(wf.getframerate(), 16000)
            self.assertEqual(wf.getsampwidth(), 2)
            self.assertEqual(wf.readframes(wf.getnframes()), pcm[:-1])
