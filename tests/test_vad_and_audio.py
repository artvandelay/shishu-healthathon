"""Behavioral tests for `intake.vad` (real Silero model) and `intake.audio`."""

from __future__ import annotations

import base64
import io
import tempfile
import time
import unittest
import wave
from pathlib import Path

import numpy as np

from intake.audio import (
    TARGET_RATE,
    CallAudioRecorder,
    pcm16_to_wav_bytes,
    pcm_bytes,
    pcm_rms,
    rate_from_mime,
    resample_int16_mono,
)
from intake.vad import SileroDetector, ThinkingVad, VadEvent

REPO_ROOT = Path(__file__).resolve().parents[1]
U4 = REPO_ROOT / "scripts" / "calibration_audio" / "u4_jaundice_16k.wav"
FRAME_MS = 32.0


def read_wav(path: Path) -> bytes:
    with wave.open(str(path), "rb") as wf:
        assert wf.getnchannels() == 1 and wf.getsampwidth() == 2 and wf.getframerate() == 16000
        return wf.readframes(wf.getnframes())


def drive(vad: ThinkingVad, probs, *, is_reflective: bool = False) -> list[VadEvent]:
    events: list[VadEvent] = []
    for prob in probs:
        events.extend(vad.on_frame(prob, is_reflective=is_reflective))
    return events


def frames(*runs) -> list[float]:
    """Build a probability trace from (probability, frame count) runs."""
    return [prob for prob, count in runs for _ in range(count)]


class SileroDetectorTests(unittest.TestCase):
    """The real ONNX graph must score the calibration clip sanely."""

    def test_u4_jaundice_is_one_frame_per_32ms_with_bounded_probabilities(self):
        pcm = read_wav(U4)
        detector = SileroDetector()
        probs = detector.feed_pcm16(pcm)
        self.assertEqual(len(probs), len(pcm) // 1024)
        self.assertEqual(len(probs), 87)
        self.assertTrue(all(0.0 <= p <= 1.0 for p in probs))
        self.assertGreater(max(probs), 0.9, "the clip is clearly speech")
        self.assertLess(min(probs), 0.2, "the clip has real silence")

    def test_u4_speech_opens_a_turn_and_records_its_internal_pause(self):
        probs = SileroDetector().feed_pcm16(read_wav(U4))
        vad = ThinkingVad()
        events = drive(vad, probs, is_reflective=True)
        self.assertEqual(events.count(VadEvent.SPEECH_START), 1)
        self.assertTrue(vad.in_turn, "the clip ends before a reflective 850 ms wait expires")
        self.assertEqual(vad.pauses_ms, [288.0], "9 silent frames inside the utterance")

    def test_u4_turn_closes_once_the_caller_actually_stops(self):
        probs = SileroDetector().feed_pcm16(read_wav(U4))
        vad = ThinkingVad()
        drive(vad, probs, is_reflective=True)
        events = [e for _ in range(60) for e in vad.on_frame(0.01, is_reflective=True)]
        self.assertEqual(events.count(VadEvent.SPEECH_END), 1)
        self.assertFalse(vad.in_turn)

    def test_partial_frames_are_buffered_until_a_full_1024_byte_frame_arrives(self):
        pcm = read_wav(U4)
        detector = SileroDetector()
        self.assertEqual(detector.feed_pcm16(pcm[:1000]), [])  # no frame yet
        self.assertEqual(len(detector._buf), 1000)
        self.assertEqual(len(detector.feed_pcm16(pcm[1000:2048])), 2)  # 2048 bytes -> two frames
        self.assertEqual(len(detector._buf), 0)

    def test_reset_clears_state_context_and_remainder_and_replays_identically(self):
        pcm = read_wav(U4)
        detector = SileroDetector()
        first = detector.feed_pcm16(pcm)
        detector.feed_pcm16(pcm[:500])
        detector.reset()
        self.assertEqual(len(detector._buf), 0)
        self.assertFalse(detector._state.any() or detector._context.any())
        np.testing.assert_allclose(detector.feed_pcm16(pcm), first, atol=1e-6)

    def test_detector_instances_share_one_session(self):
        self.assertIs(SileroDetector()._sess, SileroDetector()._sess)


class ThinkingVadPaceTests(unittest.TestCase):
    """The two-factor formula: question-kind ramp times measured caller pace."""

    def test_defaults_match_the_pack_vad_block(self):
        vad = ThinkingVad()
        self.assertEqual(
            (vad.speech_prob_threshold, vad.closed_short_ms, vad.reflective_ms),
            (0.5, 350.0, 850.0),
        )
        self.assertEqual(
            (vad.ramp_start_voiced_ms, vad.ramp_full_voiced_ms, vad.speculate_at_ms),
            (600.0, 1800.0, 280.0),
        )
        self.assertEqual(
            (vad.incomplete_hold_ms, vad.baseline_pause_ms, vad.pace_min, vad.pace_max),
            (2000.0, 260.0, 0.80, 1.45),
        )

    def test_cfg_overrides_defaults(self):
        vad = ThinkingVad({"closed_short_ms": 120.0, "pace_max": 2.0})
        self.assertEqual(vad.closed_short_ms, 120.0)
        self.assertEqual(vad.pace_max, 2.0)
        self.assertEqual(vad.reflective_ms, 850.0)

    def test_pace_factor_is_one_before_any_pause_is_observed(self):
        self.assertEqual(ThinkingVad().pace_factor, 1.0)

    def test_pace_factor_is_shrunk_by_trust_then_clamped(self):
        brisk, slow, early = ThinkingVad(), ThinkingVad(), ThinkingVad()
        brisk.pauses_ms = [170.0] * 8
        slow.pauses_ms = [390.0] * 8
        early.pauses_ms = [500.0] * 2
        self.assertAlmostEqual(brisk.pace_factor, 0.80)  # 0.65 clamped up
        self.assertAlmostEqual(slow.pace_factor, 1.45)  # 1.50 clamped down
        self.assertAlmostEqual(early.pace_factor, 1.0 + 0.25 * (500 / 260 - 1.0))

    def test_closed_wait_ramps_with_voiced_length(self):
        vad = ThinkingVad()
        vad.voiced_ms = 0.0
        self.assertAlmostEqual(vad.target_wait_ms(), 350.0)
        vad.voiced_ms = 600.0
        self.assertAlmostEqual(vad.target_wait_ms(), 350.0)
        vad.voiced_ms = 1800.0
        self.assertAlmostEqual(vad.target_wait_ms(), 850.0)
        vad.voiced_ms = 5000.0
        self.assertAlmostEqual(vad.target_wait_ms(), 850.0)  # ramp saturates

    def test_reflective_wait_is_flat_at_850_and_scales_with_pace(self):
        vad = ThinkingVad()
        vad.voiced_ms = 0.0
        self.assertAlmostEqual(vad.target_wait_ms(is_reflective=True), 850.0)
        vad.pauses_ms = [390.0] * 8
        self.assertAlmostEqual(vad.target_wait_ms(is_reflective=True), 850.0 * 1.45)
        self.assertAlmostEqual(vad.target_wait_ms(is_reflective=True), 1232.5)

    def test_incomplete_hold_overrides_both_kinds(self):
        vad = ThinkingVad()
        vad.holding_incomplete = True
        self.assertEqual(vad.target_wait_ms(), 2000.0)
        self.assertEqual(vad.target_wait_ms(is_reflective=True), 2000.0)


class ThinkingVadTurnTests(unittest.TestCase):
    """Turn boundaries, speculation, pace sampling, and the self-healing resume."""

    def test_short_closed_answer_opens_then_closes_at_350ms(self):
        vad = ThinkingVad()
        self.assertEqual(vad.on_frame(0.9), [VadEvent.SPEECH_START])
        events = drive(vad, frames((0.9, 11), (0.1, 11)))
        self.assertEqual([e for e in events if e is not VadEvent.NONE], [VadEvent.SPECULATE, VadEvent.SPEECH_END])
        self.assertEqual(vad.voiced_ms, 384.0)
        self.assertAlmostEqual(vad.post_end_silence_ms, 352.0, "11 frames of 32 ms")
        self.assertFalse(vad.in_turn)

    def test_speculation_fires_once_per_turn_inside_the_wait_window(self):
        vad = ThinkingVad()
        silent = [e for _ in range(40) for e in vad.on_frame(0.9, frame_ms=32.0)][:1]
        self.assertEqual(silent, [VadEvent.SPEECH_START])
        events = [e for _ in range(40) for e in vad.on_frame(0.05)]
        self.assertEqual(events.count(VadEvent.SPECULATE), 1)
        self.assertIn(VadEvent.SPEECH_END, events)
        self.assertLess(events.index(VadEvent.SPECULATE), events.index(VadEvent.SPEECH_END))

    def test_speculation_keeps_96ms_of_headroom_inside_the_shortest_wait(self):
        brisk = ThinkingVad()
        brisk.voiced_ms, brisk.pauses_ms = 416.0, [170.0] * 8  # "हाँ डॉक्टर", a caller who talks fast
        self.assertAlmostEqual(brisk.pace_factor, 0.80)
        self.assertAlmostEqual(brisk.target_wait_ms(), 280.0)
        self.assertAlmostEqual(brisk.speculate_deadline_ms(), 184.0)
        self.assertLess(brisk.speculate_deadline_ms(), brisk.target_wait_ms())
        self.assertEqual(ThinkingVad().speculate_deadline_ms(), 254.0, "a 350 ms wait speculates at 254 ms")
        self.assertEqual(ThinkingVad().speculate_deadline_ms(is_reflective=True), 280.0, "a long wait keeps the 280 ms default")
        self.assertEqual(ThinkingVad({"speculate_at_ms": 100.0}).speculate_deadline_ms(), 100.0, "the floor never raises a pinned value")
        self.assertEqual(ThinkingVad({"closed_short_ms": 120.0, "speculate_at_ms": 400.0}).speculate_deadline_ms(), 160.0, "160 ms is the floor")

    def test_a_mid_sentence_pause_speculates_before_it_cuts_the_turn_at_pace_min(self):
        vad = ThinkingVad()
        vad.pauses_ms = [170.0] * 8
        self.assertEqual(vad.on_frame(0.9), [VadEvent.SPEECH_START])
        voiced = [e for _ in range(12) for e in vad.on_frame(0.9)]  # 416 ms of "हाँ डॉक्टर"
        self.assertEqual([e for e in voiced if e is not VadEvent.NONE], [])
        pause = [e for _ in range(17) for e in vad.on_frame(0.05)]  # 550 ms of silence
        firing = [e for e in pause if e is not VadEvent.NONE]
        self.assertEqual(firing, [VadEvent.SPECULATE, VadEvent.SPEECH_END], "speculation always beats the endpoint")
        self.assertEqual(pause.index(VadEvent.SPECULATE), 5, "192 ms of silence, inside the 280 ms wait")
        self.assertEqual(pause.index(VadEvent.SPEECH_END), 8, "288 ms of silence, the shortened 280 ms wait")
        self.assertFalse(vad.in_turn)

    def test_speculation_precedes_the_endpoint_for_every_voiced_length_and_pace(self):
        for pauses in ([170.0] * 8, [], [390.0] * 8):  # a brisk, an uncalibrated, and a slow caller
            for voiced_frames in (2, 12, 19, 57, 90):  # 64 ms .. 2880 ms, across both ramp knees
                vad = ThinkingVad()
                vad.pauses_ms = list(pauses)
                events = drive(vad, frames((0.9, voiced_frames), (0.05, 60)))
                order = [e for e in events if e is not VadEvent.NONE]
                with self.subTest(pace=vad.pace_factor, voiced_ms=voiced_frames * 32.0):
                    self.assertEqual(order[:2], [VadEvent.SPEECH_START, VadEvent.SPECULATE])
                    self.assertLess(order.index(VadEvent.SPECULATE), order.index(VadEvent.SPEECH_END))
                    self.assertLess(vad.speculate_deadline_ms(), vad.target_wait_ms() + 1e-9)

    def test_each_new_frame_below_threshold_is_silence_not_speech(self):
        vad = ThinkingVad()
        vad.on_frame(0.4)  # just under the 0.5 threshold
        self.assertFalse(vad.in_turn)
        vad.on_frame(0.5)  # exactly at threshold counts as voiced
        self.assertTrue(vad.in_turn)

    def test_intra_turn_pause_is_recorded_in_pauses_ms(self):
        vad = ThinkingVad()
        events = drive(vad, frames((0.9, 4), (0.1, 8), (0.9, 4), (0.1, 40)))
        self.assertEqual(events.count(VadEvent.SPEECH_START), 1)
        self.assertEqual(vad.pauses_ms, [256.0])  # one V -> _ -> V boundary

    def test_gaps_outside_150_to_850_are_not_recorded(self):
        vad = ThinkingVad()
        drive(vad, frames((0.9, 2), (0.1, 3), (0.9, 2), (0.1, 40)))  # 96 ms gap: too short
        self.assertEqual(vad.pauses_ms, [])

    def test_continuation_after_endpoint_reopens_and_banks_the_gap(self):
        vad = ThinkingVad()
        drive(vad, frames((0.9, 4), (0.1, 12)))  # close at 384 ms of silence
        self.assertFalse(vad.in_turn)
        self.assertGreaterEqual(vad.post_end_silence_ms, 384.0)
        resumed = vad.on_frame(0.9)
        self.assertEqual(resumed, [VadEvent.SPEECH_START])
        self.assertTrue(vad.in_turn)
        self.assertEqual(vad.voiced_ms, 32.0, "the resumed turn starts its own voicing count")
        self.assertEqual(vad.pauses_ms, [384.0], "the endpoint gap is a real pause")

    def test_resume_beyond_1200ms_does_not_bank_a_pause(self):
        vad = ThinkingVad()
        drive(vad, frames((0.9, 2), (0.1, 40)))  # 40 silent frames of 32 ms
        self.assertGreater(vad.post_end_silence_ms, 1200.0)
        self.assertEqual(vad.on_frame(0.9), [VadEvent.SPEECH_START])
        self.assertEqual(vad.pauses_ms, [])

    def test_bot_speaking_clears_post_end_window_so_next_turn_is_fresh(self):
        vad = ThinkingVad()
        drive(vad, frames((0.9, 3), (0.1, 14)))
        vad.on_bot_speaking()
        self.assertEqual(vad.post_end_silence_ms, -1.0)
        self.assertEqual(vad.on_frame(0.9), [VadEvent.SPEECH_START])
        self.assertEqual(vad.pauses_ms, [], "a normal reply is not a mid-turn pause")

    def test_extend_for_incomplete_holds_the_endpoint_until_the_caller_resumes(self):
        vad = ThinkingVad()
        vad.on_frame(0.9)
        vad.extend_for_incomplete()
        self.assertTrue(vad.holding_incomplete)
        silent = [e for _ in range(60) for e in vad.on_frame(0.05)]
        self.assertNotIn(VadEvent.SPEECH_END, silent, "held for 2000 ms")
        vad.on_frame(0.9)
        self.assertFalse(vad.holding_incomplete, "resuming clears the hold")

    def test_extend_for_incomplete_is_ignored_outside_a_turn(self):
        vad = ThinkingVad()
        vad.extend_for_incomplete()
        self.assertFalse(vad.holding_incomplete)

    def test_endpoint_wait_uses_recorded_pace_on_the_next_turn(self):
        fast, slow = ThinkingVad(), ThinkingVad()
        fast.pauses_ms, slow.pauses_ms = [170.0] * 8, [390.0] * 8
        fast.voiced_ms = slow.voiced_ms = 0.0
        self.assertAlmostEqual(fast.target_wait_ms(), 350.0 * 0.80)
        self.assertAlmostEqual(slow.target_wait_ms(), 350.0 * 1.45)
        self.assertLess(fast.target_wait_ms(), slow.target_wait_ms())

    def test_post_end_counter_only_advances_while_awaiting_reply(self):
        vad = ThinkingVad()
        self.assertEqual(vad.on_frame(0.05), [VadEvent.NONE])
        self.assertEqual(vad.post_end_silence_ms, -1.0)
        drive(vad, frames((0.9, 3), (0.1, 14)))
        held = vad.post_end_silence_ms
        for _ in range(5):
            vad.on_frame(0.05)
        self.assertAlmostEqual(vad.post_end_silence_ms, held + 5 * 32.0)


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

    def test_pcm16_to_wav_bytes_writes_a_sixteen_kilohertz_mono_riff(self):
        pcm = b"\x00\x00\xe8\x03" * 8
        wav_bytes = pcm16_to_wav_bytes(pcm + b"\x01")
        self.assertTrue(wav_bytes.startswith(b"RIFF"))
        with wave.open(io.BytesIO(wav_bytes)) as wf:
            self.assertEqual((wf.getnchannels(), wf.getframerate(), wf.getsampwidth()), (1, 16000, 2))
            self.assertEqual(wf.readframes(wf.getnframes()), pcm)

    def test_pcm16_to_wav_bytes_rejects_a_non_positive_rate(self):
        with self.assertRaises(ValueError):
            pcm16_to_wav_bytes(b"\x00\x00", sample_rate=0)

    def test_resample_identity_and_odd_byte_trimming(self):
        self.assertEqual(resample_int16_mono(b"\x10\x00\x20\x00\xff", 16000, 16000), b"\x10\x00 \x00")
        self.assertEqual(resample_int16_mono(b"\xff", 16000, 16000), b"")
        self.assertEqual(resample_int16_mono(b"", 24000, 16000), b"")
        self.assertEqual(resample_int16_mono(b"\x10\x00\xff", 0, 16000), b"\x10\x00")

    def test_resample_down_and_up_keep_whole_samples_and_signal_shape(self):
        tone = np.repeat(np.array([0, 16000, -16000, 8000], dtype=np.int16), 25)
        down = resample_int16_mono(tone.tobytes(), 24000, 16000)
        up = resample_int16_mono(tone.tobytes(), 8000, 16000)
        self.assertEqual(len(down) % 2, 0)
        self.assertAlmostEqual(len(down) // 2, round(100 * 16000 / 24000), delta=1)
        self.assertAlmostEqual(len(up) // 2, 200, delta=1)
        self.assertEqual(int(np.frombuffer(down, dtype=np.int16).max()), 16000)
        self.assertEqual(int(np.frombuffer(up, dtype=np.int16).min()), -16000)

    def test_single_sample_resample_duplicates_instead_of_failing(self):
        self.assertEqual(len(resample_int16_mono(b"\xe8\x03", 8000, 16000)) // 2, 2)


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

    def test_mark_ended_freezes_wall_seconds_and_save_is_idempotent_in_length(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec = CallAudioRecorder(9, tmp)
            rec._started = time.monotonic() - 3.0
            rec.mark_ended()
            frozen = rec.wall_seconds()
            time.sleep(0.05)
            self.assertEqual(rec.wall_seconds(), frozen)
            name, duration = rec.save()
            self.assertEqual(duration, frozen)
            self.assertTrue((Path(tmp) / name).is_file())

    def test_output_dir_is_created_and_post_save_chunks_are_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "nested" / "audio"
            rec = CallAudioRecorder(11, target)
            self.assertTrue(target.is_dir())
            rec.add_user_pcm(b"\x10\x00" * 80, 16000)
            name, _ = rec.save()
            rec.add_user_pcm(b"\x20\x00" * 80, 16000)
            self.assertEqual(rec._chunks["user"][-1][1], b"\x10\x00" * 80)
            with wave.open(str(target / name), "rb") as wf:
                self.assertGreater(wf.getnframes(), 0)

    def test_empty_and_odd_chunks_do_not_break_the_mix(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec = CallAudioRecorder(12, tmp)
            rec.add_user_pcm(b"", 16000)
            rec.add_user_pcm(b"\x10\x00\xff", 16000)
            rec.add_bot_pcm(b"\x20\x00" * 2, 24000)
            name, _ = rec.save()
            with wave.open(str(Path(tmp) / name), "rb") as wf:
                self.assertEqual(wf.getnchannels(), 2)
                self.assertEqual(wf.getnframes() % 1, 0)

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

    def test_target_rate_constant_is_sixteen_kilohertz(self):
        self.assertEqual(TARGET_RATE, 16000)


if __name__ == "__main__":
    unittest.main()
