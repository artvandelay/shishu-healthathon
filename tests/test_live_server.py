"""Live voice server tests: config, turn-taking, quick replies, and Decision-API execution."""

from __future__ import annotations

import asyncio
import base64
import copy
import io
import json
import os
import tempfile
import time
import unittest
import wave
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from tests import TEST_DATA_DIR
from tests.helpers import (
    DummyRecorder,
    FakeLiveSession,
    FakeToolCall,
    FakeWebSocket,
    QueueWebSocket,
    SessionCtx,
    bot_audio_message,
    choice,
    clear_until,
    decision_result,
    drop_fallback,
    make_child,
    make_profile_state,
    noul,
    server_message,
    shipped_thresholds,
)

import audio_recorder  # noqa: E402
import call_cursor  # noqa: E402
import db  # noqa: E402
import decisions  # noqa: E402
import gemini_live  # noqa: E402
import live_decision_bridge  # noqa: E402
import profile_extractor  # noqa: E402
import prompts  # noqa: E402
from call_cursor import apply_turn, close_technical_failure, step_id  # noqa: E402


def _make_live(
    *, ws=None, session=None, call_id=None, recorder=None, cursor_state=None,
    thresholds=None, stop=None, reconnect=None, watchdog_timeout_sec=None,
) -> gemini_live.LiveCallSession:
    return gemini_live.LiveCallSession(
        ws=ws or FakeWebSocket(),
        session=session or FakeLiveSession(),
        call_id=db.create_call() if call_id is None else call_id,
        recorder=DummyRecorder() if recorder is None else recorder,
        cursor_state=cursor_state,
        thresholds=thresholds,
        stop=stop,
        reconnect=reconnect,
        watchdog_timeout_sec=watchdog_timeout_sec,
    )


def _eval_payload(**overrides):
    base = {
        "danger_now": noul(0.01), "caller_stop": noul(0.01), "mentions_profile": noul(0.01),
        "detail_present": noul(0.03),
        "current_outcome": choice("clear", {"clear": 0.94, "other": 0.04, "concern": 0.02}),
        "side_intent": choice("other", {"other": 0.96, "add_concern": 0.04}),
        "side_qid": choice("other", {"other": 0.96, "birth_nicu": 0.04}),
    }
    base.update(overrides)
    return base


class LiveConfigAndWsTests(unittest.TestCase):
    def test_live_config_declares_only_next_question_and_kore_vad_defaults(self):
        prompt = prompts.GEMINI_LIVE_SYSTEM_PROMPT
        self.assertIn("warm female voice", prompt)
        self.assertNotIn("Kore", prompt)
        for token in ("next_question", "say", "clarify", "BCG", "थैलेसीमिया माइनर या ट्रेट", "recorded_concerns", "stop", "SYSTEM CUE",
                      "Duchenne", "Hemophilia", "Pompe", "Gaucher", "SCID", "SMA", "MPS", "Wiskott-Aldrich", "ड्यूशेन", "हीमोफीलिया", "पॉम्पे", "गॉशर"):
            self.assertIn(token, prompt)
        for forbidden in ("target_qid", "child_ref", "outcome", "marital_status", "caller_stop", "ever_pregnant", "emergency"):
            self.assertNotIn(forbidden, prompt)

        cfg = gemini_live._live_config({"gemini_model": "gemini-3.1-flash-live-preview"})
        self.assertEqual((cfg.system_instruction.parts[0].text, cfg.speech_config.voice_config.prebuilt_voice_config.voice_name), (prompt, "Kore"))
        self.assertEqual((list(cfg.output_audio_transcription.language_codes), list(cfg.input_audio_transcription.language_codes), cfg.thinking_config.thinking_level),
                         (["hi-IN"], ["hi-IN"], "MINIMAL"))
        decls = cfg.tools[0].function_declarations
        self.assertEqual(([d.name for d in decls], dict(decls[0].parameters.properties or {})), (["next_question"], {}))
        self.assertIsNone(gemini_live._audio_from_part(SimpleNamespace(inline_data=SimpleNamespace(data=b"\x01\x00", mime_type="text/plain"))))
        self.assertIsNotNone(gemini_live._audio_from_part(SimpleNamespace(inline_data=SimpleNamespace(data=b"\x01\x00", mime_type="audio/pcm;rate=24000"))))

        env_keys = ("GEMINI_MODEL", "GEMINI_VOICE", "VAD_PREFIX_PADDING_MS", "VAD_SILENCE_MS",
                    "VAD_START_SENSITIVITY", "VAD_END_SENSITIVITY", "WATCHDOG_TIMEOUT_SEC", "DECISION_THRESHOLDS_PATH")
        clean_env = {k: v for k, v in os.environ.items() if k not in env_keys}
        with mock.patch.dict(os.environ, clean_env, clear=True):
            unset = gemini_live._session_settings({})
            self.assertEqual((unset["voice"], unset["silence_duration_ms"], unset["watchdog_timeout_sec"]), ("Kore", 1000, 8.0))
            self.assertEqual((unset["start_sens"], unset["end_sens"]),
                             (gemini_live.types.StartSensitivity.START_SENSITIVITY_LOW, gemini_live.types.EndSensitivity.END_SENSITIVITY_LOW))

            custom = gemini_live._session_settings({"gemini_voice": "Aoede", "vad_silence_ms": "1200", "vad_end_sensitivity": "HIGH", "vad_start_sensitivity": "HIGH"})
            self.assertEqual((custom["voice"], custom["silence_duration_ms"], custom["start_sens"], custom["end_sens"]),
                             ("Aoede", 1200, gemini_live.types.StartSensitivity.START_SENSITIVITY_HIGH, gemini_live.types.EndSensitivity.END_SENSITIVITY_HIGH))
            with mock.patch.dict(os.environ, {"GEMINI_MODEL": "", "GEMINI_VOICE": "Puck", "VAD_SILENCE_MS": "900", "WATCHDOG_TIMEOUT_SEC": "6.5", "DECISION_THRESHOLDS_PATH": "/tmp/env-t.json"}, clear=False):
                from_env = gemini_live.live_settings()
                self.assertEqual((from_env["gemini_model"], from_env["gemini_voice"], from_env["vad_silence_ms"], from_env["watchdog_timeout_sec"]),
                                 ("gemini-3.1-flash-live-preview", "Puck", 900, 6.5))
                overlaid = gemini_live.live_settings({"gemini_voice": "Aoede", "vad_silence_ms": None})
                self.assertEqual((overlaid["gemini_voice"], overlaid["vad_silence_ms"]), ("Aoede", 900))

    def test_handle_gemini_ws_rejects_bad_thresholds_and_malformed_env_settings(self):
        default_loaded = live_decision_bridge.load_turn_thresholds({})
        self.assertEqual(default_loaded, shipped_thresholds())
        with tempfile.TemporaryDirectory() as tmp:
            bad_path = Path(tmp) / "bad.json"
            bad_data = asdict(default_loaded)
            bad_data["danger_now_min_yes"] = 0.999
            bad_path.write_text(json.dumps(bad_data), encoding="utf-8")

            for env_patch, expected_msg in (
                ({"DECISION_THRESHOLDS_PATH": "/tmp/missing-shishu-thresholds.json", "GEMINI_API_KEY": "k"}, "Invalid or missing Decision API thresholds"),
                ({"DECISION_THRESHOLDS_PATH": str(bad_path), "GEMINI_API_KEY": "k"}, "Invalid or missing Decision API thresholds"),
                ({"VAD_SILENCE_MS": "abc", "GEMINI_API_KEY": "k"}, "Invalid live call settings"),
            ):
                ws = FakeWebSocket()
                with mock.patch.dict(os.environ, env_patch, clear=False), mock.patch.object(gemini_live, "_connect_live") as conn, mock.patch.object(db, "create_call") as create:
                    asyncio.run(gemini_live.handle_gemini_ws(ws))
                    conn.assert_not_called()
                    create.assert_not_called()
                self.assertTrue(ws.closed)
                self.assertEqual([m for m in ws.sent if m.get("type") == "error"], [{"type": "error", "message": expected_msg}])

    def test_handle_gemini_ws_wires_config_and_executes_turn_end_to_end(self):
        async def run():
            seen_configs = []
            live_sess = FakeLiveSession([
                bot_audio_message(turn_complete=True),
                server_message(user_text="रेखा"),
                server_message(turn_complete=True, tool_calls=[FakeToolCall(fid="fc-ws-1")]),
                bot_audio_message(turn_complete=True),
            ], wait_forever=True, sleep_per_msg=0.01)

            async def fake_connect(_client, _model, config):
                seen_configs.append(config)
                return SessionCtx(live_sess), live_sess, 5

            async def fake_eval(request, **_kwargs):
                return decision_result(request, _eval_payload(mentions_profile=noul(0.90), current_outcome=choice("other", {"other": 0.90, "clear": 0.10})))

            async def hook_profile(_state, **_kwargs):
                return profile_extractor.ProfileExtractionResult(ok=True, updates={"caller_name": "रेखा"}, error_reason="", extractor_ms=1)

            ws = QueueWebSocket()

            async def hangup_soon():
                await asyncio.sleep(0.15)
                await ws.push({"type": "hangup"})

            with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}), mock.patch.object(
                gemini_live, "_connect_live", side_effect=fake_connect
            ), mock.patch.object(profile_extractor, "extract_profile_entities", side_effect=hook_profile), mock.patch.object(
                decisions, "evaluate_turn", side_effect=fake_eval
            ):
                task = asyncio.create_task(hangup_soon())
                await gemini_live.handle_gemini_ws(ws)
                await task

            self.assertEqual([d.name for d in seen_configs[0].tools[0].function_declarations], ["next_question"])
            started = next(m for m in ws.sent if m.get("type") == "session_started")
            saved = db.get_call(started["call_id"])
            self.assertEqual((saved["cursor_state"]["profile"]["caller_name"], saved["cursor_state"]["active_qid"]), ("रेखा", "intro_married"))
            self.assertEqual(len([e for e in saved["events"] if e["type"] == "decision_turn"]), 1)

        asyncio.run(run())


class TurnTakingAndWatchdogTests(unittest.TestCase):
    def test_greeting_and_recorded_concerns_tool_body(self):
        session, ws = FakeLiveSession(), FakeWebSocket()
        live = _make_live(ws=ws, session=session, call_id=0)
        asyncio.run(live._send_greeting())
        self.assertEqual([i[0] for i in session.sent], ["realtime"])
        self.assertIn("नाम", session.sent[0][1]["text"])
        states = [m for m in ws.sent if m.get("type") == "question_state"]
        self.assertEqual((len(states), states[0]["step"], states[0]["expect_yes_no"]), (1, "intro_name", False))

        state = clear_until(make_profile_state(children=[make_child(4, "male")]), "birth_jaundice")
        state["answers"]["birth_nicu"] = {"state": "concern", "child_ids": ["child_1"], "notes": "साँस की तकलीफ", "followups": {"nicu_days": {"state": "concern", "note": "चार दिन"}}}
        self.assertNotIn("recorded_concerns", gemini_live._tool_body(state))
        lines = prompts.recorded_concerns_hi(state)
        self.assertEqual(len(lines), 1)
        self.assertTrue("काँच के बक्से" in lines[0] and "4 साल" in lines[0] and "साँस की तकलीफ" in lines[0] and "चार दिन" in lines[0])
        self.assertEqual(gemini_live._tool_body(state, outcome="clarify")["recorded_concerns"], lines)
        self.assertEqual(gemini_live._tool_body(clear_until(state, "fam_same_problem"))["recorded_concerns"], lines)
        self.assertEqual(gemini_live._tool_body(clear_until(make_profile_state(children=[make_child(4, "male")]), "close_anything_else"))["recorded_concerns"], [])

    def test_tool_speak_fallback_audio_cancel_send_failure_and_turn_complete_cue(self):
        thresholds = shipped_thresholds()
        state = clear_until(make_profile_state(children=[make_child(4, "male")]), "birth_jaundice")
        step = step_id(state)

        async def fake_clear(request, **_kwargs):
            return decision_result(request, _eval_payload())

        async def run():
            with mock.patch.object(decisions, "evaluate_turn", side_effect=fake_clear):
                # 1) Silent Gemini triggers fallback cue; no "wait" key in FunctionResponse
                s1 = FakeLiveSession()
                l1 = _make_live(session=s1, cursor_state=json.loads(json.dumps(state)), thresholds=thresholds)
                l1.turn.spoken_qid = l1.turn.user_turn_step = step
                l1.capture.user_turn_buffer = "नहीं"
                with mock.patch.object(gemini_live, "_TOOL_SPEAK_FALLBACK_SEC", 0.01):
                    await l1._answer_function_calls([FakeToolCall(fid="fc-s")])
                    self.assertNotIn("wait", s1.sent[0][1][0].response)
                    await l1.turn.speak_fallback
                self.assertEqual(([i[0] for i in s1.sent], l1.cursor_state["active_qid"]), (["tool", "realtime"], "birth_sudden"))

                # 2) Audio arriving early cancels fallback cue
                s2 = FakeLiveSession()
                l2 = _make_live(session=s2, cursor_state=json.loads(json.dumps(state)), thresholds=thresholds)
                l2.turn.spoken_qid = l2.turn.user_turn_step = step
                l2.capture.user_turn_buffer = "नहीं"
                with mock.patch.object(gemini_live, "_TOOL_SPEAK_FALLBACK_SEC", 0.05):
                    await l2._answer_function_calls([FakeToolCall(fid="fc-a")])
                    await l2._on_bot_audio(b"\x00\x00" * 24, 24000)
                    await l2.turn.speak_fallback
                self.assertEqual([i[0] for i in s2.sent], ["tool"])

                # 3) Failed send_tool_response cues immediately
                s3 = FakeLiveSession(fail_tool=True)
                l3 = _make_live(session=s3, cursor_state=json.loads(json.dumps(state)), thresholds=thresholds)
                l3.turn.spoken_qid = l3.turn.user_turn_step = step
                l3.capture.user_turn_buffer = "नहीं"
                await l3._answer_function_calls([FakeToolCall(fid="fc-f")])
                self.assertEqual(([i[0] for i in s3.sent], l3.turn.speak_fallback), (["tool", "realtime"], None))

                # 4) Causal speak cue on turn_complete + stop on send_realtime_input failure
                s4 = FakeLiveSession()
                l4 = _make_live(session=s4, cursor_state=json.loads(json.dumps(state)), thresholds=thresholds)
                l4.turn.spoken_qid = l4.turn.user_turn_step = step
                l4.capture.user_turn_buffer = "नहीं"
                await l4._handle_server_message(server_message(turn_complete=True, tool_calls=[FakeToolCall(fid="fc-tc")]))
                await asyncio.sleep(0)
                self.assertEqual([i[0] for i in s4.sent], ["tool", "realtime"])
                self.assertIn("इमरजेंसी", s4.sent[1][1]["text"])

                broken_ws = FakeWebSocket()
                broken = _make_live(ws=broken_ws, session=FakeLiveSession(fail_realtime=True), cursor_state=json.loads(json.dumps(l4.cursor_state)), thresholds=thresholds)
                broken.turn.pending_speak_step, broken.turn.pending_speak_kind = step_id(broken.cursor_state), "speak"
                self.assertFalse(await broken._handle_server_message(server_message(turn_complete=True)))
                self.assertTrue(broken.stop.is_set() and broken.telemetry["tool_errors"] == 1)
                self.assertEqual([m["message"] for m in broken_ws.sent if m.get("type") == "error"], ["Could not prompt Gemini to speak"])

        asyncio.run(run())

    def test_speech_end_watchdog_check_in_silence_time_limits_and_nudges(self):
        state = clear_until(make_profile_state(children=[make_child(4, "male")]), "birth_nicu")

        async def run():
            live = _make_live(ws=FakeWebSocket([{"type": "speech_end"}]), cursor_state=json.loads(json.dumps(state)))
            live.turn.spoken_qid, live.turn.playback_ends_at = step_id(state), time.time() - 1
            await live._browser_loop()
            self.assertGreaterEqual(live.turn.quiet_from, live.turn.playback_ends_at)
            self.assertFalse(live.turn.awaiting_bot_reply or live.turn.speech_end_waiting)

            cid = db.create_call()
            wd = _make_live(call_id=cid, cursor_state=json.loads(json.dumps(state)), watchdog_timeout_sec=8.0)
            wd.turn.quiet_from = time.time() - 16.0
            with mock.patch.object(gemini_live, "_WATCHDOG_INTERVAL_SEC", 0.01):
                task = asyncio.create_task(wd._watchdog_loop())
                await asyncio.sleep(0.05)
                self.assertTrue(wd.turn.checked_in)
                wd.turn.quiet_from = time.time() - 31.0
                await asyncio.sleep(0.05)
                wd.stop.set()
                await task
            events = [e["type"] for e in db.get_call(cid)["events"]]
            self.assertTrue("silence_check_in" in events and "auto_close" in events and wd.cursor_state["close_reason"] == "silence")

            soft = _make_live(cursor_state=json.loads(json.dumps(state)), watchdog_timeout_sec=8.0)
            soft.turn.call_started_ts = time.time() - gemini_live._SOFT_LIMIT_SEC - 1
            with mock.patch.object(gemini_live, "_WATCHDOG_INTERVAL_SEC", 0.01):
                task = asyncio.create_task(soft._watchdog_loop())
                await asyncio.sleep(0.05)
                self.assertTrue(soft.turn.time_cue_sent and soft.cursor_state["close_reason"] == "time_limit")
                soft.turn.call_started_ts = time.time() - gemini_live._HARD_LIMIT_SEC - 1
                await asyncio.wait_for(soft.stop.wait(), timeout=1.0)
                await task

            # Watchdog & manual nudges preserve active_speak_kind="clarify"
            nudge_sess = FakeLiveSession()
            nudge_wd = _make_live(session=nudge_sess, cursor_state=json.loads(json.dumps(state)), watchdog_timeout_sec=0.05)
            nudge_wd.turn.awaiting_bot_reply, nudge_wd.turn.last_user_turn_ts, nudge_wd.turn.active_speak_kind = True, time.time() - 5.0, "clarify"
            with mock.patch.object(gemini_live, "_WATCHDOG_INTERVAL_SEC", 0.01):
                task = asyncio.create_task(nudge_wd._watchdog_loop())
                await asyncio.sleep(0.08)
                nudge_wd.stop.set()
                await task
            self.assertTrue("she did not understand" in nudge_sess.sent[0][1]["text"] and "काँच के बक्से" in nudge_sess.sent[0][1]["text"])

            man_sess, man_cid = FakeLiveSession(), db.create_call()
            man = _make_live(ws=FakeWebSocket([{"type": "nudge"}]), session=man_sess, call_id=man_cid, cursor_state=json.loads(json.dumps(state)))
            man.turn.active_speak_kind = "clarify"
            await man._browser_loop()
            self.assertEqual(man_sess.sent[0][1]["text"], prompts.speak_cue_text(man.cursor_state, "clarify"))
            self.assertEqual(len([e for e in db.get_call(man_cid)["events"] if e["type"] == "manual_nudge"]), 1)

        asyncio.run(run())

    def test_go_away_reconnect_session_swap_and_five_attempt_cap(self):
        state = clear_until(make_profile_state(children=[make_child(4, "male")]), "birth_jaundice")

        async def run():
            missing = _make_live(cursor_state=json.loads(json.dumps(state)))
            missing.turn.bot_speaking = True
            self.assertFalse(await missing._handle_server_message(server_message(go_away=True)))

            seen = []

            async def reconnect(h):
                seen.append(h)
                return True

            stop_evt = asyncio.Event()
            wait_sess = FakeLiveSession(
                [server_message(go_away=True), server_message(turn_complete=True)],
                wait_forever=True, on_realtime=lambda *_: stop_evt.set(),
            )
            wait_ws = FakeWebSocket()
            wait_call = _make_live(ws=wait_ws, session=wait_sess, cursor_state=json.loads(json.dumps(state)), stop=stop_evt, reconnect=reconnect)
            wait_call.turn.resume_handle, wait_call.turn.awaiting_bot_reply = "resume-wait", True
            wait_call.turn.pending_speak_step = step_id(wait_call.cursor_state)
            wait_call.turn.pending_speak_kind = wait_call.turn.active_speak_kind = "clarify"
            await asyncio.wait_for(wait_call._gemini_loop(), timeout=3)
            self.assertEqual(seen, ["resume-wait"])
            self.assertIn("she did not understand", wait_sess.sent[0][1]["text"])
            self.assertIn("reconnected", [m.get("type") for m in wait_ws.sent])

            stop_drop = asyncio.Event()
            drop_sess = FakeLiveSession(error=ConnectionError("dropped"), on_realtime=lambda *_: stop_drop.set())
            drop_call = _make_live(session=drop_sess, cursor_state=json.loads(json.dumps(state)), stop=stop_drop, reconnect=reconnect)
            drop_call.turn.resume_handle, drop_call.turn.bot_speaking = "resume-drop", True
            seen.clear()
            await asyncio.wait_for(drop_call._gemini_loop(), timeout=3)
            self.assertEqual(seen, ["resume-drop"])
            self.assertIn("speak only this Hindi now", drop_sess.sent[0][1]["text"])

            # 5-attempt cap stops without attempt 6
            cap_ws = FakeWebSocket()
            cap_attempts = 0

            async def count_reconn(_h):
                nonlocal cap_attempts
                cap_attempts += 1
                return True

            cap_live = _make_live(ws=cap_ws, session=FakeLiveSession(error=ConnectionError("dropped")), reconnect=count_reconn)
            cap_live.turn.resume_handle = "handle-limit"
            await cap_live._gemini_loop()
            self.assertTrue(cap_live.stop.is_set() and cap_attempts == 5 and cap_live.telemetry["reconnects"] == 5)
            self.assertEqual([m["attempt"] for m in cap_ws.sent if m.get("type") == "reconnecting"], [1, 2, 3, 4, 5])
            self.assertTrue(any(m.get("type") == "error" and "reconnect" in m.get("message", "") for m in cap_ws.sent))

            # Full handle_gemini_ws reconnect session swap clears reconnect_requested under lock
            sess1_dead, sess1_failed, sess2_got_audio = False, asyncio.Event(), asyncio.Event()
            reconnect_entered, allow_reconnect = asyncio.Event(), asyncio.Event()

            def on_s1(_s, kw):
                if sess1_dead:
                    if "audio" in kw:
                        sess1_failed.set()
                    raise RuntimeError("old session closed")

            class _SwapSess1(FakeLiveSession):
                async def receive(self):
                    nonlocal sess1_dead
                    yield server_message(resume_handle="h-1")
                    sess1_dead = True
                    yield server_message(go_away=True)
                    await asyncio.Future()

            sess1 = _SwapSess1(on_realtime=on_s1)
            sess2 = FakeLiveSession(wait_forever=True, on_realtime=lambda _s, kw: sess2_got_audio.set() if "audio" in kw else None)
            connect_calls = 0

            async def fake_connect(_client, _model, _config):
                nonlocal connect_calls
                connect_calls += 1
                if connect_calls == 1:
                    return SessionCtx(sess1), sess1, 3
                reconnect_entered.set()
                await allow_reconnect.wait()
                return SessionCtx(sess2), sess2, 3

            fake_client = mock.MagicMock()
            fake_client.aio.aclose = mock.AsyncMock()
            ws2 = QueueWebSocket()
            loud_b64 = base64.b64encode(b"\x00\x10" * 160).decode("ascii")

            async def drive_mic():
                await asyncio.wait_for(reconnect_entered.wait(), timeout=1.0)
                await ws2.push({"type": "audio", "data": loud_b64, "sample_rate": 16000})
                await asyncio.wait_for(sess1_failed.wait(), timeout=1.0)
                allow_reconnect.set()
                for _ in range(100):
                    if any(m.get("type") == "reconnected" for m in ws2.sent):
                        break
                    await asyncio.sleep(0.01)
                await ws2.push({"type": "audio", "data": loud_b64, "sample_rate": 16000})
                await asyncio.wait_for(sess2_got_audio.wait(), timeout=1.0)
                await asyncio.sleep(0.05)
                await ws2.push({"type": "hangup"})

            with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}), mock.patch.object(
                gemini_live.genai, "Client", return_value=fake_client
            ), mock.patch.object(gemini_live, "_connect_live", side_effect=fake_connect):
                driver = asyncio.create_task(drive_mic())
                await gemini_live.handle_gemini_ws(ws2)
                await driver

            self.assertEqual((connect_calls, [m["attempt"] for m in ws2.sent if m.get("type") == "reconnected"]), (2, [1]))
            self.assertTrue(any("audio" in item[1] for item in sess2.sent))
            fake_client.aio.aclose.assert_awaited_once()

        asyncio.run(run())

    def test_dropoff_qid_and_stereo_call_audio_recorder(self):
        async def run():
            state = clear_until(make_profile_state(children=[make_child(4, "male")]), "birth_jaundice")
            call_id = db.create_call()
            live = _make_live(call_id=call_id, cursor_state=json.loads(json.dumps(state)))
            await live._mark_closing("silence")
            self.assertEqual(live.telemetry["dropoff_qid"], "birth_jaundice")
            self.assertTrue(live.turn.end_requested and live.turn.end_requested_at > 0.0)

            rec = audio_recorder.CallAudioRecorder(call_id, output_dir=str(Path(TEST_DATA_DIR) / "audio"))
            rec._started = time.monotonic() - 0.5
            rec.add_user_pcm(b"\x10\x00" * 160, 16000)
            rec.add_bot_pcm(b"\x20\x00" * 240, 24000)
            await _make_live(call_id=call_id, recorder=rec, cursor_state=json.loads(json.dumps(state)))._finalize()
            saved = db.get_call(call_id)
            listed = next(row for row in db.list_calls() if row["id"] == call_id)
            self.assertEqual(saved["telemetry"]["dropoff_qid"], "birth_jaundice")
            self.assertFalse(any(k.endswith("_json") for k in saved) or any(k.endswith("_json") for k in listed))
            for detail_only in ("transcript", "telemetry", "events", "cursor_state"):
                self.assertNotIn(detail_only, listed)
            with wave.open(str(Path(TEST_DATA_DIR) / "audio" / saved["audio_path"]), "rb") as wf:
                self.assertEqual((wf.getnchannels(), wf.getframerate()), (2, 16000))

            done_id = db.create_call()
            done_state = dict(clear_until(make_profile_state(children=[make_child(4, "male")]), "close_anything_else"), phase="closing", close_reason="complete")
            await _make_live(call_id=done_id, recorder=rec, cursor_state=done_state)._finalize()
            self.assertEqual(db.get_call(done_id)["telemetry"]["dropoff_qid"], "")

        asyncio.run(run())


class LiveDecisionFlowTests(unittest.TestCase):
    def test_quick_reply_guards_audio_flush_stale_asr_and_closing_bot_first_audio(self):
        thresholds = shipped_thresholds()

        async def run():
            kids = [make_child(6, "male", 1), make_child(2, "female", 2)]
            state = clear_until(make_profile_state(children=kids), "birth_nicu")
            step = step_id(state)

            # 1) Before spoken_qid -> ignored
            unspoken_sess = FakeLiveSession()
            await _make_live(ws=FakeWebSocket([{"type": "quick_reply", "outcome": "clear", "step": step}]), session=unspoken_sess, cursor_state=json.loads(json.dumps(state)))._browser_loop()
            self.assertEqual(unspoken_sess.sent, [])

            # 2) Spoken while bot_speaking=True -> flushes, clears ignore_bot_audio when Q2 cue fires so Q2 plays without turn_complete;
            #    model-free + stale ASR/next_question held; fresh ASR >=1.2s later advances Q2->Q3 via Decision
            ws = FakeWebSocket([
                {"type": "quick_reply", "outcome": "clear", "step": "stale"},
                {"type": "quick_reply", "outcome": "clear", "step": step},
                {"type": "quick_reply", "outcome": "concern", "step": step},
            ])
            sess = FakeLiveSession()
            live = _make_live(ws=ws, session=sess, cursor_state=json.loads(json.dumps(state)), thresholds=thresholds)
            live.turn.spoken_qid = live.turn.user_turn_step = step
            live.turn.bot_speaking, live.turn.playback_ends_at = True, time.time() + 5
            live.capture.user_turn_buffer, live.capture.user_turn_pcm = "नह", bytearray(b"\x00\x10" * 800)

            with mock.patch.object(decisions, "evaluate_turn") as eval_mock:
                await live._browser_loop()
                live.stop.clear()
                eval_mock.assert_not_called()
                self.assertEqual((live.cursor_state["active_qid"], live.turn.ignore_bot_audio, live.turn.bot_speaking, live.capture.speech_allowed),
                                 ("birth_jaundice", False, False, True))
                self.assertIn({"type": "clear"}, ws.sent)

                started = time.time() - 0.35
                live.turn.last_user_turn_ts = started
                await live._handle_server_message(server_message(user_text="नहीं"))
                self.assertAlmostEqual(live.turn.last_user_turn_ts, started, delta=0.02)
                await live._handle_server_message(server_message(turn_complete=True, tool_calls=[FakeToolCall(fid="fc-late")]))
                eval_mock.assert_not_called()
                await drop_fallback(live.turn)
            self.assertEqual([i[1] for i in sess.sent if i[0] == "tool"][0][0].response["reason"], "stale_turn")

            # Q2 audio plays immediately on birth_jaundice without Q1 turn_complete; fresh ASR advances Q2 -> Q3
            await live._handle_server_message(server_message(bot_text="क्या जन्म के बाद बच्चे को पीलिया हुआ?"))
            await live._handle_server_message(bot_audio_message(turn_complete=True))
            self.assertEqual((live.turn.spoken_qid, live.telemetry["audio_chunks_out"]), ("birth_jaundice", 1))
            live.turn.last_quick_reply_ts = time.time() - 1.5
            with mock.patch.object(decisions, "evaluate_turn", side_effect=lambda req, **_: decision_result(req, _eval_payload())):
                await live._handle_server_message(server_message(user_text="नहीं"))
                await live._handle_server_message(server_message(turn_complete=True, tool_calls=[FakeToolCall(fid="fc-fresh")]))
                await drop_fallback(live.turn)
            self.assertEqual(live.cursor_state["active_qid"], "birth_sudden")

            # 3) quick_reply on close_anything_else closes & records bot_first_audio on farewell; cue failure stops
            closing = apply_turn(clear_until(make_profile_state(children=[make_child(4, "male")]), "close_anything_else"), "concern", note="पेट दर्द")
            c_step, c_cid, c_sess = step_id(closing), db.create_call(), FakeLiveSession()
            c_live = _make_live(ws=FakeWebSocket([{"type": "quick_reply", "outcome": "clear", "step": c_step}]), session=c_sess, call_id=c_cid, cursor_state=closing)
            c_live.turn.spoken_qid, c_live.turn.bot_replied_since_user = c_step, True
            await c_live._browser_loop()
            self.assertEqual((c_live.cursor_state["phase"], c_live.cursor_state["close_reason"], c_live.turn.awaiting_bot_reply), ("closing", "complete", True))
            await asyncio.sleep(0.01)
            await c_live._handle_server_message(bot_audio_message(turn_complete=True))
            self.assertTrue(c_live.turn.closing_heard and c_live.telemetry["bot_turns"] == 1 and len(c_live.telemetry["turn_latencies_ms"]) == 1)
            self.assertTrue(any(e["type"] == "bot_first_audio" for e in db.get_call(c_cid)["events"]))

            qr_fail = _make_live(ws=FakeWebSocket([{"type": "quick_reply", "outcome": "clear", "step": step}]), session=FakeLiveSession(fail_realtime=True), cursor_state=json.loads(json.dumps(state)))
            qr_fail.turn.spoken_qid = step
            await qr_fail._browser_loop()
            self.assertTrue(qr_fail.stop.is_set())

        asyncio.run(run())

    def test_full_intro_to_clinical_gate_and_dual_intent_clarify_future_multichild(self):
        thresholds = shipped_thresholds()

        async def run():
            call_id = db.create_call()
            live = _make_live(call_id=call_id, cursor_state=call_cursor.new_cursor_state(), thresholds=thresholds)

            async def eval_intro(request, **_kwargs):
                ctx, qs = json.loads(request["input"]), request.get("questions") or {}
                is_bridge = "child_ref" in qs and "children_roster" in ctx and "side_qid" not in qs
                return decision_result(request, {
                    "danger_now": noul(0.01), "caller_stop": noul(0.01),
                    "mentions_profile": noul(0.02 if is_bridge else 0.92),
                    "current_outcome": choice("concern" if is_bridge else "other", {"concern": 0.93, "other": 0.05} if is_bridge else {"other": 0.90, "clear": 0.10}),
                })

            async def ext_intro(state, **_kwargs):
                qid = state.get("active_qid")
                up = ({"caller_name": "पूजा"} if qid == "intro_name"
                      else {"marital_status": "married"} if qid == "intro_married"
                      else {"child_count": 3, "children": [{"age_years": 11.0, "sex": "female", "name": "रिया"}, {"age_years": 6.0, "sex": "male", "name": None}, {"age_years": 0.5, "sex": "male", "name": None}]} if qid == "intro_children"
                      else {})
                return profile_extractor.ProfileExtractionResult(ok=True, updates=up, error_reason="", extractor_ms=24)

            with mock.patch.object(decisions, "evaluate_turn", side_effect=eval_intro), mock.patch.object(profile_extractor, "extract_profile_entities", side_effect=ext_intro) as ext_mock:
                for text, fid in (("मेरा नाम पूजा है", "i1"), ("हाँ, शादी हो गई है", "i2"), ("तीन बच्चे हैं", "i3"), ("हाँ, शुरू करें", "i4")):
                    live.turn.spoken_qid = step_id(live.cursor_state)
                    await live._handle_server_message(server_message(user_text=text))
                    await live._handle_server_message(server_message(turn_complete=True, tool_calls=[FakeToolCall(fid=fid)]))
                    await drop_fallback(live.turn)
                self.assertEqual(ext_mock.await_count, 3)

            self.assertEqual((live.cursor_state["active_qid"], live.cursor_state["profile"]["caller_name"], len(live.cursor_state["profile"]["children"]), live.cursor_state["profile"]["children"][2]["age_label_hi"]),
                             ("birth_mode_detail", "पूजा", 3, "6 महीने"))
            dt_ev = [e for e in db.get_call(call_id)["events"] if e["type"] == "decision_turn"]
            self.assertEqual((len(dt_ev), dt_ev[0]["meta"]["extractor_ok"]), (4, True))
            self.assertNotIn("extractor_ok", dt_ev[3]["meta"])

            # Dual-intent Smoke A + duplicate tool call hold
            s_a = clear_until(apply_turn(apply_turn(clear_until(make_profile_state(children=[make_child(4, "male")]), "fam_cancer"), "concern", note="मामा को ब्लड कैंसर"), "concern", note="मामा"), "fam_heart_kidney")
            sess_a = FakeLiveSession()
            live_a = _make_live(session=sess_a, call_id=call_id, cursor_state=s_a, thresholds=thresholds)
            live_a.turn.spoken_qid = step_id(live_a.cursor_state)
            with mock.patch.object(decisions, "evaluate_turn", side_effect=lambda req, **_: decision_result(req, _eval_payload(
                side_intent=choice("add_concern", {"add_concern": 0.93, "other": 0.05}),
                side_qid=choice("fam_cancer", {"fam_cancer": 0.91, "other": 0.04}),
            ))):
                await live_a._handle_server_message(server_message(user_text="नहीं, पर चाची को भी कैंसर हुआ था"))
                await live_a._handle_server_message(server_message(turn_complete=True, tool_calls=[FakeToolCall(fid="fc-1"), FakeToolCall(fid="fc-dup")]))
                await drop_fallback(live_a.turn)
            self.assertIn("चाची को भी कैंसर हुआ था", live_a.cursor_state["answers"]["fam_cancer"]["notes"])
            self.assertEqual(([r.response["result"] for r in [i[1] for i in sess_a.sent if i[0] == "tool"][0]], live_a.cursor_state["active_qid"]),
                             (["ok", "held"], "fam_bp_stroke"))

            # clarify_meaning rewind + future_qid_match + multi-child pre-answered followup
            live_a.cursor_state = apply_turn(clear_until(make_profile_state(children=[make_child(4, "male")]), "preg_thalassemia"), "unknown")
            live_a.turn.spoken_qid = step_id(live_a.cursor_state)
            with mock.patch.object(decisions, "evaluate_turn", side_effect=lambda req, **_: decision_result(req, _eval_payload(
                side_intent=choice("clarify_other", {"clarify_other": 0.91, "other": 0.05}),
                side_qid=choice("preg_thalassemia", {"preg_thalassemia": 0.90, "other": 0.05}),
            ))):
                await live_a._handle_server_message(server_message(user_text="थैलेसीमिया क्या होता है?"))
                await live_a._handle_server_message(server_message(turn_complete=True, tool_calls=[FakeToolCall(fid="fc-clar")]))
                await drop_fallback(live_a.turn)
            self.assertTrue(live_a.cursor_state["active_qid"] == "preg_thalassemia" and [i[1] for i in sess_a.sent if i[0] == "tool"][-1][0].response.get("clarify"))

            live_a.cursor_state = clear_until(make_profile_state(children=[make_child(4, "male")]), "birth_jaundice")
            live_a.turn.spoken_qid = step_id(live_a.cursor_state)
            with mock.patch.object(decisions, "evaluate_turn", side_effect=lambda req, **_: decision_result(req, _eval_payload(
                side_intent=choice("add_concern", {"add_concern": 0.91, "other": 0.06}),
                side_qid=choice("fam_cancer", {"fam_cancer": 0.93, "other": 0.04}),
            ))):
                await live_a._handle_server_message(server_message(user_text="नहीं, हाँ चाची को कैंसर था"))
                await live_a._handle_server_message(server_message(turn_complete=True, tool_calls=[FakeToolCall(fid="fc-fut")]))
                await drop_fallback(live_a.turn)
            self.assertEqual([e for e in db.get_call(call_id)["events"] if e["type"] == "decision_turn"][-1]["meta"]["future_qid_match"],
                             {"qid": "fam_cancer", "confidence": 0.93, "act": "add_or_correct"})
            self.assertNotIn("fam_cancer", live_a.cursor_state["answers"])

            kids2 = [make_child(9, "female", 1, name="रिया"), make_child(5, "male", 2, name="अर्जुन")]
            live_a.cursor_state = clear_until(make_profile_state(children=kids2), "birth_jaundice")
            live_a.turn.spoken_qid = step_id(live_a.cursor_state)
            with mock.patch.object(decisions, "evaluate_turn", side_effect=lambda req, **_: decision_result(req, _eval_payload(
                detail_present=noul(0.92), current_outcome=choice("concern", {"concern": 0.95, "other": 0.03}),
                child_ref=choice("child_2", {"child_2": 0.91, "child_1": 0.04, "other": 0.03}),
            ))):
                await live_a._handle_server_message(server_message(user_text="हाँ छोटे बेटे को चार दिन नीली लाइट में रखा था"))
                await live_a._handle_server_message(server_message(turn_complete=True, tool_calls=[FakeToolCall(fid="fc-multi")]))
                await drop_fallback(live_a.turn)
            self.assertEqual((live_a.cursor_state["answers"]["birth_jaundice"]["child_ids"], live_a.cursor_state["active_qid"]), (["child_2"], "birth_sudden"))
            self.assertIn("चार दिन", live_a.cursor_state["answers"]["birth_jaundice"]["followups"]["jaundice_days"]["note"])

        asyncio.run(run())

    def test_missed_tool_wav_fallback_speaker_echo_gating_and_midflight_race(self):
        thresholds = shipped_thresholds()

        async def run():
            state = clear_until(make_profile_state(children=[make_child(4, "male")]), "birth_nicu")
            ws, sess, cid = FakeWebSocket(), FakeLiveSession(), db.create_call()
            live = _make_live(ws=ws, session=sess, call_id=cid, cursor_state=json.loads(json.dumps(state)), thresholds=thresholds)
            live.turn.spoken_qid = "birth_nicu"

            # 1) Speakerphone echo during bot_speaking OR playback_ends_at window ignored
            loud_pcm = b"\x00\x10" * 1600
            loud_b64 = base64.b64encode(loud_pcm).decode("ascii")
            for bot_spk, ends_at in ((True, 0.0), (False, time.time() + 1.5)):
                live.turn.bot_speaking, live.turn.playback_ends_at, live.turn.quiet_from = bot_spk, ends_at, 77.0
                live.ws = FakeWebSocket([{"type": "audio", "data": loud_b64, "sample_rate": 16000}])
                await live._browser_loop()
                live.stop.clear()
                self.assertEqual((len(live.capture.user_turn_pcm), live.turn.quiet_from, live.telemetry["user_turns"]), (0, 77.0, 0))

            # 2) Post-playback audio-only reply (zero ASR) unlocks lingering handled_turn on step N+1 & fires missed_tool WAV fallback
            live.cursor_state = apply_turn(live.cursor_state, "clear")
            live.turn.spoken_qid, live.turn.bot_speaking, live.turn.playback_ends_at = "birth_jaundice", False, 0.0
            live.turn.user_turn_step, live.capture.handled_turn, live.capture.last_key = "birth_nicu", True, ("birth_nicu", "नहीं", 0)
            live.ws = FakeWebSocket([{"type": "audio", "data": loud_b64, "sample_rate": 16000}])
            await live._browser_loop()
            live.stop.clear()
            self.assertFalse(live.capture.handled_turn or live.capture.speech_allowed or live.capture.last_key)
            self.assertEqual((live.turn.user_turn_step, bytes(live.capture.user_turn_pcm)), ("birth_jaundice", loud_pcm))

            live.ws = ws
            await live._handle_server_message(server_message(bot_text="बिना टूल के उत्तर"))
            await live._handle_server_message(bot_audio_message(turn_complete=False))
            self.assertEqual(live.telemetry["audio_chunks_out"], 0)
            seen_reqs = []

            async def eval_cap(req, **_kw):
                seen_reqs.append(req)
                return decision_result(req, _eval_payload())

            with mock.patch.object(decisions, "evaluate_turn", side_effect=eval_cap):
                await live._handle_server_message(server_message(turn_complete=True))
            self.assertEqual((len(seen_reqs), seen_reqs[0]["input"][1]["mime_type"], live.cursor_state["active_qid"]), (1, "audio/wav", "birth_sudden"))
            self.assertTrue(any(e["type"] == "tool_call" and e["detail"] == "missed_tool_clear" for e in db.get_call(cid)["events"]))

            # 3) Same-step post-playback audio-only reply & 960_000 cap
            live.turn.spoken_qid = live.turn.user_turn_step = "birth_sudden"
            live.capture.handled_turn = True
            live.ws = FakeWebSocket([{"type": "audio", "data": loud_b64, "sample_rate": 16000}])
            await live._browser_loop()
            live.stop.clear()
            self.assertFalse(live.capture.handled_turn or live.capture.speech_allowed)

            big_b64 = base64.b64encode(b"\x00\x10" * 160_000).decode("ascii")
            cap_live = _make_live(ws=FakeWebSocket([{"type": "audio", "data": big_b64, "sample_rate": 16000} for _ in range(4)]), cursor_state=json.loads(json.dumps(live.cursor_state)))
            cap_live.turn.spoken_qid = step_id(cap_live.cursor_state)
            await cap_live._browser_loop()
            self.assertEqual(len(cap_live.capture.user_turn_pcm), 960_000)

            # 4) Finalized ASR + trailing mic audio keeps transcript; mid-flight evaluate_turn preserves next turn buffer
            live.capture.reset()
            await live._handle_server_message(server_message(user_text="हाँ, बक्से में", user_finished=True))
            snap_before = copy.deepcopy(live.capture.user_turn_cursor_snapshot)
            await live._handle_server_message(server_message(user_text="रखा था", user_finished=True))
            live.ws = FakeWebSocket([{"type": "audio", "data": loud_b64, "sample_rate": 16000}])
            await live._browser_loop()
            live.stop.clear()
            self.assertEqual((live.capture.user_turn_buffer, live.capture.user_turn_cursor_snapshot), ("हाँ, बक्से में रखा था", snap_before))

            eval_started, release_eval = asyncio.Event(), asyncio.Event()

            async def slow_eval(req, **_kw):
                eval_started.set()
                await release_eval.wait()
                return decision_result(req, _eval_payload())

            live.ws = ws
            with mock.patch.object(decisions, "evaluate_turn", side_effect=slow_eval):
                task = asyncio.create_task(live._handle_server_message(server_message(turn_complete=True, tool_calls=[FakeToolCall(fid="fc-race")])))
                await asyncio.wait_for(eval_started.wait(), timeout=1.0)
                live.capture.user_turn_buffer, live.capture.user_turn_pcm = "अगले सवाल का जवाब", bytearray(b"\x01\x00" * 320)
                release_eval.set()
                await task
                await drop_fallback(live.turn)
            self.assertEqual((live.capture.user_turn_buffer, bytes(live.capture.user_turn_pcm)), ("अगले सवाल का जवाब", b"\x01\x00" * 320))

            # 5) Empty-turn & not-spoken holds + VAD interrupted barge-in
            hold_live = _make_live(session=FakeLiveSession(), cursor_state=json.loads(json.dumps(state)))
            hold_live.turn.spoken_qid = hold_live.turn.user_turn_step = step_id(state)
            with mock.patch.object(decisions, "evaluate_turn") as eval_mock:
                await hold_live._answer_function_calls([FakeToolCall(fid="fc-empty")])
                hold_live.turn.spoken_qid = "other"
                await hold_live._answer_function_calls([FakeToolCall(fid="fc-unspoken")])
                eval_mock.assert_not_called()
            self.assertEqual([i[1][0].response["reason"] for i in hold_live.session.sent], ["empty_turn", "not_spoken"])

            barge = _make_live(cursor_state=json.loads(json.dumps(state)))
            barge.turn.bot_speaking, barge.capture.handled_turn, barge.capture.last_key = True, True, ("birth_jaundice", "", 3200)
            await barge._handle_server_message(server_message(interrupted=True))
            self.assertFalse(barge.turn.bot_speaking or barge.capture.handled_turn or barge.capture.last_key is not None)

        asyncio.run(run())

    def test_emergency_caller_stop_technical_failure_and_extractor_failures(self):
        thresholds = shipped_thresholds()

        async def run():
            # 1) Historical body_blue stays asking; active danger_now closes with 108
            hist = _make_live(cursor_state=clear_until(make_profile_state(children=[make_child(4, "male")]), "body_blue"), thresholds=thresholds)
            hist.turn.spoken_qid = step_id(hist.cursor_state)
            with mock.patch.object(decisions, "evaluate_turn", side_effect=lambda req, **_: decision_result(req, _eval_payload(danger_now=noul(0.08), current_outcome=choice("concern", {"concern": 0.92, "other": 0.08})))):
                await hist._handle_server_message(server_message(user_text="रोते समय कभी नीली पड़ती हैं"))
                await hist._handle_server_message(server_message(turn_complete=True, tool_calls=[FakeToolCall(fid="fc-h")]))
                await drop_fallback(hist.turn)
            self.assertEqual((hist.cursor_state["phase"], hist.cursor_state["answers"]["body_blue"]["state"]), ("asking", "concern"))

            emg_sess, emg_cid = FakeLiveSession(), db.create_call()
            emg = _make_live(session=emg_sess, call_id=emg_cid, cursor_state=clear_until(make_profile_state(children=[make_child(4, "male")]), "body_blue"), thresholds=thresholds)
            emg.turn.spoken_qid = step_id(emg.cursor_state)
            with mock.patch.object(decisions, "evaluate_turn", side_effect=lambda req, **_: decision_result(req, _eval_payload(danger_now=noul(0.96), current_outcome=choice("concern", {"concern": 0.90, "other": 0.10})))):
                await emg._handle_server_message(server_message(user_text="अभी साँस नहीं आ रही"))
                await emg._handle_server_message(server_message(turn_complete=True, tool_calls=[FakeToolCall(fid="fc-e")]))
                await drop_fallback(emg.turn)
            self.assertEqual((emg.cursor_state["phase"], emg.cursor_state["close_reason"]), ("closing", "emergency"))
            self.assertTrue("108" in emg_sess.sent[0][1][0].response["say"] and emg_sess.sent[0][1][0].response["stop"])

            # 2) Decision failure -> calm technical_failure, no double save_cursor_state
            f_state = apply_turn(clear_until(make_profile_state(children=[make_child(4, "male")]), "birth_nicu"), "concern", note="सांस", followup_answered=True)
            f_sess = FakeLiveSession()
            fail_live = _make_live(session=f_sess, cursor_state=f_state, thresholds=thresholds)
            fail_live.turn.spoken_qid = fail_live.turn.user_turn_step = step_id(f_state)
            fail_live.capture.user_turn_buffer = "नहीं"
            with mock.patch.object(decisions, "evaluate_turn", side_effect=lambda req, **_: decision_result(req, {}, ok=False, error_reason="timeout", attempts=2)), mock.patch.object(
                db, "save_cursor_state", wraps=db.save_cursor_state
            ) as save_spy:
                await fail_live._answer_function_calls([FakeToolCall(fid="fc-fail")])
                await drop_fallback(fail_live.turn)
            fail_say = f_sess.sent[0][1][0].response["say"]
            self.assertEqual((fail_live.cursor_state["close_reason"], save_spy.call_count), ("technical_failure", 1))
            self.assertTrue("पूजा जी" in fail_say and "अधूरी" in fail_say and "108" not in fail_say)

            # 3) Profile extractor failures: 1st re-asks, 2nd closes technical_failure (intro & mid-call hold); gate advances & resets; clarify skips
            async def fail_extract(*_a, **_kw):
                return profile_extractor.ProfileExtractionResult(ok=False, updates={}, error_reason="timeout", extractor_ms=2000)

            eval_prof = lambda req, **_: decision_result(req, {"danger_now": noul(0.01), "caller_stop": noul(0.01), "mentions_profile": noul(0.92), "current_outcome": choice("other", {"other": 0.91, "clear": 0.09})})
            ext_live = _make_live(cursor_state=call_cursor.new_cursor_state(), thresholds=thresholds)
            with mock.patch.object(decisions, "evaluate_turn", side_effect=eval_prof), mock.patch.object(profile_extractor, "extract_profile_entities", side_effect=fail_extract):
                for idx, utt in enumerate(("मेरा नाम रेखा", "रेखा"), start=1):
                    ext_live.turn.spoken_qid = step_id(ext_live.cursor_state)
                    await ext_live._handle_server_message(server_message(user_text=utt))
                    await ext_live._handle_server_message(server_message(turn_complete=True, tool_calls=[FakeToolCall(fid=f"fc-x{idx}")]))
                    await drop_fallback(ext_live.turn)
                    if idx == 1:
                        self.assertEqual((ext_live.cursor_state["phase"], ext_live.capture.extractor_failures), ("asking", 1))
            self.assertEqual(ext_live.cursor_state["close_reason"], "technical_failure")

            gate_live = _make_live(cursor_state=clear_until(make_profile_state(children=[make_child(4, "male")]), "birth_nicu"), thresholds=thresholds)
            gate_live.capture.user_turn_buffer = "नहीं"
            with mock.patch.object(decisions, "evaluate_turn", side_effect=lambda req, **_: decision_result(req, _eval_payload(mentions_profile=noul(0.85)))), mock.patch.object(
                profile_extractor, "extract_profile_entities", side_effect=fail_extract
            ):
                await live_decision_bridge.run_decision_turn(gate_live)
            self.assertEqual((gate_live.cursor_state["active_qid"], gate_live.capture.extractor_failures), ("birth_jaundice", 0))

            clar_live = _make_live(cursor_state=call_cursor.new_cursor_state(), thresholds=thresholds)
            clar_live.capture.user_turn_buffer = "क्या कहा?"
            with mock.patch.object(decisions, "evaluate_turn", side_effect=lambda req, **_: decision_result(req, {"danger_now": noul(0.01), "caller_stop": noul(0.01), "mentions_profile": noul(0.05), "current_outcome": choice("clarify", {"clarify": 0.92, "other": 0.08})})), mock.patch.object(
                profile_extractor, "extract_profile_entities"
            ) as ext_skip:
                body, clar_flag, _ = await live_decision_bridge.run_decision_turn(clar_live)
                ext_skip.assert_not_awaited()
            self.assertTrue(clar_flag and body.get("clarify"))

            hold_s, _ = call_cursor.apply_profile(clear_until(make_profile_state(children=[make_child(4, "male")]), "birth_nicu"), child_count=2, children=[{"age_years": 4.0, "sex": "male"}])
            hold_live = _make_live(cursor_state=hold_s, thresholds=thresholds)
            with mock.patch.object(decisions, "evaluate_turn", side_effect=eval_prof), mock.patch.object(profile_extractor, "extract_profile_entities", side_effect=fail_extract):
                hold_live.capture.user_turn_buffer = "छोटा दो साल"
                reask, _, out1 = await live_decision_bridge.run_decision_turn(hold_live)
                hold_live.capture.user_turn_buffer = "दो साल"
                _, _, out2 = await live_decision_bridge.run_decision_turn(hold_live)
            self.assertEqual((out1, out2, hold_live.cursor_state["close_reason"]), ("not_about_active", "technical_failure", "technical_failure"))
            self.assertIn("उम्र", reask["say"])

            # 4) Audio-only intro_name speech reaches extractor with 16k WAV & advances to intro_married
            loud_pcm = b"\x00\x10" * 1600
            audio_live = _make_live(ws=FakeWebSocket([{"type": "audio", "data": base64.b64encode(loud_pcm).decode("ascii"), "sample_rate": 16000}]), cursor_state=call_cursor.new_cursor_state(), thresholds=thresholds)
            audio_live.turn.spoken_qid = "intro_name"
            await audio_live._browser_loop()
            fake_client = mock.MagicMock()
            fake_client.aio.models.generate_content = mock.AsyncMock(return_value=SimpleNamespace(text=json.dumps({"caller_name": "पूजा", "marital_status": None, "child_count": None, "children": None})))
            with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}), mock.patch.dict(profile_extractor._clients, {}, clear=True), mock.patch.object(
                profile_extractor.genai, "Client", return_value=fake_client
            ), mock.patch.object(decisions, "evaluate_turn", side_effect=eval_prof):
                await audio_live._answer_function_calls([FakeToolCall(fid="fc-ai")])
                await drop_fallback(audio_live.turn)
            gen_kw = fake_client.aio.models.generate_content.await_args.kwargs
            self.assertEqual((audio_live.cursor_state["profile"]["caller_name"], audio_live.cursor_state["active_qid"], gen_kw["contents"][1].inline_data.mime_type),
                             ("पूजा", "intro_married", "audio/wav"))

        asyncio.run(run())

    def test_browser_text_reconnect_and_transient_db_resilience(self):
        async def run():
            async def boom():
                raise RuntimeError("task boom")

            failed_task = asyncio.create_task(boom())
            await asyncio.gather(failed_task, return_exceptions=True)
            with self.assertLogs(gemini_live.logger, level="WARNING") as cm:
                gemini_live.consume_task_error(failed_task)
            self.assertTrue(any("background live task failed" in line for line in cm.output))

            state = clear_until(make_profile_state(children=[make_child(4, "male")]), "birth_jaundice")
            with mock.patch.object(db, "append_transcript", side_effect=RuntimeError("sqlite locked")), self.assertLogs(gemini_live.logger, level="WARNING") as tr_logs:
                self.assertEqual(await _make_live(cursor_state=state)._forward_transcript("user", SimpleNamespace(text="हाँ")), "हाँ")
            self.assertTrue(any("append_transcript failed" in line for line in tr_logs.output))

            txt_reconn = _make_live(ws=FakeWebSocket(), session=FakeLiveSession(fail_realtime=True), cursor_state=json.loads(json.dumps(state)))
            txt_reconn.turn.resume_handle = "h-txt"
            self.assertTrue(await txt_reconn._on_browser_text({"text": "हाँ"}))
            self.assertTrue(txt_reconn.turn.reconnect_requested and not txt_reconn.stop.is_set())

            ws_err = FakeWebSocket([{"type": "text", "text": "हाँ"}])
            txt_err = _make_live(ws=ws_err, session=FakeLiveSession(fail_realtime=True), cursor_state=json.loads(json.dumps(state)))
            await txt_err._browser_loop()
            self.assertTrue(txt_err.stop.is_set())
            self.assertIn({"type": "error", "message": "Could not forward text to Gemini"}, ws_err.sent)

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
