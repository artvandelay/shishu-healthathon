"""Live voice server tests: config, turn-taking, quick replies, and Decision-API execution."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import sys
import tempfile
import time
import unittest
import wave
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from tests import APP_DIR, TEST_DATA_DIR  # noqa: E402

import audio_recorder  # noqa: E402
import call_cursor  # noqa: E402
import db  # noqa: E402
import decisions  # noqa: E402
import gemini_live  # noqa: E402
import live_decision_bridge  # noqa: E402
import profile_extractor  # noqa: E402
import prompts  # noqa: E402
import test_cursor as cursor_tests  # noqa: E402
from call_cursor import apply_turn, close_technical_failure, step_id  # noqa: E402


class _FakeSocket:
    def __init__(self, messages=None):
        self.client_state = gemini_live.WebSocketState.CONNECTED
        self.sent = []
        self._messages = list(messages or [])
        self.accepted = False
        self.closed = False

    async def accept(self):
        self.accepted = True
        self.client_state = gemini_live.WebSocketState.CONNECTED

    async def close(self):
        self.closed = True
        self.client_state = gemini_live.WebSocketState.DISCONNECTED

    async def send_json(self, payload):
        self.sent.append(payload)

    async def receive_text(self):
        if not self._messages:
            raise gemini_live.WebSocketDisconnect()
        return json.dumps(self._messages.pop(0), ensure_ascii=False)


class _FakeSession:
    def __init__(self, fail_tool=False, fail_realtime=False):
        self.sent = []
        self.fail_tool = fail_tool
        self.fail_realtime = fail_realtime

    async def send_tool_response(self, function_responses):
        self.sent.append(("tool", function_responses))
        if self.fail_tool:
            raise RuntimeError("tool response failed")

    async def send_realtime_input(self, **kwargs):
        self.sent.append(("realtime", kwargs))
        if self.fail_realtime:
            raise RuntimeError("realtime send failed")


class _Call:
    def __init__(self, name, args, fid):
        self.id = fid
        self.name = name
        self.args = args
        self.will_continue = False


class _Recorder:
    def add_user_pcm(self, pcm, rate):
        return None

    def add_bot_pcm(self, pcm, rate):
        return None


def _kinds(session):
    return [item[0] for item in session.sent]


async def _drop_fallback(turn):
    task = turn.speak_fallback
    turn.speak_fallback = None
    if task is None:
        return
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


def _server_message(*, turn_complete=False, user_text="", bot_text="", tool_calls=None, go_away=False):
    tool_call = SimpleNamespace(function_calls=tool_calls) if tool_calls else None
    return SimpleNamespace(
        tool_call_cancellation=None,
        server_content=SimpleNamespace(
            interrupted=False,
            input_transcription=SimpleNamespace(text=user_text) if user_text else None,
            output_transcription=SimpleNamespace(text=bot_text) if bot_text else None,
            model_turn=None,
            turn_complete=turn_complete,
        ),
        tool_call=tool_call,
        session_resumption_update=None,
        go_away=SimpleNamespace(time_left="50s") if go_away else None,
    )


def _bot_audio_msg(pcm: bytes = b"\x01\x00" * 240, *, turn_complete: bool = False):
    part = SimpleNamespace(
        thought=False,
        function_call=None,
        inline_data=SimpleNamespace(data=pcm, mime_type="audio/pcm;rate=24000"),
    )
    return SimpleNamespace(
        tool_call_cancellation=None,
        server_content=SimpleNamespace(
            interrupted=False,
            input_transcription=None,
            output_transcription=None,
            model_turn=SimpleNamespace(parts=[part]),
            turn_complete=turn_complete,
        ),
        tool_call=None,
        session_resumption_update=None,
        go_away=None,
    )


def _noul(prob: float) -> dict:
    return {"type": "noul", "noul": prob, "confidence": max(prob, 1.0 - prob)}


def _choice(winner: str, probs: dict[str, float]) -> dict:
    return {"type": "choice", "choice": winner, "probabilities": probs, "confidence": probs[winner]}


def _decision_result(request: dict, raw_answers: dict, *, ok: bool = True, error_reason: str = "", attempts: int = 1):
    questions = request.get("questions") or {}
    side_criteria = (questions.get("side_qid") or {}).get("criteria") or {}
    return decisions.DecisionCallResult(
        ok=ok,
        error_reason=error_reason,
        attempts=attempts,
        decision_ms=42,
        request_bytes=len(json.dumps(request, ensure_ascii=False).encode("utf-8")),
        question_count=len(questions),
        catalog_option_count=len(side_criteria),
        usage_present=True,
        total_input_tokens=380,
        raw_answers=raw_answers if ok else {},
    )


def _test_thresholds() -> decisions.DecisionThresholds:
    return decisions.load_thresholds(APP_DIR / "decision_thresholds.json")


def _new_telemetry():
    return gemini_live._new_telemetry("gemini-3.1-flash-live-preview", "Kore", 1000, 300, 1)


class LiveConfigAndStartupTests(unittest.TestCase):
    def test_live_config_declares_only_next_question_and_kore_vad_defaults(self):
        prompt = prompts.GEMINI_LIVE_SYSTEM_PROMPT
        self.assertIn("Kore", prompt)
        self.assertIn("next_question", prompt)
        self.assertIn("say", prompt)
        self.assertIn("clarify", prompt)
        self.assertIn("BCG", prompt)
        self.assertIn("थैलेसीमिया माइनर या ट्रेट", prompt)
        self.assertIn("recorded_concerns", prompt)
        self.assertIn("stop", prompt)
        self.assertIn("SYSTEM CUE", prompt)
        for disease in (
            "Duchenne",
            "Hemophilia",
            "Pompe",
            "Gaucher",
            "SCID",
            "SMA",
            "MPS",
            "Wiskott-Aldrich",
            "ड्यूशेन",
            "हीमोफीलिया",
            "पॉम्पे",
            "गॉशर",
            "इम्यूनोडेफिशिएंसी",
            "म्यूकोपॉलीसेकेराइडोसिस",
        ):
            self.assertIn(disease, prompt)
        for forbidden in ("target_qid", "child_ref", "outcome", "marital_status", "caller_stop", "ever_pregnant", "emergency"):
            self.assertNotIn(forbidden, prompt)

        cfg = gemini_live._live_config({"gemini_model": "gemini-3.1-flash-live-preview"})
        self.assertEqual(cfg.system_instruction.parts[0].text, prompt)
        self.assertEqual(cfg.speech_config.voice_config.prebuilt_voice_config.voice_name, "Kore")
        self.assertEqual(list(cfg.output_audio_transcription.language_codes), ["hi-IN"])
        self.assertEqual(list(cfg.input_audio_transcription.language_codes), ["hi-IN"])
        self.assertEqual(cfg.thinking_config.thinking_level, "MINIMAL")
        decls = cfg.tools[0].function_declarations
        self.assertEqual([d.name for d in decls], ["next_question"])
        self.assertEqual(dict(decls[0].parameters.properties or {}), {})

        custom_cfg = gemini_live._live_config({"gemini_model": "custom-live-model"})
        self.assertEqual(list(custom_cfg.output_audio_transcription.language_codes), ["hi-IN"])
        self.assertEqual(list(custom_cfg.input_audio_transcription.language_codes), ["hi-IN"])
        self.assertEqual(custom_cfg.thinking_config.thinking_level, "MINIMAL")
        self.assertFalse(hasattr(gemini_live, "_cancel_silence_close"))
        self.assertIsNone(
            gemini_live._audio_from_part(
                SimpleNamespace(inline_data=SimpleNamespace(data=b"\x01\x00", mime_type="text/plain"))
            )
        )
        self.assertIsNotNone(
            gemini_live._audio_from_part(
                SimpleNamespace(inline_data=SimpleNamespace(data=b"\x01\x00", mime_type="audio/pcm;rate=24000"))
            )
        )

        # Live settings defaults, env/override precedence, custom voice, and no dead DB settings table
        self.assertEqual(gemini_live.DEFAULT_LIVE_SETTINGS["gemini_model"], "gemini-3.1-flash-live-preview")
        self.assertEqual(gemini_live.DEFAULT_LIVE_SETTINGS["gemini_voice"], "Kore")
        self.assertEqual(gemini_live.DEFAULT_LIVE_SETTINGS["vad_prefix_padding_ms"], "300")
        self.assertEqual(gemini_live.DEFAULT_LIVE_SETTINGS["vad_silence_ms"], "1000")
        self.assertEqual(gemini_live.DEFAULT_LIVE_SETTINGS["vad_end_sensitivity"], "LOW")
        self.assertEqual(gemini_live.DEFAULT_LIVE_SETTINGS["vad_start_sensitivity"], "LOW")
        self.assertEqual(gemini_live.DEFAULT_LIVE_SETTINGS["watchdog_timeout_sec"], "8.0")
        self.assertNotIn("active_engine", gemini_live.DEFAULT_LIVE_SETTINGS)
        for dead_attr in ("DEFAULT_SETTINGS", "get_setting", "set_setting", "get_all_settings"):
            self.assertFalse(hasattr(db, dead_attr))
        db._init_db()
        with db._db_lock:
            dead_table = db.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='settings'"
            ).fetchone()
        self.assertIsNone(dead_table)

        env_keys = (
            "GEMINI_MODEL",
            "GEMINI_VOICE",
            "VAD_PREFIX_PADDING_MS",
            "VAD_SILENCE_MS",
            "VAD_START_SENSITIVITY",
            "VAD_END_SENSITIVITY",
            "WATCHDOG_TIMEOUT_SEC",
            "DECISION_THRESHOLDS_PATH",
        )
        clean_env = {k: v for k, v in os.environ.items() if k not in env_keys}
        with mock.patch.dict(os.environ, clean_env, clear=True):
            unset = gemini_live._session_settings({})
            self.assertEqual(unset["voice"], "Kore")
            self.assertEqual(unset["silence_duration_ms"], 1000)
            self.assertEqual(unset["end_sens"], gemini_live.types.EndSensitivity.END_SENSITIVITY_LOW)
            self.assertEqual(unset["start_sens"], gemini_live.types.StartSensitivity.START_SENSITIVITY_LOW)
            self.assertEqual(unset["watchdog_timeout_sec"], 8.0)

            custom = gemini_live._session_settings({
                "gemini_voice": "Aoede",
                "vad_silence_ms": "1200",
                "vad_end_sensitivity": "HIGH",
                "vad_start_sensitivity": "HIGH",
            })
            self.assertEqual(custom["voice"], "Aoede")
            self.assertEqual(custom["silence_duration_ms"], 1200)
            self.assertEqual(custom["end_sens"], gemini_live.types.EndSensitivity.END_SENSITIVITY_HIGH)
            self.assertEqual(custom["start_sens"], gemini_live.types.StartSensitivity.START_SENSITIVITY_HIGH)
            voice_cfg = gemini_live._live_config({"gemini_voice": "Aoede"})
            self.assertEqual(voice_cfg.speech_config.voice_config.prebuilt_voice_config.voice_name, "Aoede")

            with mock.patch.dict(
                os.environ,
                {
                    "GEMINI_MODEL": "",
                    "GEMINI_VOICE": "Puck",
                    "VAD_SILENCE_MS": "900",
                    "WATCHDOG_TIMEOUT_SEC": "6.5",
                    "DECISION_THRESHOLDS_PATH": "/tmp/env-thresholds.json",
                },
                clear=False,
            ):
                from_env = gemini_live.live_settings()
                self.assertEqual(from_env["gemini_model"], "gemini-3.1-flash-live-preview")
                self.assertEqual(from_env["gemini_voice"], "Puck")
                self.assertEqual(from_env["vad_silence_ms"], "900")
                self.assertEqual(from_env["watchdog_timeout_sec"], "6.5")
                self.assertEqual(from_env["decision_thresholds_path"], "/tmp/env-thresholds.json")
                self.assertEqual(gemini_live._session_settings()["voice"], "Puck")

                overlaid = gemini_live.live_settings({"gemini_voice": "Aoede", "vad_silence_ms": None})
                self.assertEqual(overlaid["gemini_voice"], "Aoede")
                self.assertEqual(overlaid["vad_silence_ms"], "900")

        self.assertEqual(gemini_live._TOOL_SPEAK_FALLBACK_SEC, 2.5)

    def test_handle_gemini_ws_loads_default_thresholds_and_fails_closed_on_missing_or_invalid(self):
        default_loaded = live_decision_bridge.load_turn_thresholds({})
        self.assertEqual(default_loaded, _test_thresholds())

        with tempfile.TemporaryDirectory() as tmp:
            bad_path = Path(tmp) / "bad_thresholds.json"
            bad_data = asdict(default_loaded)
            bad_data["danger_now_min_yes"] = 0.999
            bad_path.write_text(json.dumps(bad_data), encoding="utf-8")

            for bad_target in ("/tmp/does-not-exist-shishu-thresholds.json", str(bad_path)):
                ws = _FakeSocket()
                with mock.patch.dict(
                    os.environ,
                    {"DECISION_THRESHOLDS_PATH": bad_target, "GEMINI_API_KEY": "test-key"},
                    clear=False,
                ), mock.patch.object(gemini_live, "_connect_live") as connect_mock, mock.patch.object(
                    db, "create_call"
                ) as create_mock:
                    asyncio.run(gemini_live.handle_gemini_ws(ws))
                    connect_mock.assert_not_called()
                    create_mock.assert_not_called()
                self.assertTrue(ws.closed)
                errors = [m for m in ws.sent if m.get("type") == "error"]
                self.assertEqual(errors, [{"type": "error", "message": "Invalid or missing Decision API thresholds"}])

            good_path = Path(tmp) / "good_thresholds.json"
            good_path.write_text(json.dumps(asdict(default_loaded)), encoding="utf-8")
            self.assertEqual(
                live_decision_bridge.load_turn_thresholds({"decision_thresholds_path": str(good_path)}),
                default_loaded,
            )

    def test_handle_gemini_ws_wires_config_default_thresholds_and_turn_end_to_end(self):
        class _WsSession:
            def __init__(self, msgs):
                self.setup_complete = True
                self.sent = []
                self._msgs = list(msgs)

            async def send_realtime_input(self, **kwargs):
                self.sent.append(("realtime", kwargs))

            async def send_tool_response(self, function_responses):
                self.sent.append(("tool", function_responses))

            async def receive(self):
                for item in self._msgs:
                    await asyncio.sleep(0.01)
                    yield item
                await asyncio.Future()

        class _WsCtx:
            def __init__(self, sess):
                self._sess = sess

            async def __aenter__(self):
                return self._sess

            async def __aexit__(self, *_args):
                return False

        class _LiveWs(_FakeSocket):
            def __init__(self):
                super().__init__([])
                self._q: asyncio.Queue = asyncio.Queue()

            async def receive_text(self):
                item = await self._q.get()
                if item is None:
                    raise gemini_live.WebSocketDisconnect()
                return json.dumps(item, ensure_ascii=False)

        async def run():
            seen_configs = []
            live_sess = _WsSession([
                _bot_audio_msg(turn_complete=True),
                _server_message(user_text="रेखा"),
                _server_message(turn_complete=True, tool_calls=[_Call("next_question", {}, "fc-ws-1")]),
                _bot_audio_msg(turn_complete=True),
            ])

            async def fake_connect(client, model, config):
                seen_configs.append(config)
                return _WsCtx(live_sess), live_sess, 5

            async def fake_eval(request, **_kwargs):
                raw = {
                    "danger_now": _noul(0.01),
                    "caller_stop": _noul(0.01),
                    "mentions_profile": _noul(0.90),
                    "current_outcome": _choice("other", {"other": 0.90, "clear": 0.10}),
                }
                return _decision_result(request, raw)

            async def hook_profile(_state, _utt, _wav):
                return profile_extractor.ProfileExtractionResult(
                    ok=True, updates={"caller_name": "रेखा"}, error_reason="", extractor_ms=1
                )

            ws = _LiveWs()

            async def hangup_soon():
                await asyncio.sleep(0.15)
                await ws._q.put({"type": "hangup"})

            with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}), mock.patch.object(
                gemini_live, "_connect_live", side_effect=fake_connect
            ), mock.patch.object(
                live_decision_bridge, "extract_profile_updates", side_effect=hook_profile
            ), mock.patch.object(
                decisions, "evaluate_turn", side_effect=fake_eval
            ):
                task = asyncio.create_task(hangup_soon())
                await gemini_live.handle_gemini_ws(ws)
                await task

            self.assertEqual(len(seen_configs), 1)
            decls = seen_configs[0].tools[0].function_declarations
            self.assertEqual([d.name for d in decls], ["next_question"])
            started = next(m for m in ws.sent if m.get("type") == "session_started")
            saved = db.get_call(started["call_id"])
            self.assertEqual(saved["cursor_state"]["profile"]["caller_name"], "रेखा")
            self.assertEqual(saved["cursor_state"]["active_qid"], "intro_married")
            dt_ev = [e for e in saved["events"] if e["type"] == "decision_turn"]
            self.assertEqual(len(dt_ev), 1)

        asyncio.run(run())


class TurnTakingAndWatchdogTests(unittest.TestCase):
    def test_greeting_and_recorded_concerns_tool_body(self):
        cursor = gemini_live._Cursor()
        session = _FakeSession()
        ws = _FakeSocket()
        asyncio.run(gemini_live._send_greeting(session, asyncio.Lock(), ws, asyncio.Lock(), cursor))
        self.assertEqual(_kinds(session), ["realtime"])
        self.assertIn("नाम", session.sent[0][1]["text"])
        states = [item for item in ws.sent if item.get("type") == "question_state"]
        self.assertEqual(len(states), 1)
        self.assertEqual(states[0]["step"], "intro_name")
        self.assertFalse(states[0]["expect_yes_no"])
        self.assertGreater(states[0]["prompt_chars"], 0)

        state = cursor_tests._clear_until(
            cursor_tests._profile(children=[cursor_tests._child(4, "male")]),
            "birth_jaundice",
        )
        state["answers"]["birth_nicu"] = {
            "state": "concern",
            "child_ids": ["child_1"],
            "notes": "साँस की तकलीफ",
            "followups": {"nicu_days": {"state": "concern", "note": "चार दिन"}},
        }
        self.assertNotIn("recorded_concerns", gemini_live._tool_body(state))
        lines = prompts.recorded_concerns_hi(state)
        self.assertEqual(len(lines), 1)
        self.assertIn("काँच के बक्से", lines[0])
        self.assertIn("4 साल", lines[0])
        self.assertIn("साँस की तकलीफ", lines[0])
        self.assertIn("चार दिन", lines[0])
        clarify_body = gemini_live._tool_body(state, outcome="clarify")
        self.assertEqual(clarify_body["recorded_concerns"], lines)

        fam_same = cursor_tests._clear_until(state, "fam_same_problem")
        self.assertEqual(gemini_live._tool_body(fam_same)["recorded_concerns"], lines)
        close_state = cursor_tests._clear_until(
            cursor_tests._profile(children=[cursor_tests._child(4, "male")]),
            "close_anything_else",
        )
        self.assertEqual(gemini_live._tool_body(close_state)["recorded_concerns"], [])

    def test_fallback_speaks_only_when_silent_and_turn_complete_speaks_immediately(self):
        thresholds = _test_thresholds()
        state = cursor_tests._clear_until(
            cursor_tests._profile(children=[cursor_tests._child(4, "male")]),
            "birth_jaundice",
        )
        step = step_id(state)

        async def fake_clear(request, **_kwargs):
            raw = {
                "danger_now": _noul(0.01),
                "caller_stop": _noul(0.01),
                "mentions_profile": _noul(0.01),
                "detail_present": _noul(0.03),
                "current_outcome": _choice("clear", {"clear": 0.94, "other": 0.04, "concern": 0.02}),
                "side_intent": _choice("other", {"other": 0.96, "add_concern": 0.04}),
                "side_qid": _choice("other", {"other": 0.96, "birth_nicu": 0.04}),
            }
            return _decision_result(request, raw)

        async def run():
            with mock.patch.object(decisions, "evaluate_turn", side_effect=fake_clear):
                # 1) 2.5s fallback speaks when Gemini stays silent; no "wait" in FunctionResponse
                turn = gemini_live._TurnState()
                turn.spoken_qid = step
                turn.user_turn_step = step
                turn.capture.user_turn_buffer = "नहीं"
                cursor = gemini_live._Cursor()
                cursor.state = json.loads(json.dumps(state))
                session = _FakeSession()
                ws = _FakeSocket()
                with mock.patch.object(gemini_live, "_TOOL_SPEAK_FALLBACK_SEC", 0.01):
                    await gemini_live._answer_function_calls(
                        session, asyncio.Lock(), ws, asyncio.Lock(), db.create_call(),
                        _new_telemetry(), [_Call("next_question", {}, "fc-silent")],
                        set(), set(), cursor, turn, thresholds=thresholds,
                    )
                    self.assertEqual(_kinds(session), ["tool"])
                    self.assertNotIn("wait", session.sent[0][1][0].response)
                    self.assertFalse(turn.bot_replied_since_user)
                    await turn.speak_fallback
                self.assertEqual(_kinds(session), ["tool", "realtime"])
                self.assertEqual(cursor.state["active_qid"], "birth_sudden")

                # 2) Audio arriving before fallback timer cancels fallback cue
                turn_audio = gemini_live._TurnState()
                turn_audio.spoken_qid = step
                turn_audio.user_turn_step = step
                turn_audio.capture.user_turn_buffer = "नहीं"
                cursor_audio = gemini_live._Cursor()
                cursor_audio.state = json.loads(json.dumps(state))
                sess_audio = _FakeSession()
                ws_audio = _FakeSocket()
                cid = db.create_call()
                with mock.patch.object(gemini_live, "_TOOL_SPEAK_FALLBACK_SEC", 0.05):
                    await gemini_live._answer_function_calls(
                        sess_audio, asyncio.Lock(), ws_audio, asyncio.Lock(), cid,
                        _new_telemetry(), [_Call("next_question", {}, "fc-audio")],
                        set(), set(), cursor_audio, turn_audio, thresholds=thresholds,
                    )
                    await gemini_live._on_bot_audio(
                        ws_audio, asyncio.Lock(), cid, _new_telemetry(),
                        turn_audio, _Recorder(), b"\x00\x00" * 24, 24000, cursor_audio,
                    )
                    await turn_audio.speak_fallback
                self.assertEqual(_kinds(sess_audio), ["tool"])

                # 3) Failed send_tool_response speaks immediately without timer
                turn_fail = gemini_live._TurnState()
                turn_fail.spoken_qid = step
                turn_fail.user_turn_step = step
                turn_fail.capture.user_turn_buffer = "नहीं"
                cursor_fail = gemini_live._Cursor()
                cursor_fail.state = json.loads(json.dumps(state))
                sess_fail = _FakeSession(fail_tool=True)
                await gemini_live._answer_function_calls(
                    sess_fail, asyncio.Lock(), _FakeSocket(), asyncio.Lock(), db.create_call(),
                    _new_telemetry(), [_Call("next_question", {}, "fc-fail")],
                    set(), set(), cursor_fail, turn_fail, thresholds=thresholds,
                )
                self.assertIsNone(turn_fail.speak_fallback)
                self.assertEqual(_kinds(sess_fail), ["tool", "realtime"])

                # 4) Causal speak cue on server_content.turn_complete + error stop on send failure
                turn_tc = gemini_live._TurnState()
                turn_tc.spoken_qid = step
                turn_tc.user_turn_step = step
                turn_tc.capture.user_turn_buffer = "नहीं"
                cursor_tc = gemini_live._Cursor()
                cursor_tc.state = json.loads(json.dumps(state))
                sess_tc = _FakeSession()
                await gemini_live._handle_server_message(
                    _server_message(turn_complete=True, tool_calls=[_Call("next_question", {}, "fc-tc")]),
                    sess_tc, asyncio.Lock(), _FakeSocket(), asyncio.Lock(), db.create_call(),
                    _new_telemetry(), turn_tc, _Recorder(), set(), set(), cursor_tc,
                    stop=asyncio.Event(), thresholds=thresholds,
                )
                await asyncio.sleep(0)
                self.assertEqual(_kinds(sess_tc), ["tool", "realtime"])
                self.assertIn("इमरजेंसी", sess_tc.sent[1][1]["text"])
                self.assertEqual(turn_tc.pending_speak_step, "")
                self.assertIsNone(turn_tc.speak_fallback)

                broken_turn = gemini_live._TurnState()
                broken_turn.pending_speak_step = step_id(cursor_tc.state)
                broken_turn.pending_speak_kind = "speak"
                broken_ws = _FakeSocket()
                broken_tel = _new_telemetry()
                stop = asyncio.Event()
                keep = await gemini_live._handle_server_message(
                    _server_message(turn_complete=True),
                    _FakeSession(fail_realtime=True), asyncio.Lock(), broken_ws, asyncio.Lock(),
                    db.create_call(), broken_tel, broken_turn, _Recorder(), set(), set(),
                    cursor_tc, stop=stop, thresholds=thresholds,
                )
                self.assertFalse(keep)
                self.assertTrue(stop.is_set())
                self.assertEqual(broken_tel["tool_errors"], 1)
                self.assertEqual(
                    [m["message"] for m in broken_ws.sent if m.get("type") == "error"],
                    ["Could not prompt Gemini to speak"],
                )

        asyncio.run(run())

    def test_speech_end_and_watchdog_check_in_silence_close_and_time_limits(self):
        state = cursor_tests._clear_until(
            cursor_tests._profile(children=[cursor_tests._child(4, "male")]),
            "birth_nicu",
        )

        async def run():
            # 1) speech_end refreshes quiet_from after question was spoken
            turn = gemini_live._TurnState()
            turn.spoken_qid = step_id(state)
            turn.bot_speaking = False
            turn.playback_ends_at = time.time() - 1
            turn.quiet_from = 0.0
            cursor = gemini_live._Cursor()
            cursor.state = json.loads(json.dumps(state))
            ws = _FakeSocket([{"type": "speech_end"}])
            await gemini_live._browser_loop(
                _FakeSession(), asyncio.Lock(), ws, asyncio.Lock(), asyncio.Event(),
                db.create_call(), _new_telemetry(), turn, cursor, _Recorder(),
            )
            self.assertGreater(turn.quiet_from, 0)
            self.assertGreaterEqual(turn.quiet_from, turn.playback_ends_at)
            self.assertFalse(turn.awaiting_bot_reply)
            self.assertFalse(turn.speech_end_waiting)

            # 2) Watchdog 15s check-in -> 30s silence close
            wd_turn = gemini_live._TurnState()
            wd_turn.quiet_from = time.time() - 16.0
            wd_cursor = gemini_live._Cursor()
            wd_cursor.state = json.loads(json.dumps(state))
            wd_sess = _FakeSession()
            wd_ws = _FakeSocket()
            wd_tel = _new_telemetry()
            wd_stop = asyncio.Event()
            cid = db.create_call()
            with mock.patch.object(gemini_live, "_WATCHDOG_INTERVAL_SEC", 0.01):
                task = asyncio.create_task(
                    gemini_live._watchdog_loop(
                        wd_sess, asyncio.Lock(), wd_ws, asyncio.Lock(), wd_stop, cid, wd_tel, wd_turn, wd_cursor, 8.0
                    )
                )
                await asyncio.sleep(0.05)
                self.assertTrue(wd_turn.checked_in)
                wd_turn.quiet_from = time.time() - 31.0
                await asyncio.sleep(0.05)
                wd_stop.set()
                await task
            events = [e["type"] for e in db.get_call(cid)["events"]]
            self.assertIn("silence_check_in", events)
            self.assertIn("auto_close", events)
            self.assertEqual(wd_cursor.state["phase"], "closing")
            self.assertEqual(wd_cursor.state["close_reason"], "silence")

            # 3) 19m soft time cue -> 20m hard limit stop
            soft_turn = gemini_live._TurnState()
            soft_turn.call_started_ts = time.time() - gemini_live._SOFT_LIMIT_SEC - 1
            soft_cursor = gemini_live._Cursor()
            soft_cursor.state = json.loads(json.dumps(state))
            soft_stop = asyncio.Event()
            cid_soft = db.create_call()
            with mock.patch.object(gemini_live, "_WATCHDOG_INTERVAL_SEC", 0.01):
                task = asyncio.create_task(
                    gemini_live._watchdog_loop(
                        _FakeSession(), asyncio.Lock(), _FakeSocket(), asyncio.Lock(), soft_stop, cid_soft, _new_telemetry(), soft_turn, soft_cursor, 8.0
                    )
                )
                await asyncio.sleep(0.05)
                self.assertTrue(soft_turn.time_cue_sent)
                self.assertEqual(soft_cursor.state["close_reason"], "time_limit")
                soft_turn.call_started_ts = time.time() - gemini_live._HARD_LIMIT_SEC - 1
                await asyncio.wait_for(soft_stop.wait(), timeout=1.0)
                await task

        asyncio.run(run())

    def test_go_away_reconnect_resumes_dropped_turn(self):
        state = cursor_tests._clear_until(
            cursor_tests._profile(children=[cursor_tests._child(4, "male")]),
            "birth_jaundice",
        )

        async def handle(turn, message, session=None):
            cursor = gemini_live._Cursor()
            cursor.state = json.loads(json.dumps(state))
            ws = _FakeSocket()
            keep = await gemini_live._handle_server_message(
                message, session or _FakeSession(), asyncio.Lock(), ws, asyncio.Lock(),
                db.create_call(), _new_telemetry(), turn, _Recorder(), set(), set(), cursor, stop=asyncio.Event(),
            )
            return keep, cursor, ws

        class _Live:
            def __init__(self, messages, error=None):
                self.messages = list(messages)
                self.error = error
                self.stop = asyncio.Event()
                self.sent = []
                self.started = False

            async def send_realtime_input(self, **kwargs):
                self.sent.append(kwargs)
                self.stop.set()

            async def receive(self):
                if self.started:
                    await asyncio.Future()
                    return
                self.started = True
                if self.error is not None:
                    raise self.error
                for item in self.messages:
                    yield item

        async def run():
            missing = gemini_live._TurnState()
            missing.bot_speaking = True
            keep, _, _ = await handle(missing, _server_message(go_away=True))
            self.assertFalse(keep)

            speaking = gemini_live._TurnState()
            speaking.bot_speaking = True
            speaking.resume_handle = "resume-1"
            session = _FakeSession()
            keep, cursor, _ = await handle(speaking, _server_message(go_away=True), session)
            self.assertTrue(keep)
            self.assertTrue(speaking.go_away_pending)
            self.assertFalse(speaking.reconnect_requested)

            step = step_id(cursor.state)
            speaking.pending_speak_step = step
            speaking.pending_speak_kind = "clarify"
            speaking.active_speak_kind = "clarify"
            keep, _, _ = await handle(speaking, _server_message(turn_complete=True), session)
            self.assertTrue(keep)
            self.assertFalse(speaking.go_away_pending)
            self.assertTrue(speaking.reconnect_requested)
            self.assertEqual(session.sent, [])

            waiting = gemini_live._Cursor()
            waiting.state = json.loads(json.dumps(state))
            wait_turn = gemini_live._TurnState()
            wait_turn.resume_handle = "resume-wait"
            wait_turn.awaiting_bot_reply = True
            wait_turn.pending_speak_step = step_id(waiting.state)
            wait_turn.pending_speak_kind = "clarify"
            wait_turn.active_speak_kind = "clarify"
            live = _Live([_server_message(go_away=True), _server_message(turn_complete=True)])
            seen = []

            async def reconnect(h):
                seen.append(h)
                return True

            ws = _FakeSocket()
            await asyncio.wait_for(
                gemini_live._gemini_loop(
                    live, asyncio.Lock(), ws, asyncio.Lock(), live.stop,
                    db.create_call(), _new_telemetry(), wait_turn, _Recorder(), waiting, reconnect,
                ),
                timeout=3,
            )
            self.assertEqual(seen, ["resume-wait"])
            self.assertIn("she did not understand", live.sent[0]["text"])
            self.assertIn("reconnected", [m.get("type") for m in ws.sent])

            dropped = gemini_live._Cursor()
            dropped.state = json.loads(json.dumps(state))
            drop_turn = gemini_live._TurnState()
            drop_turn.resume_handle = "resume-drop"
            drop_turn.bot_speaking = True
            drop = _Live([], error=ConnectionError("dropped"))
            seen.clear()
            await asyncio.wait_for(
                gemini_live._gemini_loop(
                    drop, asyncio.Lock(), _FakeSocket(), asyncio.Lock(), drop.stop,
                    db.create_call(), _new_telemetry(), drop_turn, _Recorder(), dropped, reconnect,
                ),
                timeout=3,
            )
            self.assertEqual(seen, ["resume-drop"])
            self.assertIn("speak only this Hindi now", drop.sent[0]["text"])

        asyncio.run(run())

    def test_dropoff_qid_and_stereo_call_audio_recorder(self):
        async def run():
            state = cursor_tests._clear_until(
                cursor_tests._profile(children=[cursor_tests._child(4, "male")]),
                "birth_jaundice",
            )
            cursor = gemini_live._Cursor()
            cursor.state = json.loads(json.dumps(state))
            call_id = db.create_call()
            tel = _new_telemetry()
            for dead_key in ("end_refusals", "end_requested_at", "force_close", "force_close_at", "end_requested", "closing", "close_reason"):
                self.assertNotIn(dead_key, tel)
            turn = gemini_live._TurnState()
            await gemini_live._mark_closing(cursor, "silence", _FakeSocket(), asyncio.Lock(), call_id, tel, turn)
            self.assertEqual(tel["dropoff_qid"], "birth_jaundice")
            self.assertTrue(turn.end_requested)
            self.assertGreater(turn.end_requested_at, 0.0)
            self.assertNotIn("end_requested", tel)
            self.assertNotIn("closing", tel)

            rec = audio_recorder.CallAudioRecorder(call_id, output_dir=str(Path(TEST_DATA_DIR) / "audio"))
            rec._started = time.monotonic() - 0.5
            rec.add_user_pcm(b"\x10\x00" * 160, 16000)
            rec.add_bot_pcm(b"\x20\x00" * 240, 24000)
            abrupt = gemini_live._Cursor()
            abrupt.state = json.loads(json.dumps(state))
            await gemini_live._finalize(_FakeSocket(), asyncio.Lock(), call_id, _new_telemetry(), rec, None, abrupt)
            saved = db.get_call(call_id)
            self.assertEqual(saved["telemetry"]["dropoff_qid"], "birth_jaundice")
            self.assertFalse(any(k.endswith("_json") for k in saved))
            listed = next(row for row in db.list_calls() if row["id"] == call_id)
            self.assertFalse(any(k.endswith("_json") for k in listed))
            self.assertIn("red_flags", listed)
            self.assertIn("clinician_review", listed)
            for detail_only in ("transcript", "telemetry", "events", "cursor_state"):
                self.assertNotIn(detail_only, listed)
            with wave.open(str(Path(TEST_DATA_DIR) / "audio" / saved["audio_path"]), "rb") as wf:
                self.assertEqual(wf.getnchannels(), 2)
                self.assertEqual(wf.getframerate(), 16000)

            done_id = db.create_call()
            finished = gemini_live._Cursor()
            finished.state = cursor_tests._clear_until(
                cursor_tests._profile(children=[cursor_tests._child(4, "male")]),
                "close_anything_else",
            )
            finished.state["phase"] = "closing"
            finished.state["close_reason"] = "complete"
            await gemini_live._finalize(_FakeSocket(), asyncio.Lock(), done_id, _new_telemetry(), rec, None, finished)
            self.assertEqual(db.get_call(done_id)["telemetry"]["dropoff_qid"], "")

        asyncio.run(run())

    def test_watchdog_and_manual_nudge_preserve_active_speak_kind_and_hindi_prompt(self):
        state = cursor_tests._clear_until(
            cursor_tests._profile(children=[cursor_tests._child(4, "male")]),
            "birth_nicu",
        )

        async def run():
            # 1) Watchdog nudge uses active_speak_kind ("clarify")
            wd_cursor = gemini_live._Cursor()
            wd_cursor.state = json.loads(json.dumps(state))
            wd_turn = gemini_live._TurnState()
            wd_turn.awaiting_bot_reply = True
            wd_turn.last_user_turn_ts = time.time() - 5.0
            wd_turn.active_speak_kind = "clarify"
            wd_sess = _FakeSession()
            wd_ws = _FakeSocket()
            wd_stop = asyncio.Event()
            cid = db.create_call()
            with mock.patch.object(gemini_live, "_WATCHDOG_INTERVAL_SEC", 0.01):
                task = asyncio.create_task(
                    gemini_live._watchdog_loop(
                        wd_sess, asyncio.Lock(), wd_ws, asyncio.Lock(), wd_stop, cid,
                        _new_telemetry(), wd_turn, wd_cursor, 0.05,
                    )
                )
                await asyncio.sleep(0.08)
                wd_stop.set()
                await task
            self.assertTrue(wd_sess.sent)
            wd_cue = wd_sess.sent[0][1]["text"]
            self.assertIn("she did not understand", wd_cue)
            self.assertIn("काँच के बक्से", wd_cue)

            # 2) Manual nudge sends speak_cue_text(cursor.state, active_speak_kind) and logs it
            man_cursor = gemini_live._Cursor()
            man_cursor.state = json.loads(json.dumps(state))
            man_turn = gemini_live._TurnState()
            man_turn.active_speak_kind = "clarify"
            man_sess = _FakeSession()
            man_ws = _FakeSocket([{"type": "nudge"}])
            man_cid = db.create_call()
            await gemini_live._browser_loop(
                man_sess, asyncio.Lock(), man_ws, asyncio.Lock(), asyncio.Event(),
                man_cid, _new_telemetry(), man_turn, man_cursor, _Recorder(),
            )
            self.assertEqual(_kinds(man_sess), ["realtime"])
            man_cue = man_sess.sent[0][1]["text"]
            self.assertEqual(man_cue, prompts.speak_cue_text(man_cursor.state, "clarify"))
            self.assertIn("काँच के बक्से", man_cue)
            nudge_events = [e for e in db.get_call(man_cid)["events"] if e["type"] == "manual_nudge"]
            self.assertEqual(len(nudge_events), 1)
            self.assertEqual(nudge_events[0]["detail"], man_cue)

            # 3) _forward_transcript normalizes whitespace cleanly
            tr_ws = _FakeSocket()
            text = await gemini_live._forward_transcript(
                tr_ws, asyncio.Lock(), man_cid, "assistant",
                SimpleNamespace(text="  क्या   बच्चे को   पीलिया   हुआ? \n"),
                man_cursor,
            )
            self.assertEqual(text, "क्या बच्चे को पीलिया हुआ?")

        asyncio.run(run())


class QuickReplyProtocolTests(unittest.TestCase):
    def test_quick_reply_guards_audio_flush_and_stale_captions(self):
        async def run():
            state = cursor_tests._clear_until(
                cursor_tests._profile(children=[cursor_tests._child(4, "male")]),
                "birth_nicu",
            )
            step = step_id(state)

            # 1) Before question is spoken -> ignored
            unspoken_cursor = gemini_live._Cursor()
            unspoken_cursor.state = json.loads(json.dumps(state))
            unspoken_ws = _FakeSocket([{"type": "quick_reply", "outcome": "clear", "step": step}])
            unspoken_sess = _FakeSession()
            await gemini_live._browser_loop(
                unspoken_sess, asyncio.Lock(), unspoken_ws, asyncio.Lock(), asyncio.Event(),
                db.create_call(), _new_telemetry(), gemini_live._TurnState(), unspoken_cursor, _Recorder(),
            )
            self.assertEqual(unspoken_cursor.state["active_qid"], "birth_nicu")
            self.assertEqual(_kinds(unspoken_sess), [])

            # 2) Spoken -> flushes in-progress bot audio & stale captions until turn_complete, idempotent on dupes
            cursor = gemini_live._Cursor()
            cursor.state = json.loads(json.dumps(state))
            turn = gemini_live._TurnState()
            turn.spoken_qid = step
            turn.bot_speaking = True
            turn.playback_ends_at = time.time() + 5
            ws = _FakeSocket([
                {"type": "quick_reply", "outcome": "clear", "step": "stale"},
                {"type": "quick_reply", "outcome": "clear", "step": step},
                {"type": "quick_reply", "outcome": "concern", "step": step},
            ])
            sess = _FakeSession()
            call_id = db.create_call()
            await gemini_live._browser_loop(
                sess, asyncio.Lock(), ws, asyncio.Lock(), asyncio.Event(),
                call_id, _new_telemetry(), turn, cursor, _Recorder(),
            )
            self.assertTrue(turn.ignore_bot_audio)
            self.assertFalse(turn.bot_speaking)
            self.assertEqual(cursor.state["active_qid"], "birth_jaundice")
            self.assertIn({"type": "clear"}, ws.sent)
            self.assertEqual([m["text"] for m in ws.sent if m.get("type") == "transcript"], ["नहीं।"])
            self.assertEqual(_kinds(sess), ["realtime"])

            sent_before = list(ws.sent)
            await gemini_live._on_bot_audio(
                ws, asyncio.Lock(), call_id, _new_telemetry(), turn, _Recorder(), b"\x00\x00" * 24, 24000, cursor
            )
            await gemini_live._handle_server_message(
                _server_message(bot_text="पुराना कटा हुआ सवाल"), sess, asyncio.Lock(), ws, asyncio.Lock(),
                call_id, _new_telemetry(), turn, _Recorder(), set(), set(), cursor, stop=asyncio.Event(),
            )
            self.assertEqual(ws.sent, sent_before)

            await gemini_live._handle_server_message(
                _server_message(turn_complete=True), sess, asyncio.Lock(), ws, asyncio.Lock(),
                call_id, _new_telemetry(), turn, _Recorder(), set(), set(), cursor, stop=asyncio.Event(),
            )
            self.assertFalse(turn.ignore_bot_audio)

            await gemini_live._handle_server_message(
                _server_message(bot_text="क्या जन्म के बाद पीलिया के लिए नीली लाइट में रखना पड़ा था?"),
                sess, asyncio.Lock(), ws, asyncio.Lock(), call_id, _new_telemetry(),
                turn, _Recorder(), set(), set(), cursor, stop=asyncio.Event(),
            )
            await gemini_live._handle_server_message(
                _bot_audio_msg(turn_complete=True),
                sess, asyncio.Lock(), ws, asyncio.Lock(), call_id, _new_telemetry(),
                turn, _Recorder(), set(), set(), cursor, stop=asyncio.Event(),
            )
            self.assertIn({"type": "bot_turn_complete", "step": "birth_jaundice"}, ws.sent)
            assistant_ws = [m for m in ws.sent if m.get("type") == "transcript" and m.get("role") == "assistant"]
            self.assertEqual(len(assistant_ws), 1)
            self.assertIn("नीली लाइट", assistant_ws[0]["text"])

            # 3) quick_reply on close_anything_else closes cleanly
            closing = cursor_tests._clear_until(
                cursor_tests._profile(children=[cursor_tests._child(4, "male")]),
                "close_anything_else",
            )
            closing = apply_turn(closing, "concern", note="पेट दर्द")
            c_cursor = gemini_live._Cursor()
            c_cursor.state = closing
            c_turn = gemini_live._TurnState()
            c_step = step_id(closing)
            c_turn.spoken_qid = c_step
            c_sess = _FakeSession()
            await gemini_live._browser_loop(
                c_sess, asyncio.Lock(), _FakeSocket([{"type": "quick_reply", "outcome": "clear", "step": c_step}]),
                asyncio.Lock(), asyncio.Event(), db.create_call(), _new_telemetry(), c_turn, c_cursor, _Recorder(),
            )
            self.assertEqual(c_cursor.state["phase"], "closing")
            self.assertEqual(c_cursor.state["close_reason"], "complete")
            self.assertIn("धन्यवाद", c_sess.sent[0][1]["text"])

        asyncio.run(run())

    def test_quick_reply_is_model_free_and_holds_trailing_asr_and_late_next_question(self):
        thresholds = _test_thresholds()

        async def run():
            kids = [cursor_tests._child(6, "male", 1), cursor_tests._child(2, "female", 2)]
            state = cursor_tests._clear_until(cursor_tests._profile(children=kids), "birth_nicu")
            step = step_id(state)
            cursor = gemini_live._Cursor()
            cursor.state = state
            turn = gemini_live._TurnState()
            turn.spoken_qid = step
            turn.user_turn_step = step
            turn.capture.user_turn_buffer = "नह"
            turn.capture.user_turn_pcm = bytearray(b"\x00\x10" * 800)
            turn.capture.speech_allowed = False
            session = _FakeSession()
            ws = _FakeSocket([{"type": "quick_reply", "outcome": "clear", "step": step}])
            telemetry = _new_telemetry()
            call_id = db.create_call()

            with mock.patch.object(decisions, "evaluate_turn") as eval_mock:
                await gemini_live._browser_loop(
                    session, asyncio.Lock(), ws, asyncio.Lock(), asyncio.Event(),
                    call_id, telemetry, turn, cursor, _Recorder(),
                )
                eval_mock.assert_not_called()
                self.assertEqual(cursor.state["answers"]["birth_nicu"]["state"], "clear")
                self.assertEqual(cursor.state["active_qid"], "birth_jaundice")
                self.assertEqual(turn.capture.user_turn_buffer, "")
                self.assertEqual(len(turn.capture.user_turn_pcm), 0)
                self.assertTrue(turn.capture.speech_allowed)
                self.assertEqual(telemetry["user_turns"], 1)

                # Trailing ASR within stale window stays pinned & does not corrupt latency or revoke speech_allowed
                started = time.time() - 0.35
                turn.last_user_turn_ts = started
                await gemini_live._handle_server_message(
                    _server_message(user_text="नहीं"), session, asyncio.Lock(), ws, asyncio.Lock(),
                    call_id, telemetry, turn, _Recorder(), set(), set(), cursor, stop=asyncio.Event(),
                )
                self.assertTrue(turn.capture.speech_allowed)
                self.assertEqual(turn.user_turn_step, step)
                self.assertAlmostEqual(turn.last_user_turn_ts, started, delta=0.02)

                # Late next_question held as stale_turn without calling Decision or repeating cue
                await gemini_live._handle_server_message(
                    _server_message(turn_complete=True, tool_calls=[_Call("next_question", {}, "fc-late")]),
                    session, asyncio.Lock(), ws, asyncio.Lock(), call_id, telemetry, turn,
                    _Recorder(), set(), set(), cursor, stop=asyncio.Event(), thresholds=thresholds,
                )
                eval_mock.assert_not_called()
                await _drop_fallback(turn)

            tool_resp = [item[1] for item in session.sent if item[0] == "tool"][0][0]
            self.assertEqual(tool_resp.response["result"], "held")
            self.assertEqual(tool_resp.response["reason"], "stale_turn")
            self.assertEqual(len([item for item in session.sent if item[0] == "realtime"]), 1)
            details = [m["last_event"]["detail"] for m in ws.sent if m.get("type") == "telemetry"]
            self.assertIn("quick_reply_clear", details)
            self.assertIn("next_question_held", details)

            # Fresh answer >= 1.2s after tap on spoken Q2 advances Q2 -> Q3 via Decision
            next_step = step_id(cursor.state)
            turn.spoken_qid = next_step
            turn.ignore_bot_audio = False
            turn.last_quick_reply_ts = time.time() - 1.5

            async def fake_eval(request, **_kwargs):
                raw = {
                    "danger_now": _noul(0.01),
                    "caller_stop": _noul(0.01),
                    "mentions_profile": _noul(0.01),
                    "detail_present": _noul(0.04),
                    "current_outcome": _choice("clear", {"clear": 0.94, "other": 0.04, "concern": 0.02}),
                    "side_intent": _choice("other", {"other": 0.96, "add_concern": 0.04}),
                    "side_qid": _choice("other", {"other": 0.96, "birth_nicu": 0.04}),
                }
                return _decision_result(request, raw)

            with mock.patch.object(decisions, "evaluate_turn", side_effect=fake_eval):
                await gemini_live._handle_server_message(
                    _server_message(user_text="नहीं"), session, asyncio.Lock(), ws, asyncio.Lock(),
                    call_id, telemetry, turn, _Recorder(), set(), set(), cursor, stop=asyncio.Event(),
                )
                await gemini_live._handle_server_message(
                    _server_message(turn_complete=True, tool_calls=[_Call("next_question", {}, "fc-fresh")]),
                    session, asyncio.Lock(), ws, asyncio.Lock(), call_id, telemetry, turn,
                    _Recorder(), set(), set(), cursor, stop=asyncio.Event(), thresholds=thresholds,
                )
                await _drop_fallback(turn)
            self.assertEqual(cursor.state["answers"]["birth_jaundice"]["state"], "clear")
            self.assertEqual(cursor.state["active_qid"], "birth_sudden")

        asyncio.run(run())


class DecisionTurnAndExtractorTests(unittest.TestCase):
    def test_full_intro_flow_name_married_three_children_bridge_to_birth(self):
        thresholds = _test_thresholds()

        async def run():
            cursor = gemini_live._Cursor()
            cursor.state = call_cursor.new_cursor_state()
            turn = gemini_live._TurnState()
            ws = _FakeSocket()
            session = _FakeSession()
            telemetry = _new_telemetry()
            call_id = db.create_call()

            async def fake_eval(request, **_kwargs):
                active_q = json.loads(request["input"])["active_question"]
                is_bridge = "शुरू करें" in active_q
                raw = {
                    "danger_now": _noul(0.01),
                    "caller_stop": _noul(0.01),
                    "mentions_profile": _noul(0.02 if is_bridge else 0.92),
                    "current_outcome": _choice(
                        "concern" if is_bridge else "other",
                        {"concern": 0.93, "other": 0.05, "clear": 0.02}
                        if is_bridge
                        else {"other": 0.90, "clear": 0.06, "concern": 0.04},
                    ),
                }
                return _decision_result(request, raw)

            async def fake_extract(state, utterance_text="", audio_wav_bytes=None, **_kwargs):
                qid = state.get("active_qid")
                if qid == "intro_name":
                    updates = {"caller_name": "पूजा"}
                elif qid == "intro_married":
                    updates = {"marital_status": "married"}
                elif qid == "intro_children":
                    updates = {
                        "child_count": 3,
                        "children": [
                            {"age_years": 11.0, "sex": "female", "name": "रिया"},
                            {"age_years": 6.0, "sex": "male", "name": None},
                            {"age_years": 0.5, "sex": "male", "name": None},
                        ],
                    }
                else:
                    updates = {}
                return profile_extractor.ProfileExtractionResult(
                    ok=True, updates=updates, error_reason="", extractor_ms=24
                )

            turns = [
                ("मेरा नाम पूजा है", "fc-intro-1"),
                ("हाँ, शादी हो गई है", "fc-intro-2"),
                ("तीन बच्चे हैं — 11 साल की बड़ी बेटी रिया, 6 साल का बेटा, और 6 महीने का छोटा बेटा", "fc-intro-3"),
                ("हाँ, शुरू करें", "fc-intro-4"),
            ]
            with mock.patch.object(decisions, "evaluate_turn", side_effect=fake_eval), mock.patch.object(
                profile_extractor, "extract_profile_entities", side_effect=fake_extract
            ) as ext_mock:
                for user_text, call_sid in turns:
                    turn.spoken_qid = step_id(cursor.state)
                    await gemini_live._handle_server_message(
                        _server_message(user_text=user_text), session, asyncio.Lock(), ws, asyncio.Lock(),
                        call_id, telemetry, turn, _Recorder(), set(), set(), cursor, stop=asyncio.Event(),
                    )
                    await gemini_live._handle_server_message(
                        _server_message(turn_complete=True, tool_calls=[_Call("next_question", {}, call_sid)]),
                        session, asyncio.Lock(), ws, asyncio.Lock(), call_id, telemetry, turn,
                        _Recorder(), set(), set(), cursor, stop=asyncio.Event(), thresholds=thresholds,
                    )
                    await _drop_fallback(turn)
                self.assertEqual(ext_mock.await_count, 3)

            self.assertEqual(cursor.state["phase"], "asking")
            self.assertEqual(cursor.state["active_qid"], "birth_mode_detail")
            self.assertEqual(cursor.state["profile"]["caller_name"], "पूजा")
            self.assertEqual(cursor.state["profile"]["marital_status"], "married")
            self.assertEqual(len(cursor.state["profile"]["children"]), 3)
            self.assertEqual(cursor.state["profile"]["children"][2]["age_label_hi"], "6 महीने")
            self.assertEqual(cursor.state["answers"]["intro_bridge"]["state"], "clear")

            dt_events = [e for e in db.get_call(call_id)["events"] if e["type"] == "decision_turn"]
            self.assertEqual(len(dt_events), 4)
            self.assertTrue(dt_events[0]["meta"]["extractor_ok"])
            self.assertEqual(dt_events[0]["meta"]["extractor_ms"], 24)
            self.assertNotIn("extractor_ok", dt_events[3]["meta"])

        asyncio.run(run())

    def test_dual_intent_smoke_a_clarify_rewind_future_qid_and_multichild_followup(self):
        thresholds = _test_thresholds()

        async def run():
            # 1) Smoke A: clear current + add earlier fam_cancer concern
            state = cursor_tests._clear_until(
                cursor_tests._profile(children=[cursor_tests._child(4, "male")]),
                "fam_cancer",
            )
            state = apply_turn(state, "concern", note="मामा को ब्लड कैंसर")
            state = apply_turn(state, "concern", note="मामा")
            state = cursor_tests._clear_until(state, "fam_heart_kidney")

            cursor = gemini_live._Cursor()
            cursor.state = state
            turn = gemini_live._TurnState()
            turn.spoken_qid = step_id(cursor.state)
            ws = _FakeSocket()
            session = _FakeSession()
            telemetry = _new_telemetry()
            call_id = db.create_call()

            async def eval_smoke_a(request, **_kwargs):
                raw = {
                    "danger_now": _noul(0.02),
                    "caller_stop": _noul(0.01),
                    "mentions_profile": _noul(0.03),
                    "detail_present": _noul(0.05),
                    "current_outcome": _choice("clear", {"clear": 0.92, "other": 0.04, "concern": 0.03}),
                    "side_intent": _choice("add_concern", {"add_concern": 0.93, "other": 0.05}),
                    "side_qid": _choice("fam_cancer", {"fam_cancer": 0.91, "other": 0.04, "fam_lumps": 0.03}),
                }
                return _decision_result(request, raw)

            with mock.patch.object(decisions, "evaluate_turn", side_effect=eval_smoke_a):
                await gemini_live._handle_server_message(
                    _server_message(user_text="नहीं, पर चाची को भी कैंसर हुआ था"),
                    session, asyncio.Lock(), ws, asyncio.Lock(), call_id, telemetry,
                    turn, _Recorder(), set(), set(), cursor, stop=asyncio.Event(),
                )
                await gemini_live._handle_server_message(
                    _server_message(
                        turn_complete=True,
                        tool_calls=[_Call("next_question", {}, "fc-1"), _Call("next_question", {}, "fc-dup")],
                    ),
                    session, asyncio.Lock(), ws, asyncio.Lock(), call_id, telemetry,
                    turn, _Recorder(), set(), set(), cursor, stop=asyncio.Event(), thresholds=thresholds,
                )
                await _drop_fallback(turn)

            self.assertEqual(cursor.state["answers"]["fam_heart_kidney"]["state"], "clear")
            self.assertIn("चाची को भी कैंसर हुआ था", cursor.state["answers"]["fam_cancer"]["notes"])
            self.assertEqual(cursor.state["active_qid"], "fam_bp_stroke")
            tool_resps = [item[1] for item in session.sent if item[0] == "tool"][0]
            self.assertEqual(tool_resps[0].response["result"], "ok")
            self.assertEqual(tool_resps[1].response["result"], "held")
            self.assertEqual(tool_resps[1].response["reason"], "duplicate_turn")

            # 2) clarify_meaning rewinds to preg_thalassemia with clarify=True
            thal = cursor_tests._clear_until(
                cursor_tests._profile(children=[cursor_tests._child(4, "male")]),
                "preg_thalassemia",
            )
            cursor.state = apply_turn(thal, "unknown")
            turn.spoken_qid = step_id(cursor.state)

            async def eval_clarify(request, **_kwargs):
                raw = {
                    "danger_now": _noul(0.01),
                    "caller_stop": _noul(0.01),
                    "mentions_profile": _noul(0.01),
                    "detail_present": _noul(0.05),
                    "current_outcome": _choice("clear", {"clear": 0.91, "other": 0.05, "concern": 0.04}),
                    "side_intent": _choice("clarify_other", {"clarify_other": 0.91, "add_concern": 0.05, "other": 0.04}),
                    "side_qid": _choice("preg_thalassemia", {"preg_thalassemia": 0.90, "other": 0.05, "fam_related": 0.05}),
                }
                return _decision_result(request, raw)

            with mock.patch.object(decisions, "evaluate_turn", side_effect=eval_clarify):
                await gemini_live._handle_server_message(
                    _server_message(user_text="रिश्तेदार तो नहीं, पर थैलेसीमिया क्या होता है?"),
                    session, asyncio.Lock(), ws, asyncio.Lock(), call_id, telemetry,
                    turn, _Recorder(), set(), set(), cursor, stop=asyncio.Event(),
                )
                await gemini_live._handle_server_message(
                    _server_message(turn_complete=True, tool_calls=[_Call("next_question", {}, "fc-clar")]),
                    session, asyncio.Lock(), ws, asyncio.Lock(), call_id, telemetry,
                    turn, _Recorder(), set(), set(), cursor, stop=asyncio.Event(), thresholds=thresholds,
                )
                await _drop_fallback(turn)
            self.assertEqual(cursor.state["active_qid"], "preg_thalassemia")
            self.assertTrue([item[1] for item in session.sent if item[0] == "tool"][-1][0].response.get("clarify"))

            # 3) future_qid_match logged on birth_jaundice without writing future fam_cancer
            cursor.state = cursor_tests._clear_until(
                cursor_tests._profile(children=[cursor_tests._child(4, "male")]),
                "birth_jaundice",
            )
            turn.spoken_qid = step_id(cursor.state)

            async def eval_future(request, **_kwargs):
                raw = {
                    "danger_now": _noul(0.01),
                    "caller_stop": _noul(0.01),
                    "mentions_profile": _noul(0.01),
                    "detail_present": _noul(0.05),
                    "current_outcome": _choice("clear", {"clear": 0.90, "other": 0.06, "concern": 0.04}),
                    "side_intent": _choice("add_concern", {"add_concern": 0.91, "other": 0.06, "clarify_other": 0.03}),
                    "side_qid": _choice("fam_cancer", {"fam_cancer": 0.93, "other": 0.04, "fam_lumps": 0.03}),
                }
                return _decision_result(request, raw)

            with mock.patch.object(decisions, "evaluate_turn", side_effect=eval_future):
                await gemini_live._handle_server_message(
                    _server_message(user_text="नहीं, हाँ चाची को कैंसर था"),
                    session, asyncio.Lock(), ws, asyncio.Lock(), call_id, telemetry,
                    turn, _Recorder(), set(), set(), cursor, stop=asyncio.Event(),
                )
                await gemini_live._handle_server_message(
                    _server_message(turn_complete=True, tool_calls=[_Call("next_question", {}, "fc-fut")]),
                    session, asyncio.Lock(), ws, asyncio.Lock(), call_id, telemetry,
                    turn, _Recorder(), set(), set(), cursor, stop=asyncio.Event(), thresholds=thresholds,
                )
                await _drop_fallback(turn)
            dt_events = [e for e in db.get_call(call_id)["events"] if e["type"] == "decision_turn"]
            self.assertEqual(
                dt_events[-1]["meta"]["future_qid_match"],
                {"qid": "fam_cancer", "confidence": 0.93, "act": "add_or_correct"},
            )
            self.assertNotIn("fam_cancer", cursor.state["answers"])

            # 4) Multi-child child_ref + pre-answered followup resolves birth_jaundice in 1 turn
            kids = [
                cursor_tests._child(9, "female", 1, name="रिया"),
                cursor_tests._child(5, "male", 2, name="अर्जुन"),
            ]
            cursor.state = cursor_tests._clear_until(cursor_tests._profile(children=kids), "birth_jaundice")
            turn.spoken_qid = step_id(cursor.state)

            async def eval_multi(request, **_kwargs):
                raw = {
                    "danger_now": _noul(0.01),
                    "caller_stop": _noul(0.01),
                    "mentions_profile": _noul(0.01),
                    "detail_present": _noul(0.92),
                    "current_outcome": _choice("concern", {"concern": 0.95, "other": 0.03, "clear": 0.02}),
                    "child_ref": _choice("child_2", {"child_2": 0.91, "child_1": 0.04, "other": 0.03, "all": 0.02}),
                    "side_intent": _choice("other", {"other": 0.96, "add_concern": 0.04}),
                    "side_qid": _choice("other", {"other": 0.96, "birth_nicu": 0.04}),
                }
                return _decision_result(request, raw)

            with mock.patch.object(decisions, "evaluate_turn", side_effect=eval_multi):
                await gemini_live._handle_server_message(
                    _server_message(user_text="हाँ छोटे बेटे को चार दिन नीली लाइट में रखा था"),
                    session, asyncio.Lock(), ws, asyncio.Lock(), call_id, telemetry,
                    turn, _Recorder(), set(), set(), cursor, stop=asyncio.Event(),
                )
                await gemini_live._handle_server_message(
                    _server_message(turn_complete=True, tool_calls=[_Call("next_question", {}, "fc-multi")]),
                    session, asyncio.Lock(), ws, asyncio.Lock(), call_id, telemetry,
                    turn, _Recorder(), set(), set(), cursor, stop=asyncio.Event(), thresholds=thresholds,
                )
                await _drop_fallback(turn)
            self.assertEqual(cursor.state["answers"]["birth_jaundice"]["child_ids"], ["child_2"])
            self.assertIn("चार दिन", cursor.state["answers"]["birth_jaundice"]["followups"]["jaundice_days"]["note"])
            self.assertEqual(cursor.state["active_qid"], "birth_sudden")

        asyncio.run(run())

    def test_missed_tool_recovery_pcm_wav_fallback_midflight_and_holds(self):
        thresholds = _test_thresholds()

        async def run():
            state = cursor_tests._clear_until(
                cursor_tests._profile(children=[cursor_tests._child(4, "male")]),
                "birth_jaundice",
            )
            cursor = gemini_live._Cursor()
            cursor.state = json.loads(json.dumps(state))
            turn = gemini_live._TurnState()
            turn.spoken_qid = step_id(cursor.state)
            ws = _FakeSocket()
            session = _FakeSession()
            telemetry = _new_telemetry()
            call_id = db.create_call()

            async def eval_clear(request, **_kwargs):
                raw = {
                    "danger_now": _noul(0.01),
                    "caller_stop": _noul(0.01),
                    "mentions_profile": _noul(0.01),
                    "detail_present": _noul(0.03),
                    "current_outcome": _choice("clear", {"clear": 0.94, "other": 0.04, "concern": 0.02}),
                    "side_intent": _choice("other", {"other": 0.96, "add_concern": 0.04}),
                    "side_qid": _choice("other", {"other": 0.96, "birth_nicu": 0.04}),
                }
                return _decision_result(request, raw)

            # 1) Unprompted bot audio + caption suppressed, then missed-tool recovery at turn_complete
            with mock.patch.object(decisions, "evaluate_turn", side_effect=eval_clear) as eval_mock:
                await gemini_live._handle_server_message(
                    _server_message(user_text="नहीं"), session, asyncio.Lock(), ws, asyncio.Lock(),
                    call_id, telemetry, turn, _Recorder(), set(), set(), cursor, stop=asyncio.Event(),
                )
                self.assertFalse(turn.capture.speech_allowed)
                await gemini_live._handle_server_message(
                    _server_message(bot_text="मैं अपने मन से बोल रही हूँ"), session, asyncio.Lock(), ws, asyncio.Lock(),
                    call_id, telemetry, turn, _Recorder(), set(), set(), cursor, stop=asyncio.Event(),
                )
                await gemini_live._handle_server_message(
                    _bot_audio_msg(turn_complete=False), session, asyncio.Lock(), ws, asyncio.Lock(),
                    call_id, telemetry, turn, _Recorder(), set(), set(), cursor, stop=asyncio.Event(),
                )
                self.assertEqual(telemetry["audio_chunks_out"], 0)
                self.assertFalse(any(m.get("type") == "audio" for m in ws.sent))
                self.assertFalse(any(m.get("type") == "transcript" and m.get("role") == "assistant" for m in ws.sent))

                await gemini_live._handle_server_message(
                    _server_message(turn_complete=True), session, asyncio.Lock(), ws, asyncio.Lock(),
                    call_id, telemetry, turn, _Recorder(), set(), set(), cursor, stop=asyncio.Event(), thresholds=thresholds,
                )
                self.assertEqual(eval_mock.await_count, 1)
                self.assertEqual(cursor.state["active_qid"], "birth_sudden")
                self.assertTrue(turn.capture.speech_allowed)
                self.assertEqual(telemetry["tool_calls"], 1)
                self.assertTrue(
                    any(
                        e["type"] == "tool_call" and e["detail"] == "missed_tool_clear"
                        for e in db.get_call(call_id)["events"]
                    )
                )
                self.assertTrue(
                    any(
                        m.get("type") == "telemetry"
                        and (m.get("last_event") or {}).get("type") == "tool_call"
                        and (m.get("last_event") or {}).get("detail") == "missed_tool_clear"
                        for m in ws.sent
                    )
                )
                await gemini_live._handle_server_message(
                    _bot_audio_msg(turn_complete=True), session, asyncio.Lock(), ws, asyncio.Lock(),
                    call_id, telemetry, turn, _Recorder(), set(), set(), cursor, stop=asyncio.Event(),
                )
                self.assertEqual(turn.spoken_qid, "birth_sudden")
                self.assertEqual(telemetry["audio_chunks_out"], 1)

            # 2) Buffered 16kHz PCM audio/wav fallback when ASR is empty + 960_000 cap
            silent_chunk = base64.b64encode(b"\x00\x00" * 1600).decode("ascii")
            speech_chunk = base64.b64encode(b"\x00\x10" * 1600).decode("ascii")
            turn.capture.user_turn_buffer = ""
            turn.capture.user_turn_pcm.clear()
            await gemini_live._browser_loop(
                session, asyncio.Lock(),
                _FakeSocket([
                    {"type": "audio", "data": silent_chunk, "sample_rate": 16000},
                    {"type": "audio", "data": speech_chunk, "sample_rate": 16000},
                    {"type": "audio", "data": silent_chunk, "sample_rate": 16000},
                ]),
                asyncio.Lock(), asyncio.Event(), call_id, telemetry, turn, cursor, _Recorder(),
            )
            self.assertEqual(len(turn.capture.user_turn_pcm), 2 * 3200)
            expected_pcm = bytes(turn.capture.user_turn_pcm)
            seen_reqs = []

            async def eval_capture(request, **_kwargs):
                seen_reqs.append(request)
                return await eval_clear(request)

            with mock.patch.object(decisions, "evaluate_turn", side_effect=eval_capture):
                await gemini_live._answer_function_calls(
                    session, asyncio.Lock(), _FakeSocket(), asyncio.Lock(), call_id, telemetry,
                    [_Call("next_question", {}, "fc-wav")], set(), set(), cursor, turn, thresholds=thresholds,
                )
                await _drop_fallback(turn)
            req_in = seen_reqs[0]["input"]
            self.assertIsInstance(req_in, list)
            self.assertEqual(req_in[1]["mime_type"], "audio/wav")
            with wave.open(io.BytesIO(base64.b64decode(req_in[1]["data"])), "rb") as wf:
                self.assertEqual(wf.getframerate(), 16000)
                self.assertEqual(wf.readframes(wf.getnframes()), expected_pcm)
            self.assertEqual(len(turn.capture.user_turn_pcm), 0)

            cap_turn = gemini_live._TurnState()
            big_loud = base64.b64encode(b"\x00\x10" * 160_000).decode("ascii")
            await gemini_live._browser_loop(
                _FakeSession(), asyncio.Lock(),
                _FakeSocket([{"type": "audio", "data": big_loud, "sample_rate": 16000} for _ in range(4)]),
                asyncio.Lock(), asyncio.Event(), call_id, telemetry, cap_turn, cursor, _Recorder(),
            )
            self.assertEqual(len(cap_turn.capture.user_turn_pcm), 960_000)

            # 3) Mid-flight buffer preservation across await evaluate_turn
            turn.spoken_qid = step_id(cursor.state)
            turn.user_turn_step = turn.spoken_qid
            turn.capture.handled_turn = False
            turn.capture.last_key = None
            turn.capture.user_turn_buffer = "नहीं"
            turn.capture.user_turn_pcm = bytearray(b"\x00\x10" * 400)
            eval_started = asyncio.Event()
            release_eval = asyncio.Event()

            async def slow_eval(request, **_kwargs):
                eval_started.set()
                await release_eval.wait()
                return await eval_clear(request)

            with mock.patch.object(decisions, "evaluate_turn", side_effect=slow_eval):
                task = asyncio.create_task(
                    gemini_live._handle_server_message(
                        _server_message(turn_complete=True, tool_calls=[_Call("next_question", {}, "fc-race")]),
                        session, asyncio.Lock(), ws, asyncio.Lock(), call_id, telemetry,
                        turn, _Recorder(), set(), set(), cursor, stop=asyncio.Event(), thresholds=thresholds,
                    )
                )
                await asyncio.wait_for(eval_started.wait(), timeout=1.0)
                self.assertEqual(turn.capture.user_turn_buffer, "")
                turn.capture.user_turn_buffer = "अगले सवाल का जवाब"
                turn.capture.user_turn_pcm = bytearray(b"\x01\x00" * 320)
                release_eval.set()
                await task
                await _drop_fallback(turn)
            self.assertEqual(turn.capture.user_turn_buffer, "अगले सवाल का जवाब")
            self.assertEqual(bytes(turn.capture.user_turn_pcm), b"\x01\x00" * 320)

            # 4) Empty-turn and not-spoken holds do not call evaluate_turn
            turn.capture.user_turn_buffer = ""
            turn.capture.user_turn_pcm.clear()
            turn.capture.handled_turn = False
            turn.capture.last_key = None
            turn.spoken_qid = step_id(cursor.state)
            turn.user_turn_step = turn.spoken_qid
            hold_sess = _FakeSession()
            with mock.patch.object(decisions, "evaluate_turn") as eval_mock:
                await gemini_live._answer_function_calls(
                    hold_sess, asyncio.Lock(), _FakeSocket(), asyncio.Lock(), call_id, telemetry,
                    [_Call("next_question", {}, "fc-empty")], set(), set(), cursor, turn, thresholds=thresholds,
                )
                await _drop_fallback(turn)
                eval_mock.assert_not_called()
                self.assertEqual(hold_sess.sent[0][1][0].response["reason"], "empty_turn")

                turn.spoken_qid = "other_qid"
                await gemini_live._answer_function_calls(
                    hold_sess, asyncio.Lock(), _FakeSocket(), asyncio.Lock(), call_id, telemetry,
                    [_Call("next_question", {}, "fc-unspoken")], set(), set(), cursor, turn, thresholds=thresholds,
                )
                await _drop_fallback(turn)
                eval_mock.assert_not_called()
                self.assertEqual(hold_sess.sent[-1][1][0].response["reason"], "not_spoken")

        asyncio.run(run())

    def test_emergency_vs_history_decision_failure_and_extractor_failures(self):
        thresholds = _test_thresholds()

        async def run():
            # 1) Historical body_blue records concern WITHOUT emergency; active danger_now closes with 108
            hist_cursor = gemini_live._Cursor()
            hist_cursor.state = cursor_tests._clear_until(
                cursor_tests._profile(children=[cursor_tests._child(4, "male")], name="पूजा"),
                "body_blue",
            )
            hist_turn = gemini_live._TurnState()
            hist_turn.spoken_qid = step_id(hist_cursor.state)
            hist_sess = _FakeSession()

            async def eval_hist(request, **_kwargs):
                return _decision_result(request, {
                    "danger_now": _noul(0.08),
                    "caller_stop": _noul(0.01),
                    "mentions_profile": _noul(0.01),
                    "detail_present": _noul(0.10),
                    "current_outcome": _choice("concern", {"concern": 0.92, "other": 0.05, "clear": 0.03}),
                    "side_intent": _choice("other", {"other": 0.96, "add_concern": 0.04}),
                    "side_qid": _choice("other", {"other": 0.96, "body_fits": 0.04}),
                })

            with mock.patch.object(decisions, "evaluate_turn", side_effect=eval_hist):
                await gemini_live._handle_server_message(
                    _server_message(user_text="हाँ, रोते समय कभी नीली पड़ती हैं"),
                    hist_sess, asyncio.Lock(), _FakeSocket(), asyncio.Lock(), db.create_call(),
                    _new_telemetry(), hist_turn, _Recorder(), set(), set(), hist_cursor, stop=asyncio.Event(),
                )
                await gemini_live._handle_server_message(
                    _server_message(turn_complete=True, tool_calls=[_Call("next_question", {}, "fc-hist")]),
                    hist_sess, asyncio.Lock(), _FakeSocket(), asyncio.Lock(), db.create_call(),
                    _new_telemetry(), hist_turn, _Recorder(), set(), set(), hist_cursor, stop=asyncio.Event(), thresholds=thresholds,
                )
                await _drop_fallback(hist_turn)
            self.assertEqual(hist_cursor.state["phase"], "asking")
            self.assertEqual(hist_cursor.state["answers"]["body_blue"]["state"], "concern")

            emg_cursor = gemini_live._Cursor()
            emg_cursor.state = cursor_tests._clear_until(
                cursor_tests._profile(children=[cursor_tests._child(4, "male")], name="पूजा"),
                "body_blue",
            )
            emg_turn = gemini_live._TurnState()
            emg_turn.spoken_qid = step_id(emg_cursor.state)
            emg_sess = _FakeSession()
            emg_tel = _new_telemetry()
            emg_call = db.create_call()

            async def eval_emg(request, **_kwargs):
                return _decision_result(request, {
                    "danger_now": _noul(0.96),
                    "caller_stop": _noul(0.01),
                    "mentions_profile": _noul(0.01),
                    "detail_present": _noul(0.85),
                    "current_outcome": _choice("concern", {"concern": 0.90, "other": 0.06, "clear": 0.04}),
                    "side_intent": _choice("other", {"other": 0.96, "add_concern": 0.04}),
                    "side_qid": _choice("other", {"other": 0.96, "body_fits": 0.04}),
                })

            with mock.patch.object(decisions, "evaluate_turn", side_effect=eval_emg):
                await gemini_live._handle_server_message(
                    _server_message(user_text="बच्चे को अभी साँस नहीं आ रही और होंठ नीले हैं"),
                    emg_sess, asyncio.Lock(), _FakeSocket(), asyncio.Lock(), emg_call,
                    emg_tel, emg_turn, _Recorder(), set(), set(), emg_cursor, stop=asyncio.Event(),
                )
                await gemini_live._handle_server_message(
                    _server_message(turn_complete=True, tool_calls=[_Call("next_question", {}, "fc-emg")]),
                    emg_sess, asyncio.Lock(), _FakeSocket(), asyncio.Lock(), emg_call,
                    emg_tel, emg_turn, _Recorder(), set(), set(), emg_cursor, stop=asyncio.Event(), thresholds=thresholds,
                )
                await _drop_fallback(emg_turn)
            self.assertEqual(emg_cursor.state["phase"], "closing")
            self.assertEqual(emg_cursor.state["close_reason"], "emergency")
            tool_resp = [item[1] for item in emg_sess.sent if item[0] == "tool"][0][0]
            self.assertTrue(tool_resp.response["stop"])
            self.assertIn("108", tool_resp.response["say"])
            self.assertTrue(any(e["type"] == "end_accepted" and e["detail"] == "emergency" for e in db.get_call(emg_call)["events"]))

            # 2) Decision failure -> calm technical_failure closing, no 108, answers untouched
            fail_cursor = gemini_live._Cursor()
            f_state = cursor_tests._clear_until(
                cursor_tests._profile(children=[cursor_tests._child(4, "male")], name="पूजा"),
                "birth_nicu",
            )
            f_state = apply_turn(f_state, "concern", note="सांस", followup_answered=True)
            f_state = apply_turn(f_state, "concern", note="तीन दिन")
            fail_cursor.state = f_state
            fail_turn = gemini_live._TurnState()
            fail_turn.spoken_qid = step_id(fail_cursor.state)
            fail_sess = _FakeSession()
            fail_tel = _new_telemetry()
            fail_call = db.create_call()

            async def eval_fail(request, **_kwargs):
                return _decision_result(request, {}, ok=False, error_reason="timeout", attempts=2)

            with mock.patch.object(decisions, "evaluate_turn", side_effect=eval_fail):
                await gemini_live._handle_server_message(
                    _server_message(user_text="नहीं"), fail_sess, asyncio.Lock(), _FakeSocket(), asyncio.Lock(),
                    fail_call, fail_tel, fail_turn, _Recorder(), set(), set(), fail_cursor, stop=asyncio.Event(),
                )
                await gemini_live._handle_server_message(
                    _server_message(turn_complete=True, tool_calls=[_Call("next_question", {}, "fc-fail")]),
                    fail_sess, asyncio.Lock(), _FakeSocket(), asyncio.Lock(),
                    fail_call, fail_tel, fail_turn, _Recorder(), set(), set(), fail_cursor, stop=asyncio.Event(), thresholds=thresholds,
                )
                await _drop_fallback(fail_turn)

            self.assertEqual(fail_cursor.state["phase"], "closing")
            self.assertEqual(fail_cursor.state["close_reason"], "technical_failure")
            self.assertNotIn("birth_jaundice", fail_cursor.state["answers"])
            self.assertEqual(fail_cursor.state["answers"]["birth_nicu"]["state"], "concern")
            self.assertEqual(fail_tel["dropoff_qid"], "birth_jaundice")
            fail_line = [item[1] for item in fail_sess.sent if item[0] == "tool"][0][0].response["say"]
            self.assertIn("पूजा जी", fail_line)
            self.assertIn("अधूरी", fail_line)
            self.assertNotIn("108", fail_line)
            self.assertNotIn("अस्पताल", fail_line)
            self.assertEqual(
                close_technical_failure(
                    cursor_tests._clear_until(
                        cursor_tests._profile(children=[cursor_tests._child(4, "male")]),
                        "birth_jaundice",
                    )
                )["close_reason"],
                "technical_failure",
            )

            # 3) Profile extractor: 1st failure on intro re-asks, 2nd closes technical_failure;
            #    clinical gate 1st failure advances & resets; clarify skips extractor; ever_pregnant sync
            ext_cursor = gemini_live._Cursor()
            ext_cursor.state = call_cursor.new_cursor_state()
            ext_turn = gemini_live._TurnState()
            ext_sess = _FakeSession()
            ext_tel = _new_telemetry()
            ext_call = db.create_call()

            async def eval_profile(request, **_kwargs):
                return _decision_result(request, {
                    "danger_now": _noul(0.01),
                    "caller_stop": _noul(0.01),
                    "mentions_profile": _noul(0.92),
                    "current_outcome": _choice("other", {"other": 0.91, "clear": 0.09}),
                })

            async def fail_extract(*_args, **_kwargs):
                return profile_extractor.ProfileExtractionResult(ok=False, updates={}, error_reason="timeout", extractor_ms=2000)

            with mock.patch.object(decisions, "evaluate_turn", side_effect=eval_profile), mock.patch.object(
                profile_extractor, "extract_profile_entities", side_effect=fail_extract
            ):
                for idx, utt in enumerate(("मेरा नाम रेखा है", "रेखा"), start=1):
                    ext_turn.spoken_qid = step_id(ext_cursor.state)
                    await gemini_live._handle_server_message(
                        _server_message(user_text=utt), ext_sess, asyncio.Lock(), _FakeSocket(), asyncio.Lock(),
                        ext_call, ext_tel, ext_turn, _Recorder(), set(), set(), ext_cursor, stop=asyncio.Event(),
                    )
                    await gemini_live._handle_server_message(
                        _server_message(turn_complete=True, tool_calls=[_Call("next_question", {}, f"fc-ext-{idx}")]),
                        ext_sess, asyncio.Lock(), _FakeSocket(), asyncio.Lock(), ext_call, ext_tel,
                        ext_turn, _Recorder(), set(), set(), ext_cursor, stop=asyncio.Event(), thresholds=thresholds,
                    )
                    await _drop_fallback(ext_turn)
                    if idx == 1:
                        self.assertEqual(ext_cursor.state["phase"], "asking")
                        self.assertEqual(ext_cursor.state["active_qid"], "intro_name")
                        self.assertEqual(ext_turn.capture.extractor_failures, 1)

            self.assertEqual(ext_cursor.state["phase"], "closing")
            self.assertEqual(ext_cursor.state["close_reason"], "technical_failure")

            # Clinical gate 1st extractor failure advances gate and resets failure counter
            gate_cursor = gemini_live._Cursor()
            gate_cursor.state = cursor_tests._clear_until(
                cursor_tests._profile(children=[cursor_tests._child(4, "male")]),
                "birth_nicu",
            )
            gate_turn = gemini_live._TurnState()
            gate_turn.capture.user_turn_buffer = "नहीं"

            async def eval_gate_profile(request, **_kwargs):
                return _decision_result(request, {
                    "danger_now": _noul(0.01),
                    "caller_stop": _noul(0.01),
                    "mentions_profile": _noul(0.85),
                    "detail_present": _noul(0.05),
                    "current_outcome": _choice("clear", {"clear": 0.93, "other": 0.04, "concern": 0.03}),
                    "side_intent": _choice("other", {"other": 0.96, "add_concern": 0.04}),
                    "side_qid": _choice("other", {"other": 0.96, "birth_jaundice": 0.04}),
                })

            with mock.patch.object(decisions, "evaluate_turn", side_effect=eval_gate_profile), mock.patch.object(
                profile_extractor, "extract_profile_entities", side_effect=fail_extract
            ):
                await live_decision_bridge.run_decision_turn(
                    gate_cursor, gate_turn.capture, db.create_call(),
                    tool_body_fn=gemini_live._tool_body, mark_closing_fn=lambda _r: asyncio.sleep(0),
                    thresholds=thresholds,
                )
            self.assertEqual(gate_cursor.state["answers"]["birth_nicu"]["state"], "clear")
            self.assertEqual(gate_cursor.state["active_qid"], "birth_jaundice")
            self.assertEqual(gate_turn.capture.extractor_failures, 0)

            # intro clarify skips extractor; empty updates on intro_name still triggers hold via apply_interpreted_turn
            clar_cursor = gemini_live._Cursor()
            clar_cursor.state = call_cursor.new_cursor_state()
            clar_turn = gemini_live._TurnState()
            clar_turn.capture.user_turn_buffer = "क्या कहा आपने?"

            async def eval_clar(request, **_kwargs):
                return _decision_result(request, {
                    "danger_now": _noul(0.01),
                    "caller_stop": _noul(0.01),
                    "mentions_profile": _noul(0.05),
                    "current_outcome": _choice("clarify", {"clarify": 0.92, "other": 0.05, "clear": 0.03}),
                })

            with mock.patch.object(decisions, "evaluate_turn", side_effect=eval_clar), mock.patch.object(
                profile_extractor, "extract_profile_entities"
            ) as ext_mock:
                body, clar_flag, _ = await live_decision_bridge.run_decision_turn(
                    clar_cursor, clar_turn.capture, db.create_call(),
                    tool_body_fn=gemini_live._tool_body, mark_closing_fn=lambda _r: asyncio.sleep(0),
                    thresholds=thresholds,
                )
                ext_mock.assert_not_awaited()
            self.assertTrue(clar_flag)
            self.assertTrue(body.get("clarify"))
            self.assertIsNone(clar_cursor.state["hold_reason"])

            # Empty extractor updates on intro_name holds with name_fragment via apply_interpreted_turn
            frag_cursor = gemini_live._Cursor()
            frag_cursor.state = call_cursor.new_cursor_state()
            frag_turn = gemini_live._TurnState()
            frag_turn.capture.user_turn_buffer = "मेरा नाम"

            async def empty_extract(*_args, **_kwargs):
                return profile_extractor.ProfileExtractionResult(
                    ok=True, updates={}, error_reason="", extractor_ms=10
                )

            with mock.patch.object(decisions, "evaluate_turn", side_effect=eval_profile), mock.patch.object(
                profile_extractor, "extract_profile_entities", side_effect=empty_extract
            ):
                frag_body, _, _ = await live_decision_bridge.run_decision_turn(
                    frag_cursor, frag_turn.capture, db.create_call(),
                    tool_body_fn=gemini_live._tool_body, mark_closing_fn=lambda _r: asyncio.sleep(0),
                    thresholds=thresholds,
                )
            self.assertEqual(frag_cursor.state["hold_reason"], "name_fragment")
            self.assertIn("पूरा नाम", frag_body["say"])

        asyncio.run(run())

    def test_audio_only_intro_name_speech_reaches_extractor_and_advances_to_married(self):
        thresholds = _test_thresholds()

        async def run():
            cursor = gemini_live._Cursor()
            cursor.state = call_cursor.new_cursor_state()
            self.assertEqual(cursor.state["active_qid"], "intro_name")
            turn = gemini_live._TurnState()
            turn.spoken_qid = step_id(cursor.state)
            ws = _FakeSocket()
            session = _FakeSession()
            telemetry = _new_telemetry()
            call_id = db.create_call()

            loud_pcm = b"\x00\x10" * 1600
            quiet_pcm = b"\x00\x00" * 1600
            browser_ws = _FakeSocket([
                {"type": "audio", "data": base64.b64encode(loud_pcm).decode("ascii"), "sample_rate": 16000},
                {"type": "audio", "data": base64.b64encode(quiet_pcm).decode("ascii"), "sample_rate": 16000},
            ])
            await gemini_live._browser_loop(
                session, asyncio.Lock(), browser_ws, asyncio.Lock(), asyncio.Event(),
                call_id, telemetry, turn, cursor, _Recorder(),
            )
            self.assertEqual(turn.capture.user_turn_buffer, "")
            expected_pcm = loud_pcm + quiet_pcm
            self.assertEqual(bytes(turn.capture.user_turn_pcm), expected_pcm)
            self.assertEqual(turn.user_turn_step, "intro_name")

            seen_eval_reqs = []
            seen_extractor_calls = []
            real_extract = profile_extractor.extract_profile_entities

            async def fake_eval(request, **_kwargs):
                seen_eval_reqs.append(request)
                return _decision_result(request, {
                    "danger_now": _noul(0.01),
                    "caller_stop": _noul(0.01),
                    "mentions_profile": _noul(0.92),
                    "current_outcome": _choice("other", {"other": 0.91, "clear": 0.06, "concern": 0.03}),
                })

            async def spy_extract(state, utterance_text="", audio_wav_bytes=None, **kwargs):
                seen_extractor_calls.append((state.get("active_qid"), utterance_text, audio_wav_bytes))
                return await real_extract(
                    state, utterance_text=utterance_text, audio_wav_bytes=audio_wav_bytes, **kwargs
                )

            fake_client = mock.MagicMock()
            fake_client.aio.models.generate_content = mock.AsyncMock(
                return_value=SimpleNamespace(
                    text=json.dumps({
                        "caller_name": "पूजा",
                        "marital_status": None,
                        "child_count": None,
                        "children": None,
                    })
                )
            )

            with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}), mock.patch.dict(
                profile_extractor._clients, {}, clear=True
            ), mock.patch.object(
                profile_extractor.genai, "Client", return_value=fake_client
            ), mock.patch.object(
                decisions, "evaluate_turn", side_effect=fake_eval
            ), mock.patch.object(
                profile_extractor, "extract_profile_entities", side_effect=spy_extract
            ):
                await gemini_live._answer_function_calls(
                    session, asyncio.Lock(), ws, asyncio.Lock(), call_id, telemetry,
                    [_Call("next_question", {}, "fc-audio-intro")], set(), set(), cursor, turn,
                    thresholds=thresholds,
                )
                await _drop_fallback(turn)

            self.assertEqual(len(seen_eval_reqs), 1)
            self.assertIsInstance(seen_eval_reqs[0]["input"], list)
            self.assertEqual(seen_eval_reqs[0]["input"][1]["mime_type"], "audio/wav")

            self.assertEqual(len(seen_extractor_calls), 1)
            ext_qid, ext_utt, ext_wav = seen_extractor_calls[0]
            self.assertEqual(ext_qid, "intro_name")
            self.assertEqual(ext_utt, "")
            self.assertIsNotNone(ext_wav)
            self.assertTrue(ext_wav.startswith(b"RIFF"))
            self.assertEqual(ext_wav[8:12], b"WAVE")
            with wave.open(io.BytesIO(ext_wav), "rb") as wf:
                self.assertEqual(wf.getnchannels(), 1)
                self.assertEqual(wf.getframerate(), 16000)
                self.assertEqual(wf.readframes(wf.getnframes()), expected_pcm)

            gen_kwargs = fake_client.aio.models.generate_content.await_args.kwargs
            self.assertIsInstance(gen_kwargs["contents"], list)
            self.assertEqual(gen_kwargs["contents"][1].inline_data.data, ext_wav)
            self.assertEqual(gen_kwargs["contents"][1].inline_data.mime_type, "audio/wav")

            self.assertEqual(cursor.state["profile"]["caller_name"], "पूजा")
            self.assertEqual(cursor.state["active_qid"], "intro_married")
            self.assertEqual(len(turn.capture.user_turn_pcm), 0)
            self.assertTrue(turn.capture.speech_allowed)

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
