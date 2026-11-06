"""Checks for the checklist-driven triage flow. No network."""

import asyncio
import os
import tempfile
import time
import unittest

_TMP = tempfile.mkdtemp(prefix="shishu-triage-")
os.environ.setdefault("DATA_DIR", _TMP)

import checklist  # noqa: E402
import db  # noqa: E402
import gemini_live  # noqa: E402
import prompts  # noqa: E402
import summarizer  # noqa: E402


class ChecklistTests(unittest.TestCase):
    def test_ids_are_unique_and_groups_known(self):
        ids = [t["id"] for t in checklist.TOPICS]
        self.assertEqual(len(ids), len(set(ids)))
        for t in checklist.TOPICS:
            self.assertIn(t["group"], checklist.GROUPS)
        self.assertEqual(len(checklist.REQUIRED_IDS), 54)

    def test_group_name_marks_whole_group(self):
        cov = []
        self.assertEqual(len(checklist.mark(cov, ["birth"])), 8)
        self.assertIn("birth_bcg", cov)
        self.assertEqual(checklist.mark(cov, ["birth_bcg", "not_a_topic"]), [])

    def test_can_end_only_when_required_topics_are_covered(self):
        cov = []
        ok, note = checklist.can_end(cov)
        self.assertFalse(ok)
        self.assertIn("intro_name", note)
        checklist.mark(cov, list(checklist.REQUIRED_IDS))
        self.assertEqual(checklist.can_end(cov), (True, ""))

    def test_tool_note_and_progress_label(self):
        cov = []
        checklist.mark(cov, ["intro"])
        self.assertIn("birth_mode", checklist.tool_note(cov))
        self.assertEqual(checklist.progress_label(cov), "birth 4/54")


class PromptTests(unittest.TestCase):
    def test_prompt_lists_every_topic_and_the_outro(self):
        text = prompts.GEMINI_LIVE_SYSTEM_PROMPT
        for tid in checklist.REQUIRED_IDS:
            self.assertIn(tid, text)
        self.assertIn("UNMISTAKABLY IN HINDI", text)
        self.assertIn("Aapko iske alawa doctor ke liye kuch aur batana hai?", text)
        self.assertIn("doctor ke liye poori report", text)
        self.assertIn("chhota sa message", text)
        self.assertIn("STYLE EXAMPLES FROM REAL CALLS", text)
        for other in (
            text,
            prompts.OPENAI_LIVE_VOICE_PROMPT,
            prompts.OPENAI_RESPONSES_BACKEND_PROMPT,
        ):
            self.assertNotIn("set_progress", other)


class GeminiConfigTests(unittest.TestCase):
    def test_tool_declarations(self):
        self.assertFalse(hasattr(gemini_live, "_SET_PROGRESS_DECL"))
        props = gemini_live._RECORD_FINDING_DECL["parameters"]["properties"]
        self.assertIn("topic_ids", props)

    def test_thinking_and_vocab_only_on_gemini_3(self):
        cfg = gemini_live._live_config({})
        self.assertIn("MINIMAL", str(cfg.thinking_config.thinking_level).upper())
        self.assertIn("thalassemia", cfg.input_audio_transcription.custom_vocabulary)
        old = gemini_live._live_config({"gemini_model": "gemini-2.5-flash-native-audio-latest"})
        self.assertIsNone(old.thinking_config)


class FindingMergeTests(unittest.TestCase):
    def test_negative_does_not_overwrite_positive(self):
        call_id = db.create_call("gemini")
        db.record_finding_db(call_id, "birth_nicu", "positive", "NICU for 4 days", True)
        db.record_finding_db(call_id, "birth_nicu", "negative", "No cord bleeding", False)
        found = db.get_call(call_id)["findings"]["birth_nicu"]
        self.assertEqual(found["status"], "positive")
        self.assertTrue(found["is_red_flag"])
        self.assertIn("NICU for 4 days", found["detail"])

    def test_two_positives_are_joined(self):
        call_id = db.create_call("gemini")
        db.record_finding_db(call_id, "birth_nicu", "positive", "NICU for 4 days", True)
        db.record_finding_db(call_id, "birth_nicu", "positive", "BCG site lump", True)
        found = db.get_call(call_id)["findings"]["birth_nicu"]
        self.assertIn("NICU for 4 days; BCG site lump", found["detail"])


class IncompleteNotesTests(unittest.TestCase):
    def test_notes_only_when_topics_are_open(self):
        doctor, mother = checklist.incomplete_notes([], "silence")
        self.assertIn("Incomplete call", doctor)
        self.assertIn("silent for 30 seconds", doctor)
        self.assertIn("dobara call", mother)
        full = list(checklist.REQUIRED_IDS)
        self.assertEqual(checklist.incomplete_notes(full, "silence"), ("", ""))

    def test_prompt_has_incomplete_closings_and_cue_rule(self):
        text = prompts.GEMINI_LIVE_SYSTEM_PROMPT
        self.assertIn("INCOMPLETE CLOSING", text)
        self.assertIn("dobara call karke ise poora kar lijiye", text)
        self.assertIn("SYSTEM CUE (from the app, not the mother):", text)


class _FakeWs:
    def __init__(self):
        self.client_state = gemini_live.WebSocketState.CONNECTED
        self.sent = []

    async def send_json(self, payload):
        self.sent.append(payload)


class _FakeSession:
    def __init__(self):
        self.texts = []

    async def send_realtime_input(self, text=None, **kwargs):
        self.texts.append(text)


def _run_watchdog(state, telemetry, seconds=1.2):
    async def run():
        ws, session = _FakeWs(), _FakeSession()
        stop = asyncio.Event()
        call_id = db.create_call("gemini")
        task = asyncio.create_task(
            gemini_live._watchdog_loop(
                session, asyncio.Lock(), ws, asyncio.Lock(), stop, call_id, telemetry, state, 4.0
            )
        )
        await asyncio.sleep(seconds)
        stop.set()
        await task
        return session, ws, stop

    return asyncio.run(run())


class CallTimerTests(unittest.TestCase):
    def _telemetry(self):
        return gemini_live._new_telemetry("gemini-3.1-flash-live-preview", "Kore", 650, 300, 0)

    def test_silence_triggers_incomplete_close(self):
        state = gemini_live._TurnState()
        state.quiet_from = time.time() - (checklist.SILENCE_CLOSE_SEC + 1)
        state.checked_in = True
        telemetry = self._telemetry()
        session, _, _ = _run_watchdog(state, telemetry)
        self.assertIn(checklist.SILENCE_CLOSE_CUE, session.texts)
        self.assertTrue(telemetry["force_close"])
        self.assertEqual(telemetry["close_reason"], "silence")

    def test_check_in_comes_first(self):
        state = gemini_live._TurnState()
        state.quiet_from = time.time() - (checklist.CHECK_IN_SEC + 1)
        telemetry = self._telemetry()
        session, _, _ = _run_watchdog(state, telemetry)
        self.assertEqual(session.texts, [checklist.CHECK_IN_CUE])
        self.assertFalse(telemetry["force_close"])

    def test_hard_limit_stops_the_call(self):
        state = gemini_live._TurnState()
        state.call_started_ts = time.time() - (checklist.HARD_LIMIT_SEC + 1)
        telemetry = self._telemetry()
        _, _, stop = _run_watchdog(state, telemetry)
        self.assertTrue(stop.is_set())
        self.assertEqual(telemetry["close_reason"], "time_limit")

    def test_force_close_allows_end_call_with_open_topics(self):
        telemetry = self._telemetry()
        telemetry["force_close"] = True

        class _Fc:
            id = "fc1"
            name = "end_call"
            args = {}
            will_continue = False

        class _Sess:
            async def send_tool_response(self, function_responses=None):
                self.responses = function_responses

        async def run():
            sess = _Sess()
            await gemini_live._answer_function_calls(
                sess, asyncio.Lock(), _FakeWs(), asyncio.Lock(), db.create_call("gemini"),
                telemetry, [_Fc()], set(), set(),
            )
            return sess

        sess = asyncio.run(run())
        self.assertTrue(telemetry["end_requested"])
        self.assertEqual(sess.responses[0].response["result"], "ok")


class Call223RegressionTests(unittest.TestCase):
    def test_shilpa_removed_and_hallucinated_name_stripped(self):
        self.assertNotIn("Shilpa", prompts.DOCTOR_MOTHER_SUMMARY_PROMPT)
        shell = "".join(f"{h}\nNot reported.\n" for h in summarizer._REQUIRED_HEADINGS)
        payload = {
            "doctor_markdown": shell.replace(
                "### Patient Profile & Chief Complaint\nNot reported.",
                "### Patient Profile & Chief Complaint\nMother Shilpa has one child.",
            ),
            "mother_hindi": "मेरा नाम शिल्पा है। मेरे एक बच्चा है, जो अभी 3 साल का है।",
            "extracted_findings": {
                "profile_mother_name": {
                    "status": "info",
                    "detail": "Mother's name is Shilpa.",
                    "is_red_flag": False,
                }
            },
            "red_flags": [],
        }
        transcript = [
            {"role": "assistant", "text": "Namaste, main Shishu hoon. Aapka naam bataiyega please?"},
            {"role": "user", "text": "मेरा नाम"},
        ]
        out = summarizer._assemble(payload, {}, [], transcript)
        self.assertNotIn("शिल्पा", out["summary_mother"])
        self.assertNotIn("Shilpa", out["summary_doctor"])
        self.assertNotIn("profile_mother_name", out["findings"])

    def test_incomplete_name_fragment_keeps_intro_name_open(self):
        telemetry = gemini_live._new_telemetry("gemini-3.1-flash-live-preview", "Kore", 1300, 300, 0)

        class _Fc:
            id = "fc_name"
            name = "record_finding"
            args = {
                "domain": "profile_mother_name",
                "status": "info",
                "detail": "Mother did not state her name.",
                "is_red_flag": False,
                "topic_ids": ["intro_name"],
            }
            will_continue = False

        class _Sess:
            async def send_tool_response(self, function_responses=None):
                self.responses = function_responses

        async def run():
            sess = _Sess()
            await gemini_live._answer_function_calls(
                sess,
                asyncio.Lock(),
                _FakeWs(),
                asyncio.Lock(),
                db.create_call("gemini", caller_name="PD"),
                telemetry,
                [_Fc()],
                set(),
                set(),
                "मेरा नाम",
            )
            return sess

        sess = asyncio.run(run())
        self.assertNotIn("intro_name", telemetry["coverage"])
        self.assertIn("poora naam", sess.responses[0].response["note"])

    def test_user_caption_normalizes_foreign_no_and_hangul(self):
        self.assertEqual(prompts.clean_user_caption("Não."), "नहीं।")
        self.assertEqual(prompts.clean_user_caption("Nem."), "नहीं।")
        self.assertEqual(prompts.clean_user_caption("Nein."), "नहीं।")
        self.assertEqual(prompts.clean_user_caption("네. 아니."), "नहीं।")
        self.assertEqual(prompts.clean_user_caption("9개월"), "9 महीने")

    def test_vad_defaults_give_caller_thinking_room(self):
        s = gemini_live._session_settings({})
        self.assertEqual(s["silence_duration_ms"], 1300)
        self.assertIn("LOW", str(s["end_sens"]).upper())
        self.assertEqual(s["watchdog_timeout_sec"], 8.0)


if __name__ == "__main__":
    unittest.main()
