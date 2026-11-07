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
        self.assertEqual(s["silence_duration_ms"], 750)
        self.assertIn("HIGH", str(s["end_sens"]).upper())
        self.assertEqual(s["watchdog_timeout_sec"], 8.0)


class Call226RegressionTests(unittest.TestCase):
    def test_speaking_again_cancels_a_silence_close(self):
        state = gemini_live._TurnState()
        telemetry = gemini_live._new_telemetry("gemini-3.1-flash-live-preview", "Kore", 1300, 300, 0)
        telemetry["force_close"] = True
        telemetry["force_close_at"] = time.time() - 5
        telemetry["close_reason"] = "silence"
        gemini_live._arm_user_turn(state, telemetry)
        self.assertFalse(telemetry["force_close"])
        self.assertEqual(telemetry["close_reason"], "")
        self.assertEqual(state.quiet_from, 0.0)

    def test_a_topic_id_in_domain_still_marks_coverage(self):
        telemetry = gemini_live._new_telemetry("gemini-3.1-flash-live-preview", "Kore", 1300, 300, 0)

        class _Fc:
            id = "fc_topic"
            name = "record_finding"
            args = {
                "domain": "intro_married",
                "status": "info",
                "detail": "Mother is married.",
                "is_red_flag": False,
                "topic_ids": [],
            }
            will_continue = False

        class _Sess:
            async def send_tool_response(self, function_responses=None):
                self.responses = function_responses

        async def run():
            await gemini_live._answer_function_calls(
                _Sess(), asyncio.Lock(), _FakeWs(), asyncio.Lock(),
                db.create_call("gemini"), telemetry, [_Fc()], set(), set(), "हो गई है।",
            )

        asyncio.run(run())
        self.assertIn("intro_married", telemetry["coverage"])

    def test_marriage_answer_does_not_replace_her_name(self):
        call_id = db.create_call("gemini")
        db.record_finding_db(call_id, "profile_mother_name", "info", "Mother's name is Aradhana.", False)
        db.record_finding_db(call_id, "profile_mother_name", "info", "Mother is married.", False)
        self.assertEqual(db.get_call(call_id)["mother_name"], "Mother's name is Aradhana.")


class MotherActionableQuestionsTests(unittest.TestCase):
    def _shell(self) -> str:
        return "".join(f"{h}\nNot reported.\n" for h in summarizer._REQUIRED_HEADINGS)

    def test_nothing_flagged_still_gives_actionable_questions(self):
        payload = {
            "doctor_markdown": self._shell(),
            "mother_hindi": "आपने बताया कि सब ठीक है।",
            "extracted_findings": {},
            "red_flags": [],
        }
        out = summarizer._assemble(payload, {}, [], [])
        mother = out["summary_mother"]
        self.assertNotIn("आपने बताया कि", mother)
        self.assertIn("डॉक्टर या स्वास्थ्य कार्यकर्ता (ASHA/ANM दीदी) से मिलें", mother)
        self.assertGreaterEqual(mother.count("?"), 3)
        self.assertIn("1. ", mother)
        self.assertIn("2. ", mother)
        self.assertIn("3. ", mother)
        self.assertIn("कोई तुरंत चिंताजनक बात सामने नहीं आई है", mother)

    def test_red_flags_become_questions_and_never_cross_diagnostic_boundary(self):
        payload = {
            "doctor_markdown": self._shell(),
            "mother_hindi": (
                "1. क्या बच्चे को हीमोफीलिया (hemophilia) या Duchenne की जाँच करवानी चाहिए?\n"
                "आपने बताया कि बच्चे के घुटने में सूजन आती है।"
            ),
            "extracted_findings": {
                "bleeding_joints": {
                    "status": "positive",
                    "detail": "Knee swelling after falls resolving over 5 days.",
                    "is_red_flag": True,
                },
                "family_infant_death": {
                    "status": "positive",
                    "detail": "Previous infant died at 5 months with fits.",
                    "is_red_flag": True,
                },
            },
            "red_flags": [],
        }
        out = summarizer._assemble(payload, {}, [], [])
        mother = out["summary_mother"]
        self.assertNotIn("हीमोफीलिया", mother)
        self.assertNotIn("hemophilia", mother.lower())
        self.assertNotIn("duchenne", mother.lower())
        self.assertNotIn("आपने बताया कि", mother)
        self.assertIn("गुजर", mother)
        self.assertIn("यह किसी बीमारी की पुष्टि (डायग्नोसिस) नहीं है", mother)
        self.assertGreaterEqual(mother.count("?"), 3)

    def test_fallback_summaries_work_with_empty_call(self):
        out = summarizer.build_fallback_summaries([], {}, [])
        self.assertIn("1. ", out["summary_mother"])
        self.assertGreaterEqual(out["summary_mother"].count("?"), 3)


class Call228RegressionTests(unittest.TestCase):
    def test_pregnancy_and_birth_nicu_domains_not_treated_as_group_skip(self):
        domain, topics = gemini_live._normalize_finding(
            "pregnancy",
            "Mother had low hemoglobin in second pregnancy.",
            ["body_blood_count"],
        )
        self.assertEqual(domain, "pregnancy")
        self.assertEqual(topics, ["body_blood_count"])
        self.assertNotIn("pregnancy", topics)

        domain2, topics2 = gemini_live._normalize_finding(
            "birth_nicu",
            "Heavy cord bleeding on third child.",
            ["birth_cord_bleed"],
        )
        self.assertEqual(domain2, "birth_nicu")
        self.assertEqual(topics2, ["birth_cord_bleed"])

    def test_children_detail_does_not_overwrite_mother_name(self):
        call_id = db.create_call("gemini")
        db.record_finding_db(call_id, "profile_mother_name", "info", "Mother's name is Jia.", False)
        domain, topics = gemini_live._normalize_finding(
            "profile_mother_name",
            "Mother has three children, ages 10, 8, and 6 years old",
            ["intro_children"],
        )
        self.assertEqual(domain, "profile_child_age")
        self.assertIn("intro_children", topics)
        db.record_finding_db(
            call_id,
            "profile_mother_name",
            "info",
            "Mother has three children, ages 10, 8, and 6 years old",
            False,
        )
        call = db.get_call(call_id)
        self.assertEqual(call["mother_name"], "Mother's name is Jia.")
        self.assertEqual(call["findings"]["profile_mother_name"]["detail"], "Mother's name is Jia.")

    def test_speech_end_waiting_triggers_fast_watchdog_nudge(self):
        state = gemini_live._TurnState()
        state.awaiting_bot_reply = True
        state.speech_end_waiting = True
        state.last_user_turn_ts = time.time() - 6.3
        telemetry = gemini_live._new_telemetry("gemini-3.1-flash-live-preview", "Kore", 750, 300, 0)
        session, _, _ = _run_watchdog(state, telemetry)
        self.assertEqual(session.texts, [gemini_live._WATCHDOG_NUDGE])
        self.assertEqual(telemetry["watchdog_nudges"], 1)
        self.assertNotIn("SYSTEM CUE", gemini_live._WATCHDOG_NUDGE)
        self.assertIn("NEVER say 'Nahi' or 'Haan'", gemini_live._WATCHDOG_NUDGE)

    def test_open_capture_concern_waits_for_aur_kuch_before_closing(self):
        telemetry = gemini_live._new_telemetry("gemini-3.1-flash-live-preview", "Kore", 1300, 300, 0)
        checklist.mark(telemetry["coverage"], [t for t in checklist.REQUIRED_IDS if t != "close_anything_else"])
        call_id = db.create_call("gemini")

        class _Fc1:
            id = "fc_open_1"
            name = "record_finding"
            args = {
                "domain": "chief_complaint",
                "status": "positive",
                "detail": "Mother wants to know if she can have another child at age 40.",
                "is_red_flag": False,
                "topic_ids": ["close_anything_else"],
            }
            will_continue = False

        class _Fc2:
            id = "fc_open_2"
            name = "record_finding"
            args = {
                "domain": "coverage",
                "status": "info",
                "detail": "Nothing else to add.",
                "is_red_flag": False,
                "topic_ids": ["close_anything_else"],
            }
            will_continue = False

        class _Sess:
            async def send_tool_response(self, function_responses=None):
                self.responses = function_responses

        async def run():
            s1, s2 = _Sess(), _Sess()
            await gemini_live._answer_function_calls(
                s1, asyncio.Lock(), _FakeWs(), asyncio.Lock(), call_id, telemetry,
                [_Fc1()], set(), set(), "हां, यह बताना है कि अभी मैं और एक बच्चा कर सकती हूं कि नहीं?",
            )
            await gemini_live._answer_function_calls(
                s2, asyncio.Lock(), _FakeWs(), asyncio.Lock(), call_id, telemetry,
                [_Fc2()], set(), set(), "नहीं।",
            )
            return s1, s2

        s1, s2 = asyncio.run(run())
        self.assertIn("Aur kuch batana chahengi?", s1.responses[0].response["note"])
        self.assertIn("Do NOT speak the closing yet", s1.responses[0].response["note"])
        self.assertIn("All topics are covered", s2.responses[0].response["note"])

    def test_prompt_and_checklist_rules_for_call228_feedback(self):
        prompt = prompts.GEMINI_LIVE_SYSTEM_PROMPT
        self.assertIn("NEVER SOUND JUDGMENTAL OR SURPRISED", prompt)
        self.assertIn("MULTI-CHILD SCOPE RULE", prompt)
        self.assertIn("aapke kisi bhi bachche ko", prompt)
        self.assertIn("kisi bachche ya bade ko", prompt)
        self.assertNotIn("Pehle bachche ke janm ke baare mein kuch sawal", prompt)
        note = checklist.tool_note(["intro"])
        self.assertIn("female voice (Kore)", note)
        self.assertIn("aapke kisi bhi bachche ko", note)
        self.assertNotIn("birth_term", note)
        cfg = gemini_live._live_config({})
        self.assertEqual(cfg.temperature, 0.5)
        self.assertFalse(cfg.enable_affective_dialog)
        self.assertIsNone(cfg.context_window_compression.trigger_tokens)
        self.assertIsNone(cfg.context_window_compression.sliding_window.target_tokens)

    def test_call230_no_children_and_not_pregnant_skips_child_and_preg_delivery_topics(self):
        _, topics = gemini_live._normalize_finding(
            "profile_child_age",
            "Mother is married but has no children.",
            ["intro_married", "intro_children"],
            "मेरी शादी हो गई है। बच्चे नहीं है।",
        )
        cov = []
        checklist.mark(cov, ["intro_name", "intro_bridge"] + topics)
        nxt = checklist.open_topics(cov)[0]["id"]
        self.assertEqual(nxt, "preg_thalassemia")

        _, topics2 = gemini_live._normalize_finding(
            "pregnancy",
            "Mother clarified she has not been pregnant.",
            ["preg_conditions"],
            "No, I haven't been pregnant.",
        )
        checklist.mark(cov, topics2)
        nxt2 = checklist.open_topics(cov)[0]["id"]
        self.assertEqual(nxt2, "fam_related")

        leaked = (
            "SYSTEM CUE (from the app, not the mother): The user is speaking English.\n"
            "Aapka naam bataiyega please?"
        )
        self.assertEqual(prompts.clean_assistant_caption(leaked), "Aapka naam bataiyega please?")


if __name__ == "__main__":
    unittest.main()
