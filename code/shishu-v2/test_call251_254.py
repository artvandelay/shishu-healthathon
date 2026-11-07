"""Call 251 (wide question first) and call 254 (emergency end) regressions. No network."""

import asyncio
import os
import tempfile
import unittest

_TMP = tempfile.mkdtemp(prefix="shishu-251-254-")
os.environ.setdefault("DATA_DIR", _TMP)

import checklist  # noqa: E402
import db  # noqa: E402
import gemini_live  # noqa: E402
import mother_report  # noqa: E402
import prompts  # noqa: E402
import question_flow  # noqa: E402


class _FakeWs:
    def __init__(self):
        self.client_state = gemini_live.WebSocketState.CONNECTED
        self.sent = []

    async def send_json(self, payload):
        self.sent.append(payload)


class _Sess:
    async def send_tool_response(self, function_responses=None):
        self.responses = function_responses


class _Fc:
    def __init__(self, name, args, fc_id):
        self.id = fc_id
        self.name = name
        self.args = args
        self.will_continue = False


def _telemetry():
    return gemini_live._new_telemetry("gemini-3.1-flash-live-preview", "Kore", 750, 300, 0)


def _cov_until(topic_id):
    ids = list(checklist.REQUIRED_IDS)
    return ids[: ids.index(topic_id)]


def _run(telemetry, fc, user_text, bot_text=""):
    async def go():
        sess, ws = _Sess(), _FakeWs()
        await gemini_live._answer_function_calls(
            sess, asyncio.Lock(), ws, asyncio.Lock(), db.create_call("gemini"), telemetry,
            [fc], set(), set(), user_text, last_bot_text=bot_text,
        )
        return sess, ws

    return asyncio.run(go())


def _finding(topic_id, domain, status, detail, n):
    return _Fc(
        "record_finding",
        {"domain": domain, "status": status, "detail": detail, "is_red_flag": False, "topic_ids": [topic_id]},
        f"fc_{topic_id}_{n}",
    )


class WideQuestionFirstTests(unittest.TestCase):
    def test_ids_unchanged_and_contract_keys_valid(self):
        self.assertEqual(len(checklist.REQUIRED_IDS), 54)
        for tid in list(question_flow.WIDE_WORDS) + list(question_flow.DRILLS):
            self.assertIn(tid, checklist._BY_ID)

    def test_wide_words_hold_no_follow_up(self):
        for words in question_flow.WIDE_WORDS.values():
            low = words.lower()
            for bad in ("agar haan", "kisko", "kitni baar", "kaunsa mahina", "kaun sa mahina", "result", "maa aur pita dono"):
                self.assertNotIn(bad, low)

    def test_tool_note_uses_wide_words_and_drill(self):
        note = checklist.tool_note(_cov_until("fam_bp_stroke"))
        self.assertIn("40-50", note)
        self.assertNotIn("agar haan", note)
        self.assertIn("ONLY if she says 'Haan'", note)

    def test_section_bridge_once(self):
        cov = _cov_until("fam_related")
        bridged = ["birth", "body", "illness", "pregnancy"]
        self.assertIn("NEW SECTION (family)", checklist.tool_note(cov, bridged))
        self.assertIn("family", bridged)
        self.assertNotIn("NEW SECTION", checklist.tool_note(cov, bridged))
        self.assertNotIn("NEW SECTION", checklist.tool_note(cov))

    def test_bare_haan_holds_hearing_vision_then_detail_closes(self):
        t = _telemetry()
        t["coverage"] = _cov_until("birth_hearing_vision")
        sess, _ = _run(t, _finding("birth_hearing_vision", "milestones", "info", "Hearing and vision test done.", 1), "हां।")
        self.assertNotIn("birth_hearing_vision", t["coverage"])
        self.assertIn("report normal", sess.responses[0].response["note"])
        _run(t, _finding("birth_hearing_vision", "milestones", "info", "Hearing and vision report normal.", 2), "हां, नॉर्मल था।")
        self.assertIn("birth_hearing_vision", t["coverage"])

    def test_nahi_closes_thalassemia_at_once(self):
        t = _telemetry()
        t["coverage"] = _cov_until("preg_thalassemia")
        _run(t, _finding("preg_thalassemia", "pregnancy", "negative", "No thalassemia test.", 1), "नहीं।")
        self.assertIn("preg_thalassemia", t["coverage"])

    def test_kya_is_never_an_answer(self):
        t = _telemetry()
        t["coverage"] = _cov_until("preg_thalassemia")
        sess, ws = _run(t, _finding("preg_thalassemia", "pregnancy", "info", "Mother does not know.", 1), "क्या?")
        self.assertNotIn("preg_thalassemia", t["coverage"])
        self.assertEqual(sess.responses[0].response["note"], question_flow.CONFUSED_NOTE)
        self.assertFalse(any(m.get("type") == "finding" for m in ws.sent))

    def test_cut_off_family_bridge_does_not_cover_fam_related(self):
        t = _telemetry()
        t["coverage"] = _cov_until("fam_related")
        sess, _ = _run(
            t, _finding("fam_related", "consanguinity", "negative", "Not related.", 1), "नहीं।",
            "Ab कुछ सवाल आपके पूरे parivaar के बारे में — यानी आपके",
        )
        self.assertNotIn("fam_related", t["coverage"])
        self.assertIn("GROUP BRIDGE", sess.responses[0].response["note"])
        _run(
            t, _finding("fam_related", "consanguinity", "negative", "Not related.", 2), "नहीं।",
            "Kya aap aur aapke pati shaadi se pehle rishtedaar the?",
        )
        self.assertIn("fam_related", t["coverage"])

    def test_real_family_question_passes_first_time(self):
        t = _telemetry()
        t["coverage"] = _cov_until("fam_related")
        _run(
            t, _finding("fam_related", "consanguinity", "negative", "Not related.", 3), "नहीं।",
            "क्या आप और आपके पति शादी से पहले रिश्तेदार थे?",
        )
        self.assertIn("fam_related", t["coverage"])

    def test_bp_bare_yes_asks_which_one_not_who(self):
        t = _telemetry()
        t["coverage"] = _cov_until("fam_bp_stroke")
        sess, _ = _run(t, _finding("fam_bp_stroke", "family_history", "positive", "Someone had BP.", 1), "हाँ")
        first = question_flow.DRILLS["fam_bp_stroke"][0]
        self.assertIn(first, sess.responses[0].response["note"])
        self.assertNotIn("kisko", first)
        self.assertNotIn("fam_bp_stroke", t["coverage"])

    def test_classifiers(self):
        self.assertTrue(question_flow.is_emergency_text("वह सांस बहुत तेज ले रहा है"))
        self.assertFalse(question_flow.is_emergency_text("ek baar behosh hua tha"))
        self.assertTrue(question_flow.is_confused_reply("क्या?"))
        self.assertFalse(question_flow.is_confused_reply("क्या बताऊं, नहीं हुआ"))
        self.assertFalse(question_flow.is_bare_yes("हां, नॉर्मल था"))

    def test_prompt_rules_present(self):
        text = prompts.GEMINI_LIVE_SYSTEM_PROMPT
        for needle in ("WIDE QUESTION FIRST RULE", "EMERGENCY RULE", "CALLER STOP RULE", "108", "7. NEVER SPEAK HER ANSWER"):
            self.assertIn(needle, text)
        self.assertNotIn('say "Bas ek-do sawal aur." and ask', text)


class EmergencyEndTests(unittest.TestCase):
    def test_emergency_reason_ends_with_open_topics(self):
        t = _telemetry()
        t["coverage"] = ["intro_name", "intro_married"]
        sess, ws = _run(t, _Fc("end_call", {"reason": "emergency"}, "fc_e1"), "मुझे ऐसा लग रहा है कि वह सांस नहीं ले रहा है।")
        self.assertEqual(sess.responses[0].response["result"], "ok")
        self.assertTrue(t["end_requested"])
        self.assertEqual(t["close_reason"], "emergency")
        self.assertTrue(any(m.get("type") == "ending" for m in ws.sent))

    def test_emergency_backstop_without_reason(self):
        t = _telemetry()
        t["coverage"] = ["intro_name"]
        t["emergency_seen"] = True
        t["emergency_detail"] = "वह सांस नहीं ले रहा है"
        sess, _ = _run(t, _Fc("end_call", {}, "fc_e2"), "नमस्ते।")
        self.assertEqual(sess.responses[0].response["result"], "ok")
        self.assertEqual(t["close_reason"], "emergency")

    def test_early_end_without_reason_is_always_refused(self):
        t = _telemetry()
        t["coverage"] = ["intro_name"]
        for n in range(3):
            sess, _ = _run(t, _Fc("end_call", {}, f"fc_r{n}"), "नमस्ते।")
            self.assertEqual(sess.responses[0].response["result"], "not_yet")
            self.assertNotIn("Bas ek-do sawal aur", sess.responses[0].response["note"])
        self.assertEqual(t["end_refusals"], 3)
        self.assertFalse(t.get("end_requested"))

    def test_caller_stop_is_accepted(self):
        t = _telemetry()
        t["coverage"] = ["intro_name"]
        sess, _ = _run(t, _Fc("end_call", {"reason": "caller_stop"}, "fc_s1"), "अभी बंद कर दो, बाद में बात करते हैं।")
        self.assertEqual(sess.responses[0].response["result"], "ok")
        self.assertEqual(t["close_reason"], "caller_stop")

    def test_force_close_still_ends(self):
        t = _telemetry()
        t["force_close"] = True
        sess, _ = _run(t, _Fc("end_call", {}, "fc_f1"), "")
        self.assertEqual(sess.responses[0].response["result"], "ok")

    def test_emergency_and_stop_notes(self):
        doctor, mother = checklist.incomplete_notes(["intro_name"], "emergency", "वह सांस नहीं ले रहा है")
        self.assertIn("EMERGENCY", doctor)
        self.assertIn("सांस", doctor)
        self.assertIn("108", mother)
        doctor2, mother2 = checklist.incomplete_notes(["intro_name"], "caller_stop")
        self.assertIn("asked to stop", doctor2)
        self.assertIn("dobara call", mother2)

    def test_mother_report_emergency_band(self):
        call = {
            "id": 254, "mother_name": "Payal", "child_name": "", "child_age": "2 years", "child_gender": "",
            "started_at": "2026-11-07 10:20:00", "duration_sec": 162, "summary_mother": "",
            "findings": {}, "telemetry": {"coverage": ["intro_name"], "close_reason": "emergency"},
            "transcript": [], "status": "completed",
        }
        report = mother_report.build(call)
        self.assertTrue(report["incomplete"])
        self.assertTrue(report["emergency"])
        self.assertIn("108", mother_report.render_html(report))


if __name__ == "__main__":
    unittest.main()
