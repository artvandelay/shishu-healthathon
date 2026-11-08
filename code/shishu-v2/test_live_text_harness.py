"""Automated test harness for deterministic call state.

Includes:
1. Fast offline replay test (Call #13 scenario + multi-child & name disambiguation + PDF generation)
2. Live WebSocket text injection runner against a live SHISHU server (no microphone needed).
"""

import asyncio
import json
import sys
import unittest

import call_state
import db
import mother_report


def _up(field, value, state="answered", scope="none", child_ids=None, details=None):
    return {
        "field": field,
        "state": state,
        "value": value,
        "scope": scope,
        "child_ids": list(child_ids or []),
        "details": dict(details or {}),
    }


def _ask(state, topic_id, followup_id=None):
    return call_state.begin_question(state, {"topic_id": topic_id, "followup_id": followup_id})


class Call13DeterministicReplayTests(unittest.TestCase):
    def test_call_13_flow_with_disambiguation_and_pdf(self):
        # 1. Start call
        state = call_state.new_call_state()
        self.assertEqual(call_state.next_question(state)["topic_id"], "intro_name")

        # Turn 1: "मेरा नाम जिगर है।"
        # Model updates profile.caller_name -> automatically marks intro_name answered!
        state = call_state.apply_updates(
            state,
            [_up("profile.caller_name", "जिगर")],
            turn_id=1,
        )
        self.assertEqual(state["profile"]["caller_name"], "जिगर")
        self.assertEqual(state["topics"]["intro_name"]["state"], "answered")
        nxt = call_state.next_question(state)
        self.assertEqual(nxt["topic_id"], "intro_married")

        # Turn 3: "हो गई है।"
        state = _ask(state, nxt["topic_id"])
        state = call_state.apply_updates(
            state,
            [_up("profile.marital_status", "married")],
            turn_id=2,
        )
        self.assertEqual(state["profile"]["marital_status"], "married")
        self.assertEqual(state["topics"]["intro_married"]["state"], "answered")
        nxt = call_state.next_question(state)
        self.assertEqual(nxt["topic_id"], "intro_children")

        # Turn 5-11: "दो बच्चे हैं... एक 2 साल का है और एक 1 साल का है।"
        state = _ask(state, nxt["topic_id"])
        children = [
            {"age": {"value": 2, "unit": "years"}},
            {"age": {"value": 1, "unit": "years"}},
        ]
        state = call_state.apply_updates(
            state,
            [_up("profile.children", children)],
            turn_id=3,
        )
        self.assertEqual(len(state["profile"]["children"]), 2)
        self.assertEqual(state["topics"]["intro_children"]["state"], "answered")
        nxt = call_state.next_question(state)
        self.assertEqual(nxt["topic_id"], "intro_bridge")

        # Turn 13: "ठीक है।"
        state = _ask(state, nxt["topic_id"])
        state = call_state.apply_updates(
            state,
            [_up("topic.intro_bridge", True)],
            turn_id=4,
        )
        nxt = call_state.next_question(state)
        self.assertEqual(nxt["topic_id"], "birth_mode")

        # Turn 15: "एक नॉर्मल एक ऑपरेशन" -> positive birth_mode
        state = _ask(state, nxt["topic_id"])
        state = call_state.apply_updates(
            state,
            [_up("topic.birth_mode", True, details={"delivery_type": "operation"})],
            turn_id=5,
        )
        # Follow-ups should include delivery_type/place, operation_reason, AND affected_child (since 2 kids exist)
        pending_fids = [f["id"] for f in call_state.TOPIC_RULES["birth_mode"]["followups"]]
        self.assertIn("affected_child", pending_fids)
        self.assertIn("operation_reason", pending_fids)

        # Resolve birth_mode followups
        state = call_state.apply_updates(
            state,
            [_up(
                "topic.birth_mode",
                True,
                scope="specific",
                child_ids=["child_1"],
                details={
                    "delivery_type": "operation",
                    "place": "hospital",
                    "operation_reason": "doctor ne bola",
                    "affected_child": "child_1",
                },
            )],
            turn_id=6,
        )
        # birth_mode is now resolved
        self.assertNotEqual((call_state.next_question(state) or {}).get("topic_id"), "birth_mode")

        # Skip forward to body_joint_swelling to test UNMAPPED NAME "Ram"
        state = _ask(state, "body_joint_swelling")
        state = call_state.apply_updates(
            state,
            [_up("topic.body_joint_swelling", True, scope="none", details={
                "joint": "ghutna", "trigger": "khelna", "duration": "kuch din",
            })],
            turn_id=7,
        )
        # Caller says "राम को" -> name not yet in profile.children!
        state = call_state.apply_updates(
            state,
            [_up("topic.body_joint_swelling", True, details={"affected_child": "Ram"})],
            turn_id=8,
        )
        # affected_child should still be UNRESOLVED because Ram is not mapped to an age!
        nxt = call_state.next_question(state)
        self.assertEqual(nxt["topic_id"], "body_joint_swelling")
        self.assertEqual(nxt["followup_id"], "affected_child")
        # Prompt note explicitly instructs Gemini to ask which age child Ram is
        self.assertIn("Ram", nxt["prompt_note"])
        self.assertIn("2 saal", nxt["prompt_note"])

        # Caller clarifies Ram is the 2-year-old child -> update profile.children name + bind child_ids
        updated_children = [
            {"id": "child_1", "name": "Ram", "age": {"value": 2, "unit": "years"}},
            {"id": "child_2", "name": None, "age": {"value": 1, "unit": "years"}},
        ]
        state = call_state.apply_updates(
            state,
            [
                _up("profile.children", updated_children),
                _up(
                    "topic.body_joint_swelling",
                    True,
                    scope="specific",
                    child_ids=["child_1"],
                    details={
                        "joint": "ghutna",
                        "duration": "kuch din",
                        "trigger": "khelne ke baad",
                        "affected_child": "Ram",
                    },
                ),
            ],
            turn_id=9,
        )
        # Now child_1 has name Ram, and body_joint_swelling is resolved
        self.assertEqual(state["profile"]["children"][0]["name"], "Ram")
        self.assertNotEqual((call_state.next_question(state) or {}).get("topic_id"), "body_joint_swelling")

        # Test Family Kinship rule on fam_bp_stroke
        state = _ask(state, "fam_bp_stroke")
        state = call_state.apply_updates(
            state,
            [_up("topic.fam_bp_stroke", True, scope="none", details={"condition": "BP", "onset_age": "38"})],
            turn_id=10,
        )
        # Caller says "Ramesh ko" (personal name only without kinship)
        state = call_state.apply_updates(
            state,
            [_up("topic.fam_bp_stroke", True, details={"who": "Ramesh"})],
            turn_id=11,
        )
        nxt = call_state.next_question(state)
        self.assertEqual(nxt["topic_id"], "fam_bp_stroke")
        self.assertEqual(nxt["followup_id"], "relationship")
        self.assertIn("kinship", nxt["prompt_note"])

        # Caller provides kinship "mama"
        state = call_state.apply_updates(
            state,
            [_up(
                "topic.fam_bp_stroke",
                True,
                details={
                    "who": "Ramesh",
                    "relationship": "mama (mother's brother)",
                    "condition": "BP",
                    "onset_age": "38",
                },
            )],
            turn_id=12,
        )
        self.assertNotEqual((call_state.next_question(state) or {}).get("topic_id"), "fam_bp_stroke")

        # Caller asks to stop -> request_close("caller_stop")
        state = call_state.request_close(state, "caller_stop")
        self.assertEqual(state["phase"], "closing")
        self.assertEqual(state["close_reason"], "caller_stop")

        # Build Mother PDF from this state
        call_dict = {
            "id": 999,
            "started_at": "2026-11-08 02:36:00",
            "duration_sec": 300,
            "caller_name": "jigar",
            "mother_name": "",
            "state": state,
            "telemetry": {"close_reason": "caller_stop"},
        }
        report = mother_report.build(call_dict)

        # CRITICAL ASSERTIONS:
        # 1. Mother name is NOT "माँ" -> it must be "जिगर"
        self.assertTrue(report["who"].startswith("जिगर"), f"Expected जिगर in who, got: {report['who']}")
        # 2. Status is incomplete
        self.assertEqual(report["status"], "अधूरी कॉल")
        # 3. "इस कॉल में जो हाँ था" contains positive findings with transliterated details
        self.assertEqual(len(report["yes"]), 3, f"Expected 3 yes items, got: {report['yes']}")
        yes_text = " ".join(report["yes"])
        self.assertIn("ऑपरेशन", yes_text)
        self.assertNotIn("ओपेरतिओन", yes_text)


class AdversarialFlowTests(unittest.TestCase):
    def test_flow_4_mid_call_acute_emergency_interrupt(self):
        """Caller interrupts mid-call with an acute emergency; verifies immediate shutdown and safety banner."""
        state = call_state.new_call_state()
        state = call_state.apply_updates(state, [_up("profile.caller_name", "सीमा")], 1)
        state = _ask(state, "birth_nicu")
        state = call_state.apply_updates(
            state,
            [_up("topic.birth_nicu", True, scope="all", details={"reason": "jaundice", "days": 3})],
            2,
        )

        # Emergency happens!
        state = call_state.request_close(state, "emergency")
        self.assertEqual(state["phase"], "closing")
        self.assertEqual(state["close_reason"], "emergency")

        # Next question MUST be closing (not asking checklist topics)
        closing = call_state.next_question(state)
        self.assertEqual(closing["question_kind"], "closing")
        self.assertIn("emergency", closing["prompt_note"])

        # Mother PDF must prominently display emergency 108 banner
        call_dict = {
            "id": 777,
            "started_at": "2026-11-08 03:10:00",
            "duration_sec": 120,
            "caller_name": "seema",
            "mother_name": "",
            "state": state,
            "telemetry": {"close_reason": "emergency", "emergency_detail": "bachche ko saans nahi aa rahi"},
        }
        report = mother_report.build(call_dict)
        self.assertTrue(report["emergency"])
        self.assertTrue(report["incomplete"])
        html_out = mother_report.render_html(report)
        self.assertIn("108", html_out)
        self.assertIn("अस्पताल ले जाइए", html_out)
        # Past answered topic (NICU) is still preserved
        self.assertEqual(len(report["yes"]), 1)

    def test_a_yes_opens_only_that_topics_followups(self):
        """Near-duplicate follow-up routes, one table."""
        cases = [
            ("birth_hearing_vision", False, ["test_done"], 0, False),
            ("birth_hearing_vision", True, ["hearing_or_vision"], 0, False),
            ("birth_cleft", True, ["affected_child"], 2, False),
            ("birth_cord_3weeks", True, ["affected_child"], 2, False),
            ("body_blood_count", True, ["low_or_high", "timing"], 0, False),
            ("body_blue", True, ["location", "timing"], 0, False),
            ("body_limb_pain", True, ["location", "frequency", "hospital_visits"], 0, False),
            ("ill_fasting", True, ["symptom", "trigger", "count"], 0, False),
            ("ill_admissions", True, ["count", "reasons"], 0, False),
            ("preg_conditions", True, ["conditions", "continued_after_delivery"], 0, True),
            ("fam_infant_death", True, ["relationship", "age_at_death", "cause"], 0, False),
        ]
        for topic_id, value, expected, n_children, ever_pregnant in cases:
            with self.subTest(topic=topic_id, value=value):
                state = call_state.new_call_state()
                if n_children:
                    state = call_state.apply_updates(
                        state,
                        [_up("profile.children", [
                            {"age": {"value": age, "unit": "years"}} for age in range(1, n_children + 1)
                        ])],
                        1,
                    )
                if ever_pregnant:
                    state = call_state.apply_updates(state, [_up("profile.ever_pregnant", True)], 1)
                state = _ask(state, topic_id)
                state = call_state.apply_updates(state, [_up(f"topic.{topic_id}", value)], 2)
                pending = [item["id"] for item in call_state._pending_followups(state, topic_id)]
                for followup_id in expected:
                    self.assertIn(followup_id, pending)


_FULL_CALL_REPLIES = {
    ("intro_name", None): "मेरा नाम कविता है।",
    ("intro_married", None): "शादी हो गई है।",
    ("intro_children", None): "एक बच्चा है, तीन साल का।",
    ("intro_bridge", None): "ठीक है।",
    ("birth_mode", None): "अस्पताल में ऑपरेशन से हुआ था।",
    ("birth_mode", "delivery_type"): "ऑपरेशन से हुआ था।",
    ("birth_mode", "place"): "अस्पताल में।",
    ("birth_mode", "operation_reason"): "डॉक्टर ने पानी कम बताया था।",
    ("birth_nicu", None): "हाँ, पीलिया की वजह से 4 दिन काँच के बॉक्स में रखा था।",
    ("birth_nicu", "reason"): "पीलिया की वजह से।",
    ("birth_nicu", "days"): "चार दिन।",
    ("body_fits_milestones", None): "हाँ, एक साल पहले बुखार में झटके आए थे।",
    ("body_fits_milestones", "type"): "झटके आए थे।",
    ("body_fits_milestones", "onset"): "एक साल पहले।",
    ("fam_bp_stroke", None): "हाँ, मेरे सगे मामा को 35 साल की उम्र में बीपी हुआ था।",
    ("fam_bp_stroke", "condition"): "बीपी हुआ था।",
    ("fam_bp_stroke", "who"): "मेरे सगे मामा को।",
    ("fam_bp_stroke", "relationship"): "मेरे सगे मामा हैं।",
    ("fam_bp_stroke", "onset_age"): "35 साल की उम्र में।",
    ("profile.ever_pregnant", None): "हाँ, एक बार प्रेग्नेंट हुई थी।",
    ("profile.has_delivered", None): "हाँ, डिलीवरी हुई थी।",
    ("close_anything_else", None): "नहीं, बस इतना ही।",
}


def _reply_for_state(state: dict) -> str | None:
    if not state or state.get("phase") in ("closing", "ended"):
        return None
    cur = state.get("current_question") or {}
    tid = cur.get("topic_id")
    fid = cur.get("followup_id")
    if (tid, fid) in _FULL_CALL_REPLIES:
        return _FULL_CALL_REPLIES[(tid, fid)]
    if fid is not None:
        return "पता नहीं।"
    return "नहीं।"


async def run_live_text_call(base_url: str = "ws://127.0.0.1:8023/ws/gemini") -> dict:
    """Connect to a live SHISHU WebSocket and conduct a full state-driven screening to completion."""
    import websockets
    import app

    events = []
    current_step = 0
    call_id = None
    token = app._session_token("user", "jigar")
    from inspect import signature
    sig = signature(websockets.connect)
    if "additional_headers" in sig.parameters:
        kwargs = {"additional_headers": [("Cookie", f"{app.COOKIE_NAME}={token}")]}
    else:
        kwargs = {"extra_headers": {"Cookie": f"{app.COOKIE_NAME}={token}"}}

    try:
        async with websockets.connect(base_url, **kwargs) as ws:
            print("[live_test] Connected to SHISHU WebSocket", flush=True)
            assistant_spoke_since_turn = False

            while True:
                msg = await asyncio.wait_for(ws.recv(), timeout=90.0)
                payload = json.loads(msg)
                mtype = payload.get("type")
                events.append(payload)

                if mtype == "session_started":
                    call_id = payload.get("call_id")
                    print(f"[live_test] Session started, call_id: {call_id}", flush=True)

                elif mtype == "audio":
                    assistant_spoke_since_turn = True

                elif mtype == "transcript" and payload.get("role") == "assistant":
                    bot_text = payload.get("text", "")
                    print(f"[live_test] Assistant: {bot_text}", flush=True)
                    assistant_spoke_since_turn = True

                elif mtype == "telemetry":
                    last_evt = (payload.get("last_event") or {}).get("type")
                    if last_evt == "turn_complete" and assistant_spoke_since_turn and call_id:
                        assistant_spoke_since_turn = False
                        saved = db.get_call(call_id) or {}
                        state = saved.get("state") or {}
                        reply = _reply_for_state(state)
                        if reply and current_step < 85:
                            current_step += 1
                            cur = state.get("current_question") or {}
                            print(
                                f"[live_test] Turn {current_step} ({cur.get('topic_id')}/{cur.get('followup_id')}): {reply}",
                                flush=True,
                            )
                            await ws.send(json.dumps({"type": "text", "text": reply}))

                elif mtype == "ending":
                    print("[live_test] Ending signal received", flush=True)

                elif mtype == "call_completed":
                    print(f"[live_test] Call completed! ID: {payload.get('call_id')}", flush=True)
                    return {
                        "call_id": payload.get("call_id"),
                        "events": events,
                        "success": True,
                    }
    except Exception as exc:
        print(f"[live_test] Connection ended: {exc}")
        return {"success": False, "call_id": call_id, "error": str(exc), "events": events}


if __name__ == "__main__":
    if "--live" in sys.argv:
        port = sys.argv[sys.argv.index("--port") + 1] if "--port" in sys.argv else "8023"
        url = f"ws://127.0.0.1:{port}/ws/gemini"
        print(f"Running live text-injection test against {url}...")
        result = asyncio.run(run_live_text_call(url))
        print("Live Result:", result.get("success"), "Call ID:", result.get("call_id"))
    else:
        unittest.main()
