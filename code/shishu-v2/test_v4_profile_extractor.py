"""100% offline unit tests for Phase 5 profile_extractor."""

from __future__ import annotations

import asyncio
import json
import os
import re
import unittest
from pathlib import Path
from unittest import mock

from google.genai import types

import profile_extractor
import test_v4_cursor as cursor_tests
from call_cursor import apply_interpreted_turn, apply_profile, new_cursor_state


def _full_raw(**overrides) -> dict:
    base = {
        "caller_name": None,
        "marital_status": None,
        "child_count": None,
        "children": None,
        "ever_pregnant": None,
    }
    base.update(overrides)
    return base


class ProfileExtractorOfflineTests(unittest.TestCase):
    def test_source_guard_no_devanagari_and_no_legacy_helpers(self):
        src = (
            Path(__file__).resolve().parent / "profile_extractor.py"
        ).read_text(encoding="utf-8")
        self.assertLessEqual(len(src.splitlines()), 270)
        self.assertIsNone(
            re.search(r"[\u0900-\u097F]", src),
            "profile_extractor.py must contain zero Devanagari characters",
        )
        for forbidden in ("_just_name", "_marital", "_as_children", "_turn_view", "_invoke_generator"):
            self.assertNotIn(forbidden, src)

    def test_arbitrary_five_children_with_decimal_infant_and_names(self):
        captured = {}

        async def fake_gen(**kwargs):
            captured.update(kwargs)
            return json.dumps(_full_raw(
                child_count=5,
                children=[
                    {"age_years": 14.0, "sex": "female", "name": "  Asha  "},
                    {"age_years": 10.0, "sex": "male", "name": None},
                    {"age_years": 6.0, "sex": "female", "name": "Riya"},
                    {"age_years": 3.0, "sex": "unknown", "name": "   "},
                    {"age_years": 0.5, "sex": "male", "name": "Munna"},
                ],
            ))

        state = apply_profile(
            apply_profile(new_cursor_state(), caller_name="पूजा")[0],
            marital_status="married",
        )[0]
        self.assertEqual(state["active_qid"], "intro_children")

        res = asyncio.run(
            profile_extractor.extract_profile_entities(
                state,
                utterance_text="5 kids",
                generator=fake_gen,
            )
        )
        self.assertTrue(res.ok)
        self.assertEqual(res.error_reason, "")
        self.assertEqual(res.updates["child_count"], 5)
        self.assertEqual(len(res.updates["children"]), 5)
        self.assertEqual(res.updates["children"][0]["name"], "Asha")
        self.assertIsNone(res.updates["children"][3]["name"])
        self.assertEqual(res.updates["children"][4]["age_years"], 0.5)

        cfg = captured["config"]
        self.assertEqual(profile_extractor.DEFAULT_MODEL, "gemini-3.5-flash-lite")
        self.assertEqual(profile_extractor.DEFAULT_ATTEMPT_TIMEOUT_SEC, 3.0)
        self.assertEqual(set(captured.keys()), {"model", "contents", "config"})
        self.assertEqual(captured["model"], "gemini-3.5-flash-lite")
        self.assertEqual(cfg.response_mime_type, "application/json")
        self.assertEqual(cfg.temperature, 0.0)
        self.assertIsNone(cfg.thinking_config)

        next_state, hold = apply_profile(state, **res.updates)
        self.assertIsNone(hold)
        self.assertEqual(next_state["active_qid"], "intro_bridge")
        stored = next_state["profile"]["children"]
        self.assertEqual(len(stored), 5)
        self.assertEqual(stored[0]["age_label_hi"], "14 साल")
        self.assertEqual(stored[4]["age_label_hi"], "6 महीने")
        self.assertEqual(stored[0]["name"], "Asha")
        self.assertIsNone(stored[3]["name"])

    def test_partial_ages_allowed_and_triggers_children_ages_hold(self):
        async def fake_gen(**_kwargs):
            return _full_raw(
                child_count=3,
                children=[{"age_years": 7.0, "sex": "female", "name": None}],
            )

        state = apply_profile(
            apply_profile(new_cursor_state(), caller_name="मीना")[0],
            marital_status="married",
        )[0]
        res = asyncio.run(
            profile_extractor.extract_profile_entities(
                state,
                utterance_text="three children, eldest is 7",
                generator=fake_gen,
            )
        )
        self.assertTrue(res.ok)
        self.assertEqual(res.updates["child_count"], 3)
        self.assertEqual(len(res.updates["children"]), 1)

        held_state, hold = apply_profile(state, **res.updates)
        self.assertEqual(hold, "children_ages")
        self.assertEqual(held_state["hold_reason"], "children_ages")
        self.assertEqual(held_state["active_qid"], "intro_children")
        self.assertIsNone(held_state["profile"]["ever_pregnant"])

    def test_name_fragment_returns_empty_updates_and_holds_name_fragment(self):
        async def fake_gen(**_kwargs):
            return _full_raw(caller_name=None)

        state = new_cursor_state()
        res = asyncio.run(
            profile_extractor.extract_profile_entities(
                state,
                utterance_text="fragment",
                generator=fake_gen,
            )
        )
        self.assertTrue(res.ok)
        self.assertEqual(res.updates, {})

        held_state, hold = apply_profile(state, **res.updates)
        self.assertEqual(hold, "name_fragment")
        self.assertEqual(held_state["hold_reason"], "name_fragment")
        self.assertIsNone(held_state["profile"]["caller_name"])

    def test_midcall_marital_correction_skips_child_and_pregnancy_gates(self):
        for status in ("about_to_marry", "unmarried"):
            async def fake_gen(st=status, **_kwargs):
                return _full_raw(marital_status=st)

            state = cursor_tests._clear_until(
                cursor_tests._profile(children=[cursor_tests._child(4, "male")]),
                "birth_nicu",
            )
            res = asyncio.run(
                profile_extractor.extract_profile_entities(
                    state,
                    utterance_text="marital correction",
                    generator=fake_gen,
                )
            )
            self.assertTrue(res.ok)
            self.assertEqual(res.updates, {"marital_status": status})

            updated = apply_interpreted_turn(
                state,
                {"current_outcome": "not_about_active", "profile_updates": res.updates},
            )
            self.assertEqual(updated["profile"]["marital_status"], status)
            self.assertEqual(updated["profile"]["children"], [])
            self.assertEqual(updated["answers"]["birth_nicu"]["state"], "not_applicable")
            self.assertEqual(updated["answers"]["preg_delivery"]["state"], "not_applicable")
            self.assertEqual(updated["answers"]["fam_miscarriage"]["state"], "not_applicable")
            self.assertEqual(updated["active_qid"], "preg_thalassemia")
            if status == "unmarried":
                self.assertEqual(updated["answers"]["fam_related"]["state"], "not_applicable")
            else:
                self.assertNotIn("fam_related", updated["answers"])

    def test_zero_children_and_whitespace_stripping(self):
        async def fake_gen(**_kwargs):
            return _full_raw(
                caller_name="  Sunita   Devi  ",
                child_count=0,
                children=[],
            )

        res = asyncio.run(
            profile_extractor.extract_profile_entities(
                active_qid="intro_children",
                active_prompt="children?",
                utterance_text="none",
                generator=fake_gen,
            )
        )
        self.assertTrue(res.ok)
        self.assertEqual(res.updates["caller_name"], "Sunita Devi")
        self.assertEqual(res.updates["child_count"], 0)
        self.assertEqual(res.updates["children"], [])

    def test_audio_bytes_used_only_when_utterance_empty(self):
        seen = {}

        async def fake_gen(**kwargs):
            seen.update(kwargs)
            return _full_raw(ever_pregnant=False)

        wav = b"RIFF\x00\x00\x00\x00WAVEfmt "
        res = asyncio.run(
            profile_extractor.extract_profile_entities(
                active_qid="ever_pregnant",
                active_prompt="ever pregnant?",
                utterance_text="   ",
                audio_wav_bytes=wav,
                generator=fake_gen,
            )
        )
        self.assertTrue(res.ok)
        self.assertEqual(res.updates, {"ever_pregnant": False})
        self.assertIsInstance(seen["contents"], list)
        self.assertEqual(len(seen["contents"]), 2)
        audio_part = seen["contents"][1]
        self.assertEqual(audio_part.inline_data.data, wav)
        self.assertEqual(audio_part.inline_data.mime_type, "audio/wav")

    def test_strict_schema_rejections(self):
        bad_payloads = [
            "not-a-dict",
            [],
            _full_raw(outcome="clear"),
            _full_raw(qid="intro_name"),
            _full_raw(target_qid="birth_nicu"),
            _full_raw(phase="closing"),
            _full_raw(close_reason="complete"),
            _full_raw(emergency=True),
            _full_raw(say="hello"),
            _full_raw(marital_status="divorced"),
            _full_raw(child_count=-1),
            _full_raw(child_count=True),
            _full_raw(ever_pregnant="yes"),
            _full_raw(child_count=0, children=[{"age_years": 2.0, "sex": "male"}]),
            _full_raw(
                child_count=1,
                children=[
                    {"age_years": 2.0, "sex": "male"},
                    {"age_years": 4.0, "sex": "female"},
                ],
            ),
            _full_raw(children=[{"age_years": -0.5, "sex": "male"}]),
            _full_raw(children=[{"age_years": float("nan"), "sex": "male"}]),
            _full_raw(children=[{"age_years": float("inf"), "sex": "male"}]),
            _full_raw(children=[{"age_years": 3.0, "sex": "boy"}]),
            _full_raw(children=[{"age_years": 3.0, "sex": "male", "age_label_hi": "3"}]),
        ]
        for bad in bad_payloads:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    profile_extractor.validate_profile_payload(bad)

        async def run_bad():
            async def bad_gen(**_kwargs):
                return _full_raw(target_qid="birth_nicu", marital_status="married")

            res = await profile_extractor.extract_profile_entities(
                active_qid="intro_married",
                active_prompt="married?",
                utterance_text="yes",
                generator=bad_gen,
            )
            self.assertFalse(res.ok)
            self.assertEqual(res.error_reason, "invalid_schema")
            self.assertEqual(res.updates, {})

        asyncio.run(run_bad())

    def test_timeout_invalid_json_and_model_override(self):
        async def run():
            async def slow_gen(**_kwargs):
                await asyncio.sleep(0.2)
                return _full_raw(caller_name="A")

            res_to = await profile_extractor.extract_profile_entities(
                active_qid="intro_name",
                active_prompt="name?",
                utterance_text="A",
                attempt_timeout_sec=0.02,
                generator=slow_gen,
            )
            self.assertFalse(res_to.ok)
            self.assertEqual(res_to.error_reason, "timeout")

            async def bad_json_gen(**_kwargs):
                return "{not json"

            res_json = await profile_extractor.extract_profile_entities(
                active_qid="intro_name",
                active_prompt="name?",
                utterance_text="A",
                generator=bad_json_gen,
            )
            self.assertFalse(res_json.ok)
            self.assertEqual(res_json.error_reason, "invalid_json")

            seen_model = []

            async def model_gen(**kwargs):
                seen_model.append(kwargs["model"])
                return _full_raw(caller_name="Kiran")

            with mock.patch.dict(
                os.environ, {"PROFILE_EXTRACTOR_MODEL": "gemini-custom-test"}
            ):
                res_m = await profile_extractor.extract_profile_entities(
                    active_qid="intro_name",
                    active_prompt="name?",
                    utterance_text="Kiran",
                    generator=model_gen,
                )
            self.assertTrue(res_m.ok)
            self.assertEqual(seen_model, ["gemini-custom-test"])

        asyncio.run(run())

    def test_client_cached_per_api_key_and_sync_generator_supported(self):
        seen_sync = {}

        def sync_gen(model, contents, config):
            seen_sync["args"] = (model, contents, config)
            return _full_raw(caller_name="Meera")

        res_sync = asyncio.run(
            profile_extractor.extract_profile_entities(
                active_qid="intro_name",
                active_prompt="name?",
                utterance_text="Meera",
                generator=sync_gen,
            )
        )
        self.assertTrue(res_sync.ok)
        self.assertEqual(res_sync.updates, {"caller_name": "Meera"})
        self.assertEqual(seen_sync["args"][0], "gemini-3.5-flash-lite")

        async def fake_generate_content(*, model, contents, config):
            return json.dumps(_full_raw(caller_name="Sita"))

        fake_client = mock.MagicMock()
        fake_client.aio.models.generate_content = mock.AsyncMock(
            side_effect=fake_generate_content
        )

        with mock.patch.dict(profile_extractor._clients, {}, clear=True), mock.patch.object(
            profile_extractor.genai, "Client", return_value=fake_client
        ) as client_ctor:
            res1 = asyncio.run(
                profile_extractor.extract_profile_entities(
                    active_qid="intro_name",
                    active_prompt="name?",
                    utterance_text="Sita",
                    api_key="test-cached-key",
                )
            )
            res2 = asyncio.run(
                profile_extractor.extract_profile_entities(
                    active_qid="intro_name",
                    active_prompt="name?",
                    utterance_text="Sita",
                    api_key="test-cached-key",
                )
            )
            self.assertTrue(res1.ok)
            self.assertTrue(res2.ok)
            client_ctor.assert_called_once_with(api_key="test-cached-key")
            fake_client.close.assert_not_called()
            fake_client.aio.aclose.assert_not_called()


if __name__ == "__main__":
    unittest.main()
