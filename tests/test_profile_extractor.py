"""100% offline unit tests for Phase 5 profile_extractor."""

from __future__ import annotations

import asyncio
import json
import os
import re
import unittest
from unittest import mock

from tests import APP_DIR
from tests.helpers import clear_until, make_child, make_profile_state

import profile_extractor  # noqa: E402
from call_cursor import apply_interpreted_turn, apply_profile, new_cursor_state  # noqa: E402


def _full_raw(**overrides) -> dict:
    return {"caller_name": None, "marital_status": None, "child_count": None, "children": None, **overrides}


class ProfileExtractorOfflineTests(unittest.TestCase):
    def test_source_guard_no_devanagari(self):
        src = (APP_DIR / "profile_extractor.py").read_text(encoding="utf-8")
        self.assertIsNone(re.search(r"[\u0900-\u097F]", src), "profile_extractor.py must contain zero Devanagari")

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

        state = apply_profile(apply_profile(new_cursor_state(), caller_name="पूजा")[0], marital_status="married")[0]
        self.assertEqual(state["active_qid"], "intro_children")
        res = asyncio.run(profile_extractor.extract_profile_entities(state, utterance_text="5 kids", generator=fake_gen))
        self.assertTrue(res.ok)
        self.assertEqual(res.updates["child_count"], 5)
        self.assertEqual(len(res.updates["children"]), 5)
        self.assertEqual(res.updates["children"][0]["name"], "Asha")
        self.assertIsNone(res.updates["children"][3]["name"])
        self.assertEqual(res.updates["children"][4]["age_years"], 0.5)

        cfg = captured["config"]
        self.assertEqual(set(captured.keys()), {"model", "contents", "config"})
        self.assertEqual(captured["model"], profile_extractor.DEFAULT_MODEL)
        self.assertEqual(cfg.response_mime_type, "application/json")
        self.assertEqual(cfg.temperature, 0.0)
        self.assertIsNone(cfg.thinking_config)

        next_state, hold = apply_profile(state, **res.updates)
        self.assertIsNone(hold)
        self.assertEqual(next_state["active_qid"], "intro_bridge")
        stored = next_state["profile"]["children"]
        self.assertEqual(len(stored), 5)
        self.assertEqual((stored[0]["age_label_hi"], stored[4]["age_label_hi"]), ("14 साल", "6 महीने"))
        self.assertEqual(stored[0]["name"], "Asha")
        self.assertIsNone(stored[3]["name"])

    def test_partial_ages_hold_and_pending_child_context_preservation(self):
        self.assertEqual(profile_extractor.validate_profile_payload(_full_raw(child_count=None, children=[])), {"child_count": 0, "children": []})

        state = apply_profile(apply_profile(new_cursor_state(), caller_name="पूजा")[0], marital_status="married")[0]
        held_state, hold = apply_profile(state, child_count=2, children=[{"age_years": 7.0, "sex": "female", "name": "रिया"}])
        self.assertEqual((hold, held_state["hold_reason"], held_state["active_qid"]), ("children_ages", "children_ages", "intro_children"))
        self.assertIsNone(held_state["profile"]["ever_pregnant"])
        self.assertEqual(held_state["profile"]["pending_child_count"], 2)
        self.assertEqual(held_state["profile"]["pending_children"][0]["name"], "रिया")

        seen_contents = []

        async def fake_turn2(**kwargs):
            seen_contents.append(kwargs["contents"])
            return _full_raw(
                child_count=2,
                children=[{"age_years": 7.0, "sex": "female", "name": "रिया"}, {"age_years": 3.0, "sex": "male", "name": None}],
            )

        res2 = asyncio.run(profile_extractor.extract_profile_entities(held_state, utterance_text="छोटा बेटा 3 साल का है", generator=fake_turn2))
        self.assertTrue(res2.ok)
        ctx_prof = json.loads(dict(line.split(": ", 1) for line in seen_contents[0].splitlines())["current_profile"])
        self.assertEqual((ctx_prof["caller_name"], ctx_prof["marital_status"], ctx_prof["pending_child_count"]), ("पूजा", "married", 2))
        self.assertEqual((ctx_prof["pending_children"][0]["age_years"], ctx_prof["pending_children"][0]["name"]), (7.0, "रिया"))

        resolved, hold2 = apply_profile(held_state, **res2.updates)
        self.assertIsNone(hold2)
        self.assertEqual((resolved["active_qid"], len(resolved["profile"]["children"])), ("intro_bridge", 2))
        self.assertNotIn("pending_children", resolved["profile"])
        self.assertTrue(resolved["profile"]["children_stated"])

    def test_name_fragment_zero_children_and_whitespace_stripping(self):
        async def empty_gen(**_kwargs):
            return _full_raw(caller_name=None)

        state = new_cursor_state()
        res = asyncio.run(profile_extractor.extract_profile_entities(state, utterance_text="fragment", generator=empty_gen))
        self.assertTrue(res.ok and res.updates == {})
        held, hold = apply_profile(state, **res.updates)
        self.assertEqual((hold, held["hold_reason"], held["profile"]["caller_name"]), ("name_fragment", "name_fragment", None))

        async def zero_gen(**_kwargs):
            return _full_raw(caller_name="  Sunita   Devi  ", child_count=0, children=[])

        res_z = asyncio.run(profile_extractor.extract_profile_entities(active_qid="intro_children", active_prompt="children?", utterance_text="none", generator=zero_gen))
        self.assertEqual(res_z.updates, {"caller_name": "Sunita Devi", "child_count": 0, "children": []})

    def test_midcall_marital_correction_skips_child_and_pregnancy_gates(self):
        for status in ("about_to_marry", "unmarried"):
            async def fake_gen(st=status, **_kwargs):
                return _full_raw(marital_status=st)

            state = clear_until(make_profile_state(children=[make_child(4, "male")]), "birth_nicu")
            res = asyncio.run(profile_extractor.extract_profile_entities(state, utterance_text="marital correction", generator=fake_gen))
            self.assertEqual((res.ok, res.updates), (True, {"marital_status": status}))

            updated = apply_interpreted_turn(state, {"current_outcome": "not_about_active", "profile_updates": res.updates})
            self.assertEqual((updated["profile"]["marital_status"], updated["profile"]["children"], updated["active_qid"]), (status, [], "preg_thalassemia"))
            for qid in ("birth_nicu", "preg_delivery", "fam_miscarriage"):
                self.assertEqual(updated["answers"][qid]["state"], "not_applicable")
            if status == "unmarried":
                self.assertEqual(updated["answers"]["fam_related"]["state"], "not_applicable")
            else:
                self.assertNotIn("fam_related", updated["answers"])

    def test_audio_bytes_used_only_when_utterance_empty(self):
        seen = {}

        async def fake_gen(**kwargs):
            seen.update(kwargs)
            return _full_raw(marital_status="married")

        wav = b"RIFF\x00\x00\x00\x00WAVEfmt "
        res = asyncio.run(profile_extractor.extract_profile_entities(active_qid="intro_married", active_prompt="married?", utterance_text="   ", audio_wav_bytes=wav, generator=fake_gen))
        self.assertTrue(res.ok and res.updates == {"marital_status": "married"})
        self.assertIsInstance(seen["contents"], list)
        self.assertEqual(len(seen["contents"]), 2)
        self.assertEqual((seen["contents"][1].inline_data.data, seen["contents"][1].inline_data.mime_type), (wav, "audio/wav"))

    def test_strict_schema_rejections(self):
        bad_payloads = [
            "not-a-dict", [], _full_raw(outcome="clear"), _full_raw(qid="intro_name"),
            _full_raw(target_qid="birth_nicu"), _full_raw(phase="closing"), _full_raw(close_reason="complete"),
            _full_raw(emergency=True), _full_raw(say="hello"), _full_raw(marital_status="divorced"),
            _full_raw(child_count=-1), _full_raw(child_count=True), _full_raw(ever_pregnant=True),
            _full_raw(ever_pregnant=False), _full_raw(ever_pregnant="yes"),
            _full_raw(child_count=0, children=[{"age_years": 2.0, "sex": "male"}]),
            _full_raw(child_count=1, children=[{"age_years": 2.0, "sex": "male"}, {"age_years": 4.0, "sex": "female"}]),
            _full_raw(children=[{"age_years": -0.5, "sex": "male"}]),
            _full_raw(children=[{"age_years": float("nan"), "sex": "male"}]),
            _full_raw(children=[{"age_years": float("inf"), "sex": "male"}]),
            _full_raw(children=[{"age_years": 3.0, "sex": "boy"}]),
            _full_raw(children=[{"age_years": 3.0, "sex": "male", "age_label_hi": "3"}]),
        ]
        for bad in bad_payloads:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                profile_extractor.validate_profile_payload(bad)

        async def bad_gen(**_kwargs):
            return _full_raw(target_qid="birth_nicu", marital_status="married")

        res = asyncio.run(profile_extractor.extract_profile_entities(active_qid="intro_married", active_prompt="married?", utterance_text="yes", generator=bad_gen))
        self.assertEqual((res.ok, res.error_reason, res.updates), (False, "invalid_schema", {}))

    def test_timeout_invalid_json_and_model_override(self):
        async def run():
            async def slow_gen(**_kwargs):
                await asyncio.sleep(0.2)
                return _full_raw(caller_name="A")

            res_to = await profile_extractor.extract_profile_entities(active_qid="intro_name", active_prompt="name?", utterance_text="A", attempt_timeout_sec=0.02, generator=slow_gen)
            self.assertEqual((res_to.ok, res_to.error_reason), (False, "timeout"))

            res_json = await profile_extractor.extract_profile_entities(active_qid="intro_name", active_prompt="name?", utterance_text="A", generator=lambda **_kw: "{not json")
            self.assertEqual((res_json.ok, res_json.error_reason), (False, "invalid_json"))

            seen_model = []

            async def model_gen(**kwargs):
                seen_model.append(kwargs["model"])
                return _full_raw(caller_name="Kiran")

            with mock.patch.dict(os.environ, {"PROFILE_EXTRACTOR_MODEL": "gemini-custom-test"}):
                res_m = await profile_extractor.extract_profile_entities(active_qid="intro_name", active_prompt="name?", utterance_text="Kiran", generator=model_gen)
            self.assertTrue(res_m.ok and seen_model == ["gemini-custom-test"])

        asyncio.run(run())

    def test_client_cached_per_api_key_and_sync_generator_supported(self):
        seen_sync = {}

        def sync_gen(model, contents, config):
            seen_sync["args"] = (model, contents, config)
            return _full_raw(caller_name="Meera")

        res_sync = asyncio.run(profile_extractor.extract_profile_entities(active_qid="intro_name", active_prompt="name?", utterance_text="Meera", generator=sync_gen))
        self.assertEqual((res_sync.ok, res_sync.updates, seen_sync["args"][0]), (True, {"caller_name": "Meera"}, profile_extractor.DEFAULT_MODEL))

        fake_client = mock.MagicMock()
        fake_client.aio.models.generate_content = mock.AsyncMock(return_value=json.dumps(_full_raw(caller_name="Sita")))
        with mock.patch.dict(profile_extractor._clients, {}, clear=True), mock.patch.object(profile_extractor.genai, "Client", return_value=fake_client) as client_ctor:
            res1 = asyncio.run(profile_extractor.extract_profile_entities(active_qid="intro_name", active_prompt="name?", utterance_text="Sita", api_key="test-cached-key"))
            res2 = asyncio.run(profile_extractor.extract_profile_entities(active_qid="intro_name", active_prompt="name?", utterance_text="Sita", api_key="test-cached-key"))
            self.assertTrue(res1.ok and res2.ok)
            client_ctor.assert_called_once_with(api_key="test-cached-key")
            fake_client.close.assert_not_called()
            fake_client.aio.aclose.assert_not_called()


if __name__ == "__main__":
    unittest.main()
