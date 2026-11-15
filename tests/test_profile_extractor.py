"""Offline unit tests for the generative profile extractor."""

from __future__ import annotations

import asyncio
import json
import os
import re
import unittest
from pathlib import Path
from unittest import mock

from tests import APP_DIR

import engine
import profile_extractor
from models import (
    AnswerStatus,
    CallState,
    ChildInput,
    ChildSex,
    HoldReason,
    MaritalStatus,
    ProfilePatch,
    SubStep,
    TurnCommand,
    TurnOutcome,
)

_DEVANAGARI = re.compile(r"[\u0900-\u097F]")


def raw(**overrides) -> dict:
    """A complete extractor payload, with every field null unless overridden."""
    base = {"caller_name": None, "marital_status": None, "child_count": None, "children": None}
    base.update(overrides)
    return base


def intro(*, name: str = "पूजा", marital: MaritalStatus = MaritalStatus.MARRIED) -> CallState:
    state = engine.new_state()
    state = engine.reduce_turn(state, TurnCommand(profile_patch=ProfilePatch(caller_name=name)))
    return engine.reduce_turn(state, TurnCommand(profile_patch=ProfilePatch(marital_status=marital)))


def apply(state: CallState, patch: ProfilePatch, *, attempted: bool = False) -> CallState:
    return engine.reduce_turn(
        state, TurnCommand(profile_patch=patch, profile_turn_attempted=attempted)
    )


def extract(state: CallState | None = None, utterance: str = "", **kwargs):
    return asyncio.run(profile_extractor.extract_profile_entities(state, utterance_text=utterance, **kwargs))


class SourceGuardTests(unittest.TestCase):
    def test_no_devanagari_in_the_extractor_source(self):
        source = (APP_DIR / "profile_extractor.py").read_text(encoding="utf-8")
        self.assertIsNone(_DEVANAGARI.search(source))

    def test_extractor_never_imports_the_legacy_cursor(self):
        source = (APP_DIR / "profile_extractor.py").read_text(encoding="utf-8")
        legacy = "call" + "_cursor"
        self.assertNotIn(f"import {legacy}", source)


class ValidationTests(unittest.TestCase):
    def test_full_payload_becomes_a_typed_patch(self):
        patch = profile_extractor.validate_profile_payload(
            raw(
                caller_name="  Asha  ",
                marital_status="married",
                child_count=2,
                children=[
                    {"age_years": 5, "sex": "female", "name": "  Riya  "},
                    {"age_years": 0.5, "sex": "male", "name": "   "},
                ],
            )
        )
        self.assertEqual(patch.caller_name, "Asha")
        self.assertIs(patch.marital_status, MaritalStatus.MARRIED)
        self.assertEqual(patch.child_count, 2)
        self.assertEqual([child.age_years for child in patch.children], [5.0, 0.5])
        self.assertEqual(patch.children[0].name, "Riya")
        self.assertIsNone(patch.children[1].name)
        self.assertIs(patch.children[1].sex, ChildSex.MALE)

    def test_empty_children_list_without_count_means_zero_children(self):
        patch = profile_extractor.validate_profile_payload(raw(children=[]))
        self.assertEqual((patch.child_count, patch.children), (0, ()))

    def test_partial_ages_keep_the_count_with_fewer_children(self):
        patch = profile_extractor.validate_profile_payload(
            raw(child_count=3, children=[{"age_years": 7.0, "sex": "female"}])
        )
        self.assertEqual((patch.child_count, len(patch.children)), (3, 1))

    def test_all_null_payload_is_an_empty_patch(self):
        self.assertTrue(profile_extractor.validate_profile_payload(raw()).is_empty)

    def test_strict_rejections(self):
        bad_payloads = [
            "not-a-dict",
            [],
            raw(outcome="clear"),
            raw(qid="intro_name"),
            raw(target_qid="birth_nicu"),
            raw(emergency=True),
            raw(say="hello"),
            raw(marital_status="divorced"),
            raw(child_count=-1),
            raw(child_count=True),
            raw(ever_pregnant=True),
            raw(child_count=0, children=[{"age_years": 2.0, "sex": "male"}]),
            raw(child_count=1, children=[{"age_years": 2.0, "sex": "male"}, {"age_years": 4.0}]),
            raw(children=[{"age_years": -0.5, "sex": "male"}]),
            raw(children=[{"age_years": float("nan"), "sex": "male"}]),
            raw(children=[{"age_years": float("inf"), "sex": "male"}]),
            raw(children=[{"age_years": 3.0, "sex": "boy"}]),
            raw(children=[{"age_years": 3.0, "sex": "male", "age_label_hi": "3"}]),
            raw(children="two"),
        ]
        for bad in bad_payloads:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                profile_extractor.validate_profile_payload(bad)


class ExtractionTests(unittest.TestCase):
    def test_five_children_arrive_with_names_and_month_precision(self):
        captured = {}

        async def generator(**kwargs):
            captured.update(kwargs)
            return raw(
                child_count=5,
                children=[
                    {"age_years": 14.0, "sex": "female", "name": "  Asha  "},
                    {"age_years": 10.0, "sex": "male", "name": None},
                    {"age_years": 6.0, "sex": "female", "name": "Riya"},
                    {"age_years": 3.0, "sex": "unknown", "name": "   "},
                    {"age_years": 0.5, "sex": "male", "name": "Munna"},
                ],
            )

        state = intro()
        self.assertEqual(state.cursor.qid, "intro_children")
        result = extract(state, "5 kids", generator=generator)
        self.assertTrue(result.ok)
        self.assertEqual(result.patch.child_count, 5)
        self.assertEqual(len(result.patch.children), 5)
        self.assertEqual(result.patch.children[0].name, "Asha")
        self.assertIsNone(result.patch.children[3].name)

        self.assertEqual(set(captured), {"model", "contents", "config"})
        self.assertEqual(captured["model"], profile_extractor.DEFAULT_MODEL)
        self.assertEqual(captured["config"].response_mime_type, "application/json")
        self.assertEqual(captured["config"].temperature, 0.0)
        self.assertIsNone(captured["config"].thinking_config)

        next_state = apply(state, result.patch)
        self.assertIsNone(next_state.cursor.hold)
        self.assertEqual(next_state.cursor.qid, "intro_bridge")
        self.assertEqual(len(next_state.profile.children), 5)
        labels = [child.age_label_hi for child in next_state.profile.children]
        self.assertEqual((labels[0], labels[4]), ("14 साल", "6 महीने"))
        self.assertIsNone(next_state.profile.children[3].name)

    def test_partial_count_holds_then_pending_context_is_preserved(self):
        state = intro()
        held = apply(state, ProfilePatch(child_count=2, children=(ChildInput(age_years=7.0, sex=ChildSex.FEMALE, name="Riya"),)))
        self.assertIs(held.cursor.hold, HoldReason.CHILDREN_AGES)
        self.assertEqual(held.cursor.qid, "intro_children")
        self.assertIsNone(held.profile.ever_pregnant)
        self.assertEqual(held.profile.pending_child_count, 2)
        self.assertEqual(held.profile.pending_children[0].name, "Riya")

        seen = []

        async def generator(**kwargs):
            seen.append(kwargs["contents"])
            return raw(
                child_count=2,
                children=[
                    {"age_years": 7.0, "sex": "female", "name": "Riya"},
                    {"age_years": 3.0, "sex": "male", "name": None},
                ],
            )

        second = extract(held, "छोटा बेटा 3 साल का है", generator=generator)
        self.assertTrue(second.ok)
        context = dict(line.split(": ", 1) for line in str(seen[0]).splitlines())
        known = json.loads(context["current_profile"])
        self.assertEqual((known["caller_name"], known["pending_child_count"]), ("पूजा", 2))
        self.assertEqual(known["pending_children"][0]["age_years"], 7.0)

        resolved = apply(held, second.patch)
        self.assertIsNone(resolved.cursor.hold)
        self.assertEqual((resolved.cursor.qid, len(resolved.profile.children)), ("intro_bridge", 2))
        self.assertEqual(resolved.profile.pending_children, [])
        self.assertTrue(resolved.profile.children_stated)

    def test_name_fragment_holds_and_zero_children_resolves_the_gate(self):
        async def empty(**kwargs):
            return raw()

        fragment = extract(engine.new_state(), "fragment", generator=empty)
        self.assertTrue(fragment.ok)
        self.assertTrue(fragment.patch.is_empty)
        held = apply(engine.new_state(), fragment.patch, attempted=True)
        self.assertEqual((held.cursor.hold, held.profile.caller_name), (HoldReason.NAME_FRAGMENT, None))

        async def zero(**kwargs):
            return raw(caller_name="  Sunita   Devi  ", child_count=0, children=[])

        result = extract(active_qid="intro_children", active_prompt="children?", utterance="none", generator=zero)
        self.assertEqual(result.patch.caller_name, "Sunita Devi")
        self.assertEqual((result.patch.child_count, result.patch.children), (0, ()))

    def test_midcall_marital_correction_skips_child_and_pregnancy_gates(self):
        base = apply(
            apply(apply(engine.new_state(), ProfilePatch(caller_name="पूजा")), ProfilePatch(marital_status=MaritalStatus.MARRIED)),
            ProfilePatch(children=(ChildInput(age_years=4.0, sex=ChildSex.MALE),)),
        )
        state = base
        for _ in range(100):
            if state.cursor.qid == "birth_nicu" or state.phase.value != "asking":
                break
            state = engine.reduce_turn(state, TurnCommand(outcome=TurnOutcome.CLEAR))
        self.assertEqual(state.cursor.qid, "birth_nicu")

        for marital in (MaritalStatus.ABOUT_TO_MARRY, MaritalStatus.UNMARRIED):
            async def generator(m=marital, **kwargs):
                return raw(marital_status=m.value)

            result = extract(state, "marital correction", generator=generator)
            self.assertEqual(result.patch.marital_status, marital)
            updated = engine.reduce_turn(
                state,
                TurnCommand(profile_patch=result.patch, profile_turn_attempted=True),
            )
            self.assertIs(updated.profile.marital_status, marital)
            self.assertEqual(updated.profile.children, [])
            self.assertEqual(updated.cursor.qid, "preg_thalassemia")
            for qid in ("birth_nicu", "preg_delivery", "fam_miscarriage"):
                self.assertIs(updated.answers[qid].state, AnswerStatus.NOT_APPLICABLE)
            self.assertIs(
                updated.answers["fam_related"].state, AnswerStatus.NOT_APPLICABLE
            ) if marital is MaritalStatus.UNMARRIED else self.assertNotIn("fam_related", updated.answers)

    def test_audio_bytes_are_attached_only_when_text_is_empty(self):
        seen = {}

        async def generator(**kwargs):
            seen.update(kwargs)
            return raw(marital_status="married")

        wav = b"RIFF\x00\x00\x00\x00WAVEfmt "
        result = extract(
            active_qid="intro_married",
            active_prompt="married?",
            utterance="   ",
            audio_wav_bytes=wav,
            generator=generator,
        )
        self.assertTrue(result.ok)
        self.assertIs(result.patch.marital_status, MaritalStatus.MARRIED)
        self.assertEqual(len(seen["contents"]), 2)
        self.assertEqual(seen["contents"][1].inline_data.data, wav)
        self.assertEqual(seen["contents"][1].inline_data.mime_type, "audio/wav")

        seen.clear()
        extract(active_qid="intro_married", active_prompt="married?", utterance="yes", audio_wav_bytes=wav, generator=generator)
        self.assertIsInstance(seen["contents"], str)


class FailureTests(unittest.TestCase):
    def test_invalid_schema_timeout_and_invalid_json(self):
        async def bad(**kwargs):
            return raw(target_qid="birth_nicu", marital_status="married")

        result = extract(active_qid="intro_married", active_prompt="married?", utterance="yes", generator=bad)
        self.assertEqual((result.ok, result.error_reason), (False, "invalid_schema"))
        self.assertTrue(result.patch.is_empty)

        async def slow(**kwargs):
            await asyncio.sleep(0.2)
            return raw(caller_name="A")

        timed_out = extract(
            active_qid="intro_name", active_prompt="name?", utterance="A",
            attempt_timeout_sec=0.02, generator=slow,
        )
        self.assertEqual((timed_out.ok, timed_out.error_reason), (False, "timeout"))

        broken = extract(active_qid="intro_name", active_prompt="name?", utterance="A", generator=lambda **_: "{not json")
        self.assertEqual((broken.ok, broken.error_reason), (False, "invalid_json"))

    def test_empty_input_and_missing_key_short_circuit(self):
        empty = extract(utterance="   ")
        self.assertEqual((empty.ok, empty.error_reason), (False, "empty_input"))
        with mock.patch.dict(os.environ, {}, clear=True):
            missing = extract(active_qid="intro_name", active_prompt="name?", utterance="Kiran")
        self.assertEqual((missing.ok, missing.error_reason), (False, "missing_api_key"))

    def test_model_override_and_sync_generator(self):
        seen = {}

        def sync_generator(model, contents, config):
            seen.update(model=model, contents=contents, config=config)
            return raw(caller_name="Meera")

        result = extract(active_qid="intro_name", active_prompt="name?", utterance="Meera", generator=sync_generator)
        self.assertTrue(result.ok)
        self.assertEqual(result.patch.caller_name, "Meera")
        self.assertEqual(seen["model"], profile_extractor.DEFAULT_MODEL)

        models = []

        async def named(**kwargs):
            models.append(kwargs["model"])
            return raw(caller_name="Kiran")

        with mock.patch.dict(os.environ, {"PROFILE_EXTRACTOR_MODEL": "gemini-custom-test"}):
            overridden = extract(active_qid="intro_name", active_prompt="name?", utterance="Kiran", generator=named)
        self.assertTrue(overridden.ok)
        self.assertEqual(models, ["gemini-custom-test"])
        self.assertEqual(profile_extractor.resolve_model("explicit"), "explicit")

    def test_client_is_cached_per_api_key(self):
        client = mock.MagicMock()
        client.aio.models.generate_content = mock.AsyncMock(return_value=raw(caller_name="Sita"))
        with (
            mock.patch.dict(profile_extractor._clients, {}, clear=True),
            mock.patch.object(profile_extractor.genai, "Client", return_value=client) as build,
        ):
            first = extract(active_qid="intro_name", active_prompt="name?", utterance="Sita", api_key="test-cached-key")
            second = extract(active_qid="intro_name", active_prompt="name?", utterance="Sita", api_key="test-cached-key")
        self.assertTrue(first.ok and second.ok)
        build.assert_called_once_with(api_key="test-cached-key")


if __name__ == "__main__":
    unittest.main()
