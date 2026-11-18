"""RED-first contract tests for the lossless turn-evidence seam."""

from __future__ import annotations

import sqlite3
import unittest

import tests  # noqa: F401 — configure isolated storage before importing intake
from pydantic import ValidationError

from intake import db, engine
from intake.pack import load_pack
from intake.turns import EvidenceSource, TurnEvidence


PACK = load_pack("shishu-hi")


class TurnEvidenceContractTests(unittest.TestCase):

    def setUp(self) -> None:
        with db._LOCK:
            db.conn.execute("DELETE FROM turn_evidence")
            db.conn.execute("DELETE FROM calls")
            db.conn.commit()

    def tearDown(self) -> None:
        with db._LOCK:
            db.conn.execute("DROP TRIGGER IF EXISTS reject_turn_evidence")
            db.conn.execute("DELETE FROM turn_evidence")
            db.conn.execute("DELETE FROM calls")
            db.conn.commit()

    @staticmethod
    def evidence(step: str, text: str = "बुखार आया था") -> TurnEvidence:
        return TurnEvidence(
            source=EvidenceSource.VOICE,
            snapshot_step=step,
            transcript=text,
            received_at=1_795_000_000.25,
        )

    def test_evidence_is_an_immutable_source_record_not_a_form_answer(self):
        evidence = self.evidence("birth_bcg")

        self.assertEqual(evidence.source, EvidenceSource.VOICE)
        self.assertEqual(evidence.snapshot_step, "birth_bcg")
        self.assertFalse(hasattr(evidence, "outcome"))
        self.assertFalse(hasattr(evidence, "target_qid"))
        self.assertFalse(hasattr(evidence, "call_id"))
        self.assertFalse(hasattr(evidence, "sequence"))
        self.assertFalse(hasattr(evidence, "evidence_id"))
        with self.assertRaises(ValidationError):
            evidence.transcript = "rewritten"

    def test_turn_commit_persists_state_and_evidence_with_one_call_local_sequence(self):
        call_id = db.create_call(pack_id=PACK.id)
        before = engine.new_state(PACK)
        after = before.model_copy(
            update={"cursor": before.cursor.model_copy(update={"attempts": 1})}
        )

        stored = db.commit_turn(call_id, PACK, after, self.evidence(before.step_token))

        self.assertEqual(db.call_state_from_row(db.get_call(call_id)).to_dict(), after.to_dict())
        self.assertEqual(stored.sequence, 1)
        self.assertEqual(stored.call_id, call_id)
        self.assertEqual(db.list_turn_evidence(call_id), [stored])
        self.assertEqual(stored.snapshot_step, before.step_token)

    def test_turn_commit_rolls_back_state_when_the_evidence_insert_fails(self):
        call_id = db.create_call(pack_id=PACK.id)
        before = engine.new_state(PACK)
        db.save_cursor_state(call_id, PACK, before)
        after = before.model_copy(
            update={"cursor": before.cursor.model_copy(update={"attempts": 1})}
        )
        with db._LOCK:
            db.conn.execute(
                """
                CREATE TRIGGER reject_turn_evidence
                BEFORE INSERT ON turn_evidence
                BEGIN
                    SELECT RAISE(ABORT, 'forced evidence failure');
                END
                """
            )
            db.conn.commit()

        with self.assertRaises(sqlite3.IntegrityError):
            db.commit_turn(call_id, PACK, after, self.evidence(before.step_token))

        self.assertEqual(db.call_state_from_row(db.get_call(call_id)).to_dict(), before.to_dict())
        self.assertEqual(db.list_turn_evidence(call_id), [])


if __name__ == "__main__":
    unittest.main()
