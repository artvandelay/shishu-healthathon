"""Write the no-flag and incomplete sample PDFs. No network.

SHISHU-sample-red-flags.pdf is the Hindi mother PDF for call 253.
This script does not overwrite that file.
SHISHU-sample-red-flags-en.pdf is the English reviewer copy.
render_red_flags_en.py writes that file. This script does not.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "shishu-v2"))

import checklist  # noqa: E402
import mother_report  # noqa: E402

OUT = Path(__file__).resolve().parent / "samples"

def _answers(default: str, overrides: dict[str, str]) -> dict:
    answers = {}
    for topic in checklist.TOPICS:
        kind = overrides.get(topic["id"], default)
        answers[topic["id"]] = {"answer": kind, "evidence": ""}
    return {"version": 2, "answers": answers, "yes_hindi": []}


def _call(**extra) -> dict:
    base = {
        "id": 1,
        "mother_name": "",
        "child_name": "",
        "child_age": "",
        "child_gender": "",
        "started_at": "2026-11-06 08:40:00",
        "duration_sec": 600,
        "summary_mother": "",
        "findings": {},
        "telemetry": {"coverage": list(checklist.REQUIRED_IDS), "close_reason": ""},
        "status": "completed",
    }
    base.update(extra)
    return base


def _write(name: str, call: dict, extracted: dict) -> None:
    report = mother_report.build(call, extracted)
    pdf = mother_report.render_pdf(report)
    path = OUT / name
    path.write_bytes(pdf)
    print(f"{path.name} {len(pdf)} bytes")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    skip = {"body_puberty": "not_asked", "fam_same_problem": "not_asked"}
    _write(
        "SHISHU-sample-no-flags.pdf",
        _call(
            mother_name="Kavita",
            child_age="3 years",
            child_gender="female",
            started_at="2026-11-05 05:00:00",
            duration_sec=960,
        ),
        _answers("no", skip),
    )
    asked = {"birth_mode": "no", "birth_term": "no", "birth_nicu": "no"}
    _write(
        "SHISHU-sample-incomplete.pdf",
        _call(
            mother_name="Meera",
            child_age="1 year",
            child_gender="male",
            started_at="2026-11-07 10:32:00",
            duration_sec=370,
            telemetry={"coverage": list(asked), "close_reason": "caller_stop"},
        ),
        _answers("not_asked", asked),
    )


if __name__ == "__main__":
    main()
