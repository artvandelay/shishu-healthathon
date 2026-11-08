"""Post-call read of a saved transcript. The live call does not use this."""

import json
import os
from pathlib import Path

from dotenv import load_dotenv

import checklist
import call_state
import db
import summarizer

load_dotenv(Path(__file__).resolve().parents[2] / ".env")
load_dotenv()

_ANSWERS = ("yes", "no", "unknown", "not_asked")
_VERSION = 4

_DOMAIN_TOPIC = {
    "birth_nicu": "birth_nicu",
    "milestones": "body_fits_milestones",
    "bleeding_joints": "body_joint_swelling",
    "metabolic_fasting": "ill_fasting",
    "infections": "ill_admissions",
    "pregnancy": "preg_conditions",
    "consanguinity": "fam_related",
    "family_infant_death": "fam_infant_death",
    "family_history": "fam_fits_delay",
}

_PROMPT = """
You read a finished SHISHU screening call. You do not talk to the mother.

For every topic id, set answer to one of: yes, no, unknown, not_asked.
- not_asked: the bot never asked it, and she did not answer it in her own story.
- unknown: she said she does not know, or does not remember.
- no: she said it did not happen, or said it was normal. A normal delivery, a full-term birth, a normal newborn hearing/vision test, growing like other children, a normal scan, a delivery without problems, being married, and having children are no, not yes.
- yes: only when a screened clinical concern or positive history happened (for example: C-section or preterm birth, NICU or blue light, blood transfusions, low hemoglobin, knee swelling, high BP/sugar/thyroid or transfusion in pregnancy, consanguinity, miscarriage or infant death, or a family member needing transfusions).
A later clear answer replaces an earlier one. Do not invent. Intro topics (name, marriage, children, bridge) and close_anything_else are never yes.

yes_hindi is a list of complete, self-contained Hindi (Devanagari) sentences — ONLY for topics where you set answer to "yes" (clinical concerns or positive findings a doctor should notice).
- Do NOT put marriage, number of children, normal hearing/vision test, normal growth, normal scan, uncomplicated delivery, or close_anything_else in yes_hindi. If a topic's answer is "no", it must NEVER appear in yes_hindi.
- Every item in yes_hindi MUST state BOTH what the question/topic was AND what the mother answered (for example: "जन्म के तरीके के सवाल पर माँ ने बताया कि बच्चे का जन्म 8 महीने में ऑपरेशन से हुआ था क्योंकि बच्चा बहुत छोटा था।" or "बच्चे को खून चढ़ाने के सवाल पर माँ ने बताया कि अब तक चार बार खून चढ़ाना पड़ा है और हीमोग्लोबिन कम रहता है।").
- NEVER output a bare reply or raw transcript quote without the question context (never write "हाँ।", "हां।", "ऑपरेशन", "वह 8 महीने में पैदा हो गया था।", or "मेरे कैसे तो मौसी होती है।").
- Write in clear, natural Hindi (Devanagari). No disease names. No nos. No English.

questions_hindi is 3 to 6 Hindi questions she should ask a doctor or ASHA/ANM at the next visit.
Each item is one full question and ends with ?.
Base them on the yes answers. Do not name a disease. Do not say this is a diagnosis.

Return JSON only: {"answers": {"topic_id": {"answer": "no", "evidence": "short quote"}}, "yes_hindi": ["..."], "questions_hindi": ["..."]}
""".strip()


def _topics_brief() -> str:
    lines = []
    for topic in checklist.TOPICS:
        q_hi = checklist.question_hi(topic["id"])
        lines.append(f"{topic['id']} ({q_hi}): {topic['need']}")
    return "\n".join(lines)


def _transcript_text(transcript: list) -> str:
    lines = []
    for turn in transcript or []:
        role = str(turn.get("role") or "")
        text = str(turn.get("text") or "").strip()
        if text:
            lines.append(f"{role}: {text}")
    return "\n".join(lines)


def parse_payload(text: str) -> dict:
    raw = (text or "").strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("extractor response was not an object")
    answers: dict[str, dict] = {}
    incoming = data.get("answers") or {}
    if not isinstance(incoming, dict):
        raise ValueError("answers was not an object")
    for key, value in incoming.items():
        topic_id = str(key)
        if topic_id not in checklist._BY_ID:
            continue
        if isinstance(value, str):
            answer, evidence = value, ""
        elif isinstance(value, dict):
            answer = str(value.get("answer") or "")
            evidence = str(value.get("evidence") or "")
        else:
            continue
        if answer not in _ANSWERS:
            continue
        answers[topic_id] = {"answer": answer, "evidence": evidence[:240]}
    yes_hindi: list[str] = []
    for item in data.get("yes_hindi") or []:
        line = summarizer._scrub_diagnostic_terms(str(item).strip())
        if line and line not in yes_hindi:
            yes_hindi.append(line)
    questions_hindi: list[str] = []
    for item in data.get("questions_hindi") or []:
        line = summarizer._scrub_diagnostic_terms(str(item).strip())
        if not line:
            continue
        if "?" not in line and "？" not in line:
            line = line.rstrip("।") + "?"
        if line not in questions_hindi:
            questions_hindi.append(line)
    return {
        "version": _VERSION,
        "answers": answers,
        "yes_hindi": yes_hindi,
        "questions_hindi": questions_hindi[:6],
    }


def fallback_from_call(call: dict) -> dict:
    answers: dict[str, dict] = {}
    findings = call.get("findings") or {}
    for domain, value in findings.items():
        if not isinstance(value, dict):
            continue
        topic_id = _DOMAIN_TOPIC.get(str(domain))
        if topic_id is None and str(domain) in checklist._BY_ID:
            topic_id = str(domain)
        if topic_id is None:
            continue
        status = str(value.get("status") or "").lower()
        detail = str(value.get("detail") or "")
        if summarizer._unknown_detail(detail):
            kind = "unknown"
        elif status == "positive" or value.get("is_red_flag"):
            kind = "yes"
        elif status == "negative":
            kind = "no"
        else:
            continue
        answers[topic_id] = {"answer": kind, "evidence": detail[:240]}
    coverage = list((call.get("telemetry") or {}).get("coverage") or [])
    for topic_id in coverage:
        if topic_id in checklist._BY_ID and topic_id not in answers:
            answers[topic_id] = {"answer": "unknown", "evidence": ""}
    for topic in checklist.TOPICS:
        if topic["id"] not in answers:
            answers[topic["id"]] = {"answer": "not_asked", "evidence": ""}
    yes_hindi = []
    for topic_id, entry in answers.items():
        if entry["answer"] != "yes":
            continue
        line = checklist.question_hi(topic_id).rstrip("?")
        if line.startswith("क्या "):
            line = line[4:]
        line = summarizer._scrub_diagnostic_terms(line.strip())
        if line and not topic_id.startswith("intro_"):
            yes_hindi.append(line)
    return {
        "version": _VERSION,
        "answers": answers,
        "yes_hindi": yes_hindi,
        "questions_hindi": [],
    }


def mother_summary(
    extracted: dict,
    findings: dict | None = None,
    red_flags: list | None = None,
) -> str:
    """Hindi question list for the mother PDF. Does not write the doctor note."""
    questions = [
        str(item).strip()
        for item in (extracted or {}).get("questions_hindi") or []
        if str(item).strip()
    ]
    if len(questions) < 3:
        return summarizer.default_mother_questions(findings, red_flags)
    numbered = "\n".join(f"{idx}. {question}" for idx, question in enumerate(questions[:6], start=1))
    return (
        f"{summarizer.MOTHER_QUESTIONS_HEADING}\n\n"
        f"{numbered}\n\n"
        f"{summarizer.MOTHER_FLAG_FOOTER}"
    )


def _user_content(call: dict) -> str:
    return (
        "Topics:\n"
        + _topics_brief()
        + "\n\nTranscript:\n"
        + _transcript_text(call.get("transcript") or [])
    )


async def _from_gemini(user_content: str) -> dict:
    from google import genai

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    try:
        response = await client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents=user_content,
            config={
                "system_instruction": _PROMPT,
                "response_mime_type": "application/json",
                "temperature": 0.2,
                "max_output_tokens": 8192,
            },
        )
        return parse_payload(response.text or "")
    finally:
        await client.aio.aclose()
        client.close()


async def _from_openai(user_content: str) -> dict:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _PROMPT},
                {"role": "user", "content": user_content},
            ],
        )
        return parse_payload(response.choices[0].message.content or "")
    finally:
        await client.close()


async def _from_llm(call: dict) -> dict:
    user_content = _user_content(call)
    try:
        return await _from_gemini(user_content)
    except Exception:
        return await _from_openai(user_content)


def structured_answers(call: dict) -> dict:
    """Version 5 answers taken from the saved call state. No model call."""
    topics = ((call.get("state") or {}).get("topics") or {})
    answers: dict[str, dict] = {}
    for topic in checklist.TOPICS:
        topic_id = topic["id"]
        saved = topics.get(topic_id) if isinstance(topics.get(topic_id), dict) else {}
        status = str(saved.get("state") or "not_asked")
        value = saved.get("value")
        details = saved.get("details") or {}
        if call_state._is_positive(saved):
            kind = "yes"
        elif status == "answered" and (
            value is False
            or str(details.get("result") or "").strip().lower() in {"normal", "negative"}
            or str(value or "").strip().lower() in {"no", "nahi", "normal", "negative"}
        ):
            kind = "no"
        elif status == "answered":
            kind = "answered"
        elif status in {"unknown", "no_response", "not_asked", "not_applicable"}:
            kind = status
        else:
            kind = "not_asked"
        answers[topic_id] = {
            "answer": kind,
            "details": details,
            "instances": saved.get("instances") or [],
            "source_turn_ids": saved.get("source_turn_ids") or [],
            "reason": saved.get("reason"),
        }
    return {
        "version": 5,
        "answers": answers,
        "yes_hindi": [],
        "questions_hindi": [],
    }


async def cached_answers(call: dict) -> dict:
    if isinstance(call.get("state"), dict) and isinstance(call["state"].get("topics"), dict):
        return structured_answers(call)
    call_id = int(call["id"])
    saved = db.get_answers(call_id)
    if (
        isinstance(saved, dict)
        and saved.get("version") == _VERSION
        and isinstance(saved.get("answers"), dict)
    ):
        return saved
    try:
        result = await _from_llm(call)
    except Exception:
        result = fallback_from_call(call)
    if isinstance(result, dict):
        result["version"] = _VERSION
    db.save_answers(call_id, result)
    return result
