"""Post-call read of a saved transcript. The live call does not use this."""

import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv

import checklist
import db
import question_flow
import summarizer

load_dotenv(Path(__file__).resolve().parents[2] / ".env")
load_dotenv()

_ANSWERS = ("yes", "no", "unknown", "not_asked")
_VERSION = 6

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
For every yes, no, or unknown answer, evidence MUST be a non-empty, verbatim contiguous quote from the user turn that answers that topic. Do not translate, paraphrase, combine separate turns, or reuse a reply from a different question. A one-word answer such as "नहीं" is valid only for the topic asked just before that reply. If that question's answer has no exact user quote, use not_asked.

You may receive saved live-call findings after the transcript. They are fallible, broad notes from the call and may help identify the relevant topic. The transcript is authoritative: use a finding only when the user turn that answers that topic supports it.

Return JSON only: {"answers": {"topic_id": {"answer": "no", "evidence": "short quote"}}}
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


def _live_findings_text(call: dict) -> str:
    lines = []
    for domain, finding in (call.get("findings") or {}).items():
        if not isinstance(finding, dict):
            continue
        status = str(finding.get("status") or "").strip()
        detail = str(finding.get("detail") or "").strip()
        if status or detail:
            lines.append(f"{domain}: status={status}; detail={detail}")
    return "\n".join(lines) or "(none saved)"


# Grammar and acknowledgements. These are not question identity.
_STOP = {
    "theek", "thik", "hai", "hain", "ji", "achha", "accha", "kya", "mein", "main",
    "aur", "ya", "se", "par", "ko", "ka", "ke", "ki", "tha", "thi", "hua", "hui",
    "hue", "aap", "aapke", "aapki", "aapka", "kisi", "koi", "nahin", "nahi",
    "yeh", "woh", "toh", "bhi", "hota", "hoti", "hote", "liye", "tak", "bahut",
    "abhi", "bas", "gayi", "gaya", "gaye", "sakti", "sakta", "karte", "karti",
    "ठीक", "जी", "है", "हैं", "क्या", "में", "की", "के", "का", "को", "से", "या",
    "और", "था", "थी", "हुआ", "हुई", "किसी", "कोई", "यह", "तो", "भी", "नहीं",
    "बहुत", "अभी", "गई", "गया", "सकती", "सकता",
}


def _fold(text: str) -> str:
    """Casefold and drop virama so loose spellings of the same word still match."""
    return (text or "").replace("\u094d", "").casefold()


def _tokens(text: str) -> set[str]:
    found = set(re.findall(r"[a-z]{4,}|[\u0900-\u097f]{2,}", _fold(text)))
    return {token for token in found if token not in _STOP}


def _topic_blob(topic_id: str) -> str:
    topic = checklist._BY_ID[topic_id]
    return " ".join(
        (
            str(topic.get("words") or ""),
            question_flow.WIDE_WORDS.get(topic_id, ""),
            checklist.question_hi(topic_id),
        )
    )


_TOPIC_TOKENS = {topic["id"]: _tokens(_topic_blob(topic["id"])) for topic in checklist.TOPICS}
_TOKEN_OWNERS: dict[str, set[str]] = {}
for _topic_id, _topic_tokens in _TOPIC_TOKENS.items():
    for _token in _topic_tokens:
        _TOKEN_OWNERS.setdefault(_token, set()).add(_topic_id)


def _exact_question(assistant: str, topic_id: str) -> bool:
    text = _fold(assistant)
    topic = checklist._BY_ID[topic_id]
    phrases = (
        checklist.question_hi(topic_id).rstrip(" ?？"),
        str(topic.get("words") or ""),
        question_flow.WIDE_WORDS.get(topic_id, ""),
    )
    for phrase in phrases:
        folded = _fold(phrase).strip(" ?？")
        if len(folded) >= 12 and folded in text:
            return True
    return False


def _score_topic(assistant_tokens: set[str], topic_id: str) -> tuple[int, int]:
    found = assistant_tokens & _TOPIC_TOKENS[topic_id]
    distinctive = sum(1 for token in found if _TOKEN_OWNERS.get(token) == {topic_id})
    return distinctive, len(found)


def _match_topic(assistant: str) -> str | None:
    """Pick the one checklist question this assistant turn asked.

    Uses the question text that already belongs to the topic. A turn that
    matches two topics equally is left unmatched.
    """
    exact = [
        topic["id"]
        for topic in checklist.TOPICS
        if _exact_question(assistant, topic["id"])
    ]
    if len(exact) == 1:
        return exact[0]
    tokens = _tokens(assistant)
    ranked = sorted(
        ((_score_topic(tokens, topic["id"]), topic["id"]) for topic in checklist.TOPICS),
        reverse=True,
    )
    (best_distinct, best_total), best_id = ranked[0]
    (next_distinct, next_total), _next_id = ranked[1]
    if best_total < 2 or (best_distinct, best_total) == (next_distinct, next_total):
        return None
    if best_distinct == 0 and best_total < 3:
        return None
    return best_id


def _answer_turns(transcript: list) -> dict[str, list[str]]:
    """Map each topic to the user turns that answer its question."""
    paired: dict[str, list[str]] = {}
    assistant: str | None = None
    users: list[str] = []

    def flush() -> None:
        nonlocal assistant, users
        if assistant:
            topic_id = _match_topic(assistant)
            if topic_id:
                bucket = paired.setdefault(topic_id, [])
                for text in users:
                    if text.strip() and text not in bucket:
                        bucket.append(text)
        assistant = None
        users = []

    for turn in transcript or []:
        role = str(turn.get("role") or "")
        text = str(turn.get("text") or "")
        if role == "assistant":
            flush()
            assistant = text
            users = []
        elif role == "user" and assistant is not None:
            users.append(text)
    flush()
    return paired


def _quote_in_text(quote: str, text: str) -> bool:
    normalized_quote = " ".join(_fold(quote).split())
    if not normalized_quote:
        return False
    return normalized_quote in " ".join(_fold(text).split())


def validate_answers(call: dict, extracted: dict) -> dict:
    """Keep a verdict only when its quote is in the user turn for that topic."""
    transcript = list((call or {}).get("transcript") or [])
    turns = _answer_turns(transcript)
    answers: dict[str, dict] = {}
    for topic_id, entry in ((extracted or {}).get("answers") or {}).items():
        if topic_id not in checklist._BY_ID or not isinstance(entry, dict):
            continue
        answer = str(entry.get("answer") or "")
        evidence = str(entry.get("evidence") or "").strip()
        matched = any(_quote_in_text(evidence, text) for text in turns.get(topic_id, []))
        if answer in {"yes", "no", "unknown"} and not matched:
            answer, evidence = "not_asked", ""
        if answer in _ANSWERS:
            answers[topic_id] = {"answer": answer, "evidence": evidence[:240]}
    return {"version": _VERSION, "answers": answers}


def report_content(extracted: dict) -> dict:
    """Positive lines and doctor questions from validated yes answers only."""
    yes: list[str] = []
    questions: list[str] = []
    answers = (extracted or {}).get("answers") or {}
    for topic in checklist.TOPICS:
        topic_id = topic["id"]
        if topic_id.startswith("intro_") or topic_id == "close_anything_else":
            continue
        entry = answers.get(topic_id) or {}
        if not isinstance(entry, dict) or str(entry.get("answer") or "") != "yes":
            continue
        evidence = str(entry.get("evidence") or "").strip()
        if not evidence:
            continue
        question = checklist.question_hi(topic_id).strip()
        yes.append(f'{question} — माँ का जवाब: “{evidence}”')
        questions.append(
            f'{question} माँ ने कहा “{evidence}”. क्या डॉक्टर या ASHA/ANM दीदी से इसकी जाँच पूछें?'
        )
    return {"yes": yes, "questions": questions}


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
    return {"version": _VERSION, "answers": answers}


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
    for topic in checklist.TOPICS:
        if topic["id"] not in answers:
            answers[topic["id"]] = {"answer": "not_asked", "evidence": ""}
    return validate_answers(call, {"version": _VERSION, "answers": answers})


def _user_content(call: dict) -> str:
    return (
        "Topics:\n"
        + _topics_brief()
        + "\n\nTranscript:\n"
        + _transcript_text(call.get("transcript") or [])
        + "\n\nSaved live-call findings (hints only; transcript evidence wins):\n"
        + _live_findings_text(call)
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


async def cached_answers(call: dict) -> dict:
    call_id = int(call["id"])
    saved = db.get_answers(call_id)
    if isinstance(saved, dict) and isinstance(saved.get("answers"), dict):
        result = validate_answers(call, saved)
        if (
            saved.get("version") != _VERSION
            or "yes_hindi" in saved
            or "questions_hindi" in saved
        ):
            db.save_answers(call_id, result)
        return result
    try:
        result = await _from_llm(call)
    except Exception:
        result = fallback_from_call(call)
    result = validate_answers(call, result)
    result["version"] = _VERSION
    db.save_answers(call_id, result)
    return result
