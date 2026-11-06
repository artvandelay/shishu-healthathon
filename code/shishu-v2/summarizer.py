"""Post-call doctor and mother summaries."""

import json
import os
import re

from dotenv import load_dotenv

from prompts import DOCTOR_MOTHER_SUMMARY_PROMPT

load_dotenv("/Users/jigar/projects/0n-going/SHISHU-healthathon-koita/.env")

_REQUIRED_HEADINGS = (
    "### Patient Profile & Chief Complaint",
    "### Positive Red Flags & NRROID Gate Alerts",
    "### Birth & Neonatal History",
    "### Development, Growth & Physical Features",
    "### Systemic, Infections & Hematology",
    "### Antenatal (Pregnancy) History",
    "### Family & Pedigree History",
    "### Pertinent Negatives",
)


def _user_content(transcript: list[dict], findings: dict, red_flags: list[str]) -> str:
    payload = {
        "transcript": transcript,
        "findings": findings,
        "red_flags": red_flags,
    }
    return (
        "Write the post-call JSON from this call.\n"
        + json.dumps(payload, ensure_ascii=False)
    )


def _parse_json(text: str) -> dict:
    raw = (text or "").strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            raise
        data = json.loads(raw[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("summary response was not a JSON object")
    return data


def _flag_text(item: object) -> str:
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, dict):
        domain = str(item.get("domain") or "").strip()
        detail = str(item.get("detail") or "").strip()
        if domain and detail:
            return f"{domain}: {detail}"
        return detail or domain
    return str(item).strip()


def _validate(payload: dict) -> dict:
    doctor = payload.get("doctor_markdown")
    if not isinstance(doctor, str) or not doctor.strip():
        doctor = payload.get("summary_doctor")
    mother = payload.get("mother_hindi")
    if not isinstance(mother, str) or not mother.strip():
        mother = payload.get("summary_mother")
    if not isinstance(doctor, str) or not doctor.strip():
        raise ValueError("missing doctor_markdown")
    if not isinstance(mother, str) or not mother.strip():
        raise ValueError("missing mother_hindi")
    missing = [heading for heading in _REQUIRED_HEADINGS if heading not in doctor]
    if missing:
        raise ValueError("doctor_markdown missing headings: " + ", ".join(missing))
    extracted = payload.get("extracted_findings")
    if extracted is None:
        extracted = {}
    if not isinstance(extracted, dict):
        raise ValueError("extracted_findings was not an object")
    flags = payload.get("red_flags")
    if flags is None:
        flags = []
    if not isinstance(flags, list):
        raise ValueError("red_flags was not a list")
    return {
        "doctor_markdown": doctor.strip(),
        "mother_hindi": mother.strip(),
        "extracted_findings": extracted,
        "red_flags": flags,
    }


def _user_text(transcript: list[dict]) -> str:
    parts: list[str] = []
    for turn in transcript or []:
        role = str(turn.get("role") or "").lower()
        if role in ("user", "mother"):
            parts.append(str(turn.get("text") or ""))
    return " ".join(parts)


def _not_reported(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    detail = str(value.get("detail") or "").strip().lower()
    return (not detail) or "not reported" in detail or detail in {"not discussed", "none", "n/a"}


def _add_flag(flags: list[str], domain: str, detail: str) -> None:
    text = f"{domain}: {detail}".strip()
    if text and text not in flags:
        flags.append(text)


def _supplement_from_transcript(
    transcript: list[dict],
    findings: dict,
    red_flags: list[str],
) -> None:
    """Keep stated gates when the model drops them from red_flags."""
    text = _user_text(transcript)
    lowered = text.lower()

    def put(domain: str, detail: str, status: str, is_red_flag: bool) -> None:
        current = findings.get(domain)
        if not isinstance(current, dict) or _not_reported(current):
            findings[domain] = {
                "status": status,
                "detail": detail,
                "is_red_flag": is_red_flag,
            }
        elif is_red_flag:
            current["status"] = "positive"
            current["is_red_flag"] = True
            if not str(current.get("detail") or "").strip():
                current["detail"] = detail
        if is_red_flag:
            detail_text = str(findings[domain].get("detail") or detail)
            _add_flag(red_flags, domain, detail_text)

    if re.search(r"\bbeta\b", lowered) or "बेटा" in text:
        put("profile_child_gender", "Male. Mother said beta.", "info", False)
    elif re.search(r"\bbeti\b", lowered) or "बेटी" in text:
        put("profile_child_gender", "Female. Mother said beti.", "info", False)

    knee = re.search(
        r"(ghutn\w*|knee|joint).{0,48}(sujan|soojan|sujan|swelling)"
        r"|(sujan|soojan|sujan|swelling).{0,48}(ghutn\w*|knee|joint)",
        lowered,
    )
    if knee:
        put(
            "bleeding_joints",
            "Knee or joint swelling after a fall, resolving over days.",
            "positive",
            True,
        )

    infant_death = re.search(r"guzar gaya|guzar gayi|mar gaya|mrityu|passed away", lowered)
    young = re.search(r"mahine|month|saal se kam|under 1|1 saal", lowered)
    if infant_death and young:
        put(
            "family_infant_death",
            "A previous child died in infancy. Fits were mentioned."
            if re.search(r"fits|jhatke", lowered)
            else "A previous child died in infancy.",
            "positive",
            True,
        )


def _fill_red_flag_section(doctor: str, red_flags: list[str]) -> str:
    if not red_flags:
        return doctor
    pattern = (
        r"(### Positive Red Flags & NRROID Gate Alerts\n)"
        r"(.*?)(?=\n### )"
    )
    match = re.search(pattern, doctor, flags=re.S)
    if not match:
        return doctor
    body = match.group(2)
    if "not reported" not in body.lower():
        return doctor
    lines = "\n".join(f"- {flag}" for flag in red_flags)
    return doctor[: match.start(2)] + lines + "\n" + doctor[match.end(2) :]


def _assemble(payload: dict, findings: dict, red_flags: list[str], transcript: list[dict]) -> dict:
    cleaned = _validate(payload)
    merged_findings = dict(findings or {})
    for key, value in cleaned["extracted_findings"].items():
        if key not in merged_findings and not _not_reported(value):
            merged_findings[key] = value
    merged_flags: list[str] = []
    for item in list(red_flags or []) + list(cleaned["red_flags"]):
        text = _flag_text(item)
        if text and text not in merged_flags:
            merged_flags.append(text)
    for domain, value in merged_findings.items():
        if isinstance(value, dict) and value.get("is_red_flag"):
            _add_flag(merged_flags, str(domain), str(value.get("detail") or "").strip())
    _supplement_from_transcript(transcript, merged_findings, merged_flags)
    doctor = _fill_red_flag_section(cleaned["doctor_markdown"], merged_flags)
    if merged_findings.get("profile_child_gender", {}).get("detail", "").startswith("Male"):
        doctor = doctor.replace(
            "**Child's Gender:** Not reported",
            "**Child's Gender:** Male (mother said beta)",
        )
    elif merged_findings.get("profile_child_gender", {}).get("detail", "").startswith("Female"):
        doctor = doctor.replace(
            "**Child's Gender:** Not reported",
            "**Child's Gender:** Female (mother said beti)",
        )
    return {
        "summary_doctor": doctor,
        "summary_mother": cleaned["mother_hindi"],
        "findings": merged_findings,
        "red_flags": merged_flags,
    }


async def _from_gemini(user_content: str) -> dict:
    from google import genai

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    try:
        response = await client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents=user_content,
            config={
                "system_instruction": DOCTOR_MOTHER_SUMMARY_PROMPT,
                "response_mime_type": "application/json",
                "temperature": 0.2,
                "max_output_tokens": 8192,
            },
        )
        return _parse_json(response.text or "")
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
                {"role": "system", "content": DOCTOR_MOTHER_SUMMARY_PROMPT},
                {"role": "user", "content": user_content},
            ],
        )
        text = response.choices[0].message.content or ""
        return _parse_json(text)
    finally:
        await client.close()


async def generate_call_summaries(
    transcript: list[dict],
    findings: dict,
    red_flags: list[str],
) -> dict:
    """Return doctor and mother summaries, with findings and red flags merged."""
    user_content = _user_content(transcript, findings, red_flags)
    try:
        payload = await _from_gemini(user_content)
        return _assemble(payload, findings, red_flags, transcript)
    except Exception:
        payload = await _from_openai(user_content)
        return _assemble(payload, findings, red_flags, transcript)
