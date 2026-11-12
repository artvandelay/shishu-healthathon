"""Hindi mother PDF. Built only after the call is saved."""

import html
import json
import re
import subprocess
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import answer_extractor
import checklist
import triage_catalog

_STATIC = Path(__file__).resolve().parent / "static"
_FONT = _STATIC / "fonts" / "NotoSansDevanagari.ttf"
_LOGO = _STATIC / "shishu-logo.png"
_IST = timezone(timedelta(hours=5, minutes=30))
_MONTHS = (
    "",
    "जनवरी",
    "फ़रवरी",
    "मार्च",
    "अप्रैल",
    "मई",
    "जून",
    "जुलाई",
    "अगस्त",
    "सितम्बर",
    "अक्टूबर",
    "नवम्बर",
    "दिसम्बर",
)
THALASSEMIA_LINE = (
    "अगली प्रेग्नेंसी से पहले माँ और पिता दोनों की थैलेसीमिया जाँच (स्क्रीनिंग) करवा लें।"
)
_GROUP_HI = {
    "birth": "जन्म और नवजात",
    "body": "बढ़त और शरीर",
    "illness": "बीमारी",
    "pregnancy": "प्रेग्नेंसी",
    "family": "परिवार",
    "close": "अंत",
}
_INCOMPLETE_REASONS = frozenset({"silence", "time_limit", "emergency", "caller_stop"})
_CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

_CSS = """
@font-face {
  font-family: "Body";
  src: url("__FONT__") format("truetype");
  font-weight: 100 900;
}
@page {
  size: A4;
  margin: 14mm 16mm 16mm;
  @bottom-left {
    content: "SHISHU";
    font-family: "Body", sans-serif;
    font-size: 8pt;
    color: #8E95A3;
  }
  @bottom-right {
    content: counter(page) " / " counter(pages);
    font-family: "Body", sans-serif;
    font-size: 8pt;
    color: #8E95A3;
  }
}
* { box-sizing: border-box; }
html, body {
  margin: 0;
  padding: 0;
  font-family: "Body", sans-serif;
  color: #1E2947;
  font-size: 12pt;
  line-height: 1.45;
  -webkit-print-color-adjust: exact;
  print-color-adjust: exact;
}
.top { display: flex; align-items: center; gap: 8px; }
.top img { width: 26px; height: 26px; }
.brand {
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-size: 12pt;
  font-weight: 700;
  letter-spacing: 0.14em;
}
.kind { margin-left: auto; font-size: 11pt; font-weight: 700; color: #DAA550; }
.rule { height: 2px; background: #DAA550; margin: 6px 0 8px; }
.who { margin: 0; font-size: 12pt; font-weight: 700; }
.meta { margin: 1px 0 0; font-size: 10pt; color: #565F75; }
.notice {
  margin: 12px 0 0;
  padding: 8px 12px;
  border: 1.5px solid #1E2947;
  border-left: 5px solid #DAA550;
  font-size: 12pt;
  font-weight: 700;
}
.notice span { display: block; font-weight: 400; font-size: 10.5pt; color: #565F75; margin-top: 2px; }
.band {
  margin: 10px 0 0;
  padding: 8px 12px;
  background: #FBF3E4;
  border-left: 5px solid #DAA550;
}
h1 { margin: 18px 0 8px; font-size: 14pt; }
ol { margin: 0; padding-left: 1.2em; }
ol li { margin: 0 0 8px; }
ol li::marker { color: #DAA550; font-weight: 700; }
.note { margin: 4px 0 0; font-size: 10.5pt; color: #565F75; }
h2 { margin: 16px 0 6px; font-size: 13pt; }
.yes { margin: 0; padding: 0; list-style: none; }
.yes li { margin: 0 0 6px; padding-left: 12px; border-left: 2px solid #DAA550; }
.calm { margin: 0; color: #565F75; }
.universal {
  margin: 14px 0 0;
  padding: 8px 12px;
  border: 1px dashed #8E95A3;
  background: #F4F6F8;
  font-size: 11pt;
  break-inside: avoid;
}
.universal b { display: block; font-size: 10pt; color: #565F75; margin-bottom: 2px; }
.universal .why { display: block; font-size: 10pt; color: #565F75; margin-bottom: 4px; }
.page2 { break-before: page; }
.page2 h1 { margin-top: 0; }
.group { margin: 12px 0 2px; font-size: 11pt; font-weight: 700; color: #DAA550; break-after: avoid; }
.no-row {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  padding: 4px 0;
  border-bottom: 1px solid #E0E2E5;
  break-inside: avoid;
}
.ans { flex: 0 0 auto; color: #565F75; }
.chips { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 4px; }
.chip { padding: 3px 10px; border: 1px solid #E0E2E5; border-radius: 99px; font-size: 10pt; color: #565F75; }
"""


def _ist(started_at: str) -> datetime | None:
    text = (started_at or "").strip()
    if len(text) < 19:
        return None
    try:
        naive = datetime.strptime(text[:19], "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None
    return naive.replace(tzinfo=timezone.utc).astimezone(_IST)


def _clock(dt: datetime) -> str:
    hour = dt.hour
    suffix = "पूर्वाह्न" if hour < 12 else "अपराह्न"
    hour12 = hour % 12 or 12
    return f"{hour12}:{dt.minute:02d} {suffix}"


def _duration(seconds: int) -> str:
    minutes, secs = divmod(max(0, int(seconds or 0)), 60)
    if minutes and secs:
        return f"{minutes} मिनट {secs} सेकंड"
    if minutes:
        return f"{minutes} मिनट"
    return f"{secs} सेकंड"


_CONS = (
    ("ksh", "क्ष"), ("kh", "ख"), ("gh", "घ"), ("ng", "ङ"), ("chh", "छ"), ("ch", "च"),
    ("jh", "झ"), ("j", "ज"), ("ny", "ञ"), ("th", "थ"), ("dh", "ध"), ("ph", "फ"),
    ("bh", "भ"), ("sh", "श"), ("k", "क"), ("g", "ग"), ("t", "त"), ("d", "द"),
    ("n", "न"), ("p", "प"), ("b", "ब"), ("m", "म"), ("y", "य"), ("r", "र"),
    ("l", "ल"), ("v", "व"), ("w", "व"), ("s", "स"), ("h", "ह"), ("f", "फ"),
)
_VOW = (
    ("aa", "ा", "आ"), ("ii", "ी", "ई"), ("ee", "ी", "ई"), ("uu", "ू", "ऊ"),
    ("oo", "ू", "ऊ"), ("ai", "ै", "ऐ"), ("au", "ौ", "औ"), ("a", "", "अ"),
    ("i", "ि", "इ"), ("u", "ु", "उ"), ("e", "े", "ए"), ("o", "ो", "ओ"),
)
_KEEP_LATIN = {"asha", "anm", "bcg", "icu", "nicu", "bp", "shishu"}


def transliterate(word: str) -> str:
    """Roman Indian name to Devanagari. Shilpa -> शिल्पा, Aarav -> आरव."""
    s = (word or "").lower()
    i = 0
    out: list[str] = []
    while i < len(s):
        cons = next((pair for pair in _CONS if s.startswith(pair[0], i)), None)
        if cons is None:
            vow = next((item for item in _VOW if s.startswith(item[0], i)), None)
            if vow is None:
                out.append(s[i])
                i += 1
                continue
            out.append(vow[2])
            i += len(vow[0])
            continue
        i += len(cons[0])
        vow = next((item for item in _VOW if s.startswith(item[0], i)), None)
        if vow is None:
            nxt = any(s.startswith(pair[0], i) for pair in _CONS)
            out.append(cons[1] + ("्" if nxt else ""))
            continue
        i += len(vow[0])
        matra = "ा" if vow[0] == "a" and i >= len(s) else vow[1]
        out.append(cons[1] + matra)
    return "".join(out)


def hindi_text(text: str) -> str:
    def one(match: re.Match) -> str:
        word = match.group(0)
        if word.lower() in _KEEP_LATIN:
            return word
        if word.lower() in {"year", "years"}:
            return "साल"
        if word.lower() in {"month", "months"}:
            return "महीने"
        return transliterate(word)

    return re.sub(r"[A-Za-z]+", one, text or "")


def _name(call: dict) -> str:
    raw = str(call.get("mother_name") or "").strip()
    raw = re.sub(r"(?i)^mother'?s name is\s+", "", raw).strip(" .")
    if raw and "not reported" not in raw.lower() and "does not" not in raw.lower():
        return hindi_text(raw)
    finding = (call.get("findings") or {}).get("profile_mother_name")
    if isinstance(finding, dict):
        detail = re.sub(r"(?i)^mother'?s name is\s+", "", str(finding.get("detail") or "")).strip(" .")
        if detail and "not reported" not in detail.lower():
            return hindi_text(detail)
    return "माँ"


def _child(call: dict) -> str:
    bits = []
    name = str(call.get("child_name") or "").strip()
    age = str(call.get("child_age") or "").strip()
    gender = str(call.get("child_gender") or "").strip().lower()
    findings = call.get("findings") or {}
    if not name:
        entry = findings.get("profile_child_name")
        if isinstance(entry, dict):
            name = str(entry.get("detail") or "").strip()
    if not age:
        entry = findings.get("profile_child_age")
        if isinstance(entry, dict):
            age = str(entry.get("detail") or "").strip()
    m_age = re.search(r"(\d+\s*(?:years?|months?|साल|महीने))", age, re.I)
    if m_age and len(age.split()) > 3:
        age = m_age.group(1)
    age = re.sub(r"(?i)\byears?\b", "साल", age)
    age = re.sub(r"(?i)\bmonths?\b", "महीने", age)
    if not gender:
        entry = findings.get("profile_child_gender")
        if isinstance(entry, dict):
            gender = str(entry.get("detail") or "").lower()
    if name and "not reported" not in name.lower():
        bits.append(hindi_text(name.split(".")[0]))
    if age and "not reported" not in age.lower() and "does not" not in age.lower():
        bits.append(hindi_text(age))
    if "female" in gender or gender.startswith("girl") or "beti" in gender:
        bits.append("बेटी")
    elif "male" in gender or gender.startswith("boy") or "beta" in gender:
        bits.append("बेटा")
    return " · ".join(bits)


def _incomplete(call: dict) -> bool:
    telemetry = call.get("telemetry") or {}
    reason = str(telemetry.get("close_reason") or "")
    if reason in {"silence", "time_limit", "emergency", "caller_stop"}:
        return True
    if "coverage" not in telemetry:
        return False
    coverage = list(telemetry.get("coverage") or [])
    return bool(checklist.open_topics(coverage))


def _answer(answers: dict, topic_id: str) -> str:
    entry = (answers or {}).get(topic_id) or {}
    if isinstance(entry, dict):
        return str(entry.get("answer") or "")
    return str(entry or "")


def build(call: dict, extracted: dict | None = None) -> dict:
    extracted = extracted or {}
    answers = extracted.get("answers") or {}
    content = answer_extractor.report_content(extracted)
    when = _ist(str(call.get("started_at") or ""))
    if when:
        date = f"{when.day} {_MONTHS[when.month]} {when.year}"
        clock = _clock(when)
    else:
        date, clock = "", ""
    incomplete = _incomplete(call)
    questions = [hindi_text(question) for question in content["questions"]]
    yes = [hindi_text(line) for line in content["yes"]]
    groups = []
    unknown = []
    unasked = []
    not_asked_groups: list[str] = []
    for group in checklist.DETAIL_GROUPS:
        nos = []
        facts = []
        missed = []
        for topic in checklist.TOPICS:
            if topic["group"] != group:
                continue
            entry = answers.get(topic["id"]) if isinstance(answers.get(topic["id"]), dict) else {}
            question = checklist.question_hi(topic["id"])
            if checklist.is_concern(topic["id"], entry):
                continue
            fact = checklist.fact_label(topic["id"], entry)
            kind = _answer(answers, topic["id"]) or "not_asked"
            if fact:
                facts.append({"q": question, "a": fact})
            elif kind == "no":
                nos.append(question)
            elif kind == "unknown" or (
                kind == "yes" and checklist.report_rule(topic["id"])["polarity"] != "concern"
            ):
                unknown.append(question)
            elif kind == "not_asked" and not topic.get("optional"):
                missed.append(question)
        if nos or facts:
            groups.append({"title": checklist.GROUP_HI[group], "items": nos, "facts": facts})
        if missed:
            unasked.append({"title": checklist.GROUP_HI[group], "items": missed})
            if incomplete:
                not_asked_groups.append(checklist.GROUP_HI[group])
    child = _child(call)
    who = _name(call) + (f" · {child}" if child else "")
    return {
        "who": who,
        "date": date,
        "time": clock,
        "duration": _duration(call.get("duration_sec") or 0),
        "status": "अधूरी कॉल" if incomplete else "पूर्ण कॉल",
        "incomplete": incomplete,
        "emergency": str((call.get("telemetry") or {}).get("close_reason") or "") == "emergency",
        "questions": questions,
        "universal": THALASSEMIA_LINE,
        "yes": yes,
        "groups": groups,
        "unknown": unknown,
        "unasked": unasked,
        "not_asked": not_asked_groups,
        "started": when,
    }


def _cursor(call: dict) -> dict:
    state = call.get("cursor_state")
    if isinstance(state, dict) and state.get("active_qid"):
        return state
    raw = call.get("cursor_json") or ""
    if isinstance(raw, dict) and raw.get("active_qid"):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        if isinstance(data, dict) and data.get("active_qid"):
            return data
    return {}


def _page2_from_cursor(state: dict, incomplete: bool) -> tuple[list, list, list]:
    """Clear, unknown, and unanswered rows. not_applicable is omitted."""
    answers = state.get("answers") or {}
    buckets: dict[str, list[str]] = {}
    order: list[str] = []
    unknown: list[str] = []
    missed: dict[str, list[str]] = {}
    missed_order: list[str] = []
    for spec in triage_catalog.active_specs():
        if spec.kind in ("profile", "bridge"):
            continue
        entry = answers.get(spec.id)
        if isinstance(entry, dict) and entry.get("state") == "not_applicable":
            continue
        if isinstance(entry, dict):
            status = entry.get("state") or ""
        elif incomplete and spec.kind in ("gate", "open_capture"):
            status = "not_asked"
        else:
            continue
        if status in ("", "concern", "not_applicable"):
            continue
        title = _GROUP_HI.get(spec.group, spec.group)
        if status == "clear":
            if title not in buckets:
                order.append(title)
                buckets[title] = []
            buckets[title].append(spec.question_hi)
        elif status == "unknown":
            unknown.append(spec.question_hi)
        elif status == "not_asked" and incomplete:
            if title not in missed:
                missed_order.append(title)
                missed[title] = []
            missed[title].append(spec.question_hi)
    groups = [{"title": title, "items": buckets[title], "facts": []} for title in order]
    unasked = [{"title": title, "items": missed[title]} for title in missed_order]
    return groups, unknown, unasked


def build_v3(call: dict, synthesis: dict) -> dict:
    """Mother PDF from a v3 cursor plus one synthesis payload.

    Hindi lines are used as written. This path does not transliterate.
    """
    state = _cursor(call)
    when = _ist(str(call.get("started_at") or ""))
    if when:
        date = f"{when.day} {_MONTHS[when.month]} {when.year}"
        clock = _clock(when)
    else:
        date, clock = "", ""
    reason = str(state.get("close_reason") or (call.get("telemetry") or {}).get("close_reason") or "")
    incomplete = reason in _INCOMPLETE_REASONS or (reason != "complete" and state.get("phase") != "closing")
    if reason == "complete":
        incomplete = False
    groups, unknown, unasked = _page2_from_cursor(state, incomplete)
    name = str(synthesis.get("mother_name_hi") or "").strip() or "माँ"
    children = str(synthesis.get("children_label_hi") or "").strip()
    who = name + (f" · {children}" if children else "")
    call_id = int(call.get("id") or 0)
    return {
        "who": who,
        "ref": f"SHISHU-{call_id}" if call_id else "",
        "date": date,
        "time": clock,
        "duration": _duration(call.get("duration_sec") or 0),
        "status": "अधूरी कॉल" if incomplete else "पूर्ण कॉल",
        "incomplete": incomplete,
        "emergency": reason == "emergency",
        "questions": [str(question) for question in (synthesis.get("mother_questions_hi") or []) if str(question).strip()],
        "universal": THALASSEMIA_LINE,
        "yes": [str(line) for line in (synthesis.get("yes_summaries_hi") or []) if str(line).strip()],
        "groups": groups,
        "unknown": unknown,
        "unasked": unasked,
        "not_asked": [group["title"] for group in unasked] if incomplete else [],
        "started": when,
    }


def filename(call: dict) -> str:
    when = _ist(str(call.get("started_at") or ""))
    if when is None:
        return "SHISHU.pdf"
    return f"SHISHU-{when:%Y-%m-%d-%H%M}.pdf"


def render_html(report: dict) -> str:
    if report["questions"]:
        items = "\n".join(f"<li>{html.escape(q)}</li>" for q in report["questions"])
        questions = (
            "<h1>डॉक्टर या स्वास्थ्य कार्यकर्ता (ASHA/ANM दीदी) से ये पूछें</h1>\n"
            f"<ol>{items}</ol>\n"
        )
    else:
        questions = ""
    universal = (
        '<aside class="universal">'
        "<b>सभी के लिए एक जैसी सलाह</b>"
        '<span class="why">यह इस माँ या बच्चे के बारे में नहीं है।</span>'
        f"<span>{html.escape(report.get('universal') or THALASSEMIA_LINE)}</span>"
        "</aside>"
    )
    if report["yes"]:
        yes = '<ul class="yes">' + "".join(f"<li>{html.escape(line)}</li>" for line in report["yes"]) + "</ul>"
    elif report.get("unasked") and not report["groups"]:
        yes = '<p class="calm">इस कॉल में कोई जवाब दर्ज नहीं हुआ।</p>'
    else:
        yes = '<p class="calm">इस कॉल में कोई बात हाँ में नहीं आई।</p>'
    band = ""
    if report.get("emergency"):
        band = (
            '<div class="band"><b>बच्चे को अभी तुरंत नज़दीकी अस्पताल ले जाइए, या 108 पर कॉल कीजिए।</b> '
            "नीचे वही है जो इस कॉल में पूछा जा सका।</div>"
        )
    elif report["incomplete"]:
        band = (
            '<div class="band"><b>यह कॉल अधूरी रही।</b> '
            "नीचे वही है जो इस कॉल में पूछा जा सका। "
            "समय मिले तो दोबारा कॉल करके बाकी सवाल पूरे कर लें।</div>"
        )
    rows: list[str] = []
    for group in report["groups"]:
        rows.append(f'<div class="group">{html.escape(group["title"])}</div>')
        for question in group["items"]:
            rows.append(
                '<div class="no-row"><span>'
                + html.escape(question)
                + '</span><span class="ans">नहीं</span></div>'
            )
        for fact in group.get("facts") or []:
            rows.append(
                '<div class="no-row"><span>'
                + html.escape(fact["q"])
                + '</span><span class="ans">'
                + html.escape(fact["a"])
                + "</span></div>"
            )
    if report["unknown"]:
        rows.append('<div class="group">पक्का जवाब नहीं मिला</div>')
        for question in report["unknown"]:
            rows.append(
                '<div class="no-row"><span>'
                + html.escape(question)
                + '</span><span class="ans">पता नहीं</span></div>'
            )
    if report.get("unasked"):
        rows.append('<div class="group">अभी नहीं पूछा गया</div>')
        for group in report["unasked"]:
            rows.append(f'<div class="group">{html.escape(group["title"])}</div>')
            for question in group["items"]:
                rows.append(
                    '<div class="no-row"><span>'
                    + html.escape(question)
                    + '</span><span class="ans">पूछा नहीं गया</span></div>'
                )
    meta = " · ".join(
        bit
        for bit in (
            report.get("ref"),
            report["date"],
            report["time"],
            report["duration"],
            report["status"],
        )
        if bit
    )
    css = _CSS.replace("__FONT__", _FONT.as_uri())
    return f"""<!DOCTYPE html>
<html lang="hi">
<head><meta charset="utf-8"><style>{css}</style></head>
<body>
<header>
  <div class="top">
    <img src="{_LOGO.as_uri()}" alt="">
    <div class="brand">SHISHU</div>
    <div class="kind">कॉल रिपोर्ट</div>
  </div>
  <div class="rule"></div>
  <p class="who">{html.escape(report["who"])}</p>
  <p class="meta">{html.escape(meta)}</p>
</header>
<div class="notice">यह कोई बीमारी की पुष्टि (डायग्नोसिस) नहीं है।<span>केवल डॉक्टर जाँच करके सही सलाह दे सकते हैं।</span></div>
{band}
{questions}<p class="note">अगर इसमें कुछ गलत है, या आप कुछ और बताना चाहती हैं, तो दोबारा कॉल कर सकती हैं।</p>
<h2>इस कॉल में जो हाँ था</h2>
{yes}
{universal}
<section class="page2">
  <h1>विवरण</h1>
  <p class="meta">{html.escape(report["who"])}</p>
  {"".join(rows)}
</section>
</body>
</html>
"""


def _chrome_pdf(document: str) -> bytes:
    with tempfile.TemporaryDirectory() as folder:
        html_path = Path(folder) / "report.html"
        pdf_path = Path(folder) / "report.pdf"
        html_path.write_text(document, encoding="utf-8")
        subprocess.run(
            [
                _CHROME,
                "--headless",
                "--disable-gpu",
                "--no-pdf-header-footer",
                f"--print-to-pdf={pdf_path}",
                html_path.as_uri(),
            ],
            check=True,
            capture_output=True,
        )
        return pdf_path.read_bytes()


def render_pdf(report: dict) -> bytes:
    document = render_html(report)
    try:
        from weasyprint import HTML

        return HTML(string=document, base_url=str(_STATIC)).write_pdf()
    except Exception:
        if Path(_CHROME).is_file():
            return _chrome_pdf(document)
        raise
