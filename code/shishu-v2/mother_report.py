"""Hindi Mother PDF: one typed projection in, one WeasyPrint document out."""

import html
import os
import sys
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

if sys.platform == "darwin":
    os.environ.setdefault("DYLD_FALLBACK_LIBRARY_PATH", "/opt/homebrew/lib")

import copy_hi
import db
import summary
from summary import ScreeningSummary

_STATIC = Path(__file__).resolve().parent / "static"
_FONT = _STATIC / "fonts" / "NotoSansDevanagari.ttf"
_LOGO = _STATIC / "shishu-logo.png"
_IST = timezone(timedelta(hours=5, minutes=30))

THALASSEMIA_LINE = (
    "अगली प्रेग्नेंसी से पहले माँ और पिता दोनों की थैलेसीमिया जाँच (स्क्रीनिंग) करवा लें।"
)

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
.group { margin: 8px 0 1px; font-size: 11pt; font-weight: 700; color: #DAA550; break-after: avoid; }
.no-row {
  display: flex;
  justify-content: space-between;
  gap: 8px;
  padding: 1px 0;
  margin: 0;
  border-bottom: 1px solid #E0E2E5;
  break-inside: avoid;
  line-height: 1.35;
}
.ans { flex: 0 0 auto; color: #565F75; }
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


def _clock(when: datetime) -> str:
    suffix = "पूर्वाह्न" if when.hour < 12 else "अपराह्न"
    return f"{when.hour % 12 or 12}:{when.minute:02d} {suffix}"


def _duration(seconds: int) -> str:
    minutes, secs = divmod(max(0, int(seconds or 0)), 60)
    if minutes and secs:
        return f"{minutes} मिनट {secs} सेकंड"
    if minutes:
        return f"{minutes} मिनट"
    return f"{secs} सेकंड"


def _groups(groups) -> list[dict[str, Any]]:
    return [{"title": group.title, "items": list(group.items)} for group in groups]


def build_report(
    call: Mapping[str, Any], projected: ScreeningSummary | None = None
) -> dict[str, Any]:
    """The Mother PDF payload, projected once from the call's typed state."""
    proj = projected or summary.project_summary(db.call_state_from_row(call))
    when = _ist(str(call.get("started_at") or ""))
    return {
        "call_id": int(call.get("id") or 0),
        "date_hi": f"{when.day} {copy_hi.MONTHS_HI[when.month]} {when.year}" if when else "",
        "time_hi": _clock(when) if when else "",
        "duration_hi": _duration(call.get("duration_sec") or 0),
        "mother_name": proj.mother_name,
        "child_age": proj.children_label_hi,
        "incomplete": proj.incomplete,
        "emergency": proj.emergency,
        "questions": list(proj.mother_questions_hi),
        "yes": list(proj.yes_summaries_hi),
        "page2_groups": _groups(proj.page2_clear_groups),
        "page2_unknown": list(proj.page2_unknown),
        "page2_unasked": _groups(proj.page2_unasked_groups),
        "logo_uri": _LOGO.as_uri(),
        "font_uri": _FONT.as_uri(),
    }


def filename(call: Mapping[str, Any]) -> str:
    when = _ist(str(call.get("started_at") or ""))
    return "SHISHU.pdf" if when is None else f"SHISHU-{when:%Y-%m-%d-%H%M}.pdf"


def _who(report: Mapping[str, Any]) -> str:
    name = str(report.get("mother_name") or "").strip() or "माँ"
    children = str(report.get("child_age") or "").strip()
    return name + (f" · {children}" if children else "")


def _questions_html(questions: list[str]) -> str:
    if not questions:
        return ""
    items = "\n".join(f"<li>{html.escape(question)}</li>" for question in questions)
    return (
        "<h1>डॉक्टर या स्वास्थ्य कार्यकर्ता (ASHA/ANM दीदी) से ये पूछें</h1>\n"
        f"<ol>{items}</ol>\n"
    )


def _band_html(report: Mapping[str, Any]) -> str:
    if report.get("emergency"):
        headline = "बच्चे को अभी तुरंत नज़दीकी अस्पताल ले जाइए, या 108 पर कॉल कीजिए।"
        return (
            f'<div class="band"><b>{headline}</b> '
            "नीचे वही है जो इस कॉल में पूछा जा सका।</div>"
        )
    if report.get("incomplete"):
        return (
            '<div class="band"><b>यह कॉल अधूरी रही।</b> '
            "नीचे वही है जो इस कॉल में पूछा जा सका। "
            "समय मिले तो दोबारा कॉल करके बाकी सवाल पूरे कर लें।</div>"
        )
    return ""


def _yes_html(report: Mapping[str, Any]) -> str:
    if report.get("yes"):
        return '<ul class="yes">' + "".join(
            f"<li>{html.escape(line)}</li>" for line in report["yes"]
        ) + "</ul>"
    empty = report.get("page2_unasked") and not report.get("page2_groups")
    if empty and not report.get("page2_unknown"):
        return '<p class="calm">इस कॉल में कोई जवाब दर्ज नहीं हुआ।</p>'
    return '<p class="calm">इस कॉल में कोई बात हाँ में नहीं आई।</p>'


def _page2_html(report: Mapping[str, Any]) -> str:
    rows: list[str] = []
    for group in report.get("page2_groups") or []:
        rows.append(f'<div class="group">{html.escape(group["title"])}</div>')
        for question in group["items"]:
            rows.append(
                '<div class="no-row"><span>'
                + html.escape(question)
                + '</span><span class="ans">नहीं</span></div>'
            )
    if report.get("page2_unknown"):
        rows.append('<div class="group">पक्का जवाब नहीं मिला</div>')
        for question in report["page2_unknown"]:
            rows.append(
                '<div class="no-row"><span>'
                + html.escape(question)
                + '</span><span class="ans">पता नहीं</span></div>'
            )
    if report.get("page2_unasked"):
        rows.append('<div class="group">अभी नहीं पूछा गया</div>')
        for group in report["page2_unasked"]:
            rows.append(f'<div class="group">{html.escape(group["title"])}</div>')
            for question in group["items"]:
                rows.append(
                    '<div class="no-row"><span>'
                    + html.escape(question)
                    + '</span><span class="ans">पूछा नहीं गया</span></div>'
                )
    return "".join(rows)


def render_html(report: Mapping[str, Any]) -> str:
    """The full Hindi document, page one for her and page two for the details."""
    call_id = int(report.get("call_id") or 0)
    who = _who(report)
    meta = " · ".join(
        bit
        for bit in (
            f"SHISHU-{call_id}" if call_id else "",
            report.get("date_hi") or "",
            report.get("time_hi") or "",
            report.get("duration_hi") or "",
            "अधूरी कॉल" if report.get("incomplete") else "पूर्ण कॉल",
        )
        if bit
    )
    universal = (
        '<aside class="universal">'
        "<b>सभी के लिए एक जैसी सलाह</b>"
        '<span class="why">यह इस माँ या बच्चे के बारे में नहीं है।</span>'
        f"<span>{html.escape(THALASSEMIA_LINE)}</span>"
        "</aside>"
    )
    css = _CSS.replace("__FONT__", str(report.get("font_uri") or _FONT.as_uri()))
    return f"""<!DOCTYPE html>
<html lang="hi">
<head><meta charset="utf-8"><style>{css}</style></head>
<body>
<header>
  <div class="top">
    <img src="{report.get("logo_uri") or _LOGO.as_uri()}" alt="">
    <div class="brand">SHISHU</div>
    <div class="kind">कॉल रिपोर्ट</div>
  </div>
  <div class="rule"></div>
  <p class="who">{html.escape(who)}</p>
  <p class="meta">{html.escape(meta)}</p>
</header>
<div class="notice">यह कोई बीमारी की पुष्टि (डायग्नोसिस) नहीं है।<span>केवल डॉक्टर जाँच करके सही सलाह दे सकते हैं।</span></div>
{_band_html(report)}
{_questions_html(list(report.get("questions") or []))}<p class="note">अगर इसमें कुछ गलत है, या आप कुछ और बताना चाहती हैं, तो दोबारा कॉल कर सकती हैं।</p>
<h2>इस कॉल में जो हाँ था</h2>
{_yes_html(report)}
{universal}
<section class="page2">
  <h1>विवरण</h1>
  <p class="meta">{html.escape(who)}</p>
  {_page2_html(report)}
</section>
</body>
</html>
"""


def render_pdf(report: Mapping[str, Any]) -> bytes:
    """One crisp Devanagari PDF. WeasyPrint is the only renderer."""
    from weasyprint import HTML

    return HTML(string=render_html(report), base_url=str(_STATIC)).write_pdf()
