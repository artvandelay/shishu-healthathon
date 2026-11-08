"""Hindi mother PDF. Built only after the call is saved."""

import html
import re
import subprocess
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import checklist
import call_state

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
_ROUTINE = (
    "मेरे बच्चे की उम्र के हिसाब से उसका वज़न, लंबाई और सीखना-बढ़ना (बैठना, चलना, बोलना) ठीक चल रहा है या नहीं?",
    "बच्चे के सभी ज़रूरी टीके, और सुनने-देखने की जाँच, पूरी है या कोई टीका बाकी है?",
    "रोज़ के खान-पान के लिए अभी किन बातों का ध्यान रखूँ, और किन लक्षणों पर तुरंत डॉक्टर या स्वास्थ्य कार्यकर्ता को दिखाऊँ?",
)
THALASSEMIA_LINE = (
    "अगली प्रेग्नेंसी से पहले माँ और पिता दोनों की थैलेसीमिया जाँच (स्क्रीनिंग) करवा लें।"
)
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
    ("c", "क"), ("q", "क"), ("x", "क्स"), ("z", "ज़"),
)
_VOW = (
    ("aa", "ा", "आ"), ("ii", "ी", "ई"), ("ee", "ी", "ई"), ("uu", "ू", "ऊ"),
    ("oo", "ू", "ऊ"), ("ai", "ै", "ऐ"), ("au", "ौ", "औ"), ("a", "", "अ"),
    ("i", "ि", "इ"), ("u", "ु", "उ"), ("e", "े", "ए"), ("o", "ो", "ओ"),
)
_KEEP_LATIN = {"asha", "anm", "bcg", "icu", "nicu", "bp", "shishu"}

_PHRASE_HI: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(rf"(?i)\b{re.escape(phrase)}\b"), replacement)
    for phrase, replacement in (
        ("low birth weight", "कम वज़न"),
        ("not normal", "असामान्य"),
        ("not_normal", "असामान्य"),
        ("late walking", "चलने में देरी"),
        ("late speech", "बोलने में देरी"),
        ("late talking", "बोलने में देरी"),
        ("late sitting", "बैठने में देरी"),
        ("blood pressure", "बीपी"),
        ("high bp", "हाई बीपी"),
        ("first cousin", "सगे कज़िन"),
        ("second cousin", "दूर के कज़िन"),
        ("c-section", "ऑपरेशन"),
        ("c section", "ऑपरेशन"),
        ("mother's brother", "मामा"),
        ("father's brother", "चाचा"),
        ("mother's sister", "मौसी"),
        ("father's sister", "बुआ"),
        ("mother's mother", "नानी"),
        ("father's mother", "दादी"),
        ("mother's father", "नाना"),
        ("father's father", "दादा"),
        ("kaanch ka box", "काँच का बक्सा"),
        ("blue light", "नीली रोशनी"),
    )
)

_TOKEN_HI = {
    "years": "साल",
    "year": "साल",
    "saal": "साल",
    "months": "महीने",
    "month": "महीने",
    "mahine": "महीने",
    "mahina": "महीना",
    "days": "दिन",
    "day": "दिन",
    "din": "दिन",
    "weeks": "हफ़्ते",
    "week": "हफ़्ता",
    "hafte": "हफ़्ते",
    "hafta": "हफ़्ता",
    "male": "लड़का",
    "female": "लड़की",
    "boy": "लड़का",
    "girl": "लड़की",
    "operation": "ऑपरेशन",
    "caesarean": "ऑपरेशन",
    "cesarean": "ऑपरेशन",
    "normal": "नॉर्मल",
    "hospital": "अस्पताल",
    "clinic": "क्लिनिक",
    "home": "घर",
    "ghar": "घर",
    "early": "समय से पहले",
    "preterm": "समय से पहले",
    "late": "समय के बाद",
    "postterm": "समय के बाद",
    "jaundice": "पीलिया",
    "peeliya": "पीलिया",
    "piliya": "पीलिया",
    "breathing": "साँस की दिक्कत",
    "abnormal": "असामान्य",
    "low": "कम",
    "high": "ज़्यादा",
    "heavy": "ज़्यादा",
    "prolonged": "देर तक",
    "both": "दोनों",
    "dono": "दोनों",
    "fits": "झटके",
    "sitting": "बैठना",
    "walking": "चलना",
    "speech": "बोलना",
    "seizures": "झटके",
    "seizure": "झटके",
    "jhatke": "झटके",
    "delayed": "देरी",
    "delay": "देरी",
    "urine": "पेशाब",
    "stool": "पॉटी",
    "potty": "पॉटी",
    "fever": "बुखार",
    "bukhar": "बुखार",
    "pneumonia": "निमोनिया",
    "infection": "इन्फेक्शन",
    "swelling": "सूजन",
    "sujan": "सूजन",
    "soojan": "सूजन",
    "boil": "फोड़ा",
    "phoda": "फोड़ा",
    "lump": "गाँठ",
    "gaanth": "गाँठ",
    "bleeding": "खून आना",
    "bruising": "नील पड़ना",
    "bruises": "नील पड़ना",
    "transfusion": "खून चढ़ाना",
    "sleepiness": "सुस्ती",
    "lethargy": "सुस्ती",
    "vomiting": "उल्टी",
    "ulti": "उल्टी",
    "sugar": "शुगर",
    "bp": "बीपी",
    "diabetes": "शुगर",
    "thyroid": "थायरॉइड",
    "stroke": "लकवा",
    "skin": "खुजली",
    "cancer": "कैंसर",
    "cousin": "कज़िन",
    "uncle": "चाचा/मामा",
    "aunt": "बुआ/मौसी",
    "grandmother": "दादी/नानी",
    "grandfather": "दादा/नाना",
    "brother": "भाई",
    "bhai": "भाई",
    "sister": "बहन",
    "behen": "बहन",
    "mother": "माँ",
    "father": "पिता",
    "mama": "मामा",
    "chacha": "चाचा",
    "bua": "बुआ",
    "mausi": "मौसी",
    "masi": "मासी",
    "dadi": "दादी",
    "nani": "नानी",
    "dada": "दादा",
    "nana": "नाना",
    "ghutna": "घुटना",
    "ghutne": "घुटने",
    "knee": "घुटना",
    "joint": "जोड़",
    "isha": "ईशा",
    "ram": "राम",
    "aarav": "आरव",
}


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
    raw = text or ""
    for pattern, replacement in _PHRASE_HI:
        raw = pattern.sub(replacement, raw)

    def one(match: re.Match) -> str:
        word = match.group(0)
        lower = word.lower()
        if lower in _TOKEN_HI:
            return _TOKEN_HI[lower]
        if lower in _KEEP_LATIN:
            return word
        # A capitalized name can be written in Hindi. An unknown lowercase
        # clinical word stays as it is, instead of becoming fake Devanagari.
        if word[:1].isupper() and (len(word) == 1 or word[1:].islower()):
            return transliterate(word)
        return word

    return re.sub(r"[A-Za-z]+", one, raw)


def _name(call: dict) -> str:
    state = call.get("state")
    if isinstance(state, dict):
        prof = state.get("profile") or {}
        caller_name = call_state.spoken_person_name(prof.get("caller_name"))
        if caller_name and "not reported" not in caller_name.lower():
            return hindi_text(caller_name)
        intro_val = str(((state.get("topics") or {}).get("intro_name") or {}).get("value") or "").strip()
        if intro_val and "not reported" not in intro_val.lower():
            return hindi_text(intro_val)
    raw = str(call.get("mother_name") or "").strip()
    raw = re.sub(r"(?i)^mother'?s name is\s+", "", raw).strip(" .")
    if raw and "not reported" not in raw.lower() and "does not" not in raw.lower():
        return hindi_text(raw)
    finding = (call.get("findings") or {}).get("profile_mother_name")
    if isinstance(finding, dict):
        detail = re.sub(r"(?i)^mother'?s name is\s+", "", str(finding.get("detail") or "")).strip(" .")
        if detail and "not reported" not in detail.lower():
            return hindi_text(detail)
    caller = str(call.get("caller_name") or "").strip()
    if caller and "tester" not in caller.lower() and "bot" not in caller.lower():
        return hindi_text(caller)
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


def _questions(summary_mother: str) -> list[str]:
    found: list[str] = []
    for line in (summary_mother or "").splitlines():
        cleaned = re.sub(r"^(?:[\d१२३४५६७८९]+[\.\)\-:]|[•\-\*])\s*", "", line.strip())
        if "?" in cleaned or "？" in cleaned:
            if cleaned not in found:
                found.append(cleaned)
    return found or list(_ROUTINE)


def _incomplete(call: dict) -> bool:
    telemetry = call.get("telemetry") or {}
    reason = str(telemetry.get("close_reason") or "")
    if reason in {"silence", "time_limit", "emergency", "caller_stop"}:
        return True
    if "coverage" not in telemetry:
        return False
    coverage = list(telemetry.get("coverage") or [])
    return bool(checklist.open_topics(coverage))


def _has_hindi(text: str) -> bool:
    return any("\u0900" <= char <= "\u097f" for char in text)


def _answer(answers: dict, topic_id: str) -> str:
    entry = (answers or {}).get(topic_id) or {}
    if isinstance(entry, dict):
        return str(entry.get("answer") or "")
    return str(entry or "")


def _user_text(call: dict) -> str:
    parts = []
    for turn in call.get("transcript") or []:
        if str(turn.get("role") or "") != "user":
            continue
        text = str(turn.get("text") or "").strip()
        if text:
            parts.append(text)
    return " ".join(parts).lower()


def _covered(call: dict) -> set[str]:
    telemetry = call.get("telemetry") or {}
    return {str(topic_id) for topic_id in (telemetry.get("coverage") or [])}


def _saved_state(call: dict) -> dict | None:
    state = call.get("state")
    if isinstance(state, dict) and isinstance(state.get("topics"), dict):
        return state
    return None


def _structured_kind(state: dict, topic_id: str) -> str:
    topic = (state.get("topics") or {}).get(topic_id) or {}
    return call_state.topic_answer_kind(topic, topic_id)


def _thalassemia_from_state(state: dict) -> bool:
    topic = (state.get("topics") or {}).get("preg_thalassemia") or {}
    status = topic.get("state")
    if status in {"not_asked", "not_applicable", "no_response"}:
        return False
    caller = (topic.get("details") or {}).get("caller_test")
    if isinstance(caller, dict):
        caller = caller.get("value", caller.get("state"))
    if caller is False or caller in {"unknown", "no"}:
        return True
    if caller is True:
        return False
    if status == "unknown":
        return True
    return status == "answered" and topic.get("value") is False


_HIDDEN_DETAIL_VALUES = {
    "noted", "skip", "unknown", "no_response", "none", "true", "false", "yes", "no", "null", "n/a",
    "नहीं", "नही", "nahi", "nahin",
}


def _sex_word(child: dict | None) -> str:
    if not isinstance(child, dict):
        return ""
    if call_state._is_female(child):
        return "बेटी"
    sex = str(child.get("sex") or "").strip().lower()
    if sex in {"male", "boy", "m", "beta", "bete", "son", "बेटा", "बेटे", "लड़का", "लड़के"}:
        return "बेटा"
    return ""


def _child_label(state: dict, child_id: object) -> str:
    children = [
        child
        for child in (state.get("profile") or {}).get("children") or []
        if isinstance(child, dict)
    ]
    child = next((item for item in children if item.get("id") == child_id), None)
    if child and str(child.get("name") or "").strip():
        return str(child["name"]).strip()
    sex_word = _sex_word(child) if child else ""
    if len(children) <= 1:
        return sex_word
    ranked = sorted(
        children,
        key=lambda item: (
            call_state._child_age_years(item.get("age"))
            if call_state._child_age_years(item.get("age")) is not None
            else -1
        ),
        reverse=True,
    )
    index = next((i for i, item in enumerate(ranked) if item.get("id") == child_id), None)
    if index is None:
        return ""
    ages_differ = (
        len(ranked) >= 2
        and call_state._child_age_years(ranked[0].get("age"))
        != call_state._child_age_years(ranked[-1].get("age"))
    )
    if ages_differ and index == 0:
        return {"बेटी": "बड़ी बेटी", "बेटा": "बड़ा बेटा"}.get(sex_word, "बड़ा बच्चा")
    if ages_differ and index == len(ranked) - 1:
        return {"बेटी": "छोटी बेटी", "बेटा": "छोटा बेटा"}.get(sex_word, "छोटा बच्चा")
    if sex_word == "बेटी":
        ordinals = ("पहली", "दूसरी", "तीसरी", "चौथी", "पाँचवीं")
        noun = "बेटी"
    elif sex_word == "बेटा":
        ordinals = ("पहला", "दूसरा", "तीसरा", "चौथा", "पाँचवाँ")
        noun = "बेटा"
    else:
        ordinals = ("पहला", "दूसरा", "तीसरा", "चौथा", "पाँचवाँ")
        noun = "बच्चा"
    word = ordinals[index] if index < len(ordinals) else f"{index + 1}"
    return f"{word} {noun}"


def _shown_value(value: object) -> str:
    if isinstance(value, (list, tuple)):
        parts = [_shown_value(item) for item in value]
        return ", ".join(part for part in parts if part)
    if value is None or isinstance(value, (dict, bool)):
        return ""
    text = str(value).strip().strip("।.! ")
    if re.fullmatch(r"\d+\.0+", text):
        text = str(int(float(text)))
    if not text or text.startswith("child_") or text.lower() in _HIDDEN_DETAIL_VALUES:
        return ""
    mapped = _TOKEN_HI.get(text.lower(), text)
    cleaned = hindi_text(mapped)
    cleaned = re.sub(r"(\S+)\s*\(\1\)", r"\1", cleaned)
    return cleaned.strip()


def _detail_values(details: dict, skip_label: str = "") -> str:
    out: list[str] = []
    skip_norm = hindi_text(skip_label).strip() if skip_label else ""
    for key, value in details.items():
        if key == "affected_child" and skip_norm:
            continue
        shown = _shown_value(value)
        if key == "days" and re.fullmatch(r"\d+(?:\.\d+)?", shown):
            shown = call_state._day_phrase(float(shown))
        elif key == "count" and re.fullmatch(r"\d+(?:\.\d+)?", shown):
            shown = f"{shown} बार"
        elif key == "month" and re.fullmatch(r"\d+(?:\.\d+)?", shown):
            shown = call_state._month_phrase(float(shown))
        elif key in {"pregnancy", "pregnancy_number"} and re.fullmatch(r"\d+", shown):
            shown = _pregnancy_phrase(int(shown))
        if not shown or shown == skip_norm:
            continue
        if shown not in out:
            out.append(shown)
    kept = [
        part for part in out
        if not any(part != other and part in other for other in out)
    ]
    return ", ".join(kept)


def _pregnancy_phrase(index: int) -> str:
    """2 is दूसरी प्रेग्नेंसी, not a bare 2 beside the sentence that already said it."""
    words = {1: "पहली", 2: "दूसरी", 3: "तीसरी", 4: "चौथी", 5: "पाँचवीं"}
    word = words.get(index) or str(index)
    return f"{word} प्रेग्नेंसी"


def _number_text(value: object) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _quarter_age(value: object, unit_word: str) -> str:
    """The Hindi fraction she would have said, for years or months."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return ""
    number = float(value)
    if unit_word == "साल" and abs(number - 0.25) < 1e-6:
        return "3 महीने"
    if abs(number - 0.5) < 1e-6:
        return {"साल": "आधा साल", "हफ़्ते": "आधा हफ़्ता"}.get(unit_word, "आधा महीना")
    if abs(number - 1.25) < 1e-6:
        return {"साल": "सवा साल", "हफ़्ते": "सवा हफ़्ता"}.get(unit_word, "सवा महीना")
    if abs(number - 1.5) < 1e-6:
        return {"साल": "डेढ़ साल", "हफ़्ते": "डेढ़ हफ़्ते"}.get(unit_word, "डेढ़ महीने")
    if abs(number - 2.5) < 1e-6:
        return {"साल": "ढाई साल", "हफ़्ते": "ढाई हफ़्ते"}.get(unit_word, "ढाई महीने")
    if number > 2 and abs(number * 2 - round(number * 2)) < 1e-6 and abs(number - round(number)) == 0.5:
        return f"साढ़े {int(number)} {unit_word}"
    if (
        abs(number * 4 - round(number * 4)) < 1e-6
        and abs(number - int(number) - 0.25) < 1e-6
        and int(number) >= 2
    ):
        return f"सवा {int(number)} {unit_word}"
    if abs(number * 4 - round(number * 4)) < 1e-6 and abs(number - int(number) - 0.75) < 1e-6:
        return f"पौने {int(number) + 1} {unit_word}"
    return ""


def _format_age(age: object) -> str:
    """Hindi for a stored age. 0.5 years is आधा साल, not the same text as 6 months."""
    text = _format_age_unit(age)
    if not isinstance(age, dict) or not text:
        return text
    extra = age.get("plus_days")
    if isinstance(extra, bool) or not isinstance(extra, (int, float)):
        return text
    shown = int(extra) if float(extra).is_integer() else extra
    return f"{text} {shown} दिन"


def _format_age_unit(age: object) -> str:
    if not isinstance(age, dict) or age.get("value") is None:
        return "" if age in (None, "") else str(age)
    value = age.get("value")
    unit = str(age.get("unit") or "years").lower()
    if unit in {"month", "months", "mahine"}:
        spoken = _quarter_age(value, "महीने")
        if spoken:
            return spoken
        if isinstance(value, (int, float)) and not isinstance(value, bool) and float(value) == 1:
            return "1 महीना"
        return f"{_number_text(value)} महीने"
    if unit in {"week", "weeks", "hafte", "hafta"} and isinstance(value, (int, float)) and not isinstance(value, bool):
        spoken = _quarter_age(value, "हफ़्ते")
        if spoken:
            return spoken
        shown = _number_text(value)
        return f"{shown} {'हफ़्ता' if float(value) == 1 else 'हफ़्ते'}"
    if unit in {"day", "days", "din"} and isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{_number_text(value)} दिन"
    if unit in {"year", "years", "saal"}:
        spoken = _quarter_age(value, "साल")
        if spoken:
            return spoken
    return f"{_number_text(value)} {age.get('unit') or ''}".strip()


def _children_label(state: dict) -> str:
    parts = []
    for child in (state.get("profile") or {}).get("children") or []:
        if not isinstance(child, dict):
            continue
        label = _child_label(state, child.get("id"))
        age_text = _format_age(child.get("age"))
        bit = " ".join(part for part in (label, age_text) if part)
        if bit:
            parts.append(hindi_text(bit))
    return ", ".join(parts)


def _instance_note(state: dict, topic_id: str) -> str:
    topic = (state.get("topics") or {}).get(topic_id) or {}
    bits = []
    for instance in topic.get("instances") or []:
        if not isinstance(instance, dict):
            continue
        label = _child_label(state, instance.get("person_id"))
        details = instance.get("details") if isinstance(instance.get("details"), dict) else {}
        extra = _detail_values(details, skip_label=label)
        bit = " ".join(part for part in (label, extra) if part)
        if bit:
            bits.append(bit)
    if not bits and isinstance(topic.get("details"), dict):
        extra = _detail_values(topic["details"])
        if extra:
            bits.append(extra)
    return "; ".join(bits)


def ground_kind(call: dict, topic_id: str, answers: dict) -> str:
    """A yes, no, or unknown counts only if she said it, or the live call recorded it.

    A model that fills in 'no' for a question she never heard is not an answer.
    """
    entry = (answers or {}).get(topic_id) or {}
    kind = _answer(answers, topic_id)
    if kind not in {"yes", "no", "unknown"}:
        return "unknown" if topic_id in _covered(call) else "not_asked"
    evidence = str(entry.get("evidence") or "").strip().lower() if isinstance(entry, dict) else ""
    said = _user_text(call)
    quoted = len(evidence) >= 12 and evidence in said
    if quoted or topic_id in _covered(call):
        return kind
    return "not_asked"


def build(call: dict, extracted: dict | None = None) -> dict:
    extracted = extracted or {}
    answers = extracted.get("answers") or {}
    when = _ist(str(call.get("started_at") or ""))
    if when:
        date = f"{when.day} {_MONTHS[when.month]} {when.year}"
        clock = _clock(when)
    else:
        date, clock = "", ""
    state = _saved_state(call)
    incomplete = _incomplete(call)
    if state:
        close_reason = str(state.get("close_reason") or "")
        if close_reason == "complete":
            incomplete = False
        elif close_reason in {"emergency", "silence", "time_limit", "caller_stop", "browser_hangup"}:
            incomplete = True
    raw_summary_mother = str(call.get("summary_mother") or "").strip()
    if not raw_summary_mother and state:
        import summarizer

        st_findings, st_flags = summarizer.state_to_findings_and_flags(state)
        if st_flags or any(
            isinstance(v, dict) and v.get("is_red_flag") for v in st_findings.values()
        ):
            has_children = any(
                isinstance(child, dict) for child in ((state.get("profile") or {}).get("children") or [])
            )
            raw_summary_mother = summarizer.default_mother_questions(
                st_findings, st_flags, has_children=has_children
            )
    questions = [hindi_text(q) for q in _questions(raw_summary_mother)]
    show_thalassemia = (
        _thalassemia_from_state(state)
        if state
        else ground_kind(call, "preg_thalassemia", answers) in {"no", "unknown"}
    )
    if show_thalassemia and not any("थैलेसीमिया" in q for q in questions):
        questions.append(THALASSEMIA_LINE)
    if state:
        yes = []
        for topic in checklist.TOPICS:
            if topic["id"].startswith("intro_"):
                continue
            if _structured_kind(state, topic["id"]) != "yes":
                continue
            line = checklist.question_hi(topic["id"]).rstrip("?").strip()
            if line.startswith("क्या "):
                line = line[len("क्या "):]
            line = line.strip()
            note = _instance_note(state, topic["id"])
            if note:
                line = f"{line} ({note})"
            said = (state.get("topics") or {}).get(topic["id"], {}).get("value")
            if isinstance(said, str) and said.strip():
                spoken = hindi_text(said.strip().strip("।.! "))
                if spoken and not (call_state._content_stems(spoken) & call_state._content_stems(line)):
                    line = f"{line} ({spoken})"
            if line:
                yes.append(hindi_text(line))
    else:
        yes = [
            str(line).strip()
            for line in (extracted.get("yes_hindi") or [])
            if str(line).strip() and _has_hindi(str(line))
        ]
        yes = [hindi_text(line) for line in yes]
        grounded_yes = any(
            ground_kind(call, topic["id"], answers) == "yes"
            for topic in checklist.TOPICS
            if not topic["id"].startswith("intro_")
        )
        if not grounded_yes and not _user_text(call).strip():
            yes = []
        if not yes and _user_text(call).strip():
            for topic in checklist.TOPICS:
                if topic["id"].startswith("intro_"):
                    continue
                if ground_kind(call, topic["id"], answers) != "yes":
                    continue
                line = checklist.question_hi(topic["id"]).rstrip("?").strip()
                if line.startswith("क्या "):
                    line = line[len("क्या "):]
                line = line.strip()
                if line:
                    yes.append(line)
    groups = []
    unknown = []
    unasked = []
    not_asked_groups: list[str] = []
    for group in checklist.DETAIL_GROUPS:
        nos = []
        missed = []
        for topic in checklist.TOPICS:
            if topic["group"] != group:
                continue
            kind = _structured_kind(state, topic["id"]) if state else ground_kind(call, topic["id"], answers)
            if kind == "not_applicable":
                continue
            if kind in {"no", "answered"}:
                nos.append(checklist.question_hi(topic["id"]))
            elif kind == "unknown":
                unknown.append(checklist.question_hi(topic["id"]))
            elif kind in {"no_response", "not_asked"} and not topic.get("optional"):
                missed.append(checklist.question_hi(topic["id"]))
        if nos:
            groups.append({"title": checklist.GROUP_HI[group], "items": nos})
        if missed:
            unasked.append({"title": checklist.GROUP_HI[group], "items": missed})
            if incomplete:
                not_asked_groups.append(checklist.GROUP_HI[group])
    child = _children_label(state) if state else _child(call)
    who = _name(call) + (f" · {child}" if child else "")
    if state and str(state.get("close_reason") or "") == "emergency":
        emergency = True
    else:
        emergency = str((call.get("telemetry") or {}).get("close_reason") or "") == "emergency"
    return {
        "who": who,
        "date": date,
        "time": clock,
        "duration": _duration(call.get("duration_sec") or 0),
        "status": "अधूरी कॉल" if incomplete else "पूर्ण कॉल",
        "incomplete": incomplete,
        "emergency": emergency,
        "questions": questions,
        "yes": yes,
        "groups": groups,
        "unknown": unknown,
        "unasked": unasked,
        "not_asked": not_asked_groups,
        "started": when,
    }


def filename(call: dict) -> str:
    when = _ist(str(call.get("started_at") or ""))
    if when is None:
        return "SHISHU.pdf"
    return f"SHISHU-{when:%Y-%m-%d-%H%M}.pdf"


def render_html(report: dict) -> str:
    questions = "\n".join(f"<li>{html.escape(q)}</li>" for q in report["questions"])
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
        for bit in (report["date"], report["time"], report["duration"], report["status"])
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
<h1>डॉक्टर या स्वास्थ्य कार्यकर्ता (ASHA/ANM दीदी) से ये पूछें</h1>
<ol>{questions}</ol>
<p class="note">अगर इसमें कुछ गलत है, या आप कुछ और बताना चाहती हैं, तो दोबारा कॉल कर सकती हैं।</p>
<h2>इस कॉल में जो हाँ था</h2>
{yes}
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
