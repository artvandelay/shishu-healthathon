"""Server-owned call state, applicability rules, and question selection."""

import copy
import re

import checklist
import question_flow
import call_state_contract as contract

TOPIC_STATES = contract.TOPIC_STATES
MODEL_TOPIC_STATES = contract.MODEL_TOPIC_STATES
QUESTION_PHASES = contract.QUESTION_PHASES
END_REASONS = contract.END_REASONS
INTERNAL_CLOSE_REASONS = contract.INTERNAL_CLOSE_REASONS
PROFILE_FIELDS = contract.PROFILE_FIELDS

_SCOPES = ("all", "specific", "none")
_PARTNER_STATUSES = ("married", "about_to_marry")
_EVER_PREGNANT_NO = (
    "preg_conditions",
    "preg_scan",
    "preg_heart",
    "preg_transfusion",
    "preg_delivery",
    "fam_miscarriage",
)
_CHILD_GROUPS = ("birth", "body", "illness")
_FAMILY_YES_IDS = tuple(
    topic["id"]
    for topic in checklist.TOPICS
    if topic["group"] == "family" and topic["id"] != "fam_same_problem"
)

_NUM_WORDS = (
    ("उनतीस", "29"), ("अट्ठाईस", "28"), ("सत्ताईस", "27"), ("छब्बीस", "26"),
    ("पच्चीस", "25"), ("चौबीस", "24"), ("तेईस", "23"), ("बाईस", "22"), ("इक्कीस", "21"),
    ("अठारह", "18"), ("सत्रह", "17"), ("सोलह", "16"), ("पंद्रह", "15"), ("पन्द्रह", "15"),
    ("चौदह", "14"), ("तेरह", "13"), ("ग्यारह", "11"), ("बारह", "12"), ("आधा", "0.5"),
    ("बीस", "20"), ("तीस", "30"),
    ("पाँच", "5"), ("पांच", "5"), ("छह", "6"), ("छः", "6"),
    ("एक", "1"), ("दो", "2"), ("तीन", "3"), ("चार", "4"),
    ("सात", "7"), ("आठ", "8"), ("नौ", "9"), ("दस", "10"),
)
_MONTH_UNIT = r"(?:महीन\w*|mahine|months|month)"
_YEAR_UNIT = r"(?:साल|वर्ष|saal|years|year|varsh)"
_WEEK_UNIT = r"(?:हफ़्त\S*|हफ्त\S*|hafte|weeks|week)"
_DAY_UNIT = r"(?:दिन|din|days|day)"


def _fold_spoken_numbers(text: str) -> str:
    """Hindi number words, including the half-step ages, as digits.

    डेढ़ and ढाई are their own words. साढ़े / सवा / पौने modify the next number
    (साढ़े तीन is 3.5, सवा दो is 2.25, पौने दो is 1.75).
    """
    folded = text or ""
    folded = folded.replace("डेढ़", "1.5").replace("ढाई", "2.5")
    for word, number in _NUM_WORDS:
        # तीस must not eat तीसरे, and दो must not eat दोनों.
        folded = re.sub(rf"{re.escape(word)}(?![\u0900-\u097F])", number, folded)

    def _shift(match: re.Match[str], delta: float) -> str:
        value = float(match.group(1)) + delta
        return str(int(value) if value.is_integer() else value)

    folded = re.sub(r"साढ़े\s*(\d+(?:\.\d+)?)", lambda m: _shift(m, 0.5), folded)
    folded = re.sub(r"सवा\s*(\d+(?:\.\d+)?)", lambda m: _shift(m, 0.25), folded)
    folded = re.sub(r"पौने\s*(\d+(?:\.\d+)?)", lambda m: _shift(m, -0.25), folded)
    # आठवें महीने is the 8th month. तीसरे साल is the 3rd year. पहले stays
    # "before" unless a month, year, or week unit follows.
    for word, number in (
        ("आठवें", "8"), ("सातवें", "7"), ("नौवें", "9"), ("दसवें", "10"),
        ("छठे", "6"), ("पाँचवें", "5"), ("पांचवें", "5"),
        ("चौथे", "4"), ("तीसरे", "3"), ("दूसरे", "2"), ("पहले", "1"),
    ):
        folded = re.sub(rf"{word}(?=\s*(?:महीन|साल|वर्ष|year|हफ्त|हफ़्त))", number, folded)
    # सवा साल is one and a quarter. पौने साल is three quarters. आधे साल is half.
    # The same words before a week mean the same fraction of a week.
    unit_ahead = r"(?=\s*(?:साल|वर्ष|year|महीन|हफ्त|हफ़्त))"
    folded = re.sub(rf"सवा{unit_ahead}", "1.25", folded)
    folded = re.sub(rf"पौने{unit_ahead}", "0.75", folded)
    folded = re.sub(rf"आधे{unit_ahead}", "0.5", folded)
    return folded


# One procedure, however she says it. A nukta is already ignored by word matching.
_OPERATION_WORDS = (
    "operation", "caesarean", "cesarean", "c-section", "c section",
    "ऑपरेशन", "सीजेरियन", "सिजेरियन", "सी-सेक्शन", "सी सेक्शन",
)


def _when_operation(state: dict, topic: dict) -> bool:
    details = topic.get("details") or {}
    blob = f"{topic.get('value') or ''} {details.get('delivery_type') or ''}".lower()
    return any(_contains_word(blob, word) for word in _OPERATION_WORDS)


def _when_abnormal(state: dict, topic: dict) -> bool:
    result = str((topic.get("details") or {}).get("result") or "").lower()
    return result in {"abnormal", "not_normal", "not normal"}


def _when_partner(state: dict, topic: dict) -> bool:
    return state["profile"].get("marital_status") in _PARTNER_STATUSES


def _child_age_years(age: object) -> float | None:
    if isinstance(age, bool) or age is None:
        return None
    if isinstance(age, (int, float)):
        return float(age)
    if isinstance(age, dict):
        value = age.get("value")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        unit = str(age.get("unit") or "years").lower()
        if unit in {"month", "months", "mahine"}:
            years = float(value) / 12.0
        elif unit in {"week", "weeks", "hafte", "hafta"}:
            years = float(value) / 52.0
        elif unit in {"day", "days", "din"}:
            years = float(value) / 365.0
        else:
            years = float(value)
        plus = age.get("plus_days")
        if isinstance(plus, (int, float)) and not isinstance(plus, bool):
            years += float(plus) / 365.0
        return years
    if isinstance(age, str):
        cleaned = _fold_spoken_numbers(age.strip().lower())
        m = re.search(rf"(\d+(?:\.\d+)?)\s*{_MONTH_UNIT}", cleaned, re.I)
        if m:
            return float(m.group(1)) / 12.0
        m = re.search(rf"(\d+(?:\.\d+)?)\s*{_YEAR_UNIT}", cleaned, re.I)
        if m:
            return float(m.group(1))
        m = re.search(rf"(\d+(?:\.\d+)?)\s*{_WEEK_UNIT}", cleaned, re.I)
        if m:
            return float(m.group(1)) / 52.0
        m = re.search(rf"(\d+(?:\.\d+)?)\s*{_DAY_UNIT}", cleaned, re.I)
        if m:
            return float(m.group(1)) / 365.0
        m = re.search(r"^\s*(\d+(?:\.\d+)?)\s*$", cleaned)
        if m:
            return float(m.group(1))
    return None


def _canonical_sex(value: object) -> str | None:
    text = str(value or "").strip().lower()
    if text in {"female", "girl", "f", "beti", "daughter", "बेटी", "लड़की", "बच्ची"}:
        return "female"
    if text in {"male", "boy", "m", "beta", "bete", "son", "बेटा", "बेटे", "लड़का", "लड़के"}:
        return "male"
    return None


def _is_female(child: dict) -> bool:
    return _canonical_sex(child.get("sex")) == "female"


def _any_child_at_least(state: dict, years: float, female_only: bool = False) -> bool:
    for child in state["profile"].get("children") or []:
        if not isinstance(child, dict):
            continue
        if female_only and not _is_female(child):
            continue
        age = _child_age_years(child.get("age"))
        if age is not None and age >= years:
            return True
    return False


def _when_multiple_children(state: dict, topic: dict) -> bool:
    return len(state.get("profile", {}).get("children") or []) > 1


def _when_teeth(state: dict, topic: dict) -> bool:
    return _any_child_at_least(state, 5)


def _when_periods(state: dict, topic: dict) -> bool:
    return _any_child_at_least(state, 11, female_only=True)


def _follow(fid: str, note: str, when=None, after_no: bool = False) -> dict:
    item = {"id": fid, "note": note}
    if when is not None:
        item["when"] = when
    if after_no:
        item["after_no"] = True
    return item


def _build_topic_rules() -> dict:
    rules = {
        topic["id"]: {"skip": [], "followups": []}
        for topic in checklist.TOPICS
    }
    for topic in checklist.TOPICS:
        if topic["group"] in _CHILD_GROUPS:
            rules[topic["id"]]["skip"].append("no_children")
    for topic_id in _EVER_PREGNANT_NO:
        rules[topic_id]["skip"].append("ever_pregnant_no")
    rules["preg_delivery"]["skip"].append("has_delivered_no")
    rules["fam_related"]["skip"].append("no_spouse")
    rules["body_puberty"]["skip"].append("all_under_10")
    rules["fam_same_problem"]["skip"].append("same_problem_hidden")
    rules["birth_mode"]["followups"] = [
        _follow("delivery_type", "Ask whether the birth was a normal delivery or an operation, and where it happened."),
        _follow("place", "Ask whether the birth was at home or in a hospital."),
        _follow("operation_reason", "Ask the reason for the operation.", _when_operation),
        _follow("affected_child", "Ask which child was born by operation or had this delivery.", _when_multiple_children),
    ]
    rules["birth_term"]["followups"] = [
        _follow("amount", "Ask how early or how late the birth was."),
        _follow("affected_child", "Ask which child was early or late.", _when_multiple_children),
    ]
    rules["birth_nicu"]["followups"] = [
        _follow("reason", "Ask the reason for the NICU stay."),
        _follow("days", "Ask how many days the child stayed."),
        _follow("affected_child", "Ask which child was in NICU or kaanch ka box.", _when_multiple_children),
    ]
    rules["birth_bcg"]["followups"] = [
        _follow("reaction", "Ask whether the reaction was swelling, a boil, an armpit lump, or something else."),
        _follow("affected_child", "Ask which child had the reaction to BCG.", _when_multiple_children),
    ]
    rules["birth_hearing_vision"]["followups"] = [
        _follow("test_done", "Ask whether the newborn hearing and vision test was done."),
        _follow("result", "Ask whether the result was normal, abnormal, or unknown."),
        _follow("hearing_or_vision", "Ask whether the abnormal result was hearing, vision, or both.", _when_abnormal),
        _follow("affected_child", "Ask which child had the hearing or vision test issue.", _when_multiple_children),
    ]
    rules["birth_cleft"]["followups"] = [
        _follow("affected_child", "Ask which child has cleft lip or palate.", _when_multiple_children),
    ]
    rules["birth_cord_bleed"]["followups"] = [
        _follow("affected_child", "Ask which child had umbilical cord bleeding.", _when_multiple_children),
    ]
    rules["birth_cord_3weeks"]["followups"] = [
        _follow("affected_child", "Ask which child took more than 3 weeks for cord to fall.", _when_multiple_children),
    ]
    rules["body_fits_milestones"]["followups"] = [
        _follow("type", "Ask whether this was fits, late sitting, late walking, late speech, or more than one."),
        _follow("affected_child", "Ask which child is affected.", _when_multiple_children),
        _follow("onset", "Ask when it started."),
    ]
    rules["body_regression"]["followups"] = [
        _follow("lost_skill", "Ask which skill was lost."),
        _follow("affected_child", "Ask which child is affected.", _when_multiple_children),
    ]
    rules["body_blue"]["followups"] = [
        _follow("location", "Ask where the blue colour was."),
        _follow("timing", "Ask when it happens."),
        _follow("affected_child", "Ask which child is affected.", _when_multiple_children),
    ]
    rules["body_tummy"]["followups"] = [
        _follow("onset", "Ask when the tummy started getting bigger."),
        _follow("affected_child", "Ask which child is affected.", _when_multiple_children),
        _follow("clinician_comment", "Ask whether a clinician said the liver or spleen is enlarged."),
    ]
    rules["body_transfusion"]["followups"] = [
        _follow("affected_child", "Ask which child needed blood.", _when_multiple_children),
        _follow("count", "Ask how many transfusions."),
        _follow("last_date", "Ask the last transfusion date or age."),
    ]
    rules["body_blood_count"]["followups"] = [
        _follow("low_or_high", "Ask whether the count was low or high."),
        _follow("timing", "Ask when the count was checked."),
        _follow("affected_child", "Ask which child had abnormal blood count.", _when_multiple_children),
    ]
    rules["body_joint_swelling"]["followups"] = [
        _follow("joint", "Ask which joint swelled."),
        _follow("trigger", "Ask what brought the swelling on."),
        _follow("duration", "Ask how long it lasted."),
        _follow("affected_child", "Ask which child is affected.", _when_multiple_children),
    ]
    rules["body_bleeding_bruise"]["followups"] = [
        _follow("bleeding_or_bruises", "Ask whether the problem is prolonged bleeding, easy bruises, or both."),
        _follow(
            "teeth",
            "Ask about bleeding when a tooth falls. Ask only if a child is at least 5.",
            _when_teeth,
            after_no=True,
        ),
        _follow(
            "periods",
            "Ask about heavy periods. Ask only if a daughter is at least 11.",
            _when_periods,
            after_no=True,
        ),
        _follow("affected_child", "Ask which child had the prolonged bleeding or easy bruising.", _when_multiple_children),
    ]
    rules["body_private_parts"]["followups"] = [
        _follow("affected_child", "Ask which child is affected.", _when_multiple_children),
        _follow("description", "Ask the caller to describe what she sees. Do not add a clinical label."),
    ]
    rules["body_lumps_birth"]["followups"] = [
        _follow("location", "Ask where the lump is."),
        _follow("affected_child", "Ask which child is affected.", _when_multiple_children),
    ]
    rules["body_proportion"]["followups"] = [
        _follow("body_part", "Ask which body part is out of proportion."),
        _follow("affected_child", "Ask which child is affected.", _when_multiple_children),
    ]
    rules["body_thumb"]["followups"] = [
        _follow("finger_or_thumb", "Ask which finger or thumb is small or missing."),
        _follow("affected_child", "Ask which child is affected.", _when_multiple_children),
    ]
    rules["body_growth"]["followups"] = [
        _follow("affected_child", "Ask which child is affected.", _when_multiple_children),
        _follow("reported_by", "Ask who reported the growth concern."),
    ]
    rules["body_limb_pain"]["followups"] = [
        _follow("location", "Ask where the pain is."),
        _follow("frequency", "Ask how often it happens."),
        _follow("hospital_visits", "Ask whether it needed hospital visits."),
        _follow("affected_child", "Ask which child is affected.", _when_multiple_children),
    ]
    rules["body_puberty"]["followups"] = [
        _follow("affected_child", "Ask which child is affected.", _when_multiple_children),
        _follow("timing", "Ask whether puberty was earlier, later, or the same as peers."),
    ]
    rules["ill_admissions"]["followups"] = [
        _follow("affected_child", "Ask which child was admitted.", _when_multiple_children),
        _follow("count", "Ask how many admissions."),
        _follow("reasons", "Ask the reasons for admission."),
    ]
    rules["ill_iv_antibiotics"]["followups"] = [
        _follow("affected_child", "Ask which child is affected.", _when_multiple_children),
        _follow("count", "Ask how many times."),
        _follow("infection", "Ask which infection."),
    ]
    rules["ill_odor"]["followups"] = [
        _follow("source", "Ask whether the smell is urine, stool, or both."),
        _follow("affected_child", "Ask which child is affected.", _when_multiple_children),
    ]
    rules["ill_sudden"]["followups"] = [
        _follow("affected_child", "Ask which child is affected.", _when_multiple_children),
        _follow("event", "Ask what happened."),
        _follow("timing", "Ask when it happened."),
    ]
    rules["ill_fasting"]["followups"] = [
        _follow("symptom", "Ask whether the symptom is sleepiness, vomiting, fits, or more than one."),
        _follow("trigger", "Ask what triggered it."),
        _follow("count", "Ask how many times."),
        _follow("affected_child", "Ask which child is affected.", _when_multiple_children),
    ]
    rules["preg_conditions"]["followups"] = [
        _follow("conditions", "Ask whether it was blood pressure, sugar, thyroid, or more than one."),
        _follow("continued_after_delivery", "Ask whether each condition continued after delivery."),
    ]
    rules["preg_thalassemia"]["followups"] = [
        _follow("caller_test", "Ask whether the caller was tested for thalassemia."),
        _follow("caller_result", "Ask the caller's thalassemia result."),
        _follow("partner_test", "Ask whether the partner was tested. Skip this when no partner exists.", _when_partner),
        _follow("partner_result", "Ask the partner's thalassemia result. Skip this when no partner exists.", _when_partner),
    ]
    rules["preg_scan"]["followups"] = [
        _follow("scan_done", "Ask whether a pregnancy scan was done."),
        _follow("result", "Ask whether the scan was normal, abnormal, or unknown."),
    ]
    rules["preg_heart"]["followups"] = [
        _follow("report", "Ask what the heart report said."),
        _follow("when_found", "Ask whether it was found on a scan or at birth."),
    ]
    rules["preg_transfusion"]["followups"] = [
        _follow("pregnancy_number", "Ask which pregnancy."),
        _follow("month", "Ask which month of pregnancy."),
    ]
    rules["preg_delivery"]["followups"] = [
        _follow("problem", "Ask what the delivery problem was."),
        _follow("pregnancy", "Ask which pregnancy had the problem."),
    ]
    rules["fam_related"]["followups"] = [
        _follow("relationship", "Ask the exact relationship before marriage."),
    ]
    rules["fam_miscarriage"]["followups"] = [
        _follow("count", "Ask how many miscarriages or abortions."),
        _follow("month", "Ask the month when it is known."),
    ]
    rules["fam_infant_death"]["followups"] = [
        _follow("relationship", "Ask the relationship to the child who died."),
        _follow("age_at_death", "Ask the age at death."),
        _follow("cause", "Ask the known cause."),
    ]
    rules["fam_fits_delay"]["followups"] = [
        _follow("who", "Ask who in the family."),
        _follow("relationship", "Ask the relationship."),
        _follow("condition", "Ask whether it is fits, delay, lost skills, or more than one."),
    ]
    rules["fam_blind"]["followups"] = [
        _follow("who", "Ask who in the family."),
        _follow("relationship", "Ask the relationship."),
        _follow("onset_age", "Ask the age when vision was lost."),
    ]
    rules["fam_deaf"]["followups"] = [
        _follow("who", "Ask who in the family."),
        _follow("relationship", "Ask the relationship."),
        _follow("onset_age", "Ask the age when hearing was lost."),
    ]
    rules["fam_transfusion"]["followups"] = [
        _follow("who", "Ask who in the family."),
        _follow("relationship", "Ask the relationship."),
    ]
    rules["fam_bleeding"]["followups"] = [
        _follow("who", "Ask who in the family."),
        _follow("relationship", "Ask the relationship."),
        _follow("context", "Ask the bleeding context."),
    ]
    rules["fam_cancer"]["followups"] = [
        _follow("who", "Ask who in the family."),
        _follow("relationship", "Ask the relationship."),
    ]
    rules["fam_heart"]["followups"] = [
        _follow("who", "Ask who in the family."),
        _follow("relationship", "Ask the relationship."),
    ]
    rules["fam_kidney"]["followups"] = [
        _follow("who", "Ask who in the family."),
        _follow("relationship", "Ask the relationship."),
    ]
    rules["fam_bp_stroke"]["followups"] = [
        _follow("condition", "Ask whether it was high BP, sugar, cholesterol, or stroke."),
        _follow("who", "Ask who in the family."),
        _follow("relationship", "Ask whether the relative is immediate or distant."),
        _follow("onset_age", "Ask the age at onset."),
    ]
    rules["fam_maternal_side"]["followups"] = [
        _follow("who", "Ask who on the mother's side."),
        _follow("relationship", "Ask the relationship."),
        _follow("condition", "Ask whether it was bleeding, infection, skin patches, or more than one."),
    ]
    rules["fam_lumps"]["followups"] = [
        _follow("who", "Ask who in the family."),
        _follow("relationship", "Ask the relationship."),
        _follow("location", "Ask the location when it is known."),
    ]
    return rules


TOPIC_RULES = _build_topic_rules()


def new_call_state() -> dict:
    state = contract.new_call_state()
    _apply_rules(state)
    return state


def normalize_state(raw: object) -> dict:
    state = contract.normalize_state(raw)
    _apply_rules(state)
    return state


def _children_answer_state(state: dict) -> str | None:
    for event in reversed(state.get("history") or []):
        if event.get("kind") == "profile" and event.get("field") == "children":
            return event.get("answer_state")
    return None


def _no_children(state: dict) -> bool:
    return _children_answer_state(state) == "answered" and not state["profile"].get("children")


def _all_under_10(state: dict) -> bool:
    children = [child for child in state["profile"].get("children") or [] if isinstance(child, dict)]
    if not children:
        return False
    ages = [_child_age_years(child.get("age")) for child in children]
    if any(age is None for age in ages):
        return False
    return all(age < 10 for age in ages)


def _family_yes(state: dict) -> bool:
    for topic_id in _FAMILY_YES_IDS:
        topic = state["topics"][topic_id]
        if topic.get("state") == "answered" and _is_positive(topic, topic_id):
            return True
    return False


def _skip_reason(state: dict, topic_id: str) -> str | None:
    for name in TOPIC_RULES[topic_id]["skip"]:
        if name == "no_children" and _no_children(state):
            return "no_children"
        if name == "ever_pregnant_no" and state["profile"].get("ever_pregnant") is False:
            return "ever_pregnant_no"
        if name == "has_delivered_no" and state["profile"].get("has_delivered") is False:
            return "has_delivered_no"
        if name == "no_spouse" and state["profile"].get("marital_status") == "unmarried":
            return "no_spouse"
        if name == "all_under_10" and _all_under_10(state):
            return "all_under_10"
        if name == "same_problem_hidden" and not (state["profile"].get("children") and _family_yes(state)):
            return "same_problem_hidden"
    return None


def _apply_rules(state: dict) -> dict:
    for topic_id in contract.TOPIC_IDS:
        topic = state["topics"][topic_id]
        reason = _skip_reason(state, topic_id)
        if reason:
            if topic["state"] in ("answered", "unknown", "no_response"):
                continue
            topic["state"] = "not_applicable"
            topic["source"] = "server_rule"
            topic["reason"] = reason
            continue
        if topic["state"] == "not_applicable" and topic.get("source") == "server_rule":
            topic["state"] = "not_asked"
            topic["value"] = None
            topic["source"] = None
            topic["reason"] = None
    return state


def _detail_result(details: dict) -> str:
    for key in ("result", "caller_result", "partner_result"):
        text = str(details.get(key) or "").strip().lower()
        if text:
            return text
    return ""


def _is_reassuring_result(details: dict) -> bool:
    return _detail_result(details) in {"normal", "negative", "नॉर्मल"}


_AFFIRM_RE = re.compile(r"हाँ|हां|\bhaan\b|\byes\b", re.I)
_DENY_RE = re.compile(r"नहीं|नही|\bnahi\b|\bnahin\b|\bno\b", re.I)
_REASSURE_RE = re.compile(
    r"ठीक\s*है|ठीक\s*से|ठीक\s*रह|"
    r"theek\s*hai|thik\s*hai|कोई\s+(?:दिक्कत|परेशानी)\s+नहीं|no\s+problem|sab\s+theek|all\s+fine",
    re.I,
)


def _clauses(text: str) -> list[str]:
    return [
        bit.strip()
        for bit in re.split(r"[।!?]|[,،;]|(?<!\d)\.(?!\d)| और | and ", text or "")
        if bit.strip()
    ]


def _clause_is_reassurance(clause: str) -> bool:
    """ठीक है / ठीक से / ठीक रहा in this clause, and the clause does not also say हाँ or नहीं."""
    if not clause or _DENY_RE.search(clause) or _AFFIRM_RE.search(clause):
        return False
    return _REASSURE_RE.search(clause) is not None


_BARE_CLOSER_RE = re.compile(
    r"(?:(?:जी\s+)?(?:और\s*कुछ\s*नहीं|इतना\s*ही|नहीं|नही|बस)[।,!\s]*)+",
    re.I,
)


def _only_denial(text: str) -> bool:
    """A bare नहीं does not erase a different clause. A नहीं that names who or what does.

    और कुछ नहीं, पेट में दर्द रहता है keeps the pain. मेरे बच्चे को नहीं is the
    answer even when another clause says the problem exists for someone else.
    झटके नहीं आते is one clause, so it stays a no.
    """
    bits = _clauses(text)
    if not bits:
        return False
    content_denial = False
    bare_denial = False
    fact = False
    for bit in bits:
        if _DENY_RE.search(bit) and not _AFFIRM_RE.search(bit):
            if _BARE_CLOSER_RE.fullmatch(bit):
                bare_denial = True
            else:
                content_denial = True
            continue
        if _BARE_CLOSER_RE.fullmatch(bit):
            continue
        fact = True
    if content_denial or not bare_denial:
        return content_denial
    return not fact


def _only_reassurance(text: str) -> bool:
    """Every clause says it is fine. A different clause that names a problem does not.

    निशान ठीक से भरा है is reassurance. पेशाब में बदबू है, पाखाना ठीक है is not,
    because the smell is a separate clause.
    """
    bits = _clauses(text)
    return bool(bits) and all(_clause_is_reassurance(bit) for bit in bits)


def _stated_polarity(value: object) -> str | None:
    """Yes or no when the stored words say so, including जी हाँ and नहीं जी.

    A sentence that contains both is left unset. Delivery type and a normal
    result are decided before this.
    """
    if value is True:
        return "yes"
    if value is False:
        return "no"
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    affirmed = _AFFIRM_RE.search(text) is not None
    denied = _DENY_RE.search(text) is not None
    if affirmed and not denied:
        return "yes"
    if denied and not affirmed:
        return "no" if _only_denial(text) else None
    if _only_reassurance(text):
        return "no"
    return None


def _affirmative_value(value: object) -> bool:
    return _stated_polarity(value) == "yes"


_BARE_ACK_RE = re.compile(r"^(?:जी|ji|हम्म|hmm)[।.!\s]*$", re.I)


def _is_bare_ack(value: object) -> bool:
    """जी। by itself is an acknowledgement, not a symptom and not a no."""
    return isinstance(value, str) and _BARE_ACK_RE.match(value.strip()) is not None


def _followup_is_finding(value: object) -> bool:
    """A follow-up that describes a problem, not a no, an unknown, or a normal result."""
    if value is True:
        return True
    if value is False or value is None or isinstance(value, dict) or _is_bare_ack(value):
        return False
    if _stated_polarity(value) == "no" or _is_unknown_utterance(str(value)):
        return False
    text = str(value).strip().lower().strip("।.!")
    return text not in {"", "no", "nahi", "nahin", "normal", "negative", "नहीं", "नही", "नॉर्मल", "unknown"}


# These answers are classified by their own words (normal, operation, a test
# result, growth, puberty). A leftover sentence there is not automatically a
# finding. Every other clinical topic asks whether a problem happened.
_DESCRIBED_SKIP = {
    "birth_mode",
    "birth_term",
    "body_growth",
    "body_puberty",
    "birth_hearing_vision",
    "preg_scan",
    "preg_thalassemia",
}


def _names_reassuring_normal(value: object) -> bool:
    """normal / नॉर्मल / सामान्य, unless that word itself is denied."""
    text = str(value or "")
    for word in ("normal", "नॉर्मल", "सामान्य"):
        if _contains_word(text, word) and not _word_denied(text, word):
            return True
    return False


_QUESTION_FUNCTION_WORDS = frozenset({
    "क्या", "या", "है", "हैं", "था", "थी", "थे", "को", "के", "की", "में", "से", "का",
    "और", "नहीं", "नही", "पर", "भी", "तो", "यह", "ये", "वो", "वह", "कोई", "एक",
    "बच्चा", "बच्चे", "बच्चों", "बच्ची",
})


def _content_stems(text: str) -> set[str]:
    stems = set()
    for word in re.findall(r"[\u0900-\u097F]+", text or ""):
        if word in _QUESTION_FUNCTION_WORDS or len(word) < 3:
            continue
        stems.add(word[:3])
    return stems


def _restates_negative_question(topic_id: str | None, value: object) -> bool:
    """The question's own yes contains नहीं. Repeating that negative is a finding.

    नहीं। and नहीं जी stay a no. झटके नहीं आते stays a no, because that question
    does not contain नहीं. Denying the property (छोटी नहीं) is a no. Saying the
    named thing is absent (अंगूठा नहीं है) is a finding when the question offers
    absence as its own yes.
    """
    if not topic_id or not isinstance(value, str):
        return False
    question = checklist.question_hi(topic_id)
    if "नहीं" not in question and "नही" not in question:
        return False
    if not _DENY_RE.search(value):
        return False
    said = _content_stems(value)
    neg_stems: set[str] = set()
    for bit in re.split(r"[।,?]| या ", question):
        if "नहीं" in bit or "नही" in bit:
            neg_stems |= _content_stems(bit)
    if neg_stems:
        return bool(neg_stems & said)
    if not re.search(r"या\s+नहीं\s+है", question):
        return False
    properties = {
        match.group(1)
        for match in re.finditer(r"([\u0900-\u097F]+)\s+है", question)
        if match.group(1) not in _QUESTION_FUNCTION_WORDS and match.group(1) not in {"नहीं", "नही"}
    }
    before = re.search(r"([\u0900-\u097F]+)\s+(?:नहीं|नही)", value)
    word = before.group(1) if before else ""
    if not word or word in properties or word in _QUESTION_FUNCTION_WORDS or len(word) < 3:
        return False
    return word[:3] in _content_stems(question)


def _cord_within_three_weeks(text: str) -> bool:
    """One or two weeks is not 'more than three weeks'. ज़्यादा keeps the finding."""
    folded = _fold_spoken_numbers(text or "")
    if re.search(r"ज़्यादा|ज्यादा|more", folded, re.I):
        return False
    weeks = re.search(rf"(\d+)\s*{_WEEK_UNIT}", folded, re.I)
    if weeks and int(weeks.group(1)) <= 3:
        return True
    days = re.search(r"(\d+)\s*(?:दिन|days|day)", folded, re.I)
    if days and int(days.group(1)) <= 21:
        return True
    return bool(re.search(r"जल्दी|जल्द", folded))


def _described_problem(topic_id: str | None, value: object, details: dict) -> bool:
    """A stored sentence that names the problem, with no हाँ and no नहीं.

    A place, a normal result, a test that was merely done, and reassuring
    growth or puberty stay out of the हाँ list. A problem the details already
    recorded (a condition, a NICU reason, an early or late timing) counts.
    """
    if not topic_id or topic_id.startswith("intro_"):
        return False
    if topic_id == "close_anything_else":
        # बस। and नहीं, बस इतना ही close the call. A complaint does not.
        if re.fullmatch(
            r"\s*(?:जी\s+)?(?:बस|इतना\s*ही|और\s*कुछ\s*नहीं)[।.!\s]*",
            str(value or ""),
            re.I,
        ):
            return False
    if _is_reassuring_result(details):
        return False
    if isinstance(value, str) and _AFFIRM_RE.search(value) and _DENY_RE.search(value):
        return False
    if _is_bare_ack(value):
        return False
    if str(details.get("result") or "").strip().lower() in {"abnormal", "असामान्य", "positive", "carrier"}:
        return True
    if details.get("conditions"):
        return True
    reason = str(details.get("reason") or "").strip().lower()
    if reason and reason not in {"normal", "नॉर्मल", "unknown", "none", "noted"}:
        return True
    if str(details.get("timing") or "") in {"early", "late"}:
        return True
    if str(details.get("term") or "") in {"early", "late"}:
        return True
    if _lab_result(str(value or "")) == "abnormal":
        return True
    if _names_reassuring_normal(value):
        return False
    if topic_id == "birth_cord_3weeks" and _cord_within_three_weeks(str(value or "")):
        return False
    if topic_id in _DESCRIBED_SKIP:
        return False
    if topic_id == "preg_delivery" and _delivery_was_only_the_event(value):
        return False
    return _followup_is_finding(value)


def _delivery_was_only_the_event(value: object) -> bool:
    """प्रसव हो गया names the delivery. प्रसव में खून बह गया names a problem."""
    stems = _content_stems(str(value or ""))
    event = _content_stems("प्रसव डिलीवरी delivery हुआ हुई हुए हो गया गई")
    return bool(stems) and stems <= event


def _is_positive(topic: dict, _topic_id: str | None = None) -> bool:
    if topic.get("state") != "answered":
        return False
    details = topic.get("details") or {}
    if _is_reassuring_result(details):
        return False
    dtype = str(details.get("delivery_type") or "").strip().lower()
    if dtype in {"normal", "नॉर्मल"}:
        return False
    if any(_contains_word(dtype, word) for word in _OPERATION_WORDS):
        return True
    value = topic.get("value")
    if _stated_polarity(value) == "yes":
        return True
    if _stated_polarity(value) == "no":
        return _restates_negative_question(_topic_id, value)
    if _topic_id and _topic_id.startswith("fam_") and _topic_id != "fam_same_problem":
        blob = " ".join(
            str(part or "")
            for part in (
                value,
                details.get("relationship"),
                details.get("who"),
            )
        )
        if _has_kinship(blob):
            return True
    if isinstance(value, str) and value.strip().lower() in {
        "yes", "haan", "positive", "early", "late", "operation", "caesarean", "cesarean", "c-section", "abnormal",
        "हाँ", "हां", "ऑपरेशन", "सीजेरियन", "असामान्य",
    }:
        return True
    if _names_reassuring_normal(value):
        return False
    return _described_problem(_topic_id, value, details)


def topic_answer_kind(topic: dict, topic_id: str | None = None) -> str:
    if not isinstance(topic, dict):
        return "not_asked"
    if _is_positive(topic, topic_id):
        return "yes"
    status = str(topic.get("state") or "not_asked")
    value = topic.get("value")
    details = topic.get("details") or {}
    if status == "answered" and (
        value is False
        or str(details.get("result") or "").strip().lower() in {"normal", "negative", "नॉर्मल"}
        or str(details.get("delivery_type") or "").strip().lower() in {"normal", "नॉर्मल"}
        or _stated_polarity(value) == "no"
        or _names_reassuring_normal(value)
        or str(value or "").strip().lower() in {"no", "nahi", "nahin", "normal", "negative", "नहीं", "नही", "नॉर्मल", "सामान्य"}
    ):
        return "no"
    if status in {"unknown", "no_response", "not_asked", "not_applicable"}:
        return status
    if status == "answered":
        return "answered"
    return "not_asked"


_KINSHIP_WORDS = {
    "maata", "mata", "pita", "mummy", "papa", "mother", "father", "dad", "mom",
    "bhai", "behen", "sister", "brother", "beta", "beti", "son", "daughter",
    "pati", "husband", "patni", "wife", "dada", "dadi", "chacha", "chachi",
    "tau", "tai", "bua", "phupha", "nana", "nani", "mama", "mami", "masi",
    "mausi", "mausa", "sasur", "saas", "devar", "jeth", "nanad", "cousin",
    "door", "khandaan", "khandan", "saga", "sagga", "immediate", "distant",
    "paternal", "maternal", "relative", "rishtedaar", "rishtedar",
    "माता", "पिता", "मम्मी", "पापा", "भाई", "भैया", "भइया", "बहन", "दीदी", "बेटा", "बेटी", "पति", "पत्नी",
    "दादा", "दादी", "चाचा", "चाची", "ताऊ", "ताई", "बुआ", "फूफा", "नाना", "नानी",
    "मामा", "मामी", "मौसी", "मासी", "मौसा", "ससुर", "सास", "देवर", "जेठ", "ननद",
    "भाभी", "पोता", "पोती", "नाती", "नातिन", "साला", "साली",
    "कज़िन", "कजिन", "दूर", "खानदान", "सगा", "सगे", "सगी", "रिश्तेदार",
    "ससुराल",
}


def _token_is_kin(token: str) -> bool:
    """चाचा, चाचे, and चाचाजी are the same relative. मामला is not मामा.

    Only an ा-ending word inflects this way. रिश्तेदार must not swallow रिश्तेदारी,
    which names the idea of a relation and not which relative.
    """
    token = re.sub(r"जी$", "", token or "")
    if token in _KINSHIP_WORDS:
        return True
    for word in _KINSHIP_WORDS:
        if len(word) < 4 or not word.endswith("ा"):
            continue
        stem = word[:-1]
        if not token.startswith(stem):
            continue
        rest = token[len(stem):]
        if re.fullmatch(r"[\u093e-\u094c]*", rest or ""):
            return True
    return False


def _has_kinship(text: str) -> bool:
    raw = (text or "").lower()
    blob = re.sub(r"[^\w\s\u0900-\u097F]", " ", raw)
    return any(_token_is_kin(word) for word in blob.split())


def _find_child_id(state: dict, raw: object) -> str | None:
    if not raw or not isinstance(raw, str):
        return None
    raw_cleaned = raw.strip().lower()
    cleaned = _fold_spoken_numbers(raw_cleaned)
    children = state.get("profile", {}).get("children") or []
    for c in children:
        cid = str(c.get("id") or "").lower()
        cname = str(c.get("name") or "").lower()
        if raw_cleaned in (cid, cname) and raw_cleaned:
            return c.get("id")
        if cname and (cname == raw_cleaned or cname in raw_cleaned.split() or f"{cname} " in f"{raw_cleaned} "):
            return c.get("id")
    for c in children:
        age_dict = c.get("age")
        age = _child_age_years(age_dict)
        if age is not None:
            age_int = int(age) if age.is_integer() else age
            if (
                f"{age_int} saal" in cleaned
                or f"{age_int} year" in cleaned
                or f"{age_int} साल" in cleaned
                or f"{age_int} वर्ष" in cleaned
            ):
                return c.get("id")
            months = None
            if isinstance(age_dict, dict) and str(age_dict.get("unit") or "").lower() in {"month", "months", "mahine"}:
                months = age_dict.get("value")
            elif age < 1.0:
                months = round(age * 12)
            if months is not None:
                m_int = int(months) if isinstance(months, (int, float)) and float(months).is_integer() else months
                if (
                    f"{m_int} mahine" in cleaned
                    or f"{m_int} month" in cleaned
                    or f"{m_int} महीने" in cleaned
                    or f"{m_int} माह" in cleaned
                ):
                    return c.get("id")
    aged = [
        (c.get("id"), _child_age_years(c.get("age")))
        for c in children
        if isinstance(c, dict) and _child_age_years(c.get("age")) is not None
    ]
    if len(aged) >= 2:
        aged.sort(key=lambda item: item[1], reverse=True)
        if aged[0][1] != aged[-1][1]:
            ranked = [child_id for child_id, _age in aged]
            if any(w in cleaned for w in ("bada", "bade", "badi", "older", "elder", "बड़ा", "बड़े", "बड़ी", "pehle", "pehla", "pehli", "पहले", "पहला", "पहली", "first")):
                return ranked[0]
            if any(w in cleaned for w in ("chhota", "chhote", "chhoti", "chota", "chote", "younger", "छोटा", "छोटे", "छोटी")):
                return ranked[-1]
            # दूसरा is the second child by age, which is the younger of two and the middle of three.
            if any(w in cleaned for w in ("dusra", "dusre", "dusri", "दूसरा", "दूसरे", "दूसरी", "second")):
                return ranked[1]
            if len(ranked) >= 3 and any(w in cleaned for w in ("tisra", "tesra", "तीसरा", "तीसरे", "तीसरी", "third")):
                return ranked[2]
            if len(ranked) >= 4 and any(w in cleaned for w in ("chautha", "chauthe", "chauthi", "चौथा", "चौथे", "चौथी", "fourth")):
                return ranked[3]
            if len(ranked) >= 5 and any(w in cleaned for w in ("panchva", "paanchva", "पाँचवाँ", "पाँचवें", "पाँचवीं", "पांचवां", "fifth")):
                return ranked[4]
            if len(ranked) >= 3 and any(w in cleaned for w in ("beech", "बीच", "middle")):
                return ranked[len(ranked) // 2]
    females = [c["id"] for c in children if isinstance(c, dict) and _is_female(c) and c.get("id")]
    males = [
        c["id"] for c in children
        if isinstance(c, dict) and str(c.get("sex") or "").strip().lower() in {"male", "boy", "m", "beta", "bete", "son", "बेटा", "बेटे", "लड़का", "लड़के"} and c.get("id")
    ]
    if len(females) == 1 and any(w in cleaned for w in ("beti", "ladki", "girl", "daughter", "बेटी", "लड़की", "बच्ची")):
        return females[0]
    # बच्चा is the generic word for a child, so it does not select the only boy.
    if len(males) == 1 and any(w in cleaned for w in ("beta", "bete", "ladka", "ladke", "boy", "son", "बेटा", "बेटे", "लड़का", "लड़के")):
        return males[0]
    return None


def _followup_resolved(state: dict, topic: dict, followup_id: str) -> bool:
    details = topic.get("details") or {}
    children = state.get("profile", {}).get("children") or []

    if followup_id == "delivery_type":
        blob = f"{topic.get('value') or ''} {details.get('delivery_type') or ''}".lower()
        if any(_contains_word(blob, word) for word in ("operation", "caesarean", "cesarean", "normal", "home", "hospital")):
            return True

    if followup_id == "affected_child":
        if len(children) <= 1:
            return True
        if topic.get("instances"):
            valid_ids = _known_child_ids(children)
            if any(inst.get("person_id") in valid_ids for inst in topic["instances"]):
                return True
        val = details.get("affected_child")
        if val is not None:
            if isinstance(val, dict) and val.get("state") in ("unknown", "no_response"):
                return True
            sval = str(val).strip().lower()
            if _means_every_child(sval):
                for c in children:
                    cid = c.get("id")
                    if cid and not any(inst.get("person_id") == cid for inst in topic.get("instances") or []):
                        topic.setdefault("instances", []).append({
                            "person_id": cid,
                            "value": topic.get("value"),
                            "details": dict(details),
                        })
                return True
            matched = _find_child_id(state, val)
            if matched:
                if not any(inst.get("person_id") == matched for inst in topic.get("instances") or []):
                    topic.setdefault("instances", []).append({
                        "person_id": matched,
                        "value": topic.get("value"),
                        "details": dict(details),
                    })
                return True
            return False
        return False

    if followup_id in ("relationship", "who"):
        val = details.get(followup_id)
        other = details.get("who" if followup_id == "relationship" else "relationship")
        if isinstance(val, dict) and val.get("state") in ("unknown", "no_response"):
            return True
        if isinstance(other, dict) and other.get("state") in ("unknown", "no_response"):
            return True
        if str(val or "").strip().lower() in ("noted", "unknown", "no_response", "skip"):
            return True
        if _has_kinship(str(val or "")) or _has_kinship(str(other or "")):
            return True
        if followup_id in details and val is not None:
            if len(str(val).split()) >= 2:
                return True
            return False
        return False

    if followup_id not in details:
        return False
    return details.get(followup_id) is not None


def _pending_followups(state: dict, topic_id: str) -> list[dict]:
    topic = state["topics"][topic_id]
    positive = _is_positive(topic, topic_id)
    answered_no = topic.get("state") == "answered" and (
        topic.get("value") is False or _stated_polarity(topic.get("value")) == "no"
    )
    if not positive and not answered_no:
        return []
    pending = []
    for spec in TOPIC_RULES[topic_id]["followups"]:
        if not positive and not spec.get("after_no"):
            continue
        check = spec.get("when")
        if check is not None and not check(state, topic):
            continue
        if not _followup_resolved(state, topic, spec["id"]):
            pending.append(spec)
    return pending


def _topic_resolved(state: dict, topic_id: str) -> bool:
    topic = state["topics"][topic_id]
    status = topic.get("state")
    if status in ("unknown", "no_response", "not_applicable"):
        return True
    if status != "answered":
        return False
    return not _pending_followups(state, topic_id)


def _topic_question(topic_id: str) -> dict:
    meta = checklist._BY_ID[topic_id]
    words = question_flow.WIDE_WORDS.get(topic_id) or meta["words"]
    return {
        "topic_id": topic_id,
        "followup_id": None,
        "question_kind": "topic",
        "prompt_note": (
            f"Ask one short Hindi question for {topic_id}: {meta['need']}. "
            f"Suggested words: {words}. Then stop and wait. The server selects routing."
        ),
    }


def _followup_question(topic_id: str, spec: dict, state: dict | None = None) -> dict:
    prompt_note = (
        f"Ask only this follow-up for {topic_id}. "
        f"Record the answer in details.{spec['id']}. {spec['note']} Then stop and wait."
    )
    if state and spec["id"] == "affected_child":
        children = state.get("profile", {}).get("children") or []
        kids_desc = []
        for c in children:
            age_s = f"{c.get('age', {}).get('value')} saal" if isinstance(c.get('age'), dict) else ""
            n_s = f"'{c.get('name')}'" if c.get('name') else ""
            kids_desc.append(f"{c.get('id')} ({age_s} {n_s})".strip())
        summary = ", ".join(kids_desc)
        raw_val = (state["topics"].get(topic_id) or {}).get("details", {}).get("affected_child")
        if raw_val and str(raw_val).strip():
            prompt_note = (
                f"Caller mentioned child reference '{raw_val}', which is not mapped to an age yet. "
                f"Ask: '{raw_val} kis umar wale bachche hain?' (Known: {summary}). "
                f"When she answers, update profile.children to save the child's name, and pass child_ids."
            )
        else:
            prompt_note = (
                f"Ask which child had this problem ({summary}). "
                f"If she names a child, ask their age if not known. Record in details.affected_child and pass child_ids."
            )
    elif spec["id"] in ("relationship", "who"):
        prompt_note = (
            f"Ask who in the family and their exact kinship (e.g. mama, chacha, bua, maata-pita). "
            f"A name alone without kinship is not enough. Record in details.relationship."
        )

    return {
        "topic_id": topic_id,
        "followup_id": spec["id"],
        "question_kind": "followup",
        "prompt_note": prompt_note,
    }


def _profile_question(field: str) -> dict:
    notes = {
        "ever_pregnant": "Ask whether she has ever been pregnant. Do not ask a pregnancy-history topic yet.",
        "has_delivered": "Ask whether she has ever delivered. Do not ask the delivery topic yet.",
    }
    return {
        "topic_id": f"profile.{field}",
        "followup_id": None,
        "question_kind": "topic",
        "prompt_note": notes.get(field, f"Ask profile.{field}. The server selects routing."),
    }


def _closing_question(state: dict) -> dict:
    reason = state.get("close_reason") or "complete"
    if reason == "complete":
        note = (
            "Speak the full closing in Hindi. Do not call end_call. "
            "The server ends the call after your audio finishes."
        )
    elif reason == "emergency":
        note = "Speak the emergency closing in Hindi. The server ends the call after your audio finishes."
    elif reason == "caller_stop":
        note = "Speak the caller-stop closing in Hindi. The server ends the call after your audio finishes."
    else:
        note = (
            "Speak the incomplete closing in Hindi. Do not call end_call. "
            "The server ends the call after your audio finishes."
        )
    current = state.get("current_question") or {}
    return {
        "topic_id": current.get("topic_id"),
        "followup_id": None,
        "question_kind": "closing",
        "prompt_note": note,
    }


def state_context(state: dict) -> dict:
    _apply_rules(state)
    topics = {}
    for topic_id, topic in state["topics"].items():
        topics[topic_id] = {
            "state": topic.get("state"),
            "value": topic.get("value"),
            "details": topic.get("details") or {},
            "instances": topic.get("instances") or [],
        }
    return {
        "phase": state.get("phase"),
        "profile": state.get("profile"),
        "current_question": state.get("current_question"),
        "close_reason": state.get("close_reason"),
        "topics": topics,
    }


def _known_child_ids(children: list) -> set[str]:
    return {
        str(child.get("id"))
        for child in children
        if isinstance(child, dict) and child.get("id")
    }


def _normalize_children(value: object) -> list[dict]:
    if isinstance(value, dict):
        items = [value]
    elif isinstance(value, list):
        items = value
    else:
        items = []
    used: set[str] = set()
    children = []
    for item in items:
        raw = item if isinstance(item, dict) else {}
        child_id = str(raw.get("id") or "")
        if not child_id.startswith("child_") or child_id in used:
            number = 1
            while f"child_{number}" in used:
                number += 1
            child_id = f"child_{number}"
        used.add(child_id)
        age = raw.get("age")
        if isinstance(age, str):
            age_years = _child_age_years(age)
            if age_years is not None:
                if any(u in age.lower() for u in ("महीन", "mahine", "month")):
                    m_val = round(age_years * 12)
                    age = {"value": m_val, "unit": "months"}
                else:
                    age = {"value": int(age_years) if age_years.is_integer() else age_years, "unit": "years"}
        sex = _canonical_sex(raw.get("sex")) or raw.get("sex")
        children.append({
            "id": child_id,
            "name": raw.get("name"),
            "age": age,
            "sex": sex,
        })
    return children


def _is_duration_after(text: str, end: int) -> bool:
    """'छह महीने से' is how long something has lasted, not an age.

    The month regex can stop before a Hindi vowel sign, so skip the rest of
    that word before looking for से.
    """
    rest = re.sub(r"^\S*", "", text[end:])
    # A vowel sign is not a word character, so \b does not see the end of से.
    return re.match(r"\s*(?:से|se)(?=\s|$|[।,.])", rest, re.I) is not None


def _spoken_pregnancy_index(text: str) -> int | None:
    """Which pregnancy: पहली प्रेग्नेंसी, दूसरी गर्भावस्था, or 2nd pregnancy.

    पहली बार is the first time, so it counts only on a pregnancy question.
    """
    raw = text or ""
    for word, number in (
        ("पहली", 1), ("पहला", 1), ("first", 1),
        ("दूसरी", 2), ("दूसरा", 2), ("second", 2),
        ("तीसरी", 3), ("तीसरा", 3), ("third", 3),
        ("चौथी", 4), ("चौथा", 4), ("fourth", 4),
    ):
        if re.search(rf"{word}\s*(?:प्रेग्न|गर्भावस्थ|pregnancy|बार)", raw, re.I):
            return number
    folded = _fold_spoken_numbers(raw)
    match = re.search(r"(?<![\d.])(\d+)(?!\.\d)\s*(?:प्रेग्न|गर्भावस्थ|pregnancy)", folded, re.I)
    if match:
        return int(match.group(1))
    return None


def _spoken_times(text: str) -> int | None:
    """How many times, as in दो बार or दूसरी बार. A day count is not a time count."""
    folded = _fold_spoken_numbers(text or "")
    # 2.25 बार must not be read as 25 times. The digits after the dot are not the count.
    match = re.search(r"(?<![\d.])(\d+)(?!\.\d)\s*(?:बार|times|time)", folded, re.I)
    if match:
        return int(match.group(1))
    for word, number in (
        ("पहली", 1), ("पहला", 1), ("दूसरी", 2), ("दूसरा", 2),
        ("तीसरी", 3), ("तीसरा", 3), ("चौथी", 4), ("चौथा", 4),
    ):
        if re.search(rf"{word}\s*(?:बार|time)", text or "", re.I):
            return number
    return None


def _store_spoken_number(details: dict, key: str, parser, utterance: str) -> None:
    """The count she just said replaces a stored count. An integer stays when she did not say one."""
    parsed = parser(utterance or "")
    if parsed is not None:
        details[key] = parsed
        return
    current = details.get(key)
    if type(current) is int:
        return
    parsed = parser("" if current is None else str(current))
    if parsed is not None:
        details[key] = parsed


def _age_phrase(record: dict) -> str:
    value = record.get("value")
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    unit = {
        "months": "महीने",
        "years": "साल",
        "weeks": "हफ़्ते",
        "days": "दिन",
    }.get(record.get("unit") or "", "")
    if value is None or not unit:
        return ""
    return f"{value} {unit}"


def _whole_number(raw: str) -> int:
    number = float(raw)
    return int(number) if number.is_integer() else int(round(number))


def _month_number(raw: str) -> int | float:
    """Keep साढ़े सात as 7.5. Do not round a half month up to the next month."""
    number = float(raw)
    return int(number) if number.is_integer() else number


def _month_phrase(month: object) -> str:
    if isinstance(month, bool) or not isinstance(month, (int, float)):
        return str(month or "")
    number = float(month)
    if abs(number - 1.5) < 1e-6:
        return "डेढ़ महीने"
    if abs(number - 2.5) < 1e-6:
        return "ढाई महीने"
    if number > 2 and abs(number - int(number) - 0.5) < 1e-6:
        return f"साढ़े {int(number)} महीने"
    if abs(number - int(number) - 0.25) < 1e-6 and int(number) >= 1:
        return f"सवा {int(number)} महीने"
    if abs(number - int(number) - 0.75) < 1e-6:
        return f"पौने {int(number) + 1} महीने"
    shown = int(number) if number.is_integer() else number
    return f"{shown} महीने"


def _day_phrase(day: object) -> str:
    """डेढ़ दिन stays a day and a half. A whole number stays N दिन."""
    if isinstance(day, bool) or not isinstance(day, (int, float)):
        return str(day or "")
    number = float(day)
    if abs(number - 1.5) < 1e-6:
        return "डेढ़ दिन"
    if abs(number - 2.5) < 1e-6:
        return "ढाई दिन"
    if number > 2 and abs(number - int(number) - 0.5) < 1e-6:
        return f"साढ़े {int(number)} दिन"
    shown = int(number) if number.is_integer() else number
    return f"{shown} दिन"


def _spoken_month_count(text: str) -> int | float | None:
    """A pregnancy month. N महीने का is the child's age. N महीने पहले is how long ago."""
    folded = _fold_spoken_numbers(text or "")
    born = re.search(
        rf"(\d+(?:\.\d+)?)\s*{_MONTH_UNIT}\S*(?:\s+में|\s+का\s+पैदा)",
        folded,
        re.I,
    )
    if born:
        return _month_number(born.group(1))
    for match in re.finditer(rf"(\d+(?:\.\d+)?)\s*{_MONTH_UNIT}", folded, re.I):
        rest = re.sub(r"^\S*", "", folded[match.end():])
        if re.match(r"\s*का\b", rest):
            continue
        if re.match(r"\s*पहले", rest):
            continue
        if _is_duration_after(folded, match.end()):
            continue
        return _month_number(match.group(1))
    return None


def _spoken_day_count(text: str) -> int | None:
    """A stay of N days. A symptom's own duration is not the stay.

    N दिन का बुखार and N दिन से बुखार are the fever. N हफ़्ते रखा is that many
    weeks. N हफ़्ते का है is the child's age.
    """
    folded = _fold_spoken_numbers(text or "")
    stay_word = r"(?:रखा|रखी|रहे|रहा|रही|तक)"
    symptom = r"(?:बुखार|झटके|fits|seizure|दौरे|पीलिया|jaundice|संक्रमण|infection|इंफेक्शन|साँस)"

    def region_after(end: int) -> str:
        window = folded[end:end + 80]
        nxt = re.search(r"\d+(?:\.\d+)?\s*(?:दिन|days|day|हफ़्त|हफ्त|hafte|weeks|week)", window, re.I)
        return window[: nxt.start()] if nxt else window

    day_hits = []
    for match in re.finditer(r"(\d+(?:\.\d+)?)\s*(?:दिन|days|day)", folded, re.I):
        region = region_after(match.end())
        is_stay = re.search(stay_word, region) is not None
        is_symptom = re.match(rf"\S*\s*(?:का\s+|से\s+){symptom}", region, re.I) is not None
        day_hits.append((_month_number(match.group(1)), is_stay, is_symptom))
    for count, is_stay, _symptom in day_hits:
        if is_stay:
            return count
    for count, _stay, is_symptom in day_hits:
        if not is_symptom:
            return count

    for match in re.finditer(rf"(\d+(?:\.\d+)?)\s*{_WEEK_UNIT}", folded, re.I):
        if _followed_by_age_copula(folded, match.end()):
            continue
        region = region_after(match.end())
        if re.search(stay_word, region) or not re.search(r"(?:का|की|के)\s*(?:है|हैं|था|थी|थे)", region):
            # A bare week count with a stay, or a week count that is not an age.
            if re.search(stay_word, region) or region.strip() in {"", "।", "."}:
                days = float(match.group(1)) * 7
                return int(days) if days.is_integer() else days
    return None


def _spoken_number(raw: str) -> int | float:
    number = float(raw)
    return int(number) if number.is_integer() else number


def _followed_by_age_copula(text: str, end: int) -> bool:
    """दो हफ़्ते का है is an age. दो दिन रहा and दो दिन का बुखार are not."""
    rest = re.sub(r"^\S*", "", text[end:])
    return re.match(
        r"\s*(?:का|की|के)(?:\s*(?:है|हैं|था|थी|थे|बच्च|बेटा|बेटी|लड़क)|[।.!\s]*$)",
        rest,
    ) is not None


def _is_stay_after(text: str, end: int) -> bool:
    rest = re.sub(r"^\S*", "", text[end:])
    return re.match(r"\s*(?:रहा|रही|रहे|तक|से|के\s+लिए)", rest) is not None


def _span_is_age(text: str, end: int, require_copula: bool) -> bool:
    """A week or day count is an age with का/की है, or in a list that ends that way.

    दो हफ़्ते और एक हफ़्ता का है is two ages. दो दिन रहा is a stay.
    """
    if _is_duration_after(text, end) or (require_copula and _is_stay_after(text, end)):
        return False
    if not require_copula or _followed_by_age_copula(text, end):
        return True
    rest = re.sub(r"^\S*", "", text[end:])
    if re.match(r"\s*(?:,|और|and)\b", rest, re.I) and re.search(
        r"(?:का|की|के)(?:\s*(?:है|हैं|था|थी|थे|बच्च|बेटा|बेटी|लड़क)|[।.!\s]*$)",
        text[end:],
    ):
        return True
    # दो हफ्ते तीन दिन का है. The weeks are part of that age, not a stay.
    if require_copula and re.match(r"\s*\d+(?:\.\d+)?\s*(?:दिन|days|day)\b", text[end:], re.I):
        if re.search(
            r"(?:का|की|के)(?:\s*(?:है|हैं|था|थी|थे|बच्च|बेटा|बेटी|लड़क)|[।.!\s]*$)",
            text[end:],
        ):
            return True
    return False


def _is_event_month_after(text: str, end: int) -> bool:
    """तीन महीने में is a pregnancy month. तीन महीने का is the child's age."""
    return re.match(r"\S*(?:\s+में|\s+का\s+पैदा)", text[end:]) is not None


def _ages_with_unit(
    folded: str,
    unit_pattern: str,
    unit_name: str,
    require_copula: bool = False,
    skip_event_month: bool = False,
) -> list[tuple]:
    """Numbers that share one unit, plus a number written next to the unit.

    आठ, छह और चार साल is three ages. दो चार और छह साल is the same list:
    और sits only before the last number. एक आधा साल has no और, so it is not a list.
    """
    found = []
    covered: list[tuple[int, int]] = []
    listed = re.compile(
        rf"(\d+(?:\.\d+)?(?:(?:\s*(?:,|और|and)\s*|\s+)\d+(?:\.\d+)?)+)\s*{unit_pattern}",
        re.I,
    )
    for match in listed.finditer(folded):
        if not re.search(r",|और|and", match.group(1), re.I):
            continue
        if not _span_is_age(folded, match.end(), require_copula):
            continue
        if skip_event_month and _is_event_month_after(folded, match.end()):
            continue
        covered.append((match.start(), match.end()))
        blob_at = match.start(1)
        for num in re.finditer(r"\d+(?:\.\d+)?", match.group(1)):
            found.append((
                blob_at + num.start(),
                {"value": _spoken_number(num.group(0)), "unit": unit_name},
            ))
    single = re.compile(rf"(\d+(?:\.\d+)?)\s*{unit_pattern}", re.I)
    for match in single.finditer(folded):
        if any(start <= match.start() < end for start, end in covered):
            continue
        if not _span_is_age(folded, match.end(), require_copula):
            continue
        if skip_event_month and _is_event_month_after(folded, match.end()):
            continue
        found.append((match.start(), {"value": _spoken_number(match.group(1)), "unit": unit_name}))
    return found


def _spoken_age_records(utterance: str) -> list[dict]:
    """Ages she actually said, keeping months and years apart.

    छह महीने is 6 months. आधा साल is 0.5 years. Both are half a year for
    comparisons, and they must not be stored as the same bare 0.5.
    """
    folded = _fold_spoken_numbers(utterance or "")
    found = []
    found.extend(_ages_with_unit(folded, _MONTH_UNIT, "months", skip_event_month=True))
    found.extend(_ages_with_unit(folded, _YEAR_UNIT, "years"))
    found.extend(_ages_with_unit(folded, _WEEK_UNIT, "weeks", require_copula=True))
    found.extend(_ages_with_unit(folded, _DAY_UNIT, "days", require_copula=True))
    found.sort()
    return _merge_year_and_months(folded, _merge_weeks_and_days(folded, found))


def _merge_weeks_and_days(folded: str, found: list) -> list:
    """दो हफ्ते तीन दिन का is 17 days. दो हफ्ते और एक हफ्ता stays two ages."""
    merged = []
    index = 0
    while index < len(found):
        start, record = found[index]
        nxt = found[index + 1] if index + 1 < len(found) else None
        nxt_unit = (nxt[1].get("unit") if nxt else None)
        smaller = (
            nxt_unit == "days"
            or (nxt_unit == "weeks" and record.get("unit") in {"months", "years"})
        )
        if nxt and record.get("unit") in {"weeks", "months", "years"} and smaller:
            between = folded[start:nxt[0]]
            unit_re = {"weeks": _WEEK_UNIT, "months": _MONTH_UNIT, "years": _YEAR_UNIT}[record["unit"]]
            after_larger = re.sub(
                rf"^\d+(?:\.\d+)?\s*{unit_re}",
                "",
                between,
                count=1,
                flags=re.I,
            )
            # महीने leaves the े. That matra is still the unit, not a new word.
            after_larger = re.sub(r"^[\u093e-\u094d]+", "", after_larger or "")
            extra = nxt[1].get("value")
            if (
                re.fullmatch(r"\s*", after_larger or "")
                and isinstance(extra, (int, float))
                and not isinstance(extra, bool)
            ):
                amount = float(extra) * (7 if nxt_unit == "weeks" else 1)
                if record["unit"] == "weeks":
                    total = float(record["value"]) * 7 + amount
                    shown = int(total) if total.is_integer() else total
                    merged.append((start, {"value": shown, "unit": "days"}))
                else:
                    kept = dict(record)
                    kept["plus_days"] = int(amount) if amount.is_integer() else amount
                    merged.append((start, kept))
                index += 2
                continue
        merged.append(found[index])
        index += 1
    return merged


def _merge_year_and_months(folded: str, found: list) -> list[dict]:
    """दो साल छह महीने का is one age of ढाई साल.

    दो साल का और छह महीने का stays two ages. The का after the years is the split.
    """
    merged = []
    index = 0
    while index < len(found):
        start, record = found[index]
        nxt = found[index + 1] if index + 1 < len(found) else None
        if nxt and record.get("unit") == "years" and nxt[1].get("unit") == "months":
            between = folded[start:nxt[0]]
            after_year = re.sub(
                rf"^\d+(?:\.\d+)?\s*{_YEAR_UNIT}",
                "",
                between,
                count=1,
                flags=re.I,
            )
            months = nxt[1].get("value")
            if (
                re.fullmatch(r"\s*", after_year or "")
                and isinstance(months, (int, float))
                and not isinstance(months, bool)
                and 0 < float(months) < 12
            ):
                total = float(record["value"]) + float(months) / 12.0
                shown = int(total) if total.is_integer() else total
                merged.append({"value": shown, "unit": "years"})
                index += 2
                continue
        merged.append(record)
        index += 1
    return merged


_OTHER_PERSON_RE = re.compile(
    r"मेरी|मेरा|मेरे|मैं|पति|पत्नी|माता|पिता|मम्मी|पापा|भाई|भैया|भइया|बहन|दीदी|"
    r"दादा|दादी|चाचा|चाची|नाना|नानी|मामा|मामी|मौसी|मासी|ससुर|सास|देवर|जेठ|ननद"
)


def _child_ages_in(utterance: str) -> list[dict]:
    """Ages of her children. Someone else's उम्र, and an age at a past event, are not.

    पति की उम्र तीस साल is the husband. उम्र पाँच, तीन और एक साल after the
    children is theirs. चार साल का था जब is when something happened.
    """
    records = _spoken_age_records(utterance)
    folded = _fold_spoken_numbers(utterance or "")
    units = {
        "years": _YEAR_UNIT,
        "months": _MONTH_UNIT,
        "weeks": _WEEK_UNIT,
        "days": _DAY_UNIT,
    }
    cursor = 0
    kept = []
    for record in records:
        value = record.get("value")
        number = int(value) if isinstance(value, float) and value.is_integer() else value
        unit = units.get(record.get("unit") or "")
        match = None
        if unit is not None and number is not None:
            match = re.search(
                rf"(?<![\d.]){re.escape(str(number))}(?!\.\d)\s*{unit}",
                folded[cursor:],
            )
        if not match:
            kept.append(record)
            continue
        start = cursor + match.start()
        cursor = cursor + match.end()
        window = folded[max(0, start - 32):start]
        if "उम्र" in window:
            last_other = max((item.end() for item in _OTHER_PERSON_RE.finditer(window)), default=-1)
            last_child = max((item.end() for item in _CHILD_WORD.finditer(window)), default=-1)
            if last_other >= 0 and last_other > last_child:
                continue
        if re.match(
            rf"{re.escape(str(number))}\s*{unit}\s+(?:का|की|के)\s+(?:था|थी|थे)\s+जब",
            folded[start:],
        ):
            continue
        kept.append(record)
    return kept


def _spoken_ages(utterance: str) -> list[float]:
    return [
        age
        for age in (_child_age_years(item) for item in _spoken_age_records(utterance))
        if age is not None
    ]


def _shared_age_for_group(utterance: str, count_n: int) -> dict | None:
    """तीन साल की दो बेटियाँ is two children of that age. मेरी उम्र तीस साल is not."""
    records = _spoken_age_records(utterance)
    if count_n < 2 or len(records) != 1:
        return None
    folded = _fold_spoken_numbers(utterance or "")
    age = records[0]
    unit = {
        "years": _YEAR_UNIT,
        "months": _MONTH_UNIT,
        "weeks": _WEEK_UNIT,
        "days": _DAY_UNIT,
    }.get(age.get("unit") or "")
    if not unit:
        return None
    value = age.get("value")
    number = int(value) if isinstance(value, float) and value.is_integer() else value
    num = re.escape(str(number))
    before = re.search(
        rf"{num}\s*{unit}\s+(?:का|की|के)\s+(?:\S+\s+){{0,2}}?{count_n}\s+{_CHILD_COUNT_WORD}",
        folded,
        re.I,
    )
    after = re.search(
        rf"{count_n}\s+{_CHILD_COUNT_WORD}(.{{0,40}}?){num}\s*{unit}",
        folded,
        re.I,
    )
    if after and re.search(r"उम्र|मेरी|मैं", after.group(1)):
        after = None
    if not before and not after:
        return None
    return age


def _age_records_from_prior(count_n: int, prior: list[str] | None) -> list[dict]:
    if count_n <= 0:
        return []
    for text in reversed(list(prior or [])):
        records = _spoken_age_records(text or "")
        if not records:
            continue
        return records if len(records) == count_n else []
    return []


def _month_age_numbers(utterance: str) -> set[float]:
    """Bare numbers she used as a month age, not as a duration."""
    folded = _fold_spoken_numbers(utterance or "")
    if re.search(_YEAR_UNIT, folded, re.I):
        return set()
    found = set()
    for match in re.finditer(rf"(\d+(?:\.\d+)?)\s*{_MONTH_UNIT}", folded, re.I):
        if _is_duration_after(folded, match.end()):
            continue
        found.add(float(match.group(1)))
    return found


_CHILD_WORD = re.compile(r"बच्च|बेटा|बेटी|लड़क")


def _child_ages_already_said(prior: list[str] | None) -> list[float]:
    """Ages from the nearest earlier line that is about her child.

    A fits answer logged on the next turn often omits the child list. The line
    before it may already have said "एक बच्चा है, दो साल का". A nearer line that
    states an age and is not about a child, such as a relative's age, stops the walk.
    """
    for text in reversed(list(prior or [])):
        line = text or ""
        ages = _spoken_ages(line)
        if _CHILD_WORD.search(line) and ages:
            return ages
        if ages:
            return []
    return []


def _ages_from_prior(count_n: int, prior: list[str] | None) -> list[float]:
    """Ages from the nearest earlier line that states exactly count_n ages.

    A later "ठीक है" has no ages. The line before it often does. A nearer line
    that states a different number of ages is left alone.
    """
    if count_n <= 0:
        return []
    for text in reversed(list(prior or [])):
        ages = _spoken_ages(text or "")
        if not ages:
            continue
        return ages if len(ages) == count_n else []
    return []


def _has_year_phrase(text: str) -> bool:
    return re.search(_YEAR_UNIT, _fold_spoken_numbers(text or ""), re.I) is not None


def _retag_month_only(
    children: list[dict],
    utterance: str,
    prior: list[str] | None = None,
) -> list[dict]:
    if _has_year_phrase(utterance):
        return children
    month_ages = _month_age_numbers(utterance)
    if not month_ages:
        for text in reversed(list(prior or [])):
            if _has_year_phrase(text):
                break
            found = _month_age_numbers(text)
            if found:
                month_ages = found
                break
    if not month_ages:
        return children
    tagged = []
    for child in children:
        child = dict(child)
        age = child.get("age")
        number = None
        if isinstance(age, (int, float)) and not isinstance(age, bool):
            number = float(age)
        elif isinstance(age, dict) and not age.get("unit") and isinstance(age.get("value"), (int, float)):
            number = float(age["value"])
        if number is not None and number >= 1 and number in month_ages:
            if isinstance(age, dict):
                child["age"] = {**age, "unit": "months"}
            else:
                child["age"] = {"value": age, "unit": "months"}
        tagged.append(child)
    return tagged


def _prefer_spoken_units(
    children: list[dict],
    utterance: str,
    prior: list[str] | None = None,
) -> list[dict]:
    """A month, week, or day she said beats a year unit stamped on that number.

    छह महीने, दो हफ़्ते, or पंद्रह दिन stored as years uses the unit she said.
    दो साल stays years. A number she used for two units is left as sent.
    """
    records = _spoken_age_records(utterance)
    if not records:
        for text in reversed(list(prior or [])):
            records = _spoken_age_records(text or "")
            if records:
                break
    if not records:
        return children
    by_unit: dict[str, set] = {}
    for item in records:
        by_unit.setdefault(item.get("unit") or "", set()).add(item.get("value"))
    out = []
    for child in children:
        child = dict(child)
        age = child.get("age")
        if isinstance(age, dict):
            unit = str(age.get("unit") or "").lower()
            value = age.get("value")
            if unit in {"year", "years", "saal"}:
                for spoken in ("months", "weeks", "days"):
                    others = set().union(*(vals for name, vals in by_unit.items() if name != spoken))
                    if value in by_unit.get(spoken, ()) and value not in others:
                        child["age"] = {"value": value, "unit": spoken}
                        break
        out.append(child)
    return out


_SEX_WORDS = (
    ("बेटियाँ", "female"),
    ("बेटियां", "female"),
    ("लड़कियाँ", "female"),
    ("लड़कियां", "female"),
    ("बेटी", "female"),
    ("लड़की", "female"),
    ("बच्ची", "female"),
    ("बेटे", "male"),
    ("बेटा", "male"),
    ("लड़के", "male"),
    ("लड़का", "male"),
)
_PLURAL_SEX = {"बेटे", "बेटियाँ", "बेटियां", "लड़के", "लड़कियाँ", "लड़कियां"}


def _spoken_sex_words(utterance: str) -> list[tuple[str, str]]:
    """Sex words in the order she said them. Longer words win over prefixes."""
    text = utterance or ""
    found: list[tuple[str, str]] = []
    index = 0
    while index < len(text):
        match = next((item for item in _SEX_WORDS if text.startswith(item[0], index)), None)
        if match is None:
            index += 1
            continue
        found.append(match)
        index += len(match[0])
    return found


def _spoken_sexes(utterance: str) -> list[str]:
    return [sex for _word, sex in _spoken_sex_words(utterance)]


def _sexes_for_count(text: str, count: int) -> list[str] | None:
    """One plural (बेटे, बेटियाँ) covers every child. One singular does not."""
    words = _spoken_sex_words(text)
    sexes = [sex for _word, sex in words]
    if len(sexes) == count:
        return sexes
    if count > 1 and len(words) == 1 and words[0][0] in _PLURAL_SEX:
        return sexes * count
    return None


def _same_ages(existing: list, spoken: list) -> bool:
    if len(existing) != len(spoken):
        return False
    return [
        _child_age_years(child.get("age")) for child in existing
    ] == [
        _child_age_years(age) for age in spoken
    ]


def _children_from_ages(ages: list, sexes: list[str] | None = None) -> list[dict]:
    sexes = sexes or []
    children = []
    for index, age in enumerate(ages, start=1):
        sex = sexes[index - 1] if len(sexes) == len(ages) else None
        children.append({"id": f"child_{index}", "name": None, "age": age, "sex": sex})
    return children


def _parse_field(field: object) -> tuple[str, str]:
    text = str(field or "").strip()
    if text.startswith("profile.") or text.startswith("profiles."):
        name = text.split(".", 1)[1].strip().replace("-", "_")
        if name not in PROFILE_FIELDS:
            return "error", f"unknown field: {text}"
        return "profile", name
    if text.startswith("topic.") or text.startswith("topics."):
        topic_id = text.split(".", 1)[1].strip().replace("-", "_")
        if topic_id not in contract.TOPIC_IDS:
            topic_id = topic_id.replace(".", "_")
        if topic_id not in contract.TOPIC_IDS:
            return "error", f"unknown field: {text}"
        return "topic", topic_id
    if text in PROFILE_FIELDS:
        return "profile", text
    if text in contract.TOPIC_IDS:
        return "topic", text
    collapsed = text.replace(".", "_").replace("-", "_")
    if collapsed in contract.TOPIC_IDS:
        return "topic", collapsed
    if collapsed in PROFILE_FIELDS:
        return "profile", collapsed
    return "error", f"unknown field: {text}"


def _check_update(update: dict, current_topic: str | None) -> tuple[str, str]:
    if not isinstance(update, dict):
        return "error", "update must be an object"
    kind, name = _parse_field(update.get("field"))
    if kind == "error":
        return kind, name
    status = update.get("state")
    if status not in MODEL_TOPIC_STATES:
        return "error", f"invalid state: {status}"
    scope = update.get("scope") or "none"
    if scope not in _SCOPES:
        return "error", f"invalid scope: {scope}"
    child_ids = update.get("child_ids") or []
    if not isinstance(child_ids, list):
        return "error", "child_ids must be a list"
    return kind, name


# A vowel sign is not a word character, so Hindi must not use \w after बच्च.
# नहीं denies the child only when the sentence stops or continues with a copula.
# "मेरा बच्चा नहीं रोता" is about the child, not a count of zero.
_NO_CHILD_RE = re.compile(
    r"(?:कोई\s+)?बच्च\S*(?:\s+भी)?\s+(?:नहीं|नही)(?=\s*(?:है|हैं|था|थे|थी|हूँ|हूं)?\s*(?:[।.,]|$))"
    r"|bachch\w*\s+(?:नहीं|नही|nahi|nahin)"
    r"|no\s+child(?:ren)?|not\s+any\s+child(?:ren)?",
    re.I,
)
_CHILD_COUNT_WORD = r"(?:बच्च\S*|बेटे|बेटि\S*|बेटा|बेटी|लड़क\S*)"
_AGE_UNIT_AFTER = rf"(?:{_YEAR_UNIT}|{_MONTH_UNIT}|{_WEEK_UNIT}|{_DAY_UNIT})"


def _is_no_children_phrase(text: str) -> bool:
    """A denial of children. An age in the same sentence is about a child she has."""
    raw = (text or "").strip()
    if not raw or _spoken_age_records(raw):
        return False
    return _NO_CHILD_RE.search(raw) is not None


def _numbers_before_child_words(folded: str) -> list[tuple[int, str]]:
    """Each number that sits against a child word, with that word.

    एक लड़का और दो लड़कियाँ is (1, लड़का) and (2, लड़कियाँ).
    आठ साल का एक बच्चा is only (1, बच्चा): आठ sits against साल.
    """
    found = []
    # 1.5 बच्चा must not be read as five. The digit after the dot is not a count.
    for match in re.finditer(rf"(?<![\d.])(\d+)\s*({_CHILD_COUNT_WORD})", folded or "", re.I):
        found.append((int(match.group(1)), match.group(2)))
    if found:
        return found
    for word, number in (
        ("दोनों", 2), ("तीनों", 3), ("चारों", 4),
        ("पाँचों", 5), ("पांचों", 5), ("छहों", 6),
    ):
        if re.search(rf"{word}\s*(?:{_CHILD_COUNT_WORD}|को)", folded or ""):
            found.append((number, word))
    return found


def _count_from_child_words(folded: str) -> int | None:
    """The count is every number beside a child word, added together.

    A number after the word is a count only when an age unit does not follow.
    लड़का छह महीने is an age, not six children.
    """
    groups = _numbers_before_child_words(folded)
    if groups:
        return sum(number for number, _word in groups)
    after = re.search(
        rf"{_CHILD_COUNT_WORD}\s+(?<![\d.])(\d+)(?!\.\d)(?!\s*{_AGE_UNIT_AFTER})",
        folded or "",
        re.I,
    )
    if after:
        return int(after.group(1))
    if not re.search(_CHILD_COUNT_WORD, folded or "", re.I):
        # चार साल का है is one age, not four children. A bare number is a count
        # only when she did not also say an age. 1.5 is not one child.
        if _spoken_age_records(folded or ""):
            return None
        bare = re.search(r"(?<![\d.])(\d+)(?!\.\d)", folded or "")
        if bare:
            return int(bare.group(1))
    return None


def _sexes_from_counted_words(text: str, count: int) -> list[str] | None:
    """दो बेटे is two boys. एक लड़का और दो लड़कियाँ is one boy and two girls."""
    folded = _fold_spoken_numbers((text or "").lower())
    slots: list[str] = []
    for number, word in _numbers_before_child_words(folded):
        sexes = _spoken_sexes(word)
        if len(sexes) != 1:
            continue
        slots.extend(sexes * number)
    if len(slots) == count:
        return slots
    return None


def _sexes_for_children(text: str, count: int) -> list[str] | None:
    return _sexes_from_counted_words(text, count) or _sexes_for_count(text, count)


def _children_payload(value: object) -> tuple[str, object]:
    """none: she said there are no children. count: a number without ages. list: child records."""
    if isinstance(value, bool) or value in (0, None, []) or value == "0":
        return "none", []
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        # 0.5 is half a year, not zero children. A whole number is a count
        # until the utterance shows it is the age.
        if isinstance(value, float) and not float(value).is_integer():
            return "ignore", []
        return ("none", []) if int(value) <= 0 else ("count", int(value))
    if isinstance(value, str):
        val_clean = str(value).strip().lower()
        if val_clean in {"none", "no", "nahi", "nahin", "zero", "0", "false", "कोई नहीं", "एक भी नहीं"} or _is_no_children_phrase(value):
            return "none", []
        val_folded = _fold_spoken_numbers(val_clean)
        # The numbers beside child words are the count. आठ साल का एक बच्चा is one.
        # एक लड़का और दो लड़कियाँ is three. लड़का छह महीने is not six.
        number = _count_from_child_words(val_folded)
        if number is not None:
            return ("none", []) if number <= 0 else ("count", number)
    if isinstance(value, (list, dict)):
        return "list", _normalize_children(value)
    return "ignore", []


def _spoken_given_names(text: str) -> list[str]:
    """Names she introduced with नाम. मेरा नाम is the caller, handled separately.

    नाम आरव है, छोटे वाली ईशा है is two names. The word नाम is not repeated.
    """
    raw = text or ""
    if not re.search(r"(?<!मेरा\s)नाम", raw):
        return []
    blocked = set(_QUESTION_FUNCTION_WORDS)
    blocked.update(word for word, _number in _NUM_WORDS)
    blocked.update({
        "साल", "वर्ष", "महीने", "महीना", "हफ्ते", "हफ़्ते", "दिन", "नाम",
        "बच्चा", "बच्चे", "बच्चों", "बेटा", "बेटे", "बेटी", "बेटियाँ", "बेटियां",
        "लड़का", "लड़के", "लड़की", "लड़कियाँ", "लड़कियां",
        "वाले", "वाली", "वाला", "बड़े", "बड़ा", "बड़ी", "छोटे", "छोटा", "छोटी",
    })
    found = []

    def take(token: str) -> None:
        name = spoken_person_name(token)
        if name and name not in blocked and name not in found:
            found.append(name)

    for match in re.finditer(r"(?<!मेरा\s)नाम\s+([^\s,।]+)", raw):
        take(match.group(1))
    for match in re.finditer(r"(?:वाले|वाली|वाला|और)\s+([^\s,।]+)", raw):
        take(match.group(1))
    return found


def _attach_spoken_names(children: list, utterance: str, prior: list[str] | None = None) -> list:
    """Fill an empty child name when she said one नाम per child."""
    source = utterance if _spoken_given_names(utterance) else ""
    if not source:
        for text in reversed(list(prior or [])):
            if _spoken_given_names(text or ""):
                source = text
                break
    names = _spoken_given_names(source)
    if len(names) != len(children):
        return children
    out = []
    for child, name in zip(children, names):
        child = dict(child) if isinstance(child, dict) else {"id": None, "name": None, "age": None, "sex": None}
        if not str(child.get("name") or "").strip():
            child["name"] = name
        out.append(child)
    return out


def spoken_person_name(value: object) -> str:
    """The name inside मेरा नाम मीरा है. A bare name stays as it is."""
    text = str(value or "").strip().strip("।.! ")
    text = re.sub(r"(?i)^(?:my name is|i am|i'm|this is)\s+", "", text).strip()
    text = re.sub(r"^(?:मेरा\s+)?नाम\s+", "", text).strip()
    text = re.sub(r"\s+(?:है|हूँ|हूं)$", "", text).strip("।.! ")
    return text


def _canonical_marital(value: object) -> str | None:
    raw = str(value or "").strip()
    text = raw.lower().replace("_", " ").replace("-", " ")
    denial = (
        text in {"false", "no", "nahi", "nahin", "unmarried", "single", "not married"}
        or "not married" in text
        or re.search(r"\b(?:nahi|nahin|no)\b", text) is not None
        or any(word in raw for word in ("नहीं", "नही", "अविवाहित", "कुंवारी"))
    )
    about = (
        "about" in text
        or "hone wali" in text
        or "engaged" in text
        or "होने वाली" in raw
    )
    # हुई / ho gayi only complete a marriage. जाँच हुई and बात हुई do not.
    marriage_noun = any(
        _contains_word(raw, word) for word in ("शादी", "विवाह", "ब्याह")
    ) or any(_contains_word(text, word) for word in ("shaadi", "shadi", "vivah", "byah"))
    status_word = (
        text in {"true", "yes", "haan", "married"}
        or any(_contains_word(raw, word) for word in ("विवाहित", "शादीशुदा", "ब्याही"))
        or any(_contains_word(text, word) for word in ("shadishuda", "shaadi shuda"))
    )
    married = status_word or marriage_noun
    # A denial beats a marriage word: शादी नहीं हुई, शादीशुदा नहीं.
    if about and not denial:
        return "about_to_marry"
    if married and not denial:
        return "married"
    if denial:
        return "unmarried"
    return None


def _children_gate_open(state: dict) -> bool:
    if state["topics"]["intro_children"]["state"] == "answered":
        return True
    return any(isinstance(child, dict) for child in (state["profile"].get("children") or []))


def _topic_can_record(state: dict, topic_id: str, current: str | None) -> bool:
    topic = state["topics"].get(topic_id) or {}
    if topic_id == current or str(topic_id).startswith("intro_"):
        return True
    if not _children_gate_open(state):
        return False
    return True


def _remember_turn(topic: dict, turn_id: int) -> None:
    turns = topic.setdefault("source_turn_ids", [])
    if turn_id not in turns:
        turns.append(turn_id)


def _store_instances(topic: dict, update: dict) -> None:
    scope = update.get("scope") or "none"
    value = update.get("value")
    details = update.get("details") if isinstance(update.get("details"), dict) else {}
    child_ids = [str(child_id) for child_id in (update.get("child_ids") or [])]
    if scope == "all" and value is False:
        topic["instances"] = []
        return
    if scope != "specific":
        return
    existing = {
        item.get("person_id"): item
        for item in topic.get("instances") or []
        if isinstance(item, dict) and item.get("person_id")
    }
    for child_id in child_ids:
        instance = existing.get(child_id) or {"person_id": child_id, "value": value, "details": {}}
        instance["value"] = value
        if details:
            instance_details = instance.get("details") if isinstance(instance.get("details"), dict) else {}
            instance_details.update(details)
            instance["details"] = instance_details
        existing[child_id] = instance
    topic["instances"] = list(existing.values())


_PARENT_ECHO = {
    "operation", "normal", "caesarean", "cesarean", "c-section", "c section",
    "true", "false", "yes", "no", "haan", "nahi", "nahin", "हाँ", "नहीं", "नही",
}
_CLARIFY_RE = re.compile(
    r"^\s*(?:(?:\S+\s+){0,2})?(?:"
    r"क्या(?:\s+मतलब(?:\s+है)?|\s+होता\s+है)?"
    r"|मतलब(?:\s+क्या\s+है)?"
    r"|यानी|matlab|yaani|yani"
    r"|समझ(?:\s+में)?\s+नहीं\s+आया"
    r"|samajh(?:\s+mein)?\s+nahi(?:\s+aaya)?"
    r"|kya(?:\s+matlab(?:\s+hai)?|\s+hota\s+hai)?"
    r"|(?:फिर\s+से|दोबारा|phir\s+se|dobara)\s+(?:बोलिए|बोलो|boliye|bolo)"
    r")[।.!?]*\s*$",
    re.I,
)
_UNKNOWN_UTT_RE = re.compile(
    r"^(?:(?:\S+\s+){0,2})(?:"
    r"(?:पता|याद|मालूम)\s+(?:नहीं|नही)|(?:नहीं|नही)\s+(?:पता|मालूम)|"
    r"(?:pata|yaad|malum|maaloom)\s+nahi|don'?t\s+know"
    r")(?:\s+(?:है|था|थी|थे|जी|hai|tha|thi|the|ji))?[।.!?]*$",
    re.I,
)


def _is_unknown_utterance(text: str) -> bool:
    """She said she does not know. A number in the same line is a fact, not a blank."""
    raw = (text or "").strip()
    if not raw or re.search(r"\d", _fold_spoken_numbers(raw.lower())):
        return False
    return _UNKNOWN_UTT_RE.match(raw) is not None


def _is_parent_echo(value: object, topic_value: object) -> bool:
    """True when a string only repeats the parent answer. A boolean is a real yes or no."""
    if isinstance(value, (bool, dict)) or value is None:
        return False
    text = str(value).strip().lower()
    if text in _PARENT_ECHO:
        return True
    return topic_value is not None and text == str(topic_value).strip().lower()


def is_clarification(text: str) -> bool:
    return _CLARIFY_RE.match((text or "").strip()) is not None


_is_clarification = is_clarification


_EVERY_CHILD_WORDS = (
    "all", "both", "dono", "sabhi", "sabko", "everyone", "sab",
    "दोनों", "तीनों", "चारों", "पाँचों", "पांचों", "छहों",
    "सभी", "सबको", "सब", "हर एक",
)


def _means_every_child(text: str) -> bool:
    """सबको means every child. सबसे means the most, so it is not सब."""
    return any(_contains_word(text or "", word) for word in _EVERY_CHILD_WORDS)


def _strip_nukta(text: str) -> str:
    """सीज़ेरियन and सीजेरियन are the same word. The nukta is optional."""
    return (text or "").replace("़", "")


def _contains_word(text: str, word: str) -> bool:
    """A whole word. abnormal is not normal, and घरेलू is not घर.

    घरपे is still घर. A nukta does not make a different word.
    """
    if not text or not word:
        return False
    text = _strip_nukta(text)
    word = _strip_nukta(word)
    if re.search(r"[A-Za-z]", word):
        return re.search(rf"(?i)(?<![A-Za-z]){re.escape(word)}(?![A-Za-z])", text) is not None
    # घरपे is घर. घरेलू is not.
    return re.search(
        rf"(?<![\u0900-\u097F]){re.escape(word)}(?=(?:पे|पर|में)|[^\u0900-\u097F]|$)",
        text,
    ) is not None


def _word_denied(text: str, word: str) -> bool:
    """नॉर्मल नहीं and not normal deny that word. A later नहीं in the sentence does not."""
    denial = r"(?:नहीं|नही|\bno\b|\bnahi\b|\bnahin\b|\bnot\b)"
    bridge = r"\s+"
    token = re.escape(word)
    return (
        re.search(rf"{denial}{bridge}{token}", text, re.I) is not None
        or re.search(rf"{token}{bridge}{denial}", text, re.I) is not None
    )


def _lab_result(text: str) -> str | None:
    """normal or abnormal from this utterance. normal inside abnormal does not count."""
    if not text:
        return None
    abnormal = re.search(r"अबनॉर्मल|असामान्य|abnormal|पॉजिटिव|positive|carrier", text, re.I)
    problem = re.search(r"दिक्कत", text) and not re.search(r"(?:कोई\s+)?दिक्कत\s*नहीं", text)
    if abnormal or problem:
        return "abnormal"
    if re.search(r"(?<!ab)(?:नॉर्मल|normal)", text, re.I):
        return "normal"
    return None


def _term_label(text: str) -> str | None:
    """early, late, or full. A child's present age (6 महीने का है) is not a term."""
    if not text:
        return None
    folded = _fold_spoken_numbers(text)
    born = r"(?:\s+में|\s+का\s+पैदा)"
    early = re.search(
        rf"समय से पहले|जल्दी|preterm|early|(?:6|7|8)\s+महीन\S*{born}",
        folded,
    )
    late = re.search(
        rf"समय के बाद|देर से|postterm|late|(?:10|11|12)\s+महीन\S*{born}|ten month|eleven month",
        folded,
    )
    full = re.search(
        rf"पूरे\s*9|full term|समय पर|9\s+महीन\S*{born}|nine month",
        folded,
    )
    # साढ़े सात महीने में is 7.5, which the integer list misses. Under 9 is early.
    gestation = re.search(rf"(\d+(?:\.\d+)?)\s*महीन\S*{born}", folded)
    if gestation:
        month = float(gestation.group(1))
        early = early or month < 9
        late = late or month > 9
        full = full or month == 9
    if early and not late:
        return "early"
    if late and not early:
        return "late"
    if full and not early and not late:
        return "full"
    return None


def _named_and_not_denied(value_text: str, utterance: str, words: tuple[str, ...]) -> bool:
    """The new answer names this kind, and the utterance does not take it back."""
    named = any(_contains_word(value_text, word) or _contains_word(utterance, word) for word in words)
    if not named:
        return False
    return not any(
        _word_denied(utterance, word)
        for word in words
        if _contains_word(utterance, word) or _contains_word(value_text, word)
    )


def _absorb_stated_value(name: str, topic: dict, utterance: str, on_followup: bool = False) -> None:
    details = dict(topic.get("details") or {})
    value = topic.get("value")
    val_blob = str(value or "").lower()
    utt_blob = (utterance or "").lower()
    combined = f"{val_blob} {utt_blob}"
    if name == "birth_mode":
        op_words = _OPERATION_WORDS
        norm_words = ("normal", "नॉर्मल", "सामान्य")
        said_op = _named_and_not_denied(val_blob, utt_blob, op_words)
        said_norm = _named_and_not_denied(val_blob, utt_blob, norm_words)
        if said_op and not said_norm:
            details["delivery_type"] = "operation"
        elif said_norm and not said_op:
            details["delivery_type"] = "normal"
        hosp_words = ("hospital", "clinic", "अस्पताल", "हॉस्पिटल", "नर्सिंग")
        home_words = ("home", "ghar", "घर")
        said_hosp = _named_and_not_denied(val_blob, utt_blob, hosp_words)
        said_home = _named_and_not_denied(val_blob, utt_blob, home_words)
        if said_hosp and not said_home:
            details["place"] = "hospital"
        elif said_home and not said_hosp:
            details["place"] = "home"
        if details.get("delivery_type") == "operation" and "operation_reason" not in details:
            if re.search(r"वजह|because|reason", utt_blob):
                details["operation_reason"] = (utterance or "").strip()
    if name == "birth_term":
        named = _term_label(utt_blob) or _term_label("" if isinstance(value, bool) else val_blob)
        if named == "early":
            details["term"] = "early"
            details.pop("result", None)
            topic["value"] = True
        elif named == "late":
            details["term"] = "late"
            details.pop("result", None)
            topic["value"] = True
        elif named == "full":
            details["term"] = "full"
            details["result"] = "normal"
    if name == "birth_nicu":
        reasons = [
            label
            for label, words in (
                ("jaundice", ("पीलिया", "jaundice", "नीली रोशनी")),
                ("fits", ("झटके", "fits", "seizure", "दौरे")),
                ("infection", ("infection", "संक्रमण", "इंफेक्शन")),
                ("breathing", ("साँस", "breathing")),
            )
            if _named_and_not_denied(val_blob, utt_blob, words)
        ]
        if len(reasons) == 1:
            details["reason"] = reasons[0]
    if name == "birth_hearing_vision":
        named = _lab_result(utterance or "")
        if named:
            details["result"] = named
            if named == "normal":
                details.setdefault("test_done", True)
        if "test_done" not in details and isinstance(value, bool):
            details["test_done"] = value
    if name == "preg_scan":
        named = _lab_result(utterance or "")
        if named:
            details["result"] = named
            if named == "normal":
                details.setdefault("scan_done", True)
        if "scan_done" not in details and isinstance(value, bool):
            details["scan_done"] = value
    if name == "preg_thalassemia":
        named = _lab_result(utterance or "")
        if named:
            details["result"] = named
            if named == "normal":
                details.setdefault("caller_test", True)
    if name == "preg_delivery" and "problem" not in details:
        said = utterance or ""
        if (
            said.strip()
            and _stated_polarity(said) != "no"
            and not _delivery_was_only_the_event(said)
            and not _names_reassuring_normal(said)
        ):
            details["problem"] = said.strip()
    if name in {"preg_delivery", "preg_transfusion"}:
        key = "pregnancy" if name == "preg_delivery" else "pregnancy_number"
        if key not in details:
            index = _spoken_pregnancy_index(utterance or "")
            if index:
                details[key] = index
    keep_positive = False
    if name == "body_growth":
        text = utterance or ""
        not_growing = re.search(r"नहीं\s+बढ़|जैसा\s+नहीं|ठीक\s+से\s+नहीं|not growing", text, re.I)
        denied = re.search(r"नहीं|नही|nahi|nahin|not growing", text, re.I)
        fine = re.search(r"जैसा|ठीक|नॉर्मल|normal", text, re.I)
        if not_growing:
            details.pop("result", None)
            topic["value"] = True
            keep_positive = True
        elif denied:
            details.pop("result", None)
        elif fine and _affirmative_value(value):
            details["result"] = "normal"
    if name == "body_puberty":
        text = utterance or ""
        if re.search(r"जल्दी|देर से पहले|समय से पहले|early", text, re.I):
            details["timing"] = "early"
        elif re.search(r"देर|बाद में|late", text, re.I) and "साथ" not in text:
            details["timing"] = "late"
        elif re.search(r"साथ|बाकी बच्चों|नॉर्मल|normal|same", text, re.I):
            details["timing"] = "same"
        if str(details.get("timing") or "") in {"same", "normal", "नॉर्मल"}:
            details["result"] = "normal"
        elif str(details.get("timing") or "") in {"early", "late"}:
            details.pop("result", None)
    if name == "preg_conditions":
        found = [
            label
            for label, words in (
                ("bp", ("बीपी", "bp", "रक्तचाप", "blood pressure")),
                ("sugar", ("शुगर", "sugar", "मधुमेह")),
                ("thyroid", ("थायरॉइड", "thyroid")),
            )
            if _named_and_not_denied(val_blob, utt_blob, words)
        ]
        if found:
            details["conditions"] = found
        if re.search(r"बाद नहीं|अब नहीं|नहीं रहा", utt_blob):
            details["continued_after_delivery"] = False
        elif re.search(r"बाद भी|अब भी|continued|still", utt_blob):
            details["continued_after_delivery"] = True
    follow_ids = {spec["id"] for spec in TOPIC_RULES.get(name, {}).get("followups") or []}
    if "count" in follow_ids:
        _store_spoken_number(details, "count", _spoken_times, utterance)
    if "days" in follow_ids:
        _store_spoken_number(details, "days", _spoken_day_count, utterance)
    if "month" in follow_ids:
        _store_spoken_number(details, "month", _spoken_month_count, utterance)
    if "amount" in follow_ids and not details.get("amount"):
        month = _spoken_month_count(utterance or "")
        if month is not None:
            details["amount"] = _month_phrase(month)
    ages = _spoken_age_records(utterance or "")
    if len(ages) == 1:
        phrase = _age_phrase(ages[0])
        for fid in ("onset_age", "age_at_death"):
            if fid in follow_ids and not details.get(fid) and phrase:
                details[fid] = phrase
    if name == "fam_fits_delay" and "condition" not in details:
        named = [
            label
            for label, words in (
                ("fits", ("झटके", "fits", "seizure", "दौरे")),
                ("delay", ("देर", "delay")),
                ("lost_skill", ("भूल", "नहीं कर")),
            )
            if _named_and_not_denied(val_blob, utt_blob, words)
        ]
        if named:
            details["condition"] = named[0] if len(named) == 1 else named
    if "relationship" in follow_ids and "relationship" not in details and _has_kinship(utterance or ""):
        details["relationship"] = (utterance or "").strip()
    if "who" in follow_ids and "who" not in details and _has_kinship(utterance or ""):
        details["who"] = (utterance or "").strip()
    # The words describe the problem. A tool value of false does not hide them.
    # A denial in the utterance still wins, above.
    if topic.get("value") is False and _stated_polarity(utterance) != "no":
        if _described_problem(name, utterance, details):
            topic["value"] = True
    elif topic.get("value") is False and _restates_negative_question(name, utterance):
        topic["value"] = True
    question = checklist.question_hi(name)
    # ठीक है। by itself is an acknowledgement. It does not erase a finding the
    # tool just logged from the previous sentence. A sentence that says the
    # thing healed or is fine does.
    acknowledgement = re.fullmatch(
        r"\s*(?:जी\s+)?(?:ठीक\s*है|theek\s*hai|thik\s*hai)[।.!\s]*",
        utterance or "",
        re.I,
    )
    reassured = (
        not acknowledgement
        and _only_reassurance(utterance or "")
        and not ("ठीक" in question and "ठीक" in (utterance or ""))
    )
    if reassured:
        topic["value"] = False
    # A tool "yes" does not outvote a sentence that says no. ठीक है। by itself
    # is only an acknowledgement. A follow-up denial is about that detail.
    # A sentence that still names the problem (झटके थे, शुगर था) stays a yes.
    if (
        not on_followup
        and not keep_positive
        and not acknowledgement
        and topic.get("value") is True
        and _stated_polarity(utterance) == "no"
        and not _restates_negative_question(name, utterance)
        and not _described_problem(name, utterance, details)
    ):
        topic["value"] = False
    if name == "birth_cord_3weeks" and _cord_within_three_weeks(utterance or ""):
        topic["value"] = False
    topic["details"] = details


def apply_updates(
    state: dict,
    updates: list[dict],
    turn_id: int,
    utterance: str = "",
    prior_utterances: list[str] | None = None,
) -> dict:
    state = copy.deepcopy(state)
    if not isinstance(updates, list):
        raise ValueError("updates must be a list")
    current = (state.get("current_question") or {}).get("topic_id")
    followup_id = (state.get("current_question") or {}).get("followup_id")
    parsed = [_check_update(update, current) for update in updates]
    pairs = sorted(zip(updates, parsed), key=lambda pair: 0 if pair[1][0] == "profile" else 1)
    has_topic = any(kind == "topic" for kind, _name in parsed)
    has_children_update = any(kind == "profile" and name == "children" for kind, name in parsed)
    existing_children = [
        child for child in (state["profile"].get("children") or []) if isinstance(child, dict)
    ]
    named_children = False
    if has_topic and not has_children_update and not existing_children:
        spoken_children = _child_ages_already_said(prior_utterances)
        if spoken_children:
            state["profile"]["children"] = _children_from_ages(spoken_children)
            named_children = True
            intro_c = state["topics"]["intro_children"]
            intro_c["state"] = "answered"
            intro_c["value"] = True
            intro_c["source"] = "model"
            _remember_turn(intro_c, turn_id)
    wrote = False
    problems: list[str] = []
    for update, (kind, name) in pairs:
        if kind == "error":
            problems.append(name)
            continue
        status = update.get("state")
        if kind == "profile":
            previous = state["profile"].get(name)
            if name == "children" and status == "answered":
                kind_c, norm = _children_payload(update.get("value"))
                if kind_c in ("count", "ignore"):
                    existing = [
                        child for child in (state["profile"].get("children") or [])
                        if isinstance(child, dict)
                    ]
                    aged = [child for child in existing if _child_age_years(child.get("age")) is not None]
                    spoken_now = _child_ages_in(utterance)
                    # The tool's integer is an age when the sentence states ages
                    # and does not state that many children. आठ साल का एक बच्चा
                    # sent as 8 is one child. छह महीने sent as 6 is one child.
                    if kind_c == "count" and spoken_now:
                        recounted = _count_from_child_words(_fold_spoken_numbers(utterance or ""))
                        age_values = {
                            int(record["value"]) if isinstance(record.get("value"), float) and record["value"].is_integer()
                            else record.get("value")
                            for record in spoken_now
                        }
                        # No child-count phrase: the integer is the age (छह महीने sent as 6).
                        # A different count whose integer is one of those ages: आठ साल का एक
                        # बच्चा sent as 8, or the mother's 30 beside दो बच्चे.
                        if recounted is None:
                            kind_c = "ignore"
                        elif recounted != int(norm) and int(norm) in age_values:
                            norm = recounted
                    if (
                        kind_c == "ignore"
                        and len(spoken_now) == 1
                        and re.search(r"दोनों|जुड़वाँ|जुड़वां|जुड़वाँ|twins", utterance or "", re.I)
                        and _count_from_child_words(_fold_spoken_numbers(utterance or "")) is None
                    ):
                        # दोनों एक साल के, or जुड़वाँ, is two children of that age.
                        # बेटा और बेटी दोनों चार साल arrives as a count and becomes
                        # this case once the sentence shows no numbered count.
                        spoken_now = [spoken_now[0], spoken_now[0]]
                    count_n = int(norm) if kind_c == "count" else 0
                    shared = _shared_age_for_group(utterance, count_n)
                    if shared:
                        spoken_now = [shared] * count_n
                    spoken = spoken_now
                    if kind_c == "count" and not spoken:
                        spoken = _age_records_from_prior(count_n, prior_utterances)
                    fresh = spoken_now if kind_c == "ignore" or len(spoken_now) == count_n else []
                    # A bare count does not replace children who already have ages.
                    # The same ages said again keep the sex and name already stored.
                    # A new sentence that states its own ages replaces that list.
                    if (
                        kind_c == "count"
                        and aged
                        and len(existing) == len(aged) == count_n
                        and not fresh
                    ):
                        wrote = True
                        continue
                    if fresh and _same_ages(existing, fresh):
                        wrote = True
                        continue
                    use = fresh or (spoken if kind_c == "ignore" or len(spoken) == count_n else [])
                    if use and (not aged or fresh):
                        source_text = utterance if _spoken_age_records(utterance) else ""
                        if not source_text:
                            for earlier in reversed(list(prior_utterances or [])):
                                if len(_spoken_age_records(earlier or "")) == len(use):
                                    source_text = earlier
                                    break
                        sexes = _sexes_for_children(source_text, len(use))
                        state["profile"]["children"] = _children_from_ages(use, sexes)
                        intro_c = state["topics"]["intro_children"]
                        intro_c["state"] = "answered"
                        intro_c["value"] = True
                        intro_c["source"] = "model"
                        _remember_turn(intro_c, turn_id)
                    elif kind_c == "count" and not aged:
                        # Ages are still missing, so the children question stays open.
                        sexes = _sexes_for_children(utterance, count_n)
                        state["profile"]["children"] = _children_from_ages([None] * count_n, sexes)
                    wrote = True
                    named_children = True
                    continue
                if kind_c == "none":
                    state["profile"]["children"] = []
                    intro_c = state["topics"]["intro_children"]
                    intro_c["state"] = "answered"
                    intro_c["value"] = False
                    intro_c["source"] = "model"
                    _remember_turn(intro_c, turn_id)
                else:
                    if norm and all(_child_age_years(child.get("age")) is None for child in norm):
                        spoken_list = _spoken_age_records(utterance) or _age_records_from_prior(
                            len(norm), prior_utterances
                        )
                        if len(spoken_list) == len(norm):
                            for child, age in zip(norm, spoken_list):
                                child["age"] = age
                    norm = _retag_month_only(norm, utterance, prior_utterances)
                    norm = _prefer_spoken_units(norm, utterance, prior_utterances)
                    state["profile"]["children"] = norm
                    named_children = True
                    if norm and all(_child_age_years(c.get("age")) is not None for c in norm):
                        intro_c = state["topics"]["intro_children"]
                        intro_c["state"] = "answered"
                        intro_c["value"] = True
                        intro_c["source"] = "model"
                        _remember_turn(intro_c, turn_id)
            elif status == "answered":
                if name == "marital_status":
                    val = _canonical_marital(update.get("value"))
                    if val is None:
                        problems.append("marital status not recognized")
                        continue
                elif name == "caller_name":
                    val = spoken_person_name(update.get("value")) or spoken_person_name(utterance)
                else:
                    val = update.get("value")
                if name in {"ever_pregnant", "has_delivered"}:
                    if _is_unknown_utterance(str(val or "")) or _is_unknown_utterance(utterance or ""):
                        val = "unknown"
                    else:
                        polarity = _stated_polarity(val)
                        if polarity == "yes":
                            val = True
                        elif polarity == "no":
                            val = False
                state["profile"][name] = val
                if name == "caller_name" and val:
                    intro_n = state["topics"]["intro_name"]
                    intro_n["state"] = "answered"
                    intro_n["value"] = val
                    intro_n["source"] = "model"
                    _remember_turn(intro_n, turn_id)
                elif name == "marital_status" and val:
                    intro_m = state["topics"]["intro_married"]
                    intro_m["state"] = "answered"
                    intro_m["value"] = val
                    intro_m["source"] = "model"
                    _remember_turn(intro_m, turn_id)
            else:
                state["profile"][name] = status
            contract.append_history(state, {
                "kind": "profile",
                "field": name,
                "answer_state": status,
                "previous_value": previous,
                "turn_id": turn_id,
            })
            wrote = True
            continue
        if name == "intro_married" and status == "answered":
            canonical = _canonical_marital(update.get("value"))
            if canonical is None:
                problems.append("marital status not recognized")
                continue
            update = dict(update)
            update["value"] = canonical
        current_topic = current if not str(current or "").startswith("profile.") else None
        if not _topic_can_record(state, name, current_topic):
            continue
        known_ids = _known_child_ids(state["profile"].get("children") or [])
        bad_child = ""
        for child_id in update.get("child_ids") or []:
            if str(child_id) not in known_ids:
                bad_child = f"unknown child id: {child_id}"
                break
        if bad_child:
            problems.append(bad_child)
            continue
        raw_uval = update.get("value")
        if isinstance(raw_uval, str):
            if raw_uval.strip().lower() == "true":
                update = dict(update)
                update["value"] = True
            elif raw_uval.strip().lower() == "false":
                update = dict(update)
                update["value"] = False
        wrote = True
        topic = state["topics"][name]
        previous_state = topic.get("state")
        previous_value = topic.get("value")
        details = dict(topic.get("details") or {})
        incoming = update.get("details") if isinstance(update.get("details"), dict) else {}
        if followup_id and name == current:
            clean_utt = (utterance or "").strip()
            if _is_clarification(clean_utt):
                incoming = {k: v for k, v in incoming.items() if k != followup_id}
                details.update(incoming)
                details.pop(followup_id, None)
            elif status == "no_response":
                details[followup_id] = {"state": "no_response"}
            elif status == "unknown":
                details[followup_id] = {"state": "unknown"}
            else:
                details.update(incoming)
                explicit = followup_id in incoming and not _is_parent_echo(
                    incoming.get(followup_id), previous_value
                )
                if not explicit:
                    raw_val = update.get("value")
                    if _is_unknown_utterance(clean_utt):
                        details[followup_id] = {"state": "unknown"}
                    elif raw_val is not None and not isinstance(raw_val, bool) and not _is_parent_echo(raw_val, previous_value):
                        details[followup_id] = raw_val
                    elif clean_utt:
                        details[followup_id] = clean_utt
                    elif _is_parent_echo(details.get(followup_id), previous_value):
                        details.pop(followup_id, None)
                elif _is_unknown_utterance(str(incoming.get(followup_id) or "")):
                    details[followup_id] = {"state": "unknown"}
            topic["details"] = details
            if topic.get("state") != "answered":
                topic["state"] = "answered" if previous_state == "answered" else status
            if (
                name == "body_bleeding_bruise"
                and followup_id in {"teeth", "periods"}
                and _followup_is_finding(details.get(followup_id))
            ):
                topic["value"] = True
            topic["source"] = "model"
            _absorb_stated_value(name, topic, utterance, on_followup=True)
            _remember_turn(topic, turn_id)
            _store_instances(topic, update)
            continue
        if status == "answered" and (
            _is_unknown_utterance(str(update.get("value") or ""))
            or _is_unknown_utterance(utterance or "")
        ):
            status = "unknown"
            update = dict(update)
            update["state"] = "unknown"
            update["value"] = None
        if previous_state in MODEL_TOPIC_STATES and (
            previous_value != update.get("value") or previous_state != status
        ):
            contract.append_history(state, {
                "kind": "correction",
                "field": f"topic.{name}",
                "previous_state": previous_state,
                "previous_value": previous_value,
                "previous_turn_ids": list(topic.get("source_turn_ids") or []),
                "turn_id": turn_id,
            })
        topic["state"] = status
        topic["value"] = update.get("value")
        details.update(incoming)
        if name == "body_bleeding_bruise" and status == "answered" and update.get("value") is False:
            details.setdefault("bleeding_or_bruises", False)
        topic["details"] = details
        if name == "intro_name" and status == "answered":
            cleaned = spoken_person_name(update.get("value")) or spoken_person_name(utterance)
            if cleaned:
                state["profile"]["caller_name"] = cleaned
                topic["value"] = cleaned
        elif name == "intro_married" and status == "answered":
            state["profile"]["marital_status"] = update.get("value")
        elif name == "intro_children" and status == "answered":
            val = update.get("value")
            if val in (0, False, None, []):
                state["profile"]["children"] = []
            elif isinstance(val, (list, dict)):
                state["profile"]["children"] = _normalize_children(val)
        topic["source"] = "model"
        topic["reason"] = None
        _absorb_stated_value(name, topic, utterance)
        _remember_turn(topic, turn_id)
        _store_instances(topic, update)
    if named_children:
        state["profile"]["children"] = _attach_spoken_names(
            state["profile"].get("children") or [],
            utterance,
            prior_utterances,
        )
    if not wrote:
        raise ValueError(problems[0] if problems else "no applicable update")
    _bind_lone_child(state)
    _apply_rules(state)
    return state


def _bind_lone_child(state: dict) -> None:
    """The only child is the affected child. Which-child is not asked, but the report still names her."""
    children = [
        child
        for child in (state.get("profile") or {}).get("children") or []
        if isinstance(child, dict) and child.get("id")
    ]
    if len(children) != 1:
        return
    child_id = children[0]["id"]
    for topic_id, topic in state["topics"].items():
        followups = TOPIC_RULES.get(topic_id, {}).get("followups") or []
        if not any(spec.get("id") == "affected_child" for spec in followups):
            continue
        instances = [item for item in (topic.get("instances") or []) if isinstance(item, dict)]
        if not _is_positive(topic, topic_id):
            topic["instances"] = [item for item in instances if item.get("person_id") != child_id]
            continue
        if any(item.get("person_id") == child_id for item in instances):
            continue
        instances.append({
            "person_id": child_id,
            "value": topic.get("value"),
            "details": dict(topic.get("details") or {}),
        })
        topic["instances"] = instances


def next_question(state: dict) -> dict | None:
    _apply_rules(state)
    if state.get("phase") == "ended":
        return None
    if state.get("phase") == "closing":
        return _closing_question(state)
    current_id = (state.get("current_question") or {}).get("topic_id")
    if current_id in state["topics"]:
        pending = _pending_followups(state, current_id)
        if pending:
            return _followup_question(current_id, pending[0], state)
    for topic in checklist.TOPICS:
        topic_id = topic["id"]
        if topic_id in _EVER_PREGNANT_NO and state["profile"].get("ever_pregnant") is None:
            if not _topic_resolved(state, topic_id):
                return _profile_question("ever_pregnant")
        if _topic_resolved(state, topic_id):
            continue
        pending = _pending_followups(state, topic_id)
        if pending:
            return _followup_question(topic_id, pending[0], state)
        if state["topics"][topic_id]["state"] == "not_asked":
            return _topic_question(topic_id)
    return None


def begin_question(state: dict, question: dict) -> dict:
    state = copy.deepcopy(state)
    question = question or {}
    state["current_question"] = {
        "topic_id": question.get("topic_id"),
        "followup_id": question.get("followup_id"),
        "phase": "asking",
    }
    if state.get("phase") not in ("closing", "ended"):
        state["phase"] = "asking"
    return state


def wait_for_answer(state: dict) -> dict:
    state = copy.deepcopy(state)
    current = state.get("current_question")
    if isinstance(current, dict) and state.get("phase") not in ("closing", "ended"):
        current["phase"] = "waiting_for_answer"
    return state


def request_close(state: dict, reason: str) -> dict:
    if reason not in INTERNAL_CLOSE_REASONS:
        raise ValueError(f"invalid close reason: {reason}")
    state = copy.deepcopy(state)
    state["close_reason"] = reason
    state["phase"] = "closing"
    current = state.get("current_question")
    if not isinstance(current, dict):
        current = {"topic_id": None, "followup_id": None, "phase": "closing"}
        state["current_question"] = current
    current["phase"] = "closing"
    return state


def mark_current_no_response(state: dict) -> dict:
    state = copy.deepcopy(state)
    current = state.get("current_question") or {}
    topic_id = current.get("topic_id")
    followup_id = current.get("followup_id")
    if isinstance(topic_id, str) and topic_id.startswith("profile."):
        field = topic_id.split(".", 1)[1]
        if field in PROFILE_FIELDS:
            previous = state["profile"].get(field)
            if field != "children":
                state["profile"][field] = "no_response"
            contract.append_history(state, {
                "kind": "profile",
                "field": field,
                "answer_state": "no_response",
                "previous_value": previous,
                "turn_id": None,
            })
        return state
    if topic_id not in state["topics"]:
        return state
    topic = state["topics"][topic_id]
    if followup_id:
        details = dict(topic.get("details") or {})
        details[followup_id] = {"state": "no_response"}
        topic["details"] = details
        return state
    topic["state"] = "no_response"
    topic["source"] = "model"
    topic["reason"] = "silence"
    return state


def is_complete(state: dict) -> bool:
    _apply_rules(state)
    return all(_topic_resolved(state, topic_id) for topic_id in checklist.REQUIRED_IDS)


def unresolved_topic_ids(state: dict) -> list[str]:
    _apply_rules(state)
    return [topic["id"] for topic in checklist.TOPICS if not _topic_resolved(state, topic["id"])]


def summary_answers(state: dict) -> dict:
    _apply_rules(state)
    answers = {}
    for topic_id, topic in state["topics"].items():
        answers[topic_id] = {
            "state": topic.get("state"),
            "value": topic.get("value"),
            "details": topic.get("details") or {},
            "instances": topic.get("instances") or [],
            "source_turn_ids": list(topic.get("source_turn_ids") or []),
            "reason": topic.get("reason"),
        }
    return answers
