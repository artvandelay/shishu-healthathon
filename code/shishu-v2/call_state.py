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
    ("ग्यारह", "11"), ("बारह", "12"), ("आधा", "0.5"),
    ("पाँच", "5"), ("पांच", "5"), ("छह", "6"), ("छः", "6"),
    ("एक", "1"), ("दो", "2"), ("तीन", "3"), ("चार", "4"),
    ("सात", "7"), ("आठ", "8"), ("नौ", "9"), ("दस", "10"),
)
_MONTH_UNIT = r"(?:महीन\w*|mahine|months|month)"
_YEAR_UNIT = r"(?:साल|वर्ष|saal|years|year|varsh)"


def _when_operation(state: dict, topic: dict) -> bool:
    details = topic.get("details") or {}
    blob = f"{topic.get('value') or ''} {details.get('delivery_type') or ''}".lower()
    return any(word in blob for word in ("operation", "caesarean", "cesarean", "c-section", "c section"))


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
            return float(value) / 12.0
        return float(value)
    if isinstance(age, str):
        cleaned = age.strip().lower()
        for word, number in _NUM_WORDS:
            cleaned = cleaned.replace(word, number)
        m = re.search(rf"(\d+(?:\.\d+)?)\s*{_MONTH_UNIT}", cleaned, re.I)
        if m:
            return float(m.group(1)) / 12.0
        m = re.search(rf"(\d+(?:\.\d+)?)\s*{_YEAR_UNIT}", cleaned, re.I)
        if m:
            return float(m.group(1))
        m = re.search(r"^\s*(\d+(?:\.\d+)?)\s*$", cleaned)
        if m:
            return float(m.group(1))
    return None


def _is_female(child: dict) -> bool:
    return str(child.get("sex") or "").strip().lower() in {"female", "girl", "f", "beti", "daughter"}


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
        if topic.get("state") == "answered" and topic.get("value") is True:
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


def _is_positive(topic: dict) -> bool:
    if topic.get("state") != "answered":
        return False
    details = topic.get("details") or {}
    if str(details.get("result") or "").strip().lower() in {"normal", "negative"}:
        return False
    value = topic.get("value")
    if value is True:
        return True
    if isinstance(value, str) and value.strip().lower() in {
        "yes", "haan", "positive", "early", "late", "operation", "abnormal",
    }:
        return True
    return False


def topic_answer_kind(topic: dict) -> str:
    if not isinstance(topic, dict):
        return "not_asked"
    if _is_positive(topic):
        return "yes"
    status = str(topic.get("state") or "not_asked")
    value = topic.get("value")
    details = topic.get("details") or {}
    if status == "answered" and (
        value is False
        or str(details.get("result") or "").strip().lower() in {"normal", "negative"}
        or str(value or "").strip().lower() in {"no", "nahi", "normal", "negative"}
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
    "माता", "पिता", "मम्मी", "पापा", "भाई", "बहन", "बेटा", "बेटी", "पति", "पत्नी",
    "दादा", "दादी", "चाचा", "चाची", "ताऊ", "ताई", "बुआ", "फूफा", "नाना", "नानी",
    "मामा", "मामी", "मौसी", "मासी", "मौसा", "ससुर", "सास", "देवर", "जेठ", "ननद",
    "कज़िन", "कजिन", "दूर", "खानदान", "सगा", "सगे", "सगी", "रिश्तेदार",
}


def _has_kinship(text: str) -> bool:
    raw = (text or "").lower()
    blob = re.sub(r"[^\w\s\u0900-\u097F]", " ", raw)
    words = set(blob.split())
    return bool(words & _KINSHIP_WORDS)


def _find_child_id(state: dict, raw: object) -> str | None:
    if not raw or not isinstance(raw, str):
        return None
    raw_cleaned = raw.strip().lower()
    cleaned = raw_cleaned
    for word, number in _NUM_WORDS:
        cleaned = cleaned.replace(word, number)
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
            if any(w in cleaned for w in ("bada", "bade", "badi", "older", "elder", "बड़ा", "बड़े", "बड़ी", "pehle", "pehla", "pehli", "पहले", "पहला", "पहली", "first")):
                return aged[0][0]
            if any(w in cleaned for w in ("chhota", "chhote", "chhoti", "chota", "chote", "younger", "छोटा", "छोटे", "छोटी", "dusra", "dusre", "dusri", "दूसरा", "दूसरे", "दूसरी", "second")):
                return aged[-1][0]
    females = [c["id"] for c in children if isinstance(c, dict) and _is_female(c) and c.get("id")]
    males = [
        c["id"] for c in children
        if isinstance(c, dict) and str(c.get("sex") or "").strip().lower() in {"male", "boy", "m", "beta", "bete", "son", "बेटा", "बेटे", "लड़का", "लड़के"} and c.get("id")
    ]
    if len(females) == 1 and any(w in cleaned for w in ("beti", "ladki", "girl", "daughter", "बेटी", "लड़की", "बच्ची")):
        return females[0]
    if len(males) == 1 and any(w in cleaned for w in ("beta", "bete", "ladka", "ladke", "boy", "son", "बेटा", "बेटे", "लड़का", "लड़के", "बच्चा")):
        return males[0]
    return None


def _followup_resolved(state: dict, topic: dict, followup_id: str) -> bool:
    details = topic.get("details") or {}
    children = state.get("profile", {}).get("children") or []

    if followup_id == "delivery_type":
        blob = f"{topic.get('value') or ''} {details.get('delivery_type') or ''}".lower()
        if any(word in blob for word in ("operation", "caesarean", "cesarean", "normal", "home", "hospital")):
            return True

    if followup_id == "affected_child":
        if len(children) <= 1:
            if len(children) == 1 and _is_positive(topic):
                child_id = children[0].get("id", "child_1")
                if not any(inst.get("person_id") == child_id for inst in topic.get("instances") or []):
                    topic.setdefault("instances", []).append({
                        "person_id": child_id,
                        "value": topic.get("value"),
                        "details": dict(details),
                    })
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
            if any(w in sval for w in ("all", "both", "dono", "sabhi", "sabko", "everyone", "sab", "दोनों", "सभी", "सबको", "सब", "हर एक")):
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
    positive = _is_positive(topic)
    answered_no = topic.get("state") == "answered" and topic.get("value") is False
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
        children.append({
            "id": child_id,
            "name": raw.get("name"),
            "age": age,
            "sex": raw.get("sex"),
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


def _spoken_ages(utterance: str) -> list[float]:
    folded = utterance or ""
    for word, number in _NUM_WORDS:
        folded = folded.replace(word, number)
    found = []
    for match in re.finditer(rf"(\d+(?:\.\d+)?)\s*{_MONTH_UNIT}", folded, re.I):
        if _is_duration_after(folded, match.end()):
            continue
        found.append((match.start(), float(match.group(1)) / 12.0))
    for match in re.finditer(rf"(\d+(?:\.\d+)?)\s*{_YEAR_UNIT}", folded, re.I):
        if _is_duration_after(folded, match.end()):
            continue
        found.append((match.start(), float(match.group(1))))
    found.sort()
    return [age for _, age in found]


def _month_age_numbers(utterance: str) -> set[float]:
    """Bare numbers she used as a month age, not as a duration."""
    folded = utterance or ""
    for word, number in _NUM_WORDS:
        folded = folded.replace(word, number)
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
    folded = text or ""
    for word, number in _NUM_WORDS:
        folded = folded.replace(word, number)
    return re.search(_YEAR_UNIT, folded, re.I) is not None


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


def _children_from_ages(ages: list[float]) -> list[dict]:
    return [
        {"id": f"child_{index}", "name": None, "age": age, "sex": None}
        for index, age in enumerate(ages, start=1)
    ]


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


def _children_payload(value: object) -> tuple[str, object]:
    """none: she said there are no children. count: a number without ages. list: child records."""
    if isinstance(value, bool) or value in (0, None, []) or value == "0":
        return "none", []
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return ("none", []) if int(value) <= 0 else ("count", int(value))
    if isinstance(value, str):
        val_clean = str(value).strip().lower()
        if val_clean in {"none", "no", "nahi", "nahin", "zero", "0", "false", "कोई नहीं", "एक भी नहीं"}:
            return "none", []
        val_folded = val_clean
        for word, number in _NUM_WORDS:
            val_folded = val_folded.replace(word, number)
        m = re.search(r"\b(\d+)\b", val_folded)
        if m:
            number = int(m.group(1))
            return ("none", []) if number <= 0 else ("count", number)
    if isinstance(value, (list, dict)):
        return "list", _normalize_children(value)
    return "ignore", []


def _canonical_marital(value: object) -> str | None:
    raw = str(value or "").strip()
    text = raw.lower().replace("_", " ").replace("-", " ")
    denial = (
        text in {"false", "no", "nahi", "nahin", "unmarried", "single", "not married"}
        or "not married" in text
        or any(word in raw for word in ("नहीं", "नही", "अविवाहित", "कुंवारी"))
    )
    about = (
        "about" in text
        or "hone wali" in text
        or "engaged" in text
        or "होने वाली" in raw
    )
    married = (
        text in {"true", "yes", "haan", "married"}
        or "हो गई" in raw
        or "ho gayi" in text
        or "shaadi ho" in text
        or any(word in raw for word in ("विवाहित", "शादीशुदा"))
        or any(word in text for word in ("shadishuda", "shaadi shuda"))
    )
    if about:
        return "about_to_marry"
    if denial and not married:
        return "unmarried"
    if married:
        return "married"
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
    r"^\s*(?:(?:ये|यह|इसका|yeh|ye|iska)\s+)?(?:क्या(?:\s+मतलब(?:\s+है)?|\s+होता\s+है)?|मतलब(?:\s+क्या\s+है)?|यानी|समझ(?:\s+में)?\s+नहीं\s+आया|matlab|yaani|yani|kya(?:\s+matlab(?:\s+hai)?|\s+hota\s+hai)?|samajh(?:\s+mein)?\s+nahi(?:\s+aaya)?)[।.!?]*\s*$",
    re.I,
)
_UNKNOWN_UTT_RE = re.compile(
    r"^(?:पता\s+नहीं|याद\s+नहीं|नहीं\s+पता|pata\s+nahi|don'?t\s+know)[।.!?]*$",
    re.I,
)


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


def _absorb_stated_value(name: str, topic: dict, utterance: str) -> None:
    details = dict(topic.get("details") or {})
    value = topic.get("value")
    val_blob = str(value or "").lower()
    utt_blob = (utterance or "").lower()
    combined = f"{val_blob} {utt_blob}"
    if name == "birth_mode":
        if any(word in val_blob for word in ("operation", "caesarean", "cesarean", "c-section", "ऑपरेशन", "सीजेरियन")):
            details["delivery_type"] = "operation"
        elif any(word in val_blob for word in ("normal", "नॉर्मल")):
            details["delivery_type"] = "normal"
        elif "delivery_type" not in details:
            if any(word in utt_blob for word in ("operation", "caesarean", "cesarean", "c-section", "ऑपरेशन", "सीजेरियन")):
                details["delivery_type"] = "operation"
            elif any(word in utt_blob for word in ("normal", "नॉर्मल")):
                details["delivery_type"] = "normal"
        if any(word in val_blob for word in ("hospital", "clinic", "अस्पताल", "हॉस्पिटल", "नर्सिंग")):
            details["place"] = "hospital"
        elif any(word in val_blob for word in ("home", "ghar", "घर")):
            details["place"] = "home"
        elif "place" not in details:
            if any(word in utt_blob for word in ("hospital", "clinic", "अस्पताल", "हॉस्पिटल", "नर्सिंग")):
                details["place"] = "hospital"
            elif any(word in utt_blob for word in ("home", "ghar", "घर")):
                details["place"] = "home"
    if name == "birth_hearing_vision":
        text = utterance or ""
        if "result" not in details and re.search(r"नॉर्मल|normal", text, re.I) and not re.search(r"अबनॉर्मल|abnormal|दिक्कत", text, re.I):
            details["result"] = "normal"
            details.setdefault("test_done", True)
        if "test_done" not in details and isinstance(value, bool):
            details["test_done"] = value
    if name == "preg_scan":
        text = utterance or ""
        if "result" not in details and re.search(r"नॉर्मल|normal", text, re.I) and not re.search(r"अबनॉर्मल|abnormal|दिक्कत", text, re.I):
            details["result"] = "normal"
            details.setdefault("scan_done", True)
        if "scan_done" not in details and isinstance(value, bool):
            details["scan_done"] = value
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
    if has_topic and not has_children_update and not existing_children:
        spoken_children = _child_ages_already_said(prior_utterances)
        if spoken_children:
            state["profile"]["children"] = _children_from_ages(spoken_children)
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
                    spoken = _spoken_ages(utterance)
                    count_n = int(norm) if kind_c == "count" else 0
                    if kind_c == "count" and not spoken:
                        spoken = _ages_from_prior(count_n, prior_utterances)
                    if kind_c == "count" and aged and len(existing) == len(aged) == count_n:
                        wrote = True
                        continue
                    use = spoken if kind_c == "ignore" or len(spoken) == count_n else []
                    if use and not aged:
                        state["profile"]["children"] = _children_from_ages(use)
                        intro_c = state["topics"]["intro_children"]
                        intro_c["state"] = "answered"
                        intro_c["value"] = True
                        intro_c["source"] = "model"
                        _remember_turn(intro_c, turn_id)
                    elif kind_c == "count" and not aged:
                        state["profile"]["children"] = _children_from_ages([None] * count_n)
                    wrote = True
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
                        spoken_list = _spoken_ages(utterance) or _ages_from_prior(len(norm), prior_utterances)
                        if len(spoken_list) == len(norm):
                            for child, age in zip(norm, spoken_list):
                                child["age"] = age
                    norm = _retag_month_only(norm, utterance, prior_utterances)
                    state["profile"]["children"] = norm
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
                else:
                    val = update.get("value")
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
                    if _UNKNOWN_UTT_RE.match(clean_utt):
                        details[followup_id] = {"state": "unknown"}
                    elif raw_val is not None and not isinstance(raw_val, bool) and not _is_parent_echo(raw_val, previous_value):
                        details[followup_id] = raw_val
                    elif clean_utt:
                        details[followup_id] = clean_utt
                    elif _is_parent_echo(details.get(followup_id), previous_value):
                        details.pop(followup_id, None)
            topic["details"] = details
            if topic.get("state") != "answered":
                topic["state"] = "answered" if previous_state == "answered" else status
            if (
                name == "body_bleeding_bruise"
                and followup_id in {"teeth", "periods"}
                and _is_positive({
                    "state": status,
                    "value": incoming.get(followup_id, update.get("value")),
                })
            ):
                topic["value"] = True
            topic["source"] = "model"
            _absorb_stated_value(name, topic, utterance)
            _remember_turn(topic, turn_id)
            _store_instances(topic, update)
            continue
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
            state["profile"]["caller_name"] = update.get("value")
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
    if not wrote:
        raise ValueError(problems[0] if problems else "no applicable update")
    _apply_rules(state)
    return state


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
