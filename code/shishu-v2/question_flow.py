"""Section bridges and the emergency phrase check for SHISHU calls."""

import re

GROUP_BRIDGE_GROUPS: tuple[str, ...] = ("birth", "body", "illness", "pregnancy", "family")

_EMERGENCY_RE = re.compile(
    r"saans\s+nahi\s+(?:le|aa)|saans\s+bahut\s+tez|not\s+breathing|can'?t\s+breathe|"
    r"behosh\s+(?:hai|ho\s+gaya|ho\s+gayi)|abhi\s+behosh|unconscious|not\s+responding|"
    r"jhatke\s+ruk\s+nahi|khoon\s+ruk\s+nahi|emergency\s+hai|"
    r"सांस\s+नहीं\s+(?:ले|आ)|साँस\s+नहीं\s+(?:ले|आ)|सांस\s+बहुत\s+तेज|साँस\s+बहुत\s+तेज|"
    r"बेहोश\s+(?:है|हो\s+गया|हो\s+गई)|होश\s+नहीं\s+आ|झटके\s+रुक\s+नहीं|खून\s+रुक\s+नहीं|इमरजेंसी\s+है",
    re.I,
)


def is_emergency_text(text: str) -> bool:
    return bool(_EMERGENCY_RE.search(text or ""))
