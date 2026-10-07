"""Deterministic backstop for when the classifier is down.

This is consulted ONLY on the failure path (classifier raised, timed out, or
returned something unusable). It never runs when the classifier produced a
valid answer, so it cannot add false positives to normal operation and it
cannot override the model.

It can only ADD safety routing: it returns CRISIS, EMERGENCY, or None. It never
returns PASS. If it matches nothing, the guardrail still returns UNAVAILABLE.

The phrases are chosen for precision over recall. Idioms such as "this traffic
is killing me" or "I'm dying to see that film" must not match. English only.
"""

from __future__ import annotations

import re

from .status import Status

_CRISIS_RE = re.compile(
    r"\b("
    r"kill(?:ing)? myself|end(?:ing)? my life|take my own life|"
    r"want(?:ing)? to die|wanna die|suicidal|suicide|"
    r"hurt(?:ing)? myself|harm(?:ing)? myself|self[- ]harm|"
    r"no reason to live|better off dead"
    r")\b",
    re.IGNORECASE,
)

_EMERGENCY_RE = re.compile(
    r"\b("
    r"chest pain|can'?t breathe|cannot breathe|trouble breathing|not breathing|"
    r"heart attack|stroke|unconscious|unresponsive|"
    r"severe bleeding|bleeding (?:badly|heavily)|overdos(?:e|ed|ing)|anaphyla\w*"
    r")\b",
    re.IGNORECASE,
)


def backstop(text: str) -> Status | None:
    """Return CRISIS or EMERGENCY on a high-precision match, otherwise None.

    Crisis takes precedence over emergency, matching the classifier's own
    precedence, because a crisis message that also mentions a physical symptom
    still needs crisis resources first.
    """
    if not text:
        return None
    if _CRISIS_RE.search(text):
        return Status.CRISIS
    if _EMERGENCY_RE.search(text):
        return Status.EMERGENCY
    return None
