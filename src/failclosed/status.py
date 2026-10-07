"""The closed set of statuses the guardrail can emit.

Every status except PASS ends the graph before generation. UNAVAILABLE is never
produced by a working classifier on purpose; it is what the guardrail returns
when it could not get a usable answer, and it is routed exactly like a block.
"""

from __future__ import annotations

from enum import StrEnum


class Status(StrEnum):
    PASS = "pass"
    EMERGENCY = "emergency"
    CRISIS = "crisis"
    OFF_TOPIC = "off_topic"
    INVALID = "invalid"
    UNAVAILABLE = "unavailable"


# Labels a classifier is allowed to return. UNAVAILABLE is deliberately absent:
# a classifier that says "unavailable" is treated as a classifier failure.
CLASSIFIER_LABELS: frozenset[str] = frozenset(
    {Status.PASS, Status.EMERGENCY, Status.CRISIS, Status.OFF_TOPIC, Status.INVALID}
)

# Statuses where getting it wrong means a person in danger gets a normal answer.
SAFETY_CRITICAL: frozenset[Status] = frozenset({Status.EMERGENCY, Status.CRISIS})


def normalize_label(value: object) -> Status | None:
    """Turn a classifier's raw label into a Status, or None if it is not one.

    Lenient about case and surrounding whitespace, strict about everything else.
    None means "unknown", and the caller must fail closed on it.
    """
    if not isinstance(value, str):
        return None
    label = value.strip().lower()
    if label not in CLASSIFIER_LABELS:
        return None
    return Status(label)
