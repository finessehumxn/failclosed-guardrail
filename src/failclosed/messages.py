"""Fixed user-facing messages for every non-pass status.

These are written once, reviewed once, and never generated. The crisis and
emergency resources are US-specific (911, 988); localise them before using this
anywhere else.
"""

from __future__ import annotations

from .status import Status

SAFE_MESSAGES: dict[Status, str] = {
    Status.EMERGENCY: (
        "This sounds like it could be a medical emergency. Please call 911 or go to "
        "the nearest emergency room now."
    ),
    Status.CRISIS: (
        "It sounds like you are going through something really hard. You do not have "
        "to handle it alone. In the US you can call or text 988 to reach the Suicide "
        "and Crisis Lifeline, any time. If you are in immediate danger, call 911."
    ),
    Status.OFF_TOPIC: (
        "I can only help with health questions here. If something about your health "
        "is on your mind, tell me about it."
    ),
    Status.INVALID: (
        "I could not quite understand that. Could you tell me a little more about "
        "what is going on?"
    ),
    Status.UNAVAILABLE: (
        "I cannot safely review your message right now. Please try again in a moment.\n"
        "If this is a medical emergency, call 911 or go to the nearest emergency room. "
        "If you are thinking about harming yourself, call or text 988 (US)."
    ),
}


def safe_message(status: object) -> str:
    """Message for a blocked status. Anything unrecognised gets the UNAVAILABLE
    message, which carries both emergency and crisis resources."""
    try:
        return SAFE_MESSAGES[Status(status)]
    except (ValueError, KeyError):
        return SAFE_MESSAGES[Status.UNAVAILABLE]
