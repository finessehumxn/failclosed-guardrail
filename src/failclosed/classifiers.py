"""Classifier protocol and an offline keyword classifier.

A classifier takes the user's text and returns a label. It is allowed to be
wrong, slow, or broken; the Guardrail wraps it and decides what to do about
that. A classifier never decides on its own that a failure means "pass".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


class ClassifierError(Exception):
    """Raised by a classifier that could not produce a usable label."""


@dataclass(frozen=True)
class ClassifierResult:
    """What a classifier returns.

    `label` is a raw string on purpose. The Guardrail validates it against the
    closed set of statuses; a classifier cannot smuggle in a new route.

    There is no model-written message field. What the user sees on a blocked
    path is fixed, reviewed text (see messages.py), never generated text.
    """

    label: str


@runtime_checkable
class Classifier(Protocol):
    def classify(self, text: str) -> ClassifierResult | str: ...


def _rx(*patterns: str) -> re.Pattern[str]:
    return re.compile(r"\b(?:" + "|".join(patterns) + r")\b", re.IGNORECASE)


# Crisis phrasing, including common typos, euphemisms, and a few Spanish phrases.
# Broader than the backstop because this classifier runs on the normal path and
# its output is measured by the eval harness.
_CRISIS = _rx(
    r"kill(?:ing)? ?my ?self",
    r"end(?:ing)? (?:my life|it all)",
    r"take my (?:own )?life",
    r"want(?:ing)? to die|wanna die|wish i (?:was|were) dead",
    r"sui?ci?c?d\w*",  # suicide, suicidal, sucidal, suicde
    r"unalive",
    r"(?:want to|wanna|gonna|going to) kms",
    r"(?:hurt(?:ing)?|harm(?:ing)?|cut(?:ting)?|burn(?:ing)?) myself|self[- ]harm",
    r"no reason to (?:live|keep going)|better off dead|better off without me",
    r"(?:don'?t|do not) (?:want to|wanna) (?:be here|exist|wake up)",
    r"(?:don'?t|can'?t) see the point (?:anymore|of living|in living)",
    r"no point (?:in living|anymore)",
    # Means-seeking. These often arrive phrased as ordinary medication
    # questions, which is exactly why they must be checked before the
    # domain vocabulary decides the input is a normal "pass".
    r"lethal dose|fatal dose|to (?:not|never) wake up|enough \w+ to die|how many \w+(?: \w+){0,4} to die",
    # Spanish
    r"(?:me )?quiero morir|no quiero vivir|matarme|suicidarme|quitarme la vida|hacerme da[nñ]o",
)

_EMERGENCY = _rx(
    r"chest pain|pain in my chest|chest (?:feels )?tight",
    r"can'?t breathe?|cannot breathe|trouble breathing|not breathing|struggling to breathe",
    r"heart attack|stroke|seizure|seizing",
    r"unconscious|unresponsive|passed out|won'?t wake up",
    r"severe bleeding|bleeding (?:badly|heavily|a lot)|won'?t stop bleeding|coughing up blood|vomiting blood",
    r"overdos(?:e|ed|ing)|took too many",
    r"anaphyla\w*|throat (?:is )?(?:closing|swelling)|face is swelling",
    r"slurred speech|face (?:is )?drooping",
    # Spanish
    r"no puedo respirar|dolor (?:en el|de) pecho|ataque al coraz[oó]n",
)

# Vocabulary that marks an input as in-domain for the example app (a health
# information assistant). Swap this out for your own domain.
_DOMAIN = _rx(
    r"health\w*|medic\w*|meds?|pills?|doctor|dr|nurse|clinic|hospital|pharmac\w*",
    r"symptoms?|pain|ache|aches|hurts?|sore|fever|cough\w*|cold|flu|rash|itch\w*",
    r"headaches?|migraines?|dizz\w*|nause\w*|vomit\w*|tired|fatigue|sleep\w*|insomnia",
    r"blood|pressure|sugar|diabet\w*|insulin|cholesterol|thyroid|asthma|allerg\w*",
    r"dose|dosage|mg|prescri\w*|side effects?|ibuprofen|tylenol|acetaminophen|antibiotics?",
    r"knee|back|stomach|chest|throat|ear|eyes?|skin|heart|lungs?|kidney\w*|liver",
    r"anxi\w*|depress\w*|therap\w*|mental",
    r"lab results?|test results?|diagnos\w*|surgery|vaccin\w*|pregnan\w*",
    r"tingl\w*|numb\w*|swollen|swelling|bruis\w*|sprain\w*",
    # Spanish
    r"dolor|m[eé]dico|pastillas?|fiebre|presi[oó]n|az[uú]car|cabeza|tos",
)

# Common function words. If an input has none of these and no domain words, it
# is treated as gibberish ("invalid") rather than as a real off-topic request.
_FUNCTION_WORDS = _rx(
    r"the|a|an|is|are|was|i|my|me|you|your|what|how|why|when|where|who|can|do|does",
    r"to|for|of|in|on|with|and|or|it|this|that|should|would|best|el|la|que|mi|es",
)


class KeywordClassifier:
    """Deterministic regex classifier for tests, CI, and the offline demo.

    It is a stub, not a product. It was written alongside the eval set in this
    repo, so its score on that set says the harness works, not that keyword
    matching is a good safety classifier.
    """

    def classify(self, text: str) -> ClassifierResult:
        stripped = (text or "").strip()
        if sum(ch.isalpha() for ch in stripped) < 2:
            return ClassifierResult("invalid")
        if _CRISIS.search(stripped):
            return ClassifierResult("crisis")
        if _EMERGENCY.search(stripped):
            return ClassifierResult("emergency")
        if _DOMAIN.search(stripped):
            return ClassifierResult("pass")
        if _FUNCTION_WORDS.search(stripped):
            return ClassifierResult("off_topic")
        return ClassifierResult("invalid")
