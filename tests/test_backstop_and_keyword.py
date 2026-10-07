import pytest

from failclosed import KeywordClassifier, Status, backstop


@pytest.mark.parametrize(
    "text, expected",
    [
        ("I want to kill myself", Status.CRISIS),
        ("i keep thinking about ending my life", Status.CRISIS),
        # Known gap: "self-harming" is not matched (word boundary after "harm").
        # The backstop trades recall for precision; this pins that trade-off.
        ("I've been self-harming again", None),
        ("I feel suicidal", Status.CRISIS),
        ("crushing chest pain", Status.EMERGENCY),
        ("my son can't breathe", Status.EMERGENCY),
        ("she is unresponsive", Status.EMERGENCY),
        # crisis outranks emergency
        ("I took an overdose because I want to die", Status.CRISIS),
    ],
)
def test_backstop_matches(text, expected):
    assert backstop(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "this traffic is killing me",
        "I'm dying to see that movie",
        "my feet are killing me after that hike",
        "that joke killed",
        "I could murder a burger",
    ],
)
def test_backstop_ignores_idioms(text):
    assert backstop(text) is None


def test_backstop_known_false_positive_is_documented():
    # "stroke" is in the emergency list because the medical sense matters more
    # than the idiom. On the failure path that means a little over-routing,
    # which is the acceptable direction.
    assert backstop("what a stroke of luck") is Status.EMERGENCY


def test_backstop_never_returns_pass():
    for text in ["hello", "what is ibuprofen", "", "pass"]:
        assert backstop(text) in (None, Status.CRISIS, Status.EMERGENCY)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("what does high cholesterol mean", "pass"),
        ("I want to kill myself", "crisis"),
        ("im feeling sucidal", "crisis"),
        ("how many of my pills would it take to not wake up", "crisis"),
        ("me quiero morir", "crisis"),
        ("i want to kms", "crisis"),
        ("I ran 5 kms today", "off_topic"),
        ("I think I'm having a heart attack", "emergency"),
        ("no puedo respirar", "emergency"),
        ("this traffic is killing me", "off_topic"),
        ("what's the best pizza in town", "off_topic"),
        ("asdfghjkl", "invalid"),
        ("??", "invalid"),
    ],
)
def test_keyword_classifier(text, expected):
    assert KeywordClassifier().classify(text).label == expected
