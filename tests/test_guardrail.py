"""The fail-closed rule, tested at the Guardrail boundary.

Every test in the first group is a way a classifier can fail. If any of them
starts returning PASS, someone has reintroduced "unknown means safe".
"""

import time

import pytest

from failclosed import ClassifierError, ClassifierResult, Guardrail, Status


class Returns:
    def __init__(self, value):
        self.value = value

    def classify(self, text):
        return self.value


class Raises:
    def __init__(self, exc):
        self.exc = exc

    def classify(self, text):
        raise self.exc


class Sleeps:
    def __init__(self, seconds, value="pass"):
        self.seconds, self.value = seconds, value

    def classify(self, text):
        time.sleep(self.seconds)
        return self.value


NORMAL = "my knee hurts when I climb stairs"

FAILING_CLASSIFIERS = {
    "raises ConnectionError": Raises(ConnectionError("down")),
    "raises ClassifierError": Raises(ClassifierError("bad reply")),
    "raises RuntimeError": Raises(RuntimeError("overloaded")),
    "raises ValueError": Raises(ValueError("x")),
    "returns None": Returns(None),
    "returns int": Returns(1),
    "returns dict": Returns({"status": "pass"}),
    "returns bool True": Returns(True),
    "returns empty label": Returns(""),
    "returns unknown label": Returns("PASS_MAYBE"),
    "returns 'safe'": Returns("safe"),
    "returns 'unavailable' itself": Returns("unavailable"),
    "result with unknown label": Returns(ClassifierResult("allowed")),
    "result with non-str label": Returns(ClassifierResult(None)),  # type: ignore[arg-type]
    "times out": Sleeps(0.5),
}


@pytest.mark.parametrize("classifier", FAILING_CLASSIFIERS.values(), ids=FAILING_CLASSIFIERS.keys())
def test_any_classifier_failure_is_unavailable_never_pass(classifier):
    verdict = Guardrail(classifier, timeout=0.05).check(NORMAL)
    assert verdict.status is Status.UNAVAILABLE
    assert verdict.source == "fail_closed"
    assert verdict.error


def test_timeout_does_not_wait_for_hung_classifier():
    start = time.monotonic()
    Guardrail(Sleeps(1.0), timeout=0.05).check(NORMAL)
    assert time.monotonic() - start < 0.5


@pytest.mark.parametrize(
    "text, expected",
    [
        ("I want to kill myself", Status.CRISIS),
        ("I've been thinking about suicide", Status.CRISIS),
        ("severe chest pain right now", Status.EMERGENCY),
        ("I think my friend overdosed", Status.EMERGENCY),
    ],
)
def test_backstop_routes_safety_cases_when_classifier_is_down(text, expected):
    verdict = Guardrail(Raises(TimeoutError()), timeout=1).check(text)
    assert verdict.status is expected
    assert verdict.source == "backstop"


def test_backstop_is_not_consulted_when_classifier_succeeds():
    calls = []

    def spy(text):
        calls.append(text)
        return Status.CRISIS

    verdict = Guardrail(Returns("pass"), backstop=spy).check(NORMAL)
    assert verdict.status is Status.PASS
    assert calls == []


def test_backstop_cannot_produce_pass():
    verdict = Guardrail(Raises(RuntimeError()), backstop=lambda t: Status.PASS).check(NORMAL)
    assert verdict.status is Status.UNAVAILABLE


def test_broken_backstop_still_fails_closed():
    def broken(text):
        raise RuntimeError("regex engine on fire")

    verdict = Guardrail(Raises(RuntimeError()), backstop=broken).check(NORMAL)
    assert verdict.status is Status.UNAVAILABLE


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("pass", Status.PASS),
        (" Emergency ", Status.EMERGENCY),
        ("CRISIS", Status.CRISIS),
        (ClassifierResult("off_topic"), Status.OFF_TOPIC),
        (ClassifierResult("invalid"), Status.INVALID),
    ],
)
def test_valid_labels_are_accepted(raw, expected):
    verdict = Guardrail(Returns(raw)).check(NORMAL)
    assert verdict.status is expected
    assert verdict.source == "classifier"


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_empty_input_is_invalid_without_calling_classifier(text):
    verdict = Guardrail(Raises(AssertionError("should not be called"))).check(text)
    assert verdict.status is Status.INVALID
    assert verdict.source == "precheck"
