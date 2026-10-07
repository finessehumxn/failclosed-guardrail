"""End-to-end graph tests. The property under test: on every non-pass path,
including every way the classifier can fail, generation is never called."""

import time
import uuid
from unittest.mock import Mock

import pytest
from langgraph.types import Command

from failclosed import KeywordClassifier, Status
from failclosed.graph import DECLINED_MESSAGE, build_graph, route_after_guardrail
from failclosed.messages import SAFE_MESSAGES


def cfg():
    return {"configurable": {"thread_id": str(uuid.uuid4())}}


def spies():
    return Mock(return_value=["ctx"]), Mock(return_value="ANSWER")


class Raises:
    def classify(self, text):
        raise ConnectionError("classifier unreachable")


class Hangs:
    def classify(self, text):
        time.sleep(0.5)
        return "pass"


class Garbage:
    def __init__(self, value):
        self.value = value

    def classify(self, text):
        return self.value


NORMAL = "My blood sugar has been high and my feet tingle. What could cause that?"

BROKEN = {
    "raises": Raises(),
    "times out": Hangs(),
    "returns garbage text": Garbage("Sure! I think this is fine."),
    "returns unknown label": Garbage("pass_with_caution"),
    "returns None": Garbage(None),
    "returns a dict": Garbage({"status": "pass"}),
}


def test_normal_input_pauses_at_interrupt_then_generates_on_resume():
    retrieve, generate = spies()
    graph = build_graph(KeywordClassifier(), retrieve=retrieve, generate=generate)
    config = cfg()

    state = graph.invoke({"input": NORMAL}, config)
    assert state["status"] == "pass"
    snapshot = graph.get_state(config)
    assert snapshot.next == ("confirm",)
    payload = snapshot.tasks[0].interrupts[0].value
    assert "interpretation" in payload and "question" in payload
    # Nothing downstream of the interrupt has run yet.
    retrieve.assert_not_called()
    generate.assert_not_called()

    state = graph.invoke(Command(resume={"confirmed": True}), config)
    retrieve.assert_called_once()
    generate.assert_called_once()
    assert state["answer"] == "ANSWER"
    assert graph.get_state(config).next == ()


def test_correction_on_resume_replaces_interpretation():
    retrieve, generate = spies()
    graph = build_graph(KeywordClassifier(), retrieve=retrieve, generate=generate)
    config = cfg()
    graph.invoke({"input": NORMAL}, config)
    graph.invoke(Command(resume={"confirmed": True, "correction": "peripheral neuropathy"}), config)
    retrieve.assert_called_once_with("peripheral neuropathy")
    assert generate.call_args.args[0] == "peripheral neuropathy"


@pytest.mark.parametrize(
    "reply",
    [
        {"confirmed": False},
        {"correction": "something else"},
        {"confirmed": None},
        "yes",
        ["confirmed"],
        {"confirmed": "true"},
        {"confirmed": 1},
    ],
)
def test_anything_but_explicit_confirmation_stops_before_retrieval(reply):
    retrieve, generate = spies()
    graph = build_graph(KeywordClassifier(), retrieve=retrieve, generate=generate)
    config = cfg()
    graph.invoke({"input": NORMAL}, config)
    state = graph.invoke(Command(resume=reply), config)
    retrieve.assert_not_called()
    generate.assert_not_called()
    assert state["message"] == DECLINED_MESSAGE


@pytest.mark.parametrize(
    "text, status",
    [
        ("I want to end my life", "crisis"),
        ("my dad has crushing chest pain", "emergency"),
        ("what's the best pizza in town", "off_topic"),
        ("asdfghjkl", "invalid"),
        ("", "invalid"),
    ],
)
def test_blocked_statuses_terminate_before_generation(text, status):
    retrieve, generate = spies()
    graph = build_graph(KeywordClassifier(), retrieve=retrieve, generate=generate)
    config = cfg()
    state = graph.invoke({"input": text}, config)
    assert state["status"] == status
    assert state["message"] == SAFE_MESSAGES[Status(status)]
    assert graph.get_state(config).next == ()
    retrieve.assert_not_called()
    generate.assert_not_called()


@pytest.mark.parametrize("classifier", BROKEN.values(), ids=BROKEN.keys())
def test_broken_classifier_never_reaches_generation(classifier):
    retrieve, generate = spies()
    graph = build_graph(classifier, timeout=0.05, retrieve=retrieve, generate=generate)
    config = cfg()
    state = graph.invoke({"input": NORMAL}, config)
    assert state["status"] == "unavailable"
    assert "911" in state["message"] and "988" in state["message"]
    assert graph.get_state(config).next == ()
    retrieve.assert_not_called()
    generate.assert_not_called()


@pytest.mark.parametrize("classifier", BROKEN.values(), ids=BROKEN.keys())
def test_broken_classifier_with_crisis_text_still_routes_to_crisis(classifier):
    _, generate = spies()
    graph = build_graph(classifier, timeout=0.05, generate=generate)
    state = graph.invoke({"input": "i want to kill myself"}, cfg())
    assert state["status"] == "crisis"
    assert state["source"] == "backstop"
    generate.assert_not_called()


class ExplodingGuardrail:
    def check(self, text):
        raise RuntimeError("bug inside the guardrail itself")


def test_exception_inside_guardrail_node_fails_closed():
    _, generate = spies()
    graph = build_graph(guardrail=ExplodingGuardrail(), generate=generate)
    state = graph.invoke({"input": NORMAL}, cfg())
    assert state["status"] == "unavailable"
    generate.assert_not_called()


@pytest.mark.parametrize(
    "state, expected",
    [
        ({"status": "pass"}, "interpret"),
        ({"status": "crisis"}, "safe_exit"),
        ({"status": "emergency"}, "safe_exit"),
        ({"status": "unavailable"}, "safe_exit"),
        ({}, "safe_exit"),
        ({"status": None}, "safe_exit"),
        ({"status": "Pass"}, "safe_exit"),
        ({"status": " pass"}, "safe_exit"),
        ({"status": "weird"}, "safe_exit"),
    ],
)
def test_router_only_forwards_exact_pass(state, expected):
    assert route_after_guardrail(state) == expected
