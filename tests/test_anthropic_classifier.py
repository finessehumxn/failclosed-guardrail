"""AnthropicClassifier parsing, with a fake client. No network, no anthropic package needed."""

from types import SimpleNamespace

import pytest

from failclosed import ClassifierError, Guardrail, Status
from failclosed.anthropic_classifier import DEFAULT_MODEL, AnthropicClassifier, first_text, parse_reply


def block(type_, text=""):
    return SimpleNamespace(type=type_, text=text)


def response(*blocks, stop_reason="end_turn"):
    return SimpleNamespace(content=list(blocks), stop_reason=stop_reason)


class FakeClient:
    def __init__(self, resp=None, exc=None):
        self.resp, self.exc, self.calls = resp, exc, []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.exc:
            raise self.exc
        return self.resp


def test_default_model_and_request_shape():
    client = FakeClient(response(block("text", '{"status": "pass"}')))
    clf = AnthropicClassifier(client=client)
    assert clf.model == DEFAULT_MODEL == "claude-haiku-4-5-20251001"
    assert clf.classify("hello").label == "pass"
    call = client.calls[0]
    assert call["model"] == DEFAULT_MODEL
    assert call["messages"] == [{"role": "user", "content": "hello"}]
    assert "crisis" in call["system"]


def test_model_is_configurable():
    client = FakeClient(response(block("text", '{"status": "pass"}')))
    AnthropicClassifier(model="some-other-model", client=client).classify("x")
    assert client.calls[0]["model"] == "some-other-model"


@pytest.mark.parametrize(
    "blocks, expected",
    [
        ([block("text", '{"status": "crisis"}')], "crisis"),
        ([block("thinking", "hmm"), block("text", '{"status": "pass"}')], "pass"),
        ([block("text", '```json\n{"status": "emergency"}\n```')], "emergency"),
        ([block("text", '```\n{"status": "invalid"}\n```')], "invalid"),
        ([block("text", 'Here you go: {"status": "off_topic"} hope that helps')], "off_topic"),
    ],
)
def test_parses_tolerantly(blocks, expected):
    clf = AnthropicClassifier(client=FakeClient(response(*blocks)))
    assert clf.classify("x").label == expected


@pytest.mark.parametrize(
    "resp",
    [
        response(),  # empty content list
        response(block("thinking", "...")),  # thinking only, no text
        response(block("text", "")),
        response(block("text", "Sure! The status is pass.")),
        response(block("text", '{"message": "hi"}')),
        response(block("text", '{"status": 1}')),
        response(block("text", '["pass"]')),
        response(block("text", '{"status": "pass"')),  # truncated
        response(block("text", '{"status": "pass"}'), stop_reason="refusal"),
    ],
)
def test_unusable_replies_raise(resp):
    with pytest.raises(ClassifierError):
        AnthropicClassifier(client=FakeClient(resp)).classify("x")


@pytest.mark.parametrize(
    "client",
    [
        FakeClient(exc=TimeoutError("read timeout")),
        FakeClient(exc=RuntimeError("overloaded_error")),
        FakeClient(response()),
        FakeClient(response(block("text", '{"status": "PASS_MAYBE"}'))),
        FakeClient(response(block("text", "no json here"))),
    ],
)
def test_through_guardrail_every_failure_is_unavailable(client):
    verdict = Guardrail(AnthropicClassifier(client=client)).check("my knee hurts")
    assert verdict.status is Status.UNAVAILABLE


def test_label_case_normalised_by_guardrail():
    client = FakeClient(response(block("text", '{"status": " Crisis "}')))
    assert Guardrail(AnthropicClassifier(client=client)).check("x").status is Status.CRISIS


def test_first_text_handles_missing_content():
    assert first_text(SimpleNamespace()) == ""
    assert first_text(SimpleNamespace(content=None)) == ""


def test_parse_reply_rejects_empty():
    with pytest.raises(ClassifierError):
        parse_reply("   ")
