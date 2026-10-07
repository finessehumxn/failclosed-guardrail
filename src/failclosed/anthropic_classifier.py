"""A guardrail classifier backed by the Anthropic Messages API.

Optional: install with `pip install failclosed[anthropic]`. The `anthropic`
package is imported only when no client is passed in, so the parsing logic can
be tested without it.

Every way this can go wrong raises ClassifierError (or lets the SDK's own
exception propagate). The Guardrail turns any exception into UNAVAILABLE.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .classifiers import ClassifierError, ClassifierResult

DEFAULT_MODEL = "claude-haiku-4-5-20251001"

DEFAULT_DOMAIN = "a health information assistant that explains symptoms and medical terms"

SYSTEM_TEMPLATE = """You are a safety classifier sitting in front of {domain}.
You do not answer the user. You only classify their message into exactly one category:

- pass: an appropriate question for this assistant
- emergency: describes a possibly life-threatening situation happening now that needs emergency services
- crisis: suicidal thoughts, self-harm, or a mental health crisis
- off_topic: a real request, but unrelated to what this assistant is for
- invalid: empty, gibberish, or too vague to act on

If a message could be crisis or emergency, prefer crisis. Figures of speech
("this traffic is killing me", "I'm dying to try it") are not crisis.

Respond with only a JSON object and nothing else:
{{"status": "<category>"}}"""


def first_text(response: Any) -> str:
    """Return the first text block's text, or "" if there is none.

    Never index content[0] blindly: a refusal can come back with an empty
    content list, and a thinking-enabled model puts a thinking block first.
    """
    for block in getattr(response, "content", None) or []:
        if getattr(block, "type", None) == "text":
            return getattr(block, "text", "") or ""
    return ""


_FENCE_OPEN = re.compile(r"^```[a-zA-Z]*\s*")
_FENCE_CLOSE = re.compile(r"\s*```$")
_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def parse_reply(text: str) -> ClassifierResult:
    """Parse the model's reply into a ClassifierResult.

    Tolerates markdown code fences and JSON embedded in prose, because models
    add both. Does not validate the label against the status set; the
    Guardrail does that, so there is exactly one place where that rule lives.
    Raises ClassifierError when no JSON object with a string status is found.
    """
    if not text or not text.strip():
        raise ClassifierError("empty reply")
    body = text.strip()
    if body.startswith("```"):
        body = _FENCE_CLOSE.sub("", _FENCE_OPEN.sub("", body)).strip()
    if not body.startswith("{"):
        match = _OBJECT.search(body)
        if match is None:
            raise ClassifierError("no JSON object in reply")
        body = match.group(0)
    try:
        data = json.loads(body)
    except (ValueError, TypeError) as exc:
        raise ClassifierError(f"reply is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ClassifierError("reply JSON is not an object")
    status = data.get("status")
    if not isinstance(status, str) or not status.strip():
        raise ClassifierError("reply has no string 'status'")
    return ClassifierResult(label=status)


class AnthropicClassifier:
    """Classify with a Claude model.

    Args:
        model: model id. Defaults to a small, fast model; the guardrail runs on
            every request and only needs to emit one label.
        domain: one line describing the assistant this guardrail protects.
        client: an `anthropic.Anthropic`-compatible client. Built lazily from
            the environment (ANTHROPIC_API_KEY) if omitted.
        timeout: per-request timeout in seconds passed to the SDK. The Guardrail
            enforces its own wall-clock timeout as well.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        domain: str = DEFAULT_DOMAIN,
        client: Any | None = None,
        timeout: float = 10.0,
        max_tokens: int = 256,
    ) -> None:
        self.model = model
        self.system = SYSTEM_TEMPLATE.format(domain=domain)
        self.max_tokens = max_tokens
        if client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - depends on environment
                raise ImportError(
                    "AnthropicClassifier needs the anthropic package: "
                    "pip install 'failclosed[anthropic]'"
                ) from exc
            # Retries are left to the SDK default; the Guardrail's timeout caps
            # the total wall-clock time regardless.
            client = anthropic.Anthropic(timeout=timeout)
        self.client = client

    def classify(self, text: str) -> ClassifierResult:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.system,
            messages=[{"role": "user", "content": text}],
        )
        if getattr(response, "stop_reason", None) == "refusal":
            raise ClassifierError("model declined to classify")
        return parse_reply(first_text(response))
