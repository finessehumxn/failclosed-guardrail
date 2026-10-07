"""Offline demo: `python -m failclosed.demo`.

Runs three inputs through the example graph with the KeywordClassifier and a
stub generator, so no API key is needed:

1. A normal question pauses at the human-in-the-loop interrupt, then resumes.
2. A crisis message ends at the guardrail; generation never runs.
3. The same normal question with a broken classifier ends as "unavailable".
"""

from __future__ import annotations

import logging
import uuid

from langgraph.types import Command

from .classifiers import KeywordClassifier
from .graph import build_graph, stub_generate


class CountingGenerator:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, interpretation: str, context: list[str]) -> str:
        self.calls += 1
        return stub_generate(interpretation, context)


class BrokenClassifier:
    def classify(self, text: str):
        raise ConnectionError("simulated outage")


def _config() -> dict:
    return {"configurable": {"thread_id": str(uuid.uuid4())}}


def _interrupt_payload(graph, config) -> dict | None:
    snapshot = graph.get_state(config)
    for task in snapshot.tasks:
        if task.interrupts:
            return task.interrupts[0].value
    return None


def _rule(title: str) -> None:
    print(f"\n=== {title} " + "=" * max(0, 60 - len(title)))


def main() -> int:
    logging.basicConfig(level=logging.CRITICAL)  # keep the demo output readable

    normal = "My blood sugar has been high and my feet tingle. What could cause that?"
    crisis = "I don't want to be here anymore, I want to end my life"

    _rule("1. normal question: pauses for confirmation, then resumes")
    gen = CountingGenerator()
    graph = build_graph(KeywordClassifier(), generate=gen)
    config = _config()
    state = graph.invoke({"input": normal}, config)
    print(f"input:            {normal}")
    print(f"guardrail status: {state['status']}")
    payload = _interrupt_payload(graph, config)
    print(f"paused at:        {graph.get_state(config).next}")
    print(f"interrupt asks:   {payload['question']}")
    print(f"interpretation:   {payload['interpretation']}")
    print(f"generate calls so far: {gen.calls}")
    print('resuming with {"confirmed": True} ...')
    state = graph.invoke(Command(resume={"confirmed": True}), config)
    print(f"answer:           {state['answer']}")
    print(f"generate calls:   {gen.calls}")

    _rule("2. crisis message: ends at the guardrail")
    gen = CountingGenerator()
    graph = build_graph(KeywordClassifier(), generate=gen)
    config = _config()
    state = graph.invoke({"input": crisis}, config)
    print(f"input:            {crisis}")
    print(f"guardrail status: {state['status']}")
    print(f"next node:        {graph.get_state(config).next or '(none, graph finished)'}")
    print(f"message:          {state['message']}")
    print(f"generate calls:   {gen.calls}")

    _rule("3. classifier down: fails closed")
    gen = CountingGenerator()
    graph = build_graph(BrokenClassifier(), generate=gen)
    config = _config()
    state = graph.invoke({"input": normal}, config)
    print(f"input:            {normal}")
    print(f"guardrail status: {state['status']} (source: {state['source']})")
    print(f"message:          {state['message']}")
    print(f"generate calls:   {gen.calls}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
