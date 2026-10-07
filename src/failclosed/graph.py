"""Example LangGraph pipeline with the guardrail as its first node.

    START -> guardrail --(pass)--> interpret -> confirm [interrupt] -> retrieve -> generate -> END
                 |                                  |
                 +--(anything else)--> safe_exit    +--(declined)--> END
                                          |
                                         END

Only the exact status "pass" reaches interpret. The routing function has no
default branch that leads forward: a missing or unknown status goes to
safe_exit. Generation, retrieval, and interpretation are injectable plain
functions so the example runs with no API key and tests can spy on them.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from .classifiers import Classifier, KeywordClassifier
from .guardrail import Guardrail
from .messages import safe_message
from .status import Status

logger = logging.getLogger(__name__)


class State(TypedDict, total=False):
    input: str
    status: str
    source: str
    message: str
    interpretation: str
    confirmed: bool
    context: list[str]
    answer: str


DECLINED_MESSAGE = "Okay, I will not look that up. Tell me a bit more about what you meant."


def stub_interpret(text: str) -> str:
    return f"You are asking about: {text.strip()}"


def stub_retrieve(interpretation: str) -> list[str]:
    return [f"(retrieved reference material for: {interpretation})"]


def stub_generate(interpretation: str, context: list[str]) -> str:
    return f"[stub answer] {interpretation} | using {len(context)} source(s)"


def route_after_guardrail(state: State) -> str:
    """Forward only on an exact "pass". Everything else, including a missing
    status, a misspelt one, or a different case, goes to safe_exit."""
    return "interpret" if state.get("status") == Status.PASS.value else "safe_exit"


def route_after_confirm(state: State) -> str:
    return "retrieve" if state.get("confirmed") is True else END


def build_graph(
    classifier: Classifier | None = None,
    *,
    guardrail: Guardrail | None = None,
    timeout: float = 10.0,
    interpret: Callable[[str], str] = stub_interpret,
    retrieve: Callable[[str], list[str]] = stub_retrieve,
    generate: Callable[[str, list[str]], str] = stub_generate,
    checkpointer: Any | None = None,
):
    """Build and compile the example graph.

    Pass either a `classifier` (wrapped in a Guardrail with `timeout`) or a
    ready-made `guardrail`. Defaults to the offline KeywordClassifier.

    The graph pauses inside `confirm` with a real LangGraph interrupt, so it
    needs a checkpointer and a thread id:

        config = {"configurable": {"thread_id": "abc"}}
        graph.invoke({"input": "..."}, config)               # runs until the interrupt
        graph.invoke(Command(resume={"confirmed": True}), config)  # resumes
    """
    if guardrail is None:
        guardrail = Guardrail(classifier or KeywordClassifier(), timeout=timeout)

    def guardrail_node(state: State) -> dict:
        try:
            verdict = guardrail.check(state.get("input", ""))
            return {"status": verdict.status.value, "source": verdict.source}
        except Exception:  # noqa: BLE001 - Guardrail.check should not raise; if it does, fail closed
            logger.exception("guardrail node raised; failing closed")
            return {"status": Status.UNAVAILABLE.value, "source": "fail_closed"}

    def safe_exit_node(state: State) -> dict:
        status = state.get("status")
        try:
            status = Status(status).value
        except ValueError:
            status = Status.UNAVAILABLE.value
        return {"status": status, "message": safe_message(status)}

    def interpret_node(state: State) -> dict:
        return {"interpretation": interpret(state["input"])}

    def confirm_node(state: State) -> dict:
        # Real LangGraph interrupt: the graph checkpoints here and returns to the
        # caller. Nothing downstream has run yet, so the person is confirming a
        # lookup that has not happened, not approving one that already did.
        reply = interrupt(
            {
                "interpretation": state["interpretation"],
                "question": "Is this what you are asking about?",
            }
        )
        # The resume value comes from outside the process. Anything other than an
        # explicit {"confirmed": True} is a no.
        if not isinstance(reply, dict) or reply.get("confirmed") is not True:
            return {"confirmed": False, "message": DECLINED_MESSAGE}
        correction = reply.get("correction")
        update: dict = {"confirmed": True}
        if isinstance(correction, str) and correction.strip():
            update["interpretation"] = correction.strip()
        return update

    def retrieve_node(state: State) -> dict:
        return {"context": retrieve(state["interpretation"])}

    def generate_node(state: State) -> dict:
        answer = generate(state["interpretation"], state.get("context", []))
        return {"answer": answer, "message": answer}

    builder = StateGraph(State)
    builder.add_node("guardrail", guardrail_node)
    builder.add_node("safe_exit", safe_exit_node)
    builder.add_node("interpret", interpret_node)
    builder.add_node("confirm", confirm_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("generate", generate_node)

    builder.add_edge(START, "guardrail")
    builder.add_conditional_edges(
        "guardrail", route_after_guardrail, {"interpret": "interpret", "safe_exit": "safe_exit"}
    )
    builder.add_edge("safe_exit", END)
    builder.add_edge("interpret", "confirm")
    builder.add_conditional_edges("confirm", route_after_confirm, {"retrieve": "retrieve", END: END})
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)

    return builder.compile(checkpointer=checkpointer or MemorySaver())
