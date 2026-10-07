"""The fail-closed guardrail.

The rule: the guardrail returns PASS only when a classifier, within the time
limit, returned a label that is exactly one of the known statuses and that label
was "pass". Every other outcome (exception, timeout, None, wrong type, unknown
label, a classifier claiming "unavailable") is a failure, and a failure is
UNAVAILABLE, which ends the graph like any other block.

On the failure path only, a deterministic backstop gets one look at the text.
It can upgrade UNAVAILABLE to CRISIS or EMERGENCY so the person sees the right
resources while the classifier is down. It can never produce PASS.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from typing import Callable, Literal

from .backstop import backstop as default_backstop
from .classifiers import Classifier, ClassifierResult
from .status import Status, normalize_label

logger = logging.getLogger(__name__)

Source = Literal["classifier", "backstop", "fail_closed", "precheck"]


@dataclass(frozen=True)
class Verdict:
    status: Status
    # Where the status came from. Useful in logs and traces: a spike in
    # "backstop" or "fail_closed" means the classifier is unhealthy.
    source: Source = "classifier"
    error: str | None = None


class Guardrail:
    """Wraps a Classifier and enforces the fail-closed rule.

    Args:
        classifier: anything with `classify(text) -> ClassifierResult | str`.
        timeout: seconds to wait for the classifier. A classifier that does not
            answer in time is treated as down. The worker thread is abandoned,
            not killed (Python cannot kill threads), so classifiers that make
            network calls should also set their own client timeout.
        backstop: function consulted only on the failure path. Must return
            CRISIS, EMERGENCY, or None. Anything else it returns is ignored.
    """

    def __init__(
        self,
        classifier: Classifier,
        timeout: float = 10.0,
        backstop: Callable[[str], Status | None] = default_backstop,
    ) -> None:
        self.classifier = classifier
        self.timeout = timeout
        self.backstop = backstop

    def check(self, text: str) -> Verdict:
        if not (text or "").strip():
            return Verdict(Status.INVALID, source="precheck")

        try:
            raw = self._classify_with_timeout(text)
            status = self._validate(raw)
        except Exception as exc:  # noqa: BLE001 - any failure fails closed
            return self._fail_closed(text, exc)

        return Verdict(status, source="classifier")

    def _classify_with_timeout(self, text: str) -> object:
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="failclosed")
        try:
            future = pool.submit(self.classifier.classify, text)
            try:
                return future.result(timeout=self.timeout)
            except FutureTimeout as exc:
                raise TimeoutError(f"classifier did not answer within {self.timeout}s") from exc
        finally:
            # Do not wait for a hung classifier; that would defeat the timeout.
            pool.shutdown(wait=False, cancel_futures=True)

    @staticmethod
    def _validate(raw: object) -> Status:
        if isinstance(raw, ClassifierResult):
            label = raw.label
        elif isinstance(raw, str):
            label = raw
        else:
            raise TypeError(f"classifier returned {type(raw).__name__}, expected ClassifierResult or str")
        status = normalize_label(label)
        if status is None:
            raise ValueError(f"classifier returned unknown label {label!r}")
        return status

    def _fail_closed(self, text: str, exc: BaseException) -> Verdict:
        # Log the failure, not the user's text: the input may be sensitive.
        error = f"{type(exc).__name__}: {exc}"
        logger.error("guardrail classifier unavailable, failing closed (%s)", error)

        try:
            hit = self.backstop(text)
        except Exception:  # noqa: BLE001 - a broken backstop must not open the gate
            logger.exception("guardrail backstop raised; ignoring it")
            hit = None

        if hit in (Status.CRISIS, Status.EMERGENCY):
            logger.warning("guardrail backstop matched %s while classifier was down", hit)
            return Verdict(Status(hit), source="backstop", error=error)
        return Verdict(Status.UNAVAILABLE, source="fail_closed", error=error)
