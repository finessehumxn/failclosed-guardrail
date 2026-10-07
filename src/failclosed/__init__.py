"""failclosed: a guardrail routing node that fails closed."""

from .backstop import backstop
from .classifiers import Classifier, ClassifierError, ClassifierResult, KeywordClassifier
from .guardrail import Guardrail, Verdict
from .messages import SAFE_MESSAGES, safe_message
from .status import CLASSIFIER_LABELS, SAFETY_CRITICAL, Status, normalize_label

__all__ = [
    "CLASSIFIER_LABELS",
    "Classifier",
    "ClassifierError",
    "ClassifierResult",
    "Guardrail",
    "KeywordClassifier",
    "SAFETY_CRITICAL",
    "SAFE_MESSAGES",
    "Status",
    "Verdict",
    "backstop",
    "normalize_label",
    "safe_message",
]

__version__ = "0.1.0"
