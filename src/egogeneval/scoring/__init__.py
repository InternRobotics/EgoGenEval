"""Official end-to-end scoring interface."""

from .pipeline import EvaluatorSettings, build_official_input, score_predictions

__all__ = [
    "EvaluatorSettings",
    "build_official_input",
    "score_predictions",
]
