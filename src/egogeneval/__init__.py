"""Public package interface for EgoGenEval."""

from .versioning import PACKAGE_VERSION

__all__ = ["PACKAGE_VERSION", "score_predictions"]
__version__ = PACKAGE_VERSION


def score_predictions(*args, **kwargs):
    """One-call evaluation: generated images → CMG/SSP → aggregated result.

    Requires: pip install egogeneval[full]

    Example:
        from egogeneval import score_predictions

        results = score_predictions(
            predictions="runs/my_model/predictions.jsonl",
            manifest="data/manifests/egogeneval_v0.1.jsonl",
            eval_frames="data/hf/eval_frames",
            config="configs/evaluator.local.yaml",
            output="runs/my_model/evaluation",
        )
        print(results["overall"], results["cmg"], results["ssp"])
    """
    from .scoring import score_predictions as _score

    return _score(*args, **kwargs)
