"""Adapters that turn model outputs into a scorable predictions.jsonl.

Submitters declare one ``--model-type``:

- ``image`` — the model emits one still image per step (e.g. Qwen-Image-Edit,
  FLUX, OmniGen2).
- ``video`` — the model emits a clip or frame directory per step; the target view
  is the final frame of that step's segment (e.g. Kling, Seedance).
- ``world-model`` — rolled out autoregressively, each step conditioned on the
  model's own previous output.
- ``pose-conditioned`` — consumes a native 6-DoF camera trajectory rather than
  text (e.g. HY-WorldMirror, Lingbot-World). Reference track only.

Each (case, step) is then assigned an ``interface_type`` — the conditioning shape
— by :meth:`egogeneval.adapters.build.InterfaceType.derive`. That is a *per-case*
property, because the benchmark varies ``num_context_images`` from K=1 to K=4: one
image submission legitimately spans both ``image-editing`` and
``multi-image-editing``.

Every family ultimately reduces to *one RGB image per (case, step)*. The adapter
normalises each family to that contract, materialises the chosen frames into an
``outputs/`` directory next to the ``predictions.jsonl``, and writes a row that
passes :func:`egogeneval.predictions.validate_predictions`.
"""

from __future__ import annotations

from .build import (
    AdapterResult,
    InterfaceType,
    ModelType,
    build_predictions,
)
from .discover import discover_generations
from .frames import FrameSelection, extract_step_frame

__all__ = [
    "AdapterResult",
    "FrameSelection",
    "InterfaceType",
    "ModelType",
    "build_predictions",
    "discover_generations",
    "extract_step_frame",
]
