"""Build a scorable predictions.jsonl from a model's generation source.

The adapter is deliberately format-tolerant on input and strict on output. It
accepts a *generation source* — a JSONL where each row ties a benchmark
``sample_id`` to the images the model produced for that case — and emits the
exact ``predictions.jsonl`` + ``outputs/`` layout the scorer expects.

Two input shapes are supported per row:

1. ``generated_images``: ``[{"step": 1, "image_path": "..."}, ...]`` — the
   canonical shape used across image / video / world families. ``image_path``
   may be a still, an encoded video, or a directory of frames (see
   :mod:`egogeneval.adapters.frames`).
2. ``steps``: ``{"1": "path", "2": "path"}`` — a minimal mapping for users who
   only have per-step files and nothing else.

Absolute paths in an existing generation index can be rewritten with
``--path-map OLD=NEW`` prefix rules, without mutating the source file.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from ..data import ManifestIndex, validate_manifest
from ..errors import ValidationError
from ..io import read_jsonl, sha256_file, write_jsonl
from ..predictions import validate_predictions
from .discover import discover_generations
from .frames import FrameSelection, extract_step_frame


class ModelType(str, Enum):
    """The generator families EgoGenEval can evaluate.

    This is the only family selector users declare (``--model-type``). It
    describes *the model*, which is a stable property of a submission. The
    per-case :class:`InterfaceType` is derived from it -- see
    :meth:`InterfaceType.derive`.
    """

    IMAGE = "image"
    VIDEO = "video"
    WORLD_MODEL = "world-model"
    POSE_CONDITIONED = "pose-conditioned"

    @classmethod
    def parse(cls, value: str | ModelType) -> ModelType:
        if isinstance(value, cls):
            return value
        text = str(value or "").strip().lower()
        aliases = {
            "image": cls.IMAGE,
            "image-editing": cls.IMAGE,
            "video": cls.VIDEO,
            "world-model": cls.WORLD_MODEL,
            "world_model": cls.WORLD_MODEL,
            "pose-conditioned": cls.POSE_CONDITIONED,
            "pose_conditioned": cls.POSE_CONDITIONED,
        }
        if text in aliases:
            return aliases[text]
        choices = ", ".join(member.value for member in cls)
        raise ValidationError(f"unknown model_type {value!r}; expected one of: {choices}")


class InterfaceType(str, Enum):
    """Per-(case, step) conditioning shape, derived from the model family.

    Not user-facing: this is written into ``predictions.jsonl`` for provenance
    but is never selected by hand. A single submission legitimately spans
    several interface types, because ``num_context_images`` varies per case.
    """

    IMAGE_EDITING = "image-editing"
    MULTI_IMAGE_EDITING = "multi-image-editing"
    POSE_CONDITIONED = "pose-conditioned"
    AUTOREGRESSIVE_ROLLOUT = "autoregressive-rollout"

    @classmethod
    def derive(
        cls, model_type: str | ModelType, *, num_context_images: int = 1
    ) -> InterfaceType:
        """Map ``(model_type, K)`` to the conditioning shape for one case.

        ============== ========================= =========================
        model_type     K = 1                     K > 1
        ============== ========================= =========================
        image          image-editing             multi-image-editing
        video          autoregressive-rollout    autoregressive-rollout
        world-model    autoregressive-rollout    autoregressive-rollout
        pose-condit'd  pose-conditioned          pose-conditioned
        ============== ========================= =========================

        Video and world models share ``autoregressive-rollout``: both consume
        their own previous output as the next step's current view. What
        separates them is ``model_type``, which is recorded alongside.
        """

        family = ModelType.parse(model_type)
        if family is ModelType.POSE_CONDITIONED:
            return cls.POSE_CONDITIONED
        if family in (ModelType.VIDEO, ModelType.WORLD_MODEL):
            return cls.AUTOREGRESSIVE_ROLLOUT
        return cls.MULTI_IMAGE_EDITING if num_context_images > 1 else cls.IMAGE_EDITING


@dataclass
class AdapterResult:
    predictions_path: Path
    outputs_dir: Path
    model_id: str
    model_type: str
    #: interface_type -> number of (case, step) rows carrying it.
    interface_types: dict[str, int]
    num_cases: int
    num_steps: int
    warnings: list[str] = field(default_factory=list)



def _apply_path_map(text: str, rules: Mapping[str, str]) -> str:
    for old, new in rules.items():
        if text.startswith(old):
            return new + text[len(old) :]
    return text


def _prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _step_sources(row: dict[str, Any], path_map: Mapping[str, str]) -> dict[int, str]:
    """Extract a ``{step: source_path}`` mapping from one generation row."""

    mapping: dict[int, str] = {}
    generated = row.get("generated_images")
    if isinstance(generated, list) and generated:
        for item in generated:
            step = int(item["step"])
            raw = str(item.get("image_path") or item.get("path") or "")
            if not raw:
                raise ValidationError(f"generated_images step {step} has no image_path")
            mapping[step] = _apply_path_map(raw, path_map)
        return mapping

    steps = row.get("steps")
    if isinstance(steps, dict) and steps:
        for step_text, raw in steps.items():
            mapping[int(step_text)] = _apply_path_map(str(raw), path_map)
        return mapping

    single = row.get("final_generated_image") or row.get("generated_image")
    if single:
        mapping[1] = _apply_path_map(str(single), path_map)
        return mapping

    raise ValidationError(
        f"generation row {row.get('sample_id')!r} has no generated_images/steps/final_generated_image"
    )


def _input_ids(sample_id: str, step: int, num_context_images: int) -> list[str]:
    current = (
        f"{sample_id}:input:1"
        if step == 1
        else f"{sample_id}:generated:step{step - 1}"
    )
    auxiliary = [f"{sample_id}:input:{i}" for i in range(2, num_context_images + 1)]
    return [current, *auxiliary]


def build_predictions(
    generation_source: str | Path | Mapping[str, Any],
    *,
    manifest: str | Path,
    output: str | Path,
    model_id: str,
    model_type: str | ModelType,
    path_map: Mapping[str, str] | None = None,
    frame_selection: FrameSelection | None = None,
    output_format: str = "png",
    save_image: Callable | None = None,
    validate: bool = True,
) -> AdapterResult:
    """Materialise predictions + frames from a generation source.

    Args:
        generation_source: one of three equivalent forms --
            a **directory** in the documented generation layout (resolved by
            :func:`egogeneval.adapters.discover.discover_generations`); a
            **JSONL file** keyed by ``sample_id`` (or ``id``) with
            ``generated_images``/``steps``/``final_generated_image``; or an
            already-loaded ``{sample_id: row}`` mapping.
        manifest: benchmark manifest (defines the required (case, step) set).
        output: run directory; ``predictions.jsonl`` and ``outputs/`` land here.
        model_id: leaderboard model identifier written into every row.
        model_type: generator family (see :class:`ModelType`). The per-case
            ``interface_type`` is derived from it and the case's context size.
        path_map: ``{old_prefix: new_prefix}`` rewrite rules for source paths.
        frame_selection: which frame to pull from a video/segment (default:
            last frame, the paper's boundary-fixed-step policy).
        output_format: extension for materialised frames.
        save_image: optional ``(PIL.Image, Path) -> None`` used when re-encoding.
        validate: run :func:`validate_predictions` before returning.
    """

    manifest_index: ManifestIndex = validate_manifest(Path(manifest))
    family = ModelType.parse(model_type)

    if isinstance(generation_source, Mapping):
        by_id = {str(key): dict(value) for key, value in generation_source.items()}
    elif Path(generation_source).is_dir():
        # A directory is the documented generation layout used by
        # ``score --generations``.
        by_id = discover_generations(Path(generation_source), manifest_index)
    else:
        by_id = {}
        for row in read_jsonl(Path(generation_source)):
            sample_id = str(row.get("sample_id") or row.get("id") or "")
            if not sample_id:
                raise ValidationError("generation row missing sample_id/id")
            by_id[sample_id] = row

    path_map = dict(path_map or {})
    selection = frame_selection or FrameSelection(policy="last")
    output_root = Path(output)
    outputs_dir = output_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    predictions: list[dict[str, Any]] = []
    interface_counts: dict[str, int] = {}

    for manifest_row in manifest_index.rows:
        sample_id = str(manifest_row["sample_id"])
        if sample_id not in by_id:
            raise ValidationError(f"generation source is missing case: {sample_id}")
        gen_row = by_id[sample_id]
        num_context = int(manifest_row.get("num_context_images", 1))
        # Interface type is a per-case property: the same submission spans
        # image-editing and multi-image-editing as K varies across the manifest.
        row_interface = InterfaceType.derive(
            family, num_context_images=num_context
        ).value

        step_sources = _step_sources(gen_row, path_map)
        instructions = manifest_row["instructions"]
        for instruction in instructions:
            step = int(instruction["step"])
            if step not in step_sources:
                raise ValidationError(f"{sample_id}: generation missing step {step}")
            source = Path(step_sources[step])
            output_name = f"{sample_id}_step{step}.{output_format}"
            destination = outputs_dir / output_name
            extract_step_frame(
                source,
                destination,
                selection=selection,
                save=save_image,
            )
            interface_counts[row_interface] = interface_counts.get(row_interface, 0) + 1
            predictions.append(
                {
                    "benchmark_version": "0.1",
                    "manifest_sha256": manifest_index.sha256,
                    "model_id": model_id,
                    "model_type": family.value,
                    "case_id": sample_id,
                    "step_id": step,
                    "output_path": f"outputs/{output_name}",
                    "output_sha256": sha256_file(destination),
                    "interface_type": row_interface,
                    "input_ids": _input_ids(sample_id, step, num_context),
                    "prompt_sha256": _prompt_hash(str(instruction["text"])),
                    "generation": {
                        "source": "egogeneval-adapter",
                        "frame_policy": selection.policy,
                    },
                }
            )

    predictions_path = output_root / "predictions.jsonl"
    write_jsonl(predictions_path, predictions)

    if validate:
        validate_predictions(
            predictions_path,
            manifest_index,
            check_files=True,
            require_complete=True,
        )

    return AdapterResult(
        predictions_path=predictions_path,
        outputs_dir=outputs_dir,
        model_id=model_id,
        model_type=family.value,
        interface_types=interface_counts,
        num_cases=len(manifest_index.rows),
        num_steps=len(predictions),
        warnings=warnings,
    )
