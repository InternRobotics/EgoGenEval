"""Hash-only frozen references for independently reconstructed benchmark assets.

These records contain no pixels, depths, camera matrices, or machine-local
paths. Verify the files themselves with ``verify_data`` before comparing their
index records here: fingerprints authenticate the recorded hashes, not files.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from ..errors import ValidationError
from ..io import read_jsonl
from ..official.evaluation.eval_assets import _row_resize_contract


FINGERPRINT_FORMAT_VERSION = "egogeneval-eval-fingerprints-v1"
_IDENTITY_FIELDS = ("image_ref", "dataset", "scene_id", "loader_frame_idx")
_DIMENSION_FIELDS = ("image_height", "image_width", "depth_height", "depth_width")
_ASSET_HASH_FIELDS = ("rgb_sha256", "depth_sha256")
_CAMERA_HASH_FIELDS = (
    "effective_intrinsics_sha256",
    "effective_extrinsics_c2w_sha256",
)
_FIELDS = (
    "fingerprint_format_version",
    *_IDENTITY_FIELDS,
    *_DIMENSION_FIELDS,
    *_ASSET_HASH_FIELDS,
    *_CAMERA_HASH_FIELDS,
)
_DATASETS = {"scannet", "matterport3d", "hypersim", "scannetpp"}


def _integer(value: Any, field: str, minimum: int) -> int:
    # Avoid accepting lossy int(1.5), True, or silently changing frame identity.
    if isinstance(value, bool) or not re.fullmatch(r"[0-9]+", str(value)):
        raise ValidationError(f"invalid fingerprint {field}: expected an integer")
    result = int(value)
    if result < minimum:
        raise ValidationError(f"invalid fingerprint {field}: must be >= {minimum}")
    return result


def _metadata(row: dict[str, Any]) -> dict[str, Any]:
    dataset = row.get("dataset")
    if not isinstance(dataset, str) or dataset not in _DATASETS:
        raise ValidationError(f"invalid fingerprint dataset: {dataset!r}")
    ref = row.get("image_ref")
    root = "${" + dataset.upper() + "_ROOT}/"
    if (
        not isinstance(ref, str)
        or not ref.startswith(root)
        or ref == root
        or "\\" in ref
        or ".." in PurePosixPath(ref).parts
    ):
        raise ValidationError(
            "fingerprint image_ref must use its canonical dataset root"
        )
    scene = row.get("scene_id")
    # Hypersim identifies a scene/camera pair, e.g. ai_001_001/cam_00.
    # Preserve that canonical hierarchy while rejecting machine-local paths.
    if (
        not isinstance(scene, str)
        or not scene
        or scene == "."
        or "\\" in scene
        or ":" in scene
        or "\x00" in scene
        or PurePosixPath(scene).is_absolute()
        or ".." in PurePosixPath(scene).parts
    ):
        raise ValidationError("invalid fingerprint scene_id")
    result = {
        "image_ref": ref,
        "dataset": dataset,
        "scene_id": scene,
        "loader_frame_idx": _integer(
            row.get("loader_frame_idx"), "loader_frame_idx", 0
        ),
    }
    for field in _DIMENSION_FIELDS:
        result[field] = _integer(row.get(field), field, 1)
    for field in _ASSET_HASH_FIELDS:
        result[field] = _digest(row.get(field), field)
    return result


def _digest(value: Any, field: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValidationError(f"invalid fingerprint {field}: expected a SHA-256 digest")
    return value


def _camera(value: Any, size: int, field: str) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype="<f8").reshape(size, size).copy()
    except (ValueError, TypeError) as exc:
        raise ValidationError(
            f"{field}: expected a finite {size}x{size} matrix"
        ) from exc
    if not np.isfinite(matrix).all():
        raise ValidationError(f"{field}: expected a finite {size}x{size} matrix")
    return matrix


def _camera_digest(matrix: np.ndarray) -> str:
    # np.array_equal, also used by verify_data, treats -0.0 and 0.0 as equal.
    matrix[matrix == 0] = 0
    return hashlib.sha256(
        matrix.astype("<f8", copy=False).tobytes(order="C")
    ).hexdigest()


def fingerprint_row(row: dict[str, Any]) -> dict[str, Any]:
    """Export only canonical identity, file hashes, dimensions, and camera hashes.

    Historical ScanNet v0.1 K is normalized by the scorer's remaining half-size
    stage. Current K is already fully resized. The stage sequence and affected
    entries match ``_materialize``/``_apply_intrinsics_stages``; the hash encoding
    retains float64 precision so an altered JSON number cannot disappear in a
    float32 cast. Canonical float32 camera values are represented exactly.
    """
    result = {
        "fingerprint_format_version": FINGERPRINT_FORMAT_VERSION,
        **_metadata(row),
    }
    try:
        _, remaining_stages = _row_resize_contract(row)
    except (ValueError, TypeError) as exc:
        raise ValidationError(f"{result['image_ref']}: {exc}") from exc
    intrinsic = _camera(row.get("intrinsics"), 3, "intrinsics")
    for scale in remaining_stages:
        for i, j in ((0, 0), (1, 1), (0, 2), (1, 2)):
            intrinsic[i, j] *= scale
    pose = _camera(row.get("extrinsics_c2w"), 4, "extrinsics_c2w")
    result["effective_intrinsics_sha256"] = _camera_digest(intrinsic)
    result["effective_extrinsics_c2w_sha256"] = _camera_digest(pose)
    return result


def _reference_row(row: dict[str, Any]) -> dict[str, Any]:
    if set(row) != set(_FIELDS):
        raise ValidationError(
            "frozen fingerprint reference has missing or unexpected fields"
        )
    if row.get("fingerprint_format_version") != FINGERPRINT_FORMAT_VERSION:
        raise ValidationError("unsupported frozen fingerprint reference format")
    result = {
        "fingerprint_format_version": FINGERPRINT_FORMAT_VERSION,
        **_metadata(row),
    }
    for field in _CAMERA_HASH_FIELDS:
        result[field] = _digest(row[field], field)
    return result


def verify_fingerprints(
    records: Iterable[dict[str, Any]], reference_path: Path
) -> dict[str, Any]:
    """Match a selected asset index against a superset of frozen fingerprints.

    ``records`` are ordinary ``eval_frames.jsonl`` rows. The reference contains
    only ``fingerprint_row`` exports. Every selected row must match; unselected
    reference rows are allowed, but malformed or duplicate references are not.
    """
    references: dict[str, dict[str, Any]] = {}
    for raw in read_jsonl(Path(reference_path)):
        row = _reference_row(raw)
        ref = row["image_ref"]
        if ref in references:
            raise ValidationError(
                f"frozen fingerprint reference: duplicate image_ref {ref}"
            )
        references[ref] = row
    if not references:
        raise ValidationError("empty frozen fingerprint reference")
    seen: set[str] = set()
    for raw in records:
        row = fingerprint_row(raw)
        ref = row["image_ref"]
        if ref in seen:
            raise ValidationError(f"asset index: duplicate image_ref {ref}")
        seen.add(ref)
        if ref not in references:
            raise ValidationError(
                f"frozen fingerprint reference has no entry for {ref}"
            )
        different = [field for field in _FIELDS if row[field] != references[ref][field]]
        if different:
            raise ValidationError(
                f"{ref}: frozen fingerprint mismatch: {', '.join(different)}"
            )
    if not seen:
        raise ValidationError("empty eval-frames index for fingerprint verification")
    return {
        "status": "verified",
        "frames": len(seen),
        "reference_frames": len(references),
        "reference_checked": True,
    }
