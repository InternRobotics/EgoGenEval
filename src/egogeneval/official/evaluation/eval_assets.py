"""Load frozen, self-contained RGB-D evaluator assets.

Set ``EGOGENEVAL_EVAL_FRAMES_INDEX`` to the ``eval_frames.jsonl`` produced by
``egogeneval download-data --config eval_frames``.  The vendored evaluator then
uses these frozen, hash-verified records instead of requiring four source
dataset mounts.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
from PIL import Image


INDEX_ENV = "EGOGENEVAL_EVAL_FRAMES_INDEX"
ASSET_FORMAT_VERSION = "egogeneval-eval-assets-v0.2"
PREPROCESSING_CONTRACT_ID = "official-direct-target-rgbd-v0.2"
LEGACY_ASSET_FORMAT_VERSION = "egogeneval-private-eval-assets-v0.1"
LEGACY_PREPROCESSING_CONTRACT_ID = "official-direct-target-rgbd-v0.1"

# These are the ordered resize calls in the paper evaluator, not aggregate or
# estimated scale factors.  In particular, ScanNet executes resize_image_half
# twice; collapsing those calls into one 0.25 resize changes interpolated RGB.
_RGB_AND_INTRINSICS_RESIZE_STAGES = {
    "hypersim": (0.5,),
    "scannet": (0.5, 0.5),
    "scannetpp": (0.5,),
    "matterport3d": (0.5,),
}

_CACHE_PATH: Path | None = None
_CACHE_ROWS: list[dict[str, Any]] | None = None
_CACHE_BY_IMAGE: dict[str, dict[str, Any]] | None = None


def resize_stages_for_dataset(dataset: str) -> tuple[float, ...]:
    """Return the exact ordered resize stages used by the paper evaluator."""

    normalized = str(dataset or "").lower()
    try:
        return _RGB_AND_INTRINSICS_RESIZE_STAGES[normalized]
    except KeyError as exc:
        raise ValueError(f"unsupported eval-asset dataset: {dataset!r}") from exc


def _row_resize_contract(row: dict[str, Any]) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return RGB stages and any legacy intrinsics stages still to apply.

    Version 0.2 stores intrinsics after every declared stage.  The published
    private v0.1 assets stored ScanNet intrinsics after only the first half-size
    stage, so that exact version pair needs one compatibility-stage at load
    time.  Keeping this compensation version-scoped prevents v0.2 calibration
    from being scaled twice.
    """

    dataset = str(row.get("dataset") or "").lower()
    expected = resize_stages_for_dataset(dataset)
    version_pair = (
        str(row.get("asset_format_version") or ""),
        str(row.get("preprocessing") or ""),
    )
    current_pair = (ASSET_FORMAT_VERSION, PREPROCESSING_CONTRACT_ID)
    legacy_pair = (LEGACY_ASSET_FORMAT_VERSION, LEGACY_PREPROCESSING_CONTRACT_ID)

    if version_pair == current_pair:
        raw_stages = row.get("resize_stages")
        if not isinstance(raw_stages, (list, tuple)):
            raise ValueError(
                f"{row.get('asset_id')}: {PREPROCESSING_CONTRACT_ID} requires resize_stages"
            )
        stages = tuple(float(scale) for scale in raw_stages)
        if stages != expected:
            raise ValueError(
                f"{row.get('asset_id')}: resize_stages {stages} do not match the "
                f"paper contract for {dataset}: {expected}"
            )
        return stages, ()

    if version_pair == legacy_pair:
        # v0.1 already stored K after one 0.5 stage for every dataset.  Only
        # ScanNet's second official stage was absent from its stored K.
        remaining_intrinsics_stages = (0.5,) if dataset == "scannet" else ()
        return expected, remaining_intrinsics_stages

    raise ValueError(
        f"{row.get('asset_id')}: unsupported eval-asset contract "
        f"asset_format_version={version_pair[0]!r}, preprocessing={version_pair[1]!r}"
    )


def _apply_intrinsics_stages(intrinsic: np.ndarray, stages: tuple[float, ...]) -> np.ndarray:
    result = np.asarray(intrinsic, dtype=np.float32).copy()
    for scale in stages:
        result[0, 0] *= scale
        result[1, 1] *= scale
        result[0, 2] *= scale
        result[1, 2] *= scale
    return result


def _loader_frame_idx(row: dict[str, Any], fallback: int) -> int:
    raw = row.get("loader_frame_idx")
    if raw is None or str(raw).strip() == "":
        version_pair = (
            str(row.get("asset_format_version") or ""),
            str(row.get("preprocessing") or ""),
        )
        if version_pair == (ASSET_FORMAT_VERSION, PREPROCESSING_CONTRACT_ID):
            raise ValueError(f"{row.get('asset_id')}: current eval asset is missing loader_frame_idx")
        return fallback
    try:
        value = int(str(raw))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{row.get('asset_id')}: invalid loader_frame_idx {raw!r}") from exc
    if value < 0:
        raise ValueError(f"{row.get('asset_id')}: invalid loader_frame_idx {raw!r}")
    return value


def _scene_sort_key(row: dict[str, Any]) -> tuple[int, int | str, str]:
    raw = row.get("loader_frame_idx")
    try:
        loader_index = int(str(raw))
    except (TypeError, ValueError):
        return 1, str(row["_image_path"]), str(row.get("image_ref") or "")
    return 0, loader_index, str(row.get("image_ref") or "")


def _load_index() -> tuple[Path, list[dict[str, Any]], dict[str, dict[str, Any]]] | None:
    global _CACHE_BY_IMAGE, _CACHE_PATH, _CACHE_ROWS
    raw = os.environ.get(INDEX_ENV, "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{INDEX_ENV} does not exist: {path}")
    if _CACHE_PATH == path and _CACHE_ROWS is not None and _CACHE_BY_IMAGE is not None:
        return path, _CACHE_ROWS, _CACHE_BY_IMAGE

    rows: list[dict[str, Any]] = []
    by_image: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            image_path = (path.parent / str(row["image_path"])).resolve()
            depth_path = (path.parent / str(row["depth_path"])).resolve()
            if not image_path.is_file() or not depth_path.is_file():
                raise FileNotFoundError(
                    f"incomplete eval asset {row.get('asset_id')}: {image_path} / {depth_path}"
                )
            row["_image_path"] = image_path
            row["_depth_path"] = depth_path
            key = str(image_path)
            if key in by_image:
                raise ValueError(f"duplicate materialized eval image: {key}")
            by_image[key] = row
            rows.append(row)
    if not rows:
        raise ValueError(f"empty eval-frames index: {path}")
    _CACHE_PATH, _CACHE_ROWS, _CACHE_BY_IMAGE = path, rows, by_image
    return path, rows, by_image


def _materialize(row: dict[str, Any], frame_idx: int) -> SimpleNamespace:
    image_path = Path(row["_image_path"])
    resize_stages, remaining_intrinsics_stages = _row_resize_contract(row)
    dataset = str(row.get("dataset") or "").lower()
    if dataset == "scannetpp":
        # The frozen ScanNet++ loader decodes RGB with PIL.  Keep the original
        # encoded bytes on disk and use that same decoder here; JPEG decoding
        # is not guaranteed to be pixel-identical across decoder libraries.
        try:
            with Image.open(image_path) as handle:
                image = np.array(handle)
        except (OSError, ValueError) as exc:
            raise FileNotFoundError(f"cannot read eval RGB: {image_path}") from exc
    else:
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"cannot read eval RGB: {image_path}")
        if image.ndim == 3:
            if image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    for scale in resize_stages:
        image = cv2.resize(
            image,
            dsize=None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_LINEAR,
        )
    target_height = int(row["image_height"])
    target_width = int(row["image_width"])
    if image.shape[:2] != (target_height, target_width):
        raise ValueError(
            f"{row.get('asset_id')}: paper resize stages produced {image.shape[:2]}, "
            f"expected {(target_height, target_width)}"
        )

    depth = np.load(Path(row["_depth_path"]), allow_pickle=False)
    depth = np.asarray(depth, dtype=np.float32)
    expected_shape = (int(row["depth_height"]), int(row["depth_width"]))
    if depth.shape != expected_shape or depth.ndim != 2:
        raise ValueError(f"{row.get('asset_id')}: depth shape {depth.shape} != {expected_shape}")
    pose = np.asarray(row["extrinsics_c2w"], dtype=np.float32).reshape(4, 4)
    intrinsic = np.asarray(row["intrinsics"], dtype=np.float32).reshape(3, 3)
    intrinsic = _apply_intrinsics_stages(intrinsic, remaining_intrinsics_stages)
    return SimpleNamespace(
        frame_idx=_loader_frame_idx(row, frame_idx),
        frame_name=str(image_path),
        image=image,
        depth=depth,
        extrinsics=pose,
        intrinsics=intrinsic,
        pos=None,
        direction=None,
        pitch=None,
        height=None,
    )


def load_eval_asset_frame(path: str) -> SimpleNamespace | None:
    loaded = _load_index()
    if loaded is None:
        return None
    _, _, by_image = loaded
    key = str(Path(path).expanduser().resolve())
    row = by_image.get(key)
    if row is None:
        return None
    return _materialize(row, 0)


def load_eval_asset_scene(dataset: str, scene_id: str) -> tuple[list[Any], float] | None:
    loaded = _load_index()
    if loaded is None:
        return None
    _, rows, _ = loaded
    selected = [
        row
        for row in rows
        if str(row.get("dataset", "")).lower() == str(dataset).lower()
        and str(row.get("scene_id", "")) == str(scene_id)
    ]
    if not selected:
        return None
    selected.sort(key=_scene_sort_key)
    items = [_materialize(row, index) for index, row in enumerate(selected)]
    frame_indices = [item.frame_idx for item in items]
    if len(frame_indices) != len(set(frame_indices)):
        raise ValueError(f"duplicate loader_frame_idx in eval assets for {dataset}/{scene_id}")
    maximum = max(
        (float(np.max(item.depth)) for item in items if item.depth.size),
        default=0.0,
    )
    return items, maximum
