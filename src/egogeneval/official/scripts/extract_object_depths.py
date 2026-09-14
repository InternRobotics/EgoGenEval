#!/usr/bin/env python3
"""Extract the per-object depth medians used by the adopted Obj metric.

This GPU entry point never reruns detection or correspondence matching. It
loads frozen object boxes/matches, physical target RGB-D, and DA3 depth for the
generated target image. The CPU-only final formula is implemented separately
in ``object_metrics.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
CODE_ROOT = SCRIPT_DIR.parent
EVALUATION_ROOT = CODE_ROOT / "evaluation"
for candidate in (SCRIPT_DIR, EVALUATION_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import ssp_detection as benchmark
from direct_target_rgbd import direct_target_rgbd


INNER_MARGIN_RATIO = 0.10
MIN_VALID_PIXELS = 16
MIN_DEPTH = 0.05
MAX_DEPTH = 20.0
SCALAR_KEYS = (
    "num_gt_objects", "num_pred_all", "num_matched", "num_missing",
    "num_hallucination", "Completeness", "ObjectIntegrity",
    "Center_Error_Norm", "Topology", "Hallucination_Rate",
    "Matched_Pair_Area_Ratio_Score", "mean_identity_sim",
    "mean_struct_ssim", "mean_edge_iou", "mean_shape_sim",
    "mean_color_sim", "mean_sharpness_ratio",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if len({str(row.get("id")) for row in rows}) != len(rows):
        raise ValueError(f"{path}: duplicate IDs")
    return rows


def load_details(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    details = payload.get("details") if isinstance(payload, dict) else None
    if not isinstance(details, list):
        raise ValueError(f"{path}: details[] missing")
    if len({str(row.get("id")) for row in details}) != len(details):
        raise ValueError(f"{path}: duplicate IDs")
    return details


def is_context_id(sample_id: str) -> bool:
    return any(token.startswith("ctxK") for token in sample_id.split("__"))


def scalars_equal(compact: dict, raw: dict) -> None:
    mismatches = []
    for key in SCALAR_KEYS:
        if key not in compact or key not in raw:
            continue
        left, right = compact[key], raw[key]
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            if not math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-9):
                mismatches.append(f"{key}:{left}!={right}")
        elif left != right:
            mismatches.append(f"{key}:{left!r}!={right!r}")
    if mismatches:
        raise ValueError(f"{compact.get('id')}: frozen scalar mismatch {mismatches[:5]}")


def frozen_rows(canonical_path: Path, raw_path: Path, subset: str) -> list[dict]:
    raw = load_details(raw_path)
    raw_by_id = {str(row["id"]): row for row in raw}
    if canonical_path.suffix == ".jsonl":
        canonical = load_jsonl(canonical_path)
    else:
        canonical = load_details(canonical_path)
    if subset == "main":
        canonical = [row for row in canonical if not is_context_id(str(row["id"]))]
    elif subset == "context":
        canonical = [row for row in canonical if is_context_id(str(row["id"]))]
    elif subset != "all":
        raise ValueError(f"unknown subset {subset}")
    output = []
    for compact in canonical:
        if isinstance(compact.get("objects"), list):
            output.append(compact)
            continue
        raw_row = raw_by_id.get(str(compact["id"]))
        if raw_row is None:
            raise ValueError(f"{compact['id']}: missing complete object record")
        scalars_equal(compact, raw_row)
        restored = dict(raw_row)
        restored.update(compact)
        restored["objects"] = raw_row["objects"]
        output.append(restored)
    return output


def frame_path(frame: Any) -> str:
    if isinstance(frame, str):
        return frame
    if isinstance(frame, dict):
        for key in ("image_path", "path", "frame_name"):
            value = frame.get(key)
            if isinstance(value, str) and value:
                return value
    return ""


def generation_steps(path: Path) -> dict[str, dict]:
    steps: dict[str, dict] = {}
    for case in load_jsonl(path):
        case_id = str(case.get("id") or case.get("sample_id") or "")
        inputs = case.get("input_images")
        targets = case.get("target_images")
        generated = case.get("generated_images")
        if not case_id or not isinstance(inputs, list) or not inputs:
            raise ValueError(f"{path}: malformed case {case_id!r}")
        if not isinstance(targets, list) or not isinstance(generated, list) or len(targets) != len(generated):
            raise ValueError(f"{case_id}: target/generated length mismatch")
        generated_by_step = {}
        for local_index, item in enumerate(generated, start=1):
            step = int(item.get("step") or local_index) if isinstance(item, dict) else local_index
            generated_by_step[step] = frame_path(item)
        if set(generated_by_step) != set(range(1, len(targets) + 1)):
            raise ValueError(f"{case_id}: generated step set is incomplete")
        for index, target in enumerate(targets, start=1):
            sample_id = f"{case_id}__step{index}"
            if sample_id in steps:
                raise ValueError(f"{path}: duplicate step {sample_id}")
            steps[sample_id] = {
                "id": sample_id,
                "parent_id": case_id,
                "step": index,
                "num_steps": len(targets),
                "dataset": str(case.get("dataset") or "").lower(),
                "scene_id": str(case.get("scene_id") or ""),
                "target_path": frame_path(target),
                "generated_path": generated_by_step[index],
            }
    return steps


def object_arrays(row: dict) -> tuple[list, list, dict[int, int]]:
    gt_count = int(row["num_gt_objects"])
    pred_count = int(row["num_pred_all"])
    matched_count = int(row["num_matched"])
    gt_boxes = [None] * gt_count
    pred_boxes = [None] * pred_count
    match_map: dict[int, int] = {}
    for obj in row.get("objects") or []:
        gt_index, pred_index = obj.get("gt_idx"), obj.get("pred_idx")
        if gt_index is not None:
            gt_index = int(gt_index)
            if obj.get("gt_box") is not None:
                gt_boxes[gt_index] = [float(value) for value in obj["gt_box"][:4]]
        if pred_index is not None:
            pred_index = int(pred_index)
            if obj.get("pred_box") is not None:
                pred_boxes[pred_index] = [float(value) for value in obj["pred_box"][:4]]
        if obj.get("status") == "match":
            if gt_index is None or pred_index is None:
                raise ValueError(f"{row['id']}: matched object lacks indices")
            match_map[gt_index] = pred_index
    if any(box is None for box in gt_boxes):
        raise ValueError(f"{row['id']}: missing GT box")
    if any(pred_boxes[index] is None for index in match_map.values()):
        raise ValueError(f"{row['id']}: missing matched generated box")
    if len(match_map) != matched_count or len(set(match_map.values())) != matched_count:
        raise ValueError(f"{row['id']}: correspondence is not one-to-one")
    return gt_boxes, pred_boxes, match_map


def median_inner_box(depth: np.ndarray, box: list[float], image_shape: tuple[int, int]) -> float | None:
    values = np.asarray(depth, dtype=np.float32).squeeze()
    if values.ndim != 2:
        return None
    depth_h, depth_w = values.shape
    image_h, image_w = image_shape
    x1, y1, x2, y2 = [float(value) for value in box[:4]]
    if not np.all(np.isfinite([x1, y1, x2, y2])):
        return None
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    width, height = x2 - x1, y2 - y1
    if width < 1.0 or height < 1.0:
        return None
    margin_x = min(width * INNER_MARGIN_RATIO, max(0.0, (width - 1.0) * 0.45))
    margin_y = min(height * INNER_MARGIN_RATIO, max(0.0, (height - 1.0) * 0.45))
    x1, x2, y1, y2 = x1 + margin_x, x2 - margin_x, y1 + margin_y, y2 - margin_y
    scale_x, scale_y = depth_w / float(image_w), depth_h / float(image_h)
    ix1, ix2 = int(np.floor(x1 * scale_x)), int(np.ceil(x2 * scale_x))
    iy1, iy2 = int(np.floor(y1 * scale_y)), int(np.ceil(y2 * scale_y))
    ix1, ix2 = max(0, min(depth_w, ix1)), max(0, min(depth_w, ix2))
    iy1, iy2 = max(0, min(depth_h, iy1)), max(0, min(depth_h, iy2))
    if ix2 <= ix1 or iy2 <= iy1:
        return None
    crop = values[iy1:iy2, ix1:ix2].reshape(-1)
    valid = np.isfinite(crop) & (crop > MIN_DEPTH) & (crop < MAX_DEPTH)
    if int(valid.sum()) < MIN_VALID_PIXELS:
        return None
    return float(np.median(crop[valid]))


def da3_runner():
    return benchmark.SspDetectionRunner(
        eval_mode="none", oracle_mode=False, save_vis=False, device=None,
        da3_src_path=os.environ.get("DA3_CODE_DIR", str(CODE_ROOT / "third_party/Depth-Anything-3/src")),
        da3_model_path=os.environ.get("DA3_MODEL_DIR", "checkpoints/DA3NESTED-GIANT-LARGE"),
        proposal_mode="none", enable_sam3=False, enable_dino=False,
        enable_da3=True, enable_grounding_dino=False, enable_vlm_match=False,
        da3_max_failures=100,
    )


def generated_depth(runner, path: str, target_shape: tuple[int, int]) -> np.ndarray:
    errors = []
    for attempt in range(1, 4):
        try:
            with benchmark.torch.inference_mode():
                prediction = runner.sem_tools.da3.inference([path])
            depth = benchmark._to_numpy(prediction.depth[0]).squeeze().astype(np.float32)
            if depth.ndim != 2 or not np.isfinite(depth).any():
                raise RuntimeError(f"invalid DA3 depth {depth.shape}")
            if depth.shape != target_shape:
                depth = cv2.resize(depth, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
            return depth
        except Exception as error:
            errors.append(f"attempt{attempt}:{type(error).__name__}:{error}")
            time.sleep(0.25 * attempt)
    raise RuntimeError(f"DA3 failed for {path}: {errors}")


def evaluate(row: dict, step: dict, runner) -> dict:
    gt_boxes, pred_boxes, match_map = object_arrays(row)
    gt_depths = [None] * len(gt_boxes)
    pred_depths = [None] * len(pred_boxes)
    if len(gt_boxes) >= 2:
        dataset = step["dataset"]
        if not dataset:
            dataset, _ = benchmark.infer_dataset_and_scene_from_path(step["target_path"])
        target = direct_target_rgbd(dataset, step["target_path"])
        image_shape = tuple(int(value) for value in target.image.shape[:2])
        gt_depths = [median_inner_box(target.depth, box, image_shape) for box in gt_boxes]
        if len(match_map) >= 2:
            prediction = generated_depth(runner, step["generated_path"], tuple(target.depth.shape[:2]))
            for pred_index in match_map.values():
                pred_depths[pred_index] = median_inner_box(prediction, pred_boxes[pred_index], image_shape)
    records = []
    for gt_index in range(len(gt_boxes)):
        pred_index = match_map.get(gt_index)
        records.append({
            "gt_idx": gt_index,
            "pred_idx": pred_index,
            "gt_median_depth": gt_depths[gt_index],
            "pred_median_depth": pred_depths[pred_index] if pred_index is not None else None,
        })
    output = {
        "id": row["id"],
        "parent_id": step["parent_id"],
        "step": step["step"],
        "num_steps": step["num_steps"],
        "DepthAwareTopology_Support": "central_80_percent_of_frozen_detector_box",
        "DepthAwareTopology_ObjectDepths": records,
    }
    for key in SCALAR_KEYS:
        if key in row:
            output[key] = row[key]
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--raw-obj-json", type=Path, required=True)
    parser.add_argument("--canonical-obj", type=Path, required=True)
    parser.add_argument("--generation-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--subset", choices=("main", "context", "all"), default="main")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard specification")
    rows = frozen_rows(args.canonical_obj, args.raw_obj_json, args.subset)
    steps = generation_steps(args.generation_jsonl)
    if {str(row["id"]) for row in rows} != set(steps):
        raise ValueError("frozen object and canonical generation step IDs differ")
    selected = [row for index, row in enumerate(rows) if index % args.shard_count == args.shard_index]
    requires_da3 = any(int(row["num_gt_objects"]) >= 2 and int(row["num_matched"]) >= 2 for row in selected)
    runner = da3_runner() if requires_da3 else None
    if runner is not None and not runner.sem_tools.has_da3():
        raise RuntimeError(f"DA3 initialization failed: {runner.sem_tools.model_status}")
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_jsonl.with_name(args.output_jsonl.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(evaluate(row, steps[str(row["id"])], runner), ensure_ascii=False) + "\n")
    os.replace(temporary, args.output_jsonl)
    manifest = {
        "schema": "final-object-depth-extraction",
        "model": args.model,
        "rows": len(selected),
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "support": "central 80% frozen detector box; median; 0.05--20; at least 16 valid pixels",
        "generated_depth": "DA3 single-image depth; no rollout source; no GT scale anchor",
        "raw_obj_json": str(args.raw_obj_json.resolve()),
        "raw_obj_sha256": sha256(args.raw_obj_json),
        "canonical_obj": str(args.canonical_obj.resolve()),
        "canonical_obj_sha256": sha256(args.canonical_obj),
        "generation_jsonl": str(args.generation_jsonl.resolve()),
        "generation_jsonl_sha256": sha256(args.generation_jsonl),
        "output_jsonl": str(args.output_jsonl.resolve()),
        "output_sha256": sha256(args.output_jsonl),
    }
    manifest_path = args.output_jsonl.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
