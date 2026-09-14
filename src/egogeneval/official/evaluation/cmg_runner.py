import os
import sys
import json
import hashlib
import re
import traceback
import math
import argparse
import gc
import copy
from collections import OrderedDict
from types import SimpleNamespace
import torch
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from typing import Any, Dict, List, Optional

from benchmark_config import INPUT_JSONL, OUTPUT_DIR

_RELEASE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from dataloader import (
    load_scannetpp_scene, load_matterport3d_scene, load_scannet_scene,
    load_hypersim_scene, read_image_cv2_local
)

DATASET_REGISTRY = {
    "scannetpp": load_scannetpp_scene,
    "mp3d": load_matterport3d_scene, "matterport3d": load_matterport3d_scene,
    "scannet": load_scannet_scene,
    "hypersim": load_hypersim_scene,
}
SCENE_CACHE = {}
DEFAULT_DA3_MODEL_DIR = os.environ.get("DA3_MODEL_DIR", "checkpoints/DA3NESTED-GIANT-LARGE")
DEFAULT_DA3_CODE_DIR = os.environ.get(
    "DA3_CODE_DIR", os.path.join(_RELEASE_ROOT, "third_party", "Depth-Anything-3")
)
DA3_ASPECT_POLICIES = (
    "native",
    "gt_center_crop",
    "synthetic_pred_square_center_crop",
    "synthetic_pred_square_stretch",
)
DA3_PROCESS_RES_METHODS = (
    "upper_bound_resize",
    "upper_bound_crop",
    "lower_bound_resize",
    "lower_bound_crop",
)
DA3_REF_VIEW_STRATEGIES = (
    "first",
    "middle",
    "saddle_balanced",
    "saddle_sim_range",
)


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_pose_backends(value: Optional[Any]) -> List[str]:
    if value is None:
        value = os.environ.get("POSE_BACKENDS") or "da3"
    if isinstance(value, str):
        raw_items = re.split(r"[,:\s]+", value)
    else:
        raw_items = list(value)

    aliases = {
        "da3": "da3",
        "depth-anything-3": "da3",
        "depth_anything_3": "da3",
    }
    out = []
    for item in raw_items:
        key = str(item).strip().lower()
        if not key:
            continue
        backend = aliases.get(key)
        if backend is None:
            raise ValueError(f"Unsupported pose backend: {item}. Supported backends: da3")
        if backend not in out:
            out.append(backend)
    return out or ["da3"]


def normalize_dataset_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    n = str(name).lower()
    if n in {"mp3d", "matterport", "matterport3d"}:
        return "mp3d"
    if n in {"hypersim", "hyper-sim", "hyper_sim"}:
        return "hypersim"
    return n


def frame_image_path(frame: Any) -> str:
    if isinstance(frame, str):
        return frame
    if isinstance(frame, dict):
        for key in ("image_path", "frame_name", "path"):
            value = frame.get(key)
            if isinstance(value, str) and value:
                return value
    return ""


def frame_id_from_frame(frame: Any) -> Optional[str]:
    if isinstance(frame, dict):
        value = frame.get("frame_id")
        if value is not None:
            return str(value)
    path = frame_image_path(frame)
    if path:
        return os.path.splitext(os.path.basename(path))[0]
    return None


def cycle_step2_strict_original_step(case: Dict[str, Any]) -> Optional[int]:
    """Return the original benchmark step represented by a strict-only row.

    Strict Cycle reruns contain one local target/output, but that local element
    is the original Cycle step 2.  Keeping this distinction here prevents the
    evaluator from emitting a synthetic ``__step1`` row that must be renamed
    later by an aggregation script.
    """
    strict_meta = case.get("cycle_step2_strict_rerun")
    if not isinstance(strict_meta, dict):
        return None
    try:
        step = int(strict_meta.get("original_cycle_step") or 0)
    except (TypeError, ValueError):
        return None
    return step if step > 0 else None


def cycle_step2_strict_source_frame_ref(case: Dict[str, Any]) -> Optional[str]:
    """Return the physical source frame id for strict Cycle step-2 rows."""
    original_step = cycle_step2_strict_original_step(case)
    frame_ids = case.get("frame_ids")
    if original_step is None or not isinstance(frame_ids, list):
        return None
    source_index = original_step - 1
    if 0 <= source_index < len(frame_ids):
        value = frame_ids[source_index]
        if value is not None:
            return str(value)
    return None


def benchmark_num_steps(case: Dict[str, Any], local_target_count: int) -> int:
    """Return the original trajectory length, not the strict-row local length."""
    original_step = cycle_step2_strict_original_step(case)
    frame_ids = case.get("frame_ids")
    if original_step is not None:
        if isinstance(frame_ids, list) and len(frame_ids) >= 2:
            return max(original_step, len(frame_ids) - 1)
        return max(original_step, local_target_count)
    return local_target_count


def prediction_image_path(case: Dict[str, Any]) -> str:
    for key in ("pred", "pred_path", "prediction_path", "generated_image", "final_generated_image", "output_path"):
        value = case.get(key)
        path = frame_image_path(value)
        if path:
            return path

    for list_key in ("generated_images", "generated_step_images", "pred_images", "predictions", "outputs", "results"):
        value = case.get(list_key)
        if isinstance(value, list):
            for item in reversed(value):
                path = frame_image_path(item)
                if not path and isinstance(item, dict):
                    path = prediction_image_path(item)
                if path:
                    return path

    for container_key in ("prediction", "pred_image", "generation", "output", "result"):
        value = case.get(container_key)
        if isinstance(value, dict):
            path = prediction_image_path(value)
            if path:
                return path
    return ""


def generated_image_path_for_step(case: Dict[str, Any], step: int) -> str:
    generated = case.get("generated_images")
    if isinstance(generated, list):
        for item in generated:
            if not isinstance(item, dict):
                continue
            try:
                item_step = int(item.get("step"))
            except (TypeError, ValueError):
                continue
            if item_step == step:
                path = frame_image_path(item)
                if path:
                    return path
        index = step - 1
        if 0 <= index < len(generated):
            path = frame_image_path(generated[index])
            if path:
                return path
        return ""

    generated_steps = case.get("generated_steps")
    if isinstance(generated_steps, list):
        for item in generated_steps:
            if not isinstance(item, dict):
                continue
            raw_step = item.get("step")
            if raw_step is None:
                raw_step = item.get("step_index")
            try:
                item_step = int(raw_step)
            except (TypeError, ValueError):
                continue
            if item_step == step:
                path = prediction_image_path(item)
                if path:
                    return path
        index = step - 1
        if 0 <= index < len(generated_steps):
            item = generated_steps[index]
            path = frame_image_path(item)
            if not path and isinstance(item, dict):
                path = prediction_image_path(item)
            if path:
                return path

    generated_step_images = case.get("generated_step_images")
    if isinstance(generated_step_images, list):
        index = step - 1
        if 0 <= index < len(generated_step_images):
            path = frame_image_path(generated_step_images[index])
            if path:
                return path
    return prediction_image_path(case)


def pose_metadata_for_step(case: Dict[str, Any], step: int) -> Dict[str, Any]:
    pose_meta = case.get("pose_metadata") if isinstance(case.get("pose_metadata"), dict) else {}
    rels = pose_meta.get("relative_poses")
    if isinstance(rels, list):
        for rel in rels:
            if not isinstance(rel, dict):
                continue
            try:
                rel_step = int(rel.get("step") or -1)
            except (TypeError, ValueError):
                continue
            if rel_step == int(step):
                return rel
    if isinstance(rels, list) and 1 <= int(step) <= len(rels) and isinstance(rels[int(step) - 1], dict):
        return rels[int(step) - 1]
    instructions = case.get("instructions")
    if isinstance(instructions, list):
        for instruction in instructions:
            if not isinstance(instruction, dict):
                continue
            try:
                instruction_step = int(
                    instruction.get("original_cycle_step") or instruction.get("step") or -1
                )
            except (TypeError, ValueError):
                continue
            if instruction_step == int(step):
                return instruction
    if isinstance(instructions, list) and 1 <= int(step) <= len(instructions) and isinstance(instructions[int(step) - 1], dict):
        return instructions[int(step) - 1]
    return {}


def safe_filename_token(value: Any, max_len: int = 180) -> str:
    token = str(value or "unknown")
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", token).strip("._-")
    if not token:
        token = "unknown"
    if len(token) > max_len:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=6).hexdigest()
        token = f"{token[:max_len - 13]}_{digest}"
    return token


def model_name_from_case(case: Dict[str, Any], fallback: str = "unknown_model") -> str:
    for key in ("model", "model_name", "generator", "generator_model", "source_model", "provider"):
        value = case.get(key)
        if value:
            return safe_filename_token(value)

    generated_path = prediction_image_path(case)
    if generated_path:
        parent = os.path.basename(os.path.dirname(os.path.dirname(generated_path)))
        if parent and parent not in {"", ".", "outputs", "images", "generated_images"}:
            return safe_filename_token(parent)

    return safe_filename_token(fallback)


def group_samples_by_model(samples: List[Dict[str, Any]], fallback: str = "unknown_model") -> "OrderedDict[str, List[Dict[str, Any]]]":
    grouped = OrderedDict()
    for sample in samples:
        model_name = model_name_from_case(sample, fallback=fallback)
        grouped.setdefault(model_name, []).append(sample)
    return grouped


def normalize_benchmark_steps(case: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Expand the spatial dataset JSONL into one evaluable row per model call."""
    case_id = str(case.get("id") or case.get("sample_id") or "unknown")
    dataset_name = normalize_dataset_name(case.get("dataset"))
    scene_id = str(case.get("scene_id") or "")

    new_inputs = case.get("input_images")
    new_targets = case.get("target_images")
    if isinstance(new_inputs, list) and new_inputs and isinstance(new_targets, list) and new_targets:
        aux_frames = new_inputs[1:]
        stage1_current_frame_id = frame_id_from_frame(new_inputs[0])
        strict_original_step = cycle_step2_strict_original_step(case)
        strict_source_ref = cycle_step2_strict_source_frame_ref(case)
        original_num_steps = benchmark_num_steps(case, len(new_targets))
        steps = []
        for index, target_frame in enumerate(new_targets):
            local_step = index + 1
            step = (strict_original_step + index) if strict_original_step is not None else local_step
            current_frame = new_inputs[0] if index == 0 else new_targets[index - 1]
            geometry_current_frame = strict_source_ref if (index == 0 and strict_source_ref) else current_frame
            # Geo GT/source-warp metrics use the physical benchmark frame.  The
            # previous generated RGB remains model_source_path for predicted
            # pose estimation.
            metric_current_frame = geometry_current_frame if (index == 0 and strict_source_ref) else current_frame
            ctx_frames = [metric_current_frame] + aux_frames
            geometry_ctx_frames = [geometry_current_frame] + aux_frames
            model_source_path = (
                frame_image_path(new_inputs[0])
                if index == 0
                else generated_image_path_for_step(case, local_step - 1)
            )
            if not model_source_path:
                model_source_path = frame_image_path(current_frame)
            steps.append({
                "id": f"{case_id}__step{step}",
                "parent_id": case_id,
                "step": step,
                "num_steps": original_num_steps,
                "strict_input_local_step": local_step if strict_original_step is not None else None,
                "original_benchmark_step": step,
                "instruction_type": case.get("instruction_type"),
                "subtype": case.get("subtype"),
                "dataset": dataset_name,
                "scene_id": scene_id,
                "context_paths": [p for p in (frame_image_path(x) for x in ctx_frames) if p],
                "context_frame_ids": [fid for fid in (frame_id_from_frame(x) for x in ctx_frames) if fid],
                "context_geometry_paths": [p for p in (frame_image_path(x) for x in geometry_ctx_frames) if p],
                "context_geometry_frame_ids": [fid for fid in (frame_id_from_frame(x) for x in geometry_ctx_frames) if fid],
                "stage1_current_frame_id": stage1_current_frame_id,
                "target_path": frame_image_path(target_frame),
                "target_frame_id": frame_id_from_frame(target_frame),
                "generated_path": generated_image_path_for_step(case, local_step),
                "model_source_path": model_source_path,
                "action_metadata": pose_metadata_for_step(case, step),
                "context_metadata": case.get("context_metadata"),
            })
        return steps

    sample = normalize_benchmark_case(case)
    sample.setdefault("generated_path", prediction_image_path(case))
    sample.setdefault("parent_id", case_id)
    sample.setdefault("step", 1)
    sample.setdefault("num_steps", 1)
    sample.setdefault("model_source_path", sample.get("context_paths", [""])[0] if sample.get("context_paths") else "")
    sample.setdefault("action_metadata", pose_metadata_for_step(case, 1))
    sample.setdefault("context_metadata", case.get("context_metadata"))
    return [sample]


def normalize_benchmark_case(case: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize old runner JSONL and v5 benchmark JSONL into one internal shape."""
    case_id = str(case.get("id") or case.get("sample_id") or "unknown")
    dataset_name = normalize_dataset_name(case.get("dataset"))
    scene_id = str(case.get("scene_id") or "")

    old_context = case.get("context")
    if isinstance(old_context, list) and old_context:
        ctx_paths = [frame_image_path(x) for x in old_context]
        tgt_path = frame_image_path(case.get("target"))
        return {
            "id": case_id,
            "dataset": dataset_name,
            "scene_id": scene_id,
            "context_paths": [p for p in ctx_paths if p],
            "context_frame_ids": [fid for fid in (frame_id_from_frame(x) for x in old_context) if fid],
            "stage1_current_frame_id": frame_id_from_frame(old_context[0]) if old_context else None,
            "target_path": tgt_path,
            "target_frame_id": frame_id_from_frame(case.get("target")),
            "generated_path": prediction_image_path(case),
            "context_metadata": case.get("context_metadata"),
        }

    new_inputs = case.get("input_images")
    new_targets = case.get("target_images")
    if isinstance(new_inputs, list) and new_inputs:
        target_frame = None
        if isinstance(new_targets, list) and new_targets:
            target_frame = new_targets[0]
        elif isinstance(case.get("target_image"), dict):
            target_frame = case["target_image"]
        return {
            "id": case_id,
            "dataset": dataset_name,
            "scene_id": scene_id,
            "context_paths": [p for p in (frame_image_path(x) for x in new_inputs) if p],
            "context_frame_ids": [fid for fid in (frame_id_from_frame(x) for x in new_inputs) if fid],
            "stage1_current_frame_id": frame_id_from_frame(new_inputs[0]),
            "target_path": frame_image_path(target_frame),
            "target_frame_id": frame_id_from_frame(target_frame),
            "generated_path": generated_image_path_for_step(case, 1),
            "context_metadata": case.get("context_metadata"),
        }

    protocol = str(case.get("protocol") or "")
    inp = case.get("input") if isinstance(case.get("input"), dict) else {}
    ctx_frames = []

    if protocol == "two_image_instruction":
        for key in ("image_1", "image_2"):
            if isinstance(inp.get(key), dict):
                ctx_frames.append(inp[key])
    elif protocol == "single_image_cycle":
        inp_forward = case.get("input_forward") if isinstance(case.get("input_forward"), dict) else {}
        if isinstance(inp_forward.get("image"), dict):
            ctx_frames.append(inp_forward["image"])
    else:
        if isinstance(inp.get("image"), dict):
            ctx_frames.append(inp["image"])

    if not ctx_frames and isinstance(inp, dict):
        for key in ("image", "image_1", "image_2"):
            if isinstance(inp.get(key), dict):
                ctx_frames.append(inp[key])

    target_frame = None
    viz_frames = case.get("visualization_frames")
    if isinstance(viz_frames, list):
        for frame in viz_frames:
            if not isinstance(frame, dict):
                continue
            role = str(frame.get("role") or "").lower()
            label = str(frame.get("label") or "").lower()
            if role in {"input", "context", "source"} or label in {"input", "context", "source"}:
                ctx_frames.append(frame)
            elif role in {"target", "output"} or label in {"target", "output"}:
                target_frame = frame

    if protocol == "single_image_3step_chain" and isinstance(case.get("targets"), list) and case["targets"]:
        target_frame = case["targets"][-1]
    elif protocol == "single_image_cycle" and isinstance(case.get("target_forward"), dict):
        target_frame = case["target_forward"]
    elif isinstance(case.get("target"), dict):
        target_frame = case["target"]
    elif isinstance(case.get("targets"), list) and case["targets"]:
        target_frame = case["targets"][-1]
    elif isinstance(case.get("target_forward"), dict):
        target_frame = case["target_forward"]

    return {
        "id": case_id,
        "dataset": dataset_name,
        "scene_id": scene_id,
        "context_paths": [p for p in (frame_image_path(x) for x in ctx_frames) if p],
        "context_frame_ids": [fid for fid in (frame_id_from_frame(x) for x in ctx_frames) if fid],
        "stage1_current_frame_id": frame_id_from_frame(ctx_frames[0]) if ctx_frames else None,
        "target_path": frame_image_path(target_frame),
        "target_frame_id": frame_id_from_frame(target_frame),
        "generated_path": prediction_image_path(case),
        "context_metadata": case.get("context_metadata"),
    }
def infer_dataset_and_scene_from_path(path):
    p = path.lower().replace("\\", "/").split('/')
    if 'hypersim' in p:
        scene_name = None
        cam_name = None
        for part in p:
            if part.startswith('ai_'):
                scene_name = part
            if part.startswith('scene_cam_') and part.endswith('_final_preview'):
                cam_name = part[len('scene_'):-len('_final_preview')]
            elif part.startswith('scene_cam_') and part.endswith('_geometry_hdf5'):
                cam_name = part[len('scene_'):-len('_geometry_hdf5')]
            elif part.startswith('cam_'):
                cam_name = part
        if scene_name:
            scene_id = f"{scene_name}/{cam_name}" if cam_name else scene_name
            return 'hypersim', scene_id
    if 'scannetpp' in p: return 'scannetpp', p[p.index('scannetpp')+1]
    if 'scannet' in p: return 'scannet', p[p.index('scannet')+1]
    if 'matterport3d' in p:
        return 'mp3d', p[p.index('matterport3d')+1]
    return 'scannet', p[1]

def _normalize_frame_id_token(value):
    token = str(value or "").strip()
    if not token:
        return None
    match = re.search(r"(\d+)", token)
    if match:
        return str(int(match.group(1)))
    return token


def get_frame_from_scene(dataset_name, scene_id, frame_path):
    dataset_name = normalize_dataset_name(dataset_name)
    cache_key = f"{dataset_name}_{scene_id}"
    if cache_key not in SCENE_CACHE:
        if dataset_name not in DATASET_REGISTRY:
            raise KeyError(f"unknown dataset: {dataset_name}")
        SCENE_CACHE[cache_key], _ = DATASET_REGISTRY[dataset_name](scene_id)
    if dataset_name == "hypersim":
        target_basename = os.path.splitext(os.path.basename(frame_path))[0]
    else:
        target_basename = os.path.basename(frame_path).split('.')[0]
    for item in SCENE_CACHE[cache_key]:
        if dataset_name == "hypersim":
            item_basename = os.path.splitext(os.path.basename(item.frame_name))[0]
        else:
            item_basename = os.path.basename(item.frame_name).split('.')[0]
        if target_basename == item_basename:
            return item
    # Strict-only Cycle rows carry a bare loader-local frame_idx for the
    # physical source geometry.  Match that index before considering digits in
    # a file name; otherwise e.g. frame_idx=23/file=00230.jpg may steal a
    # request for the true frame_idx=230.
    frame_path_str = str(frame_path or "")
    is_bare_id = (
        bool(frame_path_str)
        and not any(separator in frame_path_str for separator in ("/", "\\"))
        and "." not in frame_path_str
    )
    if is_bare_id:
        wanted = _normalize_frame_id_token(frame_path_str)
        for item in SCENE_CACHE[cache_key]:
            frame_idx = getattr(item, "frame_idx", None)
            if frame_idx is not None and _normalize_frame_id_token(frame_idx) == wanted:
                return item
        for item in SCENE_CACHE[cache_key]:
            frame_name = getattr(item, "frame_name", None)
            if not frame_name:
                continue
            base = os.path.basename(str(frame_name))
            stem = os.path.splitext(base)[0]
            if any(_normalize_frame_id_token(part) == wanted for part in (base, stem)):
                return item
    raise FileNotFoundError(f"physical frame not found: {frame_path}")


def get_context_frame_for_metric(dataset_name, scene_id, image_path, geometry_path=None):
    """Load actual RGB while borrowing pose/depth from a physical frame."""
    geometry_path = geometry_path or image_path
    image_override = None
    paths_differ = str(image_path or "") != str(geometry_path or "")
    if (
        paths_differ
        and image_path
        and (os.path.isabs(str(image_path)) or os.path.exists(str(image_path)))
    ):
        image_override = read_image_cv2_local(str(image_path))

    reference_item = None
    reference_error = None
    if geometry_path:
        try:
            reference_item = get_frame_from_scene(dataset_name, scene_id, str(geometry_path))
        except Exception as error:
            reference_error = error
    if reference_item is None and image_path and str(image_path) != str(geometry_path):
        try:
            reference_item = get_frame_from_scene(dataset_name, scene_id, str(image_path))
        except Exception:
            pass

    if reference_item is not None:
        item = copy.copy(reference_item)
        item.reference_frame_name = getattr(reference_item, "frame_name", None)
        item.geometry_frame_path = geometry_path
        if image_override is not None:
            ref_h, ref_w = reference_item.image.shape[:2]
            if image_override.shape[:2] != (ref_h, ref_w):
                image_override = cv2.resize(
                    image_override, (ref_w, ref_h), interpolation=cv2.INTER_AREA
                )
            item.image = image_override
            item.frame_name = str(image_path)
            item.generated_image_path = str(image_path)
        return item

    if image_override is None and image_path and (
        os.path.isabs(str(image_path)) or os.path.exists(str(image_path))
    ):
        image_override = read_image_cv2_local(str(image_path))
    if image_override is not None:
        return SimpleNamespace(
            frame_idx=None,
            frame_name=str(image_path),
            image=image_override,
            depth=None,
            extrinsics=None,
            intrinsics=None,
            generated_image_path=str(image_path),
            reference_frame_name=None,
            geometry_frame_path=geometry_path,
        )
    if reference_error is not None:
        raise reference_error
    return get_frame_from_scene(dataset_name, scene_id, str(image_path))


def summarize_metric(values):
    vals = [float(x) for x in values if x is not None and not (isinstance(x, float) and np.isnan(x))]
    if not vals:
        return {"mean": None, "median": None, "min": None, "max": None}
    return {
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
    }


def _as_4x4(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float64)
    if mat.shape[-2:] == (4, 4):
        return mat[:4, :4].copy()
    if mat.shape[-2:] == (3, 4):
        out = np.eye(4, dtype=np.float64)
        out[:3, :4] = mat[:3, :4]
        return out
    raise ValueError(f"Expected a 3x4 or 4x4 camera matrix, got {mat.shape}")


def rotation_angle_deg(R: np.ndarray) -> float:
    R = np.asarray(R, dtype=np.float64)[:3, :3]
    cos = (np.trace(R) - 1.0) / 2.0
    return float(math.degrees(math.acos(float(np.clip(cos, -1.0, 1.0)))))


def relative_motion_from_c2w(src_c2w: np.ndarray, tgt_c2w: np.ndarray) -> Dict[str, Any]:
    src_c2w = _as_4x4(src_c2w)
    tgt_c2w = _as_4x4(tgt_c2w)
    R0, C0 = src_c2w[:3, :3], src_c2w[:3, 3]
    R1, C1 = tgt_c2w[:3, :3], tgt_c2w[:3, 3]
    t_src = R0.T @ (C1 - C0)
    R_rel = R0.T @ R1
    f_t_in_src = R_rel[:, 2]
    yaw = math.degrees(math.atan2(float(f_t_in_src[0]), float(f_t_in_src[2])))
    pitch = math.degrees(
        math.atan2(
            float(f_t_in_src[1]),
            math.sqrt(float(f_t_in_src[0] ** 2 + f_t_in_src[2] ** 2)),
        )
    )
    return {
        "t_src": t_src.astype(np.float64),
        "R_rel": R_rel.astype(np.float64),
        "tx": float(t_src[0]),
        "ty": float(t_src[1]),
        "tz": float(t_src[2]),
        "translation_norm": float(np.linalg.norm(t_src)),
        "yaw": float(yaw),
        "pitch": float(pitch),
        "rotation_angle": rotation_angle_deg(R_rel),
    }


def angle_between_vectors_deg(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> Optional[float]:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < eps or nb < eps:
        return None
    cos = float(np.dot(a, b) / (na * nb))
    return float(math.degrees(math.acos(float(np.clip(cos, -1.0, 1.0)))))


def vector_cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> Optional[float]:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < eps or nb < eps:
        return None
    return float(np.dot(a, b) / (na * nb))


def signed_direction_label(value: Optional[float], eps: float) -> Optional[str]:
    if value is None:
        return None
    value = float(value)
    if abs(value) <= eps:
        return "zero"
    return "positive" if value > 0 else "negative"


def signed_direction_correct(pred_value: Optional[float], expected_value: Optional[float], unit: Optional[str]) -> Optional[float]:
    if pred_value is None or expected_value is None:
        return None
    eps = 1e-3 if unit == "m" else 1e-2
    expected_label = signed_direction_label(expected_value, eps)
    pred_label = signed_direction_label(pred_value, eps)
    if expected_label is None or pred_label is None or expected_label == "zero":
        return None
    return 1.0 if pred_label == expected_label else 0.0


def signed_action_component(motion: Dict[str, Any], action_meta: Dict[str, Any]) -> tuple[Optional[str], Optional[float], Optional[str]]:
    action = str(action_meta.get("dominant_action") or "").lower()
    subtype = str(action_meta.get("atomic_subtype") or action_meta.get("subtype") or "").lower()
    # The frozen instruction label is authoritative.  Both A3 yaw and A4
    # pitch subtypes contain the generic token "rotation", so consulting the
    # subtype first silently routes pitch instructions to yaw.  Subtype
    # inference is retained only for legacy records without dominant_action.
    if action in {"forward", "backward"}:
        return "tz_forward_m", float(motion.get("tz", 0.0)), "m"
    if action in {"left", "right", "strafe_left", "strafe_right", "lateral"}:
        return "tx_right_m", float(motion.get("tx", 0.0)), "m"
    if action in {"up", "down", "vertical_up", "vertical_down"}:
        return "ty_down_m", float(motion.get("ty", 0.0)), "m"
    if action in {"yaw_left", "yaw_right", "turn_left", "turn_right"}:
        return "yaw_deg", float(motion.get("yaw", 0.0)), "deg"
    if action in {"pitch_up", "pitch_down"}:
        return "pitch_deg", float(motion.get("pitch", 0.0)), "deg"
    if action:
        return None, None, None
    if "forward_backward" in subtype:
        return "tz_forward_m", float(motion.get("tz", 0.0)), "m"
    if "lateral" in subtype:
        return "tx_right_m", float(motion.get("tx", 0.0)), "m"
    if "vertical" in subtype:
        return "ty_down_m", float(motion.get("ty", 0.0)), "m"
    if "pitch" in subtype:
        return "pitch_deg", float(motion.get("pitch", 0.0)), "deg"
    if "yaw" in subtype or "rotation" in subtype:
        return "yaw_deg", float(motion.get("yaw", 0.0)), "deg"
    return None, None, None


TRANSLATION_ACTION_COMPONENTS = {"tx_right_m", "ty_down_m", "tz_forward_m"}
ROTATION_ACTION_COMPONENTS = {"yaw_deg", "pitch_deg"}


def pose_action_group(component: Optional[str], unit: Optional[str]) -> Optional[str]:
    if component in TRANSLATION_ACTION_COMPONENTS or unit == "m":
        return "translation"
    if component in ROTATION_ACTION_COMPONENTS or unit == "deg":
        return "rotation"
    return None


def row_pose_action_group(row: Dict[str, Any], prefix: str) -> Optional[str]:
    return pose_action_group(
        row.get(f"{prefix}_Pose_ActionComponent"),
        row.get(f"{prefix}_Pose_ActionUnit"),
    )


def row_pose_rotation_direction_error_deg(row: Dict[str, Any], prefix: str) -> Optional[float]:
    component = row.get(f"{prefix}_Pose_ActionComponent")
    if component == "yaw_deg":
        pred = row.get(f"{prefix}_Pose_YawPredDeg")
        expected = row.get(f"{prefix}_Pose_YawExpectedDeg")
    elif component == "pitch_deg":
        pred = row.get(f"{prefix}_Pose_PitchPredDeg")
        expected = row.get(f"{prefix}_Pose_PitchExpectedDeg")
    else:
        pred = row.get(f"{prefix}_Pose_ActionPred")
        expected = row.get(f"{prefix}_Pose_ActionExpected")

    if pred is None or expected is None:
        return None
    try:
        pred = float(pred)
        expected = float(expected)
    except (TypeError, ValueError):
        return None
    if not (np.isfinite(pred) and np.isfinite(expected)):
        return None
    return abs(pred - expected)


def summarize_pose_split(results: List[Dict[str, Any]], prefix: str) -> Dict[str, Any]:
    translation_rows = [r for r in results if row_pose_action_group(r, prefix) == "translation"]
    rotation_rows = [r for r in results if row_pose_action_group(r, prefix) == "rotation"]
    return {
        "translation": {
            "count": len(translation_rows),
            "direction_error_deg": summarize_metric([r.get(f"{prefix}_Pose_TransDirErrDeg") for r in translation_rows]),
            "direction_cos": summarize_metric([r.get(f"{prefix}_Pose_TransDirCos") for r in translation_rows]),
            "direction_correct_rate": summarize_metric([r.get(f"{prefix}_Pose_TransDirCorrect") for r in translation_rows]),
            "norm_error_m": summarize_metric([r.get(f"{prefix}_Pose_TransNormErrM") for r in translation_rows]),
            "action_abs_error_m": summarize_metric([r.get(f"{prefix}_Pose_ActionAbsErr") for r in translation_rows]),
            "action_sign_correct_rate": summarize_metric([r.get(f"{prefix}_Pose_ActionSignCorrect") for r in translation_rows]),
            "rotation_error_deg": summarize_metric([r.get(f"{prefix}_Pose_RotErrDeg") for r in translation_rows]),
        },
        "rotation": {
            "count": len(rotation_rows),
            "direction_error_deg": summarize_metric([row_pose_rotation_direction_error_deg(r, prefix) for r in rotation_rows]),
            "direction_correct_rate": summarize_metric([r.get(f"{prefix}_Pose_ActionSignCorrect") for r in rotation_rows]),
            "action_abs_error_deg": summarize_metric([r.get(f"{prefix}_Pose_ActionAbsErr") for r in rotation_rows]),
            "action_sign_correct_rate": summarize_metric([r.get(f"{prefix}_Pose_ActionSignCorrect") for r in rotation_rows]),
            "rotation_error_deg": summarize_metric([r.get(f"{prefix}_Pose_RotErrDeg") for r in rotation_rows]),
            "translation_direction_error_deg": summarize_metric([r.get(f"{prefix}_Pose_TransDirErrDeg") for r in rotation_rows]),
        },
    }


def compare_pose_to_expected(
    predicted: Optional[Dict[str, Any]],
    expected: Dict[str, Any],
    action_meta: Dict[str, Any],
    metric_scale: Optional[float] = None,
) -> Dict[str, Any]:
    out = {
        "Pose_backend": None,
        "Pose_skip_reason": None,
        "Pose_RotErrDeg": None,
        "Pose_TransDirErrDeg": None,
        "Pose_TransDirCos": None,
        "Pose_TransDirCorrect": None,
        "Pose_TransNormPredRaw": None,
        "Pose_TransNormPredMetric": None,
        "Pose_TransNormExpected": float(expected.get("translation_norm", 0.0)),
        "Pose_TransNormErrM": None,
        "Pose_TxPredMetric": None,
        "Pose_TyPredMetric": None,
        "Pose_TzPredMetric": None,
        "Pose_TxExpected": float(expected.get("tx", 0.0)),
        "Pose_TyExpected": float(expected.get("ty", 0.0)),
        "Pose_TzExpected": float(expected.get("tz", 0.0)),
        "Pose_YawPredDeg": None,
        "Pose_PitchPredDeg": None,
        "Pose_YawExpectedDeg": float(expected.get("yaw", 0.0)),
        "Pose_PitchExpectedDeg": float(expected.get("pitch", 0.0)),
        "Pose_ActionComponent": None,
        "Pose_ActionPred": None,
        "Pose_ActionExpected": None,
        "Pose_ActionSignedErr": None,
        "Pose_ActionAbsErr": None,
        "Pose_ActionSignCorrect": None,
        "Pose_ActionPredDirection": None,
        "Pose_ActionExpectedDirection": None,
        "Pose_ActionUnit": None,
    }
    if predicted is None:
        out["Pose_skip_reason"] = "pose prediction unavailable"
        return out

    out["Pose_backend"] = predicted.get("backend")
    t_pred_raw = np.asarray(predicted["t_src"], dtype=np.float64)
    t_expected = np.asarray(expected["t_src"], dtype=np.float64)
    R_pred = np.asarray(predicted["R_rel"], dtype=np.float64)
    R_expected = np.asarray(expected["R_rel"], dtype=np.float64)

    out["Pose_RotErrDeg"] = rotation_angle_deg(R_expected.T @ R_pred)
    out["Pose_TransDirErrDeg"] = angle_between_vectors_deg(t_pred_raw, t_expected)
    out["Pose_TransDirCos"] = vector_cosine_similarity(t_pred_raw, t_expected)
    if out["Pose_TransDirCos"] is not None:
        out["Pose_TransDirCorrect"] = 1.0 if out["Pose_TransDirCos"] > 0.0 else 0.0
    out["Pose_TransNormPredRaw"] = float(np.linalg.norm(t_pred_raw))
    out["Pose_YawPredDeg"] = float(predicted.get("yaw", 0.0))
    out["Pose_PitchPredDeg"] = float(predicted.get("pitch", 0.0))

    motion_for_action = dict(predicted)
    if metric_scale is not None and np.isfinite(metric_scale):
        t_metric = t_pred_raw * float(metric_scale)
        out["Pose_TransNormPredMetric"] = float(np.linalg.norm(t_metric))
        out["Pose_TransNormErrM"] = abs(out["Pose_TransNormPredMetric"] - out["Pose_TransNormExpected"])
        out["Pose_TxPredMetric"] = float(t_metric[0])
        out["Pose_TyPredMetric"] = float(t_metric[1])
        out["Pose_TzPredMetric"] = float(t_metric[2])
        motion_for_action["tx"] = float(t_metric[0])
        motion_for_action["ty"] = float(t_metric[1])
        motion_for_action["tz"] = float(t_metric[2])
        motion_for_action["translation_norm"] = float(np.linalg.norm(t_metric))

    comp, pred_value, unit = signed_action_component(motion_for_action, action_meta)
    _, expected_value, _ = signed_action_component(expected, action_meta)
    out["Pose_ActionComponent"] = comp
    out["Pose_ActionPred"] = pred_value
    out["Pose_ActionExpected"] = expected_value
    out["Pose_ActionUnit"] = unit
    if pred_value is not None and expected_value is not None:
        eps = 1e-3 if unit == "m" else 1e-2
        out["Pose_ActionSignedErr"] = float(pred_value) - float(expected_value)
        out["Pose_ActionAbsErr"] = abs(out["Pose_ActionSignedErr"])
        out["Pose_ActionSignCorrect"] = signed_direction_correct(pred_value, expected_value, unit)
        out["Pose_ActionPredDirection"] = signed_direction_label(pred_value, eps)
        out["Pose_ActionExpectedDirection"] = signed_direction_label(expected_value, eps)
    return out


POSE_METRIC_SUFFIXES = [
    "Pose_backend",
    "Pose_skip_reason",
    "Pose_RotErrDeg",
    "Pose_TransDirErrDeg",
    "Pose_TransDirCos",
    "Pose_TransDirCorrect",
    "Pose_TransNormPredRaw",
    "Pose_TransNormPredMetric",
    "Pose_TransNormExpected",
    "Pose_TransNormErrM",
    "Pose_TxPredMetric",
    "Pose_TyPredMetric",
    "Pose_TzPredMetric",
    "Pose_TxExpected",
    "Pose_TyExpected",
    "Pose_TzExpected",
    "Pose_YawPredDeg",
    "Pose_PitchPredDeg",
    "Pose_YawExpectedDeg",
    "Pose_PitchExpectedDeg",
    "Pose_ActionComponent",
    "Pose_ActionPred",
    "Pose_ActionExpected",
    "Pose_ActionSignedErr",
    "Pose_ActionAbsErr",
    "Pose_ActionSignCorrect",
    "Pose_ActionPredDirection",
    "Pose_ActionExpectedDirection",
    "Pose_ActionUnit",
]


def resolve_hf_local_model_dir(model_dir: Optional[str]) -> Optional[str]:
    if not model_dir:
        return None
    model_dir = os.path.abspath(os.path.expanduser(str(model_dir)))
    if not os.path.isdir(model_dir):
        return None
    if os.path.exists(os.path.join(model_dir, "config.json")):
        return model_dir

    snapshots_dir = os.path.join(model_dir, "snapshots")
    if not os.path.isdir(snapshots_dir):
        return model_dir

    ref_path = os.path.join(model_dir, "refs", "main")
    if os.path.exists(ref_path):
        with open(ref_path, "r", encoding="utf-8") as f:
            ref = f.read().strip()
        ref_dir = os.path.join(snapshots_dir, ref)
        if os.path.isdir(ref_dir):
            return ref_dir

    snapshots = [
        os.path.join(snapshots_dir, name)
        for name in os.listdir(snapshots_dir)
        if os.path.isdir(os.path.join(snapshots_dir, name))
    ]
    if not snapshots:
        return model_dir
    snapshots.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return snapshots[0]


def add_da3_code_path(code_dir: Optional[str] = None) -> List[str]:
    """Make a local Depth Anything 3 checkout importable without requiring pip install -e."""
    candidates = []
    for item in (
        code_dir,
        os.environ.get("DA3_CODE_DIR"),
        DEFAULT_DA3_CODE_DIR,
    ):
        if item and item not in candidates:
            candidates.append(item)

    added = []
    for path in candidates:
        path = os.path.abspath(os.path.expanduser(str(path)))
        src_path = os.path.join(path, "src")
        if os.path.isdir(os.path.join(src_path, "depth_anything_3")):
            import_path = src_path
        elif os.path.isdir(os.path.join(path, "depth_anything_3")):
            import_path = path
        else:
            continue
        if import_path not in sys.path:
            sys.path.insert(0, import_path)
            added.append(import_path)
    return added


class DA3RelativePoseEstimator:
    """Lazy Depth Anything 3 relative pose wrapper.

    DA3 returns world-to-camera extrinsics. The metric code converts them to
    camera-to-world and reports target camera motion in the source camera
    frame, matching this benchmark's pose metadata convention.
    """

    def __init__(
        self,
        device,
        model_dir: Optional[str] = None,
        code_dir: Optional[str] = None,
        use_ray_pose: bool = False,
        process_res: int = 504,
        process_res_method: str = "upper_bound_resize",
        ref_view_strategy: str = "saddle_balanced",
        aspect_policy: str = "native",
    ):
        self.device = device
        self.code_dir = code_dir
        self.use_ray_pose = bool(use_ray_pose)
        self.process_res = int(process_res)
        self.process_res_method = process_res_method
        self.ref_view_strategy = ref_view_strategy
        if aspect_policy not in DA3_ASPECT_POLICIES:
            raise ValueError(
                f"Unsupported DA3 aspect policy: {aspect_policy}. "
                f"Supported policies: {', '.join(DA3_ASPECT_POLICIES)}"
            )
        self.aspect_policy = aspect_policy
        local_da3 = resolve_hf_local_model_dir(
            model_dir or os.environ.get("DA3_MODEL_DIR") or DEFAULT_DA3_MODEL_DIR
        )
        self.model_candidates = []
        for item in (
            local_da3,
            os.environ.get("DA3_MODEL_ID"),
            "depth-anything/da3nested-giant-large",
        ):
            if item and item not in self.model_candidates:
                self.model_candidates.append(item)
        self.model = None
        self.model_id = None
        self.load_error = None
        self._cache: Dict[tuple, Optional[Dict[str, Any]]] = {}

    @staticmethod
    def _transform_input(
        image_path: str,
        target_aspect_ratio: float,
        transform: str,
    ):
        """Return a DA3-compatible input plus an auditable crop/resize record."""
        with Image.open(image_path) as pil_image:
            width, height = pil_image.size
            source_aspect = float(width) / float(height)
            metadata = {
                "path": str(image_path),
                "source_width": int(width),
                "source_height": int(height),
                "source_aspect_ratio": source_aspect,
                "target_aspect_ratio": float(target_aspect_ratio),
                "transform": transform,
            }

            if transform == "center_crop":
                if abs(source_aspect - target_aspect_ratio) <= 1e-8:
                    metadata.update({
                        "applied": False,
                        "output_width": int(width),
                        "output_height": int(height),
                        "crop_box_xyxy": [0, 0, int(width), int(height)],
                    })
                    return image_path, metadata
                if source_aspect > target_aspect_ratio:
                    output_width = max(1, min(width, int(round(height * target_aspect_ratio))))
                    left = (width - output_width) // 2
                    box = (left, 0, left + output_width, height)
                else:
                    output_height = max(1, min(height, int(round(width / target_aspect_ratio))))
                    top = (height - output_height) // 2
                    box = (0, top, width, top + output_height)
                transformed = pil_image.convert("RGB").crop(box).copy()
                metadata.update({
                    "applied": True,
                    "output_width": int(transformed.width),
                    "output_height": int(transformed.height),
                    "crop_box_xyxy": [int(x) for x in box],
                })
                return transformed, metadata

            if transform == "stretch":
                side = max(1, min(width, height))
                resampling = getattr(Image, "Resampling", Image).BICUBIC
                transformed = pil_image.convert("RGB").resize((side, side), resample=resampling).copy()
                metadata.update({
                    "applied": width != height,
                    "output_width": int(side),
                    "output_height": int(side),
                    "interpolation": "PIL.Image.Resampling.BICUBIC",
                })
                return transformed, metadata

        raise ValueError(f"Unsupported DA3 input transform: {transform}")

    def _prepare_inputs(
        self,
        image_path_0: str,
        image_path_1: str,
        target_aspect_ratio: Optional[float],
        candidate_label: Optional[str],
    ):
        if self.aspect_policy == "native":
            return [image_path_0, image_path_1], []

        if self.aspect_policy == "gt_center_crop":
            if target_aspect_ratio is None or not np.isfinite(target_aspect_ratio) or target_aspect_ratio <= 0:
                raise ValueError("gt_center_crop requires a positive finite GT target aspect ratio")
            prepared = []
            metadata = []
            for path in (image_path_0, image_path_1):
                transformed, record = self._transform_input(
                    path, float(target_aspect_ratio), transform="center_crop"
                )
                prepared.append(transformed)
                metadata.append(record)
            return prepared, metadata

        # Synthetic policies leave GTImage unchanged. For Pred, the source is the
        # unmodified GT context and only the GT target is converted to a square.
        if candidate_label != "Pred":
            return [image_path_0, image_path_1], []
        transform = "center_crop" if self.aspect_policy.endswith("center_crop") else "stretch"
        transformed, record = self._transform_input(image_path_1, 1.0, transform=transform)
        return [image_path_0, transformed], [
            {
                "path": str(image_path_0),
                "transform": "none",
                "applied": False,
            },
            record,
        ]

    def _ensure_loaded(self) -> bool:
        if self.model is not None:
            return True
        added_paths = add_da3_code_path(self.code_dir)
        try:
            try:
                __import__("pycolmap")
            except Exception:
                # DA3 imports its COLMAP exporter at API import time, but pose inference does
                # not use COLMAP export. A tiny stub keeps DA3 importable in bagel envs where
                # pycolmap's C++ extension is absent or broken.
                import types
                sys.modules.pop("pycolmap", None)
                sys.modules["pycolmap"] = types.ModuleType("pycolmap")
            from depth_anything_3.api import DepthAnything3
        except Exception as e:
            hint = f" searched_code_paths={added_paths or [self.code_dir, os.environ.get('DA3_CODE_DIR'), DEFAULT_DA3_CODE_DIR]}"
            self.load_error = f"DA3 import failed: {e}.{hint}"
            return False

        errors = []
        for model_id in self.model_candidates:
            try:
                self.model = DepthAnything3.from_pretrained(model_id).eval().to(self.device)
                self.model_id = model_id
                return True
            except Exception as e:
                errors.append(f"{model_id}: {e}")
        self.load_error = " | ".join(errors) if errors else "no DA3 model candidates"
        return False

    def predict(
        self,
        image_path_0: str,
        image_path_1: str,
        target_aspect_ratio: Optional[float] = None,
        candidate_label: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        ratio_key = None if target_aspect_ratio is None else round(float(target_aspect_ratio), 8)
        key = (
            str(image_path_0),
            str(image_path_1),
            self.aspect_policy,
            ratio_key,
            candidate_label,
        )
        if key in self._cache:
            return self._cache[key]
        if not image_path_0 or not image_path_1:
            self._cache[key] = None
            return None
        if not (os.path.exists(image_path_0) and os.path.exists(image_path_1)):
            self._cache[key] = None
            return None
        if not self._ensure_loaded():
            self._cache[key] = None
            return None

        try:
            inference_inputs, transform_metadata = self._prepare_inputs(
                image_path_0,
                image_path_1,
                target_aspect_ratio=target_aspect_ratio,
                candidate_label=candidate_label,
            )
            with torch.no_grad():
                prediction = self.model.inference(
                    inference_inputs,
                    export_dir=None,
                    export_format="mini_npz",
                    process_res=self.process_res,
                    process_res_method=self.process_res_method,
                    use_ray_pose=self.use_ray_pose,
                    ref_view_strategy=self.ref_view_strategy,
                )

            extrinsic = getattr(prediction, "extrinsics", None)
            if extrinsic is None or len(extrinsic) < 2:
                self.load_error = "DA3 pose prediction returned no extrinsics"
                self._cache[key] = None
                return None
            extrinsic = np.asarray(extrinsic, dtype=np.float64)
            c2w_0 = np.linalg.inv(_as_4x4(extrinsic[0]))
            c2w_1 = np.linalg.inv(_as_4x4(extrinsic[1]))
            motion = relative_motion_from_c2w(c2w_0, c2w_1)
            backend = self.model_id or "DA3"
            if self.use_ray_pose:
                backend = f"{backend} [ray_pose]"
            motion["backend"] = backend
            motion["aspect_policy"] = self.aspect_policy
            motion["target_aspect_ratio"] = target_aspect_ratio
            motion["input_transforms"] = transform_metadata
            self._cache[key] = motion
            return motion
        except Exception as e:
            self.load_error = f"DA3 pose prediction failed: {e}"
            self._cache[key] = None
            return None
class CmgRunner:
    def __init__(
        self,
        oracle_mode=False,
        save_vis=False,
        device=None,
        pose_backends=None,
        da3_model_dir: Optional[str] = None,
        da3_code_dir: Optional[str] = None,
        da3_use_ray_pose: bool = False,
        da3_process_res: int = 504,
        da3_process_res_method: str = "upper_bound_resize",
        da3_ref_view_strategy: str = "saddle_balanced",
        da3_aspect_policy: str = "native",
        compute_photometric: bool = True,
        max_cases: Optional[int] = None,
    ):
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.oracle_mode = oracle_mode
        self.save_vis = save_vis
        self.compute_photometric = bool(compute_photometric)
        self.max_cases = int(max_cases) if max_cases is not None else None
        self.evaluator = None  # photometric disabled (--pose-only)
        self.pose_backend_names = parse_pose_backends(pose_backends)
        self.primary_pose_backend = self.pose_backend_names[0]
        self.pose_estimators = OrderedDict()
        for backend_name in self.pose_backend_names:
            if backend_name == "da3":
                self.pose_estimators[backend_name] = DA3RelativePoseEstimator(
                    device=self.device,
                    model_dir=da3_model_dir,
                    code_dir=da3_code_dir,
                    use_ray_pose=da3_use_ray_pose,
                    process_res=da3_process_res,
                    process_res_method=da3_process_res_method,
                    ref_view_strategy=da3_ref_view_strategy,
                    aspect_policy=da3_aspect_policy,
                )
        self.pose_estimator = self.pose_estimators[self.primary_pose_backend]
        


    @staticmethod
    def _float_or_none(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def select_pose_context(self, sample, ctx_items):
        ctx_frame_ids = [str(x) for x in sample.get("context_frame_ids", [])]
        target_frame_id = sample.get("target_frame_id")
        if target_frame_id is not None:
            target_frame_id = str(target_frame_id)
        step = int(sample.get("step", 1) or 1)
        num_steps = int(sample.get("num_steps", 1) or 1)
        stage1_current_frame_id = sample.get("stage1_current_frame_id")
        if stage1_current_frame_id is not None:
            stage1_current_frame_id = str(stage1_current_frame_id)
        current_matches_stage1 = bool(ctx_frame_ids) and (
            stage1_current_frame_id is None or ctx_frame_ids[0] == stage1_current_frame_id
        )
        meta = sample.get("context_metadata") if isinstance(sample.get("context_metadata"), dict) else {}

        fallback_frame_id = ctx_frame_ids[0] if ctx_frame_ids else str(getattr(ctx_items[0], "frame_idx", 0))
        candidates = [{
            "index": 0,
            "overlap": 0.0,
            "frame_id": fallback_frame_id,
            "source": "fallback_ctx0_missing_stage1_overlap",
        }]

        target_current = meta.get("target_current_scores") if isinstance(meta, dict) else None
        if isinstance(target_current, dict) and ctx_frame_ids and current_matches_stage1:
            per_target = target_current.get("per_target")
            if isinstance(per_target, list):
                for target_order, item in enumerate(per_target, start=1):
                    if not isinstance(item, dict):
                        continue
                    target_idx = item.get("target_idx")
                    target_matches = (
                        len(per_target) == 1
                        or target_order == step
                        or (
                            target_frame_id is not None
                            and target_idx is not None
                            and str(target_idx) == target_frame_id
                        )
                    )
                    if not target_matches:
                        continue
                    overlap = self._float_or_none(item.get("target_to_current_overlap"))
                    if overlap is None:
                        continue
                    candidates.append({
                        "index": 0,
                        "overlap": overlap,
                        "frame_id": ctx_frame_ids[0],
                        "source": "stage1_target_current_scores.target_to_current_overlap",
                    })

        aux_scores = meta.get("auxiliary_context_scores") if isinstance(meta, dict) else None
        if isinstance(aux_scores, list):
            aux_by_frame_id = {
                str(item.get("frame_id")): item
                for item in aux_scores
                if isinstance(item, dict) and item.get("frame_id") is not None
            }
            for ctx_index, frame_id in enumerate(ctx_frame_ids[1:], start=1):
                item = aux_by_frame_id.get(str(frame_id))
                if not item:
                    continue
                best_target_idx = item.get("best_target_idx")
                target_matches = num_steps == 1 or (
                    target_frame_id is not None
                    and best_target_idx is not None
                    and str(best_target_idx) == target_frame_id
                )
                if not target_matches:
                    continue
                overlap = self._float_or_none(item.get("target_to_aux_overlap"))
                if overlap is None:
                    continue
                candidates.append({
                    "index": ctx_index,
                    "overlap": overlap,
                    "frame_id": str(frame_id),
                    "source": "stage1_auxiliary_context_scores.target_to_aux_overlap",
                })

        best = max(candidates, key=lambda x: (x["overlap"], -x["index"]))
        if best["index"] >= len(ctx_items):
            best = candidates[0]
        return best

    def evaluate_step_sample(self, sample):
        case_id = sample["id"]
        file_case_id = safe_filename_token(case_id)
        ctx_paths = sample["context_paths"]
        tgt_gt_path = sample["target_path"]
        tgt_gen_path = sample["generated_path"]
        if not ctx_paths or not tgt_gt_path:
            return None

        dataset_name = sample.get("dataset")
        scene_id = sample.get("scene_id")
        if not dataset_name or not scene_id:
            dataset_name, scene_id = infer_dataset_and_scene_from_path(ctx_paths[0])
        ctx_geometry_paths = sample.get("context_geometry_paths") or ctx_paths
        ctx_items = [
            get_context_frame_for_metric(
                dataset_name,
                scene_id,
                path,
                ctx_geometry_paths[index] if index < len(ctx_geometry_paths) else path,
            )
            for index, path in enumerate(ctx_paths)
        ]
        tgt_item = get_frame_from_scene(dataset_name, scene_id, tgt_gt_path)

        has_generated_image = bool(tgt_gen_path) or bool(self.oracle_mode)
        gen_img_cv2 = None
        pred_skip_reason = None
        if self.oracle_mode:
            gen_img_cv2 = tgt_item.image.copy()
        elif not tgt_gen_path:
            pred_skip_reason = "missing generated_images step image_path"
        elif os.path.isabs(tgt_gen_path) or os.path.exists(tgt_gen_path):
            gen_img_cv2 = read_image_cv2_local(tgt_gen_path)
        else:
            try:
                pred_item = get_frame_from_scene(dataset_name, scene_id, tgt_gen_path)
                gen_img_cv2 = pred_item.image.copy()
            except Exception:
                gen_img_cv2 = read_image_cv2_local(tgt_gen_path)
        if gen_img_cv2 is None and pred_skip_reason is None:
            pred_skip_reason = f"failed to read generated image: {tgt_gen_path}"
            
        H_gt, W_gt = tgt_item.image.shape[:2]
        if gen_img_cv2 is not None and gen_img_cv2.shape[:2] != (H_gt, W_gt):
            gen_img_cv2 = cv2.resize(gen_img_cv2, (W_gt, H_gt))

        selected_ctx = self.select_pose_context(sample, ctx_items)

        scores = {
            "id": case_id,
            "parent_id": sample.get("parent_id"),
            "step": sample.get("step"),
            "num_steps": sample.get("num_steps"),
            "strict_input_local_step": sample.get("strict_input_local_step"),
            "original_benchmark_step": sample.get("original_benchmark_step"),
            "dataset": dataset_name,
            "scene_id": scene_id,
            "target_frame_id": sample.get("target_frame_id"),
            "target_path": tgt_gt_path,
            "generated_path": tgt_gen_path,
            "gt_pose_source_path": ctx_paths[0] if ctx_paths else "",
            "pred_pose_source_path": sample.get("model_source_path") or (ctx_paths[0] if ctx_paths else ""),
            "action_metadata": sample.get("action_metadata") or {},
            "has_generated_image": bool(has_generated_image),
            "Pose_backend": None,
            "Pose_skip_reason": None,
            "Pose_RotErrDeg": None,
            "Pose_TransDirErrDeg": None,
            "Pose_TransDirCos": None,
            "Pose_TransDirCorrect": None,
            "Pose_TransNormPredRaw": None,
            "Pose_TransNormPredMetric": None,
            "Pose_TransNormExpected": None,
            "Pose_TransNormErrM": None,
            "Pose_ActionComponent": None,
            "Pose_ActionPred": None,
            "Pose_ActionExpected": None,
            "Pose_ActionSignedErr": None,
            "Pose_ActionAbsErr": None,
            "Pose_ActionSignCorrect": None,
            "Pose_ActionPredDirection": None,
            "Pose_ActionExpectedDirection": None,
            "Pose_ActionUnit": None,
            "Pose_GTScaleCalibration": None,
            "Pose_DiagnosticPath": None,
            "Pose_DiagnosticRelPath": None,
            "Overview_DiagnosticPath": None,
            "Overview_DiagnosticRelPath": None,
            "SelectedContextIndex": selected_ctx["index"],
            "SelectedContextOverlap": selected_ctx["overlap"],
            "SelectedContextFrameId": selected_ctx["frame_id"],
            "ContextSelectionSource": selected_ctx["source"],
            "Hybrid_Score": None,
            "PSNR": None,
            "LPIPS": None,
            "LPIPS_EvalOverlap": None,
            "LPIPS_MaskMode": None,
            "SourceWarp_PSNR": None,
            "SourceWarp_LPIPS": None,
            "SourceWarp_Hybrid_Score": None,
            "SourceWarp_LPIPS_EvalOverlap": None,
            "SourceWarp_LPIPS_MaskMode": None,
            "WarpGT_PSNR": None,
            "WarpGT_LPIPS": None,
            "WarpGT_Hybrid_Score": None,
            "WarpGT_LPIPS_EvalOverlap": None,
            "WarpGT_LPIPS_MaskMode": None,
            "Overlap": None,
            "GTImage_SelectedContextIndex": selected_ctx["index"],
            "GTImage_SelectedContextOverlap": selected_ctx["overlap"],
            "GTImage_SelectedContextFrameId": selected_ctx["frame_id"],
            "GTImage_ContextSelectionSource": selected_ctx["source"],
            "GTImage_Hybrid_Score": None,
            "GTImage_PSNR": None,
            "GTImage_LPIPS": None,
            "GTImage_LPIPS_EvalOverlap": None,
            "GTImage_LPIPS_MaskMode": None,
            "GTImage_SourceWarp_PSNR": None,
            "GTImage_SourceWarp_LPIPS": None,
            "GTImage_SourceWarp_Hybrid_Score": None,
            "GTImage_SourceWarp_LPIPS_EvalOverlap": None,
            "GTImage_SourceWarp_LPIPS_MaskMode": None,
            "GTImage_WarpGT_PSNR": None,
            "GTImage_WarpGT_LPIPS": None,
            "GTImage_WarpGT_Hybrid_Score": None,
            "GTImage_WarpGT_LPIPS_EvalOverlap": None,
            "GTImage_WarpGT_LPIPS_MaskMode": None,
            "GTImage_Overlap": None,
            "GTImage_VisibleOverlapRaw": None,
            "GTImage_Pose_backend": None,
            "GTImage_Pose_skip_reason": None,
            "GTImage_Pose_RotErrDeg": None,
            "GTImage_Pose_TransDirErrDeg": None,
            "GTImage_Pose_TransDirCos": None,
            "GTImage_Pose_TransDirCorrect": None,
            "GTImage_Pose_TransNormPredRaw": None,
            "GTImage_Pose_TransNormPredMetric": None,
            "GTImage_Pose_TransNormExpected": None,
            "GTImage_Pose_TransNormErrM": None,
            "GTImage_Pose_ActionSignedErr": None,
            "GTImage_Pose_ActionAbsErr": None,
            "GTImage_Pose_ActionSignCorrect": None,
            "GTImage_Pose_ActionPredDirection": None,
            "GTImage_Pose_ActionExpectedDirection": None,
            "Pred_SelectedContextIndex": selected_ctx["index"],
            "Pred_SelectedContextOverlap": selected_ctx["overlap"],
            "Pred_SelectedContextFrameId": selected_ctx["frame_id"],
            "Pred_ContextSelectionSource": selected_ctx["source"],
            "Pred_Hybrid_Score": None,
            "Pred_PSNR": None,
            "Pred_LPIPS": None,
            "Pred_LPIPS_EvalOverlap": None,
            "Pred_LPIPS_MaskMode": None,
            "Pred_SourceWarp_PSNR": None,
            "Pred_SourceWarp_LPIPS": None,
            "Pred_SourceWarp_Hybrid_Score": None,
            "Pred_SourceWarp_LPIPS_EvalOverlap": None,
            "Pred_SourceWarp_LPIPS_MaskMode": None,
            "Pred_WarpGT_PSNR": None,
            "Pred_WarpGT_LPIPS": None,
            "Pred_WarpGT_Hybrid_Score": None,
            "Pred_WarpGT_LPIPS_EvalOverlap": None,
            "Pred_WarpGT_LPIPS_MaskMode": None,
            "Pred_Overlap": None,
            "Pred_VisibleOverlapRaw": None,
            "Pred_Pose_backend": None,
            "Pred_Pose_skip_reason": None,
            "Pred_Pose_RotErrDeg": None,
            "Pred_Pose_TransDirErrDeg": None,
            "Pred_Pose_TransDirCos": None,
            "Pred_Pose_TransDirCorrect": None,
            "Pred_Pose_TransNormPredRaw": None,
            "Pred_Pose_TransNormPredMetric": None,
            "Pred_Pose_TransNormExpected": None,
            "Pred_Pose_TransNormErrM": None,
            "Pred_Pose_ActionSignedErr": None,
            "Pred_Pose_ActionAbsErr": None,
            "Pred_Pose_ActionSignCorrect": None,
            "Pred_Pose_ActionPredDirection": None,
            "Pred_Pose_ActionExpectedDirection": None,
        }
        if "da3" in self.pose_estimators:
            scores["DA3_Pose_GTScaleCalibration"] = None
            for candidate_label in ("GTImage", "Pred"):
                for suffix in POSE_METRIC_SUFFIXES:
                    scores[f"{candidate_label}_DA3_{suffix}"] = None
        if pred_skip_reason:
            scores["Pred_skip_reason"] = pred_skip_reason

        def to_tensor(img):
            arr = np.asarray(img)
            if arr.ndim == 2:
                arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
            elif arr.ndim == 3 and arr.shape[2] > 3:
                arr = arr[..., :3]
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            arr = np.ascontiguousarray(arr)
            return (
                torch.from_numpy(arr)
                .float()
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(self.device) / 127.5 - 1.0
            )

        gt_pose_source_path = ctx_paths[0] if ctx_paths else ""
        pred_pose_source_path = sample.get("model_source_path") or gt_pose_source_path
        if not pred_pose_source_path or not os.path.exists(pred_pose_source_path):
            pred_pose_source_path = gt_pose_source_path
        scores["gt_pose_source_path"] = gt_pose_source_path
        scores["pred_pose_source_path"] = pred_pose_source_path
        action_meta = sample.get("action_metadata") if isinstance(sample.get("action_metadata"), dict) else {}
        expected_motion = relative_motion_from_c2w(ctx_items[0].extrinsics, tgt_item.extrinsics)
        pose_scale_calibrations = {name: None for name in self.pose_estimators}
        gt_img_t = to_tensor(tgt_item.image.copy()) if self.compute_photometric else None

        def add_prefixed_pose_scores(out_scores, candidate_label, pose_metrics, namespace=None):
            prefix = f"{candidate_label}_{namespace}_" if namespace else f"{candidate_label}_"
            for key, value in pose_metrics.items():
                out_scores[f"{prefix}{key}"] = value

        def compute_candidate_pose_metrics(candidate_label, candidate_path, backend_name):
            estimator = self.pose_estimators[backend_name]
            source_path = gt_pose_source_path if candidate_label == "GTImage" or self.oracle_mode else pred_pose_source_path
            if backend_name == "da3":
                predicted_motion = estimator.predict(
                    source_path,
                    candidate_path,
                    target_aspect_ratio=float(W_gt) / float(H_gt),
                    candidate_label=candidate_label,
                )
            else:
                predicted_motion = estimator.predict(source_path, candidate_path)
            metric_scale = pose_scale_calibrations.get(backend_name)
            if candidate_label == "GTImage" and predicted_motion is not None:
                raw_norm = float(np.linalg.norm(predicted_motion["t_src"]))
                expected_norm = float(expected_motion.get("translation_norm", 0.0))
                if raw_norm > 1e-8 and expected_norm > 1e-8:
                    metric_scale = expected_norm / raw_norm
                    pose_scale_calibrations[backend_name] = metric_scale
            pose_metrics = compare_pose_to_expected(
                predicted_motion,
                expected_motion,
                action_meta,
                metric_scale=metric_scale,
            )
            if predicted_motion is None and estimator.load_error:
                pose_metrics["Pose_skip_reason"] = estimator.load_error
            if predicted_motion is not None and backend_name == "da3":
                pose_metrics["Pose_AspectPolicy"] = predicted_motion.get("aspect_policy")
                pose_metrics["Pose_TargetAspectRatio"] = predicted_motion.get("target_aspect_ratio")
                pose_metrics["Pose_InputTransforms"] = predicted_motion.get("input_transforms")
            return pose_metrics

        def evaluate_candidate_image(candidate_img_cv2, candidate_label, candidate_path):
            if candidate_img_cv2.shape[:2] != (H_gt, W_gt):
                candidate_img_cv2 = cv2.resize(candidate_img_cv2, (W_gt, H_gt))

            candidate_scores = {
                f"{candidate_label}_SelectedContextIndex": selected_ctx["index"],
                f"{candidate_label}_SelectedContextOverlap": selected_ctx["overlap"],
                f"{candidate_label}_SelectedContextFrameId": selected_ctx["frame_id"],
                f"{candidate_label}_ContextSelectionSource": selected_ctx["source"],
                f"{candidate_label}_Hybrid_Score": None,
                f"{candidate_label}_PSNR": None,
                f"{candidate_label}_LPIPS": None,
                f"{candidate_label}_LPIPS_EvalOverlap": None,
                f"{candidate_label}_LPIPS_MaskMode": None,
                f"{candidate_label}_SourceWarp_PSNR": None,
                f"{candidate_label}_SourceWarp_LPIPS": None,
                f"{candidate_label}_SourceWarp_Hybrid_Score": None,
                f"{candidate_label}_SourceWarp_LPIPS_EvalOverlap": None,
                f"{candidate_label}_SourceWarp_LPIPS_MaskMode": None,
                f"{candidate_label}_WarpGT_PSNR": None,
                f"{candidate_label}_WarpGT_LPIPS": None,
                f"{candidate_label}_WarpGT_Hybrid_Score": None,
                f"{candidate_label}_WarpGT_LPIPS_EvalOverlap": None,
                f"{candidate_label}_WarpGT_LPIPS_MaskMode": None,
                f"{candidate_label}_Overlap": None,
                f"{candidate_label}_VisibleOverlapRaw": None,
                f"{candidate_label}_GeoDiagnosticPath": None,
                f"{candidate_label}_GeoDiagnosticRelPath": None,
                f"{candidate_label}_PhotoDiagnosticPath": None,
                f"{candidate_label}_PhotoDiagnosticRelPath": None,
                f"{candidate_label}_Pose_backend": None,
                f"{candidate_label}_Pose_skip_reason": None,
                f"{candidate_label}_Pose_RotErrDeg": None,
                f"{candidate_label}_Pose_TransDirErrDeg": None,
                f"{candidate_label}_Pose_TransDirCos": None,
                f"{candidate_label}_Pose_TransDirCorrect": None,
                f"{candidate_label}_Pose_TransNormPredRaw": None,
                f"{candidate_label}_Pose_TransNormPredMetric": None,
                f"{candidate_label}_Pose_TransNormExpected": None,
                f"{candidate_label}_Pose_TransNormErrM": None,
                f"{candidate_label}_Pose_TxPredMetric": None,
                f"{candidate_label}_Pose_TyPredMetric": None,
                f"{candidate_label}_Pose_TzPredMetric": None,
                f"{candidate_label}_Pose_TxExpected": None,
                f"{candidate_label}_Pose_TyExpected": None,
                f"{candidate_label}_Pose_TzExpected": None,
                f"{candidate_label}_Pose_YawPredDeg": None,
                f"{candidate_label}_Pose_PitchPredDeg": None,
                f"{candidate_label}_Pose_YawExpectedDeg": None,
                f"{candidate_label}_Pose_PitchExpectedDeg": None,
                f"{candidate_label}_Pose_ActionComponent": None,
                f"{candidate_label}_Pose_ActionPred": None,
                f"{candidate_label}_Pose_ActionExpected": None,
                f"{candidate_label}_Pose_ActionSignedErr": None,
                f"{candidate_label}_Pose_ActionAbsErr": None,
                f"{candidate_label}_Pose_ActionSignCorrect": None,
                f"{candidate_label}_Pose_ActionPredDirection": None,
                f"{candidate_label}_Pose_ActionExpectedDirection": None,
                f"{candidate_label}_Pose_ActionUnit": None,
            }
            if "da3" in self.pose_estimators:
                for suffix in POSE_METRIC_SUFFIXES:
                    candidate_scores[f"{candidate_label}_DA3_{suffix}"] = None

            for backend_name in self.pose_estimators:
                pose_metrics = compute_candidate_pose_metrics(candidate_label, candidate_path, backend_name)
                if backend_name == self.primary_pose_backend:
                    add_prefixed_pose_scores(candidate_scores, candidate_label, pose_metrics)
                if backend_name == "da3":
                    add_prefixed_pose_scores(candidate_scores, candidate_label, pose_metrics, namespace="DA3")


            if not self.compute_photometric:
                return candidate_scores

            # --- 2. Project context/source to target pose for visible mask, compare GT-vs-candidate ---
            candidate_img_t = to_tensor(candidate_img_cv2)
            T_tgt_t = torch.from_numpy(tgt_item.extrinsics.copy()).float().to(self.device).unsqueeze(0)
            K_tgt_t = torch.from_numpy(tgt_item.intrinsics.copy()).float().to(self.device).unsqueeze(0)
            ctx_imgs_t, ctx_depths_t, T_ctxs_t, K_ctxs_t = [], [], [], []

            for c_item in ctx_items:
                ctx_imgs_t.append(to_tensor(c_item.image))
                ctx_depths_t.append(torch.from_numpy(c_item.depth).float().to(self.device).unsqueeze(0).unsqueeze(0))
                T_ctxs_t.append(torch.from_numpy(c_item.extrinsics.copy()).float().to(self.device).unsqueeze(0))
                K_ctxs_t.append(torch.from_numpy(c_item.intrinsics.copy()).float().to(self.device).unsqueeze(0))

            hybrid_res = self.evaluator.calc_hybrid_reprojection(
                ctx_imgs_t,
                ctx_depths_t,
                T_ctxs_t,
                K_ctxs_t,
                T_tgt_t,
                K_tgt_t,
                gt_img_t,
                candidate_img_t,
                alpha=0.8,
                return_vis=self.save_vis,
            )

            if hybrid_res:
                for key, value in hybrid_res.items():
                    if not key.startswith("vis_"):
                        candidate_scores[f"{candidate_label}_{key}"] = value


            else:
                candidate_scores[f"{candidate_label}_Photometric_skip_reason"] = "insufficient context-to-target visible overlap"

            return candidate_scores

        scores.update(evaluate_candidate_image(tgt_item.image.copy(), "GTImage", tgt_gt_path))
        scores["Pose_GTScaleCalibration"] = pose_scale_calibrations.get(self.primary_pose_backend)
        if "da3" in self.pose_estimators:
            scores["DA3_Pose_GTScaleCalibration"] = pose_scale_calibrations.get("da3")

        if gen_img_cv2 is not None:
            scores.update(evaluate_candidate_image(gen_img_cv2, "Pred", tgt_gen_path if tgt_gen_path else tgt_gt_path))
            alias_map = {
                "SelectedContextIndex": "Pred_SelectedContextIndex",
                "SelectedContextOverlap": "Pred_SelectedContextOverlap",
                "SelectedContextFrameId": "Pred_SelectedContextFrameId",
                "ContextSelectionSource": "Pred_ContextSelectionSource",
                "Hybrid_Score": "Pred_Hybrid_Score",
                "PSNR": "Pred_PSNR",
                "LPIPS": "Pred_LPIPS",
                "LPIPS_EvalOverlap": "Pred_LPIPS_EvalOverlap",
                "LPIPS_MaskMode": "Pred_LPIPS_MaskMode",
                "SourceWarp_PSNR": "Pred_SourceWarp_PSNR",
                "SourceWarp_LPIPS": "Pred_SourceWarp_LPIPS",
                "SourceWarp_Hybrid_Score": "Pred_SourceWarp_Hybrid_Score",
                "SourceWarp_LPIPS_EvalOverlap": "Pred_SourceWarp_LPIPS_EvalOverlap",
                "SourceWarp_LPIPS_MaskMode": "Pred_SourceWarp_LPIPS_MaskMode",
                "WarpGT_PSNR": "Pred_WarpGT_PSNR",
                "WarpGT_LPIPS": "Pred_WarpGT_LPIPS",
                "WarpGT_Hybrid_Score": "Pred_WarpGT_Hybrid_Score",
                "WarpGT_LPIPS_EvalOverlap": "Pred_WarpGT_LPIPS_EvalOverlap",
                "WarpGT_LPIPS_MaskMode": "Pred_WarpGT_LPIPS_MaskMode",
                "Overlap": "Pred_Overlap",
                "VisibleOverlapRaw": "Pred_VisibleOverlapRaw",
                "Pose_backend": "Pred_Pose_backend",
                "Pose_skip_reason": "Pred_Pose_skip_reason",
                "Pose_RotErrDeg": "Pred_Pose_RotErrDeg",
                "Pose_TransDirErrDeg": "Pred_Pose_TransDirErrDeg",
                "Pose_TransDirCos": "Pred_Pose_TransDirCos",
                "Pose_TransDirCorrect": "Pred_Pose_TransDirCorrect",
                "Pose_TransNormPredRaw": "Pred_Pose_TransNormPredRaw",
                "Pose_TransNormPredMetric": "Pred_Pose_TransNormPredMetric",
                "Pose_TransNormExpected": "Pred_Pose_TransNormExpected",
                "Pose_TransNormErrM": "Pred_Pose_TransNormErrM",
                "Pose_ActionComponent": "Pred_Pose_ActionComponent",
                "Pose_ActionPred": "Pred_Pose_ActionPred",
                "Pose_ActionExpected": "Pred_Pose_ActionExpected",
                "Pose_ActionSignedErr": "Pred_Pose_ActionSignedErr",
                "Pose_ActionAbsErr": "Pred_Pose_ActionAbsErr",
                "Pose_ActionSignCorrect": "Pred_Pose_ActionSignCorrect",
                "Pose_ActionPredDirection": "Pred_Pose_ActionPredDirection",
                "Pose_ActionExpectedDirection": "Pred_Pose_ActionExpectedDirection",
                "Pose_ActionUnit": "Pred_Pose_ActionUnit",
            }
            for old_key, pred_key in alias_map.items():
                scores[old_key] = scores.get(pred_key)
        else:
            scores["Pred_skip_reason"] = pred_skip_reason or "generated image unavailable"
            scores["skip_reason"] = scores["Pred_skip_reason"]


        return scores

    def evaluate_single_sample(self, case):
        results = []
        for sample in normalize_benchmark_steps(case):
            res = self.evaluate_step_sample(sample)
            if res is not None:
                results.append(res)
        if len(results) == 1:
            return results[0]
        return results

    def _run_samples(self, samples, output_json_path):
        results = []
        failures = []

        for case in tqdm(samples, desc="evaluating"):
            try:
                res = self.evaluate_single_sample(case)
                if isinstance(res, list):
                    results.extend([x for x in res if x is not None])
                elif res is not None:
                    results.append(res)
            except Exception as e:
                case_id = case.get('id') or case.get('sample_id')
                err = f"{type(e).__name__}: {e!r}"
                print(f"Error on case {case_id}: {err}")
                traceback.print_exc(limit=3)
                failures.append({"id": case_id, "error": err})
            finally:
                if os.environ.get("CLEAR_SCENE_CACHE_EVERY_SAMPLE", "0") == "1":
                    SCENE_CACHE.clear()
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        metrics_to_avg = [
            "Hybrid_Score",
            "PSNR",
            "LPIPS",
            "LPIPS_EvalOverlap",
            "SourceWarp_PSNR",
            "SourceWarp_LPIPS",
            "SourceWarp_Hybrid_Score",
            "SourceWarp_LPIPS_EvalOverlap",
            "WarpGT_PSNR",
            "WarpGT_LPIPS",
            "WarpGT_Hybrid_Score",
            "WarpGT_LPIPS_EvalOverlap",
            "Overlap",
            "VisibleOverlapRaw",
            "Pose_RotErrDeg",
            "Pose_TransDirErrDeg",
            "Pose_TransDirCos",
            "Pose_TransDirCorrect",
            "Pose_TransNormPredRaw",
            "Pose_TransNormPredMetric",
            "Pose_TransNormExpected",
            "Pose_TransNormErrM",
            "Pose_ActionSignedErr",
            "Pose_ActionAbsErr",
            "Pose_ActionSignCorrect",
            "GTImage_Hybrid_Score",
            "GTImage_PSNR",
            "GTImage_LPIPS",
            "GTImage_LPIPS_EvalOverlap",
            "GTImage_SourceWarp_PSNR",
            "GTImage_SourceWarp_LPIPS",
            "GTImage_SourceWarp_Hybrid_Score",
            "GTImage_SourceWarp_LPIPS_EvalOverlap",
            "GTImage_WarpGT_PSNR",
            "GTImage_WarpGT_LPIPS",
            "GTImage_WarpGT_Hybrid_Score",
            "GTImage_WarpGT_LPIPS_EvalOverlap",
            "GTImage_Overlap",
            "GTImage_VisibleOverlapRaw",
            "GTImage_Pose_RotErrDeg",
            "GTImage_Pose_TransDirErrDeg",
            "GTImage_Pose_TransDirCos",
            "GTImage_Pose_TransDirCorrect",
            "GTImage_Pose_TransNormPredRaw",
            "GTImage_Pose_TransNormPredMetric",
            "GTImage_Pose_TransNormExpected",
            "GTImage_Pose_TransNormErrM",
            "GTImage_Pose_ActionSignedErr",
            "GTImage_Pose_ActionAbsErr",
            "GTImage_Pose_ActionSignCorrect",
            "Pred_Hybrid_Score",
            "Pred_PSNR",
            "Pred_LPIPS",
            "Pred_LPIPS_EvalOverlap",
            "Pred_SourceWarp_PSNR",
            "Pred_SourceWarp_LPIPS",
            "Pred_SourceWarp_Hybrid_Score",
            "Pred_SourceWarp_LPIPS_EvalOverlap",
            "Pred_WarpGT_PSNR",
            "Pred_WarpGT_LPIPS",
            "Pred_WarpGT_Hybrid_Score",
            "Pred_WarpGT_LPIPS_EvalOverlap",
            "Pred_Overlap",
            "Pred_VisibleOverlapRaw",
            "Pred_Pose_RotErrDeg",
            "Pred_Pose_TransDirErrDeg",
            "Pred_Pose_TransDirCos",
            "Pred_Pose_TransDirCorrect",
            "Pred_Pose_TransNormPredRaw",
            "Pred_Pose_TransNormPredMetric",
            "Pred_Pose_TransNormExpected",
            "Pred_Pose_TransNormErrM",
            "Pred_Pose_ActionSignedErr",
            "Pred_Pose_ActionAbsErr",
            "Pred_Pose_ActionSignCorrect",
        ]
        if "da3" in self.pose_estimators:
            for candidate_label in ("GTImage", "Pred"):
                for suffix in (
                    "Pose_RotErrDeg",
                    "Pose_TransDirErrDeg",
                    "Pose_TransDirCos",
                    "Pose_TransDirCorrect",
                    "Pose_TransNormPredRaw",
                    "Pose_TransNormPredMetric",
                    "Pose_TransNormExpected",
                    "Pose_TransNormErrM",
                    "Pose_ActionSignedErr",
                    "Pose_ActionAbsErr",
                    "Pose_ActionSignCorrect",
                ):
                    metrics_to_avg.append(f"{candidate_label}_DA3_{suffix}")
        summary = {}
        for m in metrics_to_avg:
            vals = [
                r[m] for r in results
                if r.get(m) is not None and not (isinstance(r.get(m), float) and np.isnan(r.get(m)))
            ]
            summary[f"Mean_{m}"] = float(np.mean(vals)) if vals else 0.0
        summary["num_results"] = len(results)
        summary["num_failures"] = len(failures)
        summary["Pose_Sanity"] = {
            "backend": getattr(self.pose_estimator, "model_id", None),
            "load_error": getattr(self.pose_estimator, "load_error", None),
            "gt_rotation_error_deg": summarize_metric([r.get("GTImage_Pose_RotErrDeg") for r in results]),
            "gt_translation_direction_error_deg": summarize_metric([r.get("GTImage_Pose_TransDirErrDeg") for r in results]),
            "pred_rotation_error_deg": summarize_metric([r.get("Pred_Pose_RotErrDeg") for r in results]),
            "pred_translation_direction_error_deg": summarize_metric([r.get("Pred_Pose_TransDirErrDeg") for r in results]),
            "pred_translation_direction_cos": summarize_metric([r.get("Pred_Pose_TransDirCos") for r in results]),
            "pred_translation_direction_correct_rate": summarize_metric([r.get("Pred_Pose_TransDirCorrect") for r in results]),
            "pred_translation_norm_error_m": summarize_metric([r.get("Pred_Pose_TransNormErrM") for r in results]),
            "pred_action_signed_error": summarize_metric([r.get("Pred_Pose_ActionSignedErr") for r in results]),
            "pred_action_abs_error": summarize_metric([r.get("Pred_Pose_ActionAbsErr") for r in results]),
            "pred_action_sign_correct_rate": summarize_metric([r.get("Pred_Pose_ActionSignCorrect") for r in results]),
            "gt_split": summarize_pose_split(results, "GTImage"),
            "pred_split": summarize_pose_split(results, "Pred"),
        }
        if "da3" in self.pose_estimators:
            da3_estimator = self.pose_estimators["da3"]
            summary["DA3_Pose_Sanity"] = {
                "backend": getattr(da3_estimator, "model_id", None),
                "load_error": getattr(da3_estimator, "load_error", None),
                "use_ray_pose": getattr(da3_estimator, "use_ray_pose", None),
                "process_res": getattr(da3_estimator, "process_res", None),
                "process_res_method": getattr(da3_estimator, "process_res_method", None),
                "ref_view_strategy": getattr(da3_estimator, "ref_view_strategy", None),
                "aspect_policy": getattr(da3_estimator, "aspect_policy", None),
                "gt_rotation_error_deg": summarize_metric([r.get("GTImage_DA3_Pose_RotErrDeg") for r in results]),
                "gt_translation_direction_error_deg": summarize_metric([r.get("GTImage_DA3_Pose_TransDirErrDeg") for r in results]),
                "pred_rotation_error_deg": summarize_metric([r.get("Pred_DA3_Pose_RotErrDeg") for r in results]),
                "pred_translation_direction_error_deg": summarize_metric([r.get("Pred_DA3_Pose_TransDirErrDeg") for r in results]),
                "pred_translation_direction_cos": summarize_metric([r.get("Pred_DA3_Pose_TransDirCos") for r in results]),
                "pred_translation_direction_correct_rate": summarize_metric([r.get("Pred_DA3_Pose_TransDirCorrect") for r in results]),
                "pred_translation_norm_error_m": summarize_metric([r.get("Pred_DA3_Pose_TransNormErrM") for r in results]),
                "pred_action_signed_error": summarize_metric([r.get("Pred_DA3_Pose_ActionSignedErr") for r in results]),
                "pred_action_abs_error": summarize_metric([r.get("Pred_DA3_Pose_ActionAbsErr") for r in results]),
                "pred_action_sign_correct_rate": summarize_metric([r.get("Pred_DA3_Pose_ActionSignCorrect") for r in results]),
                "gt_split": summarize_pose_split(results, "GTImage_DA3"),
                "pred_split": summarize_pose_split(results, "Pred_DA3"),
            }

        with open(output_json_path, 'w') as f:
            json.dump({"summary": summary, "details": results, "failures": failures}, f, indent=4)
        print(f"\nEvaluation complete, results saved to: {output_json_path}")


    def run(self, jsonl_path, output_json_path=None, output_dir=None):
        jsonl_path = str(jsonl_path)
        with open(jsonl_path, 'r') as f:
            samples = [json.loads(l) for l in f if l.strip()]
        if self.max_cases is not None:
            samples = samples[:max(0, self.max_cases)]


        if output_dir is not None:
            input_stem = os.path.splitext(os.path.basename(jsonl_path))[0]
            groups = group_samples_by_model(samples, fallback=input_stem)
            for model_name, model_samples in groups.items():
                model_dir = os.path.join(str(output_dir), model_name)
                os.makedirs(model_dir, exist_ok=True)
                output_json_path = os.path.join(model_dir, build_cmg_result_json_path(jsonl_path))
                print(f"\n=== Evaluating model: {model_name} ===")
                print(f"Input: {jsonl_path}")
                print(f"Sample count: {len(model_samples)}")
                print(f"Output: {output_json_path}")
                self._run_samples(model_samples, output_json_path)
            return

        if output_json_path is None:
            output_json_path = build_cmg_result_json_path(jsonl_path)
        self._run_samples(samples, output_json_path)


    def run_batch(self, jsonl_paths, output_dir=".", split_by_model=False):
        os.makedirs(output_dir, exist_ok=True)
        for jsonl_path in jsonl_paths:
            if split_by_model:
                print(f"\n=== Starting evaluation: {jsonl_path} ===")
                self.run(jsonl_path, output_dir=output_dir)
                continue
            output_json_path = os.path.join(output_dir, build_cmg_result_json_path(jsonl_path))
            print(f"\n=== Starting evaluation: {jsonl_path} ===")
            self.run(jsonl_path, output_json_path)


def build_cmg_result_json_path(jsonl_path):
    input_name = os.path.splitext(os.path.basename(jsonl_path))[0]
    return f"cmg_result_{input_name}.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EgoGenEval CMG (camera-motion geometry) metrics.")
    parser.add_argument("--input_jsonl", nargs="*", default=INPUT_JSONL, help="Input generated JSONL file(s).")
    parser.add_argument("--output_dir", default=OUTPUT_DIR, help="Directory for result JSON files.")
    parser.add_argument("--output_json_path", default=None, help="Single output JSON path; only valid with one input and --no-split-by-model.")
    parser.add_argument("--split-by-model", action="store_true", default=True, help="Write results under <output_dir>/<model>/ for each model in the JSONL.")
    parser.add_argument("--no-split-by-model", action="store_false", dest="split_by_model", help="Write each input result directly under --output_dir.")
    parser.add_argument("--save-vis", action="store_true", default=True, help="Save diagnostic visualizations.")
    parser.add_argument("--no-save-vis", action="store_false", dest="save_vis", help="Do not save diagnostic visualizations.")
    parser.add_argument("--device", default=None, help="Torch device, e.g. cuda:0 or cpu.")
    parser.add_argument("--oracle-mode", action="store_true", help="Use target GT image as prediction.")
    parser.add_argument(
        "--pose-backends",
        default=os.environ.get("POSE_BACKENDS") or "da3",
        help="Comma/space separated pose backend list. First backend fills existing Pose fields.",
    )
    parser.add_argument(
        "--da3-model-dir",
        default=os.environ.get("DA3_MODEL_DIR") or DEFAULT_DA3_MODEL_DIR,
        help="Local DA3 Hugging Face model directory.",
    )
    parser.add_argument(
        "--da3-code-dir",
        default=os.environ.get("DA3_CODE_DIR") or DEFAULT_DA3_CODE_DIR,
        help="Local Depth-Anything-3 checkout directory.",
    )
    parser.add_argument(
        "--da3-use-ray-pose",
        action="store_true",
        default=env_flag("DA3_USE_RAY_POSE", False),
        help="Use DA3 ray-based pose estimation.",
    )
    parser.add_argument(
        "--da3-process-res",
        type=int,
        default=int(os.environ.get("DA3_PROCESS_RES", "504")),
        help="DA3 inference processing resolution.",
    )
    parser.add_argument(
        "--da3-process-res-method",
        choices=DA3_PROCESS_RES_METHODS,
        default=os.environ.get("DA3_PROCESS_RES_METHOD", "upper_bound_resize"),
        help="DA3 aspect-preserving boundary resize/crop policy before patch alignment.",
    )
    parser.add_argument(
        "--da3-ref-view-strategy",
        choices=DA3_REF_VIEW_STRATEGIES,
        default=os.environ.get("DA3_REF_VIEW_STRATEGY", "saddle_balanced"),
        help="DA3 camera-decoder reference-view selection strategy.",
    )
    parser.add_argument(
        "--da3-aspect-policy",
        choices=DA3_ASPECT_POLICIES,
        default=os.environ.get("DA3_ASPECT_POLICY", "native"),
        help=(
            "DA3-only input aspect policy. native reproduces the official/main protocol; "
            "gt_center_crop center-crops both pose inputs to the GT aspect ratio."
        ),
    )
    parser.add_argument(
        "--pose-only",
        action="store_true",
        help="Compute pose metrics only; leave photometric fields empty.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Optional input-case limit for smoke tests; processes all cases by default.",
    )
    args = parser.parse_args()

    runner = CmgRunner(
        oracle_mode=args.oracle_mode,
        save_vis=args.save_vis,
        device=args.device,
        pose_backends=args.pose_backends,
        da3_model_dir=args.da3_model_dir,
        da3_code_dir=args.da3_code_dir,
        da3_use_ray_pose=args.da3_use_ray_pose,
        da3_process_res=args.da3_process_res,
        da3_process_res_method=args.da3_process_res_method,
        da3_ref_view_strategy=args.da3_ref_view_strategy,
        da3_aspect_policy=args.da3_aspect_policy,
        compute_photometric=not args.pose_only,
        max_cases=args.max_cases,
    )
    if args.output_json_path:
        if args.split_by_model or len(args.input_jsonl) != 1:
            raise ValueError("--output_json_path requires exactly one input and --no-split-by-model.")
        runner.run(args.input_jsonl[0], output_json_path=args.output_json_path)
    else:
        runner.run_batch(args.input_jsonl, output_dir=args.output_dir, split_by_model=args.split_by_model)
