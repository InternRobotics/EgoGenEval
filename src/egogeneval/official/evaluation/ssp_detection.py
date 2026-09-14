from __future__ import annotations
import os
import sys
import json
import inspect
import logging
import tempfile
import hashlib
import traceback
import re
import copy
from collections import OrderedDict
from types import SimpleNamespace
from typing import List, Tuple, Optional, Any, Dict
import gc

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torchvision.transforms as T

from benchmark_config import INPUT_JSONL, OUTPUT_DIR

logger = logging.getLogger(__name__)

# ======= Optional VLM matcher: reuses the shared local Qwen / OpenAI-compatible client =======
try:
    import vlm_client
    _VLM_IMPORT_ERROR = None
except Exception as e:
    vlm_client = None
    _VLM_IMPORT_ERROR = e

DEFAULT_VLM_BACKEND = "local_transformers"
DEFAULT_VLM_MODEL = os.environ.get("VLM_MODEL", "Qwen/Qwen3-VL-30B-A3B-Instruct")
DEFAULT_VLM_BASE_URL = None
OPENAI_COMPATIBLE_VLM_BACKENDS = {"openai_responses", "openai_chat", "openai_compatible_chat"}

# ======= SAM3 imports =======
try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    _SAM3_IMPORT_ERROR = None
except Exception as e:
    build_sam3_image_model = None
    Sam3Processor = None
    _SAM3_IMPORT_ERROR = e

def _safe_signature_params(fn) -> set:
    try:
        return set(inspect.signature(fn).parameters.keys())
    except (TypeError, ValueError):
        return set()

def safe_box_xyxy(box: List[float], h: int, w: int) -> Optional[List[int]]:
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]

# ==========================================
# Dataloader
# ==========================================
from dataloader import (
    load_scannetpp_scene,
    load_matterport3d_scene,
    load_scannet_scene,
    load_hypersim_scene,
    read_image_cv2_local,
)

DATASET_REGISTRY = {
    "scannetpp": load_scannetpp_scene,
    "mp3d": load_matterport3d_scene,
    "matterport3d": load_matterport3d_scene,
    "scannet": load_scannet_scene,
    "hypersim": load_hypersim_scene,
}

# Scene-level LRU cache to bound RAM
SCENE_CACHE = {}
MAX_CACHE_SIZE = 5


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
    """Return the original benchmark step represented by a strict-only row."""
    strict_meta = case.get("cycle_step2_strict_rerun")
    if not isinstance(strict_meta, dict):
        return None
    try:
        step = int(strict_meta.get("original_cycle_step") or 0)
    except (TypeError, ValueError):
        return None
    return step if step > 0 else None


def cycle_step2_strict_source_frame_ref(case: Dict[str, Any]) -> Optional[str]:
    """Return the physical source frame id for strict cycle-step2 reruns.

    The strict rerun stores the previous generated image as ``input_images[0]``.
    Object visibility/projection still needs the physical pose/depth of that
    generated view.  For cycle step 2, ``frame_ids`` is [source, step1-target,
    source], so the source reference is ``frame_ids[original_cycle_step - 1]``.
    """
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
    """Return original trajectory length instead of strict local row length."""
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


DETECTOR_LABEL_STOPWORDS = {
    "",
    "none",
    "unknown",
    "unclear",
    "uncertain",
    "other",
    "object",
    "objects",
    "thing",
    "things",
    "item",
    "items",
    "wall",
    "walls",
    "floor",
    "floors",
    "ceiling",
    "ceilings",
    "sky",
    "room",
    "scene",
    "background",
    "foreground",
    "layout",
}


def normalize_detector_label(label: Any) -> str:
    if isinstance(label, dict):
        for key in ("category", "label", "name", "object"):
            if label.get(key):
                label = label.get(key)
                break
    label = str(label or "").strip().lower()
    label = re.sub(r"^[\"'`]+|[\"'`]+$", "", label)
    label = re.sub(r"^(a|an|the)\s+", "", label)
    label = label.replace("_", " ").replace("-", " ")
    label = re.sub(r"\s+", " ", label)
    label = re.sub(r"[^a-z0-9 /]+", "", label).strip(" /")
    if "/" in label:
        label = label.split("/")[0].strip()
    return "" if label in DETECTOR_LABEL_STOPWORDS else label


def dedupe_detector_labels(labels: List[Any], max_labels: Optional[int] = None) -> List[str]:
    if labels is None:
        labels = []
    elif isinstance(labels, (str, dict)):
        labels = [labels]
    out: List[str] = []
    for label in labels or []:
        cleaned = normalize_detector_label(label)
        if cleaned and cleaned not in out:
            out.append(cleaned)
            if max_labels is not None and len(out) >= max_labels:
                break
    return out


def extract_detector_prompt_labels(case: Dict[str, Any], max_labels: int = 20) -> List[str]:
    """Read per-sample detector labels from a legacy VLM-tagging export.

    Unreachable in the official protocol, where frozen per-step labels from
    ``target_images.qwen3vl_gt_object_labels`` always take precedence.
    """
    labels: List[Any] = []

    def add(value: Any):
        if isinstance(value, list):
            labels.extend(value)
        elif value is not None:
            labels.append(value)

    add(case.get("object_prompt_labels"))
    add(case.get("detector_prompt_labels"))

    legacy_vlm_tags = case.get("stage2", {}) if isinstance(case.get("stage2"), dict) else {}
    add(legacy_vlm_tags.get("detector_prompt_labels"))

    vlm = legacy_vlm_tags.get("vlm", {}) if isinstance(legacy_vlm_tags.get("vlm"), dict) else {}
    obj_tags = vlm.get("object_anchor_tags", {}) if isinstance(vlm.get("object_anchor_tags"), dict) else {}
    add(obj_tags.get("detector_prompt_labels"))
    for obj in obj_tags.get("main_objects", []) or []:
        if isinstance(obj, dict):
            add(obj.get("category"))
    primary = obj_tags.get("primary_anchor", {}) or {}
    if isinstance(primary, dict):
        add(primary.get("category"))
    for group in obj_tags.get("repeated_instance_groups", []) or []:
        if isinstance(group, dict):
            add(group.get("category"))

    rel_tags = vlm.get("spatial_relation_tags", {}) if isinstance(vlm.get("spatial_relation_tags"), dict) else {}
    for rel in rel_tags.get("camera_object_relations", []) or []:
        if isinstance(rel, dict):
            add(rel.get("object"))
    for rel in rel_tags.get("object_object_depth_pairs", []) or []:
        if isinstance(rel, dict):
            add(rel.get("front"))
            add(rel.get("back"))
    for rel in rel_tags.get("object_object_left_right_pairs", []) or []:
        if isinstance(rel, dict):
            add(rel.get("left"))
            add(rel.get("right"))
    for rel in rel_tags.get("object_region_relations", []) or []:
        if isinstance(rel, dict):
            add(rel.get("object"))

    return dedupe_detector_labels(labels, max_labels=max_labels)


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def extract_frozen_gt_step_prompt_labels(
    case: Dict[str, Any],
    target_frame: Optional[Dict[str, Any]],
    step_index: int,
    benchmark_step: Optional[int] = None,
    max_labels: int = 20,
) -> Tuple[List[str], Optional[str]]:
    """Read the frozen Qwen labels for one GT target step.

    The target-frame annotation is authoritative.  The duplicated top-level
    ``gt_step_object_labels`` record is only a fallback for older exported
    ranking inputs.  Crucially, labels from different target steps are never
    unioned: chain/cycle step *t* receives only the annotation of GT step *t*.
    """

    candidates: List[Tuple[Any, str]] = []
    if isinstance(target_frame, dict):
        candidates.append(
            (target_frame.get("qwen3vl_gt_object_labels"), "target_images.qwen3vl_gt_object_labels")
        )

    expected_step = int(benchmark_step) if benchmark_step is not None else step_index + 1
    step_records = case.get("gt_step_object_labels")
    if isinstance(step_records, list):
        exact = None
        if 0 <= step_index < len(step_records):
            exact = step_records[step_index]
        if not isinstance(exact, dict) or int(exact.get("step") or expected_step) != expected_step:
            exact = next(
                (
                    row
                    for row in step_records
                    if isinstance(row, dict) and int(row.get("step") or -1) == expected_step
                ),
                None,
            )
        if isinstance(exact, dict):
            candidates.append(
                (exact.get("qwen3vl_gt_object_labels", exact), "gt_step_object_labels")
            )

    for annotation, source in candidates:
        if not isinstance(annotation, dict):
            continue
        labels = annotation.get("detector_prompt_labels")
        if not labels:
            labels = annotation.get("object_labels")
        normalized = dedupe_detector_labels(labels, max_labels=max_labels)
        if normalized:
            return normalized, source
    return [], None


def normalize_benchmark_steps(case: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Expand the spatial dataset JSONL into one evaluable row per model call."""
    case_id = str(case.get("id") or case.get("sample_id") or "unknown")
    dataset_name = normalize_dataset_name(case.get("dataset"))
    scene_id = str(case.get("scene_id") or "")
    object_prompts = extract_detector_prompt_labels(case)
    use_frozen_gt_step_prompts = _env_flag("BENCHMARK_USE_FROZEN_GT_STEP_PROMPTS", default=False)

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
            ctx_frames = [current_frame] + aux_frames
            geometry_current_frame = strict_source_ref if (index == 0 and strict_source_ref) else current_frame
            geometry_ctx_frames = [geometry_current_frame] + aux_frames
            step_object_prompts = object_prompts
            prompt_source = "case_object_labels" if object_prompts else "fixed_default_only"
            if use_frozen_gt_step_prompts:
                frozen_prompts, frozen_source = extract_frozen_gt_step_prompt_labels(
                    case, target_frame, index, benchmark_step=step
                )
                if frozen_prompts:
                    step_object_prompts = frozen_prompts
                    prompt_source = frozen_source or "frozen_gt_step_labels"
                else:
                    prompt_source = "frozen_gt_step_labels_missing_fallback"
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
                "context_metadata": case.get("context_metadata"),
                "object_prompts": step_object_prompts,
                "object_prompt_protocol": (
                    "fixed36_plus_frozen_qwen_gt_step_v1"
                    if use_frozen_gt_step_prompts
                    else "legacy_case_prompts_plus_fixed36"
                ),
                "object_prompt_source": prompt_source,
            })
        return steps

    sample = normalize_benchmark_case(case)
    sample.setdefault("generated_path", prediction_image_path(case))
    sample.setdefault("parent_id", case_id)
    sample.setdefault("step", 1)
    sample.setdefault("num_steps", 1)
    sample.setdefault("object_prompts", object_prompts)
    sample.setdefault("context_metadata", case.get("context_metadata"))
    return [sample]


def safe_filename_token(value: Any, max_len: int = 180) -> str:
    """Make sample ids safe for temp/debug filenames while keeping JSON ids unchanged."""
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


def stable_image_seed(img: np.ndarray) -> int:
    """Derive a stable OpenCV RNG seed from image bytes."""
    arr = np.ascontiguousarray(img)
    digest = hashlib.blake2b(arr.view(np.uint8), digest_size=4).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) & 0x7FFFFFFF


def normalize_benchmark_case(case: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize old runner JSONL and v5 benchmark JSONL into one internal shape."""
    case_id = str(case.get("id") or case.get("sample_id") or "unknown")
    dataset_name = normalize_dataset_name(case.get("dataset"))
    scene_id = str(case.get("scene_id") or "")
    object_prompts = extract_detector_prompt_labels(case)

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
            "object_prompts": object_prompts,
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
            "object_prompts": object_prompts,
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
        "object_prompts": object_prompts,
    }

# ==========================================
# Utility functions
# ==========================================
def infer_dataset_and_scene_from_path(path: str) -> Tuple[str, str]:
    p = path.lower().replace("\\", "/").split("/")
    if "scannetpp" in p:
        idx = p.index("scannetpp")
        return "scannetpp", p[idx + 1]
    if "scannet" in p:
        idx = p.index("scannet")
        return "scannet", p[idx + 1]
    if "matterport3d" in p:
        idx = p.index("matterport3d")
        return "matterport3d", p[idx + 1]
    if "hypersim" in p:
        idx = p.index("hypersim")
        for j in range(idx + 1, len(p)):
            if p[j].startswith("ai_"):
                return "hypersim", p[j]
        if idx + 1 < len(p):
            return "hypersim", p[idx + 1]
    raise ValueError(f"Cannot infer dataset and scene from path: {path}")

def _normalize_frame_id_token(value):
    token = str(value or "").strip()
    if not token:
        return None
    m = re.search(r"(\d+)", token)
    if m:
        return str(int(m.group(1)))
    return token

def _item_frame_id_tokens(item):
    tokens = set()
    frame_idx = getattr(item, "frame_idx", None)
    if frame_idx is not None:
        tokens.add(str(frame_idx))
        norm = _normalize_frame_id_token(frame_idx)
        if norm is not None:
            tokens.add(norm)
    frame_name = getattr(item, "frame_name", None)
    if frame_name:
        base = os.path.basename(str(frame_name))
        stem = os.path.splitext(base)[0]
        for part in (base, stem):
            norm = _normalize_frame_id_token(part)
            if norm is not None:
                tokens.add(norm)
    return tokens

def get_frame_from_scene(dataset_name: str, scene_id: str, frame_path: str):
    dataset_name = normalize_dataset_name(dataset_name)
    cache_key = f"{dataset_name}_{scene_id}"
    
    if cache_key not in SCENE_CACHE:
        if len(SCENE_CACHE) >= MAX_CACHE_SIZE:
            oldest_key = next(iter(SCENE_CACHE))
            del SCENE_CACHE[oldest_key]
            
        if dataset_name not in DATASET_REGISTRY:
            raise KeyError(f"Unknown dataset: {dataset_name}")
        SCENE_CACHE[cache_key], _ = DATASET_REGISTRY[dataset_name](scene_id)

    target_basename = os.path.splitext(os.path.basename(frame_path))[0]
    for item in SCENE_CACHE[cache_key]:
        item_basename = os.path.splitext(os.path.basename(item.frame_name))[0]
        if target_basename == item_basename:
            return item
    # Strict cycle-step2 rows may use a bare physical frame id for the geometry
    # reference while the actual model input is a generated image path.  Keep
    # numeric/id matching restricted to bare ids so generated paths such as
    # ".../step_01.png" are not accidentally mapped to dataset frame 1.
    frame_path_str = str(frame_path or "")
    is_bare_id = frame_path_str and not any(sep in frame_path_str for sep in ("/", "\\")) and "." not in frame_path_str
    if is_bare_id:
        wanted = _normalize_frame_id_token(frame_path_str)
        for item in SCENE_CACHE[cache_key]:
            frame_idx = getattr(item, "frame_idx", None)
            if frame_idx is not None and _normalize_frame_id_token(frame_idx) == wanted:
                return item
        for item in SCENE_CACHE[cache_key]:
            frame_name = getattr(item, "frame_name", None)
            if frame_name:
                base = os.path.basename(str(frame_name))
                stem = os.path.splitext(base)[0]
                if any(_normalize_frame_id_token(part) == wanted for part in (base, stem)):
                    return item
            elif wanted in _item_frame_id_tokens(item):
                return item
    raise FileNotFoundError(f"Physical frame not found: {frame_path}")

def get_context_frame_for_metric(dataset_name: str, scene_id: str, image_path: str, geometry_path: Optional[str] = None):
    """Load a context item whose image can be generated but geometry is physical."""
    geometry_path = geometry_path or image_path
    image_override = None
    if image_path and (os.path.isabs(str(image_path)) or os.path.exists(str(image_path))):
        image_override = read_image_cv2_local(str(image_path))

    ref_item = None
    ref_error = None
    if geometry_path:
        try:
            ref_item = get_frame_from_scene(dataset_name, scene_id, str(geometry_path))
        except Exception as e:
            ref_error = e
    if ref_item is None and image_path and str(image_path) != str(geometry_path):
        try:
            ref_item = get_frame_from_scene(dataset_name, scene_id, str(image_path))
        except Exception:
            pass

    if ref_item is not None:
        item = copy.copy(ref_item)
        item.reference_frame_name = getattr(ref_item, "frame_name", None)
        item.geometry_frame_path = geometry_path
        if image_override is not None:
            item.image = image_override
            item.frame_name = str(image_path)
            item.generated_image_path = str(image_path)
        return item

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
    if ref_error is not None:
        raise ref_error
    return get_frame_from_scene(dataset_name, scene_id, str(image_path))


def _box_area_xyxy(box) -> int:
    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in box[:4]]
        return max(0, x2 - x1) * max(0, y2 - y1)
    except Exception:
        return 0


def _mask_area(mask) -> int:
    if mask is None:
        return 0
    try:
        return int((np.asarray(mask) > 0).sum())
    except Exception:
        return 0


def _box_iou_xyxy(box_a, box_b) -> float:
    try:
        ax1, ay1, ax2, ay2 = [float(v) for v in box_a[:4]]
        bx1, by1, bx2, by2 = [float(v) for v in box_b[:4]]
    except Exception:
        return 0.0

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _mask_iou_binary(mask_a, mask_b) -> float:
    if mask_a is None or mask_b is None:
        return 0.0
    ma = np.asarray(mask_a) > 0
    mb = np.asarray(mask_b) > 0
    if ma.shape != mb.shape:
        return 0.0
    inter = int(np.logical_and(ma, mb).sum())
    union = int(np.logical_or(ma, mb).sum())
    return float(inter / union) if union > 0 else 0.0


def _as_rgb_uint8(img: np.ndarray) -> np.ndarray:
    """The benchmark pipeline stores images as RGB; keep that contract for PIL models."""
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    elif arr.ndim == 3 and arr.shape[2] > 3:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)



def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu().numpy()
    return np.asarray(x)


def make_json_safe(obj):
    """Recursively convert numpy/torch scalar containers into JSON-native values."""
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return make_json_safe(obj.tolist())
    if isinstance(obj, torch.Tensor):
        if obj.ndim == 0:
            return make_json_safe(obj.detach().cpu().item())
        return make_json_safe(obj.detach().cpu().tolist())
    if isinstance(obj, (np.floating, np.integer)):
        obj = obj.item()
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    return obj


def _float_or_none(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fit_tile_rgb(img: np.ndarray, size: int = 320) -> np.ndarray:
    rgb = _as_rgb_uint8(img)
    h, w = rgb.shape[:2]
    scale = min(size / max(h, 1), size / max(w, 1))
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), 245, dtype=np.uint8)
    y0 = (size - nh) // 2
    x0 = (size - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def _put_rgb_label(img: np.ndarray, text: str, color=(0, 0, 0)) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, 42), color, -1)
    cv2.addWeighted(overlay, 0.65, out, 0.35, 0, out)
    cv2.putText(out, text[:80], (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def _bool_from_vlm(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value or "").strip().lower()
    if text in {"true", "yes", "y", "1", "same", "consistent", "match", "matched"}:
        return True
    if text in {"false", "no", "n", "0", "different", "inconsistent", "mismatch", "not_match"}:
        return False
    return default


def _short_error_text(err: Any, limit: int = 600) -> str:
    text = re.sub(r"\s+", " ", str(err or "")).strip()
    if len(text) > int(limit):
        return text[: int(limit)] + "...[truncated]"
    return text


def _crop_box_with_context(img: np.ndarray, box: List[float], pad_ratio: float = 0.28) -> np.ndarray:
    rgb = _as_rgb_uint8(img)
    h, w = rgb.shape[:2]
    sb = safe_box_xyxy(box, h, w)
    if sb is None:
        return rgb
    x1, y1, x2, y2 = sb
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    pad = int(round(max(bw, bh) * pad_ratio))
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w - 1, x2 + pad)
    y2 = min(h - 1, y2 + pad)
    return rgb[y1:y2 + 1, x1:x2 + 1].copy()


def _draw_box_for_vlm(img: np.ndarray, box: List[float], color: Tuple[int, int, int], label: str) -> np.ndarray:
    rgb = _as_rgb_uint8(img).copy()
    h, w = rgb.shape[:2]
    sb = safe_box_xyxy(box, h, w)
    if sb is not None:
        x1, y1, x2, y2 = sb
        thickness = max(2, int(round(min(h, w) / 180)))
        cv2.rectangle(rgb, (x1, y1), (x2, y2), color, thickness)
        cv2.rectangle(rgb, (x1, max(0, y1 - 32)), (min(w - 1, x1 + 240), y1), color, -1)
        cv2.putText(rgb, label[:32], (x1 + 6, max(22, y1 - 9)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return rgb


def _draw_indexed_boxes_for_vlm(
    img: np.ndarray,
    boxes: List[List[float]],
    labels: Optional[List[str]],
    prefix: str,
    color: Tuple[int, int, int],
) -> np.ndarray:
    rgb = _as_rgb_uint8(img).copy()
    h, w = rgb.shape[:2]
    font_scale = max(0.52, min(0.9, min(h, w) / 560.0))
    thickness = max(2, int(round(min(h, w) / 220)))
    for idx, box in enumerate(boxes or []):
        sb = safe_box_xyxy(box, h, w)
        if sb is None:
            continue
        x1, y1, x2, y2 = sb
        obj_id = f"{prefix}{idx}"
        cat = normalize_detector_label(labels[idx]) if labels and idx < len(labels) else ""
        label = f"{obj_id} {cat}" if cat else obj_id
        label = label[:22]
        cv2.rectangle(rgb, (x1, y1), (x2, y2), color, thickness)
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        lx1 = max(0, min(x1, w - tw - 8))
        ly1 = max(0, y1 - th - baseline - 8)
        ly2 = min(h - 1, ly1 + th + baseline + 8)
        cv2.rectangle(rgb, (lx1, ly1), (min(w - 1, lx1 + tw + 8), ly2), color, -1)
        cv2.putText(
            rgb,
            label,
            (lx1 + 4, ly2 - baseline - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return rgb


def _make_vlm_crop_grid(
    img: np.ndarray,
    boxes: List[List[float]],
    labels: Optional[List[str]],
    prefix: str,
    title: str,
    width: int,
    crop_size: int = 112,
    max_crops: int = 24,
) -> np.ndarray:
    count = min(len(boxes or []), int(max_crops))
    cols = max(1, width // crop_size)
    rows = max(1, int(np.ceil(max(count, 1) / cols)))
    header_h = 34
    canvas = np.full((header_h + rows * crop_size, width, 3), 245, dtype=np.uint8)
    cv2.rectangle(canvas, (0, 0), (width - 1, header_h - 1), (70, 70, 70), -1)
    suffix = "" if len(boxes or []) <= max_crops else f" (first {max_crops}/{len(boxes)})"
    cv2.putText(canvas, f"{title}{suffix}"[:80], (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
    for out_idx in range(count):
        row, col = divmod(out_idx, cols)
        x0 = col * crop_size
        y0 = header_h + row * crop_size
        crop = _fit_tile_rgb(_crop_box_with_context(img, boxes[out_idx]), crop_size)
        obj_id = f"{prefix}{out_idx}"
        cat = normalize_detector_label(labels[out_idx]) if labels and out_idx < len(labels) else ""
        crop = _put_rgb_label(crop, f"{obj_id} {cat}" if cat else obj_id, color=(55, 55, 55))
        canvas[y0:y0 + crop_size, x0:x0 + crop_size] = crop
    return canvas


def _pad_rgb_height(img: np.ndarray, target_h: int) -> np.ndarray:
    if img.shape[0] >= target_h:
        return img
    pad = np.full((target_h - img.shape[0], img.shape[1], 3), 245, dtype=np.uint8)
    return np.concatenate([img, pad], axis=0)


def _make_vlm_global_match_montage(
    gt_img: np.ndarray,
    pred_img: np.ndarray,
    gt_boxes: List[List[float]],
    pred_boxes: List[List[float]],
    gt_labels: Optional[List[str]] = None,
    pred_labels: Optional[List[str]] = None,
    tile_size: int = 720,
    crop_atlas_max_objects: int = 32,
) -> np.ndarray:
    side = max(520, int(tile_size))
    gt_full = _fit_tile_rgb(
        _draw_indexed_boxes_for_vlm(gt_img, gt_boxes, gt_labels, "G", (235, 45, 45)),
        side,
    )
    pred_full = _fit_tile_rgb(
        _draw_indexed_boxes_for_vlm(pred_img, pred_boxes, pred_labels, "P", (40, 95, 235)),
        side,
    )
    gt_full = _put_rgb_label(gt_full, f"GT / target objects: G0-G{max(len(gt_boxes) - 1, 0)}", color=(165, 20, 20))
    pred_full = _put_rgb_label(pred_full, f"Generated/candidate objects: P0-P{max(len(pred_boxes) - 1, 0)}", color=(20, 55, 165))
    top = np.concatenate([gt_full, pred_full], axis=1)

    if len(gt_boxes) + len(pred_boxes) > int(crop_atlas_max_objects):
        return top

    gt_grid = _make_vlm_crop_grid(gt_img, gt_boxes, gt_labels, "G", "GT object crops", side)
    pred_grid = _make_vlm_crop_grid(pred_img, pred_boxes, pred_labels, "P", "Generated object crops", side)
    grid_h = max(gt_grid.shape[0], pred_grid.shape[0])
    crops = np.concatenate([_pad_rgb_height(gt_grid, grid_h), _pad_rgb_height(pred_grid, grid_h)], axis=1)
    return np.concatenate([top, crops], axis=0)


def _make_vlm_pair_montage(
    gt_img: np.ndarray,
    pred_img: np.ndarray,
    gt_box: List[float],
    pred_box: List[float],
    gt_label: str = "",
    pred_label: str = "",
    tile_size: int = 360,
) -> np.ndarray:
    gt_title = f"GT A: {gt_label or 'object'}"
    pred_title = f"PRED B: {pred_label or 'object'}"
    gt_full = _fit_tile_rgb(_draw_box_for_vlm(gt_img, gt_box, (235, 45, 45), "A"), tile_size)
    pred_full = _fit_tile_rgb(_draw_box_for_vlm(pred_img, pred_box, (40, 95, 235), "B"), tile_size)
    gt_crop = _fit_tile_rgb(_crop_box_with_context(gt_img, gt_box), tile_size)
    pred_crop = _fit_tile_rgb(_crop_box_with_context(pred_img, pred_box), tile_size)

    gt_full = _put_rgb_label(gt_full, gt_title, color=(165, 20, 20))
    pred_full = _put_rgb_label(pred_full, pred_title, color=(20, 55, 165))
    gt_crop = _put_rgb_label(gt_crop, "GT crop A", color=(165, 20, 20))
    pred_crop = _put_rgb_label(pred_crop, "Pred crop B", color=(20, 55, 165))
    return np.concatenate([
        np.concatenate([gt_full, pred_full], axis=1),
        np.concatenate([gt_crop, pred_crop], axis=1),
    ], axis=0)


VLM_MATCH_PROMPT = """You are an object matching judge for a physical-consistency benchmark.

The image is a 2x2 montage:
- Top-left: target ground-truth image with red box A.
- Top-right: candidate/generated image with blue box B.
- Bottom-left: crop around A.
- Bottom-right: crop around B.

Decide whether A and B are the same corresponding object instance for metric computation.
Use both full-image position/context and crops. Repeated identical objects should only match if they occupy the corresponding scene position. Do not match stuff regions such as wall, floor, ceiling, shadows, or background fragments unless the box clearly encloses a distinct countable object.

Then judge whether the matched object's visible orientation/pose is consistent: front/back/left/right-facing direction, object pose, plane orientation, opening direction, or distinctive side should not be flipped. If orientation is symmetric or not visually observable but there is no visible contradiction, set orientation_consistent=true and orientation_observable=false.

Return JSON only with these keys:
{
  "same_object": true/false,
  "orientation_consistent": true/false,
  "orientation_observable": true/false,
  "same_object_confidence": 0.0-1.0,
  "orientation_confidence": 0.0-1.0,
  "object_category_gt": "short category",
  "object_category_pred": "short category",
  "reason": "brief reason"
}
"""


def _format_vlm_box_catalog(prefix: str, boxes: List[List[float]], labels: Optional[List[str]]) -> str:
    lines = []
    for idx, box in enumerate(boxes or []):
        label = normalize_detector_label(labels[idx]) if labels and idx < len(labels) else "object"
        coords = [int(round(float(x))) for x in box[:4]]
        lines.append(f"- {prefix}{idx}: label={label}, box_xyxy={coords}")
    return "\n".join(lines) if lines else f"- No {prefix} objects."


def _build_vlm_global_match_prompt(
    gt_boxes: List[List[float]],
    pred_boxes: List[List[float]],
    gt_labels: Optional[List[str]],
    pred_labels: Optional[List[str]],
) -> str:
    gt_catalog = _format_vlm_box_catalog("G", gt_boxes, gt_labels)
    pred_catalog = _format_vlm_box_catalog("P", pred_boxes, pred_labels)
    return f"""You are matching object instances for a physical-consistency benchmark.

The image is a matching board:
- Left/top-left side: target ground-truth image with red boxes named G0, G1, ...
- Right/top-right side: generated/candidate image with blue boxes named P0, P1, ...
- If crop grids are present below, they show zoomed crops with the same object IDs.

Task:
Directly decide which GT objects correspond to which generated objects. A match means the same persistent object instance in the scene, not just the same category. Use full-image position, nearby context, crop appearance, and repeated-object ordering. For repeated chairs/books/monitors/etc., match only the instance at the corresponding scene position.

These matches are used to compute matched-object count, relative position/layout, depth relation, and other metrics over matched objects. Orientation/pose consistency should be reported as metadata, but an object can still be a match even if its orientation is wrong.

The catalog has already filtered out tiny clutter and partial edge fragments. Do not invent matches for objects that are not explicitly labeled in the catalog. Do not match wall, floor, ceiling, shadows, or background fragments unless the box clearly encloses a distinct countable object. Do not match objects whose category or visual type clearly differs, such as clothes vs pillow, bed vs sofa, chair vs table, or door vs window. Only allow a detector-label mismatch when the two boxes visibly enclose the same physical object and the label difference is just a naming synonym or detector ambiguity. Each G object can match at most one P object, and each P object can match at most one G object. If uncertain, leave the object unmatched.

GT object catalog:
{gt_catalog}

Generated object catalog:
{pred_catalog}

Return JSON only in this compact schema:
{{
  "matches": [
    {{
      "gt_id": "G0",
      "pred_id": "P3",
      "same_object_confidence": 0.95,
      "orientation_consistent": true,
      "orientation_observable": false,
      "orientation_confidence": 0.0,
      "reason": "brief visual/context reason"
    }}
  ],
  "unmatched_gt": ["G2"],
  "unmatched_pred": ["P1"]
}}
"""


def _parse_vlm_index(value: Any, prefix: str, limit: int) -> Optional[int]:
    if isinstance(value, (int, np.integer)):
        idx = int(value)
    else:
        text = str(value or "").strip().upper()
        match = re.search(rf"\b{re.escape(prefix.upper())}\s*#?\s*(\d+)\b", text)
        if match is None:
            match = re.fullmatch(r"\s*(\d+)\s*", text)
        if match is None:
            return None
        idx = int(match.group(1))
    if 0 <= idx < int(limit):
        return idx
    return None


_MATCH_LABEL_SYNONYM_GROUPS = [
    {"monitor", "monitor tv", "tv", "television"},
    {"sofa", "couch", "sofa bed", "bed"},
    {"bookshelf", "shelf", "bookcase"},
    {"desk", "table", "counter", "table counter", "countertop"},
    {"trash", "trash can"},
    {"bag", "backpack"},
    {"chair", "stool"},
]


def _labels_compatible_for_object_match(gt_label: Any, pred_label: Any) -> bool:
    gt = normalize_detector_label(gt_label)
    pred = normalize_detector_label(pred_label)
    if not gt or not pred:
        return True
    if gt == pred:
        return True
    for group in _MATCH_LABEL_SYNONYM_GROUPS:
        if gt in group and pred in group:
            return True
    return False


class VLMObjectMatchJudge:
    def __init__(
        self,
        backend: str,
        model: str,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        max_output_tokens: int = 500,
        timeout: float = 120.0,
        max_retries: int = 2,
        retry_sleep: float = 1.0,
        require_orientation: bool = True,
        fail_open: bool = False,
        montage_tile_size: int = 360,
        min_match_confidence: float = 0.8,
        local_device_map: str = "auto",
        local_torch_dtype: str = "auto",
        local_attn_implementation: Optional[str] = None,
        local_min_pixels: Optional[int] = None,
        local_max_pixels: Optional[int] = None,
        disable_env_proxy: bool = False,
        json_response_format: bool = False,
    ):
        self.backend = str(backend or "none")
        self.model = str(model or "")
        self.max_retries = int(max_retries)
        self.retry_sleep = float(retry_sleep)
        self.require_orientation = bool(require_orientation)
        self.fail_open = bool(fail_open)
        self.montage_tile_size = int(montage_tile_size)
        self.min_match_confidence = float(min_match_confidence)
        self.client = None
        self.status = "disabled"

        if self.backend.lower() == "none":
            return
        if vlm_client is None:
            self.status = f"disabled: vlm_client import failed: {_VLM_IMPORT_ERROR}"
            raise RuntimeError(self.status)

        self.client = vlm_client.VLMClient(
            backend=self.backend,
            model=self.model,
            api_key_env=api_key_env,
            base_url=base_url,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            timeout=timeout,
            local_device_map=local_device_map,
            local_torch_dtype=local_torch_dtype,
            local_attn_implementation=local_attn_implementation,
            local_min_pixels=local_min_pixels,
            local_max_pixels=local_max_pixels,
            disable_env_proxy=disable_env_proxy,
            json_response_format=json_response_format,
        )
        self.status = f"ok: {self.backend}/{self.model}"

    def is_available(self) -> bool:
        return self.client is not None

    def judge_pair(
        self,
        gt_img: np.ndarray,
        pred_img: np.ndarray,
        gt_box: List[float],
        pred_box: List[float],
        gt_label: str = "",
        pred_label: str = "",
    ) -> Dict[str, Any]:
        if not self.is_available():
            return {
                "accepted": bool(self.fail_open),
                "same_object": bool(self.fail_open),
                "orientation_consistent": bool(self.fail_open),
                "orientation_observable": False,
                "vlm_error": "vlm matcher unavailable",
            }

        montage = _make_vlm_pair_montage(
            gt_img=gt_img,
            pred_img=pred_img,
            gt_box=gt_box,
            pred_box=pred_box,
            gt_label=gt_label,
            pred_label=pred_label,
            tile_size=self.montage_tile_size,
        )
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(prefix="vlm_obj_match_", suffix=".jpg", delete=False) as tmp_f:
                tmp_path = tmp_f.name
            Image.fromarray(montage).save(tmp_path, quality=92)
            resp = vlm_client.call_with_retries(
                self.client,
                VLM_MATCH_PROMPT,
                tmp_path,
                self.max_retries,
                self.retry_sleep,
            )
        except Exception as e:
            return {
                "accepted": bool(self.fail_open),
                "same_object": bool(self.fail_open),
                "orientation_consistent": bool(self.fail_open),
                "orientation_observable": False,
                "same_object_confidence": 0.0,
                "orientation_confidence": 0.0,
                "vlm_error": f"{type(e).__name__}: {e}",
            }
        finally:
            if tmp_path is not None and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

        same_object = _bool_from_vlm(resp.get("same_object"), False)
        orientation_consistent = _bool_from_vlm(resp.get("orientation_consistent"), False)
        orientation_observable = _bool_from_vlm(resp.get("orientation_observable"), False)
        same_conf = _float_or_none(resp.get("same_object_confidence"))
        orient_conf = _float_or_none(resp.get("orientation_confidence"))
        if same_conf is None:
            same_conf = _float_or_none(resp.get("confidence"))
        if orient_conf is None:
            orient_conf = same_conf
        same_conf = float(np.clip(0.5 if same_conf is None else same_conf, 0.0, 1.0))
        orient_conf = float(np.clip(0.5 if orient_conf is None else orient_conf, 0.0, 1.0))
        accepted = bool(same_object and (orientation_consistent or not self.require_orientation))

        return {
            "accepted": accepted,
            "same_object": bool(same_object),
            "orientation_consistent": bool(orientation_consistent),
            "orientation_observable": bool(orientation_observable),
            "same_object_confidence": same_conf,
            "orientation_confidence": orient_conf,
            "object_category_gt": str(resp.get("object_category_gt") or ""),
            "object_category_pred": str(resp.get("object_category_pred") or ""),
            "reason": str(resp.get("reason") or resp.get("orientation_notes") or ""),
            "raw": resp,
        }

    def judge_image_matches(
        self,
        gt_img: np.ndarray,
        pred_img: np.ndarray,
        gt_boxes: List[List[float]],
        pred_boxes: List[List[float]],
        gt_labels: Optional[List[str]] = None,
        pred_labels: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        if not self.is_available():
            return {
                "matches": [],
                "rejected_matches": [],
                "vlm_error": "vlm matcher unavailable",
            }

        montage = _make_vlm_global_match_montage(
            gt_img=gt_img,
            pred_img=pred_img,
            gt_boxes=gt_boxes,
            pred_boxes=pred_boxes,
            gt_labels=gt_labels,
            pred_labels=pred_labels,
            tile_size=max(640, self.montage_tile_size * 2),
        )
        prompt = _build_vlm_global_match_prompt(gt_boxes, pred_boxes, gt_labels, pred_labels)
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(prefix="vlm_global_obj_match_", suffix=".jpg", delete=False) as tmp_f:
                tmp_path = tmp_f.name
            Image.fromarray(montage).save(tmp_path, quality=92)
            resp = vlm_client.call_with_retries(
                self.client,
                prompt,
                tmp_path,
                self.max_retries,
                self.retry_sleep,
            )
        except Exception as e:
            return {
                "matches": [],
                "rejected_matches": [],
                "vlm_error": f"{type(e).__name__}: {e}",
            }
        finally:
            if tmp_path is not None and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

        raw_matches = resp.get("matches", [])
        if isinstance(raw_matches, dict):
            raw_matches = list(raw_matches.values())
        if not isinstance(raw_matches, list):
            raw_matches = []

        accepted_candidates = []
        rejected_matches = []
        for raw_idx, item in enumerate(raw_matches):
            if not isinstance(item, dict):
                rejected_matches.append({"raw_index": raw_idx, "reason": "match entry is not an object", "raw": item})
                continue

            gt_idx = _parse_vlm_index(
                item.get("gt_id", item.get("gt", item.get("source_id", item.get("target_id")))),
                "G",
                len(gt_boxes),
            )
            pred_idx = _parse_vlm_index(
                item.get("pred_id", item.get("prediction_id", item.get("generated_id", item.get("candidate_id")))),
                "P",
                len(pred_boxes),
            )
            if gt_idx is None or pred_idx is None:
                rejected_matches.append({
                    "raw_index": raw_idx,
                    "reason": "invalid gt_id or pred_id",
                    "raw": item,
                })
                continue

            gt_label = normalize_detector_label(gt_labels[gt_idx]) if gt_labels and gt_idx < len(gt_labels) else ""
            pred_label = normalize_detector_label(pred_labels[pred_idx]) if pred_labels and pred_idx < len(pred_labels) else ""
            if not _labels_compatible_for_object_match(gt_label, pred_label):
                rejected_matches.append({
                    "raw_index": raw_idx,
                    "gt_idx": int(gt_idx),
                    "pred_idx": int(pred_idx),
                    "gt_label": gt_label,
                    "pred_label": pred_label,
                    "reason": f"label_incompatible:{gt_label}!={pred_label}",
                    "raw": item,
                })
                continue

            same_object = _bool_from_vlm(item.get("same_object", True), True)
            if not same_object:
                rejected_matches.append({
                    "raw_index": raw_idx,
                    "gt_idx": int(gt_idx),
                    "pred_idx": int(pred_idx),
                    "reason": "same_object=false",
                    "raw": item,
                })
                continue

            same_conf = _float_or_none(
                item.get("same_object_confidence", item.get("confidence", item.get("match_confidence")))
            )
            same_conf = float(np.clip(0.5 if same_conf is None else same_conf, 0.0, 1.0))
            if same_conf < self.min_match_confidence:
                rejected_matches.append({
                    "raw_index": raw_idx,
                    "gt_idx": int(gt_idx),
                    "pred_idx": int(pred_idx),
                    "reason": f"same_object_confidence_below_threshold:{same_conf:.3f}<{self.min_match_confidence:.3f}",
                    "raw": item,
                })
                continue
            orient_conf = _float_or_none(item.get("orientation_confidence"))
            orient_conf = float(np.clip(same_conf if orient_conf is None else orient_conf, 0.0, 1.0))
            orientation_consistent = _bool_from_vlm(item.get("orientation_consistent"), True)
            orientation_observable = _bool_from_vlm(item.get("orientation_observable"), False)
            accepted_candidates.append({
                "raw_index": int(raw_idx),
                "gt_idx": int(gt_idx),
                "pred_idx": int(pred_idx),
                "same_object": True,
                "orientation_consistent": bool(orientation_consistent),
                "orientation_observable": bool(orientation_observable),
                "same_object_confidence": same_conf,
                "orientation_confidence": orient_conf,
                "object_category_gt": str(item.get("object_category_gt") or item.get("gt_category") or ""),
                "object_category_pred": str(item.get("object_category_pred") or item.get("pred_category") or ""),
                "reason": str(item.get("reason") or ""),
                "raw": item,
            })

        accepted_candidates.sort(
            key=lambda x: (float(x["same_object_confidence"]), float(x["orientation_confidence"])),
            reverse=True,
        )
        used_gt, used_pred = set(), set()
        matches = []
        for item in accepted_candidates:
            gt_idx = int(item["gt_idx"])
            pred_idx = int(item["pred_idx"])
            if gt_idx in used_gt or pred_idx in used_pred:
                rejected_matches.append({
                    "raw_index": item["raw_index"],
                    "gt_idx": gt_idx,
                    "pred_idx": pred_idx,
                    "reason": "duplicate gt_id or pred_id after one-to-one filtering",
                    "raw": item.get("raw"),
                })
                continue
            used_gt.add(gt_idx)
            used_pred.add(pred_idx)
            matches.append(item)

        return {
            "matches": matches,
            "rejected_matches": rejected_matches,
            "raw": resp,
            "raw_match_count": len(raw_matches),
            "unmatched_gt": resp.get("unmatched_gt", []),
            "unmatched_pred": resp.get("unmatched_pred", []),
        }


# ==========================================
# Core tools API
# ==========================================
class SemanticToolsAPI:
    """
    Robust model tools layer:
    1) Graceful degradation when DINO/SAM3/DA3 loading fails;
    2) SAM3 and DINO ROI feature caching to avoid redundant calls;
    3) DA3 circuit breaker after repeated failures.
    """

    def __init__(
        self,
        device: torch.device,
        da3_src_path: Optional[str] = None,
        da3_model_path: Optional[str] = None,
        sam3_checkpoint_path: str = os.environ.get("SAM3_CHECKPOINT", "checkpoints/sam3.pt"),
        dino_repo_dir: str = os.environ.get("DINOV3_CODE_DIR", "third_party/dinov3"),
        dino_weight_path: str = os.environ.get("DINOV3_MODEL", "checkpoints/dinov3-vitl16/model.safetensors"),
        enable_sam3: bool = True,
        enable_dino: bool = True,
        enable_da3: bool = True,
        da3_max_failures: int = 3,
        cache_size: int = 256,
        proposal_mode: str = "text",
        grounding_dino_path: Optional[str] = None,
        enable_grounding_dino: bool = True,
        gdino_box_threshold: float = 0.25,
        gdino_text_threshold: float = 0.25,
    ):
        self.device = device
        self.proposal_mode = str(proposal_mode or "text").lower()
        # SAM3 image/text stages may have inconsistent dtype behavior across versions:
        # set_image works with bf16 autocast; set_text_prompt falls back to fp32 to avoid mat1/mat2 dtype mismatch.
        self.sam3_image_dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.sam3_text_dtype = torch.float32
        self.sam3_failures = 0
        self.sam3_max_failures = 3
        # Visualization / speed filtering for SAM3 proposals.
        # Lower threshold gives more masks but noisy and slow; 0.08~0.12 is a practical range.
        self.sam3_score_tau = 0.08
        self.sam3_max_instances = 80
        self.sam3_min_mask_area = 20
        # Prompt-free fallback proposals. This is intentionally dependency-light:
        # OpenCV color/edge proposals are weaker than SAM automatic masks, but they
        # let the benchmark avoid a fixed object vocabulary.
        self.auto_max_instances = 45
        self.auto_min_mask_area = 80
        self.auto_max_mask_area_ratio = 0.32
        self.auto_min_fill_ratio = 0.10
        self.auto_kmeans_clusters = 10
        self.gdino_path = grounding_dino_path
        self.enable_grounding_dino = enable_grounding_dino
        self.gdino_box_threshold = float(gdino_box_threshold)
        self.gdino_text_threshold = float(gdino_text_threshold)
        self.gdino_max_instances = 60
        self.gdino_min_box_area = 36
        self.gdino_max_box_area_ratio = 0.65

        self.da3_src_path = da3_src_path
        self.da3_model_path = da3_model_path
        self.sam3_checkpoint_path = sam3_checkpoint_path
        self.dino_repo_dir = dino_repo_dir
        self.dino_weight_path = dino_weight_path
        self.enable_sam3 = enable_sam3
        self.enable_dino = enable_dino
        self.enable_da3 = enable_da3
        self.da3_max_failures = da3_max_failures
        self.da3_failures = 0
        self.cache_size = cache_size

        self.sam3_processor = None
        self.sam3_model = None

        self.dino_model = None
        self.gdino_processor = None
        self.gdino_model = None
        self.da3 = None
        self.da3_load_error = ""
        self.model_status: Dict[str, str] = {}
        self.sam3_cache: OrderedDict[str, Tuple[List[List[int]], List[np.ndarray]]] = OrderedDict()
        self.auto_proposal_cache: OrderedDict[str, Tuple[List[List[int]], List[np.ndarray]]] = OrderedDict()
        self.gdino_cache: OrderedDict[str, List[dict]] = OrderedDict()
        self.roi_feature_cache: OrderedDict[str, Optional[torch.Tensor]] = OrderedDict()

        self.dino_transform = T.Compose([
            T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        if self.enable_dino:
            print("Loading DINOv3 feature model...")
            self.dino_model = self._load_dino_backbone()
            if self.dino_model is not None:
                try:
                    self.dino_model = self.dino_model.float().eval()
                    self.model_status["dino"] = "ok"
                except Exception as e:
                    print(f"DINOv3 eval/float failed, disabling feature verification: {e}")
                    self.dino_model = None
                    self.model_status["dino"] = f"disabled: {e}"
            else:
                self.model_status["dino"] = "disabled: load failed"
        else:
            self.model_status["dino"] = "disabled by config"

        if self.enable_sam3:
            print("Loading SAM3 model...")
            self.sam3_processor = self._load_sam3_processor()
            self.model_status["sam3"] = "ok" if self.sam3_processor is not None else "disabled: load failed"
        else:
            self.model_status["sam3"] = "disabled by config"

        gdino_like_mode = self.proposal_mode in {
            "gdino",
            "gdino_box",
            "grounding_dino",
            "grounding_dino_box",
            "gdino_sam3",
            "grounding_dino_sam3",
        }
        if self.enable_grounding_dino and gdino_like_mode:
            print("Loading GroundingDINO detection model...")
            self.gdino_processor, self.gdino_model = self._load_grounding_dino()
            self.model_status["grounding_dino"] = "ok" if self.gdino_model is not None else "disabled: load failed"
        elif gdino_like_mode:
            self.model_status["grounding_dino"] = "disabled by config"
        else:
            self.model_status["grounding_dino"] = "not requested"

        if self.enable_da3:
            print("Loading Depth Anything 3...")
            self.da3 = self._load_da3_model()
            self.model_status["da3"] = "ok" if self.da3 is not None else f"disabled: load failed: {self.da3_load_error or 'unknown error'}"
        else:
            self.model_status["da3"] = "disabled by config"

        self.model_status["proposal_mode"] = self.proposal_mode
        print("Model status:", json.dumps(self.model_status, ensure_ascii=False))

    def _cache_get(self, cache: OrderedDict, key: str):
        if key in cache:
            cache.move_to_end(key)
            return cache[key]
        return None

    def _cache_put(self, cache: OrderedDict, key: str, value):
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > self.cache_size:
            cache.popitem(last=False)

    def _image_hash(self, img_cv2: np.ndarray) -> str:
        arr = np.ascontiguousarray(img_cv2)
        h = hashlib.blake2b(arr.view(np.uint8), digest_size=12).hexdigest()
        return f"{arr.shape[0]}x{arr.shape[1]}x{arr.shape[2]}_{h}"

    def _find_checkpoint(self, path: Optional[str], candidates: Tuple[str, ...]) -> Optional[str]:
        if not path:
            return None
        if os.path.isfile(path):
            return path
        if os.path.isdir(path):
            for name in candidates:
                cand = os.path.join(path, name)
                if os.path.isfile(cand):
                    return cand
        return None

    def _load_state_dict_file(self, ckpt_path: str) -> dict:
        if ckpt_path.endswith(".safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(ckpt_path)
        else:
            obj = torch.load(ckpt_path, map_location="cpu")
            if isinstance(obj, dict) and "state_dict" in obj:
                state_dict = obj["state_dict"]
            elif isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
                state_dict = obj["model"]
            else:
                state_dict = obj
        if not isinstance(state_dict, dict):
            raise TypeError(f"checkpoint is not a dict: {type(state_dict)}")
        return {str(k).replace("module.", ""): v for k, v in state_dict.items()}

    def _build_sam3_processor(self, model):
        """
        Initialize SAM3 Processor compatible with different versions.
        Explicitly lowers threshold if the current version supports it; otherwise safely ignored.
        """
        params = _safe_signature_params(Sam3Processor)
        kwargs = {}

        if "device" in params:
            kwargs["device"] = self.device
        if "confidence_threshold" in params:
            kwargs["confidence_threshold"] = 0.05
        if "score_threshold" in params:
            kwargs["score_threshold"] = 0.05
        if "threshold" in params:
            kwargs["threshold"] = 0.05
        if "mask_threshold" in params:
            kwargs["mask_threshold"] = 0.0

        try:
            logger.debug("SAM3 Sam3Processor signature: %s", inspect.signature(Sam3Processor))
        except Exception:
            pass
        logger.debug("SAM3 Sam3Processor kwargs = %s", kwargs)

        processor = Sam3Processor(model, **kwargs)

        # In some versions threshold is an instance attribute, not an __init__ parameter.
        for attr, value in [
            ("confidence_threshold", 0.05),
            ("score_threshold", 0.05),
            ("threshold", 0.05),
            ("mask_threshold", 0.0),
        ]:
            if hasattr(processor, attr):
                try:
                    setattr(processor, attr, value)
                    logger.debug("SAM3 set processor.%s = %s", attr, value)
                except Exception:
                    pass

        return processor

    def _load_sam3_processor(self):
        if build_sam3_image_model is None or Sam3Processor is None:
            print(f"SAM3 import failed, skipping SAM3: {_SAM3_IMPORT_ERROR}")
            return None

        ckpt_path = self._find_checkpoint(
            self.sam3_checkpoint_path,
            candidates=("sam3.pt", "sam3.pth", "model.safetensors", "pytorch_model.bin"),
        )
        if ckpt_path is None:
            print(f"SAM3 checkpoint not found, skipping SAM3: {self.sam3_checkpoint_path}")
            return None

        # Prefer offline checkpoint_path via official builder; do not gate build on load_from_HF.
        try:
            kwargs = {}
            params = _safe_signature_params(build_sam3_image_model)
            if "checkpoint_path" in params:
                kwargs["checkpoint_path"] = ckpt_path
            if "load_from_HF" in params:
                kwargs["load_from_HF"] = False

            model = build_sam3_image_model(**kwargs)
            model = model.to(device=self.device, dtype=torch.float32).eval()
            self.sam3_model = model

            print(f"SAM3 builder offline load succeeded: {ckpt_path}, init dtype=float32")
            return self._build_sam3_processor(self.sam3_model)

        except Exception as e:
            print(f"SAM3 builder load failed, trying empty model + manual weights: {e}")

        # Fallback: if builder does not support checkpoint_path, build empty model then load weights manually.
        try:
            kwargs = {}
            params = _safe_signature_params(build_sam3_image_model)
            if "load_from_HF" in params:
                kwargs["load_from_HF"] = False

            model = build_sam3_image_model(**kwargs)
            state_dict = self._load_state_dict_file(ckpt_path)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print(f"SAM3 manual weight load done: missing={len(missing)}, unexpected={len(unexpected)}")
            if len(missing) > 500 or len(unexpected) > 500:
                print("SAM3 many missing/unexpected keys; likely weight/code version mismatch.")

            model = model.to(device=self.device, dtype=torch.float32).eval()
            self.sam3_model = model
            print("SAM3 manual load initialized as float32")
            return self._build_sam3_processor(self.sam3_model)

        except Exception as e:
            print(f"SAM3 load completely failed, disabling semantic segmentation: {e}")
            traceback.print_exc(limit=2)
            self.sam3_model = None
            return None

    def _load_dino_backbone(self):
        # Preferred path: a Hugging Face-native DINOv3 checkpoint -- a directory
        # containing config.json + model.safetensors with architecture
        # "DINOv3ViTModel". These load through transformers.AutoModel and need no
        # local hub source repo. The forward output is a BaseModelOutput whose
        # last_hidden_state/pooler_output is consumed by _select_feature_tensor.
        hf_dir = self.dino_weight_path
        if os.path.isfile(hf_dir):
            hf_dir = os.path.dirname(hf_dir)
        if os.path.isdir(hf_dir) and os.path.isfile(os.path.join(hf_dir, "config.json")):
            try:
                from transformers import AutoModel

                model = AutoModel.from_pretrained(hf_dir, local_files_only=True)
                print(f"DINOv3 HF load succeeded: {hf_dir}")
                return model.to(self.device).eval()
            except Exception as e:
                print(f"DINOv3 HF load failed, trying torch.hub local source: {e}")
                traceback.print_exc(limit=2)

        # Fallback: offline torch.hub load from a local dinov3 source repo.
        weight_path = self._find_checkpoint(self.dino_weight_path, candidates=("model.safetensors", "pytorch_model.bin", "model.pth"))
        if weight_path is None:
            print(f"DINOv3 weights not found: {self.dino_weight_path}")
            return None
        if not os.path.isdir(self.dino_repo_dir):
            print(f"DINOv3 local source directory not found: {self.dino_repo_dir}")
            return None

        print(f"   -> DINO repo: {self.dino_repo_dir}")
        print(f"   -> DINO weight: {weight_path}")
        try:
            model = torch.hub.load(self.dino_repo_dir, "dinov3_vitl16", source="local", pretrained=False)
            state_dict = self._load_state_dict_file(weight_path)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print(f"DINOv3 offline load done: missing={len(missing)}, unexpected={len(unexpected)}")
            return model.to(self.device).eval()
        except Exception as e:
            print(f"DINOv3 offline load failed, falling back to IoU-only verification: {e}")
            traceback.print_exc(limit=2)
            return None

    def _load_grounding_dino(self):
        if not self.gdino_path:
            print("GroundingDINO path is empty, gdino mode will degrade automatically.")
            return None, None
        if not os.path.exists(self.gdino_path):
            print(f"GroundingDINO path not found: {self.gdino_path}")
            return None, None

        try:
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except Exception as e:
            print(f"transformers GroundingDINO dependency import failed: {e}")
            return None, None

        try:
            processor = AutoProcessor.from_pretrained(
                self.gdino_path,
                local_files_only=True,
            )
            model = AutoModelForZeroShotObjectDetection.from_pretrained(
                self.gdino_path,
                local_files_only=True,
            ).to(self.device).eval()
            print(f"GroundingDINO local load succeeded: {self.gdino_path}")
            return processor, model
        except Exception as e:
            print(f"GroundingDINO load failed: {e}")
            traceback.print_exc(limit=2)
            return None, None

    def _load_da3_model(self):
        try:
            if self.da3_src_path and self.da3_src_path not in sys.path:
                sys.path.insert(0, self.da3_src_path)
            from depth_anything_3.api import DepthAnything3
        except Exception as e:
            self.da3_load_error = f"import failed: {_short_error_text(e)}"
            print(f"DA3 import failed, skipping DA3 metrics: {e}")
            return None

        load_errors = []
        try:
            if self.da3_model_path:
                params = _safe_signature_params(DepthAnything3.from_pretrained)
                kwargs = {}
                if "local_files_only" in params:
                    kwargs["local_files_only"] = True
                model = DepthAnything3.from_pretrained(self.da3_model_path, **kwargs).to(self.device).eval()
                print(f"DA3 from_pretrained local load succeeded: {self.da3_model_path}")
                return model
        except Exception as e:
            load_errors.append(f"from_pretrained({self.da3_model_path}) -> {_short_error_text(e)}")

        # Fallback preset mode. May still trigger internal lookup; disabled on failure to avoid blocking.
        try:
            preset_name = os.path.basename(str(self.da3_model_path).rstrip("/")) if self.da3_model_path else None
            if preset_name:
                model = DepthAnything3(model_name=preset_name).to(self.device).eval()
            else:
                model = DepthAnything3().to(self.device).eval()
            print(f"DA3 preset load succeeded: {preset_name or 'default'}")
            return model
        except Exception as e:
            load_errors.append(f"preset/default -> {_short_error_text(e)}")

        self.da3_load_error = " | ".join(load_errors)
        print("DA3 load failed, skipping DA3 metrics: " + self.da3_load_error)
        return None

    def has_dino(self) -> bool:
        return self.dino_model is not None

    def has_sam3(self) -> bool:
        return self.sam3_processor is not None

    def has_grounding_dino(self) -> bool:
        return self.gdino_processor is not None and self.gdino_model is not None

    def has_da3(self) -> bool:
        return self.da3 is not None and self.da3_failures < self.da3_max_failures
    
    def _tree_to_float_dtype(self, obj, dtype=torch.float32):
        """
        Recursively cast floating-point tensors in SAM3 inference_state to the specified dtype.
        Prevents bf16 state from set_image causing dtype mismatch in set_text_prompt fp32 branch.
        """
        if isinstance(obj, torch.Tensor):
            if obj.is_floating_point():
                return obj.to(dtype=dtype)
            return obj

        if isinstance(obj, dict):
            for k, v in obj.items():
                obj[k] = self._tree_to_float_dtype(v, dtype=dtype)
            return obj

        if isinstance(obj, list):
            for i, v in enumerate(obj):
                obj[i] = self._tree_to_float_dtype(v, dtype=dtype)
            return obj

        if isinstance(obj, tuple):
            return tuple(self._tree_to_float_dtype(v, dtype=dtype) for v in obj)

        # Handle custom state/dataclass objects
        if hasattr(obj, "__dict__"):
            for k, v in vars(obj).items():
                try:
                    setattr(obj, k, self._tree_to_float_dtype(v, dtype=dtype))
                except Exception:
                    pass
            return obj

        return obj

    def _split_first_dim(self, x) -> list:
        if x is None:
            return []
        if isinstance(x, torch.Tensor):
            if x.ndim == 0:
                return []
            return [x[i] for i in range(x.shape[0])]
        if isinstance(x, np.ndarray):
            if x.ndim == 0:
                return []
            if x.ndim == 1 and x.shape[0] == 4:
                return [x]
            return [x[i] for i in range(x.shape[0])]
        if isinstance(x, (list, tuple)):
            return list(x)
        return []

    def _get_output_field(self, output, names, default=None):
        if isinstance(output, dict):
            for name in names:
                if name in output and output[name] is not None:
                    return output[name]
            return default

        for name in names:
            if hasattr(output, name):
                value = getattr(output, name)
                if value is not None:
                    return value
        return default

    def _debug_sam3_output(self, output, prompt: str):
        if not logger.isEnabledFor(logging.DEBUG):
            return
        try:
            if isinstance(output, dict):
                logger.debug("SAM3 prompt=%r, keys=%s", prompt, list(output.keys()))
            else:
                logger.debug("SAM3 prompt=%r, output_type=%s", prompt, type(output))

            for k in ["scores", "boxes", "masks", "masks_logits"]:
                v = self._get_output_field(output, [k], None)
                if isinstance(v, torch.Tensor):
                    msg = f"SAM3 [{prompt}] {k}: shape={tuple(v.shape)}, dtype={v.dtype}"
                    if v.numel() > 0 and v.is_floating_point():
                        vv = v.detach().float()
                        msg += f", min={vv.min().item():.4f}, max={vv.max().item():.4f}, mean={vv.mean().item():.4f}"
                    logger.debug(msg)
                else:
                    logger.debug("SAM3 [%s] %s: type=%s", prompt, k, type(v))
        except Exception as e:
            logger.debug("SAM3 output debug failed for prompt=%r: %s", prompt, e)

    def _call_sam3_text_prompt(self, inference_state, prompt: str):
        """
        Compatible with different set_text_prompt(state=..., prompt=...) signatures;
        explicitly lowers threshold at call time if the current version supports it.
        """
        params = _safe_signature_params(self.sam3_processor.set_text_prompt)
        kwargs = {"state": inference_state, "prompt": prompt}

        if "confidence_threshold" in params:
            kwargs["confidence_threshold"] = 0.05
        if "score_threshold" in params:
            kwargs["score_threshold"] = 0.05
        if "threshold" in params:
            kwargs["threshold"] = 0.05
        if "mask_threshold" in params:
            kwargs["mask_threshold"] = 0.0

        return self.sam3_processor.set_text_prompt(**kwargs)

    def get_instances_from_text(
        self,
        img_cv2: np.ndarray,
        text_prompts: List[str],
        return_labels: bool = False,
    ):
        if self.sam3_processor is None:
            return ([], [], []) if return_labels else ([], [])
        if self.sam3_failures >= self.sam3_max_failures:
            return ([], [], []) if return_labels else ([], [])

        prompts = [str(p).strip() for p in text_prompts if str(p).strip()]
        if len(prompts) == 0:
            return ([], [], []) if return_labels else ([], [])

        # Do not concatenate prompts into one long sentence. SAM3 text prompt is more stable when called per class.
        prompt_str = " | ".join(prompts)
        cache_key = f"sam3:{self._image_hash(img_cv2)}:{prompt_str}"
        cached = self._cache_get(self.sam3_cache, cache_key)
        if cached is not None:
            boxes, masks = cached[:2]
            labels = cached[2] if len(cached) >= 3 else [""] * len(boxes)
            out = ([b[:] for b in boxes], [m.copy() for m in masks], list(labels))
            return out if return_labels else out[:2]

        img_rgb = _as_rgb_uint8(img_cv2)
        pil_img = Image.fromarray(img_rgb)

        all_raw_boxes = []
        all_raw_masks = []
        all_raw_scores = []
        all_raw_prompt_names = []

        try:
            with torch.inference_mode():
                if self.device.type == "cuda":
                    # Phase 1: image encoder in bf16 autocast.
                    if self.sam3_model is not None:
                        self.sam3_model.to(dtype=self.sam3_image_dtype)

                    with torch.autocast(device_type="cuda", dtype=self.sam3_image_dtype):
                        inference_state = self.sam3_processor.set_image(pil_img)

                    # Phase 2: text prompt / decoder in fp32 to avoid dtype mismatch.
                    if self.sam3_model is not None:
                        self.sam3_model.to(dtype=self.sam3_text_dtype)

                    inference_state = self._tree_to_float_dtype(
                        inference_state,
                        dtype=self.sam3_text_dtype,
                    )

                    if not hasattr(self, "_sam3_signature_printed"):
                        try:
                            logger.debug(
                                "SAM3 set_text_prompt signature: %s",
                                inspect.signature(self.sam3_processor.set_text_prompt),
                            )
                        except Exception:
                            pass
                        self._sam3_signature_printed = True

                    for p in prompts:
                        with torch.autocast(device_type="cuda", enabled=False):
                            output = self._call_sam3_text_prompt(inference_state, p)

                        self._debug_sam3_output(output, p)

                        raw_boxes_i = self._get_output_field(output, ["boxes", "pred_boxes", "box"], [])
                        raw_masks_i = self._get_output_field(output, ["masks", "pred_masks", "mask"], [])
                        raw_scores_i = self._get_output_field(output, ["scores", "score"], [])

                        raw_boxes_i = self._split_first_dim(raw_boxes_i)
                        raw_masks_i = self._split_first_dim(raw_masks_i)
                        raw_scores_i = self._split_first_dim(raw_scores_i)

                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(
                                "SAM3 prompt=%r, raw_boxes=%d, raw_masks=%d, raw_scores=%d",
                                p,
                                len(raw_boxes_i),
                                len(raw_masks_i),
                                len(raw_scores_i),
                            )

                        m = min(len(raw_boxes_i), len(raw_masks_i))
                        for bi in range(m):
                            score = 1.0
                            if bi < len(raw_scores_i):
                                try:
                                    score = float(_to_numpy(raw_scores_i[bi]).reshape(-1)[0])
                                except Exception:
                                    score = 1.0
                            all_raw_boxes.append(raw_boxes_i[bi])
                            all_raw_masks.append(raw_masks_i[bi])
                            all_raw_scores.append(score)
                            all_raw_prompt_names.append(p)

                else:
                    inference_state = self.sam3_processor.set_image(pil_img)
                    for p in prompts:
                        output = self._call_sam3_text_prompt(inference_state, p)
                        self._debug_sam3_output(output, p)

                        raw_boxes_i = self._get_output_field(output, ["boxes", "pred_boxes", "box"], [])
                        raw_masks_i = self._get_output_field(output, ["masks", "pred_masks", "mask"], [])
                        raw_scores_i = self._get_output_field(output, ["scores", "score"], [])

                        raw_boxes_i = self._split_first_dim(raw_boxes_i)
                        raw_masks_i = self._split_first_dim(raw_masks_i)
                        raw_scores_i = self._split_first_dim(raw_scores_i)

                        m = min(len(raw_boxes_i), len(raw_masks_i))
                        for bi in range(m):
                            score = 1.0
                            if bi < len(raw_scores_i):
                                try:
                                    score = float(_to_numpy(raw_scores_i[bi]).reshape(-1)[0])
                                except Exception:
                                    score = 1.0
                            all_raw_boxes.append(raw_boxes_i[bi])
                            all_raw_masks.append(raw_masks_i[bi])
                            all_raw_scores.append(score)
                            all_raw_prompt_names.append(p)

        except Exception as e:
            self.sam3_failures += 1
            print(f"SAM3 inference failed ({self.sam3_failures}/{self.sam3_max_failures}), skipping instance extraction for this image: {e}")
            traceback.print_exc(limit=2)

            if self.sam3_failures >= self.sam3_max_failures:
                print("SAM3 consecutive failures exceeded limit, auto-skipping SAM3 for remaining samples.")
                self.sam3_processor = None
                self.sam3_model = None

            return [], []

        n = min(len(all_raw_boxes), len(all_raw_masks))
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "SAM3 total prompts=%d, total_raw_boxes=%d, total_raw_masks=%d, n=%d",
                len(prompts),
                len(all_raw_boxes),
                len(all_raw_masks),
                n,
            )

        h, w = img_cv2.shape[:2]
        parsed_items = []
        score_tau = getattr(self, "sam3_score_tau", 0.08)
        max_sam3_instances = getattr(self, "sam3_max_instances", 80)
        min_mask_area = getattr(self, "sam3_min_mask_area", 20)

        for i in range(n):
            try:
                score = float(all_raw_scores[i]) if i < len(all_raw_scores) else 1.0
                prompt_name = all_raw_prompt_names[i] if i < len(all_raw_prompt_names) else ""

                box_arr = _to_numpy(all_raw_boxes[i]).reshape(-1)
                if box_arr.size < 4:
                    continue
                sb = safe_box_xyxy(box_arr[:4].tolist(), h, w)
                if sb is None:
                    continue

                mask_np = _to_numpy(all_raw_masks[i]).squeeze()
                if mask_np.ndim != 2:
                    continue
                if mask_np.shape != (h, w):
                    mask_np = cv2.resize(mask_np.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)

                mask_uint8 = ((mask_np > 0.5).astype(np.uint8) * 255)
                area = int((mask_uint8 > 0).sum())
                if area < min_mask_area:
                    continue

                parsed_items.append({
                    "score": score,
                    "area": area,
                    "box": sb,
                    "mask": mask_uint8,
                    "prompt": prompt_name,
                })

            except Exception as e:
                print(f"SAM3 output parsing failed, skipping instance {i}: {e}")
                continue

        if len(parsed_items) == 0:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("SAM3 valid_boxes=0, valid_masks=0 after parsing")
            self._cache_put(self.sam3_cache, cache_key, ([], [], []))
            return ([], [], []) if return_labels else ([], [])

        # Filter by score. If threshold removes everything, fallback to top-k of all parsed proposals.
        score_filtered = [x for x in parsed_items if x["score"] >= score_tau]
        if len(score_filtered) == 0:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "SAM3 score filter removed all instances; fallback to top-k. score_tau=%s",
                    score_tau,
                )
            score_filtered = parsed_items

        # Prefer high-confidence/large proposals, then remove duplicate hits from different prompts.
        score_filtered.sort(key=lambda x: (x["score"], x["area"]), reverse=True)
        pre_dedup_count = len(score_filtered)
        deduped_items = []
        for item in score_filtered:
            is_duplicate = False
            for kept in deduped_items:
                if (
                    _mask_iou_binary(item["mask"], kept["mask"]) >= 0.75
                    or _box_iou_xyxy(item["box"], kept["box"]) >= 0.90
                ):
                    is_duplicate = True
                    break
            if is_duplicate:
                continue
            deduped_items.append(item)
            if len(deduped_items) >= max_sam3_instances:
                break
        score_filtered = deduped_items

        valid_boxes = [x["box"] for x in score_filtered]
        valid_masks = [x["mask"] for x in score_filtered]
        valid_labels = [normalize_detector_label(x.get("prompt", "")) for x in score_filtered]

        if logger.isEnabledFor(logging.DEBUG):
            scores_np = np.array([x["score"] for x in score_filtered], dtype=np.float32)
            areas_np = np.array([x["area"] for x in score_filtered], dtype=np.float32)
            logger.debug(
                "SAM3 valid_boxes=%d, valid_masks=%d, "
                "score_range=(%.4f,%.4f), area_range=(%.0f,%.0f), "
                "score_tau=%s, dedup=%d->%d, max_instances=%d",
                len(valid_boxes),
                len(valid_masks),
                scores_np.min(),
                scores_np.max(),
                areas_np.min(),
                areas_np.max(),
                score_tau,
                pre_dedup_count,
                len(score_filtered),
                max_sam3_instances,
            )

        self._cache_put(
            self.sam3_cache,
            cache_key,
            ([b[:] for b in valid_boxes], [m.copy() for m in valid_masks], list(valid_labels)),
        )
        return (valid_boxes, valid_masks, valid_labels) if return_labels else (valid_boxes, valid_masks)

    def _rect_masks_from_boxes(
        self,
        boxes: List[List[int]],
        shape_hw: Tuple[int, int],
    ) -> List[np.ndarray]:
        h, w = shape_hw
        masks = []
        for box in boxes:
            sb = safe_box_xyxy(box, h, w)
            if sb is None:
                continue
            x1, y1, x2, y2 = sb
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[y1:y2 + 1, x1:x2 + 1] = 255
            masks.append(mask)
        return masks

    def _clean_gdino_prompts(self, text_prompts: Optional[List[str]]) -> List[str]:
        prompts = []
        for p in text_prompts or []:
            p = str(p).strip().lower()
            if not p:
                continue
            if p in {"wall", "floor", "ceiling", "sky"}:
                continue
            if p not in prompts:
                prompts.append(p)
        return prompts

    def get_instances_grounding_dino(
        self,
        img_cv2: np.ndarray,
        text_prompts: Optional[List[str]],
        return_labels: bool = False,
    ):
        """GroundingDINO bbox proposals. Masks are rectangular fallback masks."""
        if img_cv2 is None:
            return ([], [], []) if return_labels else ([], [])
        if not self.has_grounding_dino():
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("GroundingDINO unavailable; fallback to auto proposal")
            boxes, masks = self.get_instances_auto(img_cv2)
            labels = [""] * len(boxes)
            return (boxes, masks, labels) if return_labels else (boxes, masks)

        prompts = self._clean_gdino_prompts(text_prompts)
        if len(prompts) == 0:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("GroundingDINO prompts empty; fallback to auto proposal")
            boxes, masks = self.get_instances_auto(img_cv2)
            labels = [""] * len(boxes)
            return (boxes, masks, labels) if return_labels else (boxes, masks)

        img_rgb = _as_rgb_uint8(img_cv2)
        h, w = img_rgb.shape[:2]
        query = " . ".join(prompts) + " ."
        query_hash = hashlib.blake2b(query.encode("utf-8"), digest_size=8).hexdigest()
        cache_key = f"gdino:{self._image_hash(img_cv2)}:{query_hash}"
        cached = self._cache_get(self.gdino_cache, cache_key)
        if cached is not None:
            boxes = [item["box"][:] for item in cached]
            masks = self._rect_masks_from_boxes(boxes, (h, w))
            labels = [normalize_detector_label(item.get("label", "")) for item in cached]
            return (boxes, masks, labels) if return_labels else (boxes, masks)

        try:
            pil_img = Image.fromarray(img_rgb)
            inputs = self.gdino_processor(
                images=pil_img,
                text=query,
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                outputs = self.gdino_model(**inputs)

            post_fn = self.gdino_processor.post_process_grounded_object_detection
            params = _safe_signature_params(post_fn)
            kwargs = {
                "outputs": outputs,
                "target_sizes": [(h, w)],
            }
            if "input_ids" in params:
                kwargs["input_ids"] = inputs.get("input_ids")
            if "box_threshold" in params:
                kwargs["box_threshold"] = self.gdino_box_threshold
            if "text_threshold" in params:
                kwargs["text_threshold"] = self.gdino_text_threshold

            try:
                results = post_fn(**kwargs)[0]
            except TypeError:
                results = post_fn(outputs, inputs["input_ids"], target_sizes=[(h, w)])[0]

            raw_boxes = results.get("boxes", [])
            raw_scores = results.get("scores", [])
            raw_labels = results.get("labels", [])
            boxes_np = _to_numpy(raw_boxes).reshape(-1, 4) if len(raw_boxes) > 0 else np.zeros((0, 4), dtype=np.float32)
            scores_np = _to_numpy(raw_scores).reshape(-1) if len(raw_scores) > 0 else np.ones((boxes_np.shape[0],), dtype=np.float32)

            min_area = max(int(self.gdino_min_box_area), int(0.0003 * h * w))
            max_area = max(min_area + 1, int(self.gdino_max_box_area_ratio * h * w))
            items = []
            for i, box_arr in enumerate(boxes_np):
                score = float(scores_np[i]) if i < len(scores_np) else 1.0
                if score < self.gdino_box_threshold:
                    continue
                sb = safe_box_xyxy(box_arr[:4].tolist(), h, w)
                if sb is None:
                    continue
                area = _box_area_xyxy(sb)
                if area < min_area or area > max_area:
                    continue
                label = raw_labels[i] if i < len(raw_labels) else ""
                label = str(label).strip().rstrip(".").lower()
                items.append({
                    "score": score,
                    "box": sb,
                    "label": label,
                    "source": "gdino",
                })

            items.sort(key=lambda x: (x["score"], _box_area_xyxy(x["box"])), reverse=True)
            deduped = []
            for item in items:
                if any(_box_iou_xyxy(item["box"], kept["box"]) >= 0.88 for kept in deduped):
                    continue
                deduped.append(item)
                if len(deduped) >= self.gdino_max_instances:
                    break

            if logger.isEnabledFor(logging.DEBUG):
                labels = [x["label"] for x in deduped[:10]]
                logger.debug(
                    "GDINO prompts=%d, raw=%d, valid=%d, score_tau=%s, labels=%s",
                    len(prompts),
                    len(boxes_np),
                    len(deduped),
                    self.gdino_box_threshold,
                    labels,
                )

            self._cache_put(self.gdino_cache, cache_key, [dict(x) for x in deduped])
            boxes = [x["box"] for x in deduped]
            masks = self._rect_masks_from_boxes(boxes, (h, w))
            labels = [normalize_detector_label(x.get("label", "")) for x in deduped]
            return (boxes, masks, labels) if return_labels else (boxes, masks)
        except Exception as e:
            print(f"GroundingDINO inference failed, falling back to auto proposal: {e}")
            traceback.print_exc(limit=2)
            boxes, masks = self.get_instances_auto(img_cv2)
            labels = [""] * len(boxes)
            return (boxes, masks, labels) if return_labels else (boxes, masks)

    def _call_sam3_box_prompt(self, inference_state, box_cxcywh_norm: List[float]):
        return self.sam3_processor.add_geometric_prompt(
            box=box_cxcywh_norm,
            label=True,
            state=inference_state,
        )

    def _reset_sam3_prompts(self, inference_state):
        if hasattr(self.sam3_processor, "reset_all_prompts"):
            self.sam3_processor.reset_all_prompts(inference_state)
            return
        if isinstance(inference_state, dict):
            if "backbone_out" in inference_state and isinstance(inference_state["backbone_out"], dict):
                for key in ["language_features", "language_mask", "language_embeds"]:
                    inference_state["backbone_out"].pop(key, None)
            for key in ["geometric_prompt", "boxes", "masks", "masks_logits", "scores"]:
                inference_state.pop(key, None)

    def get_instances_from_boxes_sam3(
        self,
        img_cv2: np.ndarray,
        boxes: List[List[int]],
        labels: Optional[List[str]] = None,
        return_labels: bool = False,
    ):
        """Refine detector boxes into masks with SAM3 geometric prompts."""
        if img_cv2 is None or len(boxes) == 0:
            return ([], [], []) if return_labels else ([], [])
        if self.sam3_processor is None or self.sam3_failures >= self.sam3_max_failures:
            masks = self._rect_masks_from_boxes(boxes, img_cv2.shape[:2])
            out_labels = [normalize_detector_label(labels[i]) if labels and i < len(labels) else "" for i in range(len(boxes))]
            return (boxes, masks, out_labels) if return_labels else (boxes, masks)

        img_rgb = _as_rgb_uint8(img_cv2)
        h, w = img_rgb.shape[:2]
        safe_boxes = [b for b in (safe_box_xyxy(box, h, w) for box in boxes) if b is not None]
        safe_labels = []
        for i, box in enumerate(boxes):
            if safe_box_xyxy(box, h, w) is not None:
                safe_labels.append(normalize_detector_label(labels[i]) if labels and i < len(labels) else "")
        if len(safe_boxes) == 0:
            return ([], [], []) if return_labels else ([], [])

        boxes_key = hashlib.blake2b(
            json.dumps(safe_boxes, sort_keys=True).encode("utf-8"),
            digest_size=10,
        ).hexdigest()
        cache_key = f"sam3_box:{self._image_hash(img_cv2)}:{boxes_key}"
        cached = self._cache_get(self.sam3_cache, cache_key)
        if cached is not None:
            cached_boxes, cached_masks = cached[:2]
            cached_labels = cached[2] if len(cached) >= 3 else [""] * len(cached_boxes)
            out = ([b[:] for b in cached_boxes], [m.copy() for m in cached_masks], list(cached_labels))
            return out if return_labels else out[:2]

        pil_img = Image.fromarray(img_rgb)
        items = []

        try:
            with torch.inference_mode():
                if self.device.type == "cuda":
                    if self.sam3_model is not None:
                        self.sam3_model.to(dtype=self.sam3_image_dtype)
                    with torch.autocast(device_type="cuda", dtype=self.sam3_image_dtype):
                        inference_state = self.sam3_processor.set_image(pil_img)
                    if self.sam3_model is not None:
                        self.sam3_model.to(dtype=self.sam3_text_dtype)
                    inference_state = self._tree_to_float_dtype(
                        inference_state,
                        dtype=self.sam3_text_dtype,
                    )
                else:
                    inference_state = self.sam3_processor.set_image(pil_img)

                for prompt_idx, prompt_box in enumerate(safe_boxes):
                    prompt_label = safe_labels[prompt_idx] if prompt_idx < len(safe_labels) else ""
                    self._reset_sam3_prompts(inference_state)
                    x1, y1, x2, y2 = prompt_box
                    box_cxcywh = [
                        float(((x1 + x2) * 0.5) / max(w, 1)),
                        float(((y1 + y2) * 0.5) / max(h, 1)),
                        float(max(x2 - x1, 1) / max(w, 1)),
                        float(max(y2 - y1, 1) / max(h, 1)),
                    ]
                    box_cxcywh = [float(np.clip(v, 0.0, 1.0)) for v in box_cxcywh]

                    if self.device.type == "cuda":
                        with torch.autocast(device_type="cuda", enabled=False):
                            output = self._call_sam3_box_prompt(inference_state, box_cxcywh)
                    else:
                        output = self._call_sam3_box_prompt(inference_state, box_cxcywh)

                    raw_boxes = self._split_first_dim(self._get_output_field(output, ["boxes", "pred_boxes", "box"], []))
                    raw_masks = self._split_first_dim(self._get_output_field(output, ["masks", "pred_masks", "mask"], []))
                    raw_scores = self._split_first_dim(self._get_output_field(output, ["scores", "score"], []))

                    best_item = None
                    m = min(len(raw_boxes), len(raw_masks))
                    for bi in range(m):
                        box_arr = _to_numpy(raw_boxes[bi]).reshape(-1)
                        if box_arr.size < 4:
                            continue
                        sb = safe_box_xyxy(box_arr[:4].tolist(), h, w)
                        if sb is None:
                            continue

                        mask_np = _to_numpy(raw_masks[bi]).squeeze()
                        if mask_np.ndim != 2:
                            continue
                        if mask_np.shape != (h, w):
                            mask_np = cv2.resize(mask_np.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
                        mask_uint8 = ((mask_np > 0.5).astype(np.uint8) * 255)
                        area = int((mask_uint8 > 0).sum())
                        if area < self.sam3_min_mask_area:
                            continue

                        score = 1.0
                        if bi < len(raw_scores):
                            try:
                                score = float(_to_numpy(raw_scores[bi]).reshape(-1)[0])
                            except Exception:
                                score = 1.0
                        prompt_iou = _box_iou_xyxy(prompt_box, sb)
                        if prompt_iou <= 0.01:
                            continue
                        item = {
                            "score": float(score + prompt_iou),
                            "box": sb,
                            "mask": mask_uint8,
                            "label": prompt_label,
                            "source": "gdino_sam3",
                        }
                        if best_item is None or item["score"] > best_item["score"]:
                            best_item = item

                    if best_item is not None:
                        items.append(best_item)
                    else:
                        rect_mask = np.zeros((h, w), dtype=np.uint8)
                        rect_mask[y1:y2 + 1, x1:x2 + 1] = 255
                        items.append({
                            "score": 0.25,
                            "box": prompt_box,
                            "mask": rect_mask,
                            "label": prompt_label,
                            "source": "gdino_box_fallback",
                        })
        except Exception as e:
            self.sam3_failures += 1
            print(f"SAM3 box prompt failed ({self.sam3_failures}/{self.sam3_max_failures}), falling back to GDINO rect mask: {e}")
            traceback.print_exc(limit=2)
            if self.sam3_failures >= self.sam3_max_failures:
                self.sam3_processor = None
                self.sam3_model = None
            masks = self._rect_masks_from_boxes(safe_boxes, (h, w))
            return (safe_boxes, masks, safe_labels) if return_labels else (safe_boxes, masks)

        valid_boxes, valid_masks, valid_labels = self._finalize_instance_items(
            items,
            max_instances=max(self.gdino_max_instances, self.sam3_max_instances),
            mask_iou_tau=0.76,
            box_iou_tau=0.90,
            return_labels=True,
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "GDINO_SAM3 gdino_boxes=%d, refined_boxes=%d, refined_masks=%d",
                len(safe_boxes),
                len(valid_boxes),
                len(valid_masks),
            )

        self._cache_put(
            self.sam3_cache,
            cache_key,
            ([b[:] for b in valid_boxes], [m.copy() for m in valid_masks], list(valid_labels)),
        )
        return (valid_boxes, valid_masks, valid_labels) if return_labels else (valid_boxes, valid_masks)

    def get_instances_gdino_sam3(
        self,
        img_cv2: np.ndarray,
        text_prompts: Optional[List[str]],
        return_labels: bool = False,
    ):
        gdino_boxes, gdino_rect_masks, gdino_labels = self.get_instances_grounding_dino(
            img_cv2,
            text_prompts,
            return_labels=True,
        )
        if len(gdino_boxes) == 0:
            return ([], [], []) if return_labels else ([], [])
        if not self.has_sam3():
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("SAM3 unavailable; using GroundingDINO rectangular masks")
            return (gdino_boxes, gdino_rect_masks, gdino_labels) if return_labels else (gdino_boxes, gdino_rect_masks)
        return self.get_instances_from_boxes_sam3(
            img_cv2,
            gdino_boxes,
            labels=gdino_labels,
            return_labels=return_labels,
        )

    def _finalize_instance_items(
        self,
        items: List[dict],
        max_instances: Optional[int] = None,
        mask_iou_tau: float = 0.75,
        box_iou_tau: float = 0.90,
        return_labels: bool = False,
    ):
        normalized_items = []
        for item in items:
            try:
                mask = np.asarray(item.get("mask")).squeeze()
                if mask.ndim != 2:
                    continue
                mask_uint8 = ((mask > 0).astype(np.uint8) * 255)
                area = int((mask_uint8 > 0).sum())
                if area <= 0:
                    continue

                h, w = mask_uint8.shape[:2]
                box = item.get("box")
                if box is None:
                    ys, xs = np.where(mask_uint8 > 0)
                    if len(xs) == 0 or len(ys) == 0:
                        continue
                    box = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
                box = safe_box_xyxy(box, h, w)
                if box is None:
                    continue

                normalized_items.append({
                    "score": float(item.get("score", 1.0)),
                    "area": area,
                    "box": box,
                    "mask": mask_uint8,
                    "label": normalize_detector_label(item.get("label", item.get("prompt", ""))),
                    "source": item.get("source", ""),
                })
            except Exception:
                continue

        if len(normalized_items) == 0:
            return ([], [], []) if return_labels else ([], [])

        if max_instances is None:
            max_instances = self.auto_max_instances

        normalized_items.sort(key=lambda x: (x["score"], x["area"]), reverse=True)
        deduped_items = []
        for item in normalized_items:
            is_duplicate = False
            for kept in deduped_items:
                if (
                    _mask_iou_binary(item["mask"], kept["mask"]) >= mask_iou_tau
                    or _box_iou_xyxy(item["box"], kept["box"]) >= box_iou_tau
                ):
                    is_duplicate = True
                    break
            if is_duplicate:
                continue
            deduped_items.append(item)
            if len(deduped_items) >= max_instances:
                break

        boxes = [x["box"] for x in deduped_items]
        masks = [x["mask"] for x in deduped_items]
        labels = [x.get("label", "") for x in deduped_items]
        return (boxes, masks, labels) if return_labels else (boxes, masks)

    def get_instances_auto(self, img_cv2: np.ndarray) -> Tuple[List[List[int]], List[np.ndarray]]:
        """
        Prompt-free, class-agnostic proposal extractor.

        Current local SAM3 only exposes text prompt interface, no automatic mask generator.
        Uses OpenCV to generate weak proposals: color-clustering partitions + edge-closed contours,
        then connected-component analysis, area/shape filtering and NMS. Class-agnostic but covers out-of-vocabulary objects.
        """
        if img_cv2 is None:
            return [], []

        cache_key = f"auto:{self._image_hash(img_cv2)}"
        cached = self._cache_get(self.auto_proposal_cache, cache_key)
        if cached is not None:
            boxes, masks = cached
            return [b[:] for b in boxes], [m.copy() for m in masks]

        img_rgb = _as_rgb_uint8(img_cv2)
        h, w = img_rgb.shape[:2]
        total_area = h * w
        min_area = max(int(self.auto_min_mask_area), int(0.0008 * total_area))
        max_area = max(min_area + 1, int(self.auto_max_mask_area_ratio * total_area))
        min_side = max(6, int(0.015 * min(h, w)))
        kernel3 = np.ones((3, 3), np.uint8)
        kernel5 = np.ones((5, 5), np.uint8)
        candidates = []

        def add_connected_components(mask_like, base_score: float, source: str):
            mask_bin = (np.asarray(mask_like) > 0).astype(np.uint8)
            if mask_bin.shape != (h, w):
                mask_bin = cv2.resize(mask_bin, (w, h), interpolation=cv2.INTER_NEAREST)
            if int(mask_bin.sum()) < min_area:
                return

            n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)
            for label_idx in range(1, n_labels):
                area = int(stats[label_idx, cv2.CC_STAT_AREA])
                if area < min_area or area > max_area:
                    continue

                x = int(stats[label_idx, cv2.CC_STAT_LEFT])
                y = int(stats[label_idx, cv2.CC_STAT_TOP])
                bw = int(stats[label_idx, cv2.CC_STAT_WIDTH])
                bh = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
                if bw < min_side or bh < min_side:
                    continue

                fill_ratio = area / float(max(bw * bh, 1))
                if fill_ratio < self.auto_min_fill_ratio:
                    continue

                touches_border = int(x <= 1) + int(y <= 1) + int(x + bw >= w - 1) + int(y + bh >= h - 1)
                if touches_border >= 3 and area > min_area * 4:
                    continue
                if touches_border >= 2 and area > max(min_area * 8, int(0.035 * total_area)):
                    continue

                comp = ((labels == label_idx).astype(np.uint8) * 255)
                comp = cv2.morphologyEx(comp, cv2.MORPH_OPEN, kernel3, iterations=1)
                if int((comp > 0).sum()) < min_area:
                    continue

                compact_bonus = min(0.20, fill_ratio * 0.20)
                area_ratio = area / float(total_area)
                mid_size_bonus = max(0.0, 0.15 * (1.0 - min(area_ratio / 0.25, 1.0)))
                large_penalty = min(0.22, area_ratio * 0.80)
                candidates.append({
                    "score": float(base_score + compact_bonus + mid_size_bonus - large_penalty),
                    "area": area,
                    "box": [x, y, x + bw - 1, y + bh - 1],
                    "mask": comp,
                    "source": source,
                })

        # 1) Color-region proposals. Downsample before k-means to keep large images cheap.
        try:
            max_side = 480
            if max(h, w) > max_side:
                scale = max_side / float(max(h, w))
                small_w = max(1, int(round(w * scale)))
                small_h = max(1, int(round(h * scale)))
                img_small = cv2.resize(img_rgb, (small_w, small_h), interpolation=cv2.INTER_AREA)
            else:
                img_small = img_rgb
                small_h, small_w = h, w

            smooth = cv2.pyrMeanShiftFiltering(img_small, sp=8, sr=18)
            lab = cv2.cvtColor(smooth, cv2.COLOR_RGB2LAB)
            features = lab.reshape((-1, 3)).astype(np.float32)
            k = min(int(self.auto_kmeans_clusters), max(2, features.shape[0] // 256))
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 1.0)
            cv2.setRNGSeed(stable_image_seed(img_rgb))
            _, labels, _ = cv2.kmeans(features, k, None, criteria, 2, cv2.KMEANS_PP_CENTERS)
            label_map = labels.reshape((small_h, small_w))
            if (small_h, small_w) != (h, w):
                label_map = cv2.resize(label_map.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

            for cluster_id in range(k):
                add_connected_components(label_map == cluster_id, base_score=0.55, source=f"color_kmeans_{cluster_id}")
        except Exception as e:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("auto proposal color-kmeans failed: %s", e)

        # 2) Edge-closed contour proposals. This catches objects split across color clusters.
        try:
            gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(gray_blur, 40, 120)
            edges = cv2.dilate(edges, kernel3, iterations=1)
            closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel5, iterations=2)
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) < min_area:
                    continue
                contour_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(contour_mask, [contour], contourIdx=-1, color=255, thickness=-1)
                add_connected_components(contour_mask, base_score=0.48, source="edge_contour")
        except Exception as e:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("auto proposal edge-contour failed: %s", e)

        # 3) Saturation/value proposals help small colorful foreground objects.
        try:
            hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
            sat = hsv[:, :, 1]
            val = hsv[:, :, 2]
            sat_tau = max(30, int(np.percentile(sat, 70)))
            val_low = int(np.percentile(val, 12))
            val_high = int(np.percentile(val, 88))
            add_connected_components(sat >= sat_tau, base_score=0.42, source="high_saturation")
            add_connected_components(val <= val_low, base_score=0.38, source="dark_region")
            add_connected_components(val >= val_high, base_score=0.36, source="bright_region")
        except Exception as e:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("auto proposal hsv-threshold failed: %s", e)

        valid_boxes, valid_masks = self._finalize_instance_items(
            candidates,
            max_instances=self.auto_max_instances,
            mask_iou_tau=0.70,
            box_iou_tau=0.88,
        )

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "AUTO_PROPOSAL raw_candidates=%d, valid_boxes=%d, valid_masks=%d, "
                "min_area=%d, max_area=%d, mode=%s",
                len(candidates),
                len(valid_boxes),
                len(valid_masks),
                min_area,
                max_area,
                self.proposal_mode,
            )

        self._cache_put(
            self.auto_proposal_cache,
            cache_key,
            ([b[:] for b in valid_boxes], [m.copy() for m in valid_masks]),
        )
        return valid_boxes, valid_masks

    def get_instances(
        self,
        img_cv2: np.ndarray,
        text_prompts: Optional[List[str]] = None,
    ) -> Tuple[List[List[int]], List[np.ndarray]]:
        boxes, masks, _ = self.get_instances_with_labels(img_cv2, text_prompts)
        return boxes, masks

    def get_instances_with_labels(
        self,
        img_cv2: np.ndarray,
        text_prompts: Optional[List[str]] = None,
    ) -> Tuple[List[List[int]], List[np.ndarray], List[str]]:
        mode = self.proposal_mode
        prompts = text_prompts or []

        if mode in {"auto", "cv_auto", "proposal", "prompt_free"}:
            boxes, masks = self.get_instances_auto(img_cv2)
            return boxes, masks, [""] * len(boxes)

        if mode in {"gdino", "gdino_box", "grounding_dino", "grounding_dino_box"}:
            return self.get_instances_grounding_dino(img_cv2, prompts, return_labels=True)

        if mode in {"gdino_sam3", "grounding_dino_sam3"}:
            return self.get_instances_gdino_sam3(img_cv2, prompts, return_labels=True)

        if mode in {"hybrid", "auto_text", "text_auto"}:
            auto_boxes, auto_masks = self.get_instances_auto(img_cv2)
            text_boxes, text_masks, text_labels = self.get_instances_from_text(img_cv2, prompts, return_labels=True)
            items = []
            for box, mask in zip(auto_boxes, auto_masks):
                items.append({"score": 0.90, "box": box, "mask": mask, "label": "", "source": "auto"})
            for i, (box, mask) in enumerate(zip(text_boxes, text_masks)):
                label = text_labels[i] if i < len(text_labels) else ""
                items.append({"score": 1.00, "box": box, "mask": mask, "label": label, "source": "text"})
            return self._finalize_instance_items(
                items,
                max_instances=max(self.auto_max_instances, self.sam3_max_instances),
                mask_iou_tau=0.72,
                box_iou_tau=0.88,
                return_labels=True,
            )

        if mode not in {"text", "sam3", "sam3_text"}:
            print(f"Unknown proposal_mode='{self.proposal_mode}', falling back to text/SAM3 prompt mode.")
        return self.get_instances_from_text(img_cv2, prompts, return_labels=True)

    def _select_feature_tensor(self, feat: Any) -> Optional[torch.Tensor]:
        if isinstance(feat, dict):
            for key in ["x_norm_clstoken", "pooler_output", "last_hidden_state", "x_prenorm"]:
                if key in feat:
                    feat = feat[key]
                    break
            else:
                return None
        elif hasattr(feat, "last_hidden_state"):
            feat = feat.last_hidden_state

        if isinstance(feat, (list, tuple)):
            feat = feat[0] if len(feat) > 0 else None
        if not isinstance(feat, torch.Tensor):
            return None

        if feat.ndim == 3:
            feat = feat[:, 0] if feat.shape[1] > 0 else feat.mean(dim=1)
        elif feat.ndim == 4:
            feat = feat.flatten(2).mean(dim=-1)
        elif feat.ndim == 1:
            feat = feat.unsqueeze(0)

        if feat.ndim != 2:
            return None
        return feat.float()

    def extract_roi_feature(self, img_cv2: np.ndarray, box: List[float]) -> Optional[torch.Tensor]:
        if self.dino_model is None:
            return None

        sb = safe_box_xyxy(box, img_cv2.shape[0], img_cv2.shape[1])
        if sb is None:
            return None
        x1, y1, x2, y2 = sb
        if x2 - x1 < 10 or y2 - y1 < 10:
            return None

        cache_key = f"dino:{self._image_hash(img_cv2)}:{x1},{y1},{x2},{y2}"
        cached = self._cache_get(self.roi_feature_cache, cache_key)
        if cached is not None:
            return cached.clone() if isinstance(cached, torch.Tensor) else None

        crop = img_cv2[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        try:
            crop_pil = Image.fromarray(_as_rgb_uint8(crop))
            input_tensor = self.dino_transform(crop_pil).unsqueeze(0).to(self.device, non_blocking=True)
            with torch.inference_mode():
                feat_raw = self.dino_model(input_tensor)
                feat = self._select_feature_tensor(feat_raw)
            if feat is None:
                self._cache_put(self.roi_feature_cache, cache_key, None)
                return None
            feat = F.normalize(feat, p=2, dim=1).detach().cpu()
            self._cache_put(self.roi_feature_cache, cache_key, feat)
            return feat.clone()
        except Exception as e:
            print(f"DINO ROI feature failed, using IoU-only: {e}")
            traceback.print_exc(limit=2)
            # Model may be unstable; disable subsequent DINO verification to avoid repeated stalls.
            self.dino_model = None
            self.model_status["dino"] = f"disabled during inference: {e}"
            return None

    def calculate_pose_da3(self, src_img_input, tgt_img_input, depth_src_gt, pose_src_gl):
        if not self.has_da3():
            return None, None, "DA3 not enabled or circuit-breaker tripped"

        for p in [src_img_input, tgt_img_input]:
            if isinstance(p, str) and not os.path.exists(p):
                return None, None, f"DA3 input path does not exist: {p}"

        try:
            with torch.inference_mode():
                pred = self.da3.inference([src_img_input, tgt_img_input])
        except Exception as e:
            self.da3_failures += 1
            msg = f"DA3 inference failed ({self.da3_failures}/{self.da3_max_failures}): {e}"
            print(f"{msg}")
            traceback.print_exc(limit=2)
            if self.da3_failures >= self.da3_max_failures:
                print("DA3 consecutive failures exceeded limit, auto-skipping DA3 metrics for remaining samples.")
            return None, None, msg

        try:
            if getattr(pred, "depth", None) is None or getattr(pred, "extrinsics", None) is None:
                return None, None, "DA3 did not return depth/extrinsics"
            if len(pred.extrinsics) < 2 or len(pred.depth) < 2:
                return None, None, "DA3 returned insufficient outputs"

            depth_src_pred = _to_numpy(pred.depth[0]).squeeze()
            depth_tgt_pred = _to_numpy(pred.depth[1]).squeeze()
            
            if depth_src_pred.ndim != 2:
                return None, None, f"DA3 depth output shape unexpected: {depth_src_pred.shape}"

            e_src_raw = _to_numpy(pred.extrinsics[0]).astype(np.float32)
            e_tgt_raw = _to_numpy(pred.extrinsics[1]).astype(np.float32)

            def to_4x4(e):
                if e.shape == (4, 4):
                    return e
                if e.shape == (3, 4):
                    out = np.eye(4, dtype=np.float32)
                    out[:3, :4] = e
                    return out
                raise ValueError(f"extrinsics shape unexpected: {e.shape}")

            e_src = to_4x4(e_src_raw)
            e_tgt = to_4x4(e_tgt_raw)
            c2w_tgt = np.linalg.inv(e_tgt)
            t_rel = e_src @ c2w_tgt

            h_d, w_d = depth_src_pred.shape
            depth_src_gt_rs = cv2.resize(depth_src_gt, (w_d, h_d), interpolation=cv2.INTER_NEAREST)
            valid = (
                np.isfinite(depth_src_gt_rs)
                & np.isfinite(depth_src_pred)
                & (depth_src_gt_rs > 0.1)
                & (depth_src_gt_rs < 10.0)
                & (depth_src_pred > 0.1)
            )
            if int(valid.sum()) < 100:
                return None, None, "DA3 scale alignment failed: insufficient valid depth anchors"

            scale = np.median(depth_src_gt_rs[valid] / depth_src_pred[valid])
            if not np.isfinite(scale) or scale <= 0:
                return None, None, f"DA3 scale abnormal: {scale}"
            t_rel[:3, 3] *= float(scale)

            pose_tgt_pred_gl = pose_src_gl @ t_rel
            
            # Align predicted target depth to absolute physical scale using source image scale
            depth_tgt_pred_scaled = depth_tgt_pred * float(scale)
            
            return pose_tgt_pred_gl, depth_tgt_pred_scaled, None
        except Exception as e:
            return None, None, f"DA3 postprocessing failed: {e}"



# ==========================================
# Evaluator
# ==========================================
class SemanticConsistencyEvaluator:
    def __init__(self, device: torch.device):
        self.device = device

    def compute_pose_metrics(self, pose_src, pose_tgt):
        def get_angle(v1, v2):
            v1, v2 = np.array(v1, dtype=float), np.array(v2, dtype=float)
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 == 0 or n2 == 0:
                return 0.0
            cos_theta = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
            cross_product = v1[0] * v2[1] - v1[1] * v2[0]
            if cross_product == 0:
                cross_product = -1
            raw_angle = np.arccos(cos_theta) * 180.0 / np.pi * -np.sign(cross_product)
            if raw_angle < 0:
                raw_angle += 360.0
            if raw_angle > 180.0:
                raw_angle -= 360.0
            return raw_angle

        center_src, forward_src = pose_src[:3, 3], pose_src[:3, 2]
        forward_src_xy_norm = np.linalg.norm(forward_src[:2]) + 1e-8
        dir_src = forward_src[:2] / forward_src_xy_norm
        pitch_src = np.arctan2(forward_src[2], forward_src_xy_norm)

        center_tgt, forward_tgt = pose_tgt[:3, 3], pose_tgt[:3, 2]
        forward_tgt_xy_norm = np.linalg.norm(forward_tgt[:2]) + 1e-8
        dir_tgt = forward_tgt[:2] / (np.linalg.norm(forward_tgt[:2]) + 1e-8)
        pitch_tgt = np.arctan2(forward_tgt[2], forward_tgt_xy_norm)

        distance = np.linalg.norm(center_tgt[:2] - center_src[:2])
        angle = get_angle(dir_src, center_tgt[:2] - center_src[:2]) if distance > 1e-6 else 0.0
        delta_angle = get_angle(dir_src, dir_tgt)

        dx = distance * np.sin(np.deg2rad(angle))
        dy = distance * np.cos(np.deg2rad(angle))
        dz = center_tgt[2] - center_src[2]
        dphi = (pitch_tgt - pitch_src) * 180.0 / np.pi
        return {
            "dx": float(dx),
            "dy": float(dy),
            "dz": float(dz),
            "dangle": float(delta_angle),
            "dphi": float(dphi),
        }

    def project_mask_to_target(
        self,
        mask: np.ndarray,
        depth: np.ndarray,
        T_ctx: np.ndarray,
        K_ctx: np.ndarray,
        T_tgt: np.ndarray,
        K_tgt: np.ndarray,
        tgt_shape: Tuple[int, int],
    ) -> Tuple[Optional[List[int]], Optional[np.ndarray]]:
        """
        Robust projection version.

        Key fixes:
        1) Auto-try K / Kx2 / Kx0.5 and c2w / w2c inside runner without changing dataloader;
        2) Fix pixel out-of-bounds after rounding (e.g. v_float=241.7 rounds to 242);
        3) Final boundary filter before writing tgt_mask to avoid IndexError;
        4) Subsample when projected points are too many to avoid slowdown from low-score SAM3 proposals.
        """
        H_tgt, W_tgt = tgt_shape

        if mask is None or depth is None:
            return None, None

        mask = np.asarray(mask)
        depth = np.asarray(depth).astype(np.float32)

        if H_tgt <= 0 or W_tgt <= 0:
            logger.debug("PROJ reject: invalid target shape %s", tgt_shape)
            return None, None

        # Ensure mask/depth share the same pixel coordinate system.
        if mask.shape[:2] != depth.shape[:2]:
            logger.debug("PROJ resize mask %s -> depth %s", mask.shape[:2], depth.shape[:2])
            mask = cv2.resize(
                mask.astype(np.uint8),
                (depth.shape[1], depth.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        y, x = np.where(mask > 0)
        if len(x) < 10:
            logger.debug("PROJ reject: mask pixels too few = %d", len(x))
            return None, None

        depths = depth[y, x]
        valid = np.isfinite(depths) & (depths > 0.05) & (depths < 20.0)
        if not np.any(valid):
            d_valid = depth[np.isfinite(depth)]
            if d_valid.size > 0:
                logger.debug(
                    "PROJ reject: no valid depth in mask, depth_min=%.4f, depth_max=%.4f",
                    float(np.nanmin(d_valid)),
                    float(np.nanmax(d_valid)),
                )
            else:
                logger.debug("PROJ reject: depth all invalid")
            return None, None

        x = x[valid]
        y = y[valid]
        depths = depths[valid]

        # Low-threshold SAM3 produces many masks with many points each; subsample suffices for box/mask projection.
        max_points = 20000
        if len(x) > max_points:
            sel = np.linspace(0, len(x) - 1, max_points).astype(np.int64)
            x = x[sel]
            y = y[sel]
            depths = depths[sel]

        K_ctx = np.asarray(K_ctx, dtype=np.float32)[:3, :3]
        K_tgt = np.asarray(K_tgt, dtype=np.float32)[:3, :3]
        T_ctx = np.asarray(T_ctx, dtype=np.float32)
        T_tgt = np.asarray(T_tgt, dtype=np.float32)

        if T_ctx.shape != (4, 4) or T_tgt.shape != (4, 4):
            logger.debug(
                "PROJ reject: pose shape invalid, T_ctx=%s, T_tgt=%s",
                T_ctx.shape,
                T_tgt.shape,
            )
            return None, None

        def scale_K(K: np.ndarray, s: float) -> np.ndarray:
            K2 = K.copy().astype(np.float32)
            K2[0, 0] *= s
            K2[1, 1] *= s
            K2[0, 2] *= s
            K2[1, 2] *= s
            return K2

        # Auto-handle K scaling differences inside runner without changing dataloader.
        K_candidates = [
            ("K", K_ctx, K_tgt),
            ("Kx2", scale_K(K_ctx, 2.0), scale_K(K_tgt, 2.0)),
            ("Kx0.5", scale_K(K_ctx, 0.5), scale_K(K_tgt, 0.5)),
        ]

        pts_2d = np.vstack((x.astype(np.float32), y.astype(np.float32), np.ones_like(x, dtype=np.float32)))

        def try_project(Kc: np.ndarray, Kt: np.ndarray, convention: str):
            try:
                inv_Kc = np.linalg.inv(Kc)
            except np.linalg.LinAlgError:
                return None

            cam_pts = (inv_Kc @ pts_2d) * depths[None, :]
            cam_pts_homo = np.vstack((cam_pts, np.ones_like(x, dtype=np.float32)))

            try:
                if convention == "c2w":
                    pts_world = T_ctx @ cam_pts_homo
                    pts_tgt_cam = (np.linalg.inv(T_tgt) @ pts_world)[:3, :]
                elif convention == "w2c":
                    pts_world = np.linalg.inv(T_ctx) @ cam_pts_homo
                    pts_tgt_cam = (T_tgt @ pts_world)[:3, :]
                else:
                    raise ValueError(convention)
            except np.linalg.LinAlgError:
                return None

            z_all = pts_tgt_cam[2, :]
            front = np.isfinite(z_all) & (z_all > 1e-4)

            if not np.any(front):
                return {
                    "ok": False,
                    "u": np.array([], dtype=np.int32),
                    "v": np.array([], dtype=np.int32),
                    "z": np.array([], dtype=np.float32),
                    "front_count": 0,
                    "in_count": 0,
                    "z_min": float(np.nanmin(z_all)) if z_all.size > 0 else float("nan"),
                    "z_max": float(np.nanmax(z_all)) if z_all.size > 0 else float("nan"),
                    "u_min": float("nan"),
                    "u_max": float("nan"),
                    "v_min": float("nan"),
                    "v_max": float("nan"),
                }

            pts = pts_tgt_cam[:, front]
            proj = Kt @ pts
            u_float = proj[0, :] / (proj[2, :] + 1e-6)
            v_float = proj[1, :] / (proj[2, :] + 1e-6)
            z_front = pts[2, :]

            finite = np.isfinite(u_float) & np.isfinite(v_float) & np.isfinite(z_front)
            u_float = u_float[finite]
            v_float = v_float[finite]
            z_front = z_front[finite]

            # Note: filter by float range first, then floor/clip, to avoid out-of-bounds after rounding (e.g. 241.7 -> 242).
            in_bounds = (
                (u_float >= 0.0) & (u_float < float(W_tgt)) &
                (v_float >= 0.0) & (v_float < float(H_tgt))
            )

            u = np.floor(u_float[in_bounds]).astype(np.int32)
            v = np.floor(v_float[in_bounds]).astype(np.int32)
            u = np.clip(u, 0, W_tgt - 1)
            v = np.clip(v, 0, H_tgt - 1)
            z = z_front[in_bounds].astype(np.float32)

            return {
                "ok": len(u) > 0,
                "u": u,
                "v": v,
                "z": z,
                "front_count": int(front.sum()),
                "in_count": int(in_bounds.sum()),
                "u_min": float(np.nanmin(u_float)) if u_float.size > 0 else float("nan"),
                "u_max": float(np.nanmax(u_float)) if u_float.size > 0 else float("nan"),
                "v_min": float(np.nanmin(v_float)) if v_float.size > 0 else float("nan"),
                "v_max": float(np.nanmax(v_float)) if v_float.size > 0 else float("nan"),
                "z_min": float(np.nanmin(z_all)) if z_all.size > 0 else float("nan"),
                "z_max": float(np.nanmax(z_all)) if z_all.size > 0 else float("nan"),
            }

        best = None
        best_name = None

        for k_name, Kc, Kt in K_candidates:
            for convention in ["c2w", "w2c"]:
                res = try_project(Kc, Kt, convention)
                if res is None:
                    continue

                logger.debug(
                    "PROJ try %s/%s: front=%d, in=%d, z=(%.3f,%.3f), "
                    "u=(%.1f,%.1f), v=(%.1f,%.1f), target=(%d,%d)",
                    k_name,
                    convention,
                    res["front_count"],
                    res["in_count"],
                    res["z_min"],
                    res["z_max"],
                    res["u_min"],
                    res["u_max"],
                    res["v_min"],
                    res["v_max"],
                    W_tgt,
                    H_tgt,
                )

                if best is None or res["in_count"] > best["in_count"]:
                    best = res
                    best_name = f"{k_name}/{convention}"

        if best is None or best["in_count"] < 10:
            logger.debug(
                "PROJ reject: in_bounds too few = %d, best=%s",
                0 if best is None else best["in_count"],
                best_name,
            )
            return None, None

        u, v, z = best["u"], best["v"], best["z"]

        # z-buffer: far points written first, near points overwrite (painter order).
        order = np.argsort(z)[::-1]
        u_sorted = u[order]
        v_sorted = v[order]

        # Final safety: filter once more before writing mask to prevent IndexError.
        valid_pix = (
            (u_sorted >= 0) & (u_sorted < W_tgt) &
            (v_sorted >= 0) & (v_sorted < H_tgt)
        )
        u_sorted = u_sorted[valid_pix]
        v_sorted = v_sorted[valid_pix]

        if len(u_sorted) == 0:
            logger.debug("PROJ reject: no valid pixel after final filter, best=%s", best_name)
            return None, None

        tgt_mask = np.zeros((H_tgt, W_tgt), dtype=np.uint8)
        tgt_mask[v_sorted, u_sorted] = 255
        tgt_mask = cv2.dilate(tgt_mask, np.ones((3, 3), np.uint8), iterations=1)
        tgt_mask = cv2.morphologyEx(tgt_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        y_nz, x_nz = np.where(tgt_mask > 0)
        if len(x_nz) == 0:
            logger.debug("PROJ reject: empty target mask after morphology, best=%s", best_name)
            return None, None

        box = [
            int(np.min(x_nz)),
            int(np.min(y_nz)),
            int(np.max(x_nz)),
            int(np.max(y_nz)),
        ]

        logger.debug(
            "PROJ accepted: best=%s, in_bounds=%d, box=%s",
            best_name,
            best["in_count"],
            box,
        )
        return box, tgt_mask

    def compute_iou(self, box1: List[float], box2: List[float]) -> float:
        xa = max(box1[0], box2[0])
        ya = max(box1[1], box2[1])
        xb = min(box1[2], box2[2])
        yb = min(box1[3], box2[3])

        inter_w = max(0, xb - xa)
        inter_h = max(0, yb - ya)
        inter_area = inter_w * inter_h

        area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
        area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
        union_area = area1 + area2 - inter_area

        if union_area <= 0:
            return 0.0
        return inter_area / float(union_area + 1e-6)

    def merge_projections(
        self,
        boxes: List[List[float]],
        masks: List[np.ndarray],
        features: List[Optional[torch.Tensor]],
        iou_thresh: float = 0.55,
    ):
        if len(boxes) == 0:
            return [], [], []

        merged_boxes, merged_masks, merged_feats = [], [], []
        used = [False] * len(boxes)

        for i in range(len(boxes)):
            if used[i]:
                continue

            current_mask = masks[i].copy()
            current_feat = features[i]
            used[i] = True

            for j in range(i + 1, len(boxes)):
                if used[j]:
                    continue
                if self.compute_iou(boxes[i], boxes[j]) > iou_thresh:
                    current_mask = np.maximum(current_mask, masks[j])
                    used[j] = True

            y_nz, x_nz = np.where(current_mask > 0)
            if len(x_nz) > 0:
                merged_boxes.append([int(np.min(x_nz)), int(np.min(y_nz)), int(np.max(x_nz)), int(np.max(y_nz))])
                merged_masks.append(current_mask)
                merged_feats.append(current_feat)

        return merged_boxes, merged_masks, merged_feats

    def evaluate_permanence(
        self,
        gt_boxes,
        gt_features,
        pred_boxes,
        gen_img_cv2,
        sem_tools: SemanticToolsAPI,
        mode: str = "dinov3",
        iou_tau: float = 0.5,
        feat_tau: float = 0.65,
        gt_masks: Optional[List[np.ndarray]] = None,
        pred_masks: Optional[List[np.ndarray]] = None,
        gt_labels: Optional[List[str]] = None,
        pred_labels: Optional[List[str]] = None,
        require_label_match: bool = True,
        mask_iou_tau: Optional[float] = None,
    ):
        if len(gt_boxes) == 0:
            return 0.0, float("nan"), [], [], []

        center_errors = []
        verified_pred_boxes = []
        matched_pairs = []
        match_details = []

        pred_features = [sem_tools.extract_roi_feature(gen_img_cv2, pb) for pb in pred_boxes]

        candidates = []
        for i, gt_box in enumerate(gt_boxes):
            gt_feat = gt_features[i] if i < len(gt_features) else None
            gt_label = normalize_detector_label(gt_labels[i]) if gt_labels and i < len(gt_labels) else ""
            for j, pred_box in enumerate(pred_boxes):
                pred_label = normalize_detector_label(pred_labels[j]) if pred_labels and j < len(pred_labels) else ""
                labels_match = (not gt_label or not pred_label or gt_label == pred_label)
                if require_label_match and not labels_match:
                    continue
                iou = self.compute_iou(gt_box, pred_box)
                if iou < iou_tau:
                    continue
                # Main object evaluation is box-based. Masks may be saved for
                # diagnostics, but they are not used to accept/reject matches.
                mask_iou = None
                if gt_masks is not None and pred_masks is not None and i < len(gt_masks) and j < len(pred_masks):
                    mask_iou = _mask_iou_binary(gt_masks[i], pred_masks[j])

                pred_feat = pred_features[j] if j < len(pred_features) else None

                is_valid = False
                sim = 1.0
                if "dino" in mode.lower():
                    if gt_feat is not None and pred_feat is not None:
                        gt_feat = gt_feat.float()
                        pred_feat = pred_feat.float()
                        sim = F.cosine_similarity(gt_feat, pred_feat).item()
                        if sim > feat_tau:
                            is_valid = True
                elif mode == "none":
                    is_valid = True
                else:
                    is_valid = True

                if is_valid:
                    candidates.append((float(iou), float(sim), float(mask_iou) if mask_iou is not None else None, i, j))

        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        used_gt, used_pred = set(), set()
        for iou, sim, mask_iou, gt_idx, pred_idx in candidates:
            if gt_idx in used_gt or pred_idx in used_pred:
                continue
            used_gt.add(gt_idx)
            used_pred.add(pred_idx)

            matched_pred_box = pred_boxes[pred_idx]
            verified_pred_boxes.append(matched_pred_box)
            matched_pairs.append((gt_idx, pred_idx))

            gt_box = gt_boxes[gt_idx]
            c_gt = np.array([(gt_box[0] + gt_box[2]) / 2, (gt_box[1] + gt_box[3]) / 2])
            c_pred = np.array([(matched_pred_box[0] + matched_pred_box[2]) / 2, (matched_pred_box[1] + matched_pred_box[3]) / 2])
            center_err = float(np.linalg.norm(c_gt - c_pred))
            center_errors.append(center_err)
            match_details.append({
                "rank": len(match_details) + 1,
                "gt_idx": int(gt_idx),
                "pred_idx": int(pred_idx),
                "iou": float(iou),
                "mask_iou": float(mask_iou) if mask_iou is not None else None,
                "feature_sim": float(sim),
                "gt_label": gt_label,
                "pred_label": pred_label,
                "labels_match": bool(labels_match),
                "center_error": center_err,
                "gt_box": [float(x) for x in gt_box[:4]],
                "pred_box": [float(x) for x in matched_pred_box[:4]],
            })

        recall = len(matched_pairs) / len(gt_boxes) if len(gt_boxes) > 0 else 0.0
        center_error = float(np.mean(center_errors)) if len(center_errors) > 0 else float("nan")
        return recall, center_error, verified_pred_boxes, matched_pairs, match_details

    def evaluate_permanence_vlm(
        self,
        gt_boxes,
        gt_features,
        pred_boxes,
        gt_img_cv2,
        pred_img_cv2,
        sem_tools: SemanticToolsAPI,
        vlm_matcher: VLMObjectMatchJudge,
        mode: str = "dinov3",
        gt_masks: Optional[List[np.ndarray]] = None,
        pred_masks: Optional[List[np.ndarray]] = None,
        gt_labels: Optional[List[str]] = None,
        pred_labels: Optional[List[str]] = None,
        max_candidate_pairs: int = 120,
        candidate_iou_tau: float = 0.01,
        candidate_center_tau: float = 0.55,
        geometric_iou_tau: float = 0.15,
        geometric_mask_iou_tau: float = 0.05,
        geometric_center_tau: float = 0.08,
    ):
        if len(gt_boxes) == 0:
            return 0.0, float("nan"), [], [], [], {
                "vlm_calls": 0,
                "vlm_candidate_pairs": 0,
                "vlm_match_mode": "global_image",
            }

        H, W = gt_img_cv2.shape[:2]
        diag = float(max(np.hypot(H, W), 1.0))
        pred_features = []
        use_dino_sort = "dino" in str(mode).lower() and sem_tools.has_dino()
        if use_dino_sort:
            pred_features = [sem_tools.extract_roi_feature(pred_img_cv2, pb) for pb in pred_boxes]

        vlm_result = vlm_matcher.judge_image_matches(
            gt_img=gt_img_cv2,
            pred_img=pred_img_cv2,
            gt_boxes=gt_boxes,
            pred_boxes=pred_boxes,
            gt_labels=gt_labels,
            pred_labels=pred_labels,
        )

        matched_pairs, verified_pred_boxes, center_errors, match_details = [], [], [], []
        rejected_details = list(vlm_result.get("rejected_matches", []) or [])
        geometric_rejected_count = 0
        for match in vlm_result.get("matches", []):
            gt_idx = int(match["gt_idx"])
            pred_idx = int(match["pred_idx"])

            gt_box = gt_boxes[gt_idx]
            pred_box = pred_boxes[pred_idx]
            c_gt = np.array([(gt_box[0] + gt_box[2]) / 2, (gt_box[1] + gt_box[3]) / 2])
            c_pred = np.array([(pred_box[0] + pred_box[2]) / 2, (pred_box[1] + pred_box[3]) / 2])
            center_err = float(np.linalg.norm(c_gt - c_pred))
            center_norm = float(center_err / diag)

            gt_label = normalize_detector_label(gt_labels[gt_idx]) if gt_labels and gt_idx < len(gt_labels) else ""
            pred_label = normalize_detector_label(pred_labels[pred_idx]) if pred_labels and pred_idx < len(pred_labels) else ""
            iou = self.compute_iou(gt_box, pred_box)
            # VLM can propose semantic matches, but the geometric sanity gate is
            # deliberately box-only. Broken/incomplete SAM masks should not make
            # a match pass or fail.
            mask_iou = None
            if gt_masks is not None and pred_masks is not None and gt_idx < len(gt_masks) and pred_idx < len(pred_masks):
                mask_iou = _mask_iou_binary(gt_masks[gt_idx], pred_masks[pred_idx])
            geom_pass = (
                float(iou) >= float(geometric_iou_tau)
                or (float(iou) > 0.0 and center_norm <= float(geometric_center_tau))
            )
            if not geom_pass:
                geometric_rejected_count += 1
                rejected_details.append({
                    "gt_idx": int(gt_idx),
                    "pred_idx": int(pred_idx),
                    "gt_label": gt_label,
                    "pred_label": pred_label,
                    "reason": (
                        "geometric_gate_failed:"
                        f"iou={float(iou):.3f}<{float(geometric_iou_tau):.3f},"
                        f"center_norm={center_norm:.3f}>{float(geometric_center_tau):.3f}"
                    ),
                    "iou": float(iou),
                    "mask_iou": float(mask_iou) if mask_iou is not None else None,
                    "center_norm": center_norm,
                    "vlm_same_object_confidence": _float_or_none(match.get("same_object_confidence")),
                    "vlm_reason": match.get("reason"),
                })
                continue

            matched_pairs.append((gt_idx, pred_idx))
            matched_pred_box = pred_boxes[pred_idx]
            verified_pred_boxes.append(matched_pred_box)
            center_errors.append(center_err)

            sim = None
            gt_feat = gt_features[gt_idx] if gt_features is not None and gt_idx < len(gt_features) else None
            if use_dino_sort and gt_feat is not None and pred_idx < len(pred_features) and pred_features[pred_idx] is not None:
                sim = F.cosine_similarity(gt_feat.float(), pred_features[pred_idx].float()).item()

            detail = {
                "rank": len(match_details) + 1,
                "gt_idx": int(gt_idx),
                "pred_idx": int(pred_idx),
                "match_policy": "vlm_global_same_object",
                "iou": float(iou),
                "mask_iou": float(mask_iou) if mask_iou is not None else None,
                "center_norm": center_norm,
                "feature_sim": float(sim) if sim is not None else None,
                "gt_label": gt_label,
                "pred_label": pred_label,
                "labels_match": bool(gt_label and pred_label and gt_label == pred_label),
                "vlm_same_object": True,
                "vlm_orientation_consistent": bool(match.get("orientation_consistent", True)),
                "vlm_orientation_observable": bool(match.get("orientation_observable", False)),
                "vlm_same_object_confidence": _float_or_none(match.get("same_object_confidence")),
                "vlm_orientation_confidence": _float_or_none(match.get("orientation_confidence")),
                "vlm_category_gt": match.get("object_category_gt"),
                "vlm_category_pred": match.get("object_category_pred"),
                "vlm_reason": match.get("reason"),
                "center_error": center_err,
                "gt_box": [float(x) for x in gt_box[:4]],
                "pred_box": [float(x) for x in matched_pred_box[:4]],
            }
            match_details.append(detail)

        recall = len(matched_pairs) / len(gt_boxes) if len(gt_boxes) > 0 else 0.0
        center_error = float(np.mean(center_errors)) if len(center_errors) > 0 else float("nan")
        accepted_vlm_matches = [
            m for m in vlm_result.get("matches", [])
            if (int(m["gt_idx"]), int(m["pred_idx"])) in set(matched_pairs)
        ]
        orientation_pass_count = sum(1 for m in accepted_vlm_matches if bool(m.get("orientation_consistent", True)))
        stats = {
            "vlm_match_mode": "global_image",
            "vlm_candidate_pairs": int(len(gt_boxes) * len(pred_boxes)),
            "vlm_total_possible_pairs": int(len(gt_boxes) * len(pred_boxes)),
            "vlm_calls": 1,
            "vlm_raw_match_count": int(vlm_result.get("raw_match_count", len(vlm_result.get("matches", [])))),
            "vlm_same_object_pairs_raw": int(len(vlm_result.get("matches", []))),
            "vlm_same_object_pairs": int(len(matched_pairs)),
            "vlm_orientation_pass_pairs": int(orientation_pass_count),
            "vlm_geometric_rejected_pairs": int(geometric_rejected_count),
            "vlm_geometric_iou_tau": float(geometric_iou_tau),
            "vlm_geometric_mask_iou_tau": None,
            "vlm_geometric_gate": "box_iou_or_box_center",
            "vlm_geometric_center_tau": float(geometric_center_tau),
            "vlm_rejected_pairs": int(len(rejected_details)),
            "vlm_rejected_match_details": rejected_details,
            "vlm_error": vlm_result.get("vlm_error"),
        }
        return recall, center_error, verified_pred_boxes, matched_pairs, match_details, stats

    def compute_topology_consistency(
        self,
        gt_boxes: List[List[float]],
        pred_boxes: List[List[float]],
        matched_pairs: List[Tuple[int, int]],
    ) -> Dict[str, Any]:
        """
        Object-pair topology: whether relative 2D direction and distance ratios are preserved.

        Missing matched objects contribute 0 to the pair score, so this complements Recall:
        high topology means the detected/matched objects keep the GT pairwise layout.
        """
        n_gt = len(gt_boxes)
        total_pairs = n_gt * (n_gt - 1) // 2
        if n_gt < 2 or total_pairs <= 0:
            return {
                "Topology": None,
                "Topology_NumPairs": 0,
                "Topology_MatchedPairs": 0,
                "Topology_PositivePairs": 0,
                "Topology_MatchedPairCoverage": None,
            }

        match_map = {}
        for gt_idx, pred_idx in matched_pairs or []:
            if 0 <= int(gt_idx) < n_gt and 0 <= int(pred_idx) < len(pred_boxes):
                match_map[int(gt_idx)] = int(pred_idx)

        def center_xy(box):
            x1, y1, x2, y2 = [float(v) for v in box[:4]]
            return np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)

        pair_scores = []
        matched_pair_count = 0
        positive_pair_count = 0

        for i in range(n_gt):
            for j in range(i + 1, n_gt):
                if i not in match_map or j not in match_map:
                    pair_scores.append(0.0)
                    continue

                pi = match_map[i]
                pj = match_map[j]
                vec_gt = center_xy(gt_boxes[i]) - center_xy(gt_boxes[j])
                vec_pred = center_xy(pred_boxes[pi]) - center_xy(pred_boxes[pj])

                norm_gt = float(np.linalg.norm(vec_gt))
                norm_pred = float(np.linalg.norm(vec_pred))
                matched_pair_count += 1

                if norm_gt < 1e-6 or norm_pred < 1e-6:
                    pair_score = 1.0
                else:
                    cos_sim = float(np.dot(vec_gt, vec_pred) / (norm_gt * norm_pred + 1e-8))
                    dir_score = max(0.0, cos_sim)
                    dist_ratio = float(norm_pred / (norm_gt + 1e-8))
                    ratio_score = min(dist_ratio, 1.0 / (dist_ratio + 1e-8))
                    pair_score = float(dir_score * ratio_score)

                if pair_score > 0:
                    positive_pair_count += 1
                pair_scores.append(pair_score)

        topology = float(np.mean(pair_scores)) if pair_scores else None
        return {
            "Topology": topology,
            "Topology_NumPairs": int(total_pairs),
            "Topology_MatchedPairs": int(matched_pair_count),
            "Topology_PositivePairs": int(positive_pair_count),
            "Topology_MatchedPairCoverage": float(matched_pair_count / total_pairs),
        }

    def compute_matched_layout_metrics(
        self,
        gt_boxes: List[List[float]],
        pred_boxes: List[List[float]],
        matched_pairs: List[Tuple[int, int]],
        gt_masks: Optional[List[np.ndarray]] = None,
        pred_masks: Optional[List[np.ndarray]] = None,
        axis_equal_tau: float = 0.03,
    ) -> Dict[str, Any]:
        valid_pairs = [
            (int(g), int(p)) for g, p in (matched_pairs or [])
            if 0 <= int(g) < len(gt_boxes) and 0 <= int(p) < len(pred_boxes)
        ]
        if not valid_pairs:
            return {
                "Matched_Object_Area_Ratio_Mean": None,
                "Matched_Object_Area_Ratio_Score": None,
                "Matched_Location_Relation": None,
                "Matched_Location_NumPairs": 0,
                "Matched_Pair_Area_Ratio_Score": None,
            }

        def center(box):
            x1, y1, x2, y2 = [float(v) for v in box[:4]]
            return np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)

        def area(boxes, masks, idx):
            del masks
            return float(max(_box_area_xyxy(boxes[idx]), 1))

        object_area_ratios, object_area_scores = [], []
        gt_areas, pred_areas = {}, {}
        for gt_idx, pred_idx in valid_pairs:
            ga = area(gt_boxes, gt_masks, gt_idx)
            pa = area(pred_boxes, pred_masks, pred_idx)
            gt_areas[gt_idx] = ga
            pred_areas[pred_idx] = pa
            ratio = float(pa / max(ga, 1e-6))
            object_area_ratios.append(ratio)
            object_area_scores.append(float(min(ratio, 1.0 / max(ratio, 1e-6))))

        loc_scores, pair_area_scores = [], []
        for a in range(len(valid_pairs)):
            gi, pi = valid_pairs[a]
            for b in range(a + 1, len(valid_pairs)):
                gj, pj = valid_pairs[b]
                v_gt = center(gt_boxes[gi]) - center(gt_boxes[gj])
                v_pred = center(pred_boxes[pi]) - center(pred_boxes[pj])
                axis_scores = []
                for axis in (0, 1):
                    gt_delta = float(v_gt[axis])
                    pred_delta = float(v_pred[axis])
                    scale = max(1.0, abs(float(center(gt_boxes[gi])[axis])) + abs(float(center(gt_boxes[gj])[axis])))
                    if abs(gt_delta) / scale < axis_equal_tau or abs(pred_delta) / scale < axis_equal_tau:
                        continue
                    axis_scores.append(1.0 if np.sign(gt_delta) == np.sign(pred_delta) else 0.0)
                if axis_scores:
                    loc_scores.append(float(np.mean(axis_scores)))

                gt_ratio = gt_areas[gi] / max(gt_areas[gj], 1e-6)
                pred_ratio = pred_areas[pi] / max(pred_areas[pj], 1e-6)
                rel = pred_ratio / max(gt_ratio, 1e-6)
                pair_area_scores.append(float(min(rel, 1.0 / max(rel, 1e-6))))

        return {
            "Matched_Object_Area_Ratio_Mean": float(np.mean(object_area_ratios)) if object_area_ratios else None,
            "Matched_Object_Area_Ratio_Score": float(np.mean(object_area_scores)) if object_area_scores else None,
            "Matched_Location_Relation": float(np.mean(loc_scores)) if loc_scores else None,
            "Matched_Location_NumPairs": int(len(loc_scores)),
            "Matched_Pair_Area_Ratio_Score": float(np.mean(pair_area_scores)) if pair_area_scores else None,
        }

    def compute_depth_relation_accuracy_matched(
        self,
        depth_gt,
        depth_pred,
        gt_boxes,
        pred_boxes,
        matched_pairs,
        depth_equal_tau: float = 0.03,
    ) -> Dict[str, Any]:
        medians = []
        for gt_idx, pred_idx in matched_pairs or []:
            if gt_idx >= len(gt_boxes) or pred_idx >= len(pred_boxes):
                continue
            gt_med = self._median_valid_depth_in_box(depth_gt, gt_boxes[gt_idx])
            pred_med = self._median_valid_depth_in_box(depth_pred, pred_boxes[pred_idx])
            if gt_med is None or pred_med is None:
                continue
            medians.append((gt_med, pred_med))

        if len(medians) < 2:
            return {"Depth_Relation_Accuracy": None, "Depth_Relation_NumPairs": 0, "Depth_Relation_SkippedPairs": 0}

        correct, total, skipped = 0, 0, 0
        for i in range(len(medians)):
            for j in range(i + 1, len(medians)):
                diff_gt = float(medians[i][0] - medians[j][0])
                diff_pred = float(medians[i][1] - medians[j][1])
                if abs(diff_gt) < depth_equal_tau or abs(diff_pred) < depth_equal_tau:
                    skipped += 1
                    continue
                total += 1
                if (diff_gt < 0 and diff_pred < 0) or (diff_gt > 0 and diff_pred > 0):
                    correct += 1
        return {
            "Depth_Relation_Accuracy": float(correct / total) if total > 0 else None,
            "Depth_Relation_NumPairs": int(total),
            "Depth_Relation_SkippedPairs": int(skipped),
        }

    def compute_morphological_consistency(self, mask_gt, mask_pred, alpha=0.5, beta=0.5):
        def get_morph(m):
            area = np.sum(m > 0)
            if area == 0:
                return 0, 0
            y, x = np.where(m > 0)
            ar = (np.max(x) - np.min(x)) / (np.max(y) - np.min(y) + 1e-6)
            return area, ar

        A_gt, AR_gt = get_morph(mask_gt)
        A_pred, AR_pred = get_morph(mask_pred)
        if A_gt == 0:
            return None
        return alpha * abs(1 - A_pred / (A_gt + 1e-6)) + beta * abs(AR_pred - AR_gt)

    def compute_kendalls_tau(self, depth_gt, depth_pred, instance_masks):
        stats = self.compute_kendalls_tau_stats(depth_gt, depth_pred, instance_masks)
        return stats["tau"]

    def _median_valid_depth(
        self,
        depth: np.ndarray,
        mask: np.ndarray,
        min_depth: float = 0.05,
        max_depth: float = 20.0,
    ) -> Optional[float]:
        if depth is None or mask is None:
            return None

        depth_np = np.asarray(depth, dtype=np.float32).squeeze()
        mask_np = np.asarray(mask).squeeze()
        if depth_np.ndim != 2 or mask_np.ndim != 2:
            return None

        h, w = depth_np.shape[:2]
        if mask_np.shape[:2] != (h, w):
            mask_np = cv2.resize(mask_np.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

        obj = mask_np > 0
        if int(obj.sum()) == 0:
            return None

        vals = depth_np[obj]
        valid = np.isfinite(vals) & (vals > min_depth) & (vals < max_depth)
        vals = vals[valid]
        if vals.size == 0:
            return None
        return float(np.median(vals))

    def _median_valid_depth_in_box(
        self,
        depth: np.ndarray,
        box: List[float],
        min_depth: float = 0.05,
        max_depth: float = 20.0,
        inner_margin_ratio: float = 0.10,
    ) -> Optional[float]:
        if depth is None or box is None:
            return None

        depth_np = np.asarray(depth, dtype=np.float32).squeeze()
        if depth_np.ndim != 2:
            return None

        h, w = depth_np.shape[:2]
        if h <= 0 or w <= 0:
            return None

        x1, y1, x2, y2 = [float(v) for v in box[:4]]
        if not all(np.isfinite([x1, y1, x2, y2])):
            return None
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1

        bw = max(0.0, x2 - x1)
        bh = max(0.0, y2 - y1)
        if bw < 1.0 or bh < 1.0:
            return None

        # Use the central part of the box by default. This reduces background
        # leakage while staying independent of broken segmentation masks.
        mx = min(bw * float(inner_margin_ratio), max(0.0, (bw - 1.0) * 0.45))
        my = min(bh * float(inner_margin_ratio), max(0.0, (bh - 1.0) * 0.45))
        x1 += mx
        x2 -= mx
        y1 += my
        y2 -= my

        ix1 = int(np.floor(max(0.0, min(float(w), x1))))
        iy1 = int(np.floor(max(0.0, min(float(h), y1))))
        ix2 = int(np.ceil(max(0.0, min(float(w), x2))))
        iy2 = int(np.ceil(max(0.0, min(float(h), y2))))
        if ix2 <= ix1 or iy2 <= iy1:
            return None

        vals = depth_np[iy1:iy2, ix1:ix2].reshape(-1)
        valid = np.isfinite(vals) & (vals > min_depth) & (vals < max_depth)
        vals = vals[valid]
        if vals.size == 0:
            return None
        return float(np.median(vals))

    def _kendalls_tau_from_depth_medians(
        self,
        medians_gt: List[float],
        medians_pred: List[float],
        depth_equal_tau: float = 0.03,
    ) -> Dict[str, Any]:
        n_valid = len(medians_gt)
        if n_valid < 2:
            return {"tau": None, "num_objects": n_valid, "num_pairs": 0, "skipped_pairs": 0}

        concordant, discordant, skipped = 0, 0, 0
        for i in range(n_valid):
            for j in range(i + 1, n_valid):
                diff_gt = float(medians_gt[i] - medians_gt[j])
                diff_pred = float(medians_pred[i] - medians_pred[j])
                if abs(diff_gt) < depth_equal_tau or abs(diff_pred) < depth_equal_tau:
                    skipped += 1
                    continue

                same_order = (diff_gt < 0 and diff_pred < 0) or (diff_gt > 0 and diff_pred > 0)
                if same_order:
                    concordant += 1
                else:
                    discordant += 1

        compared = concordant + discordant
        tau = (concordant - discordant) / compared if compared > 0 else None
        return {
            "tau": float(tau) if tau is not None else None,
            "num_objects": int(n_valid),
            "num_pairs": int(compared),
            "skipped_pairs": int(skipped),
        }

    def compute_kendalls_tau_stats(
        self,
        depth_gt,
        depth_pred,
        instance_masks,
        depth_equal_tau: float = 0.03,
    ) -> Dict[str, Any]:
        medians_gt, medians_pred = [], []
        for mask in instance_masks or []:
            gt_med = self._median_valid_depth(depth_gt, mask)
            pred_med = self._median_valid_depth(depth_pred, mask)
            if gt_med is None or pred_med is None:
                continue
            medians_gt.append(gt_med)
            medians_pred.append(pred_med)

        return self._kendalls_tau_from_depth_medians(
            medians_gt=medians_gt,
            medians_pred=medians_pred,
            depth_equal_tau=depth_equal_tau,
        )

    def compute_kendalls_tau_stats_boxes(
        self,
        depth_gt,
        depth_pred,
        boxes,
        depth_equal_tau: float = 0.03,
    ) -> Dict[str, Any]:
        medians_gt, medians_pred = [], []
        for box in boxes or []:
            gt_med = self._median_valid_depth_in_box(depth_gt, box)
            pred_med = self._median_valid_depth_in_box(depth_pred, box)
            if gt_med is None or pred_med is None:
                continue
            medians_gt.append(gt_med)
            medians_pred.append(pred_med)

        return self._kendalls_tau_from_depth_medians(
            medians_gt=medians_gt,
            medians_pred=medians_pred,
            depth_equal_tau=depth_equal_tau,
        )

    def compute_kendalls_tau_matched(
        self,
        depth_gt,
        depth_pred,
        gt_boxes,
        pred_boxes,
        matched_pairs,
        depth_equal_tau: float = 0.03,
    ) -> Dict[str, Any]:
        medians_gt, medians_pred = [], []
        for gt_idx, pred_idx in matched_pairs or []:
            if gt_idx >= len(gt_boxes) or pred_idx >= len(pred_boxes):
                continue
            gt_med = self._median_valid_depth_in_box(depth_gt, gt_boxes[gt_idx])
            pred_med = self._median_valid_depth_in_box(depth_pred, pred_boxes[pred_idx])
            if gt_med is None or pred_med is None:
                continue
            medians_gt.append(gt_med)
            medians_pred.append(pred_med)

        return self._kendalls_tau_from_depth_medians(
            medians_gt=medians_gt,
            medians_pred=medians_pred,
            depth_equal_tau=depth_equal_tau,
        )


# ==========================================
# Main Runner
# ==========================================
class SspDetectionRunner:
    def __init__(
        self,
        eval_mode: str = "dinov3",
        oracle_mode: bool = False,
        save_vis: bool = False,
        device: Optional[str] = None,
        da3_src_path: Optional[str] = None,
        da3_model_path: Optional[str] = None,
        text_prompts: Optional[List[str]] = None,
        proposal_mode: str = "text",
        gt_object_source: str = "target_visible_seg",
        visibility_overlap_tau: float = 0.25,
        visibility_dilate_kernel: int = 11,
        enable_sam3: bool = True,
        enable_dino: bool = True,
        enable_da3: bool = True,
        grounding_dino_path: Optional[str] = None,
        enable_grounding_dino: bool = True,
        gdino_box_threshold: float = 0.25,
        gdino_text_threshold: float = 0.25,
        da3_max_failures: int = 3,
        enable_vlm_match: bool = False,
        vlm_backend: str = "none",
        vlm_model: str = DEFAULT_VLM_MODEL,
        vlm_base_url: Optional[str] = None,
        vlm_api_key_env: str = "OPENAI_API_KEY",
        vlm_timeout: float = 120.0,
        vlm_max_output_tokens: int = 2048,
        vlm_max_retries: int = 2,
        vlm_retry_sleep: float = 1.0,
        vlm_local_device_map: str = "auto",
        vlm_local_torch_dtype: str = "auto",
        vlm_local_attn_implementation: Optional[str] = None,
        vlm_local_min_pixels: Optional[int] = None,
        vlm_local_max_pixels: Optional[int] = None,
        vlm_disable_env_proxy: bool = False,
        vlm_json_response_format: bool = False,
        vlm_max_candidate_pairs: int = 120,
        vlm_candidate_iou_tau: float = 0.01,
        vlm_candidate_center_tau: float = 0.55,
        vlm_require_orientation: bool = True,
        vlm_min_match_confidence: float = 0.8,
        vlm_geometric_iou_tau: float = 0.15,
        vlm_geometric_mask_iou_tau: float = 0.05,
        vlm_geometric_center_tau: float = 0.08,
        filter_matchable_objects: bool = True,
        match_object_min_box_area_ratio: float = 0.006,
        match_object_min_side_ratio: float = 0.035,
        match_object_partial_max_area_ratio: float = 0.03,
        match_object_near_border_ratio: float = 0.04,
        match_object_near_border_thin_side_ratio: float = 0.09,
        match_object_near_border_max_area_ratio: float = 0.05,
        match_object_ignore_labels: Optional[List[str]] = None,
        match_object_max_instances: int = 12,
        match_object_max_per_label: int = 4,
        match_object_repeated_label_max_per_label: int = 3,
        match_object_dedup_iou_tau: float = 0.55,
        match_object_dedup_cover_tau: float = 0.72,
        match_object_dedup_contain_tau: float = 0.85,
        match_object_dedup_center_tau: float = 0.12,
        match_object_keep_labels: Optional[List[str]] = None,
        enable_gt_neighborhood_filter: bool = True,
        gt_neighborhood_expand_ratio: float = 0.75,
        gt_neighborhood_min_margin_ratio: float = 0.06,
        gt_neighborhood_center_tau: float = 0.18,
    ):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.eval_mode = eval_mode
        self.oracle_mode = oracle_mode
        self.save_vis = save_vis
        self.proposal_mode = str(proposal_mode or "text").lower()
        self.gt_object_source = str(gt_object_source or "target_visible_seg").lower()
        self.visibility_overlap_tau = float(visibility_overlap_tau)
        self.visibility_dilate_kernel = int(visibility_dilate_kernel)
        self.text_prompts = text_prompts if text_prompts is not None else ["object", "wall", "floor", "ceiling", "window", "door"]
        self.enable_vlm_match = bool(enable_vlm_match) and str(vlm_backend or "none").lower() != "none"
        self.vlm_max_candidate_pairs = int(vlm_max_candidate_pairs)
        self.vlm_max_output_tokens = int(vlm_max_output_tokens)
        self.vlm_local_device_map = str(vlm_local_device_map or "auto")
        self.vlm_local_torch_dtype = str(vlm_local_torch_dtype or "auto")
        self.vlm_local_attn_implementation = vlm_local_attn_implementation
        self.vlm_local_min_pixels = vlm_local_min_pixels
        self.vlm_local_max_pixels = vlm_local_max_pixels
        self.vlm_disable_env_proxy = bool(vlm_disable_env_proxy)
        self.vlm_json_response_format = bool(vlm_json_response_format)
        self.vlm_candidate_iou_tau = float(vlm_candidate_iou_tau)
        self.vlm_candidate_center_tau = float(vlm_candidate_center_tau)
        self.vlm_require_orientation = bool(vlm_require_orientation)
        self.vlm_min_match_confidence = float(vlm_min_match_confidence)
        self.vlm_geometric_iou_tau = float(vlm_geometric_iou_tau)
        self.vlm_geometric_mask_iou_tau = float(vlm_geometric_mask_iou_tau)
        self.vlm_geometric_center_tau = float(vlm_geometric_center_tau)
        self.filter_matchable_objects = bool(filter_matchable_objects)
        self.match_object_min_box_area_ratio = float(match_object_min_box_area_ratio)
        self.match_object_min_side_ratio = float(match_object_min_side_ratio)
        self.match_object_partial_max_area_ratio = float(match_object_partial_max_area_ratio)
        self.match_object_near_border_ratio = float(match_object_near_border_ratio)
        self.match_object_near_border_thin_side_ratio = float(match_object_near_border_thin_side_ratio)
        self.match_object_near_border_max_area_ratio = float(match_object_near_border_max_area_ratio)
        self.match_object_ignore_labels = {
            normalize_detector_label(x)
            for x in (match_object_ignore_labels or [])
            if normalize_detector_label(x)
        }
        self.match_object_max_instances = int(match_object_max_instances)
        self.match_object_max_per_label = int(match_object_max_per_label)
        self.match_object_repeated_label_max_per_label = int(match_object_repeated_label_max_per_label)
        self.match_object_dedup_iou_tau = float(match_object_dedup_iou_tau)
        self.match_object_dedup_cover_tau = float(match_object_dedup_cover_tau)
        self.match_object_dedup_contain_tau = float(match_object_dedup_contain_tau)
        self.match_object_dedup_center_tau = float(match_object_dedup_center_tau)
        self.match_object_keep_labels = {
            normalize_detector_label(x)
            for x in (match_object_keep_labels or [])
            if normalize_detector_label(x)
        }
        self.enable_gt_neighborhood_filter = bool(enable_gt_neighborhood_filter)
        self.gt_neighborhood_expand_ratio = float(gt_neighborhood_expand_ratio)
        self.gt_neighborhood_min_margin_ratio = float(gt_neighborhood_min_margin_ratio)
        self.gt_neighborhood_center_tau = float(gt_neighborhood_center_tau)
        self.vlm_matcher = None

        self.sem_tools = SemanticToolsAPI(
            device=self.device,
            da3_src_path=da3_src_path,
            da3_model_path=da3_model_path,
            enable_sam3=enable_sam3,
            enable_dino=enable_dino,
            enable_da3=enable_da3,
            grounding_dino_path=grounding_dino_path,
            enable_grounding_dino=enable_grounding_dino,
            gdino_box_threshold=gdino_box_threshold,
            gdino_text_threshold=gdino_text_threshold,
            da3_max_failures=da3_max_failures,
            proposal_mode=self.proposal_mode,
        )
        self.sem_evaluator = SemanticConsistencyEvaluator(device=self.device)

        if self.enable_vlm_match:
            try:
                print(f"Initializing VLM object matcher: backend={vlm_backend}, model={vlm_model}")
                self.vlm_matcher = VLMObjectMatchJudge(
                    backend=vlm_backend,
                    model=vlm_model,
                    api_key_env=vlm_api_key_env,
                    base_url=vlm_base_url,
                    max_output_tokens=vlm_max_output_tokens,
                    timeout=vlm_timeout,
                    max_retries=vlm_max_retries,
                    retry_sleep=vlm_retry_sleep,
                    require_orientation=vlm_require_orientation,
                    min_match_confidence=vlm_min_match_confidence,
                    local_device_map=vlm_local_device_map,
                    local_torch_dtype=vlm_local_torch_dtype,
                    local_attn_implementation=vlm_local_attn_implementation,
                    local_min_pixels=vlm_local_min_pixels,
                    local_max_pixels=vlm_local_max_pixels,
                    disable_env_proxy=vlm_disable_env_proxy,
                    json_response_format=vlm_json_response_format,
                )
                self.sem_tools.model_status["vlm_match"] = self.vlm_matcher.status
            except Exception as e:
                print(f"VLM object matcher init failed, falling back to default matching: {e}")
                self.vlm_matcher = None
                self.enable_vlm_match = False
                self.sem_tools.model_status["vlm_match"] = f"disabled: {e}"
        else:
            self.sem_tools.model_status["vlm_match"] = "disabled by config"

        if self.save_vis:
            self.vis_dir = "debug_vis"
            os.makedirs(self.vis_dir, exist_ok=True)

    def _prompts_for_sample(self, sample: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        case_object_prompts = dedupe_detector_labels(sample.get("object_prompts", []), max_labels=20)
        final_prompts = dedupe_detector_labels(case_object_prompts + list(self.text_prompts), max_labels=64)
        if not final_prompts:
            final_prompts = dedupe_detector_labels(self.text_prompts, max_labels=64)
        return final_prompts, case_object_prompts

    def _empty_scores(self, case_id: str, reason: Optional[str] = None) -> dict:
        scores = {
            "id": case_id,
            "mode": self.eval_mode,
            "proposal_mode": self.proposal_mode,
            "gt_object_source": self.gt_object_source,
            "has_generated_image": False,
            "Recall": 0.0,
            "Center_Error": None,
            "Morph_Loss": None,
            "Topology": None,
            "Topology_NumPairs": 0,
            "Topology_MatchedPairs": 0,
            "Topology_PositivePairs": 0,
            "Topology_MatchedPairCoverage": None,
            "Kendalls_Tau": None,
            "Kendalls_Tau_DA3_Matched": None,
            "Kendalls_Tau_Source": "da3_matched_boxes",
            "Kendalls_Tau_DA3_NumObjects": 0,
            "Kendalls_Tau_DA3_NumPairs": 0,
            "Kendalls_Tau_DA3_SkippedPairs": 0,
            "Oracle_GTDepth_Kendalls_Tau": None,
            "Oracle_GTDepth_Kendalls_NumObjects": 0,
            "Oracle_GTDepth_Kendalls_NumPairs": 0,
            "Matched_Object_Area_Ratio_Mean": None,
            "Matched_Object_Area_Ratio_Score": None,
            "Matched_Location_Relation": None,
            "Matched_Location_NumPairs": 0,
            "Matched_Pair_Area_Ratio_Score": None,
            "Depth_Relation_Accuracy": None,
            "Depth_Relation_NumPairs": 0,
            "Depth_Relation_SkippedPairs": 0,
        }
        for prefix in ("GTImage", "OraclePred", "Pred"):
            scores.update({
                f"{prefix}_Recall": 0.0,
                f"{prefix}_Center_Error": None,
                f"{prefix}_Morph_Loss": None,
                f"{prefix}_Topology": None,
                f"{prefix}_Topology_NumPairs": 0,
                f"{prefix}_Topology_MatchedPairs": 0,
                f"{prefix}_Topology_PositivePairs": 0,
                f"{prefix}_Topology_MatchedPairCoverage": None,
                f"{prefix}_Kendalls_Tau": None,
                f"{prefix}_Kendalls_Tau_DA3_Matched": None,
                f"{prefix}_Kendalls_Tau_Source": "da3_matched_boxes",
                f"{prefix}_Kendalls_Tau_DA3_NumObjects": 0,
                f"{prefix}_Kendalls_Tau_DA3_NumPairs": 0,
                f"{prefix}_Kendalls_Tau_DA3_SkippedPairs": 0,
                f"{prefix}_num_gt_boxes": 0,
                f"{prefix}_num_pred_boxes": 0,
                f"{prefix}_num_matched_pairs": 0,
                f"{prefix}_Matched_Object_Area_Ratio_Mean": None,
                f"{prefix}_Matched_Object_Area_Ratio_Score": None,
                f"{prefix}_Matched_Location_Relation": None,
                f"{prefix}_Matched_Location_NumPairs": 0,
                f"{prefix}_Matched_Pair_Area_Ratio_Score": None,
                f"{prefix}_Depth_Relation_Accuracy": None,
                f"{prefix}_Depth_Relation_NumPairs": 0,
                f"{prefix}_Depth_Relation_SkippedPairs": 0,
                f"{prefix}_vlm_match_enabled": False,
                f"{prefix}_num_vlm_candidate_pairs": 0,
                f"{prefix}_num_vlm_calls": 0,
                f"{prefix}_num_vlm_same_object_pairs": 0,
                f"{prefix}_num_vlm_orientation_pass_pairs": 0,
            })
        if reason:
            scores["skip_reason"] = reason
        return scores

    def _compute_target_visibility_mask(self, ctx_items: List[Any], tgt_item: Any, tgt_shape: Tuple[int, int]) -> np.ndarray:
        """Project valid context-depth support into the target image."""
        H_tgt, W_tgt = tgt_shape
        visible_mask = np.zeros((H_tgt, W_tgt), dtype=np.uint8)

        for ctx_i, ctx_item in enumerate(ctx_items):
            if (
                getattr(ctx_item, "depth", None) is None
                or getattr(ctx_item, "extrinsics", None) is None
                or getattr(ctx_item, "intrinsics", None) is None
            ):
                logger.debug("VISIBILITY ctx%d: missing physical geometry, skip", ctx_i)
                continue
            depth = np.asarray(ctx_item.depth, dtype=np.float32)
            valid_depth_mask = (
                np.isfinite(depth)
                & (depth > 0.05)
                & (depth < 20.0)
            ).astype(np.uint8) * 255
            if int((valid_depth_mask > 0).sum()) < 10:
                continue

            _, projected_visible = self.sem_evaluator.project_mask_to_target(
                mask=valid_depth_mask,
                depth=ctx_item.depth,
                T_ctx=ctx_item.extrinsics,
                K_ctx=ctx_item.intrinsics,
                T_tgt=tgt_item.extrinsics,
                K_tgt=tgt_item.intrinsics,
                tgt_shape=tgt_shape,
            )
            if projected_visible is None:
                logger.debug("VISIBILITY ctx%d: projection failed", ctx_i)
                continue

            visible_mask = np.maximum(visible_mask, (projected_visible > 0).astype(np.uint8) * 255)

        if int((visible_mask > 0).sum()) > 0:
            k = max(1, self.visibility_dilate_kernel)
            if k % 2 == 0:
                k += 1
            kernel = np.ones((k, k), np.uint8)
            visible_mask = cv2.dilate(visible_mask, kernel, iterations=1)
            visible_mask = cv2.morphologyEx(visible_mask, cv2.MORPH_CLOSE, kernel)

        return visible_mask

    def _filter_target_instances_by_visibility(
        self,
        boxes: List[List[int]],
        masks: List[np.ndarray],
        feats: List[Optional[torch.Tensor]],
        visible_mask: np.ndarray,
        labels: Optional[List[str]] = None,
    ) -> Tuple[List[List[int]], List[np.ndarray], List[Optional[torch.Tensor]], List[str], List[float]]:
        if visible_mask is None or int((visible_mask > 0).sum()) == 0:
            return boxes, masks, feats, list(labels or [""] * len(boxes)), []

        H, W = visible_mask.shape[:2]
        keep_boxes, keep_masks, keep_feats, keep_labels, visible_ratios = [], [], [], [], []
        visible_bool = visible_mask > 0

        for i, (box, mask) in enumerate(zip(boxes, masks)):
            mask_np = np.asarray(mask)
            if mask_np.shape[:2] != (H, W):
                mask_np = cv2.resize(mask_np.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
            obj_bool = mask_np > 0
            obj_area = int(obj_bool.sum())
            if obj_area <= 0:
                continue
            visible_area = int(np.logical_and(obj_bool, visible_bool).sum())
            visible_ratio = float(visible_area / max(obj_area, 1))
            if visible_ratio >= self.visibility_overlap_tau:
                keep_boxes.append(box)
                keep_masks.append((obj_bool.astype(np.uint8) * 255))
                keep_feats.append(feats[i] if i < len(feats) else None)
                keep_labels.append(normalize_detector_label(labels[i]) if labels and i < len(labels) else "")
                visible_ratios.append(visible_ratio)

        return keep_boxes, keep_masks, keep_feats, keep_labels, visible_ratios

    def _filter_candidate_instances_by_visibility(
        self,
        boxes: List[List[int]],
        masks: List[np.ndarray],
        visible_mask: Optional[np.ndarray],
        labels: Optional[List[str]] = None,
    ) -> Tuple[List[List[int]], List[np.ndarray], List[str], List[float]]:
        """Keep target-side candidate proposals only where source/context has support."""
        filtered_boxes, filtered_masks, _, filtered_labels, ratios = self._filter_target_instances_by_visibility(
            boxes=boxes,
            masks=masks,
            feats=[None] * len(boxes),
            visible_mask=visible_mask,
            labels=labels,
        )
        return filtered_boxes, filtered_masks, filtered_labels, ratios

    def _filter_pred_instances_by_gt_neighborhood(
        self,
        pred_boxes: List[List[int]],
        pred_masks: List[np.ndarray],
        pred_labels: Optional[List[str]],
        gt_boxes: List[List[int]],
        gt_labels: Optional[List[str]],
        image_shape: Tuple[int, int],
    ) -> Tuple[List[List[int]], List[np.ndarray], List[str], List[Dict[str, Any]]]:
        """Keep pred boxes only near a target-GT box, to avoid same-category far matches."""
        pred_labels = list(pred_labels or [""] * len(pred_boxes))
        if not self.enable_gt_neighborhood_filter or not gt_boxes:
            return pred_boxes, pred_masks, [normalize_detector_label(x) for x in pred_labels], []

        H, W = int(image_shape[0]), int(image_shape[1])
        diag = float(max(np.hypot(H, W), 1.0))
        min_margin = float(self.gt_neighborhood_min_margin_ratio * max(H, W))
        keep_boxes, keep_masks, keep_labels = [], [], []
        removed = []

        def center(box):
            x1, y1, x2, y2 = [float(v) for v in box[:4]]
            return np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)

        def expanded_contains(gt_box, pred_center) -> bool:
            x1, y1, x2, y2 = [float(v) for v in gt_box[:4]]
            bw = max(1.0, x2 - x1)
            bh = max(1.0, y2 - y1)
            mx = max(min_margin, bw * self.gt_neighborhood_expand_ratio)
            my = max(min_margin, bh * self.gt_neighborhood_expand_ratio)
            return (x1 - mx) <= float(pred_center[0]) <= (x2 + mx) and (y1 - my) <= float(pred_center[1]) <= (y2 + my)

        for idx, (box, mask) in enumerate(zip(pred_boxes, pred_masks)):
            sb = safe_box_xyxy(box, H, W)
            if sb is None:
                removed.append({"idx": int(idx), "label": normalize_detector_label(pred_labels[idx]) if idx < len(pred_labels) else "", "reasons": ["invalid_box_before_gt_neighborhood"]})
                continue
            pred_label = normalize_detector_label(pred_labels[idx]) if idx < len(pred_labels) else ""
            pc = center(sb)
            best_center_norm = None
            best_iou = 0.0
            matched_neighborhood = False
            matched_label_compatible = False

            for gt_idx, gt_box in enumerate(gt_boxes):
                gt_label = normalize_detector_label(gt_labels[gt_idx]) if gt_labels and gt_idx < len(gt_labels) else ""
                label_ok = _labels_compatible_for_object_match(gt_label, pred_label)
                if not label_ok:
                    continue
                matched_label_compatible = True
                gt_center = center(gt_box)
                center_norm = float(np.linalg.norm(pc - gt_center) / diag)
                iou = float(_box_iou_xyxy(gt_box, sb))
                best_center_norm = center_norm if best_center_norm is None else min(best_center_norm, center_norm)
                best_iou = max(best_iou, iou)
                if iou > 0.0 or expanded_contains(gt_box, pc) or center_norm <= self.gt_neighborhood_center_tau:
                    matched_neighborhood = True
                    break

            if matched_neighborhood:
                keep_boxes.append(box)
                keep_masks.append(mask)
                keep_labels.append(pred_label)
            else:
                reason = "outside_gt_neighborhood"
                if not matched_label_compatible:
                    reason = "no_label_compatible_gt_neighborhood"
                removed.append({
                    "idx": int(idx),
                    "label": pred_label,
                    "box": [float(x) for x in sb[:4]],
                    "best_iou": float(best_iou),
                    "best_center_norm": float(best_center_norm) if best_center_norm is not None else None,
                    "reasons": [reason],
                })

        return keep_boxes, keep_masks, keep_labels, removed

    def _rank_matchable_instance(
        self,
        box: List[int],
        label: str,
        image_shape: Tuple[int, int],
    ) -> float:
        H, W = int(image_shape[0]), int(image_shape[1])
        sb = safe_box_xyxy(box, H, W)
        if sb is None:
            return -1.0
        x1, y1, x2, y2 = sb
        area_ratio = float(max(1, (x2 - x1) * (y2 - y1)) / max(H * W, 1))
        label = normalize_detector_label(label)
        priority = 0.0
        if label in {"table", "desk", "counter", "table counter", "bed", "sofa", "couch", "cabinet", "bookshelf"}:
            priority += 0.08
        if label in {"shelf", "chair", "stool", "curtain", "window", "door"}:
            priority -= 0.02
        return area_ratio + priority

    def _dedupe_matchable_instances(
        self,
        items: List[Dict[str, Any]],
        image_shape: Tuple[int, int],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Suppress same-object detector fragments before object matching."""
        H, W = int(image_shape[0]), int(image_shape[1])
        diag = float(max(np.hypot(H, W), 1.0))
        structural_labels = {"window", "door", "curtain", "shelf", "bookshelf", "cabinet"}
        kept: List[Dict[str, Any]] = []
        removed: List[Dict[str, Any]] = []

        def box_stats(box_a, box_b):
            ax1, ay1, ax2, ay2 = [float(v) for v in box_a[:4]]
            bx1, by1, bx2, by2 = [float(v) for v in box_b[:4]]
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
            area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
            union = area_a + area_b - inter
            iou = float(inter / union) if union > 0 else 0.0
            cover_a = float(inter / area_a) if area_a > 0 else 0.0
            cover_b = float(inter / area_b) if area_b > 0 else 0.0
            cover_small = float(inter / min(area_a, area_b)) if min(area_a, area_b) > 0 else 0.0
            return iou, cover_a, cover_b, cover_small, area_a, area_b

        def center_norm(box_a, box_b) -> float:
            ac = np.array([(float(box_a[0]) + float(box_a[2])) * 0.5, (float(box_a[1]) + float(box_a[3])) * 0.5])
            bc = np.array([(float(box_b[0]) + float(box_b[2])) * 0.5, (float(box_b[1]) + float(box_b[3])) * 0.5])
            return float(np.linalg.norm(ac - bc) / diag)

        sorted_items = sorted(
            items,
            key=lambda x: (
                float(x.get("score", 0.0)),
                float(_box_area_xyxy(x.get("box", [0, 0, 0, 0]))),
            ),
            reverse=True,
        )

        for item in sorted_items:
            label = str(item.get("label", ""))
            sb = safe_box_xyxy(item.get("box"), H, W)
            if sb is None:
                removed.append({
                    "idx": int(item.get("idx", -1)),
                    "label": label,
                    "box": [float(x) for x in item.get("box", [])[:4]],
                    "reasons": ["invalid_box_during_dedup"],
                })
                continue

            duplicate_of = None
            duplicate_reason = None
            duplicate_stats = None
            for kept_item in kept:
                kept_label = str(kept_item.get("label", ""))
                if not _labels_compatible_for_object_match(label, kept_label):
                    continue
                kept_box = safe_box_xyxy(kept_item.get("box"), H, W)
                if kept_box is None:
                    continue

                iou, cover_item, cover_kept, cover_small, area_item, area_kept = box_stats(sb, kept_box)
                cn = center_norm(sb, kept_box)
                iou_tau = self.match_object_dedup_iou_tau
                cover_tau = self.match_object_dedup_cover_tau
                contain_tau = self.match_object_dedup_contain_tau
                center_tau = self.match_object_dedup_center_tau
                if label in structural_labels or kept_label in structural_labels:
                    iou_tau = min(iou_tau, 0.45)
                    cover_tau = min(cover_tau, 0.62)
                    contain_tau = min(contain_tau, 0.78)
                    center_tau = max(center_tau, 0.16)

                if iou >= iou_tau:
                    duplicate_reason = f"duplicate_same_object_iou:{iou:.3f}>={iou_tau:.3f}"
                elif cover_item >= cover_tau:
                    duplicate_reason = f"duplicate_candidate_covered:{cover_item:.3f}>={cover_tau:.3f}"
                elif cover_small >= contain_tau and cn <= center_tau:
                    duplicate_reason = (
                        f"duplicate_contained_same_object:contain={cover_small:.3f}>={contain_tau:.3f},"
                        f"center={cn:.3f}<={center_tau:.3f}"
                    )

                if duplicate_reason:
                    duplicate_of = kept_item
                    duplicate_stats = {
                        "iou": float(iou),
                        "candidate_covered": float(cover_item),
                        "kept_covered": float(cover_kept),
                        "smaller_covered": float(cover_small),
                        "center_norm": float(cn),
                        "candidate_area": float(area_item),
                        "kept_area": float(area_kept),
                    }
                    break

            if duplicate_of is not None:
                removed.append({
                    "idx": int(item.get("idx", -1)),
                    "label": label,
                    "box": [float(x) for x in sb[:4]],
                    "score": float(item.get("score", 0.0)),
                    "duplicate_of_idx": int(duplicate_of.get("idx", -1)),
                    "duplicate_of_label": str(duplicate_of.get("label", "")),
                    "duplicate_of_box": [float(x) for x in duplicate_of.get("box", [])[:4]],
                    **(duplicate_stats or {}),
                    "reasons": [duplicate_reason or "duplicate_same_object"],
                })
                continue

            kept.append(item)

        kept.sort(key=lambda x: int(x.get("idx", -1)))
        return kept, removed

    def _filter_instances_for_matching(
        self,
        boxes: List[List[int]],
        masks: List[np.ndarray],
        labels: Optional[List[str]],
        feats: Optional[List[Optional[torch.Tensor]]] = None,
        image_shape: Optional[Tuple[int, int]] = None,
    ) -> Tuple[List[List[int]], List[np.ndarray], List[Optional[torch.Tensor]], List[str], List[Dict[str, Any]]]:
        """Remove tiny, fragmented, or partial edge objects before VLM/object metrics."""
        labels = list(labels or [""] * len(boxes))
        feats = list(feats) if feats is not None else [None] * len(boxes)
        if not self.filter_matchable_objects or image_shape is None:
            return boxes, masks, feats, [normalize_detector_label(x) for x in labels], []

        H, W = int(image_shape[0]), int(image_shape[1])
        total_area = float(max(H * W, 1))
        min_side_base = float(max(min(H, W), 1))
        keep_items = []
        removed = []

        for idx, (box, mask) in enumerate(zip(boxes, masks)):
            label = normalize_detector_label(labels[idx]) if idx < len(labels) else ""
            sb = safe_box_xyxy(box, H, W)
            if sb is None:
                removed.append({
                    "idx": int(idx),
                    "label": label,
                    "box": [float(x) for x in box[:4]],
                    "reasons": ["invalid_box"],
                })
                continue

            x1, y1, x2, y2 = sb
            bw, bh = max(1, x2 - x1), max(1, y2 - y1)
            box_area_ratio = float((bw * bh) / total_area)
            min_side_ratio = float(min(bw, bh) / min_side_base)
            touches_border = int(x1 <= 1) + int(y1 <= 1) + int(x2 >= W - 2) + int(y2 >= H - 2)
            border_margin_x = max(2.0, float(W) * self.match_object_near_border_ratio)
            border_margin_y = max(2.0, float(H) * self.match_object_near_border_ratio)
            near_border = (
                x1 <= border_margin_x
                or y1 <= border_margin_y
                or x2 >= float(W) - border_margin_x
                or y2 >= float(H) - border_margin_y
            )
            aspect_ratio = float(max(bw, bh) / max(min(bw, bh), 1))
            reasons = []

            if label in self.match_object_ignore_labels:
                reasons.append(f"ignored_label:{label}")
            if box_area_ratio < self.match_object_min_box_area_ratio:
                reasons.append(f"small_box_area:{box_area_ratio:.4f}<{self.match_object_min_box_area_ratio:.4f}")
            if min_side_ratio < self.match_object_min_side_ratio:
                reasons.append(f"small_box_side:{min_side_ratio:.4f}<{self.match_object_min_side_ratio:.4f}")
            if touches_border > 0 and box_area_ratio < self.match_object_partial_max_area_ratio:
                reasons.append(
                    f"partial_edge_small:touches={touches_border},area={box_area_ratio:.4f}<"
                    f"{self.match_object_partial_max_area_ratio:.4f}"
                )
            if touches_border >= 2 and box_area_ratio < self.match_object_partial_max_area_ratio * 2.0:
                reasons.append(
                    f"partial_corner:touches={touches_border},area={box_area_ratio:.4f}<"
                    f"{self.match_object_partial_max_area_ratio * 2.0:.4f}"
                )
            if (
                near_border
                and box_area_ratio < self.match_object_near_border_max_area_ratio
                and (
                    min_side_ratio < self.match_object_near_border_thin_side_ratio
                    or aspect_ratio >= 3.5
                )
            ):
                reasons.append(
                    f"near_border_thin_fragment:near=1,area={box_area_ratio:.4f}<"
                    f"{self.match_object_near_border_max_area_ratio:.4f},"
                    f"side={min_side_ratio:.4f}<"
                    f"{self.match_object_near_border_thin_side_ratio:.4f},"
                    f"aspect={aspect_ratio:.2f}"
                )

            if reasons:
                removed.append({
                    "idx": int(idx),
                    "label": label,
                    "box": [float(x) for x in box[:4]],
                    "box_area_ratio": box_area_ratio,
                    "min_side_ratio": min_side_ratio,
                    "touches_border": int(touches_border),
                    "near_border": bool(near_border),
                    "aspect_ratio": float(aspect_ratio),
                    "reasons": reasons,
                })
                continue

            keep_items.append({
                "idx": int(idx),
                "box": box,
                "mask": mask,
                "feat": feats[idx] if idx < len(feats) else None,
                "label": label,
                "score": self._rank_matchable_instance(box, label, (H, W)),
            })

        keep_items, dedup_removed = self._dedupe_matchable_instances(keep_items, (H, W))
        removed.extend(dedup_removed)

        repeated_labels = {"chair", "stool", "shelf", "bookshelf", "window", "door", "curtain"}
        per_label_counts: Dict[str, int] = {}
        budgeted_items = []
        for item in sorted(keep_items, key=lambda x: float(x.get("score", 0.0)), reverse=True):
            label = str(item.get("label", ""))
            per_label_cap = (
                self.match_object_repeated_label_max_per_label
                if label in repeated_labels
                else self.match_object_max_per_label
            )
            if per_label_cap > 0 and per_label_counts.get(label, 0) >= per_label_cap:
                removed.append({
                    "idx": int(item["idx"]),
                    "label": label,
                    "box": [float(x) for x in item["box"][:4]],
                    "score": float(item.get("score", 0.0)),
                    "reasons": [f"per_label_budget:{label}>={per_label_cap}"],
                })
                continue
            budgeted_items.append(item)
            per_label_counts[label] = per_label_counts.get(label, 0) + 1

        max_instances = self.match_object_max_instances
        if max_instances > 0 and len(budgeted_items) > max_instances:
            keep_ids = {int(x["idx"]) for x in budgeted_items[:max_instances]}
            for item in budgeted_items[max_instances:]:
                removed.append({
                    "idx": int(item["idx"]),
                    "label": str(item.get("label", "")),
                    "box": [float(x) for x in item["box"][:4]],
                    "score": float(item.get("score", 0.0)),
                    "reasons": [f"global_budget>{max_instances}"],
                })
            budgeted_items = [x for x in budgeted_items if int(x["idx"]) in keep_ids]

        budgeted_items.sort(key=lambda x: int(x["idx"]))
        keep_boxes = [x["box"] for x in budgeted_items]
        keep_masks = [x["mask"] for x in budgeted_items]
        keep_feats = [x["feat"] for x in budgeted_items]
        keep_labels = [x["label"] for x in budgeted_items]
        return keep_boxes, keep_masks, keep_feats, keep_labels, removed

    def evaluate_step_sample(self, sample: dict):
        case_id = sample["id"]
        file_case_id = safe_filename_token(case_id)
        ctx_paths = sample["context_paths"]
        tgt_gt_path = sample["target_path"]
        tgt_gen_path = sample["generated_path"]

        if not ctx_paths or not tgt_gt_path:
            return self._empty_scores(case_id, "missing context/target")

        dataset_name = sample.get("dataset")
        scene_id = sample.get("scene_id")
        if not dataset_name or not scene_id:
            dataset_name, scene_id = infer_dataset_and_scene_from_path(ctx_paths[0])

        ctx_geometry_paths = sample.get("context_geometry_paths") or ctx_paths
        ctx_items = [
            get_context_frame_for_metric(
                dataset_name,
                scene_id,
                p,
                ctx_geometry_paths[i] if i < len(ctx_geometry_paths) else p,
            )
            for i, p in enumerate(ctx_paths)
        ]
        tgt_item = get_frame_from_scene(dataset_name, scene_id, tgt_gt_path)

        has_generated_image = bool(tgt_gen_path)
        gen_img_cv2 = None
        pred_skip_reason = None
        if not tgt_gen_path:
            pred_skip_reason = "missing generated_images step image_path"
        elif os.path.isabs(tgt_gen_path) or os.path.exists(tgt_gen_path):
            gen_img_cv2 = read_image_cv2_local(tgt_gen_path)
        else:
            try:
                gen_img_cv2 = get_frame_from_scene(dataset_name, scene_id, tgt_gen_path).image.copy()
            except Exception:
                gen_img_cv2 = read_image_cv2_local(tgt_gen_path)
        if gen_img_cv2 is None and pred_skip_reason is None:
            pred_skip_reason = f"failed to read generated image: {tgt_gen_path}"

        # Unify to target GT resolution (not context size) to keep projection consistent.
        H_gt, W_gt = tgt_item.image.shape[:2]
        if gen_img_cv2 is not None and gen_img_cv2.shape[:2] != (H_gt, W_gt):
            gen_img_cv2 = cv2.resize(gen_img_cv2, (W_gt, H_gt))

        sample_prompts, case_object_prompts = self._prompts_for_sample(sample)
        scores = self._empty_scores(case_id)
        scores["has_generated_image"] = bool(has_generated_image)
        scores["generated_path"] = tgt_gen_path
        scores["oracle_mode"] = bool(self.oracle_mode)
        scores["case_detector_prompt_labels"] = case_object_prompts
        scores["detector_prompt_labels"] = sample_prompts
        scores["num_case_detector_prompt_labels"] = len(case_object_prompts)
        scores["num_detector_prompt_labels"] = len(sample_prompts)
        if pred_skip_reason:
            scores["Pred_skip_reason"] = pred_skip_reason

        raw_gt_boxes, raw_gt_masks, raw_gt_feats = [], [], []
        raw_gt_labels = []
        vis_ctx_imgs, vis_ctx_boxes, vis_ctx_masks = [], [], []
        pred_boxes, pred_masks = [], []
        verified_boxes, verified_masks = [], []
        matched_pairs = []

        src_item = ctx_items[0]
        src_img_path = ctx_paths[0]           # Image 1 is the current/source view.
        tgt_img_path = tgt_gt_path

        gt_source = self.gt_object_source
        if gt_source not in {
            "projected_context",
            "target_gt_seg",
            "target_visible_seg",
            "visible_target_gt",
            "target_seg_visible",
        }:
            print(f"unknown gt_object_source='{gt_source}', fallback to target_visible_seg")
            gt_source = "target_visible_seg"
            scores["gt_object_source_fallback"] = self.gt_object_source
        scores["gt_object_source"] = gt_source
        target_visible_mask = None

        # Context is used only to compute target visible support, not as an object
        # source. The metric GT is always target GT image boxes.
        for ctx_item in ctx_items[:2]:
            vis_ctx_imgs.append(ctx_item.image)
            vis_ctx_boxes.append([])
            vis_ctx_masks.append([])
        if gt_source == "projected_context":
            scores["gt_object_source_requested"] = "projected_context"
            gt_source = "target_visible_seg"
            scores["gt_object_source_fallback"] = "projected_context disabled; using target_visible_seg"

        logger.debug(
            "%s raw_projected=%d, raw_masks=%d, "
            "ctx_object_detection=disabled, gt_source=%s",
            case_id,
            len(raw_gt_boxes),
            len(raw_gt_masks),
            gt_source,
        )

        # --- Phase 2: construct GT objects ---
        if gt_source == "projected_context":
            # Legacy path: context proposal -> depth/pose projection -> pseudo-GT.
            gt_boxes, gt_masks, gt_feats = self.sem_evaluator.merge_projections(
                raw_gt_boxes,
                raw_gt_masks,
                raw_gt_feats,
                iou_thresh=0.55,
            )
            gt_labels = [""] * len(gt_boxes)
            scores["num_raw_projected_gt_boxes"] = len(raw_gt_boxes)
        else:
            # Recommended path: segment the target GT image directly, then optionally
            # filter to regions supported by context visibility.
            target_gt_boxes, target_gt_masks, target_gt_labels = self.sem_tools.get_instances_with_labels(tgt_item.image, sample_prompts)
            target_gt_feats = [
                self.sem_tools.extract_roi_feature(tgt_item.image, box)
                for box in target_gt_boxes
            ]
            scores["num_target_gt_raw_boxes"] = len(target_gt_boxes)

            gt_boxes, gt_masks, gt_feats = target_gt_boxes, target_gt_masks, target_gt_feats
            gt_labels = target_gt_labels
            if gt_source in {"target_visible_seg", "visible_target_gt", "target_seg_visible"}:
                visible_mask = self._compute_target_visibility_mask(
                    ctx_items=ctx_items,
                    tgt_item=tgt_item,
                    tgt_shape=(H_gt, W_gt),
                )
                target_visible_mask = visible_mask
                visible_pixels = int((visible_mask > 0).sum())
                scores["visibility_pixels"] = visible_pixels
                scores["visibility_ratio"] = float(visible_pixels / max(H_gt * W_gt, 1))
                scores["visibility_overlap_tau"] = self.visibility_overlap_tau

                gt_boxes, gt_masks, gt_feats, gt_labels, visible_ratios = self._filter_target_instances_by_visibility(
                    boxes=target_gt_boxes,
                    masks=target_gt_masks,
                    feats=target_gt_feats,
                    visible_mask=visible_mask,
                    labels=target_gt_labels,
                )
                scores["num_target_gt_visible_boxes"] = len(gt_boxes)
                scores["target_gt_visible_ratio_mean"] = float(np.mean(visible_ratios)) if visible_ratios else None
                scores["target_gt_visible_ratio_min"] = float(np.min(visible_ratios)) if visible_ratios else None
            pre_matchable_gt_count = len(gt_boxes)
            gt_boxes, gt_masks, gt_feats, gt_labels, gt_removed = self._filter_instances_for_matching(
                boxes=gt_boxes,
                masks=gt_masks,
                labels=gt_labels,
                feats=gt_feats,
                image_shape=tgt_item.image.shape[:2],
            )
            scores["num_target_gt_matchable_boxes"] = len(gt_boxes)
            scores["num_target_gt_quality_filtered_boxes"] = pre_matchable_gt_count - len(gt_boxes)
            scores["target_gt_object_filter_removed_details"] = gt_removed[:80]
            scores["target_gt_labels"] = gt_labels
            scores["target_gt_boxes"] = [[float(x) for x in box[:4]] for box in gt_boxes]

        logger.debug(
            "%s merged_gt_boxes=%d, merged_gt_masks=%d, gt_source=%s",
            case_id,
            len(gt_boxes),
            len(gt_masks),
            gt_source,
        )

        if len(gt_boxes) == 0:
            print(
                f"[SKIP][{case_id}] no GT objects: "
                f"ctx_proposal_counts={[len(x) for x in vis_ctx_boxes]}, "
                f"raw_projected={len(raw_gt_boxes)}, "
                f"merged_gt_boxes={len(gt_boxes)}, "
                f"gt_source={gt_source}"
            )
            scores["skip_reason"] = f"no GT objects after gt_object_source={gt_source}"
            return scores

        # --- Phase 3: run unified target-side detection, verification, and DA3 depth ordering on candidate image ---
        auto_like_proposal = self.proposal_mode in {
            "auto",
            "cv_auto",
            "proposal",
            "prompt_free",
            "hybrid",
            "auto_text",
            "text_auto",
            "gdino",
            "gdino_box",
            "grounding_dino",
            "grounding_dino_box",
            "gdino_sam3",
            "grounding_dino_sam3",
        }
        match_iou_tau = 0.35 if auto_like_proposal else 0.5
        match_feat_tau = 0.60 if auto_like_proposal else 0.65
        match_mask_iou_tau = None

        oracle_depth_stats = self.sem_evaluator.compute_kendalls_tau_stats_boxes(
            depth_gt=tgt_item.depth,
            depth_pred=tgt_item.depth,
            boxes=gt_boxes,
        )
        scores["Oracle_GTDepth_Kendalls_Tau"] = oracle_depth_stats["tau"]
        scores["Oracle_GTDepth_Kendalls_NumObjects"] = oracle_depth_stats["num_objects"]
        scores["Oracle_GTDepth_Kendalls_NumPairs"] = oracle_depth_stats["num_pairs"]

        # Legacy GT_DA3 fields retained; uses original target path for DA3 sanity check.
        if (
            self.sem_tools.has_da3()
            and os.path.exists(src_img_path)
            and os.path.exists(tgt_img_path)
            and getattr(src_item, "depth", None) is not None
            and getattr(src_item, "extrinsics", None) is not None
        ):
            gt_pose_pred, _, gt_err = self.sem_tools.calculate_pose_da3(
                src_img_input=src_img_path,
                tgt_img_input=tgt_img_path,
                depth_src_gt=src_item.depth,
                pose_src_gl=src_item.extrinsics,
            )
            if gt_pose_pred is not None:
                scores["GT_DA3"] = self.sem_evaluator.compute_pose_metrics(src_item.extrinsics, gt_pose_pred)
            else:
                scores["GT_DA3_error"] = gt_err

        def evaluate_candidate_image(candidate_img_cv2: np.ndarray, candidate_label: str):
            candidate_scores = {
                f"{candidate_label}_Recall": 0.0,
                f"{candidate_label}_Center_Error": None,
                f"{candidate_label}_Morph_Loss": None,
                f"{candidate_label}_Topology": None,
                f"{candidate_label}_Topology_NumPairs": 0,
                f"{candidate_label}_Topology_MatchedPairs": 0,
                f"{candidate_label}_Topology_PositivePairs": 0,
                f"{candidate_label}_Topology_MatchedPairCoverage": None,
                f"{candidate_label}_Kendalls_Tau": None,
                f"{candidate_label}_Kendalls_Tau_DA3_Matched": None,
                f"{candidate_label}_Kendalls_Tau_Source": "da3_matched_boxes",
                f"{candidate_label}_Kendalls_Tau_DA3_NumObjects": 0,
                f"{candidate_label}_Kendalls_Tau_DA3_NumPairs": 0,
                f"{candidate_label}_Kendalls_Tau_DA3_SkippedPairs": 0,
                f"{candidate_label}_match_iou_tau": match_iou_tau,
                f"{candidate_label}_match_feat_tau": match_feat_tau,
                f"{candidate_label}_match_mask_iou_tau": None,
                f"{candidate_label}_match_geometry": "box",
                f"{candidate_label}_require_label_match": True,
                f"{candidate_label}_num_gt_boxes": len(gt_boxes),
                f"{candidate_label}_num_pred_boxes": 0,
                f"{candidate_label}_num_matched_pairs": 0,
                f"{candidate_label}_Matched_Object_Area_Ratio_Mean": None,
                f"{candidate_label}_Matched_Object_Area_Ratio_Score": None,
                f"{candidate_label}_Matched_Location_Relation": None,
                f"{candidate_label}_Matched_Location_NumPairs": 0,
                f"{candidate_label}_Matched_Pair_Area_Ratio_Score": None,
                f"{candidate_label}_Depth_Relation_Accuracy": None,
                f"{candidate_label}_Depth_Relation_NumPairs": 0,
                f"{candidate_label}_Depth_Relation_SkippedPairs": 0,
                f"{candidate_label}_vlm_match_enabled": bool(self.enable_vlm_match and self.vlm_matcher is not None),
                f"{candidate_label}_num_vlm_candidate_pairs": 0,
                f"{candidate_label}_num_vlm_calls": 0,
                f"{candidate_label}_num_vlm_same_object_pairs": 0,
                f"{candidate_label}_num_vlm_orientation_pass_pairs": 0,
            }
            pred_boxes, pred_masks, pred_labels = self.sem_tools.get_instances_with_labels(candidate_img_cv2, sample_prompts)
            raw_pred_count = len(pred_boxes)
            visible_ratios = []
            if target_visible_mask is not None:
                pred_boxes, pred_masks, pred_labels, visible_ratios = self._filter_candidate_instances_by_visibility(
                    boxes=pred_boxes,
                    masks=pred_masks,
                    visible_mask=target_visible_mask,
                    labels=pred_labels,
                )
                candidate_scores[f"{candidate_label}_num_pred_boxes_raw"] = raw_pred_count
                candidate_scores[f"{candidate_label}_num_pred_boxes_masked_out"] = raw_pred_count - len(pred_boxes)
                candidate_scores[f"{candidate_label}_pred_visible_ratio_mean"] = (
                    float(np.mean(visible_ratios)) if visible_ratios else None
                )
            pre_quality_pred_count = len(pred_boxes)
            pre_neighborhood_pred_count = len(pred_boxes)
            pred_boxes, pred_masks, pred_labels, pred_neighborhood_removed = self._filter_pred_instances_by_gt_neighborhood(
                pred_boxes=pred_boxes,
                pred_masks=pred_masks,
                pred_labels=pred_labels,
                gt_boxes=gt_boxes,
                gt_labels=gt_labels,
                image_shape=candidate_img_cv2.shape[:2],
            )
            candidate_scores[f"{candidate_label}_num_pred_boxes_after_visibility"] = pre_quality_pred_count
            candidate_scores[f"{candidate_label}_num_pred_boxes_neighborhood_filtered"] = pre_neighborhood_pred_count - len(pred_boxes)
            candidate_scores[f"{candidate_label}_neighborhood_filter_removed_details"] = pred_neighborhood_removed[:80]

            pre_quality_pred_count = len(pred_boxes)
            pred_boxes, pred_masks, _, pred_labels, pred_removed = self._filter_instances_for_matching(
                boxes=pred_boxes,
                masks=pred_masks,
                labels=pred_labels,
                feats=[None] * len(pred_boxes),
                image_shape=candidate_img_cv2.shape[:2],
            )
            candidate_scores[f"{candidate_label}_num_pred_boxes_quality_filtered"] = pre_quality_pred_count - len(pred_boxes)
            candidate_scores[f"{candidate_label}_object_filter_removed_details"] = pred_removed[:80]
            candidate_scores[f"{candidate_label}_num_pred_boxes"] = len(pred_boxes)
            candidate_scores[f"{candidate_label}_pred_labels"] = pred_labels
            verified_boxes, verified_masks, matched_pairs = [], [], []

            if len(pred_boxes) == 0:
                candidate_scores[f"{candidate_label}_skip_reason"] = "proposal extractor returned no pred boxes"
            else:
                effective_mode = self.eval_mode
                if "dino" in self.eval_mode.lower() and not self.sem_tools.has_dino():
                    effective_mode = "none"
                    candidate_scores[f"{candidate_label}_verification_fallback"] = "dino unavailable -> iou-only"

                if self.enable_vlm_match and self.vlm_matcher is not None and self.vlm_matcher.is_available():
                    (
                        recall,
                        center_err,
                        verified_boxes,
                        matched_pairs,
                        match_details,
                        vlm_stats,
                    ) = self.sem_evaluator.evaluate_permanence_vlm(
                        gt_boxes=gt_boxes,
                        gt_features=gt_feats,
                        pred_boxes=pred_boxes,
                        gt_img_cv2=tgt_item.image,
                        pred_img_cv2=candidate_img_cv2,
                        sem_tools=self.sem_tools,
                        vlm_matcher=self.vlm_matcher,
                        mode=effective_mode,
                        gt_masks=gt_masks,
                        pred_masks=pred_masks,
                        gt_labels=gt_labels,
                        pred_labels=pred_labels,
                        max_candidate_pairs=self.vlm_max_candidate_pairs,
                        candidate_iou_tau=self.vlm_candidate_iou_tau,
                        candidate_center_tau=self.vlm_candidate_center_tau,
                        geometric_iou_tau=self.vlm_geometric_iou_tau,
                        geometric_mask_iou_tau=self.vlm_geometric_mask_iou_tau,
                        geometric_center_tau=self.vlm_geometric_center_tau,
                    )
                    candidate_scores[f"{candidate_label}_match_policy"] = "vlm_global_same_object"
                    candidate_scores[f"{candidate_label}_require_label_match"] = False
                    candidate_scores[f"{candidate_label}_vlm_match_mode"] = vlm_stats.get("vlm_match_mode", "global_image")
                    candidate_scores[f"{candidate_label}_num_vlm_candidate_pairs"] = vlm_stats.get("vlm_candidate_pairs", 0)
                    candidate_scores[f"{candidate_label}_num_vlm_total_possible_pairs"] = vlm_stats.get("vlm_total_possible_pairs", 0)
                    candidate_scores[f"{candidate_label}_num_vlm_calls"] = vlm_stats.get("vlm_calls", 0)
                    candidate_scores[f"{candidate_label}_num_vlm_raw_matches"] = vlm_stats.get("vlm_raw_match_count", 0)
                    candidate_scores[f"{candidate_label}_num_vlm_raw_same_object_pairs"] = vlm_stats.get("vlm_same_object_pairs_raw", 0)
                    candidate_scores[f"{candidate_label}_num_vlm_same_object_pairs"] = vlm_stats.get("vlm_same_object_pairs", 0)
                    candidate_scores[f"{candidate_label}_num_vlm_orientation_pass_pairs"] = vlm_stats.get("vlm_orientation_pass_pairs", 0)
                    candidate_scores[f"{candidate_label}_num_vlm_geometric_rejected_pairs"] = vlm_stats.get("vlm_geometric_rejected_pairs", 0)
                    candidate_scores[f"{candidate_label}_vlm_geometric_iou_tau"] = vlm_stats.get("vlm_geometric_iou_tau", None)
                    candidate_scores[f"{candidate_label}_vlm_geometric_mask_iou_tau"] = vlm_stats.get("vlm_geometric_mask_iou_tau", None)
                    candidate_scores[f"{candidate_label}_vlm_geometric_center_tau"] = vlm_stats.get("vlm_geometric_center_tau", None)
                    candidate_scores[f"{candidate_label}_vlm_rejected_match_details"] = vlm_stats.get("vlm_rejected_match_details", [])
                    if vlm_stats.get("vlm_error"):
                        candidate_scores[f"{candidate_label}_vlm_error"] = vlm_stats.get("vlm_error")
                        (
                            recall,
                            center_err,
                            verified_boxes,
                            matched_pairs,
                            match_details,
                        ) = self.sem_evaluator.evaluate_permanence(
                            gt_boxes=gt_boxes,
                            gt_features=gt_feats,
                            pred_boxes=pred_boxes,
                            gen_img_cv2=candidate_img_cv2,
                            sem_tools=self.sem_tools,
                            mode=effective_mode,
                            iou_tau=match_iou_tau,
                            feat_tau=match_feat_tau,
                            gt_masks=gt_masks,
                            pred_masks=pred_masks,
                            gt_labels=gt_labels,
                            pred_labels=pred_labels,
                            require_label_match=True,
                            mask_iou_tau=None,
                        )
                        candidate_scores[f"{candidate_label}_match_policy"] = "geometry_label_dino_fallback_after_vlm_error"
                        candidate_scores[f"{candidate_label}_require_label_match"] = True
                        candidate_scores[f"{candidate_label}_vlm_fallback_reason"] = _short_error_text(vlm_stats.get("vlm_error"))
                else:
                    recall, center_err, verified_boxes, matched_pairs, match_details = self.sem_evaluator.evaluate_permanence(
                        gt_boxes=gt_boxes,
                        gt_features=gt_feats,
                        pred_boxes=pred_boxes,
                        gen_img_cv2=candidate_img_cv2,
                        sem_tools=self.sem_tools,
                        mode=effective_mode,
                        iou_tau=match_iou_tau,
                        feat_tau=match_feat_tau,
                        gt_masks=gt_masks,
                        pred_masks=pred_masks,
                        gt_labels=gt_labels,
                        pred_labels=pred_labels,
                        require_label_match=True,
                        mask_iou_tau=None,
                    )
                    candidate_scores[f"{candidate_label}_match_policy"] = "geometry_label_dino"
                candidate_scores[f"{candidate_label}_Recall"] = recall
                candidate_scores[f"{candidate_label}_Center_Error"] = center_err
                candidate_scores[f"{candidate_label}_num_matched_pairs"] = len(matched_pairs)
                candidate_scores[f"{candidate_label}_match_details"] = match_details

                if candidate_label == "OraclePred":
                    self_recall, self_center_err, _, self_pairs, _ = self.sem_evaluator.evaluate_permanence(
                        gt_boxes=pred_boxes,
                        gt_features=[None] * len(pred_boxes),
                        pred_boxes=pred_boxes,
                        gen_img_cv2=candidate_img_cv2,
                        sem_tools=self.sem_tools,
                        mode="none",
                        iou_tau=0.95,
                        feat_tau=1.0,
                        gt_masks=pred_masks,
                        pred_masks=pred_masks,
                        gt_labels=pred_labels,
                        pred_labels=pred_labels,
                        require_label_match=True,
                        mask_iou_tau=None,
                    )
                    candidate_scores["Oracle_TargetSelf_Recall"] = self_recall
                    candidate_scores["Oracle_TargetSelf_Center_Error"] = self_center_err
                    candidate_scores["Oracle_TargetSelf_Num_Proposals"] = len(pred_boxes)
                    candidate_scores["Oracle_TargetSelf_Num_Matched"] = len(self_pairs)

                valid_morph = []
                for gt_idx, pred_idx in matched_pairs:
                    if gt_idx >= len(gt_masks) or pred_idx >= len(pred_masks):
                        continue
                    g_mask = gt_masks[gt_idx]
                    p_mask = pred_masks[pred_idx]
                    morph = self.sem_evaluator.compute_morphological_consistency(g_mask, p_mask)
                    if morph is not None:
                        valid_morph.append(float(morph))
                    verified_masks.append(p_mask)
                candidate_scores[f"{candidate_label}_Morph_Loss"] = (
                    float(np.mean(valid_morph)) if len(valid_morph) > 0 else None
                )

                layout_scores = self.sem_evaluator.compute_matched_layout_metrics(
                    gt_boxes=gt_boxes,
                    pred_boxes=pred_boxes,
                    matched_pairs=matched_pairs,
                    gt_masks=None,
                    pred_masks=None,
                )
                for key, value in layout_scores.items():
                    candidate_scores[f"{candidate_label}_{key}"] = value

            topo_scores = self.sem_evaluator.compute_topology_consistency(
                gt_boxes=gt_boxes,
                pred_boxes=pred_boxes,
                matched_pairs=matched_pairs,
            )
            for key, value in topo_scores.items():
                candidate_scores[f"{candidate_label}_{key}"] = value

            if (
                self.sem_tools.has_da3()
                and os.path.exists(src_img_path)
                and getattr(src_item, "depth", None) is not None
                and getattr(src_item, "extrinsics", None) is not None
            ):
                candidate_tmp_path = None
                try:
                    with tempfile.NamedTemporaryFile(
                        prefix=f"tmp_{candidate_label.lower()}_{file_case_id}_",
                        suffix=".png",
                        delete=False,
                    ) as tmp_f:
                        candidate_tmp_path = tmp_f.name
                    Image.fromarray(_as_rgb_uint8(candidate_img_cv2)).save(candidate_tmp_path)

                    pred_pose_pred, pred_depth_tgt, pred_err = self.sem_tools.calculate_pose_da3(
                        src_img_input=src_img_path,
                        tgt_img_input=candidate_tmp_path,
                        depth_src_gt=src_item.depth,
                        pose_src_gl=src_item.extrinsics,
                    )
                    if pred_pose_pred is not None and pred_depth_tgt is not None:
                        candidate_scores[f"{candidate_label}_DA3"] = self.sem_evaluator.compute_pose_metrics(
                            src_item.extrinsics,
                            pred_pose_pred,
                        )

                        H_depth, W_depth = tgt_item.depth.shape[:2]
                        if pred_depth_tgt.shape[:2] != (H_depth, W_depth):
                            pred_depth_tgt = cv2.resize(
                                pred_depth_tgt,
                                (W_depth, H_depth),
                                interpolation=cv2.INTER_NEAREST,
                            )

                        ktau_stats = self.sem_evaluator.compute_kendalls_tau_matched(
                            depth_gt=tgt_item.depth,
                            depth_pred=pred_depth_tgt,
                            gt_boxes=gt_boxes,
                            pred_boxes=pred_boxes,
                            matched_pairs=matched_pairs,
                        )
                        candidate_scores[f"{candidate_label}_Kendalls_Tau_DA3_Matched"] = ktau_stats["tau"]
                        candidate_scores[f"{candidate_label}_Kendalls_Tau"] = ktau_stats["tau"]
                        candidate_scores[f"{candidate_label}_Kendalls_Tau_Source"] = "da3_matched_boxes"
                        candidate_scores[f"{candidate_label}_Kendalls_Tau_DA3_NumObjects"] = ktau_stats["num_objects"]
                        candidate_scores[f"{candidate_label}_Kendalls_Tau_DA3_NumPairs"] = ktau_stats["num_pairs"]
                        candidate_scores[f"{candidate_label}_Kendalls_Tau_DA3_SkippedPairs"] = ktau_stats["skipped_pairs"]
                        depth_rel = self.sem_evaluator.compute_depth_relation_accuracy_matched(
                            depth_gt=tgt_item.depth,
                            depth_pred=pred_depth_tgt,
                            gt_boxes=gt_boxes,
                            pred_boxes=pred_boxes,
                            matched_pairs=matched_pairs,
                        )
                        for key, value in depth_rel.items():
                            candidate_scores[f"{candidate_label}_{key}"] = value
                    else:
                        candidate_scores[f"{candidate_label}_DA3_error"] = pred_err
                finally:
                    if candidate_tmp_path is not None and os.path.exists(candidate_tmp_path):
                        try:
                            os.remove(candidate_tmp_path)
                        except OSError:
                            pass
            else:
                candidate_scores[f"{candidate_label}_DA3_skip_reason"] = (
                    "DA3 disabled/failed or source image path not found"
                )

            candidate_vis = {
                "pred_boxes": pred_boxes,
                "pred_masks": pred_masks,
                "verified_boxes": verified_boxes,
                "verified_masks": verified_masks,
                "matched_pairs": matched_pairs,
            }
            return candidate_scores, candidate_vis

        gtimage_scores, gtimage_vis = evaluate_candidate_image(tgt_item.image.copy(), "GTImage")
        scores.update(gtimage_scores)

        oraclepred_vis = None
        if self.oracle_mode:
            oraclepred_scores, oraclepred_vis = evaluate_candidate_image(tgt_item.image.copy(), "OraclePred")
            scores.update(oraclepred_scores)

        pred_vis = None
        if gen_img_cv2 is not None:
            pred_scores, pred_vis = evaluate_candidate_image(gen_img_cv2, "Pred")
            scores.update(pred_scores)
            alias_map = {
                "Recall": "Pred_Recall",
                "Center_Error": "Pred_Center_Error",
                "Morph_Loss": "Pred_Morph_Loss",
                "Topology": "Pred_Topology",
                "Topology_NumPairs": "Pred_Topology_NumPairs",
                "Topology_MatchedPairs": "Pred_Topology_MatchedPairs",
                "Topology_PositivePairs": "Pred_Topology_PositivePairs",
                "Topology_MatchedPairCoverage": "Pred_Topology_MatchedPairCoverage",
                "Kendalls_Tau_DA3_Matched": "Pred_Kendalls_Tau_DA3_Matched",
                "Kendalls_Tau": "Pred_Kendalls_Tau",
                "Kendalls_Tau_Source": "Pred_Kendalls_Tau_Source",
                "Kendalls_Tau_DA3_NumObjects": "Pred_Kendalls_Tau_DA3_NumObjects",
                "Kendalls_Tau_DA3_NumPairs": "Pred_Kendalls_Tau_DA3_NumPairs",
                "Kendalls_Tau_DA3_SkippedPairs": "Pred_Kendalls_Tau_DA3_SkippedPairs",
                "match_iou_tau": "Pred_match_iou_tau",
                "match_feat_tau": "Pred_match_feat_tau",
                "match_mask_iou_tau": "Pred_match_mask_iou_tau",
                "DA3_skip_reason": "Pred_DA3_skip_reason",
                "num_gt_boxes": "Pred_num_gt_boxes",
                "num_pred_boxes": "Pred_num_pred_boxes",
                "num_matched_pairs": "Pred_num_matched_pairs",
                "Matched_Object_Area_Ratio_Mean": "Pred_Matched_Object_Area_Ratio_Mean",
                "Matched_Object_Area_Ratio_Score": "Pred_Matched_Object_Area_Ratio_Score",
                "Matched_Location_Relation": "Pred_Matched_Location_Relation",
                "Matched_Location_NumPairs": "Pred_Matched_Location_NumPairs",
                "Matched_Pair_Area_Ratio_Score": "Pred_Matched_Pair_Area_Ratio_Score",
                "Depth_Relation_Accuracy": "Pred_Depth_Relation_Accuracy",
                "Depth_Relation_NumPairs": "Pred_Depth_Relation_NumPairs",
                "Depth_Relation_SkippedPairs": "Pred_Depth_Relation_SkippedPairs",
            }
            for old_key, pred_key in alias_map.items():
                scores[old_key] = scores.get(pred_key)
            if "Pred_skip_reason" in scores:
                scores["skip_reason"] = scores["Pred_skip_reason"]
        else:
            scores["Pred_skip_reason"] = pred_skip_reason or "generated image unavailable"
            scores["skip_reason"] = scores["Pred_skip_reason"]
            scores["DA3_skip_reason"] = scores["Pred_skip_reason"]

        return scores

    def evaluate_single_sample(self, case: dict):
        results = []
        for sample in normalize_benchmark_steps(case):
            res = self.evaluate_step_sample(sample)
            if res is not None:
                results.append(res)
        if len(results) == 1:
            return results[0]
        return results

    @staticmethod
    def _output_path_for_input(jsonl_path: str, output_json_path: str) -> str:
        out_dir = os.path.dirname(output_json_path)
        out_name = os.path.basename(output_json_path)
        out_stem, out_ext = os.path.splitext(out_name)
        if not out_ext:
            out_ext = ".json"

        input_stem = os.path.splitext(os.path.basename(jsonl_path))[0]
        output_name = f"{out_stem}_{input_stem}{out_ext}"
        return os.path.join(out_dir, output_name) if out_dir else output_name

    def _run_samples(self, samples: List[Dict[str, Any]], output_json_path: str, save_every: int = 20):
        output_json_path = str(output_json_path)
        results = []
        failures = []

        def dump_report(final: bool = False):
            metrics_to_avg = [
                "Recall",
                "Center_Error",
                "Morph_Loss",
                "Topology",
                "Topology_MatchedPairCoverage",
                "Kendalls_Tau_DA3_Matched",
                "Kendalls_Tau",
                "GTImage_Recall",
                "GTImage_Center_Error",
                "GTImage_Morph_Loss",
                "GTImage_Topology",
                "GTImage_Topology_MatchedPairCoverage",
                "GTImage_Kendalls_Tau_DA3_Matched",
                "GTImage_Kendalls_Tau",
                "Pred_Recall",
                "Pred_Center_Error",
                "Pred_Morph_Loss",
                "Pred_Topology",
                "Pred_Topology_MatchedPairCoverage",
                "Pred_Kendalls_Tau_DA3_Matched",
                "Pred_Kendalls_Tau",
                "OraclePred_Recall",
                "OraclePred_Center_Error",
                "OraclePred_Morph_Loss",
                "OraclePred_Topology",
                "OraclePred_Topology_MatchedPairCoverage",
                "OraclePred_Kendalls_Tau_DA3_Matched",
                "OraclePred_Kendalls_Tau",
                "Oracle_GTDepth_Kendalls_Tau",
                "Oracle_TargetSelf_Recall",
                "Oracle_TargetSelf_Center_Error",
                "Pred_Matched_Object_Area_Ratio_Score",
                "Pred_Matched_Location_Relation",
                "Pred_Matched_Pair_Area_Ratio_Score",
                "Pred_Depth_Relation_Accuracy",
                "GTImage_Matched_Object_Area_Ratio_Score",
                "GTImage_Matched_Location_Relation",
                "GTImage_Depth_Relation_Accuracy",
            ]
            summary = {}
            for m in metrics_to_avg:
                vals = [
                    r[m] for r in results
                    if m in r and r[m] is not None and not (isinstance(r[m], float) and np.isnan(r[m]))
                ]
                summary[f"Mean_{m}"] = float(np.mean(vals)) if len(vals) > 0 else None
            summary["num_results"] = len(results)
            summary["num_failures"] = len(failures)
            summary["model_status"] = self.sem_tools.model_status
            summary["gt_object_source"] = self.gt_object_source
            summary["visibility_overlap_tau"] = self.visibility_overlap_tau
            summary["enable_vlm_match"] = bool(self.enable_vlm_match)
            summary["vlm_require_orientation"] = bool(self.vlm_require_orientation)
            summary["vlm_match_policy"] = "global_same_object"
            summary["vlm_max_output_tokens"] = int(self.vlm_max_output_tokens)
            summary["vlm_max_candidate_pairs"] = int(self.vlm_max_candidate_pairs)
            summary["vlm_min_match_confidence"] = float(self.vlm_min_match_confidence)
            summary["vlm_local_device_map"] = self.vlm_local_device_map
            summary["vlm_local_torch_dtype"] = self.vlm_local_torch_dtype
            summary["vlm_local_attn_implementation"] = self.vlm_local_attn_implementation
            summary["vlm_local_min_pixels"] = self.vlm_local_min_pixels
            summary["vlm_local_max_pixels"] = self.vlm_local_max_pixels
            summary["vlm_disable_env_proxy"] = bool(self.vlm_disable_env_proxy)
            summary["vlm_json_response_format"] = bool(self.vlm_json_response_format)
            summary["vlm_geometric_iou_tau"] = float(self.vlm_geometric_iou_tau)
            summary["vlm_geometric_gate"] = "box_iou_or_box_center"
            summary["vlm_geometric_mask_iou_tau"] = None
            summary["vlm_geometric_center_tau"] = float(self.vlm_geometric_center_tau)
            summary["filter_matchable_objects"] = bool(self.filter_matchable_objects)
            summary["match_object_min_box_area_ratio"] = float(self.match_object_min_box_area_ratio)
            summary["match_object_min_side_ratio"] = float(self.match_object_min_side_ratio)
            summary["match_object_partial_max_area_ratio"] = float(self.match_object_partial_max_area_ratio)
            summary["match_object_near_border_ratio"] = float(self.match_object_near_border_ratio)
            summary["match_object_near_border_thin_side_ratio"] = float(self.match_object_near_border_thin_side_ratio)
            summary["match_object_near_border_max_area_ratio"] = float(self.match_object_near_border_max_area_ratio)
            summary["match_object_ignore_labels"] = sorted(self.match_object_ignore_labels)
            summary["match_object_max_instances"] = int(self.match_object_max_instances)
            summary["match_object_max_per_label"] = int(self.match_object_max_per_label)
            summary["match_object_repeated_label_max_per_label"] = int(self.match_object_repeated_label_max_per_label)
            summary["match_object_dedup_iou_tau"] = float(self.match_object_dedup_iou_tau)
            summary["match_object_dedup_cover_tau"] = float(self.match_object_dedup_cover_tau)
            summary["match_object_dedup_contain_tau"] = float(self.match_object_dedup_contain_tau)
            summary["match_object_dedup_center_tau"] = float(self.match_object_dedup_center_tau)
            summary["enable_gt_neighborhood_filter"] = bool(self.enable_gt_neighborhood_filter)
            summary["gt_neighborhood_expand_ratio"] = float(self.gt_neighborhood_expand_ratio)
            summary["gt_neighborhood_min_margin_ratio"] = float(self.gt_neighborhood_min_margin_ratio)
            summary["gt_neighborhood_center_tau"] = float(self.gt_neighborhood_center_tau)
            shard_count = int(os.environ.get("BENCHMARK_SHARD_COUNT", "1"))
            shard_index = int(os.environ.get("BENCHMARK_SHARD_INDEX", "0"))
            if shard_count > 1:
                summary["shard_index"] = int(shard_index)
                summary["shard_count"] = int(shard_count)
            payload = make_json_safe({"summary": summary, "details": results, "failures": failures})
            tmp_path = output_json_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=4, ensure_ascii=False)
            os.replace(tmp_path, output_json_path)
            if final:
                print(f"\nEvaluation report saved: {output_json_path}")

        for idx, case in enumerate(tqdm(samples, desc=f"Running multimodal pipeline evaluation [mode: {self.eval_mode}]")):
            try:
                res = self.evaluate_single_sample(case)
                if isinstance(res, list):
                    results.extend([x for x in res if x is not None])
                elif res is not None:
                    results.append(res)
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                print(f"Error on case {case.get('id') or case.get('sample_id') or 'unknown'}: {err}")
                traceback.print_exc(limit=3)
                failures.append({"id": case.get("id") or case.get("sample_id") or "unknown", "error": err})
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

            if save_every > 0 and (idx + 1) % save_every == 0:
                dump_report(final=False)

        dump_report(final=True)

    def run(self, jsonl_path: Any, output_json_path: Any = None, save_every: int = 20, output_dir: Optional[str] = None):
        if isinstance(jsonl_path, (list, tuple)):
            jsonl_paths = list(jsonl_path)
            if output_dir is not None:
                for input_path in jsonl_paths:
                    self.run(str(input_path), save_every=save_every, output_dir=output_dir)
                return

            if isinstance(output_json_path, (list, tuple)):
                output_json_paths = list(output_json_path)
                if len(jsonl_paths) != len(output_json_paths):
                    raise ValueError(
                        f"INPUT_LIST and OUTPUT_REPORT count mismatch: "
                        f"{len(jsonl_paths)} vs {len(output_json_paths)}"
                    )
            else:
                if output_json_path is None:
                    output_json_path = "benchmark_results.json"
                output_json_paths = [
                    self._output_path_for_input(path, str(output_json_path))
                    for path in jsonl_paths
                ]

            for input_path, report_path in zip(jsonl_paths, output_json_paths):
                print(f"\nEvaluation input: {input_path}")
                print(f"  Output report: {report_path}")
                self.run(str(input_path), str(report_path), save_every=save_every)
            return

        if isinstance(output_json_path, (list, tuple)):
            output_json_paths = list(output_json_path)
            if len(output_json_paths) != 1:
                raise ValueError("A single INPUT_LIST can only correspond to one OUTPUT_REPORT")
            output_json_path = output_json_paths[0]

        jsonl_path = str(jsonl_path)
        with open(jsonl_path, "r", encoding="utf-8") as f:
            samples = [json.loads(line) for line in f if line.strip()]

        shard_count = max(1, int(os.environ.get("BENCHMARK_SHARD_COUNT", "1")))
        shard_index = int(os.environ.get("BENCHMARK_SHARD_INDEX", "0"))
        if shard_index < 0 or shard_index >= shard_count:
            raise ValueError(f"BENCHMARK_SHARD_INDEX must be in [0, {shard_count}), got {shard_index}")
        shard_suffix = f".shard{shard_index:03d}-of-{shard_count:03d}" if shard_count > 1 else ""

        prev_vis_dir = getattr(self, "vis_dir", None)
        try:
            if output_dir is not None:
                input_stem = os.path.splitext(os.path.basename(jsonl_path))[0]
                groups = group_samples_by_model(samples, fallback=input_stem)
                model_filter = os.environ.get("BENCHMARK_MODEL_FILTER", "").strip()
                max_samples_per_model = int(os.environ.get("BENCHMARK_MAX_SAMPLES_PER_MODEL", "0"))
                for model_name, model_samples in groups.items():
                    if model_filter and model_filter not in model_name:
                        continue
                    if shard_count > 1:
                        model_samples = [
                            sample for sample_idx, sample in enumerate(model_samples)
                            if sample_idx % shard_count == shard_index
                        ]
                    if max_samples_per_model > 0:
                        model_samples = model_samples[:max_samples_per_model]
                    model_dir = os.path.join(str(output_dir), model_name)
                    os.makedirs(model_dir, exist_ok=True)
                    if self.save_vis:
                        self.vis_dir = os.path.join(model_dir, "vis_obj")
                        os.makedirs(self.vis_dir, exist_ok=True)

                    report_name = f"obj_result_{input_stem}{shard_suffix}.json"
                    report_path = os.path.join(model_dir, report_name)
                    print(f"\nEvaluating model: {model_name}")
                    print(f"  Input samples: {jsonl_path}")
                    print(f"  Sample count: {len(model_samples)}")
                    if shard_count > 1:
                        print(f"  Shard: {shard_index}/{shard_count}")
                    print(f"  Output report: {report_path}")
                    if self.save_vis:
                        print(f"  Visualization dir: {self.vis_dir}")
                    self._run_samples(model_samples, report_path, save_every=save_every)
                return

            if output_json_path is None:
                output_json_path = "benchmark_results.json"
            if shard_suffix:
                out_dir = os.path.dirname(str(output_json_path))
                out_name = os.path.basename(str(output_json_path))
                stem, ext = os.path.splitext(out_name)
                output_json_path = os.path.join(out_dir, f"{stem}{shard_suffix}{ext or '.json'}")
                samples = [
                    sample for sample_idx, sample in enumerate(samples)
                    if sample_idx % shard_count == shard_index
                ]
            self._run_samples(samples, str(output_json_path), save_every=save_every)
        finally:
            if self.save_vis:
                self.vis_dir = prev_vis_dir



if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    VERIFICATION_MODE = "dinov3"
    # text: SAM3 text prompt; auto/cv_auto: class-agnostic auto proposal;
    # gdino_box: GroundingDINO detection boxes; gdino_sam3: GroundingDINO boxes + SAM3 box prompt mask.
    # Default uses box evaluation to avoid incomplete/irregular SAM3 masks affecting object match and depth relations.
    PROPOSAL_MODE = os.environ.get("BENCHMARK_PROPOSAL_MODE", "gdino_box")
    # projected_context: legacy context mask projection as GT;
    # target_gt_seg: directly segment target GT image as GT;
    # target_visible_seg: segment target GT, keep only context-visible support regions.
    GT_OBJECT_SOURCE = "target_visible_seg"
    VISIBILITY_OVERLAP_TAU = 0.25
    ORACLE_MODE = True
    SAVE_VIS = True

    # VLM object matching: detector produces object boxes; local Qwen outputs global object correspondences.
    ENABLE_VLM_MATCH = os.environ.get("BENCHMARK_ENABLE_VLM_MATCH", "1").lower() not in {"0", "false", "no"}
    VLM_BACKEND = os.environ.get("VLM_BACKEND", DEFAULT_VLM_BACKEND)
    VLM_MODEL = os.environ.get("VLM_MODEL", DEFAULT_VLM_MODEL)
    VLM_BASE_URL = os.environ.get("VLM_BASE_URL", DEFAULT_VLM_BASE_URL)
    VLM_API_KEY_ENV = os.environ.get("VLM_API_KEY_ENV", "")
    if os.environ.get("VLM_API_KEY"):
        VLM_API_KEY_ENV = VLM_API_KEY_ENV or "SPATIAL_VLM_API_KEY"
        os.environ[VLM_API_KEY_ENV] = os.environ["VLM_API_KEY"]
    elif str(VLM_BACKEND).lower() in OPENAI_COMPATIBLE_VLM_BACKENDS and not VLM_API_KEY_ENV:
        VLM_API_KEY_ENV = "OPENAI_API_KEY"
    os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1,::1")
    os.environ.setdefault("no_proxy", os.environ["NO_PROXY"])
    VLM_REQUIRE_ORIENTATION = os.environ.get("VLM_REQUIRE_ORIENTATION", "1").lower() not in {"0", "false", "no"}
    VLM_MAX_OUTPUT_TOKENS = int(os.environ.get("VLM_MAX_OUTPUT_TOKENS", "2048"))
    VLM_MIN_MATCH_CONFIDENCE = float(os.environ.get("VLM_MIN_MATCH_CONFIDENCE", "0.8"))
    VLM_LOCAL_DEVICE_MAP = os.environ.get("VLM_LOCAL_DEVICE_MAP", "auto")
    VLM_LOCAL_TORCH_DTYPE = os.environ.get("VLM_LOCAL_TORCH_DTYPE", "bfloat16")
    VLM_LOCAL_ATTN_IMPLEMENTATION = os.environ.get("VLM_LOCAL_ATTN_IMPLEMENTATION", "auto")
    if VLM_LOCAL_ATTN_IMPLEMENTATION.lower() == "auto":
        VLM_LOCAL_ATTN_IMPLEMENTATION = None
    VLM_LOCAL_MIN_PIXELS = os.environ.get("VLM_LOCAL_MIN_PIXELS", "").strip()
    VLM_LOCAL_MAX_PIXELS = os.environ.get("VLM_LOCAL_MAX_PIXELS", "").strip()
    VLM_LOCAL_MIN_PIXELS = int(VLM_LOCAL_MIN_PIXELS) if VLM_LOCAL_MIN_PIXELS else None
    VLM_LOCAL_MAX_PIXELS = int(VLM_LOCAL_MAX_PIXELS) if VLM_LOCAL_MAX_PIXELS else None
    VLM_DISABLE_ENV_PROXY = os.environ.get("VLM_DISABLE_ENV_PROXY", "0").lower() in {"1", "true", "yes"}
    VLM_JSON_RESPONSE_FORMAT = os.environ.get("VLM_JSON_RESPONSE_FORMAT", "1").lower() not in {"0", "false", "no"}
    VLM_MAX_CANDIDATE_PAIRS = int(os.environ.get("VLM_MAX_CANDIDATE_PAIRS", "120"))
    VLM_GEOMETRIC_IOU_TAU = float(os.environ.get("VLM_GEOMETRIC_IOU_TAU", "0.15"))
    VLM_GEOMETRIC_MASK_IOU_TAU = None
    VLM_GEOMETRIC_CENTER_TAU = float(os.environ.get("VLM_GEOMETRIC_CENTER_TAU", "0.08"))
    ENABLE_SAM3 = os.environ.get("BENCHMARK_ENABLE_SAM3", "0").lower() not in {"0", "false", "no"}
    FILTER_MATCHABLE_OBJECTS = os.environ.get("BENCHMARK_FILTER_MATCHABLE_OBJECTS", "1").lower() not in {"0", "false", "no"}
    MATCH_OBJECT_MIN_BOX_AREA_RATIO = float(os.environ.get("BENCHMARK_MATCH_OBJECT_MIN_BOX_AREA_RATIO", "0.006"))
    MATCH_OBJECT_MIN_SIDE_RATIO = float(os.environ.get("BENCHMARK_MATCH_OBJECT_MIN_SIDE_RATIO", "0.035"))
    MATCH_OBJECT_PARTIAL_MAX_AREA_RATIO = float(os.environ.get("BENCHMARK_MATCH_OBJECT_PARTIAL_MAX_AREA_RATIO", "0.03"))
    MATCH_OBJECT_NEAR_BORDER_RATIO = float(os.environ.get("BENCHMARK_MATCH_OBJECT_NEAR_BORDER_RATIO", "0.04"))
    MATCH_OBJECT_NEAR_BORDER_THIN_SIDE_RATIO = float(os.environ.get("BENCHMARK_MATCH_OBJECT_NEAR_BORDER_THIN_SIDE_RATIO", "0.09"))
    MATCH_OBJECT_NEAR_BORDER_MAX_AREA_RATIO = float(os.environ.get("BENCHMARK_MATCH_OBJECT_NEAR_BORDER_MAX_AREA_RATIO", "0.05"))
    MATCH_OBJECT_MAX_INSTANCES = int(os.environ.get("BENCHMARK_MATCH_OBJECT_MAX_INSTANCES", "12"))
    MATCH_OBJECT_MAX_PER_LABEL = int(os.environ.get("BENCHMARK_MATCH_OBJECT_MAX_PER_LABEL", "4"))
    MATCH_OBJECT_REPEATED_LABEL_MAX_PER_LABEL = int(os.environ.get("BENCHMARK_MATCH_OBJECT_REPEATED_LABEL_MAX_PER_LABEL", "3"))
    MATCH_OBJECT_DEDUP_IOU_TAU = float(os.environ.get("BENCHMARK_MATCH_OBJECT_DEDUP_IOU_TAU", "0.55"))
    MATCH_OBJECT_DEDUP_COVER_TAU = float(os.environ.get("BENCHMARK_MATCH_OBJECT_DEDUP_COVER_TAU", "0.72"))
    MATCH_OBJECT_DEDUP_CONTAIN_TAU = float(os.environ.get("BENCHMARK_MATCH_OBJECT_DEDUP_CONTAIN_TAU", "0.85"))
    MATCH_OBJECT_DEDUP_CENTER_TAU = float(os.environ.get("BENCHMARK_MATCH_OBJECT_DEDUP_CENTER_TAU", "0.12"))
    ENABLE_GT_NEIGHBORHOOD_FILTER = os.environ.get("BENCHMARK_ENABLE_GT_NEIGHBORHOOD_FILTER", "1").lower() not in {"0", "false", "no"}
    GT_NEIGHBORHOOD_EXPAND_RATIO = float(os.environ.get("BENCHMARK_GT_NEIGHBORHOOD_EXPAND_RATIO", "0.75"))
    GT_NEIGHBORHOOD_MIN_MARGIN_RATIO = float(os.environ.get("BENCHMARK_GT_NEIGHBORHOOD_MIN_MARGIN_RATIO", "0.06"))
    GT_NEIGHBORHOOD_CENTER_TAU = float(os.environ.get("BENCHMARK_GT_NEIGHBORHOOD_CENTER_TAU", "0.18"))
    MATCH_OBJECT_IGNORE_LABELS = [
        x.strip()
        for x in os.environ.get(
            "BENCHMARK_MATCH_OBJECT_IGNORE_LABELS",
            "book,clothes,pillow,blanket,cup,bottle,bag,backpack,trash,trash can,box",
        ).split(",")
        if x.strip()
    ]

    DA3_SRC_PATH = os.environ.get("DA3_CODE_DIR", "third_party/Depth-Anything-3/src")
    DA3_MODEL_PATH = os.environ.get("DA3_MODEL_DIR", "checkpoints/DA3NESTED-GIANT-LARGE")
    GROUNDING_DINO_PATH = os.environ.get("GROUNDING_DINO_MODEL", "IDEA-Research/grounding-dino-base")

    # Mixed vocabulary: general fallback + forced background geometry extraction
    # Use specific object classes for SAM3 text grounding first; avoid generic/stuff prompts like object/wall/floor/ceiling initially.
    TEXT_PROMPTS = [
        "chair", "table", "sofa", "bed", "cabinet",
        "desk", "door", "window", "bookshelf",
        "counter", "toilet", "sink", "bathtub",
        "picture", "monitor", "trash can",
        "book", "bag", "backpack", "guitar", "lamp",
        "pillow", "blanket", "clothes", "cup", "bottle",
        "plant", "tv", "keyboard", "laptop", "stool",
        "bench", "shelf", "curtain", "rug", "box",
    ]

    runner = SspDetectionRunner(
        eval_mode=VERIFICATION_MODE,
        oracle_mode=ORACLE_MODE,
        save_vis=SAVE_VIS,
        device=None,
        da3_src_path=DA3_SRC_PATH,
        da3_model_path=DA3_MODEL_PATH,
        grounding_dino_path=GROUNDING_DINO_PATH,
        text_prompts=TEXT_PROMPTS,
        proposal_mode=PROPOSAL_MODE,
        gt_object_source=GT_OBJECT_SOURCE,
        visibility_overlap_tau=VISIBILITY_OVERLAP_TAU,
        enable_sam3=ENABLE_SAM3,
        enable_vlm_match=ENABLE_VLM_MATCH,
        vlm_backend=VLM_BACKEND,
        vlm_model=VLM_MODEL,
        vlm_base_url=VLM_BASE_URL,
        vlm_api_key_env=VLM_API_KEY_ENV,
        vlm_max_output_tokens=VLM_MAX_OUTPUT_TOKENS,
        vlm_require_orientation=VLM_REQUIRE_ORIENTATION,
        vlm_min_match_confidence=VLM_MIN_MATCH_CONFIDENCE,
        vlm_local_device_map=VLM_LOCAL_DEVICE_MAP,
        vlm_local_torch_dtype=VLM_LOCAL_TORCH_DTYPE,
        vlm_local_attn_implementation=VLM_LOCAL_ATTN_IMPLEMENTATION,
        vlm_local_min_pixels=VLM_LOCAL_MIN_PIXELS,
        vlm_local_max_pixels=VLM_LOCAL_MAX_PIXELS,
        vlm_disable_env_proxy=VLM_DISABLE_ENV_PROXY,
        vlm_json_response_format=VLM_JSON_RESPONSE_FORMAT,
        vlm_max_candidate_pairs=VLM_MAX_CANDIDATE_PAIRS,
        vlm_geometric_iou_tau=VLM_GEOMETRIC_IOU_TAU,
        vlm_geometric_mask_iou_tau=0.0,
        vlm_geometric_center_tau=VLM_GEOMETRIC_CENTER_TAU,
        filter_matchable_objects=FILTER_MATCHABLE_OBJECTS,
        match_object_min_box_area_ratio=MATCH_OBJECT_MIN_BOX_AREA_RATIO,
        match_object_min_side_ratio=MATCH_OBJECT_MIN_SIDE_RATIO,
        match_object_partial_max_area_ratio=MATCH_OBJECT_PARTIAL_MAX_AREA_RATIO,
        match_object_near_border_ratio=MATCH_OBJECT_NEAR_BORDER_RATIO,
        match_object_near_border_thin_side_ratio=MATCH_OBJECT_NEAR_BORDER_THIN_SIDE_RATIO,
        match_object_near_border_max_area_ratio=MATCH_OBJECT_NEAR_BORDER_MAX_AREA_RATIO,
        match_object_ignore_labels=MATCH_OBJECT_IGNORE_LABELS,
        match_object_max_instances=MATCH_OBJECT_MAX_INSTANCES,
        match_object_max_per_label=MATCH_OBJECT_MAX_PER_LABEL,
        match_object_repeated_label_max_per_label=MATCH_OBJECT_REPEATED_LABEL_MAX_PER_LABEL,
        match_object_dedup_iou_tau=MATCH_OBJECT_DEDUP_IOU_TAU,
        match_object_dedup_cover_tau=MATCH_OBJECT_DEDUP_COVER_TAU,
        match_object_dedup_contain_tau=MATCH_OBJECT_DEDUP_CONTAIN_TAU,
        match_object_dedup_center_tau=MATCH_OBJECT_DEDUP_CENTER_TAU,
        enable_gt_neighborhood_filter=ENABLE_GT_NEIGHBORHOOD_FILTER,
        gt_neighborhood_expand_ratio=GT_NEIGHBORHOOD_EXPAND_RATIO,
        gt_neighborhood_min_margin_ratio=GT_NEIGHBORHOOD_MIN_MARGIN_RATIO,
        gt_neighborhood_center_tau=GT_NEIGHBORHOOD_CENTER_TAU,
    )
    output_dir = os.environ.get("BENCHMARK_OUTPUT_DIR", OUTPUT_DIR)
    runner.run(INPUT_JSONL, output_dir=output_dir)
