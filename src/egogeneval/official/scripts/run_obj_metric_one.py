#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_ROOT = REPO_ROOT / "evaluation"
if str(EVALUATION_ROOT) not in sys.path:
    sys.path.insert(0, str(EVALUATION_ROOT))

from ssp_detection import (
    DEFAULT_VLM_BACKEND,
    DEFAULT_VLM_BASE_URL,
    DEFAULT_VLM_MODEL,
    OPENAI_COMPATIBLE_VLM_BACKENDS,
    SspDetectionRunner,
)


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


def env_bool(name: str, default: str) -> bool:
    return os.environ.get(name, default).lower() not in {"0", "false", "no"}


def env_optional_int(name: str):
    value = os.environ.get(name, "").strip()
    return int(value) if value else None


def build_runner(save_vis: bool) -> SspDetectionRunner:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    vlm_backend = os.environ.get("VLM_BACKEND", DEFAULT_VLM_BACKEND)
    vlm_model = os.environ.get("VLM_MODEL", DEFAULT_VLM_MODEL)
    vlm_base_url = os.environ.get("VLM_BASE_URL", DEFAULT_VLM_BASE_URL)
    vlm_api_key_env = os.environ.get("VLM_API_KEY_ENV", "")
    if os.environ.get("VLM_API_KEY"):
        vlm_api_key_env = vlm_api_key_env or "SPATIAL_VLM_API_KEY"
        os.environ[vlm_api_key_env] = os.environ["VLM_API_KEY"]
    elif str(vlm_backend).lower() in OPENAI_COMPATIBLE_VLM_BACKENDS and not vlm_api_key_env:
        vlm_api_key_env = "OPENAI_API_KEY"

    os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1,::1")
    os.environ.setdefault("no_proxy", os.environ["NO_PROXY"])

    return SspDetectionRunner(
        eval_mode="dinov3",
        oracle_mode=True,
        save_vis=save_vis,
        device=None,
        da3_src_path=os.environ.get("DA3_CODE_DIR", "third_party/Depth-Anything-3/src"),
        da3_model_path=os.environ.get("DA3_MODEL_DIR", "checkpoints/DA3NESTED-GIANT-LARGE"),
        grounding_dino_path=os.environ.get("GROUNDING_DINO_MODEL", "IDEA-Research/grounding-dino-base"),
        text_prompts=TEXT_PROMPTS,
        gdino_box_threshold=float(os.environ.get("GDINO_BOX_THRESHOLD", "0.25")),
        gdino_text_threshold=float(os.environ.get("GDINO_TEXT_THRESHOLD", "0.25")),
        proposal_mode=os.environ.get("BENCHMARK_PROPOSAL_MODE", "gdino_box"),
        gt_object_source="target_visible_seg",
        visibility_overlap_tau=0.25,
        enable_sam3=env_bool("BENCHMARK_ENABLE_SAM3", "0"),
        enable_vlm_match=env_bool("BENCHMARK_ENABLE_VLM_MATCH", "1"),
        vlm_backend=vlm_backend,
        vlm_model=vlm_model,
        vlm_base_url=vlm_base_url,
        vlm_api_key_env=vlm_api_key_env,
        vlm_max_output_tokens=int(os.environ.get("VLM_MAX_OUTPUT_TOKENS", "2048")),
        vlm_require_orientation=env_bool("VLM_REQUIRE_ORIENTATION", "1"),
        vlm_min_match_confidence=float(os.environ.get("VLM_MIN_MATCH_CONFIDENCE", "0.8")),
        vlm_local_device_map=os.environ.get("VLM_LOCAL_DEVICE_MAP", "auto"),
        vlm_local_torch_dtype=os.environ.get("VLM_LOCAL_TORCH_DTYPE", "bfloat16"),
        vlm_local_attn_implementation=(
            None
            if os.environ.get("VLM_LOCAL_ATTN_IMPLEMENTATION", "auto").lower() == "auto"
            else os.environ.get("VLM_LOCAL_ATTN_IMPLEMENTATION")
        ),
        vlm_local_min_pixels=env_optional_int("VLM_LOCAL_MIN_PIXELS"),
        vlm_local_max_pixels=env_optional_int("VLM_LOCAL_MAX_PIXELS"),
        vlm_disable_env_proxy=os.environ.get("VLM_DISABLE_ENV_PROXY", "0").lower() in {"1", "true", "yes"},
        vlm_json_response_format=env_bool("VLM_JSON_RESPONSE_FORMAT", "1"),
        vlm_max_candidate_pairs=int(os.environ.get("VLM_MAX_CANDIDATE_PAIRS", "120")),
        vlm_geometric_iou_tau=float(os.environ.get("VLM_GEOMETRIC_IOU_TAU", "0.15")),
        vlm_geometric_mask_iou_tau=0.0,
        vlm_geometric_center_tau=float(os.environ.get("VLM_GEOMETRIC_CENTER_TAU", "0.08")),
        filter_matchable_objects=env_bool("BENCHMARK_FILTER_MATCHABLE_OBJECTS", "1"),
        match_object_min_box_area_ratio=float(os.environ.get("BENCHMARK_MATCH_OBJECT_MIN_BOX_AREA_RATIO", "0.006")),
        match_object_min_side_ratio=float(os.environ.get("BENCHMARK_MATCH_OBJECT_MIN_SIDE_RATIO", "0.035")),
        match_object_partial_max_area_ratio=float(os.environ.get("BENCHMARK_MATCH_OBJECT_PARTIAL_MAX_AREA_RATIO", "0.03")),
        match_object_near_border_ratio=float(os.environ.get("BENCHMARK_MATCH_OBJECT_NEAR_BORDER_RATIO", "0.04")),
        match_object_near_border_thin_side_ratio=float(os.environ.get("BENCHMARK_MATCH_OBJECT_NEAR_BORDER_THIN_SIDE_RATIO", "0.09")),
        match_object_near_border_max_area_ratio=float(os.environ.get("BENCHMARK_MATCH_OBJECT_NEAR_BORDER_MAX_AREA_RATIO", "0.05")),
        match_object_ignore_labels=[
            x.strip()
            for x in os.environ.get(
                "BENCHMARK_MATCH_OBJECT_IGNORE_LABELS",
                "book,clothes,pillow,blanket,cup,bottle,bag,backpack,trash,trash can,box",
            ).split(",")
            if x.strip()
        ],
        match_object_max_instances=int(os.environ.get("BENCHMARK_MATCH_OBJECT_MAX_INSTANCES", "12")),
        match_object_max_per_label=int(os.environ.get("BENCHMARK_MATCH_OBJECT_MAX_PER_LABEL", "4")),
        match_object_repeated_label_max_per_label=int(os.environ.get("BENCHMARK_MATCH_OBJECT_REPEATED_LABEL_MAX_PER_LABEL", "3")),
        match_object_dedup_iou_tau=float(os.environ.get("BENCHMARK_MATCH_OBJECT_DEDUP_IOU_TAU", "0.55")),
        match_object_dedup_cover_tau=float(os.environ.get("BENCHMARK_MATCH_OBJECT_DEDUP_COVER_TAU", "0.72")),
        match_object_dedup_contain_tau=float(os.environ.get("BENCHMARK_MATCH_OBJECT_DEDUP_CONTAIN_TAU", "0.85")),
        match_object_dedup_center_tau=float(os.environ.get("BENCHMARK_MATCH_OBJECT_DEDUP_CENTER_TAU", "0.12")),
        enable_gt_neighborhood_filter=env_bool("BENCHMARK_ENABLE_GT_NEIGHBORHOOD_FILTER", "1"),
        gt_neighborhood_expand_ratio=float(os.environ.get("BENCHMARK_GT_NEIGHBORHOOD_EXPAND_RATIO", "0.75")),
        gt_neighborhood_min_margin_ratio=float(os.environ.get("BENCHMARK_GT_NEIGHBORHOOD_MIN_MARGIN_RATIO", "0.06")),
        gt_neighborhood_center_tau=float(os.environ.get("BENCHMARK_GT_NEIGHBORHOOD_CENTER_TAU", "0.18")),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--vis-dir", default=None)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--no-save-vis", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    runner = build_runner(save_vis=not args.no_save_vis)
    if runner.save_vis:
        runner.vis_dir = str(Path(args.vis_dir) if args.vis_dir else output_path.parent / "vis_obj")
        Path(runner.vis_dir).mkdir(parents=True, exist_ok=True)
    runner.run(args.input_jsonl, output_json_path=str(output_path), save_every=args.save_every)


if __name__ == "__main__":
    main()
