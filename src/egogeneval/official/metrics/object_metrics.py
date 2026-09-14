#!/usr/bin/env python3
"""Final object metric used by the paper release.

The detector, vocabulary, correspondence matcher, object counts, matched-object
appearance signals, and normalized center error are frozen upstream.  This
module is the single source of truth for the subsequent per-pair, per-step,
per-case, subtype, and benchmark aggregation.

The spatial pillar combines two relative relations on every GT-object pair:

* planar topology: agreement of the 2-D direction between box centers;
* depth topology: agreement of front/behind order when the GT pair has a
  sufficiently separated metric depth.

No planar distance-ratio or depth-gap-magnitude term is used. The signed
relative gap and its tie decision are invariant to a common positive scale.
Front/behind sign is also invariant to a common additive offset, although the
0.03 relative tie threshold is not. An unmatched endpoint scores zero for
every incident relation.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from itertools import combinations
from typing import Iterable, Mapping, Sequence

import numpy as np


SUBTYPES = ("atomic", "chain", "cycle")
GT_DEPTH_RELATIVE_SEPARATION = 0.03
PRED_DEPTH_RELATIVE_SEPARATION = 0.03
EPSILON = 1e-8


def clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def subtype(row: Mapping) -> str:
    return str(row["id"]).split("__")[2]


def case_id(row_or_id: Mapping | str) -> str:
    sample_id = row_or_id["id"] if isinstance(row_or_id, Mapping) else row_or_id
    return str(sample_id).split("__step")[0]


def step_index(row: Mapping) -> int:
    match = re.search(r"__step(\d+)$", str(row["id"]))
    return int(match.group(1)) if match else 0


def group_cases(rows: Iterable[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[case_id(row)].append(row)
    for steps in grouped.values():
        steps.sort(key=step_index)
    return dict(grouped)


def invalid_gt_case_ids(rows: Iterable[dict]) -> set[str]:
    """Cases with at least one required step that has no evaluable GT object."""
    return {
        parent
        for parent, steps in group_cases(rows).items()
        if any(int(row["num_gt_objects"]) <= 0 for row in steps)
    }


def filter_gt_valid_cases(
    rows: Iterable[dict], *, invalid_cases: set[str] | None = None
) -> tuple[list[dict], set[str]]:
    materialized = list(rows)
    derived = invalid_gt_case_ids(materialized)
    if invalid_cases is not None and derived != set(invalid_cases):
        raise ValueError("derived GT-invalid case mask differs from required mask")
    invalid = derived if invalid_cases is None else set(invalid_cases)
    kept = [row for row in materialized if case_id(row) not in invalid]
    if any(int(row["num_gt_objects"]) <= 0 for row in kept):
        raise ValueError("GT-valid mask retained a G=0 step")
    return kept, invalid


def _finite(row: Mapping, key: str) -> float:
    value = row.get(key)
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{row.get('id')}: required finite {key}={value!r}")
    return float(value)


def _optional_finite(value, *, name: str, sample_id: str) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{sample_id}: invalid {name}={value!r}")
    return float(value)


def _center(box: Sequence[float]) -> np.ndarray:
    x1, y1, x2, y2 = [float(value) for value in box[:4]]
    return np.asarray([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float64)


def relative_signed_depth(first: float, second: float) -> float:
    """Signed depth separation normalized by the pair mean absolute depth."""
    denominator = max(0.5 * (abs(float(first)) + abs(float(second))), EPSILON)
    return float((float(first) - float(second)) / denominator)


def _frozen_object_arrays(row: Mapping) -> dict:
    sample_id = str(row.get("id"))
    objects = row.get("objects")
    if not isinstance(objects, list):
        raise ValueError(f"{sample_id}: missing frozen objects[]")
    gt_count = int(row["num_gt_objects"])
    pred_count = int(row["num_pred_all"])
    matched_count = int(row["num_matched"])
    gt_boxes: list[list[float] | None] = [None] * gt_count
    pred_boxes: list[list[float] | None] = [None] * pred_count
    match_map: dict[int, int] = {}
    for obj in objects:
        gt_index = obj.get("gt_idx")
        pred_index = obj.get("pred_idx")
        if gt_index is not None:
            gt_index = int(gt_index)
            if not 0 <= gt_index < gt_count:
                raise ValueError(f"{sample_id}: invalid gt_idx={gt_index}")
            if obj.get("gt_box") is not None:
                gt_boxes[gt_index] = [float(value) for value in obj["gt_box"][:4]]
        if pred_index is not None:
            pred_index = int(pred_index)
            if not 0 <= pred_index < pred_count:
                raise ValueError(f"{sample_id}: invalid pred_idx={pred_index}")
            if obj.get("pred_box") is not None:
                pred_boxes[pred_index] = [float(value) for value in obj["pred_box"][:4]]
        if obj.get("status") == "match":
            if gt_index is None or pred_index is None:
                raise ValueError(f"{sample_id}: matched record lacks indices")
            if gt_index in match_map or pred_index in match_map.values():
                raise ValueError(f"{sample_id}: correspondence is not one-to-one")
            match_map[gt_index] = pred_index
    if any(box is None for box in gt_boxes):
        raise ValueError(f"{sample_id}: at least one GT box is missing")
    if any(pred_boxes[pred] is None for pred in match_map.values()):
        raise ValueError(f"{sample_id}: at least one matched prediction box is missing")
    if len(match_map) != matched_count:
        raise ValueError(
            f"{sample_id}: matched records={len(match_map)} but M={matched_count}"
        )
    return {
        "gt_boxes": gt_boxes,
        "pred_boxes": pred_boxes,
        "match_map": match_map,
    }


def _object_depth_arrays(row: Mapping, match_map: Mapping[int, int]) -> tuple[list, list]:
    sample_id = str(row.get("id"))
    gt_count = int(row["num_gt_objects"])
    pred_count = int(row["num_pred_all"])
    records = row.get("ObjectDepths")
    if records is None:
        records = row.get("DepthAwareTopology_ObjectDepths")
    if not isinstance(records, list):
        raise ValueError(f"{sample_id}: missing ObjectDepths")
    gt_depths: list[float | None] = [None] * gt_count
    pred_depths: list[float | None] = [None] * pred_count
    seen_gt: set[int] = set()
    for record in records:
        gt_index = int(record["gt_idx"])
        if not 0 <= gt_index < gt_count or gt_index in seen_gt:
            raise ValueError(f"{sample_id}: invalid/duplicate depth gt_idx={gt_index}")
        seen_gt.add(gt_index)
        gt_depths[gt_index] = _optional_finite(
            record.get("gt_median_depth"), name="gt_median_depth", sample_id=sample_id
        )
        pred_index = record.get("pred_idx")
        if pred_index is not None:
            pred_index = int(pred_index)
            if not 0 <= pred_index < pred_count:
                raise ValueError(f"{sample_id}: invalid depth pred_idx={pred_index}")
            expected = match_map.get(gt_index)
            if expected != pred_index:
                raise ValueError(
                    f"{sample_id}: depth pair ({gt_index},{pred_index}) != frozen match {expected}"
                )
            pred_depths[pred_index] = _optional_finite(
                record.get("pred_median_depth"),
                name="pred_median_depth",
                sample_id=sample_id,
            )
    if seen_gt != set(range(gt_count)):
        raise ValueError(f"{sample_id}: object-depth records do not cover every GT index")
    return gt_depths, pred_depths


def compute_relative_topology(
    row: Mapping,
    *,
    gt_relative_separation: float = GT_DEPTH_RELATIVE_SEPARATION,
    pred_relative_separation: float = PRED_DEPTH_RELATIVE_SEPARATION,
) -> dict:
    """Compute planar, depth, and pairwise-combined topology for one step.

    Planar direction is ``max(0, cosine(v_gt, v_pred))``.  A depth pair is
    eligible only when both GT depths are valid and their normalized signed
    separation exceeds ``gt_relative_separation``.  It scores one iff both
    endpoints are matched, both predicted depths are valid, the predicted pair
    is not tied, and the front/behind sign agrees.  For each GT pair, the final
    relative-relation score is the mean of planar and depth scores when depth is
    eligible, and the planar score otherwise.
    """
    sample_id = str(row.get("id"))
    arrays = _frozen_object_arrays(row)
    gt_boxes = arrays["gt_boxes"]
    pred_boxes = arrays["pred_boxes"]
    match_map = arrays["match_map"]
    gt_depths, pred_depths = _object_depth_arrays(row, match_map)
    gt_count = len(gt_boxes)
    if gt_count < 2:
        return {
            "PlanarTopology_AllGT": None,
            "PlanarTopology_MatchedPairs": None,
            "DepthTopology_AllGT": None,
            "DepthTopology_MatchedValid": None,
            "RelativeTopology_AllGT": None,
            "GT_Pair_Coverage": None,
            "DepthTopology_EligibilityRate": None,
            "DepthTopology_MatchCoverage": None,
            "DepthTopology_ValidCoverage": None,
            "DepthTopology_NumEligibleGTPairs": 0,
            "DepthTopology_NumMatchedEligiblePairs": 0,
            "DepthTopology_NumValidMatchedPairs": 0,
            "DepthTopology_NumCorrectPairs": 0,
            "DepthTopology_NumPredictedTies": 0,
        }

    planar_scores: list[float] = []
    matched_planar_scores: list[float] = []
    relative_scores: list[float] = []
    eligible = matched_eligible = valid_matched = correct = predicted_ties = 0
    total_pairs = gt_count * (gt_count - 1) // 2
    matched_pair_count = len(match_map) * (len(match_map) - 1) // 2

    for first, second in combinations(range(gt_count), 2):
        endpoints_matched = first in match_map and second in match_map
        planar = 0.0
        if endpoints_matched:
            gt_vector = _center(gt_boxes[first]) - _center(gt_boxes[second])
            pred_first = pred_boxes[match_map[first]]
            pred_second = pred_boxes[match_map[second]]
            pred_vector = _center(pred_first) - _center(pred_second)
            gt_norm = float(np.linalg.norm(gt_vector))
            pred_norm = float(np.linalg.norm(pred_vector))
            if gt_norm < EPSILON:
                raise ValueError(f"{sample_id}: GT pair ({first},{second}) has no planar direction")
            if pred_norm >= EPSILON:
                cosine = float(np.dot(gt_vector, pred_vector) / (gt_norm * pred_norm))
                planar = clip01(max(0.0, cosine))
            matched_planar_scores.append(planar)
        planar_scores.append(planar)

        gt_first, gt_second = gt_depths[first], gt_depths[second]
        depth_eligible = False
        depth_score = 0.0
        if gt_first is not None and gt_second is not None:
            gt_gap = relative_signed_depth(gt_first, gt_second)
            depth_eligible = abs(gt_gap) > gt_relative_separation
        if depth_eligible:
            eligible += 1
            if endpoints_matched:
                matched_eligible += 1
                pred_first_depth = pred_depths[match_map[first]]
                pred_second_depth = pred_depths[match_map[second]]
                if pred_first_depth is not None and pred_second_depth is not None:
                    valid_matched += 1
                    pred_gap = relative_signed_depth(pred_first_depth, pred_second_depth)
                    if abs(pred_gap) <= pred_relative_separation:
                        predicted_ties += 1
                    elif gt_gap * pred_gap > 0.0:
                        correct += 1
                        depth_score = 1.0
            relative_scores.append((planar + depth_score) / 2.0)
        else:
            relative_scores.append(planar)

    return {
        "PlanarTopology_AllGT": float(np.mean(planar_scores)),
        "PlanarTopology_MatchedPairs": (
            float(np.mean(matched_planar_scores)) if matched_planar_scores else None
        ),
        "DepthTopology_AllGT": float(correct / eligible) if eligible else None,
        "DepthTopology_MatchedValid": (
            float(correct / valid_matched) if valid_matched else None
        ),
        "RelativeTopology_AllGT": float(np.mean(relative_scores)),
        "GT_Pair_Coverage": float(matched_pair_count / total_pairs),
        "DepthTopology_EligibilityRate": float(eligible / total_pairs),
        "DepthTopology_MatchCoverage": (
            float(matched_eligible / eligible) if eligible else None
        ),
        "DepthTopology_ValidCoverage": (
            float(valid_matched / eligible) if eligible else None
        ),
        "DepthTopology_NumEligibleGTPairs": int(eligible),
        "DepthTopology_NumMatchedEligiblePairs": int(matched_eligible),
        "DepthTopology_NumValidMatchedPairs": int(valid_matched),
        "DepthTopology_NumCorrectPairs": int(correct),
        "DepthTopology_NumPredictedTies": int(predicted_ties),
    }


def f1_of(row: Mapping) -> float:
    gt = int(row["num_gt_objects"])
    pred = int(row["num_pred_all"])
    matched = int(row["num_matched"])
    return 2.0 * matched / (gt + pred) if gt + pred > 0 else 0.0


def step_obj_components(row: Mapping) -> dict:
    """Return every final per-step headline component and core diagnostic."""
    gt = int(row["num_gt_objects"])
    pred = int(row["num_pred_all"])
    matched = int(row["num_matched"])
    if gt <= 0:
        raise ValueError(f"{row.get('id')}: G={gt}; apply whole-case mask first")
    if pred < 0 or matched < 0 or matched > gt or matched > pred:
        raise ValueError(f"{row.get('id')}: invalid counts G={gt}, P={pred}, M={matched}")
    recall = matched / gt
    precision = matched / pred if pred else 0.0
    if matched:
        center_error = _finite(row, "Center_Error_Norm")
        position_matched = clip01(1.0 - center_error)
        integrity_matched = clip01(_finite(row, "ObjectIntegrity"))
    else:
        center_error = None
        position_matched = None
        integrity_matched = None
    position_all_gt = recall * (position_matched or 0.0)
    integrity_all_gt = recall * (integrity_matched or 0.0)

    if gt >= 2:
        planar = clip01(_finite(row, "PlanarTopology_AllGT"))
        depth = _optional_finite(
            row.get("DepthTopology_AllGT"),
            name="DepthTopology_AllGT",
            sample_id=str(row.get("id")),
        )
        relative = clip01(_finite(row, "RelativeTopology_AllGT"))
        pair_coverage = clip01(_finite(row, "GT_Pair_Coverage"))
        planar_matched = _optional_finite(
            row.get("PlanarTopology_MatchedPairs"),
            name="PlanarTopology_MatchedPairs",
            sample_id=str(row.get("id")),
        )
        depth_matched = _optional_finite(
            row.get("DepthTopology_MatchedValid"),
            name="DepthTopology_MatchedValid",
            sample_id=str(row.get("id")),
        )
        spatial = (position_all_gt + relative) / 2.0
    else:
        planar = depth = relative = pair_coverage = None
        planar_matched = depth_matched = None
        spatial = position_all_gt

    f1 = f1_of(row)
    return {
        "G": gt,
        "P": pred,
        "M": matched,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "center_error_matched": center_error,
        "position_matched": position_matched,
        "position_all_gt": position_all_gt,
        "planar_topology_all_gt_pairs": planar,
        "depth_topology_all_gt_pairs": depth,
        "relative_topology_all_gt_pairs": relative,
        "pair_coverage": pair_coverage,
        "planar_topology_matched_pairs": planar_matched,
        "depth_topology_matched_valid": depth_matched,
        "spatial": spatial,
        "integrity_matched": integrity_matched,
        "integrity_all_gt": integrity_all_gt,
        "obj_step": (f1 + spatial + integrity_all_gt) / 3.0,
    }


def mean_plus_worst(values: Iterable[float]) -> float:
    materialized = [float(value) for value in values]
    if not materialized:
        raise ValueError("mean-plus-worst requires at least one value")
    return (float(np.mean(materialized)) + min(materialized)) / 2.0


def case_obj_pillars(steps: Iterable[Mapping]) -> dict[str, float]:
    components = [step_obj_components(row) for row in sorted(steps, key=step_index)]
    return {
        "f1": mean_plus_worst(item["f1"] for item in components),
        "spatial": mean_plus_worst(item["spatial"] for item in components),
        "integrity": mean_plus_worst(item["integrity_all_gt"] for item in components),
    }


def aggregate_obj_scores(
    rows: Iterable[dict], *, invalid_cases: set[str] | None = None
) -> dict:
    materialized = list(rows)
    valid_rows, invalid = filter_gt_valid_cases(
        materialized, invalid_cases=invalid_cases
    )
    output: dict = {
        "num_valid_cases": len(group_cases(valid_rows)),
        "num_valid_steps": len(valid_rows),
        "num_invalid_cases": len(invalid),
        "invalid_cases": sorted(invalid),
    }
    for kind in SUBTYPES:
        cases = group_cases(row for row in valid_rows if subtype(row) == kind)
        pillars = [case_obj_pillars(steps) for steps in cases.values()]
        if not pillars:
            continue
        for pillar in ("f1", "spatial", "integrity"):
            output[f"{kind}_{pillar}"] = float(
                np.mean([case[pillar] for case in pillars])
            )
        output[f"obj_{kind}"] = float(
            np.mean([output[f"{kind}_{name}"] for name in ("f1", "spatial", "integrity")])
        )
        output[f"num_{kind}_cases"] = len(cases)
    output["obj"] = float(np.mean([output[f"obj_{kind}"] for kind in SUBTYPES]))
    return output
