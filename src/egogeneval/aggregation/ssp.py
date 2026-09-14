"""Canonical SSP aggregation from frozen step-level evaluator records."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from statistics import fmean
from typing import Any

from ..errors import ValidationError

SUBTYPES = ("atomic", "chain", "cycle")
STEP_PATTERN = re.compile(r"__step(\d+)$")


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _subtype(row: Mapping[str, Any]) -> str:
    parts = str(row.get("id", "")).split("__")
    if len(parts) < 3 or parts[2] not in SUBTYPES:
        raise ValidationError(f"{row.get('id')}: cannot determine SSP subtype")
    return parts[2]


def _case_id(row: Mapping[str, Any]) -> str:
    sample_id = str(row.get("id", ""))
    if "__step" not in sample_id:
        raise ValidationError(f"{sample_id}: SSP record lacks __step suffix")
    return sample_id.rsplit("__step", 1)[0]


def _step_index(row: Mapping[str, Any]) -> int:
    match = STEP_PATTERN.search(str(row.get("id", "")))
    if match is None:
        raise ValidationError(f"{row.get('id')}: SSP record lacks numbered step")
    return int(match.group(1))


def _integer(row: Mapping[str, Any], field: str) -> int:
    value = row.get(field)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValidationError(f"{row.get('id')}: required integer {field}")
    return value


def _finite(row: Mapping[str, Any], field: str) -> float:
    value = row.get(field)
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValidationError(f"{row.get('id')}: required finite {field}")
    return float(value)


def _step_components(row: Mapping[str, Any]) -> dict[str, float]:
    gt = _integer(row, "num_gt_objects")
    pred = _integer(row, "num_pred_all")
    matched = _integer(row, "num_matched")
    if gt <= 0:
        raise ValidationError(f"{row.get('id')}: G={gt}; apply whole-case mask first")
    if pred < 0 or matched < 0 or matched > gt or matched > pred:
        raise ValidationError(
            f"{row.get('id')}: invalid counts G={gt}, P={pred}, M={matched}"
        )
    recall = matched / gt
    f1 = 2.0 * matched / (gt + pred) if gt + pred else 0.0
    if matched:
        position_matched = _clip01(1.0 - _finite(row, "center_error_norm"))
        integrity_matched = _clip01(_finite(row, "object_integrity"))
    else:
        position_matched = 0.0
        integrity_matched = 0.0
    position_all_gt = recall * position_matched
    integrity_all_gt = recall * integrity_matched
    if gt >= 2:
        topology_row = row
        if row.get("RelativeTopology_AllGT") is None:
            # Canonical paper records store frozen boxes and per-object depths.
            # Compute the final planar+depth topology with the frozen module,
            # rather than requiring a redundant precomputed scalar. The frozen
            # module keeps the runner's own PascalCase keys, so hand it the name
            # it reads rather than this package's canonical `object_depths`.
            try:
                from ..official.metrics.object_metrics import compute_relative_topology

                frozen_row = {**row, "ObjectDepths": row.get("object_depths")}
                topology_row = {**row, **compute_relative_topology(frozen_row)}
            except (KeyError, TypeError, ValueError) as error:
                raise ValidationError(
                    f"{row.get('id')}: cannot compute final relative topology: {error}"
                ) from error
        relative = _clip01(_finite(topology_row, "RelativeTopology_AllGT"))
        spatial = (position_all_gt + relative) / 2.0
    else:
        spatial = position_all_gt
    return {
        "f1": f1,
        "spatial": spatial,
        "integrity": integrity_all_gt,
        "ssp_step": (f1 + spatial + integrity_all_gt) / 3.0,
    }


def ssp_step_score(row: Mapping[str, Any]) -> float:
    return _step_components(row)["ssp_step"]


def _group_cases(rows: Iterable[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_case_id(row)].append(row)
    for steps in grouped.values():
        steps.sort(key=_step_index)
    return dict(grouped)


def _mean_plus_worst(values: list[float]) -> float:
    if not values:
        raise ValidationError("SSP case has no steps")
    return (fmean(values) + min(values)) / 2.0


def _case_pillars(rows: list[Mapping[str, Any]]) -> dict[str, float]:
    components = [_step_components(row) for row in rows]
    return {
        pillar: _mean_plus_worst([item[pillar] for item in components])
        for pillar in ("f1", "spatial", "integrity")
    }


def aggregate_ssp(rows: Iterable[Mapping[str, Any]]) -> dict[str, float | int | list[str]]:
    materialized = list(rows)
    if not materialized:
        raise ValidationError("SSP records are empty")
    all_cases = _group_cases(materialized)
    invalid_cases = {
        case
        for case, steps in all_cases.items()
        if any(_integer(row, "num_gt_objects") <= 0 for row in steps)
    }
    valid_rows = [row for row in materialized if _case_id(row) not in invalid_cases]
    output: dict[str, float | int | list[str]] = {
        "num_valid_cases": len(_group_cases(valid_rows)),
        "num_valid_steps": len(valid_rows),
        "num_invalid_cases": len(invalid_cases),
        "invalid_cases": sorted(invalid_cases),
    }
    for kind in SUBTYPES:
        cases = _group_cases(row for row in valid_rows if _subtype(row) == kind)
        if not cases:
            raise ValidationError(f"missing SSP subtype: {kind}")
        pillars = [_case_pillars(cases[case]) for case in sorted(cases)]
        for pillar in ("f1", "spatial", "integrity"):
            output[f"{kind}_{pillar}"] = fmean(item[pillar] for item in pillars)
        output[f"ssp_{kind}"] = fmean(
            float(output[f"{kind}_{pillar}"])
            for pillar in ("f1", "spatial", "integrity")
        )
        output[f"num_{kind}_cases"] = len(cases)
    output["ssp"] = fmean(float(output[f"ssp_{kind}"]) for kind in SUBTYPES)
    return output
