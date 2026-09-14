"""Dependency-light canonical CMG aggregation."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from statistics import fmean
from typing import Any

from ..errors import ValidationError

SUBTYPES = ("atomic", "chain", "cycle")


def _subtype(row: Mapping[str, Any]) -> str:
    parts = str(row.get("id", "")).split("__")
    if len(parts) < 3 or parts[2] not in SUBTYPES:
        raise ValidationError(f"{row.get('id')}: cannot determine CMG subtype")
    return parts[2]


def _case_id(row: Mapping[str, Any]) -> str:
    sample_id = str(row.get("id", ""))
    if "__step" not in sample_id:
        raise ValidationError(f"{sample_id}: CMG record lacks __step suffix")
    return sample_id.rsplit("__step", 1)[0]


def _finite(row: Mapping[str, Any], field: str) -> float:
    value = row.get(field)
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValidationError(f"{row.get('id')}: required finite {field}")
    return float(value)


def cmg_step_score(row: Mapping[str, Any]) -> float:
    direction = _finite(row, "action_sign_correct")
    if not 0.0 <= direction <= 1.0:
        raise ValidationError(f"{row.get('id')}: invalid action_sign_correct")
    if direction == 0.0:
        return 0.0
    expected = _finite(row, "action_expected")
    absolute_error = _finite(row, "action_abs_err")
    if absolute_error < 0.0:
        raise ValidationError(f"{row.get('id')}: negative action_abs_err")
    unit = row.get("action_unit")
    if not isinstance(unit, str) or not unit:
        raise ValidationError(f"{row.get('id')}: required action_unit")
    floor = 0.1 if unit.startswith("m") else 5.0
    relative_error = absolute_error / max(abs(expected), floor)
    magnitude_quality = 1.0 / (1.0 + relative_error)
    return direction * (1.0 + magnitude_quality) / 2.0


def aggregate_cmg(rows: Iterable[Mapping[str, Any]]) -> dict[str, float | int]:
    materialized = list(rows)
    if not materialized:
        raise ValidationError("CMG records are empty")
    grouped: dict[str, dict[str, list[Mapping[str, Any]]]] = {
        kind: defaultdict(list) for kind in SUBTYPES
    }
    for row in materialized:
        grouped[_subtype(row)][_case_id(row)].append(row)
    output: dict[str, float | int] = {
        "num_valid_cases": sum(len(cases) for cases in grouped.values()),
        "num_valid_steps": len(materialized),
    }
    for kind in SUBTYPES:
        cases = grouped[kind]
        if not cases:
            raise ValidationError(f"missing CMG subtype: {kind}")
        case_scores = [
            fmean(cmg_step_score(row) for row in cases[case])
            for case in sorted(cases)
        ]
        output[f"cmg_{kind}"] = fmean(case_scores)
        output[f"num_{kind}_cases"] = len(case_scores)
    output["cmg"] = fmean(float(output[f"cmg_{kind}"]) for kind in SUBTYPES)
    return output
