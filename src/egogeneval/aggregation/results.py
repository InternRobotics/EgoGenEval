"""Build versioned evaluator-v0.1 result documents."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..errors import ValidationError
from ..io import read_jsonl
from ..schemas import validate_instance
from ..versioning import RESULT_NAMESPACE
from .cmg import aggregate_cmg
from .ssp import aggregate_ssp

PROVENANCE_FIELDS = (
    "benchmark_version",
    "evaluator_version",
    "manifest_sha256",
    "code_commit",
)
ASSET_PROVENANCE_FIELDS = (
    "asset_revision", "eval_frames_sha256", "reference_fingerprints_sha256",
)


def _record_ids(rows: list[dict[str, Any]], *, family: str) -> set[str]:
    identifiers = [str(row.get("id", "")) for row in rows]
    if any(not identifier for identifier in identifiers):
        raise ValidationError(f"{family} record lacks id")
    if len(set(identifiers)) != len(identifiers):
        raise ValidationError(f"{family} records contain duplicate IDs")
    return set(identifiers)


def summarize_records(
    cmg_path: Path,
    ssp_path: Path,
    provenance_block: Mapping[str, str],
) -> dict[str, Any]:
    for field in PROVENANCE_FIELDS:
        if not provenance_block.get(field):
            raise ValidationError(f"missing result provenance field: {field}")
    cmg_rows = read_jsonl(cmg_path)
    ssp_rows = read_jsonl(ssp_path)
    cmg_ids = _record_ids(cmg_rows, family="CMG")
    ssp_ids = _record_ids(ssp_rows, family="SSP")
    if cmg_ids != ssp_ids:
        cmg_only = sorted(cmg_ids - ssp_ids)
        ssp_only = sorted(ssp_ids - cmg_ids)
        raise ValidationError(
            "CMG and SSP record IDs differ; "
            f"CMG-only={cmg_only[:1]}, SSP-only={ssp_only[:1]}"
        )
    cmg = aggregate_cmg(cmg_rows)
    ssp = aggregate_ssp(ssp_rows)
    result: dict[str, Any] = {
        "result_namespace": RESULT_NAMESPACE,
        **{field: str(provenance_block[field]) for field in PROVENANCE_FIELDS},
        **{field: str(provenance_block[field]) for field in ASSET_PROVENANCE_FIELDS
           if field in provenance_block},
        "eligible": len(cmg_ids),
        "evaluated": len(cmg_ids),
        "missing": 0,
        "failed": 0,
        "cmg": float(cmg["cmg"]),
        "cmg_atomic": float(cmg["cmg_atomic"]),
        "cmg_chain": float(cmg["cmg_chain"]),
        "cmg_cycle": float(cmg["cmg_cycle"]),
        "ssp": float(ssp["ssp"]),
        "ssp_atomic": float(ssp["ssp_atomic"]),
        "ssp_chain": float(ssp["ssp_chain"]),
        "ssp_cycle": float(ssp["ssp_cycle"]),
    }
    result["overall"] = (result["cmg"] + result["ssp"]) / 2.0
    validate_instance(result, "result", label="result")
    return result
