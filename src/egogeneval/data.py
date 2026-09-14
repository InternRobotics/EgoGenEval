"""Manifest validation and local data preparation."""

from __future__ import annotations

import copy
import os
import re
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .errors import ValidationError
from .io import read_jsonl, sha256_file, write_jsonl
from .schemas import validate_instance

ROOT_TOKEN = re.compile(r"^\$\{([A-Z][A-Z0-9_]*)\}/(.+)$")
SAFE_CASE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]*$")


@dataclass(frozen=True)
class ManifestIndex:
    rows: tuple[dict, ...]
    expected_steps: tuple[tuple[str, int], ...]
    sha256: str


def _validate_row_contract(row: dict, *, line_number: int) -> None:
    sample_id = str(row["sample_id"])
    if not SAFE_CASE_ID.fullmatch(sample_id):
        raise ValidationError(
            f"manifest row {line_number}.sample_id: unsafe public identifier {sample_id!r}"
        )
    inputs = row["input_images"]
    targets = row["target_images"]
    instructions = row["instructions"]
    if int(row["num_context_images"]) != len(inputs):
        raise ValidationError(
            f"{sample_id}: num_context_images does not match input_images"
        )
    expected_steps = list(range(1, len(instructions) + 1))
    actual_steps = [int(item["step"]) for item in instructions]
    if actual_steps != expected_steps:
        raise ValidationError(
            f"{sample_id}: instruction steps must be contiguous from 1; got {actual_steps}"
        )
    if len(targets) != len(instructions):
        raise ValidationError(
            f"{sample_id}: target_images and instructions have different lengths"
        )
    if int(row["evaluation_protocol"]["num_model_calls"]) != len(instructions):
        raise ValidationError(
            f"{sample_id}: num_model_calls does not match instruction count"
        )
    input_paths = {str(image["image_path"]) for image in inputs}
    target_paths = {str(image["image_path"]) for image in targets}
    overlap = sorted(input_paths & target_paths)
    if overlap and str(row["instruction_type"]).lower() != "cycle":
        raise ValidationError(f"{sample_id}: target leakage through shared path {overlap[0]}")


def validate_manifest(path: Path) -> ManifestIndex:
    rows = read_jsonl(path)
    if not rows:
        raise ValidationError(f"{path}: manifest is empty")
    seen: set[str] = set()
    expected_steps: list[tuple[str, int]] = []
    for line_number, row in enumerate(rows, start=1):
        validate_instance(row, "manifest-row", label=f"manifest row {line_number}")
        _validate_row_contract(row, line_number=line_number)
        sample_id = str(row["sample_id"])
        if sample_id in seen:
            raise ValidationError(f"duplicate sample_id: {sample_id}")
        seen.add(sample_id)
        expected_steps.extend((sample_id, int(item["step"])) for item in row["instructions"])
    return ManifestIndex(
        rows=tuple(rows),
        expected_steps=tuple(expected_steps),
        sha256=sha256_file(path),
    )


def _resolve_source(path_text: str, roots: Mapping[str, Path]) -> Path:
    match = ROOT_TOKEN.fullmatch(path_text)
    if not match:
        raise ValidationError(
            f"manifest path must use a declared ${{ROOT}} token: {path_text}"
        )
    root_name, relative_text = match.groups()
    root_value = roots.get(root_name)
    if root_value is None:
        environment_value = os.environ.get(root_name)
        root_value = Path(environment_value) if environment_value else None
    if root_value is None:
        raise ValidationError(f"missing dataset root: {root_name}")
    root = Path(root_value).expanduser().resolve()
    source = (root / relative_text).resolve()
    try:
        source.relative_to(root)
    except ValueError as error:
        raise ValidationError(f"path escapes dataset root {root_name}: {path_text}") from error
    if not source.is_file():
        raise ValidationError(f"missing source file: {source}")
    return source


def _materialize(source: Path, destination: Path, *, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        destination.symlink_to(source)
    elif mode == "copy":
        shutil.copy2(source, destination)
    else:
        raise ValidationError(f"unsupported prepare mode: {mode}")


def _swap_staging(staging: Path, destination: Path) -> None:
    if not destination.exists():
        os.replace(staging, destination)
        return
    backup = destination.with_name(f".{destination.name}.previous")
    if backup.exists():
        shutil.rmtree(backup)
    os.replace(destination, backup)
    try:
        os.replace(staging, destination)
    except BaseException:
        os.replace(backup, destination)
        raise
    shutil.rmtree(backup)


def prepare_data(
    manifest: Path,
    roots: Mapping[str, Path],
    destination: Path,
    *,
    mode: Literal["symlink", "copy"] = "symlink",
) -> dict[str, int]:
    index = validate_manifest(manifest)
    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    )
    prepared_rows: list[dict] = []
    files = 0
    try:
        for row in index.rows:
            prepared = copy.deepcopy(row)
            case_id = str(row["sample_id"])
            for field, stem in (("input_images", "input"), ("target_images", "target")):
                for index_number, (source_item, prepared_item) in enumerate(
                    zip(row[field], prepared[field]), start=1
                ):
                    source = _resolve_source(str(source_item["image_path"]), roots)
                    suffix = source.suffix.lower() or ".bin"
                    relative = Path("cases") / case_id / f"{stem}_{index_number}{suffix}"
                    _materialize(source, staging / relative, mode=mode)
                    prepared_item["image_path"] = relative.as_posix()
                    files += 1
            prepared_rows.append(prepared)
        write_jsonl(staging / "prepared_manifest.jsonl", prepared_rows)
        _swap_staging(staging, destination)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return {"cases": len(prepared_rows), "files": files, "missing": 0}
