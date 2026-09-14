"""Validate generated-image prediction manifests."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from .data import ManifestIndex
from .errors import ValidationError
from .io import read_jsonl, sha256_file
from .schemas import validate_instance

PredictionKey = tuple[str, int]


@dataclass(frozen=True)
class PredictionReport:
    total: int
    missing: tuple[PredictionKey, ...]
    duplicates: tuple[PredictionKey, ...]
    unexpected: tuple[PredictionKey, ...]
    hash_mismatches: tuple[PredictionKey, ...]


def _output_path(predictions_path: Path, relative_text: str) -> Path:
    root = predictions_path.parent.resolve()
    output = (root / relative_text).resolve()
    try:
        output.relative_to(root)
    except ValueError as error:
        raise ValidationError(f"output path escapes prediction directory: {relative_text}") from error
    return output


#: Retries for a transient read error before declaring an image unreadable.
#: Generations often live on a network filesystem (NFS, FUSE, a cloud mount),
#: where a single ``read`` can be interrupted and surface as ``OSError`` even
#: though the bytes are intact -- verified by an immediate re-open. Decode
#: errors (``UnidentifiedImageError`` / ``SyntaxError``) mean the file itself is
#: wrong, so those fail on the first try; only I/O is retried.
_IMAGE_READ_RETRIES = 3
_IMAGE_READ_BACKOFF_S = 0.5


def _verify_image(path: Path, key: PredictionKey) -> None:
    # Report the full path, not ``path.name``. These files are materialised
    # copies under ``<run>/outputs/``, so the bare name is a flat
    # ``<case>_step<N>.png`` that in general does not exist anywhere in the
    # user's generation directory -- it cannot be grepped for or opened.
    if not path.is_file():
        raise ValidationError(f"{key}: output file is missing: {path}")
    for attempt in range(_IMAGE_READ_RETRIES):
        try:
            with Image.open(path) as image:
                image.verify()
            return
        except (SyntaxError, UnidentifiedImageError) as error:
            raise ValidationError(f"{key}: unreadable image: {path}") from error
        except OSError as error:
            if attempt + 1 == _IMAGE_READ_RETRIES:
                raise ValidationError(
                    f"{key}: unreadable image after {_IMAGE_READ_RETRIES} attempts: "
                    f"{path}"
                ) from error
            time.sleep(_IMAGE_READ_BACKOFF_S * (attempt + 1))


def validate_predictions(
    path: Path,
    manifest: ManifestIndex,
    *,
    check_files: bool = True,
    require_complete: bool = True,
) -> PredictionReport:
    rows = read_jsonl(path)
    if not rows:
        raise ValidationError(f"{path}: predictions file is empty")
    expected = tuple(manifest.expected_steps)
    expected_set = set(expected)
    seen: set[PredictionKey] = set()
    ordered: list[PredictionKey] = []
    duplicates: list[PredictionKey] = []
    unexpected: list[PredictionKey] = []
    hash_mismatches: list[PredictionKey] = []
    model_id: str | None = None

    for line_number, row in enumerate(rows, start=1):
        validate_instance(row, "prediction-row", label=f"prediction row {line_number}")
        if row["manifest_sha256"] != manifest.sha256:
            raise ValidationError(
                f"prediction row {line_number}.manifest_sha256 does not match the manifest"
            )
        current_model = str(row["model_id"])
        if model_id is None:
            model_id = current_model
        elif current_model != model_id:
            raise ValidationError("predictions contain more than one model_id")
        key = (str(row["case_id"]), int(row["step_id"]))
        ordered.append(key)
        if key in seen:
            duplicates.append(key)
        seen.add(key)
        if key not in expected_set:
            unexpected.append(key)
        output = _output_path(path, str(row["output_path"]))
        if check_files:
            _verify_image(output, key)
            if sha256_file(output) != row["output_sha256"]:
                hash_mismatches.append(key)

    if duplicates:
        raise ValidationError(f"duplicate prediction: {duplicates[0]}")
    if unexpected:
        raise ValidationError(f"unexpected prediction: {unexpected[0]}")
    missing = tuple(key for key in expected if key not in seen)
    if require_complete and missing:
        noun = "prediction" if len(missing) == 1 else "predictions"
        raise ValidationError(f"missing {len(missing)} {noun}; first missing key: {missing[0]}")
    if require_complete and tuple(ordered) != expected:
        raise ValidationError("predictions are not in manifest step order")
    if hash_mismatches:
        raise ValidationError(f"output SHA-256 mismatch: {hash_mismatches[0]}")
    return PredictionReport(
        total=len(rows),
        missing=missing,
        duplicates=tuple(duplicates),
        unexpected=tuple(unexpected),
        hash_mismatches=tuple(hash_mismatches),
    )
