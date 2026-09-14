"""Fetch benchmark data from the Hugging Face release.

The benchmark dataset (``longyilin/EgoGenEval``) exposes exactly two configs:

- ``test`` — the benchmark identity/protocol. Each row carries a ``manifest_json``
  string, portable ``${ROOT}`` references, a ``manifest_sha256``, and embedded
  ``input_images`` / ``target_images`` byte lists.
- ``eval_frames`` — the physical depth and camera calibration bank keyed by
  the same portable image references used by ``test``.

Training data is distributed through a separate repository and is intentionally
not a config of the benchmark dataset.

:func:`download_config` auto-detects which shape a config has from ``ds.features`` and
either reconstructs the JSONL manifest, materialises embedded images, or both — so the
caller does not have to know a config's schema in advance.

``datasets`` is an optional dependency (the ``egogeneval[data]`` extra); it is imported
lazily so the base package stays importable without it.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import tempfile
from io import BytesIO
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image as PILImage
from PIL import UnidentifiedImageError

from .data import SAFE_CASE_ID, validate_manifest
from .errors import ValidationError
from .io import sha256_file, write_jsonl

DEFAULT_REPO_ID = "longyilin/EgoGenEval"
CANONICAL_MANIFEST_NAME = "egogeneval_v0.1.jsonl"

# Configs whose rows carry a ``manifest_json`` column (reconstructable manifest).
MANIFEST_CONFIGS = frozenset({"test"})
KNOWN_CONFIGS = ("test", "eval_frames")

_INSTALL_HINT = (
    "the 'datasets' package is required to download from Hugging Face; install it with:"
    "\n  pip install 'egogeneval[data]'"
)

# Module-level indirection keeps ``datasets`` an optional, lazily loaded dependency.
load_dataset = None  # type: ignore[assignment]


@dataclass
class DownloadResult:
    repo_id: str
    config: str
    output_dir: Path
    num_rows: int
    schema: str
    manifest_path: Path | None = None
    prepared_manifest_path: Path | None = None
    manifest_sha256: str | None = None
    images_written: int = 0
    depths_written: int = 0
    warnings: list[str] = field(default_factory=list)


def _load_dataset(repo_id: str, config: str, *, revision: str | None, token: str | None):
    loader = load_dataset
    if loader is None:
        try:
            from datasets import load_dataset as loader  # type: ignore[no-redef]
        except ImportError as error:
            raise ValidationError(_INSTALL_HINT) from error
    return loader(repo_id, config, revision=revision, token=token)


def _select_split(dataset: Any):
    """Return a single split from a ``Dataset`` or ``DatasetDict``."""

    # A plain Dataset already has column_names/features; a DatasetDict is a mapping.
    if hasattr(dataset, "column_names") and not isinstance(dataset, dict):
        return dataset
    keys = list(dataset.keys())
    if not keys:
        raise ValidationError("Hugging Face returned an empty dataset")
    for preferred in ("test", "train", "validation"):
        if preferred in dataset:
            return dataset[preferred]
    return dataset[keys[0]]


def _feature_contains_image(feature: Any) -> bool:
    """Whether a datasets feature is an Image or (possibly nested) binary value."""

    if type(feature).__name__ == "Image" or feature == "Image":
        return True
    dtype = str(getattr(feature, "dtype", "")).lower()
    if dtype in {"binary", "large_binary"}:
        return True
    nested = getattr(feature, "feature", None)
    return nested is not None and _feature_contains_image(nested)


def _image_columns(features: Any) -> list[str]:
    """Column names that hold a single image or a list of image byte strings."""

    columns: list[str] = []
    for name, feature in dict(features).items():
        type_name = type(feature).__name__
        image_name = name == "image" or name.endswith(
            ("_image", "_images", "_image_bytes", "_png", "_jpg", "_jpeg")
        )
        if (
            type_name == "Image"
            or feature == "Image"
            or (image_name and _feature_contains_image(feature))
        ):
            columns.append(name)
    return columns


def _feature_contains_decoded_image(feature: Any) -> bool:
    """Whether a feature contains a real ``datasets.Image`` decoder."""

    if type(feature).__name__ == "Image":
        return True
    nested = getattr(feature, "feature", None)
    return nested is not None and _feature_contains_decoded_image(nested)


def _without_image_decoding(feature: Any) -> Any:
    """Copy a datasets feature and disable every nested Image decoder."""

    result = copy.deepcopy(feature)
    current = result
    while current is not None:
        if type(current).__name__ == "Image":
            current.decode = False
            break
        current = getattr(current, "feature", None)
    return result


class _EncodedImageSplit:
    """Row iterator over a Dataset's raw Arrow values (no media decoding)."""

    def __init__(self, split: Any):
        self._split = split.with_format("arrow")
        self.features = split.features
        self.column_names = split.column_names
        self.num_rows = split.num_rows

    def __iter__(self):
        # Bound peak memory for large training images while retaining Arrow's cheap
        # batch conversion instead of decoding one media feature at a time.
        for start in range(0, self.num_rows, 16):
            batch = self._split[start : start + 16]
            yield from batch.to_pylist()


def _preserve_encoded_images(split: Any, image_columns: list[str]) -> Any:
    """Ask datasets for its encoded ``{bytes, path}`` values, not decoded PIL images.

    Arrow formatting returns the physical Image struct without rewriting a possibly
    multi-gigabyte dataset cache. ``Dataset.cast_column`` is retained as a fallback
    for Dataset-compatible implementations without Arrow formatting. Binary columns
    need neither path.
    """

    decoded_columns = [
        column
        for column in image_columns
        if _feature_contains_decoded_image(split.features[column])
    ]
    if not decoded_columns:
        return split
    if callable(getattr(split, "with_format", None)) and hasattr(split, "num_rows"):
        return _EncodedImageSplit(split)
    cast_column = getattr(split, "cast_column", None)
    if not callable(cast_column):
        return split
    for column in decoded_columns:
        feature = split.features[column]
        split = split.cast_column(column, _without_image_decoding(feature))
    return split


def _safe_stem(value: str, fallback: str) -> str:
    stem = str(value or "").strip().replace("/", "_")
    if not stem or not SAFE_CASE_ID.fullmatch(stem):
        return fallback
    return stem


_IMAGE_EXTENSIONS = {
    "BMP": "bmp",
    "GIF": "gif",
    "JPEG": "jpg",
    "PNG": "png",
    "TIFF": "tiff",
    "WEBP": "webp",
}


def _extension(format_name: str | None) -> str:
    normalized = (format_name or "PNG").upper()
    return _IMAGE_EXTENSIONS.get(normalized, normalized.lower())


_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")


def _encoded_image_payload(image: Any) -> tuple[bytes, str | None]:
    """Return encoded bytes and their detected format.

    Hugging Face Image values are normally dictionaries because decoding is disabled
    before iteration. A PIL value is still accepted for backwards compatibility. If
    it originated from a local encoded file, ``filename`` lets us retain those exact
    bytes; only an in-memory PIL image with no encoded source must be serialized.
    """

    if isinstance(image, dict):
        embedded = image.get("bytes")
        if embedded is not None:
            image = embedded
        elif image.get("path"):
            source = Path(str(image["path"]))
            if not source.is_file():
                raise ValidationError(f"Hugging Face image path is missing: {source}")
            image = source.read_bytes()

    if isinstance(image, (bytes, bytearray, memoryview)):
        payload = bytes(image)
        try:
            with PILImage.open(BytesIO(payload)) as decoded:
                format_name = decoded.format
                decoded.verify()
        except (OSError, UnidentifiedImageError) as error:
            raise ValidationError("embedded Hugging Face image bytes are unreadable") from error
        return payload, format_name

    if not hasattr(image, "save"):
        raise ValidationError(f"unsupported Hugging Face image value: {type(image).__name__}")

    source_name = getattr(image, "filename", None)
    if source_name and Path(source_name).is_file():
        return _encoded_image_payload(Path(source_name).read_bytes())

    # Compatibility fallback for caller-created, in-memory PIL images. Real HF Image
    # columns take the decode=False path above and therefore never reach this branch.
    format_name = getattr(image, "format", None) or "PNG"
    buffer = BytesIO()
    image.save(buffer, format=format_name)
    return _encoded_image_payload(buffer.getvalue())


def _write_image_value(
    image: Any,
    destination_stem: Path,
    *,
    expected_sha256: str | None = None,
    label: str,
) -> Path:
    """Write one image value byte-for-byte and verify its optional source digest."""

    payload, format_name = _encoded_image_payload(image)
    digest = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None:
        expected = str(expected_sha256)
        if not _SHA256.fullmatch(expected):
            raise ValidationError(f"{label}: invalid image sha256 {expected!r}")
        if digest != expected.lower():
            raise ValidationError(
                f"{label}: image sha256 mismatch ({digest} != {expected.lower()})"
            )
    # ``Path.with_suffix`` would truncate sample IDs containing a dot. The stem has
    # already been validated, so append the detected encoded extension literally.
    destination = Path(f"{destination_stem}.{_extension(format_name)}")
    destination.write_bytes(payload)
    return destination


def _sha_column(row: dict[str, Any], column: str, *, only_image_column: bool) -> str | None:
    candidates = [f"{column}_sha256"]
    if column.endswith("_image"):
        candidates.append(f"{column.removesuffix('_image')}_sha256")
    if only_image_column:
        candidates.append("sha256")
    return next((candidate for candidate in candidates if candidate in row), None)


def _expected_image_shas(
    row: dict[str, Any],
    column: str,
    *,
    count: int,
    is_sequence: bool,
    only_image_column: bool,
    label: str,
) -> list[str | None]:
    sha_column = _sha_column(row, column, only_image_column=only_image_column)
    if sha_column is None:
        return [None] * count
    value = row.get(sha_column)
    if is_sequence:
        if not isinstance(value, (list, tuple)) or len(value) != count:
            actual = len(value) if isinstance(value, (list, tuple)) else type(value).__name__
            raise ValidationError(
                f"{label}: {sha_column} must contain one digest per image "
                f"(expected {count}, got {actual})"
            )
        return [str(item) for item in value]
    if not isinstance(value, str):
        raise ValidationError(f"{label}: {sha_column} must be a sha256 string")
    return [value]


def _write_images(
    split: Any, image_columns: list[str], images_dir: Path
) -> tuple[int, dict[tuple[int, str, int], Path]]:
    images_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    materialized: dict[tuple[int, str, int], Path] = {}
    destination_stems: set[Path] = set()
    id_field = next(
        (
            name
            for name in ("image_id", "asset_id", "sample_id", "pair_id")
            if name in split.column_names
        ),
        None,
    )
    for index, row in enumerate(split):
        base = _safe_stem(row.get(id_field) if id_field else "", f"row{index:06d}")
        for column in image_columns:
            value = row.get(column)
            if value is None:
                continue
            is_sequence = isinstance(value, (list, tuple))
            images = value if is_sequence else [value]
            label = f"{base}.{column}"
            expected_shas = _expected_image_shas(
                row,
                column,
                count=len(images),
                is_sequence=is_sequence,
                only_image_column=len(image_columns) == 1,
                label=label,
            )
            for image_index, image in enumerate(images):
                if image is None:
                    if expected_shas[image_index] is not None:
                        raise ValidationError(
                            f"{label}[{image_index}]: digest is present but image is null"
                        )
                    continue
                column_suffix = f"__{column}" if len(image_columns) > 1 else ""
                index_suffix = f"__{image_index:02d}" if is_sequence else ""
                destination_stem = images_dir / f"{base}{column_suffix}{index_suffix}"
                if destination_stem in destination_stems:
                    raise ValidationError(
                        f"duplicate materialized image identity: {destination_stem.name}"
                    )
                destination_stems.add(destination_stem)
                destination = _write_image_value(
                    image,
                    destination_stem,
                    expected_sha256=expected_shas[image_index],
                    label=f"{label}[{image_index}]",
                )
                key = (index, column, image_index)
                materialized[key] = destination
                written += 1
    return written, materialized


def _write_eval_frames(
    split: Any, output_dir: Path, *, image_columns: list[str] | None = None
) -> int:
    """Materialize the private physical-depth bank and its calibration index."""

    required = {
        "asset_id",
        "dataset",
        "scene_id",
        "image_ref",
        "rgb_sha256",
        "depth_npy",
        "depth_sha256",
        "depth_height",
        "depth_width",
        "image_height",
        "image_width",
        "extrinsics_c2w",
        "intrinsics",
    }
    missing = sorted(required - set(split.column_names))
    if missing:
        raise ValidationError(f"eval_frames config is missing columns: {missing}")

    depth_dir = output_dir / "depth"
    depth_dir.mkdir(parents=True, exist_ok=True)
    metadata: list[dict[str, Any]] = []
    written = 0
    seen_ids: set[str] = set()
    seen_refs: set[str] = set()
    for row in split:
        asset_id = _safe_stem(str(row.get("asset_id") or ""), "")
        image_ref = str(row.get("image_ref") or "")
        if not asset_id:
            raise ValidationError("eval_frames row has an invalid asset_id")
        if asset_id in seen_ids or image_ref in seen_refs:
            raise ValidationError(f"duplicate eval_frames identity: {asset_id} / {image_ref}")
        seen_ids.add(asset_id)
        seen_refs.add(image_ref)

        payload = row.get("depth_npy")
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise ValidationError(f"{asset_id}: depth_npy is not binary")
        payload = bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        expected = str(row.get("depth_sha256") or "")
        if digest != expected:
            raise ValidationError(f"{asset_id}: depth_npy sha256 mismatch")
        destination = depth_dir / f"{asset_id}.npy"
        destination.write_bytes(payload)

        excluded = {"depth_npy", *(image_columns or [])}
        clean = {key: value for key, value in row.items() if key not in excluded}
        if image_columns:
            candidates = sorted((output_dir / "images").glob(f"{asset_id}.*"))
            if len(candidates) != 1:
                raise ValidationError(
                    f"{asset_id}: expected one materialized RGB image, found {len(candidates)}"
                )
            if sha256_file(candidates[0]) != str(row.get("rgb_sha256") or ""):
                raise ValidationError(f"{asset_id}: materialized RGB sha256 mismatch")
            clean["image_path"] = str(candidates[0].relative_to(output_dir))
        clean["depth_path"] = str(destination.relative_to(output_dir))
        metadata.append(clean)
        written += 1

    write_jsonl(output_dir / "eval_frames.jsonl", metadata)
    return written


def _reconstruct_manifest(
    split: Any,
    output_dir: Path,
    *,
    verify_manifest_sha: str | None,
    warnings: list[str],
) -> tuple[Path, str, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    source_shas: set[str] = set()
    for row in split:
        raw = row.get("manifest_json")
        if raw is None:
            raise ValidationError("test config row is missing 'manifest_json'")
        rows.append(json.loads(raw) if isinstance(raw, str) else dict(raw))
        source = row.get("manifest_sha256") or row.get("source_manifest_sha256")
        if source:
            source_shas.add(str(source))

    manifest_path = output_dir / CANONICAL_MANIFEST_NAME
    write_jsonl(manifest_path, rows)

    # A reconstructed file that is not a valid manifest is a hard failure.
    validate_manifest(manifest_path)
    digest = sha256_file(manifest_path)

    if verify_manifest_sha and digest != verify_manifest_sha:
        raise ValidationError(
            f"reconstructed manifest sha256 {digest} != expected {verify_manifest_sha}; "
            "the Hugging Face copy has drifted from the committed manifest"
        )
    if len(source_shas) > 1:
        warnings.append(f"rows disagree on manifest_sha256 ({len(source_shas)} distinct values)")
    elif source_shas and digest not in source_shas:
        # Serialization may differ from the byte stream the release recorded; surface
        # it rather than failing, since content validity was already checked above.
        warnings.append(
            "reconstructed manifest sha256 differs from the recorded "
            f"manifest_sha256 ({next(iter(source_shas))}); content is valid but "
            "byte serialization differs"
        )
    return manifest_path, digest, rows


def _validate_reference_order(
    hf_row: dict[str, Any], manifest_row: dict[str, Any], field: str, *, sample_id: str
) -> None:
    reference_column = f"{field.removesuffix('_images')}_refs_json"
    raw = hf_row.get(reference_column)
    if raw is None:
        return
    try:
        references = json.loads(raw) if isinstance(raw, str) else list(raw)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValidationError(f"{sample_id}: invalid {reference_column}") from error
    expected = [str(item["image_path"]) for item in manifest_row[field]]
    if references != expected:
        raise ValidationError(
            f"{sample_id}: {reference_column} order disagrees with manifest_json"
        )


def _write_prepared_manifest(
    split: Any,
    manifest_rows: list[dict[str, Any]],
    materialized: dict[tuple[int, str, int], Path],
    output_dir: Path,
) -> Path | None:
    """Write a self-contained manifest when every test image is embedded.

    The canonical manifest is intentionally left untouched for identity/hash checks.
    This second manifest changes only ``image_path`` values to paths below the HF
    download directory, matching the contract of ``prepare-data``.
    """

    prepared_rows: list[dict[str, Any]] = []
    hf_rows = iter(split)
    for row_index, manifest_row in enumerate(manifest_rows):
        try:
            hf_row = next(hf_rows)
        except StopIteration as error:
            raise ValidationError(
                "test config has fewer rows than its reconstructed manifest"
            ) from error
        sample_id = str(manifest_row["sample_id"])
        hf_sample_id = hf_row.get("sample_id")
        if hf_sample_id is not None and str(hf_sample_id) != sample_id:
            raise ValidationError(
                f"test config row {row_index}: sample_id {hf_sample_id!r} "
                f"!= manifest_json sample_id {sample_id!r}"
            )
        prepared = copy.deepcopy(manifest_row)
        for image_field in ("input_images", "target_images"):
            _validate_reference_order(
                hf_row, manifest_row, image_field, sample_id=sample_id
            )
            embedded = hf_row.get(image_field)
            if not isinstance(embedded, (list, tuple)) or len(embedded) != len(
                manifest_row[image_field]
            ):
                # A source-gated release may deliberately omit a dataset's pixels.
                # Such a download cannot honestly claim to be a complete prepared
                # tree; leave only the canonical manifest and materialized subset.
                return None
            for image_index, prepared_image in enumerate(prepared[image_field]):
                path = materialized.get((row_index, image_field, image_index))
                if path is None:
                    return None
                prepared_image["image_path"] = path.relative_to(output_dir).as_posix()
        prepared_rows.append(prepared)

    try:
        next(hf_rows)
    except StopIteration:
        pass
    else:
        raise ValidationError("test config has more rows than its reconstructed manifest")

    path = output_dir / "prepared_manifest.jsonl"
    write_jsonl(path, prepared_rows)
    validate_manifest(path)
    return path


def _swap_download(staging: Path, destination: Path) -> None:
    """Atomically replace one downloader-owned config directory."""

    if destination.is_symlink():
        raise ValidationError(f"refusing to replace symlink download destination: {destination}")
    if destination.exists() and not destination.is_dir():
        raise ValidationError(f"download destination is not a directory: {destination}")
    if not destination.exists():
        os.replace(staging, destination)
        return

    backup_root = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.previous-", dir=destination.parent)
    )
    backup = backup_root / destination.name
    os.replace(destination, backup)
    try:
        os.replace(staging, destination)
    except BaseException:
        os.replace(backup, destination)
        raise
    # Installation succeeded; a cleanup failure must not report the download as
    # failed after the new directory is already live. Any residue remains a hidden,
    # recoverable previous copy next to the destination.
    shutil.rmtree(backup_root, ignore_errors=True)


def download_config(
    *,
    repo_id: str = DEFAULT_REPO_ID,
    config: str,
    output: str | Path,
    revision: str | None = None,
    token: str | None = None,
    verify_manifest_sha: str | None = None,
) -> DownloadResult:
    """Download one HF config into ``output``; reconstruct/materialise per its schema."""

    requested_output = Path(output).expanduser()
    if requested_output.name in {"", ".."}:
        raise ValidationError(f"unsafe download destination: {requested_output}")
    output_dir = requested_output.absolute()
    if output_dir in {Path.cwd().absolute(), Path.home().absolute(), Path("/")}:
        raise ValidationError(f"unsafe download destination: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.is_symlink():
        raise ValidationError(f"refusing to replace symlink download destination: {output_dir}")
    warnings: list[str] = []

    dataset = _load_dataset(repo_id, config, revision=revision, token=token)
    split = _select_split(dataset)
    features = split.features
    columns = set(split.column_names)

    # Only the benchmark ``test`` config's manifest_json represents the
    # canonical 1,400-case evaluation manifest. A schema-compatible config from
    # another repository may also carry a field with that name, but it must
    # remain ordinary config metadata.
    if config in MANIFEST_CONFIGS and "manifest_json" not in columns:
        raise ValidationError(f"{config} config is missing required column: manifest_json")
    has_manifest = config in MANIFEST_CONFIGS
    image_columns = _image_columns(features)
    split = _preserve_encoded_images(split, image_columns)

    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    )

    manifest_path: Path | None = None
    prepared_manifest_path: Path | None = None
    manifest_sha: str | None = None
    manifest_rows: list[dict[str, Any]] | None = None
    images_written = 0
    depths_written = 0
    schema_parts: list[str] = []

    try:
        if has_manifest:
            manifest_path, manifest_sha, manifest_rows = _reconstruct_manifest(
                split, staging, verify_manifest_sha=verify_manifest_sha, warnings=warnings
            )
            schema_parts.append("manifest-refs")

        materialized: dict[tuple[int, str, int], Path] = {}
        if image_columns:
            images_written, materialized = _write_images(
                split, image_columns, staging / "images"
            )
            if "depth_npy" not in columns:
                # Row metadata (minus the heavy image columns) for reference.
                meta_rows = [
                    {key: value for key, value in row.items() if key not in image_columns}
                    for row in split
                ]
                write_jsonl(staging / f"{config}.jsonl", meta_rows)
            schema_parts.append("images-embedded")

        if config == "test" and manifest_rows is not None and image_columns:
            prepared_manifest_path = _write_prepared_manifest(
                split, manifest_rows, materialized, staging
            )

        if "depth_npy" in columns:
            depths_written = _write_eval_frames(
                split, staging, image_columns=image_columns
            )
            schema_parts.append("physical-depth+calibration")

        if not schema_parts:
            # Metadata-only config from an explicitly selected repository: dump rows.
            write_jsonl(staging / f"{config}.jsonl", [dict(row) for row in split])
            schema_parts.append("metadata")

        _swap_download(staging, output_dir)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    if manifest_path is not None:
        manifest_path = output_dir / manifest_path.name
    if prepared_manifest_path is not None:
        prepared_manifest_path = output_dir / prepared_manifest_path.name

    return DownloadResult(
        repo_id=repo_id,
        config=config,
        output_dir=output_dir,
        num_rows=split.num_rows,
        schema="+".join(schema_parts),
        manifest_path=manifest_path,
        prepared_manifest_path=prepared_manifest_path,
        manifest_sha256=manifest_sha,
        images_written=images_written,
        depths_written=depths_written,
        warnings=warnings,
    )
