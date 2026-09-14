"""Prepare the complete benchmark from user-obtained official datasets."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from ..data import validate_manifest
from ..errors import ValidationError
from ..io import read_jsonl, write_jsonl
from .acquire import RAW_DATASETS
from .build import build_data
from .fingerprints import _reference_row
from .jpeg_tools import default_cache, ensure_jpeg_tools, validate_jpeg_tools
from .sources import DATASET_ROOTS, requests_from_rows


def _bundled_fingerprints() -> list[Path]:
    directory = Path(__file__).resolve().parents[3] / "data" / "manifests"
    return [directory / "egogeneval_v0.1.scannetpp-depth-v2.fingerprints.jsonl"]


def _references(paths: list[Path], expected: set[str]) -> list[dict]:
    references = {}
    for path in paths:
        if not path.is_file():
            raise ValidationError(f"benchmark fingerprints not found: {path}; run from a complete checkout")
        for raw in read_jsonl(path):
            row = _reference_row(raw)
            ref = row["image_ref"]
            if ref in references:
                raise ValidationError(f"duplicate benchmark fingerprint: {ref}")
            references[ref] = row
    missing = expected - references.keys()
    if missing:
        raise ValidationError(f"benchmark fingerprints lack {len(missing)} selected frames; first: {min(missing)}")
    return [references[ref] for ref in sorted(expected)]


def _scannetpp_environment() -> dict[str, str]:
    if sys.platform != "linux":
        raise ValidationError("ScanNet++ preparation requires Linux with EGL; "
                              "use --dataset to prepare an explicit subset elsewhere")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    os.environ.setdefault("GALLIUM_DRIVER", "llvmpipe")
    if os.environ["PYOPENGL_PLATFORM"] != "egl":
        raise ValidationError("set PYOPENGL_PLATFORM=egl for ScanNet++ preparation")
    renderer = None
    try:
        import pyrender
        from OpenGL.GL import GL_RENDERER, GL_VENDOR, GL_VERSION, glGetString
        renderer = pyrender.OffscreenRenderer(1, 1)
        return {name: glGetString(key).decode("utf-8", errors="replace")
                for name, key in (("vendor", GL_VENDOR), ("renderer", GL_RENDERER), ("version", GL_VERSION))}
    except Exception as error:
        raise ValidationError("ScanNet++ EGL renderer is unavailable; install '.[prepare,scannetpp]' "
                              f"and a working EGL implementation: {error}") from error
    finally:
        if renderer is not None:
            renderer.delete()


def prepare_benchmark(
    manifest: Path,
    roots: dict[str, Path],
    output: Path,
    *,
    embodiedscan: Path | None = None,
    datasets: list[str] | None = None,
    fingerprints: list[Path] | None = None,
    check_only: bool = False,
    workers: int = 4,
    cache: Path | None = None,
    scannet_jpeg_tools: Path | None = None,
    jpeg_archive: Path | None = None,
) -> dict[str, Any]:
    """Check all sources, provision the JPEG toolchain, and build one asset tree.

    Official source files are read-only. The destination must be new; failures
    during reconstruction leave no partially prepared benchmark at that path.
    """
    selected = set(RAW_DATASETS if datasets is None else datasets)
    if not selected or selected - set(RAW_DATASETS):
        raise ValidationError(f"choose datasets from {', '.join(RAW_DATASETS)}")
    if not 1 <= workers <= 16:
        raise ValidationError("preparation workers must be between 1 and 16")
    if not check_only and (output.exists() or output.is_symlink()):
        raise ValidationError(f"output already exists; choose a new directory: {output}")
    index = validate_manifest(manifest)
    available = {row["dataset"] for row in index.rows}
    if selected - available:
        raise ValidationError(f"selected datasets absent from the manifest: {sorted(selected - available)}; "
                              "use --dataset for an explicit subset")
    rows = tuple(row for row in index.rows if row["dataset"] in selected)
    expected = {request.image_ref for request in requests_from_rows(rows)}
    reference_paths = fingerprints or _bundled_fingerprints()
    references = _references(reference_paths, expected)
    asset_revision = "custom" if fingerprints else "scannetpp-depth-v2"
    unknown = roots.keys() - set(DATASET_ROOTS.values())
    if unknown:
        raise ValidationError(f"unknown dataset roots: {sorted(unknown)}")
    roots = {name: Path(path).expanduser().resolve() for name, path in roots.items()}
    problems = []
    for dataset in sorted(selected):
        name = DATASET_ROOTS[dataset]
        if name not in roots and os.environ.get(name):
            roots[name] = Path(os.environ[name]).expanduser().resolve()
        if name not in roots:
            problems.append(f"missing --{dataset} /path/to/{dataset} (or {name})")
        elif not roots[name].is_dir():
            problems.append(f"dataset directory does not exist: {roots[name]}")
    annotations = []
    if "matterport3d" in selected:
        if embodiedscan is None:
            problems.append("missing --embodiedscan /path/to/official-v1-annotations")
        else:
            embodiedscan = embodiedscan.expanduser().resolve()
            annotations = [embodiedscan / f"embodiedscan_infos_{split}.pkl" for split in ("train", "val", "test")]
            problems.extend(f"missing official EmbodiedScan annotation: {path}" for path in annotations if not path.is_file())
    if problems:
        report = {"status": "missing-inputs", "selected_datasets": sorted(selected),
                  "cases": len(rows), "frames": len(expected), "problem_count": len(problems),
                  "problems": problems, "missing_roots": sorted(DATASET_ROOTS[d] for d in selected
                                                                    if DATASET_ROOTS[d] not in roots)}
        if check_only:
            return report
        raise ValidationError("source check failed; no output was built: " + " | ".join(problems))
    if scannet_jpeg_tools is not None and "scannet" in selected:
        scannet_jpeg_tools = validate_jpeg_tools(scannet_jpeg_tools)
    # Resolve every selected frame before downloading/building tools or writing assets.
    options = dict(raw_sources=True, datasets=sorted(selected),
                   matterport_annotations=annotations, workers=workers, asset_revision=asset_revision)
    report = build_data(manifest, roots, output, check_only=True, **options)
    if "scannetpp" in selected:
        try:
            report["scannetpp_environment"] = _scannetpp_environment()
        except ValidationError as error:
            report["status"] = "missing-inputs"
            report["problems"].append(str(error))
            report["problem_count"] += 1
    report["fingerprint_frames"] = len(references)
    report["scannet_jpeg_backend"] = "ijg-9e-quality75" if "scannet" in selected else None
    report["jpeg_tools"] = (str(scannet_jpeg_tools) if scannet_jpeg_tools else
                            "automatically installed in cache when building") if "scannet" in selected else None
    if check_only:
        return report
    if report["status"] != "ready":
        raise ValidationError(f"source check found {report['problem_count']} problem(s); no output was built. "
                              + " | ".join(report["problems"][:5]) + "; use --check-only for the report")
    if "scannet" in selected:
        scannet_jpeg_tools = ensure_jpeg_tools(cache or default_cache(), directory=scannet_jpeg_tools,
                                             archive=jpeg_archive)
    # A single build writes the combined manifest, generation inputs and scorer
    # index atomically. No per-dataset output merging or user-side conversion.
    with tempfile.TemporaryDirectory(prefix="egogeneval-reference-") as temporary:
        combined = Path(temporary) / "fingerprints.jsonl"
        write_jsonl(combined, references)
        return build_data(manifest, roots, output, fingerprints=combined,
                          scannet_jpeg_tools=scannet_jpeg_tools, **options)
