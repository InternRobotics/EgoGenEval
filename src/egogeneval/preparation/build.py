"""Create the scorer's eval_frames contract from local, user-obtained data."""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import os
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from tqdm import tqdm

from ..data import validate_manifest
from ..errors import ValidationError
from ..io import read_json, read_jsonl, sha256_file, write_json, write_jsonl
from ..official.evaluation.eval_assets import (
    ASSET_FORMAT_VERSION,
    PREPROCESSING_CONTRACT_ID,
    _apply_intrinsics_stages,
    _materialize,
    resize_stages_for_dataset,
)
from ..official.scripts import direct_target_rgbd as direct
from .scannet import extract_frames
from .parallel import scene_jobs
from .raw_scannetpp import reconstruct_scene
from .fingerprints import verify_fingerprints
from .sources import DATASET_ROOTS, Request, Source, Sources, matrix, requests_from_rows


def _asset_path(root: Path, text: str) -> Path:
    part = PurePosixPath(text)
    if not text or part.is_absolute() or ".." in part.parts:
        raise ValidationError(f"unsafe eval asset path: {text!r}")
    return root / text


def verify_data(
    output: Path, *, reference_index: Path | None = None,
    fingerprints: Path | None = None,
) -> dict[str, Any]:
    """Verify bytes, geometry, manifest coverage, and optionally frozen references."""
    index = output / "eval_frames.jsonl" if output.is_dir() else output
    if fingerprints is None:
        bundled = index.parent / "reference_fingerprints.jsonl"
        if bundled.is_file():
            fingerprints = bundled
    report_path = index.parent / "build_report.json"
    if report_path.is_file():
        build_report = read_json(report_path)
        for name, digest in build_report.get("artifact_sha256", {}).items():
            path = _asset_path(index.parent, name)
            if not path.is_file() or sha256_file(path) != digest:
                raise ValidationError(f"missing or modified build artifact: {path}")
    records = read_jsonl(index)
    references = {}
    if reference_index is not None:
        if reference_index.is_dir():
            reference_index = reference_index / "eval_frames.jsonl"
        for row in read_jsonl(reference_index):
            ref = row["image_ref"]
            if ref in references:
                raise ValidationError(f"{reference_index}: duplicate image_ref {ref}")
            references[ref] = row
    seen, ids, frames = set(), set(), set()
    for row in records:
        ref, asset_id = row["image_ref"], row["asset_id"]
        frame = row["dataset"], row["scene_id"], int(row["loader_frame_idx"])
        if ref in seen or asset_id in ids or frame in frames:
            raise ValidationError(f"duplicate asset or frame identity: {ref}")
        seen.add(ref)
        ids.add(asset_id)
        frames.add(frame)
        image = _asset_path(index.parent, row["image_path"])
        depth = _asset_path(index.parent, row["depth_path"])
        for file, expected in [
            (image, row["rgb_sha256"]),
            (depth, row["depth_sha256"]),
        ]:
            if not file.is_file() or sha256_file(file) != expected:
                raise ValidationError(f"missing or modified asset: {file}")
        matrix(row["extrinsics_c2w"], 4, f"{ref}: pose")
        matrix(row["intrinsics"], 3, f"{ref}: K")
        try:
            loaded = _materialize(
                {**row, "_image_path": image, "_depth_path": depth}, 0
            )
        except (ValueError, OSError) as exc:
            raise ValidationError(f"{ref}: {exc}") from exc
        if (
            not np.isfinite(loaded.depth).all()
            or loaded.depth.shape != loaded.image.shape[:2]
        ):
            raise ValidationError(f"{ref}: nonfinite depth or RGB/depth shape mismatch")
        if reference_index is not None:
            if ref not in references:
                raise ValidationError(f"frozen reference has no entry for {ref}")
            expected = references[ref]
            # Normalize historical v0.1 intrinsics with the same version-scoped
            # compatibility rule used by the scorer; do not silently accept drift.
            from ..official.evaluation.eval_assets import _row_resize_contract

            _, stages = _row_resize_contract(expected)
            expected_k = _apply_intrinsics_stages(
                np.asarray(expected["intrinsics"]).reshape(3, 3), stages
            )
            keys = (
                "rgb_sha256",
                "depth_sha256",
                "image_height",
                "image_width",
                "depth_height",
                "depth_width",
                "dataset",
                "scene_id",
            )
            different = [key for key in keys if row[key] != expected.get(key)]
            if not np.array_equal(loaded.intrinsics, expected_k):
                different.append("intrinsics")
            if not np.array_equal(
                loaded.extrinsics,
                np.asarray(expected["extrinsics_c2w"]).reshape(4, 4),
            ):
                different.append("extrinsics_c2w")
            if (
                "loader_frame_idx" in expected
                and int(expected["loader_frame_idx"]) != row["loader_frame_idx"]
            ):
                different.append("loader_frame_idx")
            if different:
                raise ValidationError(
                    f"{ref}: frozen reference mismatch: {', '.join(different)}"
                )
    if not records:
        raise ValidationError(f"empty eval-frames index: {index}")
    manifest = index.parent / "manifest.jsonl"
    if manifest.is_file():
        expected_requests = {
            r.image_ref: r for r in requests_from_rows(validate_manifest(manifest).rows)
        }
        if seen != expected_requests.keys():
            raise ValidationError(
                "eval frames do not exactly cover the local canonical manifest"
            )
        for row in records:
            request = expected_requests[row["image_ref"]]
            if (request.dataset, request.scene_id, request.frame_id) != (
                row["dataset"],
                row["scene_id"],
                int(row["loader_frame_idx"]),
            ):
                raise ValidationError(
                    f"frame identity differs from manifest: {request.image_ref}"
                )
    if fingerprints is not None:
        verify_fingerprints(records, fingerprints)
    return {
        "status": "verified",
        "frames": len(records),
        "reference_checked": reference_index is not None or fingerprints is not None,
        "fingerprints_checked": fingerprints is not None,
    }


def build_data(
    manifest: Path,
    roots: dict[str, Path],
    output: Path,
    *,
    annotations: list[Path] | None = None,
    check_only: bool = False,
    mode: str = "copy",
    reference_index: Path | None = None,
    datasets: list[str] | None = None,
    raw_sources: bool = False,
    fingerprints: Path | None = None,
    scannet_jpeg_tools: Path | None = None,
    matterport_annotations: list[Path] | None = None,
    workers: int = 4,
    asset_revision: str = "custom",
) -> dict[str, Any]:
    """Rebuild frozen evaluation assets from user-obtained sources."""
    if mode not in {"copy", "symlink"}:
        raise ValidationError(f"unsupported image materialization mode: {mode}")
    if not 1 <= workers <= 16:
        raise ValidationError("preparation workers must be between 1 and 16")
    if not asset_revision or not isinstance(asset_revision, str):
        raise ValidationError("asset revision must be a nonempty string")
    if scannet_jpeg_tools is not None:
        scannet_jpeg_tools = scannet_jpeg_tools.expanduser().resolve()
        for binary in ("cjpeg", "djpeg"):
            if not (scannet_jpeg_tools / binary).is_file():
                raise ValidationError(f"missing IJG 9e tool: {scannet_jpeg_tools / binary}")
    fingerprint_digest = None
    if fingerprints is not None:
        fingerprints = fingerprints.expanduser().resolve()
        if not fingerprints.is_file():
            raise ValidationError(f"frozen fingerprints file not found: {fingerprints}")
        fingerprint_digest = sha256_file(fingerprints)
    index = validate_manifest(manifest)
    manifest_bytes = manifest.read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != index.sha256:
        raise ValidationError(f"manifest changed while being read: {manifest}")
    rows = index.rows
    available = {row["dataset"] for row in rows}
    selected = set(datasets) if datasets is not None else available
    if not selected or selected - available:
        raise ValidationError(
            f"select datasets present in the manifest: {sorted(available)}; got {sorted(selected)}"
        )
    if raw_sources and annotations:
        raise ValidationError("with --raw-sources, omit --embodiedscan-info; use --matterport-embodiedscan-info for Matterport3D only")
    if matterport_annotations and "matterport3d" not in selected:
        raise ValidationError("--matterport-embodiedscan-info requires --dataset matterport3d")
    if matterport_annotations and annotations:
        raise ValidationError("use either --matterport-embodiedscan-info or --embodiedscan-info")
    if selected != available:
        # Preserve each selected canonical JSONL line byte for byte, in order.
        lines = [line for line in manifest_bytes.splitlines(keepends=True) if line.strip()]
        manifest_bytes = b"".join(
            line for line, row in zip(lines, rows) if row["dataset"] in selected
        )
        rows = tuple(row for row in rows if row["dataset"] in selected)
    requests = requests_from_rows(rows)
    allowed_roots = {
        *DATASET_ROOTS.values(),
        "SCANNET_METADATA",
        "MATTERPORT3D_METADATA",
    }
    unknown = set(roots) - allowed_roots
    if unknown:
        raise ValidationError(f"unknown dataset roots: {sorted(unknown)}")
    roots = {name: Path(value).expanduser().resolve() for name, value in roots.items()}
    for name in allowed_roots - roots.keys():
        if os.environ.get(name):
            roots[name] = Path(os.environ[name]).expanduser().resolve()
    reader = Sources(roots, requests, annotations or [], raw_sources=raw_sources,
                     matterport_annotations=matterport_annotations)
    reader.preload_native_scannet(workers)
    problems = list(reader.errors)
    by_dataset_problems: dict[str, list[str]] = defaultdict(list)
    sources: dict[str, Source] = {}
    extract: dict[Path, list[Request]] = defaultdict(list)
    scannetpp_scenes: dict[str, list[Request]] = defaultdict(list)
    for request in requests:
        try:
            source = reader.resolve(request)
            inputs = ((source.metadata, *source.metadata_dependencies) if source.raw_scannetpp
                      else (source.rgb, source.depth))
            missing = [p for p in inputs if p is None or not p.is_file()]
            if source.raw_scannetpp:
                if missing:
                    raise ValidationError("missing original ScanNet++ files: " + ", ".join(map(str, missing)))
                scannetpp_scenes[request.scene_id].append(request)
            if (
                (missing or raw_sources)
                and request.dataset == "scannet"
                and source.sensor is not None
                and source.sensor.is_file()
            ):
                if not source.rgb.stem.isdigit():
                    raise ValidationError(
                        f"cannot derive .sens frame number from {source.rgb}"
                    )
                extract[source.sensor].append(request)
            elif missing:
                raise ValidationError(
                    "missing source files: " + ", ".join(map(str, missing))
                )
            sources[request.image_ref] = source
        except (
            ValidationError,
            OSError,
            KeyError,
            ValueError,
            IndexError,
            TypeError,
        ) as exc:
            problems.append(f"{request.dataset}/{request.scene_id}: {exc}")
            by_dataset_problems[request.dataset].append(str(exc))
    report: dict[str, Any] = {
        "status": "missing-inputs" if problems else "ready",
        "cases": len(rows),
        "frames": len(requests),
        "cases_by_dataset": dict(Counter(r["dataset"] for r in rows)),
        "frames_by_dataset": dict(Counter(r.dataset for r in requests)),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "source_manifest_sha256": index.sha256,
        "source_cases": len(index.rows),
        "selected_datasets": sorted(selected),
        "is_subset": selected != available,
        "raw_sources": raw_sources,
        "asset_revision": asset_revision,
        "workers": workers,
        "matterport_calibration": "embodiedscan-v1" if matterport_annotations else "native" if raw_sources else "auto",
        "scannet_jpeg_backend": "ijg-9e-quality75" if scannet_jpeg_tools else "installed-imageio",
        "fingerprints_sha256": fingerprint_digest,
        "scannet_frames_to_extract": sum(len(v) for v in extract.values()),
        "scannetpp_frames_to_reconstruct": sum(len(v) for v in scannetpp_scenes.values()),
        "problem_count": len(problems),
        "problems": problems[:100],
        "missing_roots": sorted(
            {DATASET_ROOTS[r.dataset] for r in requests} - roots.keys()
        ),
        "problems_by_dataset": {
            name: {
                "affected_frames": len(errors),
                "examples": list(dict.fromkeys(errors))[:5],
            }
            for name, errors in sorted(by_dataset_problems.items())
        },
    }
    if check_only:
        return report
    if problems:
        raise ValidationError(
            f"source check found {len(problems)} problem(s); no output was built. "
            + " | ".join(problems[:5])
            + "; use --check-only for the report"
        )
    output = output.expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise ValidationError(
            f"output already exists; choose a new directory: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        def extract_scene(job):
            number, (sensor, pending) = job
            directory = staging / "extracted" / str(number)
            extract_frames(
                sensor, {int(sources[r.image_ref].rgb.stem) for r in pending}, directory,
                jpeg_tools=scannet_jpeg_tools,
            )
            return pending, directory

        for pending, directory in scene_jobs(extract_scene, enumerate(sorted(extract.items())), workers):
            for request in pending:
                source = sources[request.image_ref]
                # Retain existing upstream encoded RGB even if only depth needed extraction.
                if raw_sources or not source.rgb.is_file():
                    source.rgb = directory / source.rgb.name
                if raw_sources or not source.depth.is_file():
                    source.depth = directory / source.depth.name
        scannetpp_jobs = []
        for scene_id, pending in sorted(scannetpp_scenes.items()):
            scene = sources[pending[0].image_ref].raw_scannetpp
            assert scene is not None
            directory = staging / "extracted" / "scannetpp" / scene_id
            scannetpp_jobs.append((scene, [Path(r.relative).stem for r in pending], directory))
        scannetpp_results = scene_jobs(reconstruct_scene, scannetpp_jobs, workers, processes=True)
        for (scene_id, pending), (rebuilt, renderer_info) in zip(sorted(scannetpp_scenes.items()), scannetpp_results):
            report.setdefault("scannetpp_renderers", {})[scene_id] = renderer_info
            for request in pending:
                source = sources[request.image_ref]
                source.rgb, source.depth = rebuilt[Path(request.relative).stem]
        records, by_ref = [], {}
        dependencies: dict[str, str] = {}
        for request in tqdm(
            requests,
            desc="Building local assets",
            unit="frame",
            disable=not sys.stderr.isatty(),
        ):
            source = sources[request.image_ref]
            assert source.depth is not None
            asset_id = hashlib.sha256(request.image_ref.encode()).hexdigest()[:24]
            image_path = Path("images") / f"{asset_id}{source.rgb.suffix.lower()}"
            destination = staging / image_path
            destination.parent.mkdir(exist_ok=True)
            # Symlinks into staging would break after the atomic directory rename.
            extracted_rgb = source.rgb.is_relative_to(staging)
            if mode == "symlink" and not extracted_rgb:
                destination.symlink_to(source.rgb)
            else:
                shutil.copyfile(source.rgb, destination)
            # Direct helpers preserve the evaluator's exact depth operations.
            # For extracted ScanNet, RGB and depth may now live in different roots.
            if request.dataset == "scannet":
                rgb = direct._half(direct._read_rgb(str(source.rgb)))
                depth = direct._read_depth_png(source.depth, 1000.0)
                depth[~np.isfinite(depth)] = 0.0
                if depth.shape != rgb.shape[:2]:
                    import cv2

                    depth = cv2.resize(
                        depth,
                        (rgb.shape[1], rgb.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                processed_rgb = direct._half(rgb)
                processed_depth = direct._half(direct._clip_98(depth)).astype(
                    np.float32
                )
            else:
                loader = {
                    "matterport3d": direct._matterport,
                    "scannetpp": direct._scannetpp,
                    "hypersim": direct._hypersim,
                }[request.dataset]
                rgbd = loader(str(source.rgb))
                processed_rgb, processed_depth = rgbd.image, rgbd.depth
            stages = resize_stages_for_dataset(request.dataset)
            intrinsic = _apply_intrinsics_stages(source.intrinsic, stages)
            depth_path = Path("depth") / f"{asset_id}.npy"
            (staging / depth_path).parent.mkdir(exist_ok=True)
            np.save(staging / depth_path, processed_depth, allow_pickle=False)
            row = {
                "asset_id": asset_id,
                "dataset": request.dataset,
                "scene_id": request.scene_id,
                "image_ref": request.image_ref,
                "loader_frame_idx": request.frame_id,
                "image_path": image_path.as_posix(),
                "depth_path": depth_path.as_posix(),
                "rgb_sha256": sha256_file(destination),
                "depth_sha256": sha256_file(staging / depth_path),
                "image_height": int(processed_rgb.shape[0]),
                "image_width": int(processed_rgb.shape[1]),
                "depth_height": int(processed_depth.shape[0]),
                "depth_width": int(processed_depth.shape[1]),
                "extrinsics_c2w": source.pose.tolist(),
                "intrinsics": intrinsic.tolist(),
                "asset_format_version": ASSET_FORMAT_VERSION,
                "asset_revision": asset_revision,
                "preprocessing": PREPROCESSING_CONTRACT_ID,
                "resize_stages": list(stages),
                "calibration_source": source.calibration_source,
            }
            records.append(row)
            by_ref[request.image_ref] = row
            for path in (source.metadata, *source.metadata_dependencies):
                if str(path) not in dependencies:
                    dependencies[str(path)] = sha256_file(path)
        write_jsonl(staging / "eval_frames.jsonl", records)
        # Full builds preserve the original hash; subsets preserve selected lines.
        (staging / "manifest.jsonl").write_bytes(manifest_bytes)
        prepared, model_inputs = [], []
        for row in rows:
            local = copy.deepcopy(row)
            for field in ("input_images", "target_images"):
                for item in local[field]:
                    item["image_path"] = by_ref[item["image_path"]]["image_path"]
            prepared.append(local)
            # Keep target images and numeric GT metadata out of the default generation feed.
            model_inputs.append(
                {
                    k: v
                    for k, v in local.items()
                    if k not in {"target_images", "pose_metadata"}
                }
            )
        write_jsonl(staging / "prepared_manifest.jsonl", prepared)
        write_jsonl(staging / "generation_inputs.jsonl", model_inputs)
        if fingerprints is not None:
            shutil.copyfile(fingerprints, staging / "reference_fingerprints.jsonl")
            if sha256_file(staging / "reference_fingerprints.jsonl") != fingerprint_digest:
                raise ValidationError("frozen fingerprints changed during the build")
        report["verification"] = verify_data(staging, reference_index=reference_index)
        report.update(
            {
                "status": "built",
                "output": str(output),
                "preprocessing": PREPROCESSING_CONTRACT_ID,
                "calibration_sources": dict(
                    Counter(r["calibration_source"] for r in records)
                ),
                "metadata_sha256": dependencies,
                "artifact_sha256": {
                    name: sha256_file(staging / name)
                    for name in (
                        "manifest.jsonl",
                        "prepared_manifest.jsonl",
                        "generation_inputs.jsonl",
                        "eval_frames.jsonl",
                    )
                },
                "source_roots": {k: str(v) for k, v in roots.items()},
                "versions": {
                    name: importlib.metadata.version(name)
                    for name in (("numpy", "Pillow", "imageio", "h5py") +
                                 (("opencv-python-headless", "pyrender", "trimesh", "PyOpenGL")
                                  if scannetpp_scenes else ()))
                },
                "reference_assets_verified": report["verification"]["reference_checked"],
            }
        )
        if fingerprints is not None:
            report["artifact_sha256"]["reference_fingerprints.jsonl"] = fingerprint_digest
        write_json(staging / "build_report.json", report)
        shutil.rmtree(staging / "extracted", ignore_errors=True)
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return report
