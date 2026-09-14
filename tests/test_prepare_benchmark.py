"""The unified command produces one generation/scoring contract from raw sources."""

import json
from pathlib import Path

import pytest

from egogeneval.cli import main
from egogeneval.data import validate_manifest
from egogeneval.errors import ValidationError
from egogeneval.io import read_jsonl, sha256_file, write_json, write_jsonl
from egogeneval.preparation.build import verify_data
from egogeneval.preparation.jpeg_tools import SHA256, ensure_jpeg_tools
from egogeneval.preparation.prepare import prepare_benchmark
from egogeneval.scoring.pipeline import _eval_frame_lookup, _resolve_case_images
from test_local_data import (
    _raw_fixture_fingerprints,
    dump_pickle,
    raw_source_set as _raw_source_set,
    source_set as _source_set,
)
from test_matterport_embodiedscan import annotations

source_set = _source_set
raw_source_set = _raw_source_set


def _arguments(raw_source_set, tmp_path):
    manifest, roots, _, _, house, *_ = raw_source_set
    reference = _raw_fixture_fingerprints(raw_source_set, tmp_path)
    embodied = tmp_path / "official-annotations"
    annotations(house, embodied / "embodiedscan_infos_train.pkl")
    for split in ("val", "test"):
        dump_pickle(embodied / f"embodiedscan_infos_{split}.pkl", {"data_list": []})
    roots = {name: value for name, value in roots.items() if name.endswith("_ROOT")}
    return manifest, roots, embodied, reference


def test_one_command_builds_combined_inputs_and_scoring_assets(raw_source_set, tmp_path, monkeypatch, capsys):
    manifest, roots, embodied, reference = _arguments(raw_source_set, tmp_path)
    output = tmp_path / "unified"
    # Fixtures use imageio's synthetic export; production tool provisioning is
    # covered separately. This test validates the raw-source/contract integration.
    provisioning = []
    monkeypatch.setattr("egogeneval.preparation.prepare.ensure_jpeg_tools",
                        lambda *a, **kw: provisioning.append(kw))
    arguments = ["prepare-benchmark", "--manifest", str(manifest),
                 "--embodiedscan", str(embodied), "--fingerprints", str(reference),
                 "--output", str(output), "--workers", "2"]
    for dataset in ("hypersim", "matterport3d", "scannet"):
        arguments += ["--dataset", dataset, f"--{dataset}", str(roots[dataset.upper() + "_ROOT"])]
    before = {path: sha256_file(path) for path in roots["SCANNET_ROOT"].rglob("*") if path.is_file()}
    assert main([*arguments, "--check-only"]) == 0
    checked = json.loads(capsys.readouterr().out)
    assert checked["status"] == "ready" and checked["frames"] == 6
    assert not output.exists() and not provisioning
    assert main(arguments) == 0
    built = json.loads(capsys.readouterr().out)
    assert built["reference_assets_verified"] and built["frames"] == 6 and built["cases"] == 3
    assert len(provisioning) == 1
    assert verify_data(output)["fingerprints_checked"]
    inputs = read_jsonl(output / "generation_inputs.jsonl")
    assert len(inputs) == 3
    for case in inputs:
        assert "target_images" not in case and "pose_metadata" not in case
        assert all((output / image["image_path"]).is_file() for image in case["input_images"])
    lookup = _eval_frame_lookup(output / "eval_frames.jsonl")
    for case in validate_manifest(output / "manifest.jsonl").rows:
        resolved = _resolve_case_images(case, prepared_data=None, eval_frames=lookup)
        assert all(Path(image["image_path"]).is_file() for image in resolved["target_images"])
    assert before == {path: sha256_file(path) for path in before}
    assert main(arguments) == 2  # No accidental replacement of a prior benchmark.
    assert verify_data(output)["frames"] == 6


def test_full_default_reports_missing_fourth_dataset_before_tools(raw_source_set, tmp_path, monkeypatch):
    manifest, roots, embodied, reference = _arguments(raw_source_set, tmp_path)
    # Extend references with the fixture PP rows; no image build is attempted.
    from egogeneval.preparation.build import build_data
    from egogeneval.preparation.fingerprints import fingerprint_row
    pp = tmp_path / "processed-pp-reference"
    build_data(manifest, roots, pp, datasets=["scannetpp"])
    write_jsonl(reference, [*read_jsonl(reference), *[
        fingerprint_row(row) for row in read_jsonl(pp / "eval_frames.jsonl")]])
    monkeypatch.delenv("SCANNETPP_ROOT", raising=False)
    roots.pop("SCANNETPP_ROOT")
    monkeypatch.setattr("egogeneval.preparation.prepare.ensure_jpeg_tools", lambda *a, **kw: pytest.fail("must preflight first"))
    output = tmp_path / "missing"
    report = prepare_benchmark(manifest, roots, output, embodiedscan=embodied,
                               fingerprints=[reference], check_only=True)
    assert report["selected_datasets"] == ["hypersim", "matterport3d", "scannet", "scannetpp"]
    assert report["status"] == "missing-inputs" and report["missing_roots"] == ["SCANNETPP_ROOT"]
    assert not output.exists()


def test_missing_source_preflight_does_not_provision_or_write(raw_source_set, tmp_path, monkeypatch):
    manifest, roots, embodied, reference = _arguments(raw_source_set, tmp_path)
    next(roots["SCANNET_ROOT"].rglob("*.sens")).unlink()
    monkeypatch.setattr("egogeneval.preparation.prepare.ensure_jpeg_tools", lambda *a, **kw: pytest.fail("must preflight first"))
    output = tmp_path / "missing"
    options = dict(embodiedscan=embodied, fingerprints=[reference], datasets=["hypersim", "matterport3d", "scannet"])
    report = prepare_benchmark(manifest, roots, output, check_only=True, **options)
    assert report["status"] == "missing-inputs" and not output.exists()
    with pytest.raises(ValidationError, match="no output was built"):
        prepare_benchmark(manifest, roots, output, **options)
    assert not output.exists()


def test_cached_jpeg_tools_are_version_and_digest_checked(tmp_path):
    prefix = tmp_path / "ijg-9e"
    (prefix / "bin").mkdir(parents=True)
    for name in ("cjpeg", "djpeg"):
        binary = prefix / "bin" / name
        binary.write_text(f'#!/bin/sh\necho "Independent JPEG Group\'s {name.upper()}, version 9e" >&2\ncat\n')
        binary.chmod(0o755)
    write_json(prefix / "build_identity.json", {
        "source_sha256": SHA256,
        "binary_sha256": {name: sha256_file(prefix / "bin" / name) for name in ("cjpeg", "djpeg")},
    })
    assert ensure_jpeg_tools(tmp_path) == prefix / "bin"
    (prefix / "bin" / "cjpeg").write_text((prefix / "bin" / "cjpeg").read_text() + "# modified\n")
    with pytest.raises(ValidationError, match="modified cached IJG"):
        ensure_jpeg_tools(tmp_path)


def test_wrong_offline_jpeg_archive_leaves_no_install(tmp_path):
    archive = tmp_path / "jpeg.tar.gz"
    archive.write_bytes(b"untrusted source archive")
    cache = tmp_path / "cache"
    with pytest.raises(ValidationError, match="SHA-256 mismatch"):
        ensure_jpeg_tools(cache, archive=archive)
    assert not (cache / "ijg-9e").exists()


def test_four_datasets_share_one_generation_and_scoring_bundle(raw_source_set, tmp_path, monkeypatch):
    """Mock only GPU rendering/tool setup; exercise four-way raw-source joining."""
    import importlib.metadata
    import numpy as np
    from PIL import Image
    from egogeneval.preparation.build import build_data
    from egogeneval.preparation.fingerprints import fingerprint_row
    from egogeneval.preparation.raw_scannetpp import RawScene
    from test_local_data import rgbd
    from test_raw_scannetpp import camera_fixture

    manifest, roots, embodied, reference = _arguments(raw_source_set, tmp_path)
    pp_root = roots["SCANNETPP_ROOT"]
    path, metadata, _ = camera_fixture(pp_root, scene="ppscene", filename="DSC00000.JPG")
    second = {**metadata["test_frames"][0], "file_path": "DSC00002.JPG"}
    metadata["test_frames"].append(second)
    path.write_text(json.dumps(metadata))
    for stem in ("DSC00000", "DSC00002"):
        photo = pp_root / f"data/ppscene/dslr/resized_images/{stem}.JPG"
        mask = pp_root / f"data/ppscene/dslr/resized_anon_masks/{stem}.png"
        photo.parent.mkdir(parents=True, exist_ok=True)
        mask.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (22, 18), (10, 20, 30)).save(photo)
        Image.fromarray(np.full((18, 22), 255, dtype=np.uint8)).save(mask)
    mesh = pp_root / "data/ppscene/scans/mesh_aligned_0.05.ply"
    mesh.parent.mkdir(parents=True)
    mesh.write_bytes(b"synthetic mesh placeholder; rendering is mocked")

    def render_fixture(scene, stems, destination):
        result = {}
        for stem in stems:
            assert all(path.is_file() for path in scene.files(stem))
            image, depth = destination / "images" / f"{stem}.jpg", destination / "depth" / f"{stem}.png"
            rgbd(image, depth, value=int(stem[-1]))
            result[stem] = (image, depth)
        scene.renderer_info = {"renderer": "synthetic-test-renderer"}
        return result

    version = importlib.metadata.version
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "synthetic"
                        if name in {"pyrender", "trimesh", "PyOpenGL"} else version(name))
    monkeypatch.setattr(RawScene, "reconstruct", render_fixture)
    monkeypatch.setattr("egogeneval.preparation.prepare._scannetpp_environment", lambda: {"renderer": "synthetic-test-renderer"})
    monkeypatch.setattr("egogeneval.preparation.prepare.ensure_jpeg_tools", lambda *a, **kw: None)
    all_reference = tmp_path / "synthetic-four-way-reference"
    build_data(manifest, roots, all_reference, raw_sources=True,
               matterport_annotations=[embodied / f"embodiedscan_infos_{split}.pkl" for split in ("train", "val", "test")])
    write_jsonl(reference, [fingerprint_row(row) for row in read_jsonl(all_reference / "eval_frames.jsonl")])
    before = {path: sha256_file(path) for path in pp_root.rglob("*") if path.is_file()}
    output = tmp_path / "unified-all-four"
    report = prepare_benchmark(manifest, roots, output, embodiedscan=embodied, fingerprints=[reference])
    assert report["cases"] == 4 and report["frames"] == 8 and not report["is_subset"]
    assert report["reference_assets_verified"] and report["asset_revision"] == "custom"
    assert set(report["selected_datasets"]) == {"hypersim", "matterport3d", "scannet", "scannetpp"}
    assert {row["asset_revision"] for row in read_jsonl(output / "eval_frames.jsonl")} == {"custom"}
    lookup = _eval_frame_lookup(output / "eval_frames.jsonl")
    for row in validate_manifest(output / "manifest.jsonl").rows:
        resolved = _resolve_case_images(row, prepared_data=None, eval_frames=lookup)
        assert all(Path(image["image_path"]).is_file() for image in resolved["target_images"])
    assert len(read_jsonl(output / "generation_inputs.jsonl")) == 4
    assert verify_data(output)["frames"] == 8
    assert before == {path: sha256_file(path) for path in before}
