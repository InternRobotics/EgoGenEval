"""Official-layout ZIP and raw-input integration checks using synthetic assets."""

import io
import re
import zipfile
from concurrent.futures import CancelledError
from pathlib import Path
from threading import Barrier, Lock
from urllib.error import URLError

import numpy as np
import pytest

from egogeneval.cli import main
from egogeneval.errors import ValidationError
from egogeneval.io import read_jsonl, sha256_file
from egogeneval.preparation import acquire
from egogeneval.preparation.build import build_data, verify_data
from egogeneval.preparation.scannet import extract_frames
from test_local_data import sensor_fixture, source_set as _source_set

source_set = _source_set


def test_frozen_download_plan_needs_only_original_modalities():
    plan = acquire.source_plan(Path("data/manifests/egogeneval_v0.1.jsonl"))
    assert (plan["cases"], plan["frames"]) == (397, 735)
    assert plan["excluded_datasets"] == ["scannetpp"]
    assert {k: v["scene_count"] for k, v in plan["datasets"].items()} == {
        "hypersim": 64, "matterport3d": 16, "scannet": 125,
    }
    assert all(r["path"].endswith((".sens", ".txt")) for r in plan["datasets"]["scannet"]["files"])
    assert "matterport_camera_poses" in plan["datasets"]["matterport3d"]["archive_types"]
    spp = acquire.source_plan(Path("data/manifests/egogeneval_v0.1.jsonl"), ["scannetpp"])["datasets"]["scannetpp"]
    assert spp["scene_count"] == 554
    assert spp["scene_aliases"] == {"7104910700": "7104910700000"}
    assert "7104910700000" in spp["scene_ids"] and "7104910700" not in spp["scene_ids"]
    assert len([f for f in spp["files"] if f["path"].endswith(".JPG")]) == 2930


def hypersim_zip(source_set, tmp_path):
    manifest, roots, _ = source_set
    entries = acquire.source_plan(manifest, ["hypersim"])["datasets"]["hypersim"]["files"]
    archive_path = tmp_path / "official-scene.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for row in entries:
            name = row["path"].removeprefix("evermotion_dataset/scenes/")
            archive.writestr(name, (roots["HYPERSIM_ROOT"] / row["path"]).read_bytes())
        archive.writestr("../outside.txt", b"not selected")
        archive.writestr("unselected/image.jpg", b"not selected")
    return manifest, roots, archive_path, entries


def test_official_zip_to_raw_builder_and_verifier(source_set, tmp_path):
    manifest, roots, archive, entries = hypersim_zip(source_set, tmp_path)
    fresh = tmp_path / "downloaded"
    report = acquire.unpack_sources(manifest, "hypersim", archive, fresh)
    assert report["status"] == "unpacked" and len(report["files"]) == len(entries)
    assert not (tmp_path / "outside.txt").exists()
    assert not (fresh / "unselected").exists()
    baseline = tmp_path / "baseline"
    build_data(manifest, roots, baseline, datasets=["hypersim"])
    output = tmp_path / "rebuilt"
    build_data(manifest, {"HYPERSIM_ROOT": fresh}, output, datasets=["hypersim"],
               raw_sources=True, reference_index=baseline / "eval_frames.jsonl")
    assert verify_data(output)["frames"] == len(read_jsonl(baseline / "eval_frames.jsonl"))
    # Resume validates existing files instead of silently replacing changed ones.
    acquire.unpack_sources(manifest, "hypersim", archive, fresh)
    (fresh / entries[0]["path"]).write_bytes(b"changed")
    with pytest.raises(ValidationError, match="differs"):
        acquire.unpack_sources(manifest, "hypersim", archive, fresh)


def test_raw_scannet_ignores_existing_extracted_frames(source_set, tmp_path):
    manifest, roots, _ = source_set
    sensor = roots["SCANNET_ROOT"] / "scans/scene0000_00/scene0000_00.sens"
    sensor_fixture(sensor)
    axis = np.eye(4, dtype=np.float32)
    axis[0, 3] = 7
    sensor.with_suffix(".txt").write_text("axisAlignment = " + " ".join(map(str, axis.flat)))
    expected = tmp_path / "expected"
    extract_frames(sensor, {0, 10, 20}, expected)
    output = tmp_path / "raw-scannet"
    report = build_data(manifest, roots, output, datasets=["scannet"], raw_sources=True)
    rows = read_jsonl(output / "eval_frames.jsonl")
    assert report["scannet_frames_to_extract"] == len(rows)
    assert report["calibration_sources"] == {"scannet-native-sens": len(rows)}
    for row in rows:
        name = Path(row["image_ref"]).name
        assert row["rgb_sha256"] == sha256_file(expected / name)
        np.testing.assert_array_equal(row["extrinsics_c2w"], axis)
        np.testing.assert_array_equal(row["intrinsics"], [[0.25, 0, 0], [0, 0.25, 0], [0, 0, 1]])
    with pytest.raises(ValidationError, match="source check found"):
        build_data(manifest, roots, tmp_path / "unsupported", raw_sources=True)


def test_raw_matterport_matches_legacy_without_pickle(source_set, tmp_path):
    manifest, roots, _ = source_set
    baseline = tmp_path / "legacy-mp"
    build_data(manifest, roots, baseline, datasets=["matterport3d"])
    for row in read_jsonl(baseline / "eval_frames.jsonl"):
        relative = row["image_ref"].split("}/", 1)[1]
        rgb = roots["MATTERPORT3D_ROOT"] / relative
        match = re.fullmatch(r"(.+)_i(\d+)_(\d+)\.jpg", rgb.name)
        panorama, camera, yaw = match.groups()
        pose_dir = rgb.parent.parent / "matterport_camera_poses"
        k_dir = rgb.parent.parent / "matterport_camera_intrinsics"
        pose_dir.mkdir(exist_ok=True)
        k_dir.mkdir(exist_ok=True)
        np.savetxt(pose_dir / f"{panorama}_pose_{camera}_{yaw}.txt", row["extrinsics_c2w"])
        # Fixture uses source K [[15,0,10],[0,17,8],[0,0,1]].
        (k_dir / f"{panorama}_intrinsics_{camera}.txt").write_text("22 18 15 17 10 8 0 0 0 0 0")
    archive = tmp_path / "matterport-originals.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        for path in roots["MATTERPORT3D_ROOT"].rglob("*"):
            if path.is_file():
                name = path.relative_to(roots["MATTERPORT3D_ROOT"]).as_posix().removeprefix("scans/")
                # Both native camera filename spellings are accepted by the
                # download plan, extractor and camera reader.
                name = name.replace("_pose_", "_pose").replace("_intrinsics_", "_intrinsics")
                handle.write(path, name)
    fresh = tmp_path / "original-mp"
    unpacked = acquire.unpack_sources(manifest, "matterport3d", archive, fresh)
    assert unpacked["status"] == "unpacked" and not unpacked["missing"]
    build_data(manifest, {"MATTERPORT3D_ROOT": fresh}, tmp_path / "native-mp", datasets=["matterport3d"],
               raw_sources=True, reference_index=baseline / "eval_frames.jsonl")


def test_partial_zip_reader_and_download_contract(source_set, tmp_path, monkeypatch):
    manifest, _, archive, entries = hypersim_zip(source_set, tmp_path)
    content = archive.read_bytes()
    requested = []

    class Response(io.BytesIO):
        status = 206

    def get(request, timeout):
        value = request.get_header("Range")
        if value.startswith("bytes=-"):
            start, end = len(content) - int(value[7:]), len(content) - 1
        else:
            start, end = map(int, value.removeprefix("bytes=").split("-"))
        requested.append((start, end))
        response = Response(content[start:end + 1])
        response.headers = {"Content-Range": f"bytes {start}-{end}/{len(content)}"}
        return response

    monkeypatch.setattr(acquire, "urlopen", get)
    output = tmp_path / "fresh"
    report = acquire.download_hypersim(manifest, output, workers=1)
    assert report["files"] == len(entries)
    assert len(requested) > 1
    assert main(["build-data", "--manifest", str(manifest), "--dataset", "hypersim",
                 "--raw-sources", "--root", f"HYPERSIM_ROOT={output}",
                 "--output", str(tmp_path / "built")]) == 0


def test_partial_reader_refuses_ignored_range(monkeypatch):
    class Response(io.BytesIO):
        status, headers = 200, {}

    monkeypatch.setattr(acquire, "urlopen", lambda *args, **kwargs: Response(b"ignored"))
    with pytest.raises(ValidationError, match="does not support"):
        acquire._HTTPRangeFile("https://example.invalid/scene.zip")


@pytest.mark.parametrize("dataset", ["hypersim", "matterport3d"])
def test_unrelated_zip_cannot_verify_preexisting_source_files(source_set, tmp_path, dataset):
    manifest, _, _ = source_set
    entries = acquire.source_plan(manifest, [dataset])["datasets"][dataset]["files"]
    output = tmp_path / "preexisting-sources"
    for row in entries:
        path = output / row["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"not official data")
    archive = tmp_path / "unrelated.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("unrelated.txt", b"unrelated")

    report = acquire.unpack_sources(manifest, dataset, archive, output)

    assert report["status"] == "missing-inputs"
    assert report["files"] == []
    assert set(report["missing"]) == {row["path"] for row in entries}
    # Detection must not silently delete or replace the user's existing files.
    assert all((output / row["path"]).read_bytes() == b"not official data" for row in entries)


def test_truncated_zip_reports_cli_error_without_traceback(source_set, tmp_path, capsys):
    manifest, _, _ = source_set
    archive = tmp_path / "truncated.zip"
    archive.write_bytes(b"PK")
    output = tmp_path / "sources"

    result = main([
        "unpack-sources", "--manifest", str(manifest), "--dataset", "hypersim",
        "--archives", str(archive), "--output", str(output),
    ])

    captured = capsys.readouterr()
    assert result == 2
    assert "error:" in captured.err.lower()
    assert "zip" in captured.err.lower()
    assert "Traceback" not in captured.err
    assert not (output / "source_unpack_report.json").exists()


def test_download_worker_converts_bad_zip_to_validation_error(source_set, tmp_path, monkeypatch):
    manifest, _, _ = source_set
    # The server's byte-range transport can succeed while the ZIP is truncated.
    monkeypatch.setattr(acquire, "_HTTPRangeFile", lambda *args, **kwargs: io.BytesIO(b"PK"))
    output = tmp_path / "downloaded"

    with pytest.raises(ValidationError, match="(?i)zip"):
        acquire.download_hypersim(manifest, output, workers=1)

    assert not (output / "source_download_report.json").exists()


def test_download_failure_cancels_running_and_queued_scenes(tmp_path, monkeypatch):
    workers = 4
    scenes = [f"ai_test_{index:03d}" for index in range(16)]
    plan = {
        "manifest_sha256": "synthetic-manifest",
        "datasets": {"hypersim": {"scene_ids": scenes, "files": []}},
    }
    monkeypatch.setattr(acquire, "source_plan", lambda *args, **kwargs: plan)
    initial_workers_ready = Barrier(workers)
    lock = Lock()
    started = []
    cancellation_events = []
    cancelled_workers = []
    network_error = URLError("synthetic provider failure")

    def range_reader(url, *, cancel):
        # Mirror the real reader's check before beginning a network request.
        if cancel.is_set():
            raise CancelledError()
        scene = Path(url).stem
        with lock:
            started.append(scene)
            cancellation_events.append(cancel)
        if scene in scenes[:workers]:
            initial_workers_ready.wait(timeout=2)
        # Fail a later scene while the first scene is waiting. Ordered map must
        # not prevent the parent from seeing and propagating this failure.
        if scene == scenes[workers - 1]:
            raise network_error
        assert cancel.wait(timeout=2), "running download did not receive cancellation"
        with lock:
            cancelled_workers.append(scene)
        raise CancelledError()

    monkeypatch.setattr(acquire, "_HTTPRangeFile", range_reader)
    output = tmp_path / "cancelled-download"

    with pytest.raises(URLError) as caught:
        acquire.download_hypersim(tmp_path / "unused-manifest.jsonl", output, workers=workers)

    assert caught.value is network_error
    assert workers <= len(started) < len(scenes)
    assert len({id(event) for event in cancellation_events}) == 1
    assert all(event.is_set() for event in cancellation_events)
    assert set(scenes[:workers - 1]).issubset(cancelled_workers)
    assert not (output / "source_download_report.json").exists()
