"""Parallel sensor scenes must preserve assets and atomic failure cleanup."""

import copy
import os

import pytest

from egogeneval.errors import ValidationError
from egogeneval.io import read_jsonl, write_jsonl
from egogeneval.preparation.build import build_data
from egogeneval.preparation.parallel import scene_jobs
from test_local_data import raw_source_set as _raw_source_set, source_set as _source_set

source_set = _source_set
raw_source_set = _raw_source_set


def process_identity(value):
    from multiprocessing import get_start_method
    return value, os.getpid(), get_start_method()


def test_render_workers_use_spawned_processes_and_preserve_job_order():
    results = scene_jobs(process_identity, [3, 1, 2], 2, processes=True)
    assert [value for value, _, _ in results] == [3, 1, 2]
    assert all(pid != os.getpid() and method == "spawn" for _, pid, method in results)


def test_process_worker_failure_is_propagated():
    with pytest.raises(ValueError):
        scene_jobs(int, ["1", "invalid", "2"], 2, processes=True)


@pytest.fixture
def two_scenes(raw_source_set, tmp_path):
    manifest, roots, _, sensor, *_ = raw_source_set
    new_scene = "scene0001_00"
    second = sensor.parent.parent / new_scene / f"{new_scene}.sens"
    second.parent.mkdir()
    second.write_bytes(sensor.read_bytes())
    second.with_suffix(".txt").write_text(
        "axisAlignment = 1 0 0 375 0 1 0 9 0 0 1 7 0 0 0 1\n"
    )
    rows = read_jsonl(manifest)
    added = copy.deepcopy(next(row for row in rows if row["dataset"] == "scannet"))
    added["sample_id"] += "-second-scene"
    added["scene_id"] = new_scene
    for field in ("input_images", "target_images"):
        for image in added[field]:
            image["image_path"] = image["image_path"].replace("scene0000_00", new_scene)
    selected = tmp_path / "two-scenes.jsonl"
    write_jsonl(selected, [*rows, added])
    return selected, roots, second


def test_parallel_scenes_equal_serial_assets(two_scenes, tmp_path):
    manifest, roots, _ = two_scenes
    serial, parallel = tmp_path / "serial", tmp_path / "parallel"
    options = dict(raw_sources=True, datasets=["scannet", "matterport3d", "hypersim"])
    build_data(manifest, roots, serial, workers=1, **options)
    report = build_data(manifest, roots, parallel, workers=4,
                        reference_index=serial / "eval_frames.jsonl", **options)
    assert report["reference_assets_verified"] and report["scannet_frames_to_extract"] == 4
    assert (serial / "eval_frames.jsonl").read_bytes() == (parallel / "eval_frames.jsonl").read_bytes()


def test_failed_parallel_decode_cleans_staging(two_scenes, tmp_path):
    manifest, roots, sensor = two_scenes
    content = bytearray(sensor.read_bytes())
    content[-1] ^= 255  # Corrupt depth CRC after valid calibration/frame headers.
    sensor.write_bytes(content)
    output = tmp_path / "failed-parallel"
    with pytest.raises(ValidationError, match="cannot decode frame"):
        build_data(manifest, roots, output, raw_sources=True, datasets=["scannet"], workers=4)
    assert not output.exists() and not list(tmp_path.glob(".failed-parallel.*"))
