"""Synthetic source fixtures: no upstream dataset content is redistributed."""

import copy
import io
import json
import pickle
import struct
import zlib
from pathlib import Path

import cv2
import h5py
import numpy as np
import pytest
from PIL import Image

from egogeneval.data import validate_manifest
from egogeneval.errors import ValidationError
from egogeneval.io import read_jsonl, sha256_file, write_jsonl
from egogeneval.official.evaluation import dataloader, eval_assets
from egogeneval.preparation.build import build_data, verify_data
from egogeneval.preparation.fingerprints import fingerprint_row
from egogeneval.preparation.scannet import extract_frames
from egogeneval.scoring.pipeline import _eval_frame_lookup, _resolve_case_images


def dump_pickle(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(data, handle)


def rgbd(rgb_path, depth_path, value=0, shape=(18, 22), depth_shape=None):
    rgb_path.parent.mkdir(parents=True, exist_ok=True)
    depth_path.parent.mkdir(parents=True, exist_ok=True)
    pixels = (
        (np.arange(np.prod((*shape, 3))).reshape(*shape, 3) + value) % 256
    ).astype(np.uint8)
    Image.fromarray(pixels).save(rgb_path)
    depth_shape = depth_shape or shape
    depth = (1000 + np.arange(np.prod(depth_shape)).reshape(depth_shape) * 5).astype(
        np.uint16
    )
    depth[0, 0], depth[-1, -1] = 0, 60000
    Image.fromarray(depth).save(depth_path)


@pytest.fixture
def source_set(tmp_path, monkeypatch):
    monkeypatch.delenv(eval_assets.INDEX_ENV, raising=False)
    roots = {
        name: tmp_path / name.lower()
        for name in (
            "SCANNET_ROOT",
            "SCANNET_METADATA",
            "MATTERPORT3D_ROOT",
            "MATTERPORT3D_METADATA",
            "SCANNETPP_ROOT",
            "HYPERSIM_ROOT",
        )
    }
    for name, path in roots.items():
        path.mkdir()
        monkeypatch.setenv(name, str(path))
    K = np.array([[15, 0, 10], [0, 17, 8], [0, 0, 1]], dtype=np.float32)

    def pose(x):
        value = np.eye(4, dtype=np.float32)
        value[0, 3] = x
        return value

    scene_sc = "scene0000_00"
    sc_images = []
    for number in [0, 10, 20]:
        relative = f"posed_images/{scene_sc}/{number:05d}.jpg"
        rgbd(
            roots["SCANNET_ROOT"] / relative,
            (roots["SCANNET_ROOT"] / relative).with_suffix(".png"),
            number,
            depth_shape=(7, 9),
        )
        sc_images.append(
            {
                "img_path": f"scannet/{relative}",
                "cam2global": pose(number),
                "visible_instance_ids": [],
            }
        )
    sc_scene = {
        "sample_idx": f"scannet/{scene_sc}",
        "images": sc_images,
        "instances": [],
        "cam2img": K,
        "depth2img": K,
        "axis_align_matrix": pose(100),
    }
    dump_pickle(
        roots["SCANNET_METADATA"] / f"{scene_sc}.pkl",
        {"metainfo": {"categories": {}}, "data_list": [sc_scene]},
    )
    mp_scene_id = "1mp3d_0000_region0"
    mp_images, mp_poses = [], []
    for number in [0, 1]:
        relative = f"house/matterport_color_images/view{number}_i1_2.jpg"
        rgbd(
            roots["MATTERPORT3D_ROOT"] / "scans" / relative,
            roots["MATTERPORT3D_ROOT"]
            / "scans/house/matterport_depth_images"
            / f"view{number}_d1_2.png",
        )
        mp_images.append("matterport3d/" + relative)
        mp_poses.append(pose(number))
    dump_pickle(
        roots["MATTERPORT3D_METADATA"] / f"{mp_scene_id}.pkl",
        {
            "image_paths": mp_images,
            "depth_image_paths": [
                p.replace("color_images", "depth_images").replace(
                    "_i1_2.jpg", "_d1_2.png"
                )
                for p in mp_images
            ],
            "extrinsics_c2w": mp_poses,
            "intrinsics": [K, K],
            "depth_intrinsics": [K, K],
        },
    )
    # Existing MP loader accepts a scans root; builder must support it too.
    monkeypatch.setenv("MATTERPORT3D_ROOT", str(roots["MATTERPORT3D_ROOT"] / "scans"))
    pp = roots["SCANNETPP_ROOT"] / "scannetpp_processed/ppscene"
    for number in [0, 2]:
        rgbd(
            pp / "images" / f"DSC{number:05d}.jpg",
            pp / "depth" / f"DSC{number:05d}.png",
            number,
        )
    np.savez(
        pp / "scene_metadata.npz",
        images=["DSC00002.jpg", "DSC00000.jpg"],
        trajectories=[pose(2), pose(0)],
        intrinsics=[K, K],
    )
    hs = roots["HYPERSIM_ROOT"] / "evermotion_dataset/scenes/ai_001_001"
    detail = hs / "_detail/cam_00"
    detail.mkdir(parents=True)
    (hs / "_detail/metadata_scene.csv").write_text(
        "parameter_name,parameter_value\nmeters_per_asset_unit,2\n"
    )
    for name, array in [
        ("positions", np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])),
        ("orientations", np.repeat(np.eye(3)[None], 3, axis=0)),
    ]:
        with h5py.File(detail / f"camera_keyframe_{name}.hdf5", "w") as h:
            h["dataset"] = array
    for number in [0, 2]:
        image = (
            hs / "images/scene_cam_00_final_preview" / f"frame.{number:04d}.color.jpg"
        )
        image.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (22, 18), (30, 60, number)).save(image)
        depth = (
            hs
            / "images/scene_cam_00_geometry_hdf5"
            / f"frame.{number:04d}.depth_meters.hdf5"
        )
        depth.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(depth, "w") as h:
            values = np.arange(80, dtype=np.float32).reshape(8, 10) / 10
            values[0, 0] = np.nan
            values[-1, -1] = 1000
            h["dataset"] = values
    specs = [
        (
            "scannet",
            scene_sc,
            "SCANNET_ROOT",
            [
                "posed_images/scene0000_00/00000.jpg",
                "posed_images/scene0000_00/00020.jpg",
            ],
            [0, 2],
        ),
        (
            "matterport3d",
            mp_scene_id,
            "MATTERPORT3D_ROOT",
            [p.replace("matterport3d/", "scans/") for p in mp_images],
            [0, 1],
        ),
        (
            "scannetpp",
            "ppscene",
            "SCANNETPP_ROOT",
            [f"scannetpp_processed/ppscene/images/DSC{n:05d}.jpg" for n in [0, 2]],
            [0, 1],
        ),
        (
            "hypersim",
            "ai_001_001/cam_00",
            "HYPERSIM_ROOT",
            [
                f"evermotion_dataset/scenes/ai_001_001/images/scene_cam_00_final_preview/frame.{n:04d}.color.jpg"
                for n in [0, 2]
            ],
            [0, 1],
        ),
    ]
    rows = []
    for dataset, scene, name, paths, ids in specs:
        images = [
            {
                "role": "current" if i == 0 else "target_step_1",
                "frame_id": str(ids[i]),
                "image_path": "${" + name + "}/" + path,
            }
            for i, path in enumerate(paths)
        ]
        rows.append(
            {
                "sample_id": dataset,
                "root_sample_id": dataset,
                "dataset": dataset,
                "scene_id": scene,
                "context_type": "single",
                "num_context_images": 1,
                "instruction_type": "atomic",
                "subtype": "fixture",
                "input_images": images[:1],
                "target_images": images[1:],
                "instructions": [{"step": 1, "text": "Move backward."}],
                "evaluation_protocol": {
                    "call_mode": "single_step",
                    "num_model_calls": 1,
                    "use_teacher_forcing": False,
                },
                "pose_metadata": {},
            }
        )
    manifest = tmp_path / "manifest.jsonl"
    write_jsonl(manifest, rows)
    embodied = tmp_path / "embodiedscan_infos.pkl"
    mp_scene = {
        "sample_idx": "matterport3d/house/region0",
        # MP3D benchmark cameras remain in native world coordinates even when
        # the annotation supplies a nonidentity transform for aligned boxes.
        "axis_align_matrix": pose(123),
        "images": [
            {"img_path": p, "cam2global": pose, "cam2img": K}
            for p, pose in zip(mp_images, mp_poses)
        ],
    }
    dump_pickle(
        embodied, {"metainfo": {"categories": {}}, "data_list": [sc_scene, mp_scene]}
    )
    return manifest, roots, embodied


@pytest.mark.parametrize("use_embodied", [False, True])
def test_all_sources_match_frozen_loaders(source_set, tmp_path, use_embodied):
    manifest, roots, embodied = source_set
    output = tmp_path / "built"
    annotations = [embodied] if use_embodied else []
    report = build_data(manifest, roots, output, annotations=annotations)
    assert report["cases"] == 4 and report["frames"] == 8
    assert not report["reference_assets_verified"]
    assert sha256_file(manifest) == sha256_file(output / "manifest.jsonl")
    assert verify_data(output, reference_index=output)["reference_checked"]
    lookup = _eval_frame_lookup(output / "eval_frames.jsonl")
    loaders = {
        "scannet": dataloader.load_scannet_scene,
        "matterport3d": dataloader.load_matterport3d_scene,
        "scannetpp": dataloader.load_scannetpp_scene,
        "hypersim": lambda scene: dataloader.load_hypersim_scene(
            scene, root=str(roots["HYPERSIM_ROOT"])
        ),
    }
    records = read_jsonl(output / "eval_frames.jsonl")
    for case in validate_manifest(manifest).rows:
        original, _ = loaders[case["dataset"]](case["scene_id"])
        by_name = {Path(f.frame_name).name: f for f in original}
        resolved = _resolve_case_images(case, prepared_data=None, eval_frames=lookup)
        assert Path(resolved["target_images"][0]["image_path"]).is_file()
        for row in [r for r in records if r["dataset"] == case["dataset"]]:
            expected = by_name[Path(row["image_ref"]).name]
            actual = eval_assets._materialize(
                {
                    **row,
                    "_image_path": output / row["image_path"],
                    "_depth_path": output / row["depth_path"],
                },
                0,
            )
            np.testing.assert_array_equal(actual.image, expected.image)
            np.testing.assert_array_equal(actual.depth, expected.depth)
            np.testing.assert_array_equal(actual.intrinsics, expected.intrinsics)
            np.testing.assert_array_equal(actual.extrinsics, expected.extrinsics)
            assert actual.frame_idx == expected.frame_idx
    for row in read_jsonl(output / "generation_inputs.jsonl"):
        assert "target_images" not in row and "pose_metadata" not in row


def test_preflight_no_output_and_existing_output_preserved(source_set, tmp_path):
    manifest, roots, _ = source_set
    output = tmp_path / "not_created"
    report = build_data(manifest, roots, output, check_only=True)
    assert report["status"] == "ready" and not output.exists()
    (roots["SCANNET_ROOT"] / "posed_images/scene0000_00/00000.png").unlink()
    report = build_data(manifest, roots, output, check_only=True)
    assert report["status"] == "missing-inputs" and not output.exists()
    output.mkdir()
    marker = output / "mine"
    marker.write_text("keep")
    with pytest.raises(ValidationError):
        build_data(manifest, roots, output)
    assert marker.read_text() == "keep"


def test_selected_datasets_need_no_scannetpp_and_preserve_lines(
    source_set, tmp_path, monkeypatch
):
    manifest, roots, _ = source_set
    (roots["SCANNETPP_ROOT"] / "scannetpp_processed/ppscene/scene_metadata.npz").unlink()
    roots.pop("SCANNETPP_ROOT")
    monkeypatch.delenv("SCANNETPP_ROOT")
    selected = ["hypersim", "scannet", "matterport3d"]
    output = tmp_path / "three_sources"
    preflight = build_data(manifest, roots, output, datasets=selected, check_only=True)
    assert preflight["status"] == "ready" and preflight["frames"] == 6
    assert not output.exists()
    report = build_data(manifest, roots, output, datasets=selected)
    assert report["is_subset"] and report["cases"] == 3 and report["source_cases"] == 4
    assert report["source_manifest_sha256"] == sha256_file(manifest)
    expected = b"".join(
        line for line in manifest.read_bytes().splitlines(keepends=True)
        if json.loads(line)["dataset"] in selected
    )
    assert (output / "manifest.jsonl").read_bytes() == expected
    assert report["manifest_sha256"] == sha256_file(output / "manifest.jsonl")
    assert verify_data(output)["frames"] == 6
    assert {r["dataset"] for r in read_jsonl(output / "generation_inputs.jsonl")} == set(selected)


@pytest.mark.parametrize("extra_field", ["images", "intrinsics", "both"])
def test_scannetpp_trailing_metadata_matches_frozen_loader(
    source_set, tmp_path, extra_field
):
    manifest, roots, _ = source_set
    path = roots["SCANNETPP_ROOT"] / "scannetpp_processed/ppscene/scene_metadata.npz"
    with np.load(path) as archive:
        data = dict(archive)
    if extra_field in {"images", "both"}:
        data["images"] = np.append(data["images"], "unposed.jpg")
    if extra_field in {"intrinsics", "both"}:
        data["intrinsics"] = np.concatenate(
            [data["intrinsics"], data["intrinsics"][:1]]
        )
    np.savez(path, **data)
    output = tmp_path / "built"
    build_data(manifest, roots, output)
    expected, _ = dataloader.load_scannetpp_scene("ppscene")
    by_name = {Path(frame.frame_name).name: frame for frame in expected}
    actual = [r for r in read_jsonl(output / "eval_frames.jsonl") if r["dataset"] == "scannetpp"]
    assert len(actual) == len(expected) == 2
    for row in actual:
        frame = by_name[Path(row["image_ref"]).name]
        np.testing.assert_array_equal(row["extrinsics_c2w"], frame.extrinsics)
        np.testing.assert_array_equal(row["intrinsics"], frame.intrinsics)


@pytest.mark.parametrize("short_field", ["images", "intrinsics"])
def test_scannetpp_missing_metadata_for_trajectory_fails(
    source_set, tmp_path, short_field
):
    manifest, roots, _ = source_set
    path = roots["SCANNETPP_ROOT"] / "scannetpp_processed/ppscene/scene_metadata.npz"
    with np.load(path) as archive:
        data = dict(archive)
    data[short_field] = data[short_field][:1]
    np.savez(path, **data)
    report = build_data(manifest, roots, tmp_path / "unused", check_only=True)
    assert report["status"] == "missing-inputs"
    assert all("missing image or intrinsic for trajectory" in p for p in report["problems"])


def test_corrupt_asset_and_reference_drift_fail(source_set, tmp_path):
    manifest, roots, _ = source_set
    output = tmp_path / "built"
    build_data(manifest, roots, output)
    rows = read_jsonl(output / "eval_frames.jsonl")
    reference = tmp_path / "reference.jsonl"
    altered = copy.deepcopy(rows)
    altered[0]["intrinsics"][0][0] *= 2
    write_jsonl(reference, altered)
    with pytest.raises(ValidationError, match="reference mismatch.*intrinsics"):
        verify_data(output, reference_index=reference)
    altered = copy.deepcopy(rows)
    pose = altered[0]["extrinsics_c2w"]
    pose[0][0] = float(np.nextafter(np.float32(pose[0][0]), np.float32(np.inf)))
    write_jsonl(reference, altered)
    with pytest.raises(ValidationError, match="reference mismatch.*extrinsics_c2w"):
        verify_data(output, reference_index=reference)
    (output / rows[0]["image_path"]).write_bytes(b"corrupt")
    with pytest.raises(ValidationError, match="modified asset"):
        verify_data(output)


def sensor_fixture(path):
    pixels = np.arange(18 * 22 * 3, dtype=np.uint8).reshape(18, 22, 3)
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="JPEG")
    color = buffer.getvalue()
    depth = np.arange(7 * 9, dtype="<u2").reshape(7, 9) + 1000
    compressed = zlib.compress(depth.tobytes())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(struct.pack("<IQ", 4, 4))
        f.write(b"test")
        f.write(np.tile(np.eye(4, dtype="<f4"), (4, 1)).tobytes())
        f.write(struct.pack("<iiIIIIfQ", 2, 1, 22, 18, 9, 7, 1000.0, 21))
        for _ in range(21):
            f.write(np.eye(4, dtype="<f4").tobytes())
            f.write(struct.pack("<QQQQ", 0, 0, len(color), len(compressed)))
            f.write(color)
            f.write(compressed)
    return depth


def test_sens_streaming_and_missing_depth_integration(source_set, tmp_path):
    manifest, roots, embodied = source_set
    sensor = roots["SCANNET_ROOT"] / "scans/scene0000_00/scene0000_00.sens"
    expected_depth = sensor_fixture(sensor)
    extracted = tmp_path / "extracted"
    extract_frames(sensor, {0, 20}, extracted)
    assert sorted(p.name for p in extracted.iterdir()) == [
        "00000.jpg",
        "00000.png",
        "00020.jpg",
        "00020.png",
    ]
    np.testing.assert_array_equal(
        cv2.imread(str(extracted / "00020.png"), -1), expected_depth
    )
    source = roots["SCANNET_ROOT"] / "posed_images/scene0000_00"
    original_rgb_hash = sha256_file(source / "00000.jpg")
    (source / "00000.png").unlink()  # existing RGB must be retained
    (source / "00020.jpg").unlink()
    (source / "00020.png").unlink()
    output = tmp_path / "built"
    assert (
        build_data(manifest, roots, output, annotations=[embodied], check_only=True)[
            "scannet_frames_to_extract"
        ]
        == 2
    )
    build_data(manifest, roots, output, annotations=[embodied], mode="symlink")
    assert verify_data(output)["frames"] == 8
    rows = [
        r for r in read_jsonl(output / "eval_frames.jsonl") if r["dataset"] == "scannet"
    ]
    assert rows[0]["rgb_sha256"] == original_rgb_hash
    assert not (source / "00000.png").exists()  # source dataset stays untouched
    assert not (output / "extracted").exists()


def test_truncated_sens_fails(tmp_path):
    sensor = tmp_path / "bad.sens"
    sensor_fixture(sensor)
    sensor.write_bytes(sensor.read_bytes()[:-20])
    with pytest.raises(ValidationError, match="truncated"):
        extract_frames(sensor, {20}, tmp_path / "out")


def test_mp_region_mapping_and_annotation_conflicts(source_set, tmp_path):
    manifest, roots, embodied = source_set
    with embodied.open("rb") as handle:
        data = pickle.load(handle)
    other_region = copy.deepcopy(data["data_list"][1])
    other_region["sample_idx"] = "matterport3d/house/region99"
    other_region["axis_align_matrix"][0, 3] = 999
    data["data_list"].insert(0, other_region)
    dump_pickle(embodied, data)
    output = tmp_path / "correct-region"
    build_data(manifest, roots, output, annotations=[embodied])
    mp = [
        r
        for r in read_jsonl(output / "eval_frames.jsonl")
        if r["dataset"] == "matterport3d"
    ]
    assert mp[0]["extrinsics_c2w"][0][3] == 0
    conflict = copy.deepcopy(data["data_list"][-1])
    conflict["images"][0]["cam2global"][0, 3] = 123
    data["data_list"].append(conflict)
    dump_pickle(embodied, data)
    with pytest.raises(ValidationError, match="conflicting EmbodiedScan"):
        build_data(manifest, roots, tmp_path / "conflict", annotations=[embodied])


def test_failed_build_leaves_no_partial_output(source_set, tmp_path):
    manifest, roots, _ = source_set
    # File existence alone passes preflight, so this fails after staging starts.
    depth = roots["SCANNETPP_ROOT"] / "scannetpp_processed/ppscene/depth/DSC00000.png"
    depth.write_bytes(b"invalid PNG")
    output = tmp_path / "failed"
    with pytest.raises((ValidationError, FileNotFoundError)):
        build_data(manifest, roots, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".failed.*"))


@pytest.mark.parametrize("selected", [None, ["hypersim", "matterport3d", "scannet"]])
def test_cli_build_and_verify(source_set, tmp_path, capsys, selected):
    from egogeneval.cli import main

    manifest, roots, embodied = source_set
    output = tmp_path / "cli-output"
    arguments = [
        "build-data",
        "--manifest",
        str(manifest),
        "--embodiedscan-info",
        str(embodied),
        "--output",
        str(output),
    ]
    for name, path in roots.items():
        arguments.extend(["--root", f"{name}={path}"])
    for dataset in selected or []:
        arguments.extend(["--dataset", dataset])
    assert main([*arguments, "--check-only"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "ready"
    assert not output.exists()
    assert main(arguments) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "built"
    assert main(["verify-data", "--data", str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["frames"] == (6 if selected else 8)
    assert main(arguments) == 2
    assert "already exists" in capsys.readouterr().err


def test_historical_scannet_reference_intrinsics(source_set, tmp_path):
    manifest, roots, _ = source_set
    output = tmp_path / "built"
    build_data(manifest, roots, output)
    rows = read_jsonl(output / "eval_frames.jsonl")
    for row in rows:
        row["asset_format_version"] = eval_assets.LEGACY_ASSET_FORMAT_VERSION
        row["preprocessing"] = eval_assets.LEGACY_PREPROCESSING_CONTRACT_ID
        row.pop("resize_stages")
        if row["dataset"] == "scannet":
            for i, j in [(0, 0), (0, 2), (1, 1), (1, 2)]:
                row["intrinsics"][i][j] *= 2
    reference = tmp_path / "historical.jsonl"
    write_jsonl(reference, rows)
    assert verify_data(output, reference_index=reference)["reference_checked"]


def test_changed_instructions_are_detected(source_set, tmp_path):
    manifest, roots, _ = source_set
    output = tmp_path / "built"
    build_data(manifest, roots, output)
    rows = read_jsonl(output / "manifest.jsonl")
    rows[0]["instructions"][0]["text"] = "A different task"
    write_jsonl(output / "manifest.jsonl", rows)
    with pytest.raises(ValidationError, match="modified build artifact"):
        verify_data(output)


@pytest.fixture
def raw_source_set(source_set, tmp_path):
    """Official-layout synthetic inputs alongside unusable processed metadata."""
    manifest, roots, embodied = source_set
    sensor = roots["SCANNET_ROOT"] / "scans/scene0000_00/scene0000_00.sens"
    sensor_fixture(sensor)
    axis = np.array([
        [0, -1, 0, 100], [1, 0, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1],
    ], dtype=np.float32)
    sensor.with_suffix(".txt").write_text(
        "axisAlignment = " + " ".join(str(float(v)) for v in axis.flat) + "\n"
    )
    # Existing posed_images are valid files, but are intentionally not the
    # sensor export. Raw mode must replace both RGB and depth in its own output.
    for number in (0, 20):
        processed = roots["SCANNET_ROOT"] / "posed_images/scene0000_00" / f"{number:05d}.jpg"
        Image.new("RGB", (22, 18), (255, 0, 255)).save(processed)
        Image.fromarray(np.full((7, 9), 9000, dtype=np.uint16)).save(
            processed.with_suffix(".png")
        )
    house = roots["MATTERPORT3D_ROOT"] / "scans/house"
    for folder in ("matterport_camera_poses", "matterport_camera_intrinsics"):
        (house / folder).mkdir()
    for number in (0, 1):
        pose = np.eye(4, dtype=np.float32)
        pose[0, 3] = number + 0.25
        np.savetxt(house / "matterport_camera_poses" / f"view{number}_pose1_2.txt", pose)
        (house / "matterport_camera_intrinsics" / f"view{number}_intrinsics1.txt").write_text(
            "22 18 15 17 10 8 0 0 0 0 0\n"
        )
    for path in [embodied, *roots["SCANNET_METADATA"].glob("*.pkl"),
                 *roots["MATTERPORT3D_METADATA"].glob("*.pkl")]:
        path.write_bytes(b"intentionally invalid processed metadata")
    extracted = tmp_path / "independent-sensor-export"
    extract_frames(sensor, {0, 20}, extracted)
    return manifest, roots, embodied, sensor, house, axis, extracted


@pytest.mark.parametrize("mode", ["copy", "symlink"])
def test_raw_three_datasets_ignore_processed_images_and_metadata(
    raw_source_set, tmp_path, monkeypatch, mode
):
    manifest, roots, _, _, _, axis, extracted = raw_source_set

    def forbid_pickle(*args, **kwargs):
        pytest.fail("raw build attempted to read processed pickle metadata")

    monkeypatch.setattr("egogeneval.preparation.sources._pickle", forbid_pickle)
    output = tmp_path / "raw-built"
    options = dict(datasets=["hypersim", "matterport3d", "scannet"], raw_sources=True)
    processed = roots["SCANNET_ROOT"] / "posed_images/scene0000_00"
    before = {p: sha256_file(p) for p in processed.iterdir()}
    report = build_data(manifest, roots, output, check_only=True, **options)
    assert report["status"] == "ready" and report["scannet_frames_to_extract"] == 2
    assert not output.exists()
    report = build_data(manifest, roots, output, mode=mode, **options)
    assert report["raw_sources"] and report["cases"] == 3 and report["frames"] == 6
    assert report["calibration_sources"] == {
        "scannet-native-sens": 2,
        "matterport3d-native-camera": 2,
        "hypersim-official-layout": 2,
    }
    records = read_jsonl(output / "eval_frames.jsonl")
    for row in records:
        if row["dataset"] == "scannet":
            name = Path(row["image_ref"]).name
            image = output / row["image_path"]
            assert image.read_bytes() == (extracted / name).read_bytes()
            assert sha256_file(image) != sha256_file(processed / name)
            assert not image.is_symlink()  # temporary extraction cannot be a symlink target
            np.testing.assert_array_equal(row["extrinsics_c2w"], axis)
            np.testing.assert_array_equal(row["intrinsics"], np.diag([0.25, 0.25, 1]))
        elif row["dataset"] == "matterport3d":
            assert row["extrinsics_c2w"][0][3] == row["loader_frame_idx"] + 0.25
    assert {p: sha256_file(p) for p in before} == before
    assert not (output / "extracted").exists()
    assert verify_data(output)["frames"] == 6
    assert {row["dataset"] for row in read_jsonl(output / "generation_inputs.jsonl")} == {
        "hypersim", "matterport3d", "scannet",
    }


def _raw_fixture_fingerprints(raw_source_set, tmp_path):
    # This synthetic round trip tests reference enforcement and transaction
    # wiring. It is not evidence that real official downloads match the paper.
    manifest, roots, *_ = raw_source_set
    reference_build = tmp_path / "synthetic-reference-build"
    build_data(manifest, roots, reference_build, raw_sources=True,
               datasets=["hypersim", "matterport3d", "scannet"])
    fingerprints = tmp_path / "synthetic-reference.jsonl"
    write_jsonl(fingerprints, [
        fingerprint_row(row) for row in read_jsonl(reference_build / "eval_frames.jsonl")
    ])
    return fingerprints


def test_raw_build_copies_and_automatically_rechecks_fingerprints(raw_source_set, tmp_path):
    manifest, roots, *_ = raw_source_set
    fingerprints = _raw_fixture_fingerprints(raw_source_set, tmp_path)
    output = tmp_path / "verified-raw"
    report = build_data(manifest, roots, output, raw_sources=True,
                        datasets=["hypersim", "matterport3d", "scannet"],
                        fingerprints=fingerprints)
    assert report["reference_assets_verified"]
    assert report["verification"]["fingerprints_checked"]
    bundled = output / "reference_fingerprints.jsonl"
    assert bundled.read_bytes() == fingerprints.read_bytes()
    assert report["artifact_sha256"][bundled.name] == sha256_file(fingerprints)
    fingerprints.unlink()  # subsequent verification uses only the output bundle
    assert verify_data(output)["fingerprints_checked"]
    # Keep asset/index integrity internally consistent while altering camera GT;
    # the independently bundled reference must still detect the mismatch.
    records = read_jsonl(output / "eval_frames.jsonl")
    records[0]["extrinsics_c2w"][0][3] += 0.25
    write_jsonl(output / "eval_frames.jsonl", records)
    stored_report = json.loads((output / "build_report.json").read_text())
    stored_report["artifact_sha256"]["eval_frames.jsonl"] = sha256_file(output / "eval_frames.jsonl")
    (output / "build_report.json").write_text(json.dumps(stored_report))
    with pytest.raises(ValidationError, match="frozen fingerprint mismatch.*extrinsics"):
        verify_data(output)


@pytest.mark.parametrize("field", ["rgb_sha256", "effective_extrinsics_c2w_sha256"])
def test_raw_fingerprint_mismatch_leaves_no_partial_output(raw_source_set, tmp_path, field):
    manifest, roots, *_ = raw_source_set
    fingerprints = _raw_fixture_fingerprints(raw_source_set, tmp_path)
    rows = read_jsonl(fingerprints)
    rows[0][field] = "0" * 64
    write_jsonl(fingerprints, rows)
    output = tmp_path / "failed-raw"
    with pytest.raises(ValidationError, match=f"frozen fingerprint mismatch.*{field}"):
        build_data(manifest, roots, output, raw_sources=True,
                   datasets=["hypersim", "matterport3d", "scannet"],
                   fingerprints=fingerprints)
    assert not output.exists()
    assert not list(tmp_path.glob(".failed-raw.*"))


@pytest.mark.parametrize("missing", ["sensor", "axis", "matterport_pose", "matterport_intrinsics"])
def test_raw_missing_original_files_never_falls_back(raw_source_set, tmp_path, missing):
    manifest, roots, _, sensor, house, *_ = raw_source_set
    paths = {
        "sensor": sensor,
        "axis": sensor.with_suffix(".txt"),
        "matterport_pose": house / "matterport_camera_poses/view0_pose1_2.txt",
        "matterport_intrinsics": house / "matterport_camera_intrinsics/view0_intrinsics1.txt",
    }
    paths[missing].unlink()
    output = tmp_path / "missing-raw"
    options = dict(datasets=["hypersim", "matterport3d", "scannet"], raw_sources=True)
    report = build_data(manifest, roots, output, check_only=True, **options)
    assert report["status"] == "missing-inputs" and report["problem_count"] > 0
    expected_dataset = "matterport3d" if missing.startswith("matterport") else "scannet"
    assert expected_dataset in report["problems_by_dataset"]
    assert any(str(paths[missing]) in problem for problem in report["problems"])
    with pytest.raises(ValidationError, match="source check found"):
        build_data(manifest, roots, output, **options)
    assert not output.exists()


def test_raw_scannetpp_requires_official_files_and_rejects_processed_annotations(raw_source_set, tmp_path):
    manifest, roots, embodied, *_ = raw_source_set
    output = tmp_path / "unsupported"
    with pytest.raises(ValidationError, match="nerfstudio/transforms.json"):
        build_data(manifest, roots, output, raw_sources=True, datasets=["scannetpp"])
    with pytest.raises(ValidationError, match="omit --embodiedscan-info"):
        build_data(manifest, roots, output, raw_sources=True,
                   datasets=["hypersim", "matterport3d", "scannet"], annotations=[embodied])
    assert not output.exists()
