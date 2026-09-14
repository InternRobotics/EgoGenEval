"""Synthetic checks for original DSLR calibration and stable reconstruction."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from egogeneval.errors import ValidationError
from egogeneval.preparation.raw_scannetpp import (
    CAMERA_CONVENTIONS, RawScene, camera_conventions, legacy_fisheye_matrix,
)
from egogeneval.preparation.sources import Request, Sources


def camera_fixture(tmp_path, *, scene="synthetic", filename="DSC01234.JPG"):
    pose = np.array([[0., -1., 0., 1.], [1., 0., 0., 2.],
                     [0., 0., 1., 3.], [0., 0., 0., 1.]])
    nerfstudio = pose.copy()
    nerfstudio[:3, 1:3] *= -1
    nerfstudio = nerfstudio[[1, 0, 2, 3]]
    nerfstudio[2, :] *= -1
    metadata = {
        "camera_model": "OPENCV_FISHEYE", "w": 640, "h": 480,
        "fl_x": 400., "fl_y": 420., "cx": 315., "cy": 235.,
        "k1": .01, "k2": .001, "k3": .0001, "k4": 0.,
        "frames": [],
        "test_frames": [{"file_path": filename, "transform_matrix": nerfstudio.tolist()}],
    }
    path = tmp_path / "data" / scene / "dslr/nerfstudio/transforms.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(metadata))
    return path, metadata, pose


def test_official_nerfstudio_recovers_colmap_pose_and_test_frames(tmp_path):
    path, _, expected = camera_fixture(tmp_path)
    scene = RawScene.read(tmp_path, "synthetic")
    assert scene.metadata == path
    assert scene.poses["DSC01234"][0] == "DSC01234.JPG"
    np.testing.assert_array_equal(scene.poses["DSC01234"][1], expected)
    assert scene.files("DSC01234")[1].name == "DSC01234.png"
    with pytest.raises(ValidationError, match="no official calibration"):
        scene.files("42")


def test_alias_resolves_official_directory_without_changing_frame_identity(tmp_path):
    path, _, expected = camera_fixture(tmp_path, scene="7104910700000")
    ref = "${SCANNETPP_ROOT}/scannetpp_processed/7104910700/images/DSC01234.jpg"
    request = Request("scannetpp", "7104910700", ref, 42,
                      "scannetpp_processed/7104910700/images/DSC01234.jpg")
    source = Sources({"SCANNETPP_ROOT": tmp_path}, [request], [], raw_sources=True).resolve(request)
    assert source.raw_scannetpp is not None and source.depth is None
    assert source.metadata == path
    assert source.rgb == tmp_path / "data/7104910700000/dslr/resized_images/DSC01234.JPG"
    np.testing.assert_array_equal(source.pose, expected.astype(np.float32))
    assert request.scene_id == "7104910700" and request.frame_id == 42


def test_legacy_projection_is_independent_of_new_opencv_estimator(tmp_path, monkeypatch):
    camera_fixture(tmp_path)
    scene = RawScene.read(tmp_path, "synthetic")
    baseline = scene.intrinsic()
    monkeypatch.setattr(cv2.fisheye, "estimateNewCameraMatrixForUndistortRectify",
                        lambda *a, **k: pytest.fail("version-dependent estimator used"))
    np.testing.assert_array_equal(scene.intrinsic(), baseline)
    np.testing.assert_array_equal(baseline[:2, 2], [460., 345.])
    # With isotropic centered K and zero distortion, f = half-height / tan(FOV/2).
    K = np.array([[400., 0., 320.], [0., 400., 240.], [0., 0., 1.]])
    actual = legacy_fisheye_matrix(K, np.zeros(4), (640, 480))
    expected_focal = 240. / np.tan(240. / 400.)
    np.testing.assert_allclose(actual, [[expected_focal, 0., 320.],
                                      [0., expected_focal, 240.], [0., 0., 1.]], atol=1e-12)


def test_original_scene_directory_takes_precedence_over_current_alias(tmp_path):
    original, _, _ = camera_fixture(tmp_path, scene="7104910700")
    camera_fixture(tmp_path, scene="7104910700000", filename="DSC09999.JPG")
    scene = RawScene.read(tmp_path, "7104910700")
    assert scene.metadata == original
    assert "DSC01234" in scene.poses and "DSC09999" not in scene.poses


def test_metadata_center_does_not_shift_already_matching_depth_frames(tmp_path):
    """One real scene contains both historical depth-export conventions."""
    path, metadata, _ = camera_fixture(tmp_path, scene="5b38982d25", filename="DSC04137.JPG")
    metadata["test_frames"].append({**metadata["test_frames"][0], "file_path": "DSC04452.JPG"})
    path.write_text(json.dumps(metadata))
    scene = RawScene.read(tmp_path, "5b38982d25")
    base = scene.intrinsic(center_offset=0.0)
    shifted = base.copy()
    shifted[:2, 2] += 0.5 * (690 / 480 + 1e-8)
    np.testing.assert_array_equal(scene.intrinsic(), shifted)
    np.testing.assert_array_equal(scene.depth_intrinsic("DSC04137"), base)
    np.testing.assert_array_equal(scene.depth_intrinsic("DSC04452"), shifted)
    # Both frames still use the same original RGB undistortion camera.
    baseline_root = tmp_path / "baseline"
    camera_fixture(baseline_root)
    np.testing.assert_array_equal(scene.projection()[1], RawScene.read(baseline_root, "synthetic").projection()[1])
    with pytest.raises(ValidationError, match="no official calibration"):
        scene.depth_intrinsic("DSC99999")


def test_raw_sources_use_shifted_metadata_and_record_convention_provenance(tmp_path):
    camera_fixture(tmp_path, scene="5b38982d25", filename="DSC04137.JPG")
    relative = "scannetpp_processed/5b38982d25/images/DSC04137.jpg"
    request = Request("scannetpp", "5b38982d25", "${SCANNETPP_ROOT}/" + relative, 0, relative)
    source = Sources({"SCANNETPP_ROOT": tmp_path}, [request], [], raw_sources=True).resolve(request)
    np.testing.assert_array_equal(source.intrinsic, source.raw_scannetpp.intrinsic().astype(np.float32))
    assert source.intrinsic[0, 2] != source.raw_scannetpp.depth_intrinsic("DSC04137")[0, 2]
    assert CAMERA_CONVENTIONS in source.metadata_dependencies


def test_bundled_camera_conventions_cover_only_canonical_benchmark_frames():
    from egogeneval.preparation.sources import requests_from_rows
    manifest = Path(__file__).resolve().parents[1] / "data/manifests/egogeneval_v0.1.jsonl"
    rows = tuple(json.loads(line) for line in manifest.read_text().splitlines())
    frames = {}
    for request in requests_from_rows(rows):
        if request.dataset == "scannetpp":
            frames.setdefault(request.scene_id, set()).add(Path(request.relative).stem)
    profiles = camera_conventions()
    assert len(profiles) == 133
    assert sum(len(frames[scene]) for scene in profiles) == 599
    assert sum(len(stems) for _, stems in profiles.values()) == 546
    for scene, (offset, stems) in profiles.items():
        assert offset == 0.5 and stems <= frames[scene]


def test_missing_convention_resource_fails_instead_of_silently_using_wrong_camera(tmp_path, monkeypatch):
    import egogeneval.preparation.raw_scannetpp as raw
    camera_fixture(tmp_path)
    camera_conventions.cache_clear()
    monkeypatch.setattr(raw, "CAMERA_CONVENTIONS", tmp_path / "missing.json")
    try:
        with pytest.raises(ValidationError, match="bundled ScanNet\\+\\+ camera conventions"):
            RawScene.read(tmp_path, "synthetic")
    finally:
        camera_conventions.cache_clear()


@pytest.mark.parametrize("mutation", ["duplicate", "unsafe", "nonfinite", "reflection", "undistorted", "override"])
def test_rejects_invalid_or_incompatible_original_metadata(tmp_path, mutation):
    path, metadata, _ = camera_fixture(tmp_path)
    frame = metadata["test_frames"][0]
    if mutation == "duplicate":
        metadata["frames"] = [frame.copy()]
    elif mutation == "unsafe":
        frame["file_path"] = "../DSC01234.JPG"
    elif mutation == "nonfinite":
        frame["transform_matrix"][0][0] = float("nan")
    elif mutation == "reflection":
        frame["transform_matrix"][0] = [-v for v in frame["transform_matrix"][0]]
    elif mutation == "undistorted":
        metadata["camera_model"] = "PINHOLE"
    else:
        frame["fl_x"] = 500.
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValidationError):
        RawScene.read(tmp_path, "synthetic")
