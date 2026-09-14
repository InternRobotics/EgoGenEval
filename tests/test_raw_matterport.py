"""Synthetic native-format calibration tests; no Matterport3D data is included."""

import numpy as np
import pytest

from egogeneval.errors import ValidationError
from egogeneval.preparation.raw_matterport import read_native_calibration


POSE = "0 -1 0 1.234567891\n1 0 0 -2.345678912\n0 0 1 3.456789123\n0 0 0 1\n"
INTRINSICS = "1280 1024 1078.1256789 1076.3751234 638.6254321 511.8756789 .1 -.2 .003 -.004 .05\n"


def native_files(tmp_path, separator="_", yaw=4):
    house = tmp_path / "native-house"
    rgb = house / "matterport_color_images" / f"native-panorama_i1_{yaw}.jpg"
    rgb.parent.mkdir(parents=True, exist_ok=True)
    pose = house / "matterport_camera_poses" / f"native-panorama_pose{separator}1_{yaw}.txt"
    intrinsic = house / "matterport_camera_intrinsics" / f"native-panorama_intrinsics{separator}1.txt"
    pose.parent.mkdir(parents=True, exist_ok=True)
    intrinsic.parent.mkdir(parents=True, exist_ok=True)
    pose.write_text(POSE)
    intrinsic.write_text(INTRINSICS)
    return rgb, pose, intrinsic


@pytest.mark.parametrize("separator", ["_", ""])
def test_native_cameras_preserve_world_axes_units_and_precision(tmp_path, separator):
    rgb, pose_path, k_path = native_files(tmp_path, separator)
    result = read_native_calibration(rgb)
    expected_pose = np.array(
        [[0, -1, 0, 1.234567891], [1, 0, 0, -2.345678912],
         [0, 0, 1, 3.456789123], [0, 0, 0, 1]], dtype=np.float32,
    )
    expected_k = np.array(
        [[1078.1256789, 0, 638.6254321], [0, 1076.3751234, 511.8756789], [0, 0, 1]],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(result.pose, expected_pose)
    np.testing.assert_array_equal(result.intrinsic, expected_k)
    np.testing.assert_array_equal(
        result.distortion, np.array([.1, -.2, .003, -.004, .05], dtype=np.float32),
    )
    assert result.pose.dtype == result.intrinsic.dtype == np.dtype("float32")
    assert result.image_size == (1280, 1024)
    assert result.pose_path == pose_path
    assert result.intrinsic_path == k_path


def test_yaw_selects_exact_pose_but_reuses_per_camera_intrinsics(tmp_path):
    rgb, pose_path, k_path = native_files(tmp_path, yaw=5)
    wrong_pose = pose_path.with_name("native-panorama_pose_1_0.txt")
    wrong_pose.write_text(POSE.replace("1.234567891", "99.0"))
    result = read_native_calibration(rgb)
    assert result.pose_path == pose_path
    assert result.intrinsic_path == k_path
    assert result.pose[0, 3] == np.float32(1.234567891)


@pytest.mark.parametrize("kind", ["pose", "intrinsics"])
def test_duplicate_naming_variants_do_not_silently_choose(tmp_path, kind):
    rgb, pose_path, k_path = native_files(tmp_path)
    source = pose_path if kind == "pose" else k_path
    source.with_name(source.name.replace(f"_{kind}_", f"_{kind}")).write_text(source.read_text())
    with pytest.raises(ValidationError, match="ambiguous official Matterport3D"):
        read_native_calibration(rgb)


@pytest.mark.parametrize("kind", ["pose", "intrinsics"])
def test_missing_metadata_names_the_official_download(tmp_path, kind):
    rgb, pose_path, k_path = native_files(tmp_path)
    missing = pose_path if kind == "pose" else k_path
    missing.unlink()
    archive = "matterport_camera_poses.zip" if kind == "pose" else "matterport_camera_intrinsics.zip"
    with pytest.raises(ValidationError, match=archive.replace(".", r"\.")):
        read_native_calibration(rgb)


@pytest.mark.parametrize("relative", [
    "undistorted_color_images/native-panorama_i1_4.jpg",
    "matterport_color_images/native-panorama_i7_4.jpg",
    "matterport_color_images/native-panorama_i1_8.jpg",
    "matterport_color_images/renumbered_00004.jpg",
])
def test_other_camera_conventions_and_numbered_images_are_rejected(tmp_path, relative):
    with pytest.raises(ValidationError, match="expected a native Matterport3D RGB path"):
        read_native_calibration(tmp_path / relative)


@pytest.mark.parametrize("text", [
    "1 2 3", POSE.replace("3.456789123", "nan"),
    POSE.replace("3.456789123", "inf"), POSE.replace("3.456789123", "broken"),
    POSE.replace("0 0 0 1", "0 0 0 0"), POSE.replace("0 0 1 3.456789123", "0 0 0 3.456789123"),
])
def test_corrupt_camera_pose_fails_before_reconstruction(tmp_path, text):
    rgb, pose_path, _ = native_files(tmp_path)
    pose_path.write_text(text)
    with pytest.raises(ValidationError):
        read_native_calibration(rgb)


@pytest.mark.parametrize("text", [
    "1 2 3", INTRINSICS.replace("1280", "0"), INTRINSICS.replace("1024", "10.5"),
    INTRINSICS.replace("1078.1256789", "-1"), INTRINSICS.replace(".05", "nan"),
])
def test_corrupt_intrinsics_fail_before_reconstruction(tmp_path, text):
    rgb, _, k_path = native_files(tmp_path)
    k_path.write_text(text)
    with pytest.raises(ValidationError):
        read_native_calibration(rgb)


def test_pose_can_use_single_line_ascii_format(tmp_path):
    rgb, pose_path, _ = native_files(tmp_path)
    expected = read_native_calibration(rgb).pose.copy()
    pose_path.write_text(" ".join(POSE.split()))
    np.testing.assert_array_equal(read_native_calibration(rgb).pose, expected)
