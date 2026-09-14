"""Synthetic sensor fixtures exercise the raw ScanNet calibration path."""

import io
import struct
from pathlib import Path

import numpy as np
import pytest

from egogeneval.errors import ValidationError
from egogeneval.preparation.raw_scannet import read_axis_alignment, read_calibration


def _export_roundtrip(matrix):
    # Match upstream's row-by-row %f export, independently of the reader helper.
    text = io.StringIO()
    for row in matrix:
        np.savetxt(text, row[np.newaxis], fmt="%f")
    text.seek(0)
    return np.loadtxt(text).astype(np.float32)


@pytest.fixture
def sensor_scene(tmp_path):
    sensor = tmp_path / "scene0000_00.sens"
    metadata = sensor.with_suffix(".txt")
    intrinsic = np.array([
        [571.6234, 0, 1.2345678, 0],
        [0, 572.789, 2.3456789, 0],
        [0, 0, 1, 0], [0, 0, 0, 1],
    ], dtype=np.float32)
    axis = np.array([
        [0.9238795, -0.3826834, 0, 1.0000123],
        [0.3826834, 0.9238795, 0, -2.0000234],
        [0, 0, 1, 3.0000345], [0, 0, 0, 1],
    ], dtype=np.float32)
    metadata.write_text("colorWidth = 1296\naxisAlignment = " + " ".join(str(float(v)) for v in axis.flat) + "\n")
    poses = []
    for i in range(5):
        pose = np.eye(4, dtype=np.float32)
        pose[:3, 3] = [0.12345678 + i, 1.23456789 - i, -2.34567891 + i]
        poses.append(pose)
    poses[2][:] = np.nan  # a filtered frame must not invalidate other requests
    # Deliberately non-decodable large payloads: calibration must seek past them.
    color, depth = b"x" * 10000, b"y" * 5000
    with sensor.open("wb") as stream:
        stream.write(struct.pack("<IQ", 4, 4) + b"test")
        stream.write(intrinsic.astype("<f4").tobytes())
        for _ in range(3):
            stream.write(np.eye(4, dtype="<f4").tobytes())
        stream.write(struct.pack("<iiIIIIfQ", 2, 1, 1296, 968, 640, 480, 1000.0, len(poses)))
        for pose in poses:
            stream.write(pose.astype("<f4").tobytes())
            stream.write(struct.pack("<QQQQ", 0, 0, len(color), len(depth)))
            stream.write(color)
            stream.write(depth)
    return sensor, metadata, intrinsic, axis, poses


def test_raw_calibration_matches_exported_text_and_float32_alignment(sensor_scene, monkeypatch):
    sensor, metadata, intrinsic, axis, poses = sensor_scene
    before = {path: path.read_bytes() for path in (sensor, metadata)}
    original_open = Path.open
    read_sizes = []

    class ObservedReader:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stream.close()

        def read(self, count=-1):
            read_sizes.append(count)
            assert 0 <= count <= 192  # no full image payload reads
            return self.stream.read(count)

        def tell(self):
            return self.stream.tell()

        def seek(self, *args):
            return self.stream.seek(*args)

    def observe(path, *args, **kwargs):
        stream = original_open(path, *args, **kwargs)
        return ObservedReader(stream) if path == sensor else stream

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", observe)
        result = read_calibration(sensor, metadata, {0, 4})
    assert result.frame_count == 5 and set(result.poses) == {0, 4}
    np.testing.assert_array_equal(result.intrinsic, _export_roundtrip(intrinsic)[:3, :3])
    np.testing.assert_array_equal(result.axis_align_matrix, axis)
    for frame in (0, 4):
        expected = _export_roundtrip(poses[frame])
        np.testing.assert_array_equal(result.camera_to_world[frame], expected)
        np.testing.assert_array_equal(result.poses[frame], axis @ expected)
        assert result.poses[frame].dtype == np.float32
    assert sum(read_sizes) < 1000
    assert {path: path.read_bytes() for path in before} == before


def test_native_sensor_precision_is_available_for_provenance_comparison(sensor_scene):
    sensor, metadata, intrinsic, axis, poses = sensor_scene
    native = read_calibration(sensor, metadata, {0}, roundtrip_text=False)
    exported = read_calibration(sensor, metadata, {0})
    np.testing.assert_array_equal(native.intrinsic, intrinsic[:3, :3])
    np.testing.assert_array_equal(native.camera_to_world[0], poses[0])
    np.testing.assert_array_equal(native.poses[0], axis @ poses[0])
    assert not np.array_equal(native.camera_to_world[0], exported.camera_to_world[0])


@pytest.mark.parametrize("frames", [set(), {-1}, {5}, {0.5}, {True}, {2}])
def test_invalid_requests_or_selected_pose_fail(sensor_scene, frames):
    sensor, metadata, *_ = sensor_scene
    with pytest.raises(ValidationError):
        read_calibration(sensor, metadata, frames)


@pytest.mark.parametrize("contents", [
    "colorWidth = 1296\n",
    "axisAlignment = 1 2 3\n",
    "axisAlignment = " + " ".join(["nan"] * 16),
    "axisAlignment = " + " ".join(["no"] * 16),
    "axisAlignment = " + " ".join(["0"] * 16),
    ("axisAlignment = " + " ".join(str(v) for v in np.eye(4).flat) + "\n") * 2,
])
def test_missing_or_invalid_alignment_never_falls_back_to_identity(tmp_path, contents):
    metadata = tmp_path / "scene.txt"
    metadata.write_text(contents)
    with pytest.raises(ValidationError):
        read_axis_alignment(metadata)


@pytest.mark.parametrize("cut", [3, 30, -1])
def test_truncated_sensor_metadata_or_payload_is_rejected(sensor_scene, cut):
    sensor, metadata, *_ = sensor_scene
    sensor.write_bytes(sensor.read_bytes()[:cut])
    with pytest.raises(ValidationError, match="truncated"):
        read_calibration(sensor, metadata, {4})
