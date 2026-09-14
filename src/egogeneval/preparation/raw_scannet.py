"""Read ScanNet camera calibration directly from official scene downloads.

The .sens v4 layout follows ScanNet's SensorData.py. EmbodiedScan exports its
color intrinsics and camera_to_world matrices using ``np.savetxt(fmt='%f')``;
the default below replays that six-decimal text round trip in memory before
float32 axis alignment. No images, pickle annotations, or intermediate files
are needed, and compressed RGB-D payloads are skipped rather than decoded.
"""

from __future__ import annotations

import io
import struct
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..errors import ValidationError


@dataclass(frozen=True)
class ScanNetCalibration:
    """Calibration at the source RGB resolution, before evaluator resizing."""

    intrinsic: np.ndarray
    axis_align_matrix: np.ndarray
    camera_to_world: dict[int, np.ndarray]
    poses: dict[int, np.ndarray]
    frame_count: int


def _transform(matrix: np.ndarray, label: str) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if (
        matrix.shape != (4, 4)
        or not np.isfinite(matrix).all()
        or not np.array_equal(matrix[3], [0, 0, 0, 1])
        or abs(np.linalg.det(matrix[:3, :3])) < 1e-8
    ):
        raise ValidationError(f"{label}: expected a finite, invertible 4x4 camera transform")
    return matrix


def read_axis_alignment(scene_metadata: Path) -> np.ndarray:
    """Read exactly one axisAlignment entry from the official <scene>.txt.

    Missing alignment is an error; silently using identity would change the
    benchmark world coordinates. Decimal values are converted to float32 once.
    """
    matches = []
    for line in scene_metadata.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator and key.strip() == "axisAlignment":
            matches.append(value.split())
    if len(matches) != 1 or len(matches[0]) != 16:
        raise ValidationError(f"{scene_metadata}: expected one axisAlignment entry with 16 values")
    try:
        matrix = np.asarray(matches[0], dtype=np.float32).reshape(4, 4)
    except ValueError as exc:
        raise ValidationError(f"{scene_metadata}: invalid axisAlignment values") from exc
    return _transform(matrix, f"{scene_metadata}: axisAlignment")


def _text_roundtrip(matrix: np.ndarray) -> np.ndarray:
    """Replay the official/EmbodiedScan pose export without writing to disk."""
    text = io.StringIO()
    np.savetxt(text, matrix, fmt="%f")
    text.seek(0)
    return np.loadtxt(text, dtype=np.float32)


def read_calibration(
    sensor: Path,
    scene_metadata: Path,
    frames: Iterable[int],
    *,
    roundtrip_text: bool = True,
) -> ScanNetCalibration:
    """Read requested zero-based sensor frames, not benchmark loader ordinals.

    ``poses[i]`` is float32 ``axisAlignment @ camera_to_world[i]``. Intrinsics
    are the source color camera's 3x3 matrix; apply the evaluator's resize stages
    after this call. Sensor color/depth extrinsics are not composed into poses,
    matching the upstream export of camera_to_world directly.

    ``roundtrip_text=False`` exposes native sensor precision for provenance
    comparison; the default follows the exported annotations' text precision.
    Only metadata through the largest requested frame is visited. Invalid poses
    in unrequested frames are permitted because such frames are often filtered.
    """
    requested = set(frames)
    if not requested or any(type(frame) is not int or frame < 0 for frame in requested):
        raise ValidationError(f"{sensor}: expected nonnegative integer sensor frame numbers")
    axis = read_axis_alignment(scene_metadata)
    size = sensor.stat().st_size
    with sensor.open("rb") as stream:

        def read(count: int) -> bytes:
            if count < 0 or count > size - stream.tell():
                raise ValidationError(f"{sensor}: truncated or invalid .sens metadata")
            payload = stream.read(count)
            if len(payload) != count:
                raise ValidationError(f"{sensor}: truncated .sens metadata")
            return payload

        def unpack(fmt: str):
            return struct.unpack("<" + fmt, read(struct.calcsize("<" + fmt)))

        def read_matrix() -> np.ndarray:
            return np.frombuffer(read(64), dtype="<f4").reshape(4, 4).astype(np.float32)

        if unpack("I")[0] != 4:
            raise ValidationError(f"{sensor}: only ScanNet .sens version 4 is supported")
        name_size = unpack("Q")[0]
        if name_size > size - stream.tell():
            raise ValidationError(f"{sensor}: truncated sensor name")
        stream.seek(name_size, 1)
        intrinsic = read_matrix()
        read(3 * 64)  # color extrinsics, depth intrinsics, depth extrinsics
        color_codec, depth_codec = unpack("ii")
        dimensions = unpack("IIII")
        depth_shift = unpack("f")[0]
        frame_count = unpack("Q")[0]
        if (color_codec, depth_codec) != (2, 1):
            raise ValidationError(f"{sensor}: expected JPEG color and zlib_ushort depth")
        if min(dimensions) <= 0 or depth_shift != 1000.0:
            raise ValidationError(f"{sensor}: unexpected image dimensions or depth scale")
        if max(requested) >= frame_count:
            raise ValidationError(f"{sensor}: requested frame {max(requested)} but only {frame_count} exist")
        if roundtrip_text:
            intrinsic = _text_roundtrip(intrinsic)
        intrinsic = intrinsic[:3, :3].copy()
        if not np.isfinite(intrinsic).all() or intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0:
            raise ValidationError(f"{sensor}: invalid color intrinsic matrix")

        camera_to_world, poses = {}, {}
        for frame in range(max(requested) + 1):
            native_pose = read_matrix()
            read(2 * 8)  # RGB/depth timestamps
            color_size, depth_size = unpack("QQ")
            if color_size + depth_size > size - stream.tell():
                raise ValidationError(f"{sensor}: truncated RGB-D payload at frame {frame}")
            stream.seek(color_size + depth_size, 1)
            if frame not in requested:
                continue
            pose = _text_roundtrip(native_pose) if roundtrip_text else native_pose
            pose = _transform(pose, f"{sensor}: frame {frame}")
            camera_to_world[frame] = pose
            poses[frame] = _transform(axis @ pose, f"{sensor}: aligned frame {frame}")

    return ScanNetCalibration(intrinsic, axis, camera_to_world, poses, frame_count)
