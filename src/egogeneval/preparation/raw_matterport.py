"""Read the native Matterport3D camera files distributed with RGB-D images.

Format reference (upstream commit 8cd7c81aff5824d578caa8cf5b79e2896bfe7f34):
https://github.com/niessner/Matterport/blob/8cd7c81aff5824d578caa8cf5b79e2896bfe7f34/data_organization.md

These are the cameras for ``matterport_color_images``, not the separately
undistorted images. Poses already map OpenCV camera coordinates to native world
coordinates in metres. Reading them must not invert, axis-align, rescale, or
orthogonalize them. Lens-distortion coefficients are retained for provenance;
the frozen benchmark uses the native images and pinhole K without undistortion.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..errors import ValidationError


@dataclass(frozen=True)
class NativeCalibration:
    pose: np.ndarray
    intrinsic: np.ndarray
    pose_path: Path
    intrinsic_path: Path
    image_size: tuple[int, int]  # width, height before benchmark resizing
    distortion: np.ndarray  # k1, k2, p1, p2, k3; deliberately not applied


def _one_file(directory: Path, names: tuple[str, str]) -> Path:
    # Upstream describes <uuid>_<type><camera>[_<yaw>].txt. Also accept the
    # separator before the camera index used by native dataset exports. Neither
    # choice may silently win when both are present: calibration is provenance.
    candidates = [directory / name for name in names]
    found = [path for path in candidates if path.is_file()]
    if not found:
        raise ValidationError(
            f"missing official Matterport3D camera file; expected one of "
            f"{', '.join(str(path) for path in candidates)}. Download and extract "
            f"{directory.name}.zip for this house."
        )
    if len(found) != 1:
        raise ValidationError(
            f"ambiguous official Matterport3D camera files: "
            f"{', '.join(str(path) for path in found)}"
        )
    return found[0]


def _values(path: Path, count: int) -> np.ndarray:
    try:
        # Parse tokens independently of line wrapping. A single float32 cast
        # preserves the evaluator's camera dtype without arithmetic or rounding.
        values = np.asarray(path.read_text(encoding="utf-8").split(), dtype=np.float32)
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValidationError(f"cannot read Matterport3D camera file {path}: {exc}") from exc
    if values.size != count or not np.isfinite(values).all():
        raise ValidationError(f"{path}: expected {count} finite numeric values")
    return values


def read_native_calibration(rgb: Path) -> NativeCalibration:
    """Resolve native calibration by the RGB filename, never its frame ordinal.

    ``rgb`` must have the official ``<house>/matterport_color_images/...``
    layout; its image bytes are read later by the reconstruction pipeline.
    All returned numeric arrays use float32, matching the frozen evaluator.
    Exact agreement with a historical annotation still needs reference checks;
    the public EmbodiedScan repository does not expose its MP3D camera exporter.
    """
    rgb = Path(rgb)
    match = re.fullmatch(r"(.+)_i([0-2])_([0-5])\.jpg", rgb.name)
    if rgb.parent.name != "matterport_color_images" or match is None:
        raise ValidationError(
            f"expected a native Matterport3D RGB path "
            f"<house>/matterport_color_images/<panorama>_i<camera>_<yaw>.jpg: {rgb}"
        )
    panorama, camera, yaw = match.groups()
    house = rgb.parent.parent
    pose_path = _one_file(
        house / "matterport_camera_poses",
        (f"{panorama}_pose_{camera}_{yaw}.txt", f"{panorama}_pose{camera}_{yaw}.txt"),
    )
    intrinsic_path = _one_file(
        house / "matterport_camera_intrinsics",
        (f"{panorama}_intrinsics_{camera}.txt", f"{panorama}_intrinsics{camera}.txt"),
    )
    pose = _values(pose_path, 16).reshape(4, 4)
    if (
        not np.array_equal(pose[3], np.array([0, 0, 0, 1], dtype=np.float32))
        or abs(np.linalg.det(pose[:3, :3])) < 1e-8
    ):
        raise ValidationError(f"{pose_path}: invalid camera-to-world transform")
    values = _values(intrinsic_path, 11)
    width, height, fx, fy, cx, cy = values[:6]
    if (
        min(width, height) <= 0
        or width != np.floor(width)
        or height != np.floor(height)
        or min(fx, fy) <= 0
    ):
        raise ValidationError(
            f"{intrinsic_path}: expected positive integer width/height and positive fx/fy"
        )
    intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    return NativeCalibration(
        pose=pose,
        intrinsic=intrinsic,
        pose_path=pose_path,
        intrinsic_path=intrinsic_path,
        image_size=(int(width), int(height)),
        distortion=values[6:].copy(),
    )
