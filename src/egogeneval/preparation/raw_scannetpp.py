"""Reconstruct ScanNet++ DSLR RGB-D from official images, calibration and meshes."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ..errors import ValidationError

# Preserve canonical case IDs when resolving the renamed official scene.
SCENE_ALIASES = {"7104910700": "7104910700000"}
CAMERA_FIELDS = ("fl_x", "fl_y", "cx", "cy", "k1", "k2", "k3", "k4")
CAMERA_CONVENTIONS = Path(__file__).resolve().parents[1] / "resources/scannetpp_camera_conventions.json"


def official_scene_id(scene: str) -> str:
    return SCENE_ALIASES.get(scene, scene)


@lru_cache(maxsize=1)
def camera_conventions() -> dict[str, tuple[float, frozenset[str]]]:
    """Frozen pixel-center choices; contains no images, depths or GT matrices.

    Historical DSLR metadata and depth exports used different center conventions
    for some frames. Keep those choices separate to reproduce existing assets.
    """
    try:
        data = json.loads(CAMERA_CONVENTIONS.read_text())
        if data["schema"] != "egogeneval-scannetpp-camera-conventions-v1":
            raise ValueError("unsupported schema")
        result = {}
        for scene, profile in data["scenes"].items():
            offset = profile["intrinsics_center_offset"]
            stems = profile["depth_half_pixel_frames"]
            if (type(offset) not in (int, float) or offset != 0.5
                    or not isinstance(stems, list) or len(stems) != len(set(stems))
                    or any(not isinstance(stem, str) or not stem
                           or Path(stem).name != stem or "\\" in stem for stem in stems)):
                raise ValueError(f"invalid camera convention for {scene}")
            result[scene] = float(offset), frozenset(stems)
        return result
    except (OSError, KeyError, TypeError, ValueError, AttributeError) as error:
        raise ValidationError(f"cannot read bundled ScanNet++ camera conventions: {error}") from error


def legacy_fisheye_matrix(K: np.ndarray, distortion: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Keep the historical OpenCV <=4.6 focal-length calculation.

    Later OpenCV releases changed the aspect-ratio adjustment of the center.
    Preserve the original calculation so installed OpenCV versions do not move
    the frozen benchmark's pixels. The DSLR recipe then centers the principal
    point, as in CUT3R's ScanNet++ preprocessing.
    """
    width, height = size
    samples = np.array([[[width // 2, 0], [width, height // 2],
                         [width // 2, height], [0, height // 2]]], dtype=np.float64)
    points = cv2.fisheye.undistortPoints(samples, K, distortion, R=np.eye(3)).reshape(4, 2)
    center = points.mean(axis=0)
    aspect = K[0, 0] / K[1, 1]
    center[0] *= aspect
    points[:, 1] *= aspect
    lo, hi = points.min(axis=0), points.max(axis=0)
    denominators = np.array([center[0] - lo[0], hi[0] - center[0],
                             center[1] - lo[1], hi[1] - center[1]])
    if not np.isfinite(denominators).all() or np.any(denominators <= 0):
        raise ValidationError("ScanNet++ fisheye calibration has no valid projection")
    focal = max(width * .5 / denominators[0], width * .5 / denominators[1],
                height * .5 * aspect / denominators[2], height * .5 * aspect / denominators[3])
    return np.array([[focal, 0., width / 2.],
                     [0., focal / aspect, height / 2.], [0., 0., 1.]])


@dataclass
class RawScene:
    directory: Path
    metadata: Path
    mesh: Path
    width: int
    height: int
    camera: np.ndarray
    poses: dict[str, tuple[str, np.ndarray]]
    intrinsics_center_offset: float = 0.0
    depth_half_pixel_frames: frozenset[str] = field(default_factory=frozenset)
    renderer_info: dict[str, str] = field(default_factory=dict, init=False)

    @classmethod
    def read(cls, root: Path, scene: str) -> RawScene:
        base = root / "data" if (root / "data").is_dir() else root
        directory = base / scene
        if not directory.is_dir():
            directory = base / official_scene_id(scene)
        metadata = directory / "dslr/nerfstudio/transforms.json"
        data = json.loads(metadata.read_text())
        if data.get("camera_model") != "OPENCV_FISHEYE":
            raise ValidationError(f"{metadata}: use original fisheye DSLR data, not undistorted images")
        width, height = data["w"], data["h"]
        if any(isinstance(v, bool) or not isinstance(v, int) or not 1 <= v <= 16384 for v in (width, height)):
            raise ValidationError(f"{metadata}: invalid image dimensions")
        camera = np.asarray([data[key] for key in CAMERA_FIELDS], dtype=np.float64)
        if not np.isfinite(camera).all() or np.any(camera[:2] <= 0):
            raise ValidationError(f"{metadata}: invalid fisheye parameters")
        poses = {}
        for frame in data["frames"] + data.get("test_frames", []):
            name = frame["file_path"]
            if (not isinstance(name, str) or Path(name).name != name or "\\" in name
                    or Path(name).suffix != ".JPG" or Path(name).stem in poses):
                raise ValidationError(f"{metadata}: unsafe or duplicate DSLR frame name")
            if any(key in frame and frame[key] != data[key] for key in CAMERA_FIELDS):
                raise ValidationError(f"{metadata}: per-frame camera overrides are unsupported")
            pose = np.asarray(frame["transform_matrix"], dtype=np.float64)
            if (pose.shape != (4, 4) or not np.isfinite(pose).all()
                    or not np.allclose(pose[3], [0, 0, 0, 1])
                    or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-5)
                    or not np.isclose(np.linalg.det(pose[:3, :3]), 1., atol=1e-5)):
                raise ValidationError(f"{metadata}: invalid camera transform for {name}")
            # Reverse the official toolbox's COLMAP -> Nerfstudio conversion.
            pose[2, :] *= -1
            pose = pose[[1, 0, 2, 3], :]
            pose[:3, 1:3] *= -1
            poses[Path(name).stem] = name, pose
        if not poses:
            raise ValidationError(f"{metadata}: no DSLR frames")
        offset, depth_frames = camera_conventions().get(official_scene_id(scene), (0.0, frozenset()))
        return cls(directory, metadata, directory / "scans/mesh_aligned_0.05.ply",
                   width, height, camera, poses, offset, depth_frames)

    def files(self, stem: str) -> tuple[Path, Path]:
        if stem not in self.poses:
            raise ValidationError(f"{self.metadata}: no official calibration for {stem}")
        name = self.poses[stem][0]
        return (self.directory / "dslr/resized_images" / name,
                self.directory / "dslr/resized_anon_masks" / Path(name).with_suffix(".png"))

    def projection(self) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        fx, fy, cx, cy = self.camera[:4]
        K = np.array([[fx, 0., cx - .5], [0., fy, cy - .5], [0., 0., 1.]])
        undistorted = legacy_fisheye_matrix(K, self.camera[4:], (self.width, self.height))
        scale = max(920. / self.width, 690. / self.height) + 1e-8
        size = np.floor(np.array([self.width, self.height]) * scale).astype(int)
        return K, undistorted, scale, size

    def intrinsic(self, *, center_offset: float | None = None) -> np.ndarray:
        """Metadata K, or an explicit render convention, before evaluator resize."""
        _, intrinsic, scale, size = self.projection()
        intrinsic[:2, :] *= scale
        intrinsic[:2, 2] -= .5 * (np.array([self.width, self.height]) * scale - size)
        offset = self.intrinsics_center_offset if center_offset is None else center_offset
        intrinsic[:2, 2] += offset * scale
        return intrinsic

    def depth_intrinsic(self, stem: str) -> np.ndarray:
        """Depth exports may use a different center from the stored metadata K."""
        self.files(stem)
        return self.intrinsic(center_offset=0.5 if stem in self.depth_half_pixel_frames else 0.0)

    def reconstruct(self, stems: list[str], output: Path) -> dict[str, tuple[Path, Path]]:
        if sys.platform != "linux":
            raise ValidationError("exact ScanNet++ depth reconstruction requires the validated Linux OpenGL environment")
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
        os.environ.setdefault("GALLIUM_DRIVER", "llvmpipe")
        os.environ.setdefault("LP_NUM_THREADS", "4")
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        if os.environ["PYOPENGL_PLATFORM"] != "egl":
            raise ValidationError("set PYOPENGL_PLATFORM=egl for ScanNet++ reconstruction")
        try:
            import pyrender
            import trimesh
        except ImportError as error:
            raise ValidationError("install the ScanNet++ preparation dependencies: pip install -e '.[prepare,scannetpp]'") from error
        K, undistorted, scale, size = self.projection()
        mx, my = cv2.fisheye.initUndistortRectifyMap(
            K, self.camera[4:], np.eye(3), undistorted, (self.width, self.height), cv2.CV_32FC1,
        )
        with self.mesh.open("rb") as stream:
            mesh = trimesh.Trimesh(**trimesh.exchange.ply.load_ply(stream))
        world = pyrender.Scene()
        world.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))
        (output / "images").mkdir(parents=True)
        (output / "depth").mkdir()
        renderer = pyrender.OffscreenRenderer(int(size[0]), int(size[1]))
        results = {}
        try:
            from OpenGL.GL import GL_RENDERER, GL_VENDOR, GL_VERSION, glGetString

            self.renderer_info = {
                name: glGetString(value).decode("utf-8")
                for name, value in (("vendor", GL_VENDOR), ("renderer", GL_RENDERER),
                                    ("version", GL_VERSION))
            }
            for stem in stems:
                rgb_path, mask_path = self.files(stem)
                with Image.open(rgb_path) as source:
                    rgb = np.array(source.convert("RGB"))
                with Image.open(mask_path) as source:
                    mask = np.array(source.convert("L"))
                if rgb.shape != (self.height, self.width, 3) or mask.shape != (self.height, self.width):
                    raise ValidationError(f"{rgb_path}: original RGB/mask dimensions disagree with calibration")
                rgb = cv2.remap(rgb, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
                mask = cv2.remap(mask, mx, my, cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=255)
                rgb = Image.fromarray(rgb).resize(tuple(size), Image.Resampling.LANCZOS if scale < 1 else Image.Resampling.BICUBIC)
                mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
                target = output / "images" / f"{stem}.jpg"
                rgb.save(target, quality=75, subsampling=2)
                intrinsic = self.depth_intrinsic(stem)
                camera = pyrender.IntrinsicsCamera(intrinsic[0, 0], intrinsic[1, 1],
                                                   intrinsic[0, 2], intrinsic[1, 2], znear=.05, zfar=20.)
                node = world.add(camera, pose=self.poses[stem][1] @ np.diag([1., -1., -1., 1.]))
                try:
                    depth = renderer.render(world, flags=pyrender.RenderFlags.DEPTH_ONLY)
                finally:
                    world.remove_node(node)
                depth = (depth * 1000).astype(np.uint16)
                depth[mask < 255] = 0
                depth_path = output / "depth" / f"{stem}.png"
                Image.fromarray(depth).save(depth_path)
                results[stem] = target, depth_path
        finally:
            renderer.delete()
        return results


def reconstruct_scene(job):
    """Run one independent scene in a spawned EGL worker."""
    scene, stems, output = job
    rebuilt = scene.reconstruct(stems, output)
    return rebuilt, scene.renderer_info
