"""Read calibration from EmbodiedScan annotations or existing local metadata.

RGB-D transforms live in the frozen evaluator. This module only resolves source
files and camera metadata; it never substitutes estimated camera parameters for
missing annotations. Pickles must come from a trusted upstream/team source.
"""

from __future__ import annotations

import pickle
import re
import csv
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import cv2
import h5py
import numpy as np

from ..data import ROOT_TOKEN
from ..errors import ValidationError
from ..official.evaluation import dataloader as official
from .raw_scannet import read_calibration
from .raw_matterport import read_native_calibration
from .parallel import scene_jobs
from .raw_scannetpp import CAMERA_CONVENTIONS, RawScene

DATASET_ROOTS = {
    "scannet": "SCANNET_ROOT",
    "matterport3d": "MATTERPORT3D_ROOT",
    "scannetpp": "SCANNETPP_ROOT",
    "hypersim": "HYPERSIM_ROOT",
}


@dataclass(frozen=True)
class Request:
    dataset: str
    scene_id: str
    image_ref: str
    frame_id: int
    relative: str


@dataclass
class Source:
    rgb: Path
    depth: Path | None
    pose: np.ndarray
    intrinsic: np.ndarray
    metadata: Path
    calibration_source: str
    sensor: Path | None = None
    metadata_dependencies: tuple[Path, ...] = ()
    raw_scannetpp: RawScene | None = None


def requests_from_rows(rows: tuple[dict, ...]) -> list[Request]:
    requests: dict[str, Request] = {}
    indices: dict[tuple[str, str, int], str] = {}
    for row in rows:
        dataset, scene = str(row["dataset"]), str(row["scene_id"])
        if dataset not in DATASET_ROOTS:
            raise ValidationError(f"unsupported source dataset: {dataset}")
        if PurePosixPath(scene).is_absolute() or ".." in PurePosixPath(scene).parts:
            raise ValidationError(f"unsafe scene_id: {scene!r}")
        for image in row["input_images"] + row["target_images"]:
            ref = str(image["image_path"])
            match = ROOT_TOKEN.fullmatch(ref)
            if not match or match[1] != DATASET_ROOTS[dataset]:
                raise ValidationError(
                    f"{ref}: expected ${{{DATASET_ROOTS[dataset]}}}/..."
                )
            relative = match[2]
            if (
                ".." in PurePosixPath(relative).parts
                or PurePosixPath(relative).is_absolute()
            ):
                raise ValidationError(f"unsafe source path: {ref}")
            try:
                frame = int(str(image["frame_id"]))
            except ValueError as exc:
                raise ValidationError(
                    f"{ref}: frame_id must be a nonnegative integer"
                ) from exc
            if frame < 0:
                raise ValidationError(f"{ref}: frame_id must be a nonnegative integer")
            request = Request(dataset, scene, ref, frame, relative)
            if ref in requests and requests[ref] != request:
                raise ValidationError(f"conflicting scene/frame identity for {ref}")
            key = dataset, scene, frame
            if key in indices and indices[key] != ref:
                raise ValidationError(f"multiple images for scene/frame identity {key}")
            requests[ref], indices[key] = request, ref
    return sorted(requests.values(), key=lambda r: (r.dataset, r.scene_id, r.frame_id))


def matrix(value: Any, size: int, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if size == 3 and result.shape == (4, 4):
        result = result[:3, :3]
    if result.shape != (size, size) or not np.isfinite(result).all():
        raise ValidationError(f"{label}: expected finite {size}x{size} matrix")
    if size == 4 and (
        not np.allclose(result[3], [0, 0, 0, 1])
        or abs(np.linalg.det(result[:3, :3])) < 1e-8
    ):
        raise ValidationError(f"{label}: invalid camera transform")
    if size == 3 and (result[0, 0] <= 0 or result[1, 1] <= 0):
        raise ValidationError(f"{label}: nonpositive focal length")
    return result


def _image_key(text: str, dataset: str) -> str:
    """Normalize upstream prefixes, never infer an image by its frame ordinal."""
    parts = str(text).replace("\\", "/").split("/")
    if dataset in parts:
        parts = parts[parts.index(dataset) + 1 :]
    if dataset == "matterport3d" and parts[0] == "scans":
        parts = parts[1:]
    return "/".join(parts)


def _pickle(path: Path) -> dict:
    try:
        with path.open("rb") as stream:
            result = pickle.load(stream)
        if not isinstance(result, dict):
            raise ValueError("expected a dictionary")
        return result
    except (OSError, ValueError, pickle.UnpicklingError, EOFError) as exc:
        raise ValidationError(f"cannot read trusted annotation {path}: {exc}") from exc


class Sources:
    def __init__(
        self, roots: dict[str, Path], requests: list[Request], annotations: list[Path],
        *, raw_sources: bool = False, matterport_annotations: list[Path] | None = None,
    ):
        self.roots = roots
        self.requests = requests
        self.raw_sources = raw_sources
        self.matterport_annotations = bool(matterport_annotations)
        self.calibrations: dict[
            tuple[str, str, str], tuple[np.ndarray, np.ndarray, Path, str]
        ] = {}
        self.errors: list[str] = []
        self.cache: dict[tuple[str, str], Any] = {}
        self.native_scannet_errors: dict[str, str] = {}
        wanted = {
            (r.dataset, r.scene_id, _image_key(r.relative, r.dataset)) for r in requests
        }
        by_image: dict[tuple[str, str], list[tuple[str, str, str]]] = {}
        for key in wanted:
            by_image.setdefault((key[0], key[2]), []).append(key)
        annotation_inputs = [(path, None) for path in ([] if raw_sources else annotations)]
        annotation_inputs.extend((path, "matterport3d") for path in (matterport_annotations or []))
        for path, scope in annotation_inputs:
            path = path.expanduser().resolve()
            data = _pickle(path)
            if not isinstance(data.get("data_list"), list):
                raise ValidationError(f"{path}: expected EmbodiedScan data_list")
            for scene in data["data_list"]:
                sample = str(scene.get("sample_idx", ""))
                for image in scene.get("images", []):
                    image_path = str(image.get("img_path", ""))
                    dataset = sample.split("/")[0] or image_path.split("/")[0]
                    if dataset not in {"scannet", "matterport3d"}:
                        continue
                    if scope is not None and dataset != scope:
                        continue
                    for key in by_image.get(
                        (dataset, _image_key(image_path, dataset)), []
                    ):
                        # One MP3D image may be listed in several room annotations.
                        # Match the room, not only the image, and reject ambiguity.
                        region = re.search(r"(?:^|_)region(\d+)$", key[1])
                        if dataset == "matterport3d" and region:
                            if sample.split("/")[-1] != f"region{region[1]}":
                                continue
                        try:
                            pose = matrix(
                                image["cam2global"], 4, f"{sample}: pose"
                            )
                            # EgoGenEval's frozen MP3D loader uses native-world
                            # poses. Only ScanNet applies scene axis alignment.
                            if dataset == "scannet":
                                align = matrix(
                                    scene["axis_align_matrix"],
                                    4,
                                    f"{sample}: axis alignment",
                                )
                                pose = align @ pose
                            intrinsic = matrix(
                                image.get("cam2img", scene.get("cam2img")),
                                3,
                                f"{sample}: K",
                            )
                            previous = self.calibrations.get(key)
                            if previous and (
                                not np.array_equal(previous[0], pose)
                                or not np.array_equal(previous[1], intrinsic)
                            ):
                                raise ValidationError(
                                    f"conflicting EmbodiedScan calibration for {key}"
                                )
                            self.calibrations[key] = (
                                pose,
                                intrinsic,
                                path,
                                "embodiedscan-matterport" if scope else "embodiedscan",
                            )
                        except (KeyError, TypeError, ValueError) as exc:
                            self.errors.append(f"{path}: {key}: {exc}")

    def preload_native_scannet(self, workers: int) -> None:
        """Cache each independent scene once; retain errors for the input report."""
        if not self.raw_sources or workers == 1:
            return
        scenes = {}
        for request in self.requests:
            if request.dataset == "scannet":
                scenes.setdefault(request.scene_id, request)

        def prepare(request):
            try:
                self._native(request, Path(request.relative))
            except (ValidationError, OSError, KeyError, ValueError, IndexError, TypeError) as error:
                self.native_scannet_errors[request.scene_id] = str(error)

        scene_jobs(prepare, scenes.values(), workers)

    def _root(self, dataset: str) -> Path:
        name = DATASET_ROOTS[dataset]
        if name not in self.roots:
            raise ValidationError(f"missing --root {name}=/path/to/{dataset}")
        return self.roots[name]

    def _rgb(self, request: Request) -> Path:
        root = self._root(request.dataset)
        candidates = [root / request.relative]
        prefixes = {
            "matterport3d": ("scans/",),
            "scannetpp": ("scannetpp_processed/",),
            "hypersim": ("evermotion_dataset/", "evermotion_dataset/scenes/"),
        }
        for prefix in prefixes.get(request.dataset, ()):
            if request.relative.startswith(prefix):
                candidates.append(root / request.relative[len(prefix) :])
        return next((p for p in candidates if p.is_file()), candidates[0])

    def _legacy(self, request: Request) -> tuple[np.ndarray, np.ndarray, Path, str]:
        name = (
            "SCANNET_METADATA"
            if request.dataset == "scannet"
            else "MATTERPORT3D_METADATA"
        )
        if name not in self.roots:
            raise ValidationError(
                f"{request.dataset}/{request.scene_id}: missing calibration; supply "
                f"--embodiedscan-info /path/to/embodiedscan_infos_*.pkl or --root {name}=/path/to/metadata"
            )
        path = self.roots[name] / f"{request.scene_id}.pkl"
        cache_key = request.dataset, request.scene_id
        if cache_key not in self.cache:
            data = _pickle(path)
            mapped = {}
            if request.dataset == "scannet":
                for scene in data["data_list"]:
                    for image in scene["images"]:
                        key = _image_key(image["img_path"], request.dataset)
                        pose = matrix(
                            scene["axis_align_matrix"], 4, "axis alignment"
                        ) @ matrix(image["cam2global"], 4, "pose")
                        intrinsic = matrix(
                            image.get("cam2img", scene.get("cam2img")), 3, "K"
                        )
                        mapped[key] = pose, intrinsic
            else:
                paths, poses, intrinsics = (
                    data["image_paths"],
                    data["extrinsics_c2w"],
                    data["intrinsics"],
                )
                if not len(paths) == len(poses) == len(intrinsics):
                    raise ValidationError(
                        f"{path}: unequal image/pose/intrinsic lengths"
                    )
                for image, pose, intrinsic in zip(paths, poses, intrinsics):
                    mapped[_image_key(image, request.dataset)] = (
                        matrix(pose, 4, "pose"),
                        matrix(intrinsic, 3, "K"),
                    )
            self.cache[cache_key] = mapped
        key = _image_key(request.relative, request.dataset)
        if key not in self.cache[cache_key]:
            raise ValidationError(f"{path}: no calibration for {request.relative}")
        pose, intrinsic = self.cache[cache_key][key]
        return pose, intrinsic, path, "legacy-scene-metadata"

    def _sensor(self, request: Request) -> Path:
        root = self._root("scannet")
        candidates = [root / split / request.scene_id / f"{request.scene_id}.sens"
                      for split in ("scans", "scans_test")]
        return next((p for p in candidates if p.is_file()), candidates[0])

    def _native(self, request: Request, rgb: Path):
        if request.dataset == "scannet":
            if request.scene_id in self.native_scannet_errors:
                raise ValidationError(self.native_scannet_errors[request.scene_id])
            sensor = self._sensor(request)
            scene_info = sensor.with_suffix(".txt")
            key = "native-scannet", request.scene_id
            if key not in self.cache:
                numbers = {int(Path(r.relative).stem) for r in self.requests
                           if r.dataset == "scannet" and r.scene_id == request.scene_id}
                calibration = read_calibration(sensor, scene_info, numbers)
                self.cache[key] = calibration.intrinsic, calibration.poses
            intrinsic, poses = self.cache[key]
            return poses[int(rgb.stem)], matrix(intrinsic, 3, "native K"), sensor, "scannet-native-sens", (scene_info,)

        calibration = read_native_calibration(rgb)
        return (calibration.pose, calibration.intrinsic, calibration.pose_path,
                "matterport3d-native-camera", (calibration.intrinsic_path,))

    def resolve(self, request: Request) -> Source:
        if self.raw_sources and request.dataset == "scannetpp":
            key = "raw-scannetpp", request.scene_id
            if key not in self.cache:
                self.cache[key] = RawScene.read(self._root("scannetpp"), request.scene_id)
            scene = self.cache[key]
            stem = Path(request.relative).stem
            rgb, mask = scene.files(stem)
            return Source(
                rgb=rgb, depth=None, pose=matrix(scene.poses[stem][1], 4, "DSLR pose"),
                intrinsic=matrix(scene.intrinsic(), 3, "DSLR intrinsic"),
                metadata=scene.metadata, calibration_source="scannetpp-official-dslr",
                metadata_dependencies=(scene.mesh, rgb, mask, CAMERA_CONVENTIONS), raw_scannetpp=scene,
            )
        rgb = self._rgb(request)
        sensor = None
        dependencies: tuple[Path, ...] = ()
        if request.dataset in {"scannet", "matterport3d"}:
            key = (
                request.dataset,
                request.scene_id,
                _image_key(request.relative, request.dataset),
            )
            calibration = self.calibrations.get(key)
            legacy_name = "SCANNET_METADATA" if request.dataset == "scannet" else "MATTERPORT3D_METADATA"
            if request.dataset == "matterport3d" and self.matterport_annotations:
                if calibration is None:
                    raise ValidationError(
                        f"{request.scene_id}: no matching Matterport3D calibration in the supplied "
                        "EmbodiedScan annotations; use v1 annotations with native UUID image paths"
                    )
                pose, intrinsic, metadata, origin = calibration
            elif self.raw_sources or (calibration is None and legacy_name not in self.roots):
                pose, intrinsic, metadata, origin, dependencies = self._native(request, rgb)
            else:
                pose, intrinsic, metadata, origin = (
                    calibration if calibration is not None else self._legacy(request)
                )
            if request.dataset == "scannet":
                depth = rgb.with_suffix(".png")
                sensor = self._sensor(request)
            else:
                name = re.sub(r"_i(\d+_\d+)\.jpg$", r"_d\1.png", rgb.name)
                if name == rgb.name:
                    raise ValidationError(
                        f"unexpected Matterport3D image filename: {rgb}"
                    )
                depth = rgb.parent.parent / "matterport_depth_images" / name
        elif request.dataset == "scannetpp":
            metadata = rgb.parent.parent / "scene_metadata.npz"
            cache_key = request.dataset, request.scene_id
            if cache_key not in self.cache:
                with np.load(metadata, allow_pickle=False) as data:
                    names, poses, intrinsics = (
                        data["images"],
                        data["trajectories"],
                        data["intrinsics"],
                    )
                    # The frozen loader uses len(trajectories), not len(images).
                    # Allow trailing image/K entries without a trajectory while
                    # preserving the frozen loader's exact prefix selection;
                    # never truncate missing names or calibration for a pose.
                    count = len(poses)
                    if len(names) < count or len(intrinsics) < count:
                        raise ValidationError(
                            f"{metadata}: missing image or intrinsic for trajectory "
                            f"(images={len(names)}, poses={count}, intrinsics={len(intrinsics)})"
                        )
                    mapped = {}
                    for idx in range(count):
                        name, pose, intrinsic = names[idx], poses[idx], intrinsics[idx]
                        if isinstance(name, bytes):
                            name = name.decode()
                        stem = str(name).split(".")[0]
                        if stem in mapped:
                            raise ValidationError(f"{metadata}: duplicate image {stem}")
                        mapped[stem] = (
                            matrix(pose, 4, "pose"),
                            matrix(intrinsic, 3, "K"),
                        )
                    self.cache[cache_key] = mapped
            if rgb.stem not in self.cache[cache_key]:
                raise ValidationError(f"{metadata}: no calibration for {rgb.name}")
            pose, intrinsic = self.cache[cache_key][rgb.stem]
            depth = rgb.parent.parent / "depth" / f"{rgb.stem}.png"
            origin = "scannetpp-processed"
        else:
            scene = rgb.parent.parent.parent
            cam = re.fullmatch(r"scene_(cam_\d+)_final_preview", rgb.parent.name)
            frame = re.fullmatch(r"frame\.(\d+)\.color\.jpg", rgb.name)
            if not cam or not frame:
                raise ValidationError(f"unexpected Hypersim image path: {rgb}")
            frame_number = int(
                frame[1]
            )  # NOT manifest frame_id: filtered frames change its ordinal.
            metadata = scene / "_detail" / cam[1] / "camera_keyframe_positions.hdf5"
            orientation_path = metadata.with_name("camera_keyframe_orientations.hdf5")
            units_path = scene / "_detail" / "metadata_scene.csv"
            dependencies = (orientation_path, units_path)
            cache_key = request.dataset, request.scene_id
            if cache_key not in self.cache:

                def read(path: Path):
                    with h5py.File(path, "r") as handle:
                        return np.asarray(handle["dataset"][()], dtype=np.float32)

                with units_path.open(newline="", encoding="utf-8") as handle:
                    units = [
                        row[1]
                        for row in csv.reader(handle)
                        if len(row) >= 2 and row[0].strip() == "meters_per_asset_unit"
                    ]
                if (
                    len(units) != 1
                    or not np.isfinite(float(units[0]))
                    or float(units[0]) <= 0
                ):
                    raise ValidationError(
                        f"{units_path}: missing or invalid meters_per_asset_unit"
                    )
                positions = read(metadata) * float(units[0])
                orientations = read(orientation_path)
                self.cache[cache_key] = positions, orientations
            positions, orientations = self.cache[cache_key]
            if frame_number >= min(len(positions), len(orientations)):
                raise ValidationError(
                    f"{metadata}: missing source frame {frame_number}"
                )
            pose = np.eye(4, dtype=np.float32)
            pose[:3, :3], pose[:3, 3] = (
                orientations[frame_number],
                positions[frame_number],
            )
            pose = matrix(official.blender2opencv_c2w(pose), 4, "Hypersim OpenCV pose")
            image = (
                cv2.imread(str(rgb), cv2.IMREAD_UNCHANGED) if rgb.is_file() else None
            )
            if image is None:
                raise ValidationError(f"missing or unreadable RGB: {rgb}")
            intrinsic = official.hypersim_intrinsics(image.shape[1], image.shape[0])
            depth = (
                rgb.parent.parent
                / f"scene_{cam[1]}_geometry_hdf5"
                / f"frame.{frame_number:04d}.depth_meters.hdf5"
            )
            origin = "hypersim-official-layout"
        return Source(
            rgb, depth, pose, intrinsic, metadata, origin, sensor, dependencies
        )
