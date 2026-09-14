#!/usr/bin/env python3
"""Load one physical target RGB-D frame with the benchmark's dataset rules."""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import cv2
import h5py
import numpy as np

try:
    from ..evaluation.eval_assets import load_eval_asset_frame
except ImportError:
    from eval_assets import load_eval_asset_frame


def _half(array: np.ndarray) -> np.ndarray:
    return cv2.resize(array, dsize=None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)


def _clip_98(depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    depth[~np.isfinite(depth)] = 0.0
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    valid = depth > 0
    if np.any(valid):
        depth[depth > np.percentile(depth[valid], 98)] = 0.0
    return depth


def _read_rgb(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"cannot read target RGB: {path}")
    return image


def _read_depth_png(path: Path, scale: float) -> np.ndarray:
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(f"cannot read target depth: {path}")
    return raw.astype(np.float32) / float(scale)


def _scannetpp(path: str) -> SimpleNamespace:
    rgb = _read_rgb(path)
    rgb_path = Path(path)
    depth_path = rgb_path.parent.parent / "depth" / f"{rgb_path.stem}.png"
    depth = _clip_98(_read_depth_png(depth_path, 1000.0))
    return SimpleNamespace(image=_half(rgb), depth=_half(depth).astype(np.float32))


def _scannet(path: str) -> SimpleNamespace:
    rgb_raw = _read_rgb(path)
    rgb = _half(rgb_raw)
    depth = _read_depth_png(Path(path).with_suffix(".png"), 1000.0)
    depth[~np.isfinite(depth)] = 0.0
    if depth.shape[:2] != rgb.shape[:2]:
        depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    depth = _clip_98(depth)
    # Preserve the established ScanNet loader's second half-resolution step.
    return SimpleNamespace(image=_half(rgb), depth=_half(depth).astype(np.float32))


def _hypersim(path: str) -> SimpleNamespace:
    rgb = _read_rgb(path)
    rgb_path = Path(path)
    match = re.match(r"frame\.(\d+)\.color\.jpg$", rgb_path.name)
    if not match:
        raise ValueError(f"unexpected Hypersim RGB name: {path}")
    geometry_dir = rgb_path.parent.parent / rgb_path.parent.name.replace(
        "_final_preview", "_geometry_hdf5"
    )
    depth_path = geometry_dir / f"frame.{int(match.group(1)):04d}.depth_meters.hdf5"
    if not depth_path.is_file():
        raise FileNotFoundError(f"cannot read target depth: {depth_path}")
    with h5py.File(depth_path, "r") as handle:
        key = "dataset" if "dataset" in handle else next(iter(handle.keys()))
        depth = np.asarray(handle[key][()])
    depth = _clip_98(depth)
    rgb = _half(rgb)
    depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    return SimpleNamespace(image=rgb, depth=depth.astype(np.float32))


def _matterport(path: str) -> SimpleNamespace:
    rgb = _read_rgb(path)
    rgb_path = Path(path)
    depth_name = re.sub(r"_i([0-9]+_[0-9]+)\.jpg$", r"_d\1.png", rgb_path.name)
    if depth_name == rgb_path.name:
        raise ValueError(f"unexpected Matterport3D RGB name: {path}")
    depth_path = rgb_path.parent.parent / "matterport_depth_images" / depth_name
    depth = _read_depth_png(depth_path, 4000.0)
    return SimpleNamespace(image=_half(rgb), depth=_half(depth).astype(np.float32))


def direct_target_rgbd(dataset: str, path: str) -> SimpleNamespace:
    private_asset = load_eval_asset_frame(path)
    if private_asset is not None:
        return SimpleNamespace(image=private_asset.image, depth=private_asset.depth)
    normalized = str(dataset or "").lower()
    if normalized == "scannetpp":
        return _scannetpp(path)
    if normalized == "scannet":
        return _scannet(path)
    if normalized == "hypersim":
        return _hypersim(path)
    if normalized in {"matterport3d", "matterport", "mp3d"}:
        return _matterport(path)
    raise KeyError(f"direct target RGB-D loader does not support dataset={dataset!r}")
