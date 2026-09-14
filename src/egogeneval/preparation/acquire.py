"""Plan official downloads and extract only the benchmark's frozen source files."""

from __future__ import annotations

import hashlib
import io
import os
import re
import tempfile
import sys
import zipfile
import zlib
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from http.client import IncompleteRead
from pathlib import Path
from threading import Event
from urllib.request import Request as URLRequest, urlopen

from ..data import validate_manifest
from ..errors import ValidationError
from ..io import write_json
from .sources import requests_from_rows
from .raw_scannetpp import official_scene_id

RAW_DATASETS = ("hypersim", "matterport3d", "scannet", "scannetpp")
DEFAULT_DATASETS = ("hypersim", "matterport3d", "scannet")
HYPERSIM_URL = "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/{}.zip"


def source_plan(manifest: Path, datasets: list[str] | None = None, *, matterport_calibration: str = "native") -> dict:
    if matterport_calibration not in {"native", "embodiedscan"}:
        raise ValidationError("Matterport calibration must be native or embodiedscan")
    index = validate_manifest(manifest)
    selected = set(datasets or DEFAULT_DATASETS)
    available = {r["dataset"] for r in index.rows}
    if not selected or selected - set(RAW_DATASETS) or selected - available:
        raise ValidationError(f"select available original sources: {sorted(available & set(RAW_DATASETS))}")
    rows = tuple(r for r in index.rows if r["dataset"] in selected)
    requests = requests_from_rows(rows)
    entries = {}
    for dataset in sorted(selected):
        entries[dataset] = {"scene_ids": set(), "files": {}, "archive_types": []}

    def add(dataset, path, *alternatives):
        entries[dataset]["files"][path] = {"path": path, "alternatives": list(alternatives)}

    for request in requests:
        dataset, ref = request.dataset, request.relative
        if dataset == "scannet":
            scene = request.scene_id
            entries[dataset]["scene_ids"].add(scene)
            for suffix in ("sens", "txt"):
                add(dataset, f"scans/{scene}/{scene}.{suffix}", f"scans_test/{scene}/{scene}.{suffix}")
        elif dataset == "matterport3d":
            house = ref.split("/")[1]
            entries[dataset]["scene_ids"].add(house)
            base = f"scans/{house}"
            match = re.fullmatch(r"(.+)_i(\d+)_(\d+)\.jpg", Path(ref).name)
            if not match:
                raise ValidationError(f"unexpected Matterport image: {ref}")
            panorama, camera, yaw = match.groups()
            add(dataset, ref)
            add(dataset, f"{base}/matterport_depth_images/{panorama}_d{camera}_{yaw}.png")
            if matterport_calibration == "embodiedscan":
                continue
            for folder, tag, indices in [
                ("matterport_camera_intrinsics", "intrinsics", camera),
                ("matterport_camera_poses", "pose", f"{camera}_{yaw}"),
            ]:
                add(dataset, f"{base}/{folder}/{panorama}_{tag}_{indices}.txt",
                    f"{base}/{folder}/{panorama}_{tag}{indices}.txt")
        elif dataset == "scannetpp":
            scene = official_scene_id(request.scene_id)
            entries[dataset]["scene_ids"].add(scene)
            base = f"data/{scene}"
            stem = Path(ref).stem
            add(dataset, f"{base}/dslr/resized_images/{stem}.JPG")
            add(dataset, f"{base}/dslr/resized_anon_masks/{stem}.png")
            add(dataset, f"{base}/dslr/nerfstudio/transforms.json")
            add(dataset, f"{base}/scans/mesh_aligned_0.05.ply")
        else:
            scene, camera = request.scene_id.split("/")
            entries[dataset]["scene_ids"].add(scene)
            base = f"evermotion_dataset/scenes/{scene}"
            number = re.fullmatch(r"frame\.(\d+)\.color\.jpg", Path(ref).name)
            if not number:
                raise ValidationError(f"unexpected Hypersim image: {ref}")
            add(dataset, ref)
            add(dataset, f"{base}/images/scene_{camera}_geometry_hdf5/frame.{number[1]}.depth_meters.hdf5")
            add(dataset, f"{base}/_detail/metadata_scene.csv")
            for name in ("positions", "orientations"):
                add(dataset, f"{base}/_detail/{camera}/camera_keyframe_{name}.hdf5")
    for dataset, entry in entries.items():
        entry["scene_ids"] = sorted(entry["scene_ids"])
        entry["files"] = [entry["files"][key] for key in sorted(entry["files"])]
        entry["scene_count"] = len(entry["scene_ids"])
        if dataset == "hypersim":
            entry["archives"] = [{"scene_id": s, "url": HYPERSIM_URL.format(s)} for s in entry["scene_ids"]]
        if dataset == "scannetpp":
            entry["download_assets"] = ["dslr_resized_dir", "dslr_resized_mask_dir",
                                         "dslr_nerfstudio_transform_path", "scan_mesh_path"]
            entry["scene_aliases"] = {r.scene_id: official_scene_id(r.scene_id) for r in requests
                                       if r.dataset == dataset and r.scene_id != official_scene_id(r.scene_id)}
        if dataset == "matterport3d":
            entry["archive_types"] = ["matterport_color_images", "matterport_depth_images"]
            if matterport_calibration == "native":
                entry["archive_types"] += ["matterport_camera_intrinsics", "matterport_camera_poses"]
            else:
                entry["calibration_source"] = "official-embodiedscan-v1"
                entry["annotation_files"] = [f"embodiedscan_infos_{split}.pkl" for split in ("train", "val", "test")]
    return {"format_version": 1, "manifest_sha256": index.sha256,
            "selected_datasets": sorted(selected), "cases": len(rows), "frames": len(requests),
            "excluded_datasets": sorted(available - selected), "datasets": entries}


def _archive_names(dataset: str, files: list[dict]) -> dict[str, str]:
    names = {}
    prefix = "evermotion_dataset/scenes/" if dataset == "hypersim" else "scans/"
    for row in files:
        for path in [row["path"], *row["alternatives"]]:
            # Official ZIPs are rooted at <scene>/... or <house>/....
            names[path.removeprefix(prefix)] = path
    return names


def _extract(
    archive: zipfile.ZipFile, names: dict[str, str], output: Path,
    *, cancel: Event | None = None,
) -> list[dict]:
    records = []
    seen = set()
    for info in archive.infolist():
        if cancel is not None and cancel.is_set():
            raise CancelledError()
        member = info.filename.removeprefix("./")
        if member not in names:
            continue
        if member in seen:
            raise ValidationError(f"duplicate ZIP member: {member}")
        seen.add(member)
        if info.file_size > 128 * 1024 * 1024:
            raise ValidationError(f"unexpectedly large selected source file: {member}")
        destination = output / names[member]
        if not destination.resolve().is_relative_to(output.resolve()):
            raise ValidationError(f"source output escapes destination: {destination}")
        if destination.is_symlink():
            raise ValidationError(f"refusing to overwrite source symlink: {destination}")
        if destination.exists():
            content = destination.read_bytes()
            if len(content) != info.file_size or zlib.crc32(content) != info.CRC:
                raise ValidationError(f"existing file differs from official archive: {destination}")
        else:
            content = archive.read(info)  # ZipFile verifies the member's CRC.
            destination.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(prefix=".download-", dir=destination.parent)
            try:
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(content)
                if cancel is not None and cancel.is_set():
                    raise CancelledError()
                os.replace(temporary, destination)
            finally:
                Path(temporary).unlink(missing_ok=True)
        records.append({"path": names[member], "bytes": len(content),
                        "sha256": hashlib.sha256(content).hexdigest()})
    return records


def unpack_sources(manifest: Path, dataset: str, archives: Path, output: Path, *, matterport_calibration: str = "native") -> dict:
    if dataset not in {"matterport3d", "hypersim"}:
        raise ValidationError("ZIP extraction supports matterport3d and hypersim; ScanNet downloads are .sens/.txt files")
    plan = source_plan(manifest, [dataset], matterport_calibration=matterport_calibration)
    entry = plan["datasets"][dataset]
    names = _archive_names(dataset, entry["files"])
    paths = sorted(archives.rglob("*.zip")) if archives.is_dir() else [archives]
    if not paths:
        raise ValidationError(f"no ZIP archives in {archives}")
    records = []
    for path in paths:
        try:
            with zipfile.ZipFile(path) as archive:
                records.extend(_extract(archive, names, output))
        except (zipfile.BadZipFile, EOFError, zlib.error) as error:
            raise ValidationError(f"invalid source ZIP {path}: {error}; verified files can be reused on retry") from error
    verified = {row["path"] for row in records}
    missing = [row["path"] for row in entry["files"]
               if not any(p in verified for p in [row["path"], *row["alternatives"]])]
    unverified = [row["path"] for row in entry["files"]
                  if row["path"] in missing and any((output / p).is_file()
                  for p in [row["path"], *row["alternatives"]])]
    report = {"status": "missing-inputs" if missing else "unpacked", "dataset": dataset,
              "manifest_sha256": plan["manifest_sha256"], "files": records, "missing": missing,
              "unverified_existing": unverified}
    if dataset == "matterport3d" and matterport_calibration == "embodiedscan":
        report["calibration_source"] = entry["calibration_source"]
        report["required_annotation_files"] = entry["annotation_files"]
    write_json(output / "source_unpack_report.json", report)
    return report


class _HTTPRangeFile(io.RawIOBase):
    """A bounded seekable HTTP reader for ZIPs served by the official provider."""

    def __init__(self, url: str, *, cancel: Event | None = None):
        self.url, self.position = url, 0
        self.cancel = cancel
        self._check_cancel()
        self.cached_start, self.cached = 0, b""
        with urlopen(URLRequest(url, headers={"Range": "bytes=-22", "Accept-Encoding": "identity"}), timeout=60) as response:
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
            if response.status != 206 or not match:
                raise ValidationError("server does not support partial ZIP downloads; download the archive and use unpack-sources")
            self.size = int(match[3])
            response.read(22)

    def _check_cancel(self):
        if self.cancel is not None and self.cancel.is_set():
            raise CancelledError()

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        new = offset + (self.position if whence == 1 else self.size if whence == 2 else 0)
        if whence not in {0, 1, 2} or new < 0:
            raise ValueError("invalid ZIP seek")
        self.position = new
        return new

    def read(self, size=-1):
        self._check_cancel()
        size = max(0, self.size - self.position) if size is None or size < 0 else min(size, max(0, self.size - self.position))
        if size == 0:
            return b""
        if size > 128 * 1024 * 1024:
            raise ValidationError("unexpectedly large ZIP range request")
        offset = self.position - self.cached_start
        if not 0 <= offset <= len(self.cached) - size:
            start = self.position
            end = min(self.size, start + max(size, 64 * 1024)) - 1
            request = URLRequest(self.url, headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"})
            with urlopen(request, timeout=60) as response:
                if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{self.size}":
                    raise ValidationError("official server returned an unexpected ZIP byte range")
                self.cached = response.read(end - start + 2)
                if len(self.cached) != end - start + 1:
                    raise ValidationError("truncated ZIP byte range")
            self.cached_start, offset = start, 0
        self._check_cancel()
        self.position += size
        return self.cached[offset:offset + size]


def download_hypersim(manifest: Path, output: Path, *, workers: int = 4) -> dict:
    if not 1 <= workers <= 16:
        raise ValidationError("workers must be between 1 and 16")
    plan = source_plan(manifest, ["hypersim"])
    entry = plan["datasets"]["hypersim"]
    names = _archive_names("hypersim", entry["files"])
    cancelled = Event()

    def download(scene):
        url = HYPERSIM_URL.format(scene)
        selected = {name: path for name, path in names.items() if name.startswith(scene + "/")}
        try:
            with _HTTPRangeFile(url, cancel=cancelled) as remote, zipfile.ZipFile(remote) as archive:
                records = _extract(archive, selected, output, cancel=cancelled)
        except (zipfile.BadZipFile, EOFError, zlib.error, IncompleteRead) as error:
            raise ValidationError(f"invalid official Hypersim ZIP for {scene}: {error}; verified files can be reused on retry") from error
        if {r["path"] for r in records} != set(selected.values()):
            raise ValidationError(f"{scene}: requested files missing from the official public ZIP")
        print(f"Hypersim {scene}: verified {len(records)} official source files", file=sys.stderr, flush=True)
        return {"scene_id": scene, "url": url, "files": records}

    pool = ThreadPoolExecutor(max_workers=workers)
    pending = {pool.submit(download, scene): scene for scene in entry["scene_ids"]}
    completed = {}
    try:
        for future in as_completed(pending):
            completed[pending[future]] = future.result()
    except BaseException:
        cancelled.set()
        for future in pending:
            future.cancel()
        # Running requests keep their finite timeout; no subsequent requests or
        # writes begin after cancellation. Await them before permitting a retry.
        pool.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        pool.shutdown(wait=True)
    scenes = [completed[scene] for scene in entry["scene_ids"]]
    report = {"status": "downloaded", "dataset": "hypersim", "manifest_sha256": plan["manifest_sha256"],
              "scene_count": len(scenes), "files": sum(len(s["files"]) for s in scenes), "scenes": scenes}
    write_json(output / "source_download_report.json", report)
    return report
