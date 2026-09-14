"""Official annotations can replace MP camera TXT without changing raw ScanNet."""

import json
import shutil
import zipfile

import numpy as np
import pytest

from egogeneval.cli import main
from egogeneval.errors import ValidationError
from egogeneval.preparation.acquire import source_plan, unpack_sources
from egogeneval.preparation.build import build_data, verify_data
from test_local_data import (
    _raw_fixture_fingerprints,
    dump_pickle,
    raw_source_set as _raw_source_set,
    source_set as _source_set,
)

source_set = _source_set
raw_source_set = _raw_source_set


def annotations(house, path, *, omit_last=False, offset=0):
    images = []
    for number in range(1 if omit_last else 2):
        pose = np.loadtxt(house / "matterport_camera_poses" / f"view{number}_pose1_2.txt")
        pose[0, 3] += offset
        images.append({"img_path": f"matterport3d/house/matterport_color_images/view{number}_i1_2.jpg",
                       "cam2global": pose})
    scene = {"sample_idx": "matterport3d/house/region0", "images": images,
             "cam2img": [[15, 0, 10], [0, 17, 8], [0, 0, 1]],
             "axis_align_matrix": np.full((4, 4), np.nan)}
    # Invalid ScanNet metadata must be ignored by the MP-specific option.
    scannet = {"sample_idx": "scannet/scene0000_00", "images": [
        {"img_path": "scannet/posed_images/scene0000_00/00000.jpg"}
    ]}
    dump_pickle(path, {"data_list": [scannet, scene]})
    return path


def test_rgbd_zip_and_official_mp_annotations_preserve_raw_scannet(raw_source_set, tmp_path):
    manifest, roots, _, _, house, *_ = raw_source_set
    fingerprints = _raw_fixture_fingerprints(raw_source_set, tmp_path)
    annotation = annotations(house, tmp_path / "official-v1.pkl")
    plan = source_plan(manifest, matterport_calibration="embodiedscan")
    mp = plan["datasets"]["matterport3d"]
    assert mp["archive_types"] == ["matterport_color_images", "matterport_depth_images"]
    assert len(mp["files"]) == 4 and len(mp["annotation_files"]) == 3
    archive = tmp_path / "mp-rgbd.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        for row in mp["files"]:
            handle.write(roots["MATTERPORT3D_ROOT"] / row["path"], row["path"].removeprefix("scans/"))
    fresh = tmp_path / "downloaded-mp"
    report = unpack_sources(manifest, "matterport3d", archive, fresh,
                            matterport_calibration="embodiedscan")
    assert report["status"] == "unpacked" and not report["missing"]
    assert len(report["required_annotation_files"]) == 3
    for folder in ("matterport_camera_poses", "matterport_camera_intrinsics"):
        shutil.rmtree(house / folder)
    roots = {**roots, "MATTERPORT3D_ROOT": fresh}
    output = tmp_path / "mixed-built"
    args = ["build-data", "--manifest", str(manifest), "--raw-sources",
            "--matterport-embodiedscan-info", str(annotation),
            "--fingerprints", str(fingerprints), "--output", str(output)]
    for dataset in ("scannet", "matterport3d", "hypersim"):
        args += ["--dataset", dataset]
    for name, path in roots.items():
        args += ["--root", f"{name}={path}"]
    assert main(args) == 0
    built = json.loads((output / "build_report.json").read_text())
    assert built["matterport_calibration"] == "embodiedscan-v1"
    assert built["calibration_sources"] == {
        "scannet-native-sens": 2, "embodiedscan-matterport": 2, "hypersim-official-layout": 2,
    }
    assert built["reference_assets_verified"] and verify_data(output)["fingerprints_checked"]


def test_missing_mp_annotation_never_falls_back_to_native_camera(raw_source_set, tmp_path):
    manifest, roots, _, _, house, *_ = raw_source_set
    annotation = annotations(house, tmp_path / "missing.pkl", omit_last=True)
    with pytest.raises(ValidationError, match="no matching Matterport3D calibration"):
        build_data(manifest, roots, tmp_path / "missing", raw_sources=True,
                   datasets=["matterport3d"], matterport_annotations=[annotation])
    assert not (tmp_path / "missing").exists()


def test_conflicting_official_mp_annotations_fail(raw_source_set, tmp_path):
    manifest, roots, _, _, house, *_ = raw_source_set
    paths = [annotations(house, tmp_path / "first.pkl"),
             annotations(house, tmp_path / "second.pkl", offset=1)]
    with pytest.raises(ValidationError, match="conflicting EmbodiedScan calibration"):
        build_data(manifest, roots, tmp_path / "conflict", raw_sources=True,
                   datasets=["matterport3d"], matterport_annotations=paths)


def test_mp_specific_option_requires_selected_mp(source_set, tmp_path):
    manifest, roots, annotation = source_set
    with pytest.raises(ValidationError, match="requires --dataset matterport3d"):
        build_data(manifest, roots, tmp_path / "wrong-dataset", datasets=["hypersim"],
                   matterport_annotations=[annotation])
