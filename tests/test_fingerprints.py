"""Frozen fingerprints authenticate reconstruction without publishing source data."""

import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from egogeneval.errors import ValidationError
from egogeneval.io import write_jsonl
from egogeneval.official.evaluation import eval_assets
from egogeneval.preparation.fingerprints import fingerprint_row, verify_fingerprints
from egogeneval.preparation.sources import requests_from_rows


@pytest.fixture
def asset_row():
    return {
        "asset_id": "fixture",
        "dataset": "scannet",
        "scene_id": "scene0000_00",
        "image_ref": "${SCANNET_ROOT}/posed_images/scene0000_00/00080.jpg",
        "loader_frame_idx": 8,
        "image_height": 120,
        "image_width": 160,
        "depth_height": 120,
        "depth_width": 160,
        "rgb_sha256": "a" * 64,
        "depth_sha256": "b" * 64,
        "intrinsics": [[144.125, 0, 79.5], [0, 145.25, 59.5], [0, 0, 1]],
        "extrinsics_c2w": np.eye(4).tolist(),
        "asset_format_version": eval_assets.ASSET_FORMAT_VERSION,
        "preprocessing": eval_assets.PREPROCESSING_CONTRACT_ID,
        "resize_stages": [0.5, 0.5],
        "image_path": "/private/source/rgb.jpg",
        "depth_path": "/private/source/depth.npy",
        "calibration_source": "/private/source/calibration.pkl",
    }


def test_export_is_hash_only_with_lossless_camera_encoding(asset_row):
    result = fingerprint_row(asset_row)
    assert all(isinstance(value, (str, int)) for value in result.values())
    assert not (
        {"intrinsics", "extrinsics_c2w", "image_path", "depth_path"} & result.keys()
    )
    assert "/private/" not in json.dumps(result)
    assert result["image_ref"] == asset_row["image_ref"]
    assert (
        result["effective_intrinsics_sha256"]
        == hashlib.sha256(
            np.asarray(asset_row["intrinsics"], dtype="<f8").tobytes(order="C")
        ).hexdigest()
    )
    # Byte order and an input array's native dtype must not affect the digest.
    other = copy.deepcopy(asset_row)
    other["intrinsics"] = np.asarray(other["intrinsics"], dtype=">f8")
    assert fingerprint_row(other) == result


@pytest.mark.parametrize("field", ["intrinsics", "extrinsics_c2w"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_one_ulp_camera_drift_is_rejected(asset_row, tmp_path, field, dtype):
    reference = tmp_path / "fingerprints.jsonl"
    write_jsonl(reference, [fingerprint_row(asset_row)])
    changed = copy.deepcopy(asset_row)
    changed[field][0][0] = float(
        np.nextafter(dtype(changed[field][0][0]), dtype(np.inf))
    )
    with pytest.raises(ValidationError, match="fingerprint mismatch.*effective_"):
        verify_fingerprints([changed], reference)


def test_historical_scannet_normalizes_to_same_effective_camera(asset_row, tmp_path):
    legacy = copy.deepcopy(asset_row)
    legacy["asset_format_version"] = eval_assets.LEGACY_ASSET_FORMAT_VERSION
    legacy["preprocessing"] = eval_assets.LEGACY_PREPROCESSING_CONTRACT_ID
    legacy.pop("resize_stages")
    for i, j in ((0, 0), (1, 1), (0, 2), (1, 2)):
        legacy["intrinsics"][i][j] *= 2
    _, stages = eval_assets._row_resize_contract(legacy)
    scorer_k = eval_assets._apply_intrinsics_stages(legacy["intrinsics"], stages)
    assert np.array_equal(scorer_k, asset_row["intrinsics"])
    assert fingerprint_row(legacy) == fingerprint_row(asset_row)
    reference = tmp_path / "fingerprints.jsonl"
    write_jsonl(reference, [fingerprint_row(legacy)])
    assert verify_fingerprints([asset_row], reference)["frames"] == 1


def test_selected_records_accept_superset_and_reject_missing(asset_row, tmp_path):
    other = copy.deepcopy(asset_row)
    other["image_ref"] = "${SCANNET_ROOT}/posed_images/scene0000_00/00170.jpg"
    other["loader_frame_idx"] = 17
    reference = tmp_path / "fingerprints.jsonl"
    write_jsonl(reference, [fingerprint_row(asset_row), fingerprint_row(other)])
    report = verify_fingerprints([asset_row], reference)
    assert report == {
        "status": "verified",
        "frames": 1,
        "reference_frames": 2,
        "reference_checked": True,
    }
    write_jsonl(reference, [fingerprint_row(other)])
    with pytest.raises(ValidationError, match="has no entry"):
        verify_fingerprints([asset_row], reference)


def test_duplicate_references_or_selected_rows_are_rejected(asset_row, tmp_path):
    reference = tmp_path / "fingerprints.jsonl"
    frozen = fingerprint_row(asset_row)
    write_jsonl(reference, [frozen, frozen])
    with pytest.raises(ValidationError, match="reference: duplicate image_ref"):
        verify_fingerprints([asset_row], reference)
    write_jsonl(reference, [frozen])
    with pytest.raises(ValidationError, match="asset index: duplicate image_ref"):
        verify_fingerprints([asset_row, asset_row], reference)


@pytest.mark.parametrize(
    ("field", "value"),
    [("depth_sha256", "c" * 64), ("image_height", 121), ("loader_frame_idx", 9)],
)
def test_asset_hash_dimensions_and_identity_must_match(
    asset_row, tmp_path, field, value
):
    reference = tmp_path / "fingerprints.jsonl"
    write_jsonl(reference, [fingerprint_row(asset_row)])
    asset_row[field] = value
    with pytest.raises(ValidationError, match=f"fingerprint mismatch.*{field}"):
        verify_fingerprints([asset_row], reference)


def test_empty_or_non_public_reference_rejected(asset_row, tmp_path):
    reference = tmp_path / "fingerprints.jsonl"
    write_jsonl(reference, [])
    with pytest.raises(ValidationError, match="empty frozen fingerprint"):
        verify_fingerprints([asset_row], reference)
    frozen = fingerprint_row(asset_row)
    frozen["intrinsics"] = asset_row["intrinsics"]
    write_jsonl(reference, [frozen])
    with pytest.raises(ValidationError, match="unexpected fields"):
        verify_fingerprints([asset_row], reference)
    write_jsonl(reference, [fingerprint_row(asset_row)])
    with pytest.raises(ValidationError, match="empty eval-frames"):
        verify_fingerprints([], reference)


def test_absolute_reference_and_invalid_contract_rejected(asset_row):
    asset_row["image_ref"] = "/private/source/rgb.jpg"
    with pytest.raises(ValidationError, match="canonical dataset root"):
        fingerprint_row(asset_row)
    asset_row["image_ref"] = "${SCANNET_ROOT}/posed_images/scene0000_00/00080.jpg"
    asset_row["resize_stages"] = [0.25]
    with pytest.raises(ValidationError, match="do not match the paper contract"):
        fingerprint_row(asset_row)


def test_all_735_canonical_frame_identities_can_be_fingerprinted(asset_row, tmp_path):
    """Use real manifest identities, with synthetic hashes/cameras (no source data)."""
    manifest = (
        Path(__file__).resolve().parents[1] / "data/manifests/egogeneval_v0.1.jsonl"
    )
    selected = [
        row
        for line in manifest.read_text().splitlines()
        if (row := json.loads(line))["dataset"]
        in {"hypersim", "matterport3d", "scannet"}
    ]
    assert len(selected) == 397
    requests = requests_from_rows(selected)
    assert Counter(request.dataset for request in requests) == {
        "hypersim": 250,
        "matterport3d": 37,
        "scannet": 448,
    }
    records = []
    for request in requests:
        row = copy.deepcopy(asset_row)
        row.update(
            {
                "dataset": request.dataset,
                "scene_id": request.scene_id,
                "image_ref": request.image_ref,
                "loader_frame_idx": request.frame_id,
                "resize_stages": list(
                    eval_assets.resize_stages_for_dataset(request.dataset)
                ),
            }
        )
        records.append(row)
    reference = tmp_path / "fingerprints.jsonl"
    fingerprints = [fingerprint_row(row) for row in records]
    assert any(
        row["dataset"] == "hypersim" and "/cam_" in row["scene_id"]
        for row in fingerprints
    )
    write_jsonl(reference, fingerprints)
    assert verify_fingerprints(records, reference)["frames"] == 735


def test_bundled_revision_covers_the_entire_benchmark():
    from egogeneval.preparation.prepare import _bundled_fingerprints, _references

    manifest = Path(__file__).resolve().parents[1] / "data/manifests/egogeneval_v0.1.jsonl"
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    requests = requests_from_rows(rows)
    expected = {request.image_ref for request in requests}
    references = _references(_bundled_fingerprints(), expected)
    assert len(expected) == len(references) == 3675
    assert Counter(row["dataset"] for row in references) == {
        "hypersim": 250, "matterport3d": 37, "scannet": 448, "scannetpp": 2940,
    }


@pytest.mark.parametrize(
    "scene", ["/private/source", "../camera", "scene/../camera", "C:/source", "."]
)
def test_unsafe_scene_identity_is_rejected(asset_row, scene):
    asset_row["scene_id"] = scene
    with pytest.raises(ValidationError, match="invalid fingerprint scene_id"):
        fingerprint_row(asset_row)
