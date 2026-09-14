"""Exercise the scoring join with the committed benchmark and frozen labels.

Tiny local asset files stand in for RGB-D and model outputs. No model inference
or source data download is needed to verify case/step/label selection.
"""

import copy
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from egogeneval.data import validate_manifest
from egogeneval.errors import ValidationError
from egogeneval.io import read_jsonl, sha256_file, write_json, write_jsonl
from egogeneval.scoring.pipeline import FROZEN_LABELS_NAME, _asset_provenance, build_official_input


REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "data/manifests/egogeneval_v0.1.jsonl"
LABELS = REPO / "data/evaluator" / FROZEN_LABELS_NAME
SELECTED = {"hypersim", "matterport3d", "scannet"}


@pytest.fixture
def scoring_inputs(tmp_path):
    canonical = validate_manifest(MANIFEST)
    Image.new("RGB", (2, 2)).save(tmp_path / "image.png")
    np.save(tmp_path / "depth.npy", np.ones((2, 2), dtype=np.float32))
    refs = {
        image["image_path"]
        for row in canonical.rows
        for field in ("input_images", "target_images")
        for image in row[field]
    }
    assets = tmp_path / "eval_frames.jsonl"
    write_jsonl(assets, [
        {"image_ref": ref, "image_path": "image.png", "depth_path": "depth.npy"}
        for ref in sorted(refs)
    ])

    def prepare(datasets=SELECTED):
        rows = [r for r in canonical.rows if datasets is None or r["dataset"] in datasets]
        manifest = tmp_path / "manifest.jsonl"
        write_jsonl(manifest, rows)
        predictions = tmp_path / "predictions.jsonl"
        write_jsonl(predictions, [
            {"case_id": row["sample_id"], "step_id": instruction["step"],
             "model_id": "test-model", "output_path": "image.png"}
            for row in rows for instruction in row["instructions"]
        ])
        return {
            "predictions": predictions,
            "manifest": validate_manifest(manifest),
            "prepared_data": None,
            "frozen_labels": LABELS,
            "eval_frames_index": assets,
        }

    return prepare


@pytest.mark.parametrize("datasets,cases,steps", [(SELECTED, 397, 556), (None, 1400, 2360)])
def test_frozen_bank_joins_selected_cases_without_changing_labels(
    scoring_inputs, datasets, cases, steps
):
    inputs = scoring_inputs(datasets)
    before = sha256_file(LABELS)
    joined, model_id = build_official_input(**inputs)
    expected = {row["sample_id"]: row["target_labels"] for row in read_jsonl(LABELS)}
    assert model_id == "test-model"
    assert len(joined) == cases
    assert [row["id"] for row in joined] == [row["sample_id"] for row in inputs["manifest"].rows]
    assert sum(len(row["generated_images"]) for row in joined) == steps
    for row in joined:
        assert [target["qwen3vl_gt_object_labels"] for target in row["target_images"]] == expected[row["id"]]
        assert [image["step"] for image in row["generated_images"]] == list(range(1, len(row["target_images"]) + 1))
        assert all(Path(image["image_path"]).is_file() for image in row["input_images"] + row["target_images"])
    assert sha256_file(LABELS) == before


def test_missing_selected_label_is_rejected(scoring_inputs, tmp_path):
    inputs = scoring_inputs()
    missing = inputs["manifest"].rows[0]["sample_id"]
    labels = tmp_path / "missing-labels.jsonl"
    write_jsonl(labels, [row for row in read_jsonl(LABELS) if row["sample_id"] != missing])
    inputs["frozen_labels"] = labels
    with pytest.raises(ValidationError, match="frozen-label IDs missing"):
        build_official_input(**inputs)


@pytest.mark.parametrize("invalid", ["step-count", "empty-labels", "malformed-labels"])
def test_invalid_selected_step_labels_are_rejected(scoring_inputs, tmp_path, invalid):
    inputs = scoring_inputs()
    selected = next(row for row in inputs["manifest"].rows if len(row["target_images"]) > 1)
    rows = copy.deepcopy(read_jsonl(LABELS))
    label = next(row for row in rows if row["sample_id"] == selected["sample_id"])
    if invalid == "step-count":
        label["target_labels"].pop()
        message = "frozen-label count"
    elif invalid == "empty-labels":
        label["target_labels"][1]["detector_prompt_labels"] = []
        label["target_labels"][1]["object_labels"] = []
        message = "step 2: frozen detector labels are empty"
    else:
        label["target_labels"][1] = None
        message = "step 2: malformed frozen labels"
    labels = tmp_path / "invalid-labels.jsonl"
    write_jsonl(labels, rows)
    inputs["frozen_labels"] = labels
    with pytest.raises(ValidationError, match=message):
        build_official_input(**inputs)


def test_depth_revision_is_recorded_and_modified_index_rejected(tmp_path):
    index = tmp_path / "eval_frames.jsonl"
    write_jsonl(index, [{"asset_revision": "scannetpp-depth-v2", "depth_sha256": "a" * 64}])
    write_json(tmp_path / "build_report.json", {
        "asset_revision": "scannetpp-depth-v2",
        "artifact_sha256": {"eval_frames.jsonl": sha256_file(index)},
    })
    assert _asset_provenance(index) == {
        "asset_revision": "scannetpp-depth-v2", "eval_frames_sha256": sha256_file(index),
    }
    write_jsonl(index, [{"asset_revision": "scannetpp-depth-v2", "depth_sha256": "b" * 64}])
    with pytest.raises(ValidationError, match="artifact changed"):
        _asset_provenance(index)


def test_mixed_depth_revisions_are_rejected(tmp_path):
    index = tmp_path / "eval_frames.jsonl"
    write_jsonl(index, [{"asset_revision": "paper-v0.1"}, {"asset_revision": "scannetpp-depth-v2"}])
    with pytest.raises(ValidationError, match="mixed asset revisions"):
        _asset_provenance(index)
