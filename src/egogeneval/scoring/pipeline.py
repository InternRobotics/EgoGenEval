"""Official one-command evaluator: generated images -> CMG/SSP -> scores."""

from __future__ import annotations

import copy
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from ..aggregation.results import summarize_records
from ..data import ManifestIndex, _resolve_source, validate_manifest
from ..errors import ValidationError
from ..io import read_json, read_jsonl, sha256_file, write_json, write_jsonl
from ..predictions import validate_predictions
from ..versioning import provenance

FROZEN_LABELS_NAME = "frozen_qwen3vl_gt_object_labels_v0.1.jsonl"


@dataclass(frozen=True)
class EvaluatorSettings:
    """Resolved model/runtime settings for the paper evaluator."""

    da3_code: Path
    da3_model: Path
    grounding_dino_model: Path
    dinov3_model: Path
    vlm_model: str
    device: str = "cuda:0"
    vlm_backend: str = "local_transformers"
    vlm_base_url: str = ""
    vlm_api_key_env: str = ""

    @classmethod
    def resolve(
        cls,
        config: str | Path | None = None,
        **overrides: Any,
    ) -> "EvaluatorSettings":
        values: dict[str, Any] = {}
        if config is not None:
            config_path = Path(config).expanduser().resolve()
            payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            values = payload.get("evaluator", payload)
            if not isinstance(values, dict):
                raise ValidationError(f"{config_path}: evaluator config must be a mapping")

        environment = {
            "da3_code": "DA3_CODE_DIR",
            "da3_model": "DA3_MODEL_DIR",
            "grounding_dino_model": "GROUNDING_DINO_MODEL",
            "dinov3_model": "DINOV3_MODEL",
            "vlm_model": "VLM_MODEL",
            "device": "EGOGENEVAL_DEVICE",
            "vlm_backend": "VLM_BACKEND",
            "vlm_base_url": "VLM_BASE_URL",
            "vlm_api_key_env": "VLM_API_KEY_ENV",
        }
        defaults = {
            "device": "cuda:0",
            "vlm_backend": "local_transformers",
            "vlm_base_url": "",
            "vlm_api_key_env": "",
        }
        resolved: dict[str, Any] = {}
        for key, env_name in environment.items():
            override = overrides.get(key)
            resolved[key] = (
                override
                if override not in (None, "")
                else os.environ.get(env_name, values.get(key, defaults.get(key)))
            )

        required = (
            "da3_code",
            "da3_model",
            "grounding_dino_model",
            "dinov3_model",
            "vlm_model",
        )
        missing = [key for key in required if resolved.get(key) in (None, "")]
        if missing:
            raise ValidationError(
                "missing evaluator setting(s): "
                + ", ".join(missing)
                + "; pass --evaluator-config, CLI flags, or the documented environment variables"
            )

        path_keys = ("da3_code", "da3_model", "grounding_dino_model", "dinov3_model")
        for key in path_keys:
            path = Path(str(resolved[key])).expanduser().resolve()
            if not path.exists():
                raise ValidationError(f"{key} does not exist: {path}")
            resolved[key] = path

        da3_code = Path(resolved["da3_code"])
        if (da3_code / "src" / "depth_anything_3").is_dir():
            resolved["da3_code"] = da3_code / "src"
        elif not (da3_code / "depth_anything_3").is_dir():
            raise ValidationError(
                f"da3_code does not contain depth_anything_3 (directly or under src): {da3_code}"
            )

        if str(resolved["vlm_backend"]).lower() == "local_transformers":
            vlm_path = Path(str(resolved["vlm_model"])).expanduser().resolve()
            if not vlm_path.exists():
                raise ValidationError(f"local vlm_model does not exist: {vlm_path}")
            resolved["vlm_model"] = str(vlm_path)

        return cls(**resolved)


def _frozen_labels_candidates(manifest: Path) -> list[Path]:
    """Where to look for the frozen label bank when ``--frozen-labels`` is absent.

    The manifest-relative guess only works for the repo layout
    (``data/manifests/`` next to ``data/evaluator/``). A manifest reconstructed by
    ``download-data`` lives at ``data/hf/test/`` instead, so fall back to the
    checkout's ``data/evaluator/`` — relative to the working directory, then
    relative to this package for an editable install.
    """

    repo_root = Path(__file__).resolve().parents[3]
    return [
        (manifest.parent.parent / "evaluator" / FROZEN_LABELS_NAME).resolve(),
        (Path.cwd() / "data" / "evaluator" / FROZEN_LABELS_NAME).resolve(),
        (repo_root / "data" / "evaluator" / FROZEN_LABELS_NAME).resolve(),
    ]


def _default_frozen_labels(manifest: Path) -> Path:
    candidates = _frozen_labels_candidates(manifest)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    searched = "\n  ".join(str(candidate) for candidate in dict.fromkeys(candidates))
    raise ValidationError(
        f"frozen per-step Qwen3-VL labels ({FROZEN_LABELS_NAME}) were not found. "
        f"Pass --frozen-labels explicitly. Searched:\n  {searched}"
    )


def _load_frozen_labels(path: Path) -> dict[str, list[dict[str, Any]]]:
    if not path.is_file():
        raise ValidationError(
            f"frozen per-step Qwen3-VL labels are missing: {path}; "
            "SSP must not fall back to the fixed-36 vocabulary"
        )
    rows = read_jsonl(path)
    labels: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        targets = row.get("target_labels")
        if not sample_id or not isinstance(targets, list) or not targets:
            raise ValidationError(f"{path}: malformed frozen-label row for {sample_id!r}")
        if sample_id in labels:
            raise ValidationError(f"{path}: duplicate frozen-label sample_id {sample_id}")
        labels[sample_id] = targets
    return labels


def _prepared_lookup(prepared_data: Path | None) -> dict[str, dict[str, Any]]:
    if prepared_data is None:
        return {}
    prepared_manifest = prepared_data / "prepared_manifest.jsonl"
    if not prepared_manifest.is_file():
        raise ValidationError(f"prepared manifest is missing: {prepared_manifest}")
    rows = read_jsonl(prepared_manifest)
    return {str(row["sample_id"]): row for row in rows}


def _eval_frame_lookup(index_path: Path | None) -> dict[str, dict[str, Any]]:
    if index_path is None:
        return {}
    rows = read_jsonl(index_path)
    lookup: dict[str, dict[str, Any]] = {}
    for row in rows:
        reference = str(row.get("image_ref") or "")
        relative = str(row.get("image_path") or "")
        if not reference or not relative:
            raise ValidationError(f"{index_path}: malformed eval-frame row")
        if reference in lookup:
            raise ValidationError(f"{index_path}: duplicate image_ref {reference}")
        image_path = (index_path.parent / relative).resolve()
        depth_path = (index_path.parent / str(row.get("depth_path") or "")).resolve()
        if not image_path.is_file() or not depth_path.is_file():
            raise ValidationError(
                f"{reference}: materialized RGB-D asset is missing: {image_path} / {depth_path}"
            )
        lookup[reference] = {**row, "resolved_image_path": str(image_path)}
    if not lookup:
        raise ValidationError(f"{index_path}: eval-frame index is empty")
    return lookup


def _asset_provenance(index_path: Path | None) -> dict[str, str]:
    """Identify the actual GT revision separately from the unchanged case list."""
    if index_path is None:
        return {"asset_revision": "unversioned-source-data"}
    rows = read_jsonl(index_path)
    values = [row.get("asset_revision", "unversioned-eval-frames") for row in rows]
    if not values or not all(isinstance(v, str) and v for v in values) or len(set(values)) != 1:
        raise ValidationError("eval-frame index contains missing or mixed asset revisions")
    result = {
        "asset_revision": values[0],
        "eval_frames_sha256": sha256_file(index_path),
    }
    references = index_path.parent / "reference_fingerprints.jsonl"
    if references.is_file():
        result["reference_fingerprints_sha256"] = sha256_file(references)
    report_path = index_path.parent / "build_report.json"
    if report_path.is_file():
        report = read_json(report_path)
        if report.get("asset_revision", "unversioned-eval-frames") != result["asset_revision"]:
            raise ValidationError("asset revision differs from build_report.json")
        for name, field in (("eval_frames.jsonl", "eval_frames_sha256"),
                            ("reference_fingerprints.jsonl", "reference_fingerprints_sha256")):
            expected = report.get("artifact_sha256", {}).get(name)
            if expected is not None and result.get(field) != expected:
                raise ValidationError(f"prepared benchmark artifact changed: {name}")
    return result


def _resolve_case_images(
    row: dict[str, Any],
    *,
    prepared_data: Path | None,
    eval_frames: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    resolved = copy.deepcopy(row)
    for field in ("input_images", "target_images"):
        for image in resolved[field]:
            path_text = str(image["image_path"])
            if eval_frames:
                asset = eval_frames.get(path_text)
                if asset is None:
                    raise ValidationError(f"eval-frame asset is missing for {path_text}")
                path = Path(asset["resolved_image_path"])
            elif prepared_data is None:
                path = _resolve_source(path_text, {})
            else:
                candidate = Path(path_text).expanduser()
                relative_prepared = not candidate.is_absolute()
                unresolved = candidate if not relative_prepared else prepared_data / candidate
                if (
                    field == "target_images"
                    and relative_prepared
                    and str(row.get("dataset")) != "synthetic"
                    and not unresolved.is_symlink()
                ):
                    raise ValidationError(
                        f"{unresolved}: copied target RGB cannot resolve the source physical "
                        "depth; prepare with symlink mode or omit --prepared-data and export roots"
                    )
                path = unresolved
                path = path.resolve()
                if not path.is_file():
                    raise ValidationError(f"prepared image is missing: {path}")
            image["image_path"] = str(path)
    return resolved


def build_official_input(
    predictions: Path,
    manifest: ManifestIndex,
    *,
    prepared_data: Path | None,
    frozen_labels: Path,
    eval_frames_index: Path | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """Join validated predictions, source images, and frozen SSP annotations."""

    prediction_rows = read_jsonl(predictions)
    prediction_by_key = {(str(row["case_id"]), int(row["step_id"])): row for row in prediction_rows}
    model_id = str(prediction_rows[0]["model_id"])
    prepared = _prepared_lookup(prepared_data)
    eval_frames = _eval_frame_lookup(eval_frames_index)
    label_lookup = _load_frozen_labels(frozen_labels)
    manifest_ids = {str(row["sample_id"]) for row in manifest.rows}
    # A dataset subset uses the same frozen label bank as the full benchmark.
    # Only selected cases enter evaluation, but every selected ID is required.
    missing = sorted(manifest_ids - set(label_lookup))
    if missing:
        raise ValidationError(
            f"frozen-label IDs missing from manifest selection: {missing[:1]}"
        )

    output: list[dict[str, Any]] = []
    for public_row in manifest.rows:
        sample_id = str(public_row["sample_id"])
        source_row = prepared.get(sample_id, public_row)
        case = _resolve_case_images(
            source_row,
            prepared_data=prepared_data,
            eval_frames=eval_frames,
        )
        targets = case["target_images"]
        target_labels = label_lookup[sample_id]
        if len(target_labels) != len(targets):
            raise ValidationError(
                f"{sample_id}: frozen-label count {len(target_labels)} != targets {len(targets)}"
            )
        for step, (target, annotation) in enumerate(zip(targets, target_labels), start=1):
            if not isinstance(annotation, dict):
                raise ValidationError(f"{sample_id} step {step}: malformed frozen labels")
            labels = annotation.get("detector_prompt_labels") or annotation.get("object_labels")
            if not isinstance(labels, list) or not labels:
                raise ValidationError(f"{sample_id} step {step}: frozen detector labels are empty")
            target["qwen3vl_gt_object_labels"] = annotation

        generated = []
        for instruction in public_row["instructions"]:
            step = int(instruction["step"])
            prediction = prediction_by_key[(sample_id, step)]
            generated_path = (predictions.parent / str(prediction["output_path"])).resolve()
            generated.append({"step": step, "image_path": str(generated_path)})
        case["id"] = sample_id
        case["model"] = model_id
        case["generated_images"] = generated
        output.append(case)
    return output, model_id


def _official_root() -> Path:
    return Path(__file__).resolve().parents[1] / "official"


def _official_launcher() -> Path:
    """Path to the shim that sanitises ``cv2``'s ``LD_LIBRARY_PATH`` rewrite.

    See :mod:`egogeneval.scoring._official_launcher`. Invoked by absolute path
    rather than ``-m`` so the vendored stages do not depend on ``egogeneval``
    being importable from the subprocess's ``sys.path``.
    """

    return Path(__file__).resolve().parent / "_official_launcher.py"


def _runtime_environment(
    settings: EvaluatorSettings, eval_frames_index: Path | None = None
) -> dict[str, str]:
    environment = os.environ.copy()
    official = _official_root()
    python_path = [str(official / "evaluation"), str(official / "scripts")]
    if environment.get("PYTHONPATH"):
        python_path.append(environment["PYTHONPATH"])
    environment.update(
        {
            "PYTHONPATH": os.pathsep.join(python_path),
            "PYTHONUNBUFFERED": "1",
            "DA3_CODE_DIR": str(settings.da3_code),
            "DA3_MODEL_DIR": str(settings.da3_model),
            "GROUNDING_DINO_MODEL": str(settings.grounding_dino_model),
            "DINOV3_MODEL": str(settings.dinov3_model),
            "DINOV3_HF_PATH": str(settings.dinov3_model),
            "VLM_MODEL": settings.vlm_model,
            "VLM_BACKEND": settings.vlm_backend,
            "VLM_BASE_URL": settings.vlm_base_url,
            "VLM_API_KEY_ENV": settings.vlm_api_key_env,
            "BENCHMARK_USE_FROZEN_GT_STEP_PROMPTS": "1",
            "BENCHMARK_ENABLE_SAM3": "0",
            "BENCHMARK_ENABLE_VLM_MATCH": "1",
            "GDINO_BOX_THRESHOLD": "0.25",
            "GDINO_TEXT_THRESHOLD": "0.25",
            "POSE_BACKENDS": "da3",
            "DA3_USE_RAY_POSE": "0",
            "DA3_PROCESS_RES": "504",
            "DA3_PROCESS_RES_METHOD": "upper_bound_resize",
            "DA3_REF_VIEW_STRATEGY": "saddle_balanced",
            "DA3_ASPECT_POLICY": "native",
        }
    )
    if eval_frames_index is not None:
        environment["EGOGENEVAL_EVAL_FRAMES_INDEX"] = str(eval_frames_index)
    return environment


def _run_stage(
    name: str,
    command: list[str],
    *,
    log: Path,
    environment: dict[str, str],
) -> None:
    print(f"[{name}] starting; log: {log}", file=sys.stderr, flush=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            cwd=_official_root(),
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        raise ValidationError(f"{name} failed with exit code {completed.returncode}; see {log}")
    print(f"[{name}] complete", file=sys.stderr, flush=True)


def _expected_step_ids(manifest: ManifestIndex) -> list[str]:
    return [f"{case_id}__step{step}" for case_id, step in manifest.expected_steps]


def _require_exact_ids(rows: list[dict[str, Any]], expected: list[str], family: str) -> None:
    actual = [str(row.get("id", "")) for row in rows]
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise ValidationError(
            f"{family} did not return the complete ordered benchmark; "
            f"rows={len(actual)}/{len(expected)}, missing={missing[:1]}, extra={extra[:1]}"
        )


#: Raw runner key -> canonical ``cmg.jsonl`` key. The runner's PascalCase names are
#: its own; everything this package writes is snake_case, matching results.json.
_CMG_FIELDS = {
    "Pose_ActionComponent": "action_component",
    "Pose_ActionExpected": "action_expected",
    "Pose_ActionPred": "action_pred",
    "Pose_ActionAbsErr": "action_abs_err",
    "Pose_ActionSignCorrect": "action_sign_correct",
    "Pose_ActionUnit": "action_unit",
}


def _canonical_cmg(raw_path: Path, expected: list[str]) -> list[dict[str, Any]]:
    payload = read_json(raw_path)
    details = payload.get("details") if isinstance(payload, dict) else None
    failures = payload.get("failures", []) if isinstance(payload, dict) else []
    if failures:
        raise ValidationError(f"CMG runner reported {len(failures)} failure(s)")
    if not isinstance(details, list):
        raise ValidationError(f"{raw_path}: CMG details[] missing")
    _require_exact_ids(details, expected, "CMG")
    output: list[dict[str, Any]] = []
    for detail in details:
        row: dict[str, Any] = {"id": detail["id"]}
        for field, canonical in _CMG_FIELDS.items():
            value = detail.get(field)
            if value is None:
                value = detail.get(f"Pred_DA3_{field}")
            if value is None:
                raise ValidationError(f"{detail['id']}: DA3 field {field} is undefined")
            row[canonical] = value
        output.append(row)
    return output


def _raw_object_details(raw_root: Path, expected: list[str]) -> tuple[Path, list[dict[str, Any]]]:
    candidates = sorted(raw_root.glob("*/ssp_results.json"))
    if len(candidates) != 1:
        raise ValidationError(
            f"expected one SSP ssp_results.json under {raw_root}, found {len(candidates)}"
        )
    payload = read_json(candidates[0])
    details = payload.get("details") if isinstance(payload, dict) else None
    if not isinstance(details, list):
        raise ValidationError(f"{candidates[0]}: SSP details[] missing")
    _require_exact_ids(details, expected, "SSP detection/matching")
    signal_fields = (
        "mean_identity_sim",
        "mean_struct_ssim",
        "mean_edge_iou",
        "mean_sharpness_ratio",
        "mean_color_sim",
        "mean_shape_sim",
        "ObjectIntegrity",
        "Center_Error_Norm",
    )
    for detail in details:
        sample_id = str(detail["id"])
        if detail.get("object_prompt_protocol") != "fixed36_plus_frozen_qwen_gt_step_v1":
            raise ValidationError(f"{sample_id}: formal frozen prompt protocol was not used")
        if detail.get("object_prompt_source") != "target_images.qwen3vl_gt_object_labels":
            raise ValidationError(f"{sample_id}: frozen GT step labels were not used")
        try:
            thresholds = (
                float(detail["gdino_box_threshold"]),
                float(detail["gdino_text_threshold"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValidationError(f"{sample_id}: Grounding DINO thresholds are missing") from error
        if thresholds != (0.25, 0.25):
            raise ValidationError(f"{sample_id}: Grounding DINO thresholds differ from 0.25/0.25")
        if int(detail.get("num_matched", 0)) > 0:
            if detail.get("vlm_used") is not True:
                raise ValidationError(f"{sample_id}: Qwen3-VL match/quality evaluation was not used")
            missing_signals = [field for field in signal_fields if detail.get(field) is None]
            if missing_signals:
                raise ValidationError(
                    f"{sample_id}: formal integrity signals are missing: {missing_signals}"
                )
    return candidates[0], details


def _canonical_ssp(
    raw_details: list[dict[str, Any]],
    depth_path: Path,
    expected: list[str],
) -> list[dict[str, Any]]:
    depth_rows = read_jsonl(depth_path)
    _require_exact_ids(depth_rows, expected, "SSP object-depth")
    depth_by_id = {str(row["id"]): row for row in depth_rows}
    # Raw runner key -> canonical ``ssp.jsonl`` key; see _CMG_FIELDS.
    scalar_fields = {
        "num_gt_objects": "num_gt_objects",
        "num_pred_all": "num_pred_all",
        "num_matched": "num_matched",
        "Center_Error_Norm": "center_error_norm",
        "ObjectIntegrity": "object_integrity",
    }
    object_fields = ("gt_idx", "pred_idx", "status", "gt_box", "pred_box")
    output: list[dict[str, Any]] = []
    for detail in raw_details:
        sample_id = str(detail["id"])
        row: dict[str, Any] = {"id": sample_id}
        for field, canonical in scalar_fields.items():
            if field not in detail:
                raise ValidationError(f"{sample_id}: SSP field {field} is missing")
            row[canonical] = detail[field]
        objects = detail.get("objects")
        if not isinstance(objects, list):
            raise ValidationError(f"{sample_id}: SSP objects[] is missing")
        row["objects"] = [{field: obj.get(field) for field in object_fields} for obj in objects]
        depth = depth_by_id[sample_id].get("DepthAwareTopology_ObjectDepths")
        if not isinstance(depth, list):
            raise ValidationError(f"{sample_id}: object-depth records are missing")
        row["object_depths"] = depth
        output.append(row)
    return output


def score_predictions(
    predictions: str | Path,
    manifest: str | Path,
    *,
    output: str | Path,
    prepared_data: str | Path | None = None,
    eval_frames: str | Path | None = None,
    frozen_labels: str | Path | None = None,
    config: str | Path | None = None,
    resume: bool = False,
    **settings_overrides: Any,
) -> dict[str, Any]:
    """Run the exact final paper pipeline from generated images to final scores."""

    predictions_path = Path(predictions).expanduser().resolve()
    manifest_path = Path(manifest).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()
    prepared_path = (
        Path(prepared_data).expanduser().resolve() if prepared_data is not None else None
    )
    eval_frames_path = Path(eval_frames).expanduser().resolve() if eval_frames is not None else None
    if eval_frames_path is not None and eval_frames_path.is_dir():
        eval_frames_path = eval_frames_path / "eval_frames.jsonl"
    if prepared_path is not None and eval_frames_path is not None:
        raise ValidationError("--prepared-data and --eval-frames are mutually exclusive")
    if eval_frames_path is not None and not eval_frames_path.is_file():
        raise ValidationError(f"eval-frames index is missing: {eval_frames_path}")
    labels_path = (
        Path(frozen_labels).expanduser().resolve()
        if frozen_labels is not None
        else _default_frozen_labels(manifest_path)
    )
    settings = EvaluatorSettings.resolve(config, **settings_overrides)
    manifest_index = validate_manifest(manifest_path)
    validate_predictions(
        predictions_path,
        manifest_index,
        check_files=True,
        require_complete=True,
    )
    official_rows, model_id = build_official_input(
        predictions_path,
        manifest_index,
        prepared_data=prepared_path,
        eval_frames_index=eval_frames_path,
        frozen_labels=labels_path,
    )

    output_path.mkdir(parents=True, exist_ok=True)
    official_input = output_path / "official_input.jsonl"
    if official_input.exists() and not resume:
        raise ValidationError(f"output already contains a run; use --resume: {output_path}")
    if official_input.exists() and resume:
        previous = read_jsonl(official_input)
        if previous != official_rows:
            raise ValidationError("--resume input differs from the existing official_input.jsonl")
    else:
        write_jsonl(official_input, official_rows)

    settings_record = {key: str(value) for key, value in asdict(settings).items()}
    asset_provenance = _asset_provenance(eval_frames_path)
    run_config = {
        "pipeline": "egogeneval-v0.1",
        "model_id": model_id,
        "predictions": str(predictions_path),
        "predictions_sha256": sha256_file(predictions_path),
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_index.sha256,
        "frozen_labels": str(labels_path),
        "frozen_labels_sha256": sha256_file(labels_path),
        "official_input_sha256": sha256_file(official_input),
        "eval_frames": str(eval_frames_path) if eval_frames_path is not None else None,
        "eval_frames_sha256": (
            sha256_file(eval_frames_path) if eval_frames_path is not None else None
        ),
        "evaluator_source_inventory_sha256": sha256_file(_official_root() / "VENDORED_SHA256SUMS"),
        "settings": settings_record,
        **asset_provenance,
    }
    run_config_path = output_path / "run_config.json"
    if run_config_path.exists() and resume:
        if read_json(run_config_path) != run_config:
            raise ValidationError("--resume evaluator configuration differs from run_config.json")
    else:
        write_json(run_config_path, run_config)

    expected = _expected_step_ids(manifest_index)
    raw_dir = output_path / "raw"
    logs_dir = output_path / "logs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    environment = _runtime_environment(settings, eval_frames_path)
    official = _official_root()

    raw_cmg = raw_dir / "cmg_da3.json"
    if not (resume and raw_cmg.is_file()):
        _run_stage(
            "CMG/DA3",
            [
                sys.executable,
                str(_official_launcher()),
                str(official / "evaluation" / "cmg_runner.py"),
                "--input_jsonl",
                str(official_input),
                "--output_json_path",
                str(raw_cmg),
                "--no-split-by-model",
                "--no-save-vis",
                "--device",
                settings.device,
                "--pose-backends",
                "da3",
                "--da3-model-dir",
                str(settings.da3_model),
                "--da3-code-dir",
                str(settings.da3_code),
                "--da3-process-res",
                "504",
                "--da3-process-res-method",
                "upper_bound_resize",
                "--da3-ref-view-strategy",
                "saddle_balanced",
                "--da3-aspect-policy",
                "native",
                "--pose-only",
            ],
            log=logs_dir / "cmg.log",
            environment=environment,
        )
    cmg_rows = _canonical_cmg(raw_cmg, expected)
    write_jsonl(output_path / "cmg.jsonl", cmg_rows)

    raw_ssp_root = raw_dir / "ssp_detection"
    raw_ssp_candidates = sorted(raw_ssp_root.glob("*/ssp_results.json"))
    if not (resume and len(raw_ssp_candidates) == 1):
        _run_stage(
            "SSP/GroundingDINO+Qwen3-VL+DINOv3",
            [
                sys.executable,
                str(_official_launcher()),
                str(official / "evaluation" / "ssp_runner.py"),
                "--input-jsonl",
                str(official_input),
                "--output-dir",
                str(raw_ssp_root),
                "--max-samples-per-model",
                str(len(manifest_index.rows)),
                "--mode",
                "instance",
                "--match-policy",
                "relaxed",
                "--use-vlm",
                "--no-vis",
                "--no-depth-topology",
            ],
            log=logs_dir / "ssp_detection.log",
            environment=environment,
        )
    raw_ssp, raw_details = _raw_object_details(raw_ssp_root, expected)

    raw_depth = raw_dir / "ssp_object_depths.jsonl"
    if not (resume and raw_depth.is_file()):
        _run_stage(
            "SSP/GT+DA3 object depth",
            [
                sys.executable,
                str(official / "scripts" / "extract_object_depths.py"),
                "--model",
                model_id,
                "--raw-obj-json",
                str(raw_ssp),
                "--canonical-obj",
                str(raw_ssp),
                "--generation-jsonl",
                str(official_input),
                "--output-jsonl",
                str(raw_depth),
                "--subset",
                "main",
            ],
            log=logs_dir / "ssp_depth.log",
            environment=environment,
        )
    ssp_rows = _canonical_ssp(raw_details, raw_depth, expected)
    write_jsonl(output_path / "ssp.jsonl", ssp_rows)

    result = summarize_records(
        output_path / "cmg.jsonl",
        output_path / "ssp.jsonl",
        {**provenance(manifest_index.sha256), **asset_provenance},
    )
    write_json(output_path / "results.json", result)
    run_manifest = {
        **run_config,
        "records": {"cmg": len(cmg_rows), "ssp": len(ssp_rows)},
        "results": result,
    }
    write_json(output_path / "run_manifest.json", run_manifest)
    return result
