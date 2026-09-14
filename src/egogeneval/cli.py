"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .data import prepare_data, validate_manifest
from .doctor import collect_doctor_report
from .errors import ValidationError
from .versioning import PACKAGE_VERSION

_SCORING_AVAILABLE = True
try:
    from .scoring import score_predictions as _score_predictions
except ImportError:
    _SCORING_AVAILABLE = False

_DEFAULT_MANIFEST = Path("data/manifests/egogeneval_v0.1.jsonl")

_MODEL_TYPES = ("image", "video", "world-model", "pose-conditioned")

_MODEL_TYPE_HELP = (
    "generator family: image (still-image generation/editing), video (a clip or "
    "frame directory per step), world-model (autoregressive rollout), "
    "pose-conditioned (consumes native 6-DoF). The per-case interface_type is "
    "derived from this and the case's context size."
)


def _path(value: str) -> Path:
    return Path(value).expanduser()


def _emit(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _roots(values: list[str]) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValidationError(f"--root must be NAME=PATH, got {value!r}")
        name, path_text = value.split("=", 1)
        if not name or not path_text or name in roots:
            raise ValidationError(f"invalid or duplicate --root value: {value!r}")
        roots[name] = _path(path_text)
    return roots


def _doctor_command(args: argparse.Namespace) -> int:
    report = collect_doctor_report(full=args.full, evaluator_config=args.evaluator_config)
    if args.json:
        _emit(report)
    else:
        print(
            f"EgoGenEval {report['package_version']} | "
            f"Python {report['python']['version']} | "
            f"full evaluator: {report['full_evaluator']['status']}"
        )
        da3 = report["da3"]
        if da3["status"] == "unavailable":
            # The CMG runner would otherwise swallow this and score zeroed poses.
            print(f"  DA3: unavailable — {da3['error']}")
            if "install" in da3:
                print(f"       {da3['install']}")
        elif da3["status"] != "not-checked":
            print(f"  DA3: {da3['status']}")
    return 0


def _prepare_data_command(args: argparse.Namespace) -> int:
    report = prepare_data(
        args.manifest,
        _roots(args.root),
        args.destination,
        mode=args.mode,
    )
    _emit(report)
    return 0


def _download_data_command(args: argparse.Namespace) -> int:
    from .hf import download_config

    configs = args.config or ["test"]
    token = os.environ.get(args.token_env) if args.token_env else None

    verify_sha = args.verify_manifest_sha
    if verify_sha is None and _DEFAULT_MANIFEST.exists() and "test" in configs:
        # Default to the committed manifest hash so a drifted HF copy fails loudly.
        from .io import sha256_file

        verify_sha = sha256_file(_DEFAULT_MANIFEST)

    summaries = []
    for config in configs:
        result = download_config(
            repo_id=args.repo_id,
            config=config,
            output=args.output / config,
            revision=args.revision,
            token=token,
            verify_manifest_sha=verify_sha if config == "test" else None,
        )
        summaries.append(
            {
                "repo_id": result.repo_id,
                "config": result.config,
                "output_dir": str(result.output_dir),
                "num_rows": result.num_rows,
                "schema": result.schema,
                "manifest_path": (str(result.manifest_path) if result.manifest_path else None),
                "prepared_manifest_path": (
                    str(result.prepared_manifest_path)
                    if result.prepared_manifest_path
                    else None
                ),
                "manifest_sha256": result.manifest_sha256,
                "images_written": result.images_written,
                "depths_written": result.depths_written,
                "warnings": result.warnings,
            }
        )
    _emit(summaries)
    return 0


def _prepare_benchmark_command(args: argparse.Namespace) -> int:
    try:
        from .preparation.prepare import prepare_benchmark
    except ImportError as error:
        raise ValidationError("benchmark preparation requires: pip install -e '.[prepare,scannetpp]'") from error
    roots = {
        name.upper() + "_ROOT": getattr(args, name)
        for name in ("hypersim", "matterport3d", "scannet", "scannetpp")
        if getattr(args, name) is not None
    }
    report = prepare_benchmark(
        args.manifest, roots, args.output, embodiedscan=args.embodiedscan,
        datasets=args.dataset, fingerprints=args.fingerprints,
        check_only=args.check_only, workers=args.workers, cache=args.cache_dir,
        scannet_jpeg_tools=args.scannet_jpeg_tools, jpeg_archive=args.jpeg_archive,
    )
    # The detailed provenance stays in build_report.json; keep the primary CLI
    # useful even when hundreds of scenes contribute metadata or renderer rows.
    _emit({key: value for key, value in report.items()
           if key not in {"metadata_sha256", "source_roots", "scannetpp_renderers"}})
    return 2 if report["status"] == "missing-inputs" else 0


def _build_data_command(args: argparse.Namespace) -> int:
    try:
        from .preparation.build import build_data
    except ImportError as error:
        raise ValidationError("local data building requires: pip install -e '.[prepare]'") from error
    report = build_data(
        args.manifest, _roots(args.root), args.output,
        annotations=args.embodiedscan_info, check_only=args.check_only,
        mode=args.mode, reference_index=args.reference_index, datasets=args.dataset,
        raw_sources=args.raw_sources,
        fingerprints=args.fingerprints,
        scannet_jpeg_tools=args.scannet_jpeg_tools,
        matterport_annotations=args.matterport_embodiedscan_info,
        workers=args.workers,
    )
    _emit(report)
    return 2 if report["status"] == "missing-inputs" else 0


def _verify_data_command(args: argparse.Namespace) -> int:
    try:
        from .preparation.build import verify_data
    except ImportError as error:
        raise ValidationError("local data verification requires: pip install -e '.[prepare]'") from error
    _emit(verify_data(args.data, reference_index=args.reference_index, fingerprints=args.fingerprints))
    return 0


def _source_plan_command(args: argparse.Namespace) -> int:
    from .preparation.acquire import RAW_DATASETS, source_plan
    from .io import write_json

    report = source_plan(args.manifest, args.dataset or list(RAW_DATASETS), matterport_calibration=args.matterport_calibration)
    write_json(args.output, report)
    _emit({"output": str(args.output), "cases": report["cases"], "frames": report["frames"],
           "selected_datasets": report["selected_datasets"], "excluded_datasets": report["excluded_datasets"],
           "scene_counts": {k: v["scene_count"] for k, v in report["datasets"].items()}})
    return 0


def _unpack_sources_command(args: argparse.Namespace) -> int:
    from .preparation.acquire import unpack_sources

    report = unpack_sources(args.manifest, args.dataset, args.archives, args.output,
                            matterport_calibration=args.matterport_calibration)
    _emit({k: v for k, v in report.items() if k != "files"})
    return 2 if report["missing"] else 0


def _download_hypersim_command(args: argparse.Namespace) -> int:
    from urllib.error import URLError
    from .preparation.acquire import download_hypersim

    try:
        report = download_hypersim(args.manifest, args.output, workers=args.workers)
    except (URLError, TimeoutError, OSError) as error:
        raise ValidationError(f"official Hypersim download failed: {error}; completed files can be reused on retry") from error
    _emit({k: v for k, v in report.items() if k != "scenes"})
    return 0


def _format_score_summary(
    result: dict[str, Any], *, output: Path, model_id: str | None
) -> str:
    """Render the CMG / SSP / Overall block users actually read off the terminal."""

    def line(label: str, key: str) -> str:
        return (
            f"  {label:<8} {result[key]:.4f}"
            f"   (atomic {result[f'{key}_atomic']:.4f}"
            f" | chain {result[f'{key}_chain']:.4f}"
            f" | cycle {result[f'{key}_cycle']:.4f})"
        )

    counts = f"{result['evaluated']} evaluated"
    if result.get("missing"):
        counts += f", {result['missing']} missing"
    if result.get("failed"):
        counts += f", {result['failed']} failed"

    return "\n".join(
        [
            "",
            f"EgoGenEval — {model_id}" if model_id else "EgoGenEval",
            # One record per (case, step), so this is 2360 on the full benchmark,
            # not the 1400 case count.
            f"  {'steps':<8} {counts}",
            line("CMG", "cmg"),
            line("SSP", "ssp"),
            f"  {'Overall':<8} {result['overall']:.4f}",
            "",
            f"  results.json: {output / 'results.json'}",
            "",
        ]
    )


def _score_command(args: argparse.Namespace) -> int:
    if not _SCORING_AVAILABLE:
        print(
            "error: scoring backend not available. Install with:\n  pip install egogeneval[full]",
            file=sys.stderr,
        )
        return 2

    predictions = args.predictions

    if args.generations is not None:
        # One-command path: discover the user's output directory, build
        # predictions.jsonl into --output, then score it.
        from .adapters import build_predictions, discover_generations
        from .adapters.frames import FrameSelection

        if not args.model_id:
            raise ValidationError("--generations requires --model-id")
        if not args.model_type:
            raise ValidationError("--generations requires --model-type")
        if args.output.expanduser().resolve() == args.generations.expanduser().resolve():
            # build_predictions materialises frames into <output>/outputs/, which
            # is where the discovered sources live. Overlapping the two would
            # rewrite the user's generations in place.
            raise ValidationError(
                "--output must differ from --generations; the run directory is written "
                "into, not read from (e.g. --output "
                f"{args.generations / 'evaluation'})"
            )

        manifest_index = validate_manifest(args.manifest)
        sources = discover_generations(args.generations, manifest_index)
        adapted = build_predictions(
            sources,
            manifest=args.manifest,
            output=args.output,
            model_id=args.model_id,
            model_type=args.model_type,
            frame_selection=FrameSelection(
                policy=args.frame_policy,
                index=args.frame_index,
                ratio=args.frame_ratio,
            ),
        )
        predictions = adapted.predictions_path
        print(
            f"discovered {adapted.num_steps} generated steps across "
            f"{adapted.num_cases} cases -> {predictions}",
            file=sys.stderr,
        )

    result = _score_predictions(
        predictions=predictions,
        manifest=args.manifest,
        prepared_data=args.prepared_data,
        eval_frames=args.eval_frames,
        frozen_labels=args.frozen_labels,
        output=args.output,
        config=args.evaluator_config,
        resume=args.resume,
        device=args.device,
        da3_code=args.da3_code_dir,
        da3_model=args.da3_model_dir,
        grounding_dino_model=args.grounding_dino_model,
        dinov3_model=args.dinov3_model,
        vlm_model=args.vlm_model,
        vlm_backend=args.vlm_backend,
        vlm_base_url=args.vlm_base_url,
        vlm_api_key_env=args.vlm_api_key_env,
    )
    if args.json:
        _emit(result)
    else:
        print(_format_score_summary(result, output=args.output, model_id=args.model_id))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="egogeneval")
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {PACKAGE_VERSION}",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    doctor = commands.add_parser("doctor", help="inspect environment readiness")
    doctor.add_argument("--full", action="store_true")
    doctor.add_argument(
        "--evaluator-config",
        type=_path,
        help="evaluator config to probe DA3 with; without it --full reports "
        "da3 as not-configured",
    )
    doctor.add_argument("--json", action="store_true")
    doctor.set_defaults(handler=_doctor_command)

    prepare = commands.add_parser("prepare-data", help="materialize user-owned source data")
    prepare.add_argument("--manifest", type=_path, default=_DEFAULT_MANIFEST)
    prepare.add_argument("--root", action="append", default=[])
    prepare.add_argument("--destination", type=_path, required=True)
    prepare.add_argument("--mode", choices=("symlink", "copy"), default="symlink")
    prepare.set_defaults(handler=_prepare_data_command)

    benchmark = commands.add_parser(
        "prepare-benchmark", help="prepare all four official datasets for generation and scoring in one command"
    )
    benchmark.add_argument("--manifest", type=_path, default=_DEFAULT_MANIFEST)
    for name in ("hypersim", "matterport3d", "scannet", "scannetpp"):
        benchmark.add_argument(f"--{name}", type=_path, help=f"downloaded official {name} dataset root")
    benchmark.add_argument("--embodiedscan", type=_path, help="directory containing official v1 train/val/test annotation PKLs for Matterport3D cameras")
    benchmark.add_argument("--output", type=_path, default=Path("prepared-data/egogeneval"))
    benchmark.add_argument("--dataset", action="append", choices=("hypersim", "matterport3d", "scannet", "scannetpp"), help="explicit dataset subset (repeatable); default: all four")
    benchmark.add_argument("--workers", type=int, default=4, help="parallel ScanNet scene readers/extractors, 1-16 (default: 4)")
    benchmark.add_argument("--check-only", action="store_true", help="check every selected source without building assets or installing tools")
    benchmark.add_argument("--cache-dir", type=_path, help="tool cache (default: $XDG_CACHE_HOME/egogeneval or ~/.cache/egogeneval)")
    benchmark.add_argument("--scannet-jpeg-tools", type=_path, help="optional existing IJG 9e bin directory; otherwise built automatically")
    benchmark.add_argument("--jpeg-archive", type=_path, help="official jpegsrc.v9e.tar.gz for an offline tool build")
    benchmark.add_argument("--fingerprints", type=_path, action="append", help="override bundled hash references (repeatable)")
    benchmark.set_defaults(handler=_prepare_benchmark_command)

    build = commands.add_parser("build-data", help="build local RGB-D benchmark assets from source datasets")
    build.add_argument("--manifest", type=_path, default=_DEFAULT_MANIFEST)
    build.add_argument("--dataset", action="append", choices=("scannet", "matterport3d", "hypersim", "scannetpp"), help="build only this dataset; repeat to select several (default: all manifest datasets)")
    build.add_argument("--root", action="append", default=[], help="NAME=PATH dataset or metadata root")
    build.add_argument("--embodiedscan-info", type=_path, action="append", default=[], help="trusted EmbodiedScan annotation PKL; repeat for train/val/test")
    build.add_argument("--matterport-embodiedscan-info", type=_path, action="append", default=[], help="official EmbodiedScan v1 PKL for Matterport3D cameras only; repeat for train/val/test; works with --raw-sources")
    build.add_argument("--output", type=_path, default=Path("prepared-data/egogeneval"))
    build.add_argument("--check-only", action="store_true", help="report missing source inputs without writing output")
    build.add_argument("--workers", type=int, default=4, help="parallel ScanNet scene readers/extractors, 1-16 (default: 4)")
    build.add_argument("--mode", choices=("copy", "symlink"), default="copy")
    build.add_argument("--reference-index", type=_path, help="optional frozen eval_frames.jsonl for exact reference comparison")
    build.set_defaults(handler=_build_data_command)
    build.add_argument("--raw-sources", action="store_true", help="decode ScanNet .sens and read original source metadata; Matterport cameras may use explicit --matterport-embodiedscan-info")
    build.add_argument("--scannet-jpeg-tools", type=_path, help="IJG 9e bin directory from scripts/install_scannet_jpeg.py; required for frozen ScanNet JPEG bytes")
    build.add_argument("--fingerprints", type=_path, help="hash-only frozen benchmark reference; reject any reconstructed asset mismatch")

    plan = commands.add_parser("source-plan", help="list exact official scenes, archives and files needed by the benchmark")
    plan.add_argument("--manifest", type=_path, default=_DEFAULT_MANIFEST)
    plan.add_argument("--dataset", action="append", choices=("hypersim", "matterport3d", "scannet", "scannetpp"))
    plan.add_argument("--output", type=_path, default=Path("source-plan.json"))
    plan.add_argument("--matterport-calibration", choices=("native", "embodiedscan"), default="embodiedscan", help="with embodiedscan, require only Matterport RGB-D ZIPs plus separately obtained official v1 annotations")
    plan.set_defaults(handler=_source_plan_command)

    unpack = commands.add_parser("unpack-sources", help="extract required files from user-downloaded official ZIPs")
    unpack.add_argument("--manifest", type=_path, default=_DEFAULT_MANIFEST)
    unpack.add_argument("--dataset", required=True, choices=("hypersim", "matterport3d"))
    unpack.add_argument("--archives", type=_path, required=True, help="ZIP file or directory containing ZIPs")
    unpack.add_argument("--output", type=_path, required=True, help="source dataset root")
    unpack.add_argument("--matterport-calibration", choices=("native", "embodiedscan"), default="embodiedscan", help="with embodiedscan, unpack only Matterport RGB-D; cameras are read from official annotation PKLs at build time")
    unpack.set_defaults(handler=_unpack_sources_command)

    hypersim = commands.add_parser("download-hypersim", help="download selected original files directly from official Hypersim scene ZIPs")
    hypersim.add_argument("--manifest", type=_path, default=_DEFAULT_MANIFEST)
    hypersim.add_argument("--output", type=_path, required=True, help="HYPERSIM_ROOT destination")
    hypersim.add_argument("--workers", type=int, default=4)
    hypersim.set_defaults(handler=_download_hypersim_command)

    verify = commands.add_parser("verify-data", help="verify local asset hashes, shapes, calibration and coverage")
    verify.add_argument("--data", type=_path, required=True)
    verify.add_argument("--reference-index", type=_path)
    verify.add_argument("--fingerprints", type=_path, help="hash-only reference; defaults to the copy stored by build-data")
    verify.set_defaults(handler=_verify_data_command)

    download = commands.add_parser(
        "download-data",
        help="fetch benchmark configs from the Hugging Face release",
    )
    from .hf import DEFAULT_REPO_ID, KNOWN_CONFIGS

    download.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    download.add_argument(
        "--config",
        action="append",
        choices=KNOWN_CONFIGS,
        help="HF config to fetch (repeatable); defaults to test",
    )
    download.add_argument("--output", type=_path, required=True)
    download.add_argument("--revision", help="optional commit hash or tag to pin")
    download.add_argument(
        "--token-env",
        default="HF_TOKEN",
        help="environment variable holding the HF access token (never pass the token itself)",
    )
    download.add_argument(
        "--verify-manifest-sha",
        help="expected sha256 of the reconstructed test manifest; "
        "defaults to the committed manifest's hash when present",
    )
    download.set_defaults(handler=_download_data_command, config=None)

    score = commands.add_parser(
        "score",
        help="end-to-end evaluation: generated images -> CMG/SSP -> results",
        description="Score a model end to end. Point --generations at your output "
        "directory for the one-command path, or --predictions at an already-built "
        "predictions.jsonl.",
    )
    source = score.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--generations",
        type=_path,
        help="directory of generated outputs to discover; accepts "
        "<case>/step<N>.<ext>, <case>/step<N>/, or <case>_step<N>.<ext>. "
        "Requires --model-id and --model-type",
    )
    source.add_argument(
        "--predictions",
        type=_path,
        help="an already-built predictions.jsonl",
    )
    score.add_argument("--manifest", type=_path, default=_DEFAULT_MANIFEST)
    score.add_argument(
        "--model-id",
        help="leaderboard model identifier; required with --generations",
    )
    score.add_argument(
        "--model-type",
        choices=_MODEL_TYPES,
        help=_MODEL_TYPE_HELP + " Required with --generations.",
    )
    score.add_argument(
        "--frame-policy",
        choices=("last", "index", "ratio"),
        default="last",
        help="which frame of a video/segment is the target view (default: last, "
        "the paper's boundary-fixed-step policy)",
    )
    score.add_argument("--frame-index", type=int, default=-1)
    score.add_argument("--frame-ratio", type=float, default=1.0)
    score.add_argument(
        "--prepared-data",
        type=_path,
        help="optional prepared-data tree; otherwise resolve dataset-root environment variables",
    )
    score.add_argument(
        "--eval-frames",
        type=_path,
        help="downloaded eval_frames directory or eval_frames.jsonl; replaces source roots",
    )
    score.add_argument(
        "--frozen-labels",
        type=_path,
        help="frozen detector prompt labels; defaults to the checkout's "
        "data/evaluator/ copy",
    )
    score.add_argument("--output", type=_path, required=True)
    score.add_argument(
        "--evaluator-config",
        type=_path,
        help="YAML evaluator model/runtime configuration",
    )
    score.add_argument(
        "--json",
        action="store_true",
        help="print results.json to stdout instead of the human summary",
    )
    score.add_argument("--resume", action="store_true", help="resume verified stage outputs")
    score.add_argument("--device", help="torch device; defaults to config/env or cuda:0")
    score.add_argument("--da3-code-dir", type=_path)
    score.add_argument("--da3-model-dir", type=_path)
    score.add_argument("--grounding-dino-model", type=_path)
    score.add_argument("--dinov3-model", type=_path)
    score.add_argument("--vlm-model")
    score.add_argument("--vlm-backend")
    score.add_argument("--vlm-base-url")
    score.add_argument("--vlm-api-key-env")
    score.set_defaults(handler=_score_command)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
    except SystemExit as error:
        return int(error.code)
    try:
        return int(args.handler(args))
    except (ValidationError, FileNotFoundError, NotADirectoryError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


def entrypoint() -> None:
    raise SystemExit(main())
