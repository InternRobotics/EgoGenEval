"""Environment readiness diagnostics."""

from __future__ import annotations

import importlib.util
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from .versioning import BENCHMARK_VERSION, EVALUATOR_VERSION, PACKAGE_VERSION

BASE_MODULES = ("jsonschema", "PIL", "yaml")
HEAVY_MODULES = (
    "torch",
    "torchvision",
    "transformers",
    "accelerate",
    "cv2",
    "h5py",
    "scipy",
    "skimage",
    "qwen_vl_utils",
)
DATA_ROOTS = (
    "HYPERSIM_ROOT",
    "MATTERPORT3D_ROOT",
    "SCANNET_ROOT",
    "SCANNETPP_ROOT",
)
#: Backends `adapters.frames` can decode a video model's per-step clips with.
#: Either one suffices; both absent only matters for --model-type video whose
#: steps are encoded clips rather than stills or frame directories.
VIDEO_BACKENDS = ("imageio_ffmpeg", "cv2")

#: Third-party packages Depth-Anything-3 imports on the path to
#: ``depth_anything_3.api``. They are dependencies of the DA3 checkout, not of
#: this package, so ``pip install egogeneval[full]`` cannot supply them -- which
#: is exactly why the check has to be an import, not a version pin.
DA3_HINTS = {
    "omegaconf": "pip install omegaconf",
    "moviepy": "pip install 'moviepy<2'  # 2.x dropped moviepy.editor",
    "pycolmap": "pip install pycolmap",
    "evo": "pip install evo",
}


def _availability(names: tuple[str, ...]) -> dict[str, bool]:
    return {name: importlib.util.find_spec(name) is not None for name in names}


def _da3_probe(da3_code: Path) -> dict[str, Any]:
    """Import ``depth_anything_3.api`` in a subprocess, as the CMG stage does.

    Worth the ~2 s: the frozen CMG runner catches its own import failure and
    emits ``None`` for every pose rather than aborting, so a missing DA3
    dependency otherwise costs a full evaluation run before anything complains.
    A subprocess keeps DA3's heavy imports (and its CUDA init) out of ours.
    """

    completed = subprocess.run(
        [sys.executable, "-c", "import depth_anything_3.api"],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(
            filter(None, [str(da3_code), os.environ.get("PYTHONPATH", "")])
        )},
        check=False,
    )
    if completed.returncode == 0:
        return {"status": "ready", "da3_code": str(da3_code)}

    # Name the missing module and how to install it; the raw traceback points
    # into the DA3 checkout, which tells the user nothing actionable.
    match = re.search(r"No module named '([\w.]+)'", completed.stderr)
    missing = match.group(1).split(".")[0] if match else None
    report: dict[str, Any] = {
        "status": "unavailable",
        "da3_code": str(da3_code),
        "error": (completed.stderr.strip().splitlines() or ["import failed"])[-1],
    }
    if missing:
        report["missing_module"] = missing
        report["install"] = DA3_HINTS.get(missing, f"pip install {missing}")
    return report


def collect_doctor_report(
    *, full: bool = False, evaluator_config: str | Path | None = None
) -> dict[str, Any]:
    python_supported = (3, 10) <= sys.version_info[:2] < (3, 13)
    if full:
        heavy = _availability(HEAVY_MODULES)
        full_status = "ready" if all(heavy.values()) else "unavailable"
    else:
        heavy = {}
        full_status = "not-checked"
    video_backends = _availability(VIDEO_BACKENDS)

    da3: dict[str, Any] = {"status": "not-checked"}
    if full:
        try:
            from .scoring.pipeline import EvaluatorSettings

            settings = EvaluatorSettings.resolve(evaluator_config)
        except Exception as error:  # config absent or incomplete -- not fatal here
            da3 = {"status": "not-configured", "detail": str(error)}
        else:
            da3 = _da3_probe(settings.da3_code)
        if da3["status"] == "unavailable":
            full_status = "unavailable"

    return {
        "package_version": PACKAGE_VERSION,
        "benchmark_version": BENCHMARK_VERSION,
        "evaluator_version": EVALUATOR_VERSION,
        "python": {
            "version": platform.python_version(),
            "supported": python_supported,
        },
        "platform": platform.platform(),
        "base_dependencies": _availability(BASE_MODULES),
        "dataset_roots": {name: bool(os.environ.get(name)) for name in DATA_ROOTS},
        "video_decoding": {
            "status": "ready" if any(video_backends.values()) else "unavailable",
            "backends": video_backends,
            "install": "pip install 'egogeneval[video]'",
        },
        "da3": da3,
        "full_evaluator": {
            "status": full_status,
            "dependencies": heavy,
            "historical_equivalence": True,
        },
    }
