"""Run vendored evaluators without OpenCV's bundled library search paths.

Import ``cv2`` before launching the evaluator so its import-time changes to
``LD_LIBRARY_PATH`` can be filtered. Preserve all other library paths.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

_CV2_MARKER = "/site-packages/cv2/"


def _strip_cv2_library_path(environment: dict[str, str] | os._Environ = os.environ) -> bool:
    """Remove OpenCV entries, returning whether the search path changed."""

    entries = environment.get("LD_LIBRARY_PATH", "").split(os.pathsep)
    retained = [entry for entry in entries if _CV2_MARKER not in entry]
    if retained == entries:
        return False
    if retained:
        environment["LD_LIBRARY_PATH"] = os.pathsep.join(retained)
    else:
        environment.pop("LD_LIBRARY_PATH", None)
    return True


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(
            "usage: python -m egogeneval.scoring._official_launcher <script.py> [args...]",
            file=sys.stderr,
        )
        return 2

    target = Path(argv[0]).resolve()
    if not target.is_file():
        print(f"launcher: no such script: {target}", file=sys.stderr)
        return 2

    # Load cv2 so its LD_LIBRARY_PATH rewrite happens here rather than inside
    # the evaluator. A missing/broken cv2 is not this shim's problem to report
    # -- hand over and let the evaluator raise its own import error. The strip
    # runs either way, so an already-polluted inherited value is cleaned too.
    try:
        import cv2  # noqa: F401
    except Exception:  # pragma: no cover - environment-dependent
        pass
    _strip_cv2_library_path()

    # Reproduce `python <script.py> ...`: argv[0] is the script, and the
    # script's own directory leads sys.path. runpy.run_path does not do the
    # latter for plain files.
    sys.argv = [str(target), *argv[1:]]
    sys.path.insert(0, str(target.parent))
    runpy.run_path(str(target), run_name="__main__")
    return 0


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
