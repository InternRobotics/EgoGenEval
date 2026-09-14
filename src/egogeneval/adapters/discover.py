"""Discover a model's generated outputs from a directory layout.

This is the front half of one-command evaluation. Users generate with their own
script, cluster, or API into a documented directory; ``discover_generations``
walks that directory and produces, in memory, the same ``{sample_id: row}``
shape :func:`egogeneval.adapters.build.build_predictions` already consumes. No
hand-authored index JSONL is needed.

Accepted layouts (per case, per step) -- all three may be mixed across cases,
but exactly one candidate must match for any given ``(case_id, step)``::

    <root>/<case_id>/step1.png     nested still
    <root>/<case_id>/step1.mp4     nested clip; final frame is the target view
    <root>/<case_id>/step1/        nested frame directory; last frame is scored
    <root>/<case_id>_step1.png     flat  (what build_predictions itself emits)
    <root>/<case_id>.png           bare  -- single-step cases only

``step`` numbers may be zero-padded (``step01``). ``<root>`` may be the run
directory (an ``outputs/`` subdirectory is entered automatically) or the
outputs directory itself.

Ambiguity is an error, not a heuristic: if two files could both serve as step 1
of a case, the layout must be corrected rather than resolved by guessing.
"""

from __future__ import annotations

import glob as _glob
from pathlib import Path
from typing import Any

from ..data import ManifestIndex
from ..errors import ValidationError
from .frames import IMAGE_SUFFIXES, VIDEO_SUFFIXES

#: Suffixes a discovered per-step artifact may carry. Directories are also
#: accepted (a directory of frames); see :mod:`egogeneval.adapters.frames`.
GENERATED_SUFFIXES = IMAGE_SUFFIXES | VIDEO_SUFFIXES

#: How many per-case problems to spell out before summarising the rest. A run
#: pointed at the wrong directory fails on all 1,400 cases; listing every one
#: buries the shape of the mistake.
_MAX_REPORTED = 10


def _outputs_root(root: Path) -> Path:
    """Enter ``outputs/`` when ``root`` is a run directory that has one."""

    candidate = root / "outputs"
    return candidate if candidate.is_dir() else root


def _step_tokens(step: int) -> list[str]:
    """Spellings of a step marker we accept, most canonical first.

    Zero-padding (``step01``) and an underscore separator (``step_1``,
    ``step_01``) are common in real generation scripts and unambiguous, so
    rejecting them would only force users to rename files we can already
    identify. Anything further -- a model's own segment indexing, say -- is
    genuinely model-specific and belongs in the caller's conversion, not here.
    """

    tokens = [f"step{step}", f"step{step:02d}", f"step_{step}", f"step_{step:02d}"]
    return list(dict.fromkeys(tokens))


def _step_patterns(case_id: str, step: int) -> list[str]:
    case = _glob.escape(case_id)
    patterns = []
    for token in _step_tokens(step):
        patterns += [
            f"{case}/{token}.*",
            f"{case}/{token}",  # frame directory
            f"{case}_{token}.*",
        ]
    return patterns


def _is_usable(path: Path) -> bool:
    if path.is_dir():
        return True
    return path.is_file() and path.suffix.lower() in GENERATED_SUFFIXES


def _candidates(root: Path, case_id: str, step: int, *, num_steps: int) -> list[Path]:
    patterns = _step_patterns(case_id, step)
    if num_steps == 1:
        # A single-step case may be written as a bare file named after the case.
        # `<case>.*` cannot collide with `<case>_step1.*` (the next character is
        # `_`, not `.`), so this stays unambiguous.
        patterns.append(f"{_glob.escape(case_id)}.*")

    found: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for match in sorted(root.glob(pattern)):
            resolved = match.resolve()
            if resolved in seen or not _is_usable(match):
                continue
            seen.add(resolved)
            found.append(match)
    return found


def _describe(root: Path, paths: list[Path]) -> str:
    names = []
    for path in paths:
        try:
            names.append(str(path.relative_to(root)))
        except ValueError:  # pragma: no cover - paths come from root.glob
            names.append(str(path))
    return ", ".join(names)


def discover_generations(
    root: str | Path, manifest_index: ManifestIndex
) -> dict[str, dict[str, Any]]:
    """Resolve one generated artifact per expected ``(case_id, step)``.

    Args:
        root: the run directory (or its ``outputs/``) holding the generations.
        manifest_index: the validated manifest defining which cases and steps
            are expected. Extra files in ``root`` are ignored; missing ones are
            an error, because a partial submission is not a comparable score.

    Returns:
        ``{sample_id: {"sample_id": ..., "generated_images": [{"step": n,
        "image_path": str}, ...]}}`` -- the canonical generation-source shape.

    Raises:
        ValidationError: if ``root`` is not a directory, or if any expected
            step resolves to zero or to several candidates. The message names
            the offending cases and, for ambiguity, the competing paths.
    """

    root = Path(root).expanduser()
    if not root.is_dir():
        raise ValidationError(f"generations directory not found: {root}")
    outputs = _outputs_root(root)

    steps_by_case: dict[str, list[int]] = {}
    for case_id, step in manifest_index.expected_steps:
        steps_by_case.setdefault(str(case_id), []).append(int(step))

    discovered: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    ambiguous: list[str] = []

    for case_id, steps in steps_by_case.items():
        images: list[dict[str, Any]] = []
        for step in sorted(steps):
            found = _candidates(outputs, case_id, step, num_steps=len(steps))
            if not found:
                missing.append(f"{case_id} step {step}")
                continue
            if len(found) > 1:
                ambiguous.append(
                    f"{case_id} step {step}: {_describe(outputs, found)}"
                )
                continue
            images.append({"step": step, "image_path": str(found[0])})
        discovered[case_id] = {"sample_id": case_id, "generated_images": images}

    problems: list[str] = []
    if missing:
        problems.append(
            f"no generated output found for {len(missing)} step(s) under {outputs}: "
            + _summarise(missing)
            + ". Expected one of <case>/step<N>.<ext>, <case>/step<N>/, "
            "<case>_step<N>.<ext>"
        )
    if ambiguous:
        problems.append(
            f"several candidates match {len(ambiguous)} step(s) under {outputs}, "
            "so the intended output is unclear: " + _summarise(ambiguous)
        )
    if problems:
        raise ValidationError("; ".join(problems))

    return discovered


def _summarise(items: list[str]) -> str:
    shown = "; ".join(items[:_MAX_REPORTED])
    remaining = len(items) - _MAX_REPORTED
    return shown if remaining <= 0 else f"{shown}; (+{remaining} more)"
