"""Runtime-only defaults for the public evaluator.

Every machine-specific path is supplied through an environment variable or a
CLI argument.  The historical internal release used absolute cluster paths;
those defaults are intentionally not part of the public repository.
"""

from __future__ import annotations

import os
from pathlib import Path


def _paths(name: str) -> list[str]:
    value = os.environ.get(name, "")
    return [str(Path(item).expanduser()) for item in value.split(os.pathsep) if item]


INPUT_JSONL = _paths("EGOGENEVAL_INPUT_JSONL")
OUTPUT_DIR = os.environ.get("EGOGENEVAL_OUTPUT_DIR", "outputs")
