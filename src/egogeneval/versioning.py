"""Version and provenance constants."""

from __future__ import annotations

import os

PACKAGE_VERSION = "0.1.0"
BENCHMARK_VERSION = "0.1"
EVALUATOR_VERSION = "0.1"
RESULT_NAMESPACE = "evaluator-v0.1"


def provenance(manifest_sha256: str) -> dict[str, str]:
    return {
        "benchmark_version": BENCHMARK_VERSION,
        "evaluator_version": EVALUATOR_VERSION,
        "manifest_sha256": manifest_sha256,
        "code_commit": os.environ.get("EGOGENEVAL_CODE_COMMIT", "unknown"),
    }
