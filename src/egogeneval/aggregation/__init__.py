"""Canonical aggregation from frozen EgoGenEval metric records."""

from .cmg import aggregate_cmg
from .ssp import aggregate_ssp

__all__ = ["aggregate_cmg", "aggregate_ssp"]
