"""
Tier qualification analysis (Paper 3).

Determines which intervention tiers a model qualifies for
given its current SLI measurements.
"""

from .qualify import (
    qualify_tiers,
    build_tier_matrix,
    TierQualification,
)

__all__ = [
    "qualify_tiers",
    "build_tier_matrix",
    "TierQualification",
]
