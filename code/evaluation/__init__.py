"""Evaluation primitives for VGAC.

Each module here is referenced by ``docs/METHODOLOGY.md`` so reviewers
can trace claim -> code one-to-one.

Public API:
    seeds.seed_everything   - deterministic numpy / sklearn seeding
    calibration.ece         - equal-mass Expected Calibration Error
    calibration.mce         - Maximum Calibration Error (worst bin)
    calibration.reliability_curve - bin-level mean prob vs. mean label
    calibration.brier_decomposition - Murphy 3-way (rel + res + unc)
    bootstrap.percentile_ci - 1000-iteration percentile bootstrap
"""

from .seeds import seed_everything
from .calibration import ece, mce, reliability_curve, brier_decomposition
from .bootstrap import percentile_ci

__all__ = [
    "seed_everything",
    "ece",
    "mce",
    "reliability_curve",
    "brier_decomposition",
    "percentile_ci",
]
