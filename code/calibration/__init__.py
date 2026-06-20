"""Post-hoc calibration utilities for VGAC.

VGAC uses isotonic regression for post-hoc calibration: a monotone,
non-parametric mapping from raw model score to calibrated probability,
fit on a held-out validation split. See ``isotonic.py`` for details.

Why isotonic over Platt? The paper observes non-monotone
miscalibration shapes in the raw scores (Section 4 reliability
diagrams) and we have enough validation samples (>= 100 per equal-mass
bin) even on the smaller cluster slices. Platt is reported as a
sensitivity check in the long form (ISS26 Paper 2).
"""

from .isotonic import IsotonicCalibrator, fit_isotonic

__all__ = ["IsotonicCalibrator", "fit_isotonic"]
