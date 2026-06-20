"""Post-hoc isotonic calibration.

Thin wrapper around ``sklearn.isotonic.IsotonicRegression`` that
matches the API the rest of VGAC expects: a fitted calibrator with
``.transform`` mapping raw scores -> calibrated probabilities, and a
``.summary`` dict that gets serialized into experiment artifacts.

Used by:
    - ``code/run_experiments.py`` (per-fold calibrator after CV
      training).
    - ``code/policy/inference_router.py`` (online recalibration).
    - ``code/ops/recalibrator.py`` (sliding-window recalibration).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.isotonic import IsotonicRegression

from ..evaluation.calibration import ece


@dataclass
class IsotonicCalibrator:
    """Fitted isotonic calibrator.

    ``out_of_bounds='clip'`` keeps test-time inputs within the support
    of the validation set, which is the conservative choice for an
    admission-gating system: we never extrapolate calibrated
    probabilities outside the range we have evidence for.
    """

    iso: IsotonicRegression
    val_ece_before: float
    val_ece_after: float
    n_val: int
    knots: np.ndarray = field(default_factory=lambda: np.empty(0))

    def transform(self, raw: np.ndarray) -> np.ndarray:
        return np.asarray(self.iso.transform(np.asarray(raw)))

    def __call__(self, raw: np.ndarray) -> np.ndarray:
        return self.transform(raw)

    @property
    def improvement(self) -> float:
        return float(self.val_ece_before - self.val_ece_after)

    def summary(self) -> dict:
        return {
            "method": "isotonic",
            "n_val": int(self.n_val),
            "val_ece_before": float(self.val_ece_before),
            "val_ece_after": float(self.val_ece_after),
            "improvement": self.improvement,
            "n_knots": int(len(self.knots)),
        }


def fit_isotonic(
    raw: np.ndarray,
    y: np.ndarray,
    *,
    n_bins_for_ece: int = 15,
    increasing: bool = True,
    out_of_bounds: str = "clip",
) -> IsotonicCalibrator:
    """Fit an isotonic regression on ``(raw, y)`` and report
    pre/post-calibration ECE on the same set.

    The reported ECE is the *training* fit; for unbiased estimates use
    the calibrator on a held-out test set and re-measure.
    """
    raw = np.asarray(raw, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(raw) != len(y):
        raise ValueError("raw and y length mismatch")

    iso = IsotonicRegression(out_of_bounds=out_of_bounds, increasing=increasing)
    iso.fit(raw, y)
    cal = iso.transform(raw)

    knots = np.asarray(iso.X_thresholds_) if hasattr(iso, "X_thresholds_") else np.empty(0)
    return IsotonicCalibrator(
        iso=iso,
        val_ece_before=ece(y, raw, n_bins=n_bins_for_ece),
        val_ece_after=ece(y, cal, n_bins=n_bins_for_ece),
        n_val=len(raw),
        knots=knots,
    )


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n = 2_000
    y = rng.binomial(1, 0.3, size=n)
    raw = np.clip(0.5 * (y - 0.3) + 0.3 + rng.normal(0, 0.2, size=n), 0, 1)
    raw = raw**1.5

    cal = fit_isotonic(raw, y)
    print(cal.summary())
    print(f"improvement = {cal.improvement:.4f}")
