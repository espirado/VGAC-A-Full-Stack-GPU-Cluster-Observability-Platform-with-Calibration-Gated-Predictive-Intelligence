"""Calibration metrics: ECE, MCE, reliability curve, Brier decomposition.

These are the *measurement* primitives used by the calibration-gated
decision layer described in Section 3 of the PEARC '26 paper. They are
deliberately simple and dependency-free (NumPy only) so reviewers can
audit them without running a notebook.

Notation:
    y_true  : array-like of {0, 1} ground-truth labels.
    y_prob  : array-like of P(y=1) in [0, 1].
    n_bins  : number of calibration bins; default 15 matches the paper.

ECE / MCE follow Naeini et al. (AAAI 2015) and the survey of Guo et al.
(ICML 2017). The Brier decomposition follows Murphy (1973):
    Brier = Reliability - Resolution + Uncertainty,
where Reliability is what calibration directly improves and the other
two terms are intrinsic to the data and the model's discrimination.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Equal-mass binning helper
# ---------------------------------------------------------------------------
def _equal_mass_bins(y_prob: np.ndarray, n_bins: int) -> list[np.ndarray]:
    """Return a list of index-arrays partitioning ``y_prob`` into ``n_bins``
    equal-count bins (after sorting by probability).

    Equal-mass binning is preferred over equal-width binning under class
    imbalance, where high-probability bins would otherwise be sparsely
    populated. Matches the "15 equal-mass bins" footnote in the paper.
    """
    order = np.argsort(y_prob)
    return [b for b in np.array_split(order, n_bins) if len(b) > 0]


# ---------------------------------------------------------------------------
# ECE / MCE
# ---------------------------------------------------------------------------
def ece(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15
) -> float:
    """Expected Calibration Error (equal-mass, weighted by bin size).

    ECE = sum_b (|B_b| / n) * |mean(p_b) - mean(y_b)|
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_prob)
    if n == 0:
        return 0.0
    err = 0.0
    for b in _equal_mass_bins(y_prob, n_bins):
        err += (len(b) / n) * abs(y_prob[b].mean() - y_true[b].mean())
    return float(err)


def mce(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15
) -> float:
    """Maximum Calibration Error (worst-bin gap).

    MCE = max_b |mean(p_b) - mean(y_b)|

    Operationally this is the worst-case calibration error a Tier-4
    gating decision could be exposed to before recalibration kicks in.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(y_prob) == 0:
        return 0.0
    worst = 0.0
    for b in _equal_mass_bins(y_prob, n_bins):
        worst = max(worst, abs(y_prob[b].mean() - y_true[b].mean()))
    return float(worst)


# ---------------------------------------------------------------------------
# Reliability curve
# ---------------------------------------------------------------------------
def reliability_curve(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-bin (mean predicted probability, empirical frequency, count).

    Returns three arrays of length ``<= n_bins``. Use to draw reliability
    diagrams (paper Figure 3 ``calibration_curve.png``).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bins = _equal_mass_bins(y_prob, n_bins)
    mean_p = np.array([y_prob[b].mean() for b in bins])
    mean_y = np.array([y_true[b].mean() for b in bins])
    count = np.array([len(b) for b in bins])
    return mean_p, mean_y, count


# ---------------------------------------------------------------------------
# Brier 3-way decomposition (Murphy 1973)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BrierDecomposition:
    brier: float
    reliability: float
    resolution: float
    uncertainty: float

    def check(self) -> float:
        """Identity ``Brier ~= Reliability - Resolution + Uncertainty``.

        Returns the residual; should be near zero (small numerical drift
        from binning is expected).
        """
        return float(self.brier - (self.reliability - self.resolution + self.uncertainty))


def brier_decomposition(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15
) -> BrierDecomposition:
    """Murphy's 3-way Brier decomposition.

    Reliability is the calibration-attributable component (lower is
    better). Resolution captures the model's discrimination across bins
    (higher is better). Uncertainty is the irreducible variance of the
    base rate.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_prob)
    base_rate = y_true.mean() if n else 0.0
    brier = float(np.mean((y_prob - y_true) ** 2)) if n else 0.0
    rel = res = 0.0
    for b in _equal_mass_bins(y_prob, n_bins):
        nb = len(b)
        pb = y_prob[b].mean()
        yb = y_true[b].mean()
        rel += (nb / n) * (pb - yb) ** 2
        res += (nb / n) * (yb - base_rate) ** 2
    unc = float(base_rate * (1.0 - base_rate))
    return BrierDecomposition(brier=brier, reliability=float(rel), resolution=float(res), uncertainty=unc)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n = 5_000
    y = rng.binomial(1, 0.3, size=n)
    p_well = np.clip(0.3 + 0.5 * (y - 0.3) + rng.normal(0, 0.1, size=n), 0, 1)
    p_mis = np.clip(p_well + 0.15, 0, 1)

    print("Well-calibrated:")
    print(f"  ECE = {ece(y, p_well):.4f}, MCE = {mce(y, p_well):.4f}")
    bd = brier_decomposition(y, p_well)
    print(f"  Brier = {bd.brier:.4f} (rel={bd.reliability:.4f}, res={bd.resolution:.4f}, unc={bd.uncertainty:.4f})")
    print(f"  identity residual = {bd.check():.6f}")

    print("Miscalibrated:")
    print(f"  ECE = {ece(y, p_mis):.4f}, MCE = {mce(y, p_mis):.4f}")
