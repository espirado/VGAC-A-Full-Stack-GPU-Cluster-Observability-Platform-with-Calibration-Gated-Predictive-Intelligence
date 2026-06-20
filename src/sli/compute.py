"""
Core SLI computation functions.

All functions operate on numpy arrays of (y_true, y_prob) where:
  - y_true: binary labels (0 or 1), shape (n,)
  - y_prob: predicted probability of positive class, shape (n,)
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
import json

import numpy as np


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BrierDecomposition:
    """Murphy (1973) decomposition of the Brier score."""
    reliability: float     # lower is better (calibration error)
    resolution: float      # higher is better (separation power)
    uncertainty: float     # base rate term (not under model control)
    brier_score: float     # = reliability - resolution + uncertainty

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class TailCalibrationResult:
    """Calibration analysis at high-risk probability thresholds."""
    threshold: float
    count: int
    actual_rate: float
    predicted_rate: float
    gap: float

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class SLIResult:
    """Complete SLI vector for a model on a dataset."""
    # SLI-1: ECE
    ece: float
    mce: float
    n_bins: int

    # SLI-2: Brier decomposition
    brier: BrierDecomposition

    # SLI-3: Tail calibration
    tail_calibration: List[TailCalibrationResult]
    max_tail_gap: float

    # SLI-4: PSI (None if no reference distribution provided)
    psi: Optional[float]

    # Discrimination metrics (for context, not SLIs)
    auroc: Optional[float]
    auprc: Optional[float]

    # Metadata
    n_samples: int
    positive_rate: float

    def to_dict(self) -> Dict:
        d = asdict(self)
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


@dataclass
class SLOCompliance:
    """Whether each SLO is met."""
    ece_met: bool           # ECE <= 0.05
    mce_met: bool           # MCE <= 0.10
    tail_met: bool          # max tail gap <= 0.035
    psi_met: Optional[bool] # PSI < 0.10 (None if PSI not computed)
    all_met: bool           # all of the above

    # Tier qualification (Paper 3)
    tier_qualified: int     # highest tier (1-4) the model qualifies for
    tier_name: str          # "annotate", "warn", "suggest", "gate"

    def to_dict(self) -> Dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# SLI-1: Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)
# ---------------------------------------------------------------------------

def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
    strategy: str = "quantile",
) -> Tuple[float, float, List[Dict]]:
    """
    Compute ECE and MCE with equal-mass (quantile) or equal-width binning.

    Parameters
    ----------
    y_true : array-like of shape (n,)
        Binary ground truth labels.
    y_prob : array-like of shape (n,)
        Predicted probabilities for the positive class.
    n_bins : int
        Number of bins.
    strategy : str
        "quantile" for equal-mass bins, "uniform" for equal-width.

    Returns
    -------
    ece : float
    mce : float
    bins : list of dict with keys {bin_idx, count, mean_predicted, mean_actual, gap}
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_true)
    assert n == len(y_prob), "y_true and y_prob must have the same length"

    if strategy == "quantile":
        # Equal-mass bins
        quantiles = np.linspace(0, 1, n_bins + 1)
        bin_edges = np.quantile(y_prob, quantiles)
        bin_edges = np.unique(bin_edges)  # remove duplicates
    else:
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    ece = 0.0
    mce = 0.0
    bins_info = []

    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == 0:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob > lo) & (y_prob <= hi)

        count = mask.sum()
        if count == 0:
            continue

        mean_pred = y_prob[mask].mean()
        mean_actual = y_true[mask].mean()
        gap = abs(mean_actual - mean_pred)

        ece += (count / n) * gap
        mce = max(mce, gap)

        bins_info.append({
            "bin_idx": i,
            "count": int(count),
            "mean_predicted": float(mean_pred),
            "mean_actual": float(mean_actual),
            "gap": float(gap),
        })

    return float(ece), float(mce), bins_info


def compute_mce(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
    strategy: str = "quantile",
) -> float:
    """Convenience wrapper returning only MCE."""
    _, mce, _ = compute_ece(y_true, y_prob, n_bins, strategy)
    return mce


# ---------------------------------------------------------------------------
# SLI-2: Brier Score Decomposition
# ---------------------------------------------------------------------------

def compute_brier_decomposition(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
) -> BrierDecomposition:
    """
    Murphy (1973) decomposition: Brier = Reliability - Resolution + Uncertainty.

    Uses equal-mass binning for consistency with ECE computation.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_true)

    # Base rate
    bar_o = y_true.mean()
    uncertainty = bar_o * (1 - bar_o)

    # Bin using quantiles
    quantiles = np.linspace(0, 1, n_bins + 1)
    bin_edges = np.quantile(y_prob, quantiles)
    bin_edges = np.unique(bin_edges)

    reliability = 0.0
    resolution = 0.0

    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == 0:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob > lo) & (y_prob <= hi)

        n_k = mask.sum()
        if n_k == 0:
            continue

        o_k = y_true[mask].mean()   # observed frequency in bin
        f_k = y_prob[mask].mean()   # mean forecast in bin

        reliability += (n_k / n) * (f_k - o_k) ** 2
        resolution += (n_k / n) * (o_k - bar_o) ** 2

    brier_score = reliability - resolution + uncertainty

    return BrierDecomposition(
        reliability=float(reliability),
        resolution=float(resolution),
        uncertainty=float(uncertainty),
        brier_score=float(brier_score),
    )


# ---------------------------------------------------------------------------
# SLI-3: Tail Calibration
# ---------------------------------------------------------------------------

def compute_tail_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Optional[List[float]] = None,
) -> List[TailCalibrationResult]:
    """
    Compute calibration at high-risk tails.

    For each threshold t, compute the gap between predicted and actual
    positive rates among predictions >= t.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    if thresholds is None:
        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]

    results = []
    for t in thresholds:
        mask = y_prob >= t
        count = mask.sum()
        if count < 5:  # too few samples for meaningful calibration
            continue

        actual_rate = float(y_true[mask].mean())
        predicted_rate = float(y_prob[mask].mean())
        gap = abs(actual_rate - predicted_rate)

        results.append(TailCalibrationResult(
            threshold=float(t),
            count=int(count),
            actual_rate=actual_rate,
            predicted_rate=predicted_rate,
            gap=gap,
        ))

    return results


# ---------------------------------------------------------------------------
# SLI-4: Population Stability Index (PSI)
# ---------------------------------------------------------------------------

def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
    eps: float = 1e-6,
) -> float:
    """
    Population Stability Index between reference and current distributions.

    PSI < 0.10: no significant shift
    PSI 0.10-0.20: moderate shift
    PSI >= 0.20: significant shift

    Parameters
    ----------
    reference : array-like
        Reference (training) feature distribution.
    current : array-like
        Current (serving) feature distribution.
    n_bins : int
        Number of bins for the histogram.
    eps : float
        Small value to avoid log(0).
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)

    # Use reference quantiles for bin edges
    quantiles = np.linspace(0, 1, n_bins + 1)
    bin_edges = np.quantile(reference, quantiles)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    ref_counts = np.histogram(reference, bins=bin_edges)[0]
    cur_counts = np.histogram(current, bins=bin_edges)[0]

    # Normalize to proportions
    ref_pct = ref_counts / ref_counts.sum() + eps
    cur_pct = cur_counts / cur_counts.sum() + eps

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


# ---------------------------------------------------------------------------
# Combined SLI computation
# ---------------------------------------------------------------------------

def compute_all_slis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
    reference_features: Optional[np.ndarray] = None,
    current_features: Optional[np.ndarray] = None,
    tail_thresholds: Optional[List[float]] = None,
) -> SLIResult:
    """
    Compute all four SLIs for a model on a dataset.

    Parameters
    ----------
    y_true : array-like of shape (n,)
        Binary labels.
    y_prob : array-like of shape (n,)
        Predicted probabilities.
    n_bins : int
        Number of bins for ECE and Brier.
    reference_features : array-like, optional
        Reference feature vector for PSI (e.g., training pending_at_submit).
    current_features : array-like, optional
        Current feature vector for PSI (e.g., serving pending_at_submit).
    tail_thresholds : list of float, optional
        Thresholds for tail calibration.

    Returns
    -------
    SLIResult with all four SLIs populated.
    """
    from sklearn.metrics import roc_auc_score, average_precision_score

    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    # SLI-1: ECE / MCE
    ece, mce, _ = compute_ece(y_true, y_prob, n_bins=n_bins)

    # SLI-2: Brier decomposition
    brier = compute_brier_decomposition(y_true, y_prob, n_bins=n_bins)

    # SLI-3: Tail calibration
    tail = compute_tail_calibration(y_true, y_prob, thresholds=tail_thresholds)
    max_tail_gap = max((t.gap for t in tail), default=0.0)

    # SLI-4: PSI
    psi = None
    if reference_features is not None and current_features is not None:
        psi = compute_psi(reference_features, current_features)

    # Discrimination (context, not SLIs)
    try:
        auroc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auroc = None
    try:
        auprc = float(average_precision_score(y_true, y_prob))
    except ValueError:
        auprc = None

    return SLIResult(
        ece=ece,
        mce=mce,
        n_bins=n_bins,
        brier=brier,
        tail_calibration=tail,
        max_tail_gap=max_tail_gap,
        psi=psi,
        auroc=auroc,
        auprc=auprc,
        n_samples=len(y_true),
        positive_rate=float(y_true.mean()),
    )


# ---------------------------------------------------------------------------
# SLO compliance check
# ---------------------------------------------------------------------------

# SLO thresholds (Paper 2, Table 2)
SLO_ECE = 0.05
SLO_MCE = 0.10
SLO_TAIL_GAP = 0.035
SLO_PSI = 0.10

# Tier prerequisites (Paper 3, Table 1)
TIER_ECE_THRESHOLDS = {
    1: 0.10,   # Annotate
    2: 0.07,   # Warn
    3: 0.05,   # Suggest
    4: 0.03,   # Gate
}
TIER_NAMES = {
    1: "annotate",
    2: "warn",
    3: "suggest",
    4: "gate",
}


def check_slo_compliance(sli: SLIResult) -> SLOCompliance:
    """
    Check whether SLIs meet SLO thresholds (Paper 2) and
    determine tier qualification (Paper 3).
    """
    ece_met = sli.ece <= SLO_ECE
    mce_met = sli.mce <= SLO_MCE
    tail_met = sli.max_tail_gap <= SLO_TAIL_GAP
    psi_met = sli.psi < SLO_PSI if sli.psi is not None else None

    all_met = ece_met and mce_met and tail_met
    if psi_met is not None:
        all_met = all_met and psi_met

    # Tier qualification
    tier_qualified = 0
    for tier in sorted(TIER_ECE_THRESHOLDS.keys()):
        if sli.ece <= TIER_ECE_THRESHOLDS[tier]:
            tier_qualified = tier
        else:
            break

    tier_name = TIER_NAMES.get(tier_qualified, "none")

    return SLOCompliance(
        ece_met=ece_met,
        mce_met=mce_met,
        tail_met=tail_met,
        psi_met=psi_met,
        all_met=all_met,
        tier_qualified=tier_qualified,
        tier_name=tier_name,
    )
