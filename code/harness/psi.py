"""Population Stability Index (PSI) for feature-drift detection.

Used in VGAC's per-cluster calibration harness to monitor input-feature
drift between training reference and live windows. PSI > 0.10 is one of
the recalibration triggers used by the gating layer.
"""
from __future__ import annotations

import numpy as np


def psi(reference: np.ndarray, live: np.ndarray, bins: int = 10) -> float:
    """Equal-frequency PSI per Siddiqi (banking convention).

    psi < 0.1   -> stable
    0.1 - 0.25  -> moderate shift
    > 0.25      -> severe shift
    """
    reference = np.asarray(reference)
    live = np.asarray(live)
    edges = np.quantile(reference, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    ref_hist, _ = np.histogram(reference, bins=edges)
    live_hist, _ = np.histogram(live, bins=edges)
    p = (ref_hist + 1) / (ref_hist.sum() + bins)
    q = (live_hist + 1) / (live_hist.sum() + bins)
    return float(np.sum((q - p) * np.log(q / p)))
