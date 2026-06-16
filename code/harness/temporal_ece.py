"""Temporal-ECE window sensor.

Slices a held-out prediction stream into N equal-count windows and
computes the Expected Calibration Error inside each. A recalibration
cue fires when any window's ECE exceeds the configured threshold.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class TemporalECEResult:
    windows: list[int]
    eces: list[float]
    cue_fired: bool
    first_cue_window: int | None


def ece_equalmass(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 15) -> float:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if len(y_prob) == 0:
        return 0.0
    edges = np.quantile(y_prob, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = 0.0, 1.0
    total = 0.0
    n = len(y_prob)
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (y_prob >= lo) & (y_prob <= hi) if i == bins - 1 else (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        total += (mask.sum() / n) * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(total)


def temporal_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_windows: int = 8,
    cue_threshold: float = 0.01,
    bins: int = 15,
) -> TemporalECEResult:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_prob)
    step = max(1, n // n_windows)
    eces: list[float] = []
    windows: list[int] = []
    first_cue: int | None = None
    for w in range(1, n_windows + 1):
        lo = (w - 1) * step
        hi = n if w == n_windows else w * step
        e = ece_equalmass(y_true[lo:hi], y_prob[lo:hi], bins=bins)
        windows.append(w)
        eces.append(e)
        if first_cue is None and e > cue_threshold:
            first_cue = w
    return TemporalECEResult(windows, eces, first_cue is not None, first_cue)
