"""Sliding-window recalibration trigger.

Continuously consumes ``(y, p_hat)`` pairs as the platform serves
predictions. Maintains a fixed-size window per namespace/queue and:

    1. Computes ``ECE`` over the window every ``check_every`` updates.
    2. Optionally consumes a feature stream and computes ``PSI`` against
       a fixed reference distribution.
    3. Emits a ``RecalibrationDecision`` when either signal exceeds its
       configured SLO threshold.

Defaults match the operational thresholds documented in
``docs/METHODOLOGY.md``:

    - Weekly-ECE alarm threshold:   0.07
    - PSI drift trigger:            0.1 (Siddiqi banking convention)
    - Window length:                7 days (default 7000 samples)

The trigger is *advisory*. The decision layer (`code/policy/inference_router.py`)
consumes the cue; whether to actually retrain or fall back to a lower
tier is policy.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional, Sequence

import numpy as np

from ..evaluation.calibration import ece
from ..harness.psi import psi


@dataclass(frozen=True)
class RecalibrationDecision:
    """Recalibration cue emitted by the recalibrator."""

    cue_fired: bool
    reason: Optional[str]
    rolling_ece: float
    psi: Optional[float]
    n_window: int

    def __bool__(self) -> bool:  # pragma: no cover - convenience
        return self.cue_fired


@dataclass
class SlidingWindowRecalibrator:
    """Rolling ECE / PSI monitor with hard SLO thresholds.

    Args:
        window: max number of (y, p) pairs to keep.
        ece_threshold: SLO bound on rolling ECE (default 0.07).
        psi_threshold: drift bound on the optional feature stream
            (default 0.1).
        check_every: how often (in updates) to re-evaluate. Smaller
            values catch drift earlier at higher CPU cost.
        n_bins: bins used by ECE (default 15) and PSI (default 10).
        reference_feature: optional reference distribution for PSI; if
            ``None``, PSI is not evaluated.
    """

    window: int = 7000
    ece_threshold: float = 0.07
    psi_threshold: float = 0.10
    check_every: int = 100
    n_bins_ece: int = 15
    n_bins_psi: int = 10
    reference_feature: Optional[np.ndarray] = None

    _y: Deque[float] = field(default_factory=deque, init=False, repr=False)
    _p: Deque[float] = field(default_factory=deque, init=False, repr=False)
    _x: Deque[float] = field(default_factory=deque, init=False, repr=False)
    _seen: int = field(default=0, init=False)

    def update(
        self,
        y: float | int,
        p: float,
        feature: Optional[float] = None,
    ) -> RecalibrationDecision:
        """Add one observation and (every ``check_every`` updates) emit
        a recalibration decision."""
        self._y.append(float(y))
        self._p.append(float(p))
        if feature is not None:
            self._x.append(float(feature))
        while len(self._y) > self.window:
            self._y.popleft()
            self._p.popleft()
        while len(self._x) > self.window:
            self._x.popleft()
        self._seen += 1
        if self._seen % self.check_every != 0:
            return RecalibrationDecision(False, None, 0.0, None, len(self._y))
        return self._evaluate()

    def update_batch(
        self,
        ys: Sequence[float],
        ps: Sequence[float],
        features: Optional[Sequence[float]] = None,
    ) -> RecalibrationDecision:
        """Convenience: extend with a batch and evaluate once at end."""
        feats: Sequence[Optional[float]]
        if features is None:
            feats = [None] * len(ys)
        else:
            feats = list(features)
        last: Optional[RecalibrationDecision] = None
        for y, p, x in zip(ys, ps, feats):
            last = self.update(y, p, x)
        return last or self._evaluate()

    def _evaluate(self) -> RecalibrationDecision:
        if not self._y:
            return RecalibrationDecision(False, None, 0.0, None, 0)
        y_arr = np.fromiter(self._y, dtype=float)
        p_arr = np.fromiter(self._p, dtype=float)
        rolling = float(ece(y_arr, p_arr, n_bins=self.n_bins_ece))

        psi_val: Optional[float] = None
        if self.reference_feature is not None and self._x:
            psi_val = float(psi(self.reference_feature, np.fromiter(self._x, dtype=float), bins=self.n_bins_psi))

        cue = False
        reason: Optional[str] = None
        if rolling > self.ece_threshold:
            cue = True
            reason = f"rolling_ece={rolling:.4f} > {self.ece_threshold:.4f}"
        elif psi_val is not None and psi_val > self.psi_threshold:
            cue = True
            reason = f"psi={psi_val:.4f} > {self.psi_threshold:.4f}"
        return RecalibrationDecision(
            cue_fired=cue,
            reason=reason,
            rolling_ece=rolling,
            psi=psi_val,
            n_window=len(self._y),
        )


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    rec = SlidingWindowRecalibrator(window=2000, ece_threshold=0.07, check_every=200)

    n = 4000
    y_norm = rng.binomial(1, 0.3, size=n // 2)
    p_norm = np.clip(0.3 + 0.4 * (y_norm - 0.3) + rng.normal(0, 0.05, size=n // 2), 0, 1)
    decision = rec.update_batch(y_norm.tolist(), p_norm.tolist())
    print("after healthy stream :", decision)

    y_drift = rng.binomial(1, 0.3, size=n // 2)
    p_drift = np.clip(0.3 + 0.05 * (y_drift - 0.3) + rng.normal(0, 0.05, size=n // 2) + 0.20, 0, 1)
    decision = rec.update_batch(y_drift.tolist(), p_drift.tolist())
    print("after drifted stream :", decision)
