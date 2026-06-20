"""Percentile bootstrap confidence intervals for ranking and calibration metrics.

Used by ``code/run_experiments.py`` to emit
``artifacts/bootstrap_confidence_intervals.csv``. The default of
B=1000 percentile resamples follows the paper's Methodology section
("1000-iteration percentile bootstrap for 95% CIs").

Supported metrics out of the box:
    - 'auroc'  : sklearn.metrics.roc_auc_score
    - 'auprc'  : sklearn.metrics.average_precision_score
    - 'brier'  : sklearn.metrics.brier_score_loss
    - 'ece'    : code.evaluation.calibration.ece (default 15 bins)

Custom metrics can be passed as a callable ``fn(y_true, y_prob) -> float``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Tuple, Union

import numpy as np

from .calibration import ece as _ece

MetricFn = Callable[[np.ndarray, np.ndarray], float]


@dataclass(frozen=True)
class BootstrapResult:
    metric: str
    point: float
    lo: float
    hi: float
    iters: int
    alpha: float

    def as_row(self) -> Dict[str, float | str | int]:
        return {
            "metric": self.metric,
            "point": self.point,
            "ci_lo": self.lo,
            "ci_hi": self.hi,
            "iters": self.iters,
            "alpha": self.alpha,
        }


def _resolve(metric: Union[str, MetricFn]) -> Tuple[str, MetricFn]:
    if callable(metric):
        return metric.__name__, metric
    name = metric.lower()
    if name == "auroc":
        from sklearn.metrics import roc_auc_score

        return "auroc", roc_auc_score
    if name == "auprc":
        from sklearn.metrics import average_precision_score

        return "auprc", average_precision_score
    if name == "brier":
        from sklearn.metrics import brier_score_loss

        return "brier", brier_score_loss
    if name == "ece":
        return "ece", lambda y, p: _ece(y, p, n_bins=15)
    raise ValueError(f"Unknown metric '{metric}'. Pass a callable or one of: auroc, auprc, brier, ece.")


def percentile_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: Union[str, MetricFn] = "auroc",
    n_iter: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> BootstrapResult:
    """Two-sided percentile bootstrap CI for ``metric(y_true, y_prob)``.

    Resamples observations *with* replacement at each iteration and
    re-evaluates the metric. Returns the original-sample point estimate
    along with the (alpha/2, 1-alpha/2) percentile bounds.

    Skips iterations where the resample contains a single class for
    metrics that require both (AUROC, AUPRC) - this matches the
    behaviour of ``sklearn``'s metric functions which raise on such
    inputs.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob length mismatch")

    name, fn = _resolve(metric)
    point = float(fn(y_true, y_prob))

    rng = np.random.default_rng(seed)
    n = len(y_true)
    samples = []
    for _ in range(n_iter):
        idx = rng.integers(0, n, size=n)
        ys, ps = y_true[idx], y_prob[idx]
        if name in {"auroc", "auprc"} and len(np.unique(ys)) < 2:
            continue
        try:
            samples.append(float(fn(ys, ps)))
        except (ValueError, ZeroDivisionError):
            continue

    if not samples:
        return BootstrapResult(name, point, point, point, n_iter, alpha)

    arr = np.asarray(samples)
    lo = float(np.percentile(arr, 100 * (alpha / 2)))
    hi = float(np.percentile(arr, 100 * (1 - alpha / 2)))
    return BootstrapResult(metric=name, point=point, lo=lo, hi=hi, iters=len(samples), alpha=alpha)


def percentile_ci_many(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metrics: Iterable[Union[str, MetricFn]] = ("auroc", "auprc", "brier", "ece"),
    n_iter: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> list[BootstrapResult]:
    """Convenience: run ``percentile_ci`` for several metrics on the
    same predictions, sharing a seed so iterations line up."""
    return [
        percentile_ci(y_true, y_prob, m, n_iter=n_iter, alpha=alpha, seed=seed + i)
        for i, m in enumerate(metrics)
    ]


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 1_000
    y = rng.binomial(1, 0.3, size=n)
    p = np.clip(0.3 + 0.4 * (y - 0.3) + rng.normal(0, 0.1, size=n), 0, 1)
    for r in percentile_ci_many(y, p, n_iter=200):
        print(f"{r.metric:>6}: {r.point:.4f}  [{r.lo:.4f}, {r.hi:.4f}]  ({r.iters} iters)")
