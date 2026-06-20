"""
Transfer matrix computation.

For each (source_dataset, target_dataset) pair:
  1. Train model on source
  2. Evaluate raw predictions on target → SLI vector (uncalibrated transfer)
  3. Recalibrate on small target sample → SLI vector (recalibrated transfer)
  4. Compare against in-domain baseline
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import json
import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split

from src.sli.compute import compute_all_slis, check_slo_compliance, SLIResult, SLOCompliance

logger = logging.getLogger(__name__)


@dataclass
class TransferResult:
    """Result of one (source, target, model) experiment."""
    source: str
    target: str
    model_name: str
    in_domain: bool  # True if source == target

    # Raw transfer (no recalibration)
    sli_raw: SLIResult
    slo_raw: SLOCompliance

    # After recalibration on target samples
    sli_recal: Optional[SLIResult] = None
    slo_recal: Optional[SLOCompliance] = None
    recal_samples: int = 0

    # Feature distribution shift
    psi_features: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict:
        d = {
            "source": self.source,
            "target": self.target,
            "model_name": self.model_name,
            "in_domain": self.in_domain,
            "sli_raw": self.sli_raw.to_dict(),
            "slo_raw": self.slo_raw.to_dict(),
        }
        if self.sli_recal is not None:
            d["sli_recal"] = self.sli_recal.to_dict()
            d["slo_recal"] = self.slo_recal.to_dict()
            d["recal_samples"] = self.recal_samples
        if self.psi_features is not None:
            d["psi_features"] = self.psi_features
        return d


@dataclass
class TransferMatrix:
    """Full N×N transfer matrix."""
    datasets: List[str]
    model_name: str
    results: List[TransferResult]

    def get(self, source: str, target: str) -> Optional[TransferResult]:
        for r in self.results:
            if r.source == source and r.target == target:
                return r
        return None

    def to_dataframe(self, metric: str = "ece") -> pd.DataFrame:
        """Pivot into an N×N DataFrame for a given metric."""
        rows = []
        for r in self.results:
            val = getattr(r.sli_raw, metric, None)
            if val is None and metric in ("brier_score",):
                val = r.sli_raw.brier.brier_score
            rows.append({"source": r.source, "target": r.target, "value": val})
        df = pd.DataFrame(rows)
        return df.pivot(index="source", columns="target", values="value")

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(
            {"datasets": self.datasets, "model": self.model_name,
             "results": [r.to_dict() for r in self.results]},
            indent=indent, default=str,
        )


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

MODELS = {
    "lr": lambda: LogisticRegression(max_iter=1000, random_state=42),
    "rf": lambda: RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42),
    "gb": lambda: GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42),
}


def _get_model(name: str):
    if name not in MODELS:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODELS.keys())}")
    return MODELS[name]()


# ---------------------------------------------------------------------------
# Core transfer experiment
# ---------------------------------------------------------------------------

def run_transfer_experiment(
    X_source: np.ndarray,
    y_source: np.ndarray,
    X_target: np.ndarray,
    y_target: np.ndarray,
    source_name: str,
    target_name: str,
    model_name: str = "gb",
    recal_fraction: float = 0.3,
    common_features: Optional[List[str]] = None,
) -> TransferResult:
    """
    Train on source, evaluate on target, optionally recalibrate.

    Parameters
    ----------
    X_source, y_source : source training data
    X_target, y_target : target evaluation data
    source_name, target_name : string labels
    model_name : "lr", "rf", or "gb"
    recal_fraction : fraction of target to use for recalibration
    common_features : if provided, select only these columns
    """
    in_domain = (source_name == target_name)

    # Train on source
    model = _get_model(model_name)
    model.fit(X_source, y_source)

    # Raw predictions on target
    y_prob_raw = model.predict_proba(X_target)[:, 1]

    # Compute raw SLIs
    sli_raw = compute_all_slis(y_target, y_prob_raw)
    slo_raw = check_slo_compliance(sli_raw)

    # Recalibration
    sli_recal = None
    slo_recal = None
    recal_n = 0

    if not in_domain and len(y_target) >= 30:
        # Split target into recal + eval
        recal_n = max(int(len(y_target) * recal_fraction), 20)
        idx = np.arange(len(y_target))
        np.random.RandomState(42).shuffle(idx)
        recal_idx = idx[:recal_n]
        eval_idx = idx[recal_n:]

        # Fit isotonic on recal split
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(y_prob_raw[recal_idx], y_target[recal_idx])

        # Evaluate on eval split
        y_prob_recal = iso.predict(y_prob_raw[eval_idx])
        sli_recal = compute_all_slis(y_target[eval_idx], y_prob_recal)
        slo_recal = check_slo_compliance(sli_recal)

    # Per-feature PSI (if both have same shape)
    psi_features = None
    if X_source.shape[1] == X_target.shape[1]:
        from src.sli.compute import compute_psi
        psi_features = {}
        for col_idx in range(X_source.shape[1]):
            feat_name = f"f{col_idx}" if common_features is None else common_features[col_idx]
            psi_features[feat_name] = compute_psi(
                X_source[:, col_idx], X_target[:, col_idx]
            )

    return TransferResult(
        source=source_name,
        target=target_name,
        model_name=model_name,
        in_domain=in_domain,
        sli_raw=sli_raw,
        slo_raw=slo_raw,
        sli_recal=sli_recal,
        slo_recal=slo_recal,
        recal_samples=recal_n,
        psi_features=psi_features,
    )


def build_transfer_matrix(
    datasets: Dict[str, Tuple[np.ndarray, np.ndarray]],
    model_name: str = "gb",
    recal_fraction: float = 0.3,
    common_features: Optional[List[str]] = None,
) -> TransferMatrix:
    """
    Build the full N×N transfer matrix.

    Parameters
    ----------
    datasets : dict mapping name → (X, y)
    model_name : model to use
    recal_fraction : fraction of target for recalibration
    common_features : feature names for PSI reporting

    Returns
    -------
    TransferMatrix with N² TransferResult entries.
    """
    names = sorted(datasets.keys())
    results = []

    for source_name in names:
        X_src, y_src = datasets[source_name]
        for target_name in names:
            X_tgt, y_tgt = datasets[target_name]
            logger.info(f"Transfer: {source_name} → {target_name} ({model_name})")

            result = run_transfer_experiment(
                X_source=X_src,
                y_source=y_src,
                X_target=X_tgt,
                y_target=y_tgt,
                source_name=source_name,
                target_name=target_name,
                model_name=model_name,
                recal_fraction=recal_fraction,
                common_features=common_features,
            )
            results.append(result)

    return TransferMatrix(
        datasets=names,
        model_name=model_name,
        results=results,
    )


def apply_recalibration(
    y_prob: np.ndarray,
    y_true_recal: np.ndarray,
    y_prob_recal: np.ndarray,
) -> np.ndarray:
    """
    Apply isotonic recalibration using a small labeled sample.

    Parameters
    ----------
    y_prob : predictions to recalibrate
    y_true_recal : labels from the recalibration sample
    y_prob_recal : predictions on the recalibration sample

    Returns
    -------
    Recalibrated predictions.
    """
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(y_prob_recal, y_true_recal)
    return iso.predict(y_prob)
