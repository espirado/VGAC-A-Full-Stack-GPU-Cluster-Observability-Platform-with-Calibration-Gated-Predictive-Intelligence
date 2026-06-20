"""
Tier qualification logic.

Given SLI measurements, determine which intervention tiers
a model qualifies for and compute the false-action rate at each tier.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
import json

import numpy as np
import pandas as pd

from src.sli.compute import (
    compute_all_slis,
    SLIResult,
    TIER_ECE_THRESHOLDS,
    TIER_NAMES,
)


@dataclass
class TierQualification:
    """Qualification result for one model on one dataset."""
    model_name: str
    dataset_name: str
    ece: float
    auroc: Optional[float]

    # Per-tier qualification
    tiers: Dict[int, bool]  # {1: True, 2: True, 3: False, 4: False}
    highest_tier: int
    highest_tier_name: str

    # Per-tier false-action rates (if predictions and labels provided)
    false_action_rates: Optional[Dict[int, float]] = None

    def to_dict(self) -> Dict:
        return asdict(self)


# Tier probability thresholds (Paper 3, Table 1)
TIER_PROB_THRESHOLDS = {
    1: 0.3,   # Annotate
    2: 0.5,   # Warn
    3: 0.7,   # Suggest
    4: 0.9,   # Gate
}


def qualify_tiers(
    sli: SLIResult,
    model_name: str,
    dataset_name: str,
    y_true: Optional[np.ndarray] = None,
    y_prob: Optional[np.ndarray] = None,
) -> TierQualification:
    """
    Determine tier qualification based on SLI measurements.

    If y_true and y_prob are provided, also compute false-action rates
    for each tier (the rate at which the tier triggers but the actual
    outcome was not a violation).
    """
    tiers = {}
    highest = 0

    for tier in sorted(TIER_ECE_THRESHOLDS.keys()):
        qualified = sli.ece <= TIER_ECE_THRESHOLDS[tier]
        tiers[tier] = qualified
        if qualified:
            highest = tier

    # False-action rates
    false_action_rates = None
    if y_true is not None and y_prob is not None:
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)
        false_action_rates = {}

        for tier, eps in TIER_PROB_THRESHOLDS.items():
            triggered = y_prob >= eps
            n_triggered = triggered.sum()
            if n_triggered == 0:
                false_action_rates[tier] = 0.0
                continue
            # False action = triggered but actual was NOT a violation (y_true == 0)
            false_actions = (triggered & (y_true == 0)).sum()
            false_action_rates[tier] = float(false_actions / n_triggered)

    return TierQualification(
        model_name=model_name,
        dataset_name=dataset_name,
        ece=sli.ece,
        auroc=sli.auroc,
        tiers=tiers,
        highest_tier=highest,
        highest_tier_name=TIER_NAMES.get(highest, "none"),
        false_action_rates=false_action_rates,
    )


def build_tier_matrix(
    models: Dict[str, Dict[str, tuple]],
    n_bins: int = 15,
) -> pd.DataFrame:
    """
    Build a (model × dataset) matrix of tier qualifications.

    Parameters
    ----------
    models : dict of {model_name: {dataset_name: (y_true, y_prob)}}

    Returns
    -------
    DataFrame with columns: model, dataset, ece, auroc, highest_tier, tier_name,
                            false_action_1, false_action_2, false_action_3, false_action_4
    """
    rows = []

    for model_name, datasets in models.items():
        for dataset_name, (y_true, y_prob) in datasets.items():
            sli = compute_all_slis(y_true, y_prob, n_bins=n_bins)
            qual = qualify_tiers(sli, model_name, dataset_name, y_true, y_prob)

            row = {
                "model": model_name,
                "dataset": dataset_name,
                "ece": qual.ece,
                "auroc": qual.auroc,
                "highest_tier": qual.highest_tier,
                "tier_name": qual.highest_tier_name,
            }
            if qual.false_action_rates:
                for tier, rate in qual.false_action_rates.items():
                    row[f"false_action_{tier}"] = rate

            rows.append(row)

    return pd.DataFrame(rows)
