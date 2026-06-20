"""
SLI (Service Level Indicator) computation for queue-delay prediction.

This module implements the four SLIs defined in Paper 2:
  - SLI-1: Expected Calibration Error (ECE)
  - SLI-2: Brier Score Decomposition (reliability / resolution / uncertainty)
  - SLI-3: Tail Calibration
  - SLI-4: Population Stability Index (PSI)

Usage:
    from src.sli import compute_all_slis, check_slo_compliance
    slis = compute_all_slis(y_true, y_prob, n_bins=15)
    compliance = check_slo_compliance(slis)
"""

from .compute import (
    compute_ece,
    compute_mce,
    compute_brier_decomposition,
    compute_tail_calibration,
    compute_psi,
    compute_all_slis,
    check_slo_compliance,
    SLIResult,
    BrierDecomposition,
    TailCalibrationResult,
    SLOCompliance,
)

__all__ = [
    "compute_ece",
    "compute_mce",
    "compute_brier_decomposition",
    "compute_tail_calibration",
    "compute_psi",
    "compute_all_slis",
    "check_slo_compliance",
    "SLIResult",
    "BrierDecomposition",
    "TailCalibrationResult",
    "SLOCompliance",
]
