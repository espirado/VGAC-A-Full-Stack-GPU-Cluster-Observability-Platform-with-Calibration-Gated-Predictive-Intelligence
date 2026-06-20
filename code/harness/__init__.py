"""Drift-detection harness.

    psi          - Population Stability Index (Siddiqi banking convention).
    temporal_ece - sliding-window ECE sensor.
"""

from .psi import psi
from .temporal_ece import ece_equalmass, TemporalECEResult

__all__ = ["psi", "ece_equalmass", "TemporalECEResult"]
