"""Operational helpers for VGAC.

The modules here are the runtime side of the calibration-gated
platform: continuously *observe* model calibration and trigger
recalibration when SLOs are breached.

    recalibrator.SlidingWindowRecalibrator - emits a recalibration cue
        when rolling-window ECE exceeds an SLO threshold or PSI on a
        feature exceeds Siddiqi's banking convention (PSI > 0.1).
"""

from .recalibrator import SlidingWindowRecalibrator, RecalibrationDecision

__all__ = ["SlidingWindowRecalibrator", "RecalibrationDecision"]
