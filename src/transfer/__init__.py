"""
Cross-domain transfer experiments.

Train on source dataset, evaluate on target dataset,
compute full SLI vector for each (source, target) pair.

This produces the N×N transfer matrix that is the hero figure
for Papers 2 and 4.
"""

from .matrix import (
    run_transfer_experiment,
    build_transfer_matrix,
    apply_recalibration,
    TransferResult,
    TransferMatrix,
)

__all__ = [
    "run_transfer_experiment",
    "build_transfer_matrix",
    "apply_recalibration",
    "TransferResult",
    "TransferMatrix",
]
