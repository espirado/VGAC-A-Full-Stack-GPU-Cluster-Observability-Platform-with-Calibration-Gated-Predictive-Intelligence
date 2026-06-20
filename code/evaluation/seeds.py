"""Deterministic seeding for VGAC experiments.

Used by ``code/run_experiments.py`` and ``notebooks/reproducibility.ipynb``.
The notebook section "Step 0 - Determinism" depends on this module being
importable; do not change the function name without updating callers.

Setting ``PYTHONHASHSEED`` here is best-effort: it only affects child
processes spawned after the call.
"""
from __future__ import annotations

import os
import random


def seed_everything(seed: int = 42) -> int:
    """Seed every PRNG VGAC uses and return the seed.

    Covers Python ``random``, NumPy, scikit-learn (via NumPy), and
    XGBoost / LightGBM (which honour the global NumPy state when no
    explicit seed is passed). Returns the seed for chaining.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - numpy is a hard dep
        pass
    return seed


if __name__ == "__main__":
    s = seed_everything(42)
    import numpy as np

    print(f"Seeded everything with {s}; np.random.rand() = {np.random.rand():.6f}")
