"""Prediction-to-policy translation.

Given a calibrated probability p = P(long wait), decide which scheduler
intervention band to use and emit the concrete policy record.  The three
bands --- admit, advise, guard --- map onto three integration surfaces:

  * admit:  annotation on Kubernetes Pod or Slurm job comment.
  * advise: advisory text plus a suggested alternative queue/GPU count.
  * guard:  eBPF gpu_ext hooks (memory pressure, util thresholds) to
             rate-limit the high-risk job in the presence of
             latency-sensitive workloads on the same node.

This module is deliberately small and free of network/scheduler deps so
that it can be embedded inside a Kubernetes mutating webhook, a Slurm
``job_submit`` Lua shim (via Python-Lua bridge), or an eBPF user-space
loader without additional dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


Band = Literal["admit", "advise", "guard"]


@dataclass(frozen=True)
class BandConfig:
    advise_low: float = 0.30
    advise_high: float = 0.70


@dataclass
class PolicyRecord:
    band: Band
    annotations: dict[str, str] = field(default_factory=dict)
    advisory_text: str = ""
    gpu_ext: dict[str, float] = field(default_factory=dict)


def classify(p: float, cfg: BandConfig = BandConfig()) -> Band:
    if p < cfg.advise_low:
        return "admit"
    if p < cfg.advise_high:
        return "advise"
    return "guard"


def translate(
    p: float,
    *,
    namespace: str,
    alt_queue: str | None = None,
    lower_gpus: int | None = None,
    cfg: BandConfig = BandConfig(),
) -> PolicyRecord:
    band = classify(p, cfg)
    rec = PolicyRecord(band=band)
    rec.annotations["scheduling-advisory/risk"] = {
        "admit": "low",
        "advise": "medium",
        "guard": "high",
    }[band]
    rec.annotations["scheduling-advisory/p_wait"] = f"{p:.3f}"
    if band == "advise":
        parts = [f"Queue-risk {p:.2f} is above the advisory threshold."]
        if alt_queue:
            parts.append(f"Consider routing to '{alt_queue}'.")
        if lower_gpus is not None:
            parts.append(f"Reducing GPU request to {lower_gpus} may reduce wait.")
        rec.advisory_text = " ".join(parts)
    elif band == "guard":
        rec.advisory_text = (
            f"Queue-risk {p:.2f} is in the guarded band. The scheduler will "
            "apply driver-level gpu_ext hooks to reduce interference on shared nodes."
        )
        rec.gpu_ext = {
            "util_threshold_pct": 85.0,
            "mem_pressure_pct": 90.0,
            "priority_nice": 10,
        }
    return rec
