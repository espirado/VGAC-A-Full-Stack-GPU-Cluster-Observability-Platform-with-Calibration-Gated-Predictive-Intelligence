# VGAC Architecture

This document describes the system architecture of VGAC (the artifact accompanying the
PEARC '26 short paper). It mirrors the structure of the paper but adds implementation-level
detail that did not fit in the 4-page camera-ready.

## 1. Conceptual model

VGAC is a **full-stack GPU cluster observability platform** whose predictive
intelligence is **calibration-gated**. The platform's distinguishing claim is
that it elevates the model's *measured* calibration to a first-class observability
signal — and uses it to gate its own actions. The core invariant is:

> A predictive intervention is permitted only when the underlying probability is
> calibrated well enough to support it, *as continuously observed by the platform*.

The platform has three observability layers, each with a clearly defined contract
to the layer above it:

```
+------------------------------------------------------------------+
|  L3: Decision Layer    (calibration-gated graduated intervention)|
|      - Tier rules:  apply a if  p_hat >= eps_a  AND  ECE <= R_a  |
|      - Per-tier qualification gate (Annotate, Warn, Suggest, Gate)
|      - Re-evaluated every request against L2's rolling ECE       |
+------------------------------------------------------------------+
|  L2: Prediction + Calibration-Telemetry Layer                    |
|      - Calibration-aware classifier (LR / GB / RF / XGB / LGBM)  |
|      - Isotonic post-hoc calibration                             |
|      - Rolling-window ECE / MCE / Brier monitor (the "first-     |
|        class observability signal" cited in the paper)           |
+------------------------------------------------------------------+
|  L1: Capture Layer     (submit-time observability instrumentation)|
|      - 5 s polling  -> phase-transition detection                |
|      - Records cluster state at the exact moment a job submits   |
|      - Writes to feature store consumed by L2                    |
|      - Without this, downstream features are lagging indicators  |
|        (paper's r = -0.27 -> +0.44 correction in Section 4)      |
+------------------------------------------------------------------+
```

Notice that **calibration telemetry feeds back into the decision layer in real
time**: this is what makes the platform *full-stack* observability rather than
just a pipeline. The system observes its own predictions and lets those
observations restrict its own actions — graceful degradation is therefore a
visible, measurable property of the platform, not a build-time decision.

## 2. Layer 1 — Submit-time capture

The most consequential implementation detail in VGAC is L1. Periodic snapshots
introduce a latent timing error: by the time a job is observed as `Pending`, the
queue has already drained, producing the inverted Pearson correlation
`r = -0.27` reported in the paper (Table 2).

**Implementation:**

- Poll the cluster API every 5 s.
- Maintain a per-job phase state machine: `New -> Pending -> Running -> Succeeded`.
- On each phase transition, record the global queue state vector
  `(pending_at_submit, running_at_submit, ... )`.
- Write the captured row to the feature store (parquet partition by date).

**Guarantee:** every training row carries the queue state observed at the
*moment of submission*, not at the next polling boundary. This restores the
expected correlation `r = +0.44` in Table 2 of the paper.

Code lives under `code/integration/` and (for the K8s-side admission webhook)
in the companion Sentinel repository.

## 3. Layer 2 — Prediction

VGAC trains five classifier families with identical hyperparameters across
environments, applies isotonic post-hoc calibration on a held-out validation
fold, and exposes a single inference entry point.

**Inputs (17+ features at the ceiling configuration):**

- *Job attributes:* requested CPU, memory, GPU count, ephemeral disk, RDMA
  flag, node-selector hash, affinity terms, tolerations.
- *Cluster state:* `pending_at_submit`, `running_at_submit`, `pending_ratio`,
  per-node CPU and memory pressure, capacity headroom.
- *Temporal:* hour-of-week, day-of-week, weekend flag.
- *Derived:* interaction terms (e.g., `pending_ratio * gpu_request`).

**Floor configuration (2 features):** `pending_at_submit`, `running_at_submit`.
This is the minimum-viable model and is used in the paper to demonstrate that
the tier framework gracefully accommodates clusters with no instrumentation
beyond queue depth.

**Outputs per request:**

| Field | Type | Meaning |
| --- | --- | --- |
| `prob` | `float in [0,1]` | Calibrated `P(wait > T)` |
| `qualified_tier` | `int 0..4` | Highest tier whose prerequisites currently hold |
| `confidence` | `float in [0,1]` | (Optional) variance proxy across CV folds |
| `model_id` | `str` | Hash of the deployed model + calibrator |
| `current_ece` | `float` | Rolling ECE from the calibration monitor |

The rolling ECE is computed on a sliding window of recent labelled rows. If
ECE drifts above a tier's prerequisite, that tier is automatically retired
until recalibration.

## 4. Layer 3 — Decision (graduated intervention)

The decision layer applies the rule:

> apply action `a` to job `j` iff `p_hat_j >= eps_a` **and** `ECE <= R_a`.

| Tier | Action | `eps_a` | `R_a` (ECE prereq.) | Rationale |
| --- | --- | --- | --- | --- |
| 1 — Annotate | Add risk label | 0.30 | <= 0.10 | Informational only |
| 2 — Warn | Notify user | 0.50 | <= 0.07 | Interrupts workflow |
| 3 — Suggest | Recommend reschedule | 0.70 | <= 0.05 | Implies high-confidence delay |
| 4 — Gate | Require confirmation | 0.90 | <= 0.03 | Blocks submission |

**Graceful degradation.** A model with ECE = 0.06 satisfies tiers 1 and 2 but
not 3 or 4; the system silently stops emitting Suggest/Gate annotations until
recalibration restores qualification. This makes degradation visible as
reduced functionality rather than as silent miscalibration.

## 5. Integration topologies

VGAC ships with two reference integrations.

### 5.1 Kubernetes mutating admission webhook

```
  +--- pod CREATE ---+
  |                  |
  v                  |
[K8s API server] --> [VGAC webhook] --> [predict service]
                          |
                          | annotate(prob, tier)
                          v
                       [Pod object]
```

- The webhook intercepts each `Pod` create.
- Extracts resource requests, node affinity, tolerations, queue-state.
- Calls the prediction service over gRPC (target P99 latency < 10 ms).
- Reads the *current* rolling ECE from the model registry.
- Selects the qualified tier, annotates the Pod, returns mutated object.

### 5.2 Slurm `job_submit` plugin

A C plugin attaches the predicted probability and qualified tier to job
metadata via the `Comment` field. Warn-level alerts surface through
`slurm_info` when the qualified tier is `>= 2` and `prob >= eps_2`. Suggest-
and Gate-level outcomes update Slurm partitions or fire a `slurm_strict`
rejection.

### 5.3 Common interface

Both integrations share a single contract:

```python
prob = predict(features)
tier = get_qualified_tier(prob, current_ece)
annotate(job, prob, tier)
```

The tier check is evaluated *per request* against the model's rolling ECE,
so qualification is dynamic, not a build-time decision.

## 6. Telemetry and feedback

VGAC exposes Prometheus metrics that mirror the calibration-monitor scope:

- `vgac_predict_latency_seconds` (histogram).
- `vgac_rolling_ece` (gauge).
- `vgac_tier_qualified{tier="1..4"}` (gauge, 0/1).
- `vgac_tier_emitted_total{tier="1..4"}` (counter).
- `vgac_recalibration_triggered_total` (counter).

Together they let an operator answer the only operational question that
matters: *"is the system permitted to act, and at what severity?"*

## 7. Mapping to the published paper

| Paper section | This document | Code path |
| --- | --- | --- |
| Sec. 3 — Graduated intervention framework | Sec. 4 above | `code/policy/generator.py` |
| Sec. 4 — Submit-time capture | Sec. 2 above | `code/integration/` (capture pipeline) |
| Sec. 5 — Empirical validation | (See `docs/METHODOLOGY.md`) | `code/run_experiments.py` |
| Sec. 6 — Integration reference impls | Sec. 5 above | `code/policy/inference_router.py`, `code/policy/gpu_ext_bridge.py` |

## 8. Out of scope

Three deliberate non-goals for the camera-ready artifact:

1. **Online learning** — VGAC's models are batch-trained; calibration
   monitoring is online but model updates require an explicit retraining
   trigger.
2. **Multi-tenant cost models** — the cost matrix used to choose the four
   ECE prerequisites is fixed; per-tenant cost matrices are future work.
3. **Cross-cluster transfer** — the camera-ready validates on EKS only;
   per-cluster transfer characteristics are characterised in the companion
   benchmark paper.
