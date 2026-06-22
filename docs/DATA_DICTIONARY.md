# VGAC — Data Dictionary

This file documents the schema of every CSV / JSON in `data/samples/` and
`artifacts/`. All numeric values use SI units unless otherwise stated.

## 1. `data/samples/eks_dec_sample.csv`

EKS-Dec sample — one row per submitted Pod. Sampled from the full 1.2 M-row
trace; queue-state columns are captured at *submission time* per the
methodology in §4 of the paper.

| Column | Type | Units | Description |
| --- | --- | --- | --- |
| `pending_ratio` | float | ratio | Pending GPUs divided by total cluster GPUs at submit time |
| `queue_depth_norm` | int | jobs | Normalised count of pending jobs |
| `fragmentation_score` | float | [0,1] | Heuristic fragmentation score over GPU nodes |
| `congestion_score` | float | [0,1] | Composite congestion proxy (queue × utilisation) |
| `pending_gpus` | int | GPUs | Sum of GPU requests across pending jobs |
| `total_pending` | int | jobs | Total pending jobs at submit time |
| `running_gpus` | int | GPUs | Sum of GPU allocations across running jobs |
| `gpu_nodes_alloc` | int | nodes | GPU nodes with at least one allocation |
| `gpu_nodes_total` | int | nodes | Total GPU nodes online |
| `label_long_wait` | int | 0/1 | 1 iff observed wait > P90 wait for this environment |
| `raw_score` | float | [0,1] | Pre-calibration model score (provided for reference) |

## 2. `data/samples/slurm_sample.csv`

Slurm + DCGM sample (n = 555 jobs). All columns of EKS-Dec **plus** the DCGM
telemetry summaries:

| Column | Type | Units | Description |
| --- | --- | --- | --- |
| `util_gpu_mean` | float | % | Cluster-mean SM utilisation at submit time |
| `util_gpu_std` | float | % | Cross-GPU std-dev of SM utilisation |
| `util_mem_mean` | float | % | Cluster-mean memory bus utilisation |
| `temp_c_max` | float | °C | Hottest GPU at submit time |
| `power_w_mean` | float | W | Mean GPU power draw |

DCGM aggregates are pre-computed; full per-GPU traces are not redistributed
in the sample (they are large and contain serial numbers).

## 3. `data/samples/alibaba_sample.csv` and `borg_sample.csv`

Submit-time features for the public Alibaba 2020/2023 GPU traces and Google
Borg 2019. Same schema as EKS-Dec but without DCGM telemetry. `pending_ratio`
is reconstructed via *trace replay*: at each row's submit timestamp we count
the concurrent pending jobs from the same trace.

## 4. `data/samples/cross_domain_analysis.json`

Self-describing JSON. Top-level keys:

```
{
  "psi_per_feature": { ... per-feature PSI values across env pairs ... },
  "temporal_ece":    { "windows": [...], "ece":    [...] },
  "transfer_matrix": { "<src>_to_<dst>": { "auroc_gap": ..., "ece_ratio": ...} }
}
```

`psi_per_feature.pending_ratio` reaches 12.4 across `eks_dec -> slurm`,
which the paper cites as evidence of catastrophic distribution shift.

## 5. `data/samples/drift_metrics.json`

```
{
  "window_index": [...],
  "rolling_ece":  [...],
  "rolling_brier":[...],
  "trigger":     [false, false, ..., true, ...]
}
```

`trigger == true` indicates a recalibration event under the rule
`PSI > 0.1 OR weekly_ece > 0.07`.

## 6. `artifacts/all_5_models_results.csv`

Long-form benchmark table consumed by Tables 3 and 4 of the paper.

| Column | Description |
| --- | --- |
| `env` | One of `eks_dec`, `slurm`, `alibaba`, `borg` |
| `feature_set` | `floor` (2 features) or `ceiling` (17+ features) |
| `model` | One of `lr`, `rf`, `gb`, `xgb`, `lgbm` |
| `auroc` | Area under ROC |
| `auprc` | Area under precision-recall |
| `brier` | Brier score |
| `ece` | Expected calibration error (15 equal-mass bins) |
| `mce` | Maximum calibration error |
| `n_test` | Test-fold size |
| `seed` | RNG seed (=42 throughout) |

## 7. `artifacts/bootstrap_confidence_intervals.csv`

| Column | Description |
| --- | --- |
| `env` | Environment label |
| `feature_set` | `floor` / `ceiling` |
| `model` | Model family |
| `metric` | `auroc` / `auprc` / `brier` / `ece` |
| `point` | Point estimate |
| `lower_95` | Lower 95 % bootstrap percentile (B = 1000) |
| `upper_95` | Upper 95 % bootstrap percentile |

## 8. `artifacts/paper2_paper3_experiments.json`

Snapshot of broader queue-risk experiments from the wider research program,
kept here so the cross-environment context (Slurm 555-job, tier qualification,
tail calibration, FAR) that motivates VGAC's gating tiers is fully auditable.
Top-level keys:

- `submit_only` / `dcgm_enriched` — Slurm 555 results.
- `tier_qualification` — per-model tier assignment.
- `tail_calibration` — ECE evaluated at thresholds {0.5, ..., 0.9}.
- `false_action_rates` — FAR per tier.

## 9. `data/samples/slurm_training_dataset.csv` (full Slurm training dataset)

The complete 555-job Slurm + DCGM training dataset from the wider Saint
Peter's GPU-cluster reliability research program, imported here for full
reviewability of the cross-environment context that motivates VGAC's
calibration tiers. Same columns as `slurm_sample.csv` but with all 38
features and 555 rows (no sampling). 141 KB.

## 10. `data/samples/slurm_training_dataset_summary.json`

Top-level summary computed once over `slurm_training_dataset.csv`:
positive rate, P90 wait threshold, feature counts, missing-value
fractions per column. Exists so reviewers can audit dataset shape
without loading the CSV.

## 11. `artifacts/legacy_paper4/` (upstream supporting artifacts)

Historical artifacts produced during the broader research program that
preceded this short paper. Kept for provenance and cross-reference; not
required by the camera-ready text.

| File | Schema |
| --- | --- |
| `landscape.csv`              | Cross-environment landscape: per-(env, model, feature_set) AUROC and ECE. |
| `sli_slo_analysis.json`      | SLI / SLO compliance results across the 4-environment matrix. |
| `heterogeneous_analysis.json`| Heterogeneity analysis: feature richness, label imbalance, calibration shape. |
| `deep_analysis_results.json` | Deep-dive artefact aggregating reliability-diagram bins and tail-gap statistics. |

## 12. `artifacts/legacy_paper2/` (queue-risk evaluation artifacts)

| File | Schema |
| --- | --- |
| `model_evaluation_slurm.json` | Per-model AUROC / AUPRC / Brier / ECE on the 555-job Slurm split with DCGM enrichment. |
| `paper2_notebook_results.json`| Aggregate results emitted at the end of the paper-2 reproducibility notebook. |

## 13. Notes on missing values and de-duplication

EKS-Dec rows are de-duplicated on `(pod_uid, transition_timestamp)` to remove
double-emission caused by Kubernetes informer re-deliveries. After
de-duplication n = 582 in the EKS sample (down from 650 raw events). Slurm
data are not de-duplicated because `job_id` is already unique.

Missing values in DCGM aggregates are imputed with 0.0 because the absence
of telemetry is itself signal (typically corresponds to a cold node).
