# VGAC — Claim ↔ Code ↔ Artifact Map

This file gives a one-to-one trace between every empirical claim in the PEARC '26
paper and the file/script/artifact that backs it. It is the document a reviewer
can use to verify reproducibility without reading any code.

Throughout the table, paths are relative to the repository root.

## 0. Two views of the same code: `code/` and `src/`

The repository ships **two import-compatible Python packages** that implement the
same primitives at different levels of abstraction. Reviewers can use whichever
one matches their reading style:

| Concern | Curated `code/` (paper-tight) | Upstream `src/` (production) |
| --- | --- | --- |
| Calibration metrics | `code/evaluation/calibration.py` (`ece`, `mce`, `reliability_curve`, `brier_decomposition`) | `src/sli/compute.py` (`compute_ece`, `compute_brier_decomposition`, `compute_tail_calibration`, `compute_psi`, `compute_all_slis`, `check_slo_compliance`) |
| Tier qualification | `code/policy/generator.py::PolicyGenerator` | `src/tier/qualify.py` (`qualify_tiers`, `build_tier_matrix`, `TierQualification`) |
| Bootstrap CIs | `code/evaluation/bootstrap.py` | (not yet ported upstream — `code/` is the reference) |
| Submit-time capture | `code/integration/submit_capture.py` (lightweight adapter pattern) | `src/feature/extract_k8s_submit_features.py` (production K8s extractor) |
| Drift sensors | `code/harness/{psi,temporal_ece}.py` | `src/sli/compute.py::compute_psi` |
| Multi-source loaders | (not in `code/`) | `src/data/{unified_loader,alibaba_v2020,google_2019,gpu_v2025_dlrm,logs_loader}.py` |
| Training harnesses | `code/run_experiments.py` (single-file CLI) | `src/train/{run_cv,compare_models,train_baseline_lr,export_bundle}.py` |
| Cross-cluster transfer | (deferred to follow-on paper) | `src/transfer/matrix.py` |

The two packages produce identical results on shared primitives — the curated
`code/` modules are deliberately small so the *paper's* claims map cleanly to
specific functions, and the upstream `src/` package is the larger codebase
those primitives were curated from.

## 1. Submit-time capture (Section 4 of the paper, Table 2)

| Claim | Source | Backing artifact |
| --- | --- | --- |
| Periodic 30 s polling produces `r = -0.27` between `pending_at_submit` and observed wait time | `code/integration/` (capture pipeline) | `artifacts/vgac_submit_time_correlation.json` field `periodic_snapshot_30s.pearson_r` |
| 5 s submit-time capture restores `r = +0.44` | Same code paths, post-correction | `artifacts/vgac_submit_time_correlation.json` field `submit_time_capture_5s.pearson_r` |
| Submit-time capture is required for *correct decision rules* (not just data quality) | `docs/ARCHITECTURE.md` §2 | Notebook cell **Step 1 — Submit-time capture correlation** demonstrates the *protocol*; absolute `r` on the public sample differs from the paper's full-trace headline. |

## 2. Floor model — 2 features (Section 5.2, Table 3)

| Claim | Source | Backing artifact |
| --- | --- | --- |
| Floor classifier on `pending_at_submit` + `running_at_submit` achieves AUROC = 0.756 | `code/run_experiments.py` (train_floor) | `artifacts/vgac_floor_vs_ceiling.csv`, row `floor` |
| Floor classifier ECE = 0.077 after isotonic recalibration | Same | Same row, column `ece` |
| Mid-range probabilities (0.35–0.52) show predicted-vs-actual gaps up to 0.188 | `code/evaluation/calibration.py` reliability-curve helper | Notebook cell **Step 3** + `figures/calibration_curve.png` |
| Floor qualifies Tier 1 only; Tier 2 marginal (ECE 0.077 exceeds 0.07 prereq. by 0.007) | `code/policy/generator.py::get_qualified_tier` | `artifacts/vgac_floor_vs_ceiling.csv`, columns `tier_1_annotate` … `tier_4_gate` |

## 3. Production VGAC — 17+ features (Section 5.3, Tables 4 & 5)

| Claim | Source | Backing artifact |
| --- | --- | --- |
| Production model achieves AUROC = 0.969 | `code/run_experiments.py` (train_ceiling) | `artifacts/vgac_floor_vs_ceiling.csv`, row `ceiling` |
| Production model achieves ECE = 0.005 (well below Gate prereq. of 0.03) | Same | Same row, column `ece` |
| Production model qualifies for **all four** tiers (1–4) | `code/policy/generator.py` | `artifacts/vgac_floor_vs_ceiling.csv`, columns `tier_*` |
| 12.7× ECE improvement (0.077 → 0.005) under feature enrichment | Computed directly from the two rows of the artifact | `artifacts/vgac_floor_vs_ceiling.csv` |

> **Provenance note.** `vgac_floor_vs_ceiling.csv` and `vgac_submit_time_correlation.json` reproduce the paper's headline numbers verbatim. Their underlying EKS dataset (650 raw → 582 usable rows for the floor study; 11 982 K8s+Slurm events for the ceiling study) lives on private Saint Peter's University AWS infrastructure and is not redistributed; the reconstruction protocol is documented in `code/integration/` and `docs/ARCHITECTURE.md`. The shipped `data/samples/` is a redistribution-safe cross-cluster slice that lets `notebooks/reproducibility.ipynb` execute end-to-end and reproduce the *qualitative* behaviour (tier qualification, graceful degradation, gain from feature enrichment).

## 4. Bootstrap confidence intervals

| Claim | Source | Backing artifact |
| --- | --- | --- |
| 95 % CIs reported for AUROC / AUPRC / Brier / ECE | `code/evaluation/bootstrap.py` (B = 1000 percentile) | `artifacts/bootstrap_confidence_intervals.csv` |
| Stratified k-fold (k = 5, seed = 42) | `code/run_experiments.py::cv_split` | Reproduced deterministically with the seed |

## 5. Tier qualification framework (Section 3 of the paper)

| Claim | Source | Backing artifact |
| --- | --- | --- |
| Tier 1 prereq.: ECE ≤ 0.10 | `code/policy/generator.py::TIER_PREREQS` | Reproduced via the `assert` in cell **"Step 2 — Tier prerequisite table"** |
| Tier 2 prereq.: ECE ≤ 0.07 | Same | Same |
| Tier 3 prereq.: ECE ≤ 0.05 | Same | Same |
| Tier 4 prereq.: ECE ≤ 0.03 | Same | Same |
| Action thresholds `eps_a` ∈ {0.3, 0.5, 0.7, 0.9} | Same | Same |
| Graceful degradation: ECE drift triggers tier retirement | `code/policy/inference_router.py::route` | Notebook cell **"Step 5 — Graceful degradation simulation"** |

## 6. Cross-environment positioning (referenced from the companion benchmark paper)

The PEARC '26 paper reports its primary numbers on a single EKS cluster
(`n = 650`, 582 usable). For context, the same pipeline is exercised on
EKS-Dec, Slurm + DCGM, Alibaba 2020/2023, and Google Borg 2019 in the
companion benchmark paper; the relevant artefacts ship with this repo for
inspection but are not the primary subject of the PEARC paper:

| Environment | Sample data | Cross-env metric |
| --- | --- | --- |
| EKS-Dec (1.2 M jobs) | `data/samples/eks_dec_sample.csv` | `artifacts/all_5_models_results.csv` |
| Slurm + DCGM (555 jobs) | `data/samples/slurm_sample.csv` | `artifacts/paper2_paper3_experiments.json` |
| Alibaba GPU 2020 | `data/samples/alibaba_sample.csv` | `artifacts/cross_domain_analysis.json` |
| Google Borg 2019 | `data/samples/borg_sample.csv` | `artifacts/cross_domain_analysis.json` |

Cross-environment PSI on `pending_ratio` (PSI = 12.4) and the temporal-ECE
trajectory under drift live in `artifacts/cross_domain_analysis.json` and are
plotted in the notebook.

## 7. Reproducibility contract

The notebook `notebooks/reproducibility.ipynb` provides a single-command
reproduction. Each cell is labelled with the paper section it backs and the
artifact it consumes or produces. Running the notebook end-to-end on the
sample data should regenerate every figure in the `figures/` directory and
print every numerical claim above.

If a number disagrees with the paper, it is a bug and should be reported via
the GitHub issue tracker.
