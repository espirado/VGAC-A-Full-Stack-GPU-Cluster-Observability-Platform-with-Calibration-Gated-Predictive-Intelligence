# VGAC — Claim ↔ Code ↔ Artifact Map

This file gives a one-to-one trace between every empirical claim in the PEARC '26
paper and the file/script/artifact that backs it. It is the document a reviewer
can use to verify reproducibility without reading any code.

Throughout the table, paths are relative to the repository root.

## 1. Submit-time capture (Section 4 of the paper)

| Claim | Source | Backing artifact |
| --- | --- | --- |
| Periodic 30 s polling produces `r = -0.27` between `pending_at_submit` and wait time | `code/integration/` (capture pipeline) and `code/run_experiments.py` step 1 | `artifacts/cross_domain_analysis.json` field `pre_correction_pearson` (or recomputed in the notebook) |
| 5 s submit-time capture restores `r = +0.44` | Same code paths, post-correction | `artifacts/cross_domain_analysis.json` field `post_correction_pearson` |
| Submit-time capture is required for *correct decision rules* (not just data quality) | `docs/ARCHITECTURE.md` §2 | Reproduced in `notebooks/reproducibility.ipynb`, cell **"Step 1 — Submit-time capture correlation"** |

## 2. Floor model — 2 features (Section 5.2 of the paper)

| Claim | Source | Backing artifact |
| --- | --- | --- |
| Floor classifier (logistic regression on `pending_at_submit`, `running_at_submit`) achieves AUROC = 0.756 | `code/run_experiments.py` (train_floor) | `artifacts/all_5_models_results.csv` row `eks_dec / floor / lr` |
| Floor classifier ECE = 0.077 after isotonic recalibration | Same | Same row, column `ece` |
| Mid-range probabilities (0.35–0.52) show predicted-vs-actual gaps up to 0.188 | `code/evaluation/calibration.py` reliability-curve helper | `figures/calibration_curve.png` and the bin-level table the notebook prints |
| Floor qualifies Tier 1 only; Tier 2 marginal (0.077 vs 0.07) | `code/policy/generator.py::get_qualified_tier` | Notebook cell **"Step 3 — Floor tier qualification"** |

## 3. Production VGAC — 17+ features (Section 5.3 of the paper)

| Claim | Source | Backing artifact |
| --- | --- | --- |
| Production model achieves AUROC = 0.969 | `code/run_experiments.py` (train_ceiling) | `artifacts/all_5_models_results.csv` row `eks_dec / ceiling / gb` |
| Production model achieves ECE = 0.005 (well below Gate prereq. of 0.03) | Same | Same row, column `ece` |
| Production model qualifies for **all four** tiers (1–4) | `code/policy/generator.py` | Notebook cell **"Step 4 — Ceiling tier qualification"** |
| 12.7× ECE improvement (0.077 → 0.005) under feature enrichment | Direct from the table above | `artifacts/all_5_models_results.csv` |

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
