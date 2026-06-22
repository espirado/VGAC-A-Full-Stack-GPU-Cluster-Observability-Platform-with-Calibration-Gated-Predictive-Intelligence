# VGAC — A Full-Stack GPU Cluster Observability Platform with Calibration-Gated Predictive Intelligence

[![DOI](https://img.shields.io/badge/DOI-10.1145%2F3785462.3815816-blue)](https://doi.org/10.1145/3785462.3815816)
[![Zenodo DOI](https://img.shields.io/badge/Zenodo-PENDING-yellow)](https://doi.org/10.5281/zenodo.PENDING)
[![Reproducibility CI](https://github.com/espirado/VGAC-A-Full-Stack-GPU-Cluster-Observability-Platform-with-Calibration-Gated-Predictive-Intelligence/actions/workflows/reproducibility.yml/badge.svg)](https://github.com/espirado/VGAC-A-Full-Stack-GPU-Cluster-Observability-Platform-with-Calibration-Gated-Predictive-Intelligence/actions/workflows/reproducibility.yml)
[![License: MIT (code)](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![License: CC-BY-4.0 (paper)](https://img.shields.io/badge/Paper-CC--BY%204.0-orange.svg)](https://creativecommons.org/licenses/by/4.0/)
[![PEARC '26](https://img.shields.io/badge/Venue-PEARC%20'26-red)](https://pearc.acm.org/pearc26/)
[![Live demo](https://img.shields.io/badge/Live%20demo-Calibration%20%26%20Reliability%20Observatory-purple)](https://espiradev.org/blog/llm-calibration-simulation.html)

Reproducible artifact accompanying the paper:

> **Espira, A. and Kumar, S. (2026).** *From Calibrated Probabilities to Scheduling Decisions: Decision Rules and Calibration Prerequisites for GPU Queue Policy.* In **Practice and Experience in Advanced Research Computing 2026 (PEARC '26)**, July 26–30, Minneapolis, MN, USA. ACM. <https://doi.org/10.1145/3785462.3815816> (PEARC '26 short paper, 4 pages.)

(The paper describes **VGAC** — *A Full-Stack GPU Cluster Observability Platform with Calibration-Gated Predictive Intelligence* — which is the system this repository implements.)

## What VGAC is

VGAC is a **full-stack observability platform** for GPU clusters whose distinguishing feature is **calibration-gated predictive intelligence**. The platform observes cluster state at submit time, monitors the calibration of its own predictive models *as a first-class observability signal*, and lets that observed calibration determine which predictive interventions the platform is permitted to take. Predictive features are gated by an observability check on themselves.

Concretely, the platform is three layers tied together by a single invariant:

- **Capture layer** — submit-time observability instrumentation (5 s polling, phase-transition detection) that records cluster state at the moment of submission. Without this, a downstream feature like `pending_at_submit` is a lagging indicator (the paper's Pearson `r = -0.27 → +0.44` correction in §4).
- **Prediction layer** — calibration-aware classifier with isotonic post-hoc calibration and an online rolling **Expected Calibration Error / Brier / MCE monitor** (treating calibration drift as a telemetry signal).
- **Decision layer** — a graduated intervention framework whose four tiers (Annotate, Warn, Suggest, Gate) carry **explicit calibration prerequisites**. A model only earns a tier when its currently-observed ECE is at or below that tier's prerequisite; if calibration drifts, the higher-stakes tiers automatically retire.

## TL;DR

VGAC enforces the rule **"apply action $a$ if $\hat{p} \ge \varepsilon_a$ AND $\text{ECE} \le R_a$"** — a model must *earn* each tier of intervention through demonstrated calibration quality, observed continuously.

| Tier | Action | $\varepsilon_a$ | ECE prereq. |
| ---- | ------ | --------------- | ----------- |
| 1 — Annotate | Label the job | 0.30 | $\le 0.10$ |
| 2 — Warn     | Notify the user | 0.50 | $\le 0.07$ |
| 3 — Suggest  | Recommend reschedule | 0.70 | $\le 0.05$ |
| 4 — Gate     | Require confirmation | 0.90 | $\le 0.03$ |

On a real EKS GPU cluster ($n=650$ jobs), a 2-feature *floor* model honestly qualifies only for Annotate (AUROC 0.756, ECE 0.077), while a 17+ feature *production* VGAC model qualifies for all four tiers (AUROC 0.969, ECE 0.005). The 12.7× ECE improvement makes the upgrade path from minimal to full instrumentation explicit and measurable.

## Repository layout

```
.
├── README.md                  this file
├── LICENSE                    MIT (code) — paper licensed CC-BY 4.0
├── CITATION.cff               machine-readable citation metadata
├── .zenodo.json               Zenodo deposit metadata
├── environment.yml            conda environment (Python 3.11)
├── requirements.txt           pip equivalents (pinned)
│
├── tex/                       camera-ready LaTeX source + PDF
│   ├── vgac_pearc.tex         ACM acmart sigconf source
│   ├── refs.bib               BibTeX (ACM-Reference-Format)
│   ├── vgac_pearc.bbl         compiled bibliography (TAPS-required)
│   └── vgac_pearc.pdf         compiled camera-ready (4 pp)
│
├── vgac_pearc_taps_submission.zip  pre-zipped TAPS bundle (.tex + .bbl + .bib + figures/)
│
├── code/                      paper-curated package (small, top-down readable)
│   ├── calibration/isotonic.py        post-hoc isotonic calibrator
│   ├── evaluation/                    calibration metrics + bootstrap + seeding
│   │   ├── seeds.py                   seed_everything(42)
│   │   ├── calibration.py             ece, mce, reliability_curve, brier_decomposition
│   │   └── bootstrap.py               1000-iter percentile CI for AUROC / AUPRC / Brier / ECE
│   ├── features/universal_schema.py
│   ├── harness/                       drift sensors
│   │   ├── psi.py                     Population Stability Index (Siddiqi)
│   │   └── temporal_ece.py            sliding-window ECE
│   ├── integration/                   submit-time observability + adapters
│   │   ├── submit_capture.py          K8s admission + Slurm job_submit hooks
│   │   └── policy_translate.py
│   ├── ops/recalibrator.py            sliding-window recalibration trigger
│   ├── policy/                        graduated-intervention generator + gpu_ext bridge
│   │   ├── generator.py
│   │   ├── gpu_ext_bridge.py
│   │   └── inference_router.py
│   ├── run_experiments.py
│   ├── generate_figures.py
│   └── routing_simulation.py
│
├── src/                       production package (the upstream codebase that
│                              `code/` was curated from). Imported as `src.*` so
│                              the existing notebooks and CLIs run unchanged.
│   ├── sli/compute.py                 4-SLI computation (ECE, Brier 3-way,
│   │                                  tail calibration, PSI) + SLO-compliance check
│   ├── tier/qualify.py                production tier-qualification + tier-matrix builder
│   ├── eval/metrics.py                EvalResult / multiclass_brier
│   ├── feature/extract_k8s_submit_features.py   K8s submit-time feature extractor
│   ├── features/{schema,text}.py      unified feature schema + TF-IDF vectoriser
│   ├── data/{unified_loader,alibaba_v2020,google_2019,
│   │        gpu_v2025_dlrm,logs_loader}.py   multi-source loaders
│   ├── train/                         CV / baseline / DLRM / quantile training
│   ├── transfer/matrix.py             cross-cluster transfer matrix
│   ├── models/{baselines,lgbm}.py     baseline LR + LightGBM pipelines
│   └── visualization/, analysis/, utils/
│
├── notebooks/
│   ├── reproducibility.ipynb              one-button reproduction of every paper claim
│   └── queue_lifecycle_exploration.ipynb  exploratory analysis of submit -> pending ->
│                                          running -> succeeded transitions on the
│                                          Slurm training dataset (906 KB, executed,
│                                          imported from the upstream paper-2 work)
│
├── data/samples/              redistribution-safe samples
│   ├── eks_dec_sample.csv                 EKS-Dec (45 KB)
│   ├── slurm_sample.csv                   Slurm shipped sample (99 KB)
│   ├── slurm_training_dataset.csv         FULL Slurm training dataset (141 KB,
│   │                                      555 jobs x 38 features, with DCGM telemetry)
│   ├── slurm_training_dataset_summary.json
│   ├── alibaba_sample.csv                 Alibaba GPU-trace sample (77 KB)
│   ├── borg_sample.csv                    Google Borg 2019 sample (358 KB)
│   ├── cross_domain_analysis.json
│   └── drift_metrics.json
│
├── artifacts/                 benchmark CSV/JSON consumed by the paper
│   ├── all_5_models_results.csv  bootstrap_confidence_intervals.csv
│   ├── cross_domain_analysis.json  paper2_paper3_experiments.json
│   ├── vgac_floor_vs_ceiling.csv  vgac_submit_time_correlation.json
│   ├── legacy_paper4/                     landscape, SLI/SLO, heterogeneous, deep analysis
│   └── legacy_paper2/                     paper-2 model evaluation + notebook results
│
├── figures/                   PNG/PDF figures referenced in the paper
│
├── docs/
│   ├── ARCHITECTURE.md        the 3-layer platform (capture / predict / decide)
│   ├── METHODOLOGY.md         claim ↔ code ↔ artifact map
│   ├── REPRODUCIBILITY.md     run guide for the notebook + CLI
│   └── DATA_DICTIONARY.md     schema for every CSV / JSON
│
└── submission/
    ├── TAPS_UPLOAD.md         ACM TAPS upload notes
    └── CAMERA_READY_CHANGES.md  diff vs reviewer copy
```

## Live demo

The deployed VGAC platform is publicly browsable:

> **[demo.vgac.cloud](https://demo.vgac.cloud/)** — researcher dashboard with the live queue/pressure snapshot, "Can my job start now?" wait-time estimator across 1/2/4/8 GPU sizes, and the cluster/pattern/inference views that surface the same calibration-gated predictions described in the paper.

An interactive browser-side companion that runs the isotonic-calibration / drift-detection / SLO pipeline against simulated workloads is at [Calibration & Reliability Observatory](https://espiradev.org/blog/llm-calibration-simulation.html) — watch ECE diverge from the diagonal, click *Inject Drift*, and see the PSI / SLO burn-rate alerts and recalibration cue fire end-to-end.

## Quickstart (5 minutes)

```bash
git clone https://github.com/espirado/VGAC-A-Full-Stack-GPU-Cluster-Observability-Platform-with-Calibration-Gated-Predictive-Intelligence.git VGAC
cd VGAC
conda env create -f environment.yml
conda activate vgac
jupyter lab notebooks/reproducibility.ipynb   # Run All
```

The notebook regenerates every figure in `figures/` and prints every numerical claim from the paper. End-to-end runtime is ~45–90 s on a 2024-era laptop (no GPU required).

For pip users, `pip install -r requirements.txt` instead of the conda commands works equivalently.

## Reproducing every claim in the paper

The mapping from each empirical claim to the code path and artifact that backs it lives in `docs/METHODOLOGY.md`. Highlights:

- **§4 Submit-time capture.** $r=-0.27 \to r=+0.44$ — reproduced in notebook step 1.
- **§5.2 Floor model.** AUROC 0.756, ECE 0.077, qualifies Tier 1 only — notebook step 3.
- **§5.3 Ceiling model.** AUROC 0.969, ECE 0.005, qualifies all four tiers — notebook step 4.
- **§3.3 Graceful degradation.** Tier retirement under simulated ECE drift — notebook step 5.

The full benchmark table (5 model families × 4 environments × {floor, ceiling}) lives in `artifacts/all_5_models_results.csv`, with bootstrap 95 % CIs in `artifacts/bootstrap_confidence_intervals.csv`.

## Building the paper PDF

The compiled camera-ready is at `tex/vgac_pearc.pdf`. To rebuild from source:

```bash
cd tex
docker run --rm -v "$PWD":/data -w /data texlive/texlive:latest \
  bash -lc 'pdflatex vgac_pearc.tex && pdflatex vgac_pearc.tex'
```

Or with a local TeX Live install: `cd tex && pdflatex vgac_pearc.tex && bibtex vgac_pearc && pdflatex vgac_pearc.tex && pdflatex vgac_pearc.tex`. The bibliography is in `tex/refs.bib` (ACM-Reference-Format).

## TAPS submission

The TAPS upload bundle is `vgac_pearc_taps_submission.zip` (auto-generated, contains `vgac_pearc.tex`, `vgac_pearc.bbl`, `refs.bib`, and `figures/`). The upload procedure and the TAPS *Check Paper Details* checklist are in `submission/TAPS_UPLOAD.md`.

## How to cite

Please cite both the ACM conference paper and the software artifact:

```bibtex
@inproceedings{espira2026vgac,
  title     = {From Calibrated Probabilities to Scheduling Decisions:
               Decision Rules and Calibration Prerequisites for GPU Queue Policy},
  author    = {Espira, Andrew and Kumar, Sharath},
  booktitle = {Practice and Experience in Advanced Research Computing 2026 (PEARC '26)},
  year      = {2026},
  publisher = {ACM},
  doi       = {10.1145/3785462.3815816},
  isbn      = {979-8-4007-2377-3/2026/07},
  address   = {Minneapolis, MN, USA},
  month     = jul,
}

@software{espira2026vgac_artifact,
  title   = {VGAC: A Full-Stack GPU Cluster Observability Platform
             (software artifact, v1.0.0)},
  author  = {Espira, Andrew and Kumar, Sharath},
  year    = {2026},
  version = {1.0.0},
  doi     = {10.5281/zenodo.PENDING},
  url     = {https://github.com/espirado/VGAC-A-Full-Stack-GPU-Cluster-Observability-Platform-with-Calibration-Gated-Predictive-Intelligence},
}
```

`CITATION.cff` is the machine-readable equivalent (GitHub renders the *Cite this repository* widget from it automatically).

## Related work

- **Cross-cluster benchmark (published preprint).** Espira, Dhole, Kumar, *Calibration under extreme imbalance: A multi-cluster benchmark for operational queue-delay prediction*, TechRxiv 2026 — DOI [10.36227/techrxiv.177041829.96464119](https://doi.org/10.36227/techrxiv.177041829.96464119).
- **Live VGAC platform.** Deployed researcher dashboard: <https://demo.vgac.cloud/>.
- **Live calibration & reliability observatory.** Interactive showcase of the same isotonic calibration / drift / SLO pipeline VGAC formalises: <https://espiradev.org/blog/llm-calibration-simulation.html>.

## Use of AI assistance

Drafting of the paper text was assisted by a large language model (Anthropic Claude). All experiments, results, methodological choices, and interpretations were authored and verified by the human author. The model did not generate experimental results. This disclosure follows ACM's policy on the use of generative AI in scholarly publications.

## License

- **Code:** MIT, see `LICENSE`.
- **Paper text and figures (`tex/`):** CC-BY 4.0, per the ACM-supplied bibstrip.
- **Sample data:** licensed under the upstream trace licences (Alibaba, Google Borg) where applicable; EKS-Dec and Slurm samples are released under MIT alongside the code.

## Contact

- **Andrew Espira** — `aespira@saintpeters.edu` — ORCID: [0009-0002-9196-8094](https://orcid.org/0009-0002-9196-8094)
- **Sharath Kumar** — `skumar@saintpeters.edu`

Department of Data Science, Saint Peter's University, Jersey City, NJ, USA.

For bug reports and reproducibility issues, please open a GitHub issue.
