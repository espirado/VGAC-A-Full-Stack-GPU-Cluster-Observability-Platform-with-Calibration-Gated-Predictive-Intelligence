# VGAC — A Full-Stack GPU Cluster Observability Platform with Calibration-Gated Predictive Intelligence

[![DOI](https://img.shields.io/badge/DOI-10.1145%2F3785462.3815816-blue)](https://doi.org/10.1145/3785462.3815816)
[![Zenodo DOI](https://img.shields.io/badge/Zenodo-PENDING-yellow)](https://doi.org/10.5281/zenodo.PENDING)
[![License: MIT (code)](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![License: CC-BY-4.0 (paper)](https://img.shields.io/badge/Paper-CC--BY%204.0-orange.svg)](https://creativecommons.org/licenses/by/4.0/)
[![PEARC '26](https://img.shields.io/badge/Venue-PEARC%20'26-red)](https://pearc.acm.org/pearc26/)

Reproducible artifact accompanying the paper:

> **Espira, A. (2026).** *VGAC: A Full-Stack GPU Cluster Observability Platform with Calibration-Gated Predictive Intelligence.* In **Practice and Experience in Advanced Research Computing 2026 (PEARC '26)**, July 26–30, Minneapolis, MN, USA. ACM. <https://doi.org/10.1145/3785462.3815816> (PEARC '26 short paper, 4 pages.)

## What VGAC is

VGAC is a **full-stack observability platform** for GPU clusters whose distinguishing feature is **calibration-gated predictive intelligence**. The platform observes cluster state at submit time, monitors the calibration of its own predictive models *as a first-class observability signal*, and lets that observed calibration determine which predictive interventions the platform is permitted to take. Predictive features are gated by an observability check on themselves.

Concretely, the platform is three layers tied together by a single invariant:

- **Capture layer** — submit-time observability instrumentation (5 s polling, phase-transition detection) that records cluster state at the moment of submission. Without this, a downstream feature like `pending_at_submit` is a lagging indicator (the paper's Pearson `r = -0.27 → +0.44` correction in §4).
- **Prediction layer** — calibration-aware classifier with isotonic post-hoc calibration and an online rolling **Expected Calibration Error / Brier / MCE monitor** (treating calibration drift as a telemetry signal).
- **Decision layer** — a graduated intervention framework whose four tiers (Annotate, Warn, Suggest, Gate) carry **explicit calibration prerequisites**. A model only earns a tier when its currently-observed ECE is at or below that tier's prerequisite; if calibration drifts, the higher-stakes tiers automatically retire.

> **Companion artifact.** ISS26 Paper 2 (*Reliability-First Queue Risk for GPU Clusters*) defines the SLI/SLO framework that VGAC's calibration monitor builds on. It lives in a separate repository (`espirado/Reliability-First-Queue-Risk`) and uses the IEEE conference template; see [Related work](#related-work).

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
│   └── vgac_pearc.pdf         compiled camera-ready
│
├── taps_submission/           TAPS upload bundle
├── vgac_pearc_taps_submission.zip  pre-zipped TAPS bundle
│
├── code/
│   ├── policy/                tier-qualification + gpu_ext bridge + LLM router
│   ├── integration/           submit-time capture, admission webhook glue
│   ├── evaluation/            metric helpers, bootstrap CIs
│   ├── features/              universal feature schema
│   ├── harness/               PSI, temporal-ECE drift detection
│   ├── ops/                   ops/runbook helpers
│   ├── calibration/           isotonic calibrator wrappers
│   ├── run_experiments.py     CLI: train + calibrate + qualify
│   ├── generate_figures.py    CLI: regenerate every figure in figures/
│   └── routing_simulation.py  routing-level simulation harness
│
├── notebooks/
│   └── reproducibility.ipynb  one-button reproduction of every paper claim
│
├── data/samples/              anonymised samples (EKS, Slurm, Alibaba, Borg)
├── artifacts/                 benchmark CSV/JSON consumed by the paper
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

## Quickstart (5 minutes)

```bash
git clone https://github.com/espirado/VGAC
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

Or with a local TeX Live install: `cd tex && pdflatex vgac_pearc.tex` twice. The bibliography is embedded; no `.bib`/`.bbl` is needed.

## TAPS submission

The TAPS upload bundle is `vgac_pearc_taps_submission.zip` and `taps_submission/`. The upload procedure and the TAPS *Check Paper Details* checklist are in `submission/TAPS_UPLOAD.md`.

## How to cite

Please cite both the ACM conference paper and the software artifact:

```bibtex
@inproceedings{espira2026vgac,
  title     = {VGAC: A Full-Stack GPU Cluster Observability Platform with
               Calibration-Gated Predictive Intelligence},
  author    = {Espira, Andrew},
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
  author  = {Espira, Andrew},
  year    = {2026},
  version = {1.0.0},
  doi     = {10.5281/zenodo.PENDING},
  url     = {https://github.com/espirado/VGAC},
}
```

`CITATION.cff` is the machine-readable equivalent (GitHub renders the *Cite this repository* widget from it automatically).

## Related work

- **Companion paper (SLI/SLO foundation).** Espira et al., *Reliability-First Queue Risk for GPU Clusters: Calibration, SLOs, and Reproducible Operational Integration*, ISS26 — repo: <https://github.com/espirado/Reliability-First-Queue-Risk>.
- **Cross-cluster benchmark.** Espira, Dhole, Kumar, *Calibration under extreme imbalance: A multi-cluster benchmark for operational queue-delay prediction*, TechRxiv 2026 (DOI: 10.36227/techrxiv.177041829.96464119).
- **Sentinel.** Companion eBPF / MCP-server prototype — repo: <https://github.com/espirado/SENTINEL>.

## Use of AI assistance

Drafting of the paper text was assisted by a large language model (Anthropic Claude). All experiments, results, methodological choices, and interpretations were authored and verified by the human author. The model did not generate experimental results. This disclosure follows ACM's policy on the use of generative AI in scholarly publications.

## License

- **Code:** MIT, see `LICENSE`.
- **Paper text and figures (`tex/`):** CC-BY 4.0, per the ACM-supplied bibstrip.
- **Sample data:** licensed under the upstream trace licences (Alibaba, Google Borg) where applicable; EKS-Dec and Slurm samples are released under MIT alongside the code.

## Contact

Andrew Espira — `aespira@saintpeters.edu` — Department of Data Science, Saint Peter's University, Jersey City, NJ, USA. ORCID: [0009-0002-9196-8094](https://orcid.org/0009-0002-9196-8094).

For bug reports and reproducibility issues, please open a GitHub issue.
