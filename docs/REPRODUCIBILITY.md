# VGAC — Reproducibility Guide

This guide walks a reviewer or third party from a fresh clone to a fully
reproduced set of paper results in roughly 10 minutes on a laptop.

## 1. Environment

You can use either conda or pip.

### Option A — conda (recommended)

```bash
conda env create -f environment.yml
conda activate vgac
python -m ipykernel install --user --name vgac --display-name "Python (vgac)"
```

### Option B — pip + venv

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name vgac --display-name "Python (vgac)"
```

Verify:

```bash
python -c "import pandas, numpy, sklearn, xgboost, lightgbm; print('ok')"
```

## 2. Run the reproducibility notebook

```bash
jupyter lab
# or
jupyter notebook notebooks/reproducibility.ipynb
```

Then **Run all cells**. The notebook is structured to mirror the paper:

| Notebook step | Paper section | Output |
| --- | --- | --- |
| Step 0 — Imports and data load | — | Loads `data/samples/eks_dec_sample.csv` |
| Step 1 — Submit-time capture correlation | §4 | Reproduces `r = -0.27 -> r = +0.44` |
| Step 2 — Tier prerequisite table | §3 | Reproduces Table 1 |
| Step 3 — Floor tier qualification | §5.2 | Reproduces Table 3 (AUROC 0.756, ECE 0.077) |
| Step 4 — Ceiling tier qualification | §5.3 | Reproduces Table 4 (AUROC 0.969, ECE 0.005) |
| Step 5 — Graceful degradation simulation | §3.3 | Shows tier retirement under simulated ECE drift |
| Step 6 — Figure regeneration | All | Writes PNGs/PDFs into `figures/` |

End-to-end runtime on a 2024-era laptop with 16 GB RAM is approximately
**45–90 seconds**.

## 3. Run the experiments from the command line

If you prefer the CLI rather than the notebook:

```bash
python code/run_experiments.py --env eks --features floor   --out artifacts/floor_run.json
python code/run_experiments.py --env eks --features ceiling --out artifacts/ceiling_run.json
python code/generate_figures.py --in artifacts/ceiling_run.json --out figures/
```

The output files are deterministic for the seeds shipped in
`code/evaluation/seeds.py`.

## 4. Build the paper PDF

The camera-ready PDF is already in `tex/vgac_pearc.pdf`. To rebuild it:

```bash
cd tex
docker run --rm -v "$PWD":/work -w /work texlive/texlive:latest \
  bash -lc 'pdflatex -interaction=nonstopmode vgac_pearc.tex \
    && bibtex vgac_pearc \
    && pdflatex -interaction=nonstopmode vgac_pearc.tex \
    && pdflatex -interaction=nonstopmode vgac_pearc.tex'
```

Or, with a local TeX Live install:

```bash
cd tex && pdflatex vgac_pearc.tex && bibtex vgac_pearc \
       && pdflatex vgac_pearc.tex && pdflatex vgac_pearc.tex
```

Bibliography lives in `tex/refs.bib` (ACM-Reference-Format) so all four
passes are required for the citations to resolve cleanly.

## 5. TAPS submission package

The TAPS upload bundle is the pre-zipped `vgac_pearc_taps_submission.zip` at
the repo root, containing `vgac_pearc.tex`, `vgac_pearc.bbl`, `refs.bib`, and
`figures/`. The ACM TAPS upload flow is documented in
`submission/TAPS_UPLOAD.md`.

## 6. Verifying claims numerically

Every claim in the paper has a backing artifact. The mapping is in
`docs/METHODOLOGY.md`. After running the notebook, compare the printed
numbers against:

```bash
cat artifacts/all_5_models_results.csv
cat artifacts/bootstrap_confidence_intervals.csv
cat artifacts/cross_domain_analysis.json
cat artifacts/paper2_paper3_experiments.json
```

Any disagreement larger than the third decimal place is a bug.

## 7. Full data access

The repository ships with **anonymised samples** under `data/samples/`. Full
trace data for the EKS-Dec, Slurm + DCGM, Alibaba 2020/2023, and Google Borg
2019 environments can be obtained from the corresponding upstream sources:

- EKS-Dec: collection scripts in `code/integration/`; raw data is private.
- Slurm + DCGM: collected on AWS ParallelCluster — schema in `docs/DATA_DICTIONARY.md`.
- Alibaba: <https://github.com/alibaba/clusterdata>.
- Google Borg: <https://github.com/google/cluster-data>.

The notebook degrades gracefully on the samples; full data is not required
to reproduce the headline tables and figures.

## 8. Hardware

No GPU is required to reproduce paper figures. The published numerics are
trained on aggregated CPU-side cluster state; DCGM telemetry is included as
input features only and does not require a local GPU at run time.

## 9. Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `ImportError: lightgbm` | Wrong Python | Use Python 3.11; `pip install lightgbm==4.3.*` |
| `OverflowError` in calibration | Stale cached model | `rm -rf artifacts/calibration/*.pkl` and rerun |
| Notebook hangs at "Step 4" | Insufficient RAM | Reduce sample size in cell 0 (`SAMPLE_FRAC = 0.25`) |
| Figures look wrong | Old `.png` cache | `rm figures/*.png && rerun cell 6` |

If something else breaks, please open an issue on GitHub with the full
stack trace and your environment versions.
