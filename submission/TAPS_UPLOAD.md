# PEARC '26 — TAPS Upload Notes

This file documents how the camera-ready TAPS bundle was prepared and uploaded
for the paper:

> **VGAC: A Full-Stack GPU Cluster Observability Platform with Calibration-Gated
> Predictive Intelligence**, Andrew Espira, Saint Peter's University.
> DOI: [10.1145/3785462.3815816](https://doi.org/10.1145/3785462.3815816).
> ISBN: 979-8-4007-2377-3/2026/07.

## 1. ACM rights / preamble

Per the acceptance email from `tapsadmin@aptaracorp.awsapps.com`, the
preamble in `tex/vgac_pearc.tex` includes the following — verbatim from the
ACM-supplied snippet:

```latex
\copyrightyear{2026}
\acmYear{2026}
\setcopyright{cc}
\setcctype{by}
\acmConference[PEARC '26]{Practice and Experience in Advanced Research Computing}{July 26--30, 2026}{Minneapolis, MN, USA}
\acmBooktitle{Practice and Experience in Advanced Research Computing (PEARC '26), July 26--30, 2026, Minneapolis, MN, USA}
\acmDOI{10.1145/3785462.3815816}
\acmISBN{979-8-4007-2377-3/2026/07}
```

The bibstrip therefore renders the CC-BY 4.0 licence, the conference
metadata, and the DOI on page 1, as required by the ACM publishing system.

## 2. TAPS-accepted packages

`tex/vgac_pearc.tex` uses only TAPS-allowed packages:

- `acmart` (the ACM consolidated template, document class).
- `booktabs`, `amsmath`, `xcolor`, `microtype` (all on the TAPS allow list).

No package outside the allow list is loaded. `\emergencystretch=2em` is the
only typographic override — it is a built-in primitive, not a package.

## 3. Bundle contents

The TAPS upload is a single ZIP. It contains:

```
vgac_pearc_taps_submission.zip
|-- vgac_pearc.tex   # the source
|-- vgac_pearc.pdf   # the camera-ready PDF (for previewer)
```

If reviewers ask for figures separately, add the `figures/` directory to
the bundle. The current paper renders only inline tables; no external image
assets are referenced.

## 4. Upload procedure

1. Log in to the TAPS dashboard via the link in the acceptance email
   (`http://camps.aptaracorp.com/AuthorDashboard/dashboard.html?key=0&val=...`).
2. Click **CHECK PAPER DETAILS** and verify:
   - Title: `VGAC: A Full-Stack GPU Cluster Observability Platform with
     Calibration-Gated Predictive Intelligence`.
   - Author: `Andrew Espira`.
   - Affiliation: `Saint Peter's University, Department of Data Science`.
   - DOI: `10.1145/3785462.3815816`.
3. Upload `vgac_pearc_taps_submission.zip`.
4. Wait for TAPS to compile. If compilation fails, fix the issue locally
   (rebuild via `pdflatex` once with the Docker image
   `texlive/texlive:latest`) and resubmit.
5. Approve the rendered PDF when TAPS notifies completion.
6. (If applicable) upload supplementary materials via the second link in
   the acceptance email.

## 5. Sanity checks done locally before upload

- [x] PDF page 1 shows the CC-BY bibstrip with the correct DOI.
- [x] All references resolve (no `[??]` markers).
- [x] No `Overfull \hbox` warnings.
- [x] Author email, affiliation, and country render correctly.
- [x] AI-usage disclosure is present (Section "Use of AI Assistance").
- [x] Acknowledgments paragraph is present.
- [x] CCS concept tree compiles without warnings.

## 6. Post-acceptance checklist

- [ ] TAPS upload submitted.
- [ ] Galley proof returned to ACM.
- [ ] DOI activated on the ACM Digital Library.
- [ ] Camera-ready PDF released on Zenodo (auto-deposit via GitHub release).
- [ ] CITATION.cff updated with the Zenodo DOI of v1.0.0.
- [ ] README badges updated with both the ACM DOI and the Zenodo DOI.
