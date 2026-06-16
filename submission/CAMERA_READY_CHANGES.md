# Camera-Ready Changes

Differences between the accepted reviewer copy (`paper_125.pdf`, originally
submitted in IEEE format) and the published camera-ready (`tex/vgac_pearc.pdf`,
ACM `acmart`):

## 1. Format conversion

- Re-typeset from IEEE conference template to ACM `acmart` (sigconf).
- Inserted the ACM-supplied preamble block: `\copyrightyear`, `\acmYear`,
  `\setcopyright{cc}`, `\setcctype{by}`, `\acmConference`, `\acmBooktitle`,
  `\acmDOI`, `\acmISBN`.
- Replaced IEEE-style author block with ACM `\author` + `\affiliation`.
- Inserted CCS concepts and ACM keywords block.
- Switched bibliography from numerical IEEE style to ACM-Reference-Format.

## 2. Substantive edits

- Updated abstract to mirror the four-tier-and-prerequisites framing
  exactly, with concrete numbers (`r = -0.27 → +0.44`, ECE 0.077 → 0.005,
  AUROC 0.756 → 0.969).
- Promoted the empirical floor / ceiling tables (`tab:floor`, `tab:ceiling`)
  to single-column ACM-style tables.
- Hardened the related-work section to cite `decima`, `borg`, `firmament`,
  `elkan2001`, `guo2017calibration`, `platt1999` in their canonical ACM /
  conference forms.
- Added an explicit `\acks` block thanking collaborators on the broader GPU
  reliability research programme.

## 3. Required disclosures

- Added the ACM-mandated **Use of AI Assistance** section.
- Removed all author-identifying remarks from the body (anonymous review
  was already lifted but the camera-ready re-checks for residual hints).

## 4. Typographic fixes

- Resolved all `Overfull \hbox` warnings without using forced hyphenation;
  used `\emergencystretch=2em` and rewordings of dense `\texttt{}` runs.
- Split long pseudocode lines into multiple `\\` lines inside a `quote` block
  with `\ttfamily\raggedright`.
- Tightened the integration section to ensure the `prob = predict(features);
  tier = get_qualified_tier(prob, current_ece); annotate(job, prob, tier).`
  pseudocode renders inside a single column.

## 5. Verification

- Compiled with `texlive/texlive:latest` Docker image (TeX Live 2025).
- Bibliography fully resolved (no `[??]`).
- 0 overfull boxes at the time of submission.
- Page count: well within the PEARC '26 short-paper budget.
