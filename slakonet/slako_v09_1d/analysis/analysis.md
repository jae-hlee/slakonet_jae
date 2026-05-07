# SlakoNet on Alexandria PBE 1D — Results Analysis

## Overview

SlakoNet (tight-binding neural network) was run on the Alexandria **PBE 1D**
dataset of quasi-one-dimensional crystal candidates. Each entry is a relaxed
PBE structure with reported indirect and direct band gaps. The task is to
predict the **band gap** from atomic geometry alone, benchmarked against the
PBE indirect gap (`band_gap_ind`). Full **DOS** curves are saved as a
secondary output.

- Dataset total: 13,295 entries
- Filtered out up-front (element outside `ALLOWED_SYMBOLS`, Z > 65): 3,755
- Successfully evaluated: **8,636** (904 of the 9,540 attempted are missing —
  **100% of those 904 contain an f-block lanthanide** Ce–Tb, identical to the
  v10 2D pattern; see "The 904 missing entries are 100% f-block lanthanides"
  below for the diagnosis)
- Reference: Alexandria PBE indirect gap (`band_gap_ind`); direct gap
  (`band_gap_dir`) tracked alongside

## Summary statistics

| Quantity                       |    N | mean   | median | min    | max    |
|--------------------------------|-----:|-------:|-------:|-------:|-------:|
| SlakoNet band gap (eV)         | 8,636| +1.869 | +0.308 | +0.000 | +21.0  |
| PBE indirect gap (eV)          | 8,636| +1.088 | +0.426 | +0.000 | +9.55  |
| PBE direct gap (eV)            | 8,636| +1.198 | +0.598 | +0.000 | +9.55  |

- **PBE "metals"** (`band_gap_ind == 0`): 2,940 / 8,636 = **34.0 %**
- **SlakoNet "metals"** (gap < 0.1 eV): 3,434 / 8,636 = **39.8 %**

## SlakoNet vs PBE parity

Indirect gap (SlakoNet vs `band_gap_ind`):

| subset                    |    N |  MAE  | RMSE  |   R²    |
|---------------------------|-----:|------:|------:|--------:|
| All                       | 8,636| 0.989 | 1.701 | −0.246  |
| PBE metals (gap = 0)      | 2,940| 0.160 | 0.481 | —       |
| PBE non-metals (gap > 0)  | 5,696| **1.416** | 2.065 | −0.646 |

Direct gap (SlakoNet vs `band_gap_dir`) is within **0.02 eV** of indirect on
every subset — expected, since SlakoNet returns one scalar.

The cross-dataset aggregator reports Pearson **r = 0.88** for the full set,
so the *trend* is captured well even though the regression R² goes negative
once the population mean is the comparison baseline. See `plots/parity_all.png` and
`plots/parity_nonmetals.png`.

## Metal / non-metal classification

SK threshold = 0.1 eV.

|              | PBE metal | PBE non-metal |
|--------------|----------:|--------------:|
| SK metal     |   2,123   |     1,311     |
| SK non-metal |     817   |     4,385     |

- Accuracy: **0.754**
- False negatives (PBE non-metal predicted metallic by SK): **1,311** with PBE
  gap median 0.26 eV (mean 0.53, max 4.36 eV)
- False positives (PBE metal predicted non-metal by SK): 817

See `plots/confusion_matrix.png`.

## Key observations

1. **Same failure mode as the 3D run.** The 1,311 false-negative non-metals
   look like the v03 Alexandria 3D pattern: PBE has a small-to-moderate gap
   that SlakoNet collapses to ~0. Likely culprits are the missing spin
   polarization (open-shell transition-metal cations) and weak SK parameters
   for ionic / fluoride chemistries — see
   `../../slako_v03_alex/analysis/analysis.md` for the detailed write-up.
2. **Wider non-metal spread than 3D.** Non-metal MAE is 1.42 eV (3D was
   1.78 eV in the paired ALIGNN comparison). RMSE 2.07 eV is dominated by a
   long tail of SK-overestimated insulators reaching 21 eV.
3. **High Pearson r despite negative R².** r = 0.88 says SK ranks structures
   correctly; the negative R² says the absolute scale is off and a constant
   "predict the population mean" baseline beats the regressor on
   sum-of-squares. Useful signal for screening, not for absolute gaps.
4. **No obvious correlation with formation energy.** `plots/error_vs_eform.png`
   shows the SK − PBE residual is roughly flat in `e_form`, so the error is
   not concentrated on stable / unstable structures.
5. **DOS quality.** Averaged DOS (`plots/dos_average.png`) matches the expected
   quasi-1D shape — sharp peaks at the band edges either side of E_F. Per-
   structure examples at the 10/50/90th SK-gap percentiles
   (`plots/dos_examples.png`) are qualitatively sensible.

## Interpretation

SlakoNet on the 1D Alexandria set behaves the way it does on 3D Alexandria:
high rank correlation, one-sided overprediction on insulators, and a
recurring metallic-collapse failure mode driven by the absence of spin
polarization and weak SK parameters for a handful of chemistries. The 1D
geometry itself is not the problem — the same model on 3D Alexandria has the
same residual structure. For ranking-style screening (top-k by gap) the
output is usable; for absolute gaps a calibration step or a chemistry-aware
fine-tune would be required.

## The 904 missing entries are 100% f-block lanthanides

This section overturns the "~10 % mystery dropout — rerun on one GPU to
diagnose" narrative that originally lived here. Cross-checking the 904
missing ids against the source zip:

- **904 / 904 (100%) contain at least one of Ce, Pr, Nd, Pm, Sm, Eu, Gd, Tb.**
- **0 / 8,636** completed entries contain any of those elements.

Breakdown of which lanthanide dominates the missing set (1D-specific — Pr
and Pm never appear in Alexandria 1D at all, so this list is a subset of
the v10 2D list):

| Lanthanide | Z | in missing | in completed |
|---|---:|---:|---:|
| Sm | 62 | 19.6% | 0.0% |
| Nd | 60 | 19.1% | 0.0% |
| Tb | 65 | 19.1% | 0.0% |
| Gd | 64 | 17.4% | 0.0% |
| Eu | 63 | 16.9% | 0.0% |
| Ce | 58 |  7.9% | 0.0% |

This is the same deterministic 4f-shell wall documented in
`../../slako_v10_2d/analysis/analysis.md`: `generate_shell_dict_upto_Z65()`
nominally produces a shell dict for these elements, so the `ALLOWED_SYMBOLS`
filter lets them through, but inference fails inside
`model.compute_multi_element_properties(...)` and `gpu_worker` silently
swallows the exception. A one-GPU rerun will **not** surface new structures
— the failure is not a race condition or a per-node fluke, it is a model
capability limit. The effective usable ceiling is `Z ≤ 57` (through La),
not the `Z ≤ 65` the filter advertises.

Implication: reruns / sister projects should drop Ce–Tb from
`ALLOWED_SYMBOLS` to stop burning GPU hours on guaranteed silent failures.

## Artifacts in this directory

- `plots/parity_all.png` / `plots/parity_nonmetals.png` — density scatter, SK vs PBE
- `plots/residuals_all.png` / `plots/residuals_nonmetals.png` — SK − PBE residual densities
- `plots/gap_distribution.png` — SK and PBE gap histograms overlaid
- `plots/confusion_matrix.png` — metal / non-metal classification at 0.1 eV threshold
- `plots/error_vs_eform.png` — residual vs formation energy
- `plots/dos_average.png` — mean Fermi-aligned DOS
- `plots/dos_examples.png` — representative DOS at 10/50/90th SK-gap percentiles
- `csv/summary.csv` — per-structure scalars: id, formula, nsites, spg, e_form,
  PBE indirect / direct gaps, PBE DOS(E_F), SlakoNet gap
- `csv/stats.txt` — raw headline numbers used to build this write-up
