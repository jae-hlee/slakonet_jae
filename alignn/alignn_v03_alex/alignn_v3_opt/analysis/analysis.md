# optb88vdw-ALIGNN (v03) vs SlakoNet — Alexandria PBE 3D

## What changed from v00

v00 evaluated the PBE-trained `mp_gappbe_alignn` ALIGNN checkpoint. v03 swaps
in the JARVIS-trained `jv_optb88vdw_bandgap_alignn` checkpoint on the exact
same filtered Alexandria hull set (48,764 structures, Z ≤ 65, e_above_hull = 0).
All other code and filters are identical. The SlakoNet sk_bandgap values
here are copied from v00 (same inputs, same upstream SlakoNet run), so
head-to-head numbers are directly comparable to the v00 analysis.

**Important caveat:** the reference (Alexandria `band_gap_ind`) is a PBE gap,
but the v03 model was trained on OptB88vdW gaps. OptB88vdW is a vdW-corrected
GGA — for most non-vdW solids it yields gaps comparable to PBE, but a small
systematic shift is expected. So the v03 errors reflect both (model
quality) and (functional mismatch), and must be interpreted accordingly.

Paired comparison uses the N = 31,211 structures for which SlakoNet, v00
PBE-ALIGNN, and v03 optb-ALIGNN all produced a prediction (17,553 structures
have optb predictions but no SlakoNet match — 99.9 % of those are
lanthanide-containing cells that SlakoNet silently rejects despite passing
the `ALLOWED_SYMBOLS` filter; see `../../pbe_mbj_opt_analysis/analysis.md`).

## Headline numbers — regression vs PBE indirect gap

**All structures (N = 31,211):**

| Model                         |  MAE   |  RMSE  |    R²   |    ME   |
|-------------------------------|-------:|-------:|--------:|--------:|
| SlakoNet                      | 0.930  | 1.649  | −0.008  | +0.329  |
| ALIGNN (mp_gappbe, v00)       | **0.193** | **0.463** | **0.920** | +0.017 |
| ALIGNN (optb88vdw, v03)       | 0.354  | 0.746  |  0.794  | −0.147  |

**Non-metals (PBE gap > 0, N = 16,092):**

| Model                         |  MAE   |  RMSE  |    R²   |    ME   |
|-------------------------------|-------:|-------:|--------:|--------:|
| SlakoNet                      | 1.781  | 2.291  | −1.069  | +0.615  |
| ALIGNN (mp_gappbe, v00)       | **0.274** | **0.490** | **0.906** | −0.067 |
| ALIGNN (optb88vdw, v03)       | 0.602  | 0.949  |  0.645  | −0.370  |

**Metals (PBE gap = 0, N = 15,119):**

| Model                         |  MAE   |  RMSE  |    ME   |
|-------------------------------|-------:|-------:|--------:|
| SlakoNet                      | **0.024** | 0.160 | +0.024  |
| ALIGNN (mp_gappbe, v00)       | 0.106  | 0.433  | +0.106  |
| ALIGNN (optb88vdw, v03)       | 0.090  | 0.436  | +0.090  |

## Interpretation

**optb-ALIGNN still crushes SlakoNet but does worse than PBE-ALIGNN, as
expected.** The v03 checkpoint was trained to reproduce a different functional
than the reference, so some of its error is "correct behaviour". Even so:

- Non-metal MAE drops from 1.78 eV (SlakoNet) to **0.60 eV** (optb-ALIGNN) —
  a 3× improvement — while v00's PBE-ALIGNN got 0.27 eV.
- Per-structure, **optb-ALIGNN beats SlakoNet on 81.5%** of the paired set.
- Against the matching-functional baseline, **optb-ALIGNN beats PBE-ALIGNN on
  only 44.8%** — i.e. PBE-ALIGNN wins by ~10 percentage points when the
  ground truth is PBE, which is the correct outcome.
- v03 has a clear **negative mean error on non-metals (−0.37 eV)**, and the
  linear fit gives `optb_gap ≈ 0.94 × PBE − 0.23` (non-metals only). This is
  consistent with the OptB88vdW → PBE mismatch being a downward shift of
  ~0.2–0.4 eV on insulators rather than a random scatter. If the reference
  were swapped to OptB88vdW gaps the MAE should drop meaningfully.
- v03 emits **11,213 negative predictions** (vs 4,541 for v00), clamped to 0.
  This is the main source of the `ME` becoming negative — the model is
  happier predicting small-magnitude negatives on borderline metals.

## Classification — metal vs non-metal (threshold 0.1 eV)

|              | Accuracy |   TN  |   FP  |   FN  |   TP  |
|--------------|---------:|------:|------:|------:|------:|
| SlakoNet     |  0.830   | 14715 |   404 | **4909** | 11183 |
| ALIGNN-pbe   | **0.916** | 13283 |  1836 |   777 | 15315 |
| ALIGNN-optb  |  0.892   | 13709 |  1410 |  1957 | 14135 |

FN = "PBE insulator predicted as metal" — the v00 failure cohort.

- SlakoNet's 4,909 FN (the Mn/Cr/Fe/Co/Ni collapsed-gap population diagnosed
  in v00) drops to **1,957 with v03** and **777 with v00 PBE-ALIGNN**.
  optb-ALIGNN fixes ~60% of the SlakoNet collapses but still misclassifies
  about 2.5× more insulators as metals than PBE-ALIGNN does — again
  consistent with a systematic ~0.2 eV downshift pushing small-gap
  semiconductors across the 0.1 eV decision boundary.
- optb-ALIGNN's FP count (1,410) is *lower* than PBE-ALIGNN's (1,836),
  meaning it hallucinates fewer small gaps on real metals — the metal-subset
  MAE also reflects this (0.090 vs 0.106).

## Head-to-head (paired errors)

- optb-ALIGNN closer to PBE than SlakoNet: **25,439 / 31,211 (81.5%)**
- optb-ALIGNN closer to PBE than PBE-ALIGNN: 13,974 / 31,211 (44.8%)
- PBE-ALIGNN closer to PBE than SlakoNet: 21,157 / 31,211 (67.8%)

Both ALIGNN variants are consistently better than SlakoNet on 2/3+ of the
paired set. PBE-ALIGNN beats optb-ALIGNN on the majority where the
functional-mismatch offset dominates.

## Relationship to v00's SlakoNet failure analysis

v00 identified two SlakoNet failure modes: (1) open-shell 3d/4d transition
metal oxides collapsing to `sk ≈ 0` due to absent spin polarization, and (2)
alkali/alkaline-earth fluorides collapsing due to bad SK parameters. Both of
those populations flow through the same ALIGNN pipeline in v03 and are
largely recovered (FN drops from 4909 → 1957), confirming v00's conclusion
that the SlakoNet collapses are model-specific rather than intrinsic to the
structures. optb-ALIGNN is a viable fallback for those compositions, though
PBE-ALIGNN is slightly better when the downstream target is a PBE gap.

## Figures

All in `plots/`:

- `plots/parity_three_way.png` — PBE vs {SlakoNet, PBE-ALIGNN, optb-ALIGNN}, all structures
- `plots/parity_three_way_nonmetals.png` — same, non-metals only
- `plots/residuals_three_way.png` — residual histograms, side-by-side
- `plots/confusion_three_way.png` — metal/non-metal confusion matrices
- `plots/head_to_head_error.png` — per-structure |error| (optb vs SK, optb vs PBE-ALIGNN)
- `plots/gap_distribution.png` — PBE vs SK vs both ALIGNN densities (non-metals)
- `plots/optb_vs_pbe_shift.png` — OptB88vdW–PBE systematic shift (linear fit)

## Artefacts

- `summary.txt` — raw stdout from the original analysis run (regression
  metrics, confusion counts, fit coefficients). The script that produced
  this is kept local-only.
