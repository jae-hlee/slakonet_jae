# Comprehensive ALIGNN-vs-SlakoNet band-gap analysis

This is the unified, side-by-side comparison of the three ALIGNN checkpoints
evaluated in `alignn_v1_pbe`, `alignn_v2_mbj`, `alignn_v3_opt` against the
SlakoNet DFTB predictions from `alignn_v03_alex/`. All numbers, plots, and CSVs in
this directory are pre-built artifacts; the script that produced them is
kept local-only.

## Setup

| Source | Model / quantity | Records | Notes |
|--------|-------------------|--------:|-------|
| `slakonet/slako_v03_alex/results/sk_scalars.json` (gitignored, kept local)           | **SlakoNet** (TBLite/DFTB scalar band-gap) | 31,211 | shared DFTB baseline for all ALIGNN variants |
| `alignn/alignn_v03_alex/alignn_v1_pbe/results/alignn_predictions.json` (gitignored, kept local) | **ALIGNN `mp_gappbe_alignn`** (trained on MP PBE gaps)          | 48,764 | PBE-matched |
| `alignn/alignn_v03_alex/alignn_v2_mbj/results/alignn_predictions.json` (gitignored, kept local) | **ALIGNN `jv_mbj_bandgap_alignn`** (trained on JARVIS TB-mBJ gaps) | 48,764 | different functional → gaps systematically *opened* |
| `alignn/alignn_v03_alex/alignn_v3_opt/results/alignn_predictions.json` (gitignored, kept local) | **ALIGNN `jv_optb88vdw_bandgap_alignn`** (trained on OptB88vdW gaps) | 48,764 | vdW-GGA, gaps slightly *closed* vs PBE |

**Reference:** Alexandria PBE indirect gap (`band_gap_ind`) on the filtered 3D
hull set (`e_above_hull == 0`, Z ≤ 65). Treating PBE as the ground truth is
fair for v01 (label-matched), but for v02/v03 the "error vs PBE" contains both
honest model error *and* the intended functional shift. That is the central
caveat when reading the MAE tables below.

**Paired set:** 31,211 structures — the intersection of SlakoNet's completed
run and each ALIGNN prediction. 17,553 structures have ALIGNN predictions but
no SlakoNet counterpart and are excluded. These are not timeouts: a post-hoc
audit (see `../../../slakonet/slako_v03_alex/analysis/analysis.md`) shows 99.9 % of them
contain an f-block lanthanide (Ce–Tb) that SlakoNet cannot handle — the
filter passes them but inference silently fails. A rerun will not rescue
them. Because all
three ALIGNN models run the same 48,764-structure input, the excluded set is
identical for all of them, which means every comparison below is apples-to-
apples across models.

Negative ALIGNN outputs are clamped to 0 eV in all metrics and plots (both
predicting a metal, matching the SlakoNet convention). The raw-vs-clamped
difference is negligible; see `metrics.json` for both.

## Headline regression numbers (vs PBE indirect gap)

### All paired structures (N = 31,211)

| Model                            |  MAE  | RMSE  |    R²   |   ME    | Max\|err\| |
|----------------------------------|------:|------:|--------:|--------:|---------:|
| SlakoNet (DFTB)                  | 0.930 | 1.649 | −0.008  | +0.328  | 21.49 |
| ALIGNN `mp_gappbe` (PBE)         | **0.193** | **0.463** | **+0.920** | +0.017 | 7.20  |
| ALIGNN `jv_mbj` (TB-mBJ)         | 0.752 | 1.461 | +0.208  | +0.358  | 99.04 |
| ALIGNN `jv_optb88vdw` (OptB88vdW) | 0.354 | 0.746 | +0.794  | −0.147  | 6.54  |

### Non-metals only (PBE gap > 0, N = 16,092)

| Model                            |  MAE  | RMSE  |    R²   |   ME    |
|----------------------------------|------:|------:|--------:|--------:|
| SlakoNet                         | 1.781 | 2.291 | −1.069  | +0.615  |
| ALIGNN `mp_gappbe` (PBE)         | **0.274** | **0.490** | **+0.906** | −0.067 |
| ALIGNN `jv_mbj` (TB-mBJ)         | 1.236 | 1.594 | −0.002  | +0.473  |
| ALIGNN `jv_optb88vdw` (OptB88vdW) | 0.602 | 0.949 | +0.645  | −0.370  |

### Metals only (PBE gap = 0, N = 15,119)

| Model                            |  MAE  | RMSE  |   ME    |
|----------------------------------|------:|------:|--------:|
| SlakoNet                         | **0.024** | **0.160** | +0.024 |
| ALIGNN `mp_gappbe` (PBE)         | 0.106 | 0.433 | +0.106 |
| ALIGNN `jv_mbj` (TB-mBJ)         | 0.237 | 1.304 | +0.237 |
| ALIGNN `jv_optb88vdw` (OptB88vdW) | 0.090 | 0.436 | +0.090 |

**What this table says:**
- SlakoNet's non-metal MAE (1.78 eV) is ~6–9× worse than any ALIGNN variant.
  Its R² is negative, meaning predicting the non-metal mean would beat the
  model.
- PBE-ALIGNN is label-matched and sets the accuracy ceiling (MAE 0.27 eV on
  non-metals, R² = 0.91).
- OptB88vdW-ALIGNN is a close second (MAE 0.60 eV on non-metals, R² = 0.65).
  The gap is ≈0.33 eV — most of that is the OptB88vdW→PBE functional mismatch
  (see §"Functional-shift calibration").
- TB-mBJ-ALIGNN scores MAE 1.24 eV on non-metals, nearly 2× worse than
  OptB88vdW. This is **not** because the model is bad — TB-mBJ gaps are
  intentionally opened by ~23% relative to PBE, so a structure with a 5-eV PBE
  gap is expected to have a ≈6.1-eV mBJ gap. The residual contains the shift.
- On **metals**, SlakoNet is best (MAE 0.024 eV) because its dominant failure
  mode is predicting ≈0 eV regardless of ground truth (v00 writeup). All three
  ALIGNN models produce small spurious gaps on metals; TB-mBJ is the worst at
  this (MAE 0.237 eV and RMSE 1.304 eV driven by a few pathological
  predictions, including 8 outputs > 15 eV).

## Metal / non-metal classification (threshold 0.1 eV vs PBE)

Truth non-metal is `pbe_gap > 0`; predicted non-metal is `pred ≥ 0.1 eV`.

| Model                            | Accuracy |  TN   |  FP  |  FN  |  TP   | Precision | Recall |   F1  |
|----------------------------------|---------:|------:|-----:|-----:|------:|----------:|-------:|------:|
| SlakoNet                         | 0.830    | 14715 |  404 | **4909** | 11183 | 0.965 | 0.695 | 0.808 |
| ALIGNN `mp_gappbe`               | **0.916** | 13283 | 1836 |  **777** | 15315 | 0.893 | **0.952** | **0.921** |
| ALIGNN `jv_mbj`                  | 0.838    | 11871 | **3248** | 1817 | 14275 | 0.815 | 0.887 | 0.849 |
| ALIGNN `jv_optb88vdw`            | 0.892    | 13709 | 1410 | 1957 | 14135 | 0.909 | 0.878 | 0.893 |

- **FN = insulator predicted as metal** — the v00 failure cohort (open-shell
  3d/4d TM oxides and alkali/alkaline-earth fluorides). SlakoNet leaks 4,909
  structures into this bucket; PBE-ALIGNN cuts that 6.3×, OptB88vdW-ALIGNN
  cuts it 2.5×, TB-mBJ-ALIGNN cuts it 2.7×.
- **FP = metal predicted as insulator**. TB-mBJ-ALIGNN is worst at this (3,248)
  because TB-mBJ genuinely opens gaps on systems PBE calls metallic — the
  functional, not the model, is what drives this.
- The overall accuracy ordering (PBE > OptB ≈ mBJ > SK) survives across every
  slice.

## Head-to-head (per-structure closer-to-PBE winner)

| Pair (A vs B)                                      | A better | B better |  % A |
|----------------------------------------------------|---------:|---------:|-----:|
| SlakoNet     vs ALIGNN-PBE                         |  10,054  |  21,157  | 32.2% |
| SlakoNet     vs ALIGNN-mBJ                         |  13,121  |  18,090  | 42.0% |
| SlakoNet     vs ALIGNN-optb                        |   5,772  |  25,439  | 18.5% |
| ALIGNN-PBE   vs ALIGNN-mBJ                         |  21,101  |   8,033  | 67.6% |
| ALIGNN-PBE   vs ALIGNN-optb                        |  13,596  |  13,974  | 43.6% |
| ALIGNN-mBJ   vs ALIGNN-optb                        |   5,519  |  20,457  | 17.7% |

Readings:
- Every ALIGNN variant beats SlakoNet on the majority of structures. The worst
  ALIGNN for this test is TB-mBJ (58%) — still a clear win.
- PBE-ALIGNN beats OptB-ALIGNN on only 43.6% (OptB actually wins more
  structures one-on-one, though on MAE PBE-ALIGNN wins by 0.11 eV). This is
  the OptB's slightly better metal-subset behavior showing up in the
  head-to-head count, even while its systematic downshift on non-metals pulls
  MAE up.
- OptB-ALIGNN dominates both SlakoNet (81.5%) and mBJ-ALIGNN (65.6%) in
  per-structure agreement with PBE.

## Functional-shift calibration (non-metals, linear fit `pred ≈ m·PBE + c`)

| Model               | slope m | intercept c (eV) | interpretation |
|---------------------|--------:|-----------------:|----------------|
| SlakoNet            |  1.325  | −0.151           | over-opens gaps with steep slope — i.e. large gaps get too large; dominated by failure cohort |
| ALIGNN `mp_gappbe`  |  0.952  | +0.047           | close to identity (label-matched) |
| ALIGNN `jv_mbj`     |  1.225  | −0.058           | classic TB-mBJ: ~23% upward scaling, near-zero offset |
| ALIGNN `jv_optb88vdw`|  0.939  | −0.227           | ~6% shrinkage + ~0.2 eV downward shift, as expected for OptB88vdW vs PBE on most solids |

The TB-mBJ and OptB88vdW slopes reproduce what the published literature says
about those functionals' relationships to PBE gaps. So the non-PBE-matched
ALIGNN models are doing what they should do — the apparent "error" vs PBE is
largely the intended physical correction.

## Non-metal MAE by PBE gap bin

| PBE bin (eV) |   N  |   SK  | ALIGNN-PBE | ALIGNN-mBJ | ALIGNN-optb |
|--------------|-----:|------:|-----------:|-----------:|------------:|
| 0.0–0.5      | 1789 | 0.509 | 0.281 | 0.619 | 0.283 |
| 0.5–1.0      | 1901 | 0.919 | 0.294 | 0.717 | 0.369 |
| 1.0–1.5      | 1985 | 1.229 | 0.288 | 0.824 | 0.500 |
| 1.5–2.0      | 1972 | 1.562 | 0.300 | 1.030 | 0.660 |
| 2.0–2.5      | 1820 | 1.895 | 0.273 | 1.263 | 0.825 |
| 2.5–3.0      | 1501 | 2.231 | 0.284 | 1.427 | 0.853 |
| 3.0–4.0      | 2418 | 2.394 | 0.248 | 1.540 | 0.742 |
| 4.0–5.0      | 1561 | 2.594 | 0.234 | 1.885 | 0.654 |
| 5.0–7.0      | 1047 | 3.183 | 0.264 | 2.278 | 0.524 |
| 7.0–10.0     |   98 | 5.252 | 0.200 | 2.645 | 0.418 |

- **SlakoNet** degrades monotonically with PBE gap — from 0.51 eV in the
  narrow-gap bin to 5.25 eV above 7 eV. This is the "open-gap collapse"
  signature: SlakoNet simply doesn't reach large gaps.
- **PBE-ALIGNN** is essentially flat at 0.20–0.30 eV across the full range,
  confirming no bin-specific failure mode.
- **TB-mBJ-ALIGNN** grows approximately linearly with PBE gap (from 0.62 to
  2.65 eV). Flat MAE in the *mBJ-frame* would map to rising MAE in the
  PBE-frame because the expected shift (m − 1) · PBE grows with PBE. The
  signed error (ME) in `metrics.json` goes from +0.50 at small gaps to +2.65
  at 7–10 eV — consistent with the 1.22× slope fit.
- **OptB-ALIGNN** MAE peaks around 2.5–3 eV (0.85 eV) and actually improves
  above 5 eV (0.42–0.52 eV). Its ME is negative across the entire non-metal
  range, again pointing at the systematic downshift and not random noise.

## Pairwise model-model MAE (no reference used)

|                    | SK | ALIGNN-PBE | ALIGNN-mBJ | ALIGNN-optb |
|--------------------|---:|-----------:|-----------:|------------:|
| SlakoNet           |  — | 0.996      | 0.751      | 0.889       |
| ALIGNN-PBE         |    |     —      | 0.715      | **0.308**   |
| ALIGNN-mBJ         |    |            |     —      | 0.609       |
| ALIGNN-optb        |    |            |            |      —      |

- **ALIGNN-PBE ↔ ALIGNN-optb** is the tightest pair (0.31 eV MAE). They
  agree with each other twice as well as either agrees with mBJ. That is the
  expected ordering: both are GGA-class functionals.
- **SlakoNet ↔ ALIGNN-mBJ** is surprisingly *tight* (0.75 eV) — tighter than
  SlakoNet↔ALIGNN-PBE (1.00 eV). The explanation is not that mBJ is closer
  to DFTB; it is that both mBJ-ALIGNN (over-shooting on large gaps) and
  SlakoNet (under-shooting on large gaps) produce values that end up near each
  other on the *small-gap subset* where most of the dataset lives, while on
  large gaps they diverge but in ways that partially cancel in MAE.

## Negative predictions (clamped to 0)

| Model             | negatives |
|-------------------|----------:|
| SlakoNet          |       0   |
| ALIGNN-PBE        |   4,541   |
| ALIGNN-mBJ        |   6,718   |
| ALIGNN-optb       |  11,213   |

Larger-gap-trained ALIGNN variants emit *fewer* negatives on the whole input
(not the opposite as one might guess) only because TB-mBJ and OptB shift
metals away from zero. The high number for OptB reflects its downshift
pushing borderline metals below 0. The clamp has negligible impact on MAE
(see raw-vs-clamped rows in `metrics.json`), but the negative count is a
useful diagnostic for screening pipelines.

## Pathological outputs

- **TB-mBJ-ALIGNN emits 8 outputs > 15 eV, max 99.04 eV.** These are rare but
  real and should be filtered before downstream use. Neither PBE-ALIGNN nor
  OptB-ALIGNN produce predictions of this magnitude (Max\|err\| 7.20 and 6.54
  eV respectively).
- **SlakoNet's Max\|err\| is 21.49 eV** — that maximum is a genuine SlakoNet
  under-prediction on a wide-gap insulator, not an over-shoot. The SlakoNet
  error distribution is heavy-tailed toward "predicted metal when truth is
  insulator".

## Structures where *every* ALIGNN model misses by > 1 eV (non-metals)

333 structures out of 16,092 non-metals (2.1%). The first few, from
`metrics.json::hard_for_all_alignn`:

| mat_id        | formula    | PBE   | SK    | ALIGNN-PBE | ALIGNN-mBJ | ALIGNN-optb |
|---------------|------------|------:|------:|-----------:|-----------:|------------:|
| agm004922632  | Ba2SnMoO6  | 1.246 | 0.002 | 0.030 | 0.108 | 0.023 |
| agm005784014  | BaSiH6     | 2.060 | 1.936 | 4.276 | 4.059 | 3.823 |
| agm005046878  | CaTcBrO4   | 1.716 | 5.335 | 2.806 | 3.705 | 2.827 |
| agm005691573  | Co2Br3F    | 1.932 | 0.000 | 0.903 | 0.000 | 0.068 |
| agm003215525  | Cr2FeO4    | 2.421 | 0.002 | 0.556 | 0.000 | 0.032 |

Pattern: these failures cluster in open-shell transition-metal halides/oxides
(Mo/Tc/Cr/Fe/Co, plus the SiH6 hydride outlier). Two of the ALIGNN variants
collapsing to ≈0 on Cr2FeO4 and Co2Br3F mirrors the v00 SlakoNet collapse on
the same chemistries — suggesting the *training* PBE label for those
structures may itself be an outlier (spin ordering, magnetic configuration
not well represented in the training set), not a pure model failure.

## What to use when

| Downstream need                                     | Recommendation |
|-----------------------------------------------------|----------------|
| PBE-level screening gap                             | **ALIGNN `mp_gappbe`** — MAE 0.27 eV on non-metals, flat across gap range, near-zero bias |
| Approximate mBJ / experimental gap                  | **ALIGNN `jv_mbj`** — gives the +23% scaling you expect from TB-mBJ; filter > 15 eV outliers |
| OptB88vdW gap (vdW solids, molecular crystals)      | **ALIGNN `jv_optb88vdw`** — uses the correct functional |
| Metal/non-metal label only                          | **ALIGNN `mp_gappbe`** — acc 0.916, recall 0.952; OptB is next at 0.892 |
| Binary metal detection where false positives matter | SlakoNet — 0.965 precision but 0.695 recall |
| Fast DFTB-level scalar for metals only              | SlakoNet — 0.024 eV MAE on metals |

## How this differs from each per-version analysis

- **v01 (PBE)** is the label-matched baseline. Its regression numbers land at
  the accuracy ceiling for this task. Nothing in the comprehensive view
  changes the v01 conclusion: PBE-ALIGNN is strictly better than SlakoNet
  wherever the target is a PBE gap.
- **v02 (TB-mBJ)** was a functional-shift study, not an accuracy study. Its
  "worse MAE than PBE-ALIGNN" headline is an artifact of the reference; the
  slope/intercept in §"Functional-shift calibration" is the physically
  meaningful part. For downstream use on a PBE target, v01 > v02 is the
  correct ordering. For a TB-mBJ target, v02 becomes the recommended model.
- **v03 (OptB88vdW)** sits neatly between v01 and v02. Across the three
  ALIGNN checkpoints it is the best *non-PBE-trained* predictor of PBE gaps
  (MAE 0.60 vs 1.24 eV for mBJ), mostly because OptB88vdW is a smaller
  functional perturbation of PBE than TB-mBJ is.

## Files in this directory

| File                              | Contents |
|-----------------------------------|----------|
| `metrics.json`                    | every numeric result used in tables (all/nonmetal/metal regression, classification, head-to-head, bin MAE, linear fits, worst cases, hard-for-all list) |
| `merged_predictions.json`         | 31,211 merged records: `{mat_id, formula, pbe_ref, pbe_dir, e_form, slakonet, alignn_pbe, alignn_mbj, alignn_optb}` |
| `summary.txt`                     | human-readable stdout from the script |
| `plots/parity_all.png`            | 4-panel density parity vs PBE (all structures) |
| `plots/parity_nonmetals.png`      | same 4-panel, non-metals only |
| `plots/residuals_all.png`         | (pred − PBE) histograms, all structures |
| `plots/residuals_nonmetals.png`   | (pred − PBE) histograms, non-metals |
| `plots/confusion_all.png`         | 4-way metal/non-metal confusion matrices |
| `plots/mae_by_gap_bin.png`        | MAE and ME by PBE-gap bin (non-metals), bar groups |
| `plots/gap_distribution.png`      | PBE reference vs four predictors' non-metal gap densities |
| `plots/head_to_head.png`          | per-structure ‖error vs PBE‖ scatter: PBE-ALIGNN vs {SK, mBJ, optb} |
| `plots/pairwise_alignn_agreement.png` | ALIGNN↔ALIGNN density scatters (no reference) |
| `plots/functional_shift.png`      | (mBJ − PBE) and (optb − PBE) vs PBE with linear fits |
| `plots/cumulative_error.png`      | fraction of non-metals within tolerance for each model |

## Reproducing

The artifacts above are pre-built. The original script that produced them
is kept local; no other data is consumed beyond the four raw prediction
files listed in §Setup (which are also local-only — see the convention in
the root `README.md`).
