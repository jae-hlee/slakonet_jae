# TB-mBJ ALIGNN vs SlakoNet on filtered Alexandria (v02)

## Setup

- **ALIGNN model**: `jv_mbj_bandgap_alignn` (trained on JARVIS TB-mBJ band gaps).
- **SlakoNet results**: reused from `slakonet/slako_v03_alex/results/sk_scalars.json` (gitignored, kept local)
  (same DFTB run, same filtered Alexandria hull set).
- **Reference label** in the merged records: PBE indirect gap (`band_gap_ind`)
  from the Alexandria dataset — **this is what we compare *against*, but note
  that TB-mBJ is not trained to reproduce PBE**, so a large mBJ−PBE residual
  is expected and physically meaningful, not a model defect.
- **Paired set**: 31,211 structures (intersection of 48,764 v02 ALIGNN predictions
  and 31,211 SlakoNet results). 17,553 ALIGNN-only structures are carried over
  from the v00 missing set and are not analyzed here — 99.9 % of those are
  lanthanide-containing cells that SlakoNet silently rejects (see
  `../../pbe_mbj_opt_analysis/analysis.md`), so the excluded set is not
  recoverable by rerunning.

## Key numbers

**Regression vs PBE indirect gap (31,211 structures)**

| Subset              | Model                | MAE   | RMSE  | R²     | Mean error |
|---------------------|----------------------|------:|------:|-------:|-----------:|
| All                 | SlakoNet             | 0.930 | 1.649 | −0.008 |   +0.329   |
| All                 | ALIGNN TB-mBJ        | **0.752** | **1.461** | **+0.208** |   +0.359   |
| Non-metals (16,092) | SlakoNet             | 1.781 | 2.291 | −1.069 |   +0.615   |
| Non-metals          | ALIGNN TB-mBJ        | **1.236** | **1.594** | −0.002 |   +0.473   |
| Metals (15,119)     | SlakoNet             | **0.024** | **0.160** |   n/a  |   +0.024   |
| Metals              | ALIGNN TB-mBJ        | 0.237 | 1.304 |   n/a  |   +0.237   |

**Metal / non-metal classification (threshold 0.1 eV, vs PBE)**

|                 | Accuracy | FN (insulator→metal) | FP (metal→insulator) |
|-----------------|---------:|---------------------:|---------------------:|
| SlakoNet        | 0.830    | 4,909                |   404                |
| ALIGNN TB-mBJ   | **0.838**| **1,817**            | 3,248                |

**Head-to-head vs PBE**: ALIGNN TB-mBJ closer on 58.0%, SlakoNet closer on 42.0%.

**Direct model-model agreement** (TB-mBJ vs SlakoNet, no PBE):
MAE 0.751 eV, R² 0.618 — they agree substantially more with each other than
either does with PBE on non-metals, because both models are "sane" predictors
on the clean subset; disagreement is concentrated in SlakoNet's failure cohort.

## How this differs from v00 (PBE ALIGNN)

The v00 analysis used `mp_gappbe_alignn`, which is trained to predict the same
quantity (PBE gap) as the reference. That model hit MAE = 0.19 eV overall and
0.27 eV on non-metals — effectively matched to the label. The TB-mBJ model
here is **not** label-matched: mBJ gaps are systematically larger than PBE
gaps, so any "MAE vs PBE" reading is really "gap-level shift between two
DFT functionals as learned by two ALIGNN checkpoints".

- **MAE vs PBE gets worse** with TB-mBJ (0.75 vs v00's 0.19 eV overall) —
  expected.
- **MAE vs SlakoNet** is comparable to MAE vs PBE, because SlakoNet is itself
  a PBE-ish predictor on the well-behaved subset and a mess on the failure
  cohort.
- The **linear fit** on non-metals
  `ALIGNN_mBJ ≈ 1.23 · PBE − 0.06 eV`
  reproduces the textbook TB-mBJ behavior: gaps are opened by ~23% relative
  to PBE, with near-zero offset. This is the main *physics* signal to take
  away from the run.

## How this differs from SlakoNet

- **Non-metal regression**: TB-mBJ is a clear win (MAE 1.24 vs 1.78, R² 0.00
  vs −1.07). R² ≈ 0 means TB-mBJ's errors are roughly on the order of PBE's
  variance — not because TB-mBJ is bad, but because it's predicting a
  *different gap* (mBJ ≈ 1.23·PBE), so the "residual vs PBE" contains both
  honest error *and* the intended mBJ shift. SlakoNet's R² being deeply
  negative is the catastrophic-collapse failure mode documented in v00.
- **Metal regression**: SlakoNet wins in MAE (0.024 vs 0.237), but only
  because its dominant failure mode (predicting ≈ 0 on both metals and
  Mott-insulating TM oxides) accidentally gives the right answer on metals.
  TB-mBJ's 3,248 false positives reflect the mBJ functional's tendency to
  open small gaps on systems PBE calls metallic — this is expected behavior
  of the functional, not a model bug.
- **False negatives** (insulator predicted as metal): TB-mBJ cuts this
  ~2.7×, from 4,909 → 1,817. It does not collapse the open-shell 3d/4d
  transition-metal oxides the way SlakoNet does, because it doesn't go
  through a spin-unpolarized SCF.
- **Head-to-head**: TB-mBJ is closer to PBE on 58% of structures despite
  being *trained on a different functional* — the SlakoNet failure cohort is
  large enough to outweigh mBJ's systematic upward shift.

## Artifacts worth flagging

- **6,718 negative raw predictions** (clamped to 0 in all reported stats).
  Comparable to v00 PBE model behavior.
- **8 outliers > 15 eV**, max 99.0 eV. These are rare pathological
  predictions — recommend filtering before using the output as a screening
  signal. The v00 PBE model did not produce predictions in that range.

## Files in this directory

| File                                     | What it shows                                                       |
|------------------------------------------|---------------------------------------------------------------------|
| `stats.txt`                              | Raw numeric output (MAE/RMSE/R²/classification counts)             |
| `paired_predictions.json`                | 31,211 merged records: mat_id, formula, PBE, SK, ALIGNN-mBJ, e_form |
| `plots/parity_three_way.png`                   | Density scatter: SK vs PBE, mBJ vs PBE, mBJ vs SK (all structures) |
| `plots/parity_three_way_nonmetals.png`         | Same, restricted to PBE non-metals                                 |
| `plots/residuals_sk_vs_alignn_mbj.png`         | (pred − PBE) histograms for SlakoNet and TB-mBJ side-by-side       |
| `plots/confusion_sk_vs_alignn_mbj.png`         | Metal/non-metal confusion matrices at 0.1 eV threshold             |
| `plots/head_to_head_error.png`                 | Per-structure \|err vs PBE\| scatter: SK vs TB-mBJ                 |
| `plots/gap_distribution_alignn_mbj.png`        | Overlayed densities: PBE, SK, TB-mBJ (non-metals)                  |
| `plots/mbj_correction_vs_pbe.png`              | (TB-mBJ − PBE) as a function of PBE gap with binned mean           |

## Short recommendations

1. **Treat TB-mBJ and PBE ALIGNN as different quantities.** The two models
   answer different questions; the fair comparison of model skill is
   "TB-mBJ vs experimental gap" or "TB-mBJ vs DFT-mBJ", not TB-mBJ vs PBE.
   We don't have those labels in Alexandria, so the numbers above are
   best read as *consistency checks* and *shift calibration*, not as
   ALIGNN-mBJ accuracy.
2. **For screening with an mBJ-level estimate**, prefer TB-mBJ over SlakoNet
   for compositions with open-shell 3d/4d TMs and alkali/alkaline-earth
   fluorides (the SlakoNet failure cohorts diagnosed in v00).
3. **Filter the 8 > 15 eV outliers and the 6,718 negative predictions**
   before downstream use.
4. **If a PBE-level gap is what you actually want**, use v00's
   `mp_gappbe_alignn` predictions — they are label-matched and much more
   accurate (MAE 0.19 eV).
