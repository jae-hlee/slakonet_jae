# SlakoNet on interface_db — Results Analysis

## Overview
SlakoNet (tight-binding neural network) was run on the **interface_db** dataset
of heterostructure interfaces. Each entry in the dataset is a film/substrate
stack (identified by a pair of JVASP IDs) with specified Miller planes,
thicknesses, separations, and lateral displacements. The task is to predict the
**electronic band gap** of each interface, benchmarked against DFT
(`optb88vdw_bandgap`). Full **DOS** curves are saved as a secondary output.

- Dataset total: 593 entries
- Successfully evaluated: **433** (after Z ≤ 65 element filter and prep/timeout exclusions)
- Reference: optB88-vdW DFT gaps (`optb88vdw_bandgap`, `_cbm`, `_vbm`)

## Summary statistics

| Quantity                       |   N | mean    | std   | min     | max     | median  |
|-------------------------------|----:|--------:|------:|--------:|--------:|--------:|
| SlakoNet band gap (eV)        | 433 | +1.432  | 1.036 | +0.000  | +4.716  | +1.412  |
| DFT gap (eV, raw)             | 433 | +0.308  | 0.665 | −2.658  | +2.809  | +0.334  |
| DFT gap (eV, clipped at 0)    | 433 | +0.429  | 0.476 | +0.000  | +2.809  | +0.334  |
| DFT CBM (eV)                  | 433 | +4.381  | 1.844 | +0.289  | +8.998  | +4.236  |
| DFT VBM (eV)                  | 433 | +4.690  | 1.783 | +0.665  | +10.013 | +4.574  |
| Interface offset (eV)         | 322 | −0.066  | 1.114 | −2.574  | +6.045  | −0.148  |
| Final energy (eV)             | 433 | −263.44 | 241.1 | −1297.3 | +5.71   | −191.3  |

- **DFT gaps < 0** (band overlap / metallic interface): **95 / 433 = 21.9%**
- **SlakoNet gaps ≈ 0** (< 1 meV): **15 / 433 = 3.5%**
- Interface offset populated for 322 / 433 entries (the rest have an empty-dict offset field in the raw dataset).

## SlakoNet vs DFT parity

Metrics computed against DFT gap clipped to ≥ 0 (since SlakoNet cannot predict
negative gaps):

- **MAE = 1.013 eV**
- **RMSE = 1.259 eV**
- **Pearson r = 0.730**

See `plots/parity_sk_vs_dft.png`.

## Key observations

1. **Systematic overestimation.** SlakoNet predicts gaps ~1 eV larger than DFT
   on average (mean 1.43 vs 0.43 eV clipped / 0.31 eV raw). All 15 of the
   largest |residual| outliers are positive (SK − DFT = +2.5 to +3.9 eV).
2. **Metallic interfaces are missed.** ~22% of the interfaces are metallic or
   near-metallic in DFT (gap ≤ 0), but only 3.5% of SlakoNet predictions come
   out near zero. SlakoNet tends to open a gap where DFT sees band overlap.
3. **Moderate correlation (r = 0.73).** The trend is right — SlakoNet
   distinguishes small- vs large-gap interfaces — but the quantitative spread
   is large and error is gap-dependent (see `plots/residual_vs_gap.png`).
4. **Outlier concentration.** The top-15 residuals are dominated by
   `miller_1_1_0` interfaces, suggesting a facet-specific failure mode worth
   investigating.
5. **DOS quality.** Averaged DOS curves (`plots/dos_average.png`) clearly separate
   small- vs large-gap populations around the Fermi level, and individual DOS
   examples at the 10th/50th/90th SK-gap percentiles (`plots/dos_examples.png`)
   show qualitatively reasonable shapes.

## Interpretation

SlakoNet is a tight-binding NN trained on bulk / molecular systems; applying it
to heterostructure interfaces is an extrapolation. The systematic
gap-opening bias is consistent with the model not capturing interface-specific
hybridization and charge transfer that drive metallization or gap
reduction in DFT. For screening applications the **ranking signal** (r ≈ 0.73)
is usable; for quantitative gap prediction, a calibration step or fine-tuning
on interface data would likely be needed.

## Artifacts in this directory

- `plots/parity_sk_vs_dft.png` — density scatter, SlakoNet vs clipped DFT
- `plots/distributions.png` — per-quantity histograms (SK gap, DFT gap, CBM/VBM, offset)
- `plots/gap_overlay.png` — SlakoNet vs DFT gap distributions overlaid
- `plots/residuals.png` — histogram of SK − DFT
- `plots/residual_vs_gap.png` — residual as a function of DFT gap
- `plots/dos_average.png` — mean DOS, small-gap vs large-gap populations
- `plots/dos_examples.png` — representative DOS at 10th / 50th / 90th percentile of SK gap
- `csv/summary.csv` — per-structure table of all scalar fields and residuals
