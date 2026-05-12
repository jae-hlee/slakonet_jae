# v11 e_form analysis: PBE arm (mp_e_form_alignn)

Pretrained model trained on **MP/PBE** formation energy (eV/atom). Inference on the **Alexandria PBE 3D hull (no Z filter)** set, N = 115,535. Reference is `e_form` from Alexandria PBE.

## Headline metrics

| metric | value (eV/atom) |
|---|---|
| MAE | **0.0356** |
| RMSE | 0.0667 |
| ME (bias) | +0.0110 |
| Median \|err\| | 0.0192 |
| 90th-pct \|err\| | 0.0818 |
| 95th-pct \|err\| | 0.1191 |
| 99th-pct \|err\| | 0.2527 |
| Pearson r | 0.9973 |
| ALIGNN mean / median | -1.118 / -0.792 |
| DFT mean / median | -1.129 / -0.805 |

## Cumulative error distribution

Fraction of structures with |ALIGNN - DFT| below each threshold:

| threshold (eV/atom) | count | fraction |
|---|---|---|
| 0.005 | 19,240 | 16.65% |
| 0.010 | 34,940 | 30.24% |
| 0.020 | 59,297 | 51.32% |
| 0.050 | 91,272 | 79.00% |
| 0.100 | 107,462 | 93.01% |
| 0.200 | 113,535 | 98.27% |
| 0.300 | 114,792 | 99.36% |
| 0.500 | 115,391 | 99.88% |
| 1.000 | 115,526 | 99.99% |

## Stratified metrics (metal vs non-metal)

Split by DFT `band_gap_ind`: == 0 vs > 0.

| subset | N | MAE | RMSE | ME | med \|err\| | p90 \|err\| |
|---|---|---|---|---|---|---|
| all | 115,535 | 0.0356 | 0.0667 | +0.0110 | 0.0192 | 0.0818 |
| metal | 79,214 | 0.0374 | 0.0657 | +0.0126 | 0.0217 | 0.0857 |
| non-metal | 36,321 | 0.0317 | 0.0688 | +0.0074 | 0.0153 | 0.0714 |

## Bandgap-bin stratification

Five bins by DFT `band_gap_ind` (eV).

| bin | N | MAE | RMSE | ME | med \|err\| |
|---|---|---|---|---|---|
| metal (=0) | 79,214 | 0.0374 | 0.0657 | +0.0126 | 0.0217 |
| small (0,0.5] | 6,286 | 0.0460 | 0.0857 | -0.0028 | 0.0230 |
| med  (0.5,1.5] | 9,350 | 0.0348 | 0.0733 | +0.0131 | 0.0162 |
| wide (1.5,3.5] | 13,659 | 0.0278 | 0.0613 | +0.0090 | 0.0142 |
| ultra (>3.5) | 7,026 | 0.0226 | 0.0587 | +0.0057 | 0.0135 |

## Composition cardinality stratification

By number of distinct elements per formula.

| cardinality | N | MAE | RMSE | ME | med \|err\| |
|---|---|---|---|---|---|
| 1 elem | 89 | 0.0340 | 0.0792 | +0.0312 | 0.0071 |
| 2 elem | 7,245 | 0.0364 | 0.0795 | +0.0153 | 0.0136 |
| 3 elem | 71,962 | 0.0354 | 0.0645 | +0.0114 | 0.0193 |
| 4 elem | 34,834 | 0.0369 | 0.0692 | +0.0097 | 0.0211 |
| 5+ elem | 1,405 | 0.0136 | 0.0304 | -0.0049 | 0.0111 |

## Cross-arm comparison

On N = 115,535 records (positional pairing):

| metric | value |
|---|---|
| PBE - Opt mean (signed) | -0.0411 eV/atom |
| PBE - Opt median | +0.0032 eV/atom |
| mean \|PBE - Opt\| | 0.0999 eV/atom |
| max \|PBE - Opt\| | 1.5475 eV/atom |
| Pearson r (PBE vs Opt) | 0.9891 |
| fraction \|diff\| <= 0.05 | 44.8% |
| fraction \|diff\| <= 0.10 | 68.4% |
| fraction \|diff\| <= 0.50 | 99.2% |

Top 25 cross-arm disagreements (full table at `csv/cross_arm_top_disagreements.csv`):

| id | formula | PBE | Opt | PBE - Opt |
|---|---|---|---|---|
| agm001231920 | LaEuZn2 | +1.129 | -0.418 | +1.548 |
| agm001232005 | PrEuZn2 | +0.712 | -0.380 | +1.092 |
| agm003269539 | Pu2InPt2 | +0.112 | -0.928 | +1.039 |
| agm003404222 | Pu2SnPt2 | +0.029 | -0.980 | +1.008 |
| agm002234886 | Pu2TlPt2 | +0.192 | -0.802 | +0.994 |
| agm001231951 | NdEuZn2 | +0.581 | -0.384 | +0.965 |
| agm002179958 | Pu2Pd2Pb | +0.303 | -0.648 | +0.951 |
| agm005900099 | Pu2BiPt2 | +0.028 | -0.910 | +0.938 |
| agm005832871 | Pu2SbPt2 | -0.120 | -1.002 | +0.882 |
| agm003395674 | Pu2Pt2Au | -0.010 | -0.882 | +0.872 |
| agm005410664 | Eu(MnGe)2 | -0.343 | -1.208 | +0.866 |
| agm005289126 | PuTe2 | -0.222 | -1.083 | +0.861 |
| agm003232173 | ClO2 | -0.536 | +0.295 | -0.831 |
| agm002351369 | Pu2HgPt2 | +0.055 | -0.773 | +0.828 |
| agm002261476 | Pu2CdPt2 | -0.036 | -0.863 | +0.827 |
| agm003395663 | Pu2AgPt2 | +0.004 | -0.807 | +0.810 |
| agm002234762 | Pu2TlPd2 | +0.230 | -0.577 | +0.807 |
| agm003211534 | EuVO4 | -2.836 | -3.619 | +0.783 |
| agm005543511 | NF3 | -0.910 | -0.133 | -0.778 |
| agm002281331 | Pu3In3Ir2 | +0.173 | -0.604 | +0.776 |
| agm002169632 | ClO2F | -0.809 | -0.043 | -0.766 |
| agm003231566 | FeSO4 | -2.054 | -1.291 | -0.764 |
| agm005279378 | NO2 | -0.773 | -0.009 | -0.764 |
| agm002351354 | Pu2Pt2Pb | -0.058 | -0.818 | +0.760 |
| agm003247361 | SClO2F | -1.761 | -1.007 | -0.754 |


## Per-element MAE (top 15 worst, count >= 500)

| element | count | MAE | median \|err\| |
|---|---|---|---|
| Np | 2,766 | 0.1227 | 0.0992 |
| Pu | 4,008 | 0.1009 | 0.0652 |
| Pa | 3,645 | 0.0677 | 0.0540 |
| Th | 3,931 | 0.0656 | 0.0468 |
| N | 5,825 | 0.0635 | 0.0437 |
| Ac | 4,337 | 0.0462 | 0.0368 |
| Be | 1,843 | 0.0455 | 0.0276 |
| Pt | 6,723 | 0.0454 | 0.0256 |
| H | 5,180 | 0.0448 | 0.0268 |
| F | 5,450 | 0.0443 | 0.0166 |
| Tc | 2,271 | 0.0441 | 0.0310 |
| Yb | 1,487 | 0.0433 | 0.0263 |
| U | 3,222 | 0.0425 | 0.0235 |
| Cr | 1,602 | 0.0423 | 0.0163 |
| Cl | 4,333 | 0.0418 | 0.0211 |

Full per-element table at `csv/per_element_mae.csv`.

## Top 25 worst predictions

| id | formula | DFT | ALIGNN | residual | band_gap_ind |
|---|---|---|---|---|---|
| agm002116455 | CrTcO | -4.912 | -0.711 | +4.201 | 0.0 |
| agm004676761 | Rb3CrNi2F9 | -6.193 | -2.604 | +3.589 | 5.0398 |
| agm005277902 | CsCrNiF6 | -5.719 | -2.794 | +2.925 | 2.5599 |
| agm004968779 | Pr2BeCrO6 | -5.645 | -3.035 | +2.610 | 1.3966 |
| agm005102383 | SrCrCoF6 | -5.676 | -3.107 | +2.569 | 1.5861 |
| agm004969345 | Sr2ThCrO6 | -5.326 | -2.944 | +2.382 | 0.0 |
| agm004890438 | LiCrCu2F8 | -4.404 | -2.279 | +2.125 | 0.0 |
| agm004911156 | Li2CeCrF8 | -5.571 | -3.460 | +2.111 | 0.0695 |
| agm001231920 | LaEuZn2 | -0.292 | +1.129 | +1.422 | 0.0 |
| agm001232005 | PrEuZn2 | -0.287 | +0.712 | +0.999 | 0.0 |
| agm003235847 | Li4Cr3Fe2Sb3O16 | -3.007 | -2.117 | +0.890 | 0.7417 |
| agm001231951 | NdEuZn2 | -0.288 | +0.581 | +0.869 | 0.0 |
| agm003269539 | Pu2InPt2 | -0.732 | +0.112 | +0.843 | 0.0 |
| agm002234886 | Pu2TlPt2 | -0.643 | +0.192 | +0.835 | 0.0 |
| agm005546508 | PuO5 | -2.063 | -1.237 | +0.826 | 0.21530000000000002 |
| agm003404222 | Pu2SnPt2 | -0.780 | +0.029 | +0.808 | 0.0 |
| agm005900099 | Pu2BiPt2 | -0.715 | +0.028 | +0.743 | 0.0 |
| agm002179958 | Pu2Pd2Pb | -0.439 | +0.303 | +0.742 | 0.0 |
| agm005728581 | YNp2O11 | -2.658 | -1.931 | +0.727 | 0.7453000000000001 |
| agm002351369 | Pu2HgPt2 | -0.668 | +0.055 | +0.723 | 0.0 |
| agm005234725 | SO5 | -1.368 | -0.651 | +0.717 | 1.8339 |
| agm005732821 | PmNp2O11 | -2.664 | -1.947 | +0.717 | 0.7267 |
| agm005732645 | TmNp2O11 | -2.682 | -1.974 | +0.708 | 0.7577 |
| agm003395674 | Pu2Pt2Au | -0.717 | -0.010 | +0.707 | 0.0 |
| agm001630546 | PuPt2 | -0.854 | -0.150 | +0.703 | 0.0 |

Full top-100 at `csv/worst_predictions.csv`.

## Top 10 best predictions (smallest |residual|)

| id | formula | DFT | ALIGNN | residual |
|---|---|---|---|---|
| agm004547018 | Ce2Th(SiNi)2 | -0.682 | -0.682 | +0.0000 |
| agm005670441 | Sc10(TlSb)3 | -0.583 | -0.583 | -0.0000 |
| agm002203879 | CaIn2Pd | -0.614 | -0.614 | -0.0000 |
| agm005277432 | BaSc3AgS6 | -2.099 | -2.099 | -0.0000 |
| agm003202464 | ScCo3B2 | -0.543 | -0.543 | -0.0000 |
| agm001172951 | DyLuCo4 | -0.248 | -0.248 | -0.0000 |
| agm003189739 | AlCrRu2 | -0.363 | -0.363 | -0.0000 |
| agm005014713 | LaTbSe2S | -2.191 | -2.191 | +0.0000 |
| agm004612621 | Rb3Tb(SmTe3)2 | -1.398 | -1.398 | +0.0000 |
| agm003124897 | NaNbO2 | -2.538 | -2.538 | -0.0000 |

## Files

- `csv/metrics.csv` — overall + metal/non-metal + bandgap-bin + cardinality
- `csv/per_element_mae.csv` — every element above min_count
- `csv/cumulative_error.csv` — fraction within each |err| threshold
- `csv/worst_predictions.csv` — top 100 by |residual|
- `csv/best_predictions.csv` — top 25 with smallest |residual|
- `plots/parity.png` — hexbin parity + residual panel
- `plots/residual_histogram.png` — (ALIGNN - DFT) distribution
- `plots/residual_cdf.png` — cumulative |residual| distribution
- `plots/per_element_mae.png` — bar chart of worst-15 elements
- `plots/error_vs_eform.png` — |err| vs DFT e_form scatter
- `plots/bandgap_bin_mae.png` — MAE per bandgap bin
- `plots/composition_cardinality.png` — MAE by number of elements
- `plots/cross_arm_comparison.png` — PBE vs Opt arm: scatter, signed-diff histogram, |diff| CDF

## Caveats

- Reference is Alexandria PBE `e_form`; this arm trained on **MP/PBE** e_form. Training and reference functionals match.
- Includes Z>65 elements (lanthanides Ce-Tb and actinides) absent from v03.
- All structures hull-stable (DFT e_form <= 0).
- Cross-arm comparison with `eform_v2_opt/analysis/` is in the cross-arm section above; large Opt-vs-this disagreements are characterized in `csv/worst_predictions.csv`.
