# v03 e_form analysis: PBE arm (mp_e_form_alignn)

Pretrained model trained on **MP/PBE** formation energy (eV/atom). Inference on the **Alexandria PBE 3D hull (Z<=65)** set, N = 48,764. Reference is `e_form` from Alexandria PBE.

## Headline metrics

| metric | value (eV/atom) |
|---|---|
| MAE | **0.0287** |
| RMSE | 0.0618 |
| ME (bias) | +0.0069 |
| Median \|err\| | 0.0158 |
| 90th-pct \|err\| | 0.0639 |
| 95th-pct \|err\| | 0.0910 |
| 99th-pct \|err\| | 0.1951 |
| Pearson r | 0.9978 |
| ALIGNN mean / median | -1.213 / -0.872 |
| DFT mean / median | -1.220 / -0.878 |

## Cumulative error distribution

Fraction of structures with |ALIGNN - DFT| below each threshold:

| threshold (eV/atom) | count | fraction |
|---|---|---|
| 0.005 | 9,300 | 19.07% |
| 0.010 | 16,735 | 34.32% |
| 0.020 | 28,297 | 58.03% |
| 0.050 | 41,328 | 84.75% |
| 0.100 | 46,750 | 95.87% |
| 0.200 | 48,302 | 99.05% |
| 0.300 | 48,572 | 99.61% |
| 0.500 | 48,720 | 99.91% |
| 1.000 | 48,756 | 99.98% |

## Stratified metrics (metal vs non-metal)

Split by DFT `band_gap_ind`: == 0 vs > 0.

| subset | N | MAE | RMSE | ME | med \|err\| | p90 \|err\| |
|---|---|---|---|---|---|---|
| all | 48,764 | 0.0287 | 0.0618 | +0.0069 | 0.0158 | 0.0639 |
| metal | 28,420 | 0.0297 | 0.0555 | +0.0068 | 0.0184 | 0.0673 |
| non-metal | 20,344 | 0.0273 | 0.0696 | +0.0071 | 0.0137 | 0.0571 |

## Bandgap-bin stratification

Five bins by DFT `band_gap_ind` (eV).

| bin | N | MAE | RMSE | ME | med \|err\| |
|---|---|---|---|---|---|
| metal (=0) | 28,420 | 0.0297 | 0.0555 | +0.0068 | 0.0184 |
| small (0,0.5] | 2,783 | 0.0365 | 0.0814 | +0.0102 | 0.0172 |
| med  (0.5,1.5] | 4,910 | 0.0291 | 0.0695 | +0.0088 | 0.0138 |
| wide (1.5,3.5] | 7,990 | 0.0267 | 0.0674 | +0.0072 | 0.0137 |
| ultra (>3.5) | 4,661 | 0.0212 | 0.0657 | +0.0032 | 0.0126 |

## Composition cardinality stratification

By number of distinct elements per formula.

| cardinality | N | MAE | RMSE | ME | med \|err\| |
|---|---|---|---|---|---|
| 1 elem | 65 | 0.0409 | 0.0907 | +0.0379 | 0.0093 |
| 2 elem | 3,858 | 0.0331 | 0.0740 | +0.0146 | 0.0125 |
| 3 elem | 29,803 | 0.0292 | 0.0567 | +0.0068 | 0.0163 |
| 4 elem | 13,955 | 0.0275 | 0.0695 | +0.0059 | 0.0164 |
| 5+ elem | 1,083 | 0.0133 | 0.0314 | -0.0057 | 0.0111 |

## Cross-arm comparison

On N = 48,764 records (positional pairing):

| metric | value |
|---|---|
| PBE - Opt mean (signed) | -0.0672 eV/atom |
| PBE - Opt median | -0.0102 eV/atom |
| mean \|PBE - Opt\| | 0.1131 eV/atom |
| max \|PBE - Opt\| | 1.5475 eV/atom |
| Pearson r (PBE vs Opt) | 0.9883 |
| fraction \|diff\| <= 0.05 | 43.3% |
| fraction \|diff\| <= 0.10 | 64.5% |
| fraction \|diff\| <= 0.50 | 98.8% |

Top 25 cross-arm disagreements (full table at `csv/cross_arm_top_disagreements.csv`):

| id | formula | PBE | Opt | PBE - Opt |
|---|---|---|---|---|
| agm001231920 | LaEuZn2 | +1.129 | -0.418 | +1.548 |
| agm001232005 | PrEuZn2 | +0.712 | -0.380 | +1.092 |
| agm001231951 | NdEuZn2 | +0.581 | -0.384 | +0.965 |
| agm005410664 | Eu(MnGe)2 | -0.343 | -1.208 | +0.866 |
| agm003232173 | ClO2 | -0.536 | +0.295 | -0.831 |
| agm003211534 | EuVO4 | -2.836 | -3.619 | +0.783 |
| agm005543511 | NF3 | -0.910 | -0.133 | -0.778 |
| agm002169632 | ClO2F | -0.809 | -0.043 | -0.766 |
| agm003231566 | FeSO4 | -2.054 | -1.291 | -0.764 |
| agm005279378 | NO2 | -0.773 | -0.009 | -0.764 |
| agm003247361 | SClO2F | -1.761 | -1.007 | -0.754 |
| agm003242462 | Fe2(SO4)3 | -2.021 | -1.272 | -0.749 |
| agm005545960 | S5F | -0.943 | -0.201 | -0.743 |
| agm003219343 | SCl2O | -1.239 | -0.510 | -0.729 |
| agm005848923 | Pr(ClO4)3 | -1.137 | -0.409 | -0.728 |
| agm005704712 | NO2F | -1.028 | -0.302 | -0.727 |
| agm003251572 | SO3 | -1.764 | -1.038 | -0.726 |
| agm003211653 | MnSO4 | -2.173 | -1.456 | -0.717 |
| agm005848915 | Nd(ClO4)3 | -1.134 | -0.424 | -0.710 |
| agm002176638 | MnO | -1.996 | -1.287 | -0.709 |
| agm003231932 | CoSO4 | -1.882 | -1.173 | -0.709 |
| agm005281484 | EuMoO5 | -1.945 | -2.653 | +0.708 |
| agm005505561 | MnFe4O5 | -1.744 | -1.041 | -0.703 |
| agm005230213 | Ti(NO3)4 | -1.332 | -0.631 | -0.700 |
| agm003241117 | La(ClO4)3 | -1.155 | -0.456 | -0.700 |


## Per-element MAE (top 15 worst, count >= 100)

| element | count | MAE | median \|err\| |
|---|---|---|---|
| N | 2,725 | 0.0547 | 0.0320 |
| Cr | 1,076 | 0.0444 | 0.0153 |
| Tc | 1,187 | 0.0430 | 0.0308 |
| Br | 1,900 | 0.0393 | 0.0253 |
| Cl | 2,498 | 0.0389 | 0.0185 |
| F | 3,445 | 0.0376 | 0.0137 |
| Be | 979 | 0.0373 | 0.0197 |
| Pm | 2,875 | 0.0372 | 0.0292 |
| Ce | 2,846 | 0.0369 | 0.0263 |
| I | 1,830 | 0.0366 | 0.0231 |
| C | 2,062 | 0.0360 | 0.0217 |
| H | 3,299 | 0.0353 | 0.0199 |
| La | 2,959 | 0.0343 | 0.0234 |
| Eu | 1,032 | 0.0339 | 0.0172 |
| Nb | 1,450 | 0.0307 | 0.0168 |

Full per-element table at `csv/per_element_mae.csv`.

## Top 25 worst predictions

| id | formula | DFT | ALIGNN | residual | band_gap_ind |
|---|---|---|---|---|---|
| agm002116455 | CrTcO | -4.912 | -0.711 | +4.201 | 0.0 |
| agm004676761 | Rb3CrNi2F9 | -6.193 | -2.604 | +3.589 | 5.0398 |
| agm005277902 | CsCrNiF6 | -5.719 | -2.794 | +2.925 | 2.5599 |
| agm004968779 | Pr2BeCrO6 | -5.645 | -3.035 | +2.610 | 1.3966 |
| agm005102383 | SrCrCoF6 | -5.676 | -3.107 | +2.569 | 1.5861 |
| agm004890438 | LiCrCu2F8 | -4.404 | -2.279 | +2.125 | 0.0 |
| agm004911156 | Li2CeCrF8 | -5.571 | -3.460 | +2.111 | 0.0695 |
| agm001231920 | LaEuZn2 | -0.292 | +1.129 | +1.422 | 0.0 |
| agm001232005 | PrEuZn2 | -0.287 | +0.712 | +0.999 | 0.0 |
| agm003235847 | Li4Cr3Fe2Sb3O16 | -3.007 | -2.117 | +0.890 | 0.7417 |
| agm001231951 | NdEuZn2 | -0.288 | +0.581 | +0.869 | 0.0 |
| agm005234725 | SO5 | -1.368 | -0.651 | +0.717 | 1.8339 |
| agm005251682 | ScPO6 | -2.618 | -1.924 | +0.695 | 1.6383 |
| agm003237184 | RbEuSiS4 | -1.641 | -0.968 | +0.672 | 0.0 |
| agm005289887 | Li24Mn11CrO36 | -2.846 | -2.181 | +0.665 | 0.0 |
| agm005246954 | SO12 | -0.911 | -0.261 | +0.651 | 0.2243 |
| agm005255050 | CsEuSiS4 | -1.657 | -1.045 | +0.613 | 0.0 |
| agm005247792 | BaO10 | -1.081 | -0.480 | +0.601 | 0.2278 |
| agm003267909 | LiO8 | -0.897 | -0.296 | +0.601 | 0.0 |
| agm002259422 | GaCO5 | -1.586 | -0.990 | +0.596 | 0.8879 |
| agm005260897 | NaO8 | -0.728 | -0.156 | +0.572 | 0.0 |
| agm005289855 | Nd(PO5)2 | -2.295 | -1.724 | +0.571 | 0.0025 |
| agm005239502 | SrO10 | -1.070 | -0.506 | +0.564 | 0.1622 |
| agm005545034 | CoO4 | -0.817 | -0.253 | +0.564 | 0.0 |
| agm005247895 | Pr2Te(Mo3O23)2 | -1.465 | -0.904 | +0.561 | 0.27030000000000004 |

Full top-100 at `csv/worst_predictions.csv`.

## Top 10 best predictions (smallest |residual|)

| id | formula | DFT | ALIGNN | residual |
|---|---|---|---|---|
| agm002203879 | CaIn2Pd | -0.614 | -0.614 | -0.0000 |
| agm005277432 | BaSc3AgS6 | -2.099 | -2.099 | -0.0000 |
| agm003202464 | ScCo3B2 | -0.543 | -0.543 | -0.0000 |
| agm003189739 | AlCrRu2 | -0.363 | -0.363 | -0.0000 |
| agm005014713 | LaTbSe2S | -2.191 | -2.191 | +0.0000 |
| agm004612621 | Rb3Tb(SmTe3)2 | -1.398 | -1.398 | +0.0000 |
| agm003124897 | NaNbO2 | -2.538 | -2.538 | -0.0000 |
| agm003271864 | Sb2F7 | -2.702 | -2.702 | -0.0000 |
| agm004550458 | Li2Nd(CuP)2 | -0.771 | -0.771 | +0.0000 |
| agm005581613 | Pr3(Ge3Pd10)2 | -0.650 | -0.650 | -0.0000 |

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
- Cross-arm comparison with `eform_v2_opt/analysis/` is in the cross-arm section above; large Opt-vs-this disagreements are characterized in `csv/worst_predictions.csv`.
