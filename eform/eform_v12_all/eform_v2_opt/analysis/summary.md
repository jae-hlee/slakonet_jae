# v12 e_form analysis: Opt arm (jv_formation_energy_peratom_alignn)

Pretrained model trained on **JARVIS/OptB88vdW** formation energy (eV/atom). Inference on the **Alexandria PBE 3D full set (no filters, sharded)** set, N = 4,489,295. Reference is `e_form` from Alexandria PBE.

## Headline metrics

| metric | value (eV/atom) |
|---|---|
| MAE | **0.1449** |
| RMSE | 0.2222 |
| ME (bias) | -0.0861 |
| Median \|err\| | 0.0924 |
| 90th-pct \|err\| | 0.3285 |
| 95th-pct \|err\| | 0.4649 |
| 99th-pct \|err\| | 0.8455 |
| Pearson r | 0.9802 |
| ALIGNN mean / median | -0.339 / -0.213 |
| DFT mean / median | -0.253 / -0.145 |

## Cumulative error distribution

Fraction of structures with |ALIGNN - DFT| below each threshold:

| threshold (eV/atom) | count | fraction |
|---|---|---|
| 0.005 | 147,071 | 3.28% |
| 0.010 | 291,084 | 6.48% |
| 0.020 | 564,933 | 12.58% |
| 0.050 | 1,333,456 | 29.70% |
| 0.100 | 2,378,085 | 52.97% |
| 0.200 | 3,489,919 | 77.74% |
| 0.300 | 3,960,371 | 88.22% |
| 0.500 | 4,297,562 | 95.73% |
| 1.000 | 4,466,115 | 99.48% |

## Stratified metrics (metal vs non-metal)

Split by DFT `band_gap_ind`: == 0 vs > 0.

| subset | N | MAE | RMSE | ME | med \|err\| | p90 \|err\| |
|---|---|---|---|---|---|---|
| all | 4,489,295 | 0.1449 | 0.2222 | -0.0861 | 0.0924 | 0.3285 |
| metal | 4,095,214 | 0.1434 | 0.2240 | -0.1034 | 0.0893 | 0.3277 |
| non-metal | 394,081 | 0.1610 | 0.2026 | +0.0942 | 0.1333 | 0.3321 |

## Bandgap-bin stratification

Five bins by DFT `band_gap_ind` (eV).

| bin | N | MAE | RMSE | ME | med \|err\| |
|---|---|---|---|---|---|
| metal (=0) | 4,095,214 | 0.1434 | 0.2240 | -0.1034 | 0.0893 |
| small (0,0.5] | 127,374 | 0.1451 | 0.1854 | +0.0339 | 0.1190 |
| med  (0.5,1.5] | 111,430 | 0.1487 | 0.1893 | +0.0854 | 0.1177 |
| wide (1.5,3.5] | 118,206 | 0.1755 | 0.2177 | +0.1393 | 0.1528 |
| ultra (>3.5) | 37,071 | 0.2058 | 0.2440 | +0.1847 | 0.1894 |

## e_above_hull stratification

Four bins by DFT `e_above_hull` (eV/atom). On-hull bin is directly comparable to v11; off-hull entries are v12's novel coverage relative to v11.

| bin | N | MAE | RMSE | ME | med \|err\| |
|---|---|---|---|---|---|
| on hull (=0) | 115,535 | 0.1022 | 0.1535 | +0.0521 | 0.0580 |
| near (0,0.1] | 1,339,487 | 0.0815 | 0.1159 | -0.0023 | 0.0568 |
| off (0.1,0.5] | 1,866,463 | 0.1079 | 0.1363 | -0.0569 | 0.0907 |
| far-off (>0.5) | 1,167,810 | 0.2809 | 0.3774 | -0.2425 | 0.2130 |

## Composition cardinality stratification

By number of distinct elements per formula.

| cardinality | N | MAE | RMSE | ME | med \|err\| |
|---|---|---|---|---|---|
| 1 elem | 1,736 | 0.1824 | 0.3424 | -0.1152 | 0.0655 |
| 2 elem | 241,025 | 0.1374 | 0.2227 | -0.0824 | 0.0792 |
| 3 elem | 2,963,909 | 0.1347 | 0.2078 | -0.0882 | 0.0871 |
| 4 elem | 1,267,662 | 0.1697 | 0.2527 | -0.0843 | 0.1090 |
| 5+ elem | 14,963 | 0.1759 | 0.2156 | +0.1303 | 0.1555 |

## Cross-arm comparison

On N = 4,489,295 records (positional pairing):

| metric | value |
|---|---|
| PBE - Opt mean (signed) | -0.0686 eV/atom |
| PBE - Opt median | -0.0270 eV/atom |
| mean \|PBE - Opt\| | 0.1149 eV/atom |
| max \|PBE - Opt\| | 4.1266 eV/atom |
| Pearson r (PBE vs Opt) | 0.9810 |
| fraction \|diff\| <= 0.05 | 40.5% |
| fraction \|diff\| <= 0.10 | 62.8% |
| fraction \|diff\| <= 0.50 | 97.8% |

Top 25 cross-arm disagreements (full table at `csv/cross_arm_top_disagreements.csv`):

| id | formula | PBE | Opt | PBE - Opt |
|---|---|---|---|---|
| agm005267684 | N2 | +0.110 | +4.236 | -4.127 |
| agm003157165 | N2 | +1.197 | +4.326 | -3.129 |
| agm003157166 | N2 | +1.460 | +4.133 | -2.673 |
| agm004462534 | N2 | +0.268 | +2.825 | -2.557 |
| agm003157162 | H2 | +0.000 | +2.465 | -2.465 |
| agm004442760 | H2 | +0.015 | +2.464 | -2.449 |
| agm003157654 | H2 | +0.018 | +2.458 | -2.440 |
| agm004333769 | BeBIr2 | +0.183 | +2.621 | -2.439 |
| agm003157163 | H2 | +0.019 | +2.453 | -2.434 |
| agm005499563 | La2(SiIr)3 | -0.095 | +2.325 | -2.420 |
| agm002087570 | H2 | +0.036 | +2.454 | -2.418 |
| agm001750322 | Fe(NO)2 | +0.889 | +3.301 | -2.412 |
| agm003241359 | H2 | +0.020 | +2.425 | -2.405 |
| agm005221153 | H2 | +0.043 | +2.439 | -2.396 |
| agm003219999 | H2 | +0.026 | +2.412 | -2.386 |
| agm004442754 | H2 | +0.022 | +2.402 | -2.380 |
| agm004442764 | H2 | +0.028 | +2.350 | -2.323 |
| agm003157164 | H2 | +0.028 | +2.344 | -2.316 |
| agm004442762 | H2 | +0.079 | +2.387 | -2.308 |
| agm003279036 | H2 | +0.051 | +2.335 | -2.285 |
| agm004442755 | H2 | +0.026 | +2.303 | -2.277 |
| agm004462522 | N2 | +0.402 | +2.649 | -2.246 |
| agm003732156 | AsH3N | +0.515 | +2.756 | -2.242 |
| agm003708036 | PH3N | +0.710 | +2.947 | -2.237 |
| agm003157653 | H2 | +0.029 | +2.263 | -2.234 |


## Per-element MAE (top 15 worst, count >= 5000)

| element | count | MAE | median \|err\| |
|---|---|---|---|
| W | 124,552 | 0.3015 | 0.2122 |
| C | 114,613 | 0.2526 | 0.1880 |
| B | 98,607 | 0.2385 | 0.1701 |
| Re | 125,687 | 0.2260 | 0.1343 |
| Nb | 96,916 | 0.2258 | 0.1353 |
| Os | 150,392 | 0.2156 | 0.1335 |
| I | 148,588 | 0.2003 | 0.1373 |
| F | 139,632 | 0.1997 | 0.1574 |
| Cs | 125,848 | 0.1995 | 0.1395 |
| Ta | 137,413 | 0.1983 | 0.1118 |
| Rb | 152,916 | 0.1945 | 0.1296 |
| Hf | 109,783 | 0.1945 | 0.1125 |
| Mo | 133,079 | 0.1939 | 0.1086 |
| Cl | 197,166 | 0.1929 | 0.1710 |
| V | 104,116 | 0.1899 | 0.1183 |

Full per-element table at `csv/per_element_mae.csv`.

## Top 25 worst predictions

| id | formula | DFT | ALIGNN | residual | band_gap_ind |
|---|---|---|---|---|---|
| agm002116455 | CrTcO | -4.912 | -0.782 | +4.130 | 0.0 |
| agm004676761 | Rb3CrNi2F9 | -6.193 | -2.435 | +3.758 | 5.0398 |
| agm005277902 | CsCrNiF6 | -5.719 | -2.555 | +3.164 | 2.5599 |
| agm005102383 | SrCrCoF6 | -5.676 | -2.732 | +2.944 | 1.5861 |
| agm002471662 | CsW3F | +5.115 | +2.181 | -2.935 | 0.0 |
| agm002673240 | Mn2FeF | +2.728 | +0.031 | -2.697 | 0.0 |
| agm002560930 | Ni3WO | +3.586 | +0.958 | -2.628 | 0.0 |
| agm004968779 | Pr2BeCrO6 | -5.645 | -3.027 | +2.618 | 1.3966 |
| agm002563154 | SrMn3O | +3.889 | +1.295 | -2.593 | 0.0 |
| agm002718334 | Mn2FeO | +2.706 | +0.119 | -2.586 | 0.0 |
| agm002466312 | Ni3WF | +2.977 | +0.484 | -2.493 | 0.0 |
| agm002557137 | CaMn3O | +3.648 | +1.172 | -2.476 | 0.0 |
| agm002558592 | Ni3MoO | +3.391 | +1.018 | -2.373 | 0.0 |
| agm004969345 | Sr2ThCrO6 | -5.326 | -2.958 | +2.368 | 0.0 |
| agm002471913 | NiW3F | +2.647 | +0.304 | -2.343 | 0.0 |
| agm002562072 | NaW3O | +4.531 | +2.198 | -2.333 | 0.0 |
| agm003876239 | KMo3 | +4.436 | +2.114 | -2.322 | 0.0 |
| agm002468521 | NaW3F | +4.148 | +1.827 | -2.321 | 0.0 |
| agm004890438 | LiCrCu2F8 | -4.404 | -2.094 | +2.310 | 0.0 |
| agm002469581 | BaMn3F | +3.203 | +0.893 | -2.310 | 0.0 |
| agm002561956 | W3IO | +4.312 | +2.012 | -2.300 | 0.0 |
| agm002565456 | LaW3O | +3.999 | +1.706 | -2.294 | 0.0 |
| agm004911156 | Li2CeCrF8 | -5.571 | -3.296 | +2.275 | 0.0695 |
| agm002473029 | Ni3MoF | +2.780 | +0.533 | -2.247 | 0.0 |
| agm002473990 | W3IF | +3.812 | +1.585 | -2.227 | 0.0 |

Full top-100 at `csv/worst_predictions.csv`.

## Top 10 best predictions (smallest |residual|)

| id | formula | DFT | ALIGNN | residual |
|---|---|---|---|---|
| agm004620636 | Sm2Ho(ThSb2)3 | -1.038 | -1.038 | +0.0000 |
| agm005751841 | USI2 | -1.086 | -1.086 | +0.0000 |
| agm005812104 | LiNi14Ir | +0.007 | +0.007 | -0.0000 |
| agm004349310 | SrZr2Os | +0.655 | +0.655 | +0.0000 |
| agm001644701 | GaNi2MoC | -0.089 | -0.089 | +0.0000 |
| agm004542206 | La2Sc2HPb2 | -0.567 | -0.567 | -0.0000 |
| agm005951779 | Pu2BO8 | -2.599 | -2.599 | -0.0000 |
| agm001459919 | ZnHCI2 | +1.091 | +1.091 | +0.0000 |
| agm003809470 | RbC | +0.769 | +0.769 | +0.0000 |
| agm002728958 | KTeS2 | -0.326 | -0.326 | -0.0000 |

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
- `plots/hull_bin_mae.png` — MAE per `e_above_hull` bin (v12-only)
- `plots/composition_cardinality.png` — MAE by number of elements
- `plots/cross_arm_comparison.png` — PBE vs Opt arm: scatter, signed-diff histogram, |diff| CDF

## Caveats

- Reference is Alexandria PBE `e_form`; this arm trained on **JARVIS/OptB88vdW** e_form. Training functional differs from reference; the systematic ME is the PBE-vs-OptB88vdW functional shift, not pure model error. Subtract ME to recover a rough error scale.
- Full Alexandria PBE 3D set with NO filters (no hull, no Z<=65); includes ~4.49M structures across the entire stability range.
- Both arms now have all 100 shards complete (4,489,295 entries each); cross-arm uses positional pairing.
- Cross-arm comparison with `eform_v1_pbe/analysis/` is in the cross-arm section above; large PBE-vs-this disagreements are characterized in `csv/worst_predictions.csv`.
