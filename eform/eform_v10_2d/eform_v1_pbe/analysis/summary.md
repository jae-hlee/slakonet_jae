# v10 e_form analysis: PBE arm (mp_e_form_alignn)

Pretrained model trained on **MP/PBE** formation energy (eV/atom). Inference on the **Alexandria PBE 2D (Z<=65)** set, N = 87,903. Reference is `e_form` from Alexandria PBE.

## Headline metrics

| metric | value (eV/atom) |
|---|---|
| MAE | **0.1649** |
| RMSE | 0.2378 |
| ME (bias) | -0.1425 |
| Median \|err\| | 0.1183 |
| 90th-pct \|err\| | 0.3574 |
| 95th-pct \|err\| | 0.4709 |
| 99th-pct \|err\| | 0.8029 |
| Pearson r | 0.9772 |
| ALIGNN mean / median | -0.645 / -0.416 |
| DFT mean / median | -0.502 / -0.294 |

## Cumulative error distribution

Fraction of structures with |ALIGNN - DFT| below each threshold:

| threshold (eV/atom) | count | fraction |
|---|---|---|
| 0.005 | 2,280 | 2.59% |
| 0.010 | 4,608 | 5.24% |
| 0.020 | 9,031 | 10.27% |
| 0.050 | 21,557 | 24.52% |
| 0.100 | 38,641 | 43.96% |
| 0.200 | 62,267 | 70.84% |
| 0.300 | 74,994 | 85.31% |
| 0.500 | 84,134 | 95.71% |
| 1.000 | 87,456 | 99.49% |

## Stratified metrics (metal vs non-metal)

Split by DFT `band_gap_ind`: == 0 vs > 0.

| subset | N | MAE | RMSE | ME | med \|err\| | p90 \|err\| |
|---|---|---|---|---|---|---|
| all | 87,903 | 0.1649 | 0.2378 | -0.1425 | 0.1183 | 0.3574 |
| metal | 48,542 | 0.2019 | 0.2788 | -0.1892 | 0.1535 | 0.4176 |
| non-metal | 39,361 | 0.1192 | 0.1743 | -0.0849 | 0.0813 | 0.2718 |

## Bandgap-bin stratification

Five bins by DFT `band_gap_ind` (eV).

| bin | N | MAE | RMSE | ME | med \|err\| |
|---|---|---|---|---|---|
| metal (=0) | 48,542 | 0.2019 | 0.2788 | -0.1892 | 0.1535 |
| small (0,0.5] | 13,389 | 0.1620 | 0.2205 | -0.1466 | 0.1253 |
| med  (0.5,1.5] | 10,676 | 0.1200 | 0.1678 | -0.0973 | 0.0862 |
| wide (1.5,3.5] | 10,599 | 0.0836 | 0.1278 | -0.0378 | 0.0556 |
| ultra (>3.5) | 4,697 | 0.0759 | 0.1232 | +0.0131 | 0.0479 |

## Composition cardinality stratification

By number of distinct elements per formula.

| cardinality | N | MAE | RMSE | ME | med \|err\| |
|---|---|---|---|---|---|
| 1 elem | 300 | 0.4992 | 0.7671 | -0.4774 | 0.2899 |
| 2 elem | 43,598 | 0.1974 | 0.2840 | -0.1697 | 0.1381 |
| 3 elem | 43,147 | 0.1310 | 0.1711 | -0.1136 | 0.1038 |
| 4 elem | 853 | 0.0997 | 0.1362 | -0.0959 | 0.0730 |
| 5+ elem | 5 | 0.0285 | 0.0387 | -0.0067 | 0.0140 |

## Cross-arm comparison

On N = 87,903 records (positional pairing):

| metric | value |
|---|---|
| PBE - Opt mean (signed) | -0.1162 eV/atom |
| PBE - Opt median | -0.0632 eV/atom |
| mean \|PBE - Opt\| | 0.1564 eV/atom |
| max \|PBE - Opt\| | 3.1010 eV/atom |
| Pearson r (PBE vs Opt) | 0.9700 |
| fraction \|diff\| <= 0.05 | 32.0% |
| fraction \|diff\| <= 0.10 | 51.8% |
| fraction \|diff\| <= 0.50 | 95.7% |

Top 25 cross-arm disagreements (full table at `csv/cross_arm_top_disagreements.csv`):

| id | formula | PBE | Opt | PBE - Opt |
|---|---|---|---|---|
| agm2000000337 | P | +0.263 | +3.364 | -3.101 |
| agm2000000020 | As | +0.133 | +2.764 | -2.631 |
| agm2000000303 | N2 | +0.190 | +2.663 | -2.473 |
| agm2000045656 | YN | -0.170 | +1.834 | -2.004 |
| agm2000041696 | SbAs | +0.340 | +2.316 | -1.976 |
| agm2000045569 | TbN | -0.265 | +1.658 | -1.923 |
| agm2000045542 | SmN | -0.158 | +1.765 | -1.922 |
| agm2000045447 | NdN | -0.138 | +1.742 | -1.879 |
| agm2000047957 | BaS | -1.988 | -0.141 | -1.847 |
| agm2000046334 | CaO | -2.151 | -0.401 | -1.750 |
| agm2000045397 | LaN | -0.764 | +0.964 | -1.728 |
| agm2000026957 | LaN | -1.015 | +0.684 | -1.699 |
| agm2000050804 | Rh4S7 | -0.294 | +1.394 | -1.688 |
| agm2000006294 | LaN | -1.005 | +0.648 | -1.653 |
| agm2000031633 | LaN | -1.060 | +0.575 | -1.635 |
| agm2000031706 | BaO | -2.333 | -0.735 | -1.599 |
| agm2000045391 | LaN | -0.422 | +1.175 | -1.597 |
| agm2000006707 | BaO | -2.384 | -0.791 | -1.593 |
| agm2000126634 | Cu3Pd | +0.444 | +2.036 | -1.592 |
| agm2000046566 | SrO | -2.240 | -0.673 | -1.567 |
| agm2000027088 | CaO | -2.414 | -0.879 | -1.535 |
| agm2000042416 | CO2 | +0.804 | +2.306 | -1.501 |
| agm2000004565 | CS2 | -0.142 | +1.345 | -1.487 |
| agm2000045342 | GdN | -0.119 | +1.358 | -1.476 |
| agm2000026989 | ScN | -0.671 | +0.802 | -1.472 |


## Per-element MAE (top 15 worst, count >= 200)

| element | count | MAE | median \|err\| |
|---|---|---|---|
| Tc | 321 | 0.4364 | 0.2049 |
| Pr | 565 | 0.2998 | 0.1304 |
| Nd | 1,512 | 0.2701 | 0.1954 |
| Pm | 598 | 0.2662 | 0.1046 |
| N | 5,853 | 0.2642 | 0.2166 |
| Ru | 1,556 | 0.2565 | 0.1859 |
| Tb | 1,528 | 0.2507 | 0.1638 |
| Sm | 1,379 | 0.2502 | 0.1594 |
| P | 6,984 | 0.2431 | 0.2210 |
| La | 1,425 | 0.2349 | 0.1766 |
| Y | 1,471 | 0.2283 | 0.1381 |
| Rh | 1,515 | 0.2221 | 0.1721 |
| Ba | 1,582 | 0.2210 | 0.1925 |
| Mo | 1,677 | 0.2100 | 0.1455 |
| B | 1,315 | 0.2072 | 0.1598 |

Full per-element table at `csv/per_element_mae.csv`.

## Top 25 worst predictions

| id | formula | DFT | ALIGNN | residual | band_gap_ind |
|---|---|---|---|---|---|
| agm2000000337 | P | +3.522 | +0.263 | -3.258 | 1.9201000000000001 |
| agm2000000020 | As | +2.966 | +0.133 | -2.834 | 1.6837 |
| agm2000000123 | Cr | +2.997 | +0.422 | -2.575 | 0.0 |
| agm2000000457 | V | +3.294 | +0.776 | -2.518 | 0.0 |
| agm2000041696 | SbAs | +2.840 | +0.340 | -2.500 | 1.4188 |
| agm2000045444 | NdN | +1.872 | -0.577 | -2.449 | 0.0 |
| agm2000000281 | Nb | +3.451 | +1.041 | -2.409 | 0.0 |
| agm2000045569 | TbN | +2.062 | -0.265 | -2.327 | 0.0026000000000000003 |
| agm2000045341 | GdN | +1.748 | -0.577 | -2.325 | 0.0 |
| agm2000000485 | Zr | +3.263 | +0.941 | -2.322 | 0.0 |
| agm2000000421 | Tc | +2.803 | +0.521 | -2.282 | 0.0 |
| agm2000000435 | Ti | +2.999 | +0.736 | -2.263 | 0.0 |
| agm2000045656 | YN | +2.084 | -0.170 | -2.254 | 0.0165 |
| agm2000049955 | Mo3N5 | +2.789 | +0.612 | -2.177 | 0.0 |
| agm2000000260 | Mn | +2.707 | +0.547 | -2.160 | 0.0 |
| agm2000045542 | SmN | +1.996 | -0.158 | -2.154 | 0.022500000000000003 |
| agm2000045539 | SmN | +1.908 | -0.240 | -2.148 | 0.0 |
| agm2000124717 | Tc2Mo | +2.876 | +0.734 | -2.142 | 0.0 |
| agm2000000414 | Tb | +2.577 | +0.466 | -2.111 | 0.0 |
| agm2000124704 | TcMo2 | +2.980 | +0.878 | -2.102 | 0.0 |
| agm2000000267 | Mo | +3.048 | +0.947 | -2.100 | 0.0325 |
| agm2000045447 | NdN | +1.953 | -0.138 | -2.091 | 0.0806 |
| agm2000045653 | YN | +1.936 | -0.144 | -2.080 | 0.0 |
| agm2000045397 | LaN | +1.305 | -0.764 | -2.069 | 0.0 |
| agm2000045566 | TbN | +1.899 | -0.161 | -2.060 | 0.0 |

Full top-100 at `csv/worst_predictions.csv`.

## Top 10 best predictions (smallest |residual|)

| id | formula | DFT | ALIGNN | residual |
|---|---|---|---|---|
| agm2000022144 | AgBr | -0.289 | -0.289 | +0.0000 |
| agm2000104542 | FeCuSe2 | +0.070 | +0.070 | -0.0000 |
| agm2000065579 | SiCl3F | -1.926 | -1.926 | -0.0000 |
| agm2000131990 | Sc(CuCl3)2 | -1.553 | -1.553 | +0.0000 |
| agm2000028216 | NbF4 | -3.202 | -3.202 | -0.0000 |
| agm2000065468 | PdClF | -1.095 | -1.095 | -0.0000 |
| agm2000022767 | Cd3P2 | +0.208 | +0.208 | -0.0000 |
| agm2000097976 | Li3Ni2N3 | -0.142 | -0.142 | -0.0000 |
| agm2000128872 | PI4 | +0.004 | +0.004 | +0.0000 |
| agm2000032637 | BI3 | +0.012 | +0.012 | +0.0000 |

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
- 2D structures: one vacuum direction, OOD for 3D-bulk-trained ALIGNN.
- Cross-arm comparison with `eform_v2_opt/analysis/` is in the cross-arm section above; large Opt-vs-this disagreements are characterized in `csv/worst_predictions.csv`.
