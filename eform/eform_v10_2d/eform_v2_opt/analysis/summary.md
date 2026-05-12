# v10 e_form analysis: Opt arm (jv_formation_energy_peratom_alignn)

Pretrained model trained on **JARVIS/OptB88vdW** formation energy (eV/atom). Inference on the **Alexandria PBE 2D (Z<=65)** set, N = 87,903. Reference is `e_form` from Alexandria PBE.

## Headline metrics

| metric | value (eV/atom) |
|---|---|
| MAE | **0.2013** |
| RMSE | 0.2671 |
| ME (bias) | -0.0263 |
| Median \|err\| | 0.1582 |
| 90th-pct \|err\| | 0.4173 |
| 95th-pct \|err\| | 0.5276 |
| 99th-pct \|err\| | 0.8232 |
| Pearson r | 0.9550 |
| ALIGNN mean / median | -0.528 / -0.321 |
| DFT mean / median | -0.502 / -0.294 |

## Cumulative error distribution

Fraction of structures with |ALIGNN - DFT| below each threshold:

| threshold (eV/atom) | count | fraction |
|---|---|---|
| 0.005 | 1,386 | 1.58% |
| 0.010 | 2,826 | 3.21% |
| 0.020 | 5,699 | 6.48% |
| 0.050 | 14,375 | 16.35% |
| 0.100 | 28,588 | 32.52% |
| 0.200 | 53,129 | 60.44% |
| 0.300 | 68,836 | 78.31% |
| 0.500 | 82,683 | 94.06% |
| 1.000 | 87,519 | 99.56% |

## Stratified metrics (metal vs non-metal)

Split by DFT `band_gap_ind`: == 0 vs > 0.

| subset | N | MAE | RMSE | ME | med \|err\| | p90 \|err\| |
|---|---|---|---|---|---|---|
| all | 87,903 | 0.2013 | 0.2671 | -0.0263 | 0.1582 | 0.4173 |
| metal | 48,542 | 0.2023 | 0.2697 | -0.1120 | 0.1608 | 0.4082 |
| non-metal | 39,361 | 0.2001 | 0.2638 | +0.0792 | 0.1547 | 0.4278 |

## Bandgap-bin stratification

Five bins by DFT `band_gap_ind` (eV).

| bin | N | MAE | RMSE | ME | med \|err\| |
|---|---|---|---|---|---|
| metal (=0) | 48,542 | 0.2023 | 0.2697 | -0.1120 | 0.1608 |
| small (0,0.5] | 13,389 | 0.1916 | 0.2443 | -0.0106 | 0.1569 |
| med  (0.5,1.5] | 10,676 | 0.1784 | 0.2409 | +0.0540 | 0.1338 |
| wide (1.5,3.5] | 10,599 | 0.2056 | 0.2757 | +0.1492 | 0.1544 |
| ultra (>3.5) | 4,697 | 0.2613 | 0.3313 | +0.2348 | 0.2122 |

## Composition cardinality stratification

By number of distinct elements per formula.

| cardinality | N | MAE | RMSE | ME | med \|err\| |
|---|---|---|---|---|---|
| 1 elem | 300 | 0.4646 | 0.6712 | -0.3563 | 0.3036 |
| 2 elem | 43,598 | 0.2329 | 0.3081 | -0.0342 | 0.1812 |
| 3 elem | 43,147 | 0.1685 | 0.2136 | -0.0184 | 0.1402 |
| 4 elem | 853 | 0.1503 | 0.1826 | +0.0850 | 0.1329 |
| 5+ elem | 5 | 0.2145 | 0.2410 | +0.2145 | 0.1953 |

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
| Tc | 321 | 0.4466 | 0.2964 |
| Cl | 10,850 | 0.3139 | 0.2994 |
| Pr | 565 | 0.2980 | 0.2006 |
| Pm | 598 | 0.2912 | 0.1958 |
| C | 442 | 0.2907 | 0.2255 |
| Eu | 876 | 0.2854 | 0.2597 |
| Sm | 1,379 | 0.2651 | 0.2104 |
| La | 1,425 | 0.2649 | 0.2212 |
| Tb | 1,528 | 0.2638 | 0.2104 |
| Nd | 1,512 | 0.2620 | 0.2149 |
| B | 1,315 | 0.2572 | 0.2104 |
| Mo | 1,677 | 0.2554 | 0.1890 |
| Y | 1,471 | 0.2546 | 0.2033 |
| P | 6,984 | 0.2493 | 0.2293 |
| Zr | 1,449 | 0.2372 | 0.1764 |

Full per-element table at `csv/per_element_mae.csv`.

## Top 25 worst predictions

| id | formula | DFT | ALIGNN | residual | band_gap_ind |
|---|---|---|---|---|---|
| agm2000000303 | N2 | +0.000 | +2.663 | +2.663 | 0.0 |
| agm2000000457 | V | +3.294 | +1.037 | -2.257 | 0.0 |
| agm2000000435 | Ti | +2.999 | +0.895 | -2.104 | 0.0 |
| agm2000000281 | Nb | +3.451 | +1.432 | -2.018 | 0.0 |
| agm2000000381 | Sm | +2.446 | +0.504 | -1.942 | 0.0 |
| agm2000000414 | Tb | +2.577 | +0.645 | -1.932 | 0.0 |
| agm2000000288 | Nd | +2.340 | +0.438 | -1.902 | 0.0 |
| agm2000000471 | Y | +2.585 | +0.707 | -1.878 | 0.0 |
| agm2000000153 | Fe | +2.599 | +0.753 | -1.846 | 0.0 |
| agm2000000123 | Cr | +2.997 | +1.167 | -1.830 | 0.0 |
| agm2000000421 | Tc | +2.803 | +0.988 | -1.815 | 0.0 |
| agm2000000346 | Pr | +2.279 | +0.484 | -1.795 | 0.0 |
| agm2000000331 | Pm | +2.405 | +0.646 | -1.759 | 0.0 |
| agm2000113752 | Ti3Nb2 | +2.395 | +0.647 | -1.748 | 0.0 |
| agm2000000260 | Mn | +2.707 | +1.008 | -1.699 | 0.0 |
| agm2000113744 | Ti2V3 | +2.389 | +0.694 | -1.696 | 0.0 |
| agm2000113781 | Nb2V3 | +2.545 | +0.870 | -1.676 | 0.0 |
| agm2000113786 | V3Tc2 | +2.139 | +0.464 | -1.675 | 0.0 |
| agm2000000485 | Zr | +3.263 | +1.599 | -1.664 | 0.0 |
| agm2000112287 | Cr2Tc3 | +2.448 | +0.788 | -1.660 | 0.0 |
| agm2000130273 | Tc6Mo | +2.239 | +0.624 | -1.615 | 0.0 |
| agm2000115581 | Zr3Ti4 | +2.350 | +0.750 | -1.600 | 0.0 |
| agm2000130030 | CrTc6 | +2.183 | +0.598 | -1.585 | 0.0 |
| agm2000000058 | Be | +2.104 | +0.527 | -1.577 | 0.0 |
| agm2000000098 | Ce | +1.864 | +0.289 | -1.575 | 0.0 |

Full top-100 at `csv/worst_predictions.csv`.

## Top 10 best predictions (smallest |residual|)

| id | formula | DFT | ALIGNN | residual |
|---|---|---|---|---|
| agm2000080925 | Si3(TeSe2)2 | -0.016 | -0.016 | +0.0000 |
| agm2000010052 | In2Te3 | +0.007 | +0.007 | +0.0000 |
| agm2000101130 | CeSBr2 | -1.685 | -1.685 | -0.0000 |
| agm2000132664 | Pr(NdI3)2 | -1.243 | -1.243 | +0.0000 |
| agm2000061927 | SiIN | +0.549 | +0.549 | +0.0000 |
| agm2000134240 | Sm(YI3)2 | -1.174 | -1.174 | +0.0000 |
| agm2000090113 | MnCo2Se3 | +0.040 | +0.040 | -0.0000 |
| agm2000068558 | Y(CuS)3 | -0.908 | -0.908 | +0.0000 |
| agm2000106842 | Zn2CuN2 | +0.351 | +0.351 | -0.0000 |
| agm2000067058 | Cr3AgN3 | -0.170 | -0.170 | +0.0000 |

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

- Reference is Alexandria PBE `e_form`; this arm trained on **JARVIS/OptB88vdW** e_form. Training functional differs from reference; the systematic ME is the PBE-vs-OptB88vdW functional shift, not pure model error. Subtract ME to recover a rough error scale.
- 2D structures: one vacuum direction, OOD for 3D-bulk-trained ALIGNN.
- Cross-arm comparison with `eform_v1_pbe/analysis/` is in the cross-arm section above; large PBE-vs-this disagreements are characterized in `csv/worst_predictions.csv`.
