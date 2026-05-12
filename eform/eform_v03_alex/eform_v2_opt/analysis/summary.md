# v03 e_form analysis: Opt arm (jv_formation_energy_peratom_alignn)

Pretrained model trained on **JARVIS/OptB88vdW** formation energy (eV/atom). Inference on the **Alexandria PBE 3D hull (Z<=65)** set, N = 48,764. Reference is `e_form` from Alexandria PBE.

## Headline metrics

| metric | value (eV/atom) |
|---|---|
| MAE | **0.1174** |
| RMSE | 0.1757 |
| ME (bias) | +0.0741 |
| Median \|err\| | 0.0645 |
| 90th-pct \|err\| | 0.3121 |
| 95th-pct \|err\| | 0.3830 |
| 99th-pct \|err\| | 0.5242 |
| Pearson r | 0.9869 |
| ALIGNN mean / median | -1.146 / -0.849 |
| DFT mean / median | -1.220 / -0.878 |

## Cumulative error distribution

Fraction of structures with |ALIGNN - DFT| below each threshold:

| threshold (eV/atom) | count | fraction |
|---|---|---|
| 0.005 | 2,311 | 4.74% |
| 0.010 | 4,594 | 9.42% |
| 0.020 | 8,963 | 18.38% |
| 0.050 | 20,377 | 41.79% |
| 0.100 | 30,659 | 62.87% |
| 0.200 | 38,195 | 78.33% |
| 0.300 | 43,396 | 88.99% |
| 0.500 | 48,079 | 98.60% |
| 1.000 | 48,755 | 99.98% |

## Stratified metrics (metal vs non-metal)

Split by DFT `band_gap_ind`: == 0 vs > 0.

| subset | N | MAE | RMSE | ME | med \|err\| | p90 \|err\| |
|---|---|---|---|---|---|---|
| all | 48,764 | 0.1174 | 0.1757 | +0.0741 | 0.0645 | 0.3121 |
| metal | 28,420 | 0.0618 | 0.0975 | +0.0102 | 0.0403 | 0.1393 |
| non-metal | 20,344 | 0.1952 | 0.2465 | +0.1633 | 0.1719 | 0.3905 |

## Bandgap-bin stratification

Five bins by DFT `band_gap_ind` (eV).

| bin | N | MAE | RMSE | ME | med \|err\| |
|---|---|---|---|---|---|
| metal (=0) | 28,420 | 0.0618 | 0.0975 | +0.0102 | 0.0403 |
| small (0,0.5] | 2,783 | 0.1323 | 0.1846 | +0.0756 | 0.0895 |
| med  (0.5,1.5] | 4,910 | 0.1474 | 0.2001 | +0.0930 | 0.0987 |
| wide (1.5,3.5] | 7,990 | 0.2142 | 0.2630 | +0.1910 | 0.2042 |
| ultra (>3.5) | 4,661 | 0.2508 | 0.2898 | +0.2423 | 0.2484 |

## Composition cardinality stratification

By number of distinct elements per formula.

| cardinality | N | MAE | RMSE | ME | med \|err\| |
|---|---|---|---|---|---|
| 1 elem | 65 | 0.0249 | 0.0561 | +0.0230 | 0.0056 |
| 2 elem | 3,858 | 0.1050 | 0.1761 | +0.0581 | 0.0462 |
| 3 elem | 29,803 | 0.0967 | 0.1495 | +0.0476 | 0.0536 |
| 4 elem | 13,955 | 0.1529 | 0.2107 | +0.1196 | 0.1049 |
| 5+ elem | 1,083 | 0.2819 | 0.3042 | +0.2781 | 0.2845 |

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
| Cl | 2,498 | 0.2922 | 0.2938 |
| S | 3,883 | 0.2556 | 0.2315 |
| O | 9,576 | 0.2455 | 0.2366 |
| F | 3,445 | 0.2271 | 0.2066 |
| Cs | 2,924 | 0.1739 | 0.1369 |
| Cr | 1,076 | 0.1730 | 0.1104 |
| H | 3,299 | 0.1690 | 0.1148 |
| K | 3,138 | 0.1668 | 0.1326 |
| Na | 2,779 | 0.1627 | 0.1331 |
| Rb | 2,879 | 0.1615 | 0.1223 |
| P | 4,166 | 0.1608 | 0.0967 |
| Mn | 1,981 | 0.1605 | 0.0924 |
| Fe | 1,889 | 0.1526 | 0.0602 |
| N | 2,725 | 0.1350 | 0.0791 |
| Nb | 1,450 | 0.1344 | 0.0836 |

Full per-element table at `csv/per_element_mae.csv`.

## Top 25 worst predictions

| id | formula | DFT | ALIGNN | residual | band_gap_ind |
|---|---|---|---|---|---|
| agm002116455 | CrTcO | -4.912 | -0.782 | +4.130 | 0.0 |
| agm004676761 | Rb3CrNi2F9 | -6.193 | -2.435 | +3.758 | 5.0398 |
| agm005277902 | CsCrNiF6 | -5.719 | -2.555 | +3.164 | 2.5599 |
| agm005102383 | SrCrCoF6 | -5.676 | -2.732 | +2.944 | 1.5861 |
| agm004968779 | Pr2BeCrO6 | -5.645 | -3.027 | +2.618 | 1.3966 |
| agm004890438 | LiCrCu2F8 | -4.404 | -2.094 | +2.310 | 0.0 |
| agm004911156 | Li2CeCrF8 | -5.571 | -3.296 | +2.275 | 0.0695 |
| agm003235847 | Li4Cr3Fe2Sb3O16 | -3.007 | -1.791 | +1.217 | 0.7417 |
| agm003211577 | N2O | -0.437 | +0.567 | +1.004 | 6.195 |
| agm005742851 | ClO3F | -0.829 | +0.033 | +0.862 | 1.9281000000000001 |
| agm003232173 | ClO2 | -0.551 | +0.295 | +0.847 | 0.9586 |
| agm005410664 | Eu(MnGe)2 | -0.372 | -1.208 | -0.836 | 0.0 |
| agm002169632 | ClO2F | -0.874 | -0.043 | +0.831 | 3.6203000000000003 |
| agm005279378 | NO2 | -0.810 | -0.009 | +0.801 | 2.8135 |
| agm003211534 | EuVO4 | -2.830 | -3.619 | -0.789 | 0.0 |
| agm005543511 | NF3 | -0.913 | -0.133 | +0.780 | 6.0843 |
| agm005546536 | LaCl5 | -2.039 | -1.275 | +0.765 | 2.9915000000000003 |
| agm005688321 | Fe2Cl3F | -1.769 | -1.013 | +0.756 | 3.8076 |
| agm005605090 | CoClF | -1.762 | -1.009 | +0.754 | 2.7801 |
| agm002170368 | SO2 | -1.750 | -0.999 | +0.751 | 2.843 |
| agm003247361 | SClO2F | -1.756 | -1.007 | +0.749 | 4.6555 |
| agm005704712 | NO2F | -1.050 | -0.302 | +0.748 | 4.3108 |
| agm005234725 | SO5 | -1.368 | -0.622 | +0.746 | 1.8339 |
| agm003231566 | FeSO4 | -2.035 | -1.291 | +0.745 | 4.1796 |
| agm003242462 | Fe2(SO4)3 | -2.003 | -1.272 | +0.731 | 2.287 |

Full top-100 at `csv/worst_predictions.csv`.

## Top 10 best predictions (smallest |residual|)

| id | formula | DFT | ALIGNN | residual |
|---|---|---|---|---|
| agm001214744 | TbMgSn2 | -0.500 | -0.500 | -0.0000 |
| agm005931238 | ScCd3Pd4 | -0.576 | -0.576 | +0.0000 |
| agm005675513 | TbNbN2 | -1.574 | -1.574 | -0.0000 |
| agm005975733 | Ga8Cu3Pd | -0.197 | -0.197 | +0.0000 |
| agm003241431 | Co2(GeSe)3 | -0.357 | -0.357 | -0.0000 |
| agm004509989 | Sm2ZnNi3Ge4 | -0.654 | -0.654 | -0.0000 |
| agm004810688 | Nd4GeTe2As | -1.397 | -1.397 | -0.0000 |
| agm006041961 | Pm4TeP3 | -1.594 | -1.594 | -0.0000 |
| agm004820818 | Ce4PSe2S | -1.901 | -1.901 | +0.0000 |
| agm003221352 | Pr(P3Ru)4 | -0.657 | -0.657 | -0.0000 |

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
- Cross-arm comparison with `eform_v1_pbe/analysis/` is in the cross-arm section above; large PBE-vs-this disagreements are characterized in `csv/worst_predictions.csv`.
