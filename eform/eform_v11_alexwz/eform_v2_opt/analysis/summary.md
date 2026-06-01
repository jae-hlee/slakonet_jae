# v11 e_form analysis: Opt arm (jv_formation_energy_peratom_alignn)

Pretrained model trained on **JARVIS/OptB88vdW** formation energy (eV/atom). Inference on the **Alexandria PBE 3D hull (no Z filter)** set, N = 115,535. Reference is `e_form` from Alexandria PBE.

## Headline metrics

| metric | value (eV/atom) |
|---|---|
| MAE | **0.1022** |
| RMSE | 0.1535 |
| ME (bias) | +0.0521 |
| Median \|err\| | 0.0580 |
| 90th-pct \|err\| | 0.2703 |
| 95th-pct \|err\| | 0.3466 |
| 99th-pct \|err\| | 0.4884 |
| Pearson r | 0.9883 |
| ALIGNN mean / median | -1.077 / -0.795 |
| DFT mean / median | -1.129 / -0.805 |

## Cumulative error distribution

Fraction of structures with |ALIGNN - DFT| below each threshold:

| threshold (eV/atom) | count | fraction |
|---|---|---|
| 0.005 | 5,644 | 4.89% |
| 0.010 | 11,378 | 9.85% |
| 0.020 | 22,633 | 19.59% |
| 0.050 | 51,916 | 44.94% |
| 0.100 | 77,794 | 67.33% |
| 0.200 | 96,140 | 83.21% |
| 0.300 | 106,546 | 92.22% |
| 0.500 | 114,528 | 99.13% |
| 1.000 | 115,525 | 99.99% |

## Stratified metrics (metal vs non-metal)

Split by DFT `band_gap_ind`: == 0 vs > 0.

| subset | N | MAE | RMSE | ME | med \|err\| | p90 \|err\| |
|---|---|---|---|---|---|---|
| all | 115,535 | 0.1022 | 0.1535 | +0.0521 | 0.0580 | 0.2703 |
| metal | 79,214 | 0.0635 | 0.0950 | +0.0063 | 0.0425 | 0.1462 |
| non-metal | 36,321 | 0.1866 | 0.2352 | +0.1520 | 0.1640 | 0.3792 |

## Bandgap-bin stratification

Five bins by DFT `band_gap_ind` (eV).

| bin | N | MAE | RMSE | ME | med \|err\| |
|---|---|---|---|---|---|
| metal (=0) | 79,214 | 0.0635 | 0.0950 | +0.0063 | 0.0425 |
| small (0,0.5] | 6,286 | 0.1299 | 0.1764 | +0.0618 | 0.0908 |
| med  (0.5,1.5] | 9,350 | 0.1485 | 0.1962 | +0.0993 | 0.1058 |
| wide (1.5,3.5] | 13,659 | 0.2082 | 0.2539 | +0.1859 | 0.1968 |
| ultra (>3.5) | 7,026 | 0.2461 | 0.2856 | +0.2370 | 0.2449 |

## Composition cardinality stratification

By number of distinct elements per formula.

| cardinality | N | MAE | RMSE | ME | med \|err\| |
|---|---|---|---|---|---|
| 1 elem | 89 | 0.0310 | 0.0721 | +0.0132 | 0.0054 |
| 2 elem | 7,245 | 0.0937 | 0.1572 | +0.0406 | 0.0446 |
| 3 elem | 71,962 | 0.0854 | 0.1321 | +0.0324 | 0.0489 |
| 4 elem | 34,834 | 0.1322 | 0.1830 | +0.0865 | 0.0878 |
| 5+ elem | 1,405 | 0.2695 | 0.2949 | +0.2648 | 0.2744 |

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
| Cl | 4,333 | 0.2956 | 0.3010 |
| S | 8,093 | 0.2325 | 0.2104 |
| O | 17,469 | 0.2287 | 0.2153 |
| F | 5,450 | 0.2277 | 0.2117 |
| Cs | 5,251 | 0.1649 | 0.1264 |
| K | 5,422 | 0.1589 | 0.1261 |
| Cr | 1,602 | 0.1561 | 0.0980 |
| Rb | 5,211 | 0.1537 | 0.1169 |
| H | 5,180 | 0.1518 | 0.1038 |
| Na | 4,769 | 0.1498 | 0.1177 |
| Np | 2,766 | 0.1466 | 0.1189 |
| P | 7,108 | 0.1402 | 0.0812 |
| Mn | 3,181 | 0.1400 | 0.0806 |
| I | 3,082 | 0.1232 | 0.0745 |
| Eu | 1,554 | 0.1210 | 0.0967 |

Full per-element table at `csv/per_element_mae.csv`.

## Top 25 worst predictions

| id | formula | DFT | ALIGNN | residual | band_gap_ind |
|---|---|---|---|---|---|
| agm002116455 | CrTcO | -4.912 | -0.782 | +4.130 | 0.0 |
| agm004676761 | Rb3CrNi2F9 | -6.193 | -2.435 | +3.758 | 5.0398 |
| agm005277902 | CsCrNiF6 | -5.719 | -2.555 | +3.164 | 2.5599 |
| agm005102383 | SrCrCoF6 | -5.676 | -2.732 | +2.944 | 1.5861 |
| agm004968779 | Pr2BeCrO6 | -5.645 | -3.027 | +2.618 | 1.3966 |
| agm004969345 | Sr2ThCrO6 | -5.326 | -2.958 | +2.368 | 0.0 |
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
| agm003762590 | PaIO6 | -2.193 | -1.423 | +0.770 | 2.0548 |
| agm004884803 | PaS2ClO8 | -2.436 | -1.668 | +0.768 | 1.7515 |
| agm005546536 | LaCl5 | -2.039 | -1.275 | +0.765 | 2.9915000000000003 |
| agm005688321 | Fe2Cl3F | -1.769 | -1.013 | +0.756 | 3.8076 |
| agm005605090 | CoClF | -1.762 | -1.009 | +0.754 | 2.7801 |
| agm002170368 | SO2 | -1.750 | -0.999 | +0.751 | 2.843 |
| agm003247361 | SClO2F | -1.756 | -1.007 | +0.749 | 4.6555 |
| agm005704712 | NO2F | -1.050 | -0.302 | +0.748 | 4.3108 |

Full top-100 at `csv/worst_predictions.csv`.

## Top 10 best predictions (smallest |residual|)

| id | formula | DFT | ALIGNN | residual |
|---|---|---|---|---|
| agm001214744 | TbMgSn2 | -0.500 | -0.500 | -0.0000 |
| agm005931238 | ScCd3Pd4 | -0.576 | -0.576 | +0.0000 |
| agm005675513 | TbNbN2 | -1.574 | -1.574 | -0.0000 |
| agm005988812 | CaGe7Ir3 | -0.422 | -0.422 | +0.0000 |
| agm005576839 | Pa3As2Se3 | -1.230 | -1.230 | +0.0000 |
| agm003237509 | ErMoO4F | -3.058 | -3.058 | +0.0000 |
| agm003352377 | Ce3Ir7Rh2 | -0.627 | -0.627 | -0.0000 |
| agm004511105 | Ac2ZnGe3Au4 | -0.614 | -0.614 | -0.0000 |
| agm005975733 | Ga8Cu3Pd | -0.197 | -0.197 | +0.0000 |
| agm005686430 | Tm2MoOs3 | -0.328 | -0.328 | +0.0000 |

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
- Includes Z>65 elements (lanthanides through Yb and actinides) absent from v03.
- All structures hull-stable (e_above_hull = 0).
- Cross-arm comparison with `eform_v1_pbe/analysis/` is in the cross-arm section above; large PBE-vs-this disagreements are characterized in `csv/worst_predictions.csv`.
