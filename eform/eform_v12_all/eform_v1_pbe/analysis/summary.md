# v12 e_form analysis: PBE arm (mp_e_form_alignn)

Pretrained model trained on **MP/PBE** formation energy (eV/atom). Inference on the **Alexandria PBE 3D full set (no filters, sharded)** set, N = 4,489,295. Reference is `e_form` from Alexandria PBE.

## Headline metrics

| metric | value (eV/atom) |
|---|---|
| MAE | **0.1710** |
| RMSE | 0.2960 |
| ME (bias) | -0.1547 |
| Median \|err\| | 0.0856 |
| 90th-pct \|err\| | 0.4372 |
| 95th-pct \|err\| | 0.6724 |
| 99th-pct \|err\| | 1.2007 |
| Pearson r | 0.9666 |
| ALIGNN mean / median | -0.408 / -0.246 |
| DFT mean / median | -0.253 / -0.145 |

## Cumulative error distribution

Fraction of structures with |ALIGNN - DFT| below each threshold:

| threshold (eV/atom) | count | fraction |
|---|---|---|
| 0.005 | 209,701 | 4.67% |
| 0.010 | 409,286 | 9.12% |
| 0.020 | 767,187 | 17.09% |
| 0.050 | 1,573,940 | 35.06% |
| 0.100 | 2,458,146 | 54.76% |
| 0.200 | 3,377,413 | 75.23% |
| 0.300 | 3,775,694 | 84.10% |
| 0.500 | 4,117,495 | 91.72% |
| 1.000 | 4,405,141 | 98.13% |

## Stratified metrics (metal vs non-metal)

Split by DFT `band_gap_ind`: == 0 vs > 0.

| subset | N | MAE | RMSE | ME | med \|err\| | p90 \|err\| |
|---|---|---|---|---|---|---|
| all | 4,489,295 | 0.1710 | 0.2960 | -0.1547 | 0.0856 | 0.4372 |
| metal | 4,095,214 | 0.1810 | 0.3077 | -0.1658 | 0.0929 | 0.4648 |
| non-metal | 394,081 | 0.0670 | 0.1180 | -0.0391 | 0.0350 | 0.1633 |

## Bandgap-bin stratification

Five bins by DFT `band_gap_ind` (eV).

| bin | N | MAE | RMSE | ME | med \|err\| |
|---|---|---|---|---|---|
| metal (=0) | 4,095,214 | 0.1810 | 0.3077 | -0.1658 | 0.0929 |
| small (0,0.5] | 127,374 | 0.1034 | 0.1631 | -0.0803 | 0.0650 |
| med  (0.5,1.5] | 111,430 | 0.0646 | 0.1085 | -0.0348 | 0.0369 |
| wide (1.5,3.5] | 118,206 | 0.0416 | 0.0738 | -0.0102 | 0.0239 |
| ultra (>3.5) | 37,071 | 0.0298 | 0.0620 | -0.0031 | 0.0176 |

## e_above_hull stratification

Four bins by DFT `e_above_hull` (eV/atom). On-hull bin is directly comparable to v11; off-hull entries are v12's novel coverage relative to v11.

| bin | N | MAE | RMSE | ME | med \|err\| |
|---|---|---|---|---|---|
| on hull (=0) | 115,535 | 0.0356 | 0.0667 | +0.0110 | 0.0192 |
| near (0,0.1] | 1,339,487 | 0.0424 | 0.0621 | -0.0201 | 0.0299 |
| off (0.1,0.5] | 1,866,463 | 0.1135 | 0.1406 | -0.0998 | 0.0981 |
| far-off (>0.5) | 1,167,810 | 0.4239 | 0.5479 | -0.4134 | 0.3394 |

## Composition cardinality stratification

By number of distinct elements per formula.

| cardinality | N | MAE | RMSE | ME | med \|err\| |
|---|---|---|---|---|---|
| 1 elem | 1,736 | 0.2280 | 0.4576 | -0.1897 | 0.0645 |
| 2 elem | 241,025 | 0.1547 | 0.2754 | -0.1324 | 0.0799 |
| 3 elem | 2,963,909 | 0.1607 | 0.2716 | -0.1449 | 0.0904 |
| 4 elem | 1,267,662 | 0.1995 | 0.3505 | -0.1829 | 0.0759 |
| 5+ elem | 14,963 | 0.0669 | 0.1066 | -0.0568 | 0.0278 |

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
| W | 124,552 | 0.3284 | 0.1844 |
| N | 191,521 | 0.3221 | 0.1899 |
| Re | 125,687 | 0.3166 | 0.1735 |
| C | 114,613 | 0.3159 | 0.1802 |
| B | 98,607 | 0.3026 | 0.1774 |
| I | 148,588 | 0.2893 | 0.1715 |
| Os | 150,392 | 0.2824 | 0.1417 |
| Mo | 133,079 | 0.2797 | 0.1588 |
| Ta | 137,413 | 0.2773 | 0.1575 |
| Nb | 96,916 | 0.2765 | 0.1328 |
| Hf | 109,783 | 0.2721 | 0.1366 |
| Tc | 148,676 | 0.2585 | 0.1404 |
| P | 207,281 | 0.2558 | 0.1481 |
| F | 139,632 | 0.2458 | 0.1222 |
| Cr | 135,259 | 0.2385 | 0.1305 |

Full per-element table at `csv/per_element_mae.csv`.

## Top 25 worst predictions

| id | formula | DFT | ALIGNN | residual | band_gap_ind |
|---|---|---|---|---|---|
| agm005267684 | N2 | +5.202 | +0.110 | -5.092 | 3.0083 |
| agm002116455 | CrTcO | -4.912 | -0.711 | +4.201 | 0.0 |
| agm004676761 | Rb3CrNi2F9 | -6.193 | -2.604 | +3.589 | 5.0398 |
| agm002374568 | Np2InPb | +3.955 | +0.473 | -3.482 | 0.0 |
| agm002471662 | CsW3F | +5.115 | +1.685 | -3.430 | 0.0 |
| agm004151000 | SrOs3 | +4.826 | +1.521 | -3.305 | 0.0 |
| agm003157165 | N2 | +4.491 | +1.197 | -3.294 | 0.0 |
| agm003888222 | BaOs3 | +4.834 | +1.591 | -3.243 | 0.0 |
| agm002469581 | BaMn3F | +3.203 | +0.054 | -3.149 | 0.0 |
| agm004420770 | YRe3 | +4.375 | +1.307 | -3.068 | 0.0 |
| agm003815649 | YOs3 | +4.002 | +0.999 | -3.002 | 0.0 |
| agm004119708 | YW3 | +4.371 | +1.369 | -3.002 | 0.0 |
| agm004424906 | CdRe3 | +4.552 | +1.556 | -2.996 | 0.0 |
| agm002673240 | Mn2FeF | +2.728 | -0.230 | -2.958 | 0.0 |
| agm003793858 | PbW3 | +4.505 | +1.578 | -2.927 | 0.0 |
| agm005277902 | CsCrNiF6 | -5.719 | -2.794 | +2.925 | 2.5599 |
| agm001471123 | RbOs2CN | +5.349 | +2.437 | -2.912 | 0.0 |
| agm001751163 | ScNiNO2 | +2.664 | -0.218 | -2.882 | 0.0 |
| agm004431224 | InRe3 | +4.362 | +1.482 | -2.880 | 0.0 |
| agm004102560 | Re3Pb | +4.526 | +1.646 | -2.880 | 0.0 |
| agm004462534 | N2 | +3.130 | +0.268 | -2.862 | 0.0 |
| agm004341281 | LaRe3 | +4.158 | +1.303 | -2.855 | 0.0 |
| agm004176748 | TlRe3 | +4.583 | +1.734 | -2.850 | 0.0 |
| agm002473990 | W3IF | +3.812 | +0.976 | -2.835 | 0.0 |
| agm003821515 | CdW3 | +4.399 | +1.565 | -2.835 | 0.0 |

Full top-100 at `csv/worst_predictions.csv`.

## Top 10 best predictions (smallest |residual|)

| id | formula | DFT | ALIGNN | residual |
|---|---|---|---|---|
| agm005036827 | AcTbNdTe3 | -1.300 | -1.300 | -0.0000 |
| agm005971299 | Ba3GaI7 | -1.492 | -1.492 | +0.0000 |
| agm004641920 | Na3Pr(NdS3)2 | -2.121 | -2.121 | -0.0000 |
| agm006031497 | Th(TlTe)4 | -0.636 | -0.636 | +0.0000 |
| agm005851918 | PmY2Ho9 | +0.024 | +0.024 | +0.0000 |
| agm002641927 | CaBBr2 | -0.010 | -0.010 | -0.0000 |
| agm002287084 | Sr3MgNi2 | +0.109 | +0.109 | -0.0000 |
| agm002226213 | LuNi2Sb | -0.505 | -0.505 | +0.0000 |
| agm001244049 | Th2ZnCo | -0.074 | -0.074 | +0.0000 |
| agm006035383 | Pm5YTl4 | -0.294 | -0.294 | +0.0000 |

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

- Reference is Alexandria PBE `e_form`; this arm trained on **MP/PBE** e_form. Training and reference functionals match.
- Full Alexandria PBE 3D set with NO filters (no hull, no Z<=65); includes ~4.49M structures across the entire stability range.
- Opt arm is missing shards 2 and 3 (~90k entries); cross-arm uses mat_id intersection rather than positional pairing.
- Cross-arm comparison with `eform_v2_opt/analysis/` is in the cross-arm section above; large Opt-vs-this disagreements are characterized in `csv/worst_predictions.csv`.
