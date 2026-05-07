# v10_2d: ALIGNN PBE bandgap deep analysis

Reference: DFT PBE `band_gap_ind` from Alexandria. Metal/gap split at 0.05 eV.

## Metal vs non-metal stratification

| subset | N | MAE | RMSE | ME (bias) | med \|err\| | p90 \|err\| | DFT mean | ALIGNN mean |
|---|---|---|---|---|---|---|---|---|
| all | 87,903 | 0.470 | 0.837 | +0.190 | 0.206 | 1.219 | 0.674 | 0.863 |
| DFT metals | 51,880 | 0.363 | 0.799 | +0.359 | 0.074 | 1.023 | 0.001 | 0.361 |
| DFT non-metals | 36,023 | 0.625 | 0.889 | -0.054 | 0.450 | 1.371 | 1.642 | 1.587 |

## Metal/gap classification (full set)

- Accuracy: 64.059%
- True negatives (DFT metal, ALIGNN metal): 22,564 (25.7%)
- False positives (DFT metal, ALIGNN gap):  29,316 (33.4%)
- False negatives (DFT gap, ALIGNN metal):  2,277 (2.6%)
- True positives (DFT gap, ALIGNN gap):     33,746 (38.4%)

**Failure-mode breakdown.** The dominant error mode is **false positives** (29,316 vs 2,277): ALIGNN predicts a gap where DFT says the structure is metallic.  Metal MAE (0.363) and non-metal MAE (0.625) are similar.

## Worst predictions

Top 20 entries by absolute residual (full table at `csv/worst_predictions.csv`):

| id | formula | band_gap_ind | alignn_bandgap | residual | e_form |
| --- | --- | --- | --- | --- | --- |
| agm2000128348 | BF4 | 0.000 | 8.099 | 8.099 | -2.474 |
| agm2000134425 | LiB2F8 | 0.000 | 7.941 | 7.941 | -2.919 |
| agm2000115204 | Li4F3 | 0.000 | 7.810 | 7.810 | -2.565 |
| agm2000134306 | NaB2F8 | 0.000 | 7.806 | 7.806 | -2.909 |
| agm2000134309 | RbB2F8 | 0.000 | 7.734 | 7.734 | -2.942 |
| agm2000113914 | Be2F5 | 0.000 | 7.681 | 7.681 | -2.945 |
| agm2000114564 | B3F11 | 0.000 | 7.680 | 7.680 | -2.658 |
| agm2000134300 | CsB2F8 | 0.000 | 7.561 | 7.561 | -2.954 |
| agm2000145160 | LiSiF6 | 0.000 | 7.508 | 7.508 | -2.958 |
| agm2000114340 | B3F10 | 0.000 | 7.237 | 7.237 | -2.858 |

## Per-element MAE (worst 15)

Entries containing each element (compositions overlap; the same entry appears under each of its elements). Min count threshold applied.

| element | count | MAE (eV) | median \|err\| (eV) |
|---|---|---|---|
| C | 442 | 1.087 | 0.688 |
| H | 2,250 | 0.865 | 0.431 |
| F | 7,050 | 0.854 | 0.526 |
| B | 1,315 | 0.843 | 0.609 |
| Cs | 1,624 | 0.812 | 0.473 |
| Cl | 10,850 | 0.792 | 0.497 |
| Ce | 1,236 | 0.778 | 0.265 |
| K | 2,723 | 0.750 | 0.389 |
| Be | 2,429 | 0.715 | 0.442 |
| Br | 10,812 | 0.708 | 0.455 |
| Rb | 2,331 | 0.694 | 0.374 |
| Li | 3,777 | 0.677 | 0.352 |
| Si | 1,118 | 0.676 | 0.451 |
| Sr | 1,648 | 0.674 | 0.365 |
| Ca | 1,810 | 0.672 | 0.370 |

Full table at `csv/per_element_mae.csv`; bar chart at `plots/per_element_mae.png`.

## Error vs formation energy

Pearson correlation between |ALIGNN - DFT| and `e_form`: **-0.344**. Negative: less stable structures have smaller error (often because they trend metallic in DFT and ALIGNN agrees). Plot at `plots/error_vs_eform.png`.

## Caveats

- Low-dimensional materials (1D / 2D) sit outside the 3D-bulk training distribution of `mp_gappbe_alignn`; expect roughly 3x worse MAE than the v11 Alexandria 3D bulk baseline.

## Files

- `plots/parity_all.png`, `plots/parity_metals.png`, `plots/parity_nonmetals.png` (stratified parity + residual panels)
- `csv/worst_predictions.csv` (top 20 by absolute residual)
- `csv/stratified_metrics.csv` (the all/metals/non-metals MAE table above)
- `csv/per_element_mae.csv`, `plots/per_element_mae.png`
- `plots/error_vs_eform.png`
- existing: `plots/parity.png`, `plots/confusion.png`, `plots/distribution.png`, `csv/metrics.csv`, `summary.md`

## See also

Cross-method comparison vs SlakoNet on the same matched structures: `slakonet/slako_v10_2d/analysis/plots/sk_vs_alignn.png` and `slakonet/slako_v10_2d/analysis/plots/confusion_sk_vs_alignn.png`. Metrics in `slakonet/slako_v10_2d/analysis/csv/sk_vs_alignn_metrics.csv`. Cross-dataset roll-up at `slakonet/slakonet_comprehensive_analysis/csv/sk_vs_alignn_cross_dataset.csv`.
