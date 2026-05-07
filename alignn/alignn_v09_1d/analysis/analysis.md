# v09_1d: ALIGNN PBE bandgap deep analysis

Reference: DFT PBE `band_gap_ind` from Alexandria. Metal/gap split at 0.05 eV.

## Metal vs non-metal stratification

| subset | N | MAE | RMSE | ME (bias) | med \|err\| | p90 \|err\| | DFT mean | ALIGNN mean |
|---|---|---|---|---|---|---|---|---|
| all | 9,540 | 0.485 | 0.755 | -0.143 | 0.273 | 1.255 | 1.070 | 0.927 |
| DFT metals | 3,575 | 0.186 | 0.419 | +0.180 | 0.042 | 0.528 | 0.002 | 0.183 |
| DFT non-metals | 5,965 | 0.664 | 0.899 | -0.337 | 0.501 | 1.444 | 1.709 | 1.373 |

## Metal/gap classification (full set)

- Accuracy: 72.998%
- True negatives (DFT metal, ALIGNN metal): 1,929 (20.2%)
- False positives (DFT metal, ALIGNN gap):  1,646 (17.3%)
- False negatives (DFT gap, ALIGNN metal):  930 (9.7%)
- True positives (DFT gap, ALIGNN gap):     5,035 (52.8%)

**Failure-mode breakdown.** False positives (1,646) and false negatives (930) are roughly balanced.  Metals are predicted much more accurately (MAE 0.186) than non-metals (MAE 0.664); the model defaults toward small-gap predictions which works for metals but underestimates non-metal gaps.

## Worst predictions

Top 20 entries by absolute residual (full table at `csv/worst_predictions.csv`):

| id | formula | band_gap_ind | alignn_bandgap | residual | e_form |
| --- | --- | --- | --- | --- | --- |
| agm1000001795 | CF4 | 0.000 | 5.219 | 5.219 | -1.246 |
| agm1000018471 | LaF3 | 5.984 | 0.832 | -5.152 | -4.179 |
| agm1000014506 | BeO | 5.125 | 0.240 | -4.885 | -2.353 |
| agm1000008141 | BeO | 4.844 | 0.117 | -4.727 | -2.519 |
| agm1000002197 | NaF | 0.096 | 4.806 | 4.710 | -1.828 |
| agm1000011919 | BeO | 4.802 | 0.211 | -4.591 | -2.358 |
| agm1000008998 | KCl | 0.047 | 4.596 | 4.549 | -1.600 |
| agm1000009417 | BeO | 5.133 | 0.657 | -4.476 | -2.619 |
| agm1000017161 | BeO | 5.106 | 0.630 | -4.476 | -2.470 |
| agm1000010001 | BeO | 5.152 | 0.691 | -4.461 | -2.354 |

## Per-element MAE (worst 15)

Entries containing each element (compositions overlap; the same entry appears under each of its elements). Min count threshold applied.

| element | count | MAE (eV) | median \|err\| (eV) |
|---|---|---|---|
| C | 34 | 1.566 | 1.372 |
| Be | 205 | 1.008 | 0.709 |
| Si | 69 | 0.979 | 0.789 |
| F | 735 | 0.950 | 0.755 |
| B | 156 | 0.877 | 0.725 |
| Ge | 61 | 0.845 | 0.795 |
| Sn | 198 | 0.789 | 0.738 |
| Zn | 192 | 0.778 | 0.741 |
| Cl | 731 | 0.747 | 0.617 |
| La | 142 | 0.745 | 0.509 |
| Br | 163 | 0.741 | 0.651 |
| Sb | 119 | 0.729 | 0.555 |
| Cs | 77 | 0.712 | 0.683 |
| Al | 191 | 0.698 | 0.519 |
| Rb | 143 | 0.694 | 0.621 |

Full table at `csv/per_element_mae.csv`; bar chart at `plots/per_element_mae.png`.

## Error vs formation energy

Pearson correlation between |ALIGNN - DFT| and `e_form`: **-0.365**. Negative: less stable structures have smaller error (often because they trend metallic in DFT and ALIGNN agrees). Plot at `plots/error_vs_eform.png`.

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

Cross-method comparison vs SlakoNet on the same matched structures: `slakonet/slako_v09_1d/analysis/plots/sk_vs_alignn.png` and `slakonet/slako_v09_1d/analysis/plots/confusion_sk_vs_alignn.png`. Metrics in `slakonet/slako_v09_1d/analysis/csv/sk_vs_alignn_metrics.csv`. Cross-dataset roll-up at `slakonet/slakonet_comprehensive_analysis/csv/sk_vs_alignn_cross_dataset.csv`.
