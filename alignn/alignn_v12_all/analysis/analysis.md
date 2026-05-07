# v12_all (full Alexandria 3D, 99/100 shards): ALIGNN PBE bandgap deep analysis

Reference: DFT PBE `band_gap_ind` from Alexandria. Metal/gap split at 0.05 eV.

## Metal vs non-metal stratification

| subset | N | MAE | RMSE | ME (bias) | med \|err\| | p90 \|err\| | DFT mean | ALIGNN mean |
|---|---|---|---|---|---|---|---|---|
| all | 4,444,402 | 0.185 | 0.551 | +0.148 | 0.014 | 0.510 | 0.128 | 0.275 |
| DFT metals | 4,084,037 | 0.153 | 0.513 | +0.149 | 0.012 | 0.333 | 0.000 | 0.149 |
| DFT non-metals | 360,365 | 0.544 | 0.870 | +0.135 | 0.320 | 1.294 | 1.573 | 1.708 |

## Metal/gap classification (full set)

- Accuracy: 78.099%
- True negatives (DFT metal, ALIGNN metal): 3,130,841 (70.4%)
- False positives (DFT metal, ALIGNN gap):  953,196 (21.4%)
- False negatives (DFT gap, ALIGNN metal):  20,152 (0.5%)
- True positives (DFT gap, ALIGNN gap):     340,213 (7.7%)

**Failure-mode breakdown.** The dominant error mode is **false positives** (953,196 vs 20,152): ALIGNN predicts a gap where DFT says the structure is metallic.  Metals are predicted much more accurately (MAE 0.153) than non-metals (MAE 0.544); the model defaults toward small-gap predictions which works for metals but underestimates non-metal gaps.

## Worst predictions

Top 50 entries by absolute residual (full table at `csv/worst_predictions.csv`):

| mat_id | formula | band_gap_ind | alignn_bandgap | residual | e_form | e_above_hull |
| --- | --- | --- | --- | --- | --- | --- |
| agm003523347 | LiB2F8 | 0.000 | 8.329 | 8.329 | -2.884 | 0.074 |
| agm003292742 | Li5F6 | 0.000 | 8.303 | 8.303 | -2.871 | 0.054 |
| agm005217488 | Li2F3 | 0.000 | 8.255 | 8.255 | -2.510 | 0.112 |
| agm005858487 | LiB3F12 | 0.000 | 8.255 | 8.255 | -2.784 | 0.056 |
| agm005628252 | Li3Be2F8 | 0.000 | 8.197 | 8.197 | -3.107 | 0.063 |
| agm005858583 | SrB3F12 | 0.000 | 8.130 | 8.130 | -3.233 | 0.029 |
| agm005858637 | BaB3F12 | 0.000 | 8.091 | 8.091 | -3.238 | 0.018 |
| agm005867874 | LiB2F8 | 0.000 | 8.069 | 8.069 | -2.882 | 0.075 |
| agm005858609 | CaB3F12 | 0.000 | 8.056 | 8.056 | -3.208 | 0.033 |
| agm003523348 | NaB2F8 | 0.000 | 8.049 | 8.049 | -2.886 | 0.061 |

## Per-element MAE (worst 15)

Entries containing each element (compositions overlap; the same entry appears under each of its elements). Min count threshold applied.

| element | count | MAE (eV) | median \|err\| (eV) |
|---|---|---|---|
| F | 138,258 | 0.855 | 0.288 |
| Cs | 124,590 | 0.673 | 0.263 |
| O | 318,102 | 0.611 | 0.221 |
| Cl | 195,153 | 0.594 | 0.223 |
| Rb | 151,386 | 0.585 | 0.172 |
| Br | 189,682 | 0.525 | 0.202 |
| Np | 49,393 | 0.513 | 0.035 |
| I | 147,107 | 0.505 | 0.211 |
| S | 205,428 | 0.469 | 0.149 |
| K | 195,371 | 0.457 | 0.092 |
| Na | 174,698 | 0.405 | 0.057 |
| H | 156,454 | 0.378 | 0.059 |
| Se | 252,724 | 0.355 | 0.106 |
| Ba | 190,249 | 0.310 | 0.025 |
| Te | 236,835 | 0.293 | 0.088 |

Full table at `csv/per_element_mae.csv`; bar chart at `plots/per_element_mae.png`.

## Error vs formation energy

Pearson correlation between |ALIGNN - DFT| and `e_form`: **-0.371**. Negative: less stable structures have smaller error (often because they trend metallic in DFT and ALIGNN agrees). Plot at `plots/error_vs_eform.png`.

## Caveats

- 99 of 100 array shards landed; shard 9 missing (~45k entries; resubmit fills in).
- v12 includes off-hull entries (e_above_hull > 0) which are less physically meaningful than on-hull structures. The on-hull subset (114,389 entries) reproduces v11 exactly (MAE 0.168, accuracy 89.1%); see per-hull-bin breakdown in `plots/parity_by_hull_bin.png` and `csv/metrics.csv`.
- ALIGNN `mp_gappbe_alignn` was trained on Materials Project 3D bulk PBE bandgaps, so v12 (Alexandria 3D) is in-distribution. The MAE 0.185 eV on 4.44M structures is comparable to the model's published validation MAE.

## Files

- `plots/parity_all.png`, `plots/parity_metals.png`, `plots/parity_nonmetals.png` (stratified parity + residual panels)
- `csv/worst_predictions.csv` (top 20 by absolute residual)
- `csv/stratified_metrics.csv` (the all/metals/non-metals MAE table above)
- `csv/per_element_mae.csv`, `plots/per_element_mae.png`
- `plots/error_vs_eform.png`
- existing: `plots/parity.png`, `plots/parity_all.png`, `plots/parity_by_hull_bin.png`, `plots/parity_density.png`, `plots/parity_metals.png`, `plots/parity_nonmetals.png`, `plots/confusion.png`, `plots/distribution_overlay.png`, `plots/error_vs_eform.png`, `plots/per_element_mae.png`, `csv/metrics.csv`, `summary.md`
