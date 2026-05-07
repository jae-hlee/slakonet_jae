# v11_alexwz: ALIGNN PBE bandgap deep analysis

Reference: DFT PBE `band_gap_ind` from Alexandria. Metal/gap split at 0.05 eV.

## Metal vs non-metal stratification

| subset | N | MAE | RMSE | ME (bias) | med \|err\| | p90 \|err\| | DFT mean | ALIGNN mean |
|---|---|---|---|---|---|---|---|---|
| all | 115,535 | 0.168 | 0.476 | +0.024 | 0.015 | 0.467 | 0.653 | 0.677 |
| DFT metals | 80,695 | 0.081 | 0.359 | +0.076 | 0.009 | 0.087 | 0.000 | 0.077 |
| DFT non-metals | 34,840 | 0.369 | 0.674 | -0.096 | 0.182 | 0.883 | 2.163 | 2.067 |

## Metal/gap classification (full set)

- Accuracy: 89.130%
- True negatives (DFT metal, ALIGNN metal): 69,866 (60.5%)
- False positives (DFT metal, ALIGNN gap):  10,829 (9.4%)
- False negatives (DFT gap, ALIGNN metal):  1,730 (1.5%)
- True positives (DFT gap, ALIGNN gap):     33,110 (28.7%)

**Failure-mode breakdown.** The dominant error mode is **false positives** (10,829 vs 1,730): ALIGNN predicts a gap where DFT says the structure is metallic.  Metals are predicted much more accurately (MAE 0.081) than non-metals (MAE 0.369); the model defaults toward small-gap predictions which works for metals but underestimates non-metal gaps.

## Worst predictions

Top 20 entries by absolute residual (full table at `csv/worst_predictions.csv`):

| mat_id | formula | band_gap_ind | alignn_bandgap | residual | e_form |
| --- | --- | --- | --- | --- | --- |
| agm002210810 | YbHfF6 | 7.420 | 0.022 | -7.398 | -4.376 |
| agm002147511 | YbF2 | 7.352 | 0.105 | -7.247 | -4.584 |
| agm003770263 | BaAlF6 | 0.000 | 7.201 | 7.201 | -3.568 |
| agm005938370 | NpB3F12 | 0.265 | 7.460 | 7.196 | -3.332 |
| agm002156287 | Li2YbHfF8 | 7.019 | 0.091 | -6.928 | -3.990 |
| agm002210700 | YbSiF6 | 7.407 | 0.480 | -6.928 | -3.947 |
| agm002218297 | Li2YbThF8 | 6.962 | 0.097 | -6.865 | -4.103 |
| agm002271151 | Na2YbThF8 | 6.845 | -0.007 | -6.852 | -4.049 |
| agm002149078 | YbZrF6 | 6.648 | 0.019 | -6.629 | -4.272 |
| agm005542701 | PmF3 | 7.664 | 1.184 | -6.480 | -4.450 |

## Per-element MAE (worst 15)

Entries containing each element (compositions overlap; the same entry appears under each of its elements). Min count threshold applied.

| element | count | MAE (eV) | median \|err\| (eV) |
|---|---|---|---|
| Np | 2,766 | 0.692 | 0.093 |
| F | 5,450 | 0.614 | 0.226 |
| Cl | 4,333 | 0.470 | 0.207 |
| Cs | 5,251 | 0.430 | 0.198 |
| Yb | 1,487 | 0.420 | 0.014 |
| Rb | 5,211 | 0.410 | 0.175 |
| Br | 3,381 | 0.392 | 0.180 |
| K | 5,422 | 0.360 | 0.133 |
| I | 3,082 | 0.346 | 0.117 |
| O | 17,469 | 0.335 | 0.121 |
| H | 5,180 | 0.331 | 0.093 |
| Pa | 3,645 | 0.330 | 0.017 |
| Na | 4,769 | 0.324 | 0.103 |
| Ba | 4,930 | 0.300 | 0.060 |
| S | 8,093 | 0.277 | 0.076 |

Full table at `csv/per_element_mae.csv`; bar chart at `plots/per_element_mae.png`.

## Error vs formation energy

Pearson correlation between |ALIGNN - DFT| and `e_form`: **-0.333**. Negative: less stable structures have smaller error (often because they trend metallic in DFT and ALIGNN agrees). Plot at `plots/error_vs_eform.png`.

## Caveats

- Alexandria 3D bulk crystals match `mp_gappbe_alignn`'s Materials Project training distribution. This is the cleanest cross-method comparison in the v04..v12 sweep.
- v11 has no Z<=65 element filter, so f-block lanthanides Ce-Tb and elements beyond Z=65 are present. ALIGNN handles them via the cgcnn-features-clamped-to-Z=103 path; the SK sister silently drops them. So matched (SK ∩ ALIGNN) coverage is narrower than ALIGNN-only.

## Files

- `plots/parity_all.png`, `plots/parity_metals.png`, `plots/parity_nonmetals.png` (stratified parity + residual panels)
- `csv/worst_predictions.csv` (top 20 by absolute residual)
- `csv/stratified_metrics.csv` (the all/metals/non-metals MAE table above)
- `csv/per_element_mae.csv`, `plots/per_element_mae.png`
- `plots/error_vs_eform.png`
- existing: `plots/parity.png`, `plots/confusion.png`, `plots/distribution.png`, `csv/metrics.csv`, `summary.md`
