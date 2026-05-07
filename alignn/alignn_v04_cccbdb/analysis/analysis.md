# v04 CCCBDB: ALIGNN PBE bandgap deep analysis

Reference: molecular HOMO-LUMO gap (Hartree -> eV via x27.2114). Metal/gap split at 0.05 eV.

## Metal vs non-metal stratification

| subset | N | MAE | RMSE | ME (bias) | med \|err\| | p90 \|err\| | DFT mean | ALIGNN mean |
|---|---|---|---|---|---|---|---|---|
| all | 1,330 | 3.365 | 7.931 | -3.197 | 3.019 | 5.183 | 7.023 | 3.826 |
| DFT metals | 2 | 3.742 | 3.742 | +3.742 | 3.742 | 3.742 | -0.001 | 3.741 |
| DFT non-metals | 1,328 | 3.364 | 7.936 | -3.208 | 3.016 | 5.184 | 7.033 | 3.826 |

## Metal/gap classification (full set)

- Accuracy: 99.549%
- True negatives (DFT metal, ALIGNN metal): 0 (0.0%)
- False positives (DFT metal, ALIGNN gap):  2 (0.2%)
- False negatives (DFT gap, ALIGNN metal):  4 (0.3%)
- True positives (DFT gap, ALIGNN gap):     1,324 (99.5%)

**Failure-mode breakdown.** False positives (2) and false negatives (4) are roughly balanced.  Metal MAE (3.742) and non-metal MAE (3.364) are similar.

## Worst predictions

Top 20 entries by absolute residual (full table at `csv/worst_predictions.csv`):

| jid | species | homo_lumo_gap_eV | alignn_bandgap | residual |
| --- | --- | --- | --- | --- |
| cc-699 | CCl3- | 255.762 | 3.686 | -252.076 |
| cc-761 | Ne | 53.069 | 8.771 | -44.297 |
| cc-986 | Ne2 | 48.079 | 8.351 | -39.728 |
| cc-1017 | Ar2 | 22.169 | 6.746 | -15.423 |
| cc-1257 | H3+ | 19.373 | 4.147 | -15.227 |
| cc-888 | HeH+ | 25.400 | 10.587 | -14.813 |
| cc-1151 | NH4+ | 16.754 | 4.285 | -12.470 |
| cc-993 | NeH+ | 17.278 | 7.311 | -9.967 |
| cc-114 | CF4 | 16.531 | 7.536 | -8.995 |
| cc-1107 | H2Ar | 14.141 | 5.182 | -8.959 |

## Per-element MAE (worst 15)

Entries containing each element (compositions overlap; the same entry appears under each of its elements). Min count threshold applied.

| element | count | MAE (eV) | median \|err\| (eV) |
|---|---|---|---|
| D | 16 | 5.636 | 5.465 |
| Cl | 193 | 3.933 | 2.660 |
| C | 880 | 3.517 | 3.158 |
| N | 316 | 3.387 | 3.336 |
| H | 931 | 3.343 | 3.224 |
| F | 181 | 3.233 | 3.000 |
| O | 464 | 3.179 | 3.137 |
| Se | 17 | 3.174 | 3.149 |
| Ge | 15 | 3.012 | 3.179 |
| As | 13 | 2.987 | 2.941 |
| Si | 34 | 2.942 | 2.920 |
| P | 47 | 2.846 | 2.725 |
| S | 136 | 2.641 | 2.723 |
| B | 48 | 2.626 | 2.296 |
| Zn | 13 | 2.618 | 3.248 |

Full table at `csv/per_element_mae.csv`; bar chart at `plots/per_element_mae.png`.

## Caveats

- ALIGNN `mp_gappbe_alignn` was trained on Materials Project 3D bulk crystals; CCCBDB is isolated molecules, far out-of-distribution.
- Reference is molecular DFT (Gaussian basis with various functionals), not solid-state PBE; the functional / basis difference adds to ALIGNN's error and partly explains the strong negative bias.
- 3 entries have HOMO-LUMO gap > 30 eV (CCl3- at ~256 eV, Ne and Ne2 at ~50 eV), almost certainly CCCBDB calculation artifacts. Metrics include them; the parity plot is clipped to 0-20 eV for readability.

## Files

- `plots/parity_all.png`, `plots/parity_metals.png`, `plots/parity_nonmetals.png` (stratified parity + residual panels)
- `csv/worst_predictions.csv` (top 20 by absolute residual)
- `csv/stratified_metrics.csv` (the all/metals/non-metals MAE table above)
- `csv/per_element_mae.csv`, `plots/per_element_mae.png`
- existing: `plots/parity.png`, `plots/parity_all.png`, `plots/parity_metals.png`, `plots/parity_nonmetals.png`, `plots/distribution.png`, `plots/per_element_mae.png`, `csv/metrics.csv`, `summary.md`

## See also

Cross-method comparison vs SlakoNet on the same matched structures: `slakonet/slako_v04_cccbdb/analysis/plots/sk_vs_alignn.png` and `slakonet/slako_v04_cccbdb/analysis/plots/confusion_sk_vs_alignn.png`. Metrics in `slakonet/slako_v04_cccbdb/analysis/csv/sk_vs_alignn_metrics.csv`. Cross-dataset roll-up at `slakonet/slakonet_comprehensive_analysis/csv/sk_vs_alignn_cross_dataset.csv`.
