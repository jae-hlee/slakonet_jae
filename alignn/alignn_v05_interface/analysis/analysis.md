# v05 interface_db: ALIGNN PBE bandgap deep analysis

Reference: OptB88vdW bandgap (clipped at 0). Metal/gap split at 0.05 eV.

## Metal vs non-metal stratification

| subset | N | MAE | RMSE | ME (bias) | med \|err\| | p90 \|err\| | DFT mean | ALIGNN mean |
|---|---|---|---|---|---|---|---|---|
| all | 587 | 0.531 | 0.717 | +0.487 | 0.396 | 1.052 | 0.466 | 0.953 |
| DFT metals | 143 | 0.699 | 0.973 | +0.699 | 0.565 | 1.422 | 0.004 | 0.703 |
| DFT non-metals | 444 | 0.476 | 0.612 | +0.419 | 0.375 | 0.976 | 0.615 | 1.034 |

## Metal/gap classification (full set)

- Accuracy: 77.002%
- True negatives (DFT metal, ALIGNN metal): 12 (2.0%)
- False positives (DFT metal, ALIGNN gap):  131 (22.3%)
- False negatives (DFT gap, ALIGNN metal):  4 (0.7%)
- True positives (DFT gap, ALIGNN gap):     440 (75.0%)

**Failure-mode breakdown.** The dominant error mode is **false positives** (131 vs 4): ALIGNN predicts a gap where DFT says the structure is metallic.  Metal MAE (0.699) and non-metal MAE (0.476) are similar.

## Worst predictions

Top 20 entries by absolute residual (full table at `csv/worst_predictions.csv`):

| jid | formula | optb88vdw_bandgap_clipped | alignn_bandgap | residual |
| --- | --- | --- | --- | --- |
| Interface-JVASP-39_JVASP-62940_film_miller_0_0_1_sub_miller_ |  | 0.000 | 3.708 | 3.708 |
| Interface-JVASP-39_JVASP-39_film_miller_1_0_0_sub_miller_0_0 |  | 0.000 | 3.490 | 3.490 |
| Interface-JVASP-30_JVASP-39_film_miller_1_0_0_sub_miller_0_0 |  | 0.000 | 2.880 | 2.880 |
| Interface-JVASP-39_JVASP-8003_film_miller_0_0_1_sub_miller_0 |  | 0.000 | 2.784 | 2.784 |
| Interface-JVASP-39_JVASP-8158_film_miller_0_0_1_sub_miller_0 |  | 0.000 | 2.656 | 2.656 |
| Interface-JVASP-39_JVASP-8118_film_miller_0_0_1_sub_miller_0 |  | 0.000 | 2.560 | 2.560 |
| Interface-JVASP-39_JVASP-30_film_miller_0_0_1_sub_miller_0_0 |  | 0.000 | 2.494 | 2.494 |
| Interface-JVASP-39_JVASP-1183_film_miller_0_0_1_sub_miller_0 |  | 0.000 | 2.148 | 2.148 |
| Interface-JVASP-8169_JVASP-39_film_miller_1_1_0_sub_miller_1 |  | 0.283 | 2.369 | 2.087 |
| Interface-JVASP-8003_JVASP-7836_film_miller_1_1_0_sub_miller |  | 0.891 | 2.925 | 2.034 |

## Caveats

- Reference is OptB88vdW, NOT PBE. ALIGNN predicts PBE bandgap. OptB88vdW typically gives slightly larger gaps than PBE for non-metals, so part of any ALIGNN-minus-DFT bias is the functional shift, not pure model error.
- 107 of 587 entries had `optb88vdw_bandgap < 0` (interface SCF artifact) and were clipped to 0 before parity. The metals subset is dominated by these clipped entries.
- interface_db output JSON does not carry `formula`; per-element analysis is not available for v05.

## Files

- `plots/parity_all.png`, `plots/parity_metals.png`, `plots/parity_nonmetals.png` (stratified parity + residual panels)
- `csv/worst_predictions.csv` (top 20 by absolute residual)
- `csv/stratified_metrics.csv` (the all/metals/non-metals MAE table above)
- existing: `plots/parity.png`, `plots/confusion.png`, `plots/distribution.png`, `csv/metrics.csv`, `summary.md`

## See also

Cross-method comparison vs SlakoNet on the same matched structures: `slakonet/slako_v05_interface/analysis/plots/sk_vs_alignn.png` and `slakonet/slako_v05_interface/analysis/plots/confusion_sk_vs_alignn.png`. Metrics in `slakonet/slako_v05_interface/analysis/csv/sk_vs_alignn_metrics.csv`. Cross-dataset roll-up at `slakonet/slakonet_comprehensive_analysis/csv/sk_vs_alignn_cross_dataset.csv`.
