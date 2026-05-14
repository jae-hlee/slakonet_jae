# v11_alexwz: SlakoNet vs ALIGNN vs DFT (cross-method)

SK v11 effectively complete on Rockfish /data (rsync from /scratch4 after the group inode-quota workaround). DFT reference is Alexandria PBE `band_gap_ind`.

## SK run accounting

Of 115,535 hull entries attempted:
- 45,203 (39.1%) per-id JSONs produced
  - 40,807 (35.3%) finite `sk_bandgap` → used here
  - 4,396 (3.8%) `sk_bandgap = inf` (gap-too-wide overflow; dropped)
- 70,332 (60.9%) no JSON produced — silent dropout on lanthanides + heavies (Z>65), the documented SK ceiling on this dataset

## Pairwise metrics

DFT vs SK at the **full 40,807-entry finite-SK scale** (up from the prior 6,781):

| comparison | N | MAE (eV) | RMSE | ME (bias) | metal/gap acc. |
|---|---|---|---|---|---|
| DFT_PBE_vs_SK (45k run) | 40,807 | 1.039 | 2.029 | +0.344 | 78.518% |

Three-way comparison on the 6,781-entry intersection with the prior ALIGNN-bearing matched set (the full ALIGNN-v11 predictions live on the cluster, not local; scaling the three-way to ~40k would require scp'ing `alignn_v11_alexwz/results/alignn_predictions.json`):

| comparison | N | MAE (eV) | RMSE | ME (bias) | metal/gap acc. |
|---|---|---|---|---|---|
| DFT_PBE_vs_SK | 6,781 | 1.034 | 2.027 | +0.333 | 77.820% |
| DFT_PBE_vs_ALIGNN | 6,781 | 0.177 | 0.443 | +0.019 | 90.473% |
| SK_vs_ALIGNN | 6,781 | 1.084 | 2.068 | -0.314 | 72.216% |

DFT-vs-SK MAE at full scale (1.039 eV) tracks the 6,781-subset value (1.034 eV) within 0.005 eV; the prior matched set is representative of the full SK-finite population. The "ALIGNN wins by ~5.9x on bulk crystals" headline holds.

## SK bimodal failure, surfaced at scale

Stratifying the full 40,807-entry DFT-vs-SK set by DFT gap bin:

| bin (eV) | N | SK MAE | SK ME | median \|err\| |
|---|---|---|---|---|
| metallic (0.00–0.1) | 22,340 | 0.055 | +0.055 | 0.004 |
| narrow (0.05–1.0) | 4,336 | 0.926 | +0.229 | 0.651 |
| narrow-mid (1.00–2.0) | 4,965 | 1.910 | +0.323 | 1.557 |
| mid (2.00–4.0) | 6,793 | 2.876 | +0.888 | 2.504 |
| wide (4.00–∞) | 2,373 | 3.431 | +1.765 | 3.223 |

SK is near-perfect on the 22,340 DFT metals (MAE 0.055 eV) and degrades monotonically with reference gap; by the wide-gap bin (DFT > 4 eV, N=2,373) MAE reaches 3.431 eV. The wide-gap mean error is +1.77 eV (net over-prediction on average), but the p90 |err| of 5.81 eV reflects a bimodal residual distribution: TM-compound and fluoride entries collapse to ~0 (large negative residuals) while other wide-gap chemistries over-shoot, both pulling the MAE up. This is the bimodal SK failure pattern documented in `slakonet/slako_v03_alex/analysis/analysis.md`, now visible at 40k scale rather than 7k.

## Files

- `plots/parity_dft_vs_sk.png` (now at N=40,807), `plots/parity_dft_vs_alignn.png`, `plots/parity_sk_vs_alignn.png`
- `plots/confusion_grid.png`: 1x3 panel for the three pairwise metal/gap classifications
- `plots/distribution_overlay.png`: histogram overlay on the matched subset
- `plots/sk_mae_by_gap_bin.png`: stratified DFT-vs-SK MAE by gap bin
- `csv/metrics.csv`, `csv/stratified_metrics.csv`, `csv/matched_predictions.csv` (mat_id, formula, dft_gap, sk_gap, alignn_gap, e_form)
