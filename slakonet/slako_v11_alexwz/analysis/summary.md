# v11_alexwz: SlakoNet vs ALIGNN vs DFT (cross-method)

Matched subset of 6,781 entries (SK 7,547 of 115,535 hull entries; 766 SK inf values dropped). DFT reference is Alexandria PBE `band_gap_ind`. SK v11 is partial because the Rockfish CPU run crashed at 8h40m via a single-worker torch.load failure; the rest will fill in on resubmit.

## Pairwise metrics

| comparison | N | MAE (eV) | RMSE | ME (bias) | metal/gap acc. |
|---|---|---|---|---|---|
| DFT_PBE_vs_SK | 6,781 | 1.034 | 2.027 | +0.333 | 77.820% |
| DFT_PBE_vs_ALIGNN | 6,781 | 0.177 | 0.443 | +0.019 | 90.473% |
| SK_vs_ALIGNN | 6,781 | 1.084 | 2.068 | -0.314 | 72.216% |

## Files

- `parity_dft_vs_sk.png`, `parity_dft_vs_alignn.png`, `parity_sk_vs_alignn.png`
- `confusion_grid.png`: 1x3 panel for the three pairwise metal/gap classifications
- `distribution_overlay.png`: histogram overlay of the three methods on the matched set
- `metrics.csv`, `matched_predictions.csv`
