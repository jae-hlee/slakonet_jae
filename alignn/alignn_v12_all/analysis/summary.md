# v12 alex_pbe_3d_all: ALIGNN PBE bandgap on the full Alexandria PBE 3D set

Total **4,489,295 entries** (the full Alexandria PBE 3D set). Every row has a `band_gap_ind` reference, so all 4,489,295 are usable for parity vs DFT PBE.

## Headline (full set)

- **N**: 4,489,295
- **MAE**: 0.185 eV; **RMSE**: 0.551 eV; **ME (bias)**: +0.148 eV
- **Median \|err\|**: 0.014 eV; **90th pct \|err\|**: 0.510 eV
- **Metal/gap accuracy**: 78.395% (metal cutoff: predicted gap < 0.05 eV; reference gap == 0)
- **Confusion (DFT ref, ALIGNN pred)** TN/FP/FN/TP: 3,154,044 / 941,170 / 28,732 / 365,349

## By e_above_hull bin

| bin | range (eV/atom) | N | MAE (eV) | RMSE | ME | accuracy |
|---|---|---|---|---|---|---|
| 0_hull | = 0 (on hull) | 115,535 | 0.168 | 0.476 | +0.024 | 89.312% |
| 1_near_hull | (0, 0.1] | 1,339,487 | 0.186 | 0.552 | +0.121 | 85.798% |
| 2_off_hull | (0.1, 0.5] | 1,866,463 | 0.205 | 0.624 | +0.178 | 80.142% |
| 3_far_off_hull | > 0.5 | 1,167,810 | 0.154 | 0.414 | +0.142 | 66.033% |

## Files

- `plots/parity.png`: full-set parity + residuals (with N/MAE/RMSE/ME annotations)
- `plots/parity_by_hull_bin.png`: 2x2 grid, one panel per hull bin
- `plots/confusion.png`: full-set metal/gap confusion matrix
- `plots/distribution_overlay.png`: DFT vs ALIGNN gap histograms
- `csv/metrics.csv`: full-set + per-hull-bin metrics

## Caveats

- Full coverage of the 4,489,295-entry Alexandria PBE 3D set.
- Off-hull DFT references in Alexandria are computed at the same PBE level as on-hull, but off-hull structures are less physically meaningful (high formation energy, often unstable in real synthesis). Per-bin breakdown is provided so the on-hull subset can be read separately as the cleanest comparison.
- v12 on-hull (e_above_hull = 0) is the same 115,535 structures as v11_alexwz ALIGNN (same input zip, no Z filter on either run, v11 = the hull-stable subset). The ALIGNN predictions are byte-identical between the two runs (max difference 2e-6 eV), so MAE reproduces exactly (0.168). The accuracy column here (89.3%) uses a strict band_gap_ind = 0 metal cut, which differs from the v11 analysis's 0.05 eV cut (89.1%), so the accuracy figures are not directly comparable even on these identical structures.
