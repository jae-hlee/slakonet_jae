# v12 alex_pbe_3d_all: ALIGNN PBE bandgap on the full Alexandria PBE 3D set

99 of 100 array shards landed locally; total **4,444,402 entries** (out of ~4,489,295 expected). Shard 9 is missing (its checkpoint dir is empty; the corresponding ~45k entries will fill in on Rockfish/atomgptlab resubmit). After dropping rows with missing `band_gap_ind`, **4,444,402 entries** are usable for parity vs DFT PBE.

## Headline (full set)

- **N**: 4,444,402
- **MAE**: 0.185 eV; **RMSE**: 0.551 eV; **ME (bias)**: +0.148 eV
- **Median \|err\|**: 0.014 eV; **90th pct \|err\|**: 0.510 eV
- **Metal/gap accuracy**: 78.099%
- **Confusion (DFT ref, ALIGNN pred)** TN/FP/FN/TP: 3,130,841,953,196,20,152,340,213

## By e_above_hull bin

| bin | range (eV/atom) | N | MAE (eV) | RMSE | ME | accuracy |
|---|---|---|---|---|---|---|
| 0_hull | = 0 (on hull) | 114,389 | 0.168 | 0.477 | +0.024 | 89.141% |
| 1_near_hull | (0, 0.1] | 1,325,979 | 0.186 | 0.553 | +0.121 | 85.352% |
| 2_off_hull | (0.1, 0.5] | 1,847,839 | 0.205 | 0.624 | +0.178 | 79.814% |
| 3_far_off_hull | > 0.5 | 1,156,195 | 0.154 | 0.414 | +0.142 | 65.950% |

## Files

- `plots/parity.png`: full-set parity + residuals (with N/MAE/RMSE/ME annotations)
- `plots/parity_by_hull_bin.png`: 2x2 grid, one panel per hull bin
- `plots/confusion.png`: full-set metal/gap confusion matrix
- `plots/distribution_overlay.png`: DFT vs ALIGNN gap histograms
- `csv/metrics.csv`: full-set + per-hull-bin metrics

## Caveats

- 99/100 shard coverage; shard 9 missing (~45k entries). Resubmit will fill in.
- Off-hull DFT references in Alexandria are computed at the same PBE level as on-hull, but off-hull structures are less physically meaningful (high formation energy, often unstable in real synthesis). Per-bin breakdown is provided so the on-hull subset can be read separately as the cleanest comparison.
- v12 on-hull (e_above_hull = 0) result is directly comparable to v11_alexwz ALIGNN (MAE 0.168). They share the input zip and filter; v12 numbers should match v11 closely on the same subset.
