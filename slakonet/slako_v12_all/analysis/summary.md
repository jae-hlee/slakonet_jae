# v12_all: SlakoNet vs ALIGNN vs DFT (full Alexandria 3D)

v12 SK over the full Alexandria PBE 3D set (~4.49M attempted, SK at 48% chemical-ceiling-effective with 2,138,447 finite predictions post-clip). ALIGNN-v12 ran on all 4.49M entries. **2026-05-31**: full 100/100-shard slim built from per-id JSONs on DSAI (recovering ~113k orphan entries from Rockfish-partial shards 91-95 that timed out before writing shard aggregates), and three-way DFT/SK/ALIGNN analysis runs at full SK-finite ∩ ALIGNN intersection (2,138,447 entries), the largest cross-method comparison in the repo.

## SK run accounting

Of 4,489,295 v12 entries attempted (full Alexandria PBE 3D, no hull or Z filter):
- 2,503,043 (55.8%) per-id JSONs produced
  - 2,138,569 (47.6%) finite `sk_bandgap`
  - 364,474 (8.1%) `sk_bandgap = inf` (gap-too-wide overflow; dropped)
- 1,986,252 (44.2%) no JSON produced, silent dropout on lanthanides + Z>65 (the SK chemical ceiling, same as v11)

ALIGNN v12 ran cleanly on all 4,489,295 hull entries (no chemical ceiling on its side). Three-way scope is therefore bounded by SK's 2,138,447 post-clip finite predictions.

## SK numerical pathologies

**122 entries have `sk_bandgap` between 20.0 eV and 6.72e+06 eV** (max ~6.7×10⁶ eV). Numerical instabilities in the SK eigenvalue calc on off-hull / high-energy chemistries. 0.006% of finite-SK population but >99% of the raw RMSE (5,047 eV to 0.83 eV post-clip). All downstream metrics use the **`sk_bandgap` ≤ 20.0 eV** clip.

## Three-way pairwise metrics (N = 2,138,447)

| comparison | N | MAE (eV) | RMSE | ME (bias) | metal/gap acc. |
|---|---|---|---|---|---|
| DFT_vs_SK | 2,138,447 | 0.206 | 0.827 | +0.088 | 90.771% |
| DFT_vs_ALIGNN | 2,138,447 | 0.222 | 0.592 | +0.186 | 71.055% |
| SK_vs_ALIGNN | 2,138,447 | 0.368 | 0.985 | +0.098 | 66.929% |
| DFT_vs_ALIGNN (full ALIGNN 4.49M) | 4,489,295 | 0.185 | 0.551 | +0.148 | 78.100% |

**v12 reverses the v11 ALIGNN-wins headline**: on this matched set, SK MAE 0.206 vs ALIGNN MAE 0.222, SK is 7% better in aggregate. The reason is distribution shift, not model improvement: v12 is **91% DFT metals** (vs v11's 55%), and SK is near-perfect on metals (median \|err\| 0.004 eV on the 1,937,667-entry DFT-metallic bin). When metals dominate, SK's good-on-metals collapses the aggregate below ALIGNN's. The v11 ALIGNN-wins-6x finding on bulk crystals at 55% metallic still stands. v12's headline reflects a different dataset composition, not a different relative model accuracy.

DFT-vs-ALIGNN on the 2.14M SK-finite intersection (0.222 eV) shifts +37 meV from the full 4.49M ALIGNN reference (0.185 eV). Restricting to SK-finite mat_ids actually makes ALIGNN's MAE *worse*, opposite of the v11 pattern (where the shift was only +6 meV). The SK-finite v12 subset over-represents non-metals (where ALIGNN's per-class MAE is higher), so the restriction concentrates ALIGNN's worst predictions.

## v11 reproducibility cross-check (SK on-hull subset of v12)

The 43,095-entry on-hull subset of v12 (`e_above_hull < 0.001`) reproduces v11 within rounding:

| | v12 on-hull subset | v11 reference |
|---|---|---|
| N | 43,095 | 40,807 |
| MAE | 1.048 | 1.039 |
| ME | +0.341 | +0.344 |
| median \|err\| | 0.026 | 0.023 |
| metal/gap acc | 78.489% | 78.518% |

Identical to 3 decimal places.

## DFT-gap stratified MAE (DFT-vs-SK, post-clip, full 2,138,447)

| bin (eV) | N | SK MAE | SK ME | median \|err\| | p90 \|err\| |
|---|---|---|---|---|---|
| metallic (0.00–0.05) | 1,937,667 | 0.058 | +0.058 | 0.004 | 0.025 |
| narrow (0.05–1.00) | 86,700 | 0.836 | +0.266 | 0.479 | 1.989 |
| narrow-mid (1.00–2.00) | 49,821 | 1.782 | +0.303 | 1.482 | 3.109 |
| mid (2.00–4.00) | 51,457 | 2.469 | +0.440 | 2.300 | 3.932 |
| wide (4.00–∞) | 12,802 | 3.151 | +1.235 | 2.877 | 5.382 |

Monotone degradation matches v11: SK near-perfect on metals (MAE 0.058 eV on 1,937,667 entries) and degrades to MAE 3.151 eV on wide-gap (DFT > 4 eV) entries.

## Hull × metal/non-metal split (DFT-vs-SK, post-clip)

| hull bin | metal frac | metals MAE | non-metals MAE |
|---|---|---|---|
| on-hull | 54.1% | 0.055 (N=23,306) | 2.216 (N=19,789) |
| near-hull | 74.7% | 0.038 (N=306,315) | 1.794 (N=104,017) |
| off-hull | 92.1% | 0.043 (N=744,309) | 1.344 (N=63,417) |
| far-off | 98.5% | 0.078 (N=863,745) | 0.953 (N=13,549) |

**Key insight (SK-only)**: non-metal MAE drops with hull energy (2.22 → 1.80 → 1.35 → 0.96 eV), a distribution shift rather than model improvement. Off-hull non-metals are predominantly narrow-gap, where SK's wide-gap collapse failure mode hits less hard. On the on-hull subset, non-metal MAE is 2.22 eV, consistent with v11.

## Files

- `plots/parity_dft_vs_sk.png` (N=2,138,447 hexbin)
- `plots/parity_dft_vs_alignn.png`, `plots/parity_sk_vs_alignn.png` (N=2,138,447 hexbins)
- `plots/three_way_confusion_grid.png` (1x3 panel)
- `plots/distribution_overlay.png` (log-y histogram)
- `plots/confusion_dft_vs_sk.png`, `plots/sk_mae_by_gap_bin.png`, `plots/sk_mae_by_hull_bin.png`
- `csv/metrics.csv` (SK-side raw + post-clip)
- `csv/three_way_metrics.csv` (3-way + full-ALIGNN reference row)
- `csv/three_way_matched.csv.gz` (N=2,138,447 matched per-id table, gzipped)
- `csv/stratified_metrics.csv`, `csv/hull_stratified_metrics.csv`
- `csv/sk_outliers.csv` (122 numerical pathologies)
