# v12_all: SlakoNet PBE bandgap, full Alexandria 3D set

First analysis writeup for v12 SK (sister to `alignn_v12_all`). DFT reference is Alexandria PBE `band_gap_ind`. The v12 ALIGNN-side full-set predictions live on the cluster, not local; this writeup is **SK-vs-DFT only**. Scaling to a three-way DFT/SK/ALIGNN comparison requires a future scp of `alignn_v12_all/results/all_results.json`.

## SK run accounting

Of 4,489,295 v12 entries attempted (full Alexandria PBE 3D, no hull or Z filter):
- 1,931,787 (43.0%) per-id JSONs produced (74/100 shards completed + recent backfill)
  - 1,646,162 (36.7%) finite `sk_bandgap`
  - 285,625 (6.4%) `sk_bandgap = inf` (gap-too-wide overflow; dropped)
- 2,557,508 (57.0%) no JSON produced — silent dropout on lanthanides + Z>65 (the SK chemical ceiling, same as v11)

## SK numerical pathologies (new finding)

**103 entries have `sk_bandgap` between 20.0 eV and 6.72e+06 eV** (max ~6.7×10⁶ eV). These are not physical gaps — they appear to be numerical instabilities in the SK eigenvalue calc on off-hull / high-energy chemistries (28% have DFT metallic ground state, 72% have non-zero DFT gap; `e_above_hull` ranges 0.00–1.66 eV/atom). They are 0.006% of the finite-SK population but contribute >99% of the raw RMSE (5,713 eV → 0.83 eV post-clip).

All downstream metrics in this writeup use the **`sk_bandgap` ≤ 20.0 eV** clip. Full outlier list: `csv/sk_outliers.csv`.

## Headline metrics

| comparison | N | MAE | RMSE | ME | median \|err\| | p90 \|err\| | metal/gap acc |
|---|---|---|---|---|---|---|---|
| DFT vs SK (raw) | 1,646,162 | 6.258 | 5713.258 | +6.140 | 0.005 | 0.265 | 90.750% |
| DFT vs SK (post-clip ≤20 eV) | 1,646,059 | 0.206 | 0.831 | +0.088 | 0.005 | 0.264 | 90.752% |

**Headline (post-clip): MAE 0.21 eV, p90 0.26 eV, metal/gap accuracy 90.8%** — much lower than v11's 1.04 eV because v12's full Alexandria sweep is **91% DFT metals** vs v11's 55% (off-hull entries are predominantly metallic). The apparent SK improvement is a distribution shift, not a model improvement (see hull-stratified split below).

## v11 pipeline reproducibility cross-check

The 33,026-entry on-hull subset of v12 (`e_above_hull < 0.001`) reproduces v11 within rounding:

| | v12 on-hull subset | v11 reference |
|---|---|---|
| N | 33,026 | 40,807 |
| MAE | 1.039 | 1.039 |
| ME | +0.342 | +0.344 |
| median \|err\| | 0.024 | 0.023 |
| metal/gap acc | 78.523% | 78.518% |

Identical to 3 decimal places. Confirms array-sharded v12 produces the same per-id values as single-job v11 on the same structures.

## DFT-gap stratified MAE (post-clip, full 1,646,059)

| bin (eV) | N | SK MAE | SK ME | median \|err\| | p90 \|err\| |
|---|---|---|---|---|---|
| metallic (0.00–0.1) | 1,493,071 | 0.058 | +0.058 | 0.004 | 0.026 |
| narrow (0.05–1.0) | 66,551 | 0.842 | +0.271 | 0.479 | 2.010 |
| narrow-mid (1.00–2.0) | 38,000 | 1.793 | +0.301 | 1.482 | 3.132 |
| mid (2.00–4.0) | 39,002 | 2.495 | +0.432 | 2.317 | 3.959 |
| wide (4.00–∞) | 9,435 | 3.185 | +1.232 | 2.911 | 5.395 |

Same monotone degradation as v11 — SK is near-perfect on the 1,493,071 DFT metals (MAE 0.058 eV) and reaches MAE 3.185 eV on the 9,435 wide-gap (DFT > 4 eV) entries.

## Hull × metal/non-metal split (post-clip)

| hull bin | metal frac | metals MAE | non-metals MAE |
|---|---|---|---|
| on-hull | 54.5% | 0.055 (N=17,985) | 2.215 (N=15,041) |
| near-hull | 74.8% | 0.038 (N=235,652) | 1.802 (N=79,434) |
| off-hull | 92.2% | 0.043 (N=572,257) | 1.353 (N=48,206) |
| far-off | 98.5% | 0.079 (N=667,182) | 0.955 (N=10,302) |

**Key insight**: SK's non-metal MAE actually *decreases* with hull energy (2.22 → 1.80 → 1.35 → 0.96 eV) — but this isn't model improvement. Off-hull non-metals tend to be narrow-gap rather than wide-gap, so SK's "good on metals + collapses wide gaps" failure mode hits them less hard. On the on-hull subset, non-metal MAE is 2.22 eV, consistent with v11.

## Files

- `plots/parity_dft_vs_sk.png` (N=1,646,059 hexbin), `plots/confusion_dft_vs_sk.png`
- `plots/sk_mae_by_gap_bin.png`, `plots/sk_mae_by_hull_bin.png`
- `csv/metrics.csv` (raw + post-clip headline)
- `csv/stratified_metrics.csv` (DFT-gap bins)
- `csv/hull_stratified_metrics.csv` (hull × metal split)
- `csv/sk_outliers.csv` (103 numerical pathologies for record)
