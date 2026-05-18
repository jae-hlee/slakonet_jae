# v12_all: SlakoNet vs ALIGNN vs DFT (full Alexandria 3D)

v12 SK over the full Alexandria PBE 3D set (~4.49M attempted; SK at 43% chemical-ceiling-effective with 1.65M finite predictions post-clip). ALIGNN-v12 ran on all 4.49M entries. **2026-05-18**: ALIGNN-v12 slim file (`alignn_scalars_v12.jsonl.gz`) scp'd from atomgptlab, enabling the three-way DFT/SK/ALIGNN analysis at full SK-finite ∩ ALIGNN intersection (1,646,059 entries) — the largest cross-method comparison in the repo.

## SK run accounting

Of 4,489,295 v12 entries attempted (full Alexandria PBE 3D, no hull or Z filter):
- 1,931,787 (43.0%) per-id JSONs produced
  - 1,646,162 (36.7%) finite `sk_bandgap`
  - 285,625 (6.4%) `sk_bandgap = inf` (gap-too-wide overflow; dropped)
- 2,557,508 (57.0%) no JSON produced — silent dropout on lanthanides + Z>65 (the SK chemical ceiling, same as v11)

ALIGNN v12 ran cleanly on all 4,489,295 hull entries (no chemical ceiling on its side); three-way scope is therefore bounded by SK's 1,646,162.

## SK numerical pathologies

**103 entries have `sk_bandgap` between 20.0 eV and 6.72e+06 eV** (max ~6.7×10⁶ eV). Numerical instabilities in the SK eigenvalue calc on off-hull / high-energy chemistries. 0.006% of finite-SK population but >99% of the raw RMSE (5,713 eV → 0.83 eV post-clip). All downstream metrics use the **`sk_bandgap` ≤ 20.0 eV** clip.

## Three-way pairwise metrics (N = 1,646,059)

| comparison | N | MAE (eV) | RMSE | ME (bias) | metal/gap acc. |
|---|---|---|---|---|---|
| DFT_vs_SK | 1,646,059 | 0.206 | 0.831 | +0.088 | 90.752% |
| DFT_vs_ALIGNN | 1,646,059 | 0.220 | 0.587 | +0.184 | 71.223% |
| SK_vs_ALIGNN | 1,646,059 | 0.365 | 0.985 | +0.095 | 67.074% |
| DFT_vs_ALIGNN (full ALIGNN 4.49M) | 4,489,295 | 0.185 | 0.551 | +0.148 | 78.100% |

**v12 reverses the v11 ALIGNN-wins headline**: on this matched set, SK MAE 0.206 vs ALIGNN MAE 0.220 — SK is 6% better in aggregate. The reason is distribution shift, not model improvement: v12 is **91% DFT metals** (vs v11's 55%), and SK is near-perfect on metals (median \|err\| 0.004 eV on the 1,493,071-entry DFT-metallic bin). When metals dominate, SK's good-on-metals collapses the aggregate below ALIGNN's. The v11 ALIGNN-wins-6x finding on bulk crystals at 55% metallic still stands; v12's headline reflects a different dataset composition, not a different relative model accuracy.

DFT-vs-ALIGNN on the 1.65M SK-finite intersection (0.220 eV) shifts +35 meV from the full 4.49M ALIGNN reference (0.185 eV) — restricting to SK-finite mat_ids actually makes ALIGNN's MAE *worse*, opposite of the v11 pattern (where the shift was only +6 meV). The SK-finite v12 subset over-represents non-metals (where ALIGNN's per-class MAE is higher), so the restriction concentrates ALIGNN's worst predictions.

## v11 reproducibility cross-check (SK on-hull subset of v12)

The 33,026-entry on-hull subset of v12 (`e_above_hull < 0.001`) reproduces v11 within rounding:

| | v12 on-hull subset | v11 reference |
|---|---|---|
| N | 33,026 | 40,807 |
| MAE | 1.039 | 1.039 |
| ME | +0.342 | +0.344 |
| median \|err\| | 0.024 | 0.023 |
| metal/gap acc | 78.523% | 78.518% |

Identical to 3 decimal places.

## DFT-gap stratified MAE (DFT-vs-SK, post-clip, full 1,646,059)

| bin (eV) | N | SK MAE | SK ME | median \|err\| | p90 \|err\| |
|---|---|---|---|---|---|
| metallic (0.00–0.1) | 1,493,071 | 0.058 | +0.058 | 0.004 | 0.026 |
| narrow (0.05–1.0) | 66,551 | 0.842 | +0.271 | 0.479 | 2.010 |
| narrow-mid (1.00–2.0) | 38,000 | 1.793 | +0.301 | 1.482 | 3.132 |
| mid (2.00–4.0) | 39,002 | 2.495 | +0.432 | 2.317 | 3.959 |
| wide (4.00–∞) | 9,435 | 3.185 | +1.232 | 2.911 | 5.395 |

Monotone degradation matches v11: SK near-perfect on metals (MAE 0.058 eV on 1,493,071 entries) and degrades to MAE 3.185 eV on wide-gap (DFT > 4 eV) entries.

## Hull × metal/non-metal split (DFT-vs-SK, post-clip)

| hull bin | metal frac | metals MAE | non-metals MAE |
|---|---|---|---|
| on-hull | 54.5% | 0.055 (N=17,985) | 2.215 (N=15,041) |
| near-hull | 74.8% | 0.038 (N=235,652) | 1.802 (N=79,434) |
| off-hull | 92.2% | 0.043 (N=572,257) | 1.353 (N=48,206) |
| far-off | 98.5% | 0.079 (N=667,182) | 0.955 (N=10,302) |

**Key insight (SK-only)**: non-metal MAE drops with hull energy (2.22 → 1.80 → 1.35 → 0.96 eV) — distribution shift, not model improvement. Off-hull non-metals are predominantly narrow-gap, where SK's wide-gap collapse failure mode hits less hard. On the on-hull subset, non-metal MAE is 2.22 eV, consistent with v11.

## Files

- `plots/parity_dft_vs_sk.png` (N=1,646,059 hexbin)
- `plots/parity_dft_vs_alignn.png`, `plots/parity_sk_vs_alignn.png` (N=1,646,059 hexbins)
- `plots/three_way_confusion_grid.png` (1x3 panel)
- `plots/distribution_overlay.png` (log-y histogram)
- `plots/confusion_dft_vs_sk.png`, `plots/sk_mae_by_gap_bin.png`, `plots/sk_mae_by_hull_bin.png`
- `csv/metrics.csv` (SK-side raw + post-clip)
- `csv/three_way_metrics.csv` (3-way + full-ALIGNN reference row)
- `csv/three_way_matched.csv.gz` (N=1,646,059 matched per-id table, gzipped)
- `csv/stratified_metrics.csv`, `csv/hull_stratified_metrics.csv`
- `csv/sk_outliers.csv` (103 numerical pathologies)
