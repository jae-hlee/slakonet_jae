# SlakoNet cross-dataset analysis

Aggregates the following SlakoNet runs: `v03_alex`, `v04_cccbdb`, `v05_interface`, `v06_surface`, `v07_vacancy`, `v08_supercon`, `v09_1d`, `v10_2d`, `v11_alexwz`, `v12_all`.
The pre-built artifacts in this directory (csv, plots, this writeup) are regenerated
by a local aggregator script (`build_analysis.py`, kept off-repo). It reads each
sister `slako_v*` project's per-row scalars: `analysis/csv/summary.csv` for
v03–v10, and the slim `results/sk_scalars_v*.jsonl.gz` for v11/v12 (whose per-row
data lives there rather than in a `summary.csv`).

## Datasets loaded

| key | title | kind | N | reference |
|-----|-------|------|---:|-----------|
| v03_alex | Alexandria 3D PBE | crystal | 31,211 | dft_bandgap_eV |
| v04_cccbdb | CCCBDB molecules | molecule | 1,320 | dft_bandgap_eV |
| v05_interface | Interface slabs (optB88vdW) | interface | 433 | dft_bandgap_eV |
| v06_surface | Surface slabs (PBE) | surface | 466 | dft_bandgap_eV |
| v07_vacancy | Vacancy defects | defect | 444 | — |
| v08_supercon | Alexandria supercon candidates | supercon | 4,827 | — |
| v09_1d | Alexandria 1D PBE | low_dim | 8,636 | dft_bandgap_eV |
| v10_2d | Alexandria 2D PBE | low_dim | 79,903 | dft_bandgap_eV |
| v11_alexwz | Alexandria 3D PBE, no Z≤65 filter | crystal | 40,807¹ | band_gap_ind |
| v12_all | Alexandria 3D PBE, no filters (complete 100/100 shards) | crystal | 2,138,447² | band_gap_ind |

¹ v11 N = 40,807 is the SK-finite count out of 115,535 hull entries attempted (39% success rate; rest is SK's chemical ceiling on lanthanides + Z>65, not random failures). Reference is Alexandria PBE `band_gap_ind` indirect gap.

² v12 N = 2,138,447 is the finite-SK count after a `sk_bandgap ≤ 20 eV` clip (drops 122 numerical-overflow outliers up to ~6×10⁶ eV that contribute >99% of the raw RMSE). v12 SK is **complete**: 2,503,043 per-id JSONs produced of 4,489,295 attempted (56%), the remaining 44% is silent dropout on the SK chemical ceiling (lanthanides + Z>65 + noble gases).

## Headline metrics (see `csv/summary_table.csv`)

| dataset | N | sk_mean_eV | sk_median_eV | frac_sk_metal | ref_mean_eV | MAE_eV | RMSE_eV | pearson_r |
|---|---|---|---|---|---|---|---|---|
| v03_alex | 31211 | 1.544 | 0.006 | 0.622 | 1.215 | 0.930 | 1.649 | 0.807 |
| v04_cccbdb | 1320 | 7.445 | 6.308 | 0.000 | 6.725 | 2.519 | 3.516 | 0.648 |
| v05_interface | 433 | 1.432 | 1.412 | 0.169 | 0.429 | 1.013 | 1.259 | 0.730 |
| v06_surface | 466 | 1.673 | 1.176 | 0.328 | 0.773 | 0.973 | 1.595 | 0.746 |
| v07_vacancy | 444 | 0.157 | 0.001 | 0.912 | — | — | — | — |
| v08_supercon | 4827 | 0.019 | 0.004 | 0.973 | — | — | — | — |
| v09_1d | 8636 | 1.869 | 0.308 | 0.321 | 1.088 | 0.989 | 1.701 | 0.879 |
| v10_2d | 79903 | 1.161 | 0.017 | 0.592 | 0.671 | 0.621 | 1.328 | 0.890 |
| v11_alexwz | 40807 | 1.352 | 0.006 | 0.714 | 1.008 | 1.039 | 2.029 | 0.712 |
| v12_all | 2138447 | 0.238 | 0.005 | 0.907 | 0.150 | 0.206 | 0.827 | 0.671 |

## Figures

- `plots/dataset_overview.png` — Per-dataset N, metallic fraction, SK gap median±IQR
- `plots/gap_distributions.png` — SK gap histograms (reference overlaid where available)
- `plots/parity_grid.png` — SK vs reference DFT gap, hexbin parity
- `plots/residual_distributions.png` — SK − reference residual densities
- `plots/error_summary.png` — MAE / RMSE / Pearson-r vs reference
- `plots/dos_average_grid.png` — Mean SlakoNet DOS per dataset (Fermi-aligned)
- `plots/v08_tc_correlations.png` — v08 supercon: Tc vs SK gap, DOS(E_F), λ

## Notes

- Reference gap for v03 is the paired subset of Alexandria PBE entries that also have an ALIGNN prediction (~31 k of 48 k). The per-row scalars are tracked in `../slako_v03_alex/analysis/csv/summary.csv` (regenerated from `alignn/alignn_v03_alex/pbe_mbj_opt_analysis/merged_predictions.json`); the off-repo aggregator additionally uses the cached `results/sk_scalars.json` (gitignored, kept local) to avoid re-walking all 31 k per-id JSONs.
- v04 assumes the CCCBDB HOMO/LUMO columns are in **Hartree** (the `hl_gap_hartree_eV` column in that project's `summary.csv`). The alternative eV assumption yields 0.1–0.3 eV gaps that are clearly wrong relative to SlakoNet's 4–15 eV predictions.
- v05/v06 clip the DFT reference to ≥ 0 (v06 subtracts `surf_cbm − surf_vbm`, which goes slightly negative for metals — treated as gap = 0).
- v07 (vacancy) and v08 (supercon) have **no DFT band-gap reference**, so they do not appear in the parity / residual / error plots — only in the distribution and DOS grids.
- Metallic threshold used for the `frac_sk_metal` coverage stat and the overview bars: SlakoNet gap < **0.05 eV**, applied uniformly across all 10 datasets. This matches the cross-method / three-way convention used everywhere else in the repo (the narrative "SK 91.2% metallic on vacancies", "v12 91% metals"). The legacy version of this table used 0.10 eV for v03–v10 and 0.05 eV for v11/v12; this rebuild removes that inconsistency, which shifts the v03–v10 `frac_sk_metal` values slightly downward (e.g. v09 0.398 → 0.321, v10 0.624 → 0.592).


## Metal/non-metal threshold conventions

Three thresholds for "metal" appear across this repo, applied to different things:

- `pbe_ref == 0` (strictest, DFT-side): Alexandria flags metallic structures with an exact-zero gap. Used in the v03 bootstrap CI artifact (`../../alignn/alignn_v03_alex/pbe_mbj_opt_analysis/bootstrap_ci.json`) for the most conservative ground-truth metal split, since the ranking-stability claim should be insensitive to a tolerance choice.
- `pbe_ref <= 0.05` (conventional, DFT-side): standard ML stratification threshold that absorbs sub-50 meV DFT noise (sub-zero or near-zero gaps that are physically metallic) into the metal class. Used in the manuscript Section 6 v11 stratified MAE (80,695 metals / 34,840 non-metals).
- `pred <= 0.05` (model-side): applied to *predictions*, not the reference. Reports "what fraction of structures the model predicted as metallic" with a small tolerance for near-zero outputs. Used in the `frac_sk_metal` column of `csv/summary_table.csv` (this rebuild standardizes it at 0.05 across all 10 datasets) and in the top-level README headline tables.

The two are not interchangeable: `== 0` and `<= 0.05` classify the *ground truth* (DFT label); the `frac_sk_metal` 0.05 cut classifies the *prediction*.

