# v11_alexwz: SlakoNet vs ALIGNN vs DFT (cross-method)

SK v11 effectively complete on Rockfish /data; ALIGNN v11 predictions re-sourced from atomgptlab (2026-05-14) after the prior copy went missing. DFT reference is Alexandria PBE `band_gap_ind`.

## SK run accounting

Of 115,535 hull entries attempted:
- 45,203 (39.1%) per-id JSONs produced
  - 40,807 (35.3%) finite `sk_bandgap` → used here
  - 4,396 (3.8%) `sk_bandgap = inf` (gap-too-wide overflow; dropped)
- 70,332 (60.9%) no JSON produced — silent dropout on lanthanides + heavies (Z>65), the documented SK ceiling on this dataset

ALIGNN v11 ran cleanly on all 115,535 hull entries (no chemical ceiling on its side); three-way scope is therefore bounded by SK's 40,807.

## Pairwise metrics

Three-way comparison on the full 40,807-entry SK-finite ∩ ALIGNN-finite intersection (now at SK-finite scale, not 6,781 as in the prior writeup):

| comparison | N | MAE (eV) | RMSE | ME (bias) | metal/gap acc. |
|---|---|---|---|---|---|
| DFT_PBE_vs_SK | 40,807 | 1.039 | 2.029 | +0.344 | 78.518% |
| DFT_PBE_vs_ALIGNN | 40,807 | 0.174 | 0.427 | +0.016 | 90.038% |
| SK_vs_ALIGNN | 40,807 | 1.086 | 2.068 | -0.328 | 72.399% |
| DFT_PBE_vs_ALIGNN (full ALIGNN 115k) | 115,535 | 0.168 | 0.476 | +0.024 | 89.130% |

ALIGNN headline MAE on the 40,807 SK-finite intersection is 0.174 eV vs 0.168 eV on the full 115,535 — restricting to SK-finite mat_ids shifts ALIGNN's MAE by 6 meV, i.e. the SK chemical ceiling (lanthanides + Z>65) is not where ALIGNN's worst predictions live. **ALIGNN wins by ~6.0x on bulk crystals** (MAE 1.04 vs 0.17), the same factor seen at the 6,781 scale.

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
