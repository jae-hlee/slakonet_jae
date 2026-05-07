# ALIGNN vs SlakoNet band-gap predictions — analysis (alignn_v1_pbe)

**ALIGNN:** `mp_gappbe_alignn` (pretrained PBE-gap model) run in this directory.
**SlakoNet:** predictions reused from `../../../../slakonet/slako_v03_alex/results/sk_scalars.json` (gitignored, kept local).
**Reference:** Alexandria PBE `band_gap_ind` (indirect gap) on `e_above_hull==0`,
elements with Z≤65. Materials are matched by `mat_id`; ALIGNN entries without a
SlakoNet counterpart are excluded from this comparison.

## Dataset composition (merged set)

- Merged structures: **31211**
- ALIGNN entries without SK match (excluded): **17553** (~99.9 % are lanthanide-containing cells that SlakoNet silently rejects, not recoverable by rerun — see `../../pbe_mbj_opt_analysis/analysis.md`)
- SK entries without ALIGNN match (excluded): **0**
- PBE metals (gap = 0): **15119**
- PBE non-metals (gap > 0): **16092**
- ALIGNN raw predictions < 0 (clamped to 0): **4541**

## Regression metrics vs PBE indirect gap

### All structures (N = 31211)

| Model | MAE (eV) | RMSE (eV) | R² | Mean signed err |
|-------|---------:|----------:|---:|----------------:|
| SlakoNet            | 0.9299 | 1.6487 | -0.0082 | +0.3285 |
| ALIGNN (clamped ≥0) | 0.1928 | 0.4631 | 0.9204 | +0.0172 |

### Non-metals only (N = 16092)

| Model | MAE (eV) | RMSE (eV) | R² | Mean signed err |
|-------|---------:|----------:|---:|----------------:|
| SlakoNet            | 1.7810 | 2.2909 | -1.0685 | +0.6145 |
| ALIGNN (clamped ≥0) | 0.2739 | 0.4897 | 0.9055 | -0.0666 |

### Metals only (N = 15119)

| Model | MAE (eV) | RMSE (eV) | Mean signed err |
|-------|---------:|----------:|----------------:|
| SlakoNet            | 0.0241 | 0.1596 | +0.0241 |
| ALIGNN (clamped ≥0) | 0.1064 | 0.4331 | +0.1064 |

## Metal / non-metal classification (threshold 0.1 eV)

| Model    | TN | FP | FN | TP | Accuracy | Precision | Recall | F1 |
|----------|---:|---:|---:|---:|---------:|----------:|-------:|---:|
| SlakoNet | 14715 | 404 | 4909 | 11183 | 0.8298 | 0.9651 | 0.6949 | 0.8080 |
| ALIGNN   | 13283 | 1836 | 777 | 15315 | 0.9163 | 0.8930 | 0.9517 | 0.9214 |

## Head-to-head (per structure)

- **SlakoNet closer to PBE:** 10054 (32.2%)
- **ALIGNN closer to PBE:**   21157 (67.8%)

## Error vs PBE gap (non-metals, binned)

| PBE bin (eV) | N | SK MAE | SK ME | ALIGNN MAE | ALIGNN ME |
|---|--:|--:|--:|--:|--:|
| 0.0–0.5 | 1789 | 0.509 | +0.278 | 0.281 | +0.171 |
| 0.5–1.0 | 1901 | 0.919 | +0.288 | 0.294 | +0.008 |
| 1.0–1.5 | 1985 | 1.229 | +0.326 | 0.288 | -0.057 |
| 1.5–2.0 | 1972 | 1.562 | +0.270 | 0.300 | -0.113 |
| 2.0–2.5 | 1820 | 1.895 | +0.250 | 0.273 | -0.094 |
| 2.5–3.0 | 1501 | 2.231 | +0.389 | 0.284 | -0.124 |
| 3.0–4.0 | 2418 | 2.394 | +0.771 | 0.248 | -0.124 |
| 4.0–5.0 | 1561 | 2.594 | +1.331 | 0.234 | -0.131 |
| 5.0–7.0 | 1047 | 3.183 | +2.116 | 0.264 | -0.171 |
| 7.0–10.0 | 98 | 5.252 | +4.787 | 0.200 | -0.166 |

## Plots

| File | Content |
|------|---------|
| `plots/parity_three_way.png` | Density parity: SK vs PBE, ALIGNN vs PBE, ALIGNN vs SK (all) |
| `plots/parity_three_way_nonmetals.png` | Same three panels, non-metals only |
| `plots/residuals.png` | Residual histograms for SlakoNet and ALIGNN |
| `plots/confusion_sk_vs_alignn.png` | Metal/non-metal confusion matrices, both models |
| `plots/head_to_head_error.png` | Per-structure \|SK−PBE\| vs \|ALIGNN−PBE\| |
| `plots/gap_distribution.png` | Non-metal gap distributions: PBE / SK / ALIGNN |
| `plots/mae_vs_pbe_gap.png` | MAE & mean signed error binned by PBE gap |
| `plots/error_vs_gap_hex_sk.png` | SlakoNet signed residual vs PBE gap |
| `plots/error_vs_gap_hex_alignn.png` | ALIGNN signed residual vs PBE gap |

## Artifacts

- `metrics.json` — all numeric results used above
- `worst_cases_alignn.csv` — top-50 worst ALIGNN mispredictions
- `worst_cases_sk.csv` — top-50 worst SlakoNet mispredictions
