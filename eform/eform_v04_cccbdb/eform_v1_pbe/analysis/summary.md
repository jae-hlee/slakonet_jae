# v04 e_form analysis: PBE arm (mp_e_form_alignn)

Pretrained model trained on **MP/PBE** formation energy (eV/atom). Inference on the **CCCBDB molecules (Z<=65)** set, N = 1,333. **No DFT e_form reference** ships with this dataset. Analysis is distribution-only + cross-arm comparison.

## Distribution of predicted e_form (eV/atom)

| stat | value |
|---|---|
| N | 1,333 |
| min | -3.511 |
| p05 | -1.713 |
| median | -0.279 |
| mean | -0.417 |
| p95 | +0.354 |
| max | +1.663 |
| std | 0.638 |
| n(pred < 0) | 1,071 (80.3%) |
| n(pred > 0) | 262 (19.7%) |

## Cross-arm comparison

On N = 1,333 records (positional pairing):

| metric | value |
|---|---|
| PBE - Opt mean (signed) | -0.6101 eV/atom |
| PBE - Opt median | -0.4796 eV/atom |
| mean \|PBE - Opt\| | 0.6118 eV/atom |
| max \|PBE - Opt\| | 2.9202 eV/atom |
| Pearson r (PBE vs Opt) | 0.8661 |
| fraction \|diff\| <= 0.05 | 0.4% |
| fraction \|diff\| <= 0.10 | 1.1% |
| fraction \|diff\| <= 0.50 | 51.7% |

Top 25 cross-arm disagreements (full table at `csv/cross_arm_top_disagreements.csv`):

| id | formula | PBE | Opt | PBE - Opt |
|---|---|---|---|---|
| cc-1023 | Si2 | +0.240 | +3.161 | -2.920 |
| cc-977 | SiP+ | +0.118 | +2.273 | -2.155 |
| cc-1132 | Be2 | +0.555 | +2.671 | -2.116 |
| cc-361 | SiC | +1.104 | +3.213 | -2.109 |
| cc-1019 | Ge2 | +0.187 | +2.091 | -1.905 |
| cc-1266 | Al2 | +0.177 | +2.081 | -1.904 |
| cc-1282 | AlGa | +0.082 | +1.956 | -1.874 |
| cc-692 | CS | +0.508 | +2.380 | -1.873 |
| cc-996 | BS+ | -0.279 | +1.493 | -1.771 |
| cc-79 | CS2 | -0.029 | +1.734 | -1.763 |
| cc-995 | BS- | -0.206 | +1.553 | -1.759 |
| cc-1313 | BeMg | +0.551 | +2.307 | -1.756 |
| cc-1007 | CP- | +0.802 | +2.557 | -1.755 |
| cc-964 | SiN- | -0.110 | +1.626 | -1.735 |
| cc-976 | SiP- | +0.061 | +1.788 | -1.726 |
| cc-1236 | AlN | -0.070 | +1.631 | -1.701 |
| cc-1008 | CP+ | +0.876 | +2.576 | -1.700 |
| cc-1243 | GaN | +0.257 | +1.946 | -1.689 |
| cc-953 | BC+ | +0.954 | +2.611 | -1.657 |
| cc-1278 | BeN+ | +0.121 | +1.774 | -1.653 |
| cc-1225 | AlP | -0.206 | +1.433 | -1.639 |
| cc-1021 | N3- | +0.131 | +1.743 | -1.612 |
| cc-1177 | SF- | -1.245 | +0.350 | -1.595 |
| cc-991 | Cu2 | +0.350 | +1.924 | -1.575 |
| cc-1219 | CaS | -1.641 | -0.078 | -1.564 |


## Composition cardinality stratification

By number of distinct elements per formula.

| cardinality | N | mean pred | median pred | std |
|---|---|---|---|---|
| 1 elem | 40 | +0.225 | +0.140 | 0.262 |
| 2 elem | 557 | -0.435 | -0.183 | 0.821 |
| 3 elem | 585 | -0.411 | -0.311 | 0.462 |
| 4 elem | 149 | -0.544 | -0.486 | 0.370 |
| 5+ elem | 2 | -0.576 | -0.576 | 0.156 |

## Per-element mean prediction (top 20 most-negative, count >= 10)

| element | count | mean pred | median pred | std |
|---|---|---|---|---|
| F | 181 | -1.321 | -1.188 | 0.812 |
| Ca | 10 | -1.269 | -1.569 | 0.919 |
| Mg | 16 | -1.130 | -1.373 | 0.937 |
| Al | 23 | -0.838 | -0.240 | 0.972 |
| Si | 34 | -0.718 | -0.457 | 0.943 |
| Be | 16 | -0.705 | -0.712 | 1.037 |
| O | 467 | -0.605 | -0.495 | 0.461 |
| B | 48 | -0.595 | -0.297 | 1.010 |
| Ge | 15 | -0.580 | -0.531 | 0.995 |
| Na | 18 | -0.569 | -0.633 | 0.701 |
| P | 47 | -0.567 | -0.213 | 0.899 |
| Zn | 13 | -0.552 | -0.087 | 0.788 |
| Ga | 11 | -0.542 | -0.033 | 0.866 |
| S | 136 | -0.513 | -0.281 | 0.604 |
| Li | 22 | -0.504 | -0.202 | 0.713 |
| Cl | 194 | -0.485 | -0.391 | 0.567 |
| As | 13 | -0.421 | +0.052 | 0.910 |
| D | 16 | -0.388 | -0.303 | 0.405 |
| H | 933 | -0.303 | -0.259 | 0.379 |
| C | 882 | -0.301 | -0.260 | 0.461 |

Full per-element table at `csv/per_element_pred.csv`.

## Files

- `csv/distribution.csv` — descriptive stats
- `csv/per_element_pred.csv` — per-element mean prediction
- `csv/cardinality_pred.csv` — by composition cardinality
- `csv/cross_arm_top_disagreements.csv` — top 50 records by |PBE - Opt|
- `plots/prediction_histogram.png` — this arm's predictions
- `plots/per_element_pred.png` — most-negative-predicted elements
- `plots/composition_cardinality.png` — mean pred by # elements
- `plots/cross_arm_comparison.png` — PBE-vs-Opt scatter + histogram + CDF

## Caveats

- Molecular: far-OOD for crystal-trained ALIGNN. No DFT e_form ref.
- Cross-arm comparison with `eform_v2_opt/analysis/` is the primary signal here (no DFT reference).
