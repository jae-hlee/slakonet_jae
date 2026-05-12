# v04 e_form analysis: Opt arm (jv_formation_energy_peratom_alignn)

Pretrained model trained on **JARVIS/OptB88vdW** formation energy (eV/atom). Inference on the **CCCBDB molecules (Z<=65)** set, N = 1,333. **No DFT e_form reference** ships with this dataset. Analysis is distribution-only + cross-arm comparison.

## Distribution of predicted e_form (eV/atom)

| stat | value |
|---|---|
| N | 1,333 |
| min | -3.279 |
| p05 | -1.102 |
| median | +0.121 |
| mean | +0.193 |
| p95 | +1.517 |
| max | +3.213 |
| std | 0.773 |
| n(pred < 0) | 516 (38.7%) |
| n(pred > 0) | 817 (61.3%) |

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
| 1 elem | 40 | +1.314 | +1.052 | 0.635 |
| 2 elem | 557 | +0.300 | +0.271 | 0.959 |
| 3 elem | 585 | +0.092 | +0.039 | 0.538 |
| 4 elem | 149 | -0.103 | -0.102 | 0.399 |
| 5+ elem | 2 | -0.220 | -0.220 | 0.073 |

## Per-element mean prediction (top 20 most-negative, count >= 10)

| element | count | mean pred | median pred | std |
|---|---|---|---|---|
| F | 181 | -0.624 | -0.489 | 0.944 |
| Ca | 10 | -0.286 | -0.486 | 1.195 |
| Mg | 16 | -0.196 | -0.579 | 1.243 |
| O | 467 | -0.034 | -0.073 | 0.519 |
| Cl | 194 | +0.072 | +0.036 | 0.648 |
| Al | 23 | +0.101 | +0.233 | 1.316 |
| Na | 18 | +0.123 | +0.027 | 0.948 |
| C | 882 | +0.171 | +0.066 | 0.592 |
| H | 933 | +0.193 | +0.104 | 0.515 |
| Zn | 13 | +0.213 | +0.288 | 0.887 |
| Si | 34 | +0.239 | +0.050 | 1.283 |
| S | 136 | +0.242 | +0.136 | 0.709 |
| B | 48 | +0.247 | +0.242 | 1.236 |
| Br | 52 | +0.295 | +0.255 | 0.617 |
| D | 16 | +0.300 | +0.403 | 0.518 |
| Be | 16 | +0.307 | +0.350 | 1.363 |
| Li | 22 | +0.333 | +0.571 | 0.926 |
| P | 47 | +0.342 | +0.427 | 1.207 |
| N | 318 | +0.428 | +0.254 | 0.602 |
| Ge | 15 | +0.435 | +0.602 | 1.227 |

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
- Cross-arm comparison with `eform_v1_pbe/analysis/` is the primary signal here (no DFT reference).
