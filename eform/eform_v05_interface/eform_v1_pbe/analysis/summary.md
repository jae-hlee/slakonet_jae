# v05 e_form analysis: PBE arm (mp_e_form_alignn)

Pretrained model trained on **MP/PBE** formation energy (eV/atom). Inference on the **JARVIS interface_db slabs (Z<=65)** set, N = 587. **No DFT e_form reference** ships with this dataset. Analysis is distribution-only + cross-arm comparison.

## Distribution of predicted e_form (eV/atom)

| stat | value |
|---|---|
| N | 587 |
| min | -1.649 |
| p05 | -1.215 |
| median | -0.388 |
| mean | -0.472 |
| p95 | +0.021 |
| max | +0.579 |
| std | 0.375 |
| n(pred < 0) | 551 (93.9%) |
| n(pred > 0) | 36 (6.1%) |

## Cross-arm comparison

On N = 587 records (positional pairing):

| metric | value |
|---|---|
| PBE - Opt mean (signed) | -0.0295 eV/atom |
| PBE - Opt median | -0.0122 eV/atom |
| mean \|PBE - Opt\| | 0.0571 eV/atom |
| max \|PBE - Opt\| | 0.2669 eV/atom |
| Pearson r (PBE vs Opt) | 0.9822 |
| fraction \|diff\| <= 0.05 | 58.6% |
| fraction \|diff\| <= 0.10 | 79.7% |
| fraction \|diff\| <= 0.50 | 100.0% |

Top 25 cross-arm disagreements (full table at `csv/cross_arm_top_disagreements.csv`):

| id | formula | PBE | Opt | PBE - Opt |
|---|---|---|---|---|
| Interface-JVASP-8003_JVASP-95_film_miller_1_1_0_su… |  | -0.869 | -0.603 | -0.267 |
| Interface-JVASP-8003_JVASP-95_film_miller_1_1_0_su… |  | -0.869 | -0.603 | -0.267 |
| Interface-JVASP-1702_JVASP-95_film_miller_1_1_0_su… |  | -0.950 | -0.696 | -0.253 |
| Interface-JVASP-1702_JVASP-95_film_miller_1_1_0_su… |  | -0.950 | -0.696 | -0.253 |
| Interface-JVASP-1195_JVASP-8003_film_miller_1_1_0_… |  | -1.310 | -1.071 | -0.239 |
| Interface-JVASP-1702_JVASP-1195_film_miller_1_1_0_… |  | -1.357 | -1.147 | -0.210 |
| Interface-JVASP-30_JVASP-1195_film_miller_0_0_1_su… |  | -1.140 | -0.943 | -0.196 |
| Interface-JVASP-7923_JVASP-30_film_miller_1_1_0_su… |  | -1.216 | -1.034 | -0.183 |
| Interface-JVASP-1195_JVASP-36123_film_miller_1_1_0… |  | -1.209 | -1.028 | -0.181 |
| Interface-JVASP-8003_JVASP-30_film_miller_1_1_0_su… |  | -0.667 | -0.486 | -0.181 |
| Interface-JVASP-1702_JVASP-30_film_miller_1_1_0_su… |  | -0.743 | -0.564 | -0.179 |
| Interface-JVASP-1002_JVASP-8003_film_miller_0_0_1_… |  | -0.433 | -0.256 | -0.177 |
| Interface-JVASP-1174_JVASP-8003_film_miller_0_0_1_… |  | -0.625 | -0.451 | -0.175 |
| Interface-JVASP-105410_JVASP-8003_film_miller_1_1_… |  | -0.468 | -0.294 | -0.173 |
| Interface-JVASP-105410_JVASP-8003_film_miller_1_1_… |  | -0.468 | -0.294 | -0.173 |
| Interface-JVASP-7923_JVASP-8169_film_miller_1_1_0_… |  | -1.137 | -0.964 | -0.173 |
| Interface-JVASP-8185_JVASP-95_film_miller_1_1_0_su… |  | -0.573 | -0.401 | -0.172 |
| Interface-JVASP-8185_JVASP-95_film_miller_1_1_0_su… |  | -0.573 | -0.401 | -0.172 |
| Interface-JVASP-8003_JVASP-1192_film_miller_1_1_0_… |  | -0.798 | -0.627 | -0.171 |
| Interface-JVASP-1702_JVASP-17700_film_miller_1_1_0… |  | -0.512 | -0.342 | -0.170 |
| Interface-JVASP-1702_JVASP-17700_film_miller_1_1_0… |  | -0.512 | -0.342 | -0.170 |
| Interface-JVASP-1002_JVASP-8003_film_miller_1_1_0_… |  | -0.451 | -0.282 | -0.169 |
| Interface-JVASP-8003_JVASP-7671_film_miller_1_1_0_… |  | -0.710 | -0.543 | -0.167 |
| Interface-JVASP-1702_JVASP-8184_film_miller_1_1_0_… |  | -0.670 | -0.504 | -0.166 |
| Interface-JVASP-1702_JVASP-8184_film_miller_1_1_0_… |  | -0.670 | -0.504 | -0.166 |


## Files

- `csv/distribution.csv` — descriptive stats
- `csv/cross_arm_top_disagreements.csv` — top 50 records by |PBE - Opt|
- `plots/prediction_histogram.png` — this arm's predictions
- `plots/cross_arm_comparison.png` — PBE-vs-Opt scatter + histogram + CDF

## Caveats

- Interface slabs: vacuum direction, OOD for 3D-bulk-trained ALIGNN.
- 587 records but only 433 unique jids (dataset has dup-jid quirk).
- No `formula` in output JSON; per-element / cardinality analysis skipped.
- Cross-arm comparison with `eform_v2_opt/analysis/` is the primary signal here (no DFT reference).
