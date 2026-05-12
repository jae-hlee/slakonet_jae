# v07 e_form analysis: PBE arm (mp_e_form_alignn)

Pretrained model trained on **MP/PBE** formation energy (eV/atom). Inference on the **JARVIS vacancy_db defects (Z<=65)** set, N = 470. **No DFT e_form reference** ships with this dataset. Analysis is distribution-only + cross-arm comparison.

## Distribution of predicted e_form (eV/atom)

| stat | value |
|---|---|
| N | 470 |
| min | -3.800 |
| p05 | -1.959 |
| median | -0.384 |
| mean | -0.617 |
| p95 | +0.232 |
| max | +0.707 |
| std | 0.787 |
| n(pred < 0) | 368 (78.3%) |
| n(pred > 0) | 102 (21.7%) |

## Cross-arm comparison

On N = 470 records (positional pairing):

| metric | value |
|---|---|
| PBE - Opt mean (signed) | -0.0046 eV/atom |
| PBE - Opt median | +0.0124 eV/atom |
| mean \|PBE - Opt\| | 0.0943 eV/atom |
| max \|PBE - Opt\| | 0.4875 eV/atom |
| Pearson r (PBE vs Opt) | 0.9845 |
| fraction \|diff\| <= 0.05 | 49.4% |
| fraction \|diff\| <= 0.10 | 67.0% |
| fraction \|diff\| <= 0.50 | 100.0% |

Top 25 cross-arm disagreements (full table at `csv/cross_arm_top_disagreements.csv`):

| id | formula | PBE | Opt | PBE - Opt |
|---|---|---|---|---|
| JVASP-94296_O_d_36 |  | -1.397 | -0.910 | -0.487 |
| JVASP-94296_C_a_0 |  | -1.379 | -0.910 | -0.468 |
| JVASP-53976_O_f_16 |  | -1.689 | -1.269 | -0.420 |
| JVASP-51876_O_f_16 |  | -1.693 | -1.273 | -0.419 |
| JVASP-94344_O_f_18 |  | -1.601 | -2.016 | +0.415 |
| JVASP-22523_O_c_32 |  | -1.727 | -1.327 | -0.400 |
| JVASP-664_S_h_16 |  | -1.282 | -0.886 | -0.396 |
| JVASP-664_Mo_e_0 |  | -1.285 | -0.891 | -0.393 |
| JVASP-60477_Ru_d_0 |  | -0.975 | -0.586 | -0.389 |
| JVASP-90143_O_c_24 |  | -2.044 | -1.658 | -0.386 |
| JVASP-22523_C_a_0 |  | -1.665 | -1.281 | -0.384 |
| JVASP-60497_Fe_c_0 |  | -1.039 | -0.659 | -0.380 |
| JVASP-60497_Cl_k_8 |  | -1.064 | -0.685 | -0.379 |
| JVASP-6742_Fe_c_0 |  | -1.039 | -0.660 | -0.379 |
| JVASP-94344_Mo_c_0 |  | -1.610 | -1.985 | +0.376 |
| JVASP-60477_Cl_k_8 |  | -0.947 | -0.575 | -0.373 |
| JVASP-13632_Ti_c_0 |  | -1.985 | -1.613 | -0.372 |
| JVASP-13632_Cl_k_8 |  | -1.995 | -1.629 | -0.366 |
| JVASP-13600_Fe_c_4 |  | -1.821 | -1.470 | -0.351 |
| JVASP-32_Al_c_0 |  | -3.018 | -2.670 | -0.349 |
| JVASP-8065_O_f_24 |  | -2.083 | -1.763 | -0.320 |
| JVASP-93342_O_f_24 |  | -2.082 | -1.763 | -0.319 |
| JVASP-18424_Cu_a_0 |  | -0.445 | -0.131 | -0.315 |
| JVASP-771_Ti_a_0 |  | -1.632 | -1.322 | -0.309 |
| JVASP-8065_Ge_a_0 |  | -2.052 | -1.745 | -0.306 |


## Dataset `ef` field (vacancy formation energy, eV per defect)

Carried for cross-method reference only; **not parity-comparable to ALIGNN's per-atom formation energy.**

| stat | value |
|---|---|
| N | 470 |
| min | +0.116 |
| p05 | +0.458 |
| median | +2.207 |
| mean | +2.999 |
| p95 | +7.585 |
| max | +11.867 |
| std | 2.351 |

## Files

- `csv/distribution.csv` — descriptive stats
- `csv/cross_arm_top_disagreements.csv` — top 50 records by |PBE - Opt|
- `plots/prediction_histogram.png` — this arm's predictions
- `plots/cross_arm_comparison.png` — PBE-vs-Opt scatter + histogram + CDF

## Caveats

- v07 carries `ef` (vacancy formation E, eV/defect) — NOT comparable to ALIGNN's eV/atom.
- All `formula` fields are None; element info encoded in id only.
- Cross-arm comparison with `eform_v2_opt/analysis/` is the primary signal here (no DFT reference).
