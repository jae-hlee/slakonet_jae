# v06 e_form analysis: PBE arm (mp_e_form_alignn)

Pretrained model trained on **MP/PBE** formation energy (eV/atom). Inference on the **JARVIS surface_db slabs (Z<=65)** set, N = 487. **No DFT e_form reference** ships with this dataset. Analysis is distribution-only + cross-arm comparison.

## Distribution of predicted e_form (eV/atom)

| stat | value |
|---|---|
| N | 487 |
| min | -4.152 |
| p05 | -2.361 |
| median | -0.328 |
| mean | -0.557 |
| p95 | +0.254 |
| max | +0.859 |
| std | 0.820 |
| n(pred < 0) | 354 (72.7%) |
| n(pred > 0) | 133 (27.3%) |

## Cross-arm comparison

On N = 487 records (positional pairing):

| metric | value |
|---|---|
| PBE - Opt mean (signed) | -0.0221 eV/atom |
| PBE - Opt median | +0.0012 eV/atom |
| mean \|PBE - Opt\| | 0.0809 eV/atom |
| max \|PBE - Opt\| | 0.7052 eV/atom |
| Pearson r (PBE vs Opt) | 0.9899 |
| fraction \|diff\| <= 0.05 | 49.9% |
| fraction \|diff\| <= 0.10 | 72.9% |
| fraction \|diff\| <= 0.50 | 99.4% |

Top 25 cross-arm disagreements (full table at `csv/cross_arm_top_disagreements.csv`):

| id | formula | PBE | Opt | PBE - Opt |
|---|---|---|---|---|
| Surface-JVASP-148851_miller_1_1_0_thickness_16_VAS… | ZnSO4 | -1.838 | -1.133 | -0.705 |
| Surface-JVASP-149059_miller_1_1_0_thickness_16_VAS… | ZnSO4 | -1.836 | -1.132 | -0.704 |
| Surface-JVASP-46672_miller_1_0_0_thickness_16_VASP… | Mn4OF6 | -2.700 | -2.156 | -0.545 |
| Surface-JVASP-135879_miller_1_1_0_thickness_16_VAS… | BaSO4 | -2.563 | -2.076 | -0.486 |
| Surface-JVASP-151876_miller_1_0_0_thickness_16_VAS… | AlPO4 | -3.070 | -2.601 | -0.469 |
| Surface-JVASP-14837_miller_1_1_0_thickness_16_VASP… | V | +0.591 | +0.159 | +0.431 |
| Surface-JVASP-109392_miller_1_1_0_thickness_16_VAS… | Li2SO4 | -2.145 | -1.749 | -0.396 |
| Surface-JVASP-290_miller_1_0_0_thickness_16_VASP_P… | SnS2 | -0.723 | -0.360 | -0.363 |
| Surface-JVASP-36403_miller_0_0_1_thickness_16_VASP… | SiSn | +0.401 | +0.749 | -0.348 |
| Surface-JVASP-10591_miller_0_0_1_thickness_16_VASP… | ZnS | -0.996 | -0.665 | -0.331 |
| Surface-JVASP-8003_miller_0_0_1_thickness_16_VASP_… | CdS | -0.907 | -0.610 | -0.297 |
| Surface-JVASP-8003_miller_0_0_1_thickness_16_VASP_PBE | CdS | -0.907 | -0.612 | -0.295 |
| Surface-JVASP-95_miller_1_0_0_thickness_16_VASP_PB… | CdS | -0.908 | -0.616 | -0.292 |
| Surface-JVASP-23972_miller_1_0_0_thickness_16_VASP… | H4NF | -1.155 | -0.863 | -0.292 |
| Surface-JVASP-143623_miller_1_0_0_thickness_16_VAS… | Nd4FeS6O | -2.301 | -2.019 | -0.282 |
| Surface-JVASP-110096_miller_1_0_0_thickness_16_VAS… | SrClO | -1.986 | -1.719 | -0.267 |
| Surface-JVASP-63938_miller_1_1_0_thickness_16_VASP… | BaNaBr | -1.180 | -0.916 | -0.264 |
| Surface-JVASP-111073_miller_1_1_0_thickness_16_VAS… | CuCl | -0.696 | -0.434 | -0.263 |
| Surface-JVASP-1201_miller_1_1_0_thickness_16_VASP_… | CuCl | -0.697 | -0.435 | -0.262 |
| Surface-JVASP-95_miller_1_1_0_thickness_16_VASP_PB… | CdS | -0.838 | -0.579 | -0.259 |
| Surface-JVASP-8003_miller_1_1_0_thickness_16_VASP_… | CdS | -0.872 | -0.614 | -0.259 |
| Surface-JVASP-8003_miller_1_1_0_thickness_16_VASP_… | CdS | -0.871 | -0.613 | -0.258 |
| Surface-JVASP-1702_miller_1_1_0_thickness_16_VASP_PBE | ZnS | -0.979 | -0.727 | -0.252 |
| Surface-JVASP-95_miller_1_1_0_thickness_16_VASP_PBE | CdS | -0.832 | -0.581 | -0.251 |
| Surface-JVASP-57104_miller_1_0_0_thickness_16_VASP… | ZnS | -1.041 | -0.791 | -0.251 |


## Composition cardinality stratification

By number of distinct elements per formula.

| cardinality | N | mean pred | median pred | std |
|---|---|---|---|---|
| 1 elem | 95 | +0.091 | +0.057 | 0.099 |
| 2 elem | 237 | -0.456 | -0.328 | 0.598 |
| 3 elem | 141 | -1.054 | -0.755 | 0.975 |
| 4 elem | 12 | -1.481 | -1.266 | 0.929 |
| 5+ elem | 2 | -2.762 | -2.762 | 0.282 |

## Per-element mean prediction (top 20 most-negative, count >= 5)

| element | count | mean pred | median pred | std |
|---|---|---|---|---|
| F | 12 | -3.292 | -3.675 | 0.935 |
| O | 42 | -2.372 | -2.223 | 0.899 |
| Cl | 13 | -1.904 | -2.418 | 0.977 |
| Y | 7 | -1.630 | -0.965 | 1.401 |
| Sr | 11 | -1.452 | -1.003 | 0.852 |
| Ba | 18 | -1.430 | -1.006 | 0.938 |
| Sc | 9 | -1.361 | -1.019 | 1.034 |
| Mn | 10 | -1.326 | -1.130 | 0.879 |
| S | 44 | -1.288 | -1.021 | 0.638 |
| Ca | 12 | -1.169 | -0.829 | 0.735 |
| K | 12 | -1.167 | -1.041 | 0.588 |
| Pr | 7 | -1.077 | +0.039 | 1.503 |
| Br | 11 | -1.040 | -0.857 | 0.973 |
| Mg | 27 | -0.992 | -0.708 | 0.884 |
| N | 45 | -0.911 | -0.695 | 0.667 |
| Na | 32 | -0.816 | -0.798 | 0.618 |
| Zn | 48 | -0.802 | -0.620 | 0.501 |
| Se | 31 | -0.782 | -0.608 | 0.343 |
| Nb | 10 | -0.704 | -0.223 | 1.075 |
| Cd | 29 | -0.690 | -0.544 | 0.340 |

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

- Surface slabs: vacuum direction, OOD for 3D-bulk-trained ALIGNN.
- Cross-arm comparison with `eform_v2_opt/analysis/` is the primary signal here (no DFT reference).
