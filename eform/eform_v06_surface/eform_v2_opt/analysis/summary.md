# v06 e_form analysis: Opt arm (jv_formation_energy_peratom_alignn)

Pretrained model trained on **JARVIS/OptB88vdW** formation energy (eV/atom). Inference on the **JARVIS surface_db slabs (Z<=65)** set, N = 487. **No DFT e_form reference** ships with this dataset. Analysis is distribution-only + cross-arm comparison.

## Distribution of predicted e_form (eV/atom)

| stat | value |
|---|---|
| N | 487 |
| min | -4.092 |
| p05 | -2.173 |
| median | -0.351 |
| mean | -0.535 |
| p95 | +0.219 |
| max | +0.900 |
| std | 0.784 |
| n(pred < 0) | 348 (71.5%) |
| n(pred > 0) | 139 (28.5%) |

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
| 1 elem | 95 | +0.074 | +0.051 | 0.080 |
| 2 elem | 237 | -0.418 | -0.342 | 0.585 |
| 3 elem | 141 | -1.039 | -0.815 | 0.911 |
| 4 elem | 12 | -1.395 | -1.328 | 0.859 |
| 5+ elem | 2 | -2.593 | -2.593 | 0.264 |

## Per-element mean prediction (top 20 most-negative, count >= 5)

| element | count | mean pred | median pred | std |
|---|---|---|---|---|
| F | 12 | -3.140 | -3.532 | 0.983 |
| O | 42 | -2.209 | -2.026 | 0.915 |
| Cl | 13 | -1.708 | -2.181 | 0.982 |
| Y | 7 | -1.613 | -1.064 | 1.364 |
| Sr | 11 | -1.461 | -1.067 | 0.772 |
| Sc | 9 | -1.393 | -1.078 | 0.975 |
| Ba | 18 | -1.380 | -0.958 | 0.852 |
| Mn | 10 | -1.256 | -1.100 | 0.798 |
| Ca | 12 | -1.224 | -0.942 | 0.687 |
| K | 12 | -1.179 | -1.110 | 0.552 |
| S | 44 | -1.039 | -0.791 | 0.625 |
| Mg | 27 | -1.024 | -0.841 | 0.852 |
| Br | 11 | -1.012 | -0.748 | 1.008 |
| Pr | 7 | -1.011 | +0.026 | 1.423 |
| N | 45 | -0.837 | -0.809 | 0.677 |
| Se | 31 | -0.830 | -0.636 | 0.356 |
| Na | 32 | -0.827 | -0.890 | 0.581 |
| Zr | 6 | -0.728 | -0.719 | 0.115 |
| Nb | 10 | -0.717 | -0.328 | 0.999 |
| Zn | 48 | -0.715 | -0.650 | 0.394 |

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
- Cross-arm comparison with `eform_v1_pbe/analysis/` is the primary signal here (no DFT reference).
