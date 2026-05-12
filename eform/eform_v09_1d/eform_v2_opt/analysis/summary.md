# v09 e_form analysis: Opt arm (jv_formation_energy_peratom_alignn)

Pretrained model trained on **JARVIS/OptB88vdW** formation energy (eV/atom). Inference on the **Alexandria PBE 1D (Z<=65)** set, N = 9,540. Reference is `e_form` from Alexandria PBE.

## Headline metrics

| metric | value (eV/atom) |
|---|---|
| MAE | **0.3144** |
| RMSE | 0.3988 |
| ME (bias) | +0.0899 |
| Median \|err\| | 0.2574 |
| 90th-pct \|err\| | 0.6585 |
| 95th-pct \|err\| | 0.7930 |
| 99th-pct \|err\| | 1.0598 |
| Pearson r | 0.9359 |
| ALIGNN mean / median | -0.309 / -0.142 |
| DFT mean / median | -0.399 / -0.204 |

## Cumulative error distribution

Fraction of structures with |ALIGNN - DFT| below each threshold:

| threshold (eV/atom) | count | fraction |
|---|---|---|
| 0.005 | 124 | 1.30% |
| 0.010 | 213 | 2.23% |
| 0.020 | 396 | 4.15% |
| 0.050 | 995 | 10.43% |
| 0.100 | 1,972 | 20.67% |
| 0.200 | 3,832 | 40.17% |
| 0.300 | 5,384 | 56.44% |
| 0.500 | 7,563 | 79.28% |
| 1.000 | 9,398 | 98.51% |

## Stratified metrics (metal vs non-metal)

Split by DFT `band_gap_ind`: == 0 vs > 0.

| subset | N | MAE | RMSE | ME | med \|err\| | p90 \|err\| |
|---|---|---|---|---|---|---|
| all | 9,540 | 0.3144 | 0.3988 | +0.0899 | 0.2574 | 0.6585 |
| metal | 3,177 | 0.2987 | 0.3748 | -0.0421 | 0.2566 | 0.6250 |
| non-metal | 6,363 | 0.3222 | 0.4103 | +0.1558 | 0.2576 | 0.6783 |

## Bandgap-bin stratification

Five bins by DFT `band_gap_ind` (eV).

| bin | N | MAE | RMSE | ME | med \|err\| |
|---|---|---|---|---|---|
| metal (=0) | 3,177 | 0.2987 | 0.3748 | -0.0421 | 0.2566 |
| small (0,0.5] | 2,132 | 0.3101 | 0.3860 | +0.0074 | 0.2632 |
| med  (0.5,1.5] | 1,653 | 0.2855 | 0.3698 | +0.0984 | 0.2263 |
| wide (1.5,3.5] | 1,664 | 0.3370 | 0.4373 | +0.2802 | 0.2464 |
| ultra (>3.5) | 914 | 0.3898 | 0.4782 | +0.3796 | 0.3390 |

## Composition cardinality stratification

By number of distinct elements per formula.

| cardinality | N | MAE | RMSE | ME | med \|err\| |
|---|---|---|---|---|---|
| 2 elem | 9,540 | 0.3144 | 0.3988 | +0.0899 | 0.2574 |

## Cross-arm comparison

On N = 9,540 records (positional pairing):

| metric | value |
|---|---|
| PBE - Opt mean (signed) | -0.3563 eV/atom |
| PBE - Opt median | -0.2468 eV/atom |
| mean \|PBE - Opt\| | 0.3933 eV/atom |
| max \|PBE - Opt\| | 2.3670 eV/atom |
| Pearson r (PBE vs Opt) | 0.9132 |
| fraction \|diff\| <= 0.05 | 14.1% |
| fraction \|diff\| <= 0.10 | 25.5% |
| fraction \|diff\| <= 0.50 | 70.9% |

Top 25 cross-arm disagreements (full table at `csv/cross_arm_top_disagreements.csv`):

| id | formula | PBE | Opt | PBE - Opt |
|---|---|---|---|---|
| agm1000016245 | SbP | +0.298 | +2.665 | -2.367 |
| agm1000003093 | MoP2 | -0.076 | +2.175 | -2.251 |
| agm1000001187 | MoS2 | -0.806 | +1.247 | -2.053 |
| agm1000006456 | SmN | -0.696 | +1.353 | -2.049 |
| agm1000002774 | NbS2 | -1.031 | +1.016 | -2.047 |
| agm1000009666 | NbS | -0.374 | +1.672 | -2.046 |
| agm1000013248 | NbS | -0.312 | +1.715 | -2.027 |
| agm1000009770 | VAs | -0.036 | +1.977 | -2.012 |
| agm1000010055 | FeAs | +0.123 | +2.125 | -2.002 |
| agm1000000194 | TbN | -0.679 | +1.311 | -1.990 |
| agm1000012907 | CoAs | +0.093 | +2.077 | -1.984 |
| agm1000013126 | VAs | +0.046 | +2.024 | -1.978 |
| agm1000005601 | NdN | -0.677 | +1.298 | -1.975 |
| agm1000009107 | CoAs | +0.084 | +2.055 | -1.971 |
| agm1000017172 | CoAs | +0.085 | +2.053 | -1.968 |
| agm1000009130 | VP | -0.062 | +1.905 | -1.967 |
| agm1000002991 | ScAs | -0.658 | +1.307 | -1.965 |
| agm1000009383 | NbTe | +0.293 | +2.239 | -1.946 |
| agm1000005979 | Nb2P3 | +0.077 | +2.022 | -1.945 |
| agm1000002671 | YN | -0.665 | +1.279 | -1.944 |
| agm1000011238 | CoP | -0.127 | +1.810 | -1.936 |
| agm1000004641 | YP | -0.673 | +1.261 | -1.934 |
| agm1000000299 | NbS | -0.882 | +1.034 | -1.916 |
| agm1000003460 | ZrS2 | -1.549 | +0.361 | -1.911 |
| agm1000013448 | VP | +0.011 | +1.921 | -1.909 |


## Per-element MAE (top 15 worst, count >= 50)

| element | count | MAE | median \|err\| |
|---|---|---|---|
| Cl | 731 | 0.5164 | 0.4953 |
| Ge | 61 | 0.4725 | 0.4242 |
| Si | 69 | 0.4556 | 0.4889 |
| B | 156 | 0.4290 | 0.3514 |
| Zr | 73 | 0.3965 | 0.3679 |
| Mo | 136 | 0.3942 | 0.3551 |
| Sb | 119 | 0.3924 | 0.3561 |
| Ti | 243 | 0.3714 | 0.3621 |
| S | 1,119 | 0.3642 | 0.2882 |
| Ni | 283 | 0.3608 | 0.3227 |
| Y | 162 | 0.3584 | 0.3277 |
| Co | 278 | 0.3582 | 0.2943 |
| P | 1,087 | 0.3575 | 0.3504 |
| V | 467 | 0.3496 | 0.2961 |
| Ru | 196 | 0.3485 | 0.2977 |

Full per-element table at `csv/per_element_mae.csv`.

## Top 25 worst predictions

| id | formula | DFT | ALIGNN | residual | band_gap_ind |
|---|---|---|---|---|---|
| agm1000007835 | CS2 | -0.152 | +1.713 | +1.865 | 3.6695 |
| agm1000001674 | CS2 | -0.152 | +1.591 | +1.743 | 3.6873 |
| agm1000003874 | CS2 | -0.162 | +1.560 | +1.723 | 3.4275 |
| agm1000013158 | CS2 | -0.158 | +1.462 | +1.620 | 3.382 |
| agm1000015893 | NiS | +0.029 | +1.487 | +1.458 | 0.5719000000000001 |
| agm1000006538 | CCl4 | -0.318 | +1.118 | +1.436 | 0.5578000000000001 |
| agm1000008464 | MoCl4 | -1.239 | +0.186 | +1.424 | 1.2497 |
| agm1000004678 | SbN | +0.118 | +1.532 | +1.414 | 0.2864 |
| agm1000014506 | BeO | -2.353 | -0.964 | +1.390 | 5.1248 |
| agm1000007246 | NbS2 | -0.419 | +0.969 | +1.388 | 0.3638 |
| agm1000009044 | MoS2 | -0.049 | +1.301 | +1.350 | 0.4103 |
| agm1000007631 | SiS2 | -0.280 | +1.045 | +1.325 | 3.1754 |
| agm1000003781 | CO2 | -1.731 | -0.407 | +1.325 | 7.6158 |
| agm1000008274 | PdS2 | -0.180 | +1.085 | +1.265 | 1.0525 |
| agm1000007393 | VS2 | -0.471 | +0.787 | +1.258 | 0.3829 |
| agm1000013678 | GeS2 | -0.122 | +1.123 | +1.245 | 3.0476 |
| agm1000012850 | MnO | -1.103 | +0.140 | +1.242 | 1.4412 |
| agm1000006383 | RuCl4 | -0.349 | +0.886 | +1.234 | 0.3849 |
| agm1000013621 | MgO | -1.744 | -0.523 | +1.221 | 3.0382 |
| agm1000013140 | B2S3 | -0.403 | +0.813 | +1.216 | 3.2957 |
| agm1000013121 | ZnO | -1.054 | +0.161 | +1.216 | 2.5976 |
| agm1000010462 | CrO | -0.917 | +0.293 | +1.210 | 0.0 |
| agm1000002377 | RhCl3 | -0.415 | +0.791 | +1.206 | 0.47200000000000003 |
| agm1000006329 | ZnCl2 | -1.210 | -0.013 | +1.197 | 4.6759 |
| agm1000013122 | AgCl | -0.513 | +0.678 | +1.191 | 3.1172 |

Full top-100 at `csv/worst_predictions.csv`.

## Top 10 best predictions (smallest |residual|)

| id | formula | DFT | ALIGNN | residual |
|---|---|---|---|---|
| agm1000011240 | MnSe | -0.159 | -0.159 | +0.0001 |
| agm1000015899 | CrAs | +0.308 | +0.308 | +0.0002 |
| agm1000018836 | InBr | -0.603 | -0.603 | +0.0002 |
| agm1000013834 | NbF2 | -2.003 | -2.003 | +0.0003 |
| agm1000004636 | Zn3P2 | +0.798 | +0.799 | +0.0003 |
| agm1000014525 | PdS | -0.236 | -0.235 | +0.0004 |
| agm1000019561 | Ag2Te | +0.059 | +0.060 | +0.0004 |
| agm1000018680 | CrN | -0.102 | -0.101 | +0.0004 |
| agm1000016764 | CoTe | +0.005 | +0.005 | +0.0004 |
| agm1000016729 | BAs | +0.431 | +0.430 | -0.0004 |

## Files

- `csv/metrics.csv` — overall + metal/non-metal + bandgap-bin + cardinality
- `csv/per_element_mae.csv` — every element above min_count
- `csv/cumulative_error.csv` — fraction within each |err| threshold
- `csv/worst_predictions.csv` — top 100 by |residual|
- `csv/best_predictions.csv` — top 25 with smallest |residual|
- `plots/parity.png` — hexbin parity + residual panel
- `plots/residual_histogram.png` — (ALIGNN - DFT) distribution
- `plots/residual_cdf.png` — cumulative |residual| distribution
- `plots/per_element_mae.png` — bar chart of worst-15 elements
- `plots/error_vs_eform.png` — |err| vs DFT e_form scatter
- `plots/bandgap_bin_mae.png` — MAE per bandgap bin
- `plots/composition_cardinality.png` — MAE by number of elements
- `plots/cross_arm_comparison.png` — PBE vs Opt arm: scatter, signed-diff histogram, |diff| CDF

## Caveats

- Reference is Alexandria PBE `e_form`; this arm trained on **JARVIS/OptB88vdW** e_form. Training functional differs from reference; the systematic ME is the PBE-vs-OptB88vdW functional shift, not pure model error. Subtract ME to recover a rough error scale.
- 1D structures: two vacuum directions, OOD for 3D-bulk-trained ALIGNN.
- Cross-arm comparison with `eform_v1_pbe/analysis/` is in the cross-arm section above; large PBE-vs-this disagreements are characterized in `csv/worst_predictions.csv`.
