# v09 e_form analysis: PBE arm (mp_e_form_alignn)

Pretrained model trained on **MP/PBE** formation energy (eV/atom). Inference on the **Alexandria PBE 1D (Z<=65)** set, N = 9,540. Reference is `e_form` from Alexandria PBE.

## Headline metrics

| metric | value (eV/atom) |
|---|---|
| MAE | **0.3308** |
| RMSE | 0.4757 |
| ME (bias) | -0.2664 |
| Median \|err\| | 0.2238 |
| 90th-pct \|err\| | 0.7785 |
| 95th-pct \|err\| | 1.0468 |
| 99th-pct \|err\| | 1.5319 |
| Pearson r | 0.9376 |
| ALIGNN mean / median | -0.665 / -0.448 |
| DFT mean / median | -0.399 / -0.204 |

## Cumulative error distribution

Fraction of structures with |ALIGNN - DFT| below each threshold:

| threshold (eV/atom) | count | fraction |
|---|---|---|
| 0.005 | 186 | 1.95% |
| 0.010 | 347 | 3.64% |
| 0.020 | 662 | 6.94% |
| 0.050 | 1,529 | 16.03% |
| 0.100 | 2,692 | 28.22% |
| 0.200 | 4,474 | 46.90% |
| 0.300 | 5,740 | 60.17% |
| 0.500 | 7,422 | 77.80% |
| 1.000 | 9,002 | 94.36% |

## Stratified metrics (metal vs non-metal)

Split by DFT `band_gap_ind`: == 0 vs > 0.

| subset | N | MAE | RMSE | ME | med \|err\| | p90 \|err\| |
|---|---|---|---|---|---|---|
| all | 9,540 | 0.3308 | 0.4757 | -0.2664 | 0.2238 | 0.7785 |
| metal | 3,177 | 0.3958 | 0.5389 | -0.3650 | 0.2807 | 0.9161 |
| non-metal | 6,363 | 0.2983 | 0.4408 | -0.2171 | 0.1953 | 0.6913 |

## Bandgap-bin stratification

Five bins by DFT `band_gap_ind` (eV).

| bin | N | MAE | RMSE | ME | med \|err\| |
|---|---|---|---|---|---|
| metal (=0) | 3,177 | 0.3958 | 0.5389 | -0.3650 | 0.2807 |
| small (0,0.5] | 2,132 | 0.4289 | 0.5752 | -0.4055 | 0.3283 |
| med  (0.5,1.5] | 1,653 | 0.3236 | 0.4691 | -0.2879 | 0.2285 |
| wide (1.5,3.5] | 1,664 | 0.1741 | 0.2538 | -0.0896 | 0.1214 |
| ultra (>3.5) | 914 | 0.1742 | 0.2565 | +0.1183 | 0.1017 |

## Composition cardinality stratification

By number of distinct elements per formula.

| cardinality | N | MAE | RMSE | ME | med \|err\| |
|---|---|---|---|---|---|
| 2 elem | 9,540 | 0.3308 | 0.4757 | -0.2664 | 0.2238 |

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
| Tb | 173 | 0.5976 | 0.4604 |
| N | 1,101 | 0.5789 | 0.4876 |
| Sm | 177 | 0.5693 | 0.4301 |
| Y | 162 | 0.5693 | 0.4648 |
| P | 1,087 | 0.5670 | 0.4574 |
| As | 846 | 0.5667 | 0.4641 |
| Nd | 173 | 0.5648 | 0.4590 |
| La | 142 | 0.5233 | 0.4293 |
| Gd | 157 | 0.4969 | 0.4011 |
| Sc | 169 | 0.4856 | 0.3277 |
| Ti | 243 | 0.4699 | 0.3292 |
| Eu | 153 | 0.4519 | 0.3513 |
| Nb | 333 | 0.4182 | 0.2782 |
| Co | 278 | 0.4135 | 0.2719 |
| Sr | 174 | 0.3853 | 0.3158 |

Full per-element table at `csv/per_element_mae.csv`.

## Top 25 worst predictions

| id | formula | DFT | ALIGNN | residual | band_gap_ind |
|---|---|---|---|---|---|
| agm1000016245 | SbP | +3.122 | +0.298 | -2.823 | 1.6644 |
| agm1000000194 | TbN | +2.109 | -0.679 | -2.788 | 0.43470000000000003 |
| agm1000003147 | TbP | +2.060 | -0.713 | -2.772 | 0.8046000000000001 |
| agm1000002671 | YN | +2.090 | -0.665 | -2.755 | 0.5012 |
| agm1000004641 | YP | +2.054 | -0.673 | -2.727 | 0.8316 |
| agm1000002991 | ScAs | +1.961 | -0.658 | -2.618 | 0.8646 |
| agm1000005601 | NdN | +1.823 | -0.677 | -2.500 | 0.6636000000000001 |
| agm1000001666 | YN | +2.045 | -0.422 | -2.468 | 0.0 |
| agm1000015485 | BP | +2.951 | +0.521 | -2.430 | 0.0 |
| agm1000004437 | EuN | +2.057 | -0.319 | -2.377 | 0.043500000000000004 |
| agm1000001827 | SmN | +1.949 | -0.411 | -2.360 | 0.0 |
| agm1000000905 | LaP | +1.637 | -0.631 | -2.268 | 0.8287 |
| agm1000004338 | LaP | +1.558 | -0.647 | -2.205 | 0.3183 |
| agm1000003191 | ScN | +1.971 | -0.027 | -1.998 | 0.0 |
| agm1000003792 | NdN | +1.086 | -0.885 | -1.971 | 0.5684 |
| agm1000009804 | TiAs | +1.590 | -0.378 | -1.967 | 0.0 |
| agm1000009130 | VP | +1.866 | -0.062 | -1.928 | 0.0 |
| agm1000009770 | VAs | +1.882 | -0.036 | -1.918 | 0.0 |
| agm1000009412 | CoP | +1.756 | -0.144 | -1.900 | 0.0 |
| agm1000001299 | LaN | +1.089 | -0.789 | -1.878 | 1.1702 |
| agm1000011238 | CoP | +1.751 | -0.127 | -1.877 | 0.0 |
| agm1000004901 | LaN | +1.093 | -0.784 | -1.877 | 1.1366 |
| agm1000003093 | MoP2 | +1.789 | -0.076 | -1.865 | 0.0 |
| agm1000012984 | TiAs | +1.590 | -0.275 | -1.864 | 0.0 |
| agm1000013448 | VP | +1.865 | +0.011 | -1.854 | 0.0 |

Full top-100 at `csv/worst_predictions.csv`.

## Top 10 best predictions (smallest |residual|)

| id | formula | DFT | ALIGNN | residual |
|---|---|---|---|---|
| agm1000016152 | FeS | -0.374 | -0.374 | -0.0001 |
| agm1000018307 | LiI | -1.100 | -1.100 | +0.0001 |
| agm1000017414 | GeI4 | -0.090 | -0.090 | -0.0002 |
| agm1000015558 | GdCl3 | -2.648 | -2.648 | -0.0002 |
| agm1000018170 | Rb2Se | -0.677 | -0.677 | +0.0003 |
| agm1000007905 | MnF2 | -2.763 | -2.763 | +0.0003 |
| agm1000002667 | MnF2 | -2.763 | -2.763 | +0.0003 |
| agm1000019881 | MnCl3 | -1.370 | -1.371 | -0.0003 |
| agm1000017114 | MgS | -1.401 | -1.401 | -0.0003 |
| agm1000014282 | NbS | -0.832 | -0.832 | +0.0004 |

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

- Reference is Alexandria PBE `e_form`; this arm trained on **MP/PBE** e_form. Training and reference functionals match.
- 1D structures: two vacuum directions, OOD for 3D-bulk-trained ALIGNN.
- Cross-arm comparison with `eform_v2_opt/analysis/` is in the cross-arm section above; large Opt-vs-this disagreements are characterized in `csv/worst_predictions.csv`.
