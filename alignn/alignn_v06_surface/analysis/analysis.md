# v06 surface_db: ALIGNN PBE bandgap deep analysis

Reference: max(surf_cbm - surf_vbm, 0) (PBE slab edges). Metal/gap split at 0.05 eV.

## Metal vs non-metal stratification

| subset | N | MAE | RMSE | ME (bias) | med \|err\| | p90 \|err\| | DFT mean | ALIGNN mean |
|---|---|---|---|---|---|---|---|---|
| all | 487 | 0.507 | 0.890 | +0.190 | 0.302 | 1.157 | 0.784 | 0.975 |
| DFT metals | 167 | 0.418 | 0.919 | +0.411 | 0.028 | 1.134 | 0.014 | 0.425 |
| DFT non-metals | 320 | 0.554 | 0.875 | +0.075 | 0.393 | 1.167 | 1.186 | 1.261 |

## Metal/gap classification (full set)

- Accuracy: 80.493%
- True negatives (DFT metal, ALIGNN metal): 90 (18.5%)
- False positives (DFT metal, ALIGNN gap):  77 (15.8%)
- False negatives (DFT gap, ALIGNN metal):  18 (3.7%)
- True positives (DFT gap, ALIGNN gap):     302 (62.0%)

**Failure-mode breakdown.** The dominant error mode is **false positives** (77 vs 18): ALIGNN predicts a gap where DFT says the structure is metallic.  Metal MAE (0.418) and non-metal MAE (0.554) are similar.

## Worst predictions

Top 20 entries by absolute residual (full table at `csv/worst_predictions.csv`):

| name | formula | surf_gap_clipped | alignn_bandgap | residual |
| --- | --- | --- | --- | --- |
| Surface-JVASP-819_miller_1_0_0_x1015_thickness_16_VASP_R2SCA | Ar | 0.030 | 5.763 | 5.733 |
| Surface-JVASP-23972_miller_1_0_0_thickness_16_VASP_PBE_noDP | H4NF | 0.560 | 5.887 | 5.328 |
| Surface-JVASP-151876_miller_1_0_0_thickness_16_VASP_PBE_noDP | AlPO4 | 0.091 | 4.959 | 4.869 |
| Surface-JVASP-52171_miller_1_1_0_thickness_16_VASP_PBE_noDP | NdOF | 0.036 | 4.482 | 4.446 |
| Surface-JVASP-91_miller_1_1_1_thickness_16_VASP_R2SCAN | C | 0.129 | 4.176 | 4.047 |
| Surface-JVASP-107458_miller_1_0_0_thickness_16_VASP_PBE_noDP | LiCl | 1.551 | 5.293 | 3.743 |
| Surface-JVASP-149059_miller_1_1_0_thickness_16_VASP_PBE_noDP | ZnSO4 | 0.053 | 3.745 | 3.692 |
| Surface-JVASP-148851_miller_1_1_0_thickness_16_VASP_PBE_noDP | ZnSO4 | 0.003 | 3.661 | 3.658 |
| Surface-JVASP-39_miller_0_0_1_thickness_16_VASP_R2SCAN | AlN | 0.009 | 3.450 | 3.441 |
| Surface-JVASP-819_miller_1_1_0_x1015_thickness_16_VASP_R2SCA | Ar | 8.342 | 5.340 | -3.002 |

## Per-element MAE (worst 15)

Entries containing each element (compositions overlap; the same entry appears under each of its elements). Min count threshold applied.

| element | count | MAE (eV) | median \|err\| (eV) |
|---|---|---|---|
| F | 12 | 1.721 | 1.206 |
| O | 42 | 1.205 | 0.841 |
| N | 45 | 0.810 | 0.533 |
| S | 44 | 0.799 | 0.441 |
| C | 26 | 0.776 | 0.408 |
| Cl | 13 | 0.774 | 0.428 |
| Ba | 18 | 0.729 | 0.368 |
| K | 12 | 0.702 | 0.491 |
| Al | 43 | 0.673 | 0.217 |
| Zn | 48 | 0.609 | 0.401 |
| Ca | 12 | 0.583 | 0.326 |
| B | 32 | 0.564 | 0.549 |
| Be | 13 | 0.564 | 0.218 |
| P | 42 | 0.554 | 0.302 |
| Li | 37 | 0.534 | 0.326 |

Full table at `csv/per_element_mae.csv`; bar chart at `plots/per_element_mae.png`.

## Caveats

- Reference is the slab's own band edges (`surf_*`), not the bulk-projected `scf_*` (which is wrong-scale and is plotted only for documentation in `plots/parity_scf_wrong.png`).
- Slab geometries with vacuum sit outside the 3D-bulk training distribution of `mp_gappbe_alignn`; expect errors comparable to other OOD geometries (v09/v10).

## Files

- `plots/parity_all.png`, `plots/parity_metals.png`, `plots/parity_nonmetals.png` (stratified parity + residual panels)
- `csv/worst_predictions.csv` (top 20 by absolute residual)
- `csv/stratified_metrics.csv` (the all/metals/non-metals MAE table above)
- `csv/per_element_mae.csv`, `plots/per_element_mae.png`
- existing: `plots/parity_surf.png`, `plots/parity_scf_wrong.png`, `plots/parity_all.png`, `plots/parity_metals.png`, `plots/parity_nonmetals.png`, `plots/confusion_surf.png`, `plots/distribution.png`, `plots/per_element_mae.png`, `csv/metrics.csv`, `summary.md`

## See also

Cross-method comparison vs SlakoNet on the same matched structures: `slakonet/slako_v06_surface/analysis/plots/sk_vs_alignn.png` and `slakonet/slako_v06_surface/analysis/plots/confusion_sk_vs_alignn.png`. Metrics in `slakonet/slako_v06_surface/analysis/csv/sk_vs_alignn_metrics.csv`. Cross-dataset roll-up at `slakonet/slakonet_comprehensive_analysis/csv/sk_vs_alignn_cross_dataset.csv`.
