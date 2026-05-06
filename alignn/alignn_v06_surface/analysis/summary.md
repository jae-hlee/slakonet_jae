# v06 surface_db: ALIGNN PBE bandgap

- **N**: 487
- **MAE vs `surf_cbm - surf_vbm` (clipped at 0)**: 0.507 eV
- **RMSE**: 0.890 eV; **ME (bias)**: +0.190 eV
- **Median \|err\|**: 0.302 eV; **90th pct \|err\|**: 1.157 eV
- **Confusion (metal/gap, threshold 0.05 eV), TN/FP/FN/TP**: 90,77,18,302
- **Reference choice**: surf_* (correct, slab edges). The `parity_scf_wrong.png` plot uses `scf_cbm - scf_vbm` (bulk-projected, wrong scale) for documentation purposes only.
- **Files**: `parity_surf.png`, `parity_scf_wrong.png`, `confusion_surf.png`, `distribution.png`, `metrics.csv`
