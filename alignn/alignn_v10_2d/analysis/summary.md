# v10_2d: ALIGNN PBE bandgap

- **N (with DFT ref)**: 87903
- **MAE vs `band_gap_ind`**: 0.470 eV
- **RMSE**: 0.837 eV; **ME (bias)**: +0.190 eV
- **Median \|err\|**: 0.206 eV; **90th pct \|err\|**: 1.219 eV
- **Reference**: PBE `band_gap_ind` from Alexandria
- **Confusion (metal/gap, threshold 0.05 eV), TN/FP/FN/TP**: 22564,29316,2277,33746
- **Accuracy**: 64.059%
- **Files**: `plots/parity.png`, `plots/confusion.png`, `plots/distribution.png`, `csv/metrics.csv`
