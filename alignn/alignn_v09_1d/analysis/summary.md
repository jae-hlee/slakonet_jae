# v09_1d: ALIGNN PBE bandgap

- **N (with DFT ref)**: 9540
- **MAE vs `band_gap_ind`**: 0.485 eV
- **RMSE**: 0.755 eV; **ME (bias)**: -0.143 eV
- **Median \|err\|**: 0.273 eV; **90th pct \|err\|**: 1.255 eV
- **Reference**: PBE `band_gap_ind` from Alexandria
- **Confusion (metal/gap, threshold 0.05 eV), TN/FP/FN/TP**: 1929,1646,930,5035
- **Accuracy**: 72.998%
- **Files**: `plots/parity.png`, `plots/confusion.png`, `plots/distribution.png`, `csv/metrics.csv`
