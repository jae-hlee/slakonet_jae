# v11_alexwz: ALIGNN PBE bandgap

- **N (with DFT ref)**: 115535
- **MAE vs `band_gap_ind`**: 0.168 eV
- **RMSE**: 0.476 eV; **ME (bias)**: +0.024 eV
- **Median \|err\|**: 0.015 eV; **90th pct \|err\|**: 0.467 eV
- **Reference**: PBE `band_gap_ind` from Alexandria
- **Confusion (metal/gap, threshold 0.05 eV), TN/FP/FN/TP**: 69866,10829,1730,33110
- **Accuracy**: 89.130%
- **Files**: `plots/parity.png`, `plots/confusion.png`, `plots/distribution.png`, `csv/metrics.csv`
