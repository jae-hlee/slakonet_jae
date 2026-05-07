# v07 vacancy_db: ALIGNN PBE bandgap

- **N**: 470
- **ALIGNN gap mean / std**: 0.730 / 1.413 eV
- **Fraction predicted metallic**: 55.957%
- **No DFT bandgap reference** in this dataset (only formation energy `ef`). The cross-method comparison is SK-vs-ALIGNN rather than parity-vs-DFT.
- **Pearson corr (ALIGNN gap vs ef)**: 0.366

## SlakoNet vs ALIGNN cross-check

- **N matched by ID**: 444
- **MAE (ALIGNN vs SK)**: 0.634 eV
- **RMSE**: 1.352 eV; **ME (ALIGNN - SK)**: +0.561 eV
- **Pearson corr**: 0.396
- **Metal fractions**: SK 91.216%, ALIGNN 54.279%
- **Metal/gap agreement** (threshold 0.05 eV): 61.261%
- **Confusion (SK reference, ALIGNN prediction)** TN/FP/FN/TP: 237,168,4,35
- **Files**: `plots/distribution.png`, `plots/alignn_vs_ef.png`, `csv/metrics.csv`, `plots/sk_vs_alignn.png`, `plots/confusion_sk_vs_alignn.png`
