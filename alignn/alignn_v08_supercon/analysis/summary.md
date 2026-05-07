# v08 alex_supercon: ALIGNN PBE bandgap

- **N**: 4827
- **ALIGNN gap mean / std**: 0.027 / 0.151 eV
- **Fraction predicted metallic**: 93.516%
- **Subset Tc > 5 K** (N=704): 95.312% predicted metallic
- **No DFT bandgap reference** in this Tc-focused dataset. Sanity check: do high-Tc candidates predict as metallic? Answer: yes.
- **Pearson corr** (ALIGNN gap): vs Tc 0.049, vs DOS(E_F) -0.078

## SlakoNet vs ALIGNN cross-check

- **N matched by ID**: 4827
- **MAE (ALIGNN vs SK)**: 0.039 eV
- **RMSE**: 0.176 eV; **ME (ALIGNN - SK)**: +0.008 eV
- **Pearson corr**: 0.192
- **Metal fractions**: SK 97.265%, ALIGNN 93.516%
- **Metal/gap agreement** (threshold 0.05 eV): 92.438%
- **Confusion (SK reference, ALIGNN prediction)** TN/FP/FN/TP: 4422,273,92,40
- **Files**: `plots/distribution.png`, `plots/alignn_vs_supercon.png`, `csv/metrics.csv`, `plots/sk_vs_alignn.png`, `plots/confusion_sk_vs_alignn.png`
