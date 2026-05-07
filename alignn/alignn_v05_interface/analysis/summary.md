# v05 interface_db: ALIGNN PBE bandgap

- **N total**: 587; **with DFT ref (optb88vdw_bandgap present)**: 587
- **Reference**: `optb88vdw_bandgap` (OptB88vdW DFT), clipped at 0. **NOT PBE.**
- **Negative DFT gaps clipped to 0**: 107 entries (interface artifact; surface CLAUDE.md gotcha).
- **MAE vs OptB88vdW (clipped)**: 0.531 eV; **RMSE**: 0.717 eV; **ME (bias)**: +0.487 eV
- **Median \|err\|**: 0.396 eV; **90th pct \|err\|**: 1.052 eV
- **Fraction predicted metallic** (gap <= 0.05 eV): 2.726%
- **Confusion (metal/gap, threshold 0.05 eV), TN/FP/FN/TP**: 12,131,4,440
- **Accuracy**: 77.002%
- **Caveat**: ALIGNN predicts PBE; reference is OptB88vdW. The two functionals differ on bandgap (OptB88vdW is typically larger than PBE for most materials), so part of the ME is the functional shift, not pure model error.
- **Files**: `plots/parity.png`, `plots/confusion.png`, `plots/distribution.png`, `csv/metrics.csv`
