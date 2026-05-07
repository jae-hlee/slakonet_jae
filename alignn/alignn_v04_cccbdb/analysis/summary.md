# v04 CCCBDB: ALIGNN PBE bandgap

- **N entries**: 1333
- **With DFT ref (homo + lumo present)**: 1330
- **Reference**: HOMO-LUMO gap = `lumo - homo`, in Hartree (multiply by 27.2114 to get eV).
- **Caveat**: ALIGNN `mp_gappbe_alignn` was trained on Materials Project **crystals**, not isolated molecules. CCCBDB is out-of-distribution and the parity numbers should be read accordingly. The reference itself is molecular DFT (typically Gaussian-basis), not solid-state PBE; the functional / basis difference adds to ALIGNN's error.
- **MAE**: 3.365 eV; **RMSE**: 7.931 eV; **ME (bias)**: -3.197 eV
- **Median \|err\|**: 3.019 eV; **90th pct \|err\|**: 5.183 eV
- **Files**: `plots/parity.png`, `plots/distribution.png`, `csv/metrics.csv`
