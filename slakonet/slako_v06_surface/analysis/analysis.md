# SlakoNet v6 — Surface DB Analysis

## Setup

- **Input:** JARVIS `surfacedb` (607 relaxed slab structures, PBE/R2SCAN).
- **Model:** SlakoNet (DFTB Slater-Koster neural network).
- **Coverage:** 466 / 607 slabs successfully predicted. Of the 141 missing: **120 are filtered up-front** because they contain elements heavier than Tb (Z > 65; Au, Pt, W, Hf, Ir, Os, etc.), and **21 pass the filter but silently fail inside `gpu_worker`** — 19 contain an f-block lanthanide (Ce–Tb, Z 58–65; the same 4f-shell wall documented in `../../slako_v10_2d/analysis/analysis.md`) and 2 contain argon. Nominal ceiling is Z ≤ 65; effective usable ceiling is Z ≤ 57 excluding noble gases.
- **Per-structure output:** `sk_bandgap` (eV) and a full DOS (`dos_energies`, `dos_values`).

## Prediction target

SlakoNet's `sk_bandgap` (HOMO–LUMO of the tight-binding spectrum at Γ for the slab geometry) is compared against the **slab's own DFT gap**:

```
gap_dft = max(surf_cbm − surf_vbm, 0)
```

where `surf_vbm` / `surf_cbm` are the slab's own band edges (the `scf_*` fields are bulk edges on the vacuum-aligned scale and must **not** be subtracted as a gap — doing so gives spurious 8–17 eV gaps for metals like Al, Ag, Cu). Clipping at 0 correctly assigns metallic slabs a DFT gap of 0.

## Dataset composition (DFT targets)

- Raw `surf_cbm − surf_vbm`: min 0.000, max 4.677, median 0.310, mean 0.773 eV. **No negatives** (as expected — this is the proper gap pair).
- Metallic slabs (DFT gap < 0.1 eV): **192 / 466 (41%)**.
- Non-metallic slabs (DFT gap ≥ 0.1 eV): **274 / 466 (59%)**, spanning ~0.1 – 4.7 eV.

## Overall parity

| Metric | Value |
|---|---|
| N | 466 |
| MAE | **0.97 eV** |
| RMSE | 1.59 eV |
| Pearson r | **0.75** |
| SlakoNet gap (mean / median / max) | 1.67 / 1.18 / 16.21 eV |
| DFT gap (mean / median / max) | 0.77 / 0.31 / 4.68 eV |

SlakoNet is **systematically biased high** — mean SK gap is ~0.9 eV above the mean DFT gap, and the max (16 eV) is far outside the DFT range. Correlation is nonetheless strong (r ≈ 0.75), so ordering is largely preserved.

### Non-metallic subset only (N = 274)

| Metric | Value |
|---|---|
| MAE | 1.42 eV |
| Pearson r | 0.63 |

The error is concentrated in the non-metallic entries: SlakoNet overestimates gaps for genuine semiconductors/insulators by ~1.4 eV on average.

## Metallic classification (threshold 0.1 eV)

| | SK metal | SK non-metal |
|---|---|---|
| **DFT metal (192)** | 137 (TP) | 55 (FN) |
| **DFT non-metal (274)** | 25 (FP) | 249 (TN) |

- Accuracy: **82.8 %**
- Recall (metals): 71 % (correctly flags most metals as gap ≈ 0)
- Precision (metals): 85 %

This is a qualitatively useful result — SlakoNet usually gets the metal/non-metal distinction right, even though quantitative gap values for non-metals are too large.

## Top outliers (|SK − DFT|)

All ten are in the **over-prediction** direction, and all are ionic/covalent insulators or molecular adsorbate systems:

| SK (eV) | DFT (eV) | Residual | Formula | Name |
|---|---|---|---|---|
| 16.21 | 0.56 | +15.65 | H4NF | JVASP-23972, (100) |
| 7.58 | 0.09 | +7.49 | AlPO4 | JVASP-151876, (100) |
| 7.34 | 1.55 | +5.79 | LiCl | JVASP-107458, (100) |
| 6.04 | 0.85 | +5.19 | LaOF | JVASP-116006, (110) |
| 4.75 | 0.07 | +4.68 | BaNaTiNbO6 | JVASP-101231, (110) |
| 7.29 | 2.96 | +4.34 | Ca4Cl6O | JVASP-23901, (100) |
| 5.44 | 1.22 | +4.22 | MgO | JVASP-43629, (100) |
| 5.90 | 1.89 | +4.02 | MgO | JVASP-34228, (100) |
| 3.82 | 0.14 | +3.68 | ZnS | JVASP-57104, (100) |
| 3.95 | 0.28 | +3.67 | ZnS | JVASP-10591, (001) |

The H4NF case (16 eV) is likely a molecular/ammonium-fluoride-like slab where the DFTB HOMO–LUMO is an isolated molecular gap rather than a band gap. The ionic oxide/halide outliers (MgO, LiCl, AlPO4, ZnS) are well-known cases where single-particle DFT already *under*estimates the gap and where DFTB parameterizations are known to *over*estimate ionic gaps — the two errors compound.

## DOS

- Average SlakoNet DOS, split at DFT gap = 0.5 eV, shows a clear pseudogap near E = 0 for the large-gap group and continuous spectral weight across E = 0 for the small-gap group — consistent with SlakoNet distinguishing metallic from insulating character at the DOS level even when the scalar `sk_bandgap` is quantitatively off.
- See `plots/dos_average.png` and `plots/dos_examples.png`.

## Takeaways

1. **Metal vs. insulator is well learned** — 83 % accuracy, r = 0.75. SlakoNet correctly returns near-zero gaps for most metallic slabs.
2. **Gap magnitudes are over-predicted** in the non-metallic subset by ~1.4 eV on average; ionic insulators (MgO, LiCl, ZnS, AlPO4) are the worst offenders.
3. **A handful of molecular / ammonium-like slabs** produce unphysical gaps (H4NF: 16 eV) and inflate the RMSE — filtering these or treating molecular slabs separately would meaningfully improve reported error.
4. **Z ≤ 57 effective ceiling** excludes 23 % of the dataset (20 % filtered up-front as Z > 65, plus another 3.5 % dropped silently on lanthanides/noble gases). The bulk are noble-metal and 5d transition-metal surfaces of catalytic interest; extending SlakoNet parameters upward — and adding real 4f-shell and noble-gas support — is the main path to broader coverage.

## Generated artifacts

- `plots/parity_sk_vs_dft.png` — parity plot with MAE/RMSE/r annotation
- `plots/distributions.png`, `plots/gap_distribution_comparison.png` — gap histograms
- `plots/residual_histogram.png` — SK − DFT residuals
- `plots/dos_average.png`, `plots/dos_examples.png` — DOS comparisons
- `csv/summary.csv` — per-slab table (name, formula, edges, gaps, residuals)

*Note: the parity/MAE/RMSE values currently rendered in `analysis/*.png` were produced using `scf_cbm − scf_vbm` as the DFT target and should be regenerated with `surf_cbm − surf_vbm` for the numbers in this document. The values in this markdown are the recomputed ones.*
