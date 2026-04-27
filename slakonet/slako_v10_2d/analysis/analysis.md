# SlakoNet vs PBE on Alexandria 2D

## Run summary

| | count | fraction |
|---|---:|---:|
| Alexandria 2D source entries | 137,833 | 100% |
| Skipped, Z > 65 | 49,930 | 36% |
| Valid inference targets | 87,903 | 64% |
| SlakoNet predictions (in `all_results.json`) | 79,903 | **91% of valid** |
| Missing (f-block lanthanide silent failure, see below) | 8,000 | ~9% of valid |

`all_results.json` is 15.2 GB (excluded from GitHub per the self-ignoring `.gitignore`).
Analysis is streamed — don't try to `json.load()` it into memory.

## The 8,000 missing entries are 100% f-block lanthanides

This deserves its own section because it overturns the "~5–10% dropout is timeouts
+ Cholesky noise" narrative that was previously recorded in CLAUDE.md.

After joining the 79,903 completed IDs against the 87,903 valid targets, **every
single one of the 8,000 missing entries contains at least one of Ce, Pr, Nd, Pm,
Sm, Eu, Gd, or Tb** (f-block elements with Z = 58–65). Zero lanthanide-containing
structures appear in the completed set.

| Lanthanide | Z | in missing | in completed |
|---|---:|---:|---:|
| Tb | 65 | 19.1% | 0.0% |
| Nd | 60 | 18.9% | 0.0% |
| Sm | 62 | 17.2% | 0.0% |
| Ce | 58 | 15.4% | 0.0% |
| Eu | 63 | 11.0% | 0.0% |
| Gd | 64 | 9.7% | 0.0% |
| Pm | 61 | 7.5% | 0.0% |
| Pr | 59 | 7.1% | 0.0% |

This is not a stochastic runtime pattern:

- Missing entries are **smaller** on average (mean nsites 8.8 vs 9.5 for completed),
  so it is not a timeout / big-cell bias.
- Missing entries are **more thermodynamically stable** (median `e_form`
  −1.01 vs −0.25 eV/atom), so it is not an exotic-phase bias.
- `gpu_worker` silently swallows the exception (per root CLAUDE.md), so the
  structures vanish instead of erroring out loudly.

SlakoNet's `ALLOWED_SYMBOLS` in `jslako_v10.py` passes these elements through the
filter because `generate_shell_dict_upto_Z65()` nominally produces a shell dict
for them, but the actual inference pipeline cannot handle the 4f shell — basis
or parameter coverage does not extend to open-shell f-electron chemistry.

**Implication for resubmits and sister projects:** the effective usable element
set is `Z ≤ 57` (through La), not `Z ≤ 65`. Either tighten the filter — drop
`Ce,Pr,Nd,Pm,Sm,Eu,Gd,Tb` from `ALLOWED_SYMBOLS` — or at minimum surface the
`gpu_worker` exception so future runs don't hide this kind of systematic miss
behind the same "silently swallowed" curtain.

**Cross-project verification.** After this v10 finding, every other sister
project with a non-zero "dropout" count was audited against the source zip.
Every one shows the same pattern:

| project | missing | % lanthanide | % noble gas |
|---|---:|---:|---:|
| v03 Alexandria 3D hull | 17,553 | 99.9 % | 0.1 % |
| v10 Alexandria 2D (this file) | 8,000 | 100.0 % | 0.0 % |
| v09 Alexandria 1D | 904 | 100.0 % | 0.0 % |
| v07 vacancy | 26 | 84.6 % | 15.4 % (Xe/Kr/Ne/Ar) |
| v06 surface | 21 | 90.5 % | 9.5 % (Ar) |
| v04 CCCBDB | 9 | 0.0 % | 100.0 % (He/Ne/Ar) |
| v05 interface | 0 | — | — |
| v08 supercon | 0 | — | — |

v05 and v08 are clean because their datasets contain no lanthanide or
noble-gas entries in the first place. v04's small dropout is entirely
noble gases (monatomic / rare-gas molecular species). The two walls —
lanthanide 4f and noble-gas chemistry — explain essentially all silent
dropout across the repo.

The 8,000 v10 IDs are saved in `analysis/missing_ids.txt`.

## Headline metrics (N = 79,903)

```
PBE metals (gap = 0):    44,306   (55%)
PBE non-metals (gap > 0): 35,597   (45%)

Indirect gap (SlakoNet vs PBE band_gap_ind)
  All                      MAE = 0.621 eV   RMSE = 1.328 eV   R² = -0.118
  PBE metals (gap = 0)     MAE = 0.088 eV   RMSE = 0.397 eV
  PBE non-metals (gap > 0) MAE = 1.285 eV   RMSE = 1.939 eV   R² = -0.650

Direct gap (SlakoNet vs PBE band_gap_dir)
  All                      MAE = 0.621 eV   RMSE = 1.307 eV   R² = -0.060
  PBE non-metals           MAE = 1.262 eV   RMSE = 1.908 eV

Max abs error: 21.52 eV (single outlier; PBE gap far from SlakoNet)

Metal / non-metal classification (SK threshold 0.1 eV)
  accuracy = 0.827
  TN = 40,195   FP =  4,111   FN =  9,682   TP = 25,915

False negatives (PBE says gap, SlakoNet predicts metal): 9,682
  → 27% of PBE non-metals get collapsed to a metal-like prediction.
  → PBE gap on those: median 0.22 eV, mean 0.45 eV, max 6.25 eV.
```

## How this compares with sister projects

| | v09 1D (8,636) | **v10 2D (79,903)** | v03 3D (~100k on-hull) |
|---|---:|---:|---|
| Overall MAE | 0.33 eV | **0.62 eV** | ~0.6 eV |
| Overall R² | +0.14 | **−0.12** | low |
| Non-metal MAE | 0.87 eV | **1.29 eV** | ~1 eV |
| Classification acc | 0.87 | **0.83** | ~0.84 |

2D is **harder** than 1D and roughly on par with 3D in difficulty. Two plausible
contributors (see CLAUDE.md):

1. **k-path handling is not 2D-aware.** `prepare_inputs` calls
   `Kpoints3D.kpath(atoms, line_density=20)` on a jarvis `Atoms` dict whose
   `lattice_mat` still has the ~20 Å vacuum direction along c. The sampled
   k-path therefore includes k-points along the vacuum direction, which is
   physically meaningless for a 2D Brillouin zone and biases both eigenvalues
   and DOS. Worth benchmarking a 2D-specific k-path.
2. **Same SlakoNet open-shell / ionic-halide failure mode** documented for
   v03 3D — ~27% of PBE non-metals collapse to a metal-like prediction, with
   median PBE gap on those cases at only 0.22 eV (small-gap, chemically
   borderline materials).

Negative R² overall means SlakoNet gap is a *worse* predictor of PBE gap than
a constant mean would be. This is driven by the bimodal residual structure on
non-metals (see below) and a handful of extreme outliers (MaxErr ≈ 21 eV).

## Plot catalog

- **`parity_all.png` / `parity_nonmetals.png`** — density-colored SlakoNet vs
  PBE (indirect + direct), with and without the PBE-metal rows. You can see a
  diagonal y=x cluster plus a clear horizontal band at SK ≈ 0 that contains
  most of the non-metal misses.
- **`residuals_all.png` / `residuals_nonmetals.png`** — SK − PBE histograms.
  The non-metal plot is explicitly split into "SK > 0.1 eV" (working mode,
  centered near zero) and "SK ≤ 0.1 eV" (broken mode, large negative bias).
- **`confusion_matrix.png`** — metal / non-metal at the 0.1 eV threshold.
- **`gap_distribution.png`** — PBE vs SlakoNet gap histograms (both normalized).
- **`error_vs_eform.png`** — |SK − PBE| vs formation energy, colored by PBE gap.
  Useful for seeing whether high-error outliers correlate with high-`e_form`
  (i.e. likely unstable) 2D phases.
- **`dos_average.png`** — averaged SlakoNet DOS, split by PBE metal vs
  non-metal. Confirms that metals have meaningful DOS at E_F and non-metals
  have a clear gap.
- **`dos_examples.png`** — three individual DOS curves (one metal, one
  ~1 eV semiconductor, one ~4 eV wide-gap insulator) with formula and ID.

## Outputs

```
analysis.md                    (this file)
stats.txt                      (plain-text metrics, same numbers as above)
summary.csv                    (79,903 rows; sk_bandgap_eV column read by the cross-dataset aggregator)
confusion_matrix.png
dos_average.png
dos_examples.png
error_vs_eform.png
gap_distribution.png
parity_all.png
parity_nonmetals.png
residuals_all.png
residuals_nonmetals.png
```

`summary.csv` has a `sk_bandgap_eV` column, so the local cross-dataset
aggregator's v10 stub loader picks it up automatically on the next run.

## Reproducing

The plots, `stats.txt`, and `summary.csv` in this directory are pre-built. The original analysis script is no longer kept in the repo; outputs were generated from a cumulative `results/all_results.json` (built locally via `aggregate_results.py` when per-id JSONs were present). Peak memory during the original run stayed under ~2 GB by streaming the JSON.
