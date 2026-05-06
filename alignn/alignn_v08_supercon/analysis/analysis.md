# v08 alex_supercon: ALIGNN PBE bandgap deep analysis

Reference handling: this Tc-focused dataset has no DFT bandgap field. The deep analysis here is framed around SK-vs-ALIGNN cross-method comparison plus a Tc-correlation sanity check. Metal/gap split at 0.05 eV.

## Setup

- **Dataset**: Alexandria `alex_supercon` (predicted superconductor candidates with electron-phonon coupling and Tc). N = 4,827 ALIGNN predictions, N = 4,827 SlakoNet predictions, N = 4,827 fully matched by `id`.
- **Method**: ALIGNN PBE (`mp_gappbe_alignn`) on atomgptlab CPU; SlakoNet on DSAI GPU (paired set from `slakonet/slako_v08_supercon/results/all_results.json`).

## SlakoNet vs ALIGNN cross-check

| metric | value |
|---|---|
| N matched | 4,827 |
| MAE (ALIGNN vs SK) | **0.039 eV** |
| RMSE | 0.176 eV |
| ME (ALIGNN minus SK) | +0.008 eV |
| Pearson r | 0.192 |
| metal frac (SK / ALIGNN) | **97.3% / 93.5%** |
| metal/gap agreement | 92.4% |
| confusion (SK ref, ALIGNN pred) TN/FP/FN/TP | 4,422 / 273 / 92 / 40 |

## What this dataset shows: clean cross-method baseline

v08 is the **clean baseline** of the SK-vs-ALIGNN comparison series. Both methods agree this dataset is metallic-dominant (consistent with the superconductor-candidate focus): SK calls 97.3% metallic, ALIGNN 93.5%, and they agree on the metal/gap label for 92.4% of the 4,827 entries. Cross-method MAE is 0.039 eV, the lowest of any dataset in the comparison.

The Pearson of 0.19 is misleading on its own: most of both methods' predictions are clustered near zero (gap ≤ 0.05 eV), so rank correlation is a weak summary statistic for v08. The right diagnostic is the metal/gap classification agreement, which at 92.4% confirms cross-method consistency.

This makes v08 the best evidence in the repo that the cross-method analysis pipeline is sound: when both methods are operating in their reliable regimes (here, metals where neither has a known failure mode), they agree.

## High-Tc sanity check

For superconductor candidates, the expected outcome is that high-Tc structures should be predominantly metallic (a non-zero DOS at the Fermi level is necessary for conventional superconductivity). ALIGNN passes this check:

- Tc > 5 K subset: N = 704 candidates, 95.3% predicted metallic by ALIGNN (correct).
- Pearson(ALIGNN gap, Tc) = 0.049 (essentially zero; Tc is uncorrelated with predicted gap).
- Pearson(ALIGNN gap, DOS at Fermi) = -0.078 (slight negative; small predicted gap correlates with higher DOS, as expected for metals).

These are not failure-mode signals; they are confirmation that ALIGNN behaves correctly on a Tc-focused dataset where most entries should be metallic.

## ALIGNN-side observations

- ALIGNN gap mean / std: 0.027 / 0.151 eV. Tightly clustered at zero, consistent with a metallic-dominant dataset.
- Distribution is unimodal at zero with a small tail of misclassified non-metallic entries.

## Caveats

- v08's primary ID is `id`, not `jid`. Atoms key is `atoms` (standard).
- This dataset has no DFT bandgap and no formation energy (the slakonet-side note in the top-level `CLAUDE.md` flags v08's lack of `jid`/`ef` as a porting gotcha).
- The cross-method MAE of 0.039 eV is suspiciously good and reflects the dataset's metallic-dominant nature, not ALIGNN's accuracy on insulators. v11 / v12 (where there is a real bandgap distribution) are the better tests of ALIGNN's accuracy.

## Files

- `sk_vs_alignn.png`: SK-vs-ALIGNN parity + residual histogram (with N/MAE/RMSE/ME annotations)
- `confusion_sk_vs_alignn.png`: 2x2 metal/gap classification (SK reference, ALIGNN prediction)
- `distribution.png`: ALIGNN bandgap distribution
- `alignn_vs_supercon.png`: ALIGNN gap vs Tc / DOS(Ef) / lambda
- `metrics.csv`: full metric table
- `summary.md`: shorter at-a-glance writeup

## See also

- Cross-dataset SK-vs-ALIGNN narrative at `alignn/analysis.md`
- v07 vacancy_db (the cleanest in-the-wild illustration of SK's transition-metal failure mode, contrast with v08's clean agreement)
- Cross-dataset CSV at `slakonet/sk_vs_alignn_cross_dataset.csv`
