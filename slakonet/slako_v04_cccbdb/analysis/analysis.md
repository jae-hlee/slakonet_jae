# v04_cccbdb: SlakoNet on NIST CCCBDB molecules

A far-OOD test of SlakoNet (a solid-state DFTB neural network) applied to isolated molecules with a molecular-DFT HOMO-LUMO reference. Both the inputs (no periodic boundary conditions) and the reference (Gaussian-basis molecular DFT, not solid-state PBE) sit far outside the training distribution. The numbers below are most useful as a sanity bound on cross-domain transfer, not as a quantitative claim about molecular property prediction.

The companion ALIGNN run on the same dataset is at `alignn/alignn_v04_cccbdb/`. The cross-dataset SlakoNet roll-up is at `slakonet/slakonet_comprehensive_analysis/`.

## Headline numbers

Band gap, all values in eV. Reference is the CCCBDB HOMO-LUMO gap (`lumo - homo`, Hartree converted to eV via x 27.2114). N = 1,318 valid molecules after the Z <= 65 filter and dropping 4 entries flagged as excluded.

| metric | SK vs HL-gap (eV) |
|---|---:|
| N | 1,318 |
| MAE | 2.52 |
| RMSE | 3.52 |
| ME (bias) | +0.71 |
| Pearson r | 0.65 |
| SK frac metallic | 0.00 |
| SK mean / median | 7.45 / 6.31 |
| HL-gap mean / median | 6.74 / 6.86 |

SK predicts no molecules as metallic (correctly, since CCCBDB is closed-shell molecules). The MAE of 2.52 eV with positive bias means SK systematically over-predicts the gap by about 0.7 eV on average; the 3.52 eV RMSE indicates that overprediction has a long upper tail (hydrocarbons and saturated organics push SK gaps into the 15-20 eV range while CCCBDB references stay around 7-12 eV).

## Where SK over-predicts the most

The largest residuals are saturated alkanes / alcohols / amines where SK reports gaps in the 15-25 eV range against CCCBDB references of 7-12 eV. Examples (jid: species, SK gap, HL gap):

- cc-100 CHF3: SK 23.4 eV vs HL 13.8 eV
- cc-44 C4H10O: SK 15.1 eV vs HL 9.2 eV
- cc-32 CH3OH: SK 15.7 eV vs HL 9.2 eV
- cc-88 (CH3)4: SK 15.9 eV vs HL 11.0 eV

The pattern is consistent: SK's solid-state DFTB parameters are tuned for periodic systems with delocalized states near the Fermi level. Applied to small saturated molecules, the Slater-Koster two-center approximation in vacuum gives a much wider HOMO-LUMO splitting than the CCCBDB DFT reference. This is an OOD artifact, not a tuning issue.

## SK vs ALIGNN cross-method

Both methods are out of distribution on molecules. The cross-method comparison is in `csv/sk_vs_alignn_metrics.csv`:

| metric | ALIGNN vs SK (eV) |
|---|---:|
| N matched | 1,324 |
| MAE | 3.93 |
| RMSE | 5.29 |
| ME (ALIGNN - SK) | -3.63 |
| Pearson r | 0.61 |
| SK metal frac | 0.000 |
| ALIGNN metal frac | 0.003 |
| Metal/gap agreement | 99.7% |

The two methods are OOD in opposite directions. SK over-predicts (mean 7.45 eV vs HL 6.74 eV, ME +0.71); ALIGNN under-predicts (its v04 ME against the same HL reference is -3.20 eV; see `alignn/alignn_v04_cccbdb/analysis/`). The 3.93 eV ALIGNN-minus-SK MAE is dominated by this opposite-direction bias: SK's solid-state DFTB inflates the gap, and ALIGNN's crystal-trained graph network deflates it. Pearson 0.61 between methods says the rank order is partially preserved across the two OOD methods, but the absolute scales disagree by several eV per molecule. Metal/gap agreement is trivially 99.7% because both methods see molecules as non-metallic.

## DOS observations

`plots/dos_examples.png` shows individual molecular DOS curves; `plots/dos_average.png` shows the cohort-averaged DOS on a Fermi-aligned grid. The DOS peaks are sharp (consistent with discrete molecular orbital energies) but the absolute energy positions shift relative to a periodic-system reference (no Fermi level for an isolated molecule means the SK-side Fermi alignment is a convention rather than a physical anchor). DOS-based downstream features should not be used for molecules without first re-anchoring to either HOMO or vacuum.

## Why include this dataset at all

CCCBDB is the cleanest cross-domain stress test available for this benchmark: a tightly curated set of 1,318 small molecules with well-defined HOMO-LUMO references, no periodic-boundary ambiguity, and elemental coverage that SlakoNet nominally supports (Z <= 65). The 2.52 eV MAE quantifies the upper bound of the cross-domain penalty, and the +0.71 eV bias (vs ALIGNN's -3.20 eV) pins the direction of each method's OOD drift. For users planning to apply either method outside its training distribution, this dataset is the most informative single number in the benchmark.

## Pointers

- Per-row scalars: `csv/summary.csv` (1,318 valid + 4 excluded rows; columns: `jid`, `species`, `homo_raw`, `lumo_raw`, `hl_gap_hartree_eV`, `hl_gap_ev_eV`, `sk_bandgap_eV`, `excluded`). Note: the column names `hl_gap_hartree_eV` and `hl_gap_ev_eV` are swapped relative to their values in the source CSV. The `_hartree_eV` column holds Hartree-converted-to-eV (the right reference); the `_ev_eV` column holds raw Hartree.
- SK vs ALIGNN aggregate: `csv/sk_vs_alignn_metrics.csv`.
- Cross-dataset row: `slakonet/slakonet_comprehensive_analysis/csv/summary_table.csv` (look for `v04_cccbdb`).
- Companion ALIGNN deep dive: `alignn/alignn_v04_cccbdb/analysis/analysis.md`.
- Cross-dataset SK-vs-ALIGNN narrative: `alignn/alignn_comprehensive_analysis/analysis.md` (also covers v04 in its molecular-OOD framing).
- Plots: `plots/parity_sk_vs_hl.png`, `plots/sk_vs_alignn.png`, `plots/distributions.png`, `plots/gap_distribution_comparison.png`, `plots/residual_histogram.png`, `plots/homo_vs_lumo.png`, `plots/confusion_sk_vs_alignn.png`, `plots/dos_average.png`, `plots/dos_examples.png`.
