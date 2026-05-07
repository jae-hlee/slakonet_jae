# v07 vacancy_db: ALIGNN PBE bandgap deep analysis

Reference handling: this dataset has no DFT bandgap field. The deep analysis here is framed around SK-vs-ALIGNN cross-method comparison rather than parity-vs-DFT. Metal/gap split at 0.05 eV.

## Setup

- **Dataset**: JARVIS `vacancy_db` (defective transition-metal supercells). N = 470 ALIGNN predictions, N = 444 SlakoNet predictions, N = 444 matched by `id`.
- **Method**: ALIGNN PBE (`mp_gappbe_alignn`) on atomgptlab CPU; SlakoNet on DSAI GPU (paired set from `slakonet/slako_v07_vacancy/results/all_results.json`).

## SlakoNet vs ALIGNN cross-check

| metric | value |
|---|---|
| N matched | 444 |
| MAE (ALIGNN vs SK) | **0.634 eV** |
| RMSE | 1.352 eV |
| ME (ALIGNN minus SK) | +0.561 eV |
| Pearson r | 0.396 |
| metal frac (SK / ALIGNN) | **91.2% / 54.3%** |
| metal/gap agreement | 61.3% |
| confusion (SK ref, ALIGNN pred) TN/FP/FN/TP | 237 / 168 / 4 / 35 |

## What this dataset shows: SK's transition-metal failure mode in the wild

This is the cleanest illustration of SlakoNet's documented transition-metal failure mode anywhere in the repo. SlakoNet predicts 91.2% of these defective supercells as metallic; ALIGNN predicts only 54.3% as metallic on the same 444 structures. The parity plot (`plots/sk_vs_alignn.png`) shows a vertical pile-up at SK gap = 0 against ALIGNN gaps spanning 0 to 6 eV.

Interpretation:
- v07 is dominated by transition-metal vacancies (Cu, V, Co, etc.) where SK's tight-binding parameters and its lack of spin polarization produce ~zero gap regardless of the true band structure.
- ALIGNN, trained on Materials Project bulk PBE bandgaps that include open-shell TM compounds, sees real gaps in many of these defective cells.
- The 168 false positives (SK metal, ALIGNN gap) versus 4 false negatives (SK gap, ALIGNN metal) is a 42x asymmetry. SK is missing structure that ALIGNN finds; not the other way around.

The full root-cause analysis of SK's failure mode (open-shell TM compounds + ionic fluorides) lives in `slakonet/slako_v03_alex/analysis/analysis.md`. v07 is the cleanest in-the-wild reproduction of that pattern.

## ALIGNN-side observations

- ALIGNN gap mean / std: 0.730 / 1.413 eV. Distribution is bimodal: 56% of entries predicted metallic (gap ≤ 0.05 eV), the rest spread 0.05 to 6 eV.
- Pearson(ALIGNN gap, formation energy `ef`) = 0.367. Modest positive correlation: defective cells with higher formation energy tend to have higher predicted gaps. Plot: `plots/alignn_vs_ef.png`.
- No DFT bandgap available, so no parity test against ground truth. The SK-vs-ALIGNN comparison above is the cross-method substitute.

## Caveats

- v07's `id` field is `JVASP-{parent}_{element}_{site}_{idx}` (defective vacancy variants), not `jid`. Matching across SK and ALIGNN uses this `id`.
- The atoms key is `defective_atoms` (not `atoms`); see top-level `CLAUDE.md` for the field-name gotcha. This affected predict-script extraction.
- N=470 ALIGNN vs N=444 SK: the 26-entry difference is SK silently dropping entries with elements outside `Z ≤ 65` (per the lanthanide / noble-gas wall documented in `slakonet/slako_v03_alex/analysis/analysis.md`). ALIGNN handles those elements via cgcnn-features-clamped-to-Z=103.

## Files

- `plots/sk_vs_alignn.png`: SK-vs-ALIGNN parity + residual histogram (with N/MAE/RMSE/ME annotations)
- `plots/confusion_sk_vs_alignn.png`: 2x2 metal/gap classification (SK reference, ALIGNN prediction)
- `plots/distribution.png`: ALIGNN bandgap distribution
- `plots/alignn_vs_ef.png`: ALIGNN bandgap vs formation energy
- `csv/metrics.csv`: full metric table
- `summary.md`: shorter at-a-glance writeup

## See also

- Cross-dataset SK-vs-ALIGNN narrative at `alignn/alignn_comprehensive_analysis/analysis.md`
- SK-side root-cause analysis at `slakonet/slako_v03_alex/analysis/analysis.md`
- Cross-dataset CSV at `slakonet/sk_vs_alignn_cross_dataset.csv`
