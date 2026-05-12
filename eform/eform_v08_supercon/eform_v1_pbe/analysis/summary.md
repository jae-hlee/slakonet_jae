# v08 e_form analysis: PBE arm (mp_e_form_alignn)

Pretrained model trained on **MP/PBE** formation energy (eV/atom). Inference on the **Alexandria alex_supercon candidates (Z<=65)** set, N = 4,827. **No DFT e_form reference** ships with this dataset. Analysis is distribution-only + cross-arm comparison.

## Distribution of predicted e_form (eV/atom)

| stat | value |
|---|---|
| N | 4,827 |
| min | -3.423 |
| p05 | -1.261 |
| median | -0.368 |
| mean | -0.451 |
| p95 | +0.025 |
| max | +0.309 |
| std | 0.418 |
| n(pred < 0) | 4,478 (92.8%) |
| n(pred > 0) | 349 (7.2%) |

## Cross-arm comparison

On N = 4,827 records (positional pairing):

| metric | value |
|---|---|
| PBE - Opt mean (signed) | -0.0025 eV/atom |
| PBE - Opt median | +0.0047 eV/atom |
| mean \|PBE - Opt\| | 0.0421 eV/atom |
| max \|PBE - Opt\| | 0.4257 eV/atom |
| Pearson r (PBE vs Opt) | 0.9898 |
| fraction \|diff\| <= 0.05 | 71.1% |
| fraction \|diff\| <= 0.10 | 92.0% |
| fraction \|diff\| <= 0.50 | 100.0% |

Top 25 cross-arm disagreements (full table at `csv/cross_arm_top_disagreements.csv`):

| id | formula | PBE | Opt | PBE - Opt |
|---|---|---|---|---|
| agm002311260 |  | -0.858 | -0.433 | -0.426 |
| agm003636381 |  | -0.315 | -0.683 | +0.368 |
| agm003636382 |  | -0.074 | -0.441 | +0.367 |
| agm001418576 |  | -0.559 | -0.221 | -0.338 |
| agm003447470 |  | -0.600 | -0.292 | -0.308 |
| agm001103417 |  | -1.413 | -1.109 | -0.303 |
| agm002181857 |  | -0.596 | -0.294 | -0.302 |
| agm003296043 |  | +0.309 | +0.007 | +0.302 |
| agm002163043 |  | +0.203 | +0.499 | -0.296 |
| agm002781337 |  | -1.038 | -0.743 | -0.296 |
| agm003449868 |  | -0.511 | -0.222 | -0.288 |
| agm002142631 |  | -1.429 | -1.149 | -0.280 |
| agm002142894 |  | -1.463 | -1.187 | -0.275 |
| agm002181521 |  | -0.423 | -0.153 | -0.270 |
| agm005113872 |  | -1.025 | -0.763 | -0.262 |
| agm003139380 |  | -1.009 | -0.748 | -0.262 |
| agm002165088 |  | -0.000 | +0.261 | -0.262 |
| agm002176887 |  | -2.376 | -2.115 | -0.261 |
| agm003637865 |  | -0.597 | -0.336 | -0.261 |
| agm002182830 |  | -1.047 | -0.786 | -0.261 |
| agm002170319 |  | -1.239 | -0.980 | -0.259 |
| agm002182685 |  | -0.837 | -0.579 | -0.259 |
| agm002166189 |  | -0.965 | -0.707 | -0.258 |
| agm003157485 |  | -0.515 | -0.259 | -0.256 |
| agm002163229 |  | +0.005 | +0.261 | -0.256 |


## Correlations with superconductor descriptors

| descriptor | N | Pearson r | desc min | desc median | desc max |
|---|---|---|---|---|---|
| Tc | 4,827 | +0.130 | 0.00 | 0.36 | 42.03 |
| la | 4,827 | +0.162 | 0.02 | 0.35 | 3.48 |
| dosef | 4,827 | +0.204 | -0.28 | 2.65 | 22.14 |
| wlog | 4,827 | -0.096 | 49.77 | 219.52 | 1304.81 |
| debye | 4,827 | -0.125 | 84.99 | 403.82 | 1160.14 |

### High-Tc subset (Tc > 5 K)

- N = 704 (14.6% of all)
- mean predicted e_form = -0.309 eV/atom
- median = -0.196 eV/atom
- n(<0) = 551 (78.3%)

## Files

- `csv/distribution.csv` — descriptive stats
- `csv/cross_arm_top_disagreements.csv` — top 50 records by |PBE - Opt|
- `plots/prediction_histogram.png` — this arm's predictions
- `plots/cross_arm_comparison.png` — PBE-vs-Opt scatter + histogram + CDF

## Caveats

- Supercon-focused; carries Tc/la/dosef/wlog/debye, no DFT e_form.
- All `formula` fields are None.
- Cross-arm comparison with `eform_v2_opt/analysis/` is the primary signal here (no DFT reference).
