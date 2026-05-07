# ALIGNN cross-dataset analysis

Aggregates every ALIGNN run in this repo: the three v03_alex variants
(`alignn_v1_pbe`, `alignn_v2_mbj`, `alignn_v3_opt`) on the Alexandria PBE 3D
hull-stable subset, and the single PBE-trained run on each of v04 through v12.
The pre-built artifacts in this directory mirror the structure of
`slakonet/slakonet_comprehensive_analysis/` (same column conventions in
`csv/summary_table.csv`, same plot family). This file plays both roles: a
directory-level catalog (what the artifacts are, how the numbers were
computed) and a deeper narrative (stratified parity, error-mode breakdowns,
per-element MAE, worst-prediction triage, and the SK-vs-ALIGNN cross-method
story). Per-dataset deep-analysis artifacts (the v03-style stratified plots +
`analysis.md` per project) live in each `alignn/alignn_v0*/analysis/`
directory.

## Datasets loaded

| key | title | kind | N | reference | ALIGNN checkpoint |
|-----|-------|------|---:|-----------|-------------------|
| v03_alex_pbe   | Alexandria 3D PBE (paired) | crystal   |  31,211 | Alexandria PBE `band_gap_ind`     | `mp_gappbe_alignn` (label-matched) |
| v03_alex_mbj   | Alexandria 3D PBE (paired) | crystal   |  31,211 | Alexandria PBE `band_gap_ind`     | `jv_mbj_bandgap_alignn` (functional shift) |
| v03_alex_opt   | Alexandria 3D PBE (paired) | crystal   |  31,211 | Alexandria PBE `band_gap_ind`     | `jv_optb88vdw_bandgap_alignn` (functional shift) |
| v04_cccbdb     | CCCBDB molecules           | molecule  |   1,330 | `(lumo - homo) * 27.2114`         | `mp_gappbe_alignn` |
| v05_interface  | Interface slabs            | interface |     587 | `optb88vdw_bandgap` clipped at 0  | `mp_gappbe_alignn` |
| v06_surface    | Surface slabs              | surface   |     487 | `max(surf_cbm - surf_vbm, 0)`     | `mp_gappbe_alignn` |
| v07_vacancy    | Vacancy defects            | defect    |     470 | -                                 | `mp_gappbe_alignn` |
| v08_supercon   | Alexandria supercon set    | supercon  |   4,827 | -                                 | `mp_gappbe_alignn` |
| v09_1d         | Alexandria 1D PBE          | low_dim   |   9,540 | `band_gap_ind`                    | `mp_gappbe_alignn` |
| v10_2d         | Alexandria 2D PBE          | low_dim   |  87,903 | `band_gap_ind`                    | `mp_gappbe_alignn` |
| v11_alexwz     | Alexandria 3D, no Z filter | crystal   | 115,535 | `band_gap_ind`                    | `mp_gappbe_alignn` |
| v12_all        | Alexandria 3D, full hull + off-hull | crystal | 4,444,402 | `band_gap_ind`             | `mp_gappbe_alignn` (99/100 shards) |

## Headline metrics (see `csv/summary_table.csv`)

| dataset | N | alignn_mean_eV | alignn_median_eV | frac_alignn_metal | ref_mean_eV | MAE_eV | RMSE_eV | pearson_r |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v03_alex_pbe   |    31,211 | 1.232 | 0.343 | 0.450 | 1.215 | **0.194** | **0.463** | **0.960** |
| v03_alex_mbj   |    31,211 | 1.569 | 0.251 | 0.439 | 1.215 |  0.757    |  1.462    |  0.808    |
| v03_alex_opt   |    31,211 | 1.065 | 0.095 | 0.502 | 1.215 |  0.357    |  0.746    |  0.898    |
| v04_cccbdb     |     1,330 | 3.826 | 3.855 | 0.008 | 7.023 |  3.365    |  7.931    |  0.279    |
| v05_interface  |       587 | 0.953 | 0.882 | 0.044 | 0.466 |  0.531    |  0.717    |  0.565    |
| v06_surface    |       487 | 0.975 | 0.589 | 0.269 | 0.784 |  0.507    |  0.890    |  0.692    |
| v07_vacancy    |       470 | 0.730 | 0.023 | 0.594 | -     | -         | -         | -         |
| v08_supercon   |     4,827 | 0.027 | 0.004 | 0.961 | -     | -         | -         | -         |
| v09_1d         |     9,540 | 0.927 | 0.247 | 0.384 | 1.070 |  0.485    |  0.755    |  0.873    |
| v10_2d         |    87,903 | 0.863 | 0.275 | 0.365 | 0.674 |  0.470    |  0.837    |  0.789    |
| v11_alexwz     |   115,535 | 0.677 | 0.015 | 0.655 | 0.653 | **0.168** | **0.476** | **0.933** |
| v12_all        | 4,444,402 |   -   |   -   | 0.709 |   -   |  0.185    |  0.551    |   -       |

The two strongest rows are v11_alexwz (`mp_gappbe_alignn`'s in-distribution best on Alexandria 3D bulk crystals) and v03_alex_pbe (the matched 31,211 paired subset used as the head-to-head benchmark vs SlakoNet). The mBJ and OptB88vdW v03 rows show the functional shift recovered cleanly from a non-metal slope fit (~1.23 for mBJ, ~0.94 for OptB88vdW); their MAE-against-PBE is dominated by that shift and not by model error.

## The three-regime story

The 9 single-checkpoint ALIGNN runs (v04 to v12) fall into three regimes determined by **how close the input geometry is to ALIGNN's Materials Project 3D bulk training distribution**:

| regime | datasets | typical MAE | what it tells us |
|---|---|---|---|
| **In-distribution** (3D bulk crystals at PBE) | v11_alexwz (115k), v12_all (4.44M) | **0.17 to 0.19 eV** | matches the model's published validation MAE; cleanest comparison |
| **OOD geometries** (low-dim, slabs, interfaces) | v05 interface, v06 surface, v09 1D, v10 2D | **0.47 to 0.53 eV** | ~3x worse than in-distribution; the price of vacuum / low coordination |
| **Far OOD + reference mismatch** | v04 CCCBDB molecules | **3.37 eV** with -3.2 eV bias | molecules + molecular DFT reference; useful only as a sanity bound |

The pattern cleanly maps geometry to error magnitude: removing the periodic-bulk assumption costs ~0.3 eV of MAE per step.

The **on-hull subset of v12** (114,389 entries) reproduces v11 exactly (MAE 0.168, accuracy 89.1%), confirming the array-sharded pipeline matches the single-job v11 run. v12's MAE rises with `e_above_hull` from 0.168 (on-hull) through 0.186 (near-hull) to 0.205 (off-hull), then drops to 0.154 at far-off-hull where DFT and ALIGNN both pile near zero (most far-off-hull entries are metallic in DFT). Metal/gap accuracy degrades monotonically across the same bins (89.1% to 66.0%) and bias grows from +0.024 to +0.178.

## Metal vs non-metal stratification: where the error lives

The full-set MAE numbers hide a strong asymmetry between metals (DFT gap <= 0.05 eV) and non-metals (DFT gap > 0.05 eV). This is the most important diagnostic ALIGNN's full-set MAE does not surface on its own:

| dataset | N | MAE all | MAE metals | MAE non-metals | metal/non-metal ratio |
|---|---|---|---|---|---|
| v11_alexwz | 115,535 | 0.168 | **0.081** | **0.369** | 4.6x |
| v12_all (full Alex 3D) | 4,444,402 | 0.185 | **0.153** | **0.544** | 3.6x |

**ALIGNN is much more accurate on metals than non-metals.** On v11, non-metal MAE (0.369 eV) is 4.6x the metal MAE (0.081 eV). On v12 the gap is similar (3.6x). The metals' very low MAE (0.081 eV on v11) reflects that ALIGNN's default behavior is to predict near-zero for systems that lack obvious gap signatures, which is correct on metals but underestimates wide-gap insulators.

The bias direction differs across regimes: on v11 the metal subset has positive bias (+0.076, slight overpredict) and the non-metal subset has negative bias (-0.096, underpredict the gap). On v12 both subsets share a positive bias because off-hull / far-off-hull entries dominate the non-metal pool with structurally unstable compositions where ALIGNN over-predicts small gaps.

## The dominant classification error: false positives

Across both v11 and v12, the dominant classification error is **DFT calls metallic, ALIGNN predicts a gap**:

| dataset | accuracy | TN (DFT metal, ALIGNN metal) | FP (DFT metal, ALIGNN gap) | FN (DFT gap, ALIGNN metal) | TP (DFT gap, ALIGNN gap) |
|---|---|---|---|---|---|
| v11_alexwz | 89.1% | 69,866 (60.5%) | **10,829 (9.4%)** | 1,730 (1.5%) | 33,110 (28.7%) |
| v12_all | 78.1% | 3,130,841 (70.4%) | **953,196 (21.4%)** | 20,152 (0.5%) | 340,213 (7.7%) |

False positives outnumber false negatives by **6x on v11 and 47x on v12**. ALIGNN tends to see structure where DFT does not. This is not random noise: the worst-prediction tables reveal a consistent pattern.

## Worst predictions are fluoride / fluoroborate compositions

The top-50 worst residuals on v12 are dominated by **lithium fluoroborates and alkali / alkaline-earth fluoroborates**, all with DFT gap ~0 eV and ALIGNN predicting 8 eV:

| mat_id | formula | DFT gap | ALIGNN | residual | e_above_hull |
|---|---|---|---|---|---|
| agm003523347 | LiB2F8 | 0.000 | 8.329 | +8.329 | 0.074 |
| agm003292742 | Li5F6 | 0.000 | 8.303 | +8.303 | 0.054 |
| agm005217488 | Li2F3 | 0.000 | 8.255 | +8.255 | 0.112 |
| agm005858487 | LiB3F12 | 0.000 | 8.255 | +8.255 | 0.056 |
| agm005628252 | Li3Be2F8 | 0.000 | 8.197 | +8.197 | 0.063 |
| agm005858583 | SrB3F12 | 0.000 | 8.130 | +8.130 | 0.029 |
| agm005858637 | BaB3F12 | 0.000 | 8.091 | +8.091 | 0.018 |

These are **off-hull polymorphs** (e_above_hull 0.02 to 0.11 eV/atom) of compositions that are physically known wide-gap insulators (LiBF4, Li2BeF4 etc., commonly used as solid electrolytes). The Alexandria PBE result of 0 eV likely reflects a metastable / unrealistic polymorph or a PBE artifact (PBE famously underestimates gaps and can collapse them to zero in unusual structures). **ALIGNN's 8 eV prediction may be more physically correct for the true ground-state polymorph than the Alexandria PBE reference.** The "worst predictions" are not necessarily ALIGNN errors; they are cases where ALIGNN identifies an inconsistency between the structure label and the expected wide-gap chemistry.

A similar pattern surfaces on v11 (which is on-hull only), where the worst residuals are **ytterbium fluorides and pseudo-lanthanide fluorides** (YbHfF6, YbF2, Li2YbHfF8 etc.) with DFT gaps of 6-7 eV that ALIGNN drives to ~0. There the error is in the opposite direction: ALIGNN under-predicts genuinely wide-gap fluorides containing f-block elements.

This connects directly to the SK-side story documented in `slakonet/slako_v03_alex/analysis/analysis.md`: SlakoNet ALSO struggles with fluorides and with transition-metal compositions, but its error mode is to predict near-zero gap regardless of the truth. **Both SK and ALIGNN are worst on fluorides and lanthanides**; SK's pattern is silent (predicts 0), ALIGNN's pattern is bidirectional (over-predicts for wide-gap fluorides and the structural-label mismatch flips it).

## Per-element MAE: F, Cs, O, Cl, Rb dominate

Element-resolved MAE on v12 (entries containing each element, min 500 entries):

| element | count | MAE (eV) | median \|err\| |
|---|---|---|---|
| F  | 138,258 | **0.855** | 0.288 |
| Cs | 124,590 | 0.673 | 0.263 |
| O  | 318,102 | 0.611 | 0.221 |
| Cl | 195,153 | 0.594 | 0.223 |
| Rb | 151,386 | 0.585 | 0.172 |
| Br | 189,682 | 0.525 | 0.202 |
| Np | 49,393 | 0.513 | 0.035 |
| I  | 147,107 | 0.505 | 0.211 |

The pattern is **halides + alkali metals + oxygen**: ionic compounds with wide PBE gaps that ALIGNN sometimes hits and sometimes collapses (or where the structural-label mismatch above flips the residual sign). Np stands out because the median error is small (0.035) but the MAE is large, indicating a long-tailed distribution: most Np compounds are predicted well but a handful have extreme errors.

On v11 the worst-MAE elements are similar (Np, F, Cl, Cs, Yb, Rb, Br, K, I, O) but with smaller absolute MAEs because v11 is hull-filtered and contains fewer of the structurally unusual off-hull compositions that produce v12's biggest errors.

## Error vs formation energy

|ALIGNN - DFT| has a Pearson correlation of **-0.333 on v11 and -0.371 on v12** with `e_form`. Negative correlation: less stable structures (higher formation energy) tend to have **smaller** absolute errors. This is the same artifact as the far-off-hull MAE drop: high-e_form structures are usually metallic in DFT (and ALIGNN agrees), so absolute errors stay small even though metal/gap classification accuracy is worst there.

## SlakoNet vs ALIGNN cross-method comparisons (v04 to v11)

This is the cleanest one-to-one cross-method view: same structures, same dataset, both methods predicting bandgap. DFT may or may not be available as a third reference.

| dataset | N matched | MAE (ALIGNN-SK) | RMSE | ME (ALIGNN-SK) | Pearson | metal frac (SK / ALIGNN) | metal/gap agreement |
|---|---:|---:|---:|---:|---:|---:|---:|
| v08_supercon (Tc-focused, no DFT) | 4,827 | **0.039** | 0.176 | +0.008 | 0.19 | 97.3% / 93.5% | **92.4%** |
| v06_surface (slabs) | 466 | 1.066 | 1.527 | -0.723 | 0.71 | 33% / 21% | 82.0% |
| v05_interface | 433 | 0.883 | 1.091 | -0.485 | 0.40 | 17% / 2.5% | 81.5% |
| v09_1d (Alex 1D) | 8,636 | 1.164 | 1.893 | -0.940 | **0.86** | 32% / 31% | 71.4% |
| v07_vacancy (TM defects) | 444 | 0.634 | 1.352 | +0.561 | 0.40 | 91.2% / 54.3% | 61.3% |
| v10_2d (Alex 2D) | 79,903 | 0.864 | 1.532 | -0.300 | 0.76 | **59% / 28%** | **61.9%** |
| v04_cccbdb (molecules) | 1,324 | 3.928 | 5.291 | **-3.632** | 0.61 | 0% / 0.3% | 99.7% |
| v11_alexwz (3D bulk hull, partial SK) | 6,781 | 1.084 | 2.068 | -0.314 | n/a | 64% / 54% | 72.2% |

(Per-dataset plots: `slakonet/slako_v0*_*/analysis/plots/sk_vs_alignn.png` plus matching `confusion_sk_vs_alignn.png`. Cross-dataset CSV: `slakonet/slakonet_comprehensive_analysis/csv/sk_vs_alignn_cross_dataset.csv`. The v11 row above is from the three-way analysis on the matched 6,781 subset, not the SK-vs-ALIGNN-only run.)

### Cross-cutting patterns

**SK predicts more metals than ALIGNN on every dataset except molecules.** This is the most consistent cross-method asymmetry. The gap is largest on v10 2D (SK 59% vs ALIGNN 28%, a 31-percentage-point spread) and v07 vacancy (SK 91% vs ALIGNN 54%). This is the wild-side picture of SK's documented transition-metal-and-fluoride error mode showing up across geometries: SK silently collapses to zero gap on chemistries it can't handle, inflating its metal-frac across all datasets.

**Strongest rank correlation appears on v09 1D (Pearson 0.86), highest MAE on v04 molecules (3.93 eV).** v09 is the case where the methods agree on which structures have bigger gaps than others (rank), but disagree systematically by ~1 eV on magnitude (SK biased high). For paper purposes, this is a clean signal that geometry-OOD doesn't destroy rank stability across methods, only absolute calibration.

**v07 is the cleanest illustration of SK's TM error mode.** The vertical pile-up at SK gap = 0 against ALIGNN gaps spanning 0 to 6 eV (visible in `alignn/alignn_v07_vacancy/analysis/plots/sk_vs_alignn.png`) is the visual signature of SK's silent dropout on open-shell transition metals. SK's 91.2% metal-frac on transition-metal vacancies is essentially the silent-dropout rate in disguise.

**v08 is a clean baseline.** Both methods agree the supercon candidates are metallic (92.4% classification agreement, MAE 0.039 eV between methods). This confirms the cross-method pipeline is sound when both methods are operating in their reliable regimes.

**v04 molecules show the largest absolute disagreement (3.93 eV).** Both methods are out-of-distribution (ALIGNN trained on crystals, SK trained on solid-state DFT), and they sit on opposite sides of the reference: SK predicts much larger gaps than ALIGNN (-3.6 eV ME). The 99.7% classification agreement is essentially trivial since both methods see molecules as non-metallic; the magnitude disagreement is where the real OOD signal is.

**v11 SK-vs-ALIGNN (partial, 6,781 entries)** sits in the middle: MAE 1.08 eV between methods, similar to v05/v06/v09/v10 cross-method MAEs. This is consistent with v11 being in-distribution for ALIGNN (which is why ALIGNN-vs-DFT MAE is 0.18) but mid-distribution for SK (where catastrophic outliers drag MAE to 1.0). When the v11 SK run completes, the matched subset will grow from 6,781 toward ~115k and these numbers will firm up.

## Three-way comparison on v11 (DFT vs SK vs ALIGNN, on the matched subset)

`slakonet/slako_v11_alexwz/analysis/` carries a three-way comparison on the 6,781 v11 entries that have both SK and ALIGNN predictions (after dropping 766 SK inf values and entries SK didn't run on):

| comparison | N | MAE (eV) | RMSE | ME | metal/gap acc. |
|---|---|---|---|---|---|
| DFT vs SK     | 6,781 | 1.034 | 2.027 | +0.333 | 77.8% |
| DFT vs ALIGNN | 6,781 | 0.177 | 0.443 | +0.019 | 90.5% |
| SK vs ALIGNN  | 6,781 | 1.084 | 2.068 | -0.314 | 72.2% |

**ALIGNN dominates on this subset by ~6x in MAE and 13 percentage points in metal/gap accuracy.** The SK MAE of 1.034 eV is dragged up by a long tail (median |err| 0.027 eV but p90 |err| 3.07 eV); ALIGNN has no equivalent tail (median |err| 0.015 eV, p90 |err| 0.467 eV). The SK errors are bimodal (mostly correct + catastrophic outliers); ALIGNN errors are unimodal (broader Gaussian-ish residuals with no catastrophic mode).

The full v11 SK run is incomplete (the CPU job crashed at 8h40m via a single-worker torch.load error; 7,547 of ~115,535 entries done). When the resubmit fills in, the three-way numbers will firm up on a much larger N.

## Functional-shift caveat (v05)

v05 interface_db's reference is `optb88vdw_bandgap` (OptB88vdW), **not** PBE. ALIGNN predicts PBE. OptB88vdW typically gives slightly larger gaps than PBE for non-metals. ALIGNN's +0.49 eV bias on v05 is partly the functional shift (ALIGNN's PBE prediction vs OptB88vdW reference), not pure model error. 107 of 587 entries had `optb88vdw_bandgap < 0` (interface SCF artifact) and were clipped to 0 before parity.

## Summary table: where ALIGNN works and where it doesn't

| use case | verdict | evidence |
|---|---|---|
| 3D bulk crystals at PBE level (Alexandria, Materials Project) | **Works**: MAE 0.17 to 0.19 eV, in-distribution | v11 0.168, v12 0.185, on-hull v12 0.168 reproducing v11 |
| Metals (DFT gap ~0) | **Works very well**: MAE 0.08 eV on v11 | metal/non-metal stratification |
| Wide-gap non-metals (DFT gap > 2 eV) | **Underpredicts**: MAE 0.37 eV on non-metals; ME -0.10 eV | non-metal residuals; v11 worst predictions are Yb / Pm fluorides |
| Off-hull polymorphs of ionic compounds (fluorides, fluoroborates) | **Disagrees with PBE**: predicts wide gap where Alexandria PBE says 0 | v12 worst predictions all off-hull fluoroborates; ALIGNN may be more physically correct than PBE here |
| Fluorides, halides, alkali metals (F, Cs, Cl, Rb, Br) | **Worst-MAE elements**: 0.5 to 0.85 eV per element | v11 + v12 per-element MAE tables |
| 1D / 2D / slab / interface geometries | **OOD penalty**: ~3x worse MAE (0.47 to 0.53 eV) | v05 / v06 / v09 / v10 |
| Isolated molecules | **Far OOD**: MAE 3.37 eV, -3.2 eV bias; sanity-bound only | v04 CCCBDB |
| Tc-focused or vacancy-only datasets without DFT gap | **No parity comparison possible**; SK-vs-ALIGNN cross-check works | v07 (TM disagreement), v08 (clean baseline) |
| SK vs ALIGNN cross-method (8 datasets) | **SK predicts ~30% more metals than ALIGNN** on most datasets; v07 / v10 are the biggest disagreements; v08 is the clean baseline | `slakonet/slakonet_comprehensive_analysis/csv/sk_vs_alignn_cross_dataset.csv` |

## Figures

- `plots/dataset_overview.png` - Per-dataset N, ALIGNN-predicted metallic fraction, ALIGNN gap median + IQR.
- `plots/gap_distributions.png` - ALIGNN-predicted gap histograms with DFT reference overlaid where present (12 panels, one per row in the headline table).
- `plots/parity_grid.png` - ALIGNN vs DFT reference, hexbin for large N. Inset annotates N and MAE per panel. Shows the three regimes cleanly: in-distribution v11/v03_pbe (tight diagonal), low-dim / slab / interface ~3x worse spread, far-OOD v04 molecules with cloud well below the diagonal.
- `plots/residual_distributions.png` - ALIGNN minus reference residual densities overlaid on a single axis. The v03_pbe and v11 distributions are sharp and centered; v05 / v06 / v09 / v10 are broader Gaussians; v04 is a wide negative-bias cloud (~3.2 eV under-prediction).
- `plots/error_summary.png` - MAE / RMSE / Pearson r bar charts per parity-eligible dataset.
- `plots/v08_tc_correlations.png` - ALIGNN gap vs Tc on the supercon candidate set, plus a high-Tc subset histogram. Sanity check: 95.3% of Tc > 5K candidates are predicted metallic, consistent with the dataset's superconductor framing.
- `plots/v03_functional_shift.png` - Three panels for v03 (PBE / mBJ / OptB88vdW vs Alexandria PBE reference). Hexbin parity with the y = x diagonal and a non-metal linear fit overlaid on the off-functional panels. The mBJ slope sits near 1.23 (the published TB-mBJ-vs-PBE shift); OptB88vdW slope sits near 0.94 (very close to PBE). This is the cross-functional calibration plot referenced by the head-to-head section in the top-level README.

## Notes

- v07 (vacancy) and v08 (supercon) have **no DFT bandgap reference**, so they appear in the dataset-overview and gap-distribution plots but not in parity / residual / error / functional-shift plots. The interesting analysis on those two is the SK-vs-ALIGNN cross-method comparison; per-dir details in `alignn/alignn_v07_vacancy/analysis/analysis.md` and `alignn/alignn_v08_supercon/analysis/analysis.md`.
- v04 (CCCBDB) reference is computed as `(lumo - homo) * 27.2114` from raw Hartree HOMO/LUMO values in the source. The separate column `hl_gap_hartree_eV` in `slakonet/slako_v04_cccbdb/analysis/csv/summary.csv` is misleadingly named (values are already in eV); use the formula above for clean parity.
- v06 reference uses `max(surf_cbm - surf_vbm, 0)` per the surface_db schema gotcha (the bundled `scf_*` reference is bulk-on-slab-vacuum scale and is wrong; see the slakonet-side workflow docs (kept local) for the documented fix).
- v12_all has no per-row `pred_mean` / `pred_median` / `ref_mean` / `r` because the full predictions JSON (~3 GB) is not loaded into the aggregator; numbers come from the per-shard `metrics.csv` rollup. The on-hull subset of v12 (114,389 entries) reproduces v11_alexwz exactly, which is the cross-check that the array-sharded pipeline matches the single-job v11 run.
- Metallic threshold used for coverage bars: ALIGNN gap < 0.10 eV (matches the slakonet-side convention).
- All numbers regenerated from per-row predictions in `alignn/alignn_v*/results/alignn_predictions.json` and `alignn/alignn_v03_alex/pbe_mbj_opt_analysis/merged_predictions.json`. The aggregation script is kept local off-repo (per the standing rule on Claude-generated analysis scripts).

## Metal/non-metal threshold conventions

Three thresholds for "metal" appear across this repo, applied to different things:

- `pbe_ref == 0` (strictest, DFT-side): Alexandria flags metallic structures with an exact-zero gap. Used in the v03 bootstrap CI artifact (`pbe_mbj_opt_analysis/bootstrap_ci.json`) for the most conservative ground-truth metal split, since the ranking-stability claim should be insensitive to a tolerance choice.
- `pbe_ref <= 0.05` (conventional, DFT-side): standard ML stratification threshold that absorbs sub-50 meV DFT noise (sub-zero or near-zero gaps that are physically metallic) into the metal class. Used in the manuscript Section 6 v11 stratified MAE (80,695 metals / 34,840 non-metals).
- `pred < 0.10` (or `< 0.05`) (model-side): applied to *predictions*, not the reference. Reports "what fraction of structures the model predicted as metallic" with a small tolerance for near-zero outputs. Used in the `frac_*_metal` columns of the cross-dataset summary tables (`< 0.10` in `csv/summary_table.csv`; `< 0.05` in the top-level README's headline ALIGNN table on the paired subset, which uses the tighter cut to match the SlakoNet headline's tighter cut).

The three are not interchangeable: `== 0` and `<= 0.05` classify the *ground truth* (DFT label); `< 0.10` (or `< 0.05`) classifies the *prediction*.

## See also

- `alignn/alignn_v0*/analysis/analysis.md` and the stratified plots / CSVs alongside (per-dataset deep analysis).
- `alignn/alignn_v03_alex/pbe_mbj_opt_analysis/analysis.md` - SK vs three ALIGNN variants on the v03 paired set, including bootstrap CI on the headline MAEs.
- `slakonet/slakonet_comprehensive_analysis/` - the SlakoNet counterpart to this directory (same column conventions in `csv/summary_table.csv`).
- `slakonet/slako_v11_alexwz/analysis/` - three-way SK / ALIGNN / DFT comparison on v11.
- `slakonet/slakonet_comprehensive_analysis/csv/sk_vs_alignn_cross_dataset.csv` - SK-vs-ALIGNN per-dataset cross-method comparisons.
- `slakonet/slako_v03_alex/analysis/analysis.md` - SK-side error-mode analysis (the bar this writeup tries to clear on the ALIGNN side).
