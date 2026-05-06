# slakonet_jae

Results of applying [SlakoNet](https://github.com/atomgptlab/slakonet) — a machine-learned Slater–Koster tight-binding model — to a range of crystalline, molecular, defect, surface, and interface datasets, together with [ALIGNN](https://github.com/atomgptlab/alignn) cross-checks on the same structures.

Each sub-project is a self-contained batch-inference pipeline: one script loads a dataset, runs SlakoNet (or ALIGNN) on every valid structure, and writes per-structure JSON outputs plus plots, summary CSVs, and a written `analysis.md` describing what the numbers say.

## Repository layout

```
slakonet/                     SlakoNet inference per dataset
  slako_v03_alex/             Alexandria PBE 3D crystals  (N = 31,211 paired)
  slako_v04_cccbdb/           NIST CCCBDB molecules       (N = 1,318)
  slako_v05_interface/        JARVIS interface_db slabs   (N = 433)
  slako_v06_surface/          JARVIS surface_db slabs     (N = 466)
  slako_v07_vacancy/          JARVIS vacancy_db defects   (N = 444)
  slako_v08_supercon/         Alexandria supercon set     (N = 4,827)
  slako_v09_1d/               Alexandria PBE 1D           (N = 8,636)
  slako_v10_2d/               Alexandria PBE 2D           (N = 79,903)
  slako_v11_alexwz/           Alexandria PBE 3D, no Z≤65 filter (in progress)
  slako_v12_all/              Alexandria PBE 3D, full 5M set, no filters (in progress)
  comprehensive_analysis/     Cross-dataset aggregation + unified plots

alignn/                       ALIGNN runs grouped by source dataset
  alignn_v03_alex/            Alexandria PBE 3D hull (paired with slakonet/slako_v03_alex)
    alignn_v1_pbe/            ALIGNN  mp_gappbe_alignn       (label-matched)
    alignn_v2_mbj/            ALIGNN  jv_mbj_bandgap_alignn  (TB-mBJ)
    alignn_v3_opt/            ALIGNN  jv_optb88vdw_bandgap_alignn
    comprehensive_analysis/   SlakoNet vs three ALIGNN variants, side-by-side
  alignn_v04_cccbdb/          CCCBDB molecules           jalignn4.py
  alignn_v05_interface/       JARVIS interface_db        jalignn5.py
  alignn_v06_surface/         JARVIS surface_db          jalignn6.py
  alignn_v07_vacancy/         JARVIS vacancy_db          jalignn7.py
  alignn_v08_supercon/        Alexandria supercon set    jalignn8.py
  alignn_v09_1d/              Alexandria PBE 1D          jalignn9.py
  alignn_v10_2d/              Alexandria PBE 2D          jalignn10.py
  alignn_v11_alexwz/          Alexandria PBE 3D, no Z<=65 filter (jalignn11.py)
  alignn_v12_all/             Alexandria PBE 3D, full set (jalignn12.py; 99/100 shards complete)
  analysis_summary.md         Cross-dataset roll-up of v04..v11 ALIGNN results
```

Every sub-project has a top-level inference script (`jslako_v*.py` for SlakoNet, `jalignn{N}.py` for the v04..v12 ALIGNN runs, `predict_alignn.py` for the older `alignn_v03_alex/alignn_v*` sub-runs), a `results/` directory of per-structure JSONs, and an `analysis/` directory of plots, metrics, and a written `summary.md` (or `analysis.md` in v03_alex).

## What this study finds

Two ML methods (SlakoNet tight-binding and pretrained ALIGNN graph network) are run on 9 datasets covering molecules, surfaces, interfaces, defects, superconductor candidates, and 1D/2D/3D Alexandria crystals (~4.5M structures total). Three core findings:

1. **ALIGNN wins by ~6x on the matched bulk-crystal set.** On the 6,781 Alexandria 3D crystals where both methods produced output, ALIGNN reaches MAE 0.18 eV vs PBE; SlakoNet sits at 1.03 eV. Same structures, same reference.

2. **The two methods fail in different shapes.** SlakoNet's failure is **bimodal**: median error is tiny (0.027 eV, most predictions are dead-on) but ~10% of structures collapse to predicted gap = 0 when DFT says 5+ eV. Disasters are concentrated on transition-metal compounds and ionic fluorides. ALIGNN's failure is **unimodal and gentler**: residuals spread Gaussian-ish around zero with no catastrophic collapse, but accuracy on non-metals is 4-5x worse than on metals. ALIGNN's worst residuals are also fluorides, but in the over-predict direction (off-hull fluoroborates where Alexandria PBE gives 0; ALIGNN may be more physically correct than the label here).

3. **Geometry matters more than model architecture.** ALIGNN's MAE rises cleanly as inputs leave its 3D-bulk training distribution: 3D bulk crystals 0.17-0.19 eV, slabs/interfaces/1D/2D ~0.5 eV (~3x worse), isolated molecules 3.4 eV. The same model degrades by 3x just from removing the periodic-bulk assumption.

The two methods are complementary, not redundant: SlakoNet offers physical interpretability and DOS access plus a known failure-mode set; ALIGNN offers uniformly decent accuracy within its training distribution but no transferability guarantee outside it. The v07 vacancy result (SK 91% metal vs ALIGNN 54% metal on the same 444 transition-metal-defect cells) is the cleanest direct illustration of SK's failure mode in a held-out cross-method test. The v12 fluoroborate worst-prediction pattern (DFT=0, ALIGNN=8 eV on Li/Sr/Ba/Ca-fluoroborates) is the cleanest illustration that "model error" and "label error" are not the same thing.

Full deep-analysis writeups: `slakonet/slako_v03_alex/analysis/analysis.md` (SlakoNet failure modes), `alignn/analysis.md` (ALIGNN cross-dataset narrative), `slakonet/slako_v11_alexwz/analysis/` (three-way DFT/SK/ALIGNN comparison).

## Headline SlakoNet results (from `slakonet/comprehensive_analysis/summary_table.csv`)

Band gap, all values in eV. MAE / RMSE / Pearson *r* are against the dataset's DFT reference (PBE for Alexandria and surface_db, OptB88vdW for interface_db, HOMO–LUMO for CCCBDB). Vacancy and supercon sets have no DFT gap reference available.

| Dataset        |     N  | SK mean | SK median | Frac metallic | Ref mean | MAE   | RMSE  |  r    |
|----------------|-------:|--------:|----------:|--------------:|---------:|------:|------:|------:|
| Alexandria 3D  | 31,211 | 1.54    | 0.01      | 0.63          | 1.22     | 0.93  | 1.65  | 0.81  |
| Alexandria 2D  | 79,903 | 1.16    | 0.02      | 0.62          | 0.67     | 0.62  | 1.33  | 0.89  |
| Alexandria 1D  |  8,636 | 1.87    | 0.31      | 0.40          | 1.09     | 0.99  | 1.70  | 0.88  |
| CCCBDB mols.   |  1,318 | 7.45    | 6.31      | 0.00          | 6.74     | 2.52  | 3.52  | 0.65  |
| interface_db   |    433 | 1.43    | 1.41      | 0.17          | 0.43     | 1.01  | 1.26  | 0.73  |
| surface_db     |    466 | 1.67    | 1.18      | 0.35          | 0.77     | 0.97  | 1.59  | 0.75  |
| vacancy_db     |    444 | 0.16    | 0.00      | 0.92          | —        | —     | —     | —     |
| alex_supercon  |  4,827 | 0.02    | 0.00      | 0.98          | —        | —     | —     | —     |

## Headline ALIGNN vs SlakoNet (Alexandria PBE 3D, paired N = 31,211)

From `alignn/alignn_v03_alex/comprehensive_analysis/`. Reference is Alexandria PBE indirect gap.

| Model                               | MAE   | RMSE  |   R²    | Non-metal MAE |
|-------------------------------------|------:|------:|--------:|--------------:|
| SlakoNet (DFTB)                     | 0.930 | 1.649 | −0.008  | 1.781         |
| ALIGNN `mp_gappbe_alignn` (PBE)     | **0.193** | **0.463** | **+0.920** | **0.274**     |
| ALIGNN `jv_mbj_bandgap_alignn`      | 0.752 | 1.461 | +0.208  | 1.236         |
| ALIGNN `jv_optb88vdw_bandgap_alignn`| 0.354 | 0.746 | +0.794  | 0.602         |

**What this says.** On the accuracy-matched ALIGNN checkpoint, non-metal MAE is ~0.27 eV — the accuracy ceiling for these structures. SlakoNet reaches 1.78 eV on the same subset, dominated by two failure modes (open-shell transition-metal compounds and ionic fluorides predicted as metals). On metals alone SlakoNet is actually the most accurate model (MAE 0.024 eV), because its default behaviour is to return ≈0. See `alignn/alignn_v03_alex/comprehensive_analysis/analysis.md` for the full breakdown, including the functional-shift calibration between PBE / TB-mBJ / OptB88vdW.

## Headline ALIGNN PBE results on v04-v11

ALIGNN `mp_gappbe_alignn` was run on each of v04 through v11 on the atomgptlab CPU cluster (PBE-only by design; see `alignn/CLAUDE.md` for the scope rationale and the deferred mBJ/Opt extension for v09-v12). Cross-dataset roll-up is at `alignn/analysis_summary.md` and `alignn/analysis_summary.csv`; per-dir details (parity plots, residuals, confusion matrices, metrics) live in each `alignn/alignn_v0*/analysis/`.

### Datasets with a DFT bandgap reference

| Dataset                       |     N    |  MAE (eV) |  RMSE | ME (bias) | Metal/gap acc. | Reference                                 |
|-------------------------------|---------:|----------:|------:|----------:|---------------:|-------------------------------------------|
| **v12_all** (full Alex 3D)    | 4,444,402|   0.185   | 0.551 |  +0.148   |       78.1%    | `band_gap_ind` (PBE), 99/100 shards       |
| **v11_alexwz** (3D bulk hull) |  115,535 | **0.168** | 0.476 |  +0.024   |    **89.1%**   | `band_gap_ind` (PBE)                      |
| v10_2d                        |   87,903 |   0.470   | 0.837 |  +0.190   |       64.1%    | `band_gap_ind` (PBE)                      |
| v09_1d                        |    9,540 |   0.485   | 0.755 |  −0.143   |       73.0%    | `band_gap_ind` (PBE)                      |
| v06_surface                   |      487 |   0.507   | 0.890 |  +0.190   |       (slab)   | `max(surf_cbm − surf_vbm, 0)`             |
| v05_interface                 |      587 |   0.531   | 0.717 |  +0.487   |       77.0%    | `optb88vdw_bandgap` clipped at 0 (NOT PBE)|
| v04_cccbdb (molecules)        |    1,330 |   3.365   | 7.931 |  −3.197   |       (mol.)   | `lumo − homo` (Hartree, ×27.2114 to eV)   |

**Three regimes.** v11 is in-distribution: 3D bulk Alexandria matches `mp_gappbe_alignn`'s Materials Project training distribution, and the model reaches MAE 0.17 eV on 115k structures with near-zero bias and median |err| of 0.015 eV. **v12** extends the same comparison to the full ~4.5M Alexandria 3D set (hull and off-hull); MAE rises modestly to 0.185 eV with metal/gap accuracy 78.1%, and the on-hull subset of v12 (N=114k) reproduces v11 exactly (MAE 0.168, accuracy 89.1%), confirming the array-sharded pipeline matches the single-job v11 run. v05, v06, v09, and v10 cluster around MAE 0.47 to 0.53 eV (~3x worse than v11/v12) because their geometries (1D, 2D, slab+vacuum, layered interface) sit outside the 3D-bulk training distribution. The pattern cleanly maps low-dimensional or vacuum-rich geometry to error magnitude. **v04 (CCCBDB molecules)** is a third, far-OOD regime at MAE 3.37 eV with a strong negative bias (ALIGNN under-predicts molecular HOMO-LUMO by ~3.2 eV); the number is most useful as a sanity bound, not a quantitative claim. Two compounding factors: the model is trained on crystals (graph neighborhood mismatch for isolated molecules) and the reference is molecular DFT (Gaussian basis, different functional), not solid-state PBE.

**v12 hull-bin breakdown** (in `alignn/alignn_v12_all/analysis/`): on-hull MAE 0.168 / accuracy 89.1%; near-hull (0 to 0.1 eV/atom) MAE 0.186 / 85.4%; off-hull (0.1 to 0.5) MAE 0.205 / 79.8%; far off-hull (>0.5) MAE 0.154 but accuracy drops to 66.0% as ALIGNN over-predicts small gaps on metallic far-off-hull entries. Bias grows monotonically from on-hull (+0.024) to off-hull (+0.178). Shard 9 of 100 is missing (~45k entries, easy to fill in via resubmit; numbers unlikely to shift past the 4th decimal).

**Functional-shift caveat for v05.** The interface_db reference is `optb88vdw_bandgap` (OptB88vdW), not PBE. ALIGNN predicts PBE. OptB88vdW typically gives slightly larger gaps than PBE for non-metals, so part of the +0.49 eV ALIGNN-over-DFT bias is the functional shift, not pure model error. 107 of 587 entries had negative OptB88vdW gaps (the documented interface SCF artifact) and were clipped to 0 before parity.

### Datasets without a DFT bandgap reference (SK vs ALIGNN cross-checks)

For v07 and v08 the source dataset has no DFT gap field, so the comparison is between SlakoNet and ALIGNN directly.

| Dataset                  | N (matched) | MAE vs SK (eV) | Metal frac (SK / ALIGNN) | Metal/gap agreement |
|--------------------------|------------:|---------------:|-------------------------:|--------------------:|
| **v07_vacancy** (defects)|         444 |          0.634 |        91.2% / 54.3%     |             61.3%   |
| v08_supercon             |       4,827 |          0.039 |        97.3% / 93.5%     |             92.4%   |

**v07** is the cleanest illustration of SlakoNet's documented transition-metal failure mode in this repo. SK predicts 91% of the defective cells are metallic; ALIGNN predicts only 54%. The parity plot in `alignn/alignn_v07_vacancy/analysis/sk_vs_alignn.png` shows a vertical pile-up at SK gap = 0 against ALIGNN gaps spanning 0 to 6 eV, the visual signature of SK's silent dropout on open-shell transition metals (full diagnosis in `slakonet/slako_v03_alex/analysis/analysis.md`). **v08** is a clean baseline: both methods predict the supercon candidates are metallic (consistent with the dataset's superconductor focus), with 92.4% agreement and an MAE of 0.04 eV. The high-Tc subset (Tc > 5 K, N=704) is 95.3% predicted metallic by ALIGNN, passing the sanity check.

### Schema corrections from the reruns

The first round of v04 and v05 outputs was missing DFT references (the predict scripts looked for fields that don't exist in those input zips: `hl_gap_hartree_eV` for v04, `scf_*` for v05). `jalignn4.py` and `jalignn5.py` were patched to print first-entry keys at startup and auto-propagate every scalar/string field from the input into the output JSON. Reruns on atomgptlab landed the actual reference fields:

- **v04**: `homo` and `lumo` (separately, in Hartree) plus `species` and `name`. Gap = `lumo − homo`.
- **v05**: `optb88vdw_bandgap`, `optb88vdw_cbm`, `optb88vdw_vbm`, `final_energy`, `offset`. The DFT reference is OptB88vdW, not PBE.

After the rerun both datasets are now parity-eligible (rows in the table above).

### Compute venue

- SlakoNet inference runs on Rockfish CPU (`parallel` partition, 48-core nodes). See `slakonet/job_template.sh`.
- ALIGNN inference runs on atomgptlab CPU (`main` partition, 256-core / 500 GB nodes). See `alignn/job_template.sh`. Conda lives at `/data/$USER/miniforge3` on atomgptlab; `dgl` must be installed explicitly via `pip install dgl==1.1.3 -f https://data.dgl.ai/wheels/repo.html` since it is not a transitive dependency of `pip install alignn`.

## Reproducing a run

Inference is designed for a SLURM cluster with GPUs. SLURM job scripts (`job.sh`) are not tracked in this repo — write your own based on your cluster's partition, walltime, and GPU/CPU layout. The general flow for any `slako_v*`:

```bash
# On the cluster, with the slakonet conda env active
conda activate slakonet
python jslako_v<N>.py          # auto-detects multi-GPU / single-GPU / multi-CPU
```

The inference environment needs `torch`, the full `slakonet` package (`pip install` from [atomgptlab/slakonet](https://github.com/atomgptlab/slakonet)), `jarvis-tools`, and `tqdm`. ALIGNN runs additionally need the `alignn` package and a pretrained model that `alignn.pretrained.get_figshare_model` can fetch on first use.

Each run filters to elements with Z ≤ 65 (`slako_v11_alexwz` and `slako_v12_all` are the exceptions — no element filter), checkpoints per-structure into `results/<id>.json`, and times out any single structure that exceeds 180 s. Re-running skips structures whose result file already exists.

### Data

No dataset zips ship with the repo. Download each from the [atomgptlab JARVIS databases page](https://atomgptlab.github.io/jarvis/databases/) and drop it into the matching sub-project working directory before running — the inference scripts look zips up by filename.

| Sub-project                                                                 | Expected zip                                                 |
|-----------------------------------------------------------------------------|--------------------------------------------------------------|
| `slako_v03_alex`, `slako_v11_alexwz`, `slako_v12_all`, `alignn_v03_alex/alignn_v{1,2,3}_*`   | `alexandria_pbe_3d_2024.10.1_jarvis_tools.json.zip` (1.1 GB) |
| `slako_v04_cccbdb`                                                          | `cccbdb.json.zip`                                            |
| `slako_v05_interface`                                                       | `interface_db_dd.json.zip`                                   |
| `slako_v06_surface`                                                         | `surface_db_dd.json.zip`                                     |
| `slako_v07_vacancy`                                                         | `vacancydb.json.zip`                                         |
| `slako_v08_supercon`                                                        | `alex_supercon.json.zip`                                     |
| `slako_v09_1d`                                                              | `alexandria_pbe_1d_2024.10.1_jarvis_tools.json.zip`          |
| `slako_v10_2d`                                                              | `alexandria_pbe_2d_2024.10.1_jarvis_tools.json.zip`          |

### Analysis

Each sub-project's `analysis/` directory ships with pre-built plots, a `summary.md` (or `analysis.md` in v03_alex), and a `metrics.csv` (or `summary.csv`) of the key scalars. Parity and residual plots in v04 to v11 ALIGNN dirs carry inset annotations of N, MAE, RMSE, and ME directly on the plot.

The cross-dataset layers are reader-only. Pre-built outputs live in:

- `slakonet/comprehensive_analysis/` (cross-dataset SlakoNet roll-up)
- `alignn/alignn_v03_alex/comprehensive_analysis/` (SlakoNet vs three ALIGNN variants on Alexandria 3D)
- `alignn/analysis_summary.md` and `alignn/analysis_summary.csv` (cross-dataset roll-up of the v04..v11 ALIGNN runs)

The scripts that produced them are kept local; analysis lives entirely in the artifacts on this side.

## Output schema

Each per-structure result JSON contains at least:

- `id` (or `jid` / `mat_id` depending on dataset) — structure identifier
- `sk_bandgap` — SlakoNet band gap (eV)
- `dos_values`, `dos_energies` — DOS on a Fermi-aligned grid, `E − E_F ∈ [−10, 10]` eV, 5000 points, Gaussian broadening σ = 0.1 eV
- `atoms` (or `defective_atoms` for v07) — the input geometry in JARVIS dict format
- Dataset-specific labels (PBE gap, formation energy, Tc, etc.)

ALIGNN predictions use the same per-structure layout with `alignn_bandgap` instead of `sk_bandgap`.

## Limitations

- **Element support.** SlakoNet ships Slater–Koster parameters for Z ≤ 65 (up to terbium); heavier elements are filtered out up front. For Alexandria PBE 3D, the combined Z ≤ 65 + `e_above_hull == 0` filter reduces the 4,489,295-entry dataset to 48,764 structures (see `alignn/alignn_v03_alex/alignn_v1_pbe/alignn_1282176.out`). **The nominal Z ≤ 65 ceiling is optimistic.** A post-hoc audit across every sister project shows that entries containing an f-block lanthanide (Ce–Tb, Z = 58–65) pass `ALLOWED_SYMBOLS` but silently fail inside `gpu_worker` — `generate_shell_dict_upto_Z65()` produces a shell dict that the rest of the model can't actually handle. Measured impact: v03 3D hull-filtered set drops 17,529 of 17,553 missing (99.9%) to lanthanides, v10 2D drops 8,000 of 8,000 (100%), v09 1D drops 904 of 904 (100%), v07 vacancy drops 22 of 26 (85%), v06 surface drops 19 of 21 (90%). A smaller noble-gas failure mode (Ne/Ar/Kr/Xe) accounts for the rest. **The effective usable ceiling is Z ≤ 57** (through La) excluding noble gases; full analysis in `slakonet/slako_v10_2d/analysis/analysis.md`.
- **Aggregated `all_results.json` files are not in the repo.** They exceed GitHub's 100 MB file-size limit (the Alexandria 3D file is 5.6 GB). They can be rebuilt from the per-structure JSONs, or re-generated by running the inference script.
- **Systematic failure modes for SlakoNet on non-metals.** On Alexandria 3D, ~4,909 PBE non-metals are predicted as metals — ~3,942 contain open-shell transition metals (no spin polarization in the current SlakoNet) and ~967 are ionic fluorides (bad SK parameters). Documented in `slakonet/slako_v03_alex/analysis/analysis.md`.
- **DOS broadening is fixed.** `σ = 0.1 eV` is hardcoded inside `SimpleDftb.calculate_dos`. Override requires a monkey-patch at runtime; the v08 sensitivity-test results are in `slako_v08_supercon/analysis/`.

## Upstream references

- SlakoNet — <https://github.com/atomgptlab/slakonet>
- ALIGNN — <https://github.com/atomgptlab/alignn>
- Alexandria materials database — <https://alexandria.icams.rub.de/>
- JARVIS-Tools — <https://github.com/usnistgov/jarvis>
- NIST CCCBDB — <https://cccbdb.nist.gov/>

## License

MIT — see `LICENSE`.
