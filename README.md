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
  alslak_v03_alex/            Alexandria PBE 3D hull (paired with slakonet/slako_v03_alex)
    slako_v03/                SlakoNet baseline + ALIGNN per-functional sub-runs
      alignn_v1_pbe/          ALIGNN  mp_gappbe_alignn       (label-matched)
      alignn_v2_mbj/          ALIGNN  jv_mbj_bandgap_alignn  (TB-mBJ)
      alignn_v3_opt/          ALIGNN  jv_optb88vdw_bandgap_alignn
    comprehensive_analysis/   SlakoNet vs three ALIGNN variants, side-by-side
  alslak_v04_cccbdb/          CCCBDB molecules — scaffolded for future ALIGNN runs
```

Every sub-project has a top-level `jslako_v*.py` (SlakoNet) or `predict_alignn.py` (ALIGNN), a `results/` directory of per-structure JSONs, an `analysis/` directory of plots and a written `analysis.md`, and a `summary.csv` with the key scalars.

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

From `alignn/alslak_v03_alex/comprehensive_analysis/`. Reference is Alexandria PBE indirect gap.

| Model                               | MAE   | RMSE  |   R²    | Non-metal MAE |
|-------------------------------------|------:|------:|--------:|--------------:|
| SlakoNet (DFTB)                     | 0.930 | 1.649 | −0.008  | 1.781         |
| ALIGNN `mp_gappbe_alignn` (PBE)     | **0.193** | **0.463** | **+0.920** | **0.274**     |
| ALIGNN `jv_mbj_bandgap_alignn`      | 0.752 | 1.461 | +0.208  | 1.236         |
| ALIGNN `jv_optb88vdw_bandgap_alignn`| 0.354 | 0.746 | +0.794  | 0.602         |

**What this says.** On the accuracy-matched ALIGNN checkpoint, non-metal MAE is ~0.27 eV — the accuracy ceiling for these structures. SlakoNet reaches 1.78 eV on the same subset, dominated by two failure modes (open-shell transition-metal compounds and ionic fluorides predicted as metals). On metals alone SlakoNet is actually the most accurate model (MAE 0.024 eV), because its default behaviour is to return ≈0. See `alignn/alslak_v03_alex/comprehensive_analysis/analysis.md` for the full breakdown, including the functional-shift calibration between PBE / TB-mBJ / OptB88vdW.

## Reproducing a run

Inference is designed for a SLURM cluster with GPUs. The general flow for any `slako_v*`:

```bash
# On the cluster, with the slakonet conda env active
conda activate slakonet
sbatch job.sh                  # h100 partition, 4 GPUs, ~72 hr
# — or interactively —
python jslako_v<N>.py          # auto-detects single vs multi-GPU
```

The inference environment needs `torch`, the full `slakonet` package (`pip install` from [atomgptlab/slakonet](https://github.com/atomgptlab/slakonet)), `jarvis-tools`, and `tqdm`. ALIGNN runs additionally need the `alignn` package and a pretrained model that `alignn.pretrained.get_figshare_model` can fetch on first use.

Each run filters to elements with Z ≤ 65 (`slako_v11_alexwz` and `slako_v12_all` are the exceptions — no element filter), checkpoints per-structure into `results/<id>.json`, and times out any single structure that exceeds 180 s. Re-running skips structures whose result file already exists.

### Data

No dataset zips ship with the repo. Download each from the [atomgptlab JARVIS databases page](https://atomgptlab.github.io/jarvis/databases/) and drop it into the matching sub-project working directory before running — the inference scripts look zips up by filename.

| Sub-project                                                                 | Expected zip                                                 |
|-----------------------------------------------------------------------------|--------------------------------------------------------------|
| `slako_v03_alex`, `slako_v11_alexwz`, `slako_v12_all`, `alslak_v03_alex/slako_v03/alignn_v{1,2,3}_*`   | `alexandria_pbe_3d_2024.10.1_jarvis_tools.json.zip` (1.1 GB) |
| `slako_v04_cccbdb`                                                          | `cccbdb.json.zip`                                            |
| `slako_v05_interface`                                                       | `interface_db_dd.json.zip`                                   |
| `slako_v06_surface`                                                         | `surface_db_dd.json.zip`                                     |
| `slako_v07_vacancy`                                                         | `vacancydb.json.zip`                                         |
| `slako_v08_supercon`                                                        | `alex_supercon.json.zip`                                     |
| `slako_v09_1d`                                                              | `alexandria_pbe_1d_2024.10.1_jarvis_tools.json.zip`          |
| `slako_v10_2d`                                                              | `alexandria_pbe_2d_2024.10.1_jarvis_tools.json.zip`          |

### Analysis

Each sub-project's `analysis/` directory ships with pre-built plots, an `analysis.md` write-up, and a `summary.csv` of the key scalars.

The cross-dataset layers are reader-only — no `slakonet` needed, just `pandas numpy matplotlib scipy`. Pre-built outputs live in `slakonet/comprehensive_analysis/` and `alignn/alslak_v03_alex/comprehensive_analysis/`. To regenerate the ALIGNN side:

```bash
python alignn/alslak_v03_alex/comprehensive_analysis/compare_all.py
```

## Output schema

Each per-structure result JSON contains at least:

- `id` (or `jid` / `mat_id` depending on dataset) — structure identifier
- `sk_bandgap` — SlakoNet band gap (eV)
- `dos_values`, `dos_energies` — DOS on a Fermi-aligned grid, `E − E_F ∈ [−10, 10]` eV, 5000 points, Gaussian broadening σ = 0.1 eV
- `atoms` (or `defective_atoms` for v07) — the input geometry in JARVIS dict format
- Dataset-specific labels (PBE gap, formation energy, Tc, etc.)

ALIGNN predictions use the same per-structure layout with `alignn_bandgap` instead of `sk_bandgap`.

## Limitations

- **Element support.** SlakoNet ships Slater–Koster parameters for Z ≤ 65 (up to terbium); heavier elements are filtered out up front. For Alexandria PBE 3D, the combined Z ≤ 65 + `e_above_hull == 0` filter reduces the 4,489,295-entry dataset to 48,764 structures (see `alignn/alslak_v03_alex/alignn_v1_pbe/alignn_1282176.out`). **The nominal Z ≤ 65 ceiling is optimistic.** A post-hoc audit across every sister project shows that entries containing an f-block lanthanide (Ce–Tb, Z = 58–65) pass `ALLOWED_SYMBOLS` but silently fail inside `gpu_worker` — `generate_shell_dict_upto_Z65()` produces a shell dict that the rest of the model can't actually handle. Measured impact: v03 3D hull-filtered set drops 17,529 of 17,553 missing (99.9%) to lanthanides, v10 2D drops 8,000 of 8,000 (100%), v09 1D drops 904 of 904 (100%), v07 vacancy drops 22 of 26 (85%), v06 surface drops 19 of 21 (90%). A smaller noble-gas failure mode (Ne/Ar/Kr/Xe) accounts for the rest. **The effective usable ceiling is Z ≤ 57** (through La) excluding noble gases; full analysis in `slakonet/slako_v10_2d/analysis/analysis.md`.
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
