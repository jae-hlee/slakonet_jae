# SlakoNet DB

Anonymized release of the dataset, predictions, and analysis code for
the accompanying NeurIPS 2026 Evaluations & Datasets track submission.

The benchmark is a paired band-gap evaluation on 31,211 hull-stable
Alexandria PBE 3D crystals. Each structure has predictions from four
models that span three regimes of cost and physical inductive bias:

- **SlakoNet**, a Slater–Koster tight-binding neural network that
  returns orbital-resolved electronic structure (band gap and a
  Fermi-aligned density of states on a 5,000-point grid).
- **ALIGNN** with three pre-trained checkpoints, all run as inference:
  - `mp_gappbe_alignn` (PBE-trained, label-matched to the reference)
  - `jv_mbj_bandgap_alignn` (TB-mBJ, gaps systematically opened vs PBE)
  - `jv_optb88vdw_bandgap_alignn` (OptB88vdW, vdW-corrected GGA)

The reference label is the Alexandria PBE indirect band gap on the
hull-and-Z $\leq$ 65 subset. The paired evaluation set is the
intersection of the four runs (N = 31,211).

## Repository layout

```
slakonet/
  slako_v03_alex/
    jslako_v3.py             SlakoNet inference script
    results/                 per-mat_id JSON outputs (band gap, DOS, geometry)
    analysis/                plots and analysis.md (failure-mode write-up)
alignn/
  alignn_v03_alex/
    predict_alignn.py        ALIGNN inference script (used by all three checkpoints)
    alignn_v1_pbe/           predictions from mp_gappbe_alignn
    alignn_v2_mbj/           predictions from jv_mbj_bandgap_alignn
    alignn_v3_opt/           predictions from jv_optb88vdw_bandgap_alignn
    comprehensive_analysis/
      merged_predictions.json   31,211 paired records (the central artifact)
      metrics.json              every numeric value used in the manuscript
      analysis.md               unified narrative comparison
      plots/                    parity, MAE-by-bin, confusion, etc.
```

## Headline results (paired Alexandria PBE 3D, N = 31,211)

Reference is the Alexandria PBE indirect band gap. Non-metal subset is
PBE gap > 0 (N = 16,092); metal subset is PBE gap = 0 (N = 15,119).

| Model                                | All MAE | Non-metal MAE | R² (all) |
|--------------------------------------|--------:|--------------:|---------:|
| SlakoNet (DFTB)                      | 0.930   | 1.781         | $-$0.008 |
| ALIGNN `mp_gappbe`  (PBE)            | **0.193** | **0.274**   | **+0.920** |
| ALIGNN `jv_mbj`     (TB-mBJ)         | 0.752   | 1.236         | +0.208   |
| ALIGNN `jv_optb88vdw` (OptB88vdW)    | 0.354   | 0.602         | +0.794   |

The label-matched PBE-trained ALIGNN sets the practical accuracy
ceiling at 0.27 eV non-metal MAE; SlakoNet reaches 1.78 eV on the same
subset. On metals SlakoNet is the most accurate of the four models
(0.024 eV) because its dominant failure mode is to predict gap $\approx
0$, which happens to coincide with the correct answer for metals.

The functional-shift calibration (linear fit on the non-metal subset)
recovers the published TB-mBJ vs. PBE 1.23$\times$ scaling and the
OptB88vdW vs. PBE 0.94$\times$ shrinkage from this benchmark alone, as
a measurement-validity check.

Full numerics, plots, and a documented failure-mode taxonomy
(open-shell transition-metal collapse, alkali / alkaline-earth fluoride
collapse, silent 4f-shell element wall) are in
`alignn/alignn_v03_alex/comprehensive_analysis/analysis.md` and
`slakonet/slako_v03_alex/analysis/analysis.md`.

## Reproducing

Inference was performed on a SLURM cluster with NVIDIA H100 GPUs.
SlakoNet inference per structure is roughly 12 to 15 s on an H100;
each ALIGNN inference run completes in under an hour on a single H100.

```bash
# Conda env with torch, slakonet, alignn, jarvis-tools
conda activate slakonet
cd slakonet/slako_v03_alex && python jslako_v3.py
cd ../../alignn/alignn_v03_alex/alignn_v1_pbe && python predict_alignn.py
# (repeat for alignn_v2_mbj and alignn_v3_opt)
```

The required input zip is
`alexandria_pbe_3d_2024.10.1_jarvis_tools.json.zip` (1.1 GB), available
from the upstream Alexandria materials database (cited in the
manuscript). Drop it into the SlakoNet and each ALIGNN working
directory before running.

The analysis layer that produces every figure and table in the
manuscript runs on a CPU laptop without GPU access:

```bash
# In alignn/alignn_v03_alex/comprehensive_analysis/, the released
# compare_all.py script regenerates every numeric value in the
# manuscript's headline tables and the linear-fit slopes from
# merged_predictions.json alone.
```

## Output schema

Each per-structure result JSON contains at least:

- `mat_id`, `formula`: Alexandria identifier and chemical formula
- `sk_bandgap`: SlakoNet band gap (eV)
- `dos_values`, `dos_energies`: Fermi-aligned density of states on a
  $[-10, +10]$ eV, 5,000-point grid, Gaussian broadening
  $\sigma = 0.10$ eV
- `band_gap_ind`, `band_gap_dir`: Alexandria PBE indirect / direct gaps
- `e_form`: formation energy (eV/atom)
- `atoms`: input geometry in JARVIS dict format

ALIGNN per-structure outputs use the same layout with `alignn_bandgap`
in place of `sk_bandgap`.

The merged record file
`alignn/alignn_v03_alex/comprehensive_analysis/merged_predictions.json`
combines the four predictions into one JSON keyed by `mat_id`:
`{mat_id, formula, pbe_ref, pbe_dir, e_form, slakonet, alignn_pbe,
alignn_mbj, alignn_optb}`.

## Limitations

- **Effective element coverage is Z $\leq$ 57**, not the nominal Z
  $\leq$ 65: SlakoNet silently fails on every structure containing an
  $f$-block lanthanide (Ce through Tb, Z 58 to 65) and on noble gases
  (no SK parameters). 17,529 of the 17,553 structures with an ALIGNN
  prediction but no SlakoNet prediction (99.9%) contain at least one
  lanthanide.
- **Two interpretable SlakoNet failure modes** on the non-metal subset:
  gap collapse on open-shell 3d/4d transition-metal compounds (Mn
  enriched 45.5$\times$ in the failing cohort) and gap collapse on
  alkali / alkaline-earth fluoride perovskites (incorrect cation–F
  Slater–Koster parameters). Full diagnosis in
  `slakonet/slako_v03_alex/analysis/analysis.md`.
- **DOS broadening is fixed** at $\sigma = 0.10$ eV (hard-coded in
  SlakoNet's calculate_dos). For DOS-based downstream features the
  broadening is a parameter of the dataset, not an artifact-free
  ground truth.

## Source data and pretrained models

The Alexandria PBE 3D source structures are available from the upstream
Alexandria materials database. SlakoNet and the three ALIGNN
checkpoints are released under their respective upstream licenses. All
upstream packages and source data are cited in the accompanying
manuscript bibliography.

- Alexandria materials database: <https://alexandria.icams.rub.de/>

## License

MIT, see `LICENSE`.
