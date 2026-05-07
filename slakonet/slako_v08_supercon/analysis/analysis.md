# SlakoNet on `alex_supercon` — analysis

## Dataset

`alex_supercon.json.zip` — 8,253 candidate superconductors from the Alexandria
materials database (`agm…` ids → `alexandria-mat.de`). Each entry is a crystal
structure plus DFT/DFPT-derived electron-phonon descriptors used to estimate
Tc via Allen-Dynes/McMillan:

| field    | meaning                                  | unit       |
|----------|------------------------------------------|------------|
| `id`     | Alexandria material id (`agm…`)          | —          |
| `atoms`  | jarvis Atoms dict                        | —          |
| `Tc`     | predicted critical temperature           | K          |
| `la`     | λ, electron-phonon coupling              | —          |
| `wlog`   | log-averaged phonon frequency            | K          |
| `dosef`  | DOS at Fermi level                       | states/eV  |
| `debye`  | Debye temperature                        | K          |

There is **no DFT bandgap field**: these were screened as electron-phonon
superconductors, so the implicit reference is "metallic" (gap ≈ 0).

## Run

`../jslako_v08.py` on 4 GPUs, ~4h 12m wall.

| | count |
|---|---|
| source entries        | 8,253 |
| skipped (Z > 65)      | 3,426 |
| computed              | 4,827 |
| timeouts              | 0     |
| prep failures         | 0     |

Top elements in the skipped set: Os, Ta, Hf, Nb, Pt, Re, Ti, Ru, Tc, Mo
(SlakoNet supports Z ≤ 65).

## Bandgap → metallicity check

SlakoNet should predict gap ≈ 0 for everything. It does:

| | count | fraction |
|---|---|---|
| gap < 1 meV   | 649   | 13.4% |
| gap < 50 meV  | 4,695 | 97.3% |

Median gap 3.5 meV. The 0.5% with gap > 1 eV (e.g. agm002224543 = 2.89 eV) all
have Tc ≈ 0 in the source — SlakoNet is correctly flagging them as
non-superconducting outliers in the screening list.

See `plots/distributions.png`, `plots/tc_metallic_vs_gapped.png`.

## Tc descriptor correlations

Sanity check that the source descriptors behave like superconductivity theory
expects (Spearman ρ vs Tc):

| descriptor | ρ |
|---|---|
| λ (e-ph)        | **+0.99** |
| DFT dos(Ef)     | +0.50 |
| wlog            | −0.26 |
| Debye T         | ≈ 0   |

λ ↔ Tc correlation is essentially exact, as expected from Allen-Dynes.
SlakoNet bandgap → Tc is weakly negative (ρ = −0.18): tiny gaps go with
slightly higher Tc, consistent with metallicity.

See `plots/sk_vs_super_descriptors.png`.

## SlakoNet dos(Ef) vs DFT dos(Ef) — MAE

Both methods report the same observable; this is the per-structure benchmark.
SlakoNet's dos(Ef) is read off the saved DOS by linear interpolation at E = 0
(the saved energy grid is Fermi-aligned, [-10, 10] eV, 5,000 pts).

| metric | value |
|---|---|
| N             | 4,827           |
| MAE           | 1.36 states/eV  |
| RMSE          | 1.84            |
| median \|err\|| 1.03            |
| **bias (SK − DFT)** | **−0.93**  |
| Pearson r     | +0.69           |
| best-fit slope through origin | **0.63** |

Plot: `plots/mae_dosef.png` (parity hexbin + residual histogram + MAE binned by DFT
dos(Ef)).

The slope ≈ 0.63 (not 1.0, not 0.5) rules out a clean spin-counting unit
mismatch. SlakoNet under-predicts dos(Ef) systematically, with the
under-prediction scaling with magnitude. Worst residuals are all high-DFT-DOS
materials where SlakoNet collapses near zero — e.g. agm002167590: DFT = 13.10
vs SK = 0.83.

## Per-element bias

The per-element analysis aggregated the residual over every structure
containing each element (one vote per structure); 44 elements have N ≥ 20.

**All 44 elements have negative bias.** No chemistry escapes the under-binding.

| bias range | examples |
|---|---|
| −1.5 to −1.6 | La, Ni, Cr, Mo |
| −0.7 to −1.0 | Al, Ti, Si, Zr, Nb, Ru |
| −0.2 to −0.5 | Pd, Cu, Mn, Ba |

The ranking does not follow obvious period/group lines (La and Cr both lead
the list despite different chemistries). The bias roughly tracks the mean
DFT dos(Ef) of the subset — i.e. structures that *tend* to have high DOS get
under-predicted more in absolute terms.

Combined with the slope = 0.63, this is a **global multiplicative
under-binding** signature, not chemistry-specific. Plot: `plots/mae_by_element.png`.

## Hypothesis: hardcoded Gaussian broadening

From the slakonet source (`SimpleDftb.calculate_dos`):

```python
energy_grid, dos = self.calculate_dos(
    energy_range=(-10, 10),
    num_points=5000,
    sigma=0.1,        # ← hardcoded, not exposed
    fermi_shift=True,
)
```

σ = 0.1 eV is wide. For a sharp Fermi-level peak (van Hove or flat-band feature
common in superconductors), Gaussian smearing at this width flattens the peak
and lowers dos(Ef). DFT references commonly use tighter smearing (or tetrahedron
integration), which is consistent with SlakoNet sitting *below* DFT roughly
proportionally.

The high-level `compute_multi_element_properties` does not expose σ — there is
no clean external override.

## Diagnostic to run on the cluster

The σ-sensitivity sweep (script and outputs are kept local-only) did
the following:

1. Re-derived the 30 worst-bias structures from the per-id JSONs in `results/` (gitignored).
2. Monkey-patched `SimpleDftb.calculate_dos` to force σ ∈ {0.02, 0.05, 0.10,
   0.20, 0.40} eV (0.10 reproduces the baseline).
3. Re-ran SlakoNet at each σ.

**Decision rule:**
- MAE decreases monotonically as σ → 0.02 → broadening is the cause; the fix
  is to patch slakonet to plumb σ through `compute_multi_element_properties`.
- MAE flat or worse at small σ → the under-binding is structural (basis size
  or k-mesh), and σ is not the right lever.

## Bonus — easy fix for future runs

`SimpleDftb.calculate_dos` already returns a `dos_at_fermi` value. The script
discards it; we recover it post-hoc by interpolation. Adding
`properties["dos_at_fermi"]` to the output dict in `run_inference` would make
all future dos(Ef) analyses exact and one-line.

## Artifacts

| file | what |
|---|---|
| `plots/distributions.png`            | histograms of gap, Tc, dos(Ef), λ, Θ_D, wlog |
| `plots/sk_vs_super_descriptors.png`  | scatter: gap vs Tc, gap vs DFT dos(Ef) |
| `plots/tc_metallic_vs_gapped.png`    | Tc distribution split by SlakoNet metallicity |
| `plots/dos_average.png`              | average SlakoNet DOS, low-Tc vs high-Tc |
| `plots/mae_dosef.png`                | parity, residual histogram, MAE vs DFT bin |
| `plots/mae_by_element.png`           | per-element bias and MAE bar charts |
| `csv/summary.csv`                  | one row per structure, all scalar fields |
