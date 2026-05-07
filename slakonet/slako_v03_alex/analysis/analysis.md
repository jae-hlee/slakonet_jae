# SlakoNet failure-mode analysis

## Motivating question

The confusion matrix (`plots/confusion_matrix.png`) shows **4,909 PBE non-metals
predicted as metals** by SlakoNet, while the non-metal residual histogram
(`plots/residual_histogram_nonmetals.png`) shows that SlakoNet **overestimates**
the band gap on average. These appear contradictory: if SlakoNet overshoots gaps,
how can it be turning 4,909 non-metals into metals?

## Answer (short version)

The residual histogram and the confusion matrix describe **two different failure
modes mixed together**:

1. **Working group (~11,183 materials)**: SlakoNet predicts a gap and on average
   overshoots PBE — this is the positive bump in the residual histogram.
2. **Broken group (~4,909 materials)**: SlakoNet produces `sk_bandgap ≈ 0`
   regardless of the true PBE gap. These are the "false negatives" in the
   confusion matrix.

For the broken group, the median PBE gap is **1.54 eV** (75th percentile 2.51,
max 7.3) while the median SK gap is **0.001 eV**. It is not "slight
underestimation of small-gap semiconductors" — it is catastrophic collapse on
~30% of non-metals, independent of how large the real gap is.

Analogy: a thermometer that usually reads 5° too high but occasionally breaks
and reads 0. Averaging all readings still looks biased hot, but the "0"
readings happen on hot and cold days alike.

## Root cause: E_F is placed *inside* a band, not in the gap

The average SlakoNet DOS comparison
(`plots/failure_dos_comparison.png`) is the smoking gun:

- **Failing non-metals**: sharp DOS spike (~14 arb.) right at E_F
- **Working non-metals**: proper dip at E_F (~5 arb.)

SlakoNet is not crashing. The calculation finishes, the DOS is fully populated
(failing integral ~18k vs working ~21k — nearly identical), and the code
correctly reports "gap ≈ 0" because that's literally what the DOS shows at E_F.
The bug is physical, not numerical: the Fermi level is landing inside a band.

## Why E_F lands in a band: two distinct subpopulations

### (1) Open-shell 3d/4d transition metals — dominant failure mode

Element enrichment = (fraction of failing non-metals containing the element) /
(fraction of working non-metals containing it):

| Element | Enrichment | Note                              |
|---------|-----------:|-----------------------------------|
| **Mn**  | **45.5×**  | 5 unpaired d-electrons — extreme |
| **Cr**  | **11.8×**  |                                   |
| Ni      | 7.5×       |                                   |
| Co      | 7.3×       |                                   |
| Fe      | 6.9×       |                                   |
| Ru      | 4.9×       |                                   |
| V       | 4.1×       |                                   |
| Tc      | 3.2×       |                                   |
| Mo      | 2.8×       |                                   |

These are exactly the elements whose real ground states are **magnetic
insulators** (Mott insulators, antiferromagnets) — MnO, CrO₂, NiO, FeO, CoO
families. In reality they open a gap via **spin polarization** (and often
need a Hubbard *U*). PBE captures this because it's spin-polarized.

SlakoNet's inference call runs a **single
non-spin-polarized calculation** — the returned DOS has no spin index. Without
spin splitting, partially filled d-manifolds are required by symmetry to sit
degenerate at E_F, producing a metallic DOS. The extreme Mn enrichment (45.5×)
is the signature: Mn²⁺ has the maximally half-filled d⁵ shell, where
spin-unpolarized DFTB has no mechanism to open a gap at all.

By contrast, covalent-semiconductor elements (B, N, C, Si, Ge, As, Se, S, I, Br)
are all **underrepresented** in failures (enrichment 0.4–0.6×) because their
gaps come from bonding/antibonding splitting, which spin-unpolarized tight
binding handles fine.

### (2) Simple ionic wide-gap fluorides — SK parameterization failure

The 20 failing cases with the largest PBE gap are dominated by alkali /
alkaline-earth fluoride perovskites and elpasolites:

```
mat_id          formula        PBE_gap   SK_gap   n_atoms
agm003239266    K2LiAlF6         7.33    0.0018     30
agm003267835    K2SiF6           7.25    0.0004      9
agm002471317    KMgF3            6.98    0.0014      5
agm003062555    CsCaF3           6.91    0.0006      5
agm003284674    Cs2SiF6          6.91    0.0022      9
agm002210417    K2NaAlF6         6.89    0.0002     10
agm003236371    K2BeF4           6.89    0.0024     28
agm003207098    Cs2NaYF6         6.87    0.0009     10
agm002148595    Cs2KScF6         6.84    0.0003     10
agm003207094    Cs2NaScF6        6.80    0.0007     10
...
```

These are textbook closed-shell ionic insulators — not exotic. F is enriched
1.7× and Cs/K about 1.6×. The failure here is not about magnetism; it points
to a **Slater-Koster parameter problem** for the cation–F pairs in the
`generate_shell_dict_upto_Z65` parameter set: on-site energies and/or hopping
integrals for alkali/alkaline-earth–F are not reproducing the F 2p
valence-band / cation-derived conduction-band separation, so bands end up
overlapping at E_F.

## What is *not* the cause (ruled out)

- **Structure size**: failing median = 15 atoms, working median = 18. Similar.
- **Stability**: failing structures are actually *more* stable
  (mean e_form = −1.89 vs −1.66 eV/atom). Not a relaxation/off-hull artifact.
- **Crashes/timeouts**: DOS integrals are comparable between groups, so
  calculations complete normally.
- **Small-gap edge cases being slightly underestimated**: failing median PBE
  gap is 1.54 eV with failures extending to 7.3 eV — the failure rate is
  essentially independent of the true gap magnitude.
- **Number of elements / chemistry complexity**: failing and working sets have
  nearly identical ternary/quaternary ratios.

## Follow-up verification: PBE metal contamination

**Question:** Are the 14,715 "true-negative" metals contaminated by the same
failure mode — i.e., is SlakoNet calling them metal *for the wrong reason*
(E_F pinned in a d-band spike, not a real Fermi surface)?

**Result from `investigate_pbe_metals.py`:** the TN cell is mostly clean.

| Group                          |   N   | SK gap ≤ 0.1 | Mean DOS@E_F | Median DOS@E_F |
|--------------------------------|------:|-------------:|-------------:|---------------:|
| PBE metal, has open-shell TM   |  6165 |      98.9%   |      5.61    |      3.64      |
| PBE metal, no open-shell TM    |  8954 |      96.3%   |      4.11    |      2.84      |
| Failing non-metal (FN, 4909)   |  4909 |     100.0%   |   **13.21**  |    **9.31**    |
| Working non-metal              | 11183 |       0.0%   |      4.40    |      1.66      |

Both PBE-metal subsets have moderate DOS@E_F (4–6), consistent with genuine
metallic Fermi surfaces. They do **not** share the catastrophic spike (~13)
that characterizes failing non-metals. So SlakoNet is classifying most PBE
metals as metals *correctly*, not via the broken d-band-pinning mechanism.

Separately, the TM-enrichment story holds:

- **PBE metal**:       40.8% contain Mn/Cr/Fe/Co/Ni/V/Ru/Tc/Mo
- **PBE non-metal**:   27.7%
- **Failing (FN)**:    **67.4%**

The failing set is ~2.4× enriched in open-shell TMs relative to PBE non-metals
overall — confirming that the catastrophic-collapse mechanism is
TM-driven, while the clean PBE-metal subset has TM content tracking its
real metallic chemistry.

**Bottom line:** the confusion matrix's "metal" column can be trusted at the
aggregate level. The failure mode is concentrated in the FN cell.

## Follow-up verification: does SlakoNet support spin polarization?

**Short answer: no, not in the current release.** Verified by downloading
`slakonet==2026.4.1` from PyPI and inspecting the source directly
(the version running on the HPC cluster):

- `default_model(dir_path=None, model_name="slakonet_v0")` takes only a
  path and a model name — **no spin/magnetic/nspin kwargs** (`optim.py:3023`).
  It simply loads a pretrained `.pt` checkpoint from Figshare.
- `MultiElementSkfParameterOptimizer.compute_multi_element_properties()`
  signature (`optim.py:1498`):
  ```python
  def compute_multi_element_properties(
      self, geometry=None, shell_dict=None, kpoints=None, klines=None,
      phonons=False, get_fermi=False, get_energy=False, get_forces=False,
      get_bulk_mod=False, device=None, with_eigenvectors=False, cutoff=10.0,
  ):
  ```
  **No spin, magnetic, nspin, or collinear flag.** The only "spin"
  references in the entire codebase are comments on the hardcoded `2.0 ×`
  factor in the Fermi-level search (`fermi.py:208-210`):
  ```python
  n_mid = 2.0 * weighted_occ.sum(dim=(-1, -2), keepdim=True)  # spin factor
  ```
  This is the canonical **non-spin-polarized** convention — every orbital
  is assumed to hold exactly two electrons (one up, one down, identical).
- There is no SCF loop over spin densities, no alpha/beta Hamiltonians,
  no magnetic-moment handling, and no spin index on the returned DOS.
- The trained `slakonet_v0.pt` checkpoint has no spin-dependent parameters
  to begin with, so even if a spin flag were added, the network would need
  to be retrained from scratch to produce up/down Hamiltonian corrections.

**Implication:** the "enable spin polarization" suggestion from the
initial draft is **not actionable** against the shipped code. Fixing
the TM failure mode requires either:

1. Extending the framework to carry spin-dependent Hamiltonian blocks and
   retraining — a substantial undertaking, not a config toggle; or
2. Working around it at the dataset level — flag and exclude structures
   containing Mn, Cr, Fe, Co, Ni, V, Ru, Tc, Mo from SlakoNet-based
   screening, or fall back to a different model for those compositions.

The ionic-fluoride failure mode (K₂SiF₆, KMgF₃, …) is independent of
spin and would require refitting the cation–F Slater-Koster parameters
regardless.

## Cross-check: ALIGNN as an independent predictor

To separate "is this a SlakoNet bug?" from "is this an intrinsically hard
prediction?", the pretrained ALIGNN model `mp_gappbe_alignn` (trained on
Materials Project PBE gaps) was run on the same filtered Alexandria set
(`predict_alignn.py`). 48,764 structures were predicted; 31,211 overlap with
the existing SlakoNet results and form the paired comparison set below.
The remaining 17,553 ALIGNN entries have no SK counterpart. **Those 17,553
are not "timeouts / prep failures" and should not be re-run via
`rerun_missing.py`:** a post-hoc element audit shows **99.9 % (17,529 /
17,553) contain an f-block lanthanide (Ce, Pr, Nd, Pm, Sm, Eu, Gd, Tb)**,
the remaining 19 contain a noble gas, and only 5 are truly unexplained.
Zero completed structures contain any of those elements. This is the same
deterministic 4f-shell wall documented in `../../slako_v09_1d/analysis/analysis.md`
and `../../slako_v10_2d/analysis/analysis.md`: `generate_shell_dict_upto_Z65()`
nominally produces a shell dict for those elements so the `ALLOWED_SYMBOLS`
filter passes them, but `model.compute_multi_element_properties(...)` throws
inside `gpu_worker`, which silently swallows the exception. **Effective
usable ceiling is Z ≤ 57 (through La), not Z ≤ 65.** Rerunning those ids
will just re-fail.

### Regression vs PBE indirect gap (N = 31,211 paired)

| Subset                | SlakoNet MAE | ALIGNN MAE | SlakoNet R² | ALIGNN R² |
|-----------------------|-------------:|-----------:|------------:|----------:|
| **All**               |     0.930 eV |   **0.193 eV** |      −0.008 |  **0.920** |
| **Non-metals** (16,092) |     1.781 eV |   **0.274 eV** |      −1.068 |  **0.906** |
| **Metals** (15,119)     | **0.024 eV** |     0.106 eV |         n/a |       n/a |

ALIGNN is ~5× better in aggregate and ~6× better on non-metals. SlakoNet
still wins narrowly on the PBE-metal subset because its dominant failure
mode (collapse to `sk ≈ 0`) happens to coincide with the correct answer
there. Non-metal R² flips from −1.07 (worse than predicting the mean) to
+0.91.

### Classification (metal vs non-metal, threshold 0.1 eV)

|             | Accuracy | FN (PBE insulator → predicted metal) |
|-------------|---------:|-------------------------------------:|
| **SlakoNet** | 83.0%   | 4,909                                |
| **ALIGNN**   | **91.6%** | **777**                           |

ALIGNN cuts the false-negative rate by ~6× — a direct confirmation that
the 4,909-material "collapsed" cohort is a SlakoNet-specific failure, not
something intrinsic to the structures. ALIGNN trades this for more false
positives (1,836 vs 404) because it occasionally predicts small positive
gaps on real metals.

### Head-to-head per structure

ALIGNN is closer to PBE on **67.8%** of the 31,211 paired structures;
SlakoNet wins on 32.2%, concentrated in the metal subset where its
zero-gap tendency is accidentally correct.

### Interpretation in light of the failure modes above

- The **open-shell 3d/4d TM cohort** that SlakoNet collapses (Mn, Cr, Fe,
  Co, Ni, V, Ru, Tc, Mo — diagnosed above as missing spin polarization)
  is handled reasonably by ALIGNN because ALIGNN learns the PBE gap
  end-to-end, implicitly absorbing whatever spin-polarized physics PBE
  itself encoded during training. It does not need to reproduce the
  underlying SCF mechanism.
- The **alkali/alkaline-earth fluoride cohort** (K₂SiF₆, KMgF₃, …)
  likewise does not require a correct Slater-Koster parameterization —
  ALIGNN bypasses the SK framework entirely.
- Conversely, ALIGNN's weakness is in the metal subset, where its
  similarity-to-training-distribution heuristic produces occasional
  small-but-nonzero gaps on genuinely metallic compositions.

### Figures (ALIGNN cross-check)

- `../../../alignn/alignn_v03_alex/alignn_v1_pbe/analysis/plots/parity_three_way.png` — PBE vs SK, PBE vs ALIGNN, SK vs ALIGNN
- `../../../alignn/alignn_v03_alex/alignn_v1_pbe/analysis/plots/parity_three_way_nonmetals.png` — same, non-metals only
-  — residual histograms side-by-side
-  — confusion matrices side-by-side
- `../../../alignn/alignn_v03_alex/alignn_v1_pbe/analysis/plots/head_to_head_error.png` — per-structure |error| scatter
-  — PBE vs SK vs ALIGNN densities

## Implications for the dataset

1. **Overall MAE/RMSE numbers for non-metals are inflated** by mixing two
   disjoint populations (working vs collapsed). Reporting them separately
   is more faithful.

2. **Any downstream use of `sk_bandgap` as a screening signal should exclude
   or flag** structures containing Mn, Cr, Fe, Co, Ni, V, Ru, Tc, Mo and
   the alkali/alkaline-earth fluorides. Spin polarization is not available
   in the current release as a workaround. A practical alternative is to
   substitute `alignn_bandgap` for these problematic compositions — ALIGNN
   handles both failure modes and is available per structure in
   `results/alignn_predictions.json` (gitignored, kept local).

3. **The "PBE metal" subset appears trustworthy** at the aggregate level
   (see verification above), so the 14,715 TN count is not materially
   contaminated by hidden failures.

## Suggested next steps

1. **Re-run analysis** splitting non-metals into `(collapsed | sk ≤ 0.1)` vs
   `(gapped | sk > 0.1)` subsets so that parity plots and error histograms
   aren't averaging over two disjoint failure regimes.
2. **Composition blacklist** for downstream use: filter out structures
   containing Mn/Cr/Fe/Co/Ni/V/Ru/Tc/Mo when reporting SlakoNet gaps.
3. **File upstream issue** with the slakonet maintainers noting the
   systematic failure on open-shell 3d/4d TMs and ionic fluorides, with
   this diagnosis as context.
4. **Refit SK parameters** for alkali-F / alkaline-earth-F pairs — this
   is independent of the spin issue and would recover the fluoride subset.
5. **Do not rerun SlakoNet on the 17,553 missing structures.** The
   previously-suggested `rerun_missing.py` path is effectively dead: 99.9 %
   of those ids contain an f-block lanthanide (Ce–Tb) that inference
   silently rejects — see the lanthanide note in the cross-check section.
   The paired set cannot be extended until either SlakoNet gains 4f-shell
   support or `ALLOWED_SYMBOLS` is tightened to `Z ≤ 57` (and the
   comparison framed against the smaller 31,211-entry hull subset, which
   is what's already in this repo).
6. **Composition-gated hybrid predictor**: use ALIGNN on structures with
   open-shell TMs or fluoride cation–F pairs, and SlakoNet elsewhere —
   the head-to-head plot suggests this is roughly where each model wins.

## Figures referenced

- `plots/confusion_matrix.png` — original 2×2 metal/non-metal matrix
- `plots/residual_histogram_nonmetals.png` — SK − PBE gap distribution, non-metals
- `plots/false_negatives_diagnosis.png` — PBE gap and SK gap of the 4,909 failing structures
- `plots/failure_dos_comparison.png` — average SlakoNet DOS, failing vs working non-metals
- `plots/failure_structural.png` — atom-count and formation-energy distributions

