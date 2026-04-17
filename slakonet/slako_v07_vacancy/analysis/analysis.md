# SlakoNet on the JARVIS Vacancy Database — Analysis

## What we're predicting

SlakoNet (a Slater–Koster tight-binding neural network) is used as a fast
surrogate for DFT to predict the **electronic structure of defective
crystals** — specifically the **band gap** and **density of states (DOS)**
of each single-vacancy structure in the JARVIS vacancy database. The goal
is to see how well SlakoNet reproduces the effect of a vacancy on the host
electronic structure (gap opening/closing, defect states, metallic vs.
insulating character) without running full DFT on each defective supercell.

## Dataset

- **Source**: `vacancydb.json.zip` — the JARVIS single-vacancy database.
- **Total structures**: 530 defective supercells derived from JARVIS-DFT
  bulk crystals.
- **Per-entry fields used**: `defective_atoms` (supercell with one atom
  removed — the SlakoNet input), `ef` (DFT vacancy formation energy, eV),
  `bulk_atoms`/`bulk_atoms_prim` (pristine reference),
  `vacancy_class.symbol` (removed element), `vacancy_class.wyckoff_multiplicity`,
  `material_class` (`3D`/`2D`).

## Coverage

| Quantity | Value |
|---|---|
| Vacancy DB entries | 530 |
| SlakoNet results computed | 444 |
| Skipped (unsupported elements, Z > 65) | 86 |
| Finite `sk_bandgap` & `ef` | 444 |
| Effectively metallic (gap < 1 meV) | 170 (38.3%) |

Skipped structures typically contain heavy elements outside SlakoNet's
supported set (Z ≤ 65 / up to Tb); top elements appearing in skipped cells:
Te, Se, Sm, Bi, O, S, W, Pt, Hf, Ir.

## Summary statistics (N = 444)

| Quantity | Mean | Std | Min | Median | Max |
|---|---|---|---|---|---|
| SlakoNet band gap (eV) | 0.157 | 0.764 | 0.000 | 0.0015 | 7.160 |
| Formation energy `ef` (eV) | 3.088 | 2.383 | 0.120 | 2.275 | 11.867 |

By `material_class`:

| Class | N | mean gap (eV) | mean `ef` (eV) |
|---|---|---|---|
| 3D | 394 | 0.156 | 3.023 |
| 2D |  50 | 0.164 | 3.595 |

## Band gaps by vacancy element (top 15 by count)

| Vacancy | N | mean gap (eV) | mean `ef` (eV) |
|---|---|---|---|
| O  | 64 | **0.876** | 2.985 |
| Al | 62 | 0.003 | 2.993 |
| Ni | 41 | 0.004 | 3.058 |
| Fe | 32 | 0.002 | 1.652 |
| Co | 22 | 0.002 | 3.851 |
| Cu | 21 | 0.004 | 1.160 |
| Ti | 19 | 0.003 | 2.562 |
| P  | 12 | 0.005 | 2.729 |
| Zr | 11 | 0.003 | 3.555 |
| Mn | 11 | 0.001 | 6.526 |
| Se | 10 | 0.033 | 2.958 |
| C  | 10 | 0.060 | 4.060 |
| Cr |  9 | 0.001 | 5.406 |
| S  |  9 | **1.167** | 2.632 |
| Te |  8 | 0.039 | 1.397 |

Interpretation: substantial gaps are produced almost exclusively when
vacancies are created in **insulating oxides/sulfides** (O and S vacancies).
Vacancies in transition-metal hosts (Al, Ni, Fe, Co, Cu, Ti, …) come out
essentially metallic in SlakoNet, consistent with their metallic/intermetallic
parent compounds.

## Band gap vs. formation energy

The scatter of band gap vs. `ef` shows **no global correlation** — vacancy
energetics do not track the electronic gap across the whole dataset. The
data instead **clusters by host chemistry**: oxide/sulfide hosts occupy the
high-gap region, while metallic hosts sit along the `gap ≈ 0` axis at a
wide range of formation energies. See `gap_vs_ef.png`.

## Notable entries

**Largest band gaps** (all O/S vacancies in oxide/sulfide hosts):

| Rank | id | material | vacancy | gap (eV) | `ef` (eV) |
|---|---|---|---|---|---|
| 1 | JVASP-90143_O_c_24 | 3D | O | 7.160 | 3.062 |
| 2 | JVASP-94296_O_d_36 | 3D | O | 7.111 | 3.755 |
| 3 | JVASP-22523_O_c_32 | 3D | O | 5.522 | 3.294 |
| 4 | JVASP-53976_O_f_16 | 3D | O | 5.517 | 3.270 |
| 5 | JVASP-51876_O_f_16 | 3D | O | 5.515 | 3.272 |
| 6 | JVASP-107154_O_c_27 | 3D | O | 4.708 | 2.734 |
| 7 | JVASP-93342_O_f_24 | 3D | O | 2.962 | 4.530 |
| 8 | JVASP-8065_O_f_24 | 3D | O | 2.653 | 4.527 |
| 9 | JVASP-792_S_e_12 | 2D | S | 2.589 | 0.564 |
| 10 | JVASP-32_O_e_4 | 3D | O | 2.505 | 7.196 |

**Largest formation energies** (nearly all metallic in SlakoNet):

| Rank | id | material | vacancy | `ef` (eV) | gap (eV) |
|---|---|---|---|---|---|
| 1 | JVASP-32_Al_c_0 | 3D | Al | 11.867 | 0.001 |
| 2 | JVASP-1453_Al_b_0 | 3D | Al | 11.829 | 0.001 |
| 3 | JVASP-93342_Ge_a_0 | 3D | Ge | 11.531 | 0.000 |
| 4 | JVASP-8065_Ge_a_0 | 3D | Ge | 11.527 | 0.000 |
| 5 | JVASP-13526_Y_h_0 | 2D | Y | 11.450 | 0.001 |
| 6 | JVASP-688_B_e_0 | 2D | B | 10.101 | 0.016 |
| 7 | JVASP-60525_Se_c_16 | 2D | Se | 9.478 | 0.078 |
| 8 | JVASP-100703_In_c_0 | 3D | In | 9.243 | 0.000 |
| 9 | JVASP-22523_C_a_0 | 3D | C | 9.214 | 0.006 |
| 10 | JVASP-107154_Mg_a_0 | 3D | Mg | 9.164 | 0.004 |

## DOS

`dos_average.png` splits the dataset at the median gap (~0.0015 eV) and
averages the DOS in each half. The small-gap (effectively metallic) subset
shows finite DOS at the Fermi level; the large-gap subset shows a clear
pseudogap/gap around `E = 0`. `dos_examples.png` shows representative
single-structure DOS at gap targets of 0, 1, 3, and 6 eV, illustrating the
progression from metallic to wide-gap insulating character.

## Takeaways

1. SlakoNet produces sensible qualitative behavior: **oxide/sulfide hosts
   stay insulating under O/S vacancies**, and **metallic hosts remain
   metallic** when a metal atom is removed.
2. **No simple `gap ↔ ef` relation** exists across the full dataset; any
   regression would need host chemistry (or at least vacancy element +
   material class) as a feature.
3. The **limiting factor in coverage** is SlakoNet's element support
   (Z ≤ 65) — extending the Slater–Koster parameterisation to heavier
   elements would unlock the remaining 86 structures (notably
   Te/Se/Sm/Bi/W/Pt/Hf/Ir-containing cells).
4. Next validation step: compare SlakoNet gaps against the parent
   JARVIS-DFT gaps for the pristine hosts (and, where available, DFT gaps
   for the defective cells) to quantify accuracy beyond the qualitative
   metallic/insulating split.

## Artifacts

- `analysis/distributions.png` — histograms of SlakoNet gap and `ef`.
- `analysis/gap_vs_ef.png` — scatter of gap vs. `ef`, colored by
  `material_class`.
- `analysis/gap_by_vacancy_element.png` — boxplot of gap by vacancy element
  (top 20 by count).
- `analysis/dos_average.png` — average DOS for small-gap vs. large-gap
  subsets.
- `analysis/dos_examples.png` — example DOS at target gaps of 0, 1, 3, 6 eV.
- `summary.csv` — per-structure table (id, jid, material_class,
  vacancy_symbol, wyckoff, `ef`, `sk_bandgap`).
