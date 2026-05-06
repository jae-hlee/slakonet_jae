# ALIGNN PBE inference: cross-dataset summary

All runs use `mp_gappbe_alignn` on atomgptlab CPU. MAE reported only where DFT bandgap reference is present in the dataset.

| dataset | N | MAE (eV) | notes |
|---|---|---|---|
| alignn_v04_cccbdb | 1,333 | 3.365 | vs HOMO-LUMO gap (Hartree, 1330/1333); molecular OOD |
| alignn_v05_interface | 587 | 0.531 | vs OptB88vdW (clipped at 0); functional shift in ME |
| alignn_v06_surface | 487 | 0.507 | vs max(surf_cbm-surf_vbm, 0) |
| alignn_v07_vacancy | 470 | n/a | no DFT gap; SK vs ALIGNN below |
| alignn_v08_supercon | 4,827 | n/a | no DFT gap; SK vs ALIGNN below |
| alignn_v09_1d | 9,540 | 0.485 | vs band_gap_ind (PBE) |
| alignn_v10_2d | 87,903 | 0.470 | vs band_gap_ind (PBE) |
| alignn_v11_alexwz | 115,535 | 0.168 | vs band_gap_ind (PBE) |
| alignn_v12_all | 4,444,402 | 0.185 | vs band_gap_ind (PBE), full Alexandria 3D, 99/100 shards |

## Headline observations

- v11 (Alexandria 3D bulk crystals) has the lowest MAE (~0.17 eV) of the parity-eligible datasets. mp_gappbe_alignn was trained on Materials Project 3D bulk PBE gaps, so v11 is closest to the training distribution.
- v10 (2D) and v09 (1D) have ~3x higher MAE than v11 despite the same model. Low-dimensional materials are out-of-distribution for the crystal-bulk-trained graph network.
- v06 (surfaces) and v05 (interfaces) sit in the same OOD band as v09/v10 (MAE ~0.50-0.53 eV); slabs and interfaces are 3D-like with vacuum or layered termination.
- v04 (CCCBDB molecules) has the largest MAE (~3.4 eV) and a strong negative bias (ALIGNN under-predicts molecular HOMO-LUMO by ~3.2 eV on average). Two compounding factors: the model is trained on crystals (graph neighborhood mismatch for isolated molecules) and the reference is molecular DFT (typically Gaussian-basis with different functional), not solid-state PBE. The number is most useful as a far-OOD sanity bound, not a quantitative claim.
- v05 reference is OptB88vdW (NOT PBE), so the +0.49 eV ALIGNN-minus-DFT bias is partly the PBE-vs-OptB88vdW functional shift, not pure model error.
- v07 (vacancies) and v08 (supercon) lack DFT bandgap fields by design; analysis is SK-vs-ALIGNN cross-check rather than parity. The v08 sanity check (ALIGNN agrees high-Tc candidates are metallic) passes (95.3% predicted metallic for Tc > 5 K).
- v12 (full Alexandria 3D, 4.44M of 4.49M entries, 99/100 shards) at MAE 0.185 eV with metal/gap accuracy 78.1%. Hull-bin breakdown shows the on-hull subset (N=114k) reproduces v11_alexwz exactly (MAE 0.168, accuracy 89.1%), confirming the array-sharded pipeline matches the single-job v11 run; off-hull entries drift toward higher MAE and lower accuracy with a growing positive bias as e_above_hull increases.

## SK vs ALIGNN cross-checks (v07, v08)

- **v07 vacancy** (N=444 matched by id): MAE 0.63 eV, ALIGNN biased high vs SK by +0.56 eV, metal/gap agreement 61%. Big disagreement on metallicity (SK 91% metal, ALIGNN 54% metal). Likely the documented SK failure mode for transition-metal vacancies (open-shell metals predicted with ~0 gap by SK; ALIGNN sees more gap).
- **v08 supercon** (N=4,827 fully matched): MAE 0.04 eV, both predominantly metallic (SK 97%, ALIGNN 94%), metal/gap agreement 92.4%. Both methods agree this is a metallic-dominant dataset, consistent with the superconductor focus.
