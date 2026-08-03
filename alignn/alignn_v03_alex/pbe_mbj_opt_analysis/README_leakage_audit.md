# NeurIPS 2026 leakage audit and funnel baselines

These files archive the analyses produced for the SlakoNet DB NeurIPS 2026 submission during the author response period (July to August 2026). The central item is a training-set overlap audit for the ALIGNN checkpoints evaluated on the 31,211-structure paired benchmark (this directory's `merged_predictions.json`), with leakage-controlled re-scoring and baseline comparisons for the paper's screening funnel.

## Leakage audit (`01_leakage_audit.py`)

Fingerprints every benchmark structure (reduced formula plus spglib spacegroup) and confirms candidate matches against the ALIGNN training corpora with pymatgen StructureMatcher. Corpora: Materials Project via the MEGNet set for the PBE checkpoint, JARVIS dft_3d for the mBJ and OptB88vdW checkpoints, both fetched through jarvis-tools.

- `audit_counts.md` is the overlap summary table.
- `overlap_mp.json` and `overlap_jarvis.json` are the matched-ID lists (benchmark ID to training ID, exact versus near-duplicate flag).
- `leakfree_ids.json` lists the leak-free benchmark subset per checkpoint.
- `audit_run.txt` is the run log for provenance.

## Leakage-controlled re-scoring (`02_rescore_leakfree.py`)

Re-computes the headline metrics on the de-duplicated subsets, gated on first reproducing the published full-set numbers. Results in `rescore_tables.md`. Key outcome: PBE-ALIGNN non-metal MAE moves from 0.274 eV (full set) to 0.386 eV (leak-free) and the model ranking is unchanged, with the PBE-ALIGNN advantage persisting at roughly 1.4 eV over the tight-binding baseline.

## Funnel baselines (`03_funnel_baseline.py`)

Reproduces the paper's six-stage screening funnel (with a built-in self-check against the published stage attrition) and compares it against ALIGNN-only and SlakoNet-only screening on the same pool. Results in `funnel_tables.md`. The funnel wins on precision in the 0.5 to 1.5 eV window and ALIGNN-only is preferable for wide-gap and coverage-oriented searches.

## Running

Shared helpers are in `common.py`. Scripts resolve `merged_predictions.json` from this directory and the Alexandria snapshot from the repository, so they run from here without arguments. Run order: `01` then `02` (the audit output feeds the re-scoring), `03` is independent and fast. The audit downloads the training corpora via jarvis-tools on first run, which is the dominant cost.
