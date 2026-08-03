#!/usr/bin/env python3
"""TODO item 1a: training-set overlap audit for every checkpoint.

Audits the 31,211 Alexandria benchmark structures against:
  - Materials Project / MEGNet corpus  (training set of mp_gappbe_alignn)      [jarvis data('megnet')]
  - JARVIS-DFT dft_3d                  (training set of jv_mbj / jv_optb88vdw) [jarvis data('dft_3d')]
  - optionally SlakoNet's fitting set  (--slakonet-fit-ids: file with one JARVIS jid per line)

Protocol: reduced-formula prefilter -> pymatgen StructureMatcher confirmation
(primitive cells, ltol=0.2, stol=0.3, angle_tol=5). A confirmed match is
classified 'exact' if it also passes a tight matcher (ltol=0.05, stol=0.1,
angle_tol=1), else 'near'. Stored/computed space-group agreement is recorded
per match for reporting.

Run:  python3 01_leakage_audit.py          (first run downloads ~1-2 GB of corpora)
Outputs in out/: overlap_mp.json, overlap_jarvis.json, leakfree_ids.json, audit_counts.md
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    MERGED_PREDICTIONS,
    ALEXANDRIA_ZIP,
    OUT_DIR,
    jarvis_to_pmg,
    load_benchmark_structures,
    load_merged,
    md_table,
    reduced_formula,
)

CHECKPOINT_CORPUS = {
    "PBE-ALIGNN (mp_gappbe)": "mp",
    "mBJ-ALIGNN (jv_mbj)": "jarvis",
    "OptB88vdW-ALIGNN (jv_optb88vdw)": "jarvis",
}


def entry_id(e):
    return e.get("jid") or e.get("id") or e.get("mat_id")


def entry_formula(e):
    f = e.get("formula")
    if f:
        return f
    atoms = e.get("atoms") or {}
    els = atoms.get("elements") or []
    from collections import Counter

    return "".join(f"{el}{n}" for el, n in sorted(Counter(els).items()))


def fetch_corpus(name):
    """Download (cached by jarvis-tools) and return the training corpus entries."""
    from jarvis.db.figshare import data as jdata

    os.makedirs(OUT_DIR, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(OUT_DIR)  # jarvis downloads into cwd
    try:
        t0 = time.time()
        entries = jdata("megnet" if name == "mp" else "dft_3d")
        print(f"corpus '{name}': {len(entries):,} entries ({time.time()-t0:.0f}s)")
        return entries
    finally:
        os.chdir(cwd)


def build_buckets(entries):
    buckets = {}
    for e in entries:
        rf = reduced_formula(entry_formula(e))
        if rf:
            buckets.setdefault(rf, []).append(e)
    return buckets


def audit(bench, buckets, near_m, exact_m, label):
    """Return list of match records for benchmark structures vs one corpus."""
    matches = []
    n_cand_pairs = 0
    t0 = time.time()
    items = list(bench.values())
    for i, b in enumerate(items):
        if i and i % 2000 == 0:
            print(
                f"  [{label}] {i}/{len(items)}  matches={len(matches)}  "
                f"pairs={n_cand_pairs}  ({time.time()-t0:.0f}s)"
            )
        rf = reduced_formula(b["formula"])
        cands = buckets.get(rf, [])
        if not cands:
            continue
        try:
            sb = jarvis_to_pmg(b["atoms"])
        except Exception:
            continue
        for c in cands:
            n_cand_pairs += 1
            try:
                sc = jarvis_to_pmg(c["atoms"])
            except Exception:
                continue
            try:
                if near_m.fit(sb, sc):
                    kind = "exact" if exact_m.fit(sb, sc) else "near"
                    spg_c = c.get("spg") or c.get("spg_number")
                    matches.append(
                        {
                            "mat_id": b["mat_id"],
                            "train_id": entry_id(c),
                            "kind": kind,
                            "spg_bench": b.get("spg"),
                            "spg_train": spg_c,
                        }
                    )
                    break
            except Exception:
                continue
    print(f"  [{label}] done: {len(matches)} matched of {len(items)} ({time.time()-t0:.0f}s)")
    return matches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", default=MERGED_PREDICTIONS)
    ap.add_argument("--alexandria-zip", default=ALEXANDRIA_ZIP)
    ap.add_argument("--slakonet-fit-ids", default=None,
                    help="optional file of JARVIS jids used to fit SlakoNet (one per line)")
    args = ap.parse_args()

    from pymatgen.analysis.structure_matcher import StructureMatcher

    near_m = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5, primitive_cell=True)
    exact_m = StructureMatcher(ltol=0.05, stol=0.1, angle_tol=1, primitive_cell=True)

    rows = load_merged(args.merged)
    mat_ids = [r["mat_id"] for r in rows]
    print(f"benchmark: {len(mat_ids)} structures")
    bench = load_benchmark_structures(mat_ids, args.alexandria_zip)
    print(f"loaded structures for {len(bench)} / {len(mat_ids)}")

    os.makedirs(OUT_DIR, exist_ok=True)
    results = {}
    for corpus in ("mp", "jarvis"):
        out_json = os.path.join(OUT_DIR, f"overlap_{corpus}.json")
        if os.path.exists(out_json):
            with open(out_json) as f:
                results[corpus] = json.load(f)
            print(f"reusing existing {out_json} ({len(results[corpus])} matches)")
            continue
        entries = fetch_corpus(corpus)
        buckets = build_buckets(entries)
        print(f"corpus '{corpus}': {len(buckets):,} formula buckets")
        results[corpus] = audit(bench, buckets, near_m, exact_m, corpus)
        with open(out_json, "w") as f:
            json.dump(results[corpus], f, indent=1)
        print(f"wrote {out_json}")

    # ---- summarize ----
    all_ids = set(mat_ids)
    leakfree = {}
    table = []
    for ckpt, corpus in CHECKPOINT_CORPUS.items():
        m = results[corpus]
        leaked = {x["mat_id"] for x in m}
        exact = sum(1 for x in m if x["kind"] == "exact")
        near = sum(1 for x in m if x["kind"] == "near")
        lf = sorted(all_ids - leaked)
        leakfree[ckpt] = lf
        table.append(
            (
                ckpt,
                "Materials Project" if corpus == "mp" else "JARVIS-DFT",
                exact,
                near,
                f"{100*len(leaked)/len(all_ids):.1f}%",
                len(lf),
            )
        )

    if args.slakonet_fit_ids:
        with open(args.slakonet_fit_ids) as f:
            fit_ids = {l.strip() for l in f if l.strip()}
        m = [x for x in results["jarvis"] if x["train_id"] in fit_ids]
        leaked = {x["mat_id"] for x in m}
        lf = sorted(all_ids - leaked)
        leakfree["SlakoNet (SK fit set)"] = lf
        table.append(
            (
                "SlakoNet (SK fit set)",
                f"JARVIS-DFT subset (n={len(fit_ids)})",
                sum(1 for x in m if x["kind"] == "exact"),
                sum(1 for x in m if x["kind"] == "near"),
                f"{100*len(leaked)/len(all_ids):.1f}%",
                len(lf),
            )
        )
    else:
        print(
            "NOTE: no --slakonet-fit-ids given; the JARVIS-DFT row is the upper bound "
            "for SlakoNet's fitting-set overlap (SK fit set is a subset of JARVIS-DFT)."
        )

    with open(os.path.join(OUT_DIR, "leakfree_ids.json"), "w") as f:
        json.dump(leakfree, f)
    md = "# Overlap audit (generated by 01_leakage_audit.py)\n\n" + md_table(
        ["Checkpoint", "Training corpus", "Exact matches", "Near-duplicates",
         "% of eval set", "Leak-free N"],
        table,
    )
    with open(os.path.join(OUT_DIR, "audit_counts.md"), "w") as f:
        f.write(md + "\n")
    print("\n" + md)
    print(f"\nwrote out/leakfree_ids.json and out/audit_counts.md")


if __name__ == "__main__":
    main()
