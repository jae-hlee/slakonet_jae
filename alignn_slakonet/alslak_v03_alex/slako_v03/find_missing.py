"""Identify structures that were attempted but missing from all_results.json.

Reproduces the filter logic from jslako.py, then diffs the attempted mat_ids
against the completed ones.
"""
import json
import zipfile
import os

ALLOWED_SYMBOLS = {
    'H','He','Li','Be','B','C','N','O','F','Ne',
    'Na','Mg','Al','Si','P','S','Cl','Ar',
    'K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn',
    'Ga','Ge','As','Se','Br','Kr',
    'Rb','Sr','Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd',
    'In','Sn','Sb','Te','I','Xe',
    'Cs','Ba','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb',
}


def all_elements_supported(elements):
    return all(e in ALLOWED_SYMBOLS for e in elements)


# ── Load the full Alexandria dataset and reproduce the filter ─────────────
local_zip = 'alexandria_pbe_3d_2024.10.1_jarvis_tools.json.zip'
print(f"Loading {local_zip}...", flush=True)
with zipfile.ZipFile(local_zip) as zf:
    json_name = zf.namelist()[0]
    dft_3d = json.loads(zf.read(json_name))
print(f"Loaded {len(dft_3d)} total entries", flush=True)

attempted = {}  # mat_id -> full entry
for i in dft_3d:
    if i['e_above_hull'] != 0:
        continue
    if not all_elements_supported(i['elements']):
        continue
    attempted[i['mat_id']] = i
# Free the big list so we have room for loading results
del dft_3d
print(f"Attempted (e_above_hull=0, Z<=65): {len(attempted)}", flush=True)

# ── Load completed results ───────────────────────────────────────────────
print("Loading results/all_results.json (this may take a minute)...", flush=True)
with open("results/all_results.json") as f:
    results = json.load(f)
completed = {r['mat_id'] for r in results}
print(f"Completed: {len(completed)}", flush=True)

# ── Diff ─────────────────────────────────────────────────────────────────
missing_ids = set(attempted.keys()) - completed
missing_entries = [attempted[mid] for mid in missing_ids]
print(f"\nMissing: {len(missing_entries)}")

# ── Analyze missing structures ───────────────────────────────────────────
# Element count distribution
from collections import Counter
n_elem_missing = Counter(len(set(e['elements'])) for e in missing_entries)
n_elem_complete = Counter(len(set(attempted[mid]['elements'])) for mid in completed)

print("\nUnique element count distribution:")
print(f"  {'n_elem':>8}  {'missing':>10}  {'complete':>10}  {'frac_miss':>10}")
for n in sorted(set(n_elem_missing) | set(n_elem_complete)):
    m = n_elem_missing.get(n, 0)
    c = n_elem_complete.get(n, 0)
    total = m + c
    frac = m / total if total > 0 else 0.0
    print(f"  {n:>8d}  {m:>10d}  {c:>10d}  {frac:>10.2%}")

# Structure size distribution
natoms_missing = [len(e['atoms']['elements']) for e in missing_entries]
natoms_complete = [len(attempted[mid]['atoms']['elements']) for mid in completed]
if natoms_missing:
    import statistics
    print(f"\nAtoms per cell (missing):  mean={statistics.mean(natoms_missing):.1f}, "
          f"median={statistics.median(natoms_missing)}, max={max(natoms_missing)}")
if natoms_complete:
    print(f"Atoms per cell (complete): mean={statistics.mean(natoms_complete):.1f}, "
          f"median={statistics.median(natoms_complete)}, max={max(natoms_complete)}")

# Most common elements in missing vs complete
elem_missing = Counter(e for entry in missing_entries for e in entry['elements'])
elem_complete = Counter(e for mid in completed for e in attempted[mid]['elements'])
print("\nTop 15 elements in missing structures (and their miss rate):")
print(f"  {'elem':>6}  {'missing':>10}  {'complete':>10}  {'miss_rate':>10}")
for elem, m in elem_missing.most_common(15):
    c = elem_complete.get(elem, 0)
    total = m + c
    rate = m / total if total > 0 else 0.0
    print(f"  {elem:>6s}  {m:>10d}  {c:>10d}  {rate:>10.2%}")

# ── Save missing list ────────────────────────────────────────────────────
out_file = "missing_mat_ids.json"
with open(out_file, "w") as f:
    json.dump(sorted(missing_ids), f)
print(f"\nSaved missing mat_ids to {out_file}")

# Also save full entries for easy re-running
out_full = "missing_entries.json"
with open(out_full, "w") as f:
    json.dump(missing_entries, f)
print(f"Saved full missing entries to {out_full} ({os.path.getsize(out_full)/1e6:.1f} MB)")
