"""
Concatenate every results/<id>.json into a single results/all_results.json.

Streams the output (peak memory ~= one entry), so it's safe to run on a
memory-constrained login node.

Run on the login node (no GPU needed):
    python aggregate_results.py

Safe to re-run: overwrites results/all_results.json.
"""
import json
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
OUT = os.path.join(RESULTS_DIR, "all_results.json")
TMP = OUT + ".tmp"

files = [
    f for f in os.listdir(RESULTS_DIR)
    if f.endswith(".json") and f != "all_results.json"
]
files.sort()
print(f"Found {len(files)} per-id JSON files in {RESULTS_DIR}", flush=True)

t0 = time.time()
with open(TMP, "w") as out:
    out.write("[")
    for i, fname in enumerate(files, 1):
        with open(os.path.join(RESULTS_DIR, fname)) as f:
            entry = json.load(f)
        if i > 1:
            out.write(",")
        json.dump(entry, out)
        if i % 1000 == 0:
            dt = time.time() - t0
            print(f"  {i}/{len(files)}  ({dt:.1f}s, {i/dt:.0f}/s)", flush=True)
    out.write("]")

os.replace(TMP, OUT)
size_gb = os.path.getsize(OUT) / 1e9
print(f"Done. {OUT} is {size_gb:.2f} GB ({len(files)} entries)", flush=True)
