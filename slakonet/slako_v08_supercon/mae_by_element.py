"""Per-element breakdown of SlakoNet dos(Ef) error.

For every element X, aggregate the residual over all structures that contain X.
A structure with k elements contributes its residual to k bins.
"""
import json
import re
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "analysis"
os.makedirs(OUT_DIR, exist_ok=True)

print("Loading results/all_results.json ...", flush=True)
with open("results/all_results.json") as f:
    results = json.loads(re.sub(r"\bNaN\b", "null", f.read()))

per_elem_resid = defaultdict(list)
per_elem_dft   = defaultdict(list)

for d in results:
    e = np.asarray(d["dos_energies"], dtype=float)
    v = np.asarray(d["dos_values"],   dtype=float)
    if e.size == 0 or not np.all(np.isfinite(v)):
        continue
    sk_dosef  = float(np.interp(0.0, e, v))
    dft_dosef = d["dosef"]
    if not (np.isfinite(sk_dosef) and np.isfinite(dft_dosef)):
        continue
    resid = sk_dosef - dft_dosef
    for el in set(d["atoms"]["elements"]):       # one vote per element per structure
        per_elem_resid[el].append(resid)
        per_elem_dft[el].append(dft_dosef)

MIN_N = 20
rows = []
for el, rs in per_elem_resid.items():
    if len(rs) < MIN_N:
        continue
    rs = np.asarray(rs)
    rows.append((el, len(rs), rs.mean(), np.abs(rs).mean(),
                 np.median(rs), np.mean(per_elem_dft[el])))

rows.sort(key=lambda r: r[2])  # sort by mean bias, most-underbinding first
print(f"\nElements with N >= {MIN_N}:  {len(rows)}")
print(f"{'el':>3}  {'N':>5}  {'bias':>7}  {'MAE':>6}  {'med':>7}  {'DFT⟨dosef⟩':>10}")
for el, n, b, mae, med, dft_avg in rows:
    print(f"{el:>3}  {n:5d}  {b:+7.3f}  {mae:6.3f}  {med:+7.3f}  {dft_avg:10.3f}")

# ── Plot: bias + MAE per element ─────────────────────────────────────────
els    = [r[0] for r in rows]
ns     = np.array([r[1] for r in rows])
biases = np.array([r[2] for r in rows])
maes   = np.array([r[3] for r in rows])

fig, axes = plt.subplots(2, 1, figsize=(max(12, 0.22 * len(els)), 8), sharex=True)

ax = axes[0]
colors = ["tab:red" if b < 0 else "tab:blue" for b in biases]
ax.bar(els, biases, color=colors, edgecolor="k", linewidth=0.4)
ax.axhline(0, color="k", lw=0.6)
ax.set_ylabel("mean residual  (SK − DFT)  [states/eV]")
ax.set_title(f"Per-element bias of SlakoNet dos(Ef)  "
             f"(elements with N ≥ {MIN_N}; sorted most-underbinding → least)")
for i, (b, n) in enumerate(zip(biases, ns)):
    ax.text(i, b, f"{n}", ha="center",
            va="top" if b < 0 else "bottom", fontsize=7)

ax = axes[1]
ax.bar(els, maes, color="tab:orange", edgecolor="k", linewidth=0.4)
ax.set_ylabel("MAE in element subset  [states/eV]")
ax.set_xlabel("element  (one vote per structure containing the element)")
plt.setp(ax.get_xticklabels(), rotation=70, fontsize=8)

fig.tight_layout()
out = os.path.join(OUT_DIR, "mae_by_element.png")
fig.savefig(out, dpi=180)
plt.close(fig)
print(f"\nWrote {out}")
