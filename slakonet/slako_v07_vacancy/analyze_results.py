import json
import re
import os
import csv
import zipfile
from collections import Counter, defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

OUT_DIR = "analysis"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load SlakoNet results ────────────────────────────────────────────────
print("Loading results/all_results.json ...", flush=True)
with open("results/all_results.json", "r") as f:
    content = f.read()
content = re.sub(r"\bNaN\b", "null", content)
results = json.loads(content)
print(f"  {len(results)} result entries")

# ── Load vacancy DB for metadata (vacancy_class, material_class) ─────────
print("Loading vacancydb.json.zip ...", flush=True)
with zipfile.ZipFile("vacancydb.json.zip") as z:
    with z.open(z.namelist()[0]) as f:
        vdb = json.load(f)
vdb_by_id = {e["id"]: e for e in vdb}
print(f"  {len(vdb)} vacancy DB entries")

# ── Join metadata onto results ───────────────────────────────────────────
for d in results:
    meta = vdb_by_id.get(d["id"], {})
    vc = meta.get("vacancy_class", {}) or {}
    d["vacancy_symbol"] = vc.get("symbol", "?") if isinstance(vc, dict) else "?"
    d["wyckoff"] = vc.get("wyckoff_multiplicity", "?") if isinstance(vc, dict) else "?"
    d["material_class"] = meta.get("material_class", "?")

# Valid = finite sk_bandgap and finite ef
def _finite(x):
    return x is not None and np.isfinite(x)

valid = [d for d in results if _finite(d["sk_bandgap"]) and _finite(d["ef"])]
print(f"  valid (finite sk_bandgap & ef): {len(valid)}")

ids       = [d["id"] for d in valid]
jids      = [d["jid"] for d in valid]
sk_gap    = np.array([d["sk_bandgap"] for d in valid])
ef        = np.array([d["ef"] for d in valid])
vsym      = np.array([d["vacancy_symbol"] for d in valid])
mclass    = np.array([d["material_class"] for d in valid])


# ── Helpers ──────────────────────────────────────────────────────────────
def stats(name, arr):
    print(f"  {name:28s}  N={len(arr):5d}  mean={np.mean(arr):+.4f}  "
          f"std={np.std(arr):.4f}  min={np.min(arr):+.4f}  "
          f"max={np.max(arr):+.4f}  median={np.median(arr):+.4f}")


# ── 1. Coverage ──────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("COVERAGE")
print("=" * 70)
have_ids = set(d["id"] for d in results)
missing = [e for e in vdb if e["id"] not in have_ids]
print(f"  vacancy DB entries:      {len(vdb)}")
print(f"  computed results:        {len(results)}")
print(f"  missing (skipped):       {len(missing)}")
print(f"  valid finite sk_bandgap: {len(valid)}")
zero_gap = np.sum(sk_gap < 1e-3)
print(f"  gap < 1 meV (metallic):  {zero_gap} ({100*zero_gap/len(valid):.1f}%)")

# Which elements cause skips? SlakoNet supports Z<=65 (Tb).
if missing:
    skip_elems = Counter()
    for e in missing:
        da = e.get("defective_atoms", {}) or {}
        for el in da.get("elements", []):
            skip_elems[el] += 1
    print("  top elements in skipped structures:",
          ", ".join(f"{k}:{v}" for k, v in skip_elems.most_common(10)))


# ── 2. Summary statistics ────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)
stats("SlakoNet band gap (eV)", sk_gap)
stats("Formation energy ef (eV)", ef)

print("\n  by material_class:")
for mc in sorted(set(mclass)):
    m = mclass == mc
    print(f"    {mc:6s}  N={m.sum():4d}  "
          f"gap mean={sk_gap[m].mean():.3f}  ef mean={ef[m].mean():+.3f}")

print("\n  top 15 vacancy elements (by count):")
vc_counts = Counter(vsym)
for sym, n in vc_counts.most_common(15):
    m = vsym == sym
    print(f"    {sym:3s}  N={n:4d}  gap mean={sk_gap[m].mean():.3f}  "
          f"ef mean={ef[m].mean():+.3f}")


# ── 3. Distributions ─────────────────────────────────────────────────────
print("\nPlot: distributions.png", flush=True)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].hist(sk_gap, bins=80, color="tab:blue", alpha=0.85, edgecolor="none")
axes[0].set_xlabel("SlakoNet band gap (eV)")
axes[0].set_ylabel("Count")
axes[0].set_title(f"Band gap (N={len(sk_gap)})")

axes[1].hist(ef, bins=80, color="tab:orange", alpha=0.85, edgecolor="none")
axes[1].axvline(0, color="k", lw=0.8, ls="--")
axes[1].set_xlabel("Vacancy formation energy $E_f$ (eV)")
axes[1].set_ylabel("Count")
axes[1].set_title(f"Formation energy (N={len(ef)})")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "distributions.png"), dpi=200)
plt.close(fig)


# ── 4. Gap vs ef scatter (density, colored by material_class) ────────────
print("Plot: gap_vs_ef.png", flush=True)
fig, ax = plt.subplots(figsize=(8, 6))
colors = {"3D": "tab:blue", "2D": "tab:green", "1D": "tab:red", "0D": "tab:purple"}
for mc in sorted(set(mclass)):
    m = mclass == mc
    ax.scatter(ef[m], sk_gap[m], s=14, alpha=0.7,
               label=f"{mc} (N={m.sum()})",
               color=colors.get(mc, "gray"))
# Pearson correlation
if len(sk_gap) > 2:
    r = np.corrcoef(ef, sk_gap)[0, 1]
    ax.text(0.03, 0.97, f"Pearson r = {r:+.3f}", transform=ax.transAxes,
            va="top", fontsize=11,
            bbox=dict(boxstyle="round", fc="white", alpha=0.8))
ax.set_xlabel("Vacancy formation energy $E_f$ (eV)")
ax.set_ylabel("SlakoNet band gap (eV)")
ax.set_title("Band gap vs formation energy")
ax.legend(loc="upper right", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "gap_vs_ef.png"), dpi=200)
plt.close(fig)


# ── 5. Band gap by vacancy element (boxplot, top-20 by count) ────────────
print("Plot: gap_by_vacancy_element.png", flush=True)
top_syms = [s for s, _ in vc_counts.most_common(20)]
data_by_sym = [sk_gap[vsym == s] for s in top_syms]
fig, ax = plt.subplots(figsize=(12, 5))
ax.boxplot(data_by_sym, labels=top_syms, showfliers=False)
for i, d in enumerate(data_by_sym, 1):
    ax.scatter(np.random.normal(i, 0.06, size=len(d)), d,
               s=6, alpha=0.5, color="tab:blue")
ax.set_ylabel("SlakoNet band gap (eV)")
ax.set_xlabel("Vacancy element (top 20 by count)")
ax.set_title("Band gap distribution by vacancy element")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "gap_by_vacancy_element.png"), dpi=200)
plt.close(fig)


# ── 6. DOS analysis ──────────────────────────────────────────────────────
print("Plot: dos_average.png", flush=True)
dos_energies = np.array(valid[0]["dos_energies"])
median_gap = np.median(sk_gap)
small_idx = np.where(sk_gap < median_gap)[0]
large_idx = np.where(sk_gap >= median_gap)[0]

def avg_dos(indices):
    if len(indices) == 0:
        return np.zeros_like(dos_energies)
    s = np.zeros_like(dos_energies, dtype=float)
    n = 0
    for i in indices:
        dv = np.array(valid[i]["dos_values"])
        if dv.shape == s.shape and np.all(np.isfinite(dv)):
            s += dv; n += 1
    return s / max(n, 1)

small_dos = avg_dos(small_idx)
large_dos = avg_dos(large_idx)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(dos_energies, small_dos,
        label=f"gap < {median_gap:.2f} eV (N={len(small_idx)})", alpha=0.9)
ax.plot(dos_energies, large_dos,
        label=f"gap ≥ {median_gap:.2f} eV (N={len(large_idx)})", alpha=0.9)
ax.set_xlabel("Energy (eV)")
ax.set_ylabel("Average DOS (arb. units)")
ax.set_title("Average DOS: small-gap vs large-gap defective structures")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "dos_average.png"), dpi=200)
plt.close(fig)

# Example DOS at a few target gap values
print("Plot: dos_examples.png", flush=True)
targets = [0.0, 1.0, 3.0, 6.0]
fig, axes = plt.subplots(1, len(targets), figsize=(4.5 * len(targets), 4))
for ax, tgt in zip(axes, targets):
    i = int(np.argmin(np.abs(sk_gap - tgt)))
    e = np.array(valid[i]["dos_energies"])
    d = np.array(valid[i]["dos_values"])
    ax.plot(e, d)
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("DOS")
    ax.set_title(f"{ids[i]}\nSK gap={sk_gap[i]:.2f} eV (target {tgt})")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "dos_examples.png"), dpi=200)
plt.close(fig)


# ── 7. Notable entries ───────────────────────────────────────────────────
print("\n" + "=" * 70)
print("TOP 10 LARGEST BAND GAPS")
print("=" * 70)
for rank, i in enumerate(np.argsort(sk_gap)[::-1][:10], 1):
    print(f"  {rank:2d}. {ids[i]:30s}  mc={mclass[i]:3s}  vac={vsym[i]:3s}  "
          f"gap={sk_gap[i]:.3f} eV  ef={ef[i]:+.3f} eV")

print("\nTOP 10 LOWEST (non-zero) BAND GAPS")
print("=" * 70)
nz = np.where(sk_gap > 1e-3)[0]
for rank, j in enumerate(np.argsort(sk_gap[nz])[:10], 1):
    i = nz[j]
    print(f"  {rank:2d}. {ids[i]:30s}  mc={mclass[i]:3s}  vac={vsym[i]:3s}  "
          f"gap={sk_gap[i]:.5f} eV  ef={ef[i]:+.3f} eV")

print("\nTOP 10 LARGEST FORMATION ENERGIES")
print("=" * 70)
for rank, i in enumerate(np.argsort(ef)[::-1][:10], 1):
    print(f"  {rank:2d}. {ids[i]:30s}  mc={mclass[i]:3s}  vac={vsym[i]:3s}  "
          f"ef={ef[i]:+.3f} eV  gap={sk_gap[i]:.3f} eV")


# ── 8. CSV summary ───────────────────────────────────────────────────────
print("\nWriting summary.csv ...", flush=True)
with open("summary.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "jid", "material_class", "vacancy_symbol",
                "wyckoff", "ef_eV", "sk_bandgap_eV"])
    for d in results:
        w.writerow([d["id"], d["jid"], d["material_class"],
                    d["vacancy_symbol"], d["wyckoff"],
                    f"{d['ef']:.6f}" if _finite(d["ef"]) else "",
                    f"{d['sk_bandgap']:.6f}" if _finite(d["sk_bandgap"]) else ""])
print(f"  wrote {len(results)} rows to summary.csv")

print(f"\nDone. Plots in {OUT_DIR}/")
