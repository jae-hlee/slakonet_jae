import json
import re
import os
import csv
import zipfile
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "analysis"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load SlakoNet results ────────────────────────────────────────────────
print("Loading results/all_results.json ...", flush=True)
with open("results/all_results.json", "r") as f:
    content = f.read()
content = re.sub(r"\bNaN\b", "null", content)
results = json.loads(content)
print(f"  {len(results)} result entries")

# ── Load source dataset to find skipped/missing ──────────────────────────
print("Loading alex_supercon.json.zip ...", flush=True)
with zipfile.ZipFile("alex_supercon.json.zip") as z:
    with z.open("alex_supercon.json") as f:
        src = json.load(f)
src_by_id = {e["id"]: e for e in src}
print(f"  {len(src)} source entries")


def _finite(x):
    return x is not None and np.isfinite(x)


valid = [d for d in results if _finite(d["sk_bandgap"])]
print(f"  valid (finite sk_bandgap): {len(valid)}")

ids    = np.array([d["id"]         for d in valid])
sk_gap = np.array([d["sk_bandgap"] for d in valid], dtype=float)
Tc     = np.array([d["Tc"]         for d in valid], dtype=float)
dosef  = np.array([d["dosef"]      for d in valid], dtype=float)
debye  = np.array([d["debye"]      for d in valid], dtype=float)
la     = np.array([d["la"]         for d in valid], dtype=float)
wlog   = np.array([d["wlog"]       for d in valid], dtype=float)


def stats(name, arr):
    a = arr[np.isfinite(arr)]
    print(f"  {name:28s}  N={len(a):5d}  mean={np.mean(a):+.4f}  "
          f"std={np.std(a):.4f}  min={np.min(a):+.4f}  "
          f"max={np.max(a):+.4f}  median={np.median(a):+.4f}")


# ── 1. Coverage ──────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("COVERAGE")
print("=" * 70)
have = set(d["id"] for d in results)
missing = [e for e in src if e["id"] not in have]
print(f"  source entries:          {len(src)}")
print(f"  computed results:        {len(results)}")
print(f"  missing (skipped):       {len(missing)}")
print(f"  valid finite sk_bandgap: {len(valid)}")
zero_gap = int(np.sum(sk_gap < 1e-3))
small_gap = int(np.sum(sk_gap < 0.05))
print(f"  gap < 1 meV (metallic):  {zero_gap} ({100*zero_gap/len(valid):.1f}%)")
print(f"  gap < 50 meV:            {small_gap} ({100*small_gap/len(valid):.1f}%)")

if missing:
    skip_elems = Counter()
    for e in missing:
        for el in e.get("atoms", {}).get("elements", []):
            skip_elems[el] += 1
    print("  top elements in skipped structures (Z>65 filter):")
    print("   ", ", ".join(f"{k}:{v}" for k, v in skip_elems.most_common(15)))


# ── 2. Summary statistics ────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)
stats("SlakoNet band gap (eV)", sk_gap)
stats("Tc (K)",                 Tc)
stats("dos(Ef) (states/eV)",    dosef)
stats("Debye temp (K)",         debye)
stats("lambda (e-ph)",          la)
stats("wlog (K)",               wlog)


# ── 3. Correlations of sk_bandgap with superconductor descriptors ────────
print("\n" + "=" * 70)
print("CORRELATIONS  (Pearson r,  Spearman ρ)")
print("=" * 70)
try:
    from scipy.stats import spearmanr, pearsonr
    have_scipy = True
except Exception:
    have_scipy = False


def corr(x, y, label):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        print(f"  {label:32s}  N<3, skipped")
        return
    if have_scipy:
        r, _ = pearsonr(x[m], y[m])
        rho, _ = spearmanr(x[m], y[m])
        print(f"  {label:32s}  N={m.sum():5d}  r={r:+.3f}  ρ={rho:+.3f}")
    else:
        r = np.corrcoef(x[m], y[m])[0, 1]
        print(f"  {label:32s}  N={m.sum():5d}  r={r:+.3f}")


corr(sk_gap, Tc,    "sk_bandgap  vs  Tc")
corr(sk_gap, dosef, "sk_bandgap  vs  DFT dos(Ef)")
corr(sk_gap, la,    "sk_bandgap  vs  lambda")
corr(sk_gap, debye, "sk_bandgap  vs  Debye T")
corr(sk_gap, wlog,  "sk_bandgap  vs  wlog")
print()
corr(dosef, Tc, "DFT dos(Ef)  vs  Tc")
corr(la,    Tc, "lambda       vs  Tc")
corr(debye, Tc, "Debye T      vs  Tc")
corr(wlog,  Tc, "wlog         vs  Tc")


# ── 4. Distributions ─────────────────────────────────────────────────────
print("\nPlot: distributions.png", flush=True)
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes[0, 0].hist(sk_gap, bins=80, color="tab:blue", alpha=0.85)
axes[0, 0].set_xlabel("SlakoNet band gap (eV)")
axes[0, 0].set_title(f"SlakoNet gap (N={len(sk_gap)})")
axes[0, 0].set_yscale("log")

axes[0, 1].hist(Tc[np.isfinite(Tc)], bins=80, color="tab:red", alpha=0.85)
axes[0, 1].set_xlabel("Tc (K)")
axes[0, 1].set_title("Tc")

axes[0, 2].hist(dosef[np.isfinite(dosef)], bins=80, color="tab:green", alpha=0.85)
axes[0, 2].set_xlabel("DFT DOS at Ef (states/eV)")
axes[0, 2].set_title("dos(Ef)")

axes[1, 0].hist(la[np.isfinite(la)], bins=80, color="tab:purple", alpha=0.85)
axes[1, 0].set_xlabel("lambda (e-ph)")
axes[1, 0].set_title("lambda")

axes[1, 1].hist(debye[np.isfinite(debye)], bins=80, color="tab:brown", alpha=0.85)
axes[1, 1].set_xlabel("Debye T (K)")
axes[1, 1].set_title("Debye temperature")

axes[1, 2].hist(wlog[np.isfinite(wlog)], bins=80, color="tab:olive", alpha=0.85)
axes[1, 2].set_xlabel("wlog (K)")
axes[1, 2].set_title("wlog")

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "distributions.png"), dpi=180)
plt.close(fig)


# ── 5. sk_bandgap vs Tc & dosef (scatter) ────────────────────────────────
print("Plot: sk_vs_super_descriptors.png", flush=True)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

m = np.isfinite(sk_gap) & np.isfinite(Tc)
axes[0].scatter(sk_gap[m], Tc[m], s=8, alpha=0.4, color="tab:blue")
axes[0].set_xlabel("SlakoNet band gap (eV)")
axes[0].set_ylabel("Tc (K)")
axes[0].set_title(f"Tc vs SlakoNet gap (N={m.sum()})")

m = np.isfinite(sk_gap) & np.isfinite(dosef)
axes[1].scatter(sk_gap[m], dosef[m], s=8, alpha=0.4, color="tab:green")
axes[1].set_xlabel("SlakoNet band gap (eV)")
axes[1].set_ylabel("DFT dos(Ef) (states/eV)")
axes[1].set_title(f"dos(Ef) vs SlakoNet gap (N={m.sum()})")

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "sk_vs_super_descriptors.png"), dpi=180)
plt.close(fig)


# ── 6. Metallic vs non-metallic by SlakoNet gap (Tc compared) ────────────
print("Plot: tc_metallic_vs_gapped.png", flush=True)
metallic_thr = 0.05
metallic = sk_gap < metallic_thr
gapped   = sk_gap >= metallic_thr
m_tc = np.isfinite(Tc)

fig, ax = plt.subplots(figsize=(8, 5))
bins = np.linspace(0, max(np.nanmax(Tc), 1), 60)
ax.hist(Tc[metallic & m_tc], bins=bins, alpha=0.6,
        label=f"sk_gap<{metallic_thr} eV (N={int((metallic & m_tc).sum())})",
        color="tab:blue", density=True)
ax.hist(Tc[gapped & m_tc],   bins=bins, alpha=0.6,
        label=f"sk_gap≥{metallic_thr} eV (N={int((gapped & m_tc).sum())})",
        color="tab:orange", density=True)
ax.set_xlabel("Tc (K)")
ax.set_ylabel("density")
ax.set_title("Tc distribution: SlakoNet-metallic vs SlakoNet-gapped")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "tc_metallic_vs_gapped.png"), dpi=180)
plt.close(fig)


# ── 7. DOS analysis: average DOS, high-Tc vs low-Tc ──────────────────────
print("Plot: dos_average.png", flush=True)
dos_e_ref = np.array(valid[0]["dos_energies"])

def avg_dos(indices):
    if len(indices) == 0:
        return np.zeros_like(dos_e_ref)
    s = np.zeros_like(dos_e_ref, dtype=float)
    n = 0
    for i in indices:
        dv = np.asarray(valid[i]["dos_values"], dtype=float)
        if dv.shape == s.shape and np.all(np.isfinite(dv)):
            s += dv
            n += 1
    return s / max(n, 1)


tc_finite_idx = np.where(np.isfinite(Tc))[0]
if len(tc_finite_idx):
    median_tc = float(np.median(Tc[tc_finite_idx]))
    low_idx  = tc_finite_idx[Tc[tc_finite_idx] <  median_tc]
    high_idx = tc_finite_idx[Tc[tc_finite_idx] >= median_tc]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dos_e_ref, avg_dos(low_idx),
            label=f"Tc < {median_tc:.2f} K (N={len(low_idx)})", alpha=0.9)
    ax.plot(dos_e_ref, avg_dos(high_idx),
            label=f"Tc ≥ {median_tc:.2f} K (N={len(high_idx)})", alpha=0.9)
    ax.set_xlabel("Energy (eV, raw — not Fermi-aligned)")
    ax.set_ylabel("Average DOS (arb. units)")
    ax.set_title("Average SlakoNet DOS: low-Tc vs high-Tc")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "dos_average.png"), dpi=180)
    plt.close(fig)


# ── 8. Notable entries ───────────────────────────────────────────────────
def show_top(label, order_idx, n=10):
    print("\n" + "=" * 70)
    print(label)
    print("=" * 70)
    for rank, i in enumerate(order_idx[:n], 1):
        print(f"  {rank:2d}. {ids[i]:18s}  Tc={Tc[i]:7.2f}  "
              f"sk_gap={sk_gap[i]:7.4f}  dos(Ef)={dosef[i]:7.3f}  "
              f"λ={la[i]:5.2f}  Θ_D={debye[i]:7.1f}")


tc_order = np.argsort(np.where(np.isfinite(Tc), -Tc, np.inf))
show_top("TOP 10 HIGHEST Tc", tc_order)

gap_order = np.argsort(-sk_gap)
show_top("TOP 10 LARGEST SlakoNet GAP", gap_order)

la_order = np.argsort(np.where(np.isfinite(la), -la, np.inf))
show_top("TOP 10 LARGEST lambda (e-ph coupling)", la_order)


# ── 9. CSV summary ───────────────────────────────────────────────────────
print("\nWriting summary.csv ...", flush=True)
with open("summary.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "Tc_K", "dosef", "debye_K", "lambda", "wlog_K",
                "sk_bandgap_eV"])
    for d in results:
        def g(k):
            v = d.get(k)
            return f"{v:.6f}" if _finite(v) else ""
        w.writerow([d["id"], g("Tc"), g("dosef"), g("debye"),
                    g("la"), g("wlog"), g("sk_bandgap")])
print(f"  wrote {len(results)} rows to summary.csv")

print(f"\nDone. Plots in {OUT_DIR}/")
