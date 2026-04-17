"""SlakoNet dos(Ef) vs DFT dos(Ef) — MAE / parity / residual analysis.

DOS grid is Fermi-aligned [-10, 10] eV, 5000 pts (verified empirically).
SlakoNet dos(Ef) is the linear interpolation of dos_values at E = 0.
"""
import json
import re
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "analysis"
os.makedirs(OUT_DIR, exist_ok=True)

print("Loading results/all_results.json ...", flush=True)
with open("results/all_results.json") as f:
    txt = re.sub(r"\bNaN\b", "null", f.read())
results = json.loads(txt)
print(f"  {len(results)} entries")

sk_dosef, dft_dosef, sk_gap, ids = [], [], [], []
for d in results:
    e = np.asarray(d["dos_energies"], dtype=float)
    v = np.asarray(d["dos_values"],   dtype=float)
    if e.size == 0 or not np.all(np.isfinite(v)):
        continue
    sk = float(np.interp(0.0, e, v))
    sk_dosef.append(sk)
    dft_dosef.append(d["dosef"])
    sk_gap.append(d["sk_bandgap"])
    ids.append(d["id"])

sk_dosef  = np.array(sk_dosef)
dft_dosef = np.array(dft_dosef)
sk_gap    = np.array(sk_gap)
ids       = np.array(ids)

m = np.isfinite(sk_dosef) & np.isfinite(dft_dosef)
sk_dosef, dft_dosef, sk_gap, ids = sk_dosef[m], dft_dosef[m], sk_gap[m], ids[m]

resid = sk_dosef - dft_dosef
abs_err = np.abs(resid)

mae    = abs_err.mean()
rmse   = np.sqrt(np.mean(resid**2))
medae  = np.median(abs_err)
bias   = resid.mean()
r      = np.corrcoef(sk_dosef, dft_dosef)[0, 1]
slope  = np.sum(sk_dosef * dft_dosef) / np.sum(dft_dosef**2)

print("\nSlakoNet dos(Ef)  vs  DFT dos(Ef)")
print(f"  N        = {len(sk_dosef)}")
print(f"  MAE      = {mae:.3f} states/eV")
print(f"  RMSE     = {rmse:.3f} states/eV")
print(f"  median|e|= {medae:.3f}")
print(f"  bias     = {bias:+.3f}  (sk - dft)")
print(f"  Pearson r= {r:+.3f}")
print(f"  best-fit slope (through origin) = {slope:.3f}")

# ── Plot: parity + residual histogram + MAE-binned-by-DFT ────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# (a) parity
ax = axes[0]
lim = max(np.percentile(dft_dosef, 99), np.percentile(sk_dosef, 99)) * 1.05
ax.hexbin(dft_dosef, sk_dosef, gridsize=60, mincnt=1,
          extent=(0, lim, 0, lim), cmap="viridis", bins="log")
ax.plot([0, lim], [0, lim], "r--", lw=1, label="y = x")
ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect("equal")
ax.set_xlabel("DFT dos(Ef)  (states/eV)")
ax.set_ylabel("SlakoNet dos(Ef)  (states/eV)")
ax.set_title(f"Parity  (N={len(sk_dosef)},  r={r:+.3f})")
ax.legend(loc="upper left")
ax.text(0.97, 0.03,
        f"MAE = {mae:.2f}\nRMSE = {rmse:.2f}\nbias = {bias:+.2f}\nslope = {slope:.2f}",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=10,
        bbox=dict(boxstyle="round", fc="white", alpha=0.85))

# (b) signed-residual histogram
ax = axes[1]
hi = np.percentile(np.abs(resid), 99)
ax.hist(resid, bins=80, range=(-hi, hi), color="tab:blue", alpha=0.85)
ax.axvline(0, color="k", lw=0.8, ls="--")
ax.axvline(bias, color="r", lw=1.2, label=f"bias = {bias:+.2f}")
ax.set_xlabel("residual = SlakoNet − DFT  (states/eV)")
ax.set_ylabel("count")
ax.set_title("Residual distribution")
ax.legend()

# (c) MAE binned by DFT dos(Ef)
ax = axes[2]
edges = np.linspace(0, np.percentile(dft_dosef, 98), 13)
centers = 0.5 * (edges[:-1] + edges[1:])
mae_bin, n_bin = [], []
for lo, hi in zip(edges[:-1], edges[1:]):
    sel = (dft_dosef >= lo) & (dft_dosef < hi)
    if sel.sum() >= 5:
        mae_bin.append(abs_err[sel].mean()); n_bin.append(sel.sum())
    else:
        mae_bin.append(np.nan); n_bin.append(sel.sum())
mae_bin = np.array(mae_bin); n_bin = np.array(n_bin)
ax.bar(centers, mae_bin, width=(edges[1] - edges[0]) * 0.9,
       color="tab:orange", alpha=0.85, edgecolor="k")
for c, m_, n in zip(centers, mae_bin, n_bin):
    if np.isfinite(m_):
        ax.text(c, m_, f"{n}", ha="center", va="bottom", fontsize=8)
ax.axhline(mae, color="k", lw=1, ls="--", label=f"overall MAE = {mae:.2f}")
ax.set_xlabel("DFT dos(Ef)  (states/eV)")
ax.set_ylabel("MAE in bin  (states/eV)")
ax.set_title("MAE vs DFT dos(Ef)  (counts on bars)")
ax.legend()

fig.tight_layout()
out = os.path.join(OUT_DIR, "mae_dosef.png")
fig.savefig(out, dpi=180)
plt.close(fig)
print(f"\nWrote {out}")

# ── Worst outliers ───────────────────────────────────────────────────────
print("\nTop 10 worst |residual| (likely worth inspecting):")
order = np.argsort(-abs_err)[:10]
for rank, i in enumerate(order, 1):
    print(f"  {rank:2d}. {ids[i]:18s}  DFT={dft_dosef[i]:7.3f}  "
          f"SK={sk_dosef[i]:7.3f}  Δ={resid[i]:+7.3f}  sk_gap={sk_gap[i]:.4f}")
