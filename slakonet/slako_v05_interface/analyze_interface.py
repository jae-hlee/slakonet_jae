"""Analysis for SlakoNet on interface_db dataset.

Compares SlakoNet-predicted band gaps against DFT (optb88vdw) reference gaps,
plus CBM/VBM levels, formation/offset energies, and averaged DOS.
"""
import json
import re
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.metrics import mean_absolute_error, mean_squared_error

RESULTS_JSON = "results/all_results.json"
OUT_DIR = "analysis"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load (handle NaN) ──────────────────────────────────────────────────────
print("Loading JSON...", flush=True)
with open(RESULTS_JSON) as f:
    content = f.read()
content = re.sub(r"\bNaN\b", "null", content)
data = json.loads(content)
print(f"Loaded {len(data)} entries")

valid = [d for d in data
         if d.get("sk_bandgap") is not None
         and d.get("optb88vdw_bandgap") is not None]
print(f"Valid entries: {len(valid)} / {len(data)}")

sk_gap   = np.array([d["sk_bandgap"]         for d in valid])
dft_gap  = np.array([d["optb88vdw_bandgap"]  for d in valid])
cbm      = np.array([d["optb88vdw_cbm"]      for d in valid])
vbm      = np.array([d["optb88vdw_vbm"]      for d in valid])
e_final  = np.array([d["final_energy"]       for d in valid])
offset   = np.array([d["offset"] if isinstance(d["offset"], (int, float)) else np.nan
                     for d in valid])
jids     = [d["jid"] for d in valid]

# DFT gap can be slightly negative at interfaces (band overlap). Clip to 0 for
# "physical" gap used in parity statistics but keep raw too.
dft_gap_clipped = np.clip(dft_gap, 0, None)

# ── Stats helper ────────────────────────────────────────────────────────────
def stat(name, arr):
    print(f"  {name:34s} N={len(arr):5d} "
          f"mean={np.mean(arr):+.4f}  std={np.std(arr):.4f}  "
          f"min={np.min(arr):+.4f}  max={np.max(arr):+.4f}  "
          f"median={np.median(arr):+.4f}")

print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)
stat("SlakoNet band gap (eV)",          sk_gap)
stat("DFT optb88vdw gap (eV, raw)",     dft_gap)
stat("DFT optb88vdw gap (eV, clipped)", dft_gap_clipped)
stat("DFT CBM (eV)",                    cbm)
stat("DFT VBM (eV)",                    vbm)
offset_valid = offset[~np.isnan(offset)]
stat("Interface offset (eV)",           offset_valid)
stat("Final energy (eV)",               e_final)

neg_dft = (dft_gap < 0).sum()
zero_sk = (sk_gap < 1e-3).sum()
print(f"\n  DFT gaps < 0 (band overlap): {neg_dft} ({100*neg_dft/len(valid):.1f}%)")
print(f"  SlakoNet gaps ~0 (<1meV):    {zero_sk} ({100*zero_sk/len(valid):.1f}%)")

# ── Density scatter helper ─────────────────────────────────────────────────
def density_scatter(ax, x, y, bins=60):
    h, xe, ye = np.histogram2d(x, y, bins=bins)
    xi = np.clip(np.digitize(x, xe) - 1, 0, bins - 1)
    yi = np.clip(np.digitize(y, ye) - 1, 0, bins - 1)
    c  = h[xi, yi]
    order = np.argsort(c)
    return ax.scatter(x[order], y[order], c=c[order], s=10, cmap="viridis",
                      norm=LogNorm(vmin=1), rasterized=True)

# ── Parity: SlakoNet vs DFT ────────────────────────────────────────────────
print("\nGenerating parity plot (SlakoNet vs DFT)...")
residuals = sk_gap - dft_gap_clipped
mae  = mean_absolute_error(dft_gap_clipped, sk_gap)
rmse = np.sqrt(mean_squared_error(dft_gap_clipped, sk_gap))
r    = np.corrcoef(dft_gap_clipped, sk_gap)[0, 1]

fig, ax = plt.subplots(figsize=(7, 6))
sc = density_scatter(ax, dft_gap_clipped, sk_gap)
lim = max(dft_gap_clipped.max(), sk_gap.max()) * 1.05
ax.plot([0, lim], [0, lim], "r--", lw=1, label="y = x")
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.set_xlabel("DFT (optb88vdw) band gap (eV)")
ax.set_ylabel("SlakoNet band gap (eV)")
ax.set_title(f"SlakoNet vs DFT — interface_db (N={len(valid)})")
ax.set_aspect("equal"); ax.legend()
plt.colorbar(sc, ax=ax, label="Density")
ax.text(0.05, 0.95,
        f"MAE  = {mae:.3f} eV\nRMSE = {rmse:.3f} eV\nr    = {r:.3f}",
        transform=ax.transAxes, va="top", fontsize=10,
        bbox=dict(boxstyle="round", fc="white", alpha=0.85))
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/parity_sk_vs_dft.png", dpi=200)
plt.close(fig)
print(f"  MAE={mae:.4f}  RMSE={rmse:.4f}  r={r:.4f}")

# ── Distributions ──────────────────────────────────────────────────────────
print("Generating distributions...")
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
axes[0, 0].hist(sk_gap, bins=80, alpha=0.8)
axes[0, 0].set(title="SlakoNet band gap", xlabel="eV", ylabel="count")
axes[0, 1].hist(dft_gap, bins=80, alpha=0.8, color="tab:orange")
axes[0, 1].axvline(0, color="k", lw=0.5, ls=":")
axes[0, 1].set(title="DFT band gap (raw — may be negative)", xlabel="eV", ylabel="count")
axes[1, 0].hist(cbm, bins=80, alpha=0.8, color="tab:green", label="CBM")
axes[1, 0].hist(vbm, bins=80, alpha=0.8, color="tab:red",   label="VBM")
axes[1, 0].set(title="DFT CBM / VBM", xlabel="eV", ylabel="count"); axes[1, 0].legend()
axes[1, 1].hist(offset_valid, bins=80, alpha=0.8, color="tab:purple")
axes[1, 1].set(title="Interface offset", xlabel="eV", ylabel="count")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/distributions.png", dpi=200)
plt.close(fig)

# Overlay
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(dft_gap_clipped, bins=80, alpha=0.6, label="DFT (clipped ≥0)")
ax.hist(sk_gap,          bins=80, alpha=0.6, label="SlakoNet")
ax.set(xlabel="Band gap (eV)", ylabel="count",
       title="Gap distribution: SlakoNet vs DFT")
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/gap_overlay.png", dpi=200)
plt.close(fig)

# ── Residuals ──────────────────────────────────────────────────────────────
print("Generating residuals...")
fig, ax = plt.subplots(figsize=(7, 5))
ax.hist(residuals, bins=100, alpha=0.8)
ax.axvline(0, color="r", ls="--")
ax.set(xlabel="SlakoNet − DFT (eV)", ylabel="count",
       title=f"Residuals (mean={residuals.mean():+.3f}, std={residuals.std():.3f})")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/residuals.png", dpi=200)
plt.close(fig)

# Residual vs DFT gap (is error gap-dependent?)
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(dft_gap_clipped, residuals, s=8, alpha=0.5)
ax.axhline(0, color="r", ls="--", lw=1)
ax.set(xlabel="DFT gap (eV)", ylabel="SlakoNet − DFT (eV)",
       title="Residual vs DFT gap")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/residual_vs_gap.png", dpi=200)
plt.close(fig)

# ── DOS: small vs large gap average ────────────────────────────────────────
print("Computing average DOS...")
dos_e = np.array(valid[0]["dos_energies"])
# Guard against length mismatches
lens = {len(d["dos_values"]) for d in valid}
if len(lens) == 1 and len(dos_e) in lens:
    med = np.median(sk_gap)
    large_idx = np.where(sk_gap >= med)[0]
    small_idx = np.where(sk_gap <  med)[0]
    large_dos = np.mean([valid[i]["dos_values"] for i in large_idx], axis=0)
    small_dos = np.mean([valid[i]["dos_values"] for i in small_idx], axis=0)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dos_e, small_dos, label=f"SK gap < {med:.2f} eV (N={len(small_idx)})", alpha=0.85)
    ax.plot(dos_e, large_dos, label=f"SK gap ≥ {med:.2f} eV (N={len(large_idx)})", alpha=0.85)
    ax.set(xlabel="Energy (eV)", ylabel="DOS (arb.)",
           title="Average DOS: small-gap vs large-gap interfaces")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/dos_average.png", dpi=200)
    plt.close(fig)

    # A few example DOS across the gap range
    targets = [np.percentile(sk_gap, p) for p in (10, 50, 90)]
    labels  = ["10th pct (small)", "median", "90th pct (large)"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, tg, lab in zip(axes, targets, labels):
        i = int(np.argmin(np.abs(sk_gap - tg)))
        ax.plot(dos_e, valid[i]["dos_values"])
        ax.set(xlabel="Energy (eV)", ylabel="DOS",
               title=f"{lab}\nSK={sk_gap[i]:.2f}  DFT={dft_gap[i]:.2f} eV\n{jids[i][:55]}…")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/dos_examples.png", dpi=200)
    plt.close(fig)
else:
    print(f"  Skipping DOS plots — inconsistent lengths: {lens}")

# ── Top outliers ───────────────────────────────────────────────────────────
print("\nTop 15 |residual| outliers:")
order = np.argsort(np.abs(residuals))[::-1][:15]
for rank, i in enumerate(order, 1):
    print(f"  {rank:2d}. SK={sk_gap[i]:6.3f}  DFT={dft_gap[i]:+6.3f}  "
          f"Δ={residuals[i]:+6.3f}  {jids[i]}")

# ── CSV summary ────────────────────────────────────────────────────────────
print("\nWriting summary.csv...")
with open(f"{OUT_DIR}/summary.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["jid", "sk_bandgap_eV", "dft_bandgap_eV",
                "dft_cbm_eV", "dft_vbm_eV", "offset_eV", "final_energy_eV",
                "residual_eV"])
    for i, d in enumerate(valid):
        w.writerow([d["jid"],
                    f"{sk_gap[i]:.6f}", f"{dft_gap[i]:.6f}",
                    f"{cbm[i]:.6f}",    f"{vbm[i]:.6f}",
                    f"{offset[i]:.6f}", f"{e_final[i]:.6f}",
                    f"{residuals[i]:.6f}"])

print(f"\nDone. Plots and CSV in {OUT_DIR}/")
