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

# ── Load ────────────────────────────────────────────────────────────────
print("Loading JSON...", flush=True)
with open("results/all_results.json", "r") as f:
    content = f.read()
content = re.sub(r'\bNaN\b', 'null', content)
data = json.loads(content)
print(f"Loaded {len(data)} entries", flush=True)

# DFT gap = scf_cbm - scf_vbm. Clip negatives to 0 (metallic / band-crossing).
def dft_gap(d):
    if d.get("scf_vbm") is None or d.get("scf_cbm") is None:
        return None
    return d["scf_cbm"] - d["scf_vbm"]

valid = [d for d in data if dft_gap(d) is not None and d.get("sk_bandgap") is not None]
print(f"Valid entries: {len(valid)} / {len(data)}", flush=True)

sk_gap  = np.array([d["sk_bandgap"] for d in valid], dtype=float)
dft_raw = np.array([dft_gap(d)       for d in valid], dtype=float)
dft_clip = np.clip(dft_raw, 0.0, None)   # treat VBM>CBM as metallic (gap=0)
formula = [d["formula"] for d in valid]
names   = [d["name"]    for d in valid]

out_dir = "analysis"
os.makedirs(out_dir, exist_ok=True)


def print_stats(name, arr):
    print(f"  {name:30s}  N={len(arr):5d}  mean={np.mean(arr):.4f}  "
          f"std={np.std(arr):.4f}  min={np.min(arr):.4f}  max={np.max(arr):.4f}  "
          f"median={np.median(arr):.4f}")


def density_scatter(ax, x, y, bins=80, **kwargs):
    h, xedges, yedges = np.histogram2d(x, y, bins=bins)
    xidx = np.clip(np.digitize(x, xedges) - 1, 0, bins - 1)
    yidx = np.clip(np.digitize(y, yedges) - 1, 0, bins - 1)
    c = h[xidx, yidx]
    order = np.argsort(c)
    return ax.scatter(x[order], y[order], c=c[order], s=8, cmap="viridis",
                      norm=LogNorm(vmin=1), rasterized=True, **kwargs)


# ── 1. Summary stats ────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY (surface_db)")
print("=" * 70)
print_stats("SlakoNet band gap (eV)", sk_gap)
print_stats("DFT gap raw (cbm-vbm)",  dft_raw)
print_stats("DFT gap clipped to >=0", dft_clip)
n_neg = int(np.sum(dft_raw < 0))
n_zero_dft = int(np.sum(dft_clip == 0))
n_zero_sk  = int(np.sum(sk_gap < 1e-3))
print(f"  DFT entries with vbm>cbm (negative raw gap): {n_neg}")
print(f"  DFT metallic (clipped gap == 0):             {n_zero_dft}")
print(f"  SlakoNet ~0 (gap < 1 meV):                   {n_zero_sk}")

mae  = mean_absolute_error(dft_clip, sk_gap)
rmse = np.sqrt(mean_squared_error(dft_clip, sk_gap))
r    = np.corrcoef(dft_clip, sk_gap)[0, 1]
print(f"  MAE  = {mae:.4f} eV")
print(f"  RMSE = {rmse:.4f} eV")
print(f"  Pearson r = {r:.4f}")

# ── 2. Parity plot ──────────────────────────────────────────────────────
print("Parity plot...", flush=True)
fig, ax = plt.subplots(figsize=(7, 6))
sc = density_scatter(ax, dft_clip, sk_gap)
lim = max(dft_clip.max(), sk_gap.max()) * 1.05
ax.plot([0, lim], [0, lim], "r--", lw=1, label="y = x")
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.set_xlabel("DFT band gap (eV) [scf_cbm - scf_vbm, clipped ≥0]")
ax.set_ylabel("SlakoNet band gap (eV)")
ax.set_title(f"SlakoNet vs DFT (surface_db, N={len(valid)})")
ax.set_aspect("equal"); ax.legend()
plt.colorbar(sc, ax=ax, label="Density")
ax.text(0.05, 0.95,
        f"MAE  = {mae:.3f} eV\nRMSE = {rmse:.3f} eV\nr = {r:.3f}",
        transform=ax.transAxes, va="top", fontsize=10,
        bbox=dict(boxstyle="round", fc="white", alpha=0.85))
fig.tight_layout()
fig.savefig(os.path.join(out_dir, "parity_sk_vs_dft.png"), dpi=200)
plt.close(fig)

# ── 3. Distributions ────────────────────────────────────────────────────
print("Distributions...", flush=True)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].hist(sk_gap, bins=60, alpha=0.8)
axes[0].set_xlabel("SlakoNet band gap (eV)"); axes[0].set_ylabel("Count")
axes[0].set_title(f"SlakoNet gap (N={len(sk_gap)})")

axes[1].hist(dft_clip, bins=60, alpha=0.8, color="tab:orange")
axes[1].set_xlabel("DFT band gap (eV)"); axes[1].set_ylabel("Count")
axes[1].set_title(f"DFT gap (clipped, N={len(dft_clip)})")
fig.tight_layout()
fig.savefig(os.path.join(out_dir, "distributions.png"), dpi=200)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 5))
bins = np.linspace(0, max(sk_gap.max(), dft_clip.max()), 60)
ax.hist(dft_clip, bins=bins, alpha=0.55, label="DFT (clipped)")
ax.hist(sk_gap,   bins=bins, alpha=0.55, label="SlakoNet")
ax.set_xlabel("Band gap (eV)"); ax.set_ylabel("Count")
ax.set_title("Gap distribution: SlakoNet vs DFT")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(out_dir, "gap_distribution_comparison.png"), dpi=200)
plt.close(fig)

# ── 4. Residual histogram ───────────────────────────────────────────────
residuals = sk_gap - dft_clip
fig, ax = plt.subplots(figsize=(7, 5))
ax.hist(residuals, bins=80, alpha=0.85)
ax.axvline(0, color="r", ls="--")
ax.set_xlabel("SlakoNet gap − DFT gap (eV)")
ax.set_ylabel("Count")
ax.set_title(f"Residuals (mean={residuals.mean():+.3f}, std={residuals.std():.3f})")
fig.tight_layout()
fig.savefig(os.path.join(out_dir, "residual_histogram.png"), dpi=200)
plt.close(fig)

# ── 5. Metallic confusion matrix (threshold 0.1 eV) ─────────────────────
thr = 0.1
dft_metal = dft_clip < thr
sk_metal  = sk_gap   < thr
TP = int(np.sum( dft_metal &  sk_metal))
TN = int(np.sum(~dft_metal & ~sk_metal))
FP = int(np.sum(~dft_metal &  sk_metal))  # SK says metal, DFT says not
FN = int(np.sum( dft_metal & ~sk_metal))  # SK says not metal, DFT says metal
print(f"\nMetallic classification @ {thr} eV threshold:")
print(f"  DFT metal & SK metal      (TP): {TP}")
print(f"  DFT non-met & SK non-met  (TN): {TN}")
print(f"  DFT non-met & SK metal    (FP): {FP}")
print(f"  DFT metal & SK non-met    (FN): {FN}")
acc = (TP + TN) / len(valid)
print(f"  accuracy = {acc:.3f}")

# ── 6. DOS averages split by DFT gap ────────────────────────────────────
print("\nAverage DOS...", flush=True)
dos_e = np.array(valid[0]["dos_energies"])
small_mask = dft_clip <  0.5
large_mask = dft_clip >= 0.5

def mean_dos(mask):
    if mask.sum() == 0:
        return np.zeros_like(dos_e)
    return np.mean(np.stack([np.array(valid[i]["dos_values"])
                             for i in np.where(mask)[0]]), axis=0)

small_avg = mean_dos(small_mask)
large_avg = mean_dos(large_mask)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(dos_e, small_avg, label=f"DFT gap < 0.5 eV (N={int(small_mask.sum())})")
ax.plot(dos_e, large_avg, label=f"DFT gap ≥ 0.5 eV (N={int(large_mask.sum())})")
ax.set_xlabel("Energy (eV)"); ax.set_ylabel("DOS (arb.)")
ax.set_title("Average SlakoNet DOS, split by DFT gap")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(out_dir, "dos_average.png"), dpi=200)
plt.close(fig)

# ── 7. Example DOS ──────────────────────────────────────────────────────
examples = []
for target, label in [(0.0, "metallic"), (1.0, "small gap"), (3.0, "large gap")]:
    idx = int(np.argmin(np.abs(sk_gap - target)))
    examples.append((idx, label))

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, (idx, lab) in zip(axes, examples):
    e = np.array(valid[idx]["dos_energies"])
    d = np.array(valid[idx]["dos_values"])
    ax.plot(e, d)
    ax.set_xlabel("Energy (eV)"); ax.set_ylabel("DOS")
    ax.set_title(f"{formula[idx]}  [{lab}]\n"
                 f"SK={sk_gap[idx]:.2f}  DFT={dft_clip[idx]:.2f} eV\n"
                 f"{names[idx][:60]}", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(out_dir, "dos_examples.png"), dpi=200)
plt.close(fig)

# ── 8. Top outliers ─────────────────────────────────────────────────────
print("\nTop 15 |residual| outliers:")
abs_res = np.abs(residuals)
top = np.argsort(abs_res)[::-1][:15]
for rank, i in enumerate(top, 1):
    print(f"  {rank:2d}. SK={sk_gap[i]:6.3f}  DFT={dft_clip[i]:6.3f}  "
          f"res={residuals[i]:+7.3f}  {formula[i]:15s}  {names[i]}")

# ── 9. CSV summary ──────────────────────────────────────────────────────
print("\nWriting summary.csv...", flush=True)
with open("summary.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["name", "formula", "scf_vbm", "scf_cbm",
                "dft_gap_raw_eV", "dft_gap_clipped_eV",
                "sk_bandgap_eV", "residual_eV"])
    for d in data:
        g_raw = dft_gap(d)
        if g_raw is None or d.get("sk_bandgap") is None:
            w.writerow([d.get("name",""), d.get("formula",""),
                        d.get("scf_vbm",""), d.get("scf_cbm",""),
                        "", "", d.get("sk_bandgap",""), ""])
            continue
        g_clip = max(g_raw, 0.0)
        res    = d["sk_bandgap"] - g_clip
        w.writerow([d["name"], d["formula"],
                    f"{d['scf_vbm']:.4f}", f"{d['scf_cbm']:.4f}",
                    f"{g_raw:.4f}", f"{g_clip:.4f}",
                    f"{d['sk_bandgap']:.4f}", f"{res:+.4f}"])

print(f"summary.csv written ({len(data)} rows)")
print(f"Plots saved to {out_dir}/")
print("Done.")
