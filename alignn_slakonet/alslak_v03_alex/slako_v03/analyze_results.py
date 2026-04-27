import orjson
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix, ConfusionMatrixDisplay

OUT_DIR = "analysis"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load data (two-pass: scalars first, DOS on demand) ────────────────────
print("Loading JSON with orjson...", flush=True)
with open("results/all_results.json", "rb") as f:
    data = orjson.loads(f.read())
print(f"Loaded {len(data)} entries", flush=True)

# Extract scalar arrays — release references to DOS to free memory later
sk = np.array([d["sk_bandgap"] for d in data])
ind = np.array([d["band_gap_ind"] for d in data])
dirg = np.array([d["band_gap_dir"] for d in data])
eform = np.array([d["e_form"] for d in data])
formulas = [d["formula"] for d in data]
mat_ids = [d["mat_id"] for d in data]

is_metal = ind == 0.0
is_nonmetal = ~is_metal

METAL_THRESH = 0.1  # eV threshold for SlakoNet metal classification


# ── Helper ─────────────────────────────────────────────────────────────────
def print_stats(name, pred, true):
    mae = mean_absolute_error(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
    r2 = r2_score(true, pred) if len(true) > 1 and np.std(true) > 0 else float("nan")
    maxerr = np.max(np.abs(pred - true))
    print(f"  {name:30s}  N={len(true):6d}  MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}  MaxErr={maxerr:.4f}")


def density_scatter(ax, x, y, bins=200, **kwargs):
    """Scatter colored by 2D histogram density."""
    h, xedges, yedges = np.histogram2d(x, y, bins=bins)
    xidx = np.clip(np.digitize(x, xedges) - 1, 0, bins - 1)
    yidx = np.clip(np.digitize(y, yedges) - 1, 0, bins - 1)
    c = h[xidx, yidx]
    order = np.argsort(c)
    return ax.scatter(x[order], y[order], c=c[order], s=2, cmap="viridis",
                      norm=LogNorm(vmin=1), rasterized=True, **kwargs)


# ── 1. Statistics ──────────────────────────────────────────────────────────
print("=" * 80)
print("ERROR STATISTICS (SlakoNet vs PBE)")
print("=" * 80)
print("\nIndirect gap:")
print_stats("All", sk, ind)
print_stats("Metals (PBE gap=0)", sk[is_metal], ind[is_metal])
print_stats("Non-metals (PBE gap>0)", sk[is_nonmetal], ind[is_nonmetal])
print("\nDirect gap:")
print_stats("All", sk, dirg)
print_stats("Non-metals (PBE gap>0)", sk[is_nonmetal], dirg[is_nonmetal])

# ── 2. Parity plots ───────────────────────────────────────────────────────
print("\nGenerating parity plots...", flush=True)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, (ref, label) in zip(axes, [(ind, "Indirect gap"), (dirg, "Direct gap")]):
    sc = density_scatter(ax, ref, sk)
    lim = 15.0
    ax.plot([0, lim], [0, lim], "r--", lw=1, label="y = x")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel(f"PBE {label} (eV)")
    ax.set_ylabel("SlakoNet gap (eV)")
    ax.set_title(f"SlakoNet vs PBE {label}")
    ax.set_aspect("equal")
    ax.legend()
    plt.colorbar(sc, ax=ax, label="Density")
    mae = mean_absolute_error(ref, sk)
    rmse = np.sqrt(mean_squared_error(ref, sk))
    ax.text(0.05, 0.95, f"MAE = {mae:.3f} eV\nRMSE = {rmse:.3f} eV",
            transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(boxstyle="round", fc="white", alpha=0.8))

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "parity_all.png"), dpi=200)
plt.close(fig)

# Parity for non-metals only
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, (ref, label) in zip(axes, [(ind[is_nonmetal], "Indirect gap"), (dirg[is_nonmetal], "Direct gap")]):
    pred = sk[is_nonmetal]
    sc = density_scatter(ax, ref, pred)
    lim = 15.0
    ax.plot([0, lim], [0, lim], "r--", lw=1, label="y = x")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel(f"PBE {label} (eV)")
    ax.set_ylabel("SlakoNet gap (eV)")
    ax.set_title(f"Non-metals: SlakoNet vs PBE {label}")
    ax.set_aspect("equal")
    ax.legend()
    plt.colorbar(sc, ax=ax, label="Density")
    mae = mean_absolute_error(ref, pred)
    rmse = np.sqrt(mean_squared_error(ref, pred))
    ax.text(0.05, 0.95, f"MAE = {mae:.3f} eV\nRMSE = {rmse:.3f} eV",
            transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(boxstyle="round", fc="white", alpha=0.8))
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "parity_nonmetals.png"), dpi=200)
plt.close(fig)

# Parity for metals only
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, (ref, label) in zip(axes, [(ind[is_metal], "Indirect gap"), (dirg[is_metal], "Direct gap")]):
    pred = sk[is_metal]
    sc = density_scatter(ax, ref, pred)
    lim = 15.0
    ax.plot([0, lim], [0, lim], "r--", lw=1, label="y = x")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel(f"PBE {label} (eV)")
    ax.set_ylabel("SlakoNet gap (eV)")
    ax.set_title(f"Metals: SlakoNet vs PBE {label}")
    ax.set_aspect("equal")
    ax.legend()
    plt.colorbar(sc, ax=ax, label="Density")
    mae = mean_absolute_error(ref, pred)
    rmse = np.sqrt(mean_squared_error(ref, pred))
    ax.text(0.05, 0.95, f"MAE = {mae:.3f} eV\nRMSE = {rmse:.3f} eV",
            transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(boxstyle="round", fc="white", alpha=0.8))
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "parity_metals.png"), dpi=200)
plt.close(fig)

# ── 3. Error histograms ───────────────────────────────────────────────────
print("Generating error histograms...", flush=True)
residuals = sk - ind
residuals_nm = residuals[is_nonmetal]

# All entries
fig, ax = plt.subplots(figsize=(7, 5))
ax.hist(residuals, bins=300, edgecolor="none", alpha=0.8)
ax.axvline(0, color="r", ls="--")
ax.set_xlabel("SlakoNet − PBE indirect gap (eV)")
ax.set_ylabel("Count")
ax.set_title(f"Residual distribution (all, N={len(residuals)})")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "residual_histogram_all.png"), dpi=200)
plt.close(fig)

# Non-metals only
fig, ax = plt.subplots(figsize=(7, 5))
ax.hist(residuals_nm, bins=300, edgecolor="none", alpha=0.8)
ax.axvline(0, color="r", ls="--")

# Split into working (SK > 0.1) and broken (SK <= 0.1) groups
sk_nm = sk[is_nonmetal]
working_mask = sk_nm > METAL_THRESH
broken_mask = sk_nm <= METAL_THRESH
avg_all = residuals_nm.mean()
avg_working = residuals_nm[working_mask].mean()
avg_broken = residuals_nm[broken_mask].mean()
n_working = working_mask.sum()
n_broken = broken_mask.sum()

# Plot average lines
ax.axvline(avg_all, color="k", ls="-", lw=1.5, label=f"Overall mean: {avg_all:+.2f} eV (N={len(residuals_nm)})")
ax.axvline(avg_working, color="green", ls="-", lw=1.5, label=f"Working mean: {avg_working:+.2f} eV (N={n_working})")
ax.axvline(avg_broken, color="orange", ls="-", lw=1.5, label=f"Broken mean: {avg_broken:+.2f} eV (N={n_broken})")
ax.legend(fontsize=8)

ax.set_xlabel("SlakoNet − PBE indirect gap (eV)")
ax.set_ylabel("Count")
ax.set_title(f"Residual distribution (non-metals, N={len(residuals_nm)})")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "residual_histogram_nonmetals.png"), dpi=200)
plt.close(fig)

# ── 4. Metal/insulator classification ─────────────────────────────────────
print("Computing classification metrics...", flush=True)
pbe_class = (ind > 0).astype(int)  # 1 = non-metal
sk_class = (sk > METAL_THRESH).astype(int)

cm = confusion_matrix(pbe_class, sk_class)
accuracy = np.trace(cm) / cm.sum()
print(f"\nMetal/non-metal classification (threshold={METAL_THRESH} eV):")
print(f"  Accuracy: {accuracy:.4f}")
print(f"  Confusion matrix (rows=PBE, cols=SK):  TN={cm[0,0]}  FP={cm[0,1]}  FN={cm[1,0]}  TP={cm[1,1]}")

fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(cm, display_labels=["Metal", "Non-metal"]).plot(ax=ax, cmap="Blues")
ax.set_xlabel("SlakoNet prediction")
ax.set_ylabel("PBE reference")
ax.set_title(f"Metal/Non-metal classification (acc={accuracy:.3f})")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=200)
plt.close(fig)

# ── 4b. Diagnose false negatives (PBE non-metal, SK predicted metal) ─────
fn_mask = (pbe_class == 1) & (sk_class == 0)  # false negatives
fn_pbe_gaps = ind[fn_mask]
fn_sk_gaps = sk[fn_mask]
print(f"\n--- False negatives: PBE non-metal but SK predicts metal (N={fn_mask.sum()}) ---")
print(f"  PBE indirect gap:  min={fn_pbe_gaps.min():.4f}  median={np.median(fn_pbe_gaps):.4f}  "
      f"mean={fn_pbe_gaps.mean():.4f}  max={fn_pbe_gaps.max():.4f}")
print(f"  SK bandgap:        min={fn_sk_gaps.min():.4f}  median={np.median(fn_sk_gaps):.4f}  "
      f"mean={fn_sk_gaps.mean():.4f}  max={fn_sk_gaps.max():.4f}")
for pct in [25, 50, 75, 90, 95, 99]:
    print(f"  PBE gap {pct:>2d}th percentile: {np.percentile(fn_pbe_gaps, pct):.4f} eV")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.hist(fn_pbe_gaps, bins=100, edgecolor="none", alpha=0.8)
ax.set_xlabel("PBE indirect gap (eV)")
ax.set_ylabel("Count")
ax.set_title(f"False negatives: PBE gap distribution (N={fn_mask.sum()})")
ax.axvline(np.median(fn_pbe_gaps), color="r", ls="--", label=f"median={np.median(fn_pbe_gaps):.3f}")
ax.legend()

ax = axes[1]
ax.scatter(fn_pbe_gaps, fn_sk_gaps, s=3, alpha=0.3)
ax.set_xlabel("PBE indirect gap (eV)")
ax.set_ylabel("SlakoNet gap (eV)")
ax.set_title("False negatives: PBE gap vs SK gap")
ax.axhline(METAL_THRESH, color="r", ls="--", label=f"SK metal threshold={METAL_THRESH}")
ax.plot([0, fn_pbe_gaps.max()], [0, fn_pbe_gaps.max()], "k--", lw=0.5, alpha=0.5, label="y=x")
ax.legend()

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "false_negatives_diagnosis.png"), dpi=200)
plt.close(fig)
print(f"  Saved: {OUT_DIR}/false_negatives_diagnosis.png")

# ── 5. Band gap distribution comparison ───────────────────────────────────
# All entries
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(ind, bins=200, alpha=0.6, label="PBE indirect gap", density=True)
ax.hist(sk, bins=200, alpha=0.6, label="SlakoNet gap", density=True)
ax.set_xlabel("Band gap (eV)")
ax.set_ylabel("Density")
ax.set_title(f"Band gap distribution: PBE vs SlakoNet (all, N={len(ind)})")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "gap_distribution_all.png"), dpi=200)
plt.close(fig)

# Non-metals only
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(ind[is_nonmetal], bins=200, alpha=0.6, label="PBE indirect gap", density=True)
ax.hist(sk[is_nonmetal], bins=200, alpha=0.6, label="SlakoNet gap", density=True)
ax.set_xlabel("Band gap (eV)")
ax.set_ylabel("Density")
ax.set_title(f"Band gap distribution: PBE vs SlakoNet (non-metals, N={int(is_nonmetal.sum())})")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "gap_distribution_nonmetals.png"), dpi=200)
plt.close(fig)

# ── 6. Formation energy vs error ──────────────────────────────────────────
print("Generating formation energy plot...", flush=True)
fig, ax = plt.subplots(figsize=(8, 5))
abs_err = np.abs(residuals)
sc = ax.scatter(eform, abs_err, s=1, c=ind, cmap="coolwarm", alpha=0.5, rasterized=True)
ax.set_xlabel("Formation energy (eV/atom)")
ax.set_ylabel("|SlakoNet − PBE| indirect gap (eV)")
ax.set_title("Prediction error vs formation energy")
plt.colorbar(sc, ax=ax, label="PBE indirect gap (eV)")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "error_vs_eform.png"), dpi=200)
plt.close(fig)

# ── 7. DOS analysis ───────────────────────────────────────────────────────
print("\nComputing average DOS (sampling for efficiency)...", flush=True)
metal_indices = np.where(is_metal)[0]
nonmetal_indices = np.where(is_nonmetal)[0]

# Use first entry's energy grid as reference
dos_energies = np.array(data[0]["dos_energies"])

# Average over all entries using running mean to limit memory
metal_dos_avg = np.zeros(len(dos_energies))
for i in metal_indices:
    metal_dos_avg += np.array(data[i]["dos_values"])
if len(metal_indices) > 0:
    metal_dos_avg /= len(metal_indices)

nonmetal_dos_avg = np.zeros(len(dos_energies))
for i in nonmetal_indices:
    nonmetal_dos_avg += np.array(data[i]["dos_values"])
if len(nonmetal_indices) > 0:
    nonmetal_dos_avg /= len(nonmetal_indices)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(dos_energies, metal_dos_avg, label=f"Metals (avg of {len(metal_indices)})", alpha=0.8)
ax.plot(dos_energies, nonmetal_dos_avg, label=f"Non-metals (avg of {len(nonmetal_indices)})", alpha=0.8)
ax.set_xlabel("Energy (eV)")
ax.set_ylabel("DOS (arb. units)")
ax.set_title("Average DOS: Metals vs Non-metals")
ax.legend()
ax.set_xlim(dos_energies.min(), dos_energies.max())
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "dos_average.png"), dpi=200)
plt.close(fig)

# Example DOS plots for a few structures
examples = []
for target_gap, label in [(0.0, "metal"), (1.0, "semiconductor"), (4.0, "wide-gap insulator")]:
    best_idx = int(np.argmin(np.abs(ind - target_gap)))
    examples.append((best_idx, label, formulas[best_idx], mat_ids[best_idx]))

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, (idx, label, formula, mat_id) in zip(axes, examples):
    e = np.array(data[idx]["dos_energies"])
    d = np.array(data[idx]["dos_values"])
    ax.plot(e, d)
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("DOS")
    ax.set_title(f"{formula} ({mat_id})\nPBE gap={ind[idx]:.2f}, SK gap={sk[idx]:.2f} eV\n[{label}]")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "dos_examples.png"), dpi=200)
plt.close(fig)

print(f"\nAll plots saved to {OUT_DIR}/")
print("Done.")
