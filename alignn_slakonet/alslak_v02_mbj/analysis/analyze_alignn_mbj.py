"""
Analyze TB-mBJ ALIGNN predictions (v02) against SlakoNet and PBE reference.

Mirrors the v00 analysis (which used the PBE-trained `mp_gappbe_alignn` model),
but here ALIGNN is the `jv_mbj_bandgap_alignn` checkpoint — trained on JARVIS
TB-mBJ gaps, which are systematically larger than PBE gaps. That is the point
of the comparison: mBJ should NOT track PBE one-to-one; it should sit higher.

Loads:
  - ../results/alignn_predictions.json              (v02 TB-mBJ predictions, + PBE cols)
  - ../../alslak_v00/results/sk_scalars.json        (SlakoNet scalars, cached from v00)

Produces plots + stats in this directory.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score, confusion_matrix,
    ConfusionMatrixDisplay,
)

HERE = os.path.dirname(os.path.abspath(__file__))
ALIGNN_FILE = os.path.join(HERE, "..", "results", "alignn_predictions.json")
SK_FILE = os.path.join(HERE, "..", "..", "alslak_v00", "results", "sk_scalars.json")
OUT_DIR = HERE
STATS_FILE = os.path.join(OUT_DIR, "stats.txt")
METAL_THRESH = 0.1  # eV

# TB-mBJ produces rare wild outliers (saw max=99 eV). Clip for plotting only.
PLOT_LIM = 12.0


# ── Load ─────────────────────────────────────────────────────────────────
print(f"Loading ALIGNN (TB-mBJ) predictions from {ALIGNN_FILE}...", flush=True)
with open(ALIGNN_FILE) as f:
    alignn_data = json.load(f)
print(f"  {len(alignn_data)} entries", flush=True)

print(f"Loading SlakoNet scalars from {SK_FILE}...", flush=True)
with open(SK_FILE) as f:
    sk_scalars = json.load(f)
print(f"  {len(sk_scalars)} entries", flush=True)

sk_map = {d["mat_id"]: d["sk_bandgap"] for d in sk_scalars}


# ── Merge on mat_id ──────────────────────────────────────────────────────
merged = []
missing_sk = 0
for entry in alignn_data:
    mid = entry["mat_id"]
    sk = sk_map.get(mid)
    if sk is None:
        missing_sk += 1
        continue
    merged.append({
        "mat_id": mid,
        "formula": entry["formula"],
        "pbe_ind": entry["band_gap_ind"],
        "pbe_dir": entry["band_gap_dir"],
        "e_form": entry["e_form"],
        "sk": sk,
        "alignn_mbj": entry["alignn_bandgap"],
    })

print(f"\nMerged {len(merged)} entries ({missing_sk} ALIGNN entries had no SK match)", flush=True)

pbe = np.array([d["pbe_ind"] for d in merged])
sk = np.array([d["sk"] for d in merged])
al = np.array([d["alignn_mbj"] for d in merged])
eform = np.array([d["e_form"] for d in merged])

# Clamp negative ALIGNN predictions to 0 (physical lower bound)
al_clamped = np.clip(al, 0, None)

is_metal = pbe == 0.0
is_nonmetal = ~is_metal

# Report lines buffered to write to stats.txt alongside stdout
report_lines = []

def log(msg=""):
    print(msg, flush=True)
    report_lines.append(msg)


log(f"Paired structures (v02 TB-mBJ ∩ SlakoNet): {len(merged)}")
log(f"  PBE metals (gap==0): {is_metal.sum()}   PBE non-metals: {is_nonmetal.sum()}")
log(f"  ALIGNN TB-mBJ negative predictions (clamped to 0): {(al < 0).sum()}")
log(f"  ALIGNN TB-mBJ max prediction: {al.max():.3f} eV")
log(f"  ALIGNN TB-mBJ > 15 eV outliers: {(al > 15).sum()}")


# ── Helpers ──────────────────────────────────────────────────────────────
def stats(name, pred, true):
    mae = mean_absolute_error(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
    r2 = r2_score(true, pred) if len(true) > 1 and np.std(true) > 0 else float("nan")
    me = (pred - true).mean()
    maxerr = np.max(np.abs(pred - true))
    line = (f"  {name:40s}  N={len(true):6d}  MAE={mae:.4f}  RMSE={rmse:.4f}"
            f"  R²={r2:.4f}  ME={me:+.4f}  MaxErr={maxerr:.4f}")
    log(line)


def density_scatter(ax, x, y, bins=200):
    h, xe, ye = np.histogram2d(x, y, bins=bins)
    xi = np.clip(np.digitize(x, xe) - 1, 0, bins - 1)
    yi = np.clip(np.digitize(y, ye) - 1, 0, bins - 1)
    c = h[xi, yi]
    order = np.argsort(c)
    return ax.scatter(x[order], y[order], c=c[order], s=2, cmap="viridis",
                      norm=LogNorm(vmin=1), rasterized=True)


# ── 1. Regression tables ────────────────────────────────────────────────
log("\n" + "=" * 90)
log("REGRESSION vs PBE indirect gap  (PBE is the reference label)")
log("=" * 90)
log("\nAll structures:")
stats("SlakoNet", sk, pbe)
stats("ALIGNN TB-mBJ (raw)", al, pbe)
stats("ALIGNN TB-mBJ (clamped >= 0)", al_clamped, pbe)

log("\nNon-metals only (PBE gap > 0):")
stats("SlakoNet", sk[is_nonmetal], pbe[is_nonmetal])
stats("ALIGNN TB-mBJ (raw)", al[is_nonmetal], pbe[is_nonmetal])
stats("ALIGNN TB-mBJ (clamped >= 0)", al_clamped[is_nonmetal], pbe[is_nonmetal])

log("\nMetals only (PBE gap == 0):")
stats("SlakoNet", sk[is_metal], pbe[is_metal])
stats("ALIGNN TB-mBJ (raw)", al[is_metal], pbe[is_metal])
stats("ALIGNN TB-mBJ (clamped >= 0)", al_clamped[is_metal], pbe[is_metal])

log("\n" + "=" * 90)
log("ALIGNN TB-mBJ vs SlakoNet (direct comparison, no PBE reference)")
log("=" * 90)
log("\nAll structures:")
stats("ALIGNN TB-mBJ − SlakoNet", al_clamped, sk)
log("\nNon-metals only:")
stats("ALIGNN TB-mBJ − SlakoNet", al_clamped[is_nonmetal], sk[is_nonmetal])


# ── 2. Parity (3-panel: SK vs PBE, ALIGNN-mBJ vs PBE, ALIGNN-mBJ vs SK) ─
log("\nGenerating parity plots...")
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
panels = [
    (pbe, sk, "PBE indirect gap (eV)", "SlakoNet gap (eV)", "SlakoNet vs PBE"),
    (pbe, al_clamped, "PBE indirect gap (eV)", "ALIGNN TB-mBJ gap (eV)", "ALIGNN TB-mBJ vs PBE"),
    (sk, al_clamped, "SlakoNet gap (eV)", "ALIGNN TB-mBJ gap (eV)", "ALIGNN TB-mBJ vs SlakoNet"),
]
for ax, (x, y, xl, yl, title) in zip(axes, panels):
    sc = density_scatter(ax, x, y)
    ax.plot([0, PLOT_LIM], [0, PLOT_LIM], "r--", lw=1, label="y = x")
    ax.set_xlim(0, PLOT_LIM)
    ax.set_ylim(0, PLOT_LIM)
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.legend(loc="lower right")
    plt.colorbar(sc, ax=ax, label="Density")
    mae = mean_absolute_error(x, y)
    rmse = np.sqrt(mean_squared_error(x, y))
    ax.text(0.05, 0.95, f"MAE = {mae:.3f} eV\nRMSE = {rmse:.3f} eV",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", alpha=0.8))
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "parity_three_way.png"), dpi=200)
plt.close(fig)


# ── 3. Non-metals parity ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
panels_nm = [
    (pbe[is_nonmetal], sk[is_nonmetal], "PBE indirect gap (eV)", "SlakoNet gap (eV)", "Non-metals: SlakoNet vs PBE"),
    (pbe[is_nonmetal], al_clamped[is_nonmetal], "PBE indirect gap (eV)", "ALIGNN TB-mBJ gap (eV)", "Non-metals: ALIGNN TB-mBJ vs PBE"),
    (sk[is_nonmetal], al_clamped[is_nonmetal], "SlakoNet gap (eV)", "ALIGNN TB-mBJ gap (eV)", "Non-metals: ALIGNN TB-mBJ vs SlakoNet"),
]
for ax, (x, y, xl, yl, title) in zip(axes, panels_nm):
    sc = density_scatter(ax, x, y)
    ax.plot([0, PLOT_LIM], [0, PLOT_LIM], "r--", lw=1, label="y = x")
    ax.set_xlim(0, PLOT_LIM)
    ax.set_ylim(0, PLOT_LIM)
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.legend(loc="lower right")
    plt.colorbar(sc, ax=ax, label="Density")
    mae = mean_absolute_error(x, y)
    rmse = np.sqrt(mean_squared_error(x, y))
    ax.text(0.05, 0.95, f"MAE = {mae:.3f} eV\nRMSE = {rmse:.3f} eV",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", alpha=0.8))
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "parity_three_way_nonmetals.png"), dpi=200)
plt.close(fig)


# ── 4. Residual histograms ──────────────────────────────────────────────
log("Generating residual histograms...")
res_sk = sk - pbe
res_al = al_clamped - pbe

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, res, label in zip(axes, [res_sk, res_al], ["SlakoNet", "ALIGNN TB-mBJ"]):
    ax.hist(res, bins=300, alpha=0.8)
    ax.axvline(0, color="r", ls="--")
    ax.axvline(res.mean(), color="k", ls="-", lw=1.2,
               label=f"mean = {res.mean():+.3f} eV")
    ax.set_xlabel(f"{label} − PBE indirect gap (eV)")
    ax.set_ylabel("Count")
    ax.set_title(f"Residuals: {label} − PBE  (N={len(res)}, MAE={np.abs(res).mean():.3f})")
    ax.legend()
    ax.set_xlim(-6, 8)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "residuals_sk_vs_alignn_mbj.png"), dpi=200)
plt.close(fig)


# ── 5. Classification (metal vs non-metal at 0.1 eV threshold) ─────────
log("\nComputing classification metrics (threshold = 0.1 eV)...")
pbe_cls = (pbe > 0).astype(int)
sk_cls = (sk > METAL_THRESH).astype(int)
al_cls = (al_clamped > METAL_THRESH).astype(int)

for name, pred in [("SlakoNet", sk_cls), ("ALIGNN TB-mBJ", al_cls)]:
    cm = confusion_matrix(pbe_cls, pred)
    acc = np.trace(cm) / cm.sum()
    log(f"\n{name} metal/non-metal classification:")
    log(f"  Accuracy: {acc:.4f}")
    log(f"  TN(metal→metal)={cm[0,0]}  FP(metal→nonmetal)={cm[0,1]}  "
        f"FN(nonmetal→metal)={cm[1,0]}  TP(nonmetal→nonmetal)={cm[1,1]}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, pred, name in zip(axes, [sk_cls, al_cls], ["SlakoNet", "ALIGNN TB-mBJ"]):
    cm = confusion_matrix(pbe_cls, pred)
    acc = np.trace(cm) / cm.sum()
    ConfusionMatrixDisplay(cm, display_labels=["Metal", "Non-metal"]).plot(
        ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{name} (acc={acc:.3f})")
    ax.set_xlabel(f"{name} prediction")
    ax.set_ylabel("PBE reference")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "confusion_sk_vs_alignn_mbj.png"), dpi=200)
plt.close(fig)


# ── 6. Head-to-head: which model is closer to PBE per structure ────────
log("\nGenerating head-to-head error plot (vs PBE)...")
abs_sk = np.abs(res_sk)
abs_al = np.abs(res_al)
sk_better = abs_sk < abs_al
log(f"  SlakoNet closer to PBE:     {sk_better.sum()} ({sk_better.mean()*100:.1f}%)")
log(f"  ALIGNN TB-mBJ closer to PBE: {(~sk_better).sum()} ({(~sk_better).mean()*100:.1f}%)")

fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(abs_sk[~sk_better], abs_al[~sk_better], s=2, alpha=0.3,
           c="tab:blue", label=f"ALIGNN TB-mBJ better (N={(~sk_better).sum()})", rasterized=True)
ax.scatter(abs_sk[sk_better], abs_al[sk_better], s=2, alpha=0.3,
           c="tab:orange", label=f"SlakoNet better (N={sk_better.sum()})", rasterized=True)
ax.plot([0, 10], [0, 10], "k--", lw=1)
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_xlabel("|SlakoNet − PBE| (eV)")
ax.set_ylabel("|ALIGNN TB-mBJ − PBE| (eV)")
ax.set_title("Per-structure |error| vs PBE reference")
ax.set_aspect("equal")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "head_to_head_error.png"), dpi=200)
plt.close(fig)


# ── 7. Gap distributions ────────────────────────────────────────────────
log("Generating gap distribution plot...")
fig, ax = plt.subplots(figsize=(9, 5))
bins = np.linspace(0, 10, 200)
ax.hist(pbe[is_nonmetal], bins=bins, alpha=0.5, label=f"PBE (N={is_nonmetal.sum()})", density=True)
ax.hist(sk[is_nonmetal], bins=bins, alpha=0.5, label="SlakoNet", density=True)
ax.hist(al_clamped[is_nonmetal], bins=bins, alpha=0.5, label="ALIGNN TB-mBJ", density=True)
ax.set_xlabel("Band gap (eV)")
ax.set_ylabel("Density")
ax.set_title(f"Gap distributions (non-metals, N={is_nonmetal.sum()})")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "gap_distribution_alignn_mbj.png"), dpi=200)
plt.close(fig)


# ── 8. mBJ − PBE residual vs PBE gap: is the mBJ "opening" size-dependent?
log("Generating mBJ-vs-PBE gap-size plot...")
fig, ax = plt.subplots(figsize=(9, 6))
nm_pbe = pbe[is_nonmetal]
nm_diff = al_clamped[is_nonmetal] - nm_pbe
sc = density_scatter(ax, nm_pbe, nm_diff, bins=150)
ax.axhline(0, color="r", ls="--", lw=1)
ax.set_xlabel("PBE indirect gap (eV)")
ax.set_ylabel("ALIGNN TB-mBJ − PBE (eV)")
ax.set_title("mBJ correction size vs PBE gap (non-metals)")
ax.set_xlim(0, 10)
ax.set_ylim(-4, 6)
plt.colorbar(sc, ax=ax, label="Density")
# Binned mean line
edges = np.linspace(0, 10, 41)
centers = 0.5 * (edges[1:] + edges[:-1])
means = []
for lo, hi in zip(edges[:-1], edges[1:]):
    sel = (nm_pbe >= lo) & (nm_pbe < hi)
    means.append(nm_diff[sel].mean() if sel.sum() >= 5 else np.nan)
ax.plot(centers, means, "-", color="black", lw=2, label="binned mean")
ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "mbj_correction_vs_pbe.png"), dpi=200)
plt.close(fig)


# ── 9. Linear fit: PBE gap vs ALIGNN TB-mBJ (as a sanity "rescale" model) ─
# If mBJ is a smooth upward-scaling of PBE, a ~linear fit should hold on non-metals.
log("Linear fit ALIGNN-mBJ = a * PBE + b on non-metals...")
x = pbe[is_nonmetal]
y = al_clamped[is_nonmetal]
a, b = np.polyfit(x, y, 1)
log(f"  slope = {a:.4f}   intercept = {b:+.4f} eV")
log(f"  interpretation: TB-mBJ ≈ {a:.2f}·PBE + {b:+.2f} eV on PBE non-metals")


# ── 10. Dump paired records for downstream use ─────────────────────────
log("Dumping paired_predictions.json...")
with open(os.path.join(OUT_DIR, "paired_predictions.json"), "w") as f:
    json.dump(merged, f)


# ── Finalize ────────────────────────────────────────────────────────────
with open(STATS_FILE, "w") as f:
    f.write("\n".join(report_lines) + "\n")

log(f"\nAll outputs written to {OUT_DIR}/")
log("Done.")
