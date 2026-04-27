"""
Analyze ALIGNN predictions against PBE reference and SlakoNet predictions.

Loads:
  - alignn_results/alignn_predictions.json  (mat_id, PBE gaps, alignn_bandgap)
  - results/all_results.json                (mat_id, SlakoNet sk_bandgap)

Produces plots in analysis/ comparing the three predictors.
"""

import os
import json
import orjson
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score, confusion_matrix,
    ConfusionMatrixDisplay,
)

OUT_DIR = "analysis"
os.makedirs(OUT_DIR, exist_ok=True)
ALIGNN_FILE = "alignn_results/alignn_predictions.json"
SK_FILE = "results/all_results.json"
SK_SCALARS_CACHE = "results/sk_scalars.json"
METAL_THRESH = 0.1  # eV


# ── Load ALIGNN predictions ──────────────────────────────────────────────
print(f"Loading ALIGNN predictions from {ALIGNN_FILE}...", flush=True)
with open(ALIGNN_FILE, "rb") as f:
    alignn_data = orjson.loads(f.read())
print(f"  {len(alignn_data)} entries", flush=True)

alignn_map = {d["mat_id"]: d for d in alignn_data}


# ── Load SlakoNet scalars (cache to avoid 5.6GB re-parse) ────────────────
if os.path.exists(SK_SCALARS_CACHE):
    print(f"Loading cached SK scalars from {SK_SCALARS_CACHE}...", flush=True)
    with open(SK_SCALARS_CACHE, "rb") as f:
        sk_scalars = orjson.loads(f.read())
else:
    print(f"Extracting SK scalars from {SK_FILE} (this may take a few minutes)...", flush=True)
    with open(SK_FILE, "rb") as f:
        sk_data = orjson.loads(f.read())
    print(f"  {len(sk_data)} SK entries loaded; extracting scalars...", flush=True)
    sk_scalars = [
        {
            "mat_id": d["mat_id"],
            "sk_bandgap": d["sk_bandgap"],
        }
        for d in sk_data
    ]
    del sk_data
    with open(SK_SCALARS_CACHE, "wb") as f:
        f.write(orjson.dumps(sk_scalars))
    print(f"  Cached {len(sk_scalars)} scalars to {SK_SCALARS_CACHE}", flush=True)

sk_map = {d["mat_id"]: d["sk_bandgap"] for d in sk_scalars}


# ── Merge ────────────────────────────────────────────────────────────────
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
        "alignn": entry["alignn_bandgap"],
    })

print(f"\nMerged {len(merged)} entries ({missing_sk} ALIGNN entries had no SK match)", flush=True)

pbe = np.array([d["pbe_ind"] for d in merged])
pbe_dir = np.array([d["pbe_dir"] for d in merged])
sk = np.array([d["sk"] for d in merged])
al = np.array([d["alignn"] for d in merged])
eform = np.array([d["e_form"] for d in merged])

# Clamp negative ALIGNN predictions to 0 (physical lower bound)
al_clamped = np.clip(al, 0, None)

is_metal = pbe == 0.0
is_nonmetal = ~is_metal

print(f"  PBE metals: {is_metal.sum()}   PBE non-metals: {is_nonmetal.sum()}", flush=True)
print(f"  ALIGNN negative preds (clamped to 0): {(al < 0).sum()}", flush=True)


# ── Helpers ──────────────────────────────────────────────────────────────
def stats(name, pred, true):
    mae = mean_absolute_error(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
    r2 = r2_score(true, pred) if len(true) > 1 and np.std(true) > 0 else float("nan")
    me = (pred - true).mean()
    maxerr = np.max(np.abs(pred - true))
    print(f"  {name:40s}  N={len(true):6d}  MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}  ME={me:+.4f}  MaxErr={maxerr:.4f}")


def density_scatter(ax, x, y, bins=200):
    h, xe, ye = np.histogram2d(x, y, bins=bins)
    xi = np.clip(np.digitize(x, xe) - 1, 0, bins - 1)
    yi = np.clip(np.digitize(y, ye) - 1, 0, bins - 1)
    c = h[xi, yi]
    order = np.argsort(c)
    return ax.scatter(x[order], y[order], c=c[order], s=2, cmap="viridis",
                      norm=LogNorm(vmin=1), rasterized=True)


# ── 1. Statistics ────────────────────────────────────────────────────────
print("\n" + "=" * 90)
print("REGRESSION METRICS vs PBE indirect gap")
print("=" * 90)
print("\nAll structures:")
stats("SlakoNet", sk, pbe)
stats("ALIGNN (raw)", al, pbe)
stats("ALIGNN (clamped >= 0)", al_clamped, pbe)

print("\nNon-metals only (PBE gap > 0):")
stats("SlakoNet", sk[is_nonmetal], pbe[is_nonmetal])
stats("ALIGNN (raw)", al[is_nonmetal], pbe[is_nonmetal])
stats("ALIGNN (clamped >= 0)", al_clamped[is_nonmetal], pbe[is_nonmetal])

print("\nMetals only (PBE gap == 0):")
stats("SlakoNet", sk[is_metal], pbe[is_metal])
stats("ALIGNN (raw)", al[is_metal], pbe[is_metal])
stats("ALIGNN (clamped >= 0)", al_clamped[is_metal], pbe[is_metal])


# ── 2. Parity plots (3-panel: SK vs PBE, ALIGNN vs PBE, ALIGNN vs SK) ───
print("\nGenerating parity plots...", flush=True)
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
lim = 10.0

panels = [
    (pbe, sk, "PBE indirect gap (eV)", "SlakoNet gap (eV)", "SlakoNet vs PBE"),
    (pbe, al_clamped, "PBE indirect gap (eV)", "ALIGNN gap (eV)", "ALIGNN vs PBE"),
    (sk, al_clamped, "SlakoNet gap (eV)", "ALIGNN gap (eV)", "ALIGNN vs SlakoNet"),
]
for ax, (x, y, xl, yl, title) in zip(axes, panels):
    sc = density_scatter(ax, x, y)
    ax.plot([0, lim], [0, lim], "r--", lw=1, label="y = x")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
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
    (pbe[is_nonmetal], al_clamped[is_nonmetal], "PBE indirect gap (eV)", "ALIGNN gap (eV)", "Non-metals: ALIGNN vs PBE"),
    (sk[is_nonmetal], al_clamped[is_nonmetal], "SlakoNet gap (eV)", "ALIGNN gap (eV)", "Non-metals: ALIGNN vs SlakoNet"),
]
for ax, (x, y, xl, yl, title) in zip(axes, panels_nm):
    sc = density_scatter(ax, x, y)
    ax.plot([0, lim], [0, lim], "r--", lw=1, label="y = x")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
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
print("Generating residual histograms...", flush=True)
res_sk = sk - pbe
res_al = al_clamped - pbe

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, res, label in zip(axes, [res_sk, res_al], ["SlakoNet", "ALIGNN"]):
    ax.hist(res, bins=300, alpha=0.8)
    ax.axvline(0, color="r", ls="--")
    ax.axvline(res.mean(), color="k", ls="-", lw=1.2,
               label=f"mean = {res.mean():+.3f} eV")
    ax.set_xlabel(f"{label} − PBE indirect gap (eV)")
    ax.set_ylabel("Count")
    ax.set_title(f"Residuals: {label} − PBE  (N={len(res)}, MAE={np.abs(res).mean():.3f})")
    ax.legend()
    ax.set_xlim(-6, 6)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "residuals_sk_vs_alignn.png"), dpi=200)
plt.close(fig)


# ── 5. Classification (metal vs non-metal) ──────────────────────────────
print("Computing classification metrics...", flush=True)
pbe_cls = (pbe > 0).astype(int)
sk_cls = (sk > METAL_THRESH).astype(int)
al_cls = (al_clamped > METAL_THRESH).astype(int)

for name, pred in [("SlakoNet", sk_cls), ("ALIGNN", al_cls)]:
    cm = confusion_matrix(pbe_cls, pred)
    acc = np.trace(cm) / cm.sum()
    print(f"\n{name} metal/non-metal classification (threshold={METAL_THRESH} eV):")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  TN={cm[0,0]}  FP={cm[0,1]}  FN={cm[1,0]}  TP={cm[1,1]}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, pred, name in zip(axes, [sk_cls, al_cls], ["SlakoNet", "ALIGNN"]):
    cm = confusion_matrix(pbe_cls, pred)
    acc = np.trace(cm) / cm.sum()
    ConfusionMatrixDisplay(cm, display_labels=["Metal", "Non-metal"]).plot(
        ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{name} (acc={acc:.3f})")
    ax.set_xlabel(f"{name} prediction")
    ax.set_ylabel("PBE reference")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "confusion_sk_vs_alignn.png"), dpi=200)
plt.close(fig)


# ── 6. Head-to-head: which model wins per structure ─────────────────────
print("\nGenerating head-to-head error plot...", flush=True)
abs_sk = np.abs(res_sk)
abs_al = np.abs(res_al)
sk_better = abs_sk < abs_al
print(f"  SlakoNet closer to PBE: {sk_better.sum()} ({sk_better.mean()*100:.1f}%)")
print(f"  ALIGNN closer to PBE:   {(~sk_better).sum()} ({(~sk_better).mean()*100:.1f}%)")

fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(abs_sk[~sk_better], abs_al[~sk_better], s=2, alpha=0.3,
           c="tab:blue", label=f"ALIGNN better (N={(~sk_better).sum()})", rasterized=True)
ax.scatter(abs_sk[sk_better], abs_al[sk_better], s=2, alpha=0.3,
           c="tab:orange", label=f"SlakoNet better (N={sk_better.sum()})", rasterized=True)
ax.plot([0, 10], [0, 10], "k--", lw=1)
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_xlabel("|SlakoNet − PBE| (eV)")
ax.set_ylabel("|ALIGNN − PBE| (eV)")
ax.set_title("Per-structure error: ALIGNN vs SlakoNet")
ax.set_aspect("equal")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "head_to_head_error.png"), dpi=200)
plt.close(fig)


# ── 7. Gap distributions ────────────────────────────────────────────────
print("Generating gap distribution plot...", flush=True)
fig, ax = plt.subplots(figsize=(9, 5))
bins = np.linspace(0, 10, 200)
ax.hist(pbe[is_nonmetal], bins=bins, alpha=0.5, label=f"PBE (N={is_nonmetal.sum()})", density=True)
ax.hist(sk[is_nonmetal], bins=bins, alpha=0.5, label="SlakoNet", density=True)
ax.hist(al_clamped[is_nonmetal], bins=bins, alpha=0.5, label="ALIGNN", density=True)
ax.set_xlabel("Band gap (eV)")
ax.set_ylabel("Density")
ax.set_title(f"Gap distributions (non-metals, N={is_nonmetal.sum()})")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "gap_distribution_alignn.png"), dpi=200)
plt.close(fig)


print(f"\nAll plots saved to {OUT_DIR}/")
print("Done.")
