"""
Compare v03 (optb88vdw-ALIGNN) predictions against SlakoNet and PBE reference.

Reference sets:
  - results/alignn_predictions.json
      v03 predictions from the pretrained JARVIS jv_optb88vdw_bandgap_alignn
      model on the filtered Alexandria PBE 3D hull set (48,764 structures).
      Fields: mat_id, formula, band_gap_ind, band_gap_dir, e_form, alignn_bandgap
  - ../../results/sk_scalars.json
      SlakoNet sk_bandgap per mat_id, in the parent alignn_v03_alex/ dir.
      Extracted once from the 5.6GB all_results.json that the SlakoNet
      run produced. Same Alexandria filter, so mat_id keys line up.
  - ../../alignn_v1_pbe/results/alignn_predictions.json
      sibling alignn_v1_pbe predictions from the mp_gappbe_alignn
      (PBE-trained) model, included here for an apples-to-apples
      "PBE-ALIGNN vs optb-ALIGNN" comparison on the overlapping structures.

Outputs go to analysis/plots/ and analysis/summary.txt.
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

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT_DIR = os.path.join(HERE, "plots")
os.makedirs(OUT_DIR, exist_ok=True)

ALIGNN_OPT_FILE = os.path.join(ROOT, "results/alignn_predictions.json")
ALIGNN_PBE_FILE = os.path.join(ROOT, "..", "alignn_v1_pbe", "results", "alignn_predictions.json")
SK_SCALARS_FILE = os.path.join(ROOT, "..", "results", "sk_scalars.json")

METAL_THRESH = 0.1  # eV


def load_json(path):
    with open(path, "rb") as f:
        return orjson.loads(f.read())


def stats(name, pred, true):
    mae = mean_absolute_error(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
    if len(true) > 1 and np.std(true) > 0:
        r2 = r2_score(true, pred)
    else:
        r2 = float("nan")
    me = (pred - true).mean()
    maxerr = np.max(np.abs(pred - true))
    line = (f"  {name:40s}  N={len(true):6d}  MAE={mae:.4f}  "
            f"RMSE={rmse:.4f}  R2={r2:.4f}  ME={me:+.4f}  MaxErr={maxerr:.4f}")
    print(line)
    return {"name": name, "N": int(len(true)), "MAE": float(mae),
            "RMSE": float(rmse), "R2": float(r2), "ME": float(me),
            "MaxErr": float(maxerr)}


def density_scatter(ax, x, y, bins=200):
    h, xe, ye = np.histogram2d(x, y, bins=bins)
    xi = np.clip(np.digitize(x, xe) - 1, 0, bins - 1)
    yi = np.clip(np.digitize(y, ye) - 1, 0, bins - 1)
    c = h[xi, yi]
    order = np.argsort(c)
    return ax.scatter(x[order], y[order], c=c[order], s=2, cmap="viridis",
                      norm=LogNorm(vmin=1), rasterized=True)


# ── Load data ──────────────────────────────────────────────────────────────
print(f"Loading v03 (optb88vdw-ALIGNN) predictions from {ALIGNN_OPT_FILE}")
alignn_opt = load_json(ALIGNN_OPT_FILE)
print(f"  {len(alignn_opt)} entries")

print(f"Loading v00 (PBE-ALIGNN) predictions from {ALIGNN_PBE_FILE}")
alignn_pbe_raw = load_json(ALIGNN_PBE_FILE)
alignn_pbe_map = {d["mat_id"]: d["alignn_bandgap"] for d in alignn_pbe_raw}
print(f"  {len(alignn_pbe_map)} entries")

print(f"Loading SlakoNet scalars from {SK_SCALARS_FILE}")
sk_raw = load_json(SK_SCALARS_FILE)
sk_map = {d["mat_id"]: d["sk_bandgap"] for d in sk_raw}
print(f"  {len(sk_map)} entries")


# ── Merge triple-overlap set ───────────────────────────────────────────────
triple = []
sk_missing = 0
pbe_missing = 0
for entry in alignn_opt:
    mid = entry["mat_id"]
    sk = sk_map.get(mid)
    al_pbe = alignn_pbe_map.get(mid)
    if sk is None:
        sk_missing += 1
    if al_pbe is None:
        pbe_missing += 1
    if sk is None or al_pbe is None:
        continue
    triple.append({
        "mat_id": mid,
        "formula": entry["formula"],
        "pbe_ind": entry["band_gap_ind"],
        "pbe_dir": entry["band_gap_dir"],
        "e_form": entry["e_form"],
        "sk": sk,
        "alignn_opt": entry["alignn_bandgap"],
        "alignn_pbe": al_pbe,
    })

print(f"\nMerge summary:")
print(f"  v03 optb-ALIGNN entries              = {len(alignn_opt)}")
print(f"  v03 entries with no SK match         = {sk_missing}")
print(f"  v03 entries with no v00-ALIGNN match = {pbe_missing}")
print(f"  triple-paired (PBE, SK, both ALIGNN) = {len(triple)}")

pbe = np.array([d["pbe_ind"] for d in triple])
sk = np.array([d["sk"] for d in triple])
al_opt = np.array([d["alignn_opt"] for d in triple])
al_pbe = np.array([d["alignn_pbe"] for d in triple])

# Negative ALIGNN outputs clamped to 0 (physical lower bound on band gap)
al_opt_c = np.clip(al_opt, 0, None)
al_pbe_c = np.clip(al_pbe, 0, None)

is_metal = pbe == 0.0
is_nonmetal = ~is_metal

print(f"\n  PBE metals: {is_metal.sum()}   PBE non-metals: {is_nonmetal.sum()}")
print(f"  optb-ALIGNN negatives (clamped): {(al_opt < 0).sum()}")
print(f"  pbe-ALIGNN  negatives (clamped): {(al_pbe < 0).sum()}")


# ── Regression metrics vs PBE ──────────────────────────────────────────────
print("\n" + "=" * 92)
print("REGRESSION METRICS vs PBE indirect gap  (triple-paired subset)")
print("=" * 92)
print("\nAll structures:")
stats("SlakoNet",                     sk,         pbe)
stats("ALIGNN-pbe  (v00, clamped)",   al_pbe_c,   pbe)
stats("ALIGNN-optb (v03, clamped)",   al_opt_c,   pbe)

print("\nNon-metals only (PBE gap > 0):")
stats("SlakoNet",                     sk[is_nonmetal],       pbe[is_nonmetal])
stats("ALIGNN-pbe  (v00, clamped)",   al_pbe_c[is_nonmetal], pbe[is_nonmetal])
stats("ALIGNN-optb (v03, clamped)",   al_opt_c[is_nonmetal], pbe[is_nonmetal])

print("\nMetals only (PBE gap == 0):")
stats("SlakoNet",                     sk[is_metal],       pbe[is_metal])
stats("ALIGNN-pbe  (v00, clamped)",   al_pbe_c[is_metal], pbe[is_metal])
stats("ALIGNN-optb (v03, clamped)",   al_opt_c[is_metal], pbe[is_metal])


# ── Parity: SlakoNet / PBE-ALIGNN / optb-ALIGNN  vs PBE ────────────────────
print("\nGenerating 3-panel parity plot (all structures)...")
lim = 10.0
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
panels = [
    (pbe, sk,       "PBE indirect gap (eV)", "SlakoNet gap (eV)",         "SlakoNet vs PBE"),
    (pbe, al_pbe_c, "PBE indirect gap (eV)", "ALIGNN (mp_gappbe) (eV)",   "PBE-ALIGNN vs PBE"),
    (pbe, al_opt_c, "PBE indirect gap (eV)", "ALIGNN (optb88vdw) (eV)",   "optb-ALIGNN vs PBE"),
]
for ax, (x, y, xl, yl, title) in zip(axes, panels):
    sc = density_scatter(ax, x, y)
    ax.plot([0, lim], [0, lim], "r--", lw=1, label="y = x")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel(xl); ax.set_ylabel(yl)
    ax.set_title(title); ax.set_aspect("equal")
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


# ── Parity: non-metals only ────────────────────────────────────────────────
print("Generating 3-panel parity plot (non-metals only)...")
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
panels_nm = [
    (pbe[is_nonmetal], sk[is_nonmetal],       "PBE indirect gap (eV)", "SlakoNet gap (eV)",       "Non-metals: SlakoNet vs PBE"),
    (pbe[is_nonmetal], al_pbe_c[is_nonmetal], "PBE indirect gap (eV)", "ALIGNN (mp_gappbe) (eV)", "Non-metals: PBE-ALIGNN vs PBE"),
    (pbe[is_nonmetal], al_opt_c[is_nonmetal], "PBE indirect gap (eV)", "ALIGNN (optb88vdw) (eV)", "Non-metals: optb-ALIGNN vs PBE"),
]
for ax, (x, y, xl, yl, title) in zip(axes, panels_nm):
    sc = density_scatter(ax, x, y)
    ax.plot([0, lim], [0, lim], "r--", lw=1, label="y = x")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel(xl); ax.set_ylabel(yl)
    ax.set_title(title); ax.set_aspect("equal")
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


# ── Residual histograms ────────────────────────────────────────────────────
print("Generating residual histograms...")
res_sk     = sk - pbe
res_al_pbe = al_pbe_c - pbe
res_al_opt = al_opt_c - pbe

fig, axes = plt.subplots(1, 3, figsize=(20, 5))
for ax, res, label in zip(axes,
                          [res_sk, res_al_pbe, res_al_opt],
                          ["SlakoNet", "ALIGNN (PBE)", "ALIGNN (optb88vdw)"]):
    ax.hist(res, bins=300, alpha=0.8)
    ax.axvline(0, color="r", ls="--")
    ax.axvline(res.mean(), color="k", ls="-", lw=1.2,
               label=f"mean = {res.mean():+.3f} eV")
    ax.set_xlabel(f"{label} - PBE indirect gap (eV)")
    ax.set_ylabel("Count")
    ax.set_title(f"Residuals: {label} - PBE  (N={len(res)}, MAE={np.abs(res).mean():.3f})")
    ax.legend()
    ax.set_xlim(-6, 6)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "residuals_three_way.png"), dpi=200)
plt.close(fig)


# ── Classification (metal vs non-metal, 0.1 eV threshold) ──────────────────
print("\nClassification metrics (metal/non-metal, 0.1 eV threshold):")
pbe_cls    = (pbe > 0).astype(int)
sk_cls     = (sk > METAL_THRESH).astype(int)
al_pbe_cls = (al_pbe_c > METAL_THRESH).astype(int)
al_opt_cls = (al_opt_c > METAL_THRESH).astype(int)

cls_rows = []
for name, pred in [("SlakoNet", sk_cls),
                   ("ALIGNN-pbe",  al_pbe_cls),
                   ("ALIGNN-optb", al_opt_cls)]:
    cm = confusion_matrix(pbe_cls, pred)
    acc = np.trace(cm) / cm.sum()
    print(f"\n  {name}")
    print(f"    Accuracy: {acc:.4f}")
    print(f"    TN={cm[0,0]}  FP={cm[0,1]}  FN={cm[1,0]}  TP={cm[1,1]}")
    cls_rows.append((name, acc, cm))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (name, acc, cm) in zip(axes, cls_rows):
    ConfusionMatrixDisplay(cm, display_labels=["Metal", "Non-metal"]).plot(
        ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{name} (acc={acc:.3f})")
    ax.set_xlabel(f"{name} prediction")
    ax.set_ylabel("PBE reference")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "confusion_three_way.png"), dpi=200)
plt.close(fig)


# ── Head-to-head per structure ─────────────────────────────────────────────
print("\nHead-to-head errors:")
abs_sk     = np.abs(res_sk)
abs_al_pbe = np.abs(res_al_pbe)
abs_al_opt = np.abs(res_al_opt)

opt_vs_pbe_alignn_win = abs_al_opt < abs_al_pbe
print(f"  optb-ALIGNN closer to PBE than PBE-ALIGNN: "
      f"{opt_vs_pbe_alignn_win.sum()} ({opt_vs_pbe_alignn_win.mean()*100:.1f}%)")
opt_vs_sk_win = abs_al_opt < abs_sk
print(f"  optb-ALIGNN closer to PBE than SlakoNet:   "
      f"{opt_vs_sk_win.sum()} ({opt_vs_sk_win.mean()*100:.1f}%)")
pbe_alignn_vs_sk_win = abs_al_pbe < abs_sk
print(f"  PBE-ALIGNN  closer to PBE than SlakoNet:   "
      f"{pbe_alignn_vs_sk_win.sum()} ({pbe_alignn_vs_sk_win.mean()*100:.1f}%)")

fig, axes = plt.subplots(1, 2, figsize=(14, 7))
# Panel 1: optb-ALIGNN vs SlakoNet
ax = axes[0]
better = abs_al_opt < abs_sk
ax.scatter(abs_sk[~better], abs_al_opt[~better], s=2, alpha=0.3,
           c="tab:orange", label=f"SlakoNet better (N={(~better).sum()})", rasterized=True)
ax.scatter(abs_sk[better], abs_al_opt[better], s=2, alpha=0.3,
           c="tab:blue", label=f"optb-ALIGNN better (N={better.sum()})", rasterized=True)
ax.plot([0, 10], [0, 10], "k--", lw=1)
ax.set_xlim(0, 10); ax.set_ylim(0, 10)
ax.set_xlabel("|SlakoNet - PBE| (eV)")
ax.set_ylabel("|optb-ALIGNN - PBE| (eV)")
ax.set_title("Per-structure error: optb-ALIGNN vs SlakoNet")
ax.set_aspect("equal"); ax.legend()
# Panel 2: optb-ALIGNN vs PBE-ALIGNN
ax = axes[1]
better = abs_al_opt < abs_al_pbe
ax.scatter(abs_al_pbe[~better], abs_al_opt[~better], s=2, alpha=0.3,
           c="tab:orange", label=f"PBE-ALIGNN better (N={(~better).sum()})", rasterized=True)
ax.scatter(abs_al_pbe[better], abs_al_opt[better], s=2, alpha=0.3,
           c="tab:blue", label=f"optb-ALIGNN better (N={better.sum()})", rasterized=True)
ax.plot([0, 10], [0, 10], "k--", lw=1)
ax.set_xlim(0, 10); ax.set_ylim(0, 10)
ax.set_xlabel("|ALIGNN (PBE) - PBE| (eV)")
ax.set_ylabel("|ALIGNN (optb88vdw) - PBE| (eV)")
ax.set_title("Per-structure error: optb-ALIGNN vs PBE-ALIGNN")
ax.set_aspect("equal"); ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "head_to_head_error.png"), dpi=200)
plt.close(fig)


# ── Gap distributions (non-metals) ─────────────────────────────────────────
print("Generating gap distribution plot...")
fig, ax = plt.subplots(figsize=(10, 5))
bins = np.linspace(0, 10, 200)
ax.hist(pbe[is_nonmetal],      bins=bins, alpha=0.45, label=f"PBE (N={is_nonmetal.sum()})", density=True)
ax.hist(sk[is_nonmetal],       bins=bins, alpha=0.45, label="SlakoNet",                   density=True)
ax.hist(al_pbe_c[is_nonmetal], bins=bins, alpha=0.45, label="ALIGNN (mp_gappbe)",         density=True)
ax.hist(al_opt_c[is_nonmetal], bins=bins, alpha=0.45, label="ALIGNN (optb88vdw)",         density=True)
ax.set_xlabel("Band gap (eV)")
ax.set_ylabel("Density")
ax.set_title(f"Gap distributions (PBE non-metals, N={is_nonmetal.sum()})")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "gap_distribution.png"), dpi=200)
plt.close(fig)


# ── optb88vdw vs PBE systematic shift ──────────────────────────────────────
# optb88vdw is a vdW functional: gaps are typically similar to PBE but can differ
# by a small offset. Fit the slope/intercept on the non-metal set to characterize.
nm_x = pbe[is_nonmetal]
nm_y = al_opt_c[is_nonmetal]
slope, intercept = np.polyfit(nm_x, nm_y, 1)
print(f"\noptb-ALIGNN = {slope:.4f} * PBE + {intercept:+.4f}  (non-metals)")

fig, ax = plt.subplots(figsize=(7, 6))
sc = density_scatter(ax, nm_x, nm_y)
xx = np.linspace(0, 10, 2)
ax.plot(xx, xx, "r--", lw=1, label="y = x")
ax.plot(xx, slope * xx + intercept, "k-", lw=1.2,
        label=f"fit: y = {slope:.3f}x + {intercept:+.3f}")
ax.set_xlim(0, 10); ax.set_ylim(0, 10)
ax.set_xlabel("PBE indirect gap (eV)")
ax.set_ylabel("optb-ALIGNN gap (eV)")
ax.set_title("optb88vdw-ALIGNN systematic shift vs PBE (non-metals)")
ax.set_aspect("equal"); ax.legend(loc="lower right")
plt.colorbar(sc, ax=ax, label="Density")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "optb_vs_pbe_shift.png"), dpi=200)
plt.close(fig)


print(f"\nAll plots saved to {OUT_DIR}/")
print("Done.")
