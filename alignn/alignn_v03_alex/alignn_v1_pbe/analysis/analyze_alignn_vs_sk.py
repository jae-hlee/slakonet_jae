"""
Compare ALIGNN vs SlakoNet band-gap predictions against PBE reference.

ALIGNN predictions come from this directory (ran with the mp_gappbe_alignn model):
    results/alignn_predictions.json
        entry: {mat_id, formula, band_gap_ind, band_gap_dir, e_form, alignn_bandgap}

SlakoNet predictions come from the parent alignn_v03_alex/ dir (already produced there):
    ../../results/sk_scalars.json
        entry: {mat_id, sk_bandgap}

Merges on mat_id and evaluates both models against PBE `band_gap_ind`.

Outputs (analysis/):
  plots/parity_three_way.png         SK vs PBE | ALIGNN vs PBE | ALIGNN vs SK (all)
  plots/parity_three_way_nonmetals.png  same, non-metals only
  plots/residuals.png                 Residual histograms SK & ALIGNN
  plots/confusion_sk_vs_alignn.png    Metal/non-metal confusion, both models
  plots/head_to_head_error.png        Per-structure |err_SK| vs |err_ALIGNN|
  plots/gap_distribution.png          Non-metal gap distributions (PBE/SK/ALIGNN)
  plots/mae_vs_pbe_gap.png            MAE vs PBE gap bins, both models
  plots/error_vs_gap_hex_alignn.png   ALIGNN signed residual vs PBE gap
  plots/error_vs_gap_hex_sk.png       SlakoNet signed residual vs PBE gap
  metrics.json                        numeric summary
  worst_cases_alignn.csv              top-50 worst ALIGNN mispredictions
  worst_cases_sk.csv                  top-50 worst SlakoNet mispredictions
  analysis.md                         written summary
"""

import csv
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import orjson
from matplotlib.colors import LogNorm
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ALIGNN_FILE = os.path.join(ROOT, "results", "alignn_predictions.json")
SK_FILE = os.path.join(ROOT, "..", "results", "sk_scalars.json")

OUT_DIR = HERE
PLOT_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

METAL_THRESH = 0.1  # eV, predicted-gap threshold for "metal"
LIM = 10.0          # parity axis limit (eV)


# ── Load ALIGNN predictions ─────────────────────────────────────────────
print(f"Loading ALIGNN predictions from {ALIGNN_FILE}...", flush=True)
with open(ALIGNN_FILE, "rb") as f:
    alignn_data = orjson.loads(f.read())
print(f"  {len(alignn_data)} ALIGNN entries", flush=True)


# ── Load SlakoNet scalars (cached in v00) ──────────────────────────────
print(f"Loading SlakoNet scalars from {SK_FILE}...", flush=True)
with open(SK_FILE, "rb") as f:
    sk_scalars = orjson.loads(f.read())
print(f"  {len(sk_scalars)} SlakoNet entries", flush=True)
sk_map = {d["mat_id"]: d["sk_bandgap"] for d in sk_scalars}


# ── Merge on mat_id ────────────────────────────────────────────────────
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
print(f"  Merged {len(merged)} entries "
      f"(ALIGNN-only: {missing_sk}, SK-only: {len(sk_scalars) - len(merged)})")

mat_ids = [d["mat_id"] for d in merged]
formulas = [d["formula"] for d in merged]
pbe = np.array([d["pbe_ind"] for d in merged], dtype=float)
pbe_dir = np.array([d["pbe_dir"] for d in merged], dtype=float)
eform = np.array([d["e_form"] for d in merged], dtype=float)
sk = np.array([d["sk"] for d in merged], dtype=float)
al_raw = np.array([d["alignn"] for d in merged], dtype=float)

# Clamp negative ALIGNN predictions to 0 (physical lower bound)
al = np.clip(al_raw, 0, None)
n_neg = int((al_raw < 0).sum())

is_metal = pbe == 0.0
is_nonmetal = ~is_metal
N = len(merged)
print(f"  PBE metals: {is_metal.sum()}   non-metals: {is_nonmetal.sum()}")
print(f"  ALIGNN negative raw preds clamped: {n_neg}")


# ── Helpers ─────────────────────────────────────────────────────────────
def reg_stats(pred, true):
    if len(true) == 0:
        return {"N": 0, "MAE": float("nan"), "RMSE": float("nan"),
                "R2": float("nan"), "ME": float("nan"), "MaxErr": float("nan")}
    mae = float(mean_absolute_error(true, pred))
    rmse = float(np.sqrt(mean_squared_error(true, pred)))
    r2 = float(r2_score(true, pred)) if len(true) > 1 and np.std(true) > 0 else float("nan")
    me = float((pred - true).mean())
    maxerr = float(np.max(np.abs(pred - true)))
    return {"N": int(len(true)), "MAE": mae, "RMSE": rmse, "R2": r2, "ME": me, "MaxErr": maxerr}


def fmt(name, s):
    return (f"  {name:32s}  N={s['N']:6d}  MAE={s['MAE']:.4f}  "
            f"RMSE={s['RMSE']:.4f}  R²={s['R2']:.4f}  ME={s['ME']:+.4f}  MaxErr={s['MaxErr']:.4f}")


def density_scatter(ax, x, y, bins=200):
    h, xe, ye = np.histogram2d(x, y, bins=bins)
    xi = np.clip(np.digitize(x, xe) - 1, 0, bins - 1)
    yi = np.clip(np.digitize(y, ye) - 1, 0, bins - 1)
    c = h[xi, yi]
    order = np.argsort(c)
    return ax.scatter(x[order], y[order], c=c[order], s=2, cmap="viridis",
                      norm=LogNorm(vmin=1), rasterized=True)


def parity_panel(ax, x, y, xl, yl, title, lim=LIM):
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
            bbox=dict(boxstyle="round", fc="white", alpha=0.85))


# ── Regression metrics ─────────────────────────────────────────────────
metrics = {
    "dataset": {
        "N_merged": N,
        "N_alignn_only_unmatched": missing_sk,
        "N_sk_only_unmatched": len(sk_scalars) - N,
        "N_metals_pbe": int(is_metal.sum()),
        "N_nonmetals_pbe": int(is_nonmetal.sum()),
        "N_alignn_negative_clamped": n_neg,
        "metal_threshold_eV": METAL_THRESH,
    },
    "vs_pbe_indirect": {
        "sk_all": reg_stats(sk, pbe),
        "alignn_raw_all": reg_stats(al_raw, pbe),
        "alignn_clamped_all": reg_stats(al, pbe),
        "sk_nonmetals": reg_stats(sk[is_nonmetal], pbe[is_nonmetal]),
        "alignn_clamped_nonmetals": reg_stats(al[is_nonmetal], pbe[is_nonmetal]),
        "sk_metals": reg_stats(sk[is_metal], pbe[is_metal]),
        "alignn_clamped_metals": reg_stats(al[is_metal], pbe[is_metal]),
    },
}

print("\n" + "=" * 90)
print("REGRESSION METRICS vs PBE indirect gap")
print("=" * 90)
print("All structures:")
print(fmt("SlakoNet",              metrics["vs_pbe_indirect"]["sk_all"]))
print(fmt("ALIGNN (raw)",          metrics["vs_pbe_indirect"]["alignn_raw_all"]))
print(fmt("ALIGNN (clamped ≥ 0)",  metrics["vs_pbe_indirect"]["alignn_clamped_all"]))
print("\nNon-metals (PBE gap > 0):")
print(fmt("SlakoNet",              metrics["vs_pbe_indirect"]["sk_nonmetals"]))
print(fmt("ALIGNN (clamped ≥ 0)",  metrics["vs_pbe_indirect"]["alignn_clamped_nonmetals"]))
print("\nMetals (PBE gap == 0):")
print(fmt("SlakoNet",              metrics["vs_pbe_indirect"]["sk_metals"]))
print(fmt("ALIGNN (clamped ≥ 0)",  metrics["vs_pbe_indirect"]["alignn_clamped_metals"]))


# ── Parity plots (three-way) ──────────────────────────────────────────
print("\nGenerating 3-way parity plots...", flush=True)
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
parity_panel(axes[0], pbe, sk, "PBE indirect gap (eV)", "SlakoNet gap (eV)",
             f"SlakoNet vs PBE (N={N})")
parity_panel(axes[1], pbe, al, "PBE indirect gap (eV)", "ALIGNN gap (eV)",
             f"ALIGNN vs PBE (N={N})")
parity_panel(axes[2], sk, al, "SlakoNet gap (eV)", "ALIGNN gap (eV)",
             f"ALIGNN vs SlakoNet (N={N})")
fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "parity_three_way.png"), dpi=200)
plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
parity_panel(axes[0], pbe[is_nonmetal], sk[is_nonmetal],
             "PBE indirect gap (eV)", "SlakoNet gap (eV)",
             f"Non-metals: SlakoNet vs PBE (N={is_nonmetal.sum()})")
parity_panel(axes[1], pbe[is_nonmetal], al[is_nonmetal],
             "PBE indirect gap (eV)", "ALIGNN gap (eV)",
             f"Non-metals: ALIGNN vs PBE (N={is_nonmetal.sum()})")
parity_panel(axes[2], sk[is_nonmetal], al[is_nonmetal],
             "SlakoNet gap (eV)", "ALIGNN gap (eV)",
             f"Non-metals: ALIGNN vs SlakoNet (N={is_nonmetal.sum()})")
fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "parity_three_way_nonmetals.png"), dpi=200)
plt.close(fig)


# ── Residual histograms ───────────────────────────────────────────────
print("Generating residual histograms...", flush=True)
res_sk = sk - pbe
res_al = al - pbe

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, res, name in zip(axes, [res_sk, res_al], ["SlakoNet", "ALIGNN"]):
    ax.hist(res, bins=300, alpha=0.8)
    ax.axvline(0, color="r", ls="--")
    ax.axvline(float(res.mean()), color="k", ls="-", lw=1.2,
               label=f"mean = {res.mean():+.3f} eV")
    ax.set_xlabel(f"{name} − PBE indirect (eV)")
    ax.set_ylabel("Count")
    ax.set_title(f"{name} − PBE  (N={len(res)}, MAE={np.abs(res).mean():.3f})")
    ax.legend()
    ax.set_xlim(-6, 6)
fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "residuals.png"), dpi=200)
plt.close(fig)


# ── Metal/non-metal classification, both models ──────────────────────
print("Computing classification metrics...", flush=True)
pbe_cls = (pbe > 0).astype(int)
sk_cls = (sk > METAL_THRESH).astype(int)
al_cls = (al > METAL_THRESH).astype(int)

metrics["classification"] = {"threshold_eV": METAL_THRESH}
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, pred, name in zip(axes, [sk_cls, al_cls], ["SlakoNet", "ALIGNN"]):
    cm = confusion_matrix(pbe_cls, pred)
    TN, FP, FN, TP = map(int, (cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]))
    acc = (TP + TN) / cm.sum()
    prec = TP / (TP + FP) if (TP + FP) else float("nan")
    rec = TP / (TP + FN) if (TP + FN) else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else float("nan")
    metrics["classification"][name.lower()] = {
        "TN": TN, "FP": FP, "FN": FN, "TP": TP,
        "accuracy": float(acc),
        "precision_nonmetal": float(prec),
        "recall_nonmetal": float(rec),
        "f1_nonmetal": float(f1),
    }
    print(f"  {name}: acc={acc:.4f}  TN={TN} FP={FP} FN={FN} TP={TP}  "
          f"prec={prec:.4f} rec={rec:.4f} F1={f1:.4f}")
    ConfusionMatrixDisplay(cm, display_labels=["Metal", "Non-metal"]).plot(
        ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{name} (acc={acc:.3f})")
    ax.set_xlabel(f"{name} prediction")
    ax.set_ylabel("PBE reference")
fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "confusion_sk_vs_alignn.png"), dpi=200)
plt.close(fig)


# ── Head-to-head per-structure error ─────────────────────────────────
print("Generating head-to-head error plot...", flush=True)
abs_sk = np.abs(res_sk)
abs_al = np.abs(res_al)
sk_better = abs_sk < abs_al
n_sk_better = int(sk_better.sum())
n_al_better = int((~sk_better).sum())
print(f"  SlakoNet closer to PBE: {n_sk_better} ({100*n_sk_better/N:.1f}%)")
print(f"  ALIGNN closer to PBE:   {n_al_better} ({100*n_al_better/N:.1f}%)")
metrics["head_to_head"] = {
    "N_total": N,
    "N_sk_better": n_sk_better,
    "N_alignn_better": n_al_better,
    "frac_sk_better": n_sk_better / N,
    "frac_alignn_better": n_al_better / N,
}

fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(abs_sk[~sk_better], abs_al[~sk_better], s=2, alpha=0.3,
           c="tab:blue", label=f"ALIGNN better (N={n_al_better})", rasterized=True)
ax.scatter(abs_sk[sk_better], abs_al[sk_better], s=2, alpha=0.3,
           c="tab:orange", label=f"SlakoNet better (N={n_sk_better})", rasterized=True)
ax.plot([0, 10], [0, 10], "k--", lw=1)
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_xlabel("|SlakoNet − PBE| (eV)")
ax.set_ylabel("|ALIGNN − PBE| (eV)")
ax.set_title("Per-structure error: ALIGNN vs SlakoNet")
ax.set_aspect("equal")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "head_to_head_error.png"), dpi=200)
plt.close(fig)


# ── Gap distributions (non-metals) ───────────────────────────────────
print("Generating gap distribution plot...", flush=True)
fig, ax = plt.subplots(figsize=(9, 5))
bins = np.linspace(0, 10, 200)
ax.hist(pbe[is_nonmetal], bins=bins, alpha=0.5, label=f"PBE (N={is_nonmetal.sum()})", density=True)
ax.hist(sk[is_nonmetal], bins=bins, alpha=0.5, label="SlakoNet", density=True)
ax.hist(al[is_nonmetal], bins=bins, alpha=0.5, label="ALIGNN", density=True)
ax.set_xlabel("Band gap (eV)")
ax.set_ylabel("Density")
ax.set_title(f"Gap distributions (non-metals, N={is_nonmetal.sum()})")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "gap_distribution.png"), dpi=200)
plt.close(fig)


# ── MAE binned by PBE gap (non-metals), both models ─────────────────
print("Computing MAE vs PBE-gap bins...", flush=True)
edges = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0])
centers = 0.5 * (edges[:-1] + edges[1:])
bin_sk, bin_al = [], []
for lo, hi in zip(edges[:-1], edges[1:]):
    mask = (pbe >= lo) & (pbe < hi) & is_nonmetal
    n = int(mask.sum())
    if n == 0:
        bin_sk.append({"lo": float(lo), "hi": float(hi), "N": 0, "MAE": float("nan"), "ME": float("nan")})
        bin_al.append({"lo": float(lo), "hi": float(hi), "N": 0, "MAE": float("nan"), "ME": float("nan")})
        continue
    r_sk = sk[mask] - pbe[mask]
    r_al = al[mask] - pbe[mask]
    bin_sk.append({"lo": float(lo), "hi": float(hi), "N": n,
                   "MAE": float(np.abs(r_sk).mean()), "ME": float(r_sk.mean())})
    bin_al.append({"lo": float(lo), "hi": float(hi), "N": n,
                   "MAE": float(np.abs(r_al).mean()), "ME": float(r_al.mean())})
metrics["mae_by_pbe_gap_nonmetals"] = {"sk": bin_sk, "alignn": bin_al}

fig, ax = plt.subplots(figsize=(10, 5.5))
widths = (edges[1:] - edges[:-1]) * 0.4
offset = widths / 2
ax.bar(centers - offset, [b["MAE"] for b in bin_sk], width=widths, alpha=0.8,
       color="tab:orange", label="SlakoNet MAE")
ax.bar(centers + offset, [b["MAE"] for b in bin_al], width=widths, alpha=0.8,
       color="tab:blue", label="ALIGNN MAE")
ax.plot(centers, [b["ME"] for b in bin_sk], "o--", color="tab:red",
        label="SlakoNet mean signed err", lw=1.3)
ax.plot(centers, [b["ME"] for b in bin_al], "s--", color="tab:green",
        label="ALIGNN mean signed err", lw=1.3)
ax.axhline(0, color="k", lw=0.6)
for x, b in zip(centers, bin_sk):
    ax.text(x, 0.01, f"N={b['N']}", ha="center", va="bottom", fontsize=8,
            transform=ax.get_xaxis_transform())
ax.set_xlabel("PBE indirect gap (eV)")
ax.set_ylabel("Error (eV)")
ax.set_title("Error vs PBE gap (non-metals) — SlakoNet vs ALIGNN")
ax.legend(loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "mae_vs_pbe_gap.png"), dpi=200)
plt.close(fig)


# ── Hex signed-residual vs PBE gap, per model ────────────────────────
print("Generating hex residual plots...", flush=True)
for name, residual, fname in [
    ("SlakoNet", res_sk, "error_vs_gap_hex_sk.png"),
    ("ALIGNN", res_al, "error_vs_gap_hex_alignn.png"),
]:
    fig, ax = plt.subplots(figsize=(9, 6))
    hb = ax.hexbin(pbe[is_nonmetal], residual[is_nonmetal],
                   gridsize=80, bins="log", cmap="viridis", mincnt=1,
                   extent=(0, LIM, -6, 6))
    ax.axhline(0, color="r", ls="--", lw=1)
    ax.set_xlabel("PBE indirect gap (eV)")
    ax.set_ylabel(f"{name} − PBE (eV)")
    ax.set_title(f"{name} signed residual vs PBE gap (non-metals)")
    plt.colorbar(hb, ax=ax, label="log10(count)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, fname), dpi=200)
    plt.close(fig)


# ── Worst cases, per model ───────────────────────────────────────────
def write_worst(residual, fname):
    order = np.argsort(-np.abs(residual))[:50]
    rows = []
    for i in order:
        rows.append({
            "mat_id": mat_ids[i],
            "formula": formulas[i],
            "pbe_ind": float(pbe[i]),
            "pbe_dir": float(pbe_dir[i]),
            "sk": float(sk[i]),
            "alignn": float(al[i]),
            "alignn_raw": float(al_raw[i]),
            "e_form": float(eform[i]),
            "abs_err": float(abs(residual[i])),
            "signed_err": float(residual[i]),
        })
    with open(os.path.join(OUT_DIR, fname), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

write_worst(res_al, "worst_cases_alignn.csv")
write_worst(res_sk, "worst_cases_sk.csv")


# ── Save numeric metrics ─────────────────────────────────────────────
with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)


# ── Written summary ──────────────────────────────────────────────────
all_sk = metrics["vs_pbe_indirect"]["sk_all"]
all_al = metrics["vs_pbe_indirect"]["alignn_clamped_all"]
nm_sk = metrics["vs_pbe_indirect"]["sk_nonmetals"]
nm_al = metrics["vs_pbe_indirect"]["alignn_clamped_nonmetals"]
m_sk = metrics["vs_pbe_indirect"]["sk_metals"]
m_al = metrics["vs_pbe_indirect"]["alignn_clamped_metals"]
cls_sk = metrics["classification"]["slakonet"]
cls_al = metrics["classification"]["alignn"]
h2h = metrics["head_to_head"]

summary = f"""# ALIGNN vs SlakoNet band-gap predictions — analysis (alignn_v1_pbe)

**ALIGNN:** `mp_gappbe_alignn` (pretrained PBE-gap model) run in this directory.
**SlakoNet:** predictions reused from `../../results/sk_scalars.json`.
**Reference:** Alexandria PBE `band_gap_ind` (indirect gap) on `e_above_hull==0`,
elements with Z≤65. Materials are matched by `mat_id`; ALIGNN entries without a
SlakoNet counterpart are excluded from this comparison.

## Dataset composition (merged set)

- Merged structures: **{N}**
- ALIGNN entries without SK match (excluded): **{missing_sk}**
- SK entries without ALIGNN match (excluded): **{len(sk_scalars) - N}**
- PBE metals (gap = 0): **{int(is_metal.sum())}**
- PBE non-metals (gap > 0): **{int(is_nonmetal.sum())}**
- ALIGNN raw predictions < 0 (clamped to 0): **{n_neg}**

## Regression metrics vs PBE indirect gap

### All structures (N = {N})

| Model | MAE (eV) | RMSE (eV) | R² | Mean signed err |
|-------|---------:|----------:|---:|----------------:|
| SlakoNet            | {all_sk['MAE']:.4f} | {all_sk['RMSE']:.4f} | {all_sk['R2']:.4f} | {all_sk['ME']:+.4f} |
| ALIGNN (clamped ≥0) | {all_al['MAE']:.4f} | {all_al['RMSE']:.4f} | {all_al['R2']:.4f} | {all_al['ME']:+.4f} |

### Non-metals only (N = {nm_sk['N']})

| Model | MAE (eV) | RMSE (eV) | R² | Mean signed err |
|-------|---------:|----------:|---:|----------------:|
| SlakoNet            | {nm_sk['MAE']:.4f} | {nm_sk['RMSE']:.4f} | {nm_sk['R2']:.4f} | {nm_sk['ME']:+.4f} |
| ALIGNN (clamped ≥0) | {nm_al['MAE']:.4f} | {nm_al['RMSE']:.4f} | {nm_al['R2']:.4f} | {nm_al['ME']:+.4f} |

### Metals only (N = {m_sk['N']})

| Model | MAE (eV) | RMSE (eV) | Mean signed err |
|-------|---------:|----------:|----------------:|
| SlakoNet            | {m_sk['MAE']:.4f} | {m_sk['RMSE']:.4f} | {m_sk['ME']:+.4f} |
| ALIGNN (clamped ≥0) | {m_al['MAE']:.4f} | {m_al['RMSE']:.4f} | {m_al['ME']:+.4f} |

## Metal / non-metal classification (threshold {METAL_THRESH} eV)

| Model    | TN | FP | FN | TP | Accuracy | Precision | Recall | F1 |
|----------|---:|---:|---:|---:|---------:|----------:|-------:|---:|
| SlakoNet | {cls_sk['TN']} | {cls_sk['FP']} | {cls_sk['FN']} | {cls_sk['TP']} | {cls_sk['accuracy']:.4f} | {cls_sk['precision_nonmetal']:.4f} | {cls_sk['recall_nonmetal']:.4f} | {cls_sk['f1_nonmetal']:.4f} |
| ALIGNN   | {cls_al['TN']} | {cls_al['FP']} | {cls_al['FN']} | {cls_al['TP']} | {cls_al['accuracy']:.4f} | {cls_al['precision_nonmetal']:.4f} | {cls_al['recall_nonmetal']:.4f} | {cls_al['f1_nonmetal']:.4f} |

## Head-to-head (per structure)

- **SlakoNet closer to PBE:** {h2h['N_sk_better']} ({100*h2h['frac_sk_better']:.1f}%)
- **ALIGNN closer to PBE:**   {h2h['N_alignn_better']} ({100*h2h['frac_alignn_better']:.1f}%)

## Error vs PBE gap (non-metals, binned)

| PBE bin (eV) | N | SK MAE | SK ME | ALIGNN MAE | ALIGNN ME |
|---|--:|--:|--:|--:|--:|
"""
for bs, ba in zip(bin_sk, bin_al):
    if bs["N"] == 0:
        continue
    summary += (f"| {bs['lo']:.1f}–{bs['hi']:.1f} | {bs['N']} | "
                f"{bs['MAE']:.3f} | {bs['ME']:+.3f} | "
                f"{ba['MAE']:.3f} | {ba['ME']:+.3f} |\n")

summary += f"""
## Plots

| File | Content |
|------|---------|
| `plots/parity_three_way.png` | Density parity: SK vs PBE, ALIGNN vs PBE, ALIGNN vs SK (all) |
| `plots/parity_three_way_nonmetals.png` | Same three panels, non-metals only |
| `plots/residuals.png` | Residual histograms for SlakoNet and ALIGNN |
| `plots/confusion_sk_vs_alignn.png` | Metal/non-metal confusion matrices, both models |
| `plots/head_to_head_error.png` | Per-structure \\|SK−PBE\\| vs \\|ALIGNN−PBE\\| |
| `plots/gap_distribution.png` | Non-metal gap distributions: PBE / SK / ALIGNN |
| `plots/mae_vs_pbe_gap.png` | MAE & mean signed error binned by PBE gap |
| `plots/error_vs_gap_hex_sk.png` | SlakoNet signed residual vs PBE gap |
| `plots/error_vs_gap_hex_alignn.png` | ALIGNN signed residual vs PBE gap |

## Artifacts

- `metrics.json` — all numeric results used above
- `worst_cases_alignn.csv` — top-50 worst ALIGNN mispredictions
- `worst_cases_sk.csv` — top-50 worst SlakoNet mispredictions
"""

with open(os.path.join(OUT_DIR, "analysis.md"), "w") as f:
    f.write(summary)

print(f"\nAll outputs written to {OUT_DIR}/")
print("Done.")
