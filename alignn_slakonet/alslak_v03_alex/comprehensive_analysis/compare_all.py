"""Comprehensive comparison of three ALIGNN checkpoints vs SlakoNet on the
filtered Alexandria 3D PBE hull set.

Models merged here:
  - SlakoNet  (slako_v03/results/sk_scalars.json)
  - ALIGNN mp_gappbe_alignn      (alslak_v1_pbe/results/alignn_predictions.json)
  - ALIGNN jv_mbj_bandgap_alignn (alslak_v2_mbj/results/alignn_predictions.json)
  - ALIGNN jv_optb88vdw_bandgap_alignn (alslak_v3_opt/results/alignn_predictions.json)

Reference: Alexandria PBE indirect gap (band_gap_ind) on e_above_hull == 0.
"""

import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "comprehensive_analysis"
PLOTS = OUT / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

V00_SK = ROOT / "slako_v03/results/sk_scalars.json"
V01_AL = ROOT / "alslak_v1_pbe/results/alignn_predictions.json"
V02_AL = ROOT / "alslak_v2_mbj/results/alignn_predictions.json"
V03_AL = ROOT / "alslak_v3_opt/results/alignn_predictions.json"

MODELS = ["slakonet", "alignn_pbe", "alignn_mbj", "alignn_optb"]
LABELS = {
    "slakonet":   "SlakoNet (DFTB)",
    "alignn_pbe": "ALIGNN mp_gappbe (PBE)",
    "alignn_mbj": "ALIGNN jv_mbj (TB-mBJ)",
    "alignn_optb":"ALIGNN jv_optb88vdw (OptB88vdW)",
}
SHORT_LABELS = {
    "slakonet":   "SlakoNet",
    "alignn_pbe": "ALIGNN PBE",
    "alignn_mbj": "ALIGNN TB-mBJ",
    "alignn_optb":"ALIGNN OptB88vdW",
}
COLORS = {
    "slakonet":   "#e45756",
    "alignn_pbe": "#4c78a8",
    "alignn_mbj": "#54a24b",
    "alignn_optb":"#b279a2",
}
METAL_THR = 0.1

# -----------------------------------------------------------------------------
# Load & merge
# -----------------------------------------------------------------------------

def load_json(p: Path):
    with p.open() as f:
        return json.load(f)

def build_merged():
    sk   = {r["mat_id"]: r["sk_bandgap"]       for r in load_json(V00_SK)}
    pbe  = {r["mat_id"]: r                     for r in load_json(V01_AL)}
    mbj  = {r["mat_id"]: r["alignn_bandgap"]   for r in load_json(V02_AL)}
    optb = {r["mat_id"]: r["alignn_bandgap"]   for r in load_json(V03_AL)}

    merged = []
    for mid, sk_val in sk.items():
        rec_pbe = pbe.get(mid)
        if rec_pbe is None:
            continue
        merged.append({
            "mat_id": mid,
            "formula": rec_pbe["formula"],
            "pbe_ref": rec_pbe["band_gap_ind"],
            "pbe_dir": rec_pbe["band_gap_dir"],
            "e_form":  rec_pbe["e_form"],
            "slakonet":    sk_val,
            "alignn_pbe":  rec_pbe["alignn_bandgap"],
            "alignn_mbj":  mbj.get(mid),
            "alignn_optb": optb.get(mid),
        })
    return merged


def to_arrays(merged):
    pbe  = np.array([r["pbe_ref"]      for r in merged], dtype=float)
    sk   = np.array([r["slakonet"]     for r in merged], dtype=float)
    apb  = np.array([r["alignn_pbe"]   for r in merged], dtype=float)
    amb  = np.array([r["alignn_mbj"]   for r in merged], dtype=float)
    aop  = np.array([r["alignn_optb"]  for r in merged], dtype=float)
    return pbe, sk, apb, amb, aop


def clamp0(x):
    return np.maximum(x, 0.0)


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def regression_stats(pred, ref):
    err = pred - ref
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((ref - ref.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    me = float(np.mean(err))
    return {
        "N": int(len(pred)),
        "MAE": mae, "RMSE": rmse, "R2": r2,
        "ME": me, "MaxErr": float(np.max(np.abs(err))),
    }


def classification_stats(pred, ref, thr=METAL_THR):
    # Match v00/v01/v02/v03 convention: truth non-metal is strictly pbe > 0
    # (Alexandria stores metals as gap = 0 exactly), predicted non-metal uses
    # the threshold `thr`.
    pred_gap  = pred >= thr
    truth_gap = ref  > 0
    tp = int(np.sum(pred_gap &  truth_gap))
    tn = int(np.sum(~pred_gap & ~truth_gap))
    fp = int(np.sum(pred_gap & ~truth_gap))
    fn = int(np.sum(~pred_gap &  truth_gap))
    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec  = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else float("nan")
    return {"TN": tn, "FP": fp, "FN": fn, "TP": tp,
            "accuracy": acc, "precision_nonmetal": prec,
            "recall_nonmetal": rec, "f1_nonmetal": f1}


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------

def _hex_parity(ax, x, y, title, xlim, ylim, xlabel, ylabel):
    hb = ax.hexbin(x, y, gridsize=70, cmap="viridis", mincnt=1,
                   norm=LogNorm(vmin=1, vmax=max(2, len(x))),
                   extent=(xlim[0], xlim[1], ylim[0], ylim[1]))
    lo = min(xlim[0], ylim[0])
    hi = max(xlim[1], ylim[1])
    ax.plot([lo, hi], [lo, hi], "w--", lw=1)
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(title)
    return hb


def plot_parity_grid(pbe, preds_clamped, keys, fname, nonmetal_mask=None, tag=""):
    mask = nonmetal_mask if nonmetal_mask is not None else slice(None)
    xref = pbe[mask]
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.2), constrained_layout=True)
    xmax = float(np.quantile(xref, 0.999)) if len(xref) else 8.0
    for ax, key in zip(axes, keys):
        y = preds_clamped[key][mask]
        ymax = float(np.quantile(y, 0.999)) if len(y) else xmax
        hi = max(xmax, ymax, 6.0)
        hb = _hex_parity(ax, xref, y, LABELS[key],
                         (0, hi), (0, hi),
                         "PBE indirect gap (eV)",
                         f"{LABELS[key]} (eV)")
    cb = fig.colorbar(hb, ax=axes.ravel().tolist(), label="count (log)", shrink=0.85)
    fig.suptitle(f"Parity: predictions vs PBE indirect gap{tag}", fontsize=14)
    fig.savefig(PLOTS / fname, dpi=140)
    plt.close(fig)


def plot_residuals(pbe, preds_clamped, keys, fname, tag=""):
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5), constrained_layout=True, sharey=True)
    bins = np.linspace(-5, 5, 101)
    for ax, key in zip(axes, keys):
        err = preds_clamped[key] - pbe
        ax.hist(err, bins=bins, color=COLORS[key], alpha=0.85, edgecolor="black", linewidth=0.3)
        ax.axvline(0, color="k", lw=0.6)
        ax.axvline(float(np.mean(err)), color="crimson", lw=1, ls="--",
                   label=f"mean={np.mean(err):+.3f}")
        ax.set_title(LABELS[key])
        ax.set_xlabel("prediction − PBE (eV)")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("count")
    fig.suptitle(f"Residual distributions vs PBE{tag}", fontsize=14)
    fig.savefig(PLOTS / fname, dpi=140)
    plt.close(fig)


def plot_confusion(pbe, preds_clamped, keys, fname):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), constrained_layout=True)
    for ax, key in zip(axes, keys):
        c = classification_stats(preds_clamped[key], pbe)
        mat = np.array([[c["TN"], c["FP"]], [c["FN"], c["TP"]]])
        im = ax.imshow(mat, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(mat[i, j]), ha="center", va="center",
                        color="white" if mat[i, j] > mat.max() / 2 else "black",
                        fontsize=11)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["metal", "non-metal"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["metal", "non-metal"])
        ax.set_xlabel("predicted"); ax.set_ylabel("PBE truth")
        ax.set_title(f"{LABELS[key]}\nacc={c['accuracy']:.3f}")
    fig.suptitle(f"Metal / non-metal confusion (threshold {METAL_THR} eV vs PBE)", fontsize=13)
    fig.savefig(PLOTS / fname, dpi=140)
    plt.close(fig)


def plot_mae_bins(pbe, preds_clamped, keys, fname):
    mask = pbe > 0
    bins = [(0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 2.5),
            (2.5, 3.0), (3.0, 4.0), (4.0, 5.0), (5.0, 7.0), (7.0, 10.0)]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5), constrained_layout=True)
    width = 0.2
    x = np.arange(len(bins))
    for i, key in enumerate(keys):
        y_pred = preds_clamped[key][mask]
        y_ref  = pbe[mask]
        mae = []
        me  = []
        for lo, hi in bins:
            m = (y_ref >= lo) & (y_ref < hi)
            if m.sum():
                err = y_pred[m] - y_ref[m]
                mae.append(np.mean(np.abs(err)))
                me.append(np.mean(err))
            else:
                mae.append(np.nan); me.append(np.nan)
        offset = (i - 1.5) * width
        ax1.bar(x + offset, mae, width, label=LABELS[key], color=COLORS[key])
        ax2.bar(x + offset, me,  width, label=LABELS[key], color=COLORS[key])
    for ax in (ax1, ax2):
        ax.set_xticks(x)
        ax.set_xticklabels([f"{lo}-{hi}" for lo, hi in bins], rotation=30)
        ax.set_xlabel("PBE gap bin (eV)")
        ax.legend(fontsize=9)
        ax.axhline(0, color="k", lw=0.5)
    ax1.set_ylabel("MAE (eV)")
    ax2.set_ylabel("mean signed error (eV)")
    ax1.set_title("Non-metal MAE by PBE gap bin")
    ax2.set_title("Non-metal signed error by PBE gap bin")
    fig.savefig(PLOTS / fname, dpi=140)
    plt.close(fig)


def plot_gap_distribution(pbe, preds_clamped, keys, fname):
    mask = pbe > 0
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    bins = np.linspace(0, 10, 81)
    ax.hist(pbe[mask], bins=bins, histtype="step", lw=1.8, color="black",
            label=f"PBE reference (N={mask.sum()})", density=True)
    for key in keys:
        ax.hist(preds_clamped[key][mask], bins=bins, histtype="step", lw=1.5,
                color=COLORS[key], label=LABELS[key], density=True)
    ax.set_xlabel("band gap (eV)")
    ax.set_ylabel("density")
    ax.set_xlim(0, 10)
    ax.set_title("Non-metal gap distribution: PBE vs four predictors")
    ax.legend(fontsize=9)
    fig.savefig(PLOTS / fname, dpi=140)
    plt.close(fig)


def plot_head_to_head(pbe, preds_clamped, keys, fname):
    ref_key = "alignn_pbe"
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    others = [k for k in keys if k != ref_key]
    for ax, other in zip(axes, others):
        a = np.abs(preds_clamped[ref_key] - pbe)
        b = np.abs(preds_clamped[other]   - pbe)
        hi = float(np.quantile(np.concatenate([a, b]), 0.995))
        hi = max(hi, 2.0)
        hb = ax.hexbin(a, b, gridsize=70, cmap="magma", mincnt=1,
                       norm=LogNorm(vmin=1, vmax=max(2, len(a))),
                       extent=(0, hi, 0, hi))
        ax.plot([0, hi], [0, hi], "w--", lw=0.8)
        wins_ref   = int(np.sum(a < b))
        wins_other = int(np.sum(b < a))
        ax.set_title(f"{SHORT_LABELS[other]}  vs  {SHORT_LABELS[ref_key]}\n"
                     f"ref-better: {wins_ref}   other-better: {wins_other}",
                     fontsize=10)
        ax.set_xlabel(f"|{LABELS[ref_key]} − PBE|")
        ax.set_ylabel(f"|{LABELS[other]} − PBE|")
        ax.set_xlim(0, hi); ax.set_ylim(0, hi)
    fig.savefig(PLOTS / fname, dpi=140)
    plt.close(fig)


def plot_pairwise_model_agreement(preds_clamped, keys, fname):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    pairs = [("alignn_pbe", "alignn_mbj"),
             ("alignn_pbe", "alignn_optb"),
             ("alignn_mbj", "alignn_optb")]
    for ax, (a, b) in zip(axes, pairs):
        xa = preds_clamped[a]; yb = preds_clamped[b]
        hi = float(np.quantile(np.concatenate([xa, yb]), 0.999))
        hi = max(hi, 6.0)
        hb = ax.hexbin(xa, yb, gridsize=70, cmap="viridis", mincnt=1,
                       norm=LogNorm(vmin=1, vmax=max(2, len(xa))),
                       extent=(0, hi, 0, hi))
        ax.plot([0, hi], [0, hi], "w--", lw=0.8)
        mae_ab = float(np.mean(np.abs(xa - yb)))
        ax.set_title(f"{SHORT_LABELS[a]}  vs  {SHORT_LABELS[b]}\nMAE={mae_ab:.3f} eV",
                     fontsize=10)
        ax.set_xlabel(f"{SHORT_LABELS[a]} (eV)")
        ax.set_ylabel(f"{SHORT_LABELS[b]} (eV)")
    fig.savefig(PLOTS / fname, dpi=140)
    plt.close(fig)


def plot_functional_shift(pbe, preds_clamped, fname):
    mask = pbe > 0
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), constrained_layout=True)

    # mBJ - PBE
    ax = axes[0]
    y = preds_clamped["alignn_mbj"][mask] - pbe[mask]
    hi = float(np.quantile(pbe[mask], 0.999))
    hb = ax.hexbin(pbe[mask], y, gridsize=70, cmap="cividis", mincnt=1,
                   norm=LogNorm(vmin=1, vmax=max(2, mask.sum())),
                   extent=(0, hi, -2, 6))
    # Linear fit
    m, c = np.polyfit(pbe[mask], preds_clamped["alignn_mbj"][mask], 1)
    xs = np.linspace(0, hi, 50)
    ax.plot(xs, (m - 1) * xs + c, "r--", lw=1.5,
            label=f"fit: mBJ ≈ {m:.2f}·PBE {'+' if c>=0 else '−'} {abs(c):.2f}")
    ax.axhline(0, color="w", lw=0.6)
    ax.set_xlabel("PBE gap (eV)")
    ax.set_ylabel("ALIGNN-mBJ − PBE (eV)")
    ax.set_title("TB-mBJ correction vs PBE (non-metals)")
    ax.legend(fontsize=9, loc="upper left")

    # optb - PBE
    ax = axes[1]
    y = preds_clamped["alignn_optb"][mask] - pbe[mask]
    hb = ax.hexbin(pbe[mask], y, gridsize=70, cmap="cividis", mincnt=1,
                   norm=LogNorm(vmin=1, vmax=max(2, mask.sum())),
                   extent=(0, hi, -3, 3))
    m, c = np.polyfit(pbe[mask], preds_clamped["alignn_optb"][mask], 1)
    ax.plot(xs, (m - 1) * xs + c, "r--", lw=1.5,
            label=f"fit: optb ≈ {m:.2f}·PBE {'+' if c>=0 else '−'} {abs(c):.2f}")
    ax.axhline(0, color="w", lw=0.6)
    ax.set_xlabel("PBE gap (eV)")
    ax.set_ylabel("ALIGNN-optb88vdw − PBE (eV)")
    ax.set_title("OptB88vdW shift vs PBE (non-metals)")
    ax.legend(fontsize=9, loc="upper left")

    fig.savefig(PLOTS / fname, dpi=140)
    plt.close(fig)


def plot_cumulative_error(pbe, preds_clamped, keys, fname):
    mask = pbe > 0
    fig, ax = plt.subplots(figsize=(9, 5.4), constrained_layout=True)
    grid = np.linspace(0, 5, 500)
    for key in keys:
        err = np.abs(preds_clamped[key][mask] - pbe[mask])
        err_sorted = np.sort(err)
        y = np.searchsorted(err_sorted, grid) / len(err_sorted)
        ax.plot(grid, y, lw=2, color=COLORS[key], label=LABELS[key])
    ax.set_xlabel("|prediction − PBE| (eV)")
    ax.set_ylabel("fraction of non-metals within tolerance")
    ax.set_title("Cumulative error distribution (non-metals)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 1.02)
    fig.savefig(PLOTS / fname, dpi=140)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main driver
# -----------------------------------------------------------------------------

def main():
    merged = build_merged()
    print(f"Merged records: {len(merged)}")

    pbe, sk, apb, amb, aop = to_arrays(merged)
    preds_raw     = {"slakonet": sk, "alignn_pbe": apb, "alignn_mbj": amb, "alignn_optb": aop}
    preds_clamped = {k: clamp0(v) for k, v in preds_raw.items()}

    nonmetal_mask = pbe > 0
    metal_mask    = pbe == 0
    N_nm = int(nonmetal_mask.sum()); N_m = int(metal_mask.sum())

    negatives = {k: int((v < 0).sum()) for k, v in preds_raw.items()}

    # Regression
    reg_all, reg_nm, reg_m = {}, {}, {}
    for key in MODELS:
        reg_all[key] = regression_stats(preds_clamped[key], pbe)
        reg_nm[key]  = regression_stats(preds_clamped[key][nonmetal_mask], pbe[nonmetal_mask])
        reg_m[key]   = regression_stats(preds_clamped[key][metal_mask],    pbe[metal_mask])

    # Classification
    cls_stats = {k: classification_stats(preds_clamped[k], pbe) for k in MODELS}

    # Head-to-head each pair (i closer to PBE than j)
    h2h = {}
    for i, a in enumerate(MODELS):
        for b in MODELS[i+1:]:
            ea = np.abs(preds_clamped[a] - pbe)
            eb = np.abs(preds_clamped[b] - pbe)
            h2h[f"{a}_vs_{b}"] = {
                "a_better": int(np.sum(ea < eb)),
                "b_better": int(np.sum(eb < ea)),
                "ties":     int(np.sum(ea == eb)),
                "frac_a_better": float(np.mean(ea < eb)),
            }

    # Bin MAE (non-metals only)
    bin_edges = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0]
    bins_out = {k: [] for k in MODELS}
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        m = (pbe >= lo) & (pbe < hi) & nonmetal_mask
        for key in MODELS:
            err = preds_clamped[key][m] - pbe[m]
            if m.sum():
                bins_out[key].append({
                    "lo": lo, "hi": hi, "N": int(m.sum()),
                    "MAE": float(np.mean(np.abs(err))),
                    "ME":  float(np.mean(err)),
                })
            else:
                bins_out[key].append({"lo": lo, "hi": hi, "N": 0,
                                      "MAE": None, "ME": None})

    # Linear fit (non-metals) for each model: pred = m*PBE + c
    linfits = {}
    for key in MODELS:
        m_, c_ = np.polyfit(pbe[nonmetal_mask], preds_clamped[key][nonmetal_mask], 1)
        linfits[key] = {"slope": float(m_), "intercept": float(c_)}

    # Worst cases per model (over all merged)
    worst = {}
    for key in MODELS:
        err = preds_clamped[key] - pbe
        idx = np.argsort(-np.abs(err))[:50]
        worst[key] = [{
            "mat_id": merged[i]["mat_id"],
            "formula": merged[i]["formula"],
            "pbe": float(pbe[i]),
            "pred_raw": float(preds_raw[key][i]),
            "pred_clamped": float(preds_clamped[key][i]),
            "error": float(err[i]),
        } for i in idx]

    # All-model worst cases (structures where every ALIGNN model disagrees with PBE by >1 eV)
    align_keys = ["alignn_pbe", "alignn_mbj", "alignn_optb"]
    mask_all_bad = np.all(np.stack([
        np.abs(preds_clamped[k] - pbe) > 1.0 for k in align_keys
    ]), axis=0) & nonmetal_mask
    multi_hard = []
    for i in np.where(mask_all_bad)[0][:100]:
        multi_hard.append({
            "mat_id": merged[i]["mat_id"],
            "formula": merged[i]["formula"],
            "pbe": float(pbe[i]),
            "slakonet": float(preds_clamped["slakonet"][i]),
            "alignn_pbe": float(preds_clamped["alignn_pbe"][i]),
            "alignn_mbj": float(preds_clamped["alignn_mbj"][i]),
            "alignn_optb": float(preds_clamped["alignn_optb"][i]),
        })

    # Pairwise model-model MAE (no reference)
    mm_mae = {}
    for i, a in enumerate(MODELS):
        for b in MODELS[i+1:]:
            mm_mae[f"{a}_vs_{b}"] = float(np.mean(np.abs(preds_clamped[a] - preds_clamped[b])))

    # Dump metrics
    metrics = {
        "dataset": {
            "N_merged": len(merged),
            "N_metals_pbe": N_m,
            "N_nonmetals_pbe": N_nm,
            "metal_threshold_eV": METAL_THR,
            "negative_predictions_clamped": negatives,
        },
        "regression_vs_pbe_all":       reg_all,
        "regression_vs_pbe_nonmetals": reg_nm,
        "regression_vs_pbe_metals":    reg_m,
        "classification_vs_pbe":       cls_stats,
        "head_to_head":                h2h,
        "linear_fit_nonmetals":        linfits,
        "pairwise_model_mae":          mm_mae,
        "mae_by_pbe_gap_nonmetals":    bins_out,
        "worst_per_model":             worst,
        "hard_for_all_alignn":         multi_hard,
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))

    # Merged predictions (lightweight)
    (OUT / "merged_predictions.json").write_text(json.dumps(merged, default=float))

    # Plots
    keys = MODELS
    plot_parity_grid(pbe, preds_clamped, keys, "parity_all.png")
    plot_parity_grid(pbe, preds_clamped, keys, "parity_nonmetals.png",
                     nonmetal_mask=nonmetal_mask, tag=" (non-metals)")
    plot_residuals(pbe, preds_clamped, keys, "residuals_all.png")
    plot_residuals(pbe[nonmetal_mask],
                   {k: v[nonmetal_mask] for k, v in preds_clamped.items()},
                   keys, "residuals_nonmetals.png", tag=" (non-metals)")
    plot_confusion(pbe, preds_clamped, keys, "confusion_all.png")
    plot_mae_bins(pbe, preds_clamped, keys, "mae_by_gap_bin.png")
    plot_gap_distribution(pbe, preds_clamped, keys, "gap_distribution.png")
    plot_head_to_head(pbe, preds_clamped, keys, "head_to_head.png")
    plot_pairwise_model_agreement(preds_clamped, keys, "pairwise_alignn_agreement.png")
    plot_functional_shift(pbe, preds_clamped, "functional_shift.png")
    plot_cumulative_error(pbe, preds_clamped, keys, "cumulative_error.png")

    # Human-readable summary
    lines = []
    lines.append(f"N merged (SlakoNet ∩ ALIGNN): {len(merged)}")
    lines.append(f"  PBE non-metals (>{METAL_THR} eV): {N_nm}")
    lines.append(f"  PBE metals:                      {N_m}")
    lines.append("")
    lines.append("Negative predictions clamped to 0:")
    for k, v in negatives.items():
        lines.append(f"  {k:12s}: {v}")
    lines.append("")
    def fmt_row(name, d):
        r2 = f"{d['R2']:+.3f}" if not math.isnan(d['R2']) else "   n/a"
        return f"  {name:32s}  MAE={d['MAE']:.3f}  RMSE={d['RMSE']:.3f}  R2={r2}  ME={d['ME']:+.3f}"
    lines.append("Regression vs PBE (ALL):")
    for k in MODELS: lines.append(fmt_row(LABELS[k], reg_all[k]))
    lines.append("")
    lines.append("Regression vs PBE (NON-METALS):")
    for k in MODELS: lines.append(fmt_row(LABELS[k], reg_nm[k]))
    lines.append("")
    lines.append("Regression vs PBE (METALS):")
    for k in MODELS: lines.append(fmt_row(LABELS[k], reg_m[k]))
    lines.append("")
    lines.append("Classification vs PBE (threshold 0.1 eV):")
    for k in MODELS:
        c = cls_stats[k]
        lines.append(f"  {LABELS[k]:32s}  acc={c['accuracy']:.3f}  TN={c['TN']}  FP={c['FP']}  "
                     f"FN={c['FN']}  TP={c['TP']}")
    lines.append("")
    lines.append("Head-to-head (fraction of structures where first is closer to PBE):")
    for key, v in h2h.items():
        lines.append(f"  {key:40s}  a_better={v['a_better']}  b_better={v['b_better']}  "
                     f"frac_a={v['frac_a_better']:.3f}")
    lines.append("")
    lines.append("Linear fits (non-metals): pred ≈ slope * PBE + intercept")
    for k, v in linfits.items():
        lines.append(f"  {LABELS[k]:32s}  slope={v['slope']:.3f}  intercept={v['intercept']:+.3f}")
    lines.append("")
    lines.append("Pairwise model-model MAE (no reference):")
    for k, v in mm_mae.items():
        lines.append(f"  {k:40s}  MAE={v:.3f}")
    lines.append("")
    lines.append(f"Structures where every ALIGNN model is off by >1 eV on non-metals: "
                 f"{int(mask_all_bad.sum())}")
    (OUT / "summary.txt").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
