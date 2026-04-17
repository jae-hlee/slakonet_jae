"""
Analyze SlakoNet predictions on Alexandria 1D PBE against DFT references.

Reads:
  results/all_results.json              (8,636 SlakoNet predictions, ~1.5 GB)
  alexandria_pbe_1d_2024.10.1_jarvis_tools.json.zip  (13,295 source entries)

Writes:
  analysis/summary.csv                 (cross-dataset aggregator input)
  analysis/*.png                       (parity, residual, confusion, DOS)
  analysis/stats.txt                   (headline metrics)
"""
import os
import zipfile
import json
import orjson
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    confusion_matrix, ConfusionMatrixDisplay,
)

HERE = os.path.dirname(os.path.abspath(__file__))        # .../slako_v09_1d/analysis
PROJ = os.path.dirname(HERE)                             # .../slako_v09_1d
OUT_DIR = HERE
os.makedirs(OUT_DIR, exist_ok=True)
RESULTS = os.path.join(PROJ, "results", "all_results.json")
SOURCE_ZIP = os.path.join(PROJ, "alexandria_pbe_1d_2024.10.1_jarvis_tools.json.zip")

METAL_THRESH = 0.1  # eV


def log(msg):
    print(msg, flush=True)


# ── Load source (for DFT reference columns) ───────────────────────────────
log("Loading source zip...")
with zipfile.ZipFile(SOURCE_ZIP) as z:
    with z.open("alexandria_pbe_1d_2024.10.1_jarvis_tools.json") as f:
        src = json.load(f)
src_by_id = {e["id"]: e for e in src}
log(f"  source entries: {len(src)}")

# ── Load SlakoNet predictions ─────────────────────────────────────────────
log("Loading SlakoNet predictions...")
with open(RESULTS, "rb") as f:
    preds = orjson.loads(f.read())
log(f"  predictions: {len(preds)}")

# ── Join ──────────────────────────────────────────────────────────────────
rows = []
missing = 0
for p in preds:
    s = src_by_id.get(p["id"])
    if s is None:
        missing += 1
        continue
    rows.append({
        "id": p["id"],
        "formula": s["formula"],
        "nsites": s["nsites"],
        "spg": s["spg"],
        "e_form": s["e_form"],
        "dft_bandgap_eV": s["band_gap_ind"],
        "dft_bandgap_dir_eV": s["band_gap_dir"],
        "dos_ef_dft": s["dos_ef"],
        "sk_bandgap_eV": p["sk_bandgap"],
    })
df = pd.DataFrame(rows)
log(f"  joined rows: {len(df)} (missing in source: {missing})")

sk = df["sk_bandgap_eV"].to_numpy()
ind = df["dft_bandgap_eV"].to_numpy()
dirg = df["dft_bandgap_dir_eV"].to_numpy()
eform = df["e_form"].to_numpy()

is_pbe_metal = ind <= 0.0
is_pbe_nm = ~is_pbe_metal


def stats_line(name, pred, true):
    mae = mean_absolute_error(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
    r2 = r2_score(true, pred) if len(true) > 1 and np.std(true) > 0 else float("nan")
    maxerr = np.max(np.abs(pred - true)) if len(true) else float("nan")
    return f"  {name:32s}  N={len(true):6d}  MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}  MaxErr={maxerr:.4f}"


def density_scatter(ax, x, y, bins=200):
    h, xe, ye = np.histogram2d(x, y, bins=bins)
    xi = np.clip(np.digitize(x, xe) - 1, 0, bins - 1)
    yi = np.clip(np.digitize(y, ye) - 1, 0, bins - 1)
    c = h[xi, yi]
    order = np.argsort(c)
    return ax.scatter(x[order], y[order], c=c[order], s=3, cmap="viridis",
                      norm=LogNorm(vmin=1), rasterized=True)


# ── Error stats ───────────────────────────────────────────────────────────
lines = []
lines.append("=" * 80)
lines.append("SlakoNet vs PBE on Alexandria 1D")
lines.append("=" * 80)
lines.append(f"Source entries:         {len(src)}")
lines.append(f"SlakoNet completed:     {len(preds)}")
lines.append(f"Joined (id matched):    {len(df)}")
lines.append(f"PBE metals (gap≤0):     {int(is_pbe_metal.sum())}")
lines.append(f"PBE non-metals:         {int(is_pbe_nm.sum())}")
lines.append("")
lines.append("Indirect gap (SlakoNet vs PBE band_gap_ind):")
lines.append(stats_line("All", sk, ind))
lines.append(stats_line("PBE metals (gap=0)", sk[is_pbe_metal], ind[is_pbe_metal]))
lines.append(stats_line("PBE non-metals (gap>0)", sk[is_pbe_nm], ind[is_pbe_nm]))
lines.append("")
lines.append("Direct gap (SlakoNet vs PBE band_gap_dir):")
lines.append(stats_line("All", sk, dirg))
lines.append(stats_line("PBE non-metals", sk[is_pbe_nm], dirg[is_pbe_nm]))

# Metal / non-metal classification (threshold-based)
pbe_class = (ind > 0).astype(int)
sk_class = (sk > METAL_THRESH).astype(int)
cm = confusion_matrix(pbe_class, sk_class)
acc = np.trace(cm) / cm.sum()
tn, fp, fn, tp = cm.ravel()
lines.append("")
lines.append(f"Metal / non-metal classification  (SK threshold = {METAL_THRESH} eV)")
lines.append(f"  accuracy = {acc:.4f}")
lines.append(f"  TN={tn}  FP={fp}  FN={fn}  TP={tp}")

# False-negative diagnostic
fn_mask = (pbe_class == 1) & (sk_class == 0)
if fn_mask.sum():
    lines.append("")
    lines.append(f"False negatives (PBE says non-metal, SK predicts metal): N={int(fn_mask.sum())}")
    fn_pbe = ind[fn_mask]
    lines.append(f"  PBE gap:  min={fn_pbe.min():.3f}  median={np.median(fn_pbe):.3f}  "
                 f"mean={fn_pbe.mean():.3f}  max={fn_pbe.max():.3f}")

stats_text = "\n".join(lines)
log(stats_text)
with open(os.path.join(OUT_DIR, "stats.txt"), "w") as f:
    f.write(stats_text + "\n")


# ── summary.csv (for comprehensive_analysis aggregator) ───────────────────
df.to_csv(os.path.join(OUT_DIR, "summary.csv"), index=False)
log(f"Wrote {OUT_DIR}/summary.csv")


# ── Parity plots ──────────────────────────────────────────────────────────
log("Plotting parity...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, (ref, label) in zip(axes, [(ind, "Indirect gap"), (dirg, "Direct gap")]):
    sc = density_scatter(ax, ref, sk)
    lim = max(ref.max(), sk.max()) + 0.5
    ax.plot([0, lim], [0, lim], "r--", lw=1, label="y = x")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel(f"PBE {label} (eV)")
    ax.set_ylabel("SlakoNet gap (eV)")
    ax.set_title(f"1D Alexandria — SlakoNet vs PBE {label}")
    ax.set_aspect("equal")
    mae = mean_absolute_error(ref, sk)
    rmse = np.sqrt(mean_squared_error(ref, sk))
    ax.text(0.05, 0.95, f"N = {len(ref)}\nMAE = {mae:.3f} eV\nRMSE = {rmse:.3f} eV",
            transform=ax.transAxes, va="top",
            bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    ax.legend(loc="lower right")
    plt.colorbar(sc, ax=ax, label="Density")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "parity_all.png"), dpi=180)
plt.close(fig)

# Non-metals only
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, (ref, label) in zip(axes, [(ind[is_pbe_nm], "Indirect gap"),
                                   (dirg[is_pbe_nm], "Direct gap")]):
    pred = sk[is_pbe_nm]
    sc = density_scatter(ax, ref, pred)
    lim = max(ref.max(), pred.max()) + 0.5
    ax.plot([0, lim], [0, lim], "r--", lw=1, label="y = x")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel(f"PBE {label} (eV)")
    ax.set_ylabel("SlakoNet gap (eV)")
    ax.set_title(f"Non-metals only — SlakoNet vs PBE {label}")
    ax.set_aspect("equal")
    mae = mean_absolute_error(ref, pred)
    rmse = np.sqrt(mean_squared_error(ref, pred))
    ax.text(0.05, 0.95, f"N = {len(ref)}\nMAE = {mae:.3f} eV\nRMSE = {rmse:.3f} eV",
            transform=ax.transAxes, va="top",
            bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    ax.legend(loc="lower right")
    plt.colorbar(sc, ax=ax, label="Density")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "parity_nonmetals.png"), dpi=180)
plt.close(fig)


# ── Residuals ─────────────────────────────────────────────────────────────
log("Plotting residuals...")
res = sk - ind
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(res, bins=200, alpha=0.85)
ax.axvline(0, color="r", ls="--")
ax.axvline(res.mean(), color="k", ls="-", lw=1.2,
           label=f"mean = {res.mean():+.2f} eV")
ax.set_xlabel("SlakoNet − PBE indirect gap (eV)")
ax.set_ylabel("Count")
ax.set_title(f"Residuals (all, N={len(res)})")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "residuals_all.png"), dpi=180)
plt.close(fig)

res_nm = res[is_pbe_nm]
sk_nm = sk[is_pbe_nm]
working = sk_nm > METAL_THRESH
broken = ~working
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(res_nm, bins=200, alpha=0.85)
ax.axvline(0, color="r", ls="--")
ax.axvline(res_nm.mean(), color="k", ls="-", lw=1.2,
           label=f"overall mean = {res_nm.mean():+.2f} eV  (N={len(res_nm)})")
if working.any():
    ax.axvline(res_nm[working].mean(), color="green", ls="-", lw=1.2,
               label=f"SK>0.1 mean = {res_nm[working].mean():+.2f} eV  (N={int(working.sum())})")
if broken.any():
    ax.axvline(res_nm[broken].mean(), color="orange", ls="-", lw=1.2,
               label=f"SK≤0.1 mean = {res_nm[broken].mean():+.2f} eV  (N={int(broken.sum())})")
ax.legend(fontsize=8)
ax.set_xlabel("SlakoNet − PBE indirect gap (eV)")
ax.set_ylabel("Count")
ax.set_title(f"Residuals (PBE non-metals, N={len(res_nm)})")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "residuals_nonmetals.png"), dpi=180)
plt.close(fig)


# ── Confusion matrix ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(cm, display_labels=["Metal", "Non-metal"]).plot(ax=ax, cmap="Blues")
ax.set_xlabel("SlakoNet prediction")
ax.set_ylabel("PBE reference")
ax.set_title(f"Metal / non-metal classification  (acc = {acc:.3f})")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=180)
plt.close(fig)


# ── Band gap distribution ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
bins = np.linspace(0, max(ind.max(), sk.max()), 200)
ax.hist(ind, bins=bins, alpha=0.6, label="PBE indirect gap", density=True)
ax.hist(sk, bins=bins, alpha=0.6, label="SlakoNet gap", density=True)
ax.set_xlabel("Band gap (eV)")
ax.set_ylabel("Density")
ax.set_title(f"Gap distribution: PBE vs SlakoNet  (N={len(ind)})")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "gap_distribution.png"), dpi=180)
plt.close(fig)


# ── Formation energy vs absolute error ────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
sc = ax.scatter(eform, np.abs(res), c=ind, cmap="coolwarm", s=3, alpha=0.5, rasterized=True)
ax.set_xlabel("Formation energy (eV/atom)")
ax.set_ylabel("|SlakoNet − PBE| indirect gap (eV)")
ax.set_title("Error vs formation energy")
plt.colorbar(sc, ax=ax, label="PBE indirect gap (eV)")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "error_vs_eform.png"), dpi=180)
plt.close(fig)


# ── DOS analysis (stream predictions; avoid holding 1.5GB twice) ──────────
log("Loading DOS arrays...")
# preds already in memory; build aligned indices back to df rows
df_idx_by_id = {rid: i for i, rid in enumerate(df["id"].tolist())}
dos_energies = np.asarray(preds[0]["dos_energies"])
metal_sum = np.zeros_like(dos_energies)
nm_sum = np.zeros_like(dos_energies)
n_met = n_nm = 0
for p in preds:
    i = df_idx_by_id.get(p["id"])
    if i is None:
        continue
    d = np.asarray(p["dos_values"])
    if is_pbe_metal[i]:
        metal_sum += d
        n_met += 1
    else:
        nm_sum += d
        n_nm += 1
metal_avg = metal_sum / max(n_met, 1)
nm_avg = nm_sum / max(n_nm, 1)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(dos_energies, metal_avg, label=f"PBE metals (avg of {n_met})", alpha=0.8)
ax.plot(dos_energies, nm_avg, label=f"PBE non-metals (avg of {n_nm})", alpha=0.8)
ax.axvline(0, color="k", lw=0.5, ls="--")
ax.set_xlabel("E − E_F (eV)")
ax.set_ylabel("DOS (arb. units)")
ax.set_title("Average SlakoNet DOS — 1D Alexandria")
ax.set_xlim(-10, 10)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "dos_average.png"), dpi=180)
plt.close(fig)

# Example DOS at three PBE gap targets
targets = [(0.0, "metal"), (1.0, "semiconductor"), (4.0, "wide-gap insulator")]
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, (tgt, label) in zip(axes, targets):
    i = int(np.argmin(np.abs(ind - tgt)))
    pid = df.iloc[i]["id"]
    p = next(p for p in preds if p["id"] == pid)
    e = np.asarray(p["dos_energies"])
    d = np.asarray(p["dos_values"])
    ax.plot(e, d)
    ax.axvline(0, color="k", lw=0.5, ls="--")
    ax.set_xlabel("E − E_F (eV)")
    ax.set_ylabel("DOS")
    ax.set_title(f"{df.iloc[i]['formula']} ({pid})\n"
                 f"PBE = {ind[i]:.2f}  SK = {sk[i]:.2f} eV  [{label}]")
    ax.set_xlim(-10, 10)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "dos_examples.png"), dpi=180)
plt.close(fig)

log(f"\nAll outputs in {OUT_DIR}/")
log("Done.")
