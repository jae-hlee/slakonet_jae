"""
Analyze SlakoNet predictions on Alexandria 2D PBE against DFT references.

Reads:
  results/all_results.json  (streamed; can be >>10 GB)
  alexandria_pbe_2d_2024.10.1_jarvis_tools.json.zip

Writes:
  analysis/summary.csv
  analysis/*.png
  analysis/stats.txt
"""
import os
import zipfile
import json
import ijson
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

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
OUT_DIR = HERE
os.makedirs(OUT_DIR, exist_ok=True)
RESULTS = os.path.join(PROJ, "results", "all_results.json")
SOURCE_ZIP = os.path.join(PROJ, "alexandria_pbe_2d_2024.10.1_jarvis_tools.json.zip")

METAL_THRESH = 0.1  # eV


def log(msg):
    print(msg, flush=True)


# ── Load source scalar metadata (small; ~138k entries) ────────────────────
log("Loading source zip...")
with zipfile.ZipFile(SOURCE_ZIP) as z:
    with z.open("alexandria_pbe_2d_2024.10.1_jarvis_tools.json") as f:
        src = json.load(f)
src_by_id = {}
for e in src:
    src_by_id[e["id"]] = {
        "formula": e["formula"],
        "nsites": e["nsites"],
        "spg": e["spg"],
        "e_form": e["e_form"],
        "e_above_hull": e.get("e_above_hull"),
        "band_gap_ind": e["band_gap_ind"],
        "band_gap_dir": e["band_gap_dir"],
        "dos_ef": e.get("dos_ef"),
    }
log(f"  source entries: {len(src)}")
del src  # free ~few GB


# ── Pass 1: stream predictions, extract scalars ───────────────────────────
log("Streaming all_results.json (pass 1: scalars)...")
rows = []
missing = 0
n = 0
with open(RESULTS, "rb") as f:
    for p in ijson.items(f, "item"):
        n += 1
        s = src_by_id.get(p["id"])
        if s is None:
            missing += 1
            continue
        rows.append((
            p["id"],
            s["formula"],
            s["nsites"],
            s["spg"],
            s["e_form"],
            s["e_above_hull"],
            s["band_gap_ind"],
            s["band_gap_dir"],
            s["dos_ef"],
            float(p["sk_bandgap"]),
        ))
        if n % 10000 == 0:
            log(f"  {n} predictions streamed")
log(f"  total streamed: {n}  joined: {len(rows)}  missing in source: {missing}")

df = pd.DataFrame(rows, columns=[
    "id", "formula", "nsites", "spg", "e_form", "e_above_hull",
    "dft_bandgap_eV", "dft_bandgap_dir_eV", "dos_ef_dft", "sk_bandgap_eV",
])

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
    return f"  {name:32s}  N={len(true):6d}  MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}  MaxErr={maxerr:.4f}"


def density_scatter(ax, x, y, bins=200):
    h, xe, ye = np.histogram2d(x, y, bins=bins)
    xi = np.clip(np.digitize(x, xe) - 1, 0, bins - 1)
    yi = np.clip(np.digitize(y, ye) - 1, 0, bins - 1)
    c = h[xi, yi]
    order = np.argsort(c)
    return ax.scatter(x[order], y[order], c=c[order], s=3, cmap="viridis",
                      norm=LogNorm(vmin=1), rasterized=True)


# ── Stats ─────────────────────────────────────────────────────────────────
lines = []
lines.append("=" * 80)
lines.append("SlakoNet vs PBE on Alexandria 2D")
lines.append("=" * 80)
lines.append(f"Source entries:         {len(src_by_id)}")
lines.append(f"SlakoNet completed:     {n}")
lines.append(f"Joined (id matched):    {len(df)}")
lines.append(f"PBE metals (gap<=0):    {int(is_pbe_metal.sum())}")
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

pbe_class = (ind > 0).astype(int)
sk_class = (sk > METAL_THRESH).astype(int)
cm = confusion_matrix(pbe_class, sk_class)
acc = np.trace(cm) / cm.sum()
tn, fp, fn, tp = cm.ravel()
lines.append("")
lines.append(f"Metal / non-metal classification  (SK threshold = {METAL_THRESH} eV)")
lines.append(f"  accuracy = {acc:.4f}")
lines.append(f"  TN={tn}  FP={fp}  FN={fn}  TP={tp}")

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
    ax.set_title(f"2D Alexandria - SlakoNet vs PBE {label}")
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
    ax.set_title(f"Non-metals only - SlakoNet vs PBE {label}")
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
ax.set_xlabel("SlakoNet - PBE indirect gap (eV)")
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
               label=f"SK<=0.1 mean = {res_nm[broken].mean():+.2f} eV  (N={int(broken.sum())})")
ax.legend(fontsize=8)
ax.set_xlabel("SlakoNet - PBE indirect gap (eV)")
ax.set_ylabel("Count")
ax.set_title(f"Residuals (PBE non-metals, N={len(res_nm)})")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "residuals_nonmetals.png"), dpi=180)
plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(cm, display_labels=["Metal", "Non-metal"]).plot(ax=ax, cmap="Blues")
ax.set_xlabel("SlakoNet prediction")
ax.set_ylabel("PBE reference")
ax.set_title(f"Metal / non-metal classification  (acc = {acc:.3f})")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=180)
plt.close(fig)


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


fig, ax = plt.subplots(figsize=(8, 5))
sc = ax.scatter(eform, np.abs(res), c=ind, cmap="coolwarm", s=3, alpha=0.5, rasterized=True)
ax.set_xlabel("Formation energy (eV/atom)")
ax.set_ylabel("|SlakoNet - PBE| indirect gap (eV)")
ax.set_title("Error vs formation energy")
plt.colorbar(sc, ax=ax, label="PBE indirect gap (eV)")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "error_vs_eform.png"), dpi=180)
plt.close(fig)


# ── Pass 2: stream predictions for DOS ────────────────────────────────────
# Pre-compute which IDs are metal/non-metal and pick example IDs first.
log("Picking DOS example IDs...")
metal_ids = set(df.loc[is_pbe_metal, "id"].tolist())
nm_ids = set(df.loc[is_pbe_nm, "id"].tolist())
targets = [(0.0, "metal"), (1.0, "semiconductor"), (4.0, "wide-gap insulator")]
example_ids = {}
example_meta = {}
for tgt, label in targets:
    i = int(np.argmin(np.abs(ind - tgt)))
    pid = df.iloc[i]["id"]
    example_ids[pid] = label
    example_meta[pid] = {
        "label": label,
        "formula": df.iloc[i]["formula"],
        "pbe": float(ind[i]),
        "sk": float(sk[i]),
    }
log(f"  examples: {example_ids}")

log("Streaming all_results.json (pass 2: DOS averages + examples)...")
dos_energies = None
metal_sum = None
nm_sum = None
n_met = n_nm = 0
example_dos = {}
n_seen = 0
with open(RESULTS, "rb") as f:
    for p in ijson.items(f, "item"):
        n_seen += 1
        pid = p["id"]
        if dos_energies is None:
            dos_energies = np.asarray([float(x) for x in p["dos_energies"]])
            metal_sum = np.zeros_like(dos_energies)
            nm_sum = np.zeros_like(dos_energies)
        if pid in metal_ids:
            metal_sum += np.asarray([float(x) for x in p["dos_values"]])
            n_met += 1
        elif pid in nm_ids:
            nm_sum += np.asarray([float(x) for x in p["dos_values"]])
            n_nm += 1
        if pid in example_ids:
            example_dos[pid] = (
                np.asarray([float(x) for x in p["dos_energies"]]),
                np.asarray([float(x) for x in p["dos_values"]]),
            )
        if n_seen % 10000 == 0:
            log(f"  {n_seen} DOS streamed")
log(f"  metals: {n_met}  non-metals: {n_nm}  examples: {len(example_dos)}")

metal_avg = metal_sum / max(n_met, 1)
nm_avg = nm_sum / max(n_nm, 1)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(dos_energies, metal_avg, label=f"PBE metals (avg of {n_met})", alpha=0.8)
ax.plot(dos_energies, nm_avg, label=f"PBE non-metals (avg of {n_nm})", alpha=0.8)
ax.axvline(0, color="k", lw=0.5, ls="--")
ax.set_xlabel("E - E_F (eV)")
ax.set_ylabel("DOS (arb. units)")
ax.set_title("Average SlakoNet DOS - 2D Alexandria")
ax.set_xlim(-10, 10)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "dos_average.png"), dpi=180)
plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, pid in zip(axes, example_ids):
    m = example_meta[pid]
    if pid in example_dos:
        e, d = example_dos[pid]
        ax.plot(e, d)
        ax.axvline(0, color="k", lw=0.5, ls="--")
        ax.set_xlim(-10, 10)
    ax.set_xlabel("E - E_F (eV)")
    ax.set_ylabel("DOS")
    ax.set_title(f"{m['formula']} ({pid})\n"
                 f"PBE = {m['pbe']:.2f}  SK = {m['sk']:.2f} eV  [{m['label']}]")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "dos_examples.png"), dpi=180)
plt.close(fig)

log(f"\nAll outputs in {OUT_DIR}/")
log("Done.")
