import json
import re
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.metrics import mean_absolute_error, mean_squared_error

HARTREE_TO_EV = 27.211386245988

# ── Load data (handle NaN values in JSON) ────────────────────────────────
print("Loading JSON...", flush=True)
with open("results/all_results.json", "r") as f:
    content = f.read()
content = re.sub(r'\bNaN\b', 'null', content)
data = json.loads(content)
print(f"Loaded {len(data)} entries", flush=True)

# Filter entries with valid homo/lumo
valid = [d for d in data if d["homo"] is not None and d["lumo"] is not None
         and d["jid"] != "cc-699"]  # CCl3- outlier (HOMO=-9.26 Ha → 255 eV HL gap)
print(f"Valid entries (non-null homo & lumo, no outliers): {len(valid)} / {len(data)}", flush=True)

# Extract arrays (raw values from JSON)
sk_gap = np.array([d["sk_bandgap"] for d in valid])
homo_raw = np.array([d["homo"] for d in valid])
lumo_raw = np.array([d["lumo"] for d in valid])
species = [d["species"] for d in valid]
jids = [d["jid"] for d in valid]


# ── Helpers ───────────────────────────────────────────────────────────────
def print_stats(name, arr):
    print(f"  {name:30s}  N={len(arr):6d}  mean={np.mean(arr):.4f}  "
          f"std={np.std(arr):.4f}  min={np.min(arr):.4f}  max={np.max(arr):.4f}  "
          f"median={np.median(arr):.4f}")


def density_scatter(ax, x, y, bins=100, **kwargs):
    """Scatter colored by 2D histogram density."""
    h, xedges, yedges = np.histogram2d(x, y, bins=bins)
    xidx = np.clip(np.digitize(x, xedges) - 1, 0, bins - 1)
    yidx = np.clip(np.digitize(y, yedges) - 1, 0, bins - 1)
    c = h[xidx, yidx]
    order = np.argsort(c)
    return ax.scatter(x[order], y[order], c=c[order], s=2, cmap="viridis",
                      norm=LogNorm(vmin=1), rasterized=True, **kwargs)


def run_analysis(out_dir, unit_label, homo, lumo, hl_gap):
    """Run full analysis suite and save plots to out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    residuals = sk_gap - hl_gap

    # ── 1. Summary statistics ────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print(f"SUMMARY STATISTICS (assuming HOMO/LUMO in {unit_label})")
    print(f"{'=' * 80}")
    print_stats("SlakoNet band gap (eV)", sk_gap)
    print_stats(f"HOMO ({unit_label})", homo_raw)
    print_stats(f"LUMO ({unit_label})", lumo_raw)
    print_stats("HOMO-LUMO gap (eV)", hl_gap)

    # ── 2. Parity plot ───────────────────────────────────────────────────
    print("  Generating parity plot...", flush=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = density_scatter(ax, hl_gap, sk_gap)
    lim = max(hl_gap.max(), sk_gap.max()) * 1.05
    ax.plot([0, lim], [0, lim], "r--", lw=1, label="y = x")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel(f"HOMO-LUMO gap (eV) [from {unit_label}]")
    ax.set_ylabel("SlakoNet band gap (eV)")
    ax.set_title(f"SlakoNet vs HOMO-LUMO gap (HOMO/LUMO in {unit_label})")
    ax.set_aspect("equal")
    ax.legend()
    plt.colorbar(sc, ax=ax, label="Density")
    mae = mean_absolute_error(hl_gap, sk_gap)
    rmse = np.sqrt(mean_squared_error(hl_gap, sk_gap))
    ax.text(0.05, 0.95, f"MAE = {mae:.3f} eV\nRMSE = {rmse:.3f} eV",
            transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "parity_sk_vs_hl.png"), dpi=200)
    plt.close(fig)
    print(f"  MAE={mae:.4f}  RMSE={rmse:.4f}")

    # ── 3. Distribution plots ────────────────────────────────────────────
    print("  Generating distribution plots...", flush=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].hist(sk_gap, bins=100, edgecolor="none", alpha=0.8)
    axes[0, 0].set_xlabel("SlakoNet band gap (eV)")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].set_title(f"SlakoNet band gap distribution (N={len(sk_gap)})")

    axes[0, 1].hist(hl_gap, bins=100, edgecolor="none", alpha=0.8, color="tab:orange")
    axes[0, 1].set_xlabel("HOMO-LUMO gap (eV)")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].set_title(f"HOMO-LUMO gap distribution (N={len(hl_gap)})")

    axes[1, 0].hist(homo, bins=100, edgecolor="none", alpha=0.8, color="tab:green")
    axes[1, 0].set_xlabel(f"HOMO (eV)")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].set_title(f"HOMO distribution (N={len(homo)})")

    axes[1, 1].hist(lumo, bins=100, edgecolor="none", alpha=0.8, color="tab:red")
    axes[1, 1].set_xlabel(f"LUMO (eV)")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].set_title(f"LUMO distribution (N={len(lumo)})")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "distributions.png"), dpi=200)
    plt.close(fig)

    # Overlay comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(hl_gap, bins=100, alpha=0.6, label="HOMO-LUMO gap")
    ax.hist(sk_gap, bins=100, alpha=0.6, label="SlakoNet band gap")
    ax.set_xlabel("Gap (eV)")
    ax.set_ylabel("Count")
    ax.set_title(f"Gap distribution comparison (HOMO/LUMO in {unit_label})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "gap_distribution_comparison.png"), dpi=200)
    plt.close(fig)

    # ── 4. Residual histogram ────────────────────────────────────────────
    print("  Generating residual histogram...", flush=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(residuals, bins=200, edgecolor="none", alpha=0.8)
    ax.axvline(0, color="r", ls="--")
    ax.set_xlabel("SlakoNet gap − HOMO-LUMO gap (eV)")
    ax.set_ylabel("Count")
    ax.set_title(f"Residual distribution (N={len(residuals)})")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "residual_histogram.png"), dpi=200)
    plt.close(fig)

    # ── 5. HOMO vs LUMO scatter ──────────────────────────────────────────
    print("  Generating HOMO vs LUMO scatter...", flush=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(homo, lumo, c=sk_gap, s=2, cmap="coolwarm", rasterized=True)
    ax.set_xlabel("HOMO (eV)")
    ax.set_ylabel("LUMO (eV)")
    ax.set_title(f"HOMO vs LUMO (colored by SlakoNet gap)")
    plt.colorbar(sc, ax=ax, label="SlakoNet band gap (eV)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "homo_vs_lumo.png"), dpi=200)
    plt.close(fig)

    # ── 6. DOS analysis ──────────────────────────────────────────────────
    print("  Computing average DOS...", flush=True)
    dos_energies = np.array(valid[0]["dos_energies"])

    median_gap = np.median(sk_gap)
    large_gap_idx = np.where(sk_gap >= median_gap)[0]
    small_gap_idx = np.where(sk_gap < median_gap)[0]

    large_dos_avg = np.zeros(len(dos_energies))
    for i in large_gap_idx:
        large_dos_avg += np.array(valid[i]["dos_values"])
    large_dos_avg /= len(large_gap_idx)

    small_dos_avg = np.zeros(len(dos_energies))
    for i in small_gap_idx:
        small_dos_avg += np.array(valid[i]["dos_values"])
    small_dos_avg /= len(small_gap_idx)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dos_energies, small_dos_avg, label=f"Small gap < {median_gap:.1f} eV (N={len(small_gap_idx)})", alpha=0.8)
    ax.plot(dos_energies, large_dos_avg, label=f"Large gap ≥ {median_gap:.1f} eV (N={len(large_gap_idx)})", alpha=0.8)
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("DOS (arb. units)")
    ax.set_title("Average DOS: small-gap vs large-gap molecules")
    ax.legend()
    ax.set_xlim(dos_energies.min(), dos_energies.max())
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "dos_average.png"), dpi=200)
    plt.close(fig)

    # Example DOS plots
    examples = []
    for target_gap, label in [(0.5, "small gap"), (5.0, "medium gap"), (15.0, "large gap")]:
        best_idx = int(np.argmin(np.abs(sk_gap - target_gap)))
        examples.append((best_idx, label, species[best_idx], jids[best_idx]))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, (idx, label, sp, jid) in zip(axes, examples):
        e = np.array(valid[idx]["dos_energies"])
        d = np.array(valid[idx]["dos_values"])
        ax.plot(e, d)
        ax.set_xlabel("Energy (eV)")
        ax.set_ylabel("DOS")
        ax.set_title(f"{sp} ({jid})\nSK gap={sk_gap[idx]:.2f}, HL gap={hl_gap[idx]:.2f} eV\n[{label}]")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "dos_examples.png"), dpi=200)
    plt.close(fig)

    # ── 7. Top outliers ──────────────────────────────────────────────────
    print(f"\n  Top 10 largest |residual| outliers ({unit_label}):")
    abs_res = np.abs(residuals)
    top_idx = np.argsort(abs_res)[::-1][:10]
    for rank, i in enumerate(top_idx, 1):
        print(f"    {rank:2d}. {jids[i]:10s}  {species[i]:20s}  "
              f"SK={sk_gap[i]:.3f}  HL={hl_gap[i]:.3f}  residual={residuals[i]:+.3f}")

    print(f"  Plots saved to {out_dir}/")


# ── Run analysis assuming HOMO/LUMO in Hartree ──────────────────────────
print("\n>>> Analysis 1: HOMO/LUMO assumed in Hartree (converted to eV)")
homo_ev_from_ha = homo_raw * HARTREE_TO_EV
lumo_ev_from_ha = lumo_raw * HARTREE_TO_EV
hl_gap_from_ha = lumo_ev_from_ha - homo_ev_from_ha
run_analysis("analysis/hartree", "Hartree", homo_ev_from_ha, lumo_ev_from_ha, hl_gap_from_ha)

# ── Run analysis assuming HOMO/LUMO already in eV ───────────────────────
print("\n>>> Analysis 2: HOMO/LUMO assumed in eV")
hl_gap_ev = lumo_raw - homo_raw
run_analysis("analysis/ev", "eV", homo_raw, lumo_raw, hl_gap_ev)

# ── Write summary CSV (all 1324 entries) ─────────────────────────────────
import csv

print("\nWriting summary.csv...", flush=True)
all_entries = json.loads(content)  # already has NaN replaced with null
with open("summary.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["jid", "species",
                     "homo_raw", "lumo_raw",
                     "hl_gap_hartree_eV", "hl_gap_ev_eV",
                     "sk_bandgap_eV", "excluded"])
    for d in all_entries:
        homo = d["homo"]
        lumo = d["lumo"]
        jid = d["jid"]
        excluded = ""
        if homo is None or lumo is None:
            excluded = "null_homo_lumo"
        elif jid == "cc-699":
            excluded = "outlier"

        if homo is not None and lumo is not None:
            hl_ha = (lumo - homo) * HARTREE_TO_EV
            hl_ev = lumo - homo
            writer.writerow([jid, d["species"],
                             f"{homo:.6f}", f"{lumo:.6f}",
                             f"{hl_ha:.6f}", f"{hl_ev:.6f}",
                             f"{d['sk_bandgap']:.6f}", excluded])
        else:
            writer.writerow([jid, d["species"],
                             homo if homo is not None else "",
                             lumo if lumo is not None else "",
                             "", "",
                             f"{d['sk_bandgap']:.6f}", excluded])

print(f"summary.csv written ({len(all_entries)} rows)")
print("Done. Results in analysis/hartree/ and analysis/ev/")
