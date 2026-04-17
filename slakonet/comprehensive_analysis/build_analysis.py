"""
Cross-dataset analysis for SlakoNet runs v03–v09 (extensible to v10/v11).

Loads the per-dataset summaries and per-structure results already produced by each
sister project and emits a unified set of comparison plots into this directory:

  - dataset_overview.png         : N structures / coverage / metallic fraction
  - gap_distributions.png        : SlakoNet bandgap histograms across datasets
  - parity_grid.png              : SK vs DFT parity for datasets with a reference gap
  - residual_distributions.png   : SK − DFT residual densities
  - error_summary.png            : MAE / RMSE / Pearson-r bars across datasets
  - dos_average_grid.png         : Fermi-aligned mean DOS per dataset
  - v08_tc_correlations.png      : Tc vs {sk_bandgap, dosef, lambda} for supercon
  - summary_table.csv / .md      : Headline stats

Silently skips datasets that have no `results/all_results.json` yet, so the same
script picks up v10_2d / v11_alexwz once those runs finish.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
OUT.mkdir(exist_ok=True)

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 140,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "font.size": 9,
    }
)

METAL_THR = 0.1  # eV — SlakoNet gap below this counts as metallic


# -------------------------------------------------------------------------
# Dataset specs
# -------------------------------------------------------------------------
@dataclass
class Dataset:
    key: str              # short label, e.g. "v03_alex"
    title: str            # pretty title
    kind: str             # "crystal" | "molecule" | "interface" | "surface" | "defect" | "supercon" | "low_dim"
    reference: str | None # name of the DFT/reference gap column in the frame (None if no reference)
    frame: pd.DataFrame   # columns: id, sk_bandgap_eV, optional <reference>

    @property
    def n(self) -> int:
        return len(self.frame)

    @property
    def has_reference(self) -> bool:
        return self.reference is not None and self.reference in self.frame.columns


def _safe_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"[warn] could not read {path}: {exc}")
        return None


def load_v03() -> Dataset | None:
    """v03_alex — Alexandria 3D PBE. Use sk_scalars.json + alignn_predictions.json
    (alignn_predictions.json holds PBE band_gap_ind). Merge on mat_id."""
    base = ROOT / "slako_v03_alex" / "results"
    sk_path = base / "sk_scalars.json"
    alignn_path = base / "alignn_predictions.json"
    if not (sk_path.exists() and alignn_path.exists()):
        return None
    sk = pd.DataFrame(json.loads(sk_path.read_text()))
    al = pd.DataFrame(json.loads(alignn_path.read_text()))
    df = sk.merge(al[["mat_id", "band_gap_ind", "alignn_bandgap"]], on="mat_id", how="inner")
    df = df.rename(columns={"mat_id": "id", "sk_bandgap": "sk_bandgap_eV",
                            "band_gap_ind": "dft_bandgap_eV",
                            "alignn_bandgap": "alignn_bandgap_eV"})
    df["dft_bandgap_eV"] = df["dft_bandgap_eV"].clip(lower=0)
    return Dataset("v03_alex", "Alexandria 3D PBE", "crystal",
                   reference="dft_bandgap_eV", frame=df)


def load_v04() -> Dataset | None:
    """v04_cccbdb — CCCBDB molecules. Hartree→eV HL gap is the right reference."""
    csv = ROOT / "slako_v04_cccbdb" / "summary.csv"
    df = _safe_read_csv(csv)
    if df is None:
        return None
    df = df.rename(columns={"jid": "id", "hl_gap_hartree_eV": "dft_bandgap_eV"})
    df = df[["id", "species", "sk_bandgap_eV", "dft_bandgap_eV"]].dropna(subset=["sk_bandgap_eV"])
    # drop a few zero/negative HL gaps that are dataset artefacts
    df = df[df["dft_bandgap_eV"].between(0, 50)]
    return Dataset("v04_cccbdb", "CCCBDB molecules", "molecule",
                   reference="dft_bandgap_eV", frame=df)


def load_v05() -> Dataset | None:
    csv = ROOT / "slako_v05_interface" / "analysis" / "summary.csv"
    df = _safe_read_csv(csv)
    if df is None:
        return None
    df = df.rename(columns={"jid": "id", "dft_bandgap_eV": "dft_bandgap_eV"})
    # v05 stores the raw (possibly negative) DFT gap; clip to physical range
    df["dft_bandgap_eV"] = df["dft_bandgap_eV"].clip(lower=0)
    return Dataset("v05_interface", "Interface slabs (optB88vdW)", "interface",
                   reference="dft_bandgap_eV", frame=df[["id", "sk_bandgap_eV", "dft_bandgap_eV"]])


def load_v06() -> Dataset | None:
    """v06 — surface slabs. The CSV in that project uses scf_cbm−scf_vbm which its
    own CLAUDE.md flags as wrong (bulk-referenced, unrelated to the slab gap). Pull
    surf_cbm/surf_vbm from the zipped source instead."""
    import zipfile
    csv = ROOT / "slako_v06_surface" / "analysis" / "summary.csv"
    zipp = ROOT / "slako_v06_surface" / "surface_db_dd.json.zip"
    df = _safe_read_csv(csv)
    if df is None:
        return None
    df = df.rename(columns={"name": "id"})
    if zipp.exists():
        with zipfile.ZipFile(zipp) as zf:
            raw = json.loads(zf.read(zf.namelist()[0]))
        slab = pd.DataFrame([{"id": r["name"],
                              "surf_vbm": r.get("surf_vbm"),
                              "surf_cbm": r.get("surf_cbm")} for r in raw])
        df = df.merge(slab, on="id", how="left")
        gap = (df["surf_cbm"] - df["surf_vbm"]).clip(lower=0)
        df["dft_bandgap_eV"] = gap
    else:
        df["dft_bandgap_eV"] = df["dft_gap_clipped_eV"]
    df = df.dropna(subset=["sk_bandgap_eV", "dft_bandgap_eV"])
    return Dataset("v06_surface", "Surface slabs (PBE)", "surface",
                   reference="dft_bandgap_eV", frame=df[["id", "sk_bandgap_eV", "dft_bandgap_eV"]])


def load_v07() -> Dataset | None:
    csv = ROOT / "slako_v07_vacancy" / "analysis" / "summary.csv"
    df = _safe_read_csv(csv)
    if df is None:
        return None
    return Dataset("v07_vacancy", "Vacancy defects", "defect",
                   reference=None,
                   frame=df[["id", "jid", "material_class", "vacancy_symbol",
                             "ef_eV", "sk_bandgap_eV"]])


def load_v08() -> Dataset | None:
    csv = ROOT / "slako_v08_supercon" / "analysis" / "summary.csv"
    df = _safe_read_csv(csv)
    if df is None:
        return None
    return Dataset("v08_supercon", "Alexandria supercon candidates", "supercon",
                   reference=None,
                   frame=df[["id", "Tc_K", "dosef", "debye_K", "lambda", "wlog_K",
                             "sk_bandgap_eV"]])


# Extension hooks — these just need to land a summary.csv with sk_bandgap_eV
def load_stub(key: str, title: str, kind: str, sub: str) -> Dataset | None:
    """Pick up v09/v10/v11 once results land."""
    candidates = [
        ROOT / sub / "analysis" / "summary.csv",
        ROOT / sub / "summary.csv",
    ]
    for c in candidates:
        df = _safe_read_csv(c)
        if df is None:
            continue
        if "sk_bandgap_eV" not in df.columns:
            continue
        df = df.rename(columns={df.columns[0]: "id"})
        ref = "dft_bandgap_eV" if "dft_bandgap_eV" in df.columns else None
        cols = ["id", "sk_bandgap_eV"] + ([ref] if ref else [])
        return Dataset(key, title, kind, reference=ref, frame=df[cols])
    return None


def load_all() -> list[Dataset]:
    candidates: list[Callable[[], Dataset | None]] = [
        load_v03, load_v04, load_v05, load_v06, load_v07, load_v08,
        lambda: load_stub("v09_1d", "Alexandria 1D PBE", "low_dim", "slako_v09_1d"),
        lambda: load_stub("v10_2d", "Alexandria 2D PBE", "low_dim", "slako_v10_2d"),
        lambda: load_stub("v11_alexwz", "Alexandria WZ PBE", "crystal", "slako_v11_alexwz"),
    ]
    out: list[Dataset] = []
    for fn in candidates:
        try:
            ds = fn()
        except Exception as exc:
            print(f"[warn] loader {fn} failed: {exc}")
            continue
        if ds is None:
            continue
        print(f"[ok] {ds.key:<14}  N={ds.n:>6}  reference={ds.reference}")
        out.append(ds)
    return out


# -------------------------------------------------------------------------
# Metric helpers
# -------------------------------------------------------------------------
def regression_stats(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]; y_pred = y_pred[mask]
    if len(y_true) < 2:
        return {"mae": np.nan, "rmse": np.nan, "pearson_r": np.nan, "n": int(len(y_true))}
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    if np.std(y_true) > 0 and np.std(y_pred) > 0:
        r = float(stats.pearsonr(y_true, y_pred)[0])
    else:
        r = np.nan
    return {"mae": mae, "rmse": rmse, "pearson_r": r, "n": int(len(y_true))}


# -------------------------------------------------------------------------
# Plotting
# -------------------------------------------------------------------------
def palette(n: int) -> list:
    cmap = plt.get_cmap("tab10")
    return [cmap(i % 10) for i in range(n)]


def plot_dataset_overview(datasets: list[Dataset]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    keys = [d.key for d in datasets]
    colors = palette(len(datasets))

    # (1) N structures
    ns = [d.n for d in datasets]
    axes[0].barh(keys, ns, color=colors)
    for i, n in enumerate(ns):
        axes[0].text(n, i, f" {n:,}", va="center", fontsize=8)
    axes[0].set_xlabel("# structures evaluated")
    axes[0].set_title("Coverage per dataset")
    axes[0].set_xscale("log")
    axes[0].invert_yaxis()

    # (2) Fraction metallic (sk_bandgap < METAL_THR)
    frac_metal = [float((d.frame["sk_bandgap_eV"] < METAL_THR).mean()) for d in datasets]
    axes[1].barh(keys, frac_metal, color=colors)
    for i, f in enumerate(frac_metal):
        axes[1].text(f, i, f" {f:.0%}", va="center", fontsize=8)
    axes[1].set_xlim(0, 1.05)
    axes[1].set_xlabel(f"fraction SlakoNet metallic (gap < {METAL_THR} eV)")
    axes[1].set_title("Predicted metallicity")
    axes[1].invert_yaxis()

    # (3) sk_bandgap median ± IQR
    meds = [float(d.frame["sk_bandgap_eV"].median()) for d in datasets]
    q1 = [float(d.frame["sk_bandgap_eV"].quantile(0.25)) for d in datasets]
    q3 = [float(d.frame["sk_bandgap_eV"].quantile(0.75)) for d in datasets]
    xpos = np.arange(len(datasets))
    axes[2].errorbar(meds, xpos, xerr=[np.array(meds) - np.array(q1), np.array(q3) - np.array(meds)],
                     fmt="o", color="black", capsize=3)
    for i, m in enumerate(meds):
        axes[2].text(m, i - 0.25, f" {m:.2f}", fontsize=8)
    axes[2].set_yticks(xpos)
    axes[2].set_yticklabels(keys)
    axes[2].set_xlabel("SlakoNet band gap (eV) — median ± IQR")
    axes[2].set_title("Predicted gap magnitude")
    axes[2].invert_yaxis()

    fig.suptitle("SlakoNet cross-dataset overview", fontsize=12)
    fig.savefig(OUT / "dataset_overview.png")
    plt.close(fig)


def plot_gap_distributions(datasets: list[Dataset]) -> None:
    n = len(datasets)
    ncol = 3
    nrow = math.ceil(n / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow), sharex=True, constrained_layout=True)
    axes = np.atleast_2d(axes)
    bins = np.linspace(0, 15, 61)
    for ax, ds in zip(axes.flat, datasets):
        sk = ds.frame["sk_bandgap_eV"].values
        ax.hist(sk, bins=bins, color="#2166ac", alpha=0.7, label="SlakoNet")
        if ds.has_reference:
            ref = ds.frame[ds.reference].values
            ax.hist(ref, bins=bins, color="#b2182b", alpha=0.5, label="Reference")
        med = np.nanmedian(sk)
        ax.axvline(med, ls="--", color="#2166ac", alpha=0.9, lw=1)
        ax.set_title(f"{ds.title}\n(N={ds.n})", fontsize=9)
        ax.set_xlabel("band gap (eV)")
        ax.set_ylabel("count")
        ax.set_xlim(0, 15)
        ax.legend(fontsize=7)
    # hide unused subplots
    for ax in axes.flat[n:]:
        ax.axis("off")
    fig.suptitle("SlakoNet gap distributions (reference gap overlaid where available)")
    fig.savefig(OUT / "gap_distributions.png")
    plt.close(fig)


def plot_parity_grid(datasets: list[Dataset]) -> None:
    ref_sets = [d for d in datasets if d.has_reference]
    if not ref_sets:
        return
    n = len(ref_sets)
    ncol = min(3, n)
    nrow = math.ceil(n / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.5 * ncol, 4.3 * nrow), constrained_layout=True)
    axes = np.atleast_1d(axes).flatten()
    for ax, ds in zip(axes, ref_sets):
        ref = ds.frame[ds.reference].values
        sk = ds.frame["sk_bandgap_eV"].values
        mask = np.isfinite(ref) & np.isfinite(sk)
        ref = ref[mask]; sk = sk[mask]
        stats_ = regression_stats(ref, sk)
        hb = ax.hexbin(ref, sk, gridsize=40, mincnt=1, cmap="viridis", bins="log")
        lim = max(np.percentile(ref, 99), np.percentile(sk, 99), 5)
        ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.7)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_xlabel("reference gap (eV)")
        ax.set_ylabel("SlakoNet gap (eV)")
        ax.set_title(f"{ds.title}  (N={stats_['n']})\n"
                     f"MAE={stats_['mae']:.2f} eV  RMSE={stats_['rmse']:.2f} eV  r={stats_['pearson_r']:.2f}")
        fig.colorbar(hb, ax=ax, shrink=0.85, label="log10 count")
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("SlakoNet parity against reference DFT gaps")
    fig.savefig(OUT / "parity_grid.png")
    plt.close(fig)


def plot_residuals(datasets: list[Dataset]) -> None:
    ref_sets = [d for d in datasets if d.has_reference]
    if not ref_sets:
        return
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    colors = palette(len(ref_sets))
    bins = np.linspace(-8, 10, 73)
    for ds, color in zip(ref_sets, colors):
        r = (ds.frame["sk_bandgap_eV"] - ds.frame[ds.reference]).dropna().values
        weights = np.ones_like(r) / len(r)
        ax.hist(r, bins=bins, weights=weights, alpha=0.45, label=f"{ds.key} (μ={r.mean():+.2f})", color=color)
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("SlakoNet − reference gap (eV)")
    ax.set_ylabel("density")
    ax.set_title("Residual distributions (SK − reference)")
    ax.legend(fontsize=8, loc="upper right")
    fig.savefig(OUT / "residual_distributions.png")
    plt.close(fig)


def plot_error_summary(datasets: list[Dataset]) -> pd.DataFrame:
    rows = []
    for ds in datasets:
        row = {"dataset": ds.key, "kind": ds.kind, "N": ds.n,
               "frac_sk_metal": float((ds.frame["sk_bandgap_eV"] < METAL_THR).mean()),
               "sk_mean_eV": float(ds.frame["sk_bandgap_eV"].mean()),
               "sk_median_eV": float(ds.frame["sk_bandgap_eV"].median())}
        if ds.has_reference:
            s = regression_stats(ds.frame[ds.reference].values, ds.frame["sk_bandgap_eV"].values)
            row.update({"ref_mean_eV": float(ds.frame[ds.reference].mean()),
                        "MAE_eV": s["mae"], "RMSE_eV": s["rmse"], "pearson_r": s["pearson_r"]})
        else:
            row.update({"ref_mean_eV": np.nan, "MAE_eV": np.nan, "RMSE_eV": np.nan, "pearson_r": np.nan})
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "summary_table.csv", index=False)

    # bar chart
    ref_df = df.dropna(subset=["MAE_eV"]).copy()
    if len(ref_df):
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
        colors = palette(len(ref_df))
        axes[0].bar(ref_df["dataset"], ref_df["MAE_eV"], color=colors)
        axes[0].set_title("MAE vs reference (eV)")
        axes[0].set_ylabel("MAE (eV)")
        axes[0].tick_params(axis="x", rotation=30)

        axes[1].bar(ref_df["dataset"], ref_df["RMSE_eV"], color=colors)
        axes[1].set_title("RMSE vs reference (eV)")
        axes[1].set_ylabel("RMSE (eV)")
        axes[1].tick_params(axis="x", rotation=30)

        axes[2].bar(ref_df["dataset"], ref_df["pearson_r"], color=colors)
        axes[2].set_title("Pearson r")
        axes[2].set_ylabel("r")
        axes[2].set_ylim(-0.2, 1.0)
        axes[2].axhline(0, color="k", lw=0.7)
        axes[2].tick_params(axis="x", rotation=30)

        fig.suptitle("Regression quality on datasets with a DFT reference")
        fig.savefig(OUT / "error_summary.png")
        plt.close(fig)
    return df


def _iter_dos_files(dataset_key: str) -> Iterable[dict]:
    """Stream entries of all_results.json that carry DOS arrays. Returns dicts."""
    # Mapping from dataset key to results json
    mapping = {
        "v03_alex": None,  # too large; skip
        "v04_cccbdb": ROOT / "slako_v04_cccbdb" / "results" / "all_results.json",
        "v05_interface": ROOT / "slako_v05_interface" / "results" / "all_results.json",
        "v06_surface": ROOT / "slako_v06_surface" / "results" / "all_results.json",
        "v07_vacancy": ROOT / "slako_v07_vacancy" / "results" / "all_results.json",
        "v08_supercon": ROOT / "slako_v08_supercon" / "results" / "all_results.json",
        "v09_1d": ROOT / "slako_v09_1d" / "results" / "all_results.json",
    }
    path = mapping.get(dataset_key)
    if path is None or not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        print(f"[warn] could not parse {path}: {exc}")
        return []
    return data


def plot_dos_grid(datasets: list[Dataset]) -> None:
    series = []
    for ds in datasets:
        entries = list(_iter_dos_files(ds.key))
        if not entries:
            continue
        # Sample up to 1000 for average
        if len(entries) > 1000:
            rng = np.random.default_rng(0)
            idx = rng.choice(len(entries), size=1000, replace=False)
            entries = [entries[i] for i in idx]
        dos_list = []
        e_grid = None
        for e in entries:
            dos = e.get("dos_values"); en = e.get("dos_energies")
            if dos is None or en is None:
                continue
            dos = np.asarray(dos, dtype=float)
            en = np.asarray(en, dtype=float)
            if dos.shape != en.shape or len(dos) < 10:
                continue
            if e_grid is None:
                e_grid = en
            if len(en) != len(e_grid):
                try:
                    dos = np.interp(e_grid, en, dos)
                except Exception:
                    continue
            dos_list.append(dos)
        if not dos_list or e_grid is None:
            continue
        mean_dos = np.mean(np.stack(dos_list), axis=0)
        series.append((ds, e_grid, mean_dos, len(dos_list)))

    if not series:
        return
    n = len(series)
    ncol = 3
    nrow = math.ceil(n / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.5 * ncol, 3.3 * nrow), sharex=True, constrained_layout=True)
    axes = np.atleast_1d(axes).flatten()
    for ax, (ds, en, dos, nsamp) in zip(axes, series):
        ax.plot(en, dos, color="#2166ac", lw=1.2)
        ax.axvline(0, color="k", lw=0.6, ls="--")
        ax.set_title(f"{ds.title}\n(avg of {nsamp})", fontsize=9)
        ax.set_xlim(-8, 8)
        ax.set_xlabel("E − E_F (eV)")
        ax.set_ylabel("DOS (arb.)")
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Mean SlakoNet DOS per dataset (Fermi-aligned)")
    fig.savefig(OUT / "dos_average_grid.png")
    plt.close(fig)


def plot_v08_extras(datasets: list[Dataset]) -> None:
    v08 = next((d for d in datasets if d.key == "v08_supercon"), None)
    if v08 is None:
        return
    df = v08.frame.copy()
    df = df[(df["Tc_K"] >= 0) & (df["Tc_K"] < 200)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    axes[0].scatter(df["sk_bandgap_eV"], df["Tc_K"], s=4, alpha=0.35, color="#2166ac")
    axes[0].set_xscale("symlog", linthresh=0.01)
    axes[0].set_xlabel("SlakoNet gap (eV, symlog)")
    axes[0].set_ylabel("Tc (K)")
    axes[0].set_title("Tc vs predicted gap")

    axes[1].scatter(df["dosef"], df["Tc_K"], s=4, alpha=0.35, color="#b2182b")
    axes[1].set_xlabel("DFT DOS(E_F) (states/eV)")
    axes[1].set_ylabel("Tc (K)")
    axes[1].set_title("Tc vs DFT DOS(E_F)")

    axes[2].scatter(df["lambda"], df["Tc_K"], s=4, alpha=0.35, color="#1b7837")
    axes[2].set_xlabel("electron–phonon coupling λ")
    axes[2].set_ylabel("Tc (K)")
    axes[2].set_title("Tc vs λ (Eliashberg)")

    fig.suptitle("v08 supercon: Tc correlations")
    fig.savefig(OUT / "v08_tc_correlations.png")
    plt.close(fig)


def write_markdown(datasets: list[Dataset], summary_df: pd.DataFrame) -> None:
    lines: list[str] = []
    lines.append("# SlakoNet cross-dataset analysis")
    lines.append("")
    loaded_keys = ", ".join(f"`{ds.key}`" for ds in datasets)
    lines.append(f"Aggregates the following SlakoNet runs: {loaded_keys}.")
    lines.append("Re-running `build_analysis.py` automatically picks up any sister `slako_v*`")
    lines.append("project once it drops a `summary.csv` (or `results/all_results.json`).")
    lines.append("")
    lines.append("## Datasets loaded")
    lines.append("")
    lines.append("| key | title | kind | N | reference |")
    lines.append("|-----|-------|------|---:|-----------|")
    for ds in datasets:
        lines.append(f"| {ds.key} | {ds.title} | {ds.kind} | {ds.n:,} | "
                     f"{ds.reference or '—'} |")
    lines.append("")
    lines.append("## Headline metrics (see `summary_table.csv`)")
    lines.append("")
    cols = ["dataset", "N", "sk_mean_eV", "sk_median_eV", "frac_sk_metal",
            "ref_mean_eV", "MAE_eV", "RMSE_eV", "pearson_r"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, row in summary_df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, float) and not math.isnan(v):
                vals.append(f"{v:.3f}")
            elif isinstance(v, float):
                vals.append("—")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    lines.append("")
    lines.append("## Figures")
    lines.append("")
    figs = [
        ("dataset_overview.png",       "Per-dataset N, metallic fraction, SK gap median±IQR"),
        ("gap_distributions.png",      "SK gap histograms (reference overlaid where available)"),
        ("parity_grid.png",            "SK vs reference DFT gap, hexbin parity"),
        ("residual_distributions.png", "SK − reference residual densities"),
        ("error_summary.png",          "MAE / RMSE / Pearson-r vs reference"),
        ("dos_average_grid.png",       "Mean SlakoNet DOS per dataset (Fermi-aligned)"),
        ("v08_tc_correlations.png",    "v08 supercon: Tc vs SK gap, DOS(E_F), λ"),
    ]
    for name, caption in figs:
        if (OUT / name).exists():
            lines.append(f"- `{name}` — {caption}")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- Reference gap for v03 is the paired subset of Alexandria PBE entries that also have an "
        "ALIGNN prediction (~31 k of 48 k). This is the scalar cache produced by "
        "`slako_v03_alex/analyze_alignn.py`.")
    lines.append(
        "- v04 assumes the CCCBDB HOMO/LUMO columns are in **Hartree** (the `hl_gap_hartree_eV` "
        "column in that project's `summary.csv`). The alternative eV assumption yields 0.1–0.3 eV "
        "gaps that are clearly wrong relative to SlakoNet's 4–15 eV predictions.")
    lines.append(
        "- v05/v06 clip the DFT reference to ≥ 0 (v06 subtracts `surf_cbm − surf_vbm`, which goes "
        "slightly negative for metals — treated as gap = 0).")
    lines.append(
        "- v07 (vacancy) and v08 (supercon) have **no DFT band-gap reference**, so they do not "
        "appear in the parity / residual / error plots — only in the distribution and DOS grids.")
    lines.append(
        f"- Metallic threshold used for coverage bars: SlakoNet gap < {METAL_THR:.2f} eV.")
    lines.append("")
    (OUT / "analysis.md").write_text("\n".join(lines))


def main() -> None:
    datasets = load_all()
    if not datasets:
        print("no datasets loaded; nothing to do")
        return
    plot_dataset_overview(datasets)
    plot_gap_distributions(datasets)
    plot_parity_grid(datasets)
    plot_residuals(datasets)
    summary_df = plot_error_summary(datasets)
    plot_dos_grid(datasets)
    plot_v08_extras(datasets)
    write_markdown(datasets, summary_df)
    print(f"[done] wrote outputs into {OUT}")


if __name__ == "__main__":
    main()
