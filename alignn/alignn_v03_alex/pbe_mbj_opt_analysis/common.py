"""Shared helpers for the SlakoNet DB rebuttal analyses (leakage audit, re-scoring, funnel)."""

import gzip
import json
import os
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
MERGED_PREDICTIONS = os.path.join(HERE, "merged_predictions.json")
ALEXANDRIA_ZIP = os.path.join(
    REPO, "slakonet/slako_v11_alexwz/alexandria_pbe_3d_2024.10.1_jarvis_tools.json.zip"
)
OUT_DIR = HERE

GAP_KEYS = {
    "SlakoNet": "slakonet",
    "PBE-ALIGNN": "alignn_pbe",
    "mBJ-ALIGNN": "alignn_mbj",
    "OptB88vdW-ALIGNN": "alignn_optb",
}


def load_merged(path=MERGED_PREDICTIONS):
    with open(path) as f:
        rows = json.load(f)
    assert isinstance(rows, list) and rows and "mat_id" in rows[0]
    return rows


def stream_alexandria_entries(zip_path=ALEXANDRIA_ZIP, chunk_bytes=1 << 24):
    """Yield entries from the (huge) Alexandria snapshot without loading it whole.

    The archive holds a single JSON array of objects; we incrementally
    raw_decode objects out of a growing text buffer.
    """
    dec = json.JSONDecoder()
    skip = " \t\r\n,"
    z = zipfile.ZipFile(zip_path)
    with z.open(z.namelist()[0]) as f:
        buf, idx, started, eof = "", 0, False, False
        while not eof:
            chunk = f.read(chunk_bytes)
            if chunk:
                buf = buf[idx:] + chunk.decode("utf-8")  # one compaction per chunk
                idx = 0
            else:
                eof = True
            if not started:
                while idx < len(buf) and buf[idx] in " \t\r\n":
                    idx += 1
                if idx < len(buf) and buf[idx] == "[":
                    idx += 1
                    started = True
            while True:
                while idx < len(buf) and buf[idx] in skip:
                    idx += 1
                if idx >= len(buf) or buf[idx] == "]":
                    break
                try:
                    obj, idx = dec.raw_decode(buf, idx)
                except json.JSONDecodeError:
                    break  # incomplete object - need more data
                yield obj


def load_benchmark_structures(mat_ids, zip_path=ALEXANDRIA_ZIP, cache=None):
    """Return {mat_id: entry} for the requested ids, using a gzip cache after first pass."""
    cache = cache or os.path.join(OUT_DIR, "benchmark_structures.json.gz")
    if os.path.exists(cache):
        with gzip.open(cache, "rt") as f:
            found = json.load(f)
        missing = set(mat_ids) - set(found)
        if not missing:
            return {k: found[k] for k in mat_ids if k in found}
        print(f"cache incomplete ({len(missing)} missing) - rescanning snapshot")
    wanted = set(mat_ids)
    found = {}
    n = 0
    for entry in stream_alexandria_entries(zip_path):
        n += 1
        if n % 200000 == 0:
            print(f"  scanned {n:,} entries, found {len(found):,}/{len(wanted):,}", flush=True)
        mid = entry.get("mat_id") or entry.get("id")
        if mid in wanted:
            found[mid] = {
                "mat_id": mid,
                "formula": entry.get("formula"),
                "spg": entry.get("spg"),
                "atoms": entry["atoms"],
            }
            if len(found) == len(wanted):
                break
    os.makedirs(OUT_DIR, exist_ok=True)
    with gzip.open(cache, "wt") as f:
        json.dump(found, f)
    print(f"cached {len(found):,} benchmark structures -> {cache}")
    return found


def jarvis_to_pmg(atoms_dict):
    from jarvis.core.atoms import Atoms

    return Atoms.from_dict(atoms_dict).pymatgen_converter()


def reduced_formula(formula_str):
    from pymatgen.core import Composition

    try:
        return Composition(formula_str).reduced_formula
    except Exception:
        return None


# ---------------- metrics ----------------

def mae(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean(np.abs(y - p)))


def rmse(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.sqrt(np.mean((y - p) ** 2)))


def r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def bootstrap_ci(y, p, fn=mae, n_boot=1000, seed=0):
    """95% percentile CI of fn(y, p) under paired resampling."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    rng = np.random.default_rng(seed)
    stats = []
    idx = np.arange(len(y))
    for _ in range(n_boot):
        s = rng.choice(idx, size=len(idx), replace=True)
        stats.append(fn(y[s], p[s]))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def split_masks(rows):
    """Metal (pbe_ref <= 0) vs non-metal (pbe_ref > 0) masks over merged rows."""
    ref = np.array([r["pbe_ref"] for r in rows], float)
    return ref <= 0.0, ref > 0.0


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)
