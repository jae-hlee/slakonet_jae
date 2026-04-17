"""Check whether the 'PBE metal' subset is contaminated by open-shell
transition-metal compounds, which would suggest hidden false negatives
(SlakoNet silently collapsing to zero for materials PBE also reports as
metal, making the confusion matrix's metal cell untrustworthy)."""
import orjson
import numpy as np
from collections import Counter

print("Loading results...", flush=True)
with open("results/all_results.json", "rb") as f:
    data = orjson.loads(f.read())
print(f"Loaded {len(data)} entries", flush=True)

sk = np.array([d["sk_bandgap"] for d in data])
ind = np.array([d["band_gap_ind"] for d in data])
elem_sets = [set(d["geometry_out"]["elements"]) for d in data]

is_pbe_metal = ind == 0.0
is_pbe_nonmetal = ~is_pbe_metal
is_fail_nm = is_pbe_nonmetal & (sk <= 0.1)  # the 4909

# Problem elements flagged in analysis.md
PROBLEM_TMS = {"Mn", "Cr", "Ni", "Co", "Fe", "Ru", "V", "Tc", "Mo"}

def has_tm(i):
    return bool(elem_sets[i] & PROBLEM_TMS)

pbe_metal_idx = np.where(is_pbe_metal)[0]
pbe_nonmetal_idx = np.where(is_pbe_nonmetal)[0]
fail_nm_idx = np.where(is_fail_nm)[0]

# ── Fraction of each group containing a problem TM ──────────────────────
n_metal_tm = sum(has_tm(i) for i in pbe_metal_idx)
n_nm_tm = sum(has_tm(i) for i in pbe_nonmetal_idx)
n_fail_tm = sum(has_tm(i) for i in fail_nm_idx)

print(f"\n=== Fraction of each group containing >=1 open-shell TM ===")
print(f"  PBE metal       (N={len(pbe_metal_idx):5d}): "
      f"{n_metal_tm:5d} with TM  ({100*n_metal_tm/len(pbe_metal_idx):.1f}%)")
print(f"  PBE non-metal   (N={len(pbe_nonmetal_idx):5d}): "
      f"{n_nm_tm:5d} with TM  ({100*n_nm_tm/len(pbe_nonmetal_idx):.1f}%)")
print(f"  Failing (FN)    (N={len(fail_nm_idx):5d}): "
      f"{n_fail_tm:5d} with TM  ({100*n_fail_tm/len(fail_nm_idx):.1f}%)")

# ── Per-element frequency in PBE-metal vs PBE-nonmetal ──────────────────
print(f"\n=== Per-element prevalence in PBE metals (problem TMs only) ===")
metal_counts = Counter()
nm_counts = Counter()
for i in pbe_metal_idx:
    for e in elem_sets[i]:
        metal_counts[e] += 1
for i in pbe_nonmetal_idx:
    for e in elem_sets[i]:
        nm_counts[e] += 1

print(f"{'Elem':>5} {'PBE_metal':>10} {'PBE_metal%':>11} {'PBE_nm':>8} {'PBE_nm%':>9}")
for e in sorted(PROBLEM_TMS):
    mf = metal_counts[e] / len(pbe_metal_idx)
    nf = nm_counts[e] / len(pbe_nonmetal_idx)
    print(f"{e:>5} {metal_counts[e]:>10d} {100*mf:>10.2f}% "
          f"{nm_counts[e]:>8d} {100*nf:>8.2f}%")

# ── Average SK DOS right at E_F ─────────────────────────────────────────
print(f"\n=== Average SK DOS at E_F (closest bin to 0) ===")
e_grid = np.array(data[0]["dos_energies"])
ef_bin = int(np.argmin(np.abs(e_grid)))

def avg_dos_at_ef(indices):
    vals = [data[i]["dos_values"][ef_bin] for i in indices]
    return np.mean(vals), np.median(vals)

pbe_metal_tm = [i for i in pbe_metal_idx if has_tm(i)]
pbe_metal_notm = [i for i in pbe_metal_idx if not has_tm(i)]
working_nm = [i for i in pbe_nonmetal_idx if sk[i] > 0.1]

for label, idx in [
    ("PBE metal, has TM     ", pbe_metal_tm),
    ("PBE metal, no TM      ", pbe_metal_notm),
    ("Failing non-metal (FN)", list(fail_nm_idx)),
    ("Working non-metal     ", working_nm),
]:
    if not idx:
        continue
    m, med = avg_dos_at_ef(idx)
    print(f"  {label} (N={len(idx):5d}): mean DOS@E_F = {m:7.3f}  median = {med:7.3f}")

# ── How many of the "PBE metal, has TM" have SK also predicting ~0? ─────
tm_metal_sk_zero = sum(1 for i in pbe_metal_tm if sk[i] <= 0.1)
notm_metal_sk_zero = sum(1 for i in pbe_metal_notm if sk[i] <= 0.1)

print(f"\n=== SK gap <=0.1 eV rate, split by TM content ===")
print(f"  PBE metal WITH  problem TM: {tm_metal_sk_zero}/{len(pbe_metal_tm)} "
      f"({100*tm_metal_sk_zero/max(1,len(pbe_metal_tm)):.1f}% predicted metal by SK)")
print(f"  PBE metal WITHOUT problem TM: {notm_metal_sk_zero}/{len(pbe_metal_notm)} "
      f"({100*notm_metal_sk_zero/max(1,len(pbe_metal_notm)):.1f}% predicted metal by SK)")
