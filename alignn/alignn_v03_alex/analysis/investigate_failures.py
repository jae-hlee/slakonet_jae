"""Investigate why SlakoNet catastrophically fails (sk_bandgap ≈ 0) on ~4909
PBE non-metals. Looks for patterns in composition, size, DOS shape, etc."""
import orjson
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter

OUT_DIR = "analysis"
os.makedirs(OUT_DIR, exist_ok=True)

print("Loading results...", flush=True)
with open("results/all_results.json", "rb") as f:
    data = orjson.loads(f.read())
print(f"Loaded {len(data)} entries", flush=True)

sk = np.array([d["sk_bandgap"] for d in data])
ind = np.array([d["band_gap_ind"] for d in data])
eform = np.array([d["e_form"] for d in data])
formulas = [d["formula"] for d in data]
mat_ids = [d["mat_id"] for d in data]

# Number of atoms + element sets from stored geometry
n_atoms = np.array([len(d["geometry_out"]["elements"]) for d in data])
elem_sets = [set(d["geometry_out"]["elements"]) for d in data]
elem_lists = [d["geometry_out"]["elements"] for d in data]

METAL_THRESH = 0.1

is_nonmetal = ind > 0
is_fail = is_nonmetal & (sk <= METAL_THRESH)   # the 4909
is_ok = is_nonmetal & (sk > METAL_THRESH)      # the 11183
is_pbe_metal = ind == 0.0

print(f"\nPBE non-metals:        {is_nonmetal.sum()}")
print(f"  SK failure (≤0.1):   {is_fail.sum()}")
print(f"  SK predicts gap:     {is_ok.sum()}")
print(f"PBE metals:            {is_pbe_metal.sum()}")

# ── 1. Element frequency analysis ────────────────────────────────────────
print("\n=== Element frequency in failing vs working non-metals ===")
fail_counts = Counter()
ok_counts = Counter()
for i in np.where(is_fail)[0]:
    for e in elem_sets[i]:
        fail_counts[e] += 1
for i in np.where(is_ok)[0]:
    for e in elem_sets[i]:
        ok_counts[e] += 1

n_fail = is_fail.sum()
n_ok = is_ok.sum()

rows = []
for e in set(fail_counts) | set(ok_counts):
    f_frac = fail_counts[e] / n_fail
    o_frac = ok_counts[e] / n_ok
    if fail_counts[e] + ok_counts[e] < 50:
        continue  # skip rare elements
    enrichment = f_frac / o_frac if o_frac > 0 else float("inf")
    rows.append((e, fail_counts[e], ok_counts[e], f_frac, o_frac, enrichment))

rows.sort(key=lambda r: -r[5])
print(f"{'Elem':>5} {'Fail':>6} {'OK':>6} {'Fail%':>7} {'OK%':>7} {'Enrich':>8}")
for r in rows:
    print(f"{r[0]:>5} {r[1]:>6d} {r[2]:>6d} {r[3]*100:>6.2f}% {r[4]*100:>6.2f}% {r[5]:>8.2f}")

# ── 2. Structure size ────────────────────────────────────────────────────
print("\n=== Structure size (# atoms in unit cell) ===")
print(f"  Failing:  mean={n_atoms[is_fail].mean():.1f}  median={np.median(n_atoms[is_fail]):.0f}  "
      f"min={n_atoms[is_fail].min()}  max={n_atoms[is_fail].max()}")
print(f"  Working:  mean={n_atoms[is_ok].mean():.1f}  median={np.median(n_atoms[is_ok]):.0f}  "
      f"min={n_atoms[is_ok].min()}  max={n_atoms[is_ok].max()}")

# ── 3. Formation energy ──────────────────────────────────────────────────
print("\n=== Formation energy (eV/atom) ===")
print(f"  Failing:  mean={eform[is_fail].mean():.3f}  median={np.median(eform[is_fail]):.3f}")
print(f"  Working:  mean={eform[is_ok].mean():.3f}  median={np.median(eform[is_ok]):.3f}")

# ── 4. DOS sanity — is SlakoNet producing a nonzero DOS at all? ──────────
print("\n=== DOS integral (sanity check — is DOS nonzero for failures?) ===")
fail_dos_sums = []
ok_dos_sums = []
fail_idx = np.where(is_fail)[0]
ok_idx = np.where(is_ok)[0]

# Sample 500 from each for speed
rng = np.random.default_rng(0)
for i in rng.choice(fail_idx, size=min(500, len(fail_idx)), replace=False):
    d = np.array(data[i]["dos_values"])
    fail_dos_sums.append(d.sum())
for i in rng.choice(ok_idx, size=min(500, len(ok_idx)), replace=False):
    d = np.array(data[i]["dos_values"])
    ok_dos_sums.append(d.sum())

print(f"  Failing DOS integral: mean={np.mean(fail_dos_sums):.2f}  median={np.median(fail_dos_sums):.2f}")
print(f"  Working DOS integral: mean={np.mean(ok_dos_sums):.2f}  median={np.median(ok_dos_sums):.2f}")

# ── 5. Average DOS shape: failing non-metals vs working non-metals ──────
print("\n=== Averaging DOS shapes (failing vs working non-metals) ===")
e_grid = np.array(data[0]["dos_energies"])
fail_avg = np.zeros(len(e_grid))
ok_avg = np.zeros(len(e_grid))
for i in fail_idx:
    fail_avg += np.array(data[i]["dos_values"])
fail_avg /= len(fail_idx)
for i in ok_idx:
    ok_avg += np.array(data[i]["dos_values"])
ok_avg /= len(ok_idx)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(e_grid, fail_avg, label=f"Failing non-metals (N={len(fail_idx)})", color="tab:red")
ax.plot(e_grid, ok_avg, label=f"Working non-metals (N={len(ok_idx)})", color="tab:blue")
ax.axvline(0, color="k", ls=":", lw=0.8, label="E_F")
ax.set_xlabel("Energy relative to E_F (eV)")
ax.set_ylabel("Average DOS (arb.)")
ax.set_title("Average SlakoNet DOS: PBE non-metals where SlakoNet collapses vs not")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "failure_dos_comparison.png"), dpi=200)
plt.close(fig)
print(f"  Saved: {OUT_DIR}/failure_dos_comparison.png")

# ── 6. Element count distribution (binary / ternary / ...) ──────────────
print("\n=== Number of distinct elements per formula ===")
fail_nel = [len(elem_sets[i]) for i in fail_idx]
ok_nel = [len(elem_sets[i]) for i in ok_idx]
print(f"  Failing:  {Counter(fail_nel)}")
print(f"  Working:  {Counter(ok_nel)}")

# ── 7. Top failing formulas ──────────────────────────────────────────────
print("\n=== 20 failing examples with largest PBE gap (clearest failures) ===")
fail_sorted = sorted(fail_idx, key=lambda i: -ind[i])[:20]
print(f"{'mat_id':<20} {'formula':<20} {'PBE_gap':>8} {'SK_gap':>8} {'natoms':>6}")
for i in fail_sorted:
    print(f"{mat_ids[i]:<20} {formulas[i]:<20} {ind[i]:>8.3f} {sk[i]:>8.4f} {n_atoms[i]:>6d}")

# ── 8. Histogram of # atoms: fail vs ok ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
bins = np.arange(0, max(n_atoms.max(), 50) + 2)
axes[0].hist(n_atoms[is_fail], bins=bins, alpha=0.6, label="failing", color="tab:red")
axes[0].hist(n_atoms[is_ok], bins=bins, alpha=0.6, label="working", color="tab:blue")
axes[0].set_xlabel("Atoms in unit cell")
axes[0].set_ylabel("Count")
axes[0].set_title("Structure size")
axes[0].set_xlim(0, 60)
axes[0].legend()

axes[1].hist(eform[is_fail], bins=80, alpha=0.6, label="failing", color="tab:red")
axes[1].hist(eform[is_ok], bins=80, alpha=0.6, label="working", color="tab:blue")
axes[1].set_xlabel("Formation energy (eV/atom)")
axes[1].set_ylabel("Count")
axes[1].set_title("Formation energy")
axes[1].legend()

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "failure_structural.png"), dpi=200)
plt.close(fig)
print(f"\nSaved: {OUT_DIR}/failure_structural.png")

print("\nDone.")
