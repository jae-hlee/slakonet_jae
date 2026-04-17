"""Plot the result of sigma_diagnostic.py.

Reads analysis/sigma_diagnostic.json (produced on the cluster) and shows:
  • MAE vs sigma across the worst-bias subset
  • Per-structure dos(Ef) vs sigma trajectories
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("analysis/sigma_diagnostic.json") as f:
    data = json.load(f)

sigmas = data["sigmas"]
rows = data["structures"]

by_id = {}
for r in rows:
    by_id.setdefault(r["id"], {})[r["sigma"]] = r["sk_dosef"]
dft = {r["id"]: r["dft_dosef"] for r in rows}

mae_per_sigma = []
for s in sigmas:
    errs = []
    for id_, dft_v in dft.items():
        sk = by_id[id_].get(s)
        if sk is None or not np.isfinite(sk):
            continue
        errs.append(abs(sk - dft_v))
    mae_per_sigma.append(np.mean(errs) if errs else np.nan)

print("σ (eV)    MAE (states/eV)")
for s, m in zip(sigmas, mae_per_sigma):
    print(f"  {s:.2f}      {m:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# (a) MAE vs sigma
ax = axes[0]
ax.plot(sigmas, mae_per_sigma, "o-", color="tab:blue", lw=2)
ax.axvline(0.10, color="k", ls="--", lw=0.8, label="baseline σ = 0.10")
ax.set_xlabel("Gaussian broadening σ (eV)")
ax.set_ylabel(f"MAE on worst-bias subset (N={len(dft)})")
ax.set_title("dos(Ef) MAE vs σ")
ax.legend()

# (b) per-structure trajectories
ax = axes[1]
for id_, dft_v in dft.items():
    ys = [by_id[id_].get(s) for s in sigmas]
    ax.plot(sigmas, ys, "o-", alpha=0.5, color="tab:blue", lw=1)
    ax.axhline(dft_v, color="tab:red", ls=":", lw=0.5, alpha=0.4)
ax.set_xlabel("σ (eV)")
ax.set_ylabel("SlakoNet dos(Ef)  (states/eV)")
ax.set_title("Per-structure: dotted red = DFT target, blue = SlakoNet vs σ")

fig.tight_layout()
fig.savefig("analysis/sigma_diagnostic.png", dpi=180)
print("\nWrote analysis/sigma_diagnostic.png")
