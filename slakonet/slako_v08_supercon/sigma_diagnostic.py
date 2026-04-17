"""Sigma-broadening diagnostic — RUN ON THE CLUSTER (full slakonet env).

Hypothesis: SlakoNet under-predicts dos(Ef) because the hardcoded Gaussian
broadening (sigma = 0.1 eV in SimpleDftb.calculate_dos) is too wide and
flattens sharp Fermi-level peaks.

What this does:
  1. Reads results/all_results.json to identify worst-bias structures
     (ones where the original sigma=0.1 SlakoNet dos(Ef) is far below
     the DFT dosef in alex_supercon).
  2. Monkey-patches slakonet.main.SimpleDftb.calculate_dos to force a
     chosen sigma value.
  3. Re-runs SlakoNet for several sigma values on the worst-bias subset.
  4. Records the new dos(Ef) (linear interp of dos_values at E=0) for
     each (id, sigma).
  5. Saves analysis/sigma_diagnostic.json — pull this back to the laptop
     and run sigma_diagnostic_plot.py to visualize.

If MAE drops monotonically as sigma → 0.02, broadening is the cause.
If MAE is flat or increases at small sigma, the under-binding is
structural (basis size / k-mesh) and σ is not the lever.
"""
import json
import re
import os
import functools
import numpy as np
import torch

import slakonet.main as sk_main
from slakonet.optim import default_model, kpts_to_klines
from slakonet.atoms import Geometry
from slakonet.main import generate_shell_dict_upto_Z65
from jarvis.core.atoms import Atoms
from jarvis.core.kpoints import Kpoints3D as Kpoints

SIGMAS = [0.02, 0.05, 0.10, 0.20, 0.40]   # 0.10 reproduces baseline
N_WORST = 30
OUT = "analysis/sigma_diagnostic.json"


def patch_sigma(new_sigma):
    """Force SimpleDftb.calculate_dos to use new_sigma regardless of caller."""
    if not hasattr(sk_main.SimpleDftb, "_orig_calculate_dos"):
        sk_main.SimpleDftb._orig_calculate_dos = sk_main.SimpleDftb.calculate_dos

    @functools.wraps(sk_main.SimpleDftb._orig_calculate_dos)
    def wrapped(self, *args, **kwargs):
        kwargs["sigma"] = new_sigma
        return sk_main.SimpleDftb._orig_calculate_dos(self, *args, **kwargs)
    sk_main.SimpleDftb.calculate_dos = wrapped


def sk_dosef_from(properties):
    e = properties["dos_energy_grid_tensor"].detach().cpu().numpy()
    v = properties["dos_values_tensor"].detach().cpu().numpy()
    return float(np.interp(0.0, e, v))


def main():
    print("Loading results/all_results.json ...", flush=True)
    with open("results/all_results.json") as f:
        results = json.loads(re.sub(r"\bNaN\b", "null", f.read()))

    rows = []
    for d in results:
        e = np.asarray(d["dos_energies"], dtype=float)
        v = np.asarray(d["dos_values"],   dtype=float)
        if e.size == 0 or not np.all(np.isfinite(v)):
            continue
        sk0  = float(np.interp(0.0, e, v))
        dft  = d["dosef"]
        if not (np.isfinite(sk0) and np.isfinite(dft)):
            continue
        rows.append((d["id"], dft, sk0, sk0 - dft, d["atoms"]))

    rows.sort(key=lambda r: r[3])              # most-underbinding first
    worst = rows[:N_WORST]
    print(f"Selected {len(worst)} worst-bias structures (most negative residual).")
    for i, (id_, dft, sk, r, _) in enumerate(worst[:10], 1):
        print(f"  {i:2d}. {id_:18s}  DFT={dft:7.3f}  SK(σ=0.1)={sk:7.3f}  Δ={r:+7.3f}")

    shell_dict = generate_shell_dict_upto_Z65()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}\n", flush=True)

    out = {"sigmas": SIGMAS, "structures": []}
    for sigma in SIGMAS:
        patch_sigma(sigma)
        model = default_model().float().eval()
        print(f"=== σ = {sigma} eV ===", flush=True)
        for id_, dft, sk0, _, atoms_dict in worst:
            try:
                atoms = Atoms.from_dict(atoms_dict)
                geom = Geometry.from_ase_atoms([atoms.ase_converter()])
                kpoints = Kpoints().kpath(atoms, line_density=20)
                klines = kpts_to_klines(kpoints.kpts, default_points=2)
                with torch.no_grad():
                    props, ok = model.compute_multi_element_properties(
                        geometry=geom, shell_dict=shell_dict, klines=klines,
                        get_fermi=True, with_eigenvectors=True, device=device,
                    )
                if not ok:
                    sk_new = None
                else:
                    sk_new = sk_dosef_from(props)
            except Exception as e:
                print(f"  FAIL {id_}: {e}")
                sk_new = None
            print(f"  {id_:18s}  DFT={dft:7.3f}  σ={sigma:.2f} → SK={sk_new}", flush=True)
            out["structures"].append({
                "id": id_, "sigma": sigma,
                "dft_dosef": dft, "sk_dosef_baseline": sk0, "sk_dosef": sk_new,
            })

    os.makedirs("analysis", exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
