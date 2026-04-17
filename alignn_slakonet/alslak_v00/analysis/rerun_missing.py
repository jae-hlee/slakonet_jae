"""Re-run SlakoNet on the missing structures with verbose error logging.

Reads missing_entries.json (produced by find_missing.py) and runs inference,
logging exactly why each structure fails (prep error, timeout, SlakoNet
success=False, or unexpected exception) to missing_failures.log.
"""
import json
import os
import traceback
import torch
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, TimeoutError
from tqdm import tqdm
from jarvis.core.atoms import Atoms
from jarvis.core.kpoints import Kpoints3D as Kpoints
from slakonet.optim import (
    MultiElementSkfParameterOptimizer,
    kpts_to_klines,
    default_model,
)
from slakonet.atoms import Geometry
from slakonet.main import generate_shell_dict_upto_Z65
from jarvis.db.jsonutils import dumpjson

TIMEOUT_SECONDS = 300  # longer timeout on the rerun


def prepare_inputs(item):
    atoms = Atoms.from_dict(item['atoms'])
    geometry = Geometry.from_ase_atoms([atoms.ase_converter()])
    kpoints = Kpoints().kpath(atoms, line_density=20)
    klines = kpts_to_klines(kpoints.kpts, default_points=2)
    return {'geometry': geometry, 'klines': klines, 'item': item}


def run_inference(prepared, model, shell_dict, device):
    with torch.no_grad():
        properties, success = model.compute_multi_element_properties(
            geometry=prepared['geometry'],
            shell_dict=shell_dict,
            klines=prepared['klines'],
            get_fermi=True,
            with_eigenvectors=True,
            device=device,
        )
    if not success:
        return None
    item = prepared['item']
    return {
        'mat_id': item['mat_id'],
        'formula': item['formula'],
        'band_gap_ind': item['band_gap_ind'],
        'band_gap_dir': item['band_gap_dir'],
        'e_form': item['e_form'],
        'sk_bandgap': float(properties["bandgap"].detach().cpu()),
        'dos_values': properties["dos_values_tensor"].detach().cpu().numpy().tolist(),
        'dos_energies': properties["dos_energy_grid_tensor"].detach().cpu().numpy().tolist(),
        'geometry_out': item['atoms'],
    }


def run_with_timeout(fn, *args, timeout=TIMEOUT_SECONDS):
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(fn, *args)
        return future.result(timeout=timeout)


def gpu_worker(gpu_id, chunk, shell_dict):
    device = f"cuda:{gpu_id}"
    model = default_model().float().eval().to(device)
    results = []
    failures = []  # list of (mat_id, stage, error_str)

    for item in tqdm(chunk, desc=f"GPU {gpu_id}", position=gpu_id):
        mat_id = item['mat_id']
        fname = os.path.join("results_rerun", f"{mat_id}.json")
        if os.path.exists(fname):
            continue

        # Stage 1: prep
        try:
            p = run_with_timeout(prepare_inputs, item, timeout=60)
        except TimeoutError:
            failures.append((mat_id, "PREP_TIMEOUT", "prep exceeded 60s"))
            continue
        except Exception as e:
            failures.append((mat_id, "PREP_FAIL", f"{type(e).__name__}: {e}"))
            continue

        # Stage 2: inference
        try:
            info = run_with_timeout(run_inference, p, model, shell_dict, device)
        except TimeoutError:
            failures.append((mat_id, "INFER_TIMEOUT", f"exceeded {TIMEOUT_SECONDS}s"))
            continue
        except Exception as e:
            tb = traceback.format_exc().strip().split("\n")[-1]
            failures.append((mat_id, "INFER_FAIL", f"{type(e).__name__}: {tb}"))
            continue

        if info is None:
            failures.append((mat_id, "SK_SUCCESS_FALSE", "compute_multi_element_properties returned success=False"))
            continue

        dumpjson(data=info, filename=fname)
        results.append(info)

    return results, failures


def main():
    with open("missing_entries.json") as f:
        missing = json.load(f)
    print(f"Loaded {len(missing)} missing entries", flush=True)

    os.makedirs("results_rerun", exist_ok=True)
    shell_dict = generate_shell_dict_upto_Z65()

    num_gpus = torch.cuda.device_count()
    print(f"Using {num_gpus} GPUs", flush=True)

    if num_gpus > 1:
        chunks = [missing[g::num_gpus] for g in range(num_gpus)]
        import torch.multiprocessing as mp
        mp.set_start_method('spawn', force=True)
        all_results = []
        all_failures = []
        with ProcessPoolExecutor(max_workers=num_gpus) as pool:
            futures = [pool.submit(gpu_worker, g, chunks[g], shell_dict) for g in range(num_gpus)]
            for f in futures:
                res, fails = f.result()
                all_results.extend(res)
                all_failures.extend(fails)
    else:
        all_results, all_failures = gpu_worker(0, missing, shell_dict)

    print(f"\nRerun completed: {len(all_results)} / {len(missing)} successful")
    print(f"Failures: {len(all_failures)}")

    # Write failure log
    with open("missing_failures.log", "w") as f:
        f.write(f"# Rerun of {len(missing)} missing structures\n")
        f.write(f"# Successful: {len(all_results)}\n")
        f.write(f"# Failed: {len(all_failures)}\n\n")
        # Summary by stage
        from collections import Counter
        stage_counts = Counter(stage for _, stage, _ in all_failures)
        f.write("## Failures by stage\n")
        for stage, count in stage_counts.most_common():
            f.write(f"  {stage:20s} {count}\n")
        f.write("\n## Detailed failures\n")
        for mat_id, stage, err in all_failures:
            f.write(f"{mat_id}\t{stage}\t{err}\n")

    print("\nFailure breakdown:")
    from collections import Counter
    for stage, count in Counter(s for _, s, _ in all_failures).most_common():
        print(f"  {stage:20s} {count}")
    print(f"\nDetailed log: missing_failures.log")

    # Save combined rerun results
    if all_results:
        dumpjson(data=all_results, filename="results_rerun/all_rerun_results.json")


if __name__ == '__main__':
    main()
