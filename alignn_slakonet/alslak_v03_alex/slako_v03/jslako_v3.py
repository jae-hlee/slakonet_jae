import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import signal
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, TimeoutError
from tqdm import tqdm
from jarvis.core.atoms import Atoms
from jarvis.core.kpoints import Kpoints3D as Kpoints
from slakonet.optim import (
    MultiElementSkfParameterOptimizer,
    get_atoms,
    kpts_to_klines,
    default_model,
)
from slakonet.atoms import Geometry
from slakonet.main import generate_shell_dict_upto_Z65
from jarvis.db.figshare import data
from jarvis.db.jsonutils import dumpjson

TIMEOUT_SECONDS = 180  # 3 minutes max per structure

ALLOWED_SYMBOLS = {
    'H','He','Li','Be','B','C','N','O','F','Ne',
    'Na','Mg','Al','Si','P','S','Cl','Ar',
    'K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn',
    'Ga','Ge','As','Se','Br','Kr',
    'Rb','Sr','Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd',
    'In','Sn','Sb','Te','I','Xe',
    'Cs','Ba','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb',
}


def all_elements_supported(elements):
    """Check that every element in the structure has Z <= 65."""
    return all(e in ALLOWED_SYMBOLS for e in elements)


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
    bandgap = float(properties["bandgap"].detach().cpu())
    return {
        'mat_id': item['mat_id'],
        'formula': item['formula'],
        'band_gap_ind': item['band_gap_ind'],
        'band_gap_dir': item['band_gap_dir'],
        'e_form': item['e_form'],
        'sk_bandgap': bandgap,
        'dos_values': properties["dos_values_tensor"].detach().cpu().numpy().tolist(),
        'dos_energies': properties["dos_energy_grid_tensor"].detach().cpu().numpy().tolist(),
        'geometry_out': item['atoms'],
    }


def run_inference_with_timeout(prepared, model, shell_dict, device, timeout=TIMEOUT_SECONDS):
    """Run inference in a thread with a timeout."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(run_inference, prepared, model, shell_dict, device)
        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            return "TIMEOUT"


def run_single_gpu(valid_entries, shell_dict):
    model = default_model().float().eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}", flush=True)

    mem = []
    n_timeout = 0
    n_prep_fail = 0
    for item in tqdm(valid_entries, desc="Processing"):
        try:
            fname = os.path.join("results", f"{item['mat_id']}.json")
            if os.path.exists(fname):
                continue

            try:
                p = prepare_inputs(item)
            except Exception as e:
                n_prep_fail += 1
                tqdm.write(f"PREP FAIL: {item['mat_id']} {item['formula']} — {e}")
                continue

            info = run_inference_with_timeout(p, model, shell_dict, device)
            if info == "TIMEOUT":
                n_timeout += 1
                tqdm.write(
                    f"TIMEOUT ({TIMEOUT_SECONDS}s): {item['mat_id']} — skipping (timeouts so far: {n_timeout})"
                )
                continue
            if info:
                tqdm.write(
                    f"{info['mat_id']} {info['formula']} pbe_gap={info['band_gap_ind']} sk={info['sk_bandgap']:.4f} n={len(mem)}"
                )
                dumpjson(data=info, filename=fname)
                mem.append(info)
        except Exception as e:
            tqdm.write(f"ERROR: {item['mat_id']} — {e}")
    print(f"\nTotal timeouts: {n_timeout}", flush=True)
    print(f"Total prep failures: {n_prep_fail}", flush=True)
    return mem


def gpu_worker(gpu_id, chunk, shell_dict):
    device = f"cuda:{gpu_id}"
    model = default_model().float().eval().to(device)
    results = []
    n_timeout = 0
    for item in tqdm(chunk, desc=f"GPU {gpu_id}", position=gpu_id):
        try:
            fname = os.path.join("results", f"{item['mat_id']}.json")
            if os.path.exists(fname):
                continue

            p = prepare_inputs(item)
            info = run_inference_with_timeout(p, model, shell_dict, device)
            if info == "TIMEOUT":
                n_timeout += 1
                tqdm.write(
                    f"[GPU {gpu_id}] TIMEOUT ({TIMEOUT_SECONDS}s): {item['mat_id']} — skipping (timeouts: {n_timeout})"
                )
                continue
            if info:
                tqdm.write(
                    f"[GPU {gpu_id}] {info['mat_id']} {info['formula']} pbe_gap={info['band_gap_ind']} sk={info['sk_bandgap']:.4f}"
                )
                dumpjson(data=info, filename=fname)
                results.append(info)
        except Exception:
            pass
    print(f"[GPU {gpu_id}] Total timeouts: {n_timeout}")
    return results


def run_multi_gpu(valid_entries, shell_dict):
    num_gpus = torch.cuda.device_count()
    print(f"Using {num_gpus} GPUs")
    chunks = [valid_entries[g::num_gpus] for g in range(num_gpus)]

    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)

    with ProcessPoolExecutor(max_workers=num_gpus) as pool:
        futures = [
            pool.submit(gpu_worker, g, chunks[g], shell_dict)
            for g in range(num_gpus)
        ]
        mem = []
        for f in tqdm(futures, desc="Collecting GPU results"):
            mem.extend(f.result())
    return mem


if __name__ == '__main__':
    import json
    import zipfile

    local_zip = 'alexandria_pbe_3d_2024.10.1_jarvis_tools.json.zip'
    print(f"Loading from local file: {local_zip}")
    with zipfile.ZipFile(local_zip) as zf:
        json_name = zf.namelist()[0]
        print(f"Reading {json_name} from zip...")
        dft_3d = json.loads(zf.read(json_name))
    print(f"Loaded {len(dft_3d)} entries", flush=True)

    RESULTS_DIR = "results"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Precompute shell_dict once
    shell_dict = generate_shell_dict_upto_Z65()

    # Filter: e_above_hull == 0, all elements Z <= 65, and json doesn't already exist
    valid_entries = []
    skipped_existing = 0
    skipped_hull = 0
    skipped_elements = 0
    for i in tqdm(dft_3d, desc="Filtering entries"):
        if i['e_above_hull'] != 0:
            skipped_hull += 1
            continue
        if not all_elements_supported(i['elements']):
            skipped_elements += 1
            continue
        fname = os.path.join(RESULTS_DIR, f"{i['mat_id']}.json")
        if os.path.exists(fname):
            skipped_existing += 1
            continue
        valid_entries.append(i)

    print(f"Skipped (not on hull): {skipped_hull}", flush=True)
    print(f"Skipped (elements Z > 65): {skipped_elements}", flush=True)
    print(f"Skipped (already computed): {skipped_existing}", flush=True)
    print(f"Total valid entries to run: {len(valid_entries)}", flush=True)

    if torch.cuda.device_count() > 1:
        mem = run_multi_gpu(valid_entries, shell_dict)
    else:
        mem = run_single_gpu(valid_entries, shell_dict)

    print(f"\nCompleted: {len(mem)} structures", flush=True)
    dumpjson(data=mem, filename=os.path.join(RESULTS_DIR, 'all_results.json'))
