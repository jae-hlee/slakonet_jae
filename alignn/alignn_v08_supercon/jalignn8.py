"""
Predict band gaps with ALIGNN on Alexandria alex_supercon candidates.

Mirrors slako_v08_supercon's filter (Z <= 65). The dataset has no DFT band gap
reference — it carries electron-phonon descriptors (Tc, λ, dos(Ef), wlog,
Debye-T) instead. ALIGNN's predicted gap should be near 0 for true superconductor
candidates.

Outputs: results/alignn_predictions.json
  Each entry: {id, formula, alignn_bandgap, Tc, la, dosef, wlog, debye}
"""

import json
import os
import zipfile
import torch
from tqdm import tqdm
from alignn.pretrained import get_figshare_model
from torch.utils.data import DataLoader
from alignn.dataset import get_torch_dataset

MODEL_NAME = "mp_gappbe_alignn"
INPUT_ZIP = "alex_supercon.json.zip"
OUTPUT_FILE = "results/alignn_predictions.json"
CHECKPOINT_DIR = "results/alignn_checkpoints"
BATCH_SIZE = 32
NUM_WORKERS = 0
CHECKPOINT_EVERY = 500

ID_KEY = "id"
ATOMS_KEY = "atoms"

ALLOWED_SYMBOLS = {
    'H','He','Li','Be','B','C','N','O','F','Ne',
    'Na','Mg','Al','Si','P','S','Cl','Ar',
    'K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn',
    'Ga','Ge','As','Se','Br','Kr',
    'Rb','Sr','Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd',
    'In','Sn','Sb','Te','I','Xe',
    'Cs','Ba','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb',
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_entries(path):
    print(f"Loading {path}...", flush=True)
    with zipfile.ZipFile(path) as zf:
        json_name = zf.namelist()[0]
        data = json.loads(zf.read(json_name))
    print(f"Loaded {len(data)} entries", flush=True)

    valid = []
    for i in tqdm(data, desc="Filtering"):
        atoms = i.get(ATOMS_KEY)
        if atoms is None:
            continue
        if not all(e in ALLOWED_SYMBOLS for e in atoms.get('elements', [])):
            continue
        valid.append(i)
    print(f"After filtering: {len(valid)} structures", flush=True)
    return valid


def load_checkpoint():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    done = {}
    for fname in os.listdir(CHECKPOINT_DIR):
        if fname.endswith(".json"):
            with open(os.path.join(CHECKPOINT_DIR, fname), "r") as f:
                for entry in json.load(f):
                    done[entry[ID_KEY]] = entry["alignn_bandgap"]
    print(f"Found {len(done)} existing ALIGNN predictions", flush=True)
    return done


def save_checkpoint(results, chunk_idx):
    path = os.path.join(CHECKPOINT_DIR, f"chunk_{chunk_idx:06d}.json")
    with open(path, "w") as f:
        json.dump(results, f)


def main():
    entries = load_entries(INPUT_ZIP)
    done = load_checkpoint()

    todo = [e for e in entries if e[ID_KEY] not in done]
    print(f"{len(todo)} structures remaining to predict", flush=True)

    if todo:
        print(f"Loading ALIGNN model: {MODEL_NAME}...", flush=True)
        model = get_figshare_model(MODEL_NAME)
        model.to(device)
        model.eval()

        mem = []
        for entry in todo:
            mem.append({
                "atoms": entry[ATOMS_KEY],
                "prop": -9999,
                "jid": entry[ID_KEY],
            })

        print("Building ALIGNN graph dataset...", flush=True)
        test_data = get_torch_dataset(
            dataset=mem,
            target="prop",
            neighbor_strategy="k-nearest",
            atom_features="cgcnn",
            use_canonize=True,
            line_graph=True,
        )

        collate_fn = test_data.collate_line_graph
        test_loader = DataLoader(
            test_data,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn,
            drop_last=False,
            num_workers=NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
        )

        print("Running ALIGNN inference...", flush=True)
        results_chunk = []
        chunk_idx = len(os.listdir(CHECKPOINT_DIR))
        id_iter = iter(test_loader.dataset.ids)

        with torch.no_grad():
            for dat in tqdm(test_loader, desc="ALIGNN inference"):
                g, lg, lat, target = dat
                out = model([g.to(device), lg.to(device), lat.to(device)])
                preds = out.cpu().numpy().flatten().tolist()

                for pred_val in preds:
                    sid = next(id_iter)
                    done[sid] = pred_val
                    results_chunk.append({
                        ID_KEY: sid,
                        "alignn_bandgap": pred_val,
                    })

                if len(results_chunk) >= CHECKPOINT_EVERY:
                    save_checkpoint(results_chunk, chunk_idx)
                    chunk_idx += 1
                    results_chunk = []

        if results_chunk:
            save_checkpoint(results_chunk, chunk_idx)

    print("Assembling final output...", flush=True)
    output = []
    for entry in entries:
        sid = entry[ID_KEY]
        if sid not in done:
            continue
        out = {
            ID_KEY: sid,
            "formula": entry.get("formula"),
            "alignn_bandgap": done[sid],
        }
        for k in ("Tc", "la", "dosef", "wlog", "debye"):
            if k in entry:
                out[k] = entry[k]
        output.append(out)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    print(f"Writing {len(output)} entries to {OUTPUT_FILE}")
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f)
    print("Done.")


if __name__ == "__main__":
    main()
