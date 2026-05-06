"""
Predict PBE band gaps with ALIGNN on the Alexandria PBE 3D dataset
matching slako_v11_alexwz's filtering: e_above_hull == 0 only, NO Z<=65 filter.

Outputs: results/alignn_predictions.json
  Each entry: {mat_id, formula, band_gap_ind, band_gap_dir, e_form, alignn_bandgap}
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
INPUT_ZIP = "alexandria_pbe_3d_2024.10.1_jarvis_tools.json.zip"
OUTPUT_FILE = "results/alignn_predictions.json"
CHECKPOINT_DIR = "results/alignn_checkpoints"
BATCH_SIZE = 32
NUM_WORKERS = 0
CHECKPOINT_EVERY = 500

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_filtered_alexandria(path):
    """Load and filter Alexandria — hull only (no Z filter, matches slako_v11)."""
    print(f"Loading {path}...", flush=True)
    with zipfile.ZipFile(path) as zf:
        json_name = zf.namelist()[0]
        dft_3d = json.loads(zf.read(json_name))
    print(f"Loaded {len(dft_3d)} entries", flush=True)

    valid = []
    for i in tqdm(dft_3d, desc="Filtering"):
        if i['e_above_hull'] != 0:
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
                    done[entry["mat_id"]] = entry["alignn_bandgap"]
    print(f"Found {len(done)} existing ALIGNN predictions", flush=True)
    return done


def save_checkpoint(results, chunk_idx):
    path = os.path.join(CHECKPOINT_DIR, f"chunk_{chunk_idx:06d}.json")
    with open(path, "w") as f:
        json.dump(results, f)


def main():
    entries = load_filtered_alexandria(INPUT_ZIP)
    done = load_checkpoint()

    todo = [e for e in entries if e["mat_id"] not in done]
    print(f"{len(todo)} structures remaining to predict", flush=True)

    if todo:
        print(f"Loading ALIGNN model: {MODEL_NAME}...", flush=True)
        model = get_figshare_model(MODEL_NAME)
        model.to(device)
        model.eval()

        mem = []
        for entry in todo:
            mem.append({
                "atoms": entry["atoms"],
                "prop": -9999,
                "jid": entry["mat_id"],
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
                    jid = next(id_iter)
                    done[jid] = pred_val
                    results_chunk.append({
                        "mat_id": jid,
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
        mid = entry["mat_id"]
        if mid not in done:
            continue
        output.append({
            "mat_id": mid,
            "formula": entry["formula"],
            "band_gap_ind": entry["band_gap_ind"],
            "band_gap_dir": entry["band_gap_dir"],
            "e_form": entry["e_form"],
            "alignn_bandgap": done[mid],
        })

    print(f"Writing {len(output)} entries to {OUTPUT_FILE}")
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f)
    print("Done.")


if __name__ == "__main__":
    main()
