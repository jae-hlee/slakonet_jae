"""
Predict PBE band gaps with ALIGNN on the full Alexandria PBE 3D dataset
matching slako_v12_all: NO e_above_hull filter, NO Z<=65 filter.

Sharded via SLURM array. Reads SLURM_ARRAY_TASK_ID / SLURM_ARRAY_TASK_COUNT
and slices dft_3d[shard_id::shard_count] so a --array=0-N submission
splits the ~4.5M entries across tasks. Each shard writes its own
checkpoint dir and final per-shard output JSON.

Outputs: results/alignn_predictions_shard{NNN}.json
  Each entry: {mat_id, formula, band_gap_ind, band_gap_dir, e_form, e_above_hull, alignn_bandgap}
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
BATCH_SIZE = 32
NUM_WORKERS = 0
CHECKPOINT_EVERY = 500

SHARD_ID = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
SHARD_COUNT = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))

# Per-shard paths so concurrent array tasks don't collide
CHECKPOINT_DIR = f"results/alignn_checkpoints_shard{SHARD_ID:03d}"
OUTPUT_FILE = f"results/alignn_predictions_shard{SHARD_ID:03d}.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_full_alexandria(path):
    """Load Alexandria — no filters (matches slako_v12_all)."""
    print(f"Loading {path}...", flush=True)
    with zipfile.ZipFile(path) as zf:
        json_name = zf.namelist()[0]
        dft_3d = json.loads(zf.read(json_name))
    print(f"Loaded {len(dft_3d)} entries", flush=True)
    return dft_3d


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
    entries = load_full_alexandria(INPUT_ZIP)

    if SHARD_COUNT > 1:
        entries = entries[SHARD_ID::SHARD_COUNT]
        print(
            f"[array {SHARD_ID}/{SHARD_COUNT}] this task owns {len(entries)} entries",
            flush=True,
        )

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
            "e_above_hull": entry["e_above_hull"],
            "alignn_bandgap": done[mid],
        })

    print(f"Writing {len(output)} entries to {OUTPUT_FILE}")
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f)
    print("Done.")


if __name__ == "__main__":
    main()
