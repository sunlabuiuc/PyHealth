# run_informer_mimic3.py
#
# End-to-end pipeline training both Informer and Transformer on
# MIMIC-III lab-value forecasting using the correct PyHealth 2.0 API.
#
# Usage:
#   python run_informer_mimic3.py --root /path/to/mimic-iii/1.4
#   python run_informer_mimic3.py --root /path/to/mimic-iii/1.4 --dev
#
# Why labevents instead of chartevents:
#   The mimic3.yaml config only defines these tables:
#     patients, admissions, icustays, diagnoses_icd,
#     prescriptions, procedures_icd, labevents, noteevents
#   chartevents is NOT in the config → ValueError at load time.
#   labevents is the correct continuous numeric table available here.
#   It contains itemid, valuenum, and charttime — sufficient for
#   building hourly time-series windows of lab measurements.
# python run_informer_mimic3.py --root '..\pyhealth\datasets\mimic3'

import argparse
import math
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch import optim

from pyhealth.datasets import MIMIC3Dataset, get_dataloader
from pyhealth.models.informer import Informer
from pyhealth.models.transformer import Transformer
from pyhealth.tasks import BaseTask


# ================================================================== #
# Shared hyper-parameters                                            #
# ================================================================== #

# Common lab-test item IDs in MIMIC-III labevents
# (among the most frequently measured):
#   50809  Glucose            50912  Creatinine
#   50902  Chloride           50882  Bicarbonate
#   50868  Anion Gap          50971  Potassium
LAB_ITEM_IDS = [50809, 50912, 50902, 50882, 50868, 50971]
N_FEATURES   = len(LAB_ITEM_IDS)   # 6 channels

# Window sizes (in hours; lab events are not perfectly hourly so we
# use a coarser 6-hour grid that better matches lab draw frequency)
GRID_HOURS     = 6    # resample every N hours
SEQ_LEN        = 16   # encoder: 16 grid-steps = 4 days
LABEL_LEN      = 8    # Informer decoder start-token overlap
PRED_LEN       = 4    # forecast: 4 grid-steps = 1 day
MIN_STAY_STEPS = SEQ_LEN + PRED_LEN   # minimum usable patient record

BATCH_SIZE     = 32
LEARNING_RATE  = 1e-4
TRAIN_EPOCHS   = 3

# Informer
INFORMER_D_MODEL  = 256
INFORMER_N_HEADS  = 4
INFORMER_E_LAYERS = 2
INFORMER_D_LAYERS = 1
INFORMER_D_FF     = 512
INFORMER_DROPOUT  = 0.05
INFORMER_FACTOR   = 5

# Transformer
TRANSFORMER_EMB_DIM  = 128
TRANSFORMER_HEADS    = 4
TRANSFORMER_DROPOUT  = 0.1
TRANSFORMER_N_LAYERS = 2


# ================================================================== #
# Step 1 — Task class (PyHealth 2.0: must inherit BaseTask)          #
# ================================================================== #

class LabValuesForecastingMIMIC3(BaseTask):
    """Sliding-window lab-value forecasting task for MIMIC-III.

    Uses the labevents table (available in mimic3.yaml) to build
    time-series windows of 6 common lab measurements, resampled onto
    a 6-hour grid with forward-fill imputation.

    PyHealth 2.0 contract:
        task_name     : str            (required by set_task)
        input_schema  : Dict[str, str] (maps keys to processor names)
        output_schema : Dict[str, str] (maps label key to processor)
        __call__(patient) -> List[Dict]

    Sample fields:
        x_enc      : (List[datetime], ndarray [SEQ_LEN, N_FEATURES])
                     Encoder input — used by Informer + Transformer.
        x_mark_enc : (List[datetime], ndarray [SEQ_LEN, 4])
                     Sinusoidal time-stamp features — Informer only.
        x_dec      : (List[datetime], ndarray [LABEL_LEN+PRED_LEN, N_FEATURES])
                     Decoder input — Informer only.
        x_mark_dec : (List[datetime], ndarray [LABEL_LEN+PRED_LEN, 4])
                     Decoder time-stamp features — Informer only.
        label      : float  (mean of all features over prediction window)
    """

    task_name: str = "LabValuesForecastingMIMIC3"

    input_schema: Dict[str, str] = {
        "x_enc":      "timeseries",   # Informer + Transformer
        "x_mark_enc": "timeseries",   # Informer only
        "x_dec":      "timeseries",   # Informer only
        "x_mark_dec": "timeseries",   # Informer only
    }

    output_schema: Dict[str, str] = {"label": "regression"}

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Extract sliding lab-value windows from one patient.

        Args:
            patient: PyHealth 2.0 Patient object.
                event.timestamp → Python datetime (normalised by PyHealth)
                event.attr_dict → lowercased column values, e.g.
                                  {"itemid": "50809", "valuenum": "120.0"}

        Returns:
            List of sample dicts, one per sliding window.
        """
        samples = []

        # labevents is the correct event_type for the labevents table
        labevents = patient.get_events(event_type="labevents")
        if not labevents:
            return samples

        # Collect (timestamp, value) pairs per lab item ID
        item_series: Dict[int, List[Tuple[datetime, float]]] = {
            iid: [] for iid in LAB_ITEM_IDS
        }
        for event in labevents:
            try:
                iid   = int(event.attr_dict.get("itemid", -1))
                value = float(event.attr_dict.get("valuenum", float("nan")))
            except (TypeError, ValueError):
                continue
            if iid in item_series and not math.isnan(value):
                item_series[iid].append((event.timestamp, value))

        for iid in LAB_ITEM_IDS:
            item_series[iid].sort(key=lambda x: x[0])

        all_times = [t for s in item_series.values() for t, _ in s]
        if not all_times:
            return samples

        t_start  = min(all_times)
        t_end    = max(all_times)
        duration = (t_end - t_start).total_seconds() / 3600.0
        n_steps  = int(duration / GRID_HOURS) + 1

        if n_steps < MIN_STAY_STEPS:
            return samples

        # Resample onto a GRID_HOURS-spaced grid with forward-fill
        grid_ts  = [t_start + timedelta(hours=h * GRID_HOURS)
                    for h in range(n_steps)]
        grid_val = np.zeros((n_steps, N_FEATURES), dtype=np.float32)

        for f_idx, iid in enumerate(LAB_ITEM_IDS):
            series = item_series[iid]
            if not series:
                continue
            ptr, last_val = 0, series[0][1]
            for h in range(n_steps):
                t = grid_ts[h]
                while ptr + 1 < len(series) and series[ptr + 1][0] <= t:
                    ptr     += 1
                    last_val = series[ptr][1]
                grid_val[h, f_idx] = last_val

        # Per-feature z-score normalisation
        mean     = grid_val.mean(axis=0, keepdims=True)
        std      = grid_val.std(axis=0,  keepdims=True) + 1e-6
        grid_val = (grid_val - mean) / std

        def time_features(timestamps: List[datetime]) -> np.ndarray:
            """Sinusoidal encoding: [sin(h), cos(h), sin(dow), cos(dow)]."""
            feats = np.zeros((len(timestamps), 4), dtype=np.float32)
            for i, t in enumerate(timestamps):
                hr  = 2 * math.pi * t.hour    / 24.0
                dow = 2 * math.pi * t.weekday() / 7.0
                feats[i] = [math.sin(hr), math.cos(hr),
                            math.sin(dow), math.cos(dow)]
            return feats

        # Slide windows with stride PRED_LEN
        window = SEQ_LEN + PRED_LEN
        for start in range(0, n_steps - window + 1, PRED_LEN):
            enc_end = start + SEQ_LEN
            dec_end = enc_end + PRED_LEN

            enc_ts  = grid_ts[start:enc_end]
            dec_ts  = grid_ts[enc_end - LABEL_LEN:dec_end]

            samples.append({
                "patient_id": patient.patient_id,
                "visit_id":   f"{patient.patient_id}_w{start}",
                "x_enc":      (enc_ts, grid_val[start:enc_end]),
                "x_mark_enc": (enc_ts, time_features(enc_ts)),
                "x_dec":      (dec_ts, grid_val[enc_end - LABEL_LEN:dec_end]),
                "x_mark_dec": (dec_ts, time_features(dec_ts)),
                "label":      float(grid_val[enc_end:dec_end].mean()),
            })

        return samples


# ================================================================== #
# Step 2 — Build shared SampleDataset via set_task()                 #
# ================================================================== #

def build_dataset(mimic3_root: str, dev: bool = False):
    """Load MIMIC-III with labevents and apply LabValuesForecastingMIMIC3.

    Args:
        mimic3_root: Path to the MIMIC-III 1.4 root directory.
        dev: If True, cap at 1000 patients for fast iteration.

    Returns:
        SampleDataset used by both Informer and Transformer.
    """
    print("=" * 60)
    print("Loading MIMIC-III dataset (labevents) ...")
    base_dataset = MIMIC3Dataset(
        root=mimic3_root,
        tables=["labevents"],   # defined in mimic3.yaml
        dev=dev,
    )

    print("Applying LabValuesForecastingMIMIC3 task via set_task() ...")
    sample_dataset = base_dataset.set_task(LabValuesForecastingMIMIC3())
    print(f"Generated {len(sample_dataset):,} sliding-window samples.")
    return sample_dataset


# ================================================================== #
# Step 3 — Build models                                              #
# ================================================================== #

def build_informer(dataset, device: torch.device) -> Informer:
    """Instantiate Informer from the shared SampleDataset."""
    return Informer(
        dataset=dataset,
        enc_in=N_FEATURES,
        dec_in=N_FEATURES,
        c_out=N_FEATURES,
        seq_len=SEQ_LEN,
        label_len=LABEL_LEN,
        out_len=PRED_LEN,
        factor=INFORMER_FACTOR,
        d_model=INFORMER_D_MODEL,
        n_heads=INFORMER_N_HEADS,
        e_layers=INFORMER_E_LAYERS,
        d_layers=INFORMER_D_LAYERS,
        d_ff=INFORMER_D_FF,
        dropout=INFORMER_DROPOUT,
        attn="prob",
        embed="timeF",
        freq="h",
        activation="gelu",
        output_attention=False,
        distil=True,
        mix=True,
        device=device,
    ).to(device)


def build_transformer(dataset, device: torch.device) -> Transformer:
    """Instantiate Transformer from the shared SampleDataset."""
    return Transformer(
        dataset=dataset,
        embedding_dim=TRANSFORMER_EMB_DIM,
        heads=TRANSFORMER_HEADS,
        dropout=TRANSFORMER_DROPOUT,
        num_layers=TRANSFORMER_N_LAYERS,
    ).to(device)


# ================================================================== #
# Step 4 — Training loop                                             #
# ================================================================== #

def train_one_epoch(model, loader, optimizer, device) -> float:
    """Run one epoch, return average loss."""
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        optimizer.zero_grad()
        loss = model(**batch)["loss"]
        loss.backward()
        optimizer.step()
        total += loss.item()
        n     += 1
    return total / max(n, 1)


def train(model, name: str, loader, device,
          epochs: int = TRAIN_EPOCHS) -> List[float]:
    """Train for `epochs` epochs, print progress, return loss history."""
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    losses    = []

    print(f"\n{'─' * 60}")
    print(f"  Training {name}")
    print(f"  Parameters: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"{'─' * 60}")

    for epoch in range(1, epochs + 1):
        avg = train_one_epoch(model, loader, optimizer, device)
        losses.append(avg)
        print(f"  Epoch {epoch:>2}/{epochs}  |  avg loss: {avg:.6f}")

    print(f"  {name} training complete.")
    return losses



# ================================================================== #
# Step 5 — Hyperparameter search for Informer                        #
# ================================================================== #

# Grid of hyperparameter combinations to try.
# Varies three axes independently:
#   learning_rate : controls optimiser step size
#   d_model       : hidden embedding dimension (model capacity)
#   dropout       : regularisation strength
HP_GRID = [
    {"learning_rate": 1e-3,  "d_model": 128, "dropout": 0.0},
    {"learning_rate": 1e-3,  "d_model": 256, "dropout": 0.1},
    {"learning_rate": 1e-4,  "d_model": 128, "dropout": 0.1},
    {"learning_rate": 1e-4,  "d_model": 256, "dropout": 0.05},
    {"learning_rate": 1e-4,  "d_model": 256, "dropout": 0.2},
    {"learning_rate": 5e-5,  "d_model": 512, "dropout": 0.1},
]


def build_informer_with_hp(
    dataset,
    device: torch.device,
    d_model: int,
    dropout: float,
) -> Informer:
    """Instantiate Informer with custom d_model and dropout.

    All other architectural parameters are kept at their defaults so
    that only the swept dimensions change between runs. n_heads is
    adjusted so that d_model remains divisible by n_heads.

    Args:
        dataset: Shared SampleDataset.
        device:  Target device.
        d_model: Hidden embedding dimension.
        dropout: Dropout rate applied throughout the model.

    Returns:
        Informer model moved to device.
    """
    # Ensure d_model is divisible by n_heads
    n_heads = 4
    while d_model % n_heads != 0 and n_heads > 1:
        n_heads -= 1

    return Informer(
        dataset=dataset,
        enc_in=N_FEATURES,
        dec_in=N_FEATURES,
        c_out=N_FEATURES,
        seq_len=SEQ_LEN,
        label_len=LABEL_LEN,
        out_len=PRED_LEN,
        factor=INFORMER_FACTOR,
        d_model=d_model,
        n_heads=n_heads,
        e_layers=INFORMER_E_LAYERS,
        d_layers=INFORMER_D_LAYERS,
        d_ff=d_model * 2,      # keep d_ff proportional to d_model
        dropout=dropout,
        attn="prob",
        embed="timeF",
        freq="h",
        activation="gelu",
        output_attention=False,
        distil=True,
        mix=True,
        device=device,
    ).to(device)


def train_with_lr(
    model,
    name: str,
    loader,
    device,
    learning_rate: float,
    epochs: int = TRAIN_EPOCHS,
) -> List[float]:
    """Train a model with a specific learning rate, return loss history.

    Identical to train() but accepts learning_rate as an explicit
    argument so each hyperparameter run uses its own optimiser.

    Args:
        model:         PyHealth BaseModel instance.
        name:          Display name for progress output.
        loader:        DataLoader.
        device:        Target device.
        learning_rate: Adam learning rate for this run.
        epochs:        Number of training epochs.

    Returns:
        List of per-epoch average losses.
    """
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    losses    = []
    n_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    lr={learning_rate:.0e}  d_model={model.enc_embedding.value_embedding.tokenConv.out_channels}"
          f"  dropout={model.enc_embedding.dropout.p}  params={n_params:,}")

    for epoch in range(1, epochs + 1):
        avg = train_one_epoch(model, loader, optimizer, device)
        losses.append(avg)
        print(f"      Epoch {epoch:>2}/{epochs}  |  loss: {avg:.6f}")

    return losses


def run_hp_search(
    dataset,
    loader,
    device: torch.device,
    epochs: int = TRAIN_EPOCHS,
) -> List[dict]:
    """Run Informer training over every combination in HP_GRID.

    Builds a fresh model for each configuration, trains it for the
    specified number of epochs, and records the final-epoch loss.
    Results are printed as a ranked table at the end.

    Args:
        dataset: Shared SampleDataset.
        loader:  Shared DataLoader (same data as main training).
        device:  Target device.
        epochs:  Epochs per configuration.

    Returns:
        List of result dicts sorted by final loss (ascending), each
        containing the hyperparameters and the full loss history.
    """
    results = []

    print(f"\n{'=' * 60}")
    print(f"  Informer Hyperparameter Search  ({len(HP_GRID)} configs)")
    print(f"{'=' * 60}")

    for i, hp in enumerate(HP_GRID, 1):
        lr      = hp["learning_rate"]
        d_model = hp["d_model"]
        dropout = hp["dropout"]

        print(f"\n  [{i}/{len(HP_GRID)}]  "
              f"lr={lr:.0e}  d_model={d_model}  dropout={dropout}")

        model  = build_informer_with_hp(dataset, device, d_model, dropout)
        losses = train_with_lr(
            model, f"Informer-hp-{i}", loader, device, lr, epochs
        )

        results.append({
            "run":           i,
            "learning_rate": lr,
            "d_model":       d_model,
            "dropout":       dropout,
            "final_loss":    losses[-1],
            "loss_history":  losses,
        })

    # Sort by final loss ascending (lower is better)
    results.sort(key=lambda r: r["final_loss"])

    print(f"\n{'=' * 60}")
    print(f"  Hyperparameter Search Results (ranked by final loss)")
    print(f"{'=' * 60}")
    print(f"  {'Rank':<5}  {'lr':>8}  {'d_model':>8}  "
          f"{'dropout':>8}  {'final loss':>12}")
    print(f"  {'─'*5}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*12}")
    for rank, r in enumerate(results, 1):
        marker = "  ◀ best" if rank == 1 else ""
        print(f"  {rank:<5}  {r['learning_rate']:>8.0e}  "
              f"{r['d_model']:>8}  {r['dropout']:>8.2f}  "
              f"{r['final_loss']:>12.6f}{marker}")
    print(f"{'=' * 60}")
    print(f"  Best: lr={results[0]['learning_rate']:.0e}  "
          f"d_model={results[0]['d_model']}  "
          f"dropout={results[0]['dropout']}")

    return results

# ================================================================== #
# Entry point                                                        #
# ================================================================== #

if __name__ == "__main__":
    root='..\\pyhealth\\datasets\\mimic3'
    epochs=TRAIN_EPOCHS
    batch_size=BATCH_SIZE
    gpu=-1
    dev=False

    device = (
        torch.device(f"cuda:{gpu}")
        if gpu >= 0 and torch.cuda.is_available()
        else torch.device("cpu")
    )
    # print(torch.cuda.is_available())
    print(f"Device: {device}")
    # parser = argparse.ArgumentParser(
    #     description="Train Informer + Transformer on MIMIC-III lab-value forecasting"
    # )
    # parser.add_argument("--root",       type=str, required=True,
    #                     help="Path to MIMIC-III 1.4 root directory")
    # parser.add_argument("--epochs",     type=int, default=TRAIN_EPOCHS)
    # parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    # parser.add_argument("--gpu",        type=int, default=0,
    #                     help="GPU index; use -1 for CPU")
    # parser.add_argument("--dev",        action="store_true", default=False,
    #                     help="Cap at 1000 patients for fast testing")
    # args = parser.parse_args()

    device = (
        torch.device(f"cuda:{gpu}")
        if gpu >= 0 and torch.cuda.is_available()
        else torch.device("cpu")
    )
    # print(f"Device: {device}")

    # 1. Shared SampleDataset
    dataset = build_dataset(root, dev=dev)
    loader  = get_dataloader(dataset, batch_size=batch_size, shuffle=True)
    print(f"Dataset: {len(dataset):,} samples  |  "
          f"Batches/epoch: {math.ceil(len(dataset) / batch_size)}")

    # 2. Informer
    informer_losses = train(
        build_informer(dataset, device), "Informer",
        loader, device, epochs
    )

    # 3. Transformer
    transformer_losses = train(
        build_transformer(dataset, device), "Transformer",
        loader, device, epochs
    )

    # 4. Informer hyperparameter search
    print("\n" + "=" * 60)
    print("  Starting Informer hyperparameter search ...")
    hp_results = run_hp_search(dataset, loader, device, epochs=epochs)

    # 5. Comparison summary
    print("\n" + "=" * 60)
    print("  Final loss comparison")
    print("=" * 60)
    print(f"  {'Model':<15}  {'Final loss':>12}  All epochs")
    print(f"  {'─'*15}  {'─'*12}  {'─'*30}")
    for name, losses in [("Informer",    informer_losses),
                          ("Transformer", transformer_losses)]:
        print(f"  {name:<15}  {losses[-1]:>12.6f}  "
              f"{'  '.join(f'{l:.4f}' for l in losses)}")
    print("=" * 60)
