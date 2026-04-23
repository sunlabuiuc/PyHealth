from __future__ import annotations

"""
Ablation study: lab-only vs chart-only vs lab+chart for TPC remaining LoS.

Converted from examples/ablation.ipynb for local / CLI runs.
Install deps first: pip install -e . scikit-learn pandas
"""

import argparse
import os
import sys
from typing import List

import numpy as np
import pandas as pd
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

os.environ.setdefault(
    "PYHEALTH_CACHE_PATH",
    os.path.join(_REPO_ROOT, ".pyhealth_cache"),
)

from pyhealth.datasets import MIMIC4EHRDataset, get_dataloader, split_by_patient
from pyhealth.models import TPC
from pyhealth.tasks import RemainingLengthOfStayTPC_MIMIC4
from pyhealth.tasks.length_of_stay_prediction import categorize_los
from pyhealth.trainer import Trainer


def bin_remaining_los_days(y_days: np.ndarray) -> np.ndarray:
    flat = y_days.reshape(-1)
    out = np.zeros_like(flat, dtype=int)
    for i, value in enumerate(flat):
        if value == 0:
            out[i] = -1
        else:
            out[i] = categorize_los(int(np.floor(value)))
    return out.reshape(y_days.shape)


def masked_kappa_and_accuracy(y_true_days: np.ndarray, y_pred_days: np.ndarray):
    from sklearn.metrics import cohen_kappa_score

    y_true_bin = bin_remaining_los_days(y_true_days)
    y_pred_bin = bin_remaining_los_days(y_pred_days)

    mask = y_true_bin != -1
    yt = y_true_bin[mask]
    yp = y_pred_bin[mask]

    acc = float((yt == yp).mean()) if yt.size else float("nan")
    kappa = float(cohen_kappa_score(yt, yp)) if yt.size else float("nan")
    return kappa, acc


def safe_variable_length_inference(model, dataloader, device=None):
    """Run inference batch-by-batch and stack predictions for metric computation.

    TPC returns ``y_true`` / ``y_prob`` as **1-D** masked tensors (all valid
    timesteps in the batch flattened). Some models return **2-D** ``(B, T)``
    padded sequences; both layouts are supported.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    y_true_all: list[np.ndarray] = []
    y_pred_all: list[np.ndarray] = []
    loss_all: list[float] = []
    max_len = 0
    use_2d = False

    with torch.no_grad():
        for data in dataloader:
            batch = {}
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
                else:
                    batch[k] = v

            output = model(**batch)

            y_true = output["y_true"].detach().cpu().numpy()
            y_pred = output["y_prob"].detach().cpu().numpy()
            loss_all.append(float(output["loss"].item()))

            if y_true.ndim == 1:
                y_true_all.append(y_true)
                y_pred_all.append(y_pred)
            elif y_true.ndim == 2:
                use_2d = True
                max_len = max(max_len, y_true.shape[1])
                y_true_all.append(y_true)
                y_pred_all.append(y_pred)
            else:
                raise ValueError(f"Unexpected y_true shape: {y_true.shape}")

    if not use_2d:
        y_true_concat = np.concatenate(y_true_all, axis=0) if y_true_all else np.array([])
        y_pred_concat = np.concatenate(y_pred_all, axis=0) if y_pred_all else np.array([])
    else:
        padded_true = []
        padded_pred = []
        for yt, yp in zip(y_true_all, y_pred_all):
            pad_width = max_len - yt.shape[1]
            if pad_width > 0:
                yt = np.pad(yt, ((0, 0), (0, pad_width)), mode="constant", constant_values=0)
                yp = np.pad(yp, ((0, 0), (0, pad_width)), mode="constant", constant_values=0)
            padded_true.append(yt)
            padded_pred.append(yp)
        y_true_concat = np.concatenate(padded_true, axis=0)
        y_pred_concat = np.concatenate(padded_pred, axis=0)

    loss_mean = float(np.mean(loss_all)) if loss_all else float("nan")
    return y_true_concat, y_pred_concat, loss_mean


def run_experiment(
    name: str,
    root: str,
    cache_dir: str,
    labevent_itemids: List[str],
    chartevent_itemids: List[str],
    epochs: int = 1,
):
    print(f"\n========== Running experiment: {name} ==========")
    print(f"Lab features: {len(labevent_itemids)}")
    print(f"Chart features: {len(chartevent_itemids)}")

    dataset = MIMIC4EHRDataset(
        root=root,
        tables=["patients", "admissions", "icustays", "labevents", "chartevents"],
        dev=False,
        num_workers=2,
        cache_dir=cache_dir,
    )

    task = RemainingLengthOfStayTPC_MIMIC4(
        labevent_itemids=labevent_itemids,
        chartevent_itemids=chartevent_itemids,
    )

    sample_dataset = dataset.set_task(task)
    print("Task dataset built. Number of samples:", len(sample_dataset))

    train_ds, val_ds, test_ds = split_by_patient(sample_dataset, ratios=[0.8, 0.1, 0.1])

    print("Train:", len(train_ds))
    print("Val:", len(val_ds))
    print("Test:", len(test_ds))

    train_loader = get_dataloader(train_ds, batch_size=8, shuffle=True)
    val_loader = get_dataloader(val_ds, batch_size=8, shuffle=False)
    test_loader = get_dataloader(test_ds, batch_size=8, shuffle=False)

    model = TPC(
        dataset=sample_dataset,
        temporal_channels=11,
        pointwise_channels=5,
        num_layers=8,
        kernel_size=5,
        main_dropout=0.0,
        temporal_dropout=0.05,
        use_batchnorm=True,
        final_hidden=36,
    )

    trainer = Trainer(
        model=model,
        metrics=["mae", "mse"],
        enable_logging=False,
    )

    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=None,
        test_dataloader=None,
        epochs=epochs,
        monitor="mae",
        monitor_criterion="min",
        optimizer_params={"lr": 0.00221},
    )

    y_true, y_pred, _ = safe_variable_length_inference(model, test_loader)

    kappa, acc = masked_kappa_and_accuracy(y_true, y_pred)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    mse = float(np.mean((y_true - y_pred) ** 2))

    result = {
        "experiment": name,
        "mae": mae,
        "mse": mse,
        "kappa": kappa,
        "accuracy": acc,
    }

    print("Result:", result)
    return result


def _check_required_files(data_root: str) -> bool:
    required_files = [
        ("hosp", "patients.csv.gz"),
        ("hosp", "admissions.csv.gz"),
        ("icu", "icustays.csv.gz"),
        ("hosp", "labevents.csv.gz"),
        ("hosp", "d_labitems.csv.gz"),
        ("icu", "chartevents.csv.gz"),
    ]
    print("Checking required files...\n")
    all_ok = True
    for sub, fn in required_files:
        path = os.path.join(data_root, sub, fn)
        exists = os.path.exists(path)
        print(f"{path}: {exists}")
        if not exists:
            all_ok = False
    print("\nAll files found:", all_ok)
    return all_ok


def main() -> None:
    default_data = os.path.join(_REPO_ROOT, "datasets", "mimic-iv-demo", "2.2")
    default_csv = os.path.join(_REPO_ROOT, "ablation_results.csv")

    parser = argparse.ArgumentParser(description="Run TPC ablation experiments (from ablation.ipynb).")
    parser.add_argument(
        "--data-root",
        default=os.environ.get("MIMIC4_DATA_ROOT", default_data),
        help="MIMIC-IV (or demo) root containing hosp/ and icu/ (default: repo demo path or MIMIC4_DATA_ROOT).",
    )
    parser.add_argument(
        "--output-csv",
        default=default_csv,
        help="Where to write the results table.",
    )
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs per experiment.")
    parser.add_argument(
        "--skip-sanity",
        action="store_true",
        help="Skip the small dev=True dataset load used in the notebook as a sanity check.",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip the lab_only_debug run before the three named ablations.",
    )
    args = parser.parse_args()

    data_root = os.path.abspath(args.data_root)
    print("DATA_ROOT:", data_root)

    if not _check_required_files(data_root):
        raise SystemExit(1)

    labevent_itemids = [
        "50824", "52455", "50983", "52623", "50822", "52452", "50971",
        "52610", "50806", "52434", "50902", "52535", "50803", "50804",
        "50809", "52027", "50931", "52569", "50808", "51624", "50960",
        "50868", "52500", "52031", "50964", "51701", "50970",
    ]

    chart_vitals_itemids = [
        "220045",
        "220179",
        "220180",
        "220181",
        "220210",
        "224690",
        "223761",
        "223762",
        "220277",
        "225664",
        "220621",
        "226537",
    ]

    if not args.skip_sanity:
        dataset = MIMIC4EHRDataset(
            root=data_root,
            tables=["patients", "admissions", "icustays", "labevents", "chartevents"],
            dev=True,
            num_workers=2,
            cache_dir=os.path.join(_REPO_ROOT, ".sanity_cache"),
        )
        print(dataset)
        task = RemainingLengthOfStayTPC_MIMIC4(
            labevent_itemids=labevent_itemids,
            chartevent_itemids=[],
        )
        sample_dataset = dataset.set_task(task)
        print(sample_dataset)
        print("Number of samples:", len(sample_dataset))

    if not args.skip_baseline:
        run_experiment(
            name="lab_only_debug",
            root=data_root,
            cache_dir=os.path.join(_REPO_ROOT, ".ablation_cache", "lab_only_debug"),
            labevent_itemids=labevent_itemids,
            chartevent_itemids=[],
            epochs=args.epochs,
        )

    experiments = [
        {"name": "lab_only", "labevent_itemids": labevent_itemids, "chartevent_itemids": []},
        {
            "name": "chart_only",
            "labevent_itemids": [],
            "chartevent_itemids": chart_vitals_itemids,
        },
        {
            "name": "lab_plus_chart",
            "labevent_itemids": labevent_itemids,
            "chartevent_itemids": chart_vitals_itemids,
        },
    ]

    all_results = []
    for exp in experiments:
        result = run_experiment(
            name=exp["name"],
            root=data_root,
            cache_dir=os.path.join(_REPO_ROOT, ".ablation_cache", exp["name"]),
            labevent_itemids=exp["labevent_itemids"],
            chartevent_itemids=exp["chartevent_itemids"],
            epochs=args.epochs,
        )
        all_results.append(result)

    results_df = pd.DataFrame(all_results)
    print("\nResults:")
    print(results_df)
    print("\nSorted by MAE:")
    print(results_df.sort_values("mae"))

    out_path = os.path.abspath(args.output_csv)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    results_df.to_csv(out_path, index=False)
    print("Saved to:", out_path)

    best_row = results_df.sort_values("mae").iloc[0]
    print("\nBest configuration:")
    print(best_row)
    print("\nInterpretation template:")
    print(
        f"The best-performing ablation setting was {best_row['experiment']} "
        f"with MAE={best_row['mae']:.4f}, MSE={best_row['mse']:.4f}, "
        f"Kappa={best_row['kappa']:.4f}, Accuracy={best_row['accuracy']:.4f}."
    )


if __name__ == "__main__":
    main()
