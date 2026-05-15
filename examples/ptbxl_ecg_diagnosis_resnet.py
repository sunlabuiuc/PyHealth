"""
PTB-XL ECG Diagnosis — Ablation Study (Real Data)
===================================================

Reproduces part of the benchmark from:
    Nonaka, K., & Seita, D. (2021). In-depth Benchmarking of Deep Neural
    Network Architectures for ECG Diagnosis.
    Proceedings of Machine Learning Research, 149, 414-424.
    https://proceedings.mlr.press/v149/nonaka21a.html

This script runs an ablation study on the real PTB-XL dataset using
PyHealth's pipeline: PTBXLDataset → PTBXLDiagnosis / PTBXLMulticlassDiagnosis
→ MLP / CNN / Transformer → Trainer.

Ablation dimensions
--------------------
1. Task type        — multilabel (PTBXLDiagnosis) vs multiclass
                      (PTBXLMulticlassDiagnosis)
2. Model arch       — MLP, CNN, Transformer
3. Hidden dim       — 32, 64, 128

Usage
-----
    # Auto-downloads PTB-XL if not present (resumes on failure):
    python examples/ptbxl_ecg_diagnosis_resnet.py

    # Or point to an existing download:
    python examples/ptbxl_ecg_diagnosis_resnet.py --root /path/to/ptb-xl

Requirements
------------
    pip install pandas numpy torch wfdb pyhealth requests

Author:
    Ankita Jain (ankitaj3@illinois.edu), Manish Singh (manishs4@illinois.edu)
"""

import argparse
import os
import time
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import requests
import torch
import wfdb

from pyhealth.datasets import create_sample_dataset, get_dataloader, split_by_sample
from pyhealth.datasets.ptbxl import PTBXLDataset
from pyhealth.models import MLP, CNN, Transformer
from pyhealth.trainer import Trainer
from pyhealth.tasks.ptbxl_diagnosis import (
    PTBXLDiagnosis,
    PTBXLMulticlassDiagnosis,
    SUPERCLASSES,
)

# ── Reproducibility ──────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)

# ── PTB-XL download constants ────────────────────────────────────────────────
PTBXL_URL = "https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip"
PTBXL_ZIP = "ptb-xl-1.0.3.zip"
PTBXL_EXTRACTED_DIR = "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
DEFAULT_DATA_DIR = os.path.join(Path.home(), ".pyhealth", "ptbxl")


def download_ptbxl(dest_dir: str, max_retries: int = 5) -> str:
    """Download and extract PTB-XL dataset with resume-on-failure support.

    Args:
        dest_dir: Directory to store the downloaded/extracted data.
        max_retries: Maximum number of retry attempts per failure.

    Returns:
        Path to the extracted PTB-XL root directory.
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    extracted_root = dest / PTBXL_EXTRACTED_DIR

    # Already extracted? Skip everything.
    if (extracted_root / "ptbxl_database.csv").exists():
        print(f"  PTB-XL already available at {extracted_root}")
        return str(extracted_root)

    zip_path = dest / PTBXL_ZIP

    print(f"  Downloading PTB-XL to {zip_path} ...")
    print(f"  Source: {PTBXL_URL}")

    for attempt in range(1, max_retries + 1):
        try:
            downloaded = zip_path.stat().st_size if zip_path.exists() else 0
            headers = {}
            if downloaded > 0:
                headers["Range"] = f"bytes={downloaded}-"
                print(f"  Resuming from byte {downloaded} (attempt {attempt})")

            resp = requests.get(PTBXL_URL, headers=headers, stream=True, timeout=60)

            if resp.status_code == 200:
                downloaded = 0
                mode = "wb"
            elif resp.status_code == 206:
                mode = "ab"
            elif resp.status_code == 416:
                print("  Download already complete.")
                break
            else:
                resp.raise_for_status()

            total = int(resp.headers.get("content-length", 0)) + downloaded
            chunk_size = 1024 * 1024  # 1 MB

            with open(zip_path, mode) as f:
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        pct = (downloaded / total * 100) if total else 0
                        print(
                            f"\r  Downloaded {downloaded / 1e6:.1f} MB"
                            f" / {total / 1e6:.1f} MB ({pct:.1f}%)",
                            end="", flush=True,
                        )

            print("\n  Download complete.")
            break

        except (requests.RequestException, IOError) as e:
            print(f"\n  Download error (attempt {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                wait = min(2 ** attempt, 30)
                print(f"  Retrying in {wait}s ...")
                time.sleep(wait)
            else:
                raise RuntimeError(
                    f"Failed to download PTB-XL after {max_retries} attempts. "
                    f"Partial file kept at {zip_path} — re-run to resume."
                )

    print(f"  Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)
    print(f"  Extracted to {extracted_root}")

    if not (extracted_root / "ptbxl_database.csv").exists():
        raise RuntimeError(
            f"Extraction succeeded but ptbxl_database.csv not found in "
            f"{extracted_root}. Check the archive contents."
        )

    return str(extracted_root)


# ── Signal loading ────────────────────────────────────────────────────────────

def load_ecg_signal(root: str, signal_file: str):
    """Load a 12-lead ECG signal from a WFDB record file.

    Returns:
        Flattened list of signal values, or None on failure.
    """
    record_path = os.path.join(root, signal_file)
    try:
        signal, _ = wfdb.rdsamp(record_path)
        # signal shape: (num_samples, 12) — e.g. (1000, 12) at 100 Hz
        return signal.flatten().tolist()
    except Exception as e:
        return None


# ── Build samples using iter_patients + task ──────────────────────────────────

def collect_task_samples(dataset, task):
    """Apply a task to all patients via iter_patients and collect samples."""
    samples = []
    for patient in dataset.iter_patients():
        samples.extend(task(patient))
    return samples


def build_dataset_from_samples(task_samples, root, task_type="multiclass"):
    """Load real ECG signals and build a PyHealth SampleDataset.

    Args:
        task_samples: List of dicts from PTBXLDiagnosis or PTBXLMulticlassDiagnosis.
        root: Path to the PTB-XL dataset root (for wfdb loading).
        task_type: "multiclass" or "multilabel".

    Returns:
        A PyHealth SampleDataset, or None if no valid samples.
    """
    feature_samples = []
    skipped = 0
    total = len(task_samples)

    for i, s in enumerate(task_samples):
        if (i + 1) % 500 == 0 or i == 0:
            print(f"\r  Loading signals: {i + 1}/{total}", end="", flush=True)

        signal = load_ecg_signal(root, s["signal_file"])
        if signal is None:
            skipped += 1
            continue

        sample = {
            "patient_id": s["patient_id"],
            "visit_id": str(s["record_id"]),
            "ecg_signal": signal,
        }

        if task_type == "multilabel":
            sample["labels"] = s["labels"]
        else:
            sample["label"] = s["label"]

        feature_samples.append(sample)

    print(f"\r  Loading signals: {total}/{total} done.")

    if skipped:
        print(f"  Skipped {skipped} records (unreadable signal files)")

    if not feature_samples:
        print("  No valid samples — cannot build dataset.")
        return None

    label_key = "labels" if task_type == "multilabel" else "label"
    label_mode = "multilabel" if task_type == "multilabel" else "multiclass"

    print(f"  Building SampleDataset ({len(feature_samples)} samples) ...")
    return create_sample_dataset(
        samples=feature_samples,
        input_schema={"ecg_signal": "sequence"},
        output_schema={label_key: label_mode},
        dataset_name="ptbxl_real",
    )


# ── Run one ablation config ──────────────────────────────────────────────────

def run_ablation(
    sample_ds, model_cls, model_name, hidden_dim, epochs, batch_size, lr,
    metrics_list,
):
    """Train + evaluate a single model configuration."""
    train_ds, val_ds, test_ds = split_by_sample(sample_ds, [0.7, 0.15, 0.15])

    train_loader = get_dataloader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = get_dataloader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = get_dataloader(test_ds, batch_size=batch_size, shuffle=False)

    if model_cls is MLP:
        model = model_cls(dataset=sample_ds, hidden_dim=hidden_dim, n_layers=2)
    elif model_cls is CNN:
        model = model_cls(dataset=sample_ds, hidden_dim=hidden_dim, num_layers=2)
    elif model_cls is Transformer:
        model = model_cls(dataset=sample_ds, embedding_dim=hidden_dim, num_layers=2)
    else:
        raise ValueError(f"Unknown model class: {model_cls}")

    trainer = Trainer(model=model, metrics=metrics_list, enable_logging=False)
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=epochs,
        optimizer_params={"lr": lr},
    )

    val_scores = trainer.evaluate(val_loader)
    test_scores = trainer.evaluate(test_loader)

    return {"val": val_scores, "test": test_scores}


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="PTB-XL ECG Diagnosis Ablation Study"
    )
    parser.add_argument(
        "--root", type=str, default=None,
        help="Path to PTB-XL dataset root. If omitted, auto-downloads to "
             "~/.pyhealth/ptbxl/",
    )
    parser.add_argument(
        "--sampling-rate", type=int, default=100, choices=[100, 500],
        help="Sampling rate: 100 or 500 Hz (default: 100)",
    )
    parser.add_argument(
        "--epochs", type=int, default=10,
        help="Training epochs per config (default: 10)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size (default: 32)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Learning rate (default: 1e-3)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Auto-download if no root provided
    if args.root is None:
        print("No --root specified. Auto-downloading PTB-XL ...")
        root = download_ptbxl(DEFAULT_DATA_DIR)
    elif not os.path.isfile(os.path.join(args.root, "ptbxl_database.csv")):
        print(f"ptbxl_database.csv not found in {args.root}")
        print("Downloading PTB-XL to that location ...")
        root = download_ptbxl(args.root)
    else:
        root = args.root

    print(f"Loading PTB-XL from: {root}")
    print(f"Sampling rate: {args.sampling_rate} Hz")

    # ── Load dataset ──────────────────────────────────────────────────────────
    dataset = PTBXLDataset(root=root, sampling_rate=args.sampling_rate)

    # ── Generate task samples via iter_patients ───────────────────────────────
    print("Generating task samples ...")
    ml_task = PTBXLDiagnosis()
    mc_task = PTBXLMulticlassDiagnosis()

    ml_samples = collect_task_samples(dataset, ml_task)
    mc_samples = collect_task_samples(dataset, mc_task)

    # ── Ablation 1: Task type — label distribution ────────────────────────────
    print()
    print("=" * 70)
    print("Ablation 1: Task type — multilabel vs multiclass")
    print("=" * 70)
    print(f"  PTBXLDiagnosis (multilabel)           : {len(ml_samples)} samples")
    print(f"  PTBXLMulticlassDiagnosis (multiclass) : {len(mc_samples)} samples")

    ml_dist = Counter(l for s in ml_samples for l in s["labels"])
    mc_dist = Counter(s["label"] for s in mc_samples)
    print(f"\n  Multilabel distribution : {dict(sorted(ml_dist.items()))}")
    print(f"  Multiclass distribution : {dict(sorted(mc_dist.items()))}")

    # ── Ablation 2: Model architecture × hidden dim ───────────────────────────
    MODEL_CONFIGS = [
        (MLP, "MLP"),
        (CNN, "CNN"),
        (Transformer, "Transformer"),
    ]
    HIDDEN_DIMS = [32, 64, 128]

    # --- Multiclass ablation ---
    print()
    print("=" * 70)
    print("Ablation 2a: Multiclass — Model × hidden_dim")
    print(f"  epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")
    print("=" * 70)

    print("  Loading ECG signals for multiclass task ...")
    mc_ds = build_dataset_from_samples(mc_samples, root, task_type="multiclass")
    if mc_ds is not None:
        mc_metrics = ["f1_weighted", "accuracy"]
        header = f"  {'Model':<14} {'hidden_dim':<12} {'val_f1':<10} {'test_f1':<10} {'test_acc':<10} {'test_loss'}"
        print(header)
        print(f"  {'-' * 68}")

        for model_cls, model_name in MODEL_CONFIGS:
            for hd in HIDDEN_DIMS:
                try:
                    result = run_ablation(
                        mc_ds, model_cls, model_name, hd,
                        args.epochs, args.batch_size, args.lr,
                        mc_metrics,
                    )
                    v, t = result["val"], result["test"]
                    print(
                        f"  {model_name:<14} {hd:<12} "
                        f"{v.get('f1_weighted', 0):<10.4f} "
                        f"{t.get('f1_weighted', 0):<10.4f} "
                        f"{t.get('accuracy', 0):<10.4f} "
                        f"{t.get('loss', 0):.4f}"
                    )
                except Exception as e:
                    print(f"  {model_name:<14} {hd:<12} FAILED: {e}")

        mc_ds.close()

    # --- Multilabel ablation ---
    print()
    print("=" * 70)
    print("Ablation 2b: Multilabel — Model × hidden_dim")
    print(f"  epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")
    print("=" * 70)

    print("  Loading ECG signals for multilabel task ...")
    ml_ds = build_dataset_from_samples(ml_samples, root, task_type="multilabel")
    if ml_ds is not None:
        ml_metrics = ["roc_auc_samples", "f1_weighted"]
        header = f"  {'Model':<14} {'hidden_dim':<12} {'val_auroc':<12} {'test_auroc':<12} {'test_f1':<10} {'test_loss'}"
        print(header)
        print(f"  {'-' * 72}")

        for model_cls, model_name in MODEL_CONFIGS:
            for hd in HIDDEN_DIMS:
                try:
                    result = run_ablation(
                        ml_ds, model_cls, model_name, hd,
                        args.epochs, args.batch_size, args.lr,
                        ml_metrics,
                    )
                    v, t = result["val"], result["test"]
                    print(
                        f"  {model_name:<14} {hd:<12} "
                        f"{v.get('roc_auc_samples', 0):<12.4f} "
                        f"{t.get('roc_auc_samples', 0):<12.4f} "
                        f"{t.get('f1_weighted', 0):<10.4f} "
                        f"{t.get('loss', 0):.4f}"
                    )
                except Exception as e:
                    print(f"  {model_name:<14} {hd:<12} FAILED: {e}")

        ml_ds.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("Ablation complete.")
    print("=" * 70)
    print()
    print("Reference: Nonaka & Seita (2021) reported on real PTB-XL:")
    print("  - Multilabel ROC-AUC ~0.93 (ResNet)")
    print("  - Multiclass F1      ~0.82 (ResNet)")
    print()
    print("To improve results, consider:")
    print("  1. More epochs (--epochs 50)")
    print("  2. Larger hidden dims")
    print("  3. Learning rate scheduling")
    print("  4. 500 Hz signals (--sampling-rate 500)")
