#!/usr/bin/env python3
"""
Multi-View Contrastive Learning: SleepEEG -> Epilepsy Domain Adaptation
=========================================================================

Reproduces Oh & Bui (2025), CHIL 2025 Best Paper:
  "Multi-View Contrastive Learning for Robust Domain Adaptation
   in Medical Time Series Analysis"

Paper: https://proceedings.mlr.press/v287/oh25a.html

This script demonstrates:
  1. Downloading preprocessed SleepEEG (source) and Epilepsy (target) data
  2. Contrastive pre-training on the source domain
  3. Fine-tuning and evaluation on the target domain
  4. Ablation Study 1: Encoder backbone comparison
     (Transformer vs. 1D-CNN vs. GRU)
  5. Ablation Study 2: Fusion strategy comparison
     (Attention vs. Concatenation vs. Mean Pooling)

Metrics reported match the paper's Table 2 (Accuracy, Precision, Recall,
F1-macro). A single run is produced by this script for reproducibility; the
checked-in ablation_results.json reflects a 10-seed aggregation (mean +/- std)
of the same configurations for more robust comparison.

Usage:
    python sleepEEG_epilepsy_multiview_contrastive.py

    Runs on CPU by default; uses CUDA if available.
    Full training on Colab T4 takes ~30 min for pre-training.
    Set QUICK_MODE = True below for a fast demo (~2 min on CPU).

Results (reported in paper Table 2, SleepEEG -> Epilepsy, Proposed method):
    Accuracy  0.956 +/- 0.002
    Precision 0.936 +/- 0.004
    Recall    0.935 +/- 0.004
    F1        0.931 +/- 0.003
    TFC baseline (Table 2):  Acc 0.950, F1 0.915
"""

import os
import sys
import time
import json
from collections import defaultdict
from functools import partial

# Force unbuffered output so we can monitor progress
print = partial(print, flush=True)

import requests
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import MultiViewContrastive

# =====================================================================
# Configuration
# =====================================================================

QUICK_MODE = False  # Set False for full reproduction

PRETRAIN_EPOCHS = 5 if QUICK_MODE else 200
FINETUNE_EPOCHS = 5 if QUICK_MODE else 100
PRETRAIN_BATCH = 128
FINETUNE_BATCH = 16
PRETRAIN_LR = 3e-4
FINETUNE_LR = 1e-3
WEIGHT_DECAY = 1e-5
TEMPERATURE = 0.07
SEED = 42

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "datasets")
RESULTS_FILE = "ablation_results.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FIGSHARE_IDS = {
    "SleepEEG": 19930178,
    "Epilepsy": 19930199,
}


# =====================================================================
# Data download and loading
# =====================================================================


def download_dataset(name: str, article_id: int, data_dir: str) -> str:
    """Download a dataset from figshare if not already present."""
    ds_dir = os.path.join(data_dir, name)
    os.makedirs(ds_dir, exist_ok=True)

    if os.path.exists(os.path.join(ds_dir, "train.pt")):
        print(f"  {name}: already downloaded.")
        return ds_dir

    print(f"  Downloading {name}...")
    api_url = f"https://api.figshare.com/v2/articles/{article_id}/files"
    resp = requests.get(api_url, timeout=15)
    resp.raise_for_status()
    for finfo in resp.json():
        fname = finfo["name"]
        furl = finfo["download_url"]
        dest = os.path.join(ds_dir, fname)
        if os.path.exists(dest):
            continue
        r = requests.get(furl, stream=True, timeout=300)
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(65536):
                f.write(chunk)
        print(f"    {fname} done.")
    return ds_dir


def load_tensors(ds_dir: str, split: str = "train"):
    """Load .pt file and return (X, y) tensors."""
    data = torch.load(
        os.path.join(ds_dir, f"{split}.pt"),
        map_location="cpu",
        weights_only=False,
    )
    if isinstance(data, dict):
        X = data.get("samples", data.get("X"))
        y = data.get("labels", data.get("y"))
    else:
        raise ValueError(f"Unexpected data format in {split}.pt")
    X = X.float()
    # Normalize per-feature (zero mean, unit std) to prevent NaN
    mean = X.mean(dim=(0, 2), keepdim=True)
    std = X.std(dim=(0, 2), keepdim=True).clamp(min=1e-8)
    X = (X - mean) / std
    return X, y.long()


# =====================================================================
# Contrastive loss
# =====================================================================


def nt_xent_loss(z_list, temperature=0.07):
    """NT-Xent contrastive loss across all view pairs.

    Uses L2-normalized embeddings. Temperature 0.07 matches the original
    paper's setting (sharper similarity distribution for harder negatives).
    """
    loss = 0.0
    n_pairs = 0
    for i in range(len(z_list)):
        for j in range(i + 1, len(z_list)):
            zi = F.normalize(z_list[i], dim=1)
            zj = F.normalize(z_list[j], dim=1)
            B = zi.size(0)
            sim = torch.mm(zi, zj.t()) / temperature
            labels = torch.arange(B, device=sim.device)
            loss += (
                F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)
            ) / 2
            n_pairs += 1
    return loss / max(n_pairs, 1)


# =====================================================================
# Training loops
# =====================================================================


def pretrain(model, src_X, epochs, batch_size, lr, weight_decay):
    """Contrastive pre-training on source domain."""
    # Build DataLoader from raw tensors
    loader = DataLoader(
        TensorDataset(src_X),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    # Only train encoder + projection, not classifier head
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        nan_detected = False
        for (x_batch,) in loader:
            x_batch = x_batch.to(DEVICE)
            latents = model.encode_views(x_batch)
            z_list = list(latents.values())
            loss = nt_xent_loss(z_list, TEMPERATURE)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"    WARNING: NaN/Inf loss at epoch {epoch+1}, skipping batch")
                nan_detected = True
                optimizer.zero_grad()
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        if nan_detected:
            print(f"    Epoch {epoch+1}: had NaN batches, continuing...")

        if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == 0:
            avg = total_loss / max(len(loader), 1)
            print(f"    Pre-train epoch {epoch+1}/{epochs}: loss={avg:.4f}")


def finetune_and_eval(model, tgt_train_X, tgt_train_y, tgt_test_X, tgt_test_y,
                      epochs, batch_size, lr, num_classes):
    """Fine-tune on target train set and evaluate on target test set."""
    # Create PyHealth dataset for fine-tuning
    train_samples = []
    for i in range(tgt_train_X.shape[0]):
        train_samples.append({
            "patient_id": f"p{i}",
            "visit_id": "v0",
            "signal": tgt_train_X[i].numpy(),
            "label": int(tgt_train_y[i].item()),
        })

    train_ds = create_sample_dataset(
        samples=train_samples,
        input_schema={"signal": "tensor"},
        output_schema={"label": "multiclass"},
        dataset_name="epilepsy_train",
    )
    train_loader = get_dataloader(train_ds, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            ret = model(**batch)
            optimizer.zero_grad()
            ret["loss"].backward()
            optimizer.step()
            total_loss += ret["loss"].item()

        if (epoch + 1) % max(1, epochs // 5) == 0 or epoch == 0:
            avg = total_loss / len(train_loader)
            print(f"    Fine-tune epoch {epoch+1}/{epochs}: loss={avg:.4f}")

    # Evaluate
    model.eval()
    all_preds = []
    all_true = []

    # Process test data in batches
    test_loader = DataLoader(
        TensorDataset(tgt_test_X, tgt_test_y),
        batch_size=64,
        shuffle=False,
    )

    feat_key = model.feature_keys[0]
    label_key = model.label_keys[0]

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            # Use the same forward path as fine-tuning to guarantee
            # identical preprocessing, view construction, and fusion.
            batch = {feat_key: x_batch, label_key: y_batch}
            ret = model(**batch)
            logits = ret["logit"]
            all_preds.append(logits.argmax(dim=-1).cpu())
            all_true.append(y_batch)

    preds = torch.cat(all_preds).numpy()
    true = torch.cat(all_true).numpy()

    acc = accuracy_score(true, preds)
    prec = precision_score(true, preds, average="macro", zero_division=0)
    rec = recall_score(true, preds, average="macro", zero_division=0)
    f1 = f1_score(true, preds, average="macro")

    return {
        "accuracy": acc,
        "precision_macro": prec,
        "recall_macro": rec,
        "f1_macro": f1,
    }


# =====================================================================
# Main
# =====================================================================


def run_experiment(
    src_X, tgt_train_X, tgt_train_y, tgt_test_X, tgt_test_y,
    encoder_type, view_type, fusion_type, num_classes
):
    """Run one full experiment: pretrain + finetune + eval."""
    print(f"\n  Config: encoder={encoder_type}, view={view_type}, "
          f"fusion={fusion_type}")

    # Create a minimal dataset for model init (must include all classes)
    init_samples = []
    seen_labels = set()
    for i in range(tgt_train_X.shape[0]):
        lbl = int(tgt_train_y[i].item())
        if lbl not in seen_labels or len(init_samples) < num_classes * 2:
            init_samples.append({
                "patient_id": f"p{i}",
                "visit_id": "v0",
                "signal": tgt_train_X[i].numpy(),
                "label": lbl,
            })
            seen_labels.add(lbl)
        if len(seen_labels) == num_classes and len(init_samples) >= num_classes * 2:
            break
    init_ds = create_sample_dataset(
        samples=init_samples,
        input_schema={"signal": "tensor"},
        output_schema={"label": "multiclass"},
        dataset_name="init",
    )

    model = MultiViewContrastive(
        dataset=init_ds,
        encoder_type=encoder_type,
        view_type=view_type,
        fusion_type=fusion_type,
        num_embedding=64,
        num_hidden=128,
        num_head=4,
        num_layers=3,
        dropout=0.2,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    t0 = time.time()
    pretrain(model, src_X, PRETRAIN_EPOCHS, PRETRAIN_BATCH, PRETRAIN_LR, WEIGHT_DECAY)
    pretrain_time = time.time() - t0

    t0 = time.time()
    metrics = finetune_and_eval(
        model, tgt_train_X, tgt_train_y, tgt_test_X, tgt_test_y,
        FINETUNE_EPOCHS, FINETUNE_BATCH, FINETUNE_LR, num_classes
    )
    finetune_time = time.time() - t0

    metrics["pretrain_time_s"] = round(pretrain_time, 1)
    metrics["finetune_time_s"] = round(finetune_time, 1)
    metrics["n_params"] = n_params

    print(f"  Results: acc={metrics['accuracy']:.4f}, "
          f"prec={metrics['precision_macro']:.4f}, "
          f"rec={metrics['recall_macro']:.4f}, "
          f"f1={metrics['f1_macro']:.4f}")
    print(f"  Time: pretrain={pretrain_time:.1f}s, finetune={finetune_time:.1f}s")
    return metrics


def main():
    print("=" * 70)
    print("Multi-View Contrastive Learning Reproduction")
    print("SleepEEG (source) -> Epilepsy (target)")
    print(f"Device: {DEVICE}")
    print(f"Mode: {'QUICK (demo)' if QUICK_MODE else 'FULL reproduction'}")
    print("=" * 70)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Download data
    print("\n--- Step 1: Downloading data ---")
    os.makedirs(DATA_DIR, exist_ok=True)
    for name, aid in FIGSHARE_IDS.items():
        download_dataset(name, aid, DATA_DIR)

    # Load data
    print("\n--- Step 2: Loading data ---")
    src_dir = os.path.join(DATA_DIR, "SleepEEG")
    tgt_dir = os.path.join(DATA_DIR, "Epilepsy")

    src_X, _ = load_tensors(src_dir, "train")
    tgt_train_X, tgt_train_y = load_tensors(tgt_dir, "train")
    tgt_test_X, tgt_test_y = load_tensors(tgt_dir, "test")

    # Subsample source for quick mode
    if QUICK_MODE:
        src_X = src_X[:2048]

    num_classes = int(tgt_train_y.max().item()) + 1

    print(f"  Source (SleepEEG): {src_X.shape}")
    print(f"  Target train:     {tgt_train_X.shape}, classes={num_classes}")
    print(f"  Target test:      {tgt_test_X.shape}")

    all_results = {}

    # -----------------------------------------------------------------
    # Main reproduction: ALL views, Transformer, Attention fusion
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("MAIN REPRODUCTION: Transformer + ALL views + Attention fusion")
    print("=" * 70)

    main_metrics = run_experiment(
        src_X, tgt_train_X, tgt_train_y, tgt_test_X, tgt_test_y,
        encoder_type="transformer",
        view_type="ALL",
        fusion_type="attention",
        num_classes=num_classes,
    )
    all_results["main_reproduction"] = main_metrics

    # -----------------------------------------------------------------
    # Ablation 1: Encoder backbone comparison
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ABLATION 1: Encoder Backbone Comparison")
    print("  (Transformer vs. 1D-CNN vs. GRU, ALL views, Attention fusion)")
    print("=" * 70)

    for enc in ["transformer", "cnn", "gru"]:
        key = f"encoder_{enc}"
        if enc == "transformer":
            all_results[key] = main_metrics  # reuse
            print(f"\n  {enc}: (reusing main result)")
            continue
        metrics = run_experiment(
            src_X, tgt_train_X, tgt_train_y, tgt_test_X, tgt_test_y,
            encoder_type=enc,
            view_type="ALL",
            fusion_type="attention",
            num_classes=num_classes,
        )
        all_results[key] = metrics

    # -----------------------------------------------------------------
    # Ablation 2: Fusion strategy comparison
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ABLATION 2: Fusion Strategy Comparison")
    print("  (Attention vs. Concat vs. Mean, ALL views, Transformer)")
    print("=" * 70)

    for fus in ["attention", "concat", "mean"]:
        key = f"fusion_{fus}"
        if fus == "attention":
            all_results[key] = main_metrics  # reuse
            print(f"\n  {fus}: (reusing main result)")
            continue
        metrics = run_experiment(
            src_X, tgt_train_X, tgt_train_y, tgt_test_X, tgt_test_y,
            encoder_type="transformer",
            view_type="ALL",
            fusion_type=fus,
            num_classes=num_classes,
        )
        all_results[key] = metrics

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(
        f"\n{'Configuration':<45} {'Acc':>8} {'Prec':>8} {'Rec':>8} "
        f"{'F1':>8} {'Params':>10}"
    )
    print("-" * 93)
    for key, m in all_results.items():
        print(
            f"{key:<45} {m['accuracy']:>8.4f} "
            f"{m['precision_macro']:>8.4f} {m['recall_macro']:>8.4f} "
            f"{m['f1_macro']:>8.4f} {m['n_params']:>10,}"
        )

    # Save results
    results_path = os.path.join(
        os.path.dirname(__file__), "..", RESULTS_FILE
    )
    serializable = {}
    for k, v in all_results.items():
        serializable[k] = {
            kk: float(vv) if isinstance(vv, (float, np.floating)) else vv
            for kk, vv in v.items()
        }
    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {results_path}")

    print("\n" + "=" * 70)
    print("Paper reference (Oh & Bui 2025, Table 2, SleepEEG -> Epilepsy):")
    print("  Proposed:    acc 0.956  prec 0.936  rec 0.935  f1 0.931")
    print("  TFC baseline: acc 0.950  prec 0.946  rec 0.891  f1 0.915")
    if not QUICK_MODE:
        print(
            f"  Our reproduction: acc {main_metrics['accuracy']:.3f}  "
            f"prec {main_metrics['precision_macro']:.3f}  "
            f"rec {main_metrics['recall_macro']:.3f}  "
            f"f1 {main_metrics['f1_macro']:.3f}"
        )
    else:
        print("  (Quick mode - not comparable to paper results)")
    print("=" * 70)


if __name__ == "__main__":
    main()
