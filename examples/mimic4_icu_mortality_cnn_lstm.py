"""Ablation study for CNNLSTMPredictor on MIMIC-IV ICU mortality prediction

This script demonstrates the CNNLSTMPredictor model from:
    "Robust Mortality Prediction in the Intensive Care Unit using
    Temporal Difference Learning" (Frost et al.)
    Paper: https://github.com/tdgfrost/td-icu-mortality

Ablation Study
--------------
The original paper uses fixed hyperparameters (learning rate = 1/n_params,
hidden dimension = 128, dropout = 0.3, batch_size = 64). This ablation
investigates whether varying these hyperparameters improves model
performance on 28-day ICU mortality prediction.

Hyperparameters varied:
    1. Learning rate: [0.0001, 0.0005, 0.001, 0.005, 0.01]
    2. Hidden dimension: [32, 64, 128, 256]
    3. Dropout: [0.0, 0.1, 0.3, 0.5]
    4. Batch size: [16, 64, 128]

For each ablation dimension, one hyperparameter is varied while others are
held at their default values (lr=0.001, hidden_dim=128, dropout=0.3,
batch_size=64).

Setup:
    - Dataset: MIMIC-IV v3.0 (diagnoses_icd + procedures_icd)
    - Task: 28-day ICU mortality prediction
    - Split: 80/10/10 train/val/test
    - Training: PyHealth Trainer, 2 epochs, Adam optimizer
    - Metrics: AUROC, PR-AUC, F1

Usage:
    python examples/mimic4_icu_mortality_cnn_lstm.py \\
        --mimic4_root /path/to/mimiciv/3.0/

Results:
    Learning Rate Ablation (hidden_dim=128, dropout=0.3, batch_size=64):
    | LR     | AUROC  | PR-AUC | F1     |
    |--------|--------|--------|--------|
    | 0.0001 | 0.9368 | 0.6466 | 0.5235 |
    | 0.0005 | 0.9418 | 0.6343 | 0.5626 |
    | 0.001  | 0.9514 | 0.6847 | 0.6252 |
    | 0.005  | 0.9514 | 0.6614 | 0.5036 |
    | 0.01   | 0.9271 | 0.5854 | 0.0056 |

    Hidden Dimension Ablation (lr=0.001, dropout=0.3, batch_size=64):
    | Hidden | AUROC  | PR-AUC | F1     |
    |--------|--------|--------|--------|
    | 32     | 0.9471 | 0.6570 | 0.5727 |
    | 64     | 0.9492 | 0.6713 | 0.6382 |
    | 128    | 0.9519 | 0.6773 | 0.5442 |
    | 256    | 0.9524 | 0.6636 | 0.5948 |

    Dropout Ablation (lr=0.001, hidden_dim=128, batch_size=64):
    | Drop   | AUROC  | PR-AUC | F1     |
    |--------|--------|--------|--------|
    | 0.0    | 0.9518 | 0.6833 | 0.6026 |
    | 0.1    | 0.9491 | 0.6839 | 0.6287 |
    | 0.3    | 0.9496 | 0.6763 | 0.6166 |
    | 0.5    | 0.9512 | 0.6774 | 0.5621 |

    Batch Size Ablation (lr=0.001, hidden_dim=128, dropout=0.3):
    | Batch  | AUROC  | PR-AUC | F1     |
    |--------|--------|--------|--------|
    | 16     | 0.9458 | 0.6577 | 0.5151 |
    | 64     | 0.9436 | 0.6608 | 0.6037 |
    | 128    | 0.9480 | 0.6832 | 0.5643 |

Best Hyperparameters:
    The best AUROC configuration is lr=0.001, hidden_dim=256, dropout=0.0,
    batch_size=128. However, performance differences are small (AUROC range:
    0.9271-0.9524), indicating robustness to hyperparameter choices.

    - Learning rate: Best at 0.001 (mid), AUROC=0.9514.
    - Hidden dimension: Best at 256 (high), AUROC=0.9524.
    - Dropout: Best at 0.0 (low), AUROC=0.9518.
    - Batch size: Best at 128 (high), AUROC=0.9480.
"""

import argparse
import os
import pickle
import random
import tempfile
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn

from pyhealth.datasets import (
    MIMIC4EHRDataset,
    get_dataloader,
    split_by_sample,
)
from pyhealth.models.cnn_lstm import CNNLSTMPredictor
from pyhealth.tasks import BaseTask
from pyhealth.trainer import Trainer


# ---------- demo dataset (no MIMIC-IV required) ----------
def make_demo_dataset():
    """Build a small synthetic SampleDataset for demo mode (no real data).

    Creates 30 patients with random condition/procedure codes and
    alternating mortality labels so the full ablation sweep can run
    without MIMIC-IV access.

    Returns:
        SampleDataset backed by a temporary directory.
    """
    import litdata
    from pyhealth.datasets.sample_dataset import SampleBuilder, SampleDataset

    code_pool_c = [f"C{i:04d}" for i in range(50)]
    code_pool_p = [f"P{i:04d}" for i in range(30)]
    samples = []
    for i in range(30):
        samples.append(
            {
                "patient_id": f"demo_{i}",
                "visit_id": f"visit_{i}",
                "conditions": random.sample(
                    code_pool_c, random.randint(3, 8)
                ),
                "procedures": random.sample(
                    code_pool_p, random.randint(1, 4)
                ),
                "mortality": i % 2,
            }
        )

    input_schema = {"conditions": "sequence", "procedures": "sequence"}
    output_schema = {"mortality": "binary"}

    builder = SampleBuilder(input_schema, output_schema)
    builder.fit(samples)

    tmpdir = tempfile.mkdtemp()
    builder.save(os.path.join(tmpdir, "schema.pkl"))

    litdata.optimize(
        fn=builder.transform,
        inputs=[{"sample": pickle.dumps(s)} for s in samples],
        output_dir=tmpdir,
        chunk_bytes="64MB",
        num_workers=0,
    )
    return SampleDataset(path=tmpdir)


# ---------- reproducibility ----------
SEED = 1
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
os.environ["PYTHONHASHSEED"] = str(SEED)


# ---------- task definition ----------
class ICUMortalityPredictionMIMIC4(BaseTask):
    """Multi-horizon ICU mortality prediction task for MIMIC-IV

    Predicts whether a patient will die within horizon_days of
    hospital admission.

    Args:
        horizon_days: number of days for the mortality prediction
            window. Common values: 1, 3, 7, 14, 28.
    """

    task_name: str = "ICUMortalityPredictionMIMIC4"
    input_schema: Dict[str, str] = {
        "conditions": "sequence",
        "procedures": "sequence",
    }
    output_schema: Dict[str, str] = {"mortality": "binary"}

    def __init__(self, horizon_days: int = 28):
        self.horizon_days = horizon_days
        self.task_name = f"ICUMortality{horizon_days}d_MIMIC4"

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Extract mortality prediction samples from a patient"""
        samples: List[Dict[str, Any]] = []

        admissions = patient.get_events(event_type="admissions")
        if not admissions:
            return []

        patient_info = patient.get_events(event_type="patients")
        dod = None
        # Get the day of death (dod) from patient info if available
        if patient_info:
            dod_raw = getattr(patient_info[0], "dod", None)
            if dod_raw and str(dod_raw) not in ("", "None"):
                try:
                    dod = (
                        datetime.strptime(dod_raw, "%Y-%m-%d")
                        if isinstance(dod_raw, str)
                        else dod_raw
                    )
                except (ValueError, TypeError):
                    pass

        # Loop through each hospital admission and create a sample for each
        for admission in admissions:
            hadm_id = admission.hadm_id
            admit_time = admission.timestamp
            mortality = 0
            expired = getattr(
                admission, "hospital_expire_flag", None
            )

            # Check if patient expired within the horizon
            if expired and int(expired) == 1:
                dischtime = getattr(admission, "dischtime", None)
                if dischtime and admit_time:
                    try:
                        los = (
                            dischtime - admit_time
                        ).total_seconds() / 86400
                        mortality = (
                            1 if los <= self.horizon_days else 0
                        )
                    except (TypeError, AttributeError):
                        mortality = 1
                else:
                    mortality = 1
            # Check recorded date of death against horizon
            elif dod and admit_time:
                try:
                    admit_dt = (
                        admit_time
                        if isinstance(admit_time, datetime)
                        else datetime.strptime(
                            str(admit_time)[:19],
                            "%Y-%m-%d %H:%M:%S",
                        )
                    )
                    dod_dt = (
                        dod
                        if isinstance(dod, datetime)
                        else datetime.strptime(
                            str(dod)[:10], "%Y-%m-%d"
                        )
                    )
                    days = (
                        dod_dt - admit_dt
                    ).total_seconds() / 86400
                    if 0 <= days <= self.horizon_days:
                        mortality = 1
                except (TypeError, ValueError, AttributeError):
                    pass

            # Extract diagnoses and procedures for this admission
            diagnoses = patient.get_events(
                event_type="diagnoses_icd",
                filters=[("hadm_id", "==", hadm_id)],
            )
            conditions = list(
                {
                    e.icd_code
                    for e in diagnoses
                    if hasattr(e, "icd_code")
                }
            )

            procs = patient.get_events(
                event_type="procedures_icd",
                filters=[("hadm_id", "==", hadm_id)],
            )
            procedures = list(
                {
                    e.icd_code
                    for e in procs
                    if hasattr(e, "icd_code")
                }
            )

            if not conditions and not procedures:
                continue

            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "visit_id": str(hadm_id),
                    "conditions": conditions,
                    "procedures": procedures,
                    "mortality": mortality,
                }
            )

        return samples


# ---------- training helper ----------
def train_and_evaluate(
    samples,
    train_loader,
    val_loader,
    test_loader,
    model_kwargs,
    lr=0.001,
    epochs=2,
):
    """Train a CNNLSTMPredictor and return test metrics

    Args:
        samples: SampleDataset for model initialization.
        train_loader: training DataLoader.
        val_loader: validation DataLoader.
        test_loader: test DataLoader.
        model_kwargs: dict of model hyperparameters.
        lr: learning rate.
        epochs: number of training epochs.

    Returns:
        dict with roc_auc, pr_auc, f1 metrics on the test set.
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    model = CNNLSTMPredictor(dataset=samples, **model_kwargs)
    trainer = Trainer(
        model=model,
        metrics=["roc_auc", "pr_auc", "f1"],
        device="cpu",
    )
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=epochs,
        optimizer_class=torch.optim.Adam,
        optimizer_params={"lr": lr, "weight_decay": 1e-5},
        max_grad_norm=5.0,
        monitor="roc_auc",
        monitor_criterion="max",
        load_best_model_at_last=True,
    )
    return trainer.evaluate(test_loader)


def main():
    parser = argparse.ArgumentParser(
        description="CNNLSTMPredictor ablation on MIMIC-IV"
    )
    parser.add_argument(
        "--mimic4_root",
        type=str,
        default=None,
        help="Path to MIMIC-IV v3.0 root directory",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run with synthetic data (no MIMIC-IV required)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of training epochs (default: 2)",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Print progress updates every 10 minutes",
    )
    args = parser.parse_args()

    use_demo = args.demo or args.mimic4_root is None

    if use_demo:
        print("Running in DEMO mode with synthetic data (no MIMIC-IV required).")
        samples = make_demo_dataset()
        batch_size = 8
    else:
        os.environ["MEMORY_FS_ROOT"] = "/tmp/pandarallel_memory"
        print("Loading MIMIC-IV dataset...")
        dataset = MIMIC4EHRDataset(
            root=args.mimic4_root,
            tables=["diagnoses_icd", "procedures_icd"],
        )
        print(f"Patients: {len(dataset.unique_patient_ids)}")
        task = ICUMortalityPredictionMIMIC4(horizon_days=28)
        samples = dataset.set_task(task)
        print(f"Samples: {len(samples)}")
        batch_size = 64

    # ---- Split ----
    train_s, val_s, test_s = split_by_sample(
        samples, [0.8, 0.1, 0.1]
    )
    train_loader = get_dataloader(
        train_s, batch_size=batch_size, shuffle=True
    )
    val_loader = get_dataloader(
        val_s, batch_size=batch_size, shuffle=False
    )
    test_loader = get_dataloader(
        test_s, batch_size=batch_size, shuffle=False
    )
    print(
        f"Train: {len(train_s)}, Val: {len(val_s)}, "
        f"Test: {len(test_s)}"
    )

    # ---- Progress tracking ----
    run_start = time.time()
    last_status_time = run_start
    models_completed = 0
    total_models = 5 + 4 + 4 + 3  # lr + hd + dp + bs

    def print_progress(label, results):
        nonlocal models_completed, last_status_time
        models_completed += 1
        if not args.progress:
            return
        now = time.time()
        elapsed = now - run_start
        elapsed_min = elapsed / 60
        avg_per_model = elapsed / models_completed
        remaining = avg_per_model * (total_models - models_completed)
        # Print every 10 minutes or on first/last model
        if (now - last_status_time >= 600
                or models_completed == 1
                or models_completed == total_models):
            print(
                f"\n[PROGRESS] {models_completed}/{total_models} models "
                f"| Elapsed: {elapsed_min:.1f}min "
                f"| Est. remaining: {remaining / 60:.1f}min "
                f"| Last: {label}"
            )
            last_status_time = now

    # ---- Default model kwargs ----
    default_kwargs = {
        "embedding_dim": 128,
        "hidden_dim": 128,
        "num_cnn_layers": 2,
        "num_lstm_layers": 2,
        "dropout": 0.3,
    }

    # ============================================================
    # Ablation 1: Learning Rate
    # ============================================================
    print("\n" + "=" * 60)
    print("ABLATION 1: LEARNING RATE")
    print("=" * 60)

    learning_rates = [0.0001, 0.0005, 0.001, 0.005, 0.01]
    lr_results = {}
    for lr in learning_rates:
        print(f"\n--- LR = {lr} ---")
        results = train_and_evaluate(
            samples,
            train_loader,
            val_loader,
            test_loader,
            default_kwargs,
            lr=lr,
            epochs=args.epochs,
        )
        lr_results[lr] = results
        print(
            f"  AUROC={results['roc_auc']:.4f}  "
            f"PR-AUC={results['pr_auc']:.4f}  "
            f"F1={results['f1']:.4f}"
        )
        print_progress(f"LR={lr}", results)

    # ============================================================
    # Ablation 2: Hidden Dimension
    # ============================================================
    print("\n" + "=" * 60)
    print("ABLATION 2: HIDDEN DIMENSION")
    print("=" * 60)

    hidden_dims = [32, 64, 128, 256]
    hd_results = {}
    for hd in hidden_dims:
        print(f"\n--- Hidden Dim = {hd} ---")
        kwargs = dict(default_kwargs)
        kwargs["embedding_dim"] = hd
        kwargs["hidden_dim"] = hd
        results = train_and_evaluate(
            samples,
            train_loader,
            val_loader,
            test_loader,
            kwargs,
            lr=0.001,
            epochs=args.epochs,
        )
        hd_results[hd] = results
        print(
            f"  AUROC={results['roc_auc']:.4f}  "
            f"PR-AUC={results['pr_auc']:.4f}  "
            f"F1={results['f1']:.4f}"
        )
        print_progress(f"HiddenDim={hd}", results)

    # ============================================================
    # Ablation 3: Dropout
    # ============================================================
    print("\n" + "=" * 60)
    print("ABLATION 3: DROPOUT")
    print("=" * 60)

    dropouts = [0.0, 0.1, 0.3, 0.5]
    dp_results = {}
    for dp in dropouts:
        print(f"\n--- Dropout = {dp} ---")
        kwargs = dict(default_kwargs)
        kwargs["dropout"] = dp
        results = train_and_evaluate(
            samples,
            train_loader,
            val_loader,
            test_loader,
            kwargs,
            lr=0.001,
            epochs=args.epochs,
        )
        dp_results[dp] = results
        print(
            f"  AUROC={results['roc_auc']:.4f}  "
            f"PR-AUC={results['pr_auc']:.4f}  "
            f"F1={results['f1']:.4f}"
        )
        print_progress(f"Dropout={dp}", results)

    # ============================================================
    # Ablation 4: Batch Size
    # ============================================================
    print("\n" + "=" * 60)
    print("ABLATION 4: BATCH SIZE")
    print("=" * 60)

    batch_sizes = [16, 64, 128]
    bs_results = {}
    for bs in batch_sizes:
        print(f"\n--- Batch Size = {bs} ---")
        bs_train_loader = get_dataloader(
            train_s, batch_size=bs, shuffle=True
        )
        bs_val_loader = get_dataloader(
            val_s, batch_size=bs, shuffle=False
        )
        bs_test_loader = get_dataloader(
            test_s, batch_size=bs, shuffle=False
        )
        results = train_and_evaluate(
            samples,
            bs_train_loader,
            bs_val_loader,
            bs_test_loader,
            default_kwargs,
            lr=0.001,
            epochs=args.epochs,
        )
        bs_results[bs] = results
        print(
            f"  AUROC={results['roc_auc']:.4f}  "
            f"PR-AUC={results['pr_auc']:.4f}  "
            f"F1={results['f1']:.4f}"
        )
        print_progress(f"BatchSize={bs}", results)

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    best_lr = max(
        lr_results, key=lambda x: lr_results[x]["roc_auc"]
    )
    best_hd = max(
        hd_results, key=lambda x: hd_results[x]["roc_auc"]
    )
    best_dp = max(
        dp_results, key=lambda x: dp_results[x]["roc_auc"]
    )
    best_bs = max(
        bs_results, key=lambda x: bs_results[x]["roc_auc"]
    )

    print(
        f"Best LR: {best_lr} "
        f"(AUROC={lr_results[best_lr]['roc_auc']:.4f})"
    )
    print(
        f"Best Hidden Dim: {best_hd} "
        f"(AUROC={hd_results[best_hd]['roc_auc']:.4f})"
    )
    print(
        f"Best Dropout: {best_dp} "
        f"(AUROC={dp_results[best_dp]['roc_auc']:.4f})"
    )
    print(
        f"Best Batch Size: {best_bs} "
        f"(AUROC={bs_results[best_bs]['roc_auc']:.4f})"
    )

    # Paper's suggested LR
    ref_model = CNNLSTMPredictor(
        dataset=samples, **default_kwargs
    )
    n_params = sum(
        p.numel()
        for p in ref_model.parameters()
        if p.requires_grad
    )
    paper_lr = 1.0 / n_params
    print(f"\nPaper's LR (1/n_params): {paper_lr:.8f}")
    print(f"Total trainable parameters: {n_params:,}")


if __name__ == "__main__":
    main()
