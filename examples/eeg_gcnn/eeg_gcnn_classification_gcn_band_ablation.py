"""Band ablation study for shallow EEG-GCNN — inference-only, no retraining.

Runs 13 conditions on the existing alpha=0.5 checkpoints produced by
training_pipeline_shallow_gcnn.py:
  - Baseline:       all 6 bands active
  - Leave-one-out:  one band zeroed, 6 conditions
  - Keep-one-in:    all bands zeroed except one, 6 conditions

The litdata SampleDataset cache is cleared between conditions so each
EEGGCNNClassification(excluded_bands=...) call loads the correct samples.

Usage (from the examples/eeg_gcnn directory):
    conda activate pyhealth (assuming PyHealth is installed in this conda env)
    python eeg_gcnn_classification_gcn_band_ablation.py

    Requires checkpoints in output_data/ produced by
    training_pipeline_shallow_gcnn.py.

Ablations:
    Edge weight mix (ALPHA):
        Must match the value used during training. Update ALPHA in the
        Configuration section to evaluate the corresponding checkpoints:
            ALPHA = 1.0   # geodesic only
            ALPHA = 0.0   # coherence only
            ALPHA = 0.5   # equal mix (default)

    Patient subset (MAX_PATIENTS):
        Must match the value used during training to reproduce the same
        held-out test split:
            MAX_PATIENTS = None  # full dataset (default)
            MAX_PATIENTS = 20    # matches a capped training run
"""

import os
import shutil
import sys
import statistics as stats
from itertools import chain
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyhealth.datasets import EEGGCNNDataset
from pyhealth.datasets.collate import collate_temporal
from pyhealth.tasks import EEGGCNNClassification
from pyhealth.tasks.eeg_gcnn_classification import BAND_NAMES

from pyhealth.models import EEGGraphConvNet

# ---------------------------------------------------------------------------
# Configuration  (must match training_pipeline_shallow_gcnn.py exactly)
# ---------------------------------------------------------------------------

DATA_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "precomputed_data"
)
ALPHA           = 0.5
CHECKPOINT_NAME = f"psd_gnn_shallow_ph_alpha{ALPHA:.2f}"
NUM_FOLDS       = 10
BATCH_SIZE      = 512
NUM_WORKERS     = 0
SEED            = 42
TEST_RATIO      = 0.30
MAX_PATIENTS: Optional[int] = None  # must match training pipeline


# ---------------------------------------------------------------------------
# Build the 13 ablation conditions
# ---------------------------------------------------------------------------

# Each entry has a human-readable label and the list of bands to zero out.
CONDITIONS: List[Dict] = [
    {"label": "baseline",   "excluded_bands": []},
    *[
        {"label": f"loo_{b}", "excluded_bands": [b]}
        for b in BAND_NAMES
    ],
    *[
        {"label": f"koi_{b}", "excluded_bands": [x for x in BAND_NAMES if x != b]}
        for b in BAND_NAMES
    ],
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _MapStyleDataset(torch.utils.data.Dataset):
    """Thin map-style wrapper around a plain list of processed sample dicts.

    Args:
        samples: List of processed sample dicts from a SampleDataset.
    """

    def __init__(self, samples: List[dict]) -> None:
        self._data = samples

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> dict:
        return self._data[idx]


def _clear_pyhealth_cache() -> None:
    """Remove all cached PyHealth datasets from disk.

    Clears both macOS (Library/Caches) and Linux (~/.cache) cache locations
    so that EEGGCNNDataset.set_task() rebuilds the sample dataset from scratch
    with the correct band mask for the next ablation condition.
    """
    for cache_root in [
        Path.home() / "Library" / "Caches" / "pyhealth",
        Path.home() / ".cache" / "pyhealth",
    ]:
        for d in cache_root.glob("*"):
            if d.is_dir():
                shutil.rmtree(d, ignore_errors=True)


def get_patient_predictions(
    patient_ids: List[str],
    y_prob_windows: np.ndarray,
    y_true_windows: np.ndarray,
) -> Tuple[List[str], np.ndarray, List[int]]:
    """Aggregate window-level predictions to patient level.

    Window probabilities are averaged per patient. Asserts that all windows
    belonging to a patient share the same ground-truth label.

    Args:
        patient_ids: Length-N list of patient ID strings, one per window.
        y_prob_windows: (N,) array of sigmoid probabilities for class 1.
        y_true_windows: (N,) array of integer ground-truth labels (0 or 1).

    Returns:
        A tuple of (patient_ids_out, y_prob_patients, y_true_patients) where:
            patient_ids_out: List of unique patient IDs.
            y_prob_patients: (P, 2) array with columns [prob_class0, prob_class1].
            y_true_patients: Length-P list of patient-level integer labels.
    """
    df = pd.DataFrame({
        "patient_id": patient_ids,
        "y_prob_1":   y_prob_windows,
        "y_true":     y_true_windows,
    })
    rows = []
    for pid, grp in df.groupby("patient_id"):
        assert grp["y_true"].nunique() == 1, (
            f"Patient {pid} has inconsistent labels across windows."
        )
        prob_1 = grp["y_prob_1"].mean()
        rows.append({
            "patient_id": pid,
            "y_prob_1":   prob_1,
            "y_prob_0":   1.0 - prob_1,
            "y_true":     int(grp["y_true"].iloc[0]),
        })
    result = pd.DataFrame(rows)
    y_prob_pat = np.column_stack(
        [result["y_prob_0"].values, result["y_prob_1"].values]
    )
    return list(result["patient_id"]), y_prob_pat, result["y_true"].tolist()


def compute_metrics(
    y_prob_windows: np.ndarray,
    y_true_windows: np.ndarray,
    patient_ids: List[str],
) -> Dict[str, float]:
    """Compute window-level and patient-level metrics for one fold.

    The decision threshold is chosen via Youden's J statistic on the
    patient-level ROC curve. Precision, recall, and F1 use pos_label=0
    (diseased class).

    Args:
        y_prob_windows: (N,) float array of sigmoid probabilities for class 1.
        y_true_windows: (N,) int array of ground-truth labels.
        patient_ids: Length-N list of patient ID strings, one per window.

    Returns:
        Dictionary with keys: auroc_window, auroc_patient, precision,
        recall, f1, bal_acc.
    """
    auroc_window = roc_auc_score(y_true_windows, y_prob_windows)

    _, y_prob_pat, y_true_pat = get_patient_predictions(
        patient_ids, y_prob_windows, y_true_windows
    )

    auroc_patient = roc_auc_score(y_true_pat, y_prob_pat[:, 1])

    fpr, tpr, thresholds = roc_curve(y_true_pat, y_prob_pat[:, 1], pos_label=1)
    optimal_threshold = sorted(
        zip(np.abs(tpr - fpr), thresholds),
        key=lambda x: x[0],
        reverse=True,
    )[0][1]

    roc_predictions = [
        1 if p >= optimal_threshold else 0 for p in y_prob_pat[:, 1]
    ]

    precision = precision_score(
        y_true_pat, roc_predictions, pos_label=0, zero_division=0
    )
    recall = recall_score(
        y_true_pat, roc_predictions, pos_label=0, zero_division=0
    )
    f1 = f1_score(
        y_true_pat, roc_predictions, pos_label=0, zero_division=0
    )
    bal_acc = balanced_accuracy_score(y_true_pat, roc_predictions)

    return {
        "auroc_window":  auroc_window,
        "auroc_patient": auroc_patient,
        "precision":     precision,
        "recall":        recall,
        "f1":            f1,
        "bal_acc":       bal_acc,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[MAIN] Using device: {device}")

    output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "output_data"
    )

    all_results: Dict[str, Dict[str, Tuple[float, float]]] = {}

    for condition in CONDITIONS:
        label          = condition["label"]
        excluded_bands = condition["excluded_bands"]

        print(f"\n{'='*60}")
        print(f"[ABLATION] Condition: {label}  excluded={excluded_bands}")
        print(f"{'='*60}")

        # Clear litdata cache so set_task() rebuilds with correct band mask
        _clear_pyhealth_cache()

        dataset = EEGGCNNDataset(root=DATA_ROOT, alpha=ALPHA)
        sample_ds = dataset.set_task(
            EEGGCNNClassification(excluded_bands=excluded_bands)
        )

        all_samples = list(sample_ds)
        patient_to_index = sample_ds.patient_to_index

        all_patients = np.array(sorted(patient_to_index.keys()))
        if MAX_PATIENTS is not None:
            rng = np.random.default_rng(SEED)
            all_patients = rng.choice(
                all_patients,
                size=min(MAX_PATIENTS, len(all_patients)),
                replace=False,
            )
            all_patients = np.sort(all_patients)

        _, test_patients = train_test_split(
            all_patients, test_size=TEST_RATIO, random_state=SEED
        )

        test_indices = list(chain.from_iterable(
            patient_to_index[pid] for pid in test_patients
        ))
        test_samples = [all_samples[i] for i in test_indices]

        test_loader = DataLoader(
            _MapStyleDataset(test_samples),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=False,
            collate_fn=collate_temporal,
        )

        fold_metrics: List[Dict[str, float]] = []

        for fold_idx in range(NUM_FOLDS):
            ckpt_path = os.path.join(
                output_dir, f"{CHECKPOINT_NAME}_fold_{fold_idx}.ckpt"
            )
            if not os.path.exists(ckpt_path):
                print(f"[ABLATION] Checkpoint not found: {ckpt_path} — skipping.")
                continue

            model = EEGGraphConvNet(dataset=sample_ds)
            model.load_state_dict(
                torch.load(ckpt_path, map_location=device, weights_only=True)
            )
            model.to(device)
            model.eval()

            all_patient_ids: List[str] = []
            all_y_prob: List[float] = []
            all_y_true: List[float] = []

            with torch.no_grad():
                for batch in test_loader:
                    output = model(**batch)
                    y_prob = output["y_prob"].cpu().numpy().squeeze(-1)
                    y_true = output["y_true"].cpu().numpy().squeeze(-1)
                    all_patient_ids.extend(batch["patient_id"])
                    all_y_prob.extend(y_prob.tolist())
                    all_y_true.extend(y_true.tolist())

            metrics = compute_metrics(
                np.array(all_y_prob),
                np.array(all_y_true, dtype=int),
                all_patient_ids,
            )
            fold_metrics.append(metrics)

        if not fold_metrics:
            print(f"[ABLATION] No folds completed for {label}, skipping.")
            continue

        summary = {
            k: (
                stats.mean([m[k] for m in fold_metrics]),
                stats.stdev([m[k] for m in fold_metrics]) if len(fold_metrics) > 1 else 0.0,
            )
            for k in fold_metrics[0]
        }
        all_results[label] = summary

        print(f"\n[ABLATION] {label} results:")
        for k, (mean, std) in summary.items():
            print(f"  {k:20s}: {mean:.4f} ± {std:.4f}")

    # -----------------------------------------------------------------------
    # Final summary table
    # -----------------------------------------------------------------------
    print(f"\n\n{'='*60}")
    print("BAND ABLATION SUMMARY (auroc_patient | bal_acc | f1)")
    print(f"{'='*60}")
    print(f"{'Condition':<20} {'auroc_patient':>15} {'bal_acc':>10} {'f1':>10}")
    print("-" * 60)
    for label, summary in all_results.items():
        auc  = f"{summary['auroc_patient'][0]:.4f}±{summary['auroc_patient'][1]:.4f}"
        bacc = f"{summary['bal_acc'][0]:.4f}±{summary['bal_acc'][1]:.4f}"
        f1   = f"{summary['f1'][0]:.4f}±{summary['f1'][1]:.4f}"
        print(f"{label:<20} {auc:>15} {bacc:>10} {f1:>10}")
