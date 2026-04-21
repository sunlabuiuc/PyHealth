"""PyHealth 2.0 heldout test evaluation for shallow EEG-GCNN.

Loads the fold checkpoints produced by eeg_gcnn_classification_gcn_training.py,
runs inference on the 30% held-out test subjects, and reports both
window-level and patient-level metrics.

Patient-level aggregation:
    - Window probabilities are averaged per patient.
    - The optimal decision threshold is selected using Youden's J statistic
      on the patient-level ROC curve.
    - Precision, recall, F1, and balanced accuracy are computed at
      pos_label=0 (diseased), matching the original pipeline convention.

Checkpoint format:
    Pure state-dicts saved by trainer.save_ckpt() under output_data/:
        {EXPERIMENT_NAME}_fold_{fold_idx}.ckpt

Usage (from the examples/eeg_gcnn directory):
    conda activate pyhealth (assuming PyHealth is installed in this conda env)
    python eeg_gcnn_classification_gcn_evaluation.py

Ablations:
    Edge weight mix (ALPHA):
        Must match the value used in eeg_gcnn_classification_gcn_training.py.
        Set ALPHA in the Configuration section to evaluate the corresponding
        checkpoints:
            ALPHA = 1.0   # geodesic only
            ALPHA = 0.0   # coherence only
            ALPHA = 0.5   # equal mix (default)

    Frequency band ablation (EXCLUDED_BANDS):
        Set EXCLUDED_BANDS to a list of band names to zero out those bands
        at inference time without retraining:
            EXCLUDED_BANDS = []              # all bands active (default)
            EXCLUDED_BANDS = ["delta"]       # leave-one-out: exclude delta
            EXCLUDED_BANDS = ["theta", "alpha"]  # exclude multiple bands

    Patient subset (MAX_PATIENTS):
        Must match the value used during training to reproduce the same
        held-out test split:
            MAX_PATIENTS = None  # full dataset (default)
            MAX_PATIENTS = 20    # matches a capped training run
"""

import os
import sys
import statistics as stats
from itertools import chain
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
# Configuration  (must match eeg_gcnn_classification_gcn_training.py exactly)
# ---------------------------------------------------------------------------

DATA_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "precomputed_data"
)
ALPHA           = 0.5     # must match eeg_gcnn_classification_gcn_training.py
EXCLUDED_BANDS  = []      # [] = all bands active; e.g. ["delta"] for LOO
EXPERIMENT_NAME = f"psd_gnn_shallow_ph_alpha{ALPHA:.2f}"
NUM_FOLDS       = 10    # minimum 2 (one train/val split); 10 for full 10-fold CV
BATCH_SIZE      = 512
NUM_WORKERS     = 0
SEED            = 42
TEST_RATIO      = 0.30
MAX_PATIENTS: Optional[int] = None  # must match training pipeline


# ---------------------------------------------------------------------------
# Map-style wrapper (same as training pipeline)
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


# ---------------------------------------------------------------------------
# Patient-level aggregation
# ---------------------------------------------------------------------------

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
    y_prob_patients = np.column_stack(
        [result["y_prob_0"].values, result["y_prob_1"].values]
    )
    return list(result["patient_id"]), y_prob_patients, result["y_true"].tolist()


# ---------------------------------------------------------------------------
# Per-fold metric computation
# ---------------------------------------------------------------------------

def compute_metrics(
    y_prob_windows: np.ndarray,
    y_true_windows: np.ndarray,
    patient_ids: List[str],
    fold_idx: int,
) -> Dict[str, float]:
    """Compute window-level and patient-level metrics for one fold.

    The decision threshold is chosen via Youden's J statistic on the
    patient-level ROC curve, matching the original pipeline convention.
    Precision, recall, and F1 use pos_label=0 (diseased class).

    Args:
        y_prob_windows: (N,) float array of sigmoid probabilities for class 1.
        y_true_windows: (N,) int array of ground-truth labels.
        patient_ids: Length-N list of patient ID strings, one per window.
        fold_idx: Fold index used for log prefixes.

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
    print(
        f"[Fold {fold_idx}] Optimal threshold (Youden's J): {optimal_threshold:.4f}"
    )

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

    print(
        f"[Fold {fold_idx}] window AUROC={auroc_window:.4f} | "
        f"patient AUROC={auroc_patient:.4f} | "
        f"precision={precision:.4f} recall={recall:.4f} "
        f"F1={f1:.4f} bal_acc={bal_acc:.4f}"
    )

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

    # ------------------------------------------------------------------
    # Load dataset — reuses the cache built during training
    # ------------------------------------------------------------------
    print(f"[MAIN] Loading EEGGCNNDataset from: {DATA_ROOT}")
    dataset = EEGGCNNDataset(root=DATA_ROOT, alpha=ALPHA)
    dataset.stats()

    print(f"[MAIN] Loading sample dataset (excluded_bands={EXCLUDED_BANDS})...")
    sample_ds = dataset.set_task(EEGGCNNClassification(excluded_bands=EXCLUDED_BANDS))

    # Materialise all samples once (same approach as training pipeline)
    print("[MAIN] Materialising sample dataset into memory...")
    all_samples = list(sample_ds)
    patient_to_index = sample_ds.patient_to_index

    # ------------------------------------------------------------------
    # Recreate the exact same patient selection and 70/30 split as training
    # ------------------------------------------------------------------
    all_patients = np.array(sorted(patient_to_index.keys()))
    if MAX_PATIENTS is not None:
        rng = np.random.default_rng(SEED)
        all_patients = rng.choice(
            all_patients,
            size=min(MAX_PATIENTS, len(all_patients)),
            replace=False,
        )
        all_patients = np.sort(all_patients)
        print(
            f"[MAIN] Capped to {len(all_patients)} patients "
            f"(MAX_PATIENTS={MAX_PATIENTS})"
        )

    _, test_patients = train_test_split(
        all_patients, test_size=TEST_RATIO, random_state=SEED
    )
    print(f"[MAIN] Held-out test patients: {len(test_patients)}")

    test_indices = list(chain.from_iterable(
        patient_to_index[pid] for pid in test_patients
    ))
    test_samples = [all_samples[i] for i in test_indices]
    print(f"[MAIN] Held-out test windows: {len(test_samples)}")

    test_loader = DataLoader(
        _MapStyleDataset(test_samples),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
        collate_fn=collate_temporal,
    )

    # ------------------------------------------------------------------
    # Evaluate each fold checkpoint on the held-out test set
    # ------------------------------------------------------------------
    output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "output_data"
    )

    fold_metrics: List[Dict[str, float]] = []

    for fold_idx in range(NUM_FOLDS):
        print(f"\n[MAIN] ========== Fold {fold_idx + 1}/{NUM_FOLDS} ==========")

        ckpt_path = os.path.join(output_dir, f"{EXPERIMENT_NAME}_fold_{fold_idx}.ckpt")
        if not os.path.exists(ckpt_path):
            print(f"[MAIN] Checkpoint not found: {ckpt_path} — skipping.")
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

                y_prob = output["y_prob"].cpu().numpy().squeeze(-1)  # (B,)
                y_true = output["y_true"].cpu().numpy().squeeze(-1)  # (B,)

                all_patient_ids.extend(batch["patient_id"])
                all_y_prob.extend(y_prob.tolist())
                all_y_true.extend(y_true.tolist())

        metrics = compute_metrics(
            np.array(all_y_prob),
            np.array(all_y_true, dtype=int),
            all_patient_ids,
            fold_idx,
        )
        fold_metrics.append(metrics)

    # ------------------------------------------------------------------
    # Cross-fold summary
    # ------------------------------------------------------------------
    print(f"\n[MAIN] ========== {NUM_FOLDS}-Fold Heldout Test Summary ==========")

    def _summary(fold_metrics: List[Dict[str, float]], key: str) -> None:
        """Print mean ± std for a single metric across folds.

        Args:
            fold_metrics: List of per-fold metric dicts.
            key: Metric name to summarise.
        """
        vals = [m[key] for m in fold_metrics if key in m]
        if not vals:
            return
        std = stats.stdev(vals) if len(vals) > 1 else 0.0
        print(f"  {key:20s}: {stats.mean(vals):.4f} ± {std:.4f}")

    for key in fold_metrics[0].keys():
        _summary(fold_metrics, key)

    print("[MAIN] Done.")
