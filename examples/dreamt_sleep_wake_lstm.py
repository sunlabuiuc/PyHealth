"""DREAMT Sleep Staging LSTM — Ablation Study.

This script runs two ablation experiments on the DREAMT
sleep staging task using ``SleepStagingDREAMT``:

1. **Signal-subset ablation** (binary wake/sleep):
   ACC-only vs BVP/HRV-only vs EDA+TEMP-only vs ALL signals.

2. **Label-granularity ablation** (ALL signals):
   2-class (wake/sleep) vs 5-class (W/N1/N2/N3/R).

A single-layer unidirectional LSTM is trained with 5-fold
participant-level cross-validation (no subject leakage).

Each 30-second epoch's raw multi-channel signal is reduced to
per-channel statistics (mean, std, min, max) to form a compact
feature vector suitable for the LSTM.

Usage — full DREAMT run::

    python dreamt_sleep_wake_lstm.py --root /path/to/dreamt

Usage — synthetic demo (no dataset required)::

    python dreamt_sleep_wake_lstm.py --demo

Metrics: F1, AUROC, Accuracy, Cohen's Kappa.

Results / Findings
------------------

**Demo mode** (synthetic data, 3 patients, 2 training epochs):

Results are non-meaningful and serve only to verify that the
full pipeline (epoching -> feature extraction -> LSTM training
-> evaluation) runs end-to-end without error.  Expected output
is near-random performance (F1 ~ 0.2-0.5, Kappa ~ 0).

Reference:
    Wang et al. "Addressing wearable sleep tracking inequity:
    a new dataset and novel methods for a population with sleep
    disorders." CHIL 2024, PMLR 248:380-396.
"""

import argparse
import os
import tempfile
import warnings
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from pyhealth.tasks.sleep_staging_dreamt import (
    ALL_SIGNAL_COLUMNS,
    SleepStagingDREAMT,
)

warnings.filterwarnings("ignore")

EPOCH_LEN: int = 30 * 64  # 1920 samples per 30-s epoch at 64 Hz

SIGNAL_SUBSETS: Dict[str, List[str]] = {
    "ACC": ["ACC_X", "ACC_Y", "ACC_Z"],
    "BVP_HRV": ["BVP", "HR", "IBI"],
    "EDA_TEMP": ["EDA", "TEMP"],
    "ALL": list(ALL_SIGNAL_COLUMNS),
}


def _epoch_features(signal: np.ndarray) -> np.ndarray:
    """Convert a raw epoch signal to a compact feature vector.

    Computes mean, std, min, and max per channel.

    Args:
        signal: Array of shape ``(n_channels, epoch_len)``.

    Returns:
        1-D feature vector of length ``4 * n_channels``.
    """
    if isinstance(signal, torch.Tensor):
        signal = signal.numpy()
    feats: List[float] = []
    for ch in range(signal.shape[0]):
        s = signal[ch].astype(np.float64)
        feats.extend([
            float(np.mean(s)),
            float(np.std(s)),
            float(np.min(s)),
            float(np.max(s)),
        ])
    return np.array(feats, dtype=np.float32)


# -----------------------------------------------------------
# LSTM model
# -----------------------------------------------------------


class SleepLSTM(nn.Module):
    """Single-layer unidirectional LSTM for sleep tasks."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor ``(batch, seq_len, features)``.

        Returns:
            Logits ``(batch, seq_len, num_classes)``.
        """
        out, _ = self.lstm(x)
        return self.fc(out)


# -----------------------------------------------------------
# Dataset wrapper
# -----------------------------------------------------------


class SequenceDataset(Dataset):
    """Groups epoch samples by patient into sequences."""

    def __init__(
        self,
        samples: List[Dict[str, Any]],
    ) -> None:
        patient_map: Dict[str, list] = defaultdict(list)
        for s in samples:
            patient_map[s["patient_id"]].append(s)

        self.sequences: List[torch.Tensor] = []
        self.labels_list: List[torch.Tensor] = []
        for pid in sorted(patient_map):
            epochs = sorted(
                patient_map[pid],
                key=lambda e: e["epoch_index"],
            )
            signals = np.stack(
                [_epoch_features(e["signal"]) for e in epochs],
                axis=0,
            )
            labels = np.array(
                [e["label"] for e in epochs],
            )
            self.sequences.append(
                torch.tensor(signals, dtype=torch.float32)
            )
            self.labels_list.append(
                torch.tensor(labels, dtype=torch.long)
            )

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(
        self,
        idx: int,
    ) -> tuple:
        return self.sequences[idx], self.labels_list[idx]


def collate_fn(
    batch: list,
) -> tuple:
    """Pad variable-length sequences in a batch."""
    seqs, labels = zip(*batch)
    max_len = max(s.shape[0] for s in seqs)
    feat_dim = seqs[0].shape[1]
    padded_seqs = torch.zeros(
        len(seqs),
        max_len,
        feat_dim,
    )
    padded_labels = torch.full(
        (len(seqs), max_len),
        -1,
        dtype=torch.long,
    )
    masks = torch.zeros(
        len(seqs),
        max_len,
        dtype=torch.bool,
    )
    for i, (s, lbl) in enumerate(zip(seqs, labels)):
        length = s.shape[0]
        padded_seqs[i, :length] = s
        padded_labels[i, :length] = lbl
        masks[i, :length] = True
    return padded_seqs, padded_labels, masks


# -----------------------------------------------------------
# Training / evaluation
# -----------------------------------------------------------


def train_and_evaluate(
    train_samples: List[Dict[str, Any]],
    test_samples: List[Dict[str, Any]],
    num_classes: int = 2,
    epochs: int = 30,
    lr: float = 1e-3,
    hidden_dim: int = 64,
    device: str = "cpu",
) -> Dict[str, float]:
    """Train LSTM and return test metrics."""
    if not train_samples or not test_samples:
        return {}

    n_channels = train_samples[0]["signal"].shape[0]
    feat_dim = 4 * n_channels

    train_ds = SequenceDataset(train_samples)
    test_ds = SequenceDataset(test_samples)

    train_loader = DataLoader(
        train_ds,
        batch_size=8,
        shuffle=True,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=8,
        shuffle=False,
        collate_fn=collate_fn,
    )

    model = SleepLSTM(
        feat_dim,
        hidden_dim,
        num_classes,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    model.train()
    for _ in tqdm(range(epochs), desc="    Training", unit="epoch", leave=False):
        for seqs, labels, masks in train_loader:
            seqs = seqs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(seqs)
            loss = criterion(
                logits.reshape(-1, num_classes),
                labels.reshape(-1),
            )
            loss.backward()
            optimizer.step()

    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []
    all_probs: list = []
    with torch.no_grad():
        for seqs, labels, masks in test_loader:
            seqs = seqs.to(device)
            logits = model(seqs)
            probs = torch.softmax(logits, dim=-1).cpu()
            preds = logits.argmax(dim=-1).cpu()
            for i in range(seqs.shape[0]):
                valid = masks[i]
                all_preds.extend(preds[i][valid].numpy().tolist())
                all_labels.extend(labels[i][valid].numpy().tolist())
                if num_classes == 2:
                    all_probs.extend(
                        probs[i][valid][:, 1].numpy().tolist()
                    )
                else:
                    all_probs.extend(
                        probs[i][valid].numpy().tolist()
                    )

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    avg = "binary" if num_classes == 2 else "macro"

    results: Dict[str, float] = {
        "f1": f1_score(
            y_true,
            y_pred,
            average=avg,
            zero_division=0,
        ),
        "accuracy": accuracy_score(y_true, y_pred),
        "kappa": cohen_kappa_score(y_true, y_pred),
    }

    try:
        if num_classes == 2:
            results["auroc"] = roc_auc_score(
                y_true,
                np.array(all_probs),
            )
        else:
            results["auroc"] = roc_auc_score(
                y_true,
                np.array(all_probs),
                multi_class="ovr",
                average="macro",
            )
    except ValueError:
        results["auroc"] = float("nan")

    return results


def participant_cv(
    samples: List[Dict[str, Any]],
    n_folds: int = 5,
    num_classes: int = 2,
    **kwargs: Any,
) -> Dict[str, str]:
    """5-fold participant-level cross-validation."""
    patient_ids = sorted(set(s["patient_id"] for s in samples))
    np.random.seed(42)
    np.random.shuffle(patient_ids)

    fold_size = max(1, len(patient_ids) // n_folds)
    fold_results: List[Dict[str, float]] = []

    for fold in tqdm(range(n_folds), desc="  CV folds", unit="fold"):
        start = fold * fold_size
        end = (
            start + fold_size
            if fold < n_folds - 1
            else len(patient_ids)
        )
        test_ids = set(patient_ids[start:end])
        train_ids = set(patient_ids) - test_ids

        if not train_ids or not test_ids:
            continue

        train_s = [
            s for s in samples if s["patient_id"] in train_ids
        ]
        test_s = [
            s for s in samples if s["patient_id"] in test_ids
        ]

        res = train_and_evaluate(
            train_s,
            test_s,
            num_classes=num_classes,
            **kwargs,
        )
        if res:
            fold_results.append(res)
            tqdm.write(
                f"  Fold {fold + 1}: "
                f"F1={res['f1']:.3f}  "
                f"AUROC={res['auroc']:.3f}  "
                f"Acc={res['accuracy']:.3f}  "
                f"Kappa={res['kappa']:.3f}"
            )

    if not fold_results:
        return {}

    avg: Dict[str, str] = {}
    for key in fold_results[0]:
        vals = [r[key] for r in fold_results]
        avg[key] = f"{np.mean(vals):.3f} +/- {np.std(vals):.3f}"
    return avg


# -----------------------------------------------------------
# Demo data generation
# -----------------------------------------------------------


def _generate_demo_csv(
    tmpdir: str,
    patient_id: str,
    n_epochs: int,
    rng: np.random.RandomState,
) -> str:
    """Create one synthetic 64 Hz CSV file.

    Args:
        tmpdir: Directory to write the CSV.
        patient_id: Used in the filename.
        n_epochs: Number of 30-s epochs to generate.
        rng: Random state for reproducibility.

    Returns:
        Path to the written CSV file.
    """
    import pandas as pd

    stages_pool = ["W", "N1", "N2", "N3", "R"]
    rows = n_epochs * EPOCH_LEN
    data = {
        "TIMESTAMP": np.arange(rows) / 64.0,
        "BVP": rng.randn(rows) * 50,
        "IBI": np.clip(rng.rand(rows) * 0.2 + 0.7, 0, 2),
        "EDA": rng.rand(rows) * 5 + 0.1,
        "TEMP": rng.rand(rows) * 4 + 33,
        "ACC_X": rng.randn(rows) * 10,
        "ACC_Y": rng.randn(rows) * 10,
        "ACC_Z": rng.randn(rows) * 10,
        "HR": rng.rand(rows) * 30 + 60,
    }
    stage_col = []
    for i in range(n_epochs):
        st = stages_pool[i % len(stages_pool)]
        stage_col.extend([st] * EPOCH_LEN)
    data["Sleep_Stage"] = stage_col

    csv_path = os.path.join(tmpdir, f"{patient_id}_whole_df.csv")
    pd.DataFrame(data).to_csv(csv_path, index=False)
    return csv_path


def _generate_demo_samples(
    n_classes: int = 2,
    signal_columns: Optional[List[str]] = None,
    n_patients: int = 3,
    epochs_per_patient: int = 20,
    seed: int = 123,
) -> List[Dict[str, Any]]:
    """Create synthetic samples for demo mode.

    Args:
        n_classes: Number of label classes (2, 3, or 5).
        signal_columns: Which signal columns to include.
        n_patients: Number of synthetic patients.
        epochs_per_patient: 30-s epochs per patient.
        seed: Random seed for reproducibility.

    Returns:
        List of sample dicts ready for training.
    """
    from types import SimpleNamespace

    rng = np.random.RandomState(seed)
    all_samples: List[Dict[str, Any]] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for p in range(n_patients):
            pid = f"DEMO_{p:03d}"
            csv_path = _generate_demo_csv(
                tmpdir, pid, epochs_per_patient, rng,
            )

            evt = SimpleNamespace(file_64hz=csv_path)
            patient = SimpleNamespace(
                patient_id=pid,
                get_events=lambda et=None, e=evt: [e],
            )

            task = SleepStagingDREAMT(
                n_classes=n_classes,
                signal_columns=signal_columns,
                apply_filters=False,
            )
            all_samples.extend(task(patient))

    return all_samples


# -----------------------------------------------------------
# Main
# -----------------------------------------------------------

DEFAULT_ROOT = os.path.expanduser("~/.pyhealth/dreamt")


def _resolve_root(
    root_arg: Optional[str],
) -> str:
    """Find a valid DREAMT root, or exit with guidance.

    Args:
        root_arg: User-supplied ``--root`` value, or None.

    Returns:
        Absolute path to the DREAMT version directory.

    Raises:
        SystemExit: If no valid directory is found.
    """
    candidates = (
        [root_arg]
        if root_arg
        else [
            DEFAULT_ROOT,
            os.path.expanduser("~/data/dreamt"),
            os.path.expanduser("~/dreamt"),
        ]
    )
    for path in candidates:
        if path and os.path.isdir(path):
            info = os.path.join(path, "participant_info.csv")
            if os.path.isfile(info):
                return path
            for sub in sorted(os.listdir(path)):
                subpath = os.path.join(path, sub)
                if os.path.isdir(subpath) and os.path.isfile(
                    os.path.join(subpath, "participant_info.csv")
                ):
                    return subpath
    print(
        "ERROR: Could not find the DREAMT dataset.\n"
        "\n"
        "Download from PhysioNet (credentialed access):\n"
        "  https://physionet.org/content/dreamt/\n"
        "\n"
        "Then either:\n"
        f"  - Extract to {DEFAULT_ROOT}/\n"
        "  - Or pass --root /path/to/dreamt/version/\n"
        "\n"
        "The directory must contain participant_info.csv\n"
        "and a data_64Hz/ folder with per-participant CSVs."
    )
    raise SystemExit(1)


def _run_ablations_real(args: argparse.Namespace) -> None:
    """Run ablations on the real DREAMT dataset.

    Args:
        args: Parsed command-line arguments.
    """
    from pyhealth.datasets import DREAMTDataset

    root = _resolve_root(args.root)
    print(f"Loading DREAMT dataset from {root} ...")
    dataset = DREAMTDataset(root=root)

    print("\n" + "=" * 60)
    print("ABLATION 1: Signal Subset (2-class wake/sleep)")
    print("=" * 60)

    for subset_name, columns in tqdm(SIGNAL_SUBSETS.items(), desc="Signal subsets", unit="subset"):
        tqdm.write(f"\n--- Signal subset: {subset_name} ---")
        task = SleepStagingDREAMT(
            n_classes=2,
            signal_columns=columns,
        )
        sample_ds = dataset.set_task(task)
        samples = [sample_ds[i] for i in tqdm(range(len(sample_ds)), desc="  Loading samples", leave=False)]
        tqdm.write(f"  Total samples: {len(samples)}")
        avg = participant_cv(
            samples,
            num_classes=2,
            epochs=args.epochs,
            hidden_dim=args.hidden_dim,
            device=args.device,
        )
        tqdm.write(f"  Average: {avg}")

    print("\n" + "=" * 60)
    print("ABLATION 2: Label Granularity (ALL signals)")
    print("=" * 60)

    print("\n--- 2-class (wake vs sleep) ---")
    task_2 = SleepStagingDREAMT(n_classes=2)
    sd_2 = dataset.set_task(task_2)
    samps_2 = [sd_2[i] for i in tqdm(range(len(sd_2)), desc="  Loading samples", leave=False)]
    avg_2 = participant_cv(
        samps_2,
        num_classes=2,
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        device=args.device,
    )
    print(f"  Average: {avg_2}")

    print("\n--- 5-class (W/N1/N2/N3/R) ---")
    task_5 = SleepStagingDREAMT(n_classes=5)
    sd_5 = dataset.set_task(task_5)
    samps_5 = [sd_5[i] for i in tqdm(range(len(sd_5)), desc="  Loading samples", leave=False)]
    avg_5 = participant_cv(
        samps_5,
        num_classes=5,
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        device=args.device,
    )
    print(f"  Average: {avg_5}")


def _run_ablations_demo(args: argparse.Namespace) -> None:
    """Run ablations on synthetic demo data.

    Args:
        args: Parsed command-line arguments.
    """
    print("=== DEMO MODE (synthetic data) ===\n")
    print("Generating 3 synthetic patients (20 epochs each) ...")

    demo_epochs = min(args.epochs, 2)

    print("\n" + "=" * 60)
    print("ABLATION 1: Signal Subset (2-class, demo)")
    print("=" * 60)

    for subset_name, columns in tqdm(SIGNAL_SUBSETS.items(), desc="Signal subsets", unit="subset"):
        tqdm.write(f"\n--- Signal subset: {subset_name} ---")
        seed = abs(hash(subset_name)) % (2**31)
        sub_samples = _generate_demo_samples(
            n_classes=2,
            signal_columns=columns,
            n_patients=3,
            seed=seed,
        )
        tqdm.write(f"  Total samples: {len(sub_samples)}")
        n_pids = len(set(s["patient_id"] for s in sub_samples))
        avg = participant_cv(
            sub_samples,
            n_folds=min(3, n_pids),
            num_classes=2,
            epochs=demo_epochs,
            hidden_dim=args.hidden_dim,
            device=args.device,
        )
        tqdm.write(f"  Average: {avg}")

    print("\n" + "=" * 60)
    print("ABLATION 2: Label Granularity (demo)")
    print("=" * 60)

    print("\n--- 2-class (wake vs sleep) ---")
    samples_2 = _generate_demo_samples(
        n_classes=2, n_patients=3, seed=123,
    )
    n_pids_2 = len(set(s["patient_id"] for s in samples_2))
    avg_2 = participant_cv(
        samples_2,
        n_folds=min(3, n_pids_2),
        num_classes=2,
        epochs=demo_epochs,
        hidden_dim=args.hidden_dim,
        device=args.device,
    )
    print(f"  Average: {avg_2}")

    print("\n--- 5-class (W/N1/N2/N3/R) ---")
    samples_5 = _generate_demo_samples(
        n_classes=5, n_patients=3, seed=123,
    )
    n_pids_5 = len(set(s["patient_id"] for s in samples_5))
    avg_5 = participant_cv(
        samples_5,
        n_folds=min(3, n_pids_5),
        num_classes=5,
        epochs=demo_epochs,
        hidden_dim=args.hidden_dim,
        device=args.device,
    )
    print(f"  Average: {avg_5}")

    print("\nDemo complete.")


def main() -> None:
    """Entry point for the DREAMT LSTM ablation study."""
    parser = argparse.ArgumentParser(
        description="DREAMT LSTM ablation study",
    )
    parser.add_argument(
        "--root",
        default=None,
        help=(
            "Path to DREAMT dataset. "
            f"Default: {DEFAULT_ROOT}"
        ),
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help=(
            "Run with synthetic data instead of real "
            "DREAMT (no dataset download required)."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Training epochs per fold",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=64,
        help="LSTM hidden dimension",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device (cpu or cuda)",
    )
    args = parser.parse_args()

    if args.demo:
        _run_ablations_demo(args)
    else:
        _run_ablations_real(args)


if __name__ == "__main__":
    main()
