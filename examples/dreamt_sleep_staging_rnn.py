"""DREAMT Sleep Staging — Ablation Study with PyHealth RNN.

This script runs two ablation experiments on the DREAMT
sleep staging task using ``SleepStagingDREAMT``:

1. **Signal-subset ablation** (binary wake/sleep):
   ACC-only vs BVP/HRV-only vs EDA+TEMP-only vs ALL signals.

2. **Label-granularity ablation** (ALL signals):
   2-class (wake/sleep) vs 5-class (W/N1/N2/N3/R).

The model is PyHealth's built-in ``RNN`` (LSTM variant), trained
using PyHealth's ``Trainer`` with patient-level data splits.

Each 30-second epoch's raw multi-channel signal is reduced to
per-channel statistics (mean, std, min, max) to form a compact
feature vector fed to the model.

Usage — full DREAMT run::

    python dreamt_sleep_staging_rnn.py --root /path/to/dreamt

Usage — synthetic demo (no dataset required)::

    python dreamt_sleep_staging_rnn.py --demo

Metrics: F1 (macro), Accuracy, Cohen's Kappa.

Results / Findings
------------------

**Demo mode** (synthetic data, 6 patients, 2 training epochs):

Results are non-meaningful and serve only to verify that the
full pipeline (epoching -> feature extraction -> PyHealth RNN
training -> evaluation) runs end-to-end without error.  Expected
output is near-random performance.

**Paper reference** (Wang et al. CHIL 2024, Table 2):

The original paper reports wake/sleep (2-class) detection on
80 participants (after artifact QC) using LightGBM / GPBoost
with hand-crafted features from all 8 E4 channels and 5-fold
participant-level CV.  Key results from Table 2:

- Baseline LightGBM:  F1 = 0.777, Acc = 0.816, Kappa = 0.605
- Best (GPBoost + Apnea Severity RE + LSTM post-processing):
  F1 = 0.823, Acc = 0.857, Kappa = 0.694

The paper does not report per-signal-subset ablations; those
are original to this script.  This ablation also uses a simpler
feature set (4 summary stats per channel) and a neural model
(LSTM via PyHealth RNN) with a 70/10/20 patient-level split,
so results are expected to differ from the paper.

Reference:
    Wang et al. "Addressing wearable sleep tracking inequity:
    a new dataset and novel methods for a population with sleep
    disorders." CHIL 2024, PMLR 248:380-396.
"""

import argparse
import os
import tempfile
import warnings
from typing import Any, Dict, List, Optional

import numpy as np

from pyhealth.datasets import (
    create_sample_dataset,
    get_dataloader,
    split_by_patient,
)
from pyhealth.models import RNN
from pyhealth.tasks.sleep_staging_dreamt import (
    ALL_SIGNAL_COLUMNS,
    SleepStagingDREAMT,
)
from pyhealth.trainer import Trainer

warnings.filterwarnings("ignore", category=FutureWarning)

EPOCH_LEN: int = 30 * 64  # 1920 samples per 30-s epoch at 64 Hz

SIGNAL_SUBSETS: Dict[str, List[str]] = {
    "ACC": ["ACC_X", "ACC_Y", "ACC_Z"],
    "BVP_HRV": ["BVP", "HR", "IBI"],
    "EDA_TEMP": ["EDA", "TEMP"],
    "ALL": list(ALL_SIGNAL_COLUMNS),
}


def _epoch_features(signal: np.ndarray) -> List[float]:
    """Convert a raw epoch signal to a compact feature vector.

    Computes mean, std, min, and max per channel.

    Args:
        signal: Array of shape ``(n_channels, epoch_len)``.

    Returns:
        Flat list of length ``4 * n_channels``.
    """
    feats: List[float] = []
    for ch in range(signal.shape[0]):
        s = signal[ch].astype(np.float64)
        feats.extend([
            float(np.mean(s)),
            float(np.std(s)),
            float(np.min(s)),
            float(np.max(s)),
        ])
    return feats


def _prepare_samples(
    raw_samples: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Transform task-output samples into feature-vector samples.

    Each raw sample has ``signal`` (n_channels, epoch_len).
    This replaces it with a flat feature vector suitable for
    PyHealth's ``TensorProcessor``.

    Args:
        raw_samples: Output of ``SleepStagingDREAMT(patient)``.

    Returns:
        List of dicts with ``patient_id``, ``features``, ``label``.
    """
    return [
        {
            "patient_id": s["patient_id"],
            "features": _epoch_features(s["signal"]),
            "label": s["label"],
        }
        for s in raw_samples
    ]


def run_config(
    raw_samples: List[Dict[str, Any]],
    num_classes: int,
    device: str = "cpu",
    epochs: int = 5,
    hidden_dim: int = 64,
    split_ratios: Optional[List[float]] = None,
) -> Dict[str, float]:
    """Run one ablation configuration with PyHealth RNN and Trainer.

    Args:
        raw_samples: Epoch samples from ``SleepStagingDREAMT``.
        num_classes: Number of classification classes (2 or 5).
        device: Torch device string.
        epochs: Training epochs.
        hidden_dim: LSTM hidden dimension.
        split_ratios: Patient-level train/val/test ratios.

    Returns:
        Dictionary of evaluation metric scores.
    """
    if split_ratios is None:
        split_ratios = [0.7, 0.1, 0.2]

    samples = _prepare_samples(raw_samples)

    dataset = create_sample_dataset(
        samples=samples,
        input_schema={"features": "tensor"},
        output_schema={"label": "multiclass"},
        dataset_name="dreamt",
        task_name="sleep_staging",
    )

    train_ds, val_ds, test_ds = split_by_patient(
        dataset, split_ratios, seed=42,
    )

    train_loader = get_dataloader(train_ds, batch_size=32, shuffle=True)
    val_loader = get_dataloader(val_ds, batch_size=32, shuffle=False)
    test_loader = get_dataloader(test_ds, batch_size=32, shuffle=False)

    model = RNN(
        dataset=dataset,
        embedding_dim=hidden_dim,
        hidden_dim=hidden_dim,
        rnn_type="LSTM",
        num_layers=1,
        dropout=0.0,
    )

    trainer = Trainer(
        model=model,
        metrics=["accuracy", "f1_macro", "cohen_kappa"],
        device=device,
        enable_logging=False,
    )
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=epochs,
    )

    results = trainer.evaluate(test_loader)
    return results


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
    n_patients: int = 6,
    epochs_per_patient: int = 15,
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
# Ablation runners
# -----------------------------------------------------------

DEFAULT_ROOT = os.path.expanduser("~/.pyhealth/dreamt")


def _resolve_root(root_arg: Optional[str]) -> str:
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

    for subset_name, columns in SIGNAL_SUBSETS.items():
        print(f"\n--- Signal subset: {subset_name} ---")
        task = SleepStagingDREAMT(
            n_classes=2,
            signal_columns=columns,
        )
        sample_ds = dataset.set_task(task)
        raw = [sample_ds[i] for i in range(len(sample_ds))]
        print(f"  Total samples: {len(raw)}")
        results = run_config(
            raw,
            num_classes=2,
            epochs=args.epochs,
            hidden_dim=args.hidden_dim,
            device=args.device,
        )
        print(f"  Results: {results}")

    print("\n" + "=" * 60)
    print("ABLATION 2: Label Granularity (ALL signals)")
    print("=" * 60)

    for nc in [2, 5]:
        print(f"\n--- {nc}-class ---")
        task = SleepStagingDREAMT(n_classes=nc)
        sd = dataset.set_task(task)
        raw = [sd[i] for i in range(len(sd))]
        print(f"  Total samples: {len(raw)}")
        results = run_config(
            raw,
            num_classes=nc,
            epochs=args.epochs,
            hidden_dim=args.hidden_dim,
            device=args.device,
        )
        print(f"  Results: {results}")


def _run_ablations_demo(args: argparse.Namespace) -> None:
    """Run ablations on synthetic demo data.

    Args:
        args: Parsed command-line arguments.
    """
    print("=== DEMO MODE (synthetic data) ===\n")
    print("Generating 6 synthetic patients (15 epochs each) ...")

    demo_epochs = min(args.epochs, 2)

    print("\n" + "=" * 60)
    print("ABLATION 1: Signal Subset (2-class, demo)")
    print("=" * 60)

    for subset_name, columns in SIGNAL_SUBSETS.items():
        print(f"\n--- Signal subset: {subset_name} ---")
        seed = abs(hash(subset_name)) % (2**31)
        raw = _generate_demo_samples(
            n_classes=2,
            signal_columns=columns,
            n_patients=6,
            seed=seed,
        )
        print(f"  Total samples: {len(raw)}")
        results = run_config(
            raw,
            num_classes=2,
            epochs=demo_epochs,
            hidden_dim=args.hidden_dim,
            device=args.device,
            split_ratios=[0.5, 0.17, 0.33],
        )
        print(f"  Results: {results}")

    print("\n" + "=" * 60)
    print("ABLATION 2: Label Granularity (demo)")
    print("=" * 60)

    for nc in [2, 5]:
        print(f"\n--- {nc}-class ---")
        raw = _generate_demo_samples(
            n_classes=nc, n_patients=6, seed=123,
        )
        print(f"  Total samples: {len(raw)}")
        results = run_config(
            raw,
            num_classes=nc,
            epochs=demo_epochs,
            hidden_dim=args.hidden_dim,
            device=args.device,
            split_ratios=[0.5, 0.17, 0.33],
        )
        print(f"  Results: {results}")

    print("\nDemo complete.")


def main() -> None:
    """Entry point for the DREAMT sleep staging ablation study."""
    parser = argparse.ArgumentParser(
        description="DREAMT sleep staging ablation (PyHealth RNN)",
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
        help="Training epochs (default: 30)",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=64,
        help="RNN hidden dimension (default: 64)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device: cpu or cuda (default: cpu)",
    )
    args = parser.parse_args()

    if args.demo:
        _run_ablations_demo(args)
    else:
        _run_ablations_real(args)


if __name__ == "__main__":
    main()
