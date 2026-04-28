"""WESAD Stress Classification Example and Ablation Study.

This script demonstrates the WESADDataset + StressClassificationWESAD
pipeline and runs an ablation study over different window sizes and
model configurations.

Paper:
    Toye, Gomez, Kleinberg, "Simulation of Health Time Series with
    Nonstationarity", CHIL 2024.

Ablation Study
--------------
We investigate how two factors affect stress classification performance:

1. **Window size** (task configuration):
   - 5 seconds, 10 seconds, 20 seconds
   - Smaller windows yield more samples but less context per sample.
   - Larger windows capture more temporal patterns but reduce data size.

2. **Model architecture** (PyHealth models):
   - MLP with different hidden dimensions (32, 64, 128)
   - Demonstrates how model capacity interacts with feature granularity.

Results are printed as a summary table at the end.

Usage:
    python examples/wesad_stress_classification_mlp.py

    To use the real WESAD dataset, set ROOT below to the path
    containing S2/, S3/, ..., S17/ directories.
"""

import os
import pickle
import shutil
import tempfile
import warnings

import numpy as np

warnings.filterwarnings("ignore")


def create_synthetic_wesad(root: str, n_subjects: int = 3) -> str:
    """Create a minimal synthetic WESAD dataset for testing.

    Args:
        root: Directory to create synthetic data in.
        n_subjects: Number of synthetic subjects.

    Returns:
        Path to the synthetic dataset root.
    """
    subject_ids = [f"S{i}" for i in range(2, 2 + n_subjects)]

    for sid in subject_ids:
        subject_dir = os.path.join(root, sid)
        os.makedirs(subject_dir, exist_ok=True)

        n_eda = 4800  # 20 minutes at 4 Hz
        n_labels = int(n_eda * 175)  # ~700 Hz / 4 Hz ratio

        eda = np.random.rand(n_eda, 1).astype(np.float64)
        # Add realistic structure: stress periods have higher EDA
        stress_start = int(n_eda * 0.4)
        stress_end = int(n_eda * 0.7)
        eda[stress_start:stress_end] += 2.0

        labels = np.ones(n_labels, dtype=np.int32)
        labels_stress_start = int(n_labels * 0.4)
        labels_stress_end = int(n_labels * 0.7)
        labels_amuse_start = int(n_labels * 0.85)
        labels[labels_stress_start:labels_stress_end] = 2
        labels[labels_amuse_start:] = 3

        data = {
            "signal": {
                "wrist": {
                    "EDA": eda,
                    "BVP": np.random.rand(n_eda * 16, 1),
                    "ACC": np.random.rand(n_eda * 8, 3),
                    "TEMP": np.random.rand(n_eda, 1),
                },
                "chest": {
                    "ECG": np.random.rand(n_labels, 1),
                    "EDA": np.random.rand(n_labels, 1),
                    "EMG": np.random.rand(n_labels, 1),
                    "Temp": np.random.rand(n_labels, 1),
                    "ACC": np.random.rand(n_labels, 3),
                    "Resp": np.random.rand(n_labels, 1),
                },
            },
            "label": labels,
            "subject": sid,
        }

        pkl_path = os.path.join(subject_dir, f"{sid}.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(data, f)

    return root


def run_ablation():
    """Run the ablation study over window sizes and hidden dims."""
    from pyhealth.datasets import WESADDataset, split_by_patient, get_dataloader
    from pyhealth.tasks import StressClassificationWESAD
    from pyhealth.models import MLP
    from pyhealth.trainer import Trainer

    # -- Configuration --
    # To use the REAL WESAD dataset, change ROOT to the actual path:
    #   ROOT = "/path/to/WESAD/"
    # For this demo, we create synthetic data.
    use_synthetic = True

    if use_synthetic:
        tmp_dir = tempfile.mkdtemp(prefix="wesad_ablation_")
        ROOT = create_synthetic_wesad(tmp_dir, n_subjects=4)
    else:
        ROOT = "/path/to/WESAD/"  # <-- UPDATE THIS
        tmp_dir = None

    window_sizes = [5.0, 10.0, 20.0]
    hidden_dims = [32, 64, 128]

    results = []

    try:
        for window_sec in window_sizes:
            # Load dataset for this window size
            dataset = WESADDataset(root=ROOT)

            task = StressClassificationWESAD(
                window_size_sec=window_sec,
                use_features=True,
            )

            sample_dataset = dataset.set_task(task)

            train_ds, val_ds, test_ds = split_by_patient(
                sample_dataset, [0.6, 0.2, 0.2]
            )

            if len(train_ds) == 0 or len(test_ds) == 0:
                print(
                    f"  Skipping window={window_sec}s: "
                    f"train={len(train_ds)}, test={len(test_ds)}"
                )
                continue

            train_loader = get_dataloader(
                train_ds, batch_size=32, shuffle=True
            )
            val_loader = get_dataloader(
                val_ds, batch_size=32, shuffle=False
            )
            test_loader = get_dataloader(
                test_ds, batch_size=32, shuffle=False
            )

            for hidden_dim in hidden_dims:
                print(
                    f"\n--- Window: {window_sec}s | "
                    f"Hidden dim: {hidden_dim} ---"
                )

                model = MLP(
                    dataset=sample_dataset,
                    feature_keys=["signal"],
                    label_key="label",
                    mode="binary",
                    embedding_dim=hidden_dim,
                    hidden_dim=hidden_dim,
                )

                trainer = Trainer(model=model)
                trainer.train(
                    train_dataloader=train_loader,
                    val_dataloader=val_loader,
                    epochs=5,
                    monitor="pr_auc",
                )

                metrics = trainer.evaluate(test_loader)
                print(f"  Results: {metrics}")

                results.append({
                    "window_sec": window_sec,
                    "hidden_dim": hidden_dim,
                    **metrics,
                })

        # -- Print summary --
        print("\n" + "=" * 70)
        print("ABLATION STUDY RESULTS")
        print("=" * 70)
        print(
            f"{'Window (s)':<12} {'Hidden dim':<12} "
            f"{'Accuracy':<12} {'F1':<12} {'PR-AUC':<12}"
        )
        print("-" * 70)
        for r in results:
            print(
                f"{r['window_sec']:<12.0f} {r['hidden_dim']:<12} "
                f"{r.get('accuracy', 'N/A'):<12} "
                f"{r.get('f1', 'N/A'):<12} "
                f"{r.get('pr_auc', 'N/A'):<12}"
            )
        print("=" * 70)

    finally:
        if tmp_dir and os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    run_ablation()
