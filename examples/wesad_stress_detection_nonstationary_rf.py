from __future__ import annotations
import os
import pickle
import tempfile
from typing import List, Tuple
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from pyhealth.datasets.wesad_nonstationary import WESADNonstationaryDataset
from pyhealth.tasks.wesad_stress_detection import wesad_stress_detection_fn


def load_features_and_labels(samples) -> Tuple[np.ndarray, np.ndarray]:
    """Loads simple statistical features from saved epoch files."""
    features: List[List[float]] = []
    labels: List[int] = []

    for sample in samples:
        with open(sample["epoch_path"], "rb") as f:
            epoch = pickle.load(f)

        signal = np.asarray(epoch["signal"], dtype=float)
        feat = [
            float(np.mean(signal)),
            float(np.std(signal)),
            float(np.min(signal)),
            float(np.max(signal)),
        ]
        features.append(feat)
        labels.append(int(sample["label"]))

    return np.asarray(features, dtype=float), np.asarray(labels, dtype=int)


def _write_synthetic_subject(root: str, subject_id: str, n_samples: int = 160, fs: int = 4):
    """Writes a small synthetic WESAD-like subject file.

    The first half is baseline (label=1), second half is stress (label=2).
    """
    t = np.linspace(0, 8 * np.pi, n_samples)
    baseline = 0.2 * np.sin(t[: n_samples // 2])
    stress = 0.6 * np.sin(t[n_samples // 2 :]) + 0.3
    eda = np.concatenate([baseline, stress])
    label = np.array([1] * (n_samples // 2) + [2] * (n_samples // 2))

    path = os.path.join(root, f"{subject_id}.pkl")
    with open(path, "wb") as f:
        pickle.dump({"eda": eda, "label": label, "fs": fs}, f)


def build_synthetic_dataset_root() -> str:
    """Creates a temporary synthetic dataset root for the example."""
    tmpdir = tempfile.mkdtemp(prefix="wesad_nonstationary_example_")
    _write_synthetic_subject(tmpdir, "S1", n_samples=160, fs=4)
    _write_synthetic_subject(tmpdir, "S2", n_samples=160, fs=4)
    _write_synthetic_subject(tmpdir, "S3", n_samples=160, fs=4)
    return tmpdir


def run_experiment(root: str, augmentation_mode: str, change_type: str):
    dataset = WESADNonstationaryDataset(
        root=root,
        augmentation_mode=augmentation_mode,
        change_type=change_type,
        magnitude=0.5,
        duration_ratio=0.25,
        refresh_cache=True,
        random_state=42,
    )

    sample_dataset = dataset.set_task(
        wesad_stress_detection_fn,
        window_sec=10,
        shift_sec=10,
        stress_label=2,
        baseline_label=1,
        keep_baseline_only=True,
    )
    samples = sample_dataset.samples

    X, y = load_features_and_labels(samples)

    if len(X) < 4 or len(np.unique(y)) < 2:
        print(
            f"Skipping mode={augmentation_mode}, change_type={change_type} "
            f"because data is insufficient."
        )
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.4,
        random_state=42,
        stratify=y,
    )

    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred)

    print(
        f"mode={augmentation_mode:>7} | change_type={change_type:>4} "
        f"| accuracy={acc:.4f} | f1={f1:.4f} | n_samples={len(samples)}"
    )


def main():
    """Runs ablations for the WESAD nonstationarity example.

    Ablation 1: augmentation mode
        - none
        - random
        - learned

    Ablation 2: change type
        - mean
        - std
        - both

    If WESAD_ROOT is not set, the script automatically falls back to a small
    synthetic dataset so the example remains runnable.
    """
    root = os.environ.get("WESAD_ROOT", "")
    if not root:
        print("WESAD_ROOT not set. Using synthetic example data instead.")
        root = build_synthetic_dataset_root()

    print("Ablation 1: augmentation mode")
    run_experiment(root=root, augmentation_mode="none", change_type="mean")
    run_experiment(root=root, augmentation_mode="random", change_type="mean")
    run_experiment(root=root, augmentation_mode="learned", change_type="mean")

    print("\nAblation 2: change type")
    run_experiment(root=root, augmentation_mode="learned", change_type="mean")
    run_experiment(root=root, augmentation_mode="learned", change_type="std")
    run_experiment(root=root, augmentation_mode="learned", change_type="both")


if __name__ == "__main__":
    main()