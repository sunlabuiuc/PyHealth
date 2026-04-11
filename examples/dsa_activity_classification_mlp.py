"""Synthetic ablation for the DSA standalone task using an existing PyHealth model.

This example creates a tiny DSA-shaped dataset on disk, applies the
``DSAActivityClassification`` task with different unit selections, and compares
test accuracy with the built-in ``MLP`` model.

The goal is to demonstrate task configurations rather than reproduce the paper's
reported numbers. The synthetic signals are intentionally constructed so the
torso channels carry the strongest class information while the right-arm
channels are much less informative.

Expected outcome:
    ``all_units`` and ``torso_only`` should outperform ``right_arm_only`` on
    this synthetic setup.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pyhealth.datasets import DSADataset, get_dataloader, split_by_patient
from pyhealth.tasks import DSAActivityClassification
from pyhealth.trainer import Trainer
from pyhealth.models.mlp import MLP


def write_segment(path: Path, values: np.ndarray) -> None:
    """Write one synthetic DSA segment to disk."""
    lines = []
    for row in values:
        lines.append(",".join(f"{value:.6f}" for value in row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_activity_pattern(activity_code: str, length: int) -> np.ndarray:
    """Create a class-dependent temporal pattern that survives min-max scaling."""
    if activity_code == "A01":
        return np.linspace(-1.0, 1.0, length, dtype=np.float64)
    if activity_code == "A02":
        return np.linspace(1.0, -1.0, length, dtype=np.float64)
    raise ValueError(f"Unsupported synthetic activity code: {activity_code}")


def build_synthetic_segment(
    activity_code: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build one synthetic DSA segment with shape ``(125, 45)``."""
    length = 125
    n_channels = 45
    signal = rng.normal(loc=0.0, scale=0.08, size=(length, n_channels))
    activity_pattern = make_activity_pattern(activity_code, length).reshape(-1, 1)

    # Torso channels are highly informative for the class.
    torso_offsets = np.linspace(-0.2, 0.2, 9, dtype=np.float64).reshape(1, -1)
    signal[:, 0:9] += activity_pattern + torso_offsets

    # Right-arm channels have weak, mostly noisy information.
    weak_pattern = np.sin(np.linspace(0.0, np.pi, length, dtype=np.float64)).reshape(
        -1, 1
    )
    signal[:, 9:18] += 0.15 * weak_pattern + rng.normal(
        loc=0.0,
        scale=0.12,
        size=(length, 9),
    )

    # Remaining body sites are mostly nuisance/noise features.
    signal[:, 18:] += rng.normal(loc=0.0, scale=0.12, size=(length, 27))
    return signal


def build_synthetic_dsa_tree(root: Path) -> None:
    """Create a small synthetic DSA directory tree for the ablation script."""
    rng = np.random.default_rng(598)
    activities = ("a01", "a02")
    subjects = tuple(f"p{index}" for index in range(1, 7))
    segments = tuple(f"s{index:02d}.txt" for index in range(1, 5))

    for activity_dir in activities:
        activity_code = activity_dir.upper()
        for subject in subjects:
            for segment_name in segments:
                segment_dir = root / activity_dir / subject
                segment_dir.mkdir(parents=True, exist_ok=True)
                segment = build_synthetic_segment(activity_code, rng)
                write_segment(segment_dir / segment_name, segment)


def run_single_configuration(
    dataset: DSADataset,
    dataset_root: str,
    selected_units: Optional[Sequence[str]],
) -> Dict[str, float]:
    """Train and evaluate one task configuration."""
    task = DSAActivityClassification(
        dataset_root=dataset_root,
        selected_units=selected_units,
    )
    sample_dataset = dataset.set_task(task, num_workers=1)

    train_dataset, val_dataset, test_dataset = split_by_patient(
        sample_dataset,
        [0.5, 0.25, 0.25],
        seed=598,
    )

    train_loader = get_dataloader(train_dataset, batch_size=16, shuffle=True)
    val_loader = get_dataloader(val_dataset, batch_size=16, shuffle=False)
    test_loader = get_dataloader(test_dataset, batch_size=16, shuffle=False)

    model = MLP(
        dataset=sample_dataset,
        embedding_dim=32,
        hidden_dim=32,
        n_layers=1,
    )
    trainer = Trainer(
        model=model,
        device="cuda" if torch.cuda.is_available() else "cpu",
        metrics=["accuracy"],
    )
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=3,
        monitor="accuracy",
    )
    results = trainer.evaluate(test_loader)
    return {"accuracy": float(results["accuracy"])}


def main() -> None:
    """Run a small unit-selection ablation on synthetic DSA-shaped data."""
    configurations: Dict[str, Optional[Tuple[str, ...]]] = {
        "all_units": None,
        "torso_only": ("T",),
        "right_arm_only": ("RA",),
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        build_synthetic_dsa_tree(root)
        dataset = DSADataset(root=tmpdir)

        print("Synthetic DSA ablation with DSAActivityClassification + MLP")
        print("Dataset root:", tmpdir)
        print()

        all_results: Dict[str, Dict[str, float]] = {}
        for config_name, selected_units in configurations.items():
            print(f"Running configuration: {config_name}")
            results = run_single_configuration(dataset, tmpdir, selected_units)
            all_results[config_name] = results
            print(f"  accuracy: {results['accuracy']:.4f}")
            print()

        print("Summary")
        for config_name, results in all_results.items():
            print(f"  {config_name}: accuracy={results['accuracy']:.4f}")


if __name__ == "__main__":
    main()
