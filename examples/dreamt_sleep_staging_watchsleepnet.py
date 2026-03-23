"""Synthetic DREAMT sleep-staging example with WatchSleepNet ablations.

This example builds a tiny DREAMT-compatible directory in a temporary folder,
creates a task-specific SampleDataset, and runs a lightweight ablation over a
few WatchSleepNet configurations. It is intended for reproducibility demos and
does not require access to the real DREAMT release.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pyhealth.datasets import DREAMTDataset, get_dataloader
from pyhealth.models import WatchSleepNet
from pyhealth.tasks import SleepStagingDREAMT


def build_synthetic_dreamt_root(root: Path, num_subjects: int = 4) -> Path:
    """Create a tiny DREAMT-style directory with wearable CSV files."""
    dreamt_root = root / "dreamt"
    (dreamt_root / "data_64Hz").mkdir(parents=True)

    participant_rows = []
    for subject_index in range(num_subjects):
        patient_id = f"S{subject_index + 1:03d}"
        participant_rows.append(
            {
                "SID": patient_id,
                "AGE": 25 + subject_index,
                "GENDER": "F" if subject_index % 2 == 0 else "M",
                "BMI": 22.0 + subject_index,
                "OAHI": 1.0,
                "AHI": 2.0,
                "Mean_SaO2": "97%",
                "Arousal Index": 10.0,
                "MEDICAL_HISTORY": "None",
                "Sleep_Disorders": "None",
            }
        )

        labels = ["P"] * 32 + ["W"] * 32 + ["N2"] * 32 + ["R"] * 32
        timestamps = np.arange(len(labels), dtype=np.float32) / 64.0
        frame = pd.DataFrame(
            {
                "TIMESTAMP": timestamps,
                "IBI": np.sin(timestamps * (subject_index + 1)) + 1.0,
                "HR": 60.0 + 3.0 * np.cos(timestamps),
                "BVP": np.sin(timestamps * 0.5),
                "EDA": np.linspace(0.01, 0.04, len(labels)),
                "TEMP": np.full(len(labels), 33.0 + 0.1 * subject_index),
                "ACC_X": np.zeros(len(labels)),
                "ACC_Y": np.ones(len(labels)),
                "ACC_Z": np.full(len(labels), 2.0),
                "Sleep_Stage": labels,
            }
        )
        frame.to_csv(
            dreamt_root / "data_64Hz" / f"{patient_id}_whole_df.csv",
            index=False,
        )

    pd.DataFrame(participant_rows).to_csv(
        dreamt_root / "participant_info.csv",
        index=False,
    )
    return dreamt_root


def run_ablation(sample_dataset) -> None:
    """Run a tiny illustrative ablation with one epoch per configuration."""
    configs = [
        {"name": "baseline", "hidden_dim": 32, "use_tcn": True, "use_attention": True},
        {"name": "no_attention", "hidden_dim": 32, "use_tcn": True, "use_attention": False},
        {"name": "no_tcn", "hidden_dim": 32, "use_tcn": False, "use_attention": True},
        {"name": "small_hidden", "hidden_dim": 16, "use_tcn": True, "use_attention": True},
    ]

    loader = get_dataloader(sample_dataset, batch_size=4, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Ablation results")
    for config in configs:
        model = WatchSleepNet(
            dataset=sample_dataset,
            hidden_dim=config["hidden_dim"],
            conv_channels=config["hidden_dim"],
            num_attention_heads=4,
            use_tcn=config["use_tcn"],
            use_attention=config["use_attention"],
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        running_loss = 0.0
        num_batches = 0
        for batch in loader:
            batch = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            optimizer.zero_grad()
            output = model(**batch)
            output["loss"].backward()
            optimizer.step()
            running_loss += float(output["loss"].item())
            num_batches += 1

        mean_loss = running_loss / max(num_batches, 1)
        print(
            f"{config['name']:>12} | hidden_dim={config['hidden_dim']:>2} "
            f"| use_tcn={config['use_tcn']} | use_attention={config['use_attention']} "
            f"| mean_loss={mean_loss:.4f}"
        )


def main() -> None:
    temp_dir = Path(tempfile.mkdtemp(prefix="pyhealth_dreamt_example_"))
    try:
        dreamt_root = build_synthetic_dreamt_root(temp_dir)
        dataset = DREAMTDataset(root=str(dreamt_root), cache_dir=temp_dir / "cache")
        task = SleepStagingDREAMT(
            window_size=32,
            stride=32,
            source_preference="wearable",
        )
        sample_dataset = dataset.set_task(task=task, num_workers=1)

        print(f"Generated {len(sample_dataset)} synthetic windows")
        print(f"Input schema: {sample_dataset.input_schema}")
        print(f"Output schema: {sample_dataset.output_schema}")

        run_ablation(sample_dataset)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
