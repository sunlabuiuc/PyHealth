"""Synthetic DREAMT CSVs and task samples for example scripts (no dataset download)."""

import os
import tempfile
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from pyhealth.tasks.sleep_staging_dreamt import SleepStagingDREAMT

EPOCH_LEN: int = 30 * 64  # 1920 samples per 30-s epoch at 64 Hz


def generate_demo_csv(
    tmpdir: str,
    patient_id: str,
    n_epochs: int,
    rng: np.random.RandomState,
) -> str:
    """Write one synthetic 64 Hz CSV and return its path."""
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


def generate_demo_samples(
    n_classes: int = 2,
    signal_columns: Optional[List[str]] = None,
    n_patients: int = 6,
    epochs_per_patient: int = 15,
    seed: int = 123,
) -> List[Dict[str, Any]]:
    """Build epoch samples via ``SleepStagingDREAMT`` on synthetic CSVs."""
    rng = np.random.RandomState(seed)
    all_samples: List[Dict[str, Any]] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for p in range(n_patients):
            pid = f"DEMO_{p:03d}"
            csv_path = generate_demo_csv(
                tmpdir, pid, epochs_per_patient, rng,
            )

            evt = SimpleNamespace(file_64hz=csv_path)

            def get_events(event_type=None, **kwargs):
                return [evt]

            patient = SimpleNamespace(
                patient_id=pid,
                get_events=get_events,
            )

            task = SleepStagingDREAMT(
                n_classes=n_classes,
                signal_columns=signal_columns,
                apply_filters=False,
            )
            all_samples.extend(task(patient))

    return all_samples
