from pathlib import Path
import pandas as pd

from pyhealth.datasets.wearable_dataset import WearableDataset
from pyhealth.tasks.wearable_illness_task import WearableIllnessPrediction


def test_wearable_dataset_and_task(tmp_path):
    # Create a small synthetic dataset to verify the pipeline runs
    data = pd.DataFrame(
        [
            {"patient_id": "P1", "day_index": 0, "resting_heart_rate": 60, "sleep_duration": 7.0, "is_ill": 0},
            {"patient_id": "P1", "day_index": 1, "resting_heart_rate": 61, "sleep_duration": 7.0, "is_ill": 0},
            {"patient_id": "P1", "day_index": 2, "resting_heart_rate": 62, "sleep_duration": 6.5, "is_ill": 0},
            {"patient_id": "P1", "day_index": 3, "resting_heart_rate": 63, "sleep_duration": 6.0, "is_ill": 0},
            {"patient_id": "P1", "day_index": 4, "resting_heart_rate": 65, "sleep_duration": 5.5, "is_ill": 1},
            {"patient_id": "P2", "day_index": 0, "resting_heart_rate": 70, "sleep_duration": 8.0, "is_ill": 0},
            {"patient_id": "P2", "day_index": 1, "resting_heart_rate": 71, "sleep_duration": 7.5, "is_ill": 0},
            {"patient_id": "P2", "day_index": 2, "resting_heart_rate": 72, "sleep_duration": 7.0, "is_ill": 0},
            {"patient_id": "P2", "day_index": 3, "resting_heart_rate": 75, "sleep_duration": 6.0, "is_ill": 1},
        ]
    )

    csv_path = tmp_path / "wearable.csv"
    data.to_csv(csv_path, index=False)

    dataset = WearableDataset(root=str(tmp_path))
    task = WearableIllnessPrediction(baseline_window=2)

    samples = dataset.set_task(task)

    # Basic sanity checks to make sure pipeline produces usable output
    assert len(samples) > 0

    first_sample = samples[0]
    assert "patient_id" in first_sample
    assert "visit_id" in first_sample
    assert "conditions" in first_sample
    assert "label" in first_sample