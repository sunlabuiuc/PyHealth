"""Fit z-score normalization on training patients and reuse it on held-out data.

Run with: pixi run -e test python examples/timeseries_normalization.py
All samples are synthetic; no downloads or model training are required.
"""

from datetime import datetime, timedelta

import numpy as np

from pyhealth.datasets import create_sample_dataset


def main() -> None:
    """Demonstrate ordinary and temporal normalization without split leakage."""
    start = datetime(2026, 1, 1)
    raw_samples = [
        {
            "patient_id": "train-patient",
            "vitals": ([start, start + timedelta(hours=1)], np.array([[10.0], [30.0]])),
        },
        {"patient_id": "validation-patient", "vitals": ([start], np.array([[40.0]]))},
        {"patient_id": "test-patient", "vitals": ([start], np.array([[1000.0]]))},
    ]
    # Split raw samples by patient BEFORE fitting any normalization statistics.
    split_patients = {
        "train": {"train-patient"},
        "validation": {"validation-patient"},
        "test": {"test-patient"},
    }
    splits = {
        name: [s for s in raw_samples if s["patient_id"] in patients]
        for name, patients in split_patients.items()
    }

    for alias in ("timeseries", "temporal_timeseries"):
        schema = {"vitals": (alias, {"normalize_strategy": "standard"})}
        train = create_sample_dataset(
            samples=splits["train"], input_schema=schema, output_schema={}
        )
        for name, expected in (("validation", 2.0), ("test", 98.0)):
            held_out = create_sample_dataset(
                samples=splits[name],
                input_schema=schema,
                output_schema={},
                # Supplying processors here prevents fitting them on held-out data.
                input_processors=train.input_processors,
            )
            value = held_out[0]["vitals"]
            if isinstance(value, dict):
                np.testing.assert_array_equal(value["time"], [0.0])
                value = value["value"]
            # Training mean=20, scale=10. The test outlier does not change them.
            np.testing.assert_array_equal(value, [[expected]])
            print(f"{alias}: {name} normalized values = {value.tolist()}")


if __name__ == "__main__":
    main()
