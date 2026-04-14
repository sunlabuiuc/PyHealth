"""
Utility functions for generating small synthetic data used in tests.

These helpers ensure tests run quickly, use no real datasets, and
remain fully isolated from external dependencies.
"""

import json
import numpy as np
import tempfile
from pathlib import Path


# ---------------------------------------------------
# Synthetic DATASET generator (for dataset tests)
# ---------------------------------------------------

def create_temp_dataset():
    """
    Create a temporary JSON dataset with 2 synthetic patients.

    Returns
    -------
    temp_dir : TemporaryDirectory
        Temporary directory object (must be cleaned up by tests).
    data_path : Path
        Path to the generated JSON file.
    """
    temp_dir = tempfile.TemporaryDirectory()
    data_path = Path(temp_dir.name) / "patients.json"

    patients = create_synthetic_patient_records()

    with open(data_path, "w") as f:
        json.dump(patients, f)

    return temp_dir, data_path


# ---------------------------------------------------
# Synthetic PATIENT records (for dataset + task tests)
# ---------------------------------------------------

def create_synthetic_patient_records():
    """
    Generate tiny synthetic EHR patient records.

    Returns
    -------
    list
        List of patient dictionaries with visit information.
    """
    return [
        {
            "patient_id": "p1",
            "visits": [
                {
                    "conditions": ["c1", "c2"],
                    "procedures": ["p1"],
                    "drugs": ["d1"],
                    "label": 1,
                },
                {
                    "conditions": ["c3"],
                    "procedures": ["p2"],
                    "drugs": ["d2", "d3"],
                    "label": 0,
                },
            ],
        },
        {
            "patient_id": "p2",
            "visits": [
                {
                    "conditions": ["c4"],
                    "procedures": [],
                    "drugs": ["d4"],
                    "label": 0,
                }
            ],
        },
    ]


# ---------------------------------------------------
# Synthetic MODEL tensors (for model tests)
# ---------------------------------------------------

def create_synthetic_ehr(n_samples: int = 4, n_features: int = 8):
    """
    Generate tiny synthetic tensors for model testing.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    n_features : int
        Number of input features.

    Returns
    -------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Binary labels.
    """
    X = np.random.rand(n_samples, n_features).astype(np.float32)
    y = np.random.randint(0, 2, size=(n_samples,)).astype(np.float32)
    return X, y