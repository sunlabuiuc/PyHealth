import numpy as np
from califorest_tests.utils import create_synthetic_patient_records

"""
Task pipeline tests using synthetic patient records.

These tests validate:
- Sample processing
- Feature extraction
- Label generation
- Edge case handling
"""

def process_samples(patients):
    """Fake task pipeline"""
    X = []
    y = []

    for p in patients:
        for v in p["visits"]:
            features = len(v["conditions"]) + len(v["drugs"])
            X.append(features)
            y.append(v["label"])

    return np.array(X), np.array(y)


def test_sample_processing():
    patients = create_synthetic_patient_records()
    X, y = process_samples(patients)

    assert len(X) == len(y)
    assert X.ndim == 1


def test_label_generation():
    patients = create_synthetic_patient_records()
    _, y = process_samples(patients)

    assert set(y).issubset({0,1})


def test_edge_cases_empty_patient():
    X, y = process_samples([])
    assert len(X) == 0