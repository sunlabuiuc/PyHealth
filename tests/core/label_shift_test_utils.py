"""Shared fixture for the label-shift tests.

This module holds the small 6-sample dataset and MLP used by the
end-to-end tests in test_label_shift.py, test_bbse.py, and test_em.py,
so the same sample list is not copied into each file. It defines no
test cases, so unittest discovery does not collect it.
"""

import copy

from pyhealth.datasets import create_sample_dataset
from pyhealth.models import MLP

SAMPLES = [
    {
        "patient_id": "patient-0",
        "visit_id": "visit-0",
        "conditions": ["cond-33", "cond-86", "cond-80", "cond-12"],
        "procedures": [1.0, 2.0, 3.5, 4.0],
        "label": 0,
    },
    {
        "patient_id": "patient-1",
        "visit_id": "visit-1",
        "conditions": ["cond-33", "cond-86", "cond-80"],
        "procedures": [5.0, 2.0, 3.5, 4.0],
        "label": 1,
    },
    {
        "patient_id": "patient-2",
        "visit_id": "visit-2",
        "conditions": ["cond-10", "cond-20", "cond-30"],
        "procedures": [2.0, 3.0, 4.5, 5.0],
        "label": 2,
    },
    {
        "patient_id": "patient-3",
        "visit_id": "visit-3",
        "conditions": ["cond-40", "cond-50"],
        "procedures": [1.5, 2.5, 3.0, 4.5],
        "label": 0,
    },
    {
        "patient_id": "patient-4",
        "visit_id": "visit-4",
        "conditions": ["cond-60", "cond-70", "cond-80"],
        "procedures": [3.0, 4.0, 5.0, 6.0],
        "label": 1,
    },
    {
        "patient_id": "patient-5",
        "visit_id": "visit-5",
        "conditions": ["cond-90", "cond-100"],
        "procedures": [2.5, 3.5, 4.0, 5.5],
        "label": 2,
    },
]


def make_sample_dataset_and_model():
    """Return the shared (dataset, MLP model) pair, model in eval mode.

    The sample list is deep-copied so tests cannot mutate each other's
    fixture through the shared module-level list.
    """
    dataset = create_sample_dataset(
        samples=copy.deepcopy(SAMPLES),
        input_schema={"conditions": "sequence", "procedures": "tensor"},
        output_schema={"label": "multiclass"},
        dataset_name="test",
    )
    model = MLP(
        dataset=dataset,
        feature_keys=["conditions", "procedures"],
        label_key="label",
        mode="multiclass",
    )
    model.eval()
    return dataset, model
