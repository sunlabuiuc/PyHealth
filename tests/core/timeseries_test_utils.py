"""Shared fixtures for the per-step time-series tests.

These small datasets are used by test_mlp.py and test_logistic_regression.py,
so the same builders are not copied into each file. It defines no test
cases, so unittest discovery does not collect it.
"""

from datetime import datetime, timedelta

from pyhealth.datasets import create_sample_dataset


def padded_series():
    """Series of 2, 5 and 3 hourly steps; the last one is all zeros."""
    t0 = datetime(2020, 1, 1)
    samples = []
    for i, n in enumerate([2, 5, 3]):
        steps = [[float(h + 1), 2.0 * i + 1, -1.0] for h in range(n)]
        samples.append({
            "patient_id": f"patient-{i}",
            "vitals": (
                [t0 + timedelta(hours=h) for h in range(n)],
                steps if i < 2 else [[0.0, 0.0, 0.0]] * n,
            ),
            "label": i % 2,
        })
    return create_sample_dataset(
        samples=samples,
        input_schema={"vitals": "timeseries"},
        output_schema={"label": "binary"},
        dataset_name="test_padding",
    )


def nested_multihot():
    """Two visits per patient as per-visit multi-hot rows."""
    samples = [
        {
            "patient_id": f"patient-{i}",
            "visits": [["a", "b"], ["c"] if i % 2 else ["a"]],
            "label": i % 2,
        }
        for i in range(4)
    ]
    return create_sample_dataset(
        samples=samples,
        input_schema={"visits": "nested_multihot"},
        output_schema={"label": "binary"},
        dataset_name="test_multihot",
    )


# Other 3D numeric inputs embedded by an nn.Linear over their last dimension;
# no step is all zeros, so every step counts in the mean.
NUMERIC_INPUTS = {
    "nested_sequence_floats": lambda i: [[1.0 + i, 2.0], [3.0, 4.0 - i]],
    "stagenet_tensor": lambda i: (
        [0.0, 1.5, 3.0],
        [[1.0, 2.0 + i, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0 - i]],
    ),
    "tensor": lambda i: [[1.0, 2.0 + i], [3.0, 4.0], [5.0 - i, 6.0]],
}


def numeric_input(input_type):
    """Four samples of one field of the given input type."""
    samples = [
        {
            "patient_id": f"patient-{i}",
            "x": NUMERIC_INPUTS[input_type](i),
            "label": i % 2,
        }
        for i in range(4)
    ]
    return create_sample_dataset(
        samples=samples,
        input_schema={"x": input_type},
        output_schema={"label": "binary"},
        dataset_name=f"test_{input_type}",
    )
