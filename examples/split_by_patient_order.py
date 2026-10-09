"""split_by_patient selects the same patients however the samples are ordered.

The same synthetic samples are built twice in different orders and split with
the same seed; the test patients match.

Run: python examples/split_by_patient_order.py
"""

import random

from pyhealth.datasets import create_sample_dataset, split_by_patient


def build(order_seed):
    samples = [
        {
            "patient_id": f"p{p:02d}",
            "visit_id": f"p{p:02d}-v{v}",
            "codes": [f"c{(p + v) % 5}"],
            "label": (p + v) % 2,
        }
        for p in range(20)
        for v in range(2)
    ]
    random.Random(order_seed).shuffle(samples)
    return create_sample_dataset(
        samples=samples,
        input_schema={"codes": "sequence"},
        output_schema={"label": "binary"},
    )


def test_patients(dataset):
    _, _, test = split_by_patient(dataset, [0.7, 0.1, 0.2], seed=42)
    return sorted({test[i]["patient_id"] for i in range(len(test))})


if __name__ == "__main__":
    a, b = build(order_seed=0), build(order_seed=1)
    print("first patients, order A:", list(a.patient_to_index)[:4])
    print("first patients, order B:", list(b.patient_to_index)[:4])
    print("test patients, order A: ", test_patients(a))
    print("test patients, order B: ", test_patients(b))
    assert test_patients(a) == test_patients(b)
