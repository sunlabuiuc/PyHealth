"""Patient-level splits must not depend on the order of the samples."""

import random
import unittest

from pyhealth.datasets import (
    create_sample_dataset,
    split_by_patient,
    split_by_patient_conformal,
)

N_PATIENTS = 30


def _samples(order_seed):
    """Two samples per patient, shuffled into an order set by ``order_seed``."""
    samples = [
        {
            "patient_id": f"p{p:02d}",
            "visit_id": f"p{p:02d}-v{v}",
            "codes": [f"c{(p + v) % 7}"],
            "label": (p + v) % 2,
        }
        for p in range(N_PATIENTS)
        for v in range(2)
    ]
    random.Random(order_seed).shuffle(samples)
    return samples


def _dataset(order_seed):
    return create_sample_dataset(
        samples=_samples(order_seed),
        input_schema={"codes": "sequence"},
        output_schema={"label": "binary"},
        in_memory=True,
    )


def _patients(part):
    return sorted({part[i]["patient_id"] for i in range(len(part))})


class TestSplitByPatientOrder(unittest.TestCase):
    def setUp(self):
        self.a = _dataset(order_seed=0)
        self.b = _dataset(order_seed=1)
        # The two datasets list their patients in different orders.
        self.assertNotEqual(
            list(self.a.patient_to_index), list(self.b.patient_to_index)
        )

    def test_split_by_patient_ignores_sample_order(self):
        parts_a = split_by_patient(self.a, [0.6, 0.2, 0.2], seed=5)
        parts_b = split_by_patient(self.b, [0.6, 0.2, 0.2], seed=5)
        for part_a, part_b in zip(parts_a, parts_b):
            self.assertEqual(_patients(part_a), _patients(part_b))
        self.assertEqual(sum(len(_patients(p)) for p in parts_a), N_PATIENTS)

    def test_split_by_patient_conformal_ignores_sample_order(self):
        ratios = [0.5, 0.2, 0.1, 0.2]
        parts_a = split_by_patient_conformal(self.a, ratios, seed=5)
        parts_b = split_by_patient_conformal(self.b, ratios, seed=5)
        for part_a, part_b in zip(parts_a, parts_b):
            self.assertEqual(_patients(part_a), _patients(part_b))

    def test_seed_still_changes_the_split(self):
        test_5 = _patients(split_by_patient(self.a, [0.6, 0.2, 0.2], seed=5)[2])
        test_6 = _patients(split_by_patient(self.a, [0.6, 0.2, 0.2], seed=6)[2])
        self.assertNotEqual(test_5, test_6)


if __name__ == "__main__":
    unittest.main()
