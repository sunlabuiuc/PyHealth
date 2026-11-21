import unittest

import pyhealth.datasets as D
from pyhealth.datasets import create_sample_dataset


def make_dataset(order=None, split_key="split", train_value="train",
                 eval_value="eval"):
    """10 patients x 3 samples; patients 0-6 are train, 7-9 are eval."""
    samples = [
        {
            "patient_id": f"p{p}",
            "record_id": f"p{p}_{j}",
            "feature": p * 10 + j,
            "label": j % 2,
            split_key: train_value if p < 7 else eval_value,
        }
        for p in range(10)
        for j in range(3)
    ]
    if order is not None:
        samples = [samples[i] for i in order]
    return create_sample_dataset(samples, {"feature": "raw"}, {"label": "raw"})


def patients_of(dataset, index):
    return {dataset[int(i)]["patient_id"] for i in index}


def as_lists(parts):
    return [p.tolist() for p in parts]


# Indices returned by the previous implementations (upstream/master) for
# make_dataset() with seed=42 and get_index=True.
EXPECTED = {
    "split_by_sample_tuh": [
        [16, 10, 15, 20, 12, 14, 7, 6, 9, 3],
        [0, 17, 5, 11, 19, 2, 4, 18, 1, 13, 8],
        [21, 22, 23, 24, 25, 26, 27, 28, 29],
    ],
    "split_by_patient_tuh": [
        [9, 10, 11, 6, 7, 8, 18, 19, 20],
        [12, 13, 14, 3, 4, 5, 15, 16, 17, 0, 1, 2],
        [21, 22, 23, 24, 25, 26, 27, 28, 29],
    ],
    "split_by_sample_conformal_tuh": [
        [16, 10, 15, 20, 12, 14, 7, 6, 9, 3],
        [0, 17, 5, 11, 19],
        [2, 4, 18, 1, 13, 8],
        [21, 22, 23, 24, 25, 26, 27, 28, 29],
    ],
    "split_by_patient_conformal_tuh": [
        [9, 10, 11, 6, 7, 8, 18, 19, 20],
        [12, 13, 14, 3, 4, 5],
        [15, 16, 17, 0, 1, 2],
        [21, 22, 23, 24, 25, 26, 27, 28, 29],
    ],
}

WRAPPER_RATIOS = {
    "split_by_sample_tuh": [0.5, 0.5],
    "split_by_patient_tuh": [0.5, 0.5],
    "split_by_sample_conformal_tuh": [0.5, 0.25, 0.25],
    "split_by_patient_conformal_tuh": [0.5, 0.25, 0.25],
}


class TestSplitByPredefined(unittest.TestCase):
    def test_exported_from_package(self):
        from pyhealth.datasets import split_by_predefined
        self.assertTrue(callable(split_by_predefined))

    def test_heldout_is_eval_partition(self):
        ds = make_dataset()
        *_, test = D.split_by_predefined(
            ds, [0.5, 0.25, 0.25], seed=0, get_index=True
        )
        self.assertEqual(patients_of(ds, test), {"p7", "p8", "p9"})
        self.assertEqual(len(test), 9)

    def test_by_patient_no_patient_in_two_parts(self):
        ds = make_dataset()
        parts = D.split_by_predefined(
            ds, [0.5, 0.25, 0.25], seed=0, get_index=True, by_patient=True
        )
        sets = [patients_of(ds, p) for p in parts]
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                self.assertFalse(sets[i] & sets[j])

    def test_same_seed_same_split(self):
        ds = make_dataset()
        a = D.split_by_predefined(ds, [0.5, 0.5], seed=3, get_index=True)
        b = D.split_by_predefined(ds, [0.5, 0.5], seed=3, get_index=True)
        self.assertEqual(as_lists(a), as_lists(b))

    def test_by_patient_independent_of_sample_order(self):
        ds_a = make_dataset()
        ds_b = make_dataset(order=list(reversed(range(30))))
        a = D.split_by_predefined(
            ds_a, [0.5, 0.25, 0.25], seed=1, get_index=True, by_patient=True
        )
        b = D.split_by_predefined(
            ds_b, [0.5, 0.25, 0.25], seed=1, get_index=True, by_patient=True
        )
        for pa, pb in zip(a, b):
            self.assertEqual(patients_of(ds_a, pa), patients_of(ds_b, pb))

    def test_sample_level_three_ratios(self):
        ds = make_dataset()
        parts = D.split_by_predefined(
            ds, [0.5, 0.25, 0.25], seed=0, get_index=True
        )
        self.assertEqual(len(parts), 4)
        self.assertEqual(
            sorted(int(i) for p in parts for i in p), list(range(30))
        )
        self.assertEqual(len(parts[3]), 9)

    def test_custom_split_key_and_train_value(self):
        ds = make_dataset(
            split_key="partition", train_value="fit", eval_value="holdout"
        )
        train, val, test = D.split_by_predefined(
            ds, [0.5, 0.5], seed=0, get_index=True,
            split_key="partition", train_value="fit",
        )
        self.assertEqual(len(test), 9)
        self.assertEqual(len(train) + len(val), 21)

    def test_missing_split_key_raises(self):
        ds = make_dataset(split_key="other")
        for by_patient in (False, True):
            with self.assertRaises(ValueError):
                D.split_by_predefined(
                    ds, [0.5, 0.5], seed=0, by_patient=by_patient
                )

    def test_bad_ratios_raise(self):
        ds = make_dataset()
        for ratios in ([0.5, 0.4], [1.0], [0.25] * 4):
            with self.assertRaises(ValueError):
                D.split_by_predefined(ds, ratios, seed=0)

    def test_wrappers_forward_to_split_by_predefined(self):
        ds = make_dataset()
        cases = [
            ("split_by_sample_tuh", [0.5, 0.5], False),
            ("split_by_patient_tuh", [0.5, 0.5], True),
            ("split_by_sample_conformal_tuh", [0.5, 0.25, 0.25], False),
            ("split_by_patient_conformal_tuh", [0.5, 0.25, 0.25], True),
        ]
        for name, ratios, by_patient in cases:
            wrapped = getattr(D, name)(ds, ratios, seed=42, get_index=True)
            direct = D.split_by_predefined(
                ds, ratios, seed=42, get_index=True, by_patient=by_patient
            )
            self.assertEqual(as_lists(wrapped), as_lists(direct), name)

    def test_wrappers_match_previous_behavior(self):
        ds = make_dataset()
        for name, ratios in WRAPPER_RATIOS.items():
            parts = getattr(D, name)(ds, ratios, seed=42, get_index=True)
            self.assertEqual(as_lists(parts), EXPECTED[name], name)
