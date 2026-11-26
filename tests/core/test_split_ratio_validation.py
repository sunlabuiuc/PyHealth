"""Regression tests for the public splitters' ratio contract."""

import unittest

import numpy as np

from pyhealth.datasets.sample_dataset import create_sample_dataset
from pyhealth.datasets.splitter import (
    split_by_patient,
    split_by_patient_conformal,
    split_by_patient_conformal_tuh,
    split_by_patient_tuh,
    split_by_sample,
    split_by_sample_conformal,
    split_by_sample_conformal_tuh,
    split_by_sample_tuh,
    split_by_visit,
    split_by_visit_conformal,
)

THREE_WAY = (split_by_visit, split_by_patient, split_by_sample)
FOUR_WAY = (
    split_by_visit_conformal,
    split_by_patient_conformal,
    split_by_sample_conformal,
)
TUH_THREE_WAY = (split_by_patient_conformal_tuh, split_by_sample_conformal_tuh)
TUH_TWO_WAY = (split_by_patient_tuh, split_by_sample_tuh)


class UnreadableDataset:
    """Any attempted dataset access means validation occurred too late."""

    def __getattr__(self, name):
        raise AssertionError(f"dataset was accessed through {name}")

    def __len__(self):
        raise AssertionError("dataset length was accessed")

    def __getitem__(self, index):
        raise AssertionError("dataset item was accessed")


def make_dataset(in_memory=True):
    samples = [
        {
            "patient_id": f"p{i // 2}",
            "record_id": f"r{i}",
            "feature": i,
            "label": i % 2,
            "split": "train" if i < 16 else "eval",
        }
        for i in range(20)
    ]
    return create_sample_dataset(
        samples=samples,
        input_schema={"feature": "raw"},
        output_schema={"label": "raw"},
        in_memory=in_memory,
    )


def ids(partition):
    return {partition[i]["record_id"] for i in range(len(partition))}


class TestSplitRatioValidation(unittest.TestCase):
    def test_negative_ratios_rejected_before_dataset_access_in_all_splitters(self):
        cases = [
            *((fn, [0.8, -0.2, 0.4]) for fn in THREE_WAY),
            *((fn, [0.8, -0.2, 0.1, 0.3]) for fn in FOUR_WAY),
            *((fn, [0.8, -0.2, 0.4]) for fn in TUH_THREE_WAY),
            *((fn, [1.2, -0.2]) for fn in TUH_TWO_WAY),
        ]
        for splitter, ratios in cases:
            offending_index = 0 if splitter in TUH_TWO_WAY else 1
            with self.subTest(splitter=splitter.__name__), self.assertRaisesRegex(
                ValueError, rf"ratios\[{offending_index}\]"
            ):
                splitter(UnreadableDataset(), ratios, seed=42)

    def test_malformed_containers_values_lengths_and_totals(self):
        cases = [
            (0.5, TypeError),
            ("0.5,0.3,0.2", TypeError),
            ((x for x in [0.5, 0.3, 0.2]), TypeError),
            (np.array([[0.5, 0.3, 0.2]]), ValueError),
            ([0.5, 0.5], ValueError),
            ([0.5, 0.3, 0.1, 0.1], ValueError),
            ([True, 0.0, 0.0], TypeError),
            ([np.bool_(True), 0.0, 0.0], TypeError),
            ([0.5, "0.3", 0.2], TypeError),
            ([0.5, complex(0.3), 0.2], TypeError),
            ([0.5, float("nan"), 0.5], ValueError),
            ([0.5, float("inf"), 0.5], ValueError),
            ([1.0 + 1e-12, 0.0, -1e-12], ValueError),
            ([0.5, -1e-12, 0.500000000001], ValueError),
            ([0.5, 0.3, 0.3], ValueError),
        ]
        for ratios, exception in cases:
            with self.subTest(ratios=str(ratios)), self.assertRaisesRegex(
                exception, "ratios"
            ):
                split_by_sample(UnreadableDataset(), ratios)

    def test_length_and_index_paths_validate_before_dataset_access(self):
        for splitters, wrong_length in [
            (THREE_WAY, [0.5, 0.5]),
            (FOUR_WAY, [0.5, 0.3, 0.2]),
            (TUH_THREE_WAY, [0.5, 0.5]),
            (TUH_TWO_WAY, [0.5, 0.3, 0.2]),
        ]:
            for splitter in splitters:
                with self.subTest(splitter=splitter.__name__), self.assertRaisesRegex(
                    ValueError, "ratios"
                ):
                    splitter(UnreadableDataset(), wrong_length)
        for splitter, ratios in [
            (split_by_sample, [0.8, -0.2, 0.4]),
            (split_by_sample_conformal, [0.8, -0.2, 0.1, 0.3]),
            (split_by_patient_tuh, [1.2, -0.2]),
            (split_by_sample_tuh, [1.2, -0.2]),
            (split_by_patient_conformal_tuh, [0.8, -0.2, 0.4]),
            (split_by_sample_conformal_tuh, [0.8, -0.2, 0.4]),
        ]:
            with self.subTest(index_path=splitter.__name__), self.assertRaisesRegex(
                ValueError, "ratios"
            ):
                splitter(UnreadableDataset(), ratios, get_index=True)

    def test_valid_ratios_preserve_patient_partition_and_seeded_membership(self):
        dataset = make_dataset()
        ratios = [0.5, 0.25, 0.25]
        partitions = split_by_patient(dataset, ratios, seed=42)
        patient_order = list(dataset.patient_to_index)
        np.random.default_rng(42).shuffle(patient_order)
        expected = [set(patient_order[:5]), set(patient_order[5:7]), set(patient_order[7:])]
        observed = [{part[i]["patient_id"] for i in range(len(part))} for part in partitions]
        self.assertEqual(observed, expected)
        self.assertEqual([len(part) for part in partitions], [10, 4, 6])
        self.assertEqual(
            [[part[i]["record_id"] for i in range(len(part))] for part in partitions],
            [
                ["r10", "r11", "r12", "r13", "r0", "r1", "r14", "r15", "r6", "r7"],
                ["r4", "r5", "r8", "r9"],
                ["r18", "r19", "r2", "r3", "r16", "r17"],
            ],
        )
        self.assertEqual(set.union(*observed), set(dataset.patient_to_index))
        self.assertEqual(sum(map(len, observed)), len(set.union(*observed)))
        self.assertEqual(ratios, [0.5, 0.25, 0.25])

    def test_valid_tuh_ratios_keep_eval_as_test(self):
        dataset = make_dataset()
        for splitter, ratios in [
            (split_by_patient_tuh, (0.5, 0.5)),
            (split_by_sample_tuh, (0.5, 0.5)),
            (split_by_patient_conformal_tuh, (0.5, 0.25, 0.25)),
            (split_by_sample_conformal_tuh, (0.5, 0.25, 0.25)),
        ]:
            with self.subTest(splitter=splitter.__name__):
                partitions = splitter(dataset, ratios, seed=42)
                observed = [ids(part) for part in partitions]
                self.assertEqual(observed[-1], {f"r{i}" for i in range(16, 20)})
                self.assertEqual(set.union(*observed), {f"r{i}" for i in range(20)})
                self.assertEqual(sum(map(len, observed)), 20)

    def test_valid_generic_splitters_cover_every_record_once(self):
        dataset = make_dataset()
        for splitter, ratios in [
            *((fn, (0.5, 0.25, 0.25)) for fn in THREE_WAY),
            *((fn, (0.5, 0.2, 0.2, 0.1)) for fn in FOUR_WAY),
        ]:
            with self.subTest(splitter=splitter.__name__):
                partitions = splitter(dataset, ratios, seed=42)
                observed = [ids(part) for part in partitions]
                self.assertEqual(set.union(*observed), {f"r{i}" for i in range(20)})
                self.assertEqual(sum(map(len, observed)), 20)

    def test_valid_index_paths_match_returned_subsets(self):
        dataset = make_dataset()
        cases = [
            (split_by_sample, (0.5, 0.25, 0.25)),
            (split_by_sample_conformal, (0.5, 0.2, 0.2, 0.1)),
            (split_by_patient_tuh, (0.5, 0.5)),
            (split_by_sample_tuh, (0.5, 0.5)),
            (split_by_patient_conformal_tuh, (0.5, 0.25, 0.25)),
            (split_by_sample_conformal_tuh, (0.5, 0.25, 0.25)),
        ]
        for splitter, ratios in cases:
            with self.subTest(splitter=splitter.__name__):
                partitions = splitter(dataset, ratios, seed=42)
                indices = splitter(dataset, ratios, seed=42, get_index=True)
                self.assertEqual(len(indices), len(partitions))
                for partition, index_vector in zip(partitions, indices):
                    self.assertEqual(
                        [partition[i]["record_id"] for i in range(len(partition))],
                        [dataset[int(index)]["record_id"] for index in index_vector],
                    )

    def test_disk_backed_patient_split_preserves_partition_mappings(self):
        dataset = make_dataset(in_memory=False)
        partitions = split_by_patient(dataset, (0.5, 0.25, 0.25), seed=42)
        observed = [ids(part) for part in partitions]
        self.assertEqual(set.union(*observed), {f"r{i}" for i in range(20)})
        self.assertEqual(sum(map(len, observed)), 20)
        for partition in partitions:
            self.assertEqual(
                set(partition.patient_to_index),
                {partition[i]["patient_id"] for i in range(len(partition))},
            )

    def test_numpy_ratios_and_floating_sum_tolerance(self):
        dataset = make_dataset()
        ratios = np.array([0.7, 0.2, 0.1])
        partitions = split_by_sample(dataset, ratios, seed=42)
        observed = [ids(part) for part in partitions]
        self.assertEqual(set.union(*observed), {f"r{i}" for i in range(20)})
        self.assertEqual(sum(map(len, observed)), 20)
        # A total just inside the documented tolerance is accepted unchanged.
        near_one = [0.5, 0.3, 0.2000005]
        split_by_sample(dataset, near_one, seed=42)
        self.assertEqual(near_one, [0.5, 0.3, 0.2000005])
        with self.assertRaisesRegex(ValueError, "sum"):
            split_by_sample(UnreadableDataset(), [0.5, 0.3, 0.200002])

    def test_zero_ratios_keep_empty_adjustable_partitions(self):
        dataset = make_dataset()
        train, val, test = split_by_sample(dataset, [1.0, 0.0, 0.0], seed=42)
        self.assertEqual([len(train), len(val), len(test)], [20, 0, 0])
        train, val, test = split_by_patient_tuh(dataset, (1.0, 0.0), seed=42)
        self.assertEqual([len(train), len(val), len(test)], [16, 0, 4])


if __name__ == "__main__":
    unittest.main()
