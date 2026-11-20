"""Tests for GroupBalancedSampler and get_group_balanced_dataloader."""

import unittest
from collections import Counter

import numpy as np
import torch

from pyhealth.datasets import create_sample_dataset
from pyhealth.sampler import GroupBalancedSampler, get_group_balanced_dataloader

SEXES = ["F"] * 6 + ["M"] * 2
LABELS = [0, 0, 0, 0, 1, 1, 0, 1]


def _dataset():
    samples = [
        {"patient_id": f"p{i}", "x": [float(i)], "y": y, "sex": s}
        for i, (s, y) in enumerate(zip(SEXES, LABELS))
    ]
    return create_sample_dataset(samples, {"x": "tensor"}, {"y": "binary"})


class TestGroupBalancedSampler(unittest.TestCase):
    def test_weights_are_inverse_group_size(self):
        sampler = GroupBalancedSampler(SEXES)
        expected = [1 / 6] * 6 + [1 / 2] * 2
        torch.testing.assert_close(
            sampler.weights, torch.tensor(expected, dtype=torch.double)
        )

    def test_group_by_label_cells(self):
        sampler = GroupBalancedSampler(SEXES, labels=LABELS)
        self.assertAlmostEqual(float(sampler.weights[0]), 1 / 4)
        self.assertAlmostEqual(float(sampler.weights[4]), 1 / 2)
        self.assertAlmostEqual(float(sampler.weights[6]), 1.0)

    def test_draws_groups_evenly(self):
        sampler = GroupBalancedSampler(SEXES, num_samples=4000, seed=0)
        counts = Counter(SEXES[i] for i in sampler)
        self.assertEqual(len(sampler), 4000)
        self.assertAlmostEqual(counts["M"] / 4000, 0.5, delta=0.03)

    def test_reproducible_and_changes_per_epoch(self):
        a = GroupBalancedSampler(SEXES, seed=3)
        b = GroupBalancedSampler(SEXES, seed=3)
        first = list(a)
        self.assertEqual(first, list(b))
        self.assertNotEqual(first, list(a))

    def test_seeds_do_not_overlap_across_epochs(self):
        a = GroupBalancedSampler(SEXES, num_samples=64, seed=0)
        b = GroupBalancedSampler(SEXES, num_samples=64, seed=1)
        list(a)
        self.assertNotEqual(list(a), list(b))

    def test_set_epoch_replays_a_pass(self):
        sampler = GroupBalancedSampler(SEXES, seed=5)
        first = list(sampler)
        sampler.set_epoch(0)
        self.assertEqual(list(sampler), first)

    def test_tensor_and_numpy_groups(self):
        expected = GroupBalancedSampler(SEXES).weights
        codes = [int(s == "M") for s in SEXES]
        for groups in (
            torch.tensor(codes),
            np.array(codes),
            [torch.tensor(c) for c in codes],
        ):
            torch.testing.assert_close(GroupBalancedSampler(groups).weights, expected)

    def test_rejects_missing_groups(self):
        for missing in (None, float("nan")):
            with self.assertRaises(ValueError):
                GroupBalancedSampler(["F", "M", missing])

    def test_rejects_mismatched_labels(self):
        with self.assertRaises(ValueError):
            GroupBalancedSampler(SEXES, labels=LABELS[:-1])
        with self.assertRaises(ValueError):
            GroupBalancedSampler([])


class TestGroupBalancedDataloader(unittest.TestCase):
    def setUp(self):
        self.dataset = _dataset()

    def tearDown(self):
        self.dataset.close()

    def test_batches_keep_raw_fields_and_balance(self):
        loader = get_group_balanced_dataloader(
            self.dataset, batch_size=50, group_key="sex", num_samples=2000, seed=0
        )
        sexes = [s for batch in loader for s in batch["sex"]]
        self.assertEqual(len(sexes), 2000)
        self.assertAlmostEqual(sexes.count("M") / 2000, 0.5, delta=0.04)
        batch = next(iter(loader))
        self.assertEqual(tuple(batch["x"].shape), (50, 1))

    def test_label_key_reads_processed_label(self):
        loader = get_group_balanced_dataloader(
            self.dataset, batch_size=4, group_key="sex", label_key="y", seed=0
        )
        self.assertAlmostEqual(float(loader.sampler.weights[6]), 1.0)

    def test_groups_given_directly(self):
        loader = get_group_balanced_dataloader(
            self.dataset, batch_size=4, groups=SEXES, labels=LABELS, seed=0
        )
        self.assertAlmostEqual(float(loader.sampler.weights[6]), 1.0)

    def test_bad_arguments_raise(self):
        with self.assertRaises(ValueError):
            get_group_balanced_dataloader(self.dataset, 4)
        with self.assertRaises(ValueError):
            get_group_balanced_dataloader(
                self.dataset, 4, group_key="sex", groups=SEXES
            )
        with self.assertRaises(ValueError):
            get_group_balanced_dataloader(self.dataset, 4, groups=SEXES[:-1])

    def test_missing_key_raises(self):
        with self.assertRaises(KeyError):
            get_group_balanced_dataloader(self.dataset, 4, group_key="age")


if __name__ == "__main__":
    unittest.main()
