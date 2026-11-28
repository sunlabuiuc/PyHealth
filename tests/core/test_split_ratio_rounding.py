"""Split sizes must not depend on floating-point error in the ratios.

``0.7 + 0.1`` is ``0.7999999999999999`` in floating point. Truncating
``n * (0.7 + 0.1)`` therefore gave one item too few before the second cut
whenever ``n * 0.8`` is a whole number: with 10 patients and ratios
``[0.7, 0.1, 0.2]`` the validation part was empty.
"""

import itertools
import unittest
from fractions import Fraction
from math import floor

import numpy as np

from pyhealth.datasets.sample_dataset import create_sample_dataset
from pyhealth.datasets.splitter import (
    _CUT_RULE,
    PatientSplit,
    _cut_index,
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


def exact_sizes(n: int, ratios) -> list[int]:
    """Part sizes when the ratios are read as the decimals that were written.

    Each cut is ``n`` times the cumulated ratio, rounded down, in exact
    rational arithmetic.
    """
    cuts, total = [], Fraction(0)
    for ratio in ratios[:-1]:
        total += Fraction(str(ratio))
        cuts.append(floor(n * total))
    edges = [0, *cuts, n]
    return [stop - start for start, stop in itertools.pairwise(edges)]


def make_dataset(n_train: int, n_eval: int = 0):
    """One sample per patient; the last ``n_eval`` are the TUH eval split."""
    samples = [
        {
            "patient_id": f"p{i:03d}",
            "record_id": f"r{i}",
            "feature": i,
            "label": i % 2,
            "split": "train" if i < n_train else "eval",
        }
        for i in range(n_train + n_eval)
    ]
    return create_sample_dataset(
        samples=samples,
        input_schema={"feature": "raw"},
        output_schema={"label": "raw"},
    )


def sizes(parts) -> list[int]:
    return [len(part) for part in parts]


class TestCutIndex(unittest.TestCase):
    def test_matches_exact_arithmetic_for_decimal_ratios(self):
        ratio_sets = (
            (0.7, 0.1, 0.2),
            (0.6, 0.2, 0.2),
            (0.8, 0.1, 0.1),
            (0.7, 0.15, 0.15),
            (0.6, 0.1, 0.3),
            (0.58, 0.21, 0.21),
            (0.6, 0.1, 0.1, 0.2),
            (0.4999999, 0.25, 0.2500001),
            (0.3333333, 0.3333333, 0.3333334),
        )
        for ratios in ratio_sets:
            for n in range(501):
                cumulated, total = [], 0.0
                for ratio in ratios[:-1]:
                    total += ratio
                    cumulated.append(_cut_index(n, total))
                edges = [0, *cumulated, n]
                got = [stop - start for start, stop in itertools.pairwise(edges)]
                self.assertEqual(got, exact_sizes(n, ratios), (n, ratios))

    def test_the_documented_example(self):
        self.assertEqual(int(10 * (0.7 + 0.1)), 7)  # the float error
        self.assertEqual(_cut_index(10, 0.7 + 0.1), 8)
        self.assertEqual(_cut_index(100, 0.58), 58)

    def test_products_that_are_not_whole_still_round_down(self):
        self.assertEqual(_cut_index(7, 0.5), 3)
        self.assertEqual(_cut_index(10, 0.79), 7)
        self.assertEqual(_cut_index(3, 2 / 3), 2)
        self.assertEqual(_cut_index(0, 0.7), 0)
        self.assertEqual(_cut_index(10, 1.0), 10)

    def test_ratios_with_more_than_six_decimals_are_not_rounded_up(self):
        # Only float error is corrected: 2 * 0.4999999 is 0.9999998, not 1.
        self.assertEqual(_cut_index(2, 0.4999999), 0)
        self.assertEqual(_cut_index(10**6, 0.4999999), 499999)
        self.assertEqual(_cut_index(10**7, 0.4999999), 4999999)

    def test_decimal_ratios_closest_to_an_integer_are_not_snapped(self):
        # With 7 decimals and n * 10**7 < 10**12, choose the ratio that puts
        # n * ratio exactly 1e-7 below an integer: it must still round down.
        for n in (99999, 99991, 12347):
            k = -pow(n, -1, 10**7) % 10**7
            ratio = float(f"0.{k:07d}")
            with self.subTest(n=n, ratio=ratio):
                exact = n * Fraction(k, 10**7)
                self.assertEqual(exact - floor(exact), Fraction(10**7 - 1, 10**7))
                self.assertEqual(_cut_index(n, ratio), floor(exact))

    def test_large_sizes_match_exact_arithmetic(self):
        for n in (10**6 - 1, 10**6, 10**8, 10**9 + 7):
            for ratios in ((0.7, 0.1, 0.2), (0.6, 0.1, 0.1, 0.2)):
                cumulated, total = [], 0.0
                for ratio in ratios[:-1]:
                    total += ratio
                    cumulated.append(_cut_index(n, total))
                edges = [0, *cumulated, n]
                got = [stop - start for start, stop in itertools.pairwise(edges)]
                self.assertEqual(got, exact_sizes(n, ratios), (n, ratios))


class TestSplitSizes(unittest.TestCase):
    def test_three_way_splitters(self):
        dataset = make_dataset(10)
        for splitter in THREE_WAY:
            with self.subTest(splitter=splitter.__name__):
                parts = splitter(dataset, [0.7, 0.1, 0.2], seed=0)
                self.assertEqual(sizes(parts), [7, 1, 2])

    def test_four_way_splitters(self):
        dataset = make_dataset(10)
        for splitter in FOUR_WAY:
            with self.subTest(splitter=splitter.__name__):
                parts = splitter(dataset, [0.6, 0.1, 0.1, 0.2], seed=0)
                self.assertEqual(sizes(parts), [6, 1, 1, 2])

    def test_tuh_splitters_cut_the_train_split(self):
        dataset = make_dataset(10, n_eval=3)
        for splitter in TUH_THREE_WAY:
            with self.subTest(splitter=splitter.__name__):
                parts = splitter(dataset, [0.7, 0.1, 0.2], seed=0)
                self.assertEqual(sizes(parts), [7, 1, 2, 3])
        dataset = make_dataset(100, n_eval=3)
        for splitter in TUH_TWO_WAY:
            with self.subTest(splitter=splitter.__name__):
                parts = splitter(dataset, [0.58, 0.42], seed=0)
                self.assertEqual(sizes(parts), [58, 42, 3])

    def test_patient_split(self):
        patient_to_index = {f"p{i:03d}": [i] for i in range(10)}
        parts = PatientSplit((0.7, 0.1, 0.2)).split_indices(patient_to_index)
        self.assertEqual(sizes(parts), [7, 1, 2])

    def test_patient_split_description_names_the_cut_rule(self):
        # The description is the processed-cache key: it has to differ from
        # the one used when some sizes were off by one, which had no "cuts".
        description = PatientSplit((0.7, 0.1, 0.2), seed=3).to_dict()
        self.assertEqual(
            description,
            {
                "kind": "patient",
                "ratios": [0.7, 0.1, 0.2],
                "seed": 3,
                "cuts": _CUT_RULE,
            },
        )

    def test_only_one_item_moves_between_adjacent_parts(self):
        # Same shuffle as before the fix: the second cut moves by one, so one
        # patient goes from the test part to the validation part.
        patient_to_index = {f"p{i:03d}": [i] for i in range(10)}
        train, val, test = PatientSplit((0.7, 0.1, 0.2), seed=0).split_indices(
            patient_to_index
        )
        order = np.random.default_rng(0).permutation(10)
        self.assertEqual(sorted(train), sorted(order[:7]))
        self.assertEqual(sorted(val), sorted(order[7:8]))  # was order[7:7]
        self.assertEqual(sorted(test), sorted(order[8:]))  # was order[7:]

    def test_sizes_without_float_error_are_unchanged(self):
        for n, ratios in ((7, [0.5, 0.25, 0.25]), (10, [0.55, 0.25, 0.2])):
            dataset = make_dataset(n)
            for splitter in THREE_WAY:
                with self.subTest(splitter=splitter.__name__, n=n):
                    parts = splitter(dataset, ratios, seed=0)
                    self.assertEqual(sizes(parts), exact_sizes(n, ratios))
                    self.assertEqual(
                        sizes(parts),
                        [
                            int(n * ratios[0]),
                            int(n * (ratios[0] + ratios[1])) - int(n * ratios[0]),
                            n - int(n * (ratios[0] + ratios[1])),
                        ],
                    )


if __name__ == "__main__":
    unittest.main()
