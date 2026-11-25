"""Tests for NHANES dataset preprocessing."""

import unittest

import numpy as np
import pandas as pd

from pyhealth.datasets import BaseDataset
from pyhealth.datasets.nhanes import (
    NHANESDataset,
    create_5min_epochs,
    detect_nonwear_choi,
    merge_metadata,
)


class TestNHANESDataset(unittest.TestCase):
    """Tests NHANES dataset preprocessing functions."""

    def setUp(self) -> None:
        self.participant = pd.DataFrame(
            {
                "SEQN": np.full(10080, 21005),
                "PAXN": np.arange(1, 10081),
                "PAXINTEN": np.full(10080, 100.0),
            }
        )

    def test_dataset_inheritance(self) -> None:
        self.assertTrue(issubclass(NHANESDataset, BaseDataset))

    def test_nonwear_below_threshold(self) -> None:
        counts = np.zeros(89)
        result = detect_nonwear_choi(counts)

        self.assertFalse(result.any())

    def test_nonwear_at_threshold(self) -> None:
        counts = np.zeros(90)
        result = detect_nonwear_choi(counts)

        self.assertTrue(result.all())

    def test_nonwear_with_short_spike(self) -> None:
        counts = np.zeros(120)
        counts[60:62] = 10

        result = detect_nonwear_choi(counts)

        self.assertTrue(result.all())

    def test_nonwear_with_long_spike(self) -> None:
        counts = np.zeros(120)
        counts[60:63] = 10

        result = detect_nonwear_choi(counts)

        self.assertFalse(result.any())

    def test_invalid_activity_counts(self) -> None:
        for counts in [
            np.array([1, np.nan]),
            np.array([1, -1]),
            np.array([1, np.inf]),
        ]:
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                detect_nonwear_choi(counts)

    def test_epoch_aggregation(self) -> None:
        epochs = create_5min_epochs(self.participant)

        self.assertEqual(epochs.shape, (2016, 7))
        self.assertEqual(epochs["wear_minutes"].sum(), 10080)

        np.testing.assert_allclose(epochs["intensity"], 100.0)
        np.testing.assert_allclose(epochs["wear_fraction"], 1.0)

    def test_epoch_aggregation_nonwear(self) -> None:
        participant = self.participant.copy()
        participant["PAXINTEN"] = 0.0

        epochs = create_5min_epochs(participant)

        self.assertEqual(epochs["wear_minutes"].sum(), 0)
        self.assertTrue(epochs["intensity"].eq(0).all())
        self.assertTrue(epochs["wear_fraction"].eq(0).all())

    def test_epoch_aggregation_partial_wear(self) -> None:
        participant = self.participant.copy()
        participant.loc[:91, "PAXINTEN"] = 0.0

        epochs = create_5min_epochs(participant)

        epoch = epochs.iloc[18]

        self.assertEqual(epoch["wear_minutes"], 3)
        self.assertAlmostEqual(epoch["wear_fraction"], 0.6)
        self.assertAlmostEqual(epoch["raw_intensity"], 60.0)
        self.assertAlmostEqual(epoch["intensity"], 100.0)

    def test_incomplete_activity(self) -> None:
        with self.assertRaises(ValueError):
            create_5min_epochs(self.participant.iloc[:-1])

    def test_duplicate_epoch_index(self) -> None:
        participant = self.participant.copy()
        participant.loc[1, "PAXN"] = 1

        with self.assertRaises(ValueError):
            create_5min_epochs(participant)

    def test_multiple_participants(self) -> None:
        participant = self.participant.copy()
        participant.loc[0, "SEQN"] = 21006

        with self.assertRaises(ValueError):
            create_5min_epochs(participant)

    def test_metadata_merge(self) -> None:
        activity = pd.DataFrame(
            {
                "patient_id": ["1", "2"],
                "cycle": ["2003-2004", "2003-2004"],
                "activity_file": ["a.parquet", "b.parquet"],
            }
        )

        clinical = pd.DataFrame(
            {
                "patient_id": ["1"],
                "cycle": ["2003-2004"],
                "age": [50],
            }
        )

        mortality = pd.DataFrame(
            {
                "patient_id": ["1"],
                "cycle": ["2003-2004"],
                "mortstat": [1],
            }
        )

        result = merge_metadata(activity, clinical, mortality)

        self.assertEqual(len(result), 2)
        self.assertTrue(result["patient_id"].is_unique)
        self.assertEqual(result.iloc[0]["age"], 50)
        self.assertTrue(pd.isna(result.iloc[1]["age"]))


if __name__ == "__main__":
    unittest.main()
