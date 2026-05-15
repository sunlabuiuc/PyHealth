"""
Unit tests for the DailyAndSportActivitiesDataset and
DailyAndSportActivitiesClassification classes.

Authors:
    Niam Pattni (npattni2@illinois.edu)
    Sezim Zamirbekova (szami2@illinois.edu)
"""
from pathlib import Path
import shutil
import unittest

import numpy as np
import pandas as pd
import torch

from pyhealth.datasets import DailyAndSportActivitiesDataset
from pyhealth.tasks import DailyAndSportActivitiesClassification

class TestDailyAndSportActivityDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = (
            Path(__file__).parent.parent.parent
            / "test-resources"
            / "core"
            / "daily-sport-activities"
        )
        cls.generate_fake_dataset()
        cls.dataset = DailyAndSportActivitiesDataset(cls.root)
        cls.samples = cls.dataset.set_task(cls.dataset.default_task)

    @classmethod
    def tearDownClass(cls):
        cls.samples.close()
        for activity_dir in cls.root.glob("a*"):
            if activity_dir.is_dir():
                shutil.rmtree(activity_dir)

    @classmethod
    def generate_fake_dataset(cls):
        """
        Creates a tiny synthetic dataset with the same folder structure as the real one.

        Structure:
            root/
                a01/
                    p1/
                        s01.txt
                        s02.txt
                    p7/
                        s01.txt
                    p8/
                        s01.txt
                a02/
                    p1/
                        s01.txt
                    p7/
                        s01.txt
                    p8/
                        s01.txt

        Args:
            root (Path): The root directory to create the dummy folder structure.
        """
        files = [
            ("a01", "p1", "s01.txt"),
            ("a01", "p1", "s02.txt"),
            ("a01", "p7", "s01.txt"),
            ("a01", "p8", "s01.txt"),
            ("a02", "p1", "s01.txt"),
            ("a02", "p7", "s01.txt"),
            ("a02", "p8", "s01.txt"),
        ]

        rng = np.random.default_rng(123)

        for activity, subject, segment in files:
            path = cls.root / activity / subject / segment
            path.parent.mkdir(parents=True, exist_ok=True)
            data = rng.normal(size=(125, 45)).astype(np.float32)
            np.savetxt(path, data, delimiter=",", fmt="%.6f")
    
    def test_stats(self):
        self.dataset.stats()

    def test_num_patients(self):
        self.assertEqual(len(self.dataset.unique_patient_ids), 3)

    def test_load_signal_returns_expected_shape_and_dtype(self):
        file_path = self.root / "a01" / "p1" / "s01.txt"
        signal = self.dataset.load_signal(file_path)

        self.assertEqual(signal.shape, (125, 45))
        self.assertEqual(signal.dtype, np.float32)

    def test_load_signal_invalid_shape_raises(self):
        bad_path = self.root / "a03" / "p1" / "s01.txt"
        bad_path.parent.mkdir(parents=True, exist_ok=True)

        bad_signal = np.random.randn(124, 45).astype(np.float32)
        np.savetxt(bad_path, bad_signal, delimiter=",", fmt="%.6f")

        with self.assertRaises(ValueError):
            self.dataset.load_signal(bad_path)

        bad_path.unlink()
        bad_path.parent.rmdir()
        bad_path.parent.parent.rmdir()

    def test_load_data_returns_event_dataframe(self):
        df = self.dataset.load_data().compute()

        self.assertEqual(len(df), 7)
        self.assertIn("patient_id", df.columns)
        self.assertIn("event_type", df.columns)
        self.assertIn("timestamp", df.columns)
        self.assertIn("daily_sport_activities/record_id", df.columns)
        self.assertIn("daily_sport_activities/visit_id", df.columns)
        self.assertIn("daily_sport_activities/activity_id", df.columns)
        self.assertIn("daily_sport_activities/activity", df.columns)
        self.assertIn("daily_sport_activities/file_path", df.columns)
        self.assertIn("daily_sport_activities/n_rows", df.columns)
        self.assertIn("daily_sport_activities/n_cols", df.columns)

        self.assertEqual(set(df["event_type"].unique()), {"daily_sport_activities"})

    def test_get_patient_p1(self):
        events = self.dataset.get_patient("p1").get_events(
            event_type="daily_sport_activities"
        )

        self.assertEqual(len(events), 3)

        self.assertEqual(events[0]["visit_id"], "s01")
        self.assertEqual(events[1]["visit_id"], "s02")
        self.assertIn(events[2]["activity"], {"sitting", "standing"})

    def test_get_patient_p7(self):
        events = self.dataset.get_patient("p7").get_events(
            event_type="daily_sport_activities"
        )

        self.assertEqual(len(events), 2)
        self.assertEqual(set(event["activity_id"] for event in events), {"a01", "a02"})

    def test_get_patient_p8(self):
        events = self.dataset.get_patient("p8").get_events(
            event_type="daily_sport_activities"
        )

        self.assertEqual(len(events), 2)
        self.assertEqual(set(event["activity_id"] for event in events), {"a01", "a02"})

    def test_event_metadata_is_parsed_correctly(self):
        df = self.dataset.load_data().compute()
        row = df.iloc[0]

        self.assertIn(row["patient_id"], {"p1", "p7", "p8"})
        self.assertIn(row["daily_sport_activities/activity_id"], {"a01", "a02"})
        self.assertIn(row["daily_sport_activities/activity"], {"sitting", "standing"})
        self.assertIn(row["daily_sport_activities/visit_id"], {"s01", "s02"})
        self.assertEqual(row["daily_sport_activities/n_rows"], 125)
        self.assertEqual(row["daily_sport_activities/n_cols"], 45)
        self.assertEqual(row["daily_sport_activities/sampling_rate_hz"], 25)
        self.assertEqual(row["daily_sport_activities/duration_seconds"], 5)

    def test_default_task(self):
        self.assertIsInstance(
            self.dataset.default_task,
            DailyAndSportActivitiesClassification,
        )

    def test_task_generates_expected_number_of_samples(self):
        self.assertEqual(len(self.samples), 28)

    def test_task_sample_structure(self):
        sample = self.samples[0]

        self.assertIn("patient_id", sample)
        self.assertIn("visit_id", sample)
        self.assertIn("record_id", sample)
        self.assertIn("signal", sample)
        self.assertIn("label", sample)

        self.assertEqual(sample["signal"].shape, (50, 45))
        self.assertEqual(sample["signal"].dtype, torch.float32)
        self.assertIsInstance(sample["label"], torch.Tensor)
        self.assertEqual(sample["label"].dtype, torch.int64)

    def test_task_labels_are_correct(self):
        labels = [sample["label"] for sample in self.samples]
        self.assertEqual(labels.count(0), 16)
        self.assertEqual(labels.count(1), 12)

    def test_task_selected_features_reduces_dimension(self):
        samples = self.dataset.set_task(
            DailyAndSportActivitiesClassification(
                signal_loader=self.dataset.load_signal,
                window_size=50,
                stride=25,
                selected_features=[0, 1, 2, 3],
            )
        )

        self.assertGreater(len(samples), 0)
        self.assertEqual(samples[0]["signal"].shape, (50, 4))
        samples.close()

    def test_task_invalid_feature_index_raises(self):
        with self.assertRaises(ValueError):
            self.dataset.set_task(
                DailyAndSportActivitiesClassification(
                    signal_loader=self.dataset.load_signal,
                    window_size=50,
                    stride=25,
                    selected_features=[999],
                )
            )

    def test_task_empty_selected_features_raises(self):
        with self.assertRaises(ValueError):
            self.dataset.set_task(
                DailyAndSportActivitiesClassification(
                    signal_loader=self.dataset.load_signal,
                    window_size=50,
                    stride=25,
                    selected_features=[],
                )
            )

    def test_task_invalid_window_size_raises(self):
        with self.assertRaises(ValueError):
            DailyAndSportActivitiesClassification(
                signal_loader=self.dataset.load_signal,
                window_size=0,
                stride=25,
            )

    def test_task_invalid_stride_raises(self):
        with self.assertRaises(ValueError):
            DailyAndSportActivitiesClassification(
                signal_loader=self.dataset.load_signal,
                window_size=50,
                stride=0,
            )

    def test_task_window_larger_than_signal_raises(self):
        with self.assertRaises(ValueError):
            self.dataset.set_task(
                DailyAndSportActivitiesClassification(
                    signal_loader=self.dataset.load_signal,
                    window_size=200,
                    stride=25,
                )
            )


if __name__ == "__main__":
    unittest.main()