"""Tests for MetabricDataset, its paper splits and MetabricSurvivalPrediction.

All data are synthetic: four patients written as Survival MDN style ``.npz``
split files in a temporary directory.
"""

import io
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pyhealth.datasets import MetabricDataset, SubjectSplit
from pyhealth.datasets.metabric import METABRIC_FEATURES, prepare_metabric
from pyhealth.tasks import MetabricSurvivalPrediction

# Raw features: mki67, egfr, pgr, erbb2, age, hormone, radio, chemo, er.
RAW = np.array(
    [
        [5.6, 7.8, 10.8, 6.0, 56.84, 1, 1, 0, 1],
        [6.7, 5.3, 8.6, 5.7, 66.91, 0, 0, 0, 0],
        [5.5, 5.3, 10.6, 6.0, 67.85, 1, 0, 0, 1],
        [6.1, 6.0, 6.2, 5.5, 41.20, 0, 1, 1, 0],
    ],
    dtype=np.float32,
)
# Months 120, 36, 0 (zero follow-up) and 240, stored as months / 365.25.
MONTHS = np.array([120.0, 36.0, 0.0, 240.0])
EVENTS = np.array([0.0, 1.0, 1.0, 0.0])
LABELS = np.stack([MONTHS / 365.25, EVENTS], axis=1).astype(np.float32)
# Rows of each part for split 1 and for splits 2..10.
SPLIT_1 = {"train": [0, 1], "valid": [2], "test": [3]}
SPLIT_OTHER = {"train": [3, 2], "valid": [0], "test": [1]}


def write_npz(directory: Path) -> None:
    """Writes 30 synthetic split files in the Survival MDN layout."""
    for split_id in range(1, 11):
        parts = SPLIT_1 if split_id == 1 else SPLIT_OTHER
        for part, rows in parts.items():
            raw = RAW[rows]
            mean, std = RAW[parts["train"]].mean(0), RAW[parts["train"]].std(0) + 1
            np.savez(
                directory / f"{part}_{split_id}.npz",
                (raw - mean) / std,
                LABELS[rows],
                raw,
            )


class TestPrepareMetabric(unittest.TestCase):
    """``prepare_metabric`` turns the split files into one patient table."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        write_npz(self.root)

    def tearDown(self):
        shutil.rmtree(self.root)

    def test_table_contents(self):
        table = pd.read_csv(prepare_metabric(str(self.root)))
        self.assertEqual(len(table), 4)
        self.assertEqual(
            list(table["patient_id"]), ["MB0001", "MB0002", "MB0003", "MB0004"]
        )
        for name in METABRIC_FEATURES:
            self.assertIn(name, table.columns)
        np.testing.assert_allclose(table["duration_months"], MONTHS, atol=1e-3)
        self.assertEqual(list(table["event"]), [0, 1, 1, 0])
        self.assertEqual(list(table["split_1"]), ["train", "train", "valid", "test"])
        self.assertEqual(list(table["split_7"]), ["valid", "test", "train", "train"])

    def test_raw_features_kept(self):
        table = pd.read_csv(prepare_metabric(str(self.root)))
        np.testing.assert_array_equal(
            table[list(METABRIC_FEATURES)].to_numpy().astype(np.float32),
            RAW.astype(np.float32),
        )

    def test_feature_names_match_task(self):
        from pyhealth.tasks import survival_metabric

        self.assertEqual(METABRIC_FEATURES, survival_metabric.METABRIC_FEATURES)

    def test_missing_file(self):
        (self.root / "test_5.npz").unlink()
        with self.assertRaises(FileNotFoundError):
            prepare_metabric(str(self.root))

    def test_mismatched_split(self):
        np.savez(self.root / "test_4.npz", RAW[:1], LABELS[:1] + 1, RAW[:1] + 1)
        with self.assertRaises(ValueError):
            prepare_metabric(str(self.root))

    def test_download_falls_back_to_mirror(self):
        """The first source fails; every file comes from the second one."""
        from unittest import mock
        from urllib.error import HTTPError

        from pyhealth.datasets import metabric

        local = self.root
        out = Path(tempfile.mkdtemp())
        first, second = metabric._SOURCE_URLS
        requested = []

        def fake_urlopen(url, timeout=None):
            requested.append(url)
            if url.startswith(first):
                raise HTTPError(url, 404, "Not Found", None, None)
            return io.BytesIO((local / url.rsplit("/", 1)[1]).read_bytes())

        try:
            with mock.patch.object(metabric.urllib.request, "urlopen", fake_urlopen):
                table = pd.read_csv(prepare_metabric(str(out), download=True))
            self.assertEqual(len(table), 4)
            self.assertEqual(sum(u.startswith(second) for u in requested), 30)
        finally:
            shutil.rmtree(out)

    def test_download_fails_on_all_sources(self):
        from unittest import mock
        from urllib.error import URLError

        from pyhealth.datasets import metabric

        def fail(url, timeout=None):
            raise URLError("offline")

        with mock.patch.object(metabric.urllib.request, "urlopen", fail):
            with self.assertRaisesRegex(OSError, "any source"):
                prepare_metabric(str(self.root), download=True)

    def test_fractional_event_rejected(self):
        # 0.4 must not be rounded to a censored observation.
        labels = LABELS[SPLIT_1["train"]].copy()
        labels[0, 1] = 0.4
        raw = RAW[SPLIT_1["train"]]
        np.savez(self.root / "train_1.npz", raw, labels, raw)
        with self.assertRaisesRegex(ValueError, "event"):
            prepare_metabric(str(self.root))

    def test_non_binary_indicator_rejected(self):
        raw = RAW[SPLIT_1["train"]].copy()
        raw[1, 6] = 0.5
        np.savez(self.root / "train_1.npz", raw, LABELS[SPLIT_1["train"]], raw)
        with self.assertRaisesRegex(ValueError, "exactly 0 or 1"):
            prepare_metabric(str(self.root))

    def test_negative_time_rejected(self):
        labels = LABELS[SPLIT_1["train"]].copy()
        labels[0, 0] = -0.1
        raw = RAW[SPLIT_1["train"]]
        np.savez(self.root / "train_1.npz", raw, labels, raw)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            prepare_metabric(str(self.root))

    def test_wrong_feature_count(self):
        np.savez(self.root / "train_1.npz", RAW[:2, :5], LABELS[:2], RAW[:2, :5])
        with self.assertRaises(ValueError):
            prepare_metabric(str(self.root))


class TestMetabricDataset(unittest.TestCase):
    """Loading, patients and the published splits."""

    @classmethod
    def setUpClass(cls):
        cls.root = Path(tempfile.mkdtemp())
        write_npz(cls.root)
        cls.cache = Path(tempfile.mkdtemp())
        cls.dataset = MetabricDataset(root=str(cls.root), cache_dir=str(cls.cache))
        # Load the event table once here so each test runs in milliseconds.
        cls.dataset.get_patient("MB0001")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root)
        shutil.rmtree(cls.cache)

    def test_builds_csv_and_loads(self):
        self.assertTrue((self.root / "metabric.csv").exists())
        self.assertEqual(self.dataset.dataset_name, "metabric")
        self.assertEqual(len(self.dataset.unique_patient_ids), 4)

    def test_patient_event(self):
        patient = self.dataset.get_patient("MB0002")
        events = patient.get_events(event_type="metabric")
        self.assertEqual(len(events), 1)
        self.assertAlmostEqual(float(events[0]["age_at_diagnosis"]), 66.91, places=2)
        self.assertEqual(int(float(events[0]["event"])), 1)

    def test_missing_files_raise(self):
        empty = Path(tempfile.mkdtemp())
        try:
            with self.assertRaises(FileNotFoundError):
                MetabricDataset(root=str(empty))
        finally:
            shutil.rmtree(empty)

    def test_paper_split_membership(self):
        split = self.dataset.paper_split(1)
        self.assertIsInstance(split, SubjectSplit)
        index = {
            pid: [i] for i, pid in enumerate(["MB0001", "MB0002", "MB0003", "MB0004"])
        }
        train, valid, test = split.split_indices(index)
        self.assertEqual(train.tolist(), [0, 1])
        self.assertEqual(valid.tolist(), [2])
        self.assertEqual(test.tolist(), [3])
        self.assertNotEqual(split.to_dict(), self.dataset.paper_split(2).to_dict())

    def test_paper_split_validation(self):
        for split_id in (0, 11):
            with self.assertRaises(ValueError):
                self.dataset.paper_split(split_id)


class TestMetabricSurvivalPrediction(unittest.TestCase):
    """Samples, labels, feature sets and options of the survival task."""

    @classmethod
    def setUpClass(cls):
        cls.root = Path(tempfile.mkdtemp())
        write_npz(cls.root)
        cls.cache = Path(tempfile.mkdtemp())
        cls.dataset = MetabricDataset(root=str(cls.root), cache_dir=str(cls.cache))
        # Load the event table once here so each test runs in milliseconds.
        cls.dataset.get_patient("MB0001")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root)
        shutil.rmtree(cls.cache)

    def sample(self, task, patient_id="MB0001"):
        return task(self.dataset.get_patient(patient_id))[0]

    def test_schema(self):
        task = MetabricSurvivalPrediction()
        self.assertEqual(task.task_name, "MetabricSurvivalPrediction")
        self.assertEqual(task.input_schema["features"][0], "tensor")
        self.assertEqual(task.output_schema, {"duration": "tensor", "event": "tensor"})
        self.assertEqual(
            MetabricSurvivalPrediction(standardize=False).input_schema,
            {"features": "tensor"},
        )

    def test_sample_and_labels(self):
        sample = self.sample(MetabricSurvivalPrediction(standardize=False))
        self.assertEqual(sample["patient_id"], "MB0001")
        np.testing.assert_allclose(sample["features"], RAW[0], atol=1e-2)
        self.assertAlmostEqual(sample["duration"][0], 120.0, places=2)
        self.assertEqual(sample["event"], [0.0])

    def test_feature_sets(self):
        for name, size in (("all", 9), ("clinical", 5), ("genes", 4)):
            task = MetabricSurvivalPrediction(feature_set=name)
            self.assertEqual(len(self.sample(task)["features"]), size)
        clinical = self.sample(MetabricSurvivalPrediction(feature_set="clinical"))
        self.assertAlmostEqual(clinical["features"][0], 56.84, places=2)

    def test_time_units(self):
        years = self.sample(MetabricSurvivalPrediction(time_unit="years"))
        paper = self.sample(MetabricSurvivalPrediction(time_unit="paper"))
        self.assertAlmostEqual(years["duration"][0], 10.0, places=3)
        self.assertAlmostEqual(paper["duration"][0], 120.0 / 365.25, places=5)

    def test_horizon_censors(self):
        task = MetabricSurvivalPrediction(horizon_months=60.0)
        late = self.sample(task, "MB0004")
        early = self.sample(task, "MB0002")
        self.assertEqual((late["duration"][0], late["event"][0]), (60.0, 0.0))
        self.assertAlmostEqual(early["duration"][0], 36.0, places=2)
        self.assertEqual(early["event"], [1.0])

    def test_zero_duration_raised(self):
        sample = self.sample(MetabricSurvivalPrediction(), "MB0003")
        self.assertAlmostEqual(sample["duration"][0], 0.01)
        self.assertEqual(sample["event"], [1.0])

    def test_invalid_options(self):
        for kwargs in (
            {"feature_set": "imaging"},
            {"time_unit": "days"},
            {"horizon_months": 0},
            {"min_duration_months": 0},
            {"horizon_months": 0.005},
            {"horizon_months": float("inf")},
            {"min_duration_months": float("inf")},
            {"min_duration_months": float("nan")},
            {"horizon_months": 1.0, "min_duration_months": 2.0},
        ):
            with self.assertRaises(ValueError):
                MetabricSurvivalPrediction(**kwargs)

    def test_set_task_with_paper_split(self):
        train, valid, test = self.dataset.set_task(
            MetabricSurvivalPrediction(), split=self.dataset.paper_split(1)
        )
        self.assertEqual((len(train), len(valid), len(test)), (2, 1, 1))
        item = train[0]
        self.assertIsInstance(item["features"], torch.Tensor)
        self.assertEqual(tuple(item["features"].shape), (9,))
        features = torch.stack([train[i]["features"] for i in range(len(train))])
        # z-scored with the two training patients' statistics.
        torch.testing.assert_close(features.mean(0), torch.zeros(9), atol=1e-5, rtol=0)
        self.assertEqual(tuple(item["duration"].shape), (1,))
        self.assertTrue(torch.all(item["duration"] > 0))


if __name__ == "__main__":
    unittest.main()
