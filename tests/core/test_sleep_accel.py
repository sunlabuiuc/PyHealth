"""Tests for the Sleep-Accel dataset and heart-rate IBI task."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from pyhealth.datasets import SleepAccelDataset
from pyhealth.datasets.sleep_accel import (
    SLEEP_ACCEL_STAGE_MAP,
    load_sleep_accel_heart_rate,
    load_sleep_accel_stages,
)
from pyhealth.tasks import SleepStagingIBISleepAccel
from pyhealth.tasks.sleep_staging_ibi import _heart_rate_to_grid


def _write_subject(
    root: Path,
    subject: str,
    scores: list[int],
    rates: list[float] | None = None,
) -> None:
    """Writes one synthetic Sleep-Accel subject with all four modalities."""
    for folder in ("heart_rate", "labels", "motion", "steps"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    labels = "".join(f"{i * 30} {score}\n" for i, score in enumerate(scores))
    (root / "labels" / f"{subject}_labeled_sleep.txt").write_text(labels)
    if rates is None:
        rates = [60.0] * (len(scores) + 2)
    times = list(range(-30, 30 * (len(scores) + 1), 30))
    rows = "".join(
        f"{time},{rate}\n" for time, rate in zip(times, rates, strict=False)
    )
    (root / "heart_rate" / f"{subject}_heartrate.txt").write_text(rows)
    (root / "motion" / f"{subject}_acceleration.txt").write_text("0,0,0,0\n")
    (root / "steps" / f"{subject}_steps.txt").write_text("0,0\n")


def _event(root: Path, subject: str) -> SimpleNamespace:
    """Builds one event like the rows of the Sleep-Accel metadata CSV."""
    return SimpleNamespace(
        patient_id=subject,
        record_name=subject,
        label_file=str(root / "labels" / f"{subject}_labeled_sleep.txt"),
        heart_rate_file=str(
            root / "heart_rate" / f"{subject}_heartrate.txt"
        ),
    )


def _patient(subject: str, events: list) -> SimpleNamespace:
    return SimpleNamespace(
        patient_id=subject,
        get_events=lambda event_type=None: events,
    )


class TestSleepAccelDataset(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name) / "sleep-accel"
        self.root.mkdir()

    def test_metadata_columns_and_absolute_paths(self) -> None:
        _write_subject(self.root, "101", [0, 5])

        dataset = SleepAccelDataset(
            root=str(self.root), cache_dir=str(self.root / "cache"), num_workers=1
        )
        frame = pd.read_csv(
            self.root / "sleep_accel-pyhealth.csv", dtype=str, keep_default_na=False
        )

        self.assertEqual(list(frame.columns)[:2], ["patient_id", "record_name"])
        self.assertEqual(frame["patient_id"].tolist(), ["101"])
        row = frame.iloc[0]
        for column in frame.columns[2:]:
            self.assertTrue(Path(row[column]).is_absolute())
            self.assertTrue(Path(row[column]).is_file())
        self.assertEqual(dataset.dataset_name, "sleep_accel")

    def test_incomplete_record_is_skipped_with_warning(self) -> None:
        _write_subject(self.root, "101", [0])
        _write_subject(self.root, "102", [0])
        (self.root / "steps" / "102_steps.txt").unlink()

        with self.assertLogs("pyhealth.datasets.sleep_accel", "WARNING") as logs:
            dataset = SleepAccelDataset(
                root=str(self.root),
                cache_dir=str(self.root / "cache"),
                num_workers=1,
            )

        frame = pd.read_csv(self.root / "sleep_accel-pyhealth.csv", dtype=str)
        self.assertEqual(frame["patient_id"].tolist(), ["101"])
        self.assertIn("missing steps_file", logs.output[0])
        self.assertEqual(len(dataset.unique_patient_ids), 1)

    def test_invalid_root_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "not a directory"):
            SleepAccelDataset(root=str(self.root / "missing"))

    def test_dev_uses_isolated_cache_key(self) -> None:
        _write_subject(self.root, "101", [0])
        cache_dir = str(self.root / "cache")
        dev = SleepAccelDataset(
            root=str(self.root), cache_dir=cache_dir, dev=True, num_workers=1
        )
        full = SleepAccelDataset(
            root=str(self.root), cache_dir=cache_dir, dev=False, num_workers=1
        )

        self.assertTrue(dev.dev)
        self.assertFalse(full.dev)
        self.assertNotEqual(dev.cache_dir, full.cache_dir)


class TestSleepAccelHelpers(unittest.TestCase):
    def test_stage_map_and_loaders(self) -> None:
        self.assertEqual(
            SLEEP_ACCEL_STAGE_MAP,
            {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 4, -1: -1},
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_subject(root, "1", [0, 5, -1])
            stages, starts = load_sleep_accel_stages(
                root / "labels" / "1_labeled_sleep.txt"
            )
            times, bpm = load_sleep_accel_heart_rate(
                root / "heart_rate" / "1_heartrate.txt"
            )
        self.assertEqual(stages.tolist(), [0, 4, -1])
        self.assertEqual(starts.tolist(), [0.0, 30.0, 60.0])
        self.assertEqual(times[0], -30.0)
        self.assertTrue(np.all(bpm == 60.0))

    def test_heart_rate_grid_validation(self) -> None:
        row = _heart_rate_to_grid([0.0, 30.0], [60.0, 120.0], 4, 30.0)
        np.testing.assert_allclose(row, [1.0, 0.875, 0.75, 0.625], atol=1e-6)
        self.assertEqual(row.dtype, np.float32)
        zeroed = _heart_rate_to_grid([0.0, 30.0], [30.0, 300.0], 4, 30.0)
        np.testing.assert_array_equal(zeroed, np.zeros(4))


class TestSleepStagingIBISleepAccel(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name) / "sleep-accel"
        self.root.mkdir()

    def test_label_map_shape_and_unknown(self) -> None:
        _write_subject(self.root, "1", [0, 1, 2, 3, 4, 5, -1])

        samples = SleepStagingIBISleepAccel()(
            _patient("1", [_event(self.root, "1")])
        )

        self.assertEqual(len(samples), 1)
        sample = samples[0]
        self.assertEqual(sample["label"].tolist(), [0, 1, 1, 1, 1, 2, -1])
        self.assertEqual(sample["label"].dtype, np.int64)
        self.assertEqual(sample["signal"].shape, (7, 750))
        self.assertEqual(sample["signal"].dtype, np.float32)
        self.assertEqual(sample["lengths"], 7)
        self.assertEqual(sample["record_id"], "1")
        np.testing.assert_allclose(sample["signal"][0], 1.0)

    def test_max_epochs_truncates(self) -> None:
        _write_subject(self.root, "1", [0, 5, 0, 5])

        samples = SleepStagingIBISleepAccel(max_epochs=2)(
            _patient("1", [_event(self.root, "1")])
        )

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["signal"].shape, (2, 750))
        self.assertEqual(samples[0]["label"].tolist(), [0, 2])

    def test_set_task_integration(self) -> None:
        _write_subject(self.root, "1", [0, 5])
        _write_subject(self.root, "2", [1, -1])

        dataset = SleepAccelDataset(
            root=str(self.root), cache_dir=str(self.root / "cache"), num_workers=1
        )
        samples = dataset.set_task(
            SleepStagingIBISleepAccel(), num_workers=1
        )

        self.assertEqual(len(samples), 2)
        self.assertEqual(
            sorted(sample["patient_id"] for sample in samples), ["1", "2"]
        )
        for sample in samples:
            self.assertEqual(sample["signal"].shape, (2, 750))
            self.assertEqual(sample["label"].dtype, torch.int64)


if __name__ == "__main__":
    unittest.main()
