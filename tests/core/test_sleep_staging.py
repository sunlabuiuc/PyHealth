"""Regression tests for the legacy per-record sleep staging task functions.

``pyhealth.tasks.sleep_staging`` is loaded by file path rather than through the
``pyhealth.tasks`` package so the tests only need ``mne``, ``numpy`` and
``pandas`` (the functions' real I/O is stubbed) instead of the package's full
dependency set.
"""

import importlib.util
import os
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

_MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "pyhealth" / "tasks" / "sleep_staging.py"
)


def _load_sleep_staging():
    spec = importlib.util.spec_from_file_location(
        "pyhealth.tasks.sleep_staging", _MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sleep_staging = _load_sleep_staging()


class _FakeRaw:
    """Minimal stand-in for an ``mne.io.Raw`` object."""

    def __init__(self, data):
        self._data = data

    def get_data(self):
        return self._data

    def to_data_frame(self):
        return self._data


class TestSleepStagingShhsFn(unittest.TestCase):
    """``sleep_staging_shhs_fn`` must not index past the parsed labels."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.load_dir = Path(self._tmp.name) / "in"
        self.save_dir = Path(self._tmp.name) / "out"
        self.load_dir.mkdir()
        self.save_dir.mkdir()
        self.label_file = "shhs1-200001-nsrr.xml"

    def _write_labels(self, labels) -> None:
        stages = "".join(f"<SleepStage>{label}</SleepStage>" for label in labels)
        (self.load_dir / self.label_file).write_text(
            f"<StudyAnnotation><SleepStages>{stages}</SleepStages></StudyAnnotation>"
        )

    def _record(self):
        return [
            {
                "load_from_path": str(self.load_dir),
                "signal_file": "shhs1-200001.edf",
                "label_file": self.label_file,
                "save_to_path": str(self.save_dir),
            }
        ]

    def _run(self, n_epochs, labels):
        self._write_labels(labels)
        sample_length = 125 * 30
        data = np.zeros((14, n_epochs * sample_length))
        with mock.patch.object(
            sleep_staging.mne.io, "read_raw_edf", return_value=_FakeRaw(data)
        ):
            return sleep_staging.sleep_staging_shhs_fn(self._record())

    def test_more_signal_epochs_than_labels(self) -> None:
        """Signal epochs beyond available labels are skipped, not indexed."""

        samples = self._run(n_epochs=3, labels=["0", "1"])
        self.assertEqual(len(samples), 2)
        self.assertEqual([s["label"] for s in samples], ["0", "1"])
        for i, sample in enumerate(samples):
            self.assertEqual(sample["record_id"], f"shhs1-200001-{i}")
            self.assertEqual(sample["patient_id"], "shhs1-200001")
            self.assertTrue(os.path.isfile(sample["epoch_path"]))
            with open(sample["epoch_path"], "rb") as f:
                epoch = pickle.load(f)
            self.assertEqual(epoch["signal"].shape, (2, 125 * 30))
            self.assertEqual(epoch["label"], sample["label"])

    def test_more_labels_than_signal_epochs(self) -> None:
        """Extra labels without a complete signal epoch are ignored."""

        samples = self._run(n_epochs=2, labels=["0", "1", "2", "3"])
        self.assertEqual(len(samples), 2)
        self.assertEqual([s["label"] for s in samples], ["0", "1"])

    def test_matching_epochs_and_labels(self) -> None:
        samples = self._run(n_epochs=3, labels=["0", "1", "2"])
        self.assertEqual(len(samples), 3)
        self.assertEqual([s["label"] for s in samples], ["0", "1", "2"])


class TestSleepStagingIsrucFn(unittest.TestCase):
    """``sleep_staging_isruc_fn`` splits labelled 30 s epochs into sub-epochs."""

    # Shuffled, with reference suffixes, plus columns the task must drop.
    _COLUMNS = ("time", "O2-M1", "F3-A2", "C3-A2", "ECG", "F4-M1", "C4-M1", "O1-M2")

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.load_dir = Path(self._tmp.name) / "in"
        self.save_dir = Path(self._tmp.name) / "out"
        self.load_dir.mkdir()
        self.save_dir.mkdir()

    def _run(self, labels, epoch_seconds=10, label_id=1, label2=None):
        (self.load_dir / "1_1.txt").write_text("".join(f"{x}\n" for x in labels))
        (self.load_dir / "1_2.txt").write_text(
            "".join(f"{x}\n" for x in (label2 or labels))
        )
        n_rows = 200 * 30 * len(labels)
        # Each column holds its own index, so channel order is checkable.
        frame = pd.DataFrame(
            {c: np.full(n_rows, float(i)) for i, c in enumerate(self._COLUMNS)}
        )
        record = [
            {
                "load_from_path": str(self.load_dir),
                "signal_file": "1.rec",
                "label1_file": "1_1.txt",
                "label2_file": "1_2.txt",
                "save_to_path": str(self.save_dir),
                "subject_id": "1",
            }
        ]
        with mock.patch.object(
            sleep_staging.mne.io, "read_raw_edf", return_value=_FakeRaw(frame)
        ):
            return sleep_staging.sleep_staging_isruc_fn(
                record, epoch_seconds=epoch_seconds, label_id=label_id
            )

    def test_splits_epochs_and_orders_channels(self) -> None:
        samples = self._run(labels=[0, 5])
        self.assertEqual([s["label"] for s in samples], ["W"] * 3 + ["R"] * 3)
        self.assertEqual(
            [s["record_id"] for s in samples], [f"1-{i}" for i in range(6)]
        )
        self.assertEqual(
            Path(samples[0]["epoch_path"]).parent, self.save_dir / "10-1" / "1"
        )
        with open(samples[0]["epoch_path"], "rb") as f:
            epoch = pickle.load(f)
        self.assertEqual(epoch["signal"].shape, (6, 200 * 10))
        # F3, F4, C3, C4, O1, O2 by their column index in _COLUMNS.
        self.assertEqual(list(epoch["signal"][:, 0]), [2.0, 5.0, 3.0, 6.0, 7.0, 1.0])

    def test_uses_second_expert_labels(self) -> None:
        samples = self._run(labels=[0], epoch_seconds=30, label_id=2, label2=[3])
        self.assertEqual([s["label"] for s in samples], ["N3"])
        self.assertIn(os.path.join("30-2", "1"), samples[0]["epoch_path"])

    def test_rejects_unknown_label(self) -> None:
        with self.assertRaises(AssertionError):
            self._run(labels=[0, 4])


class TestSleepStagingSleepedfFn(unittest.TestCase):
    """``sleep_staging_sleepedf_fn`` keeps only scored sleep-stage epochs."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.save_dir = Path(self._tmp.name)

    def _run(self, n_epochs, annotations):
        data = np.zeros((7, n_epochs * 100 * 30))
        ann = mock.Mock(
            duration=[d for d, _ in annotations],
            description=[des for _, des in annotations],
        )
        record = [
            {
                "load_from_path": "/data",
                "signal_file": "SC4001E0-PSG.edf",
                "label_file": "SC4001EC-Hypnogram.edf",
                "save_to_path": str(self.save_dir),
            }
        ]
        with (
            mock.patch.object(
                sleep_staging.mne.io, "read_raw_edf", return_value=_FakeRaw(data)
            ),
            mock.patch.object(sleep_staging.mne, "read_annotations", return_value=ann),
        ):
            return sleep_staging.sleep_staging_sleepedf_fn(record)

    def test_expands_durations_and_skips_unscored_epochs(self) -> None:
        samples = self._run(
            n_epochs=5,
            annotations=[
                (60.0, "Sleep stage W"),
                (30.0, "Movement time"),
                (30.0, "Sleep stage ?"),
                (30.0, "Sleep stage R"),
            ],
        )
        self.assertEqual([s["label"] for s in samples], ["W", "W", "R"])
        self.assertEqual(
            [s["record_id"] for s in samples], ["SC4001-0", "SC4001-1", "SC4001-4"]
        )
        self.assertEqual(samples[0]["patient_id"], "SC4001")
        with open(samples[2]["epoch_path"], "rb") as f:
            epoch = pickle.load(f)
        self.assertEqual(epoch["signal"].shape, (7, 100 * 30))
        self.assertEqual(epoch["label"], "R")

    def test_stops_at_shorter_of_signal_and_labels(self) -> None:
        many_labels = self._run(n_epochs=2, annotations=[(120.0, "Sleep stage 2")])
        self.assertEqual(len(many_labels), 2)
        few_labels = self._run(n_epochs=4, annotations=[(30.0, "Sleep stage 3")])
        self.assertEqual([s["label"] for s in few_labels], ["3"])


if __name__ == "__main__":
    unittest.main()
