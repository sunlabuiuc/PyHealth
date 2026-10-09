"""Regression tests for the legacy per-record sleep staging task functions.

``pyhealth.tasks.sleep_staging`` is loaded by file path rather than through the
``pyhealth.tasks`` package so the tests only need ``mne`` (the function's real
I/O is stubbed) instead of the package's full dependency set.
"""

import importlib.util
import os
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

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
            f"<StudyAnnotation><SleepStages>{stages}</SleepStages>"
            "</StudyAnnotation>"
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


if __name__ == "__main__":
    unittest.main()
