import os
import pickle
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd

from pyhealth.tasks import sleep_staging
from pyhealth.tasks.sleep_staging import (
    sleep_staging_isruc_fn,
    sleep_staging_sleepedf_fn,
)


class _FakeRawEDF:
    """Minimal stand-in for the object returned by mne.io.read_raw_edf."""

    def __init__(self, data=None, frame=None):
        self._data = data
        self._frame = frame

    def get_data(self):
        return self._data

    def to_data_frame(self):
        return self._frame


class TestSleepStagingISRUC(unittest.TestCase):
    """Test cases for sleep_staging_isruc_fn."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = self._tmp.name
        with open(os.path.join(self.root, "1_1.txt"), "w") as f:
            f.write("0\n5\n2\n")
        self.record = [
            {
                "load_from_path": self.root,
                "signal_file": "1.edf",
                "label1_file": "1_1.txt",
                "save_to_path": os.path.join(self.root, "epochs"),
                "subject_id": "1",
            }
        ]

    def test_splits_labeled_epochs_on_the_six_eeg_channels(self):
        """Reference suffixes are stripped and non-EEG columns are dropped."""
        num_rows = 3 * 30 * 200
        eeg = ["F3-A2", "C3-A2", "O1-A2", "F4-A1", "C4-A1", "O2-A1"]
        frame = pd.DataFrame(
            {
                "time": np.zeros(num_rows),
                **{c: np.full(num_rows, float(i)) for i, c in enumerate(eeg)},
                "ROC-A1": np.full(num_rows, -1.0),
            }
        )
        with mock.patch.object(
            sleep_staging.mne.io,
            "read_raw_edf",
            return_value=_FakeRawEDF(frame=frame),
        ):
            samples = sleep_staging_isruc_fn(self.record, epoch_seconds=10)

        # each 30 s label covers three 10 s epochs
        self.assertEqual(
            [s["label"] for s in samples], ["W"] * 3 + ["R"] * 3 + ["N2"] * 3
        )
        self.assertEqual(samples[0]["record_id"], "1-0")
        self.assertEqual(samples[0]["patient_id"], "1")
        with open(samples[0]["epoch_path"], "rb") as f:
            epoch = pickle.load(f)
        self.assertEqual(epoch["signal"].shape, (6, 2000))
        # channel order is F3, F4, C3, C4, O1, O2
        np.testing.assert_array_equal(epoch["signal"][:, 0], [0, 3, 1, 4, 2, 5])


class TestSleepStagingSleepEDF(unittest.TestCase):
    """Test cases for sleep_staging_sleepedf_fn."""

    def test_keeps_only_sleep_stage_epochs(self):
        """Unscored and movement epochs are skipped."""
        with tempfile.TemporaryDirectory() as tmp:
            record = [
                {
                    "load_from_path": tmp,
                    "signal_file": "SC4001E0-PSG.edf",
                    "label_file": "SC4001EC-Hypnogram.edf",
                    "save_to_path": tmp,
                }
            ]
            annotations = SimpleNamespace(
                duration=[60.0, 30.0, 30.0, 30.0],
                description=[
                    "Sleep stage W",
                    "Sleep stage ?",
                    "Sleep stage R",
                    "Movement time",
                ],
            )
            signal = np.zeros((2, 5 * 30 * 100))
            with mock.patch.object(
                sleep_staging.mne.io,
                "read_raw_edf",
                return_value=_FakeRawEDF(data=signal),
            ), mock.patch.object(
                sleep_staging.mne, "read_annotations", return_value=annotations
            ):
                samples = sleep_staging_sleepedf_fn(record)

            self.assertEqual(
                [s["record_id"] for s in samples],
                ["SC4001-0", "SC4001-1", "SC4001-3"],
            )
            self.assertEqual([s["label"] for s in samples], ["W", "W", "R"])
            with open(samples[0]["epoch_path"], "rb") as f:
                epoch = pickle.load(f)
            self.assertEqual(epoch["signal"].shape, (2, 3000))


if __name__ == "__main__":
    unittest.main()
