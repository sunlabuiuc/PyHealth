"""Regression tests for epoch-file cleanup in the legacy cardiology tasks."""

import builtins
import importlib.util
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.io import savemat

# These standalone legacy functions do not require the other task dependencies.
MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "pyhealth/tasks/cardiology_detect.py"
)
spec = importlib.util.spec_from_file_location("cardiology_detect", MODULE_PATH)
cardiology = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cardiology)

TASK_NAMES = (
    "cardiology_isAR_fn",
    "cardiology_isBBBFB_fn",
    "cardiology_isAD_fn",
    "cardiology_isCD_fn",
    "cardiology_isWA_fn",
)


class TestCardiologyEpochFiles(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.signal = np.arange(12 * 7500).reshape(12, 7500)
        savemat(root / "record.mat", {"val": self.signal})
        (root / "record.hea").write_text(
            "#Age: 34\n#Sex: Female\n#Dx: 164889003\n"
            "#Rx: Unknown\n#Hx: Unknown\n#Sx: Unknown\n"
        )
        self.record = [
            {
                "load_from_path": str(root),
                "patient_id": "patient-1",
                "signal_file": "record.mat",
                "label_file": "record.hea",
                "save_to_path": str(root),
            }
        ]
        self.output_files = []

    def tracked_open(self, path, mode):
        handle = self.enterContext(builtins.open(path, mode))
        if mode == "wb":
            # Keep handles alive: garbage collection must not hide the leak.
            self.output_files.append(handle)
        return handle

    def test_outputs_are_closed_and_preserve_epoch_data(self):
        for name in TASK_NAMES:
            with self.subTest(task=name):
                self.output_files.clear()
                with patch.object(cardiology, "open", self.tracked_open, create=True):
                    samples = getattr(cardiology, name)(self.record)
                self.assertEqual(len(samples), 2)
                self.assertEqual(len(self.output_files), 2)
                self.assertTrue(all(handle.closed for handle in self.output_files))
                for index, sample in enumerate(samples):
                    self.assertEqual(sample["patient_id"], "patient-1")
                    self.assertEqual(sample["record_id"], index + 1)
                    self.assertEqual(sample["Sex"], ["Female"])
                    self.assertEqual(sample["Age"], ["34"])
                    with builtins.open(sample["epoch_path"], "rb") as epoch_file:
                        epoch = pickle.load(epoch_file)
                    np.testing.assert_array_equal(
                        epoch["signal"],
                        self.signal[:, index * 2500 : index * 2500 + 5000],
                    )
                    self.assertEqual(epoch["label"], sample["label"])

    def test_output_is_closed_when_serialization_fails(self):
        for name in TASK_NAMES:
            with self.subTest(task=name):
                self.output_files.clear()
                with (
                    patch.object(cardiology, "open", self.tracked_open, create=True),
                    patch.object(
                        cardiology.pickle, "dump", side_effect=OSError("write failed")
                    ),
                    self.assertRaisesRegex(OSError, "write failed"),
                ):
                    getattr(cardiology, name)(self.record)
                self.assertEqual(len(self.output_files), 1)
                self.assertTrue(self.output_files[0].closed)


if __name__ == "__main__":
    unittest.main()
