import os
import pickle
import tempfile
import unittest
import xml.etree.ElementTree as ET
from unittest import mock

import numpy as np

from pyhealth.tasks import sleep_staging
from pyhealth.tasks.sleep_staging import sleep_staging_shhs_fn

SHHS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<PSGAnnotation>
    <EpochLength>30</EpochLength>
    <SleepStages>
        <SleepStage>0</SleepStage>
        <SleepStage>4</SleepStage>
        <SleepStage>2</SleepStage>
        <SleepStage>1</SleepStage>
    </SleepStages>
</PSGAnnotation>
"""


class _FakeRawEDF:
    """Minimal stand-in for the object returned by mne.io.read_raw_edf."""

    def __init__(self, data):
        self._data = data

    def get_data(self):
        return self._data


class TestSleepStagingSHHS(unittest.TestCase):
    """Test cases for sleep_staging_shhs_fn."""

    def setUp(self):
        """Set up a fake SHHS recording (signal path + XML label file)."""
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = os.path.join(self._tmp.name, "shhs")
        self.save_path = os.path.join(self._tmp.name, "epochs")
        os.makedirs(self.root)
        os.makedirs(self.save_path)

        self.signal_file = "shhs1-200001.edf"
        self.label_file = "shhs1-200001.xml"
        with open(os.path.join(self.root, self.label_file), "w") as f:
            f.write(SHHS_XML)

        self.record = [
            {
                "load_from_path": self.root,
                "signal_file": self.signal_file,
                "label_file": self.label_file,
                "save_to_path": self.save_path,
            }
        ]

    def test_xml_parser_available_at_module_level(self):
        """ET must be bound unconditionally so label parsing cannot NameError."""
        self.assertIs(sleep_staging.ET, ET)

    def test_parses_sleep_stage_labels_from_xml(self):
        """The task reads SleepStage labels via ElementTree, with no extra deps."""
        num_epochs = 4
        # 14 channels: skips both missing-channel branches in the task fn.
        signal = np.zeros((14, num_epochs * 125))
        with mock.patch.object(
            sleep_staging.mne.io,
            "read_raw_edf",
            return_value=_FakeRawEDF(signal),
        ):
            samples = sleep_staging_shhs_fn(self.record, epoch_seconds=1)

        self.assertEqual(len(samples), num_epochs)
        for i, sample in enumerate(samples):
            self.assertEqual(sample["record_id"], f"shhs1-200001-{i}")
            self.assertEqual(sample["patient_id"], "shhs1-200001")
            self.assertEqual(sample["label"], ["0", "4", "2", "1"][i])
            self.assertTrue(os.path.exists(sample["epoch_path"]))
            with open(sample["epoch_path"], "rb") as f:
                epoch = pickle.load(f)
            self.assertEqual(epoch["signal"].shape, (2, 125))
            self.assertEqual(epoch["label"], sample["label"])

    def _run_epochs(self, num_epochs):
        signal = np.zeros((14, num_epochs * 125))
        with mock.patch.object(
            sleep_staging.mne.io,
            "read_raw_edf",
            return_value=_FakeRawEDF(signal),
        ):
            return sleep_staging_shhs_fn(self.record, epoch_seconds=1)

    def test_skips_signal_epochs_beyond_available_labels(self):
        """Six signal epochs and four labels yield the four labeled epochs."""
        samples = self._run_epochs(6)
        self.assertEqual([s["label"] for s in samples], ["0", "4", "2", "1"])
        self.assertEqual(
            [s["record_id"] for s in samples],
            [f"shhs1-200001-{i}" for i in range(4)],
        )

    def test_ignores_labels_without_a_complete_signal_epoch(self):
        """Two signal epochs and four labels yield the first two labels."""
        samples = self._run_epochs(2)
        self.assertEqual([s["label"] for s in samples], ["0", "4"])

    def test_drops_missing_channel_rows_before_selecting_eeg(self):
        """15- and 16-channel recordings still yield the two EEG channels."""
        for num_channels in (15, 16):
            with self.subTest(num_channels=num_channels):
                # each row holds its own channel index
                signal = np.repeat(
                    np.arange(num_channels, dtype=float)[:, None], 4 * 125, axis=1
                )
                with mock.patch.object(
                    sleep_staging.mne.io,
                    "read_raw_edf",
                    return_value=_FakeRawEDF(signal),
                ):
                    samples = sleep_staging_shhs_fn(self.record, epoch_seconds=1)

                with open(samples[0]["epoch_path"], "rb") as f:
                    epoch = pickle.load(f)
                np.testing.assert_array_equal(epoch["signal"][:, 0], [2, 7])


if __name__ == "__main__":
    unittest.main()
