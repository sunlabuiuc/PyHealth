from pathlib import Path
import shutil
import tempfile
from typing import List
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from pyhealth.datasets import CCEPECoGDataset
from pyhealth.tasks.localize_soz import LocalizeSOZ


class _DummyEvent:
    """Minimal stand-in for a BIDS ECoG run event."""

    def __init__(
        self,
        header_file="dummy.vhdr",
        events_file="dummy_events.tsv",
        channels_file="dummy_channels.tsv",
        electrodes_file="dummy_electrodes.tsv",
        session_id="1",
        task_id="ecog",
        run_id="1",
    ):
        self.header_file = header_file
        self.events_file = events_file
        self.channels_file = channels_file
        self.electrodes_file = electrodes_file
        self.session_id = session_id
        self.task_id = task_id
        self.run_id = run_id


class _DummyPatient:
    """Minimal stand-in for a PyHealth Patient."""

    def __init__(self, patient_id: str, events: List[_DummyEvent]):
        self.patient_id = patient_id
        self._events = events

    def get_events(self, event_type=None) -> List[_DummyEvent]:
        # Return events only for the "ecog" split; treat all others as empty
        # so each event is processed exactly once.
        if event_type in ("train", "eval"):
            return []
        return self._events


class TestCCEPECoGDataset(unittest.TestCase):
    """Tests for CCEPECoGDataset indexing and validation."""

    def setUp(self):
        """Generate a minimal synthetic BIDS directory."""
        self.temp_dir = tempfile.mkdtemp()
        root = Path(self.temp_dir)

        for i in range(1, 3):
            sub = f"{i:02d}"
            session = "1"
            task = "ecog"
            run = "1"

            patient_dir = root / f"sub-{sub}" / f"ses-{session}" / "ieeg"
            patient_dir.mkdir(parents=True, exist_ok=True)
            prefix = f"sub-{sub}_ses-{session}_task-{task}_run-{run}"

            (patient_dir / f"{prefix}_ieeg.vhdr").write_text("Dummy VHDR file content")

            soz_vals = ["yes", "no"] if i == 1 else ["no", "no"]
            pd.DataFrame({
                "name": ["PT01", "PT02"],
                "x": [-45.2, -47.7],
                "y": [-81.2, -80.2],
                "z": [-1.6, 2.8],
                "size": [4.2, 4.2],
                "material": ["Platinum", "Platinum"],
                "manufacturer": ["AdTech", "AdTech"],
                "group": ["grid", "grid"],
                "hemisphere": ["L", "L"],
                "silicon": ["no", "no"],
                "soz": soz_vals,
                "resected": ["no", "no"],
                "edge": ["no", "no"],
            }).to_csv(patient_dir / f"sub-{sub}_ses-{session}_electrodes.tsv", sep="\t", index=False)

            pd.DataFrame({
                "name": ["PT01", "PT02"],
                "type": ["ECOG", "ECOG"],
                "units": ["µV", "µV"],
                "low_cutoff": [232, 232],
                "high_cutoff": [0.15, 0.15],
                "reference": ["G2", "G2"],
                "group": ["grid", "grid"],
                "sampling_frequency": [512, 512],
                "notch": ["n/a", "n/a"],
                "status": ["good", "good"],
                "status_description": ["included", "included"],
            }).to_csv(patient_dir / f"{prefix}_channels.tsv", sep="\t", index=False)

            pd.DataFrame({
                "onset": [507.00585],
                "duration": [0.23242],
                "trial_type": ["artefact"],
                "sub_type": ["n/a"],
                "electrodes_involved_onset": ["all"],
                "electrodes_involved_offset": ["all"],
                "offset": [507.23828],
                "sample_start": [259587],
                "sample_end": [259706],
                "electrical_stimulation_type": ["n/a"],
                "electrical_stimulation_site": ["n/a"],
                "electrical_stimulation_current": ["n/a"],
                "electrical_stimulation_frequency": ["n/a"],
                "electrical_stimulation_pulsewidth": ["n/a"],
                "notes": ["n/a"],
            }).to_csv(patient_dir / f"{prefix}_events.tsv", sep="\t", index=False)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_index_data_positive_soz(self):
        """CCEPECoGDataset correctly identifies positive SOZ cases."""
        dataset = CCEPECoGDataset(root=self.temp_dir, dev=True)
        p1_events = dataset.get_patient("01").get_events()
        self.assertEqual(len(p1_events), 1)
        e1 = p1_events[0]
        self.assertEqual(e1["session_id"], "1")
        self.assertEqual(e1["task_id"], "ecog")
        self.assertEqual(e1["run_id"], "1")
        self.assertEqual(str(e1["has_soz"]), "True")
        self.assertIn("sub-01_ses-1_task-ecog_run-1_ieeg.vhdr", e1["header_file"])
        self.assertIn("sub-01_ses-1_electrodes.tsv", e1["electrodes_file"])
        self.assertIn("sub-01_ses-1_task-ecog_run-1_channels.tsv", e1["channels_file"])
        self.assertIn("sub-01_ses-1_task-ecog_run-1_events.tsv", e1["events_file"])

    def test_index_data_negative_soz(self):
        """CCEPECoGDataset correctly identifies negative SOZ cases."""
        dataset = CCEPECoGDataset(root=self.temp_dir, dev=True)
        p2_events = dataset.get_patient("02").get_events()
        self.assertEqual(len(p2_events), 1)
        e2 = p2_events[0]
        self.assertEqual(e2["session_id"], "1")
        self.assertEqual(e2["task_id"], "ecog")
        self.assertEqual(e2["run_id"], "1")
        self.assertEqual(str(e2["has_soz"]), "False")
        self.assertIn("sub-02_ses-1_task-ecog_run-1_ieeg.vhdr", e2["header_file"])
        self.assertIn("sub-02_ses-1_electrodes.tsv", e2["electrodes_file"])
        self.assertIn("sub-02_ses-1_task-ecog_run-1_channels.tsv", e2["channels_file"])
        self.assertIn("sub-02_ses-1_task-ecog_run-1_events.tsv", e2["events_file"])

    def test_default_task(self):
        """CCEPECoGDataset.default_task returns a LocalizeSOZ instance."""
        dataset = CCEPECoGDataset(root=self.temp_dir, dev=True)
        self.assertIsInstance(dataset.default_task, LocalizeSOZ)

    def test_verify_data_no_root(self):
        """_verify_data raises FileNotFoundError for a non-existent root."""
        with self.assertRaises(FileNotFoundError):
            CCEPECoGDataset(root="/tmp/non_existent_bids_root")

    def test_verify_data_no_subjects(self):
        """_verify_data raises ValueError for an empty BIDS root."""
        with tempfile.TemporaryDirectory() as bad_dir:
            with self.assertRaisesRegex(ValueError, "contains no 'sub-\\*' subject folders"):
                CCEPECoGDataset(root=bad_dir)

    def test_verify_data_no_vhdr(self):
        """_verify_data raises ValueError when .vhdr recordings are missing."""
        with tempfile.TemporaryDirectory() as bad_dir:
            (Path(bad_dir) / "sub-01" / "ses-1" / "ieeg").mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "contains no '.vhdr' files"):
                CCEPECoGDataset(root=bad_dir)

    def test_verify_data_no_electrodes(self):
        """_verify_data raises ValueError when electrodes.tsv is missing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            patient_dir = Path(temp_dir) / "sub-01" / "ses-1" / "ieeg"
            patient_dir.mkdir(parents=True, exist_ok=True)
            (patient_dir / "sub-01_ses-1_task-ecog_run-1_ieeg.vhdr").write_text("dummy")
            with self.assertRaisesRegex(ValueError, "contains no 'electrodes.tsv' file"):
                CCEPECoGDataset(root=temp_dir)

    def test_verify_data_no_channels(self):
        """_verify_data raises ValueError when channels.tsv is missing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            patient_dir = Path(temp_dir) / "sub-01" / "ses-1" / "ieeg"
            patient_dir.mkdir(parents=True, exist_ok=True)
            (patient_dir / "sub-01_ses-1_task-ecog_run-1_ieeg.vhdr").write_text("dummy")
            (patient_dir / "sub-01_ses-1_electrodes.tsv").write_text("dummy")
            with self.assertRaisesRegex(ValueError, "contains no 'channels.tsv' files"):
                CCEPECoGDataset(root=temp_dir)

    def test_verify_data_no_events(self):
        """_verify_data raises ValueError when events.tsv is missing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            patient_dir = Path(temp_dir) / "sub-01" / "ses-1" / "ieeg"
            patient_dir.mkdir(parents=True, exist_ok=True)
            (patient_dir / "sub-01_ses-1_task-ecog_run-1_ieeg.vhdr").write_text("dummy")
            (patient_dir / "sub-01_ses-1_electrodes.tsv").write_text("dummy")
            (patient_dir / "sub-01_ses-1_task-ecog_run-1_channels.tsv").write_text("dummy")
            with self.assertRaisesRegex(ValueError, "contains no 'events.tsv' files"):
                CCEPECoGDataset(root=temp_dir)


class TestLocalizeSOZ(unittest.TestCase):
    """Tests for the LocalizeSOZ task class."""

    _EXPECTED_SAMPLE_KEYS = {
        "patient_id", "visit_id", "record_id", "session_id", "task_id",
        "run_id", "channel", "electrode_index", "header_file", "events_file",
        "channels_file", "electrodes_file", "soz", "X_stim", "X_recording",
        "electrode_lobes", "electrode_coords",
    }

    @staticmethod
    def _make_processed(n: int = 2):
        """Return a fake process_for_analysis result with *n* electrodes."""
        channels = [f"CH{i:02d}" for i in range(n)]
        lobes = list(range(n))
        y = np.zeros(n, dtype=np.int32)
        if n > 0:
            y[0] = 1
        coords = [[-45.2, -81.2, -1.6]] * n
        X_stim = np.zeros((n, 2, 5, 10), dtype=np.float32)
        X_recording = np.zeros((n, 2, 8, 10), dtype=np.float32)
        return channels, lobes, y, coords, X_stim, X_recording

    def test_task_name(self):
        self.assertEqual(LocalizeSOZ().task_name, "LocalizeSOZ")

    def test_input_schema_keys_and_types(self):
        schema = LocalizeSOZ().input_schema
        for key in ("X_stim", "X_recording", "electrode_lobes", "electrode_coords"):
            with self.subTest(key=key):
                self.assertIn(key, schema)
                self.assertEqual(schema[key], "tensor")

    def test_output_schema(self):
        schema = LocalizeSOZ().output_schema
        self.assertIn("soz", schema)
        self.assertEqual(schema["soz"], "binary")

    def test_call_empty_patient_returns_empty_list(self):
        """Patient with no events returns an empty sample list."""
        samples = LocalizeSOZ()(_DummyPatient("sub-01", []))
        self.assertEqual(samples, [])

    @patch("pyhealth.tasks.localize_soz.DatasetCreator.process_for_analysis")
    @patch("pyhealth.tasks.localize_soz.StimulationDataProcessor.process_run_data")
    @patch("pyhealth.tasks.localize_soz.pd.read_csv")
    @patch("pyhealth.tasks.localize_soz.mne.io.read_raw_brainvision")
    def test_call_sample_count_matches_electrode_count(self, mock_eeg, mock_csv, mock_proc, mock_analysis):
        """One sample is emitted per electrode."""
        mock_eeg.return_value = MagicMock()
        mock_csv.return_value = pd.DataFrame()
        mock_proc.return_value = pd.DataFrame()
        mock_analysis.return_value = self._make_processed(n=3)

        samples = LocalizeSOZ()(_DummyPatient("sub-01", [_DummyEvent()]))
        self.assertEqual(len(samples), 3)

    @patch("pyhealth.tasks.localize_soz.DatasetCreator.process_for_analysis")
    @patch("pyhealth.tasks.localize_soz.StimulationDataProcessor.process_run_data")
    @patch("pyhealth.tasks.localize_soz.pd.read_csv")
    @patch("pyhealth.tasks.localize_soz.mne.io.read_raw_brainvision")
    def test_call_sample_has_all_expected_keys(self, mock_eeg, mock_csv, mock_proc, mock_analysis):
        """Every sample dict contains exactly the expected keys."""
        mock_eeg.return_value = MagicMock()
        mock_csv.return_value = pd.DataFrame()
        mock_proc.return_value = pd.DataFrame()
        mock_analysis.return_value = self._make_processed(n=2)

        samples = LocalizeSOZ()(_DummyPatient("sub-01", [_DummyEvent()]))
        self.assertEqual(set(samples[0].keys()), self._EXPECTED_SAMPLE_KEYS)

    @patch("pyhealth.tasks.localize_soz.DatasetCreator.process_for_analysis")
    @patch("pyhealth.tasks.localize_soz.StimulationDataProcessor.process_run_data")
    @patch("pyhealth.tasks.localize_soz.pd.read_csv")
    @patch("pyhealth.tasks.localize_soz.mne.io.read_raw_brainvision")
    def test_call_soz_labels_match_y_array(self, mock_eeg, mock_csv, mock_proc, mock_analysis):
        """soz labels in samples exactly match the y array from processing."""
        mock_eeg.return_value = MagicMock()
        mock_csv.return_value = pd.DataFrame()
        mock_proc.return_value = pd.DataFrame()
        channels, lobes, _, coords, X_stim, X_recording = self._make_processed(n=2)
        y = np.array([1, 0], dtype=np.int32)
        mock_analysis.return_value = (channels, lobes, y, coords, X_stim, X_recording)

        samples = LocalizeSOZ()(_DummyPatient("sub-01", [_DummyEvent()]))
        self.assertEqual(samples[0]["soz"], 1)
        self.assertEqual(samples[1]["soz"], 0)

    @patch("pyhealth.tasks.localize_soz.DatasetCreator.process_for_analysis")
    @patch("pyhealth.tasks.localize_soz.StimulationDataProcessor.process_run_data")
    @patch("pyhealth.tasks.localize_soz.pd.read_csv")
    @patch("pyhealth.tasks.localize_soz.mne.io.read_raw_brainvision")
    def test_call_visit_id_format(self, mock_eeg, mock_csv, mock_proc, mock_analysis):
        """visit_id follows the {pid}-{session}-{run}-{channel} format."""
        mock_eeg.return_value = MagicMock()
        mock_csv.return_value = pd.DataFrame()
        mock_proc.return_value = pd.DataFrame()
        channels, lobes, y, coords, X_stim, X_recording = self._make_processed(n=1)
        channels = ["PT01"]
        mock_analysis.return_value = (channels, lobes, y, coords, X_stim, X_recording)

        samples = LocalizeSOZ()(_DummyPatient("sub-01", [_DummyEvent(session_id="2", run_id="3")]))
        self.assertEqual(samples[0]["visit_id"], "sub-01-2-3-PT01")

    @patch("pyhealth.tasks.localize_soz.DatasetCreator.process_for_analysis")
    @patch("pyhealth.tasks.localize_soz.StimulationDataProcessor.process_run_data")
    @patch("pyhealth.tasks.localize_soz.pd.read_csv")
    @patch("pyhealth.tasks.localize_soz.mne.io.read_raw_brainvision")
    def test_call_process_run_data_none_skips_event(self, mock_eeg, mock_csv, mock_proc, mock_analysis):
        """When process_run_data returns None the event is skipped entirely."""
        mock_eeg.return_value = MagicMock()
        mock_csv.return_value = pd.DataFrame()
        mock_proc.return_value = None

        samples = LocalizeSOZ()(_DummyPatient("sub-01", [_DummyEvent()]))
        self.assertEqual(samples, [])
        mock_analysis.assert_not_called()

    @patch("pyhealth.tasks.localize_soz.DatasetCreator.process_for_analysis")
    @patch("pyhealth.tasks.localize_soz.StimulationDataProcessor.process_run_data")
    @patch("pyhealth.tasks.localize_soz.pd.read_csv")
    @patch("pyhealth.tasks.localize_soz.mne.io.read_raw_brainvision")
    def test_call_multiple_events_aggregate(self, mock_eeg, mock_csv, mock_proc, mock_analysis):
        """Samples accumulate across multiple events for one patient."""
        mock_eeg.return_value = MagicMock()
        mock_csv.return_value = pd.DataFrame()
        mock_proc.return_value = pd.DataFrame()
        mock_analysis.return_value = self._make_processed(n=2)

        events = [_DummyEvent(run_id="1"), _DummyEvent(run_id="2")]
        samples = LocalizeSOZ()(_DummyPatient("sub-01", events))
        # 2 electrodes × 2 events = 4 samples
        self.assertEqual(len(samples), 4)


if __name__ == "__main__":
    unittest.main()
