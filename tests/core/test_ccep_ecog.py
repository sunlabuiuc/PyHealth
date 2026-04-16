import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from pyhealth.datasets import CCEPECoGDataset
from pyhealth.tasks.localize_soz.py import LocalizeSOZ, DatasetCreator, StimulationDataProcessor


class TestCCEPECoGDataset(unittest.TestCase):
    def setUp(self):
        """Generates a minimal synthetic BIDS directory for testing."""
        self.temp_dir = tempfile.mkdtemp()
        root = Path(self.temp_dir)
        
        for i in range(1, 3):
            sub = f"{i:02d}"
            session = "1"
            task = "ecog"
            run = "1"

            # Create BIDS directories
            patient_dir = root / f"sub-{sub}" / f"ses-{session}" / "ieeg"
            patient_dir.mkdir(parents=True, exist_ok=True)

            prefix = f"sub-{sub}_ses-{session}_task-{task}_run-{run}"

            vhdr_path = patient_dir / f"{prefix}_ieeg.vhdr"
            vhdr_path.write_text("Dummy VHDR file content")

            # sub-01: has 'yes' in soz column
            # sub-02: has 'no' in soz column
            soz_vals = ["yes", "no"] if i == 1 else ["no", "no"]
            
            electrodes_path = patient_dir / f"sub-{sub}_ses-{session}_electrodes.tsv"
            df = pd.DataFrame({
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
                "edge": ["no", "no"]
            })
            df.to_csv(electrodes_path, sep="\t", index=False)

            channels_path = patient_dir / f"sub-{sub}_ses-{session}_task-{task}_run-{run}_channels.tsv"
            df_chan = pd.DataFrame({
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
                "status_description": ["included", "included"]
            })
            df_chan.to_csv(channels_path, sep="\t", index=False)
            
            events_path = patient_dir / f"sub-{sub}_ses-{session}_task-{task}_run-{run}_events.tsv"
            df_events = pd.DataFrame({
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
                "notes": ["n/a"]
            })
            df_events.to_csv(events_path, sep="\t", index=False)

    def tearDown(self):
        """Cleanup."""
        shutil.rmtree(self.temp_dir)

    def test_index_data_positive_soz(self):
        """Test CCEPECoGDataset correctly identifies positive SOZ cases."""
        dataset = CCEPECoGDataset(root=self.temp_dir, dev=True)
        
        # Verify Subject 01 (Positive SOZ label case)
        p1_events = dataset.get_patient("01").get_events()
        self.assertEqual(len(p1_events), 1)
        e1 = p1_events[0]
        self.assertEqual(e1["session_id"], "1")
        self.assertEqual(e1["task_id"], "ecog")
        self.assertEqual(e1["run_id"], "1")
        self.assertEqual(str(e1["has_soz"]), 'True')
        self.assertTrue("sub-01_ses-1_task-ecog_run-1_ieeg.vhdr" in e1["header_file"])
        self.assertTrue("sub-01_ses-1_electrodes.tsv" in e1["electrodes_file"])
        self.assertTrue("sub-01_ses-1_task-ecog_run-1_channels.tsv" in e1["channels_file"])
        self.assertTrue("sub-01_ses-1_task-ecog_run-1_events.tsv" in e1["events_file"])

    def test_index_data_negative_soz(self):
        """Test CCEPECoGDataset correctly identifies negative SOZ cases."""
        dataset = CCEPECoGDataset(root=self.temp_dir, dev=True)

        # Verify Subject 02 (Negative SOZ label case)
        p2_events = dataset.get_patient("02").get_events()
        self.assertEqual(len(p2_events), 1)
        e2 = p2_events[0]
        self.assertEqual(e2["session_id"], "1")
        self.assertEqual(e2["task_id"], "ecog")
        self.assertEqual(e2["run_id"], "1")
        self.assertEqual(str(e2["has_soz"]), 'False' )
        self.assertTrue("sub-02_ses-1_task-ecog_run-1_ieeg.vhdr" in e2["header_file"])
        self.assertTrue("sub-02_ses-1_electrodes.tsv" in e2["electrodes_file"])
        self.assertTrue("sub-02_ses-1_task-ecog_run-1_channels.tsv" in e2["channels_file"])
        self.assertTrue("sub-02_ses-1_task-ecog_run-1_events.tsv" in e2["events_file"])

    def test_verify_data_no_root(self):
        """Test _verify_data raises FileNotFoundError for non-existent root."""
        with self.assertRaises(FileNotFoundError):
            CCEPECoGDataset(root="/tmp/non_existent_bids_root")

    def test_verify_data_no_subjects(self):
        """Test _verify_data raises ValueError for empty BIDS root."""
        with tempfile.TemporaryDirectory() as bad_dir:
            with self.assertRaisesRegex(ValueError, "contains no 'sub-\\*' subject folders"):
                CCEPECoGDataset(root=bad_dir)

    def test_verify_data_no_vhdr(self):
        """Test _verify_data raises ValueError when recordings are missing."""
        with tempfile.TemporaryDirectory() as bad_dir:
            root = Path(bad_dir)
            (root / "sub-01" / "ses-1" / "ieeg").mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "contains no '.vhdr' files"):
                CCEPECoGDataset(root=bad_dir)

    def test_verify_data_no_electrodes(self):
        """Test _verify_data raises ValueError when electrodes.tsv is missing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            patient_dir = temp_path / "sub-01" / "ses-1" / "ieeg"
            patient_dir.mkdir(parents=True, exist_ok=True)
            (patient_dir / "sub-01_ses-1_task-ecog_run-1_ieeg.vhdr").write_text("dummy")
            with self.assertRaisesRegex(ValueError, "contains no 'electrodes.tsv' file"):
                CCEPECoGDataset(root=temp_dir)

    def test_verify_data_no_channels(self):
        """Test _verify_data raises ValueError when channels.tsv is missing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            patient_dir = temp_path / "sub-01" / "ses-1" / "ieeg"
            patient_dir.mkdir(parents=True, exist_ok=True)
            (patient_dir / "sub-01_ses-1_task-ecog_run-1_ieeg.vhdr").write_text("dummy")
            (patient_dir / "sub-01_ses-1_electrodes.tsv").write_text("dummy")
            with self.assertRaisesRegex(ValueError, "contains no 'channels.tsv' files"):
                CCEPECoGDataset(root=temp_dir)

    def test_verify_data_no_events(self):
        """Test _verify_data raises ValueError when events.tsv is missing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            patient_dir = temp_path / "sub-01" / "ses-1" / "ieeg"
            patient_dir.mkdir(parents=True, exist_ok=True)
            (patient_dir / "sub-01_ses-1_task-ecog_run-1_ieeg.vhdr").write_text("dummy")
            (patient_dir / "sub-01_ses-1_electrodes.tsv").write_text("dummy")
            (patient_dir / "sub-01_ses-1_task-ecog_run-1_channels.tsv").write_text("dummy")
            with self.assertRaisesRegex(ValueError, "contains no 'events.tsv' files"):
                CCEPECoGDataset(root=temp_dir)

class _DummyPatient:
	def __init__(self, patient_id: str, events: List[_DummyEvent]):
		self.patient_id = patient_id
		self._events = events

	def get_events(self, event_type=None) -> List[_DummyEvent]:
		# Treat all dummy events as belonging to the train split so each event
		# is processed exactly once (eval returns empty).
		if event_type == "eval":
			return []
		return self._events

class TestLocalizeSOZ(unittest.TestCase):
    
    def test_default_task(self):
        self.assertIsInstance(self.dataset.default_task, LocalizeSOZ)

    def test_call_returns_samples_length(self):
        task = LocalizeSOZ()
        

        self.temp_dir = tempfile.mkdtemp()
        root = Path(self.temp_dir)
        dataset = CCEPECoGDataset(root=self.temp_dir, dev=True)
        

        
        self.assertEqual(len(samples[0]), 15)
        self.assertEqual(samples[0]["patient_id"], whatever the entered pid was)
        self.assertIn("header_file",samples[0])
        self.assertIn("events_file",samples[0])
        self.assertIn("channels_file",samples[0])
        self.assertIn("electrodes_file",samples[0])
        self.assertIn("mean_y",samples[0])
        self.assertIn("mean_X_stim",samples[0])
        self.assertIn("mean_X_recording",samples[0])
        self.assertIn("mean_electrode_lobes",samples[0])
        self.assertIn("mean_electrode_coords",samples[0])
        self.assertIn("std_y",samples[0])
        self.assertIn("std_X_stim",samples[0])
        self.assertIn("std_X_recording",samples[0])
        self.assertIn("std_electrode_lobes",samples[0])
        self.assertIn("std_electrode_coords",samples[0])
        
        
if __name__ == "__main__":
    unittest.main()
