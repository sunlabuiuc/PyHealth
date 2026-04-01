"""
Unit tests for the RESPectCCEPDataset.

Author:
    [Your Name]
"""
import os
import shutil
import tempfile
import unittest
import pandas as pd
from pyhealth.datasets import RESPectCCEPDataset

class TestRESPectCCEPDataset(unittest.TestCase):
    def setUp(self):
        # Create a mock BIDS root
        self.root = "test_bids_data"
        if os.path.exists(self.root):
            shutil.rmtree(self.root)
        
        self.cache_dir = tempfile.mkdtemp()
        
        # 1. Create directory structure
        # sub-01 has one session with two runs
        # sub-02 has one session with one run
        self.ieeg_path_1 = os.path.join(self.root, "sub-01/ses-1/ieeg")
        self.ieeg_path_2 = os.path.join(self.root, "sub-02/ses-1/ieeg")
        os.makedirs(self.ieeg_path_1)
        os.makedirs(self.ieeg_path_2)

        # 2. Create mock participants.tsv
        participants_data = {
            "participant_id": ["sub-01", "sub-02"],
            "session": ["ses-1", "ses-1"],
            "age": [25, 40],
            "sex": ["M", "F"],
        }
        pd.DataFrame(participants_data).to_csv(
            os.path.join(self.root, "participants.tsv"), sep="\t", index=False
        )

        # 3. Create mock session-level file (electrodes)
        open(os.path.join(self.ieeg_path_1, "sub-01_ses-1_electrodes.tsv"), 'a').close()
        open(os.path.join(self.ieeg_path_2, "sub-02_ses-1_electrodes.tsv"), 'a').close()

        # 4. Create mock run-level files (the triplets + channels/events)
        runs = [
            (self.ieeg_path_1, "sub-01_ses-1_task-SPES_run-01"),
            (self.ieeg_path_1, "sub-01_ses-1_task-SPES_run-02"),
            (self.ieeg_path_2, "sub-02_ses-1_task-SPES_run-01"),
        ]

        for path, prefix in runs:
            open(os.path.join(path, f"{prefix}_events.tsv"), 'a').close()
            open(os.path.join(path, f"{prefix}_channels.tsv"), 'a').close()
            open(os.path.join(path, f"{prefix}_ieeg.vhdr"), 'a').close()
            open(os.path.join(path, f"{prefix}_ieeg.eeg"), 'a').close()
            open(os.path.join(path, f"{prefix}_ieeg.vmrk"), 'a').close()

        # Initialize dataset
        self.dataset = RESPectCCEPDataset(root=self.root, cache_dir=self.cache_dir)

    def tearDown(self):
        if os.path.exists(self.root):
            shutil.rmtree(self.root)
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)

    def test_stats(self):
        # Ensure stats() doesn't crash
        self.dataset.stats()

    def test_num_patients(self):
        # We created sub-01 and sub-02
        self.assertEqual(len(self.dataset.unique_patient_ids), 2)

    def test_indexing_logic(self):
        """Verify that the indexer correctly found all 3 runs across patients."""
        # Check the generated pyhealth metadata CSV
        metadata_path = os.path.join(self.root, "respect_ccep_metadata-pyhealth.csv")
        self.assertTrue(os.path.exists(metadata_path))
        
        df = pd.read_csv(metadata_path)
        self.assertEqual(len(df), 3) # 2 runs for sub-01, 1 for sub-02

    def test_get_patient_records(self):
        """Verify patient-specific event loading."""
        # Patient 1 should have 2 recording events (runs)
        patient_01 = self.dataset.get_patient("sub-01")
        events = patient_01.get_events()
        self.assertEqual(len(events), 2)
        
        # Verify demographics were merged correctly
        self.assertEqual(int(events[0]['age']), 25)
        self.assertEqual(events[0]['sex'], 'M')

    def test_file_pointers(self):
        """Verify the file paths are correctly stored in attributes."""
        patient_02 = self.dataset.get_patient("sub-02")
        event = patient_02.get_events()[0]
        
        # Check that the pointer to the header file exists and is correct
        self.assertIn('vhdr_file_path', event)
        self.assertTrue(event['vhdr_file_path'].endswith(".vhdr"))
        
        # Verify the file actually exists where the pointer says it does
        full_vhdr_path = os.path.join(self.root, event['vhdr_file_path'])
        self.assertTrue(os.path.exists(full_vhdr_path))

if __name__ == "__main__":
    unittest.main()