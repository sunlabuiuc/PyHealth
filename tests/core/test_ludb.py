import unittest
import tempfile
import shutil
import numpy as np
from pathlib import Path

try:
    import wfdb
except ImportError:
    wfdb = None

from pyhealth.datasets import LUDBDataset


class TestLUDBDataset(unittest.TestCase):
    """Test LUDB dataset with mock test data."""
    
    def setUp(self):
        """Set up mock WFDB files in temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.root = Path(self.temp_dir)
        
        if wfdb is None:
            self.skipTest("wfdb library not available")
        
        self.num_recordings = 3
        self._create_mock_files()
    
    def _create_mock_files(self):
        """Create mock .hea and .dat files using wfdb library."""
        # Create mock signal data: 12 leads, 5000 samples (10s at 500Hz)
        num_leads = 12
        num_samples = 5000
        sampling_rate = 500
        
        lead_names = ['i', 'ii', 'iii', 'avr', 'avl', 'avf', 
                      'v1', 'v2', 'v3', 'v4', 'v5', 'v6']
        
        for rec_num in range(1, self.num_recordings + 1):
            # Generate random ECG-like signal data (float values in mV)
            signals = np.random.randn(num_samples, num_leads) * 0.5
            
            # Record name should be just the number (wfdb will add .hea/.dat)
            record_name = str(rec_num)
            
            # Write using wfdb.wrsamp
            wfdb.wrsamp(
                record_name=record_name,
                fs=sampling_rate,
                units=['mV'] * num_leads,
                sig_name=lead_names,
                p_signal=signals,
                fmt=['16'] * num_leads,  # 16-bit format
                write_dir=str(self.root),
            )
            
            # Manually add diagnostic metadata to .hea file
            # wfdb doesn't directly support custom comments, so we append them
            hea_file = self.root / f"{record_name}.hea"
            with open(hea_file, 'a') as f:
                f.write(f"#<age>: {np.random.randint(25, 80)}\n")
                f.write(f"#<sex>: {np.random.choice(['M', 'F'])}\n")
                f.write("#<diagnoses>:\n")
                f.write("#Rhythm: Sinus rhythm.\n")
                f.write("#Electric axis of the heart: normal.\n")
    
    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)

    def test_dataset_initialization(self):
        """Test LUDBDataset initialization."""
        dataset = LUDBDataset(root=str(self.root), dev=True)
        
        self.assertIsNotNone(dataset)
        self.assertEqual(dataset.dataset_name, "ludb")
        self.assertEqual(dataset.root, str(self.root))

    def test_metadata_file_created(self):
        """Test ludb-pyhealth.csv created."""
        dataset = LUDBDataset(root=str(self.root), dev=True)
        metadata_file = self.root / "ludb-pyhealth.csv"
        self.assertTrue(metadata_file.exists())

    def test_patient_count(self):
        """Test all recordings are added as patients."""
        dataset = LUDBDataset(root=str(self.root), dev=True)
        self.assertEqual(len(dataset.unique_patient_ids), self.num_recordings)

    def test_stats_method(self):
        """Test stats method."""
        dataset = LUDBDataset(root=str(self.root), dev=True)
        dataset.stats()

    def test_get_patient(self):
        """Test get_patient method."""
        dataset = LUDBDataset(root=str(self.root), dev=True)
        patient_id = dataset.unique_patient_ids[0]
        patient = dataset.get_patient(patient_id)
        
        self.assertIsNotNone(patient)
        self.assertEqual(patient.patient_id, patient_id)

    def test_parse_hea_file(self):
        """Test parse_hea_file static method."""
        hea_path = str(self.root / "1.hea")
        metadata = LUDBDataset.parse_hea_file(hea_path)
        
        self.assertIsInstance(metadata, dict)
        self.assertIn("record_name", metadata)
        self.assertIn("sampling_rate", metadata)
        self.assertIn("num_leads", metadata)
        self.assertEqual(metadata["sampling_rate"], 500)
        self.assertEqual(metadata["num_leads"], 12)

    def test_parse_dat_file(self):
        """Test parse_dat_file static method."""
        dat_path = str(self.root / "1.dat")
        signal = LUDBDataset.parse_dat_file(dat_path)
        
        self.assertIsNotNone(signal)
        self.assertEqual(signal.shape[0], 12)  # 12 leads
        self.assertEqual(signal.shape[1], 5000)  # 5000 samples
        self.assertEqual(signal.dtype.name, "float64")

    def test_split_path(self):
        """Test _split_path helper method."""
        test_path = "/path/to/file.hea"
        directory, filename = LUDBDataset._split_path(test_path)
        
        self.assertIsInstance(directory, str)
        self.assertIsInstance(filename, str)
        self.assertEqual(filename, "file.hea")


if __name__ == "__main__":
    unittest.main()

