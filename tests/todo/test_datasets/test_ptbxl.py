"""Test PTBXLDataset"""

import unittest
import os
from pyhealth.datasets import PTBXLDataset


class TestPTBXLDataset(unittest.TestCase):

    def test_initialization(self):
        """Test dataset can be initialized"""
        # Skip if data not available
        data_root = os.environ.get('PTBXL_ROOT', '/path/to/ptb-xl')

        if not os.path.exists(os.path.join(data_root, 'ptbxl_database.csv')):
            self.skipTest("PTB-XL data not available")

        dataset = PTBXLDataset(
            root=data_root,
            sampling_rate=100,
            dev=True
        )

        self.assertGreater(len(dataset.unique_patient_ids), 0)
        self.assertEqual(dataset.sampling_rate, 100)

    def test_invalid_sampling_rate(self):
        """Test invalid sampling rate raises error"""
        with self.assertRaises(ValueError):
            PTBXLDataset(root="/tmp", sampling_rate=250)

    def test_get_patient(self):
        """Test getting a patient from the dataset"""
        data_root = os.environ.get('PTBXL_ROOT', '/path/to/ptb-xl')

        if not os.path.exists(os.path.join(data_root, 'ptbxl_database.csv')):
            self.skipTest("PTB-XL data not available")

        dataset = PTBXLDataset(
            root=data_root,
            sampling_rate=100,
            dev=True
        )

        # Get first patient
        patient_id = dataset.unique_patient_ids[0]
        patient = dataset.get_patient(patient_id)

        self.assertEqual(patient.patient_id, patient_id)
        self.assertIsNotNone(patient.data_source)


if __name__ == '__main__':
    unittest.main()
