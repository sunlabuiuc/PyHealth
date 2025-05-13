import os
import pandas as pd
import tempfile
import numbers
import unittest
from pyhealth.datasets import NFGDataset  # ensure NFGDataset is importable from the library

class TestNFGDataset(unittest.TestCase):
    def setUp(self):
        # Create a temporary CSV file for testing
        self.temp_file = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv")
        # Write header and some sample rows
        self.temp_file.write("age,sex,cr_type\n")
        self.temp_file.write("65,1,1\n")   # patient 0: age 65, male(1), label 1
        self.temp_file.write("70,0,0\n")   # patient 1: age 70, female(0), label 0
        self.temp_file.flush()
        self.temp_file_name = self.temp_file.name
        self.temp_file.close()
    
    def tearDown(self):
        # Remove temporary file after test
        if os.path.exists(self.temp_file_name):
            os.remove(self.temp_file_name)
    
    def test_load_and_contents(self):
        # Initialize the dataset with the CSV path
        dataset = NFGDataset(csv_path=self.temp_file_name, dataset_name="NFGTest")
        
        # The dataset should have 2 samples (rows)
        self.assertEqual(len(dataset), 2)
        
        # Each sample should be a dict with the proper keys
        sample0 = dataset[0]
        sample1 = dataset[1]
        # Check presence of required keys
        for sample in [sample0, sample1]:
            self.assertIn("patient_id", sample)
            self.assertIn("visit_id", sample)
            self.assertIn("age", sample)
            self.assertIn("sex", sample)
            self.assertIn("cr_type", sample)
        
        # Verify that patient_id and visit_id are unique and correctly assigned
        self.assertNotEqual(sample0["patient_id"], sample1["patient_id"])
        self.assertNotEqual(sample0["visit_id"], sample1["visit_id"])
        
        # Verify the feature values remain raw (no transformations)
        # and label is binary int as expected
        self.assertEqual(sample0["age"], 65)
        self.assertEqual(sample0["sex"], 1)
        self.assertIsInstance(sample0["age"], numbers.Number)
        self.assertIsInstance(sample0["sex"], numbers.Number)
        # Label should be int 0 or 1
        self.assertEqual(sample0["cr_type"], 1)
        self.assertIsInstance(sample0["cr_type"], int)
        
        self.assertEqual(sample1["age"], 70)
        self.assertEqual(sample1["sex"], 0)
        self.assertEqual(sample1["cr_type"], 0)
        self.assertIsInstance(sample1["cr_type"], int)
        '''
        # (Optional) Check dataset input_info for correctness
        info = dataset.input_info
        self.assertEqual(info["age"]["type"], int)
        self.assertEqual(info["age"]["dim"], 0)
        self.assertEqual(info["sex"]["type"], int)
        self.assertEqual(info["sex"]["dim"], 0)
        self.assertEqual(info["cr_type"]["type"], int)
        self.assertEqual(info["cr_type"]["dim"], 0)
        
        # (Optional) Check that the dataset stats report correct counts
        self.assertEqual(len(dataset.patient_to_index), 2)  # 2 patients
        self.assertEqual(len(dataset.visit_to_index), 2)    # 2 visits
        # Each patient_id maps to one index, each visit_id maps to one index
        for pid, indices in dataset.patient_to_index.items():
            self.assertEqual(len(indices), 1)
        for vid, indices in dataset.visit_to_index.items():
            self.assertEqual(len(indices), 1)
        '''
        
# Run the tests
if __name__ == "__main__":
    unittest.main()
