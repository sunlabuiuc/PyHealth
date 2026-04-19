import tempfile
import shutil
import unittest
from pathlib import Path

import dask.dataframe as dd
import numpy as np
import pandas as pd
import scipy.io

from pyhealth.datasets import PTBXLDataset
from pyhealth.tasks.ptbxl_multilabel_classification import PTBXLMultilabelClassification

def write_hea_file(path, record_id, age, sex, dx):
    with open(path, "w") as f:
        f.write(f"{record_id} 12 500 5000\n")
        for lead in ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]:
            f.write(f"{record_id}.mat 16+24 200/mV 16 0 0 0 0 {lead}\n")
        f.write(f"#Age: {age}\n")
        f.write(f"#Sex: {sex}\n")
        f.write(f"#Dx: {dx}\n")
        f.write("#Rx: Unknown\n")
        f.write("#Hx: Unknown\n")
        f.write("#Sx: Unknown\n")

def write_mat_file(path):
    scipy.io.savemat(str(path), {"val": np.random.randn(12, 5000)})

def write_database_csv(path, records):
    pd.DataFrame({
        "ecg_id":     [int(r[0].replace("HR", "")) for r in records],
        "strat_fold": [r[4] for r in records],
    }).to_csv(path, index=False)

class TestPTBXLDataset(unittest.TestCase):
    """Test PTBXLDataset with synthetic test data"""

    # Create records with (record_id, age, sex, dx_codes, strat_fold); at minimum need 1 and 8 for 
    RECORDS = [
        ("HR00001", 56, "Female", "251146004,426783006", 1),  # train
        ("HR00002", 37, "Female", "426783006",           8),  # train
        ("HR00003", 24, "Male",   "426783006",           9),  # val
        ("HR00004", 45, "Female", "164889003",           10), # test
    ]

    @classmethod
    def setUp(cls):
        """Create a temporary directory with 4 synthetic .hea/.mat pairs and a matching ptbxl_database.csv"""
        cls.test_dir = tempfile.mkdtemp()
        for record_id, age, sex, dx, _ in cls.RECORDS:
            write_hea_file(
                Path(cls.test_dir) / f"{record_id}.hea",
                record_id, age, sex, dx
            )
            write_mat_file(Path(cls.test_dir) / f"{record_id}.mat")
        write_database_csv(
            Path(cls.test_dir) / "ptbxl_database.csv",
            cls.RECORDS
        )
        
        cls.dataset = PTBXLDataset(root=cls.test_dir)
        cls.df = cls.dataset.load_data().compute().set_index("patient_id")

    @classmethod
    def tearDown(cls):
        shutil.rmtree(cls.test_dir)

    def test_instantiation(self):
        """Test 1 - Dataset can be instantiated"""
        self.assertIsNotNone(self.dataset)

    def test_dataset_name_default(self):
        """Test 2 - Default dataset name is ptbxl"""
        self.assertEqual(self.dataset.dataset_name, "ptbxl")

    def test_dataset_name_custom(self):
        """Test 3 - Can set a custom dataset name"""
        dataset = PTBXLDataset(root=self.test_dir, dataset_name="my_ptbxl")
        self.assertEqual(dataset.dataset_name, "my_ptbxl")

    def test_default_task_returns_task_instance(self):
        """Test 4 - default_task() returns a PTBXLMultilabelClassification instance"""
        self.assertIsInstance(self.dataset.default_task, PTBXLMultilabelClassification)

    def test_classes_attribute(self):
        """Test 5 - The list of strings CLASSES exists and is not empty"""
        self.assertIsInstance(PTBXLDataset.CLASSES, list)
        self.assertGreater(len(PTBXLDataset.CLASSES), 0)
        self.assertTrue(all(isinstance(c, str) for c in PTBXLDataset.CLASSES))

    def test_load_data_returns_dask_dataframe(self):
        """Test 6 - load_data() returns a Dask DataFrame"""
        self.assertIsInstance(self.dataset.load_data(), dd.DataFrame)

    def test_load_data_row_count(self):
        """Test 7 - load_data() returns one row per .hea file"""
        self.assertEqual(len(self.df), len(self.RECORDS))

    def test_load_data_required_columns(self):
        """Test 8 - load_data() output contains all required BaseDataset columns"""
        for col in ["patient_id", "event_type", "timestamp"]:
            self.assertIn(col, self.df.reset_index().columns, f"Missing required column: {col}")

    def test_load_data_attribute_columns(self):
        """Test 9 - load_data() output contains all ptbxl/ attribute columns"""
        for col in ["ptbxl/mat", "ptbxl/age", "ptbxl/sex",
                    "ptbxl/dx_codes", "ptbxl/dx_abbreviations", "ptbxl/split"]:
            self.assertIn(col, self.df.columns, f"Missing attribute column: {col}")

    def test_load_data_event_type(self):
        """Test 10 - All rows have event_type == ptbxl"""
        self.assertTrue((self.df["event_type"] == "ptbxl").all())

    def test_age_parsed_correctly(self):
        """Test 11 - Ages are parsed correctly from .hea files"""
        self.assertEqual(self.df.loc["HR00001", "ptbxl/age"], 56)

    def test_sex_parsed_correctly(self):
        """Test 12 - Sex is parsed correctly from .hea files"""
        self.assertEqual(self.df.loc["HR00001", "ptbxl/sex"], "Female")
        self.assertEqual(self.df.loc["HR00003", "ptbxl/sex"], "Male")

    def test_dx_codes_parsed_correctly(self):
        """Test 13 - SNOMED CT codes are parsed correctly from .hea files."""
        self.assertEqual(self.df.loc["HR00001", "ptbxl/dx_codes"], "251146004,426783006")
        self.assertEqual(self.df.loc["HR00003", "ptbxl/dx_codes"], "426783006")

    def test_dx_abbreviations_mapped_correctly(self):
        """Test 14 - SNOMED CT codes are mapped to correct abbreviations"""
        self.assertIn("NSR",   self.df.loc["HR00001", "ptbxl/dx_abbreviations"])
        self.assertIn("AF",    self.df.loc["HR00004", "ptbxl/dx_abbreviations"])

    def test_mat_file_path_correct(self):
        """Test 15 - .mat file paths point to the correct location"""
        expected = str(Path(self.test_dir) / "HR00001.mat")
        self.assertEqual(self.df.loc["HR00001", "ptbxl/mat"], expected)

    def test_split_values(self):
        """Test 16 - Split column only contains train, val, or test"""
        self.assertTrue(self.df["ptbxl/split"].isin(["train", "val", "test"]).all())

    def test_split_from_strat_fold(self):
        """Test 17 - Splits are correctly assigned from strat_fold values"""
        self.assertEqual(self.df.loc["HR00001", "ptbxl/split"], "train")  # fold 1
        self.assertEqual(self.df.loc["HR00002", "ptbxl/split"], "train")  # fold 8
        self.assertEqual(self.df.loc["HR00003", "ptbxl/split"], "val")    # fold 9
        self.assertEqual(self.df.loc["HR00004", "ptbxl/split"], "test")   # fold 10

    def test_unknown_snomed_code_skipped(self):
        """Test 18 - SNOMED codes not in mapping are skipped without error"""
        write_hea_file(
            Path(self.test_dir) / "HR00099.hea",
            "HR00099", 30, "Male", "999999999,426783006"
        )
        write_mat_file(Path(self.test_dir) / "HR00099.mat")
        pd.DataFrame({
            "ecg_id":     [int(r[0].replace("HR", "")) for r in self.RECORDS] + [99],
            "strat_fold": [r[4] for r in self.RECORDS] + [1],
        }).to_csv(Path(self.test_dir) / "ptbxl_database.csv", index=False)

        df = PTBXLDataset(root=self.test_dir).load_data().compute()
        df = df.set_index("patient_id")
        self.assertEqual(df.loc["HR00099", "ptbxl/dx_abbreviations"], "NSR")

    def test_invalid_age_handled(self):
        """Test 19 - Non-integer age values result in None without error"""
        write_hea_file(
            Path(self.test_dir) / "HR00098.hea",
            "HR00098", "NaN", "Female", "426783006"
        )
        write_mat_file(Path(self.test_dir) / "HR00098.mat")
        pd.DataFrame({
            "ecg_id":     [int(r[0].replace("HR", "")) for r in self.RECORDS] + [98],
            "strat_fold": [r[4] for r in self.RECORDS] + [1],
        }).to_csv(Path(self.test_dir) / "ptbxl_database.csv", index=False)

        df = PTBXLDataset(root=self.test_dir).load_data().compute()
        df = df.set_index("patient_id")
        self.assertTrue(pd.isna(df.loc["HR00098", "ptbxl/age"]))

    def test_no_hea_files_raises_error(self):
        """Test 20 - FileNotFoundError raised if no .hea files found"""
        empty_dir = tempfile.mkdtemp()
        try:
            dataset = PTBXLDataset(root=empty_dir)
            with self.assertRaises(FileNotFoundError):
                dataset.load_data().compute()
        finally:
            shutil.rmtree(empty_dir)

    def test_missing_csv_raises_error(self):
        """Test 21 - FileNotFoundError raised if ptbxl_database.csv is missing"""
        no_csv_dir = tempfile.mkdtemp()
        try:
            write_hea_file(
                Path(no_csv_dir) / "HR00001.hea",
                "HR00001", 56, "Female", "426783006"
            )
            write_mat_file(Path(no_csv_dir) / "HR00001.mat")
            dataset = PTBXLDataset(root=no_csv_dir)
            with self.assertRaises(FileNotFoundError):
                dataset.load_data().compute()
        finally:
            shutil.rmtree(no_csv_dir)

if __name__ == "__main__":
    unittest.main()