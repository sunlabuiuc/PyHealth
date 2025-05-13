import os
import shutil
import tempfile
import unittest
import pandas as pd

from pyhealth.datasets.mortality_notes import MortalityNotesDataset

# To test locally I ran: PYTHONPATH="$PWD" pytest ./pyhealth/unittests/test_datasets/test_mortality_notes.py
class TestMortalityNotesDatasetBasic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        adm = pd.DataFrame([
            # hadm_id, subject_id, dischtime, hospital_expire_flag
            [1,  10, "2025-01-02 12:00:00", 1],
            [2,  20, "2025-01-01 00:00:00", 0],
        ], columns=["hadm_id","subject_id","dischtime","hospital_expire_flag"])
        adm.to_csv(os.path.join(cls.tmp,"ADMISSIONS.csv"), index=False)

        notes = pd.DataFrame([
            # row_id, subject_id, hadm_id, chartdate, charttime, category, text
            [100, 10, 1, "2025-01-01", "2025-01-0108:00:00", "Progress note",       "foo"],
            [101, 10, 1, "2025-01-01",                 None, "Progress note",       "bar"], 
            [200, 20, 2, "2025-01-01", "2025-01-0112:00:00", "Progress note",       "baz"],
            [201, 10, 1, "2025-01-02", "2025-01-0207:00:00", "Discharge summary",  "qux"],
        ], columns=["row_id","subject_id","hadm_id","chartdate","charttime","category","text"])
        notes.to_csv(os.path.join(cls.tmp,"NOTEEVENTS.csv"), index=False)

        cls.ds = MortalityNotesDataset(root=cls.tmp, dev=True)
        cls.ds_dev = MortalityNotesDataset(root=cls.tmp, dev=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def test_initialization(self):
        self.assertEqual(self.ds.root, self.tmp)
        self.assertEqual(self.ds.dataset_name, "mortality_notes")
    
    def test_get_final_notes_columns(self):
        final = self.ds.get_final_notes()
        expected = ["Adm_ID","note_id","text","chartdate","charttime","Label"]
        self.assertListEqual(final.columns, expected)

if __name__ == "__main__":
    unittest.main()
