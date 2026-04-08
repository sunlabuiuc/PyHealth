import os
import tempfile
import unittest

from pyhealth.datasets import PTBXLDataset


class TestPTBXLDataset(unittest.TestCase):
    def test_load_data_dev_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "ptbxl_database.csv")

            with open(csv_path, "w") as f:
                f.write("ecg_id,patient_id,filename_lr,filename_hr,scp_codes\n")
                f.write('1,100,records100/00000/00001_lr,records500/00000/00001_hr,"{\'MI\': 1}"\n')
                f.write('2,101,records100/00000/00002_lr,records500/00000/00002_hr,"{\'NORM\': 1}"\n')

            dataset = PTBXLDataset(
                root=tmpdir,
                dev=True,
            )

            df = dataset.load_data().compute()

            self.assertEqual(len(df), 2)
            self.assertIn("patient_id", df.columns)
            self.assertIn("event_type", df.columns)
            self.assertIn("ptbxl/ecg_id", df.columns)
            self.assertIn("ptbxl/filename_lr", df.columns)
            self.assertIn("ptbxl/scp_codes", df.columns)
            self.assertEqual(str(df.iloc[0]["patient_id"]), "100")
            self.assertEqual(df.iloc[0]["event_type"], "ptbxl")


if __name__ == "__main__":
    unittest.main()