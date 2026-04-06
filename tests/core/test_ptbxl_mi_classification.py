import unittest
import pandas as pd
import polars as pl

from pyhealth.tasks.ptbxl_mi_classification import PTBXLMIClassificationTask
from pyhealth.data import Patient


class TestPTBXLTask(unittest.TestCase):

    def test_mi_label_extraction(self):
        # synthetic patient data
        df = pd.DataFrame({
            "patient_id": ["1", "1"],
            "event_type": ["ptbxl", "ptbxl"],
            "timestamp": [None, None],
            "ptbxl/ecg_id": [100, 101],
            "ptbxl/filename_lr": ["a", "b"],
            "ptbxl/filename_hr": ["a", "b"],
            "ptbxl/scp_codes": [
                "{'MI': 1}",     # should be label = 1
                "{'NORM': 1}"    # should be label = 0
            ],
        })

        pl_df = pl.from_pandas(df)

        patient = Patient(
            patient_id="1",
            data_source=pl_df
        )

        task = PTBXLMIClassificationTask()
        samples = task(patient)

        self.assertEqual(len(samples), 2)
        self.assertEqual(samples[0]["label"], 1)
        self.assertEqual(samples[1]["label"], 0)


if __name__ == "__main__":
    unittest.main()