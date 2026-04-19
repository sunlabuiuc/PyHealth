import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import polars as pl

from pyhealth.tasks.ptbxl_mi_classification import PTBXLMIClassificationTask
from pyhealth.data import Patient


class TestPTBXLTask(unittest.TestCase):
    @patch.object(PTBXLMIClassificationTask, "_load_ecg_signal")
    def test_mi_label_extraction(self, mock_load_signal):
        mock_load_signal.return_value = np.zeros((12, 1000), dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            scp_path = os.path.join(tmpdir, "scp_statements.csv")

            # minimal synthetic SCP mapping
            scp_df = pd.DataFrame(
                {
                    "diagnostic_class": ["MI", "NORM"],
                },
                index=["IMI", "NORM"],
            )
            scp_df.to_csv(scp_path)

            df = pd.DataFrame(
                {
                    "patient_id": ["1", "1"],
                    "event_type": ["ptbxl", "ptbxl"],
                    "timestamp": [None, None],
                    "ptbxl/ecg_id": [100, 101],
                    "ptbxl/record_path": [
                        "records100/00000/00001_lr",
                        "records100/00000/00002_lr",
                    ],
                    "ptbxl/scp_codes": [
                        "{'IMI': 1}",
                        "{'NORM': 1}",
                    ],
                }
            )

            patient = Patient(
                patient_id="1",
                data_source=pl.from_pandas(df),
            )

            task = PTBXLMIClassificationTask(root=tmpdir)
            samples = task(patient)

            self.assertEqual(len(samples), 2)
            self.assertEqual(samples[0]["label"], 1)
            self.assertEqual(samples[1]["label"], 0)
            self.assertEqual(np.array(samples[0]["signal"]).shape, (12, 1000))


if __name__ == "__main__":
    unittest.main()