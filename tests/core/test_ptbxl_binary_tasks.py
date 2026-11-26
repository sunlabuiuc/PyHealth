import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from pyhealth.tasks import PTBXLCD, PTBXLHYP, PTBXLMI, PTBXLSTTC
from pyhealth.tasks.ptbxl_binary import PTBXLBinaryClassification

SCP_CSV = """,diagnostic,diagnostic_class
IMI,1,MI
LVH,1,HYP
NDT,1,STTC
LAFB,1,CD
NORM,1,NORM
SR,,
"""

TASKS = {"MI": PTBXLMI, "HYP": PTBXLHYP, "STTC": PTBXLSTTC, "CD": PTBXLCD}


def _fake_patient(scp_codes):
    event = SimpleNamespace(
        scp_codes=scp_codes, signal_file="unused", record_id=7, strat_fold=3
    )
    return SimpleNamespace(
        patient_id="1", get_events=lambda event_type=None: [event]
    )


class TestPTBXLBinaryTasks(unittest.TestCase):
    """Tests for the single-diagnosis PTB-XL tasks."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.scp_path = Path(self.tmp.name) / "scp_statements.csv"
        self.scp_path.write_text(SCP_CSV)
        patcher = patch(
            "pyhealth.datasets.ptbxl.load_ptbxl_record",
            return_value=np.zeros((12, 5000), dtype=np.float32),
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.tmp.cleanup)

    def test_labels_for_each_class(self):
        cases = {
            "{'IMI': 80.0, 'SR': 0.0}": {"MI": 1, "HYP": 0, "STTC": 0, "CD": 0},
            "{'LVH': 100.0}": {"MI": 0, "HYP": 1, "STTC": 0, "CD": 0},
            "{'NDT': 100.0}": {"MI": 0, "HYP": 0, "STTC": 1, "CD": 0},
            "{'LAFB': 100.0}": {"MI": 0, "HYP": 0, "STTC": 0, "CD": 1},
            "{'NORM': 100.0, 'SR': 0.0}": {"MI": 0, "HYP": 0, "STTC": 0, "CD": 0},
            "{'IMI': 50.0, 'LVH': 50.0}": {"MI": 1, "HYP": 1, "STTC": 0, "CD": 0},
        }
        for scp_codes, expected in cases.items():
            for name, task_cls in TASKS.items():
                with self.subTest(scp_codes=scp_codes, task=name):
                    task = task_cls(self.scp_path)
                    samples = task(_fake_patient(scp_codes))
                    self.assertEqual(len(samples), 1)
                    self.assertEqual(samples[0]["label"], expected[name])

    def test_sample_fields(self):
        sample = PTBXLMI(self.scp_path)(_fake_patient("{'IMI': 80.0}"))[0]
        self.assertEqual(
            set(sample), {"patient_id", "record_id", "signal", "label", "strat_fold"}
        )
        self.assertEqual(sample["record_id"], "7")
        self.assertEqual(sample["strat_fold"], 3)
        self.assertEqual(sample["signal"].shape, (12, 5000))

    def test_names_and_schema(self):
        expected = {
            "MI": "PTBXL_MI", "HYP": "PTBXL_HYP",
            "STTC": "PTBXL_STTC", "CD": "PTBXL_CD",
        }
        for name, task_cls in TASKS.items():
            task = task_cls(self.scp_path)
            self.assertEqual(task.task_name, expected[name])
            self.assertEqual(task.diagnostic_class, name)
            self.assertEqual(task.input_schema, {"signal": "tensor"})
            self.assertEqual(task.output_schema, {"label": "binary"})

    def test_invalid_diagnostic_class(self):
        with self.assertRaises(ValueError):
            PTBXLBinaryClassification(self.scp_path, "XYZ")


if __name__ == "__main__":
    unittest.main()
