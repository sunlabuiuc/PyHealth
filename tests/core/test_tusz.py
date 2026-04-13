import os
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import List
import pandas as pd
import numpy as np
from unittest.mock import patch

from pyhealth.datasets import TUSZDataset
from pyhealth.tasks import TUSZTask

SAMPLE_RATE = 200
FEATURE_SAMPLE_RATE = 50

@dataclass
class _DummyEvent:
	signal_file: str


class _DummyPatient:
	def __init__(self, patient_id: str, events: List[_DummyEvent]):
		self.patient_id = patient_id
		self._events = events

	def get_events(self, event_type=None) -> List[_DummyEvent]:
		# Treat all dummy events as belonging to the train split so each event
		# is processed exactly once (eval returns empty).
		if event_type == "eval":
			return []
		return self._events

class TestTUSZDataset(unittest.TestCase):

    def _touch(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")

    def test_prepare_metadata_creates_expected_csvs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            # Minimal filesystem layout: train has normal + abnormal, eval too
            train_normal_dir = root / "train" / "01_tcp_ar"
            train_abnormal_dir = root / "train" / "01_tcp_ar"
            eval_normal_dir = root / "eval" / "01_tcp_ar"
            eval_abnormal_dir = root / "eval" / "01_tcp_ar"

            self._touch(train_normal_dir / "aaaaaaac_s005_2002.edf")
            self._touch(train_abnormal_dir / "aaaaaayf_s003_2003.edf")
            self._touch(eval_normal_dir / "aaaaaarq_s016_2014.edf")
            self._touch(eval_abnormal_dir / "aaaaaabg_s002_2006.edf")

            # Call prepare_metadata without invoking BaseDataset init
            ds = TUSZDataset.__new__(TUSZDataset)
            ds.root = str(root)
            ds.root_path = Path(root)
            ds.dataset_name = 'tusz'
            ds.cache_dir = Path.home() / ".cache" / "pyhealth" / 'tusz'
            ds.final_tables = ['train', 'eval']
            ds.prepare_metadata()

            train_csv = root / "tusz-train-pyhealth.csv"
            eval_csv = root / "tusz-eval-pyhealth.csv"

            self.assertTrue(train_csv.exists())
            self.assertTrue(eval_csv.exists())

            train_df = pd.read_csv(train_csv)
            eval_df = pd.read_csv(eval_csv)

            print(train_df)
            print(eval_df)


if __name__ == "__main__":
    unittest.main()
