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

            # Minimal filesystem layout
            train_dir = root / "train"
            eval_dir = root / "eval"

            self._touch(train_dir / "aaaaaaac" / "s005_2002" / "01_tcp_ar" / "aaaaaaac_s005_2002.edf")
            self._touch(train_dir / "aaaaaayf" / "s003_2003" / "02_tcp_le" / "aaaaaayf_s003_2003.edf")
            self._touch(eval_dir / "aaaaaabg" / "s002_2006" / "01_tcp_ar" / "aaaaaabg_s002_2006.edf")
            self._touch(eval_dir / "aaaaaarq" / "s016_2014" / "03_tcp_ar_a" / "aaaaaarq_s016_2014.edf")

            # Call prepare_metadata without invoking BaseDataset init
            ds = TUSZDataset.__new__(TUSZDataset)
            ds.root = str(root)
            ds.root_path = Path(root)
            ds.dataset_name = 'tusz'
            ds.cache_dir = Path.home() / ".cache" / "pyhealth" / "tusz"
            ds.use_cache = False
            ds.final_tables = ['train', 'eval']
            ds.prepare_metadata()

            train_csv = root / "tusz-train-pyhealth.csv"
            eval_csv = root / "tusz-eval-pyhealth.csv"

            self.assertTrue(train_csv.exists())
            self.assertTrue(eval_csv.exists())

            train_df = pd.read_csv(train_csv)
            eval_df = pd.read_csv(eval_csv)

            self.assertEqual(len(train_df), 2)
            self.assertEqual(len(eval_df), 2)

            self.assertIn("patient_id", train_df.columns)
            self.assertIn("record_id", train_df.columns)
            self.assertIn("signal_file", train_df.columns)

            self.assertEqual(train_df.loc[0, "patient_id"], "aaaaaaac")
            self.assertEqual(train_df.loc[0, "record_id"], "s005_2002")
            self.assertTrue(str(train_df.loc[0, "signal_file"]).endswith("aaaaaaac_s005_2002.edf"))
            self.assertEqual(train_df.loc[1, "patient_id"], "aaaaaayf")
            self.assertEqual(train_df.loc[1, "record_id"], "s003_2003")
            self.assertTrue(str(train_df.loc[1, "signal_file"]).endswith("aaaaaayf_s003_2003.edf"))

            self.assertIn("patient_id", eval_df.columns)
            self.assertIn("record_id", eval_df.columns)
            self.assertIn("signal_file", eval_df.columns)

            self.assertEqual(eval_df.loc[0, "patient_id"], "aaaaaabg")
            self.assertEqual(eval_df.loc[0, "record_id"], "s002_2006")
            self.assertTrue(str(eval_df.loc[0, "signal_file"]).endswith("aaaaaabg_s002_2006.edf"))
            self.assertEqual(eval_df.loc[1, "patient_id"], "aaaaaarq")
            self.assertEqual(eval_df.loc[1, "record_id"], "s016_2014")
            self.assertTrue(str(eval_df.loc[1, "signal_file"]).endswith("aaaaaarq_s016_2014.edf"))

            # Idempotency: should not crash when CSVs already exist
            ds.prepare_metadata()

            train_df = pd.read_csv(train_csv)
            eval_df = pd.read_csv(eval_csv)

            self.assertEqual(len(train_df), 2)
            self.assertEqual(len(eval_df), 2)

            # test subsets after finish to avoid interfering cache dir
            self._test_subsets()
            
    def _test_subsets(self):
        with tempfile.TemporaryDirectory() as tmp:
            for data_type in ["train", "eval", "dev"]:
                ds = TUSZDataset(root=tmp, subset=data_type, use_cache=False)
                self.assertEqual(ds.final_tables, [data_type])

            ds = TUSZDataset(root=tmp, subset="train,eval", use_cache=False)
            self.assertEqual(ds.final_tables, ["train", "eval"])

            ds = TUSZDataset(root=tmp, subset="all", use_cache=False)
            self.assertEqual(ds.final_tables, ["train", "eval", "dev"])

    def test_invalid_subset_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                TUSZDataset(root=tmp, subset="nope")

    def test_default_task_returns_task_instance(self):
        ds = TUSZDataset.__new__(TUSZDataset)
        task = ds.default_task
        self.assertIsInstance(task, TUSZTask)


if __name__ == "__main__":
    unittest.main()
