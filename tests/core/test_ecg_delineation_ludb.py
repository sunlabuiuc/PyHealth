"""Tiny offline native task checks, including ignored points and cache APIs.

Contributors: Thi Van Trinh Tran and Zexi Li.
"""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import polars as pl
import torch

from pyhealth.data import Patient
from pyhealth.datasets import LUDBDataset, create_sample_dataset, get_dataloader
from pyhealth.tasks.ecg_delineation_ludb import ECGDelineationLUDB
from tests.core.test_ludb import _artifacts


def _patient(args, pid=2):
    rows = pd.read_csv(args["manifest_path"])
    rows = rows[rows["patient_id"] == pid].to_dict("records")
    events = [{"patient_id": str(pid), "event_type": "windows", "timestamp": None,
               "_source_row": index,
               **{f"windows/{key}": value for key, value in row.items()
                  if key != "patient_id"}}
              for index, row in enumerate(rows)]
    frame = pl.DataFrame(events).with_columns(pl.col("timestamp").cast(pl.Datetime))
    return Patient(str(pid), frame)


def _record(*args):
    return SimpleNamespace(
        p_signal=np.tile(np.arange(5000, dtype=float)[:, None], (1, 12)), fs=500,
        sig_name=["i", "ii", "iii", "avr", "avl", "avf",
                  "v1", "v2", "v3", "v4", "v5", "v6"],
    )


class TestECGDelineationLUDB(unittest.TestCase):
    def test_frozen_arrays_and_native_label_packing(self):
        with tempfile.TemporaryDirectory() as directory:
            args = _artifacts(Path(directory))
            task = LUDBDataset(**args).default_task
            with patch.dict("sys.modules", {"wfdb": SimpleNamespace(rdrecord=_record)}):
                samples = task(_patient(args))
            self.assertEqual(len(samples), 2)  # all-ignored sample retained
            np.testing.assert_array_equal(samples[0]["signal"], np.arange(512))
            self.assertEqual(samples[0]["ignore_mask"].sum(), 10)
            self.assertEqual(len(samples[0]["label"]["time"]), 502)
            self.assertNotIn(20, samples[0]["label"]["time"])
            self.assertEqual(samples[1]["label"], {"time": [], "values": []})
            self.assertEqual(samples[1]["dense_label"].shape, (512,))
            self.assertIn('"qc":"keep"', samples[0]["patient_context"])
            dataset = create_sample_dataset(
                samples, task.input_schema, task.output_schema, in_memory=True,
            )
            batch = next(iter(get_dataloader(dataset, batch_size=2)))
            values, time, offsets = batch["label"]
            self.assertEqual(offsets.tolist(), [0, 502, 502])
            self.assertEqual(values.dtype, torch.long)
            self.assertEqual(time.shape, (502,))
            self.assertEqual(batch["signal"].dtype, torch.float32)
            self.assertEqual(batch["ignore_mask"].dtype, torch.bool)
            self.assertNotIn("dense_label", dataset.input_schema)

    def test_native_dataset_set_task_end_to_end(self):
        with tempfile.TemporaryDirectory() as directory:
            args = _artifacts(Path(directory))
            raw = LUDBDataset(**args)
            with patch.dict("sys.modules", {"wfdb": SimpleNamespace(rdrecord=_record)}):
                train, test = raw.set_task(split=raw.manifest_split, num_workers=1)
            self.assertEqual(len(train), 2)
            self.assertEqual(len(test), 2)
            self.assertEqual({int(row["patient_id"]) for row in train}, {2})
            self.assertEqual({int(row["patient_id"]) for row in test}, {10})
            self.assertEqual(train.output_processors["label"].output_size(), 4)
            self.assertEqual(sum(int(row["ignore_mask"].sum()) for row in train), 522)

    def test_bad_lead_order_and_bounds_are_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            args = _artifacts(Path(directory))
            task = LUDBDataset(**args).default_task
            record = _record()
            record.sig_name = list(reversed(record.sig_name))
            with patch.dict("sys.modules", {"wfdb": SimpleNamespace(
                rdrecord=lambda _: record,
            )}):
                with self.assertRaisesRegex(ValueError, "lead orders"):
                    task(_patient(args))
            frame = pd.read_csv(args["manifest_path"])
            frame.loc[frame.patient_id == 2, "window_length"] = 511
            frame.to_csv(args["manifest_path"], index=False)
            with patch.dict("sys.modules", {"wfdb": SimpleNamespace(rdrecord=_record)}):
                with self.assertRaisesRegex(ValueError, "window bounds"):
                    task(_patient(args))

    def test_task_is_stateless_across_patient_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            args = _artifacts(Path(directory))
            task = ECGDelineationLUDB(
                args["records_root"], args["mask_root"], args["metadata_path"],
                patient_ids=[2, 10],
            )
            before = repr(vars(task))
            with patch.dict("sys.modules", {"wfdb": SimpleNamespace(rdrecord=_record)}):
                task(_patient(args, 2))
                task(_patient(args, 10))
            self.assertEqual(before, repr(vars(task)))
