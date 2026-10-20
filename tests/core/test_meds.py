"""Tests for MEDSDataset (synthetic Parquet fixtures, fixed seeds).

Optional smoke on a real export: set ``MEDS_DEMO_ROOT`` to the dataset version
directory (the folder containing ``data/`` and ``metadata/``).
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from unittest.mock import patch

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from pyhealth.datasets import MEDSDataset, MIMIC3Dataset
from pyhealth.tasks.base_task import BaseTask

T1 = datetime(2024, 1, 1, 8, 0, 0)
T2 = datetime(2024, 1, 2, 9, 30, 0)

MEDS_DEMO_ROOT = os.environ.get("MEDS_DEMO_ROOT")
_DEFAULT_DEMO = (
    Path(__file__).parent.parent.parent / "test-resources" / "meds_demo"
)
MEDS_DEMO_PATH = (
    Path(MEDS_DEMO_ROOT).expanduser()
    if MEDS_DEMO_ROOT
    else _DEFAULT_DEMO
)

# Fixed split assignment for deterministic assertions.
_SYNTHETIC_SPLITS: Dict[str, List[int]] = {
    "train": [1001, 1002, 1003, 1004],
    "tuning": [1005, 1006],
    "held_out": [1007, 1008],
}


def write_synthetic_meds(root: Path, *, seed: int = 42, rows_per_shard: int = 25) -> None:
    """Write a MEDS-shaped tree under ``root`` (``data/<split>/*.parquet`` + metadata)."""
    rng = np.random.default_rng(seed)
    data_root = root / "data"
    for split, subjects in _SYNTHETIC_SPLITS.items():
        split_dir = data_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for shard in range(2):
            n = rows_per_shard
            pq.write_table(
                pa.table(
                    {
                        "subject_id": pa.array(rng.choice(subjects, n), type=pa.int64()),
                        "time": pa.array(
                            pd.date_range("2020-01-01", periods=n, freq="h"),
                            type=pa.timestamp("us"),
                        ),
                        "code": pa.array(
                            [f"LAB//{i % 5}" for i in range(n)], type=pa.string()
                        ),
                        "numeric_value": pa.array(
                            rng.normal(size=n).astype(np.float32), type=pa.float32()
                        ),
                    }
                ),
                split_dir / f"{shard}.parquet",
            )

    meta = root / "metadata"
    meta.mkdir(exist_ok=True)
    all_subjects = [sid for ids in _SYNTHETIC_SPLITS.values() for sid in ids]
    all_splits = [
        split for split, ids in _SYNTHETIC_SPLITS.items() for _ in ids
    ]
    pq.write_table(
        pa.table(
            {
                "subject_id": pa.array(all_subjects, type=pa.int64()),
                "split": pa.array(all_splits, type=pa.string()),
            }
        ),
        meta / "subject_splits.parquet",
    )


class _MedsSmokeTask(BaseTask):
    """Minimal task: one sample per patient with at least one meds event."""

    task_name: str = "MedsSmokeTask"
    input_schema: Dict[str, str] = {"codes": "sequence"}
    output_schema: Dict[str, str] = {"has_events": "binary"}

    def __call__(self, patient):
        meds = patient.get_events(event_type="meds")
        if not meds:
            return []
        codes = [event.code for event in meds if event.code]
        if not codes:
            return []
        return [
            {
                "patient_id": patient.patient_id,
                "codes": codes,
                "has_events": int(patient.patient_id) % 2,
            }
        ]


class TestMEDSDatasetSynthetic(unittest.TestCase):
    """MEDSDataset against a local synthetic MEDS export."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.mkdtemp(prefix="meds_synthetic_")
        cls.root = Path(cls._tmp)
        write_synthetic_meds(cls.root)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls._tmp, ignore_errors=True)

    def _dataset(self, **kwargs) -> MEDSDataset:
        return MEDSDataset(
            root=str(self.root),
            cache_dir=self._tmp,
            num_workers=1,
            **kwargs,
        )

    def test_load_table_schema_and_dtypes(self) -> None:
        ds = self._dataset()
        df = ds.load_table("meds").compute()
        self.assertIn("patient_id", df.columns)
        self.assertIn("timestamp", df.columns)
        self.assertIn("event_type", df.columns)
        self.assertIn("meds/code", df.columns)
        self.assertIn("meds/numeric_value", df.columns)
        self.assertEqual(str(df["patient_id"].dtype), "string")
        self.assertEqual(str(df["timestamp"].dtype), "datetime64[ms]")
        self.assertTrue((df["event_type"] == "meds").all())

    def test_loads_all_patients(self) -> None:
        ds = self._dataset()
        expected = {str(sid) for sid in _SYNTHETIC_SPLITS["train"]}
        expected |= {str(sid) for sid in _SYNTHETIC_SPLITS["tuning"]}
        expected |= {str(sid) for sid in _SYNTHETIC_SPLITS["held_out"]}
        self.assertEqual(set(ds.unique_patient_ids), expected)

    def test_subset_train_via_metadata(self) -> None:
        ds = self._dataset(subset="train", split_source="metadata")
        expected = {str(sid) for sid in _SYNTHETIC_SPLITS["train"]}
        self.assertEqual(set(ds.unique_patient_ids), expected)

    def test_subset_tuning_via_directory(self) -> None:
        ds = self._dataset(subset="tuning", split_source="directory")
        expected = {str(sid) for sid in _SYNTHETIC_SPLITS["tuning"]}
        self.assertEqual(set(ds.unique_patient_ids), expected)

    def test_subject_splits_exposed_as_events(self) -> None:
        ds = self._dataset(tables=["meds", "subject_splits"])
        patient_id = str(_SYNTHETIC_SPLITS["train"][0])
        patient = ds.get_patient(patient_id)
        split_events = patient.get_events(event_type="subject_splits")
        self.assertEqual(len(split_events), 1)
        self.assertEqual(split_events[0].split, "train")

    def test_patient_meds_event_attributes(self) -> None:
        ds = self._dataset(subset="train")
        patient = ds.get_patient(str(_SYNTHETIC_SPLITS["train"][0]))
        meds = patient.get_events(event_type="meds")
        self.assertGreater(len(meds), 0)
        self.assertTrue(str(meds[0].code).startswith("LAB//"))

    def test_invalid_subset_raises(self) -> None:
        with self.assertRaises(ValueError):
            self._dataset(subset="validation")

    def test_schema_violations_raise_type_error_at_construction(self):
        """The footer guard rejects non-conforming `time` before any Dask.

        Covers the ADR 002 T5 hazard (date-like ints such as 20240101 would
        otherwise parse silently) plus strings, timezone-aware timestamps,
        and a missing column.
        """
        cases = {
            "int64": (
                "time",
                pl.Series([20240101, 20240102], dtype=pl.Int64),
            ),
            "string": (
                "time",
                pl.Series(["2024-01-01", "2024-01-02"], dtype=pl.String),
            ),
            "tz_aware": (
                "time",
                pl.Series([T1, T2], dtype=pl.Datetime("us", "UTC")),
            ),
            "missing": ("ts", pl.Series([T1, T2], dtype=pl.Datetime("us"))),
        }
        for label, (col_name, series) in cases.items():
            with self.subTest(time=label):
                bad_root = Path(self._tmp) / f"meds_bad_{label}"
                (bad_root / "data").mkdir(parents=True)
                pl.DataFrame(
                    {
                        "subject_id": pl.Series([1, 2], dtype=pl.Int64),
                        col_name: series,
                        "code": pl.Series(["A", "B"], dtype=pl.String),
                        "numeric_value": pl.Series(
                            [None, None], dtype=pl.Float32
                        ),
                    }
                ).write_parquet(bad_root / "data" / "0.parquet")
                with self.assertRaises(TypeError):
                    MEDSDataset(root=str(bad_root), cache_dir=self._tmp)

    def test_cache_dir_varies_with_subset(self) -> None:
        with patch(
            "pyhealth.datasets.base_dataset.platformdirs.user_cache_dir",
            return_value=self._tmp,
        ):
            all_ds = MEDSDataset(
                root=str(self.root),
                cache_dir=self._tmp,
                num_workers=1,
            )
            train_ds = MEDSDataset(
                root=str(self.root),
                cache_dir=self._tmp,
                subset="train",
                split_source="metadata",
                num_workers=1,
            )
            self.assertNotEqual(all_ds.cache_dir, train_ds.cache_dir)

    def test_set_task_smoke(self) -> None:
        ds = self._dataset(subset="train")
        sample_ds = ds.set_task(_MedsSmokeTask(), num_workers=1)
        self.assertGreater(len(sample_ds), 0)
        sample = sample_ds[0]
        self.assertIn("codes", sample)
        self.assertEqual(sample["has_events"], int(sample["patient_id"]) % 2)

    def test_mimic3_csv_path_unchanged(self) -> None:
        """Non-regression: CSV-backed datasets still load after MEDSDataset addition."""
        demo = (
            Path(__file__).parent.parent.parent
            / "test-resources"
            / "core"
            / "mimic3demo"
        )
        ds = MIMIC3Dataset(
            root=str(demo),
            tables=["diagnoses_icd"],
            cache_dir=self._tmp,
            num_workers=1,
        )
        self.assertGreater(len(ds.unique_patient_ids), 0)


@unittest.skipUnless(
    MEDS_DEMO_PATH.is_dir()
    and (MEDS_DEMO_PATH / "data").is_dir()
    and (MEDS_DEMO_PATH / "metadata" / "subject_splits.parquet").is_file(),
    "Download mimic-iv-demo-meds into test-resources/meds_demo or set MEDS_DEMO_ROOT",
)
class TestMEDSDatasetDemoSmoke(unittest.TestCase):
    """Smoke on mimic-iv-demo-meds (partial export is enough for dtype checks)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.root = MEDS_DEMO_PATH.resolve()
        cls.cache = tempfile.mkdtemp(prefix="meds_demo_")

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.cache, ignore_errors=True)

    def test_demo_load_table_dtypes(self) -> None:
        ds = MEDSDataset(
            root=str(self.root),
            cache_dir=self.cache,
            num_workers=1,
        )
        df = ds.load_table("meds").compute()
        self.assertEqual(str(df["patient_id"].dtype), "string")
        self.assertEqual(str(df["timestamp"].dtype), "datetime64[ms]")
        self.assertGreater(len(df), 0)

    def test_demo_stats_and_subset(self) -> None:
        ds = MEDSDataset(
            root=str(self.root),
            cache_dir=self.cache,
            subset="train",
            num_workers=1,
        )
        ds.stats()
        self.assertGreater(len(ds.unique_patient_ids), 0)


if __name__ == "__main__":
    unittest.main()
