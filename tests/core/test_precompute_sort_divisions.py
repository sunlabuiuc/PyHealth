"""Tests for the Python 3.11 dask divisions workaround in base_dataset.

The deadlock itself only shows up with a real dataset build (for example
``test_apn``) on Python 3.11, which the ``oldest-python`` CI job runs with a
per-test timeout. These tests check that the helper leaves the sort correct,
including when the frame contains a P2P merge, which only runs on distributed
workers (full MIMIC-IV hits this; the demo datasets are too small).
"""

import tempfile
import unittest

import dask
import dask.dataframe as dd
import pandas as pd
from distributed import Client, LocalCluster

from pyhealth.datasets.base_dataset import _precompute_sort_divisions


def _cluster() -> LocalCluster:
    return LocalCluster(
        n_workers=2, threads_per_worker=1, processes=False, dashboard_address=None
    )


def _sort_and_write(df: dd.DataFrame, out_dir: str) -> pd.DataFrame:
    df = df.sort_values("patient_id")
    _precompute_sort_divisions(df, num_workers=1)
    df.to_parquet(out_dir, write_index=False)
    return pd.read_parquet(out_dir)


class TestPrecomputeSortDivisions(unittest.TestCase):
    def test_sort_and_write_with_distributed_client(self):
        pdf = pd.DataFrame(
            {"patient_id": [str(i % 7) for i in range(50)], "v": range(50)}
        )
        # Unknown divisions, as for a frame read from CSV: the sort must compute them.
        df = dd.from_pandas(pdf, npartitions=3).clear_divisions()
        with tempfile.TemporaryDirectory() as out_dir, _cluster() as cluster:
            with Client(cluster):
                out = _sort_and_write(df, out_dir)
        self.assertEqual(len(out), 50)
        self.assertEqual(list(out["patient_id"]), sorted(out["patient_id"]))

    def test_sort_after_p2p_merge(self):
        events = pd.DataFrame({"hadm_id": [i % 40 for i in range(400)], "v": range(400)})
        admissions = pd.DataFrame(
            {"hadm_id": range(40), "patient_id": [f"p{i % 13:02d}" for i in range(40)]}
        )
        left = dd.from_pandas(events, npartitions=4).clear_divisions()
        right = dd.from_pandas(admissions, npartitions=3).clear_divisions()
        with tempfile.TemporaryDirectory() as out_dir, _cluster() as cluster:
            with Client(cluster), dask.config.set({"dataframe.shuffle.method": "p2p"}):
                df = left.merge(right, on="hadm_id", how="inner", broadcast=False)
                out = _sort_and_write(df, out_dir)
        self.assertEqual(len(out), 400)
        self.assertEqual(list(out["patient_id"]), sorted(out["patient_id"]))


if __name__ == "__main__":
    unittest.main()
