"""Smoke test for the Python 3.11 dask divisions workaround in base_dataset.

The deadlock itself only shows up with a real dataset build (for example
``test_apn``) on Python 3.11, which the ``oldest-python`` CI job runs with a
per-test timeout. This test checks that the helper leaves the sort correct.
"""

import tempfile
import unittest

import dask.dataframe as dd
import pandas as pd
from distributed import Client, LocalCluster

from pyhealth.datasets.base_dataset import _compute_divisions_locally


def _sort_and_write(out_dir: str) -> pd.DataFrame:
    pdf = pd.DataFrame({"patient_id": [str(i % 7) for i in range(50)], "v": range(50)})
    # Unknown divisions, as for a frame read from CSV: the sort must compute them.
    df = dd.from_pandas(pdf, npartitions=3).clear_divisions()
    with (
        LocalCluster(
            n_workers=1, threads_per_worker=1, processes=False, dashboard_address=None
        ) as cluster,
        Client(cluster),
    ):
        df = df.sort_values("patient_id")
        _compute_divisions_locally(df, num_workers=1)
        df.to_parquet(out_dir, write_index=False)
    return pd.read_parquet(out_dir)


class TestComputeDivisionsLocally(unittest.TestCase):
    def test_sort_and_write_with_distributed_client(self):
        with tempfile.TemporaryDirectory() as out_dir:
            out = _sort_and_write(out_dir)
        self.assertEqual(len(out), 50)
        self.assertEqual(list(out["patient_id"]), sorted(out["patient_id"]))


if __name__ == "__main__":
    unittest.main()
