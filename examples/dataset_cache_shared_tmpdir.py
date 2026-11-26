"""Jobs that share a cache_dir do not delete each other's temporary files.

While it builds, a dataset keeps temporary files (converted source tables, Dask
scratch space) in folders under ``tmp/`` in its cache directory, and removes
them when the build ends. Two jobs started with the same configuration, such as
two runs of a hyperparameter sweep, share that cache directory. Each job removes
only the folders it created, so the job that finishes first no longer deletes
files the other job is still reading.

Runs in a few seconds on CPU; nothing is built or downloaded.

Usage:
    python examples/dataset_cache_shared_tmpdir.py
"""

import tempfile

from pyhealth.datasets import BaseDataset


def open_dataset(cache: str) -> BaseDataset:
    return BaseDataset(
        root="", tables=[], dataset_name="shared_tmpdir_demo", cache_dir=cache
    )


def main():
    with tempfile.TemporaryDirectory() as cache:
        job_a = open_dataset(cache)
        job_b = open_dataset(cache)
        print("same cache directory:", job_a.cache_dir == job_b.cache_dir)

        # Job A is mid-build and has converted a source table into tmp/.
        table = job_a.create_tmpdir() / "table.parquet"
        table.write_text("job A's converted table")

        # Job B finishes its own build, which ends with clean_tmpdir().
        job_b.create_tmpdir()
        job_b.clean_tmpdir()
        kept = table.exists()
        print("job A's file after job B's cleanup:", "kept" if kept else "deleted")

        job_a.clean_tmpdir()
        tmp_root = job_a.cache_dir / "tmp"
        print("tmp/ removed once both jobs are done:", not tmp_root.exists())


if __name__ == "__main__":
    main()
