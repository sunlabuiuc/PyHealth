# Authors: Aritra Saha and Ugochukwu Njoku.
# Survival Mixture Density Networks, Han et al., MLHC 2022.
# https://proceedings.mlr.press/v182/han22a.html
"""Small infrastructure helpers for the survival integration tests."""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

from distributed import LocalCluster


@contextmanager
def threaded_csv_loading() -> Iterator[None]:
    """Keep real Dask CSV processing without starting child Python processes.

    These tests check data/task behavior, not Dask's process isolation. Only
    the worker transport changes; CSV parsing, task execution, processors and
    on-disk caches still run normally. Other dataset tests retain the default
    process-based loader.
    """

    def cluster(**kwargs: Any) -> LocalCluster:
        return LocalCluster(**{**kwargs, "processes": False})

    with patch("pyhealth.datasets.base_dataset.DaskCluster", cluster):
        yield
