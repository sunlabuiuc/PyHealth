"""
Contributors: Manling Tan, Yining Jin, Xiao Luo
NetIDs: manling4, yining34, xiaoluo6

Paper Title:
Temporal Pointwise Convolutional Networks for
Length of Stay Prediction in the Intensive Care Unit

Paper Authors:
Emma Rocheteau, Pietro Liò, and Stephanie Hyland (2021)

Paper URL:
https://arxiv.org/abs/2007.09483
"""


"""Opt-in real CSV loader/task smoke check; fast task tests live in core.

Set PYHEALTH_RUN_TPC_INTEGRATION=1 to run this slower local-server check.
"""

import os

import pytest

from examples.tpc_support.checkpoint import run


@pytest.mark.skipif(
    os.environ.get("PYHEALTH_RUN_TPC_INTEGRATION") != "1",
    reason="Opt-in CSV/Dask/task check; set PYHEALTH_RUN_TPC_INTEGRATION=1",
)
def test_csv_to_task(tmp_path):
    result = run(tmp_path)
    assert result["status"] == "passed"
    assert result["scope"] == "CSV loading, cohort, static features, hourly labels"
