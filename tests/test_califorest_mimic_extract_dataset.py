# Authors: Cesar Jesus Giglio Badoino (cesarjg2@illinois.edu)
#          Arjun Tangella (avtange2@illinois.edu)
#          Tony Nguyen (tonyln2@illinois.edu)
# Paper: CaliForest: Calibrated Random Forest for Health Data
# Paper link: https://doi.org/10.1145/3368555.3384461
# Description: Tests for CaliForest MIMIC-Extract dataset
"""Tests for CaliForestMIMICExtractDataset."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from pyhealth.datasets.califorest_mimic_extract import (
    CaliForestMIMICExtractDataset,
    _normalize_columns,
    _simple_imputer,
)


def _make_fake_hdf5(path):
    """Create a minimal fake all_hourly_data.h5."""
    patients = pd.DataFrame(
        {
            "mort_hosp": [0, 1, 0],
            "mort_icu": [0, 1, 0],
            "los_icu": [2.0, 8.0, 4.0],
            "max_hours": [48, 72, 50],
            "age": [65, 70, 55],
            "gender": ["M", "F", "M"],
            "ethnicity": ["W", "W", "B"],
        },
        index=pd.MultiIndex.from_tuples(
            [(1, 100, 1000), (2, 200, 2000), (3, 300, 3000)],
            names=["subject_id", "hadm_id", "icustay_id"],
        ),
    )
    rows = []
    for sid, hid, iid in [(1, 100, 1000), (2, 200, 2000), (3, 300, 3000)]:
        for h in range(3):
            rows.append((sid, hid, iid, h))
    idx = pd.MultiIndex.from_tuples(
        rows, names=["subject_id", "hadm_id", "icustay_id", "hours_in"],
    )
    np.random.seed(0)
    n = len(rows)
    data = {
        ("heart_rate", "mean"): np.random.randn(n),
        ("heart_rate", "count"): np.random.randint(0, 3, n).astype(float),
        ("heart_rate", "std"): np.abs(np.random.randn(n)),
        ("sodium", "mean"): np.random.randn(n),
        ("sodium", "count"): np.random.randint(0, 3, n).astype(float),
        ("sodium", "std"): np.abs(np.random.randn(n)),
    }
    vitals = pd.DataFrame(data, index=idx)
    vitals.columns = pd.MultiIndex.from_tuples(
        vitals.columns, names=["LEVEL2", "Aggregation Function"]
    )
    patients.to_hdf(path, key="patients", mode="w")
    vitals.to_hdf(path, key="vitals_labs", mode="a")


class TestDatasetInstantiation:
    def test_creates_with_valid_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_fake_hdf5(os.path.join(tmpdir, "all_hourly_data.h5"))
            ds = CaliForestMIMICExtractDataset(root=tmpdir)
            assert ds.dataset_name == "califorest_mimic_extract"

    def test_raises_on_missing_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                CaliForestMIMICExtractDataset(root=tmpdir)


class TestNormalizeColumns:
    def test_swaps_when_agg_is_level1(self):
        cols = pd.MultiIndex.from_tuples(
            [("hr", "mean"), ("hr", "count")],
            names=["LEVEL2", "Aggregation Function"],
        )
        df = pd.DataFrame(np.ones((2, 2)), columns=cols)
        result = _normalize_columns(df)
        assert result.columns.get_level_values(0)[0] in {"mean", "count"}

    def test_noop_when_agg_is_level0(self):
        cols = pd.MultiIndex.from_tuples(
            [("mean", "hr"), ("count", "hr")],
            names=["Aggregation Function", "LEVEL2"],
        )
        df = pd.DataFrame(np.ones((2, 2)), columns=cols)
        result = _normalize_columns(df)
        assert result.columns.get_level_values(0)[0] == "mean"


class TestSimpleImputer:
    def test_output_has_three_channels(self):
        cols = pd.MultiIndex.from_tuples(
            [("mean", "hr"), ("count", "hr"), ("std", "hr")],
            names=["Aggregation Function", "LEVEL2"],
        )
        idx = pd.MultiIndex.from_tuples(
            [(1, 1, 1, 0), (1, 1, 1, 1)],
            names=["subject_id", "hadm_id", "icustay_id", "hours_in"],
        )
        df = pd.DataFrame(
            [[1.0, 1.0, 0.1], [np.nan, 0.0, 0.0]],
            index=idx, columns=cols,
        )
        result = _simple_imputer(df)
        agg_funcs = set(result.columns.get_level_values(0).unique())
        assert "mean" in agg_funcs
        assert "mask" in agg_funcs
        assert "time_since_measured" in agg_funcs


class TestLoadCaliforestData:
    def test_returns_aligned_arrays(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_fake_hdf5(os.path.join(tmpdir, "all_hourly_data.h5"))
            ds = CaliForestMIMICExtractDataset(root=tmpdir)
            X_tr, X_te, y_tr, y_te = ds.load_califorest_data(
                target="mort_hosp", train_frac=0.67,
            )
            assert X_tr.ndim == 2
            assert X_te.ndim == 2
            assert len(y_tr) == X_tr.shape[0]
            assert len(y_te) == X_te.shape[0]
            assert set(np.unique(np.concatenate([y_tr, y_te]))).issubset({0, 1})
