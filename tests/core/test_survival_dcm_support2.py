# Authors: Murphy Crosby and Chidozie Opara.
# Deep Cox Mixtures for Survival Regression, Nagpal et al., MLHC 2021.
# https://proceedings.mlr.press/v149/nagpal21a.html
"""SUPPORT2 time-to-event task in the Deep Cox Mixtures representation."""

import csv
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import polars as pl
import torch
import yaml
from distributed import LocalCluster

import pyhealth.datasets
from pyhealth.data import Patient
from pyhealth.datasets import Support2Dataset
from pyhealth.tasks import SurvivalDCMSupport2
from pyhealth.tasks.survival_dcm_support2 import (
    DCM_SUPPORT2_CATEGORIES,
    DCM_SUPPORT2_FEATURES,
    DCM_SUPPORT2_NUMERIC,
)

N_NUMERIC = len(DCM_SUPPORT2_NUMERIC)
N_FEATURES = len(DCM_SUPPORT2_FEATURES)


def record(**overrides):
    """One synthetic SUPPORT2 record, as the strings the dataset provides."""
    values = {name: str(10.0 + i) for i, name in enumerate(DCM_SUPPORT2_NUMERIC)}
    values.update(
        {
            "sex": "female",
            "dzgroup": "CHF",
            "dzclass": "COPD/CHF/Cirrhosis",
            "income": "under $11k",
            "race": "white",
            "ca": "no",
            "d.time": "120",
            "death": "1",
        }
    )
    values.update(overrides)
    return values


def patient(*records, patient_id="1"):
    """A Patient holding one ``support2`` event per record."""
    events = [
        {
            "patient_id": patient_id,
            "event_type": "support2",
            "timestamp": None,
            **{f"support2/{name}": value for name, value in r.items()},
        }
        for r in records
    ]

    schema = {
        "patient_id": pl.String,
        "event_type": pl.String,
        "timestamp": pl.Datetime,
    }
    frame = pl.DataFrame(events) if events else pl.DataFrame(schema=schema)
    return Patient(patient_id, frame)


def one_hot(sample, name):
    """The one-hot columns of one categorical covariate."""
    levels = dict(DCM_SUPPORT2_CATEGORIES)[name]
    start = DCM_SUPPORT2_FEATURES.index(f"{name}={levels[0]}")
    return sample["features"][start : start + len(levels)]


class TestSurvivalDCMSupport2(unittest.TestCase):
    def test_feature_names(self):
        self.assertEqual(N_NUMERIC, 18)
        self.assertEqual(N_FEATURES, 44)
        self.assertEqual(DCM_SUPPORT2_FEATURES[:N_NUMERIC], DCM_SUPPORT2_NUMERIC)
        self.assertEqual(DCM_SUPPORT2_FEATURES[N_NUMERIC], "sex=female")
        self.assertEqual(len(set(DCM_SUPPORT2_FEATURES)), N_FEATURES)

    def test_schema(self):
        task = SurvivalDCMSupport2()
        self.assertEqual(task.task_name, "SurvivalDCMSupport2")
        self.assertEqual(task.input_schema, {"features": "tensor"})
        self.assertEqual(task.output_schema, {"duration": "tensor", "event": "tensor"})

    def test_raw_sample(self):
        samples = SurvivalDCMSupport2()(patient(record(), patient_id="7"))
        self.assertEqual(len(samples), 1)
        sample = samples[0]
        self.assertEqual(sample["patient_id"], "7")
        self.assertEqual(len(sample["features"]), N_FEATURES)
        self.assertEqual(
            sample["features"][:N_NUMERIC], [10.0 + i for i in range(N_NUMERIC)]
        )
        self.assertEqual(sample["duration"], [120.0])
        self.assertEqual(sample["event"], [1.0])

    def test_one_hot_encoding(self):
        sample = SurvivalDCMSupport2()(patient(record()))[0]
        self.assertEqual(one_hot(sample, "sex"), [1.0, 0.0])
        self.assertEqual(one_hot(sample, "race"), [0.0, 0.0, 0.0, 0.0, 1.0])
        self.assertEqual(one_hot(sample, "ca"), [0.0, 1.0, 0.0])
        for name, levels in DCM_SUPPORT2_CATEGORIES:
            self.assertEqual(sum(one_hot(sample, name)), 1.0, name)
            self.assertEqual(len(one_hot(sample, name)), len(levels), name)

    def test_missing_numeric_is_nan_in_raw_mode(self):
        for missing in ("NaN", "", None):
            sample = SurvivalDCMSupport2()(patient(record(alb=missing)))[0]
            index = DCM_SUPPORT2_FEATURES.index("alb")
            self.assertTrue(math.isnan(sample["features"][index]), repr(missing))
            others = [v for i, v in enumerate(sample["features"]) if i != index]
            self.assertFalse(any(math.isnan(v) for v in others))

    def test_missing_categorical_is_all_zeros(self):
        for missing in ("NaN", "", None):
            sample = SurvivalDCMSupport2()(
                patient(record(race=missing, income=missing))
            )[0]
            self.assertEqual(one_hot(sample, "race"), [0.0] * 5, repr(missing))
            self.assertEqual(one_hot(sample, "income"), [0.0] * 4, repr(missing))
            self.assertEqual(one_hot(sample, "sex"), [1.0, 0.0])

    def test_standardized_sample(self):
        raw = SurvivalDCMSupport2()(patient(record()))[0]["features"]
        mean = [float(i) for i in range(N_FEATURES)]
        scale = [1.0 + 0.5 * i for i in range(N_FEATURES)]
        task = SurvivalDCMSupport2(feature_mean=mean, feature_scale=scale)
        sample = task(patient(record()))[0]
        expected = (np.asarray(raw) - np.asarray(mean)) / np.asarray(scale)
        np.testing.assert_allclose(sample["features"], expected)
        self.assertEqual(sample["duration"], [120.0])

    def test_missing_numeric_is_zero_when_standardized(self):
        task = SurvivalDCMSupport2(
            feature_mean=[5.0] * N_FEATURES, feature_scale=[2.0] * N_FEATURES
        )
        sample = task(patient(record(alb="NaN", glucose=None)))[0]
        self.assertEqual(sample["features"][DCM_SUPPORT2_FEATURES.index("alb")], 0.0)
        self.assertEqual(
            sample["features"][DCM_SUPPORT2_FEATURES.index("glucose")], 0.0
        )
        self.assertFalse(any(math.isnan(v) for v in sample["features"]))

    def test_censored_patient(self):
        sample = SurvivalDCMSupport2()(patient(record(death="0")))[0]
        self.assertEqual(sample["event"], [0.0])

    def test_invalid_records_raise(self):
        bad_records = {
            "unknown category": record(race="martian"),
            "non-numeric value": record(age="old"),
            "infinite value": record(age="inf"),
            "zero time": record(**{"d.time": "0"}),
            "missing time": record(**{"d.time": "NaN"}),
            "invalid event": record(death="2"),
            "missing event": record(death=None),
        }
        for label, bad in bad_records.items():
            with self.subTest(label), self.assertRaises(ValueError):
                SurvivalDCMSupport2()(patient(bad))

    def test_requires_exactly_one_record(self):
        for label, records in {"none": (), "two": (record(), record())}.items():
            with self.subTest(label), self.assertRaises(ValueError):
                SurvivalDCMSupport2()(patient(*records))

    def test_invalid_statistics_raise(self):
        ok = [1.0] * N_FEATURES
        bad_arguments = {
            "mean only": {"feature_mean": ok},
            "scale only": {"feature_scale": ok},
            "wrong length": {"feature_mean": ok[:-1], "feature_scale": ok},
            "zero scale": {"feature_mean": ok, "feature_scale": [0.0] * N_FEATURES},
            "nan mean": {"feature_mean": [math.nan] * N_FEATURES, "feature_scale": ok},
            "negative scale": {
                "feature_mean": ok,
                "feature_scale": [-1.0] * N_FEATURES,
            },
            "infinite scale": {
                "feature_mean": ok,
                "feature_scale": [math.inf] * N_FEATURES,
            },
        }
        for label, kwargs in bad_arguments.items():
            with self.subTest(label), self.assertRaises(ValueError):
                SurvivalDCMSupport2(**kwargs)


class TestSurvivalDCMSupport2WithDataset(unittest.TestCase):
    """One end-to-end check through ``Support2Dataset`` and ``set_task``.

    This loads a CSV, so it takes about half a second rather than milliseconds.
    It guards the layer the in-memory tests cannot see: how the dataset names,
    types and represents missing values before they reach the task.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        config = Path(pyhealth.datasets.__file__).parent / "configs" / "support2.yaml"
        columns = ["sno"] + yaml.safe_load(config.read_text())["tables"]["support2"][
            "attributes"
        ]
        rows = [
            {"sno": "1", **record()},
            {"sno": "2", **record(alb="NaN", race="NaN", death="0")},
            {"sno": "3", **record(glucose="", income="", **{"d.time": "30"})},
        ]
        with (self.root / "support2.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, restval="")
            writer.writeheader()
            writer.writerows(rows)

        # Run the dataset's Dask workers as threads, so loading the CSV starts
        # no child processes. Everything else in the loader runs unchanged.
        def threaded_cluster(**kwargs):
            kwargs["processes"] = False
            return LocalCluster(**kwargs)

        patcher = patch("pyhealth.datasets.base_dataset.DaskCluster", threaded_cluster)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_dataset_to_samples(self):
        dataset = Support2Dataset(
            root=str(self.root),
            tables=["support2"],
            cache_dir=self.root / "cache",
            num_workers=1,
        )
        samples = dataset.set_task(SurvivalDCMSupport2(), num_workers=1)
        self.assertEqual(len(samples), 3)
        by_id = {sample["patient_id"]: sample for sample in samples}
        self.assertEqual(set(by_id), {"1", "2", "3"})
        for sample in by_id.values():
            self.assertEqual(tuple(sample["features"].shape), (N_FEATURES,))
        alb = DCM_SUPPORT2_FEATURES.index("alb")
        glucose = DCM_SUPPORT2_FEATURES.index("glucose")
        self.assertFalse(torch.isnan(by_id["1"]["features"]).any())
        self.assertTrue(torch.isnan(by_id["2"]["features"][alb]))
        self.assertTrue(torch.isnan(by_id["3"]["features"][glucose]))
        self.assertEqual(one_hot(by_id["1"], "race").tolist(), [0.0] * 4 + [1.0])
        self.assertEqual(one_hot(by_id["2"], "race").tolist(), [0.0] * 5)
        self.assertEqual(one_hot(by_id["3"], "income").tolist(), [0.0] * 4)
        self.assertEqual(by_id["2"]["event"].item(), 0.0)
        self.assertEqual(by_id["3"]["duration"].item(), 30.0)


if __name__ == "__main__":
    unittest.main()
