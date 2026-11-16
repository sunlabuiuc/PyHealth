"""Unit tests for the NHANESMortalityPrediction task.

Patients are lightweight fakes whose epoch files are tiny synthetic parquet
files, so these tests do not build a full dataset. No real NHANES data is used.
"""

import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from pyhealth.datasets import create_sample_dataset
from pyhealth.tasks.nhanes_mortality_prediction import (
    CATEGORY_SIZES,
    NHANESMortalityPrediction,
)

SEQ_LEN = 12  # One hour of 5-minute epochs keeps the tests tiny.


class _Patient:
    """Minimal stand-in for a PyHealth patient with one ``nhanes`` event."""

    def __init__(self, patient_id: str, event: SimpleNamespace) -> None:
        self.patient_id = patient_id
        self._event = event

    def get_events(self, event_type: str) -> list:
        return [self._event] if event_type == "nhanes" else []


class _TaskTestCase(unittest.TestCase):
    """Writes epoch files into a temporary directory."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.count = 0

    def tearDown(self):
        self._tmp.cleanup()

    def patient(self, wear, mortstat=0, permth_int=200, intensity=None, **attrs):
        """Builds a fake patient whose epochs have the given wear pattern."""
        wear = np.asarray(wear, dtype=np.int8)
        n = wear.size
        index = np.arange(n)
        if intensity is None:
            intensity = np.where(wear == 1, 100.0 + index, 0.0)
        epochs = pd.DataFrame(
            {
                "epoch": index.astype(np.int32),
                "day": ((index // 288) % 7).astype(np.int8),
                "hour": ((index % 288) // 12).astype(np.int8),
                "minute": ((index % 12) * 5).astype(np.int8),
                "intensity": np.asarray(intensity, dtype=np.float32),
                "wear_minutes": (5 * wear).astype(np.int8),
                "wear_mask": wear,
            }
        )
        self.count += 1
        path = self.dir / f"{self.count}.parquet"
        epochs.to_parquet(path, index=False)
        fields = {
            "age": 50.0,
            "sex": 1,
            "income": 5,
            "bmi": 27.0,
            "waist": 95.0,
            "dbp": 70.0,
            "sbp": 120.0,
            "permth_exm": permth_int,
        }
        fields.update(attrs)
        event = SimpleNamespace(
            epoch_file=str(path), mortstat=mortstat, permth_int=permth_int, **fields
        )
        return _Patient(str(self.count), event)


class TestLabels(_TaskTestCase):
    """Death within the horizon, measured from the interview."""

    def label(self, task=None, **kwargs):
        task = task or NHANESMortalityPrediction(seq_len=SEQ_LEN)
        samples = task(self.patient(np.ones(SEQ_LEN), **kwargs))
        return samples[0]["mortality"] if samples else None

    def test_death_at_horizon_is_positive(self):
        """Death at month 60 is 1; at month 61 it is 0."""
        self.assertEqual(self.label(mortstat=1, permth_int=60), 1)
        self.assertEqual(self.label(mortstat=1, permth_int=61), 0)

    def test_survivors(self):
        """Survivors followed 60+ months are 0; shorter follow-up is skipped."""
        self.assertEqual(self.label(mortstat=0, permth_int=60), 0)
        self.assertIsNone(self.label(mortstat=0, permth_int=59))

    def test_missing_follow_up_is_skipped(self):
        """Missing status or months gives no samples."""
        self.assertIsNone(self.label(mortstat=None, permth_int=100))
        self.assertIsNone(self.label(mortstat=1, permth_int=None))
        self.assertIsNone(self.label(mortstat=math.nan, permth_int=100))

    def test_uses_months_from_interview(self):
        """PERMTH_INT decides the label, not PERMTH_EXM."""
        self.assertEqual(self.label(mortstat=1, permth_int=61, permth_exm=59), 0)
        self.assertEqual(self.label(mortstat=1, permth_int=59, permth_exm=61), 1)

    def test_custom_horizon(self):
        """followup_months changes the horizon."""
        task = NHANESMortalityPrediction(seq_len=SEQ_LEN, followup_months=12)
        self.assertEqual(self.label(task, mortstat=1, permth_int=20), 0)
        self.assertEqual(self.label(task, mortstat=1, permth_int=12), 1)


class TestWindows(_TaskTestCase):
    """Fixed-length windows and the wear-coverage filter."""

    def starts(self, wear, **kwargs):
        task = NHANESMortalityPrediction(seq_len=SEQ_LEN, **kwargs)
        return [s["window_start"] for s in task(self.patient(wear))]

    def test_non_overlapping_by_default(self):
        """Windows start every seq_len epochs; a partial tail is dropped."""
        self.assertEqual(self.starts(np.ones(36)), [0, 12, 24])
        self.assertEqual(self.starts(np.ones(30)), [0, 12])

    def test_stride(self):
        """A smaller stride gives overlapping windows."""
        self.assertEqual(self.starts(np.ones(36), stride=6), [0, 6, 12, 18, 24])

    def test_low_coverage_windows_are_dropped(self):
        """Windows with under 30% worn epochs are dropped."""
        wear = np.r_[
            np.ones(12), np.r_[np.ones(3), np.zeros(9)], np.r_[np.ones(4), np.zeros(8)]
        ]
        self.assertEqual(self.starts(wear), [0, 24])  # 3/12 dropped, 4/12 kept
        self.assertEqual(self.starts(wear, min_coverage=0.0), [0, 12, 24])

    def test_too_short_gives_no_samples(self):
        """A participant shorter than one window contributes nothing."""
        self.assertEqual(self.starts(np.ones(SEQ_LEN - 1)), [])


class TestSampleContents(_TaskTestCase):
    """Raw sample fields before PyHealth's processors."""

    def test_sequence_fields(self):
        """Shapes, ranges, and NaN intensity where the device was not worn."""
        wear = np.r_[np.ones(8), np.zeros(4)]
        task = NHANESMortalityPrediction(seq_len=SEQ_LEN)
        sample = task(self.patient(wear))[0]
        self.assertEqual(sample["intensity"].shape, (SEQ_LEN, 1))
        self.assertTrue(np.isnan(sample["intensity"][8:]).all())
        self.assertEqual(sample["intensity"][0, 0], 100.0)
        np.testing.assert_array_equal(sample["wear_mask"], wear)
        self.assertEqual(sample["hour"].tolist(), [0] * 12)
        self.assertEqual(sample["day"].dtype, np.int64)
        self.assertEqual((sample["age"], sample["sex"]), (50.0, 1))

    def test_covariates_and_codes(self):
        """Numeric covariates keep NaN; bad or missing codes become 0."""
        task = NHANESMortalityPrediction(seq_len=SEQ_LEN)
        self.assertEqual(task.numeric_covariates, ["age", "bmi", "waist", "dbp", "sbp"])
        self.assertEqual(task.categorical_covariates, ["sex", "income"])
        self.assertEqual(
            task.category_sizes, [CATEGORY_SIZES["sex"], CATEGORY_SIZES["income"]]
        )
        sample = task(self.patient(np.ones(SEQ_LEN), bmi=math.nan, income=77))[0]
        np.testing.assert_array_equal(
            sample["covariates"],
            np.array([50.0, np.nan, 95.0, 70.0, 120.0], np.float32),
        )
        self.assertEqual(sample["covariate_codes"].tolist(), [1, 0])
        for income, code in ((13, 13), (14, 0), (None, 0), (2.5, 0)):
            sample = task(self.patient(np.ones(SEQ_LEN), income=income, sex=None))[0]
            self.assertEqual(sample["covariate_codes"].tolist(), [0, code])

    def test_wear_aware_off(self):
        """Same windows; mask all ones and non-wear kept as real zeros."""
        wear = np.r_[np.ones(6), np.zeros(6), np.zeros(12)]
        on = NHANESMortalityPrediction(seq_len=SEQ_LEN)(self.patient(wear))
        off = NHANESMortalityPrediction(seq_len=SEQ_LEN, wear_aware=False)(
            self.patient(wear)
        )
        self.assertEqual(
            [s["window_start"] for s in off], [s["window_start"] for s in on]
        )
        self.assertTrue((off[0]["wear_mask"] == 1).all())
        self.assertEqual(off[0]["intensity"][6:].ravel().tolist(), [0.0] * 6)

    def test_covariate_sets(self):
        """'age_sex' keeps age and sex; None drops all covariate fields."""
        age_sex = NHANESMortalityPrediction(seq_len=SEQ_LEN, covariate_set="age_sex")
        sample = age_sex(self.patient(np.ones(SEQ_LEN)))[0]
        self.assertEqual(sample["covariates"].tolist(), [50.0])
        self.assertEqual(sample["covariate_codes"].tolist(), [1])
        none = NHANESMortalityPrediction(seq_len=SEQ_LEN, covariate_set=None)
        sample = none(self.patient(np.ones(SEQ_LEN)))[0]
        self.assertNotIn("covariates", sample)
        self.assertNotIn("covariates", none.input_schema)
        self.assertEqual(none.category_sizes, [])

    def test_invalid_arguments(self):
        """Out-of-range settings raise ValueError."""
        for kwargs in (
            {"seq_len": 0},
            {"stride": 0},
            {"min_coverage": 1.5},
            {"followup_months": 0},
            {"covariate_set": "labs"},
        ):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                NHANESMortalityPrediction(**kwargs)


class TestProcessedSamples(_TaskTestCase):
    """Samples after PyHealth's processors, fitted on these samples."""

    def test_processors(self):
        """Intensity z-scored on worn epochs, codes as long, missing flags."""
        task = NHANESMortalityPrediction(seq_len=SEQ_LEN)
        wear = np.r_[np.ones(9), np.zeros(3)]
        samples = []
        for i, (mortstat, months) in enumerate(((1, 30), (0, 200), (0, 200))):
            intensity = np.where(wear == 1, 50.0 * (i + 1) + np.arange(12), 0.0)
            patient = self.patient(
                wear, mortstat, months, intensity=intensity, waist=math.nan
            )
            samples += task(patient)
        dataset = create_sample_dataset(
            samples, task.input_schema, task.output_schema, dataset_name="nhanes"
        )
        processed = [dataset[i] for i in range(len(dataset))]
        worn = torch.cat([s["intensity"][:9].ravel() for s in processed])
        self.assertAlmostEqual(worn.mean().item(), 0.0, places=5)
        for sample in processed:
            self.assertEqual(tuple(sample["intensity"].shape), (SEQ_LEN, 1))
            self.assertTrue((sample["intensity"][9:] == 0).all())
            self.assertEqual(sample["hour"].dtype, torch.long)
            self.assertEqual(sample["covariate_codes"].dtype, torch.long)
            self.assertEqual(tuple(sample["covariates"].shape), (10,))
            # Values then missing flags; waist (index 2) is missing.
            self.assertEqual(sample["covariates"][5:].tolist(), [0, 0, 1, 0, 0])
        labels = sorted(int(s["mortality"].item()) for s in processed)
        self.assertEqual(labels, [0, 0, 1])


if __name__ == "__main__":
    unittest.main()
