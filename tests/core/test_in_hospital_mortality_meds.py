"""Tests for InHospitalMortalityMEDS.

The synthetic tests build a small MEDS-shaped dataset (typed Parquet shards
with a ``hadm_id`` column, plus a stay-aware config) with no real data and no
downloads. Feature-content assertions apply the task directly to real
``Patient`` objects (``dataset.get_patient(...)``), because that yields the
task's raw ``List[Dict]`` output; going through ``set_task`` instead would
tokenize the ``codes`` sequence into integer indices (the ``sequence``
processor), which is the correct user path but hides the string codes the
leakage tests must inspect. A separate test exercises ``set_task`` end to end
to confirm the intended path produces the expected number of samples.

The central property under test is the absence of label leakage: for a stay
that ends in death, neither the discharge event nor the ``MEDS_DEATH``
sentinel may appear in the emitted feature sequence.
``TestInHospitalMortalityMEDSDemoSmoke`` optionally exercises the public
MIMIC-IV demo in MEDS when it is available locally, and is skipped otherwise.
"""

import os
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

from pyhealth.datasets import MEDSDataset
from pyhealth.tasks import InHospitalMortalityMEDS
from pyhealth.tasks.in_hospital_mortality_meds import (
    DEATH_CODE,
    DISCHARGE_PREFIX,
)

T0 = datetime(2024, 1, 1, 8, 0, 0)

# Stay-aware config: exposes hadm_id, which the task requires.
_CONFIG = """version: "1.0"
tables:
  meds:
    file_path: "data"
    patient_id: "subject_id"
    timestamp: "time"
    attributes:
      - "code"
      - "numeric_value"
      - "hadm_id"
"""


def _events_to_frame(rows):
    """Rows are (subject_id, offset_hours, code, hadm_id_or_None)."""
    return pl.DataFrame(
        {
            "subject_id": pl.Series([r[0] for r in rows], dtype=pl.Int64),
            "time": pl.Series(
                [T0 + timedelta(hours=r[1]) for r in rows], dtype=pl.Datetime("us")
            ),
            "code": pl.Series([r[2] for r in rows], dtype=pl.String),
            "numeric_value": pl.Series([None] * len(rows), dtype=pl.Float32),
            "hadm_id": pl.Series(
                [r[3] for r in rows],
                dtype=pl.Int64,  # nullable
            ),
        }
    )


class TestInHospitalMortalityMEDS(unittest.TestCase):
    """Task behavior on a synthetic MEDS dataset via set_task."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.root = self.temp_dir / "meds"
        (self.root / "data").mkdir(parents=True)
        self.cache_root = self.temp_dir / "cache"
        self.config_path = self.temp_dir / "meds_hadm.yaml"
        self.config_path.write_text(_CONFIG)
        self._write_default_cohort()

    def tearDown(self):
        if self.temp_dir.exists():
            # ignore_errors: litdata may keep chunk handles open on Windows.
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _write_default_cohort(self):
        # Subject 1: a stay ending in death (hadm 555), a MEDS_DEATH a few
        # hours later (null hadm), a post-death stray event, then a second
        # stay ending at home (hadm 777). Subject 2: a survived stay whose
        # subject later dies out of hospital.
        rows = [
            (1, 0, "HOSPITAL_ADMISSION//EW", 555),
            (1, 2, "LAB//50912", 555),
            (1, 6, "MED//aspirin", 555),
            (1, 10, "HOSPITAL_DISCHARGE//DIED", 555),
            (1, 14, DEATH_CODE, None),
            (1, 16, "LAB//stray", None),
            (1, 120, "HOSPITAL_ADMISSION//OBS", 777),
            (1, 121, "LAB//x", 777),
            (1, 144, "HOSPITAL_DISCHARGE//HOME", 777),
            (2, 0, "HOSPITAL_ADMISSION//EW", 999),
            (2, 5, "HOSPITAL_DISCHARGE//HOME", 999),
            (2, 200, DEATH_CODE, None),
        ]
        _events_to_frame(rows).write_parquet(self.root / "data" / "0.parquet")

    def _dataset(self):
        return MEDSDataset(
            root=str(self.root),
            config_path=str(self.config_path),
            cache_dir=self.cache_root,
        )

    def _apply(self, task=None):
        """Applies the task to every patient, returning raw sample dicts.

        This mirrors what ``set_task`` does per patient but keeps the task's
        untokenized output so feature sequences remain inspectable.
        """
        task = task or InHospitalMortalityMEDS()
        dataset = self._dataset()
        samples = []
        for pid in dataset.unique_patient_ids:
            samples.extend(task(dataset.get_patient(pid)))
        return samples

    def _by_hadm(self, samples):
        return {s["hadm_id"]: s for s in samples}

    def test_one_sample_per_completed_stay(self):
        samples = self._apply()
        # Three completed stays across the two subjects.
        self.assertEqual(len(samples), 3)
        self.assertEqual(sorted(s["hadm_id"] for s in samples), [555, 777, 999])
        # ids are integral, not promoted floats
        self.assertTrue(all(isinstance(s["hadm_id"], int) for s in samples))

    def test_labels_from_discharge_code(self):
        by = self._by_hadm(self._apply())
        self.assertEqual(by[555]["mortality"], 1)  # DIED
        self.assertEqual(by[777]["mortality"], 0)  # HOME
        self.assertEqual(by[999]["mortality"], 0)  # HOME (subject dies later)

    def test_no_label_leakage_in_positive_stay(self):
        """The defining safety property: outcome never enters the features."""
        died = self._by_hadm(self._apply())[555]
        for code in died["codes"]:
            self.assertFalse(code.startswith(DISCHARGE_PREFIX))
            self.assertNotEqual(code, DEATH_CODE)
        # Exactly the pre-discharge, non-death events, in order.
        self.assertEqual(
            died["codes"],
            ["HOSPITAL_ADMISSION//EW", "LAB//50912", "MED//aspirin"],
        )
        # The stray post-death event is excluded too.
        self.assertNotIn("LAB//stray", died["codes"])

    def test_meds_death_without_hadm_never_labels_a_stay(self):
        # Subject 2 dies out of hospital; the in-hospital stay stays negative.
        self.assertEqual(self._by_hadm(self._apply())[999]["mortality"], 0)

    def test_first_hours_requires_sufficient_length_of_stay(self):
        # Default cohort: no stay exceeds 48h, so the early-warning variant
        # yields nothing.
        task = InHospitalMortalityMEDS(observation_window="first_hours")
        self.assertEqual(self._apply(task), [])

    def test_first_hours_observes_only_the_window(self):
        # A single long stay (LOS 100h) observed for its first 48h.
        (self.root / "data" / "0.parquet").unlink()
        rows = [
            (7, 0, "HOSPITAL_ADMISSION//EW", 900),
            (7, 12, "LAB//a", 900),
            (7, 47, "LAB//b", 900),  # inside 48h
            (7, 60, "LAB//c", 900),  # outside 48h
            (7, 100, "HOSPITAL_DISCHARGE//DIED", 900),
        ]
        _events_to_frame(rows).write_parquet(self.root / "data" / "0.parquet")
        task = InHospitalMortalityMEDS(
            observation_window="first_hours", window_hours=48.0
        )
        samples = self._apply(task)
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["mortality"], 1)  # eventual outcome
        self.assertEqual(
            samples[0]["codes"],
            ["HOSPITAL_ADMISSION//EW", "LAB//a", "LAB//b"],
        )
        self.assertNotIn("LAB//c", samples[0]["codes"])

    def test_discharge_boundary_is_half_open(self):
        (self.root / "data" / "0.parquet").unlink()
        rows = [
            (8, 0, "HOSPITAL_ADMISSION//EW", 111),
            (8, 4, "LAB//inside", 111),
            (8, 5, "LAB//at_discharge", 111),  # exactly at t_discharge
            (8, 5, "HOSPITAL_DISCHARGE//HOME", 111),
        ]
        _events_to_frame(rows).write_parquet(self.root / "data" / "0.parquet")
        samples = self._apply()
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["codes"], ["HOSPITAL_ADMISSION//EW", "LAB//inside"])

    def test_invalid_parameters_raise(self):
        with self.assertRaises(ValueError):
            InHospitalMortalityMEDS(observation_window="bogus")
        with self.assertRaises(ValueError):
            InHospitalMortalityMEDS(window_hours=0)
        with self.assertRaises(ValueError):
            InHospitalMortalityMEDS(window_hours=-3)

    def test_set_task_integration_yields_expected_count(self):
        """The intended user path runs and produces one sample per stay.

        Codes are tokenized by the sequence processor here, so only the
        sample count (structure) is asserted; content/leakage is covered by
        the get_patient-based tests above.
        """
        dataset = self._dataset()
        sample_dataset = dataset.set_task(InHospitalMortalityMEDS())
        self.assertEqual(len(sample_dataset), 3)


def _demo_root() -> str:
    env = os.environ.get("MEDS_DEMO_ROOT")
    if env:
        return env
    test_dir = Path(__file__).parent.parent.parent
    return str(test_dir / "test-resources" / "meds_demo")


@unittest.skipUnless(
    Path(_demo_root()).is_dir(),
    "MIMIC-IV demo in MEDS format not available locally "
    "(set MEDS_DEMO_ROOT or place it under test-resources/meds_demo)",
)
class TestInHospitalMortalityMEDSDemoSmoke(unittest.TestCase):
    """Smoke test on the public MIMIC-IV demo in MEDS format.

    Requires a config exposing hadm_id; this test writes one next to a
    temporary cache. The demo (PhysioNet, https://doi.org/10.13026/t2y8-ea41,
    ODbL v1.0) is never downloaded here.
    """

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config_path = self.temp_dir / "meds_hadm.yaml"
        self.config_path.write_text(_CONFIG)

    def tearDown(self):
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_produces_stays_without_leakage(self):
        dataset = MEDSDataset(
            root=_demo_root(),
            config_path=str(self.config_path),
            cache_dir=self.temp_dir,
        )
        task = InHospitalMortalityMEDS()
        # Apply per patient to inspect raw (untokenized) code sequences.
        samples = []
        for pid in dataset.unique_patient_ids:
            samples.extend(task(dataset.get_patient(pid)))
        self.assertGreater(len(samples), 0)
        # No sample may contain a discharge or death code (leakage guard).
        for sample in samples:
            for code in sample["codes"]:
                self.assertFalse(code.startswith(DISCHARGE_PREFIX))
                self.assertNotEqual(code, DEATH_CODE)
        n_positive = sum(int(s["mortality"]) for s in samples)
        self.assertGreater(n_positive, 0)
        self.assertLess(n_positive, len(samples))


if __name__ == "__main__":
    unittest.main()
