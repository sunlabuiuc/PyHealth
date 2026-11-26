"""Tests for the PhysioNet 2012 in-hospital mortality task (synthetic data only).

Authors: Sadegh Nassiri, Kaushal Seelam (Georgia Tech, CSE 6250 project).

Paper: Xu et al., "TransEHR: Self-Supervised Transformer for Clinical Time Series
Data", ML4H 2023, https://proceedings.mlr.press/v225/xu23a.html

Covers sample building, label generation, feature groups, the observation window,
ICU-type filtering, normalization and the end-to-end ``set_task`` path.
"""

import gc
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from pyhealth.datasets import PhysioNet2012Dataset
from pyhealth.tasks import (
    PhysioNet2012InHospitalMortality,
    build_mortality_sample,
    physionet2012_normalization,
)
from pyhealth.tasks.physionet2012_inhospital_mortality import (
    LAB_VARIABLES,
    MORTALITY_VARIABLES,
    VITAL_VARIABLES,
)

HEADER = "Time,Parameter,Value\n"


def _record(record_id, age, gender, height, icu_type, lines):
    descriptors = (
        f"00:00,Age,{age}\n00:00,Gender,{gender}\n00:00,Height,{height}\n"
        f"00:00,ICUType,{icu_type}\n00:00,Weight,80\n"
    )
    return HEADER + f"00:00,RecordID,{record_id}\n" + descriptors + "".join(lines)


def _write_fake_root(root: Path) -> None:
    """Three synthetic stays: 100001 and 100003 die, 100002 survives."""
    folder = root / "set-a"
    folder.mkdir(parents=True)
    (folder / "100001.txt").write_text(_record(100001, 60, 1, 170, 3, [
        "00:10,HR,70\n", "00:40,HR,80\n", "01:05,HR,90\n", "01:05,Temp,37\n",
        "02:00,Creatinine,1.0\n", "05:30,Creatinine,1.2\n", "05:45,Creatinine,1.4\n",
        "05:30,Lactate,2.0\n", "30:00,HR,100\n",
    ]))
    (folder / "100002.txt").write_text(_record(100002, 40, 0, -1, 1, [
        "03:00,HR,60\n", "03:00,Weight,82\n", "10:00,Lactate,1.0\n",
    ]))
    (folder / "100003.txt").write_text(_record(100003, 75, 1, 160, 4, [
        "30:00,HR,70\n",
    ]))
    (root / "Outcomes-a.txt").write_text(
        "RecordID,SAPS-I,SOFA,Length_of_stay,Survival,In-hospital_death\n"
        "100001,6,1,5,-1,1\n100002,16,8,8,-1,0\n100003,10,3,4,3,1\n"
    )


class TestBuildMortalitySample(unittest.TestCase):
    """The pure NumPy sample builder: no dataset needed."""

    def build(self, minutes, parameters, values, **kwargs):
        kwargs.setdefault("variables", ["HR", "Temp"])
        kwargs.setdefault("lab_variables", ["Lactate", "Creatinine"])
        kwargs.setdefault("observation_hours", 4)
        kwargs.setdefault("max_events", 6)
        return build_mortality_sample(minutes, parameters, values, **kwargs)

    def test_hourly_grid_and_masks(self):
        out = self.build([10, 40, 65, 125], ["HR", "HR", "Temp", "HR"],
                         [70.0, 80.0, 37.0, 90.0])
        self.assertEqual(out["measurements"].shape, (4, 2))
        np.testing.assert_allclose(out["measurements"][:, 0], [75.0, 0.0, 90.0, 0.0])
        np.testing.assert_allclose(out["measurements"][:, 1], [0.0, 37.0, 0.0, 0.0])
        np.testing.assert_array_equal(out["observed"][:, 0], [1, 0, 1, 0])
        np.testing.assert_array_equal(out["padding_mask"], [1, 1, 1, 0])

    def test_window_and_missing_values_are_dropped(self):
        out = self.build([10, 30, 4 * 60, 5 * 60], ["HR", "HR", "HR", "HR"],
                         [70.0, np.nan, 99.0, 99.0])
        np.testing.assert_allclose(out["measurements"][:, 0], [70.0, 0, 0, 0])
        np.testing.assert_array_equal(out["padding_mask"], [1, 0, 0, 0])

    def test_nothing_observed_gives_none(self):
        self.assertIsNone(self.build([], [], []))
        self.assertIsNone(self.build([5 * 60], ["HR"], [1.0]))  # after the window
        self.assertIsNone(self.build([10], ["Na"], [1.0]))  # not a grid variable
        self.assertIsNone(self.build([10], ["HR"], [np.nan]))

    def test_normalization(self):
        out = self.build([10, 70], ["HR", "Temp"], [70.0, 37.0],
                         center={"HR": 60.0, "Temp": 36.0},
                         scale={"HR": 20.0, "Temp": 0.0})
        self.assertAlmostEqual(float(out["measurements"][0, 0]), 0.5)
        self.assertEqual(float(out["measurements"][1, 1]), 0.0)  # zero scale
        self.assertEqual(float(out["measurements"][2, 0]), 0.0)  # unobserved
        with self.assertRaises(ValueError):
            self.build([10], ["HR"], [1.0], center={"HR": 0.0})

    def test_clip(self):
        kwargs = {"center": {"HR": 60.0, "Temp": 36.0},
                  "scale": {"HR": 1.0, "Temp": 1.0}}
        out = self.build([10, 70], ["HR", "Temp"], [7270.0, 30.0], clip=5.0, **kwargs)
        self.assertEqual(float(out["measurements"][0, 0]), 5.0)
        self.assertEqual(float(out["measurements"][1, 1]), -5.0)
        out = self.build([10], ["HR"], [7270.0], **kwargs)  # off by default
        self.assertEqual(float(out["measurements"][0, 0]), 7210.0)
        with self.assertRaises(ValueError):  # clipping needs normalization
            self.build([10], ["HR"], [1.0], clip=5.0)
        with self.assertRaises(ValueError):
            self.build([10], ["HR"], [1.0], clip=-1.0, **kwargs)

    def test_event_stream(self):
        out = self.build(
            [30, 90, 90, 150, 200], ["Creatinine", "Lactate", "Creatinine",
                                    "HR", "Creatinine"], [1.0] * 5,
        )
        # Lactate is type 1, Creatinine type 2; ties are ordered by type.
        np.testing.assert_array_equal(out["event_types"], [2, 1, 2, 2, 0, 0])
        np.testing.assert_allclose(out["event_times"],
                                   [0.5, 1.5, 1.5, 200 / 60, 0, 0], rtol=1e-6)
        self.assertEqual(out["event_types"].dtype, np.int64)

    def test_event_caps(self):
        minutes = [10, 20, 30, 40]
        out = self.build(minutes + [15], ["Lactate"] * 4 + ["HR"], [1.0] * 5,
                         max_events_per_variable=2)
        np.testing.assert_array_equal(out["event_types"][:3], [1, 1, 0])
        out = self.build(minutes + [15], ["Lactate"] * 4 + ["HR"], [1.0] * 5,
                         max_events=3)
        np.testing.assert_array_equal(out["event_types"], [1, 1, 1])
        np.testing.assert_allclose(out["event_times"], [10 / 60, 20 / 60, 30 / 60],
                                   rtol=1e-6)


class TestPhysioNet2012InHospitalMortality(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Windows keeps memory-mapped cache files open; do not fail on cleanup.
        cls.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls.root = Path(cls.tmp.name) / "p12"
        _write_fake_root(cls.root)
        cls.dataset = PhysioNet2012Dataset(
            root=str(cls.root), cache_dir=Path(cls.tmp.name) / "cache"
        )
        cls.dataset.get_patient("100001")  # build the lazy event cache once

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def samples(self, **kwargs):
        task = PhysioNet2012InHospitalMortality(**kwargs)
        out = {}
        for pid in self.dataset.unique_patient_ids:
            for sample in task(self.dataset.get_patient(pid)):
                out[pid] = sample
        return out

    def test_variable_groups(self):
        self.assertEqual(len(MORTALITY_VARIABLES), 36)
        self.assertNotIn("MechVent", MORTALITY_VARIABLES)
        self.assertIn("Weight", MORTALITY_VARIABLES)
        self.assertEqual(set(VITAL_VARIABLES) | set(LAB_VARIABLES),
                         set(MORTALITY_VARIABLES))
        self.assertFalse(set(VITAL_VARIABLES) & set(LAB_VARIABLES))

    def test_labels_and_shapes(self):
        out = self.samples()
        self.assertEqual(sorted(out), ["100001", "100002", "100003"])
        self.assertEqual([out[p]["mortality"] for p in sorted(out)], [1, 0, 1])
        s = out["100001"]
        self.assertEqual(s["measurements"].shape, (48, 36))
        self.assertEqual(s["observed"].shape, (48, 36))
        self.assertEqual(s["padding_mask"].shape, (48,))
        self.assertEqual(s["event_times"].shape, (400,))
        self.assertEqual(s["static"].shape, (8,))

    def test_grid_values(self):
        s = self.samples()["100001"]
        hr = MORTALITY_VARIABLES.index("HR")
        creat = MORTALITY_VARIABLES.index("Creatinine")
        self.assertAlmostEqual(float(s["measurements"][0, hr]), 75.0)  # hourly mean
        self.assertAlmostEqual(float(s["measurements"][30, hr]), 100.0)
        self.assertAlmostEqual(float(s["measurements"][5, creat]), 1.3, places=5)
        self.assertEqual(int(s["padding_mask"].sum()), 31)  # hours 0..30

    def test_weight_column_holds_in_stay_measurements_only(self):
        # Every record has "00:00,Weight,80", a descriptor that goes to the
        # patients table and so never reaches the grid.
        out = self.samples()
        w = MORTALITY_VARIABLES.index("Weight")
        self.assertEqual(float(out["100001"]["observed"][:, w].sum()), 0.0)
        self.assertEqual(float(out["100001"]["measurements"][:, w].sum()), 0.0)
        # 100002 also has an in-stay "03:00,Weight,82".
        self.assertEqual(float(out["100002"]["observed"][:, w].sum()), 1.0)
        self.assertAlmostEqual(float(out["100002"]["measurements"][3, w]), 82.0)

    def test_event_stream_from_dataset(self):
        s = self.samples()["100001"]
        creat = LAB_VARIABLES.index("Creatinine") + 1
        lactate = LAB_VARIABLES.index("Lactate") + 1
        np.testing.assert_array_equal(s["event_types"][:5],
                                      [creat, creat, lactate, creat, 0])
        np.testing.assert_allclose(s["event_times"][:4], [2.0, 5.5, 5.5, 5.75])

    def test_static_features(self):
        out = self.samples(height_fill=165.0)
        # age, gender=0, gender=1, icutype=1..4, height
        np.testing.assert_allclose(out["100001"]["static"],
                                   [60, 0, 1, 0, 0, 1, 0, 170])
        np.testing.assert_allclose(out["100002"]["static"],
                                   [40, 1, 0, 1, 0, 0, 0, 165])  # height missing

    def test_feature_groups(self):
        vitals = PhysioNet2012InHospitalMortality(feature_groups=("vitals",))
        self.assertEqual(sorted(vitals.variables), sorted(VITAL_VARIABLES))
        s = self.samples(feature_groups=("vitals",))["100001"]
        self.assertEqual(s["measurements"].shape, (48, len(VITAL_VARIABLES)))
        self.assertEqual(int(s["event_types"].sum()), 0)  # no event stream
        labs = self.samples(feature_groups=("labs",))["100001"]
        self.assertEqual(labs["measurements"].shape, (48, len(LAB_VARIABLES)))
        self.assertGreater(int(labs["event_types"].sum()), 0)

    def test_observation_window_drops_empty_stays(self):
        # 100003 is only observed at hour 30.
        self.assertNotIn("100003", self.samples(observation_hours=24))
        self.assertIn("100001", self.samples(observation_hours=24))

    def test_icu_type_filter(self):
        self.assertEqual(sorted(self.samples(icu_types=[1, 3])), ["100001", "100002"])
        self.assertEqual(self.samples(icu_types=[2]), {})

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            PhysioNet2012InHospitalMortality(feature_groups=("bogus",))
        with self.assertRaises(ValueError):
            PhysioNet2012InHospitalMortality(feature_groups=())
        with self.assertRaises(ValueError):
            PhysioNet2012InHospitalMortality(center={"HR": 0.0})
        with self.assertRaises(ValueError):
            PhysioNet2012InHospitalMortality(observation_hours=0)
        with self.assertRaises(ValueError):  # clip needs normalization statistics
            PhysioNet2012InHospitalMortality(clip=5.0)
        with self.assertRaises(ValueError):  # statistics lack the task's variables
            PhysioNet2012InHospitalMortality(center={"HR": 0.0}, scale={"HR": 1.0})

    def test_normalization_statistics(self):
        center, scale = physionet2012_normalization(self.dataset, variables=["HR"])
        hourly_hr = np.array([75.0, 90.0, 100.0, 60.0, 70.0])
        self.assertAlmostEqual(center["HR"], np.median(hourly_hr), places=5)
        p5, p95 = np.percentile(hourly_hr, [5, 95])
        self.assertAlmostEqual(scale["HR"], p95 - p5, places=5)
        # Restricting to some records changes the statistics.
        center2, _ = physionet2012_normalization(
            self.dataset, patient_ids=["100002"], variables=["HR"]
        )
        self.assertEqual(center2["HR"], 60.0)
        with self.assertRaises(ValueError):
            physionet2012_normalization(self.dataset, variables=["TroponinT"])

    def test_set_task_end_to_end(self):
        variables = ["HR", "Temp", "Creatinine", "Lactate"]
        center = {v: 0.0 for v in variables}
        scale = {v: 1.0 for v in variables}
        task = PhysioNet2012InHospitalMortality(
            center=center, scale=scale,
            vital_variables=["HR", "Temp"], lab_variables=["Creatinine", "Lactate"],
        )
        self.assertEqual(task.variables, ["Creatinine", "HR", "Lactate", "Temp"])
        samples = self.dataset.set_task(task)
        self.assertEqual(len(samples), 3)
        item = samples[0]
        self.assertEqual(item["event_types"].dtype, torch.long)
        self.assertEqual(item["measurements"].dtype, torch.float32)
        self.assertEqual(tuple(item["measurements"].shape), (48, 4))
        self.assertIn(float(item["mortality"]), (0.0, 1.0))
        del item, samples  # release the memory-mapped cache before cleanup
        gc.collect()


if __name__ == "__main__":
    unittest.main()
