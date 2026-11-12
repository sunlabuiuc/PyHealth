"""Tests for the DSA activity classification task (synthetic data only)."""

import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from pyhealth.datasets.dsa import compute_channel_ranges
from pyhealth.tasks.dsa_activity_classification import DSAActivityClassification


class _FakePatient:
    """Minimal stand-in for a PyHealth patient."""

    def __init__(self, patient_id, events):
        self.patient_id = patient_id
        self._events = events

    def get_events(self, event_type=None):
        return self._events


class TestDSAActivityClassification(unittest.TestCase):
    """Tests sample building, sensor selection and validation."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.raw = np.arange(125 * 45, dtype=np.float64).reshape(125, 45)
        events = []
        for segment, activity in ((1, 1), (2, "3")):  # activity may be a str
            path = os.path.join(cls.tmp, f"s{segment:02d}.txt")
            np.savetxt(path, cls.raw + segment, delimiter=",")
            events.append(
                SimpleNamespace(
                    signal_file=path, activity=activity, segment=segment
                )
            )
        cls.patient = _FakePatient("p1", events)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def test_default_schema(self):
        task = DSAActivityClassification()
        self.assertEqual(task.target_sensor, "right_arm")
        self.assertNotIn("right_arm", task.source_sensors)
        self.assertEqual(len(task.source_sensors), 4)
        self.assertEqual(task.input_schema["signal"], "tensor")
        self.assertEqual(task.input_schema["source_torso"], "tensor")
        self.assertEqual(task.output_schema, {"label": "multiclass"})

    def test_samples_shapes_labels_and_ids(self):
        samples = DSAActivityClassification()(self.patient)
        self.assertEqual(len(samples), 2)
        self.assertEqual([s["label"] for s in samples], [0, 2])
        self.assertEqual({s["patient_id"] for s in samples}, {"p1"})
        for sample in samples:
            self.assertEqual(sample["signal"].shape, (125, 9))
            self.assertEqual(sample["source_torso"].shape, (125, 9))

    def test_target_and_sources_are_time_aligned_slices(self):
        task = DSAActivityClassification(
            target_sensor="right_arm", source_sensors=["torso"]
        )
        sample = task(self.patient)[0]
        raw = (self.raw + 1).astype(np.float32)
        # torso = columns 0-8, right arm = columns 9-17 of the same rows.
        np.testing.assert_array_equal(sample["source_torso"], raw[:, 0:9])
        np.testing.assert_array_equal(sample["signal"], raw[:, 9:18])

    def test_minmax_scales_every_channel_to_unit_range(self):
        files = [e.signal_file for e in self.patient._events]
        ranges = compute_channel_ranges(files)
        task = DSAActivityClassification(channel_ranges=ranges)
        samples = task(self.patient)
        for key in ("signal", "source_torso"):
            stacked = np.concatenate([s[key] for s in samples])
            self.assertAlmostEqual(float(stacked.min()), -1.0, places=5)
            self.assertAlmostEqual(float(stacked.max()), 1.0, places=5)
        self.assertNotEqual(task.task_name, DSAActivityClassification().task_name)

    def test_bad_channel_ranges_shape_raises(self):
        with self.assertRaises(ValueError):
            DSAActivityClassification(channel_ranges=(np.zeros(3), np.ones(3)))

    def test_target_only_has_no_source_keys(self):
        task = DSAActivityClassification(target_sensor="torso", source_sensors=[])
        self.assertEqual(list(task.input_schema), ["signal"])
        sample = task(self.patient)[0]
        self.assertNotIn("source_right_arm", sample)

    def test_task_name_depends_on_sensor_setup(self):
        a = DSAActivityClassification(target_sensor="torso")
        b = DSAActivityClassification(target_sensor="left_leg")
        self.assertNotEqual(a.task_name, b.task_name)

    def test_task_name_is_short_and_depends_on_sources(self):
        full = DSAActivityClassification()
        fewer = DSAActivityClassification(source_sensors=["torso"])
        self.assertNotEqual(full.task_name, fewer.task_name)
        self.assertLess(len(full.task_name), 40)

    def test_invalid_configs_raise(self):
        with self.assertRaises(ValueError):
            DSAActivityClassification(target_sensor="wrist")
        with self.assertRaises(ValueError):
            DSAActivityClassification(source_sensors=["elbow"])
        with self.assertRaises(ValueError):
            DSAActivityClassification(
                target_sensor="torso", source_sensors=["torso"]
            )
        with self.assertRaises(ValueError):
            DSAActivityClassification(source_sensors=["torso", "torso"])

    def test_patient_without_events_gives_no_samples(self):
        self.assertEqual(DSAActivityClassification()(_FakePatient("p9", [])), [])


if __name__ == "__main__":
    unittest.main()
