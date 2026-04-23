"""Unit tests for DecompensationDSA task and make_synthetic_dsa_samples.

These tests are fully self-contained — they import only numpy and the task
module directly (no full PyHealth installation required).

Run from the repository root::

    python -m pytest tests/core/test_decompensation_dsa.py -v
"""

from __future__ import annotations

import importlib
import importlib.util
import pathlib
import sys
import types
import unittest

import numpy as np

# ---------------------------------------------------------------------------
# Import the task module directly (bypasses full pyhealth package init)
# ---------------------------------------------------------------------------

_repo   = pathlib.Path(__file__).parents[2]
_target = _repo / "pyhealth" / "tasks" / "decompensation_dsa.py"

# Stub base_task so we don't need polars
_tasks_pkg = types.ModuleType("pyhealth.tasks")
_tasks_pkg.base_task = types.ModuleType("pyhealth.tasks.base_task")

class _BaseTask:
    task_name:     str  = ""
    input_schema:  dict = {}
    output_schema: dict = {}

    def __init__(self, code_mapping=None):
        pass

    def __call__(self, patient):
        raise NotImplementedError

_tasks_pkg.base_task.BaseTask = _BaseTask
sys.modules["pyhealth"]             = types.ModuleType("pyhealth")
sys.modules["pyhealth.tasks"]       = _tasks_pkg
sys.modules["pyhealth.tasks.base_task"] = _tasks_pkg.base_task

_spec = importlib.util.spec_from_file_location("pyhealth.tasks.decompensation_dsa", _target)
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

DecompensationDSA          = _mod.DecompensationDSA
make_synthetic_dsa_samples = _mod.make_synthetic_dsa_samples


# ---------------------------------------------------------------------------
# Tests for make_synthetic_dsa_samples
# ---------------------------------------------------------------------------

class TestMakeSyntheticDSASamples(unittest.TestCase):

    def setUp(self) -> None:
        self.samples = make_synthetic_dsa_samples(
            n_patients=20, n_features=8, horizon=24,
            max_seq_len=50, event_rate=0.5, seed=0,
        )

    def test_correct_count(self) -> None:
        self.assertEqual(len(self.samples), 20)

    def test_required_keys(self) -> None:
        for s in self.samples:
            for k in ("patient_id", "timeseries", "label"):
                self.assertIn(k, s)

    def test_timeseries_shape(self) -> None:
        for s in self.samples:
            ts = np.array(s["timeseries"])
            self.assertEqual(ts.shape, (50, 8))

    def test_label_binary(self) -> None:
        for s in self.samples:
            self.assertIn(s["label"], (0, 1))

    def test_has_positive_and_negative(self) -> None:
        labels = [s["label"] for s in self.samples]
        self.assertGreater(sum(labels), 0)
        self.assertGreater(len(labels) - sum(labels), 0)

    def test_unique_patient_ids(self) -> None:
        ids = [s["patient_id"] for s in self.samples]
        self.assertEqual(len(ids), len(set(ids)))

    def test_reproducibility(self) -> None:
        a = make_synthetic_dsa_samples(n_patients=5, seed=7)
        b = make_synthetic_dsa_samples(n_patients=5, seed=7)
        for sa, sb in zip(a, b):
            np.testing.assert_array_equal(
                np.array(sa["timeseries"]), np.array(sb["timeseries"])
            )

    def test_different_seeds_differ(self) -> None:
        a = make_synthetic_dsa_samples(n_patients=5, seed=1)
        b = make_synthetic_dsa_samples(n_patients=5, seed=2)
        self.assertFalse(
            np.allclose(np.array(a[0]["timeseries"]), np.array(b[0]["timeseries"]))
        )

    def test_zero_event_rate(self) -> None:
        s = make_synthetic_dsa_samples(n_patients=10, event_rate=0.0, seed=0)
        self.assertTrue(all(x["label"] == 0 for x in s))

    def test_full_event_rate(self) -> None:
        s = make_synthetic_dsa_samples(n_patients=10, event_rate=1.0, seed=0)
        self.assertTrue(all(x["label"] == 1 for x in s))

    def test_default_max_seq_len(self) -> None:
        s = make_synthetic_dsa_samples(n_patients=3)
        ts = np.array(s[0]["timeseries"])
        self.assertEqual(ts.shape[0], 100)   # default max_seq_len


# ---------------------------------------------------------------------------
# Tests for DecompensationDSA schema
# ---------------------------------------------------------------------------

class TestDecompensationDSASchema(unittest.TestCase):

    def test_task_name(self) -> None:
        self.assertEqual(DecompensationDSA.task_name, "DecompensationDSA")

    def test_input_schema_timeseries(self) -> None:
        self.assertIn("timeseries", DecompensationDSA.input_schema)

    def test_output_schema_label(self) -> None:
        self.assertIn("label", DecompensationDSA.output_schema)
        self.assertEqual(DecompensationDSA.output_schema["label"], "binary")

    def test_inherits_base_task(self) -> None:
        self.assertTrue(issubclass(DecompensationDSA, _BaseTask))


if __name__ == "__main__":
    unittest.main()
