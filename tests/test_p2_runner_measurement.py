"""Proof that paired runs cannot overwrite each other or report train as test.

The run directory was ``{model}_seed{seed}``. ``--task labs`` and
``--task notes_labs`` at seed 42 resolved to one path and the second run
destroyed the first. ``split_by_patient`` fell back to ``split_by_sample``
with no warning (patient leak). Predictions came from
``test_loader or val_loader or train_loader``.

The sixth backbone was also missing: ``--model`` had no ``mlp``.

Repro::

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. \\
      python -m pytest tests/test_p2_runner_measurement.py -q
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


RUNNER = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "mortality_prediction"
    / "unified_embedding_e2e_mimic4.py"
)


def _load_runner():
    spec = importlib.util.spec_from_file_location("e2e_runner_measurement", RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _parse(mod, *argv):
    with mock.patch.object(sys, "argv", ["e2e.py", "--ehr-root", "/tmp", *argv]):
        return mod.parse_args()


def _exp_name_assignment() -> str:
    for line in RUNNER.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("exp_name = "):
            return stripped
    raise AssertionError("exp_name is no longer assigned in the runner")


class TestP2RunnerMeasurement(unittest.TestCase):
    def test_run_directory_includes_the_task(self):
        assignment = _exp_name_assignment()
        self.assertIn("args.task", assignment)
        template = assignment.split("=", 1)[1].strip()
        args = SimpleNamespace(task="labs", model="transformer", seed=42)
        labs = eval(template, {"args": args})  # noqa: S307
        args.task = "notes_labs"
        notes = eval(template, {"args": args})  # noqa: S307
        self.assertEqual(labs, "labs_transformer_seed42")
        self.assertEqual(notes, "notes_labs_transformer_seed42")
        self.assertNotEqual(labs, notes)

    def test_cli_accepts_mlp(self):
        from pyhealth.models import MLP

        self.assertTrue(hasattr(MLP, "_forward_unified"))
        mod = _load_runner()
        args = _parse(mod, "--model", "mlp")
        self.assertEqual(args.model, "mlp")

    def test_split_warns_and_labels_leaky_fallback(self):
        from pyhealth.datasets import create_sample_dataset

        mod = _load_runner()
        samples = [
            {
                "patient_id": "only",
                "visit_id": "v0",
                "labs": [1.0, 2.0],
                "label": 0,
            },
            {
                "patient_id": "only",
                "visit_id": "v1",
                "labs": [3.0, 4.0],
                "label": 1,
            },
        ]
        dataset = create_sample_dataset(
            samples=samples,
            input_schema={"labs": "tensor"},
            output_schema={"label": "binary"},
            in_memory=True,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _train, _val, _test, mode = mod._split_dataset(dataset, seed=1)
        self.assertEqual(mode, "by_sample_fallback_leaky")
        self.assertTrue(any("split_by_sample" in str(w.message) for w in caught))

    def test_eval_split_is_named_and_recorded(self):
        src = RUNNER.read_text()
        self.assertIn('inference_loader, eval_split = test_loader, "test"', src)
        self.assertIn("write_run_config", src)
        self.assertIn("These are not test metrics", src)
        self.assertNotIn(
            "inference_loader = test_loader or val_loader or train_loader",
            src,
        )

    def test_jamba_cli_default_matches_the_library(self):
        from pyhealth.models.jamba_ehr import JambaLayer

        self.assertEqual(JambaLayer.__init__.__defaults__[1], 6)
        mod = _load_runner()
        args = _parse(mod, "--model", "jambaehr")
        self.assertEqual(args.jamba_transformer_layers, 2)
        self.assertEqual(args.jamba_mamba_layers, 6)

    def test_observation_window_defaults_to_full_stay(self):
        mod = _load_runner()
        args = _parse(mod)
        self.assertIsNone(args.observation_window_hours)
        args_24 = _parse(mod, "--observation-window-hours", "24")
        self.assertEqual(args_24.observation_window_hours, 24)
