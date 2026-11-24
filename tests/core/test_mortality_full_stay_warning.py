"""Warn when a mortality run keeps the full stay.

``window_hours=0`` (the default) keeps the full stay through discharge, so
mortality features can include data up to discharge or death and the label
can leak into the inputs. A run that starts with an effective observation
window of 0 on a mortality task logs one warning naming the flag that sets
a window. A positive window stays silent, and so does a non-mortality task
(the runner currently offers only mortality tasks, so that case runs
through ``run()`` with a non-mortality task name on a synthetic dataset).
"""

from __future__ import annotations

import unittest
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory


def _runner():
    import importlib.util

    path = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "mortality_prediction"
        / "unified_embedding_e2e_mimic4.py"
    )
    spec = importlib.util.spec_from_file_location(
        "unified_e2e_runner_full_stay", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _twenty_patient_samples():
    samples = []
    for i in range(20):
        samples.append(
            {
                "patient_id": f"patient-{i}",
                "visit_id": "visit-0",
                "labs": (
                    [0.0, 2.0],
                    [[140.0 + i, 1.0], [141.0 + i, 1.1]],
                ),
                "label": i % 2,
            }
        )
    return samples


def _run_args(runner, tmp, task, window_hours):
    argv = [
        "--ehr-root",
        "/tmp/does-not-exist",
        "--model",
        "mlp",
        "--task",
        task,
        "--epochs",
        "0",
        "--device",
        "cpu",
        "--seed",
        "42",
        "--no-cache-frozen-text",
        "--output-dir",
        tmp,
    ]
    if window_hours != 0:
        argv += ["--observation-window-hours", str(window_hours)]
    return Namespace(
        **{
            **vars(runner.parse_args(argv)),
            "embedding_dim": 8,
            "hidden_dim": 8,
        }
    )


class TestMortalityFullStayWarning(unittest.TestCase):
    def test_mortality_tasks_come_from_output_schema(self):
        runner = _runner()
        self.assertTrue(runner._is_mortality_task("labs"))
        self.assertTrue(runner._is_mortality_task("notes_labs"))
        self.assertFalse(runner._is_mortality_task("length_of_stay"))

    def test_warns_at_zero_for_each_mortality_task(self):
        from pyhealth.datasets import create_sample_dataset

        runner = _runner()
        for task in ("labs", "notes_labs"):
            dataset = create_sample_dataset(
                samples=_twenty_patient_samples(),
                input_schema={"labs": "stagenet_tensor"},
                output_schema={"label": "binary"},
                dataset_name=f"test_full_stay_{task}",
            )
            with TemporaryDirectory() as tmp:
                args = _run_args(runner, tmp, task, 0)
                with self.assertLogs(level="WARNING") as logs:
                    runner.run(args, sample_dataset=dataset)
            output = "\n".join(logs.output)
            self.assertIn("full stay through discharge", output)
            self.assertIn("--observation-window-hours", output)

    def test_silent_at_positive_window(self):
        from pyhealth.datasets import create_sample_dataset

        runner = _runner()
        self.assertTrue(runner._is_mortality_task("labs"))
        dataset = create_sample_dataset(
            samples=_twenty_patient_samples(),
            input_schema={"labs": "stagenet_tensor"},
            output_schema={"label": "binary"},
            dataset_name="test_full_stay_window",
        )
        with TemporaryDirectory() as tmp:
            args = _run_args(runner, tmp, "labs", 24)
            with self.assertNoLogs(level="WARNING"):
                runner.run(args, sample_dataset=dataset)

    def test_silent_for_non_mortality_task_at_zero(self):
        from pyhealth.datasets import create_sample_dataset

        runner = _runner()
        self.assertFalse(runner._is_mortality_task("length_of_stay"))
        dataset = create_sample_dataset(
            samples=_twenty_patient_samples(),
            input_schema={"labs": "stagenet_tensor"},
            output_schema={"label": "binary"},
            dataset_name="test_full_stay_other_task",
        )
        with TemporaryDirectory() as tmp:
            # No --task value outside the parser choices can reach run()
            # from the CLI today; a future non-mortality task would arrive
            # as a task name whose class lacks "mortality".
            args = _run_args(runner, tmp, "labs", 0)
            args.task = "length_of_stay"
            with self.assertNoLogs(level="WARNING"):
                runner.run(args, sample_dataset=dataset)
