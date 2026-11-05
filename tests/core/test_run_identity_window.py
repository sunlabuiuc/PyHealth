"""Run identity covers the observation window.

An observation-window arm is a different experiment from the full-stay run
at the same task/model/seed; without the window in the name the two share
an output directory and collide in W&B.
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
        "unified_e2e_runner_window", path
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


class TestWindowAwareRunIdentity(unittest.TestCase):
    def test_window_suffix(self):
        runner = _runner()
        self.assertEqual(runner._window_suffix(0), "")
        self.assertEqual(runner._window_suffix(0.0), "")
        self.assertEqual(runner._window_suffix(24), "_w24")
        self.assertEqual(runner._window_suffix(24.0), "_w24")
        self.assertEqual(runner._window_suffix(24.5), "_w24.5")
        self.assertEqual(runner._window_suffix(0.5), "_w0.5")

    def test_wandb_logger_carries_group_and_job_type(self):
        runner = _runner()
        logger = runner.WandbLogger(
            enabled=False,
            project="p",
            entity=None,
            run_name="r",
            tags=["labs"],
            config={},
            group="labs_w24",
            job_type="mlp",
        )
        self.assertEqual(logger.group, "labs_w24")
        self.assertEqual(logger.job_type, "mlp")

    def test_window_arm_lands_in_its_own_directory(self):
        from pyhealth.datasets import create_sample_dataset

        runner = _runner()
        dataset = create_sample_dataset(
            samples=_twenty_patient_samples(),
            input_schema={"labs": "stagenet_tensor"},
            output_schema={"label": "binary"},
            dataset_name="test_window",
        )
        with TemporaryDirectory() as tmp:
            args = Namespace(
                **{
                    **vars(
                        runner.parse_args(
                            [
                                "--ehr-root",
                                "/tmp/does-not-exist",
                                "--task",
                                "labs",
                                "--model",
                                "mlp",
                                "--epochs",
                                "0",
                                "--batch-size",
                                "4",
                                "--device",
                                "cpu",
                                "--seed",
                                "42",
                                "--observation-window-hours",
                                "24",
                                "--no-cache-frozen-text",
                                "--output-dir",
                                tmp,
                            ]
                        )
                    ),
                    "embedding_dim": 8,
                    "hidden_dim": 8,
                }
            )
            output_csv = runner.run(args, sample_dataset=dataset)
            self.assertIn("labs_mlp_seed42_w24", str(output_csv))
            full_stay = str(output_csv).replace("_w24", "")
            self.assertNotEqual(str(output_csv), full_stay)
