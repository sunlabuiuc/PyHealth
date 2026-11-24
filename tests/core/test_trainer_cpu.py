"""Per-epoch CPU accounting next to the existing wall-time accounting.

Self-only CPU time badly understates a run whose cost is data loading,
since workers are separate processes, so the clock counts the process tree.
The series records wall time per epoch in the runner (the Trainer returns
``None``), so CPU rides the same epoch boundaries.
"""

from __future__ import annotations

import json
import unittest
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory

import torch


def _runner():
    import importlib.util

    path = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "mortality_prediction"
        / "unified_embedding_e2e_mimic4.py"
    )
    spec = importlib.util.spec_from_file_location(
        "unified_e2e_runner_cpu", path
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


class TestTrainerCpuAccounting(unittest.TestCase):
    def test_cpu_seconds_is_monotonic(self):
        from pyhealth.trainer import _cpu_seconds

        first = _cpu_seconds()
        total = 0
        for i in range(1000):
            total += i * i
        second = _cpu_seconds()
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertGreaterEqual(first, 0.0)
        self.assertGreaterEqual(second, first)
        self.assertGreater(total, 0)

    def test_run_records_per_epoch_cpu(self):
        from pyhealth.datasets import create_sample_dataset

        runner = _runner()
        dataset = create_sample_dataset(
            samples=_twenty_patient_samples(),
            input_schema={"labs": "stagenet_tensor"},
            output_schema={"label": "binary"},
            dataset_name="test_cpu",
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
                                "1",
                                "--batch-size",
                                "4",
                                "--device",
                                "cpu",
                                "--seed",
                                "42",
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
            metrics = json.loads(
                (Path(output_csv).parent / "metrics.json").read_text()
            )
            self.assertEqual(
                len(metrics["epoch_train_cpu_seconds"]),
                len(metrics["epoch_train_seconds"]),
            )
            for value in metrics["epoch_train_cpu_seconds"]:
                self.assertTrue(torch.isfinite(torch.tensor(value)).item())
                self.assertGreaterEqual(value, 0.0)
            self.assertTrue(
                torch.isfinite(torch.tensor(metrics["train_cpu_seconds"]))
            )
            self.assertGreaterEqual(metrics["train_cpu_util_pct"], 0.0)
