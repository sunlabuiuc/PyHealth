"""Test metrics persist to disk whether or not W&B logging is on.

``metrics.json`` carries validation only, so the test numbers worth reporting
used to live nowhere on disk: only in stdout and W&B, recoverable
afterwards only by re-scoring predictions by hand.
"""

from __future__ import annotations

import json
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
        "unified_e2e_runner_metrics", path
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


class TestPersistTestMetrics(unittest.TestCase):
    def test_test_metrics_json_written_next_to_predictions(self):
        from pyhealth.datasets import create_sample_dataset

        runner = _runner()
        dataset = create_sample_dataset(
            samples=_twenty_patient_samples(),
            input_schema={"labs": "stagenet_tensor"},
            output_schema={"label": "binary"},
            dataset_name="test_persist_test_metrics",
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
            test_path = Path(output_csv).parent / "test_metrics.json"
            self.assertTrue(test_path.is_file())
            test_metrics = json.loads(test_path.read_text())
            self.assertEqual(test_metrics["eval_split"], "test")
            scores = {
                k: v for k, v in test_metrics.items() if k != "eval_split"
            }
            self.assertGreater(len(scores), 0)
            metrics = json.loads(
                (Path(output_csv).parent / "metrics.json").read_text()
            )
            self.assertEqual(scores.keys(), metrics["test_scores"].keys())

    def test_create_directory_survives_existing_dirs(self):
        from pyhealth.utils import create_directory

        with TemporaryDirectory() as tmp:
            create_directory(tmp)
            create_directory(tmp)
            create_directory(str(Path(tmp) / "new" / "nested"))
            self.assertTrue((Path(tmp) / "new" / "nested").is_dir())
