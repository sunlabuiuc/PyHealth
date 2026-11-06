"""Paired runs keep their own directories; test is never reported as train.

Two arms of one comparison (``--task labs`` vs ``--task notes_labs`` at one
seed) used to resolve to one directory, so the second run destroyed the
first. And predictions came from ``test or val or train`` with nothing
saying which, so a run without a test split reported training metrics as
test. The split mode and the eval split are now named, warned about, and
recorded in ``run_config.json``.
"""

from __future__ import annotations

import json
import unittest
import warnings
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
    spec = importlib.util.spec_from_file_location("unified_e2e_runner_paired", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _single_patient_samples():
    return [
        {
            "patient_id": "only",
            "visit_id": "v0",
            "labs": (
                [0.0, 2.0],
                [[140.0, 1.0], [141.0, 1.1]],
            ),
            "label": 0,
        },
        {
            "patient_id": "only",
            "visit_id": "v1",
            "labs": (
                [0.0, 2.0],
                [[142.0, 1.0], [143.0, 1.1]],
            ),
            "label": 1,
        },
    ]


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


class TestPairedRunsAndEvalSplit(unittest.TestCase):
    def test_split_warns_and_labels_leaky_fallback(self):
        from pyhealth.datasets import create_sample_dataset

        runner = _runner()
        dataset = create_sample_dataset(
            samples=_single_patient_samples(),
            input_schema={"labs": "stagenet_tensor"},
            output_schema={"label": "binary"},
            in_memory=True,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            train, _val, _test, mode = runner._split_dataset(dataset, seed=1)
        self.assertEqual(mode, "by_sample_fallback_leaky")
        self.assertTrue(
            any("split_by_sample" in str(w.message) for w in caught)
        )
        self.assertGreater(len(train), 0)

    def test_patient_split_mode_is_clean(self):
        from pyhealth.datasets import create_sample_dataset

        runner = _runner()
        dataset = create_sample_dataset(
            samples=_twenty_patient_samples(),
            input_schema={"labs": "stagenet_tensor"},
            output_schema={"label": "binary"},
            in_memory=True,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _train, _val, _test, mode = runner._split_dataset(dataset, seed=42)
        self.assertEqual(mode, "by_patient")
        self.assertEqual(
            [w for w in caught if "split_by_sample" in str(w.message)], []
        )

    def test_select_inference_loader_names_each_split(self):
        runner = _runner()
        test_loader, val_loader, train_loader = object(), object(), object()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            loader, split = runner._select_inference_loader(
                test_loader, val_loader, train_loader
            )
        self.assertIs(loader, test_loader)
        self.assertEqual(split, "test")
        self.assertEqual(caught, [])

        with self.assertWarns(RuntimeWarning):
            loader, split = runner._select_inference_loader(
                None, val_loader, train_loader
            )
        self.assertIs(loader, val_loader)
        self.assertEqual(split, "val")

        with self.assertWarns(RuntimeWarning):
            loader, split = runner._select_inference_loader(
                None, None, train_loader
            )
        self.assertIs(loader, train_loader)
        self.assertEqual(split, "train")

    def test_run_directory_includes_task_and_records_eval_split(self):
        import csv

        from pyhealth.datasets import create_sample_dataset

        runner = _runner()
        dataset = create_sample_dataset(
            samples=_twenty_patient_samples(),
            input_schema={"labs": "stagenet_tensor"},
            output_schema={"label": "binary"},
            dataset_name="test_paired_runs",
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
            self.assertIn("labs_mlp_seed42", str(output_csv))
            self.assertNotIn("_dev", str(output_csv))
            run_config = json.loads(
                (Path(output_csv).parent / "run_config.json").read_text()
            )
            self.assertEqual(run_config["config"]["task"], "labs")
            self.assertEqual(run_config["config"]["split_mode"], "by_patient")
            self.assertEqual(run_config["config"]["eval_split"], "test")
            with open(output_csv, newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue(rows)
            self.assertIn("eval_split", rows[0])
            self.assertEqual(
                {row["eval_split"] for row in rows}, {"test"}
            )

    def test_run_directory_gains_dev_suffix(self):
        from pyhealth.datasets import create_sample_dataset

        runner = _runner()
        dataset = create_sample_dataset(
            samples=_twenty_patient_samples(),
            input_schema={"labs": "stagenet_tensor"},
            output_schema={"label": "binary"},
            dataset_name="test_dev_suffix",
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
                                "--dev",
                                "1000",
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
            self.assertIn("labs_mlp_seed42_dev1000", str(output_csv))
