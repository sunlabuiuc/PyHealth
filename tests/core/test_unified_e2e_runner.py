"""Smoke tests for the unified-embedding e2e runner (no MIMIC data).

Covers the CLI surface the cache A/B depends on (--help, the cache flags,
the window default) plus one tiny synthetic end-to-end run through model
training, evaluation, inference, and ``metrics.json``.
"""

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
    spec = importlib.util.spec_from_file_location("unified_e2e_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _samples(n=20):
    samples = []
    for i in range(n):
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


class TestUnifiedE2ERunner(unittest.TestCase):
    def test_help(self):
        runner = _runner()
        with self.assertRaises(SystemExit) as ctx:
            runner.main(["--help"])
        self.assertEqual(ctx.exception.code, 0)

    def test_defaults(self):
        runner = _runner()
        args = runner.parse_args(["--ehr-root", "/tmp/does-not-exist"])
        self.assertEqual(args.task, "labs")
        self.assertEqual(args.observation_window_hours, 0)
        self.assertTrue(args.cache_frozen_text)
        self.assertEqual(args.max_frozen_text_cache, 1_000_000)
        self.assertEqual(args.accumulation_steps, 1)
        self.assertFalse(args.use_amp)

    def test_no_cache_flag(self):
        runner = _runner()
        args = runner.parse_args(
            ["--ehr-root", "/tmp/does-not-exist", "--no-cache-frozen-text"]
        )
        self.assertFalse(args.cache_frozen_text)

    def test_synthetic_end_to_end(self):
        from pyhealth.datasets import create_sample_dataset

        runner = _runner()
        dataset = create_sample_dataset(
            samples=_samples(),
            input_schema={"labs": "stagenet_tensor"},
            output_schema={"label": "binary"},
            dataset_name="test_e2e_runner_synthetic",
        )
        with TemporaryDirectory() as tmp:
            args = Namespace(
                **{
                    **vars(
                        runner.parse_args(
                            [
                                "--ehr-root",
                                "/tmp/does-not-exist",
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
            self.assertTrue(Path(output_csv).is_file())
            metrics_path = Path(output_csv).parent / "metrics.json"
            self.assertTrue(metrics_path.is_file())
            metrics = json.loads(metrics_path.read_text())
            for key in (
                "config",
                "split",
                "epoch_train_seconds",
                "eval_seconds",
                "inference_seconds",
                "frozen_text_cache",
                "test_scores",
            ):
                self.assertIn(key, metrics)
            self.assertEqual(len(metrics["epoch_train_seconds"]), 1)
            self.assertGreaterEqual(metrics["inference_seconds"], 0.0)
            self.assertEqual(
                sorted(metrics["frozen_text_cache"]),
                ["hits", "misses", "size"],
            )
            self.assertTrue(torch.isfinite(torch.tensor(metrics["eval_seconds"])))


DEMO_ROOT = (
    Path(__file__).resolve().parents[2] / "test-resources" / "core" / "mimic4demo"
)

_EXPECTED_METRICS_KEYS = (
    "config",
    "split",
    "epoch_train_seconds",
    "eval_seconds",
    "inference_seconds",
    "frozen_text_cache",
    "test_scores",
)


def _demo_args(runner, tmp, model, task="labs", extra=()):
    # Seed 2: split_by_patient gives val=[1 event]/test=[2,2] (no empty
    # val/test batches). Seed 42 gives val=[0 events], and the RNN's
    # pack_padded_sequence cannot pack an all-empty batch.
    return Namespace(
        **vars(
            runner.parse_args(
                [
                    "--ehr-root",
                    str(DEMO_ROOT),
                    "--task",
                    task,
                    "--model",
                    model,
                    "--epochs",
                    "1",
                    "--batch-size",
                    "8",
                    "--embedding-dim",
                    "8",
                    "--hidden-dim",
                    "8",
                    "--device",
                    "cpu",
                    "--seed",
                    "2",
                    "--no-cache-frozen-text",
                    "--output-dir",
                    str(Path(tmp) / "out"),
                    "--cache-dir",
                    str(Path(tmp) / "cache"),
                    "--num-workers",
                    "0",
                    *extra,
                ]
            )
        )
    )


def _assert_demo_metrics(testcase, output_csv):
    testcase.assertTrue(Path(output_csv).is_file())
    metrics_path = Path(output_csv).parent / "metrics.json"
    testcase.assertTrue(metrics_path.is_file())
    metrics = json.loads(metrics_path.read_text())
    testcase.assertEqual(set(metrics.keys()), set(_EXPECTED_METRICS_KEYS))
    testcase.assertEqual(len(metrics["epoch_train_seconds"]), 1)
    for v in metrics["epoch_train_seconds"]:
        testcase.assertTrue(torch.isfinite(torch.tensor(float(v))).item())
    testcase.assertTrue(
        torch.isfinite(torch.tensor(float(metrics["eval_seconds"]))).item()
    )
    testcase.assertTrue(
        torch.isfinite(torch.tensor(float(metrics["inference_seconds"]))).item()
    )
    testcase.assertEqual(
        sorted(metrics["frozen_text_cache"]), ["hits", "misses", "size"]
    )
    testcase.assertGreater(metrics["split"]["train"], 0)
    for v in metrics["test_scores"].values():
        testcase.assertTrue(torch.isfinite(torch.tensor(float(v))).item())


class TestUnifiedE2ERunnerDemoData(unittest.TestCase):
    """Real-path runner test on the MIMIC-IV demo fixture (seconds, CPU).

    Drives ``run`` end to end on demo data for ``labs`` (no text model
    needed) with two backbones. The ``notes_labs`` case needs the
    Bio_ClinicalBERT weights; it is skipped unless they are already in the
    local HF cache -- tests never download.
    """

    @classmethod
    def setUpClass(cls):
        # One real set_task shared by both backbones: run() splits/trains
        # on these demo samples (seed 2 in _demo_args). Separated so the
        # slow base build happens once, not per backbone.
        import os

        from pyhealth.datasets import MIMIC4Dataset
        from pyhealth.tasks.multimodal_mimic4 import LabsMIMIC4

        cls._tmp = TemporaryDirectory()
        dataset = MIMIC4Dataset(
            ehr_root=str(DEMO_ROOT),
            ehr_tables=["labevents"],
            cache_dir=os.path.join(cls._tmp.name, "cache"),
        )
        cls.sample_dataset = dataset.set_task(LabsMIMIC4())

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_labs_mlp_and_rnn_finish_with_finite_scores(self):
        runner = _runner()
        for model in ("mlp", "rnn"):
            with self.subTest(model=model):
                with TemporaryDirectory() as tmp:
                    args = _demo_args(runner, tmp, model, task="labs")
                    output_csv = runner.run(args, sample_dataset=self.sample_dataset)
                    _assert_demo_metrics(self, output_csv)

    def test_notes_labs_skipped_without_cached_text_model(self):
        try:
            from transformers import AutoTokenizer

            AutoTokenizer.from_pretrained(
                "emilyalsentzer/Bio_ClinicalBERT", local_files_only=True
            )
        except Exception as exc:  # noqa: BLE001 -- any load failure means skip
            self.skipTest(f"text model not in local HF cache: {exc}")
        import gzip

        runner = _runner()
        with TemporaryDirectory() as tmp:
            note_dir = Path(tmp) / "notes" / "note"
            note_dir.mkdir(parents=True, exist_ok=True)
            import csv

            def _write(path, rows):
                with gzip.open(path, "wt", encoding="utf-8", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(
                        [
                            "subject_id",
                            "charttime",
                            "note_id",
                            "hadm_id",
                            "note_type",
                            "note_seq",
                            "storetime",
                            "text",
                        ]
                    )
                    w.writerows(rows)

            _write(
                note_dir / "discharge.csv.gz",
                [
                    (10008, "2151-10-05", 1, 20012, "discharge", 1,
                     "2151-10-25 12:00:00", "Chief Complaint:\n\nshortness of breath\n\n"),
                    (10008, "2151-12-15", 2, 20013, "discharge", 1,
                     "2151-12-20 12:00:00", "Chief Complaint:\n\nchest pain\n\n"),
                ],
            )
            _write(
                note_dir / "radiology.csv.gz",
                [
                    (10008, "2151-10-06 09:00:00", 3, 20012, "radiology", 1,
                     "2151-10-06 10:00:00", "Impression:\n\nno acute findings\n\n"),
                ],
            )
            args = _demo_args(
                runner,
                tmp,
                "rnn",
                task="notes_labs",
                extra=(
                    "--note-root", str(Path(tmp) / "notes"),
                    "--freeze-encoder",
                    # run() builds the base dataset + set_task inside, and
                    # set_task(num_workers=0) is invalid upstream
                    # (ZeroDivisionError); loaders are still fast enough.
                    "--num-workers", "1",
                ),
            )
            output_csv = runner.run(args)
            _assert_demo_metrics(self, output_csv)


if __name__ == "__main__":
    unittest.main()
