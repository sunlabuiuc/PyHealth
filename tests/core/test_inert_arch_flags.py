"""Flags a backbone ignores are warned about and recorded.

A launcher could appear to set something that never reached the model:
MLP has no ``dropout`` parameter and takes ``**kwargs`` (swallowed
silently); JambaEHR ignores ``--num-layers``; transformer/bottleneck/
ehrmamba/jamba ignore ``--hidden-dim``. The ignored flags are now logged at
startup and stored in ``run_config.json`` next to the resolved settings.
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
        "unified_e2e_runner_inert", path
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


class TestInertArchFlags(unittest.TestCase):
    def test_map_matches_what_each_backbone_consumes(self):
        runner = _runner()
        inert = {
            model: runner._inert_arch_flags(model)
            for model in runner._ARCH_FLAGS_USED
        }
        self.assertEqual(set(inert), {
            "mlp",
            "rnn",
            "transformer",
            "bottleneck_transformer",
            "ehrmamba",
            "jambaehr",
        })
        # Spot-check against the verified model wiring.
        self.assertIn("--dropout", inert["mlp"])
        self.assertNotIn("--dropout", inert["rnn"])
        self.assertIn("--num-layers", inert["jambaehr"])
        self.assertNotIn("--num-layers", inert["transformer"])
        self.assertIn("--hidden-dim", inert["transformer"])
        self.assertNotIn("--hidden-dim", inert["mlp"])
        # Every named flag is a real CLI option, so the map cannot drift
        # from the parser.
        args = runner.parse_args(["--ehr-root", "/tmp/does-not-exist"])
        for flag in runner._ARCH_FLAGS_ALL:
            self.assertTrue(
                hasattr(args, flag.lstrip("-").replace("-", "_")), flag
            )

    def test_build_model_warns_about_ignored_flags(self):
        from pyhealth.datasets import create_sample_dataset

        runner = _runner()
        dataset = create_sample_dataset(
            samples=_twenty_patient_samples(),
            input_schema={"labs": "stagenet_tensor"},
            output_schema={"label": "binary"},
            dataset_name="test_inert",
        )
        args = runner.parse_args(
            ["--ehr-root", "/tmp/does-not-exist", "--model", "mlp",
             "--dropout", "0.5"]
        )
        unified = runner._build_unified_embedding(args, dataset)
        with self.assertLogs(level="WARNING") as logs:
            runner._build_model(args, dataset, unified)
        self.assertTrue(any("--dropout" in line for line in logs.output))

    def test_build_model_stays_silent_at_defaults(self):
        # Every backbone ignores some flag, but a user who set nothing
        # gets no warning: only non-default inert values warn.
        from pyhealth.datasets import create_sample_dataset

        runner = _runner()
        dataset = create_sample_dataset(
            samples=_twenty_patient_samples(),
            input_schema={"labs": "stagenet_tensor"},
            output_schema={"label": "binary"},
            dataset_name="test_inert_silent",
        )
        for model in runner._ARCH_FLAGS_USED:
            args = runner.parse_args(
                ["--ehr-root", "/tmp/does-not-exist", "--model", model]
            )
            unified = runner._build_unified_embedding(args, dataset)
            with self.assertNoLogs(level="WARNING"):
                runner._build_model(args, dataset, unified)

    def test_run_config_records_inert_flags(self):
        from pyhealth.datasets import create_sample_dataset

        runner = _runner()
        dataset = create_sample_dataset(
            samples=_twenty_patient_samples(),
            input_schema={"labs": "stagenet_tensor"},
            output_schema={"label": "binary"},
            dataset_name="test_inert_run",
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
                                "0",
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
            run_config = json.loads(
                (Path(output_csv).parent / "run_config.json").read_text()
            )
            self.assertIn(
                "--dropout", run_config["config"]["inert_flags"]
            )
