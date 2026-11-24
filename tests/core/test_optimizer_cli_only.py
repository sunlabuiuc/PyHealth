"""Optimizer settings come from the CLI only.

A per-model branch used to give bottleneck_transformer max_grad_norm=0.5
and Adam eps=1e-6 while every other backbone got 1.0 and 1e-8, so a
six-backbone table that read as compute-matched was not, and the recorded
config stored the raw flag (null) instead of the value the optimizer used.
"""

from __future__ import annotations

import unittest
from pathlib import Path

MODELS = [
    "mlp",
    "rnn",
    "transformer",
    "bottleneck_transformer",
    "ehrmamba",
    "jambaehr",
]


def _runner():
    import importlib.util

    path = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "mortality_prediction"
        / "unified_embedding_e2e_mimic4.py"
    )
    spec = importlib.util.spec_from_file_location(
        "unified_e2e_runner_optimizer", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestOptimizerCliOnly(unittest.TestCase):
    def test_defaults_are_model_independent(self):
        runner = _runner()
        args = runner.parse_args(["--ehr-root", "/tmp/does-not-exist"])
        self.assertEqual(args.lr, 1e-4)
        self.assertEqual(args.adam_eps, 1e-8)
        self.assertEqual(args.max_grad_norm, 1.0)

    def test_every_backbone_resolves_the_same_settings(self):
        runner = _runner()
        seen = set()
        for model in MODELS:
            args = runner.parse_args(
                ["--ehr-root", "/tmp/does-not-exist", "--model", model]
            )
            lr, clip, params = runner._optimizer_settings(args)
            self.assertEqual((lr, clip), (1e-4, 1.0))
            self.assertEqual(params, {"lr": 1e-4, "eps": 1e-8})
            seen.add((lr, clip, params["eps"]))
        self.assertEqual(len(seen), 1)

    def test_explicit_flags_still_win(self):
        runner = _runner()
        args = runner.parse_args(
            [
                "--ehr-root",
                "/tmp/does-not-exist",
                "--model",
                "bottleneck_transformer",
                "--lr",
                "3e-4",
                "--adam-eps",
                "1e-7",
                "--max-grad-norm",
                "0.5",
            ]
        )
        lr, clip, params = runner._optimizer_settings(args)
        self.assertEqual((lr, clip, params["eps"]), (3e-4, 0.5, 1e-7))
