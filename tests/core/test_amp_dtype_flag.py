"""--amp-dtype without --use-amp is a hard error, not a silent fp32 run.

The flag list says --amp-dtype "bf16" and never mentions --use-amp, so
following it literally ran fp32 while the recorded config claimed bf16.
Passing a dtype alone now fails parsing; omitting it defaults to bf16 for
the run that does pass --use-amp.
"""

from __future__ import annotations

import unittest
from pathlib import Path


def _runner():
    import importlib.util

    path = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "mortality_prediction"
        / "unified_embedding_e2e_mimic4.py"
    )
    spec = importlib.util.spec_from_file_location(
        "unified_e2e_runner_amp", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestAmpDtypeRequiresUseAmp(unittest.TestCase):
    def test_dtype_alone_is_an_error(self):
        runner = _runner()
        with self.assertRaises(SystemExit):
            runner.parse_args(
                ["--ehr-root", "/tmp/does-not-exist", "--amp-dtype", "fp16"]
            )

    def test_dtype_defaults_to_bf16(self):
        runner = _runner()
        args = runner.parse_args(["--ehr-root", "/tmp/does-not-exist"])
        self.assertEqual(args.amp_dtype, "bf16")
        self.assertFalse(args.use_amp)

    def test_dtype_with_use_amp_is_accepted(self):
        runner = _runner()
        args = runner.parse_args(
            [
                "--ehr-root",
                "/tmp/does-not-exist",
                "--use-amp",
                "--amp-dtype",
                "fp16",
            ]
        )
        self.assertTrue(args.use_amp)
        self.assertEqual(args.amp_dtype, "fp16")
