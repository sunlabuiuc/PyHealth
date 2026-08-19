"""Proof that a finished run records the conditions that produced the score.

``metrics_history.json`` stores PR-AUC but not whether the encoder was frozen,
which split ran, or which code produced it. Cluster jobs start from an unpacked
archive, so git is often empty; the SHA-256 of the package source still
identifies the code.

Repro::

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. \\
      python -m pytest tests/test_p2_run_config.py -q
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path


class TestP2RunConfig(unittest.TestCase):
    def test_write_run_config_records_resolved_fields(self):
        from pyhealth.utils import write_run_config

        with tempfile.TemporaryDirectory() as tmp:
            path = write_run_config(
                tmp, {"resolved_lr": 1e-4, "split_mode": "by_patient"}
            )
            data = json.loads(Path(path).read_text())
        self.assertEqual(data["config"]["resolved_lr"], 0.0001)
        self.assertIn("git", data)
        self.assertIn("source_sha256", data)
        self.assertEqual(len(data["source_sha256"]), 64)
        self.assertTrue(path.endswith("run_config.json"))
