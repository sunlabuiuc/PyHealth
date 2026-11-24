"""run_config.json next to the metrics of a finished run.

``metrics.json`` stores the score but not the conditions, so a run that
changed one flag is indistinguishable from one that did not after the job
log is gone. The provenance half (git revision + source digest) keeps
"which code produced this result" answerable when the run executes from an
unpacked tarball with no git metadata.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


class TestWriteRunConfig(unittest.TestCase):
    def test_records_config_provenance_and_torch(self):
        from pyhealth.utils import write_run_config

        with TemporaryDirectory() as tmp:
            path = write_run_config(tmp, {"lr": 1e-4, "split": "by_patient"})
            self.assertTrue(path.endswith("run_config.json"))
            data = json.loads(Path(path).read_text())
        self.assertEqual(
            set(data), {"config", "git", "source_sha256", "torch"}
        )
        self.assertEqual(data["config"]["lr"], 0.0001)
        self.assertIn("commit", data["git"])
        self.assertEqual(len(data["source_sha256"]), 64)
        self.assertTrue(data["torch"])

    def test_non_jsonable_values_become_strings(self):
        from pyhealth.utils import write_run_config

        with TemporaryDirectory() as tmp:
            path = write_run_config(tmp, {"tags": ("a", "b"), "obj": object()})
            data = json.loads(Path(path).read_text())
        self.assertEqual(data["config"]["tags"], ["a", "b"])
        self.assertIsInstance(data["config"]["obj"], str)

    def test_no_partial_file_left_behind(self):
        from pyhealth.utils import write_run_config

        with TemporaryDirectory() as tmp:
            write_run_config(tmp, {"lr": 1e-4})
            leftovers = [
                p for p in Path(tmp).iterdir() if p.suffix.startswith(".tmp")
            ]
            self.assertEqual(leftovers, [])
            self.assertTrue((Path(tmp) / "run_config.json").is_file())
