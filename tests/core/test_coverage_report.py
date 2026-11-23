"""Tests for the coverage merge in tools/coverage_report.py.

PR runs execute only the selected tests and merge their coverage with the
baseline master's last full run saved; these check that merge's rules on
hand-built coverage data files.
"""

import importlib.util
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

try:
    from coverage import CoverageData
except ImportError:  # the test env installs it with pytest-cov
    CoverageData = None

_TOOL = Path(__file__).resolve().parents[2] / "tools" / "coverage_report.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("coverage_report", _TOOL)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@unittest.skipIf(CoverageData is None, "coverage is not installed")
class CoverageMergeTest(unittest.TestCase):
    def setUp(self):
        self.tool = _load_tool()
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root)
        for name in ("unchanged", "changed", "changed_unrun"):
            path = self.root / "pyhealth" / f"{name}.py"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("x = 1\n" * 5)
        # pyhealth/deleted.py exists only in the baseline's data

    def path(self, name: str) -> str:
        return str(self.root / "pyhealth" / f"{name}.py")

    def data(self, filename: str, lines: dict) -> Path:
        out = self.root / filename
        data = CoverageData(basename=str(out))
        data.add_lines({self.path(k): v for k, v in lines.items()})
        data.write()
        return out

    def merged(self, fresh_lines: dict = None) -> dict:
        baseline = self.data(".coverage.baseline", {
            "unchanged": [1, 2],
            "changed": [1, 2, 3],
            "changed_unrun": [1, 2],
            "deleted": [1],
        })
        fresh = (self.data(".coverage", fresh_lines) if fresh_lines is not None
                 else self.root / ".coverage")
        changed = {"pyhealth/changed.py", "pyhealth/changed_unrun.py",
                   "pyhealth/deleted.py"}
        out = self.root / ".coverage.merged"
        self.tool.merge(baseline, fresh, changed, out, root=self.root)
        result = CoverageData(basename=str(out))
        result.read()
        return {Path(f).stem: sorted(result.lines(f) or [])
                for f in result.measured_files()}

    def test_unchanged_file_takes_the_union(self):
        self.assertEqual(self.merged({"unchanged": [3]})["unchanged"], [1, 2, 3])

    def test_changed_file_takes_only_the_fresh_data(self):
        # line numbers moved, so the baseline's lines 2-3 no longer apply
        self.assertEqual(self.merged({"changed": [1]})["changed"], [1])

    def test_deleted_file_is_dropped(self):
        self.assertNotIn("deleted", self.merged({"unchanged": [3]}))

    def test_changed_file_no_test_ran_counts_as_uncovered(self):
        # still measured, so it counts in the total at 0%, as in a full run
        self.assertEqual(self.merged({"unchanged": [3]})["changed_unrun"], [])

    def test_no_tests_ran_keeps_the_baseline_for_unchanged_files(self):
        merged = self.merged(None)
        self.assertEqual(merged["unchanged"], [1, 2])
        self.assertEqual(merged["changed"], [])
        self.assertNotIn("deleted", merged)


if __name__ == "__main__":
    unittest.main()
