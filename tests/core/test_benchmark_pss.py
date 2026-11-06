"""PSS next to summed RSS in benchmark_perf.

Summing RSS over forked workers counts every shared page once per process;
PSS divides it among the mappers, so the total is additive. Each harness
tracks both and records ``peak_pss_bytes`` alongside ``peak_rss_bytes``.
Where PSS is unavailable for any process in the tree the sample is not a
PSS total, so the column is left empty for the run (no RSS fallback, and
the reset runs inside the sampler lock).

Some harnesses do not import in every environment (``meds_reader`` is an
optional dependency; one legacy file names a task helper this tree no
longer exports). Those get a structural check of the added lines; the
importable ones also run the tracker.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import time
import unittest
from pathlib import Path

import psutil

HARNESSES = [
    "benchmark_workers_n.py",
    "benchmark_workers_n_drug_recommendation.py",
    "benchmark_workers_n_length_of_stay.py",
    "legacy_ver/benchmark_legacy_drug_rec.py",
    "legacy_ver/benchmark_legacy_los.py",
    "legacy_ver/benchmark_legacy_mortality.py",
    "meds_reader_ver/benchmark_meds_reader_drug_rec.py",
    "meds_reader_ver/benchmark_meds_reader_drug_rec_pyhealth_etl.py",
    "meds_reader_ver/benchmark_meds_reader_los.py",
    "meds_reader_ver/benchmark_meds_reader_los_pyhealth_etl.py",
    "meds_reader_ver/benchmark_meds_reader_mortality.py",
    "meds_reader_ver/benchmark_meds_reader_mortality_pyhealth_etl.py",
    "patient_exploration/benchmark_patient_access_legacy.py",
    "patient_exploration/benchmark_patient_access_meds_reader.py",
    "patient_exploration/benchmark_patient_access_pandas.py",
    "patient_exploration/benchmark_patient_access_pyhealth2.py",
]

PSS_MARKERS = (
    "peak_pss_bytes",
    "_total_pss_bytes",
    "memory_full_info().pss",
)


def _path(name):
    return (
        Path(__file__).resolve().parents[2] / "examples" / "benchmark_perf" / name
    )


def _load(name):
    import sys

    mod_name = "bench_pss_" + name.replace("/", "_").replace(".py", "")
    spec = importlib.util.spec_from_file_location(mod_name, _path(name))
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def _result_class(module):
    for value in vars(module).values():
        if (
            isinstance(value, type)
            and dataclasses.is_dataclass(value)
            and any(f.name == "peak_rss_bytes" for f in dataclasses.fields(value))
        ):
            return value
    raise AssertionError(f"no peak_rss_bytes dataclass in {module.__name__}")


class TestBenchmarkPss(unittest.TestCase):
    def test_all_harnesses_report_pss(self):
        self.assertEqual(len(HARNESSES), 16)
        exercised = 0
        for name in HARNESSES:
            with self.subTest(harness=name):
                src = _path(name).read_text()
                for marker in PSS_MARKERS:
                    self.assertIn(marker, src)
                try:
                    module = _load(name)
                except ImportError as exc:
                    self.skipTest(f"{name} does not import here: {exc}")
                cls = _result_class(module)
                self.assertIn(
                    "peak_pss_bytes", {f.name for f in dataclasses.fields(cls)}
                )
                tracker = module.PeakMemoryTracker(poll_interval_s=0.01)
                self.assertIsNone(tracker.peak_pss_bytes())
                sample = tracker._total_pss_bytes()
                self.assertTrue(sample is None or sample >= 0)
                tracker.start()
                time.sleep(0.05)
                tracker.stop()
                peak = tracker.peak_pss_bytes()
                self.assertTrue(peak is None or peak >= 0)
                self.assertGreaterEqual(tracker.peak_bytes(), 0)
                tracker.reset()
                self.assertIsNone(tracker.peak_pss_bytes())
                self.assertEqual(tracker.peak_bytes(), 0)
                exercised += 1
        self.assertGreater(exercised, 0)


class _FakeMemory:
    """Stand-in for the object ``memory_full_info()`` returns."""

    def __init__(self, pss):
        self.pss = pss


class _FakeProcess:
    """Minimal ``psutil.Process`` stand-in for PSS sampling tests."""

    def __init__(self, pss=None, error=None, children=()):
        self._pss = pss
        self._error = error
        self._children = list(children)

    def children(self, recursive=False):
        return list(self._children)

    def memory_full_info(self):
        if self._error is not None:
            raise self._error
        return _FakeMemory(self._pss)


class TestTotalPssProcessErrors(unittest.TestCase):
    """Denied processes drop the sample; exited ones are skipped."""

    def _tracker(self):
        for name in HARNESSES:
            try:
                module = _load(name)
            except ImportError:
                continue
            return module.PeakMemoryTracker(poll_interval_s=3600)
        self.skipTest("no harness imports here")

    def test_access_denied_drops_sample(self):
        tracker = self._tracker()
        tracker._proc = _FakeProcess(
            pss=50,
            children=[
                _FakeProcess(pss=100),
                _FakeProcess(error=psutil.AccessDenied()),
            ],
        )
        self.assertIsNone(tracker._total_pss_bytes())

    def test_no_such_process_skipped(self):
        tracker = self._tracker()
        tracker._proc = _FakeProcess(
            pss=50,
            children=[
                _FakeProcess(pss=100),
                _FakeProcess(error=psutil.NoSuchProcess(9999)),
            ],
        )
        self.assertEqual(tracker._total_pss_bytes(), 150)
