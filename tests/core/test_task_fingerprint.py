"""Property tests for deterministic task fingerprinting (issue #916).

Each test maps to a numbered finding in the audit. They assert *properties* --
equal configs produce equal keys, different configs produce different keys --
rather than recomputing the hashing expression, so the implementation can be
changed without rewriting the tests.
"""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

import numpy as np

from pyhealth.tasks.base_task import BaseTask
from pyhealth.tasks.fingerprint import (
    FINGERPRINT_VERSION,
    UnfingerprintableError,
    processors_fingerprint,
    slugify,
    task_cache_name,
    task_fingerprint,
    task_spec,
    write_task_metadata,
)

_FINGERPRINT_PATH = Path(importlib.util.find_spec("pyhealth.tasks.fingerprint").origin)


class _Task(BaseTask):
    """Minimal concrete task; subclasses override what each test needs."""

    task_name = "unit_test_task"
    input_schema = {"x": "sequence"}
    output_schema = {"y": "binary"}

    def __call__(self, patient):
        return []


class Readmission(_Task):
    task_name = "ReadmissionPredictionMIMIC3"

    def __init__(self, window=timedelta(days=15), exclude_minors=True, num_workers=4):
        super().__init__()
        self.window = window
        self.exclude_minors = exclude_minors
        self.num_workers = num_workers


class TestDeterminismAcrossProcesses(unittest.TestCase):
    """Findings 1 & 2: keys must not depend on PYTHONHASHSEED.

    The module is loaded standalone by file path so each subprocess stays fast
    (importing the pyhealth package would pull in torch).
    """

    def _fingerprint_with_seed(self, seed, body):
        snippet = (
            "import importlib.util\n"
            f"s = importlib.util.spec_from_file_location('fp', r'{_FINGERPRINT_PATH}')\n"
            "fp = importlib.util.module_from_spec(s); s.loader.exec_module(fp)\n"
            f"{body}\n"
            "print(fp.task_fingerprint(T()))"
        )
        result = subprocess.run(
            [sys.executable, "-c", snippet],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONHASHSEED": str(seed)},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout.strip()

    def test_set_valued_argument_is_seed_independent(self):
        body = (
            "class T:\n"
            "    task_name='T'; input_schema={}; output_schema={}\n"
            "    def __init__(self): self.codes={'A','B','C','D','E','F'}"
        )
        keys = {self._fingerprint_with_seed(s, body) for s in (0, 1, 7)}
        self.assertEqual(len(keys), 1, f"set iteration order leaked into the key: {keys}")

    def test_object_argument_is_seed_independent(self):
        body = (
            "class Cfg:\n"
            "    def __init__(self): self.a=1; self.b={'x','y'}\n"
            "class T:\n"
            "    task_name='T'; input_schema={}; output_schema={}\n"
            "    def __init__(self): self.cfg=Cfg()"
        )
        keys = {self._fingerprint_with_seed(s, body) for s in (0, 1, 7)}
        self.assertEqual(len(keys), 1, f"object identity leaked into the key: {keys}")


class TestNoCollisions(unittest.TestCase):
    def test_large_arrays_do_not_collide(self):
        """Finding 3: str() truncates arrays past 1000 elements; digests do not."""
        a, b = np.arange(2000), np.arange(2000)
        b[500] = -999

        class T(_Task):
            def __init__(self, bins):
                super().__init__()
                self.bins = bins

        self.assertNotEqual(task_fingerprint(T(a)), task_fingerprint(T(b)))
        self.assertEqual(task_fingerprint(T(a)), task_fingerprint(T(np.arange(2000))))

    def test_class_level_config_is_captured(self):
        """Finding 5: vars() cannot see class attributes."""

        class T30(_Task):
            task_name = "Shared"
            horizon_days = 30

        class T365(_Task):
            task_name = "Shared"
            horizon_days = 365

        self.assertNotEqual(task_fingerprint(T30()), task_fingerprint(T365()))

    def test_types_are_not_conflated(self):
        class T(_Task):
            def __init__(self, v):
                super().__init__()
                self.v = v

        self.assertNotEqual(task_fingerprint(T(1)), task_fingerprint(T("1")))
        self.assertNotEqual(task_fingerprint(T([1, 2])), task_fingerprint(T((1, 2))))
        self.assertNotEqual(task_fingerprint(T(0.0)), task_fingerprint(T(-0.0)))
        self.assertNotEqual(task_fingerprint(T(1)), task_fingerprint(T(True)))

    def test_version_bump_changes_key(self):
        """Finding 7: changing __call__ logic must be expressible."""

        class T1(_Task):
            version = "1"

        class T2(_Task):
            version = "2"

        self.assertNotEqual(task_fingerprint(T1()), task_fingerprint(T2()))

    def test_code_mapping_changes_key(self):
        """BaseTask.__init__ rewrites input_schema; that must be reflected."""
        plain = task_fingerprint(_Task())
        mapped = task_fingerprint(_Task(code_mapping={"x": ("ICD9CM", "CCSCM")}))
        other = task_fingerprint(_Task(code_mapping={"x": ("NDC", "ATC")}))
        self.assertEqual(len({plain, mapped, other}), 3)


class TestNoCrashes(unittest.TestCase):
    def test_mixed_key_dict_does_not_crash(self):
        """Finding 4: json.dumps(sort_keys=True) raises on mixed key types."""

        class T(_Task):
            def __init__(self, mapping):
                super().__init__()
                self.mapping = mapping

        self.assertEqual(len(task_fingerprint(T({1: "a", "b": 2, (3, 4): None}))), 64)

    def test_slots_task_does_not_crash(self):
        """Finding 6: vars() requires __dict__."""

        class T(BaseTask):
            __slots__ = ("w",)
            task_name = "slotted"
            input_schema = {}
            output_schema = {}

            def __init__(self):
                self.w = 7

            def __call__(self, patient):
                return []

        self.assertEqual(len(task_fingerprint(T())), 64)


class TestFailLoud(unittest.TestCase):
    def test_unfingerprintable_argument_raises_with_guidance(self):
        class Opaque:
            __slots__ = ()

        class T(_Task):
            def __init__(self):
                super().__init__()
                self.thing = Opaque()

        with self.assertRaises(UnfingerprintableError) as ctx:
            task_fingerprint(T())
        message = str(ctx.exception)
        self.assertIn("__pyhealth_fingerprint__", message)
        self.assertIn("fingerprint_exclude", message)

    def test_user_hook_is_honoured(self):
        class Mapper:
            __slots__ = ("src", "tgt")

            def __init__(self, src, tgt):
                self.src, self.tgt = src, tgt

            def __pyhealth_fingerprint__(self):
                return {"src": self.src, "tgt": self.tgt}

        class T(_Task):
            def __init__(self, mapper):
                super().__init__()
                self.mapper = mapper

        self.assertEqual(
            task_fingerprint(T(Mapper("ICD9CM", "CCSCM"))),
            task_fingerprint(T(Mapper("ICD9CM", "CCSCM"))),
        )
        self.assertNotEqual(
            task_fingerprint(T(Mapper("ICD9CM", "CCSCM"))),
            task_fingerprint(T(Mapper("NDC", "ATC"))),
        )


class TestInitArgs(unittest.TestCase):
    def test_explicit_default_equals_omitted_default(self):
        self.assertEqual(
            task_fingerprint(Readmission()),
            task_fingerprint(Readmission(window=timedelta(days=15))),
        )

    def test_changed_argument_changes_key(self):
        self.assertNotEqual(
            task_fingerprint(Readmission()),
            task_fingerprint(Readmission(window=timedelta(days=30))),
        )

    def test_non_semantic_argument_does_not_change_key(self):
        self.assertEqual(
            task_fingerprint(Readmission(num_workers=1)),
            task_fingerprint(Readmission(num_workers=16)),
        )

    def test_init_args_are_recorded_with_defaults_applied(self):
        recorded = getattr(Readmission(exclude_minors=False), "_pyhealth_init_args")
        self.assertIs(recorded["exclude_minors"], False)
        self.assertEqual(recorded["window"], timedelta(days=15))

    def test_fingerprint_exclude_is_honoured(self):
        class T(_Task):
            fingerprint_exclude = frozenset({"scratch"})

            def __init__(self, scratch, real):
                super().__init__()
                self.scratch = scratch
                self.real = real

        self.assertEqual(task_fingerprint(T("a", 1)), task_fingerprint(T("b", 1)))
        self.assertNotEqual(task_fingerprint(T("a", 1)), task_fingerprint(T("a", 2)))


class TestProcessors(unittest.TestCase):
    def test_processor_with_address_bearing_attribute(self):
        """Finding 8: SequenceProcessor._mapper is a CrossMap with a raw repr."""

        class CrossMapLike:
            def __init__(self, src, tgt):
                self.src, self.tgt = src, tgt

        class Proc:
            def __init__(self, mapping):
                self.code_vocab = {"<pad>": 0, "<unk>": 1}
                self._mapper = CrossMapLike(*mapping)

        a = processors_fingerprint({"conditions": Proc(("ICD9CM", "CCSCM"))}, None)
        b = processors_fingerprint({"conditions": Proc(("ICD9CM", "CCSCM"))}, None)
        c = processors_fingerprint({"conditions": Proc(("NDC", "ATC"))}, None)
        self.assertEqual(a, b, "identical processors must share a cache key")
        self.assertNotEqual(a, c)

    def test_mixed_key_vocabulary_does_not_crash(self):
        """code_vocab is typed Dict[Any, int]; mixed keys crash sort_keys=True."""

        class Proc:
            def __init__(self):
                self.code_vocab = {"<pad>": 0, 1: 1, ("a", "b"): 2}

        self.assertEqual(len(processors_fingerprint({"x": Proc()}, None)), 64)

    def test_none_processors_are_stable(self):
        self.assertEqual(processors_fingerprint(None, None), processors_fingerprint({}, {}))


class TestPathSafety(unittest.TestCase):
    def test_slash_in_task_name_is_neutralised(self):
        """Finding 9: BenchmarkEHRShot sets task_name = 'BenchmarkEHRShot/{task}'."""
        self.assertEqual(slugify("BenchmarkEHRShot/guo_los"), "BenchmarkEHRShot-guo_los")
        self.assertNotIn("/", slugify("a/b/../c"))

    def test_cache_name_is_a_single_path_component(self):
        class T(_Task):
            task_name = "Bench/mark: v2"

        self.assertEqual(len(Path(task_cache_name(T())).parts), 1)


class TestMetadataSidecar(unittest.TestCase):
    def test_sidecar_is_written_atomically_and_is_readable(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_task_metadata(Path(tmp), Readmission(window=timedelta(days=30)))
            payload = json.loads(Path(path).read_text())
            self.assertEqual(payload["spec"]["fingerprint_version"], FINGERPRINT_VERSION)
            self.assertEqual(payload["spec"]["task_name"], "ReadmissionPredictionMIMIC3")
            self.assertIn("30", json.dumps(payload["spec"]["init_args"]))
            self.assertFalse(list(Path(tmp).glob("*.tmp")), "temp file left behind")

    def test_spec_is_json_serialisable(self):
        json.dumps(task_spec(Readmission()))

    def test_second_call_does_not_refresh_created_at(self):
        """created_at must mean created, not last accessed."""
        with tempfile.TemporaryDirectory() as tmp:
            first = json.loads(write_task_metadata(Path(tmp), Readmission()).read_text())
            second = json.loads(write_task_metadata(Path(tmp), Readmission()).read_text())
            self.assertEqual(first["created_at"], second["created_at"])

    def test_concurrent_writers_leave_no_temp_files(self):
        """Runs outside the build lock, so parallel jobs reach it at once."""
        from concurrent.futures import ThreadPoolExecutor

        with tempfile.TemporaryDirectory() as tmp:
            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(
                    lambda _: write_task_metadata(Path(tmp), Readmission(), overwrite=True),
                    range(24),
                ))
            self.assertFalse(list(Path(tmp).glob("*.tmp")), "temp files left behind")
            payload = json.loads((Path(tmp) / "task_meta.json").read_text())
            self.assertEqual(payload["fingerprint"], task_fingerprint(Readmission()))

    def test_write_failure_is_not_fatal(self):
        """A diagnostic file must never abort a multi-hour build."""
        with tempfile.TemporaryDirectory() as tmp:
            unwritable = Path(tmp) / "does" / "not" / "exist"
            self.assertIsNotNone(write_task_metadata(unwritable, Readmission()))


class TestRealTasks(unittest.TestCase):
    """Smoke test against tasks actually shipped in the package."""

    def test_shipped_task_arguments_change_the_key(self):
        from pyhealth.tasks.readmission_prediction import ReadmissionPredictionMIMIC3

        t15 = ReadmissionPredictionMIMIC3()
        t30 = ReadmissionPredictionMIMIC3(window=timedelta(days=30))
        self.assertNotEqual(task_fingerprint(t15), task_fingerprint(t30))
        self.assertEqual(task_fingerprint(t15), task_fingerprint(t15))
        self.assertTrue(task_cache_name(t15).startswith("ReadmissionPredictionMIMIC3_"))

    def test_benchmark_ehrshot_name_is_slugified(self):
        from pyhealth.tasks.benchmark_ehrshot import BenchmarkEHRShot

        name = task_cache_name(BenchmarkEHRShot(task="guo_los"))
        self.assertNotIn("/", name)
        self.assertEqual(len(Path(name).parts), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
