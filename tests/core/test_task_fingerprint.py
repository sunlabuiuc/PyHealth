"""Property tests for deterministic task fingerprinting (issue #916).

Each test maps to a numbered finding in the audit. They assert *properties* --
equal configs produce equal keys, different configs produce different keys --
rather than recomputing the hashing expression, so the implementation can be
changed without rewriting the tests.
"""

import collections
import copy
import dataclasses
import decimal
import enum
import functools
import importlib.util
import json
import os
import pickle
import re
import subprocess
import sys
import tempfile
import unittest
import uuid
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path, PurePosixPath
from unittest import mock

import numpy as np
import pandas as pd
import polars as pl
import torch

from pyhealth.tasks import fingerprint
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

    def test_spec_has_no_init_args(self):
        self.assertNotIn("init_args", task_spec(Readmission()))

    def test_base_task_has_no_init_wrapper(self):
        self.assertNotIn("__init_subclass__", vars(BaseTask))
        self.assertFalse(getattr(Readmission.__init__, "_pyhealth_recorded", False))

    def test_copies_and_pickles_keep_the_fingerprint(self):
        task = Readmission(window=timedelta(days=30))
        expected = task_fingerprint(task)
        clones = (
            copy.copy(task),
            copy.deepcopy(task),
            pickle.loads(pickle.dumps(task)),
        )
        for clone in clones:
            self.assertEqual(task_fingerprint(clone), expected)
        self.assertNotEqual(expected, task_fingerprint(Readmission()))

    def test_fingerprint_exclude_is_honoured(self):
        class T(_Task):
            fingerprint_exclude = frozenset({"scratch"})

            def __init__(self, scratch, real):
                super().__init__()
                self.scratch = scratch
                self.real = real

        self.assertEqual(task_fingerprint(T("a", 1)), task_fingerprint(T("b", 1)))
        self.assertNotEqual(task_fingerprint(T("a", 1)), task_fingerprint(T("a", 2)))


class _Holder(_Task):
    """Task whose only configuration is one arbitrary value."""

    def __init__(self, value):
        super().__init__()
        self.value = value


def _key(value):
    return task_fingerprint(_Holder(value))


def _module_function(x, missing=object()):
    return x


class _Scaler:
    def __init__(self, factor):
        self.factor = factor

    def scale(self, x):
        return x * self.factor

    @classmethod
    def build(cls):
        return cls(1)


class _FingerprintAssertions(unittest.TestCase):
    def assertSameKey(self, a, b):
        self.assertEqual(_key(a), _key(b))

    def assertDistinctKeys(self, *values):
        self.assertEqual(len({_key(v) for v in values}), len(values), values)


class TestCallables(_FingerprintAssertions):
    def test_bound_method_includes_its_instance(self):
        self.assertDistinctKeys(_Scaler(1).scale, _Scaler(2).scale)
        self.assertSameKey(_Scaler(1).scale, _Scaler(1).scale)
        # A method bound to a class, not to an instance.
        self.assertSameKey(_Scaler.build, _Scaler.build)
        self.assertDistinctKeys(_Scaler.build, _Scaler(1).scale)

    def test_builtin_bound_method_includes_its_owner(self):
        self.assertDistinctKeys({"a": 1}.get, {"a": 2}.get)
        self.assertSameKey({"a": 1}.get, {"a": 1}.get)
        self.assertDistinctKeys(np.add.reduce, np.multiply.reduce)

    def test_module_level_builtins_are_identified_by_name(self):
        self.assertDistinctKeys(len, sum, torch.sum, torch.special.expit)
        self.assertSameKey(len, len)

    def test_method_of_the_task_itself(self):
        class T(_Task):
            def __init__(self, n):
                super().__init__()
                self.n = n
                self.handler = self.handle
                self.scale = lambda x: x * self.n

            def handle(self):
                return self.n

        self.assertEqual(task_fingerprint(T(1)), task_fingerprint(T(1)))
        self.assertNotEqual(task_fingerprint(T(1)), task_fingerprint(T(2)))

    def test_closures_include_captured_values(self):
        def closure(n):
            def f(x):
                return x + n

            return f

        def default(n):
            def f(x, k=n):
                return x + k

            return f

        def kwdefault(n):
            def f(x, *, k=n):
                return x + k

            return f

        def attribute(n):
            def f(x):
                return x

            f.n = n
            return f

        for factory in (closure, default, kwdefault, attribute):
            with self.subTest(factory.__name__):
                self.assertSameKey(factory(1), factory(1))
                self.assertDistinctKeys(factory(1), factory(2))

    def test_wrapped_function_is_included(self):
        def wrap(func):
            @functools.wraps(func)
            def wrapper(*args):
                return func(*args)

            return wrapper

        def low(x):
            return x

        def high(x):
            return x + 1

        self.assertDistinctKeys(wrap(low), wrap(high), low, high)

    def test_recursive_and_unassigned_closures(self):
        def recursive(n):
            return recursive(n - 1) if n else 0

        def outer():
            def inner():
                return later  # deleted below, so the cell stays empty

            later = 1
            del later
            return inner

        self.assertSameKey(recursive, recursive)
        self.assertSameKey(outer(), outer())

    def test_closure_over_a_module(self):
        def make(module):
            def f(x):
                return module.dumps(x)

            return f

        self.assertDistinctKeys(make(json), make(pickle))

    def test_lambdas(self):
        low = lambda x: x
        high = lambda x: x + 1
        self.assertDistinctKeys(low, high)
        self.assertSameKey(low, low)
        pair = (lambda x: x, lambda x: x + 1)  # indistinguishable by source line
        with self.assertRaises(UnfingerprintableError):
            _key(pair[0])

    def test_function_without_source(self):
        namespace = {}
        exec("def f(x): return x", namespace)
        with self.assertRaises(UnfingerprintableError):
            _key(namespace["f"])

    def test_importable_function(self):
        self.assertSameKey(_module_function, _module_function)
        self.assertDistinctKeys(_module_function, _key)

    def test_sentinel_default_in_a_local_function(self):
        def f(x, missing=object()):
            return x

        self.assertSameKey(f, f)

    def test_partial(self):
        partial = functools.partial
        self.assertSameKey(partial(_key, 1), partial(_key, 1))
        self.assertDistinctKeys(
            partial(_key, 1), partial(_key, 2), partial(_key, value=1), partial(len, 1)
        )


class TestTables(_FingerprintAssertions):
    def test_pandas_column_labels(self):
        self.assertDistinctKeys(
            pd.DataFrame({"a": [1, 2]}),
            pd.DataFrame({"b": [1, 2]}),
            pd.DataFrame({0: [1, 2]}),
            pd.DataFrame({"0": [1, 2]}),
        )
        self.assertSameKey(pd.DataFrame({"a": [1]}), pd.DataFrame({"a": [1]}))

    def test_pandas_names_of_series_index_and_columns(self):
        self.assertDistinctKeys(
            pd.Series([1], name="a"),
            pd.Series([1], name="b"),
            pd.Series([1], name="a", index=pd.Index([0], name="i")),
            pd.Series([1], name="a", index=pd.Index([1], name="i")),
            pd.DataFrame({"a": [1]}),
            pd.DataFrame([[1]], columns=pd.Index(["a"], name="c")),
        )
        tuples = [("x", 1), ("y", 2)]
        self.assertDistinctKeys(
            pd.Series([1, 2], index=pd.MultiIndex.from_tuples(tuples)),
            pd.Series([1, 2], index=pd.MultiIndex.from_tuples(tuples, names=["a", "b"])),
            pd.Index([1, 2]),
            pd.Index([1, 3]),
        )

    def test_pandas_dtypes(self):
        self.assertDistinctKeys(
            pd.Series([1], dtype="int64"),
            pd.Series([1], dtype="int32"),
            pd.Series([1], dtype="Int64"),
            pd.Series([1], dtype=object),
            pd.Series(["1"], dtype=object),
            pd.Series([True], dtype=object),
        )
        self.assertDistinctKeys(
            pd.Series(pd.Categorical(["a"], ["a", "b"])),
            pd.Series(pd.Categorical(["a"], ["a", "c"])),
            pd.Series(pd.Categorical(["a"], ["a", "b"], ordered=True)),
        )

    def test_pandas_values(self):
        times = pd.to_datetime(["2020-01-01", None])
        for values in ([1, 2], ["a", "b"], [[1], [2]], [None, 1.5], times):
            with self.subTest(values=values):
                self.assertSameKey(pd.Series(values), pd.Series(values))
                self.assertDistinctKeys(pd.Series(values), pd.Series(values[::-1]))
        long = pd.Series([np.arange(2000)])
        other = pd.Series([np.arange(2000)])
        other[0][500] = -1
        self.assertDistinctKeys(long, other)

    def test_pandas_subclass_needs_a_hook(self):
        class Frame(pd.DataFrame):
            pass

        class Described(pd.DataFrame):
            def __pyhealth_fingerprint__(self):
                return pd.DataFrame(self)

        with self.assertRaises(UnfingerprintableError):
            _key(Frame({"a": [1]}))
        self.assertDistinctKeys(Described({"a": [1]}), Described({"a": [2]}))

    def test_polars_labels_dtypes_and_values(self):
        self.assertDistinctKeys(
            pl.DataFrame({"a": [1]}),
            pl.DataFrame({"b": [1]}),
            pl.DataFrame({"a": [2]}),
            pl.DataFrame({"a": [1]}, schema={"a": pl.Int32}),
            pl.DataFrame({"a": [[1]]}),
            pl.Series("a", [1]),
            pl.Series("b", [1]),
        )
        self.assertSameKey(pl.DataFrame({"a": [1]}), pl.DataFrame({"a": [1]}))
        many = [str(i) for i in range(100)]
        self.assertDistinctKeys(
            pl.Series("a", ["0"], dtype=pl.Enum(many)),
            pl.Series("a", ["0"], dtype=pl.Enum(many[:50] + ["x"] + many[51:])),
        )

    def test_table_values_stay_out_of_the_spec(self):
        spec = json.dumps(task_spec(_Holder(pd.DataFrame({"a": ["secret"]}))))
        self.assertNotIn("secret", spec)


class TestArrays(_FingerprintAssertions):
    def test_torch(self):
        self.assertDistinctKeys(
            torch.tensor([1.0]),
            torch.tensor([2.0]),
            torch.tensor(1.0),
            torch.tensor([1.0], dtype=torch.float64),
            torch.tensor([1.0], dtype=torch.bfloat16),
            torch.tensor([2.0], dtype=torch.bfloat16),
            torch.tensor([True]),
            torch.tensor([1j]),
            torch.tensor([1j]).conj(),
            torch.empty(0),
            torch.float32,
            torch.device("cpu"),
        )
        grid = torch.arange(6).reshape(2, 3)
        self.assertSameKey(grid.T, grid.T.contiguous())
        self.assertSameKey(torch.ones(1), torch.ones(1, requires_grad=True))

    def test_tensors_without_a_plain_buffer(self):
        for value in (torch.eye(2).to_sparse(), torch.empty(2, device="meta")):
            with self.subTest(layout=value.layout, device=value.device):
                with self.assertRaises(UnfingerprintableError):
                    _key(value)

    def test_subclasses_include_their_values(self):
        class Array(np.ndarray):
            pass

        class Tensor(torch.Tensor):
            pass

        base = np.array([1, 2])
        self.assertDistinctKeys(base, base.view(Array), (base + 1).view(Array))
        self.assertDistinctKeys(
            torch.ones(1),
            torch.ones(1).as_subclass(Tensor),
            torch.zeros(1).as_subclass(Tensor),
        )

    def test_numpy(self):
        def record(*names):
            return np.array([(1, 2)], dtype=[(name, "i4") for name in names])

        self.assertDistinctKeys(record("a", "b"), record("c", "d"))
        self.assertDistinctKeys(
            np.ma.array([1, 2], mask=[0, 1]), np.ma.array([1, 2], mask=[0, 0])
        )
        self.assertDistinctKeys(
            np.array(["a", 1], dtype=object), np.array(["a", "1"], dtype=object)
        )
        self.assertDistinctKeys(np.dtype("i4"), np.dtype("i8"))
        self.assertDistinctKeys(
            np.datetime64(5, "ns"), np.timedelta64(5, "ns"), np.datetime64(5, "us"), 5
        )
        self.assertDistinctKeys(np.longdouble(1), np.longdouble(2))
        self.assertSameKey(np.float32(0.5), np.float32(0.5))
        long_strings = np.array(["x" * 100], dtype=np.dtypes.StringDType())
        self.assertSameKey(long_strings, long_strings.copy())
        with tempfile.TemporaryDirectory() as tmp:
            np.save(Path(tmp) / "a.npy", np.arange(3))
            mapped = np.load(Path(tmp) / "a.npy", mmap_mode="r")
            self.assertSameKey(mapped, np.arange(3))
            del mapped


class _Colour(enum.Enum):
    RED = "red"


class _Level(enum.IntEnum):
    LOW = 1


class _Grade(enum.IntEnum):
    LOW = 1


class _Unit(str, enum.Enum):
    MG = "mg"


@dataclasses.dataclass
class _Window:
    days: int = 1


class _NamedWindow(_Window):
    def __init__(self, label):
        super().__init__()
        self.label = label


class _Codes(dict):
    def __init__(self, items, label):
        super().__init__(items)
        self.label = label


class TestValueTypes(_FingerprintAssertions):
    def test_enums(self):
        self.assertDistinctKeys(_Colour.RED, "red", _Unit.MG, "mg")
        self.assertDistinctKeys(_Level.LOW, _Grade.LOW, 1)
        self.assertSameKey(_Level.LOW, _Level(1))

    def test_dataclasses(self):
        self.assertSameKey(_Window(), _Window(1))
        self.assertDistinctKeys(_Window(1), _Window(2))
        self.assertDistinctKeys(_NamedWindow("a"), _NamedWindow("b"), _Window())

    def test_paths_patterns_and_scalars(self):
        self.assertDistinctKeys(PurePosixPath("a"), PurePosixPath("b"), "a")
        prefix = "a" * 300  # repr() of a pattern stops at 200 characters
        self.assertDistinctKeys(
            re.compile(prefix + "b"), re.compile(prefix + "c"), re.compile("a", re.I)
        )
        self.assertDistinctKeys(b"a", "a", 1j, range(3), range(4))
        self.assertDistinctKeys(
            decimal.Decimal("1.0"),
            datetime(2020, 1, 1),
            datetime(2020, 1, 1, tzinfo=timezone.utc),
            date(2020, 1, 1),
            time(1, 2),
            uuid.UUID(int=1),
            uuid.UUID(int=2),
        )
        self.assertDistinctKeys({1, 2}, {1, "2"}, frozenset({1, 2}), {3})
        self.assertSameKey({1, "a", (2, 3)}, {(2, 3), "a", 1})
        self.assertDistinctKeys(pd.Timedelta(1, "ns"), pd.Timedelta(2, "ns"))
        self.assertSameKey(pd.Timedelta(timedelta(days=1)), timedelta(days=1))

    def test_container_subclasses(self):
        a = collections.namedtuple("Window", "start")
        b = collections.namedtuple("Window2", "start")
        self.assertDistinctKeys(a(1), b(1), (1,))
        self.assertDistinctKeys(
            collections.defaultdict(int, a=1),
            collections.defaultdict(list, a=1),
            {"a": 1},
        )
        self.assertDistinctKeys(
            _Codes({"a": 1}, "x"), _Codes({"a": 1}, "y"), _Codes({"a": 2}, "x")
        )
        self.assertSameKey(_Codes({"a": 1}, "x"), _Codes({"a": 1}, "x"))

    def test_dict_that_is_its_own_namespace(self):
        class AttrDict(dict):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.__dict__ = self

        self.assertSameKey(AttrDict(a=1), AttrDict(a=1))
        self.assertDistinctKeys(AttrDict(a=1), AttrDict(a=2))

    def test_slots(self):
        class Single:
            __slots__ = "width"

            def __init__(self, width):
                self.width = width

        class _Private:
            __slots__ = ("__width",)

            def __init__(self, width):
                self.__width = width

        for cls in (Single, _Private):
            with self.subTest(cls.__name__):
                self.assertSameKey(cls(1), cls(1))
                self.assertDistinctKeys(cls(1), cls(2))

    def test_deep_nesting_raises(self):
        nested = []
        for _ in range(100):
            nested = [nested]
        with self.assertRaises(UnfingerprintableError):
            _key(nested)

    def test_circular_reference_raises(self):
        loop = []
        loop.append(loop)
        with self.assertRaises(UnfingerprintableError):
            _key(loop)


class TestModes(unittest.TestCase):
    """The two documented environment variables."""

    def _run(self, body, **env):
        snippet = (
            "import importlib.util\n"
            f"s = importlib.util.spec_from_file_location('fp', r'{_FINGERPRINT_PATH}')\n"
            "fp = importlib.util.module_from_spec(s); s.loader.exec_module(fp)\n"
            f"{body}"
        )
        return subprocess.run(
            [sys.executable, "-c", snippet],
            capture_output=True,
            text=True,
            env={**os.environ, **env},
        )

    _OPAQUE_TASK = (
        "class T:\n"
        "    task_name='T'; input_schema={}; output_schema={}\n"
        "    def __init__(self): self.x = iter([])\n"
        "print(len(fp.task_fingerprint(T())))"
    )

    def test_strict_variable(self):
        strict = self._run(self._OPAQUE_TASK)
        self.assertNotEqual(strict.returncode, 0)
        self.assertIn("UnfingerprintableError", strict.stderr)
        relaxed = self._run(self._OPAQUE_TASK, PYHEALTH_FINGERPRINT_STRICT="0")
        self.assertEqual(relaxed.returncode, 0, relaxed.stderr)
        self.assertEqual(relaxed.stdout.strip(), "64")
        self.assertIn("unstable repr()", relaxed.stderr)

    def test_source_variable(self):
        body = "print(fp._HASH_SOURCE)"
        self.assertEqual(self._run(body).stdout.strip(), "False")
        enabled = self._run(body, PYHEALTH_FINGERPRINT_SOURCE="1")
        self.assertEqual(enabled.stdout.strip(), "True")

    def test_relaxed_mode_warns_and_returns_a_key(self):
        namespace = {}
        exec("def f(x): return x", namespace)
        for value in (iter([]), namespace["f"], torch.eye(2).to_sparse()):
            with self.subTest(type=type(value).__name__):
                with self.assertRaises(UnfingerprintableError):
                    _key(value)
                with (
                    mock.patch.object(fingerprint, "_STRICT", False),
                    self.assertLogs(fingerprint.logger, "WARNING"),
                ):
                    self.assertEqual(len(_key(value)), 64)

    def _variant(self, call):
        # Same qualified name for every variant, so only the source differs.
        return type("Variant", (_Task,), {"__call__": call})()

    def test_source_mode_tracks_call_edits_only_when_enabled(self):
        def call(self, patient):
            return [1]

        original = self._variant(call)

        def call(self, patient):
            return [2]

        edited = self._variant(call)

        def call(self, patient):
            """Docstring."""
            # comment
            return [1]

        cosmetic = self._variant(call)

        self.assertEqual(task_fingerprint(original), task_fingerprint(edited))
        keys = [
            task_fingerprint(t, include_source=True)
            for t in (original, edited, cosmetic)
        ]
        self.assertNotEqual(keys[0], keys[1])
        self.assertEqual(keys[0], keys[2])
        with mock.patch.object(fingerprint, "_HASH_SOURCE", True):
            self.assertEqual(task_fingerprint(original), keys[0])

    def test_source_mode_warns_without_source(self):
        namespace = {}
        exec("def call(self, patient): return []", namespace)
        with self.assertLogs(fingerprint.logger, "WARNING"):
            task_fingerprint(self._variant(namespace["call"]), include_source=True)


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
            self.assertIn("30", json.dumps(payload["spec"]["instance_state"]))
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

    def test_normalised_init_args_share_a_cache_key(self):
        # Tasks normalise some arguments on purpose so that equivalent spellings
        # share a cache; the fingerprint must not undo that (review on #1207).
        from pyhealth.tasks.lab_nowcasting import LabNowcasting, LabNowcastingMIMIC3
        from pyhealth.tasks.seizure_detection_chbmit import SeizureDetectionCHBMIT

        self.assertEqual(
            task_fingerprint(SeizureDetectionCHBMIT(subjects=["chb01"])),
            task_fingerprint(SeizureDetectionCHBMIT(subjects=["1"])),
        )
        for cls in (LabNowcasting, LabNowcastingMIMIC3):
            self.assertEqual(
                task_fingerprint(cls()),
                task_fingerprint(cls(event_tables=["labevents"])),
            )
        # Normalisation must not merge configurations that really differ.
        self.assertNotEqual(
            task_fingerprint(SeizureDetectionCHBMIT(subjects=["chb01"])),
            task_fingerprint(SeizureDetectionCHBMIT(subjects=["chb02"])),
        )

    def test_memoized_lookups_do_not_change_the_key(self):
        from pyhealth.tasks import MortalityPredictionMIMIC4
        from pyhealth.tasks.ehr_text_serialization import (
            CodeDescriber,
            EHRTextSerializationTask,
        )

        def task(**kwargs):
            return EHRTextSerializationTask(
                MortalityPredictionMIMIC4(), describer=CodeDescriber(**kwargs)
            )

        def lookup_in(table):
            def lookup(code, vocabularies):
                return table.get(code)

            return lookup

        used = task(lookup=lookup_in({"4019": "Hypertension"}))
        before = task_fingerprint(used)
        used.describer("4019")
        self.assertEqual(task_fingerprint(used), before)
        self.assertNotEqual(
            task_fingerprint(task(lookup=lookup_in({"4019": "Other"}))), before
        )
        self.assertNotEqual(
            task_fingerprint(task(name="a")), task_fingerprint(task(name="b"))
        )

    def test_wrapped_task_version_changes_the_key(self):
        from pyhealth.tasks import MortalityPredictionMIMIC4
        from pyhealth.tasks.ehr_text_serialization import EHRTextSerializationTask

        class Revised(MortalityPredictionMIMIC4):
            version = "2"

        # Same qualified name, so that only the version differs.
        Revised.__module__ = MortalityPredictionMIMIC4.__module__
        Revised.__qualname__ = MortalityPredictionMIMIC4.__qualname__
        self.assertNotEqual(
            task_fingerprint(EHRTextSerializationTask(MortalityPredictionMIMIC4())),
            task_fingerprint(EHRTextSerializationTask(Revised())),
        )

    def test_benchmark_ehrshot_name_is_slugified(self):
        from pyhealth.tasks.benchmark_ehrshot import BenchmarkEHRShot

        name = task_cache_name(BenchmarkEHRShot(task="guo_los"))
        self.assertNotIn("/", name)
        self.assertEqual(len(Path(name).parts), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
