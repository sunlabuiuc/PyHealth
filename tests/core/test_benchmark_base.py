# Author: Felipe Amaral Bonchristiano
# NetID: felipea5
# Description: Tests for the pyhealth.benchmarks base abstractions.

import json
import types
import unittest
from collections import OrderedDict

from pyhealth.benchmarks import Benchmark, Evaluation, Result


def _module_function():
    return None


class _Outer:
    class Inner:
        pass


# Named differently from the variable, so its path doesn't resolve.
_Alias = type("_NotBound", (), {})

# Its name is rebound below, so its path resolves to a different class.
_Shadowed = type("_Shadowed", (), {})
_OldShadowed = _Shadowed
_Shadowed = type("_Shadowed", (), {})


class _CountingBenchmark(Benchmark):
    """Returns one success per evaluation and counts ``_evaluate`` calls."""

    def __init__(self, results=None):
        self.calls = 0
        self.results = results

    @property
    def name(self):
        return "counting"

    def _evaluate(self, evaluations, cache_dir, output_dir):
        self.calls += 1
        if self.results is not None:
            return self.results
        return {ev: Result("success", {"x": ev.params["x"]}) for ev in evaluations}


class TestEvaluation(unittest.TestCase):
    def test_id_independent_of_key_order(self):
        a = Evaluation({"a": 1, "b": {"c": 2, "d": 3}})
        b = Evaluation({"b": {"d": 3, "c": 2}, "a": 1})
        self.assertEqual(a.id, b.id)
        self.assertEqual(a, b)

    def test_id_changes_with_any_value(self):
        base = {"a": 1, "b": [1, 2], "c": {"d": "x"}, "e": json.JSONDecoder}
        variants = [
            {**base, "a": 2},
            {**base, "a": 1.0},
            {**base, "b": [1, 3]},
            {**base, "c": {"d": "y"}},
            {**base, "e": json.JSONEncoder},
        ]
        ids = {Evaluation(base).id} | {Evaluation(v).id for v in variants}
        self.assertEqual(len(ids), len(variants) + 1)

    def test_class_serializes_to_defining_module(self):
        ev = Evaluation({"cls": json.JSONDecoder})
        self.assertEqual(
            ev.to_dict(), {"cls": {"__type__": "json.decoder.JSONDecoder"}}
        )

    def test_function_serializes_by_path(self):
        ev = Evaluation({"fn": json.dumps})
        self.assertEqual(ev.to_dict(), {"fn": {"__type__": "json.dumps"}})

    def test_rejects_arbitrary_object(self):
        with self.assertRaises(TypeError):
            Evaluation({"x": object()})

    def test_rejects_lambda(self):
        with self.assertRaises(TypeError):
            Evaluation({"fn": lambda: None})

    def test_rejects_local_class(self):
        class Local:
            pass

        with self.assertRaises(TypeError):
            Evaluation({"cls": Local})

    def test_rejects_main_class(self):
        cls = type("Script", (), {"__module__": "__main__"})
        with self.assertRaisesRegex(TypeError, "importable module"):
            Evaluation({"cls": cls})

    def test_rejects_type_alias(self):
        with self.assertRaises(TypeError):
            Evaluation({"cls": _Alias})

    def test_rejects_rebound_class(self):
        with self.assertRaises(TypeError):
            Evaluation({"cls": _OldShadowed})
        Evaluation({"cls": _Shadowed})

    def test_rejects_unresolvable_builtin_type(self):
        for cls in [type(None), types.FunctionType]:
            with self.subTest(cls=cls), self.assertRaises(TypeError):
                Evaluation({"cls": cls})

    def test_rejects_nan(self):
        with self.assertRaises(ValueError):
            Evaluation({"x": float("nan")})

    def test_golden_id(self):
        ev = Evaluation(
            {
                "cls": OrderedDict,
                "n": 3,
                "lr": 0.5,
                "flag": True,
                "none": None,
                "name": "x",
                "seq": [1, 2.5, "a"],
                "nested": {"b": 1, "a": (True, None)},
            }
        )
        self.assertEqual(
            ev.id,
            "d3322d1ed849f5b773ab44fc80198a64a0d6533ef515c7006577abd3a31fd613",
        )

    def test_dedupes_in_set_and_dict(self):
        a = Evaluation({"x": 1, "fn": _module_function})
        b = Evaluation({"fn": _module_function, "x": 1})
        self.assertIsNot(a, b)
        self.assertEqual(len({a, b}), 1)
        self.assertEqual(len({a: 1, b: 2}), 1)


class TestEvaluationFromDict(unittest.TestCase):
    def test_round_trip_plain_params(self):
        ev = Evaluation({"a": 1, "b": [1.5, "x", None], "c": {"d": True}})
        self.assertEqual(Evaluation.from_dict(ev.to_dict()), ev)

    def test_round_trip_types(self):
        ev = Evaluation(
            {
                "cls": json.JSONDecoder,
                "fn": _module_function,
                "nested": _Outer.Inner,
                "seq": [OrderedDict, {"fn": json.dumps}],
            }
        )
        rebuilt = Evaluation.from_dict(ev.to_dict())
        self.assertEqual(rebuilt, ev)
        self.assertIs(rebuilt.params["nested"], _Outer.Inner)
        self.assertIs(rebuilt.params["fn"], _module_function)

    def test_reexport_path_uses_defining_module(self):
        ev = Evaluation.from_dict({"c": {"__type__": "json.JSONDecoder"}})
        self.assertEqual(ev, Evaluation({"c": json.decoder.JSONDecoder}))
        self.assertEqual(
            ev.to_dict(), {"c": {"__type__": "json.decoder.JSONDecoder"}}
        )

    def test_unresolvable_path_raises(self):
        for path in ["json.NoSuchThing", "no_such_module_xyz.Thing", ""]:
            with self.subTest(path=path), self.assertRaises(ValueError):
                Evaluation.from_dict({"c": {"__type__": path}})

    def test_non_type_target_raises(self):
        with self.assertRaisesRegex(ValueError, "math.pi"):
            Evaluation.from_dict({"c": {"__type__": "math.pi"}})

    def test_type_with_extra_keys_raises(self):
        with self.assertRaisesRegex(ValueError, "json.JSONDecoder"):
            Evaluation.from_dict({"c": {"__type__": "json.JSONDecoder", "extra": 1}})

    def test_type_key_in_params_raises(self):
        for params in [
            {"__type__": "x"},
            {"a": {"__type__": "json.JSONDecoder"}},
            {"a": [{"b": {"__type__": 1}}]},
        ]:
            with self.subTest(params=params), self.assertRaises(ValueError):
                Evaluation(params)

    def test_non_str_key_raises(self):
        for params in [
            {"p": {1: "a"}},
            {"p": [{None: "a"}]},
            {"p": {"q": {True: "a"}}},
            {"p": {1.5: "a"}},
        ]:
            with self.subTest(params=params), self.assertRaises(TypeError):
                Evaluation(params)


class TestBenchmark(unittest.TestCase):
    def test_evaluate_returns_results(self):
        evs = [Evaluation({"x": 1}), Evaluation({"x": 2})]
        results = _CountingBenchmark().evaluate(evs)
        self.assertEqual(results[evs[1]].metrics, {"x": 2})

    def test_duplicates_raise_before_evaluate(self):
        bench = _CountingBenchmark()
        evs = [Evaluation({"x": 1}), Evaluation({"x": 2}), Evaluation({"x": 1})]
        with self.assertRaises(ValueError):
            bench.evaluate(evs)
        self.assertEqual(bench.calls, 0)

    def test_missing_results_raise(self):
        evs = [Evaluation({"x": 1}), Evaluation({"x": 2})]
        bench = _CountingBenchmark(results={evs[0]: Result("success", {})})
        with self.assertRaisesRegex(ValueError, "missing"):
            bench.evaluate(evs)

    def test_extra_results_raise(self):
        evs = [Evaluation({"x": 1})]
        extra = Evaluation({"x": 2})
        bench = _CountingBenchmark(
            results={
                evs[0]: Result("success", {}),
                extra: Result("success", {}),
            }
        )
        with self.assertRaisesRegex(ValueError, "unexpected"):
            bench.evaluate(evs)

    def test_mixed_success_and_failure_returned(self):
        class Flaky(Benchmark):
            @property
            def name(self):
                return "flaky"

            def _evaluate(self, evaluations, cache_dir, output_dir):
                results = {}
                for ev in evaluations:
                    try:
                        if ev.params["x"] < 0:
                            raise ValueError("negative")
                        results[ev] = Result("success", {"x": ev.params["x"]})
                    except ValueError as exc:
                        results[ev] = Result(
                            "failed",
                            {},
                            {"error": str(exc), "error_type": type(exc).__name__},
                        )
                return results

        evs = [Evaluation({"x": 1}), Evaluation({"x": -1})]
        results = Flaky().evaluate(evs)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[evs[0]].status, "success")
        self.assertEqual(results[evs[1]].status, "failed")
        self.assertEqual(results[evs[1]].metadata["error_type"], "ValueError")

    def test_missing_name_cannot_instantiate(self):
        class NoName(Benchmark):
            def _evaluate(self, evaluations, cache_dir, output_dir):
                return {}

        with self.assertRaises(TypeError):
            NoName()

    def test_missing_evaluate_cannot_instantiate(self):
        class NoEvaluate(Benchmark):
            @property
            def name(self):
                return "noevaluate"

        with self.assertRaises(TypeError):
            NoEvaluate()


if __name__ == "__main__":
    unittest.main()
