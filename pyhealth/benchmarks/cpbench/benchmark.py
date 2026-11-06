"""Evaluate the supplied experiments and return their results."""

from collections.abc import Sequence
from pathlib import Path

from pyhealth.benchmarks import Benchmark, Evaluation, Result

from . import adapters
from .evaluation import validate_evaluation
from .preparation import seeded


class CPBench(Benchmark):
    """Run each evaluation's method and return one Result per evaluation."""

    @property
    def name(self) -> str:
        return "cpbench"

    def _evaluate(
        self,
        evaluations: Sequence[Evaluation],
        cache_dir: Path | None,
        output_dir: Path | None,
    ) -> dict[Evaluation, Result]:
        results = {}
        for evaluation in evaluations:
            try:
                validate_evaluation(**evaluation.params)
                with seeded(evaluation.params["protocol"].get("seed", 0)):
                    result = adapters.run(evaluation, cache_dir, output_dir)
                if not isinstance(result, Result):
                    raise TypeError("Adapter must return a Result")
                results[evaluation] = result
            except Exception as exc:
                results[evaluation] = Result(
                    "failed",
                    {},
                    {
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                    },
                )
        return results
