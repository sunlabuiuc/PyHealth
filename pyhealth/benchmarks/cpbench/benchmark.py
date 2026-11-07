"""Evaluate the supplied experiments and return their results."""

import random
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from pyhealth.benchmarks import Benchmark, Evaluation, Result
from pyhealth.utils import preserve_rng_state

from . import adapters


class CPBench(Benchmark):
    """Run each evaluation's method and return one Result per evaluation.

    Examples:
        >>> benchmark = CPBench()
        >>> benchmark.evaluate([])
        {}
    """

    @property
    def name(self) -> str:
        return "cpbench"

    def validate_evaluations(self, evaluations: Sequence[Evaluation]) -> None:
        """Reject malformed specifications and unsupported methods before running."""
        super().validate_evaluations(evaluations)

        for evaluation in evaluations:
            validate_evaluation(**evaluation.params)

            method = evaluation.params["method"]["class"]
            if method not in adapters.ADAPTERS:
                raise ValueError(f"No CPBench adapter for {method!r}")

    def validate_results(
        self,
        evaluations: Sequence[Evaluation],
        results: Mapping[Evaluation, Result],
    ) -> None:
        """Check result correspondence and adapter return values.

        Metric names and values are checked by each adapter's scorer.
        """
        super().validate_results(evaluations, results)

        for evaluation, result in results.items():
            if not isinstance(result, Result):
                raise TypeError(f"{evaluation.id}: adapter must return a Result")

            if result.status not in ("success", "failed"):
                raise ValueError(
                    f"{evaluation.id}: invalid result status {result.status!r}"
                )

            if not isinstance(result.metrics, Mapping):
                raise TypeError(f"{evaluation.id}: result.metrics must be a mapping")

    def _evaluate(
        self,
        evaluations: Sequence[Evaluation],
        cache_dir: Path | None,
        output_dir: Path | None,
    ) -> dict[Evaluation, Result]:
        results = {}

        for evaluation in evaluations:
            try:
                with _seeded(evaluation.params["protocol"].get("seed", 0)):
                    result = adapters.run(evaluation, cache_dir)

                results[evaluation] = result

            except Exception as exc:  # noqa: BLE001 - Isolate evaluation failures.
                results[evaluation] = Result(
                    "failed",
                    {},
                    {
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                    },
                )

        return results


def validate_evaluation(*, dataset, task, model, method, metrics, protocol):
    """Validate every evaluation before adapter dispatch or any model/data work.

    dataset/task/model/method name importable constructors with optional params.
    model selects exactly one of training, checkpoint or predictions. method
    options are interpreted by its adapter. protocol declares named split parts,
    one run seed, batching/device and ordering/feedback rules. Metrics are explicit.

    Examples:
        >>> from examples.cpbench.synthetic_data import SyntheticDataset, SyntheticTask
        >>> from pyhealth.calib.predictionset import BaseConformal
        >>> from pyhealth.datasets import PatientSplit
        >>> from pyhealth.models import MLP
        >>> validate_evaluation(
        ...     dataset={"class": SyntheticDataset},
        ...     task={"class": SyntheticTask},
        ...     model={"class": MLP, "training": {"epochs": 1}},
        ...     method={"class": BaseConformal},
        ...     metrics=["miscoverage_overall_ps"],
        ...     protocol={"split": {
        ...         "class": PatientSplit,
        ...         "params": {"ratios": [0.4, 0.4, 0.2]},
        ...         "roles": ["train", "calibration", "test"],
        ...     }},
        ... )
    """
    _component(dataset, "dataset")
    _component(task, "task")
    _component(model, "model", {"training", "checkpoint", "predictions"})
    _component(method, "method", {"options"})

    sources = {"training", "checkpoint", "predictions"} & model.keys()
    if len(sources) != 1:
        raise ValueError(
            "model must select exactly one of training, checkpoint, predictions"
        )

    if "training" in model:
        _mapping(model["training"], "model.training")

    else:
        source = next(iter(sources))
        if not isinstance(model[source], str) or not model[source]:
            raise TypeError(f"model.{source} must be a nonempty path string")

    _mapping(method.get("options", {}), "method.options")

    if (
        not isinstance(metrics, (list, tuple))
        or not metrics
        or any(not isinstance(name, str) or not name for name in metrics)
    ):
        raise ValueError("metrics must be a nonempty list or tuple of names")

    _mapping(protocol, "protocol")
    _validate_protocol(**protocol)

    if "training" in model and protocol["split"]["roles"][0] != "train":
        raise ValueError(
            "training requires train as the first split role for processor fitting"
        )


def _mapping(value, name):
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")


def _component(value, name, extra_keys=()):
    _mapping(value, name)

    unknown = set(value) - {"class", "params", *extra_keys}
    if unknown:
        raise ValueError(f"unknown {name} keys: {sorted(unknown)}")

    if not callable(value.get("class")):
        raise TypeError(f'{name}["class"] must be an importable class or factory')

    _mapping(value.get("params", {}), f"{name}.params")


def _validate_protocol(
    *, split, seed=0, batch_size=32, device="cpu", order="dataset", feedback="none"
):
    _component(split, "protocol.split", {"roles"})

    roles = split.get("roles")
    if (
        not isinstance(roles, (list, tuple))
        or not roles
        or any(not isinstance(role, str) or not role for role in roles)
        or len(set(roles)) != len(roles)
    ):
        raise ValueError("split.roles must be a nonempty list of distinct role names")

    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("protocol.seed must be an integer in [0, 2**32)")

    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
    ):
        raise ValueError("protocol.batch_size must be a positive integer")

    if not isinstance(device, str):
        raise TypeError("protocol.device must be a string")

    if (
        not isinstance(order, str)
        or not order
        or not isinstance(feedback, str)
        or not feedback
    ):
        raise TypeError("protocol.order and protocol.feedback must be nonempty strings")


@contextmanager
def _seeded(seed):
    """Seed an evaluation and restore the caller's random states on exit."""
    with preserve_rng_state():
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        yield
