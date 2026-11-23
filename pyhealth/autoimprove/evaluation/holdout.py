"""The one evaluation on the held-out test split."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import numpy as np

from pyhealth.datasets import SampleDataset
from pyhealth.metrics import bootstrap_ci, clinical_summary, paired_bootstrap_diff
from pyhealth.trainer import get_metrics_fn

from . import guardrails, report
from .training import fit_predict


def evaluate_on_test(
    state,
    test_dataset: SampleDataset,
    n_boot: int = 1000,
    return_predictions: bool = False,
    subgroups: Mapping[str, Sequence[Any]] | None = None,
) -> dict[str, Any]:
    """Retrains the selected and the baseline configs and scores them on test.

    Backs :meth:`~pyhealth.autoimprove.AutoImprover.evaluate_on_test`; see there.
    ``state`` is the run's :class:`RunState`.

    Examples:
        >>> evaluate_on_test(improver.state, test_ds, n_boot=200)["best"]  # doctest: +SKIP
    """
    result = state.result()
    if result.best is None:
        raise ValueError("no successful trial to evaluate")
    configs = {"best": result.best}
    if (result.baseline is not None and result.baseline.error is None
            and result.baseline is not result.best):
        configs["baseline"] = result.baseline
    if result.top is not None and result.top is not result.best:
        configs["top"] = result.top
    held = report.held_back(state, result.best)
    if held:
        configs["without_flagged"] = replace(result.best, features=[
            s for s in result.best.features if s["name"] not in held])
    probs, out = {}, {"metric": state.metric}
    if held:
        out["held_back"] = sorted(held)
    for name, trial in configs.items():
        config = trial.config
        train, val = state._datasets(trial.features)
        test_ds = state._test_dataset(trial.features, test_dataset, train)
        fitted = fit_predict(
            config, state.space, train, val, [test_ds], state.metric, state.seed,
            state.device, state.code_timeout, state.store.log_dir(), f"test_{name}",
        )
        y_true, y_prob, patient_ids = fitted["predictions"][0]
        probs[name] = y_prob
        groups = np.asarray(patient_ids) if patient_ids else None
        if state.mode == "binary":
            out[name] = bootstrap_ci(
                y_true, y_prob, state.metric, groups=groups,
                n_boot=n_boot, seed=state.seed,
            )
        else:
            value = get_metrics_fn(state.mode)(y_true, y_prob, metrics=[state.metric])
            out[name] = {"estimate": float(value[state.metric])}
        out[name]["config"] = config.to_dict()
        out[name]["trial_id"] = trial.trial_id
        if state.mode == "binary":
            out[name]["clinical"] = clinical_summary(
                np.ravel(y_true), np.ravel(y_prob), groups=groups,
                subgroups=subgroups,
            )
    pairs = {"difference": ("best", "baseline"), "top_minus_best": ("top", "best")}
    for key, (a, b) in pairs.items():
        if a in probs and b in probs and state.mode == "binary":
            out[key] = paired_bootstrap_diff(
                y_true, probs[a], probs[b], state.metric,
                groups=groups, n_boot=n_boot, seed=state.seed,
            )
    on_test = guardrails.test_guardrails(state, test_dataset, y_true, probs)
    if on_test:
        out["guardrails"] = on_test
    state.store.save_result("test.json", out, default=float)
    if return_predictions:
        out["predictions"] = {"y_true": y_true, "patient_ids": patient_ids, **probs}
    return out
