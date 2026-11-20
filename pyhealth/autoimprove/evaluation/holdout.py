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
    imp,
    test_dataset: SampleDataset,
    n_boot: int = 1000,
    return_predictions: bool = False,
    subgroups: Mapping[str, Sequence[Any]] | None = None,
) -> dict[str, Any]:
    """Retrains the selected and the baseline configs and scores them on test.

    Run this once, after the search. For binary labels the result holds
    patient-level bootstrap intervals (``bootstrap_ci``) and a paired
    bootstrap interval for selected minus baseline
    (``paired_bootstrap_diff``). Other label types get the metric's value
    on test. When a simpler model was selected over the highest-scoring
    candidate, that candidate is scored too, so the price of the simpler
    model is measured on test as well. With a ``run_dir`` the result is
    saved to ``test.json``; it holds aggregate scores only.

    For binary labels each config also gets its clinical performance
    (:func:`~pyhealth.metrics.clinical_summary`): calibration, what
    flagging the top 5, 10 and 20% of risk would catch, net benefit, and
    results within each subgroup. Counts below 11 patients are
    suppressed.

    Args:
        test_dataset: The held-out test samples, never seen by the search.
        n_boot: Bootstrap resamples (of patients) for the intervals.
        return_predictions: Also return the test predictions, e.g. to
            compute other metrics without training again. They are not
            written to ``test.json``.
        subgroups: Attribute name to one label per test sample, in the
            dataset's order, e.g. ``{"sex": [...], "age band": [...]}``.

    Returns:
        ``{"best": {...}, "baseline": {...}, "difference": {...}}``, with
        ``best`` the selected config; ``baseline`` and ``difference`` are
        present when a baseline was given and not selected (``difference``
        only for binary labels), and ``top`` and ``top_minus_best`` when
        the highest-scoring candidate differs. With ``return_predictions``,
        ``"predictions"`` holds ``y_true``, ``patient_ids`` and each
        config's ``y_prob`` under its name.

    Review gate: engineered features with an open flag (no clinical
    rationale, an effect opposite to the expected direction, or a
    suspicious score) are held back until :meth:`approve`; the selection
    is then also scored without them (``without_flagged``).

    Examples:
        >>> improver.evaluate_on_test(test_ds)  # doctest: +SKIP
    """
    result = imp.result()
    if result.best is None:
        raise ValueError("no successful trial to evaluate")
    configs = {"best": result.best}
    if (result.baseline is not None and result.baseline.error is None
            and result.baseline is not result.best):
        configs["baseline"] = result.baseline
    if result.top is not None and result.top is not result.best:
        configs["top"] = result.top
    held = report.held_back(imp, result.best)
    if held:
        configs["without_flagged"] = replace(result.best, features=[
            s for s in result.best.features if s["name"] not in held])
    probs, out = {}, {"metric": imp.metric}
    if held:
        out["held_back"] = sorted(held)
    for name, trial in configs.items():
        config = trial.config
        train, val = imp._datasets(trial.features)
        test_ds = imp._test_dataset(trial.features, test_dataset, train)
        fitted = fit_predict(
            config, imp.space, train, val, [test_ds], imp.metric, imp.seed,
            imp.device, imp.code_timeout, imp.store.log_dir(), f"test_{name}",
        )
        y_true, y_prob, patient_ids = fitted["predictions"][0]
        probs[name] = y_prob
        groups = np.asarray(patient_ids) if patient_ids else None
        if imp.mode == "binary":
            out[name] = bootstrap_ci(
                y_true, y_prob, imp.metric, groups=groups,
                n_boot=n_boot, seed=imp.seed,
            )
        else:
            value = get_metrics_fn(imp.mode)(y_true, y_prob, metrics=[imp.metric])
            out[name] = {"estimate": float(value[imp.metric])}
        out[name]["config"] = config.to_dict()
        out[name]["trial_id"] = trial.trial_id
        if imp.mode == "binary":
            out[name]["clinical"] = clinical_summary(
                np.ravel(y_true), np.ravel(y_prob), groups=groups,
                subgroups=subgroups,
            )
    pairs = {"difference": ("best", "baseline"), "top_minus_best": ("top", "best")}
    for key, (a, b) in pairs.items():
        if a in probs and b in probs and imp.mode == "binary":
            out[key] = paired_bootstrap_diff(
                y_true, probs[a], probs[b], imp.metric,
                groups=groups, n_boot=n_boot, seed=imp.seed,
            )
    on_test = guardrails.test_guardrails(imp, test_dataset, y_true, probs)
    if on_test:
        out["guardrails"] = on_test
    imp.store.save_result("test.json", out, default=float)
    if return_predictions:
        out["predictions"] = {"y_true": y_true, "patient_ids": patient_ids, **probs}
    return out
