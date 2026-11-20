"""Evaluation turns: a proposer writes a new way to score predictions.

The proposal is a Python module that defines::

    def evaluate(y_true, y_prob, info) -> dict[str, float]

called once per cross-validation fold on that fold's held-out predictions,
usually with :mod:`pyhealth.metrics` (calibration, subgroup gaps, alert
yield, net benefit...). ``info`` holds per-sample columns the proposal asked
for (``needs``); the function never learns which attempt it is scoring.

Every finished trial keeps its out-of-fold predictions, so an evaluation is
run on all of them, old and new, without retraining. It becomes an **active
guardrail** automatically if it passes :func:`decide`: it runs, is
deterministic, is defined on every fold, points the way it says, varies
between trials, would not remove more than half of the candidates, and fewer
than ``max_guardrails`` are active. Otherwise it is kept for the report only
(``report_only``) or ``rejected``.

An active guardrail removes a trial from the candidates when the trial is
clearly worse than the baseline on it (:func:`clearly_worse`): the same
fold-by-fold comparison the main metric uses, in the guardrail's direction.
It never changes the main metric, and the baseline always passes.
"""

from __future__ import annotations

import inspect
import math
import os
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..utils.sandbox import ContractError, SandboxError, check_or_raise, run_sandboxed
from .records import Trial

ENTRY = "evaluate"

STATUSES = ("active", "report_only", "rejected")

EVALUATION_CONTRACT = """\
Write one Python module that defines

    def evaluate(y_true, y_prob, info):
        ...
        return {"<metric name>": float, ...}

- It is called once per cross-validation fold with that fold's held-out
  labels and predictions, as numpy arrays in the label processor's layout
  (binary: y_true and y_prob of shape (n, 1)), and `info`, a dict of numpy
  arrays with one value per sample: "patient" (an opaque id) and every column
  you list in "needs".
- Return the same metric names on every call, as floats (NaN if undefined).
- It must be deterministic and must not depend on anything but its inputs.
- Use pyhealth.metrics where it fits (listed below). Allowed imports: numpy,
  scipy, sklearn, pandas, pyhealth, math, statistics, collections,
  itertools, functools. No file, network or shell access."""


@dataclass
class EvaluationSpec:
    """A proposed evaluation: its code, metrics and the columns it needs.

    Args:
        name: Short identifier, e.g. ``"calibration"``.
        code: The module defining ``evaluate(y_true, y_prob, info)``.
        metrics: Metric name to direction, ``"higher"`` or ``"lower"`` (is
            better).
        needs: Sample columns passed in ``info``.
        rationale: Why it matters clinically.
        kind: e.g. ``"calibration"``, ``"subgroup"``, ``"utility"``.

    Examples:
        >>> spec = EvaluationSpec.from_dict(
        ...     {"name": "calibration", "metrics": {"ece": "lower"}},
        ...     "def evaluate(y_true, y_prob, info):\\n    return {'ece': 0.0}\\n")
        >>> spec.metrics
        {'ece': 'lower'}
    """

    name: str
    code: str
    metrics: dict[str, str]
    needs: list[str] = field(default_factory=list)
    rationale: str = ""
    kind: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any], code: str) -> EvaluationSpec:
        """Builds and checks a spec from a reply's JSON and code.

        Raises:
            ValueError: If the name, metrics or needs are malformed, or the
                code fails the static check.
        """
        name = str(data.get("name", "")).strip().lower()
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,39}", name):
            raise ValueError('"name" must be a short identifier, e.g. "calibration"')
        raw = data.get("metrics")
        if not isinstance(raw, dict) or not raw:
            raise ValueError('"metrics" must map each returned metric to "higher" '
                             'or "lower"')
        metrics = {}
        for metric, value in raw.items():
            direction = value.get("direction") if isinstance(value, dict) else value
            if direction not in ("higher", "lower"):
                raise ValueError(f'metric {metric!r} needs direction "higher" or '
                                 '"lower"')
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,39}", str(metric)):
                raise ValueError(f"metric name {metric!r} must be an identifier")
            metrics[str(metric)] = direction
        needs = data.get("needs") or []
        if not isinstance(needs, list) or not all(isinstance(n, str) for n in needs):
            raise ValueError('"needs" must be a list of column names')
        check_or_raise(code, ENTRY)
        return cls(name, code, metrics, list(needs),
                   str(data.get("rationale", "")), str(data.get("kind", "")))

    def keys(self) -> list[str]:
        """The guardrail keys, ``"<name>.<metric>"``."""
        return [f"{self.name}.{m}" for m in self.metrics]

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "metrics": dict(self.metrics),
                "needs": list(self.needs), "rationale": self.rationale,
                "kind": self.kind}


MENU = {
    "binary": [
        ("pyhealth.metrics", "binary_metrics_fn"),
        ("pyhealth.metrics.calibration", "ece_confidence_binary"),
        ("pyhealth.metrics", "calibration_bins"),
        ("pyhealth.metrics", "subgroup_metrics"),
        ("pyhealth.metrics", "fairness_metrics_fn"),
        ("pyhealth.metrics", "alert_rate_metrics"),
        ("pyhealth.metrics", "net_benefit"),
    ],
    "multiclass": [
        ("pyhealth.metrics", "multiclass_metrics_fn"),
        ("pyhealth.metrics.calibration", "ece_confidence_multiclass"),
        ("pyhealth.metrics.calibration", "ece_classwise"),
        ("pyhealth.metrics.calibration", "brier_top1"),
    ],
    "multilabel": [
        ("pyhealth.metrics", "multilabel_metrics_fn"),
        ("pyhealth.metrics", "ranking_metrics_fn"),
    ],
    "regression": [("pyhealth.metrics", "regression_metrics_fn")],
}
"""The :mod:`pyhealth.metrics` functions an evaluation prompt lists, by label."""


def metrics_menu(mode: str) -> str:
    """The pyhealth.metrics functions for a label kind, for a prompt.

    Each line is ``module.function(signature)`` and the first line of its
    docstring.

    Examples:
        >>> "binary_metrics_fn(" in metrics_menu("binary")
        True
    """
    import importlib

    lines = [f"pyhealth.metrics functions for {mode} labels:"]
    for module, name in MENU.get(mode, []):
        try:
            fn = getattr(importlib.import_module(module), name)
            signature = str(inspect.signature(fn))
            doc = (inspect.getdoc(fn) or "").strip().splitlines()
        except (ImportError, AttributeError, ValueError, TypeError):
            continue
        lines.append(f"- {module}.{name}{signature}: {doc[0] if doc else ''}")
    return "\n".join(lines)


def run_evaluations(
    spec: EvaluationSpec,
    y_true: np.ndarray,
    fold: np.ndarray,
    info: dict[str, np.ndarray],
    y_probs: Sequence[np.ndarray],
    checks: bool = True,
    timeout: float = 300.0,
    log_path: str | None = None,
) -> dict[str, Any]:
    """Runs an evaluation on several trials' predictions, fold by fold.

    Args:
        spec: The evaluation.
        y_true: Labels, one row per sample.
        fold: Fold id per sample (all 0 for a single split).
        info: The columns the spec needs, plus ``patient``.
        y_probs: One prediction array per trial, aligned with ``y_true``.
        checks: Also run the determinism and direction checks.

    Returns:
        ``{"values": [[{metric: value} per fold] per trial], "deterministic":
        bool, "perfect": {...}, "constant": {...}}``.

    Raises:
        SandboxError: If the code raised or timed out.
    """
    payload = {"y_true": y_true, "fold": fold, "info": info, "y_probs": list(y_probs),
               "metrics": list(spec.metrics), "checks": checks}
    return run_sandboxed(spec.code, f"{__name__}:run_evaluation",
                         payload, timeout=timeout, log_path=log_path,
                         label=f"evaluation_{spec.name}")


def run_evaluation(
    namespace: dict[str, Any], payload: dict[str, Any]
) -> dict[str, Any]:
    """The child side of :func:`run_evaluations`."""
    evaluate = namespace.get(ENTRY)
    if not callable(evaluate):
        raise ContractError(f"the module does not define {ENTRY}()")
    y_true, fold, info = payload["y_true"], payload["fold"], payload["info"]
    names = payload["metrics"]
    folds = sorted(set(np.asarray(fold).tolist()))

    def call(y_prob: np.ndarray, k: int) -> dict[str, float]:
        mask = np.asarray(fold) == k
        out = evaluate(np.array(y_true[mask]), np.array(y_prob[mask]),
                       {c: np.array(v[mask]) for c, v in info.items()})
        if not isinstance(out, dict):
            raise ContractError(
                f"evaluate must return a dict, not {type(out).__name__}")
        missing = [n for n in names if n not in out]
        if missing:
            raise ContractError(f"evaluate did not return {', '.join(missing)}")
        values = {}
        for n in names:
            try:
                values[n] = float(out[n])
            except (TypeError, ValueError):
                raise ContractError(f"{n} must be a number, not "
                                    f"{type(out[n]).__name__}") from None
        return values

    values = [[call(np.asarray(p), k) for k in folds] for p in payload["y_probs"]]
    result: dict[str, Any] = {"values": values, "folds": folds}
    if payload["checks"] and payload["y_probs"]:
        first = np.asarray(payload["y_probs"][0])
        result["deterministic"] = _same(call(first, folds[0]), values[0][0])
        perfect = perfect_predictions(np.asarray(y_true), first.shape)
        constant = np.broadcast_to(perfect.mean(axis=0), perfect.shape).copy()
        result["perfect"] = call(perfect, folds[0])
        result["constant"] = call(constant, folds[0])
    return result


def _same(a: dict[str, float], b: dict[str, float]) -> bool:
    return all((math.isnan(a[k]) and math.isnan(b[k])) or a[k] == b[k] for k in a)


def perfect_predictions(y_true: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Predictions that equal the labels, in the layout of ``shape``.

    Examples:
        >>> perfect_predictions(np.array([0, 2]), (2, 3)).tolist()
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        >>> perfect_predictions(np.array([[1.0], [0.0]]), (2, 1)).tolist()
        [[1.0], [0.0]]
    """
    y = np.asarray(y_true)
    if len(shape) == 2 and y.ndim == 1 and shape[1] > 1:  # class indices
        out = np.zeros(shape)
        out[np.arange(len(y)), y.astype(int)] = 1.0
        return out
    return y.astype(float).reshape(shape)


def clearly_worse(
    trial: Sequence[float], baseline: Sequence[float], direction: str, se: float = 1.0
) -> bool:
    """Whether a trial is clearly worse than the baseline on a guardrail.

    The fold-by-fold loss (baseline minus trial, in the guardrail's
    direction) has a lower bound, mean minus ``se`` standard errors, above 0.
    A fold where either value is undefined is left out; with fewer than two
    folds left nothing is clear.

    Examples:
        >>> clearly_worse([0.10, 0.12, 0.11], [0.05, 0.06, 0.05], "lower")
        True
        >>> clearly_worse([0.06, 0.04, 0.05], [0.05, 0.06, 0.05], "lower")
        False
    """
    pairs = [(t, b) for t, b in zip(trial, baseline)
             if not (math.isnan(t) or math.isnan(b))]
    if len(pairs) < 2:
        return False
    loss = np.array([b - t if direction == "higher" else t - b for t, b in pairs])
    bound = loss.mean() - se * loss.std(ddof=1) / math.sqrt(len(loss))
    return bool(bound > 0)


def decide(
    spec: EvaluationSpec,
    result: dict[str, Any],
    trial_ids: Sequence[int],
    baseline_id: int,
    candidates: Sequence[int],
    n_active: int,
    max_guardrails: int,
    se: float = 1.0,
    min_defined: float = 0.8,
) -> tuple[str, str]:
    """Whether a run evaluation becomes an active guardrail.

    Args:
        spec: The evaluation.
        result: From :func:`run_evaluations` on ``trial_ids``.
        trial_ids: The trials scored, in the order of ``result["values"]``.
        baseline_id: The baseline's trial id (it must be in ``trial_ids``).
        candidates: The current candidates (non-baseline trial ids).
        n_active: Guardrails already active.
        max_guardrails: Most active guardrails.
        se: Standard errors for :func:`clearly_worse`.
        min_defined: Share of trials that must be defined on every fold.

    Returns:
        ``(status, reason)``, status one of :data:`STATUSES`.
    """
    values = result["values"]
    index = {t: i for i, t in enumerate(trial_ids)}
    if baseline_id not in index:
        return "rejected", "the baseline has no stored predictions to compare with"
    if not result.get("deterministic", True):
        return "rejected", "two calls on the same predictions gave different values"
    names = list(spec.metrics)

    def defined(i: int) -> bool:
        return all(math.isfinite(v[n]) for v in values[i] for n in names)

    if not defined(index[baseline_id]):
        return "rejected", "undefined (NaN) on a fold of the baseline"
    share = sum(defined(i) for i in range(len(values))) / len(values)
    if share < min_defined:
        return "rejected", (f"defined on every fold for only {share:.0%} of trials "
                            f"(needs {min_defined:.0%})")
    for n, direction in spec.metrics.items():
        perfect, constant = result["perfect"][n], result["constant"][n]
        if not (math.isfinite(perfect) and math.isfinite(constant)):
            continue
        if (perfect < constant) if direction == "higher" else (perfect > constant):
            return "rejected", (f"{n}: perfect predictions score worse than a "
                                f"constant guess, so '{direction} is better' is "
                                "backwards")
    means = {n: [np.mean([v[n] for v in values[i]]) for i in range(len(values))
                 if defined(i)] for n in names}
    if all(np.ptp(m) < 1e-9 for m in means.values() if m):
        return "report_only", ("it gives every trial the same value, so it cannot "
                               "tell them apart")
    base = values[index[baseline_id]]
    removed = [t for t in candidates if t in index and any(
        clearly_worse([v[n] for v in values[index[t]]], [v[n] for v in base],
                      spec.metrics[n], se) for n in names)]
    if candidates and len(removed) > len(candidates) / 2:
        return "report_only", (f"it would remove {len(removed)} of {len(candidates)} "
                               "candidates, too sweeping to adopt automatically")
    if n_active >= max_guardrails:
        return "report_only", f"{max_guardrails} guardrails are already active"
    return "active", "passed every check"


def failing(
    spec: EvaluationSpec,
    scores: dict[int, list[dict[str, float]]],
    baseline_id: int,
    trial_id: int,
    se: float = 1.0,
) -> list[str]:
    """The guardrail keys a trial fails (empty if it passes or is the baseline).

    A trial with no scores for this evaluation (e.g. no stored predictions)
    fails every key, since it cannot be shown to pass.
    """
    if trial_id == baseline_id:
        return []
    if trial_id not in scores or baseline_id not in scores:
        return spec.keys()
    trial, base = scores[trial_id], scores[baseline_id]
    return [f"{spec.name}.{n}" for n, d in spec.metrics.items()
            if clearly_worse([v[n] for v in trial], [v[n] for v in base], d, se)]


def admissible_filter(
    evaluations: Sequence[dict[str, Any]], baseline_id: int | None, se: float = 1.0
) -> Callable[[Any], list[str]]:
    """A function from a trial to the active guardrails it fails.

    ``evaluations`` are the improver's records (``{"spec", "status",
    "scores"}``).
    """
    active = [e for e in evaluations if e["status"] == "active"]

    def fails(trial: Any) -> list[str]:
        if baseline_id is None or trial.trial_id == baseline_id:
            return []
        out: list[str] = []
        for e in active:
            out += failing(e["spec"], e["scores"], baseline_id, trial.trial_id, se)
        return out

    return fails


def evaluation_inputs(imp, spec, trials: Sequence[Trial]):
    oof = imp.oof_predictions(trials[0].trial_id)
    columns = imp._columns()
    info = {c: columns[c][oof["index"]] for c in ["patient", *spec.needs]}
    y_probs = [imp.oof_predictions(t.trial_id)["y_prob"] for t in trials]
    return oof["y_true"], oof["fold"], info, y_probs


def score_with_evaluations(imp, trial: Trial) -> None:
    """Scores a new trial with every adopted or report-only evaluation."""
    from .guardrails import run_evaluations

    kept = [e for e in imp.evaluations if e["status"] in ("active", "report_only")]
    oof = imp.oof_predictions(trial.trial_id)
    if not kept or oof is None or oof["split"] != "train":
        return
    for e in kept:
        spec = e["spec"]
        try:
            result = run_evaluations(
                spec, *evaluation_inputs(imp, spec, [trial]), checks=False,
                timeout=imp.eval_timeout,
                log_path=imp.store.log_path(f"evaluation_{spec.name}"))
        except SandboxError as err:
            trial.notes.append(f"evaluation {spec.name} failed on this trial: "
                               f"{err.brief}")
            continue
        e["scores"][trial.trial_id] = result["values"][0]


def save_evaluations(imp) -> None:
    if not imp.run_dir:
        return
    records = []
    fails = imp.guardrail_failures()
    for e in imp.evaluations:
        spec = e["spec"]
        imp.store.write_text(os.path.join("evaluations", f"{spec.name}.py"), spec.code)
        records.append({
            **spec.to_dict(), "status": e["status"], "reason": e["reason"],
            "cycle": e["cycle"], "agent": e["agent"],
            "scores": {str(k): v for k, v in e["scores"].items()},
            "fails": {str(t): [k for k in keys if k.startswith(spec.name + ".")]
                      for t, keys in fails.items()
                      if any(k.startswith(spec.name + ".") for k in keys)},
        })
    imp.store.write_json("evaluations.json", records, default=float)


def load_evaluations(imp) -> None:
    """Restores a resumed run's evaluations, with their code and scores."""
    records = imp.store.read_json("evaluations.json") or []
    for r in records:
        code = imp.store.read_text(f"evaluations/{r['name']}.py")
        if code is None:
            continue
        spec = EvaluationSpec(r["name"], code, dict(r["metrics"]),
                              list(r.get("needs", [])), r.get("rationale", ""),
                              r.get("kind", ""))
        imp.evaluations.append({
            "spec": spec, "status": r["status"], "reason": r.get("reason", ""),
            "cycle": r.get("cycle", -1), "agent": r.get("agent", ""),
            "scores": {int(k): v for k, v in r.get("scores", {}).items()},
        })


def test_guardrails(imp, test_dataset, y_true, probs) -> dict[str, Any]:
    """Each active guardrail's value on test, for every config scored there."""
    from ..search.analysis import sample_columns
    from .guardrails import run_evaluations

    active = [e for e in imp.evaluations if e["status"] == "active"]
    if not active:
        return {}
    columns = sample_columns(test_dataset)
    names = list(probs)
    out: dict[str, Any] = {}
    for e in active:
        spec = e["spec"]
        info = {c: columns[c] for c in ["patient", *spec.needs] if c in columns}
        try:
            res = run_evaluations(spec, np.asarray(y_true),
                                  np.zeros(len(y_true), dtype=np.int16), info,
                                  [np.asarray(probs[n]) for n in names],
                                  checks=False, timeout=imp.eval_timeout,
                                  log_path=imp.store.log_path(f"test_{spec.name}"))
        except SandboxError as err:
            out[spec.name] = {"error": err.brief}
            continue
        for metric, direction in spec.metrics.items():
            out[f"{spec.name}.{metric}"] = {
                "direction": direction,
                **{n: res["values"][i][0][metric] for i, n in enumerate(names)},
            }
    return out


def guardrail_summary(imp) -> list[dict[str, Any]]:
    """Each evaluation for the report: spec, status, reason, failing trials."""
    fails = imp.guardrail_failures()
    return [{
        **e["spec"].to_dict(), "status": e["status"], "reason": e["reason"],
        "fails": {t: [k for k in keys if k.startswith(e["spec"].name + ".")]
                  for t, keys in fails.items()
                  if any(k.startswith(e["spec"].name + ".") for k in keys)},
    } for e in imp.evaluations]
