"""Re-scoring the leading candidates on new folds and seeds, and pruning features."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ..search.space import _derived
from .cv import patient_folds, run_config_cv
from .records import Trial
from .selection import paired_gain


def _feature_groups(best: Trial, trials: Sequence[Trial]) -> list[tuple[int, list[str]]]:
    """The selected trial's features, grouped by the accepted trial that added them."""
    groups: dict[int, list[str]] = {}
    for spec in best.features:
        name = spec["name"]
        owner = next((t.trial_id for t in trials if name in t.added and t.accepted),
                     -1 - len(groups))
        groups.setdefault(owner, []).append(name)
    return list(groups.items())


def confirm(imp) -> dict[str, Any]:
    """Re-scores the leading candidates on new folds and several seeds.

    The baseline (if any), the selection and the top ``confirm_top`` candidates
    by mean are scored again on folds drawn with a new seed, each
    averaged over ``confirm_seeds`` training seeds. The selection is then
    made on these scores, with the stricter bound ``confirm_se``; after
    :meth:`prune`, the pruned selection is the result.
    Candidates chosen among many on the same folds are partly lucky (the
    winner's curse); new folds and seeds remove part of that luck before
    the one test evaluation. The same training
    patients are reused, so only the test split is fully independent.

    Returns:
        ``{"seeds", "fold_seed", "scores": {trial_id: [fold scores]}}``,
        also saved to ``confirm.json``.

    Examples:
        >>> improver.confirm()  # doctest: +SKIP
    """
    result = imp.result()
    baseline = result.baseline
    reference = baseline if baseline is not None and baseline.error is None else None
    scored = [t for t in imp.trials if t.error is None and t.fold_scores]
    if not scored or (reference is not None and not reference.fold_scores):
        return {}
    n = len((reference or scored[0]).fold_scores)
    fails = imp._guardrail_fails()
    candidates = [t for t in scored if not t.stopped_early
                  and len(t.fold_scores) == n and t is not reference
                  and t.accepted is not False and not fails(t)]
    sign = 1 if imp.maximize else -1

    def clears(t: Trial) -> bool:
        if reference is None:  # nothing to compare with: every trial
            return True
        gain, se = paired_gain(t.fold_scores, reference.fold_scores, imp.maximize)
        return gain - (imp.promote_se or 1.0) * se > 0

    bound = [t for t in candidates if clears(t)]
    chosen = sorted(bound, key=lambda t: -sign * t.score)[: imp.confirm_top]
    if result.best is not None and result.best not in [*chosen, reference]:
        chosen.append(result.best)
    imp._confirm_folds = patient_folds(imp.train_dataset, n, imp.seed + 1)
    for t in [reference, *chosen] if reference is not None else chosen:
        imp.confirmed[t.trial_id] = confirm_scores(imp, t.config, t.features)
    out = {"seeds": [imp.seed + k for k in range(imp.confirm_seeds)],
           "fold_seed": imp.seed + 1,
           "scores": {str(k): v for k, v in imp.confirmed.items()}}
    imp.store.write_json("confirm.json", out)
    return out


def confirm_scores(imp, config, specs) -> list[float]:
    train, val = imp._datasets(specs)
    runs = [
        run_config_cv(config, imp.space, train, val, imp.metric,
                      imp._confirm_folds, (), imp.seed + k, imp.device,
                      timeout=imp.code_timeout, log_dir=imp.store.log_dir(),
                      label=f"confirm_seed{k}")["fold_scores"]
        for k in range(imp.confirm_seeds)
    ]
    return [float(v) for v in np.mean(np.array(runs), axis=0)]


def prune(imp) -> dict[str, Any]:
    """Drops accepted feature groups that the selected model doesn't need.

    On the confirmation folds and seeds, each feature group of the
    selected trial (the features one trial added together) is removed in
    turn. Its contribution is the score lost without it, with a paired
    SE: what the group bought (the ablation). The group whose removal
    costs least is dropped if that cost is under one SE, and the rest are
    checked again, until every remaining group pays for itself. A pruned
    config becomes a new trial and the result: the shortest feature list
    that keeps the selection's confirmed score within one SE per step.

    Returns:
        ``{"selected", "ablation": [{"features", "trial_id",
        "contribution", "se"}], "dropped": [[names]], "final"}``, also
        saved to ``prune.json``.

    Examples:
        >>> improver.prune()["dropped"]  # doctest: +SKIP
        [['exists_90d__med_orders_SABA']]
    """
    best = imp.result().best
    if best is None or not best.features or best.trial_id not in imp.confirmed:
        return {}
    groups = _feature_groups(best, imp.trials)
    current = list(best.features)
    scores = imp.confirmed[best.trial_id]
    ablation, dropped = [], []
    while groups:
        rows = []
        for trial_id, names in groups:
            without = [s for s in current if s["name"] not in names]
            lost = confirm_scores(imp, best.config, without)
            gain, se = paired_gain(scores, lost, imp.maximize)
            rows.append((gain - se, gain, se, trial_id, names, without, lost))
        if not ablation:
            ablation = [{"features": r[4], "trial_id": r[3], "contribution": r[1],
                         "se": r[2]} for r in rows]
        rows.sort(key=lambda r: r[0])
        margin, _, _, trial_id, names, without, lost = rows[0]
        if margin > 0:
            break
        dropped.append(names)
        groups = [g for g in groups if g[0] != trial_id]
        current, scores = without, lost
    out = {"selected": best.trial_id, "ablation": ablation, "dropped": dropped,
           "final": [s["name"] for s in current]}
    if dropped:
        pruned = imp._evaluate(
            _derived(best.config,
                     "the selection without features it does not need"),
            max((t.cycle for t in imp.trials), default=0), "pruned", "prune",
            features=current, early_stop=False,
        )
        if pruned.error is None:
            imp.confirmed[pruned.trial_id] = scores
            imp.pruned_id = pruned.trial_id
            out["pruned_trial"] = pruned.trial_id
    imp.store.write_json("prune.json", out)
    saved = imp.store.read_json("confirm.json")
    if saved is not None:
        saved["scores"] = {str(k): v for k, v in imp.confirmed.items()}
        imp.store.write_json("confirm.json", saved)
    return out
