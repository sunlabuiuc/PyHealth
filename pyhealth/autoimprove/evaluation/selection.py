"""Which trial to deploy: paired comparisons with the baseline and the one-SE rule."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from ..search.space import PipelineConfig, _derived
from .records import Trial
from .training import patient_folds, run_config_cv


FOLD_TIE = 1e-4
"""Fold scores closer than this count as a tie, not as one fold being better."""


def _rule_text(promote_se: float | None, one_se: bool, compared: bool) -> str:
    """The selection rule in plain words.

    ``compared`` says whether trials are compared with a starting pipeline
    that ran.
    """
    if promote_se is None:
        return "The selection is the trial with the best score."
    simplest = (
        "the most interpretable model (lowest rung) whose score is within one "
        "paired standard error of the best"
    )
    if not compared:
        if not one_se:
            return ("There is no starting pipeline to compare with, so the "
                    "selection is the trial with the best mean score.")
        return ("There is no starting pipeline to compare with, so every trial "
                f"is a candidate. The selection is {simplest}: simpler models "
                "win ties.")
    bound = (
        f"Every trial is compared with the starting pipeline fold by fold. "
        f"A trial is a candidate if its gain is above 0 by more than "
        f"{promote_se:g} standard error(s), so a gain must be "
        f"consistent across folds, not only high on average."
    )
    if not one_se:
        return bound + " The selection is the candidate with the highest bound."
    return bound + (
        " Among the candidates, the selection is the most interpretable "
        "model (lowest rung) whose score is within one paired standard error "
        "of the best candidate's: simpler models win ties."
    )


def paired_gain(
    candidate: Sequence[float], incumbent: Sequence[float], maximize: bool
) -> tuple[float, float]:
    """Mean and standard error of the fold-by-fold gain of one config over another.

    Examples:
        >>> paired_gain([0.5, 0.6, 0.7], [0.4, 0.5, 0.6], maximize=True)
        (0.1, 0.0)
    """
    diff = np.asarray(candidate, dtype=float) - np.asarray(incumbent, dtype=float)
    if not maximize:
        diff = -diff
    se = float(diff.std(ddof=1) / math.sqrt(len(diff))) if len(diff) > 1 else math.inf
    return round(float(diff.mean()), 12), round(se, 12)


def select_trial(
    trials: Sequence[Trial], maximize: bool, promote_se: float | None = None
) -> Trial | None:
    """The trial a search would deploy.

    Without ``promote_se``: the best score. With it, every trial is compared
    with the baseline trial (``agent == "baseline"``) on the shared folds,
    and the trial with the highest lower bound on its gain,
    ``mean - promote_se * se``, is selected if that bound is above 0;
    otherwise the baseline stays selected. Trials without fold scores
    matching the baseline's are not candidates. Without a finished baseline
    trial with fold scores there is nothing to compare with, and the best
    score is selected. Ties go to the lower ``trial_id``, so the result does
    not depend on the order the trials ran in.

    Examples:
        >>> a = Trial(0, -1, "baseline", "baseline", PipelineConfig("RNN"),
        ...           score=0.50, fold_scores=[0.48, 0.50, 0.52])
        >>> b = Trial(1, 0, "agent0", "explore", PipelineConfig("MLP"),
        ...           score=0.52, fold_scores=[0.40, 0.58, 0.58])
        >>> select_trial([a, b], True).trial_id, select_trial([a, b], True, 1.0).trial_id
        (1, 0)
    """
    done = sorted((t for t in trials if t.error is None and t.score is not None),
                  key=lambda t: t.trial_id)
    if not done:
        return None
    best = min(done, key=lambda t: -t.score if maximize else t.score)
    if promote_se is None:
        return best
    reference = next((t for t in done if t.agent == "baseline"), None)
    if reference is None or not reference.fold_scores:
        return best
    selected, best_bound = reference, 0.0
    for t in done:
        if t is reference or len(t.fold_scores) != len(reference.fold_scores):
            continue
        gain, se = paired_gain(t.fold_scores, reference.fold_scores, maximize)
        bound = gain - promote_se * se
        if bound > best_bound:
            selected, best_bound = t, bound
    return selected


def select_simplest(
    trials: Sequence[Trial],
    maximize: bool,
    promote_se: float = 1.0,
    rung: Callable[[PipelineConfig], int] | None = None,
) -> tuple[Trial | None, Trial | None]:
    """The most interpretable trial that is as good as the best: the one-SE rule.

    1. The candidates are the baseline trial and every trial whose
       fold-by-fold gain over it has a lower bound,
       ``mean - promote_se * se``, above 0. Without a finished baseline
       there is nothing to compare with, and every finished trial is a
       candidate.
    2. The top candidate is the one with the best mean score.
    3. The selection is the candidate on the lowest ``rung`` whose score is
       at most one paired standard error below the top candidate's; among
       those, the one with the fewest engineered features, then the best
       mean.

    Without fold scores, the best score is both selected and top.

    Args:
        trials: The trials of a search.
        maximize: Whether larger scores are better.
        promote_se: Standard errors a gain must clear to make a candidate.
        rung: A config's place on the interpretability ladder (lower is
            simpler), e.g. :meth:`SearchSpace.rung`. ``None`` puts every
            config on one rung, so the top candidate is selected.

    Returns:
        ``(selected, top)``.

    Examples:
        >>> base = Trial(0, -1, "baseline", "baseline", PipelineConfig("XGBoost"),
        ...              score=0.50, fold_scores=[0.48, 0.50, 0.52])
        >>> lr = Trial(1, 0, "a0", "explore", PipelineConfig("LogisticRegression"),
        ...            score=0.55, fold_scores=[0.52, 0.57, 0.56])
        >>> mlp = Trial(2, 0, "a1", "explore", PipelineConfig("MLP"),
        ...             score=0.56, fold_scores=[0.55, 0.555, 0.575])
        >>> selected, top = select_simplest([base, lr, mlp], True, 1.0,
        ...                                 SearchSpace().rung)
        >>> selected.trial_id, top.trial_id
        (1, 2)
    """
    done = sorted((t for t in trials if t.error is None and t.score is not None),
                  key=lambda t: t.trial_id)
    if not done:
        return None, None
    candidates, top, near = _one_se_sets(done, maximize, promote_se)
    if not candidates:
        best = max(done, key=lambda t: (1 if maximize else -1) * t.score)
        return best, best
    rung = rung or (lambda config: 0)
    # simpler wins: the lower rung, then fewer engineered features
    selected = min(near, key=lambda t: (rung(t.config), len(t.features),
                                        (-1 if maximize else 1) * t.score))
    return selected, top


def _one_se_sets(
    trials: Sequence[Trial],
    maximize: bool,
    promote_se: float = 1.0,
    top: Trial | None = None,
) -> tuple[list[Trial], Trial | None, list[Trial]]:
    """The candidates, the top candidate, and the candidates within one SE of it.

    These are the sets :func:`select_simplest` chooses from (see its steps 1
    to 3); the report shows them. Trials are compared on their fold scores.

    Args:
        trials: The trials of a search; failed ones are ignored.
        maximize: Whether larger scores are better.
        promote_se: Standard errors a gain must clear to make a candidate.
        top: The trial to measure "within one SE" against. Defaults to the
            candidate with the best mean score.

    Returns:
        ``(candidates, top, near)``, in trial order; empty lists and ``None``
        when no trial can be compared.

    Examples:
        >>> base = Trial(0, -1, "baseline", "baseline", PipelineConfig("RNN"),
        ...              score=0.50, fold_scores=[0.48, 0.50, 0.52])
        >>> lr = Trial(1, 0, "a0", "explore", PipelineConfig("LogisticRegression"),
        ...            score=0.55, fold_scores=[0.52, 0.57, 0.56])
        >>> mlp = Trial(2, 0, "a1", "explore", PipelineConfig("MLP"),
        ...             score=0.56, fold_scores=[0.55, 0.555, 0.575])
        >>> candidates, top, near = _one_se_sets([base, lr, mlp], True)
        >>> [t.trial_id for t in candidates], top.trial_id, [t.trial_id for t in near]
        ([0, 1, 2], 2, [1, 2])
    """
    done = sorted((t for t in trials if t.error is None and t.score is not None),
                  key=lambda t: t.trial_id)
    sign = 1 if maximize else -1
    reference = next((t for t in done if t.agent == "baseline"), None)
    if reference is None:
        candidates = [t for t in done if t.fold_scores]
    elif reference.fold_scores:
        candidates = [reference]
        for t in done:
            if t is reference or len(t.fold_scores) != len(reference.fold_scores):
                continue
            gain, se = paired_gain(t.fold_scores, reference.fold_scores, maximize)
            if gain - promote_se * se > 0:
                candidates.append(t)
    else:
        candidates = []
    if not candidates:
        return [], None, []
    top = top or max(candidates, key=lambda t: sign * t.score)

    def within(t: Trial) -> bool:
        if t is top:
            return True
        if len(t.fold_scores) != len(top.fold_scores):
            return False
        gap, se = paired_gain(top.fold_scores, t.fold_scores, maximize)
        return gap <= se

    return candidates, top, [t for t in candidates if within(t)]


def _better(new: float | None, old: float | None, maximize: bool) -> bool:
    if new is None:
        return False
    if old is None:
        return True
    return new > old if maximize else new < old


# ---- confirming and pruning the selection -----------------------------------
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

    Backs :meth:`~pyhealth.autoimprove.AutoImprover.confirm`; see there.
    ``imp`` is the improver.
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

    Backs :meth:`~pyhealth.autoimprove.AutoImprover.prune`; see there.
    ``imp`` is the improver.
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
