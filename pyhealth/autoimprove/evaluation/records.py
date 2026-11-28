"""The records of a run: one Trial per attempt, and the run's result."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..search.space import PipelineConfig, config_diff


@dataclass
class Trial:
    """One proposed configuration and how it did on validation.

    Together the trials are the run's decision log: the config with the
    proposer's rationale and hypothesis (written before the trial ran), the
    trial it was changed from (``parent``) and what changed (``change``),
    and the scores. With feature search, a trial also records its feature
    specs (``features``), the ones it added (``added``), whether the group
    was accepted, and features that looked too good (``suspicious``).

    ``error`` holds the full error text and is kept in ``trials.jsonl``.
    ``error_brief`` leaves out the exception's message, which can quote a raw
    value from a sample, and is the form shown to a proposer.

    Examples:
        >>> trial = Trial(0, 0, "baseline", "baseline", PipelineConfig("RNN"),
        ...               metric_name="pr_auc", score=0.3)
        >>> Trial.from_dict(trial.to_dict()).score
        0.3
    """

    trial_id: int
    cycle: int
    agent: str
    strategy: str
    config: PipelineConfig
    metric_name: str = ""
    score: float | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    seconds: float = 0.0
    n_params: int = 0
    error: str | None = None
    fold_scores: list[float] = field(default_factory=list)
    error_brief: str | None = None
    parent: int | None = None
    change: dict[str, str] = field(default_factory=dict)
    features: list[dict[str, Any]] = field(default_factory=list)
    added: list[str] = field(default_factory=list)
    accepted: bool | None = None
    suspicious: list[str] = field(default_factory=list)
    stopped_early: bool = False

    def to_dict(self) -> dict[str, Any]:
        """The trial as JSON-ready types."""
        data = dict(self.__dict__)
        data["config"] = {**self.config.to_dict(), "rationale": self.config.rationale,
                          "hypothesis": self.config.hypothesis}
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Trial:
        """Rebuilds a trial saved with :meth:`to_dict` (unknown keys are ignored)."""
        known = {f for f in cls.__dataclass_fields__}
        data = {k: v for k, v in data.items() if k in known}
        data["config"] = PipelineConfig.from_dict(data["config"])
        return cls(**data)


@dataclass
class ImproveResult:
    """The outcome of :meth:`AutoImprover.run`.

    Args:
        trials: Every trial, in the order run.
        best: The selected trial (see :func:`select_simplest` and
            :func:`select_trial`).
        baseline: The trial of the starting config, if one was given.
        lessons: The lessons at the end of the run.
        maximize: Whether larger scores are better.
        dropped_models: Models removed from the space before the search
            because they did not run on the training data, with the error.
        top: The highest-scoring candidate. It differs from ``best`` when a
            simpler model within one standard error was selected instead.

    Examples:
        >>> trial = Trial(0, -1, "baseline", "baseline", PipelineConfig("MLP"),
        ...               metric_name="pr_auc", score=0.3)
        >>> result = ImproveResult([trial], trial, trial, [], maximize=True)
        >>> result.leaderboard(k=1)[0].score
        0.3
    """

    trials: list[Trial]
    best: Trial | None
    baseline: Trial | None
    lessons: list[str]
    maximize: bool
    dropped_models: dict[str, str] = field(default_factory=dict)
    top: Trial | None = None

    def leaderboard(self, k: int = 10) -> list[Trial]:
        """The ``k`` best finished trials, best first."""
        done = [t for t in self.trials if t.error is None and not t.stopped_early]
        return sorted(done, key=lambda t: t.score, reverse=self.maximize)[:k]

    def summary(self, k: int = 10) -> str:
        """A markdown table of the leaderboard.

        Examples:
            >>> print(result.summary())  # doctest: +SKIP
        """
        rows = ["| # | agent | model | score | seconds | config |",
                "|---|---|---|---|---|---|"]
        for t in self.leaderboard(k):
            mark = " (selected)" if self.best is not None and t is self.best else ""
            rows.append(
                f"| {t.trial_id}{mark} | {t.agent} | {t.config.model} | {t.score:.4f} | "
                f"{t.seconds:.0f} | `{json.dumps(t.config.to_dict()['model_params'])}`"
                f" `{json.dumps(t.config.to_dict()['train_params'])}` |"
            )
        for name, error in self.dropped_models.items():
            rows.append(f"\nNot searched: {name} does not run on this data ({error})")
        return "\n".join(rows)


def _parent_of(config: PipelineConfig, trials: Sequence[Trial]) -> Trial | None:
    """The trial a config was changed from.

    The proposer's ``based_on`` when it names an earlier trial; otherwise
    the closest earlier trial: same model first, then fewest differing
    settings, then the most recent.
    """
    by_id = {t.trial_id: t for t in trials}
    based_on = config.hypothesis.get("based_on")
    try:
        if based_on is not None and int(based_on) in by_id:
            return by_id[int(based_on)]
    except (TypeError, ValueError):
        pass
    if not trials:
        return None
    return min(
        trials,
        key=lambda t: (t.config.model != config.model,
                       len(config_diff(t.config, config)), -t.trial_id),
    )
