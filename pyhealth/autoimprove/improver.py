"""The improvement loop: propose, train, score, learn, repeat."""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import math
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from pyhealth.datasets import SampleDataset
from pyhealth.metrics import bootstrap_ci, clinical_summary, paired_bootstrap_diff
from pyhealth.processors import PROCESSOR_REGISTRY
from pyhealth.tasks.task_card import DIRECTIONS, FeatureInfo, TaskCard
from pyhealth.trainer import get_metrics_fn

from .evaluate import (
    MAX_LABEL_SCAN,
    check_models,
    describe_error,
    fold_parts,
    greater_is_better,
    patient_folds,
    predict,
    run_config,
    run_config_cv,
    train_pipeline,
)
from .features import EngineeredDataset, FeatureSpace, FeatureSpec
from .knowledge import KnowledgeBase
from .proposers import (
    FEATURE_STRATEGIES,
    STRATEGIES,
    ProposalContext,
    Proposer,
    ProposerExhausted,
)
from .space import PipelineConfig, SearchSpace, config_diff

logger = logging.getLogger(__name__)

DEFAULT_METRIC = {
    "binary": "pr_auc",
    "multiclass": "f1_macro",
    "multilabel": "pr_auc_samples",
    "regression": "mse",
}
"""Metric optimised by default for each label type."""

MAX_PROPOSER_FAILURES = 3
"""Proposer errors in a row (API errors, refusals) before a run stops."""

FOLD_TIE = 1e-4
"""Fold scores closer than this count as a tie, not as one fold being better."""


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


class AutoImprover:
    """Improves a PyHealth pipeline automatically, by guided search.

    A small population of agents takes turns proposing configurations
    (model, arguments, training settings). Each one is checked against the
    search space, trained on the training split and scored on the validation
    split. The agents see each other's results and follow rotating
    strategies (explore, refine, crossover, contrast), as in
    `codegen-classroom <https://github.com/google-research/codegen-classroom>`_.
    After every cycle the proposer rewrites a short list of lessons from the
    results, which later prompts (and later runs, with ``lessons_path``) build
    on. The loop stops after ``n_cycles``, after ``patience`` cycles without
    progress (no new selection and no better best score), or when the trial
    or time budget runs out.

    The test split is never used during the search. Call
    :meth:`evaluate_on_test` once afterwards to compare the best config with
    the starting one on held-out patients.

    Proposals are data, never code: nothing a proposer returns is executed.

    Args:
        train_dataset: Training samples, e.g. from
            ``set_task(task, split=PatientSplit(...))``.
        val_dataset: Validation samples, used to score every trial (or, with
            ``cv_folds``, only for early stopping).
        proposer: Where proposals come from, e.g. ``LLMProposer(ClaudeLLM())``
            or ``RandomProposer()``.
        baseline: The current pipeline, tried first so later trials are
            compared with it.
        space: Allowed models and ranges. Defaults to ``SearchSpace()``.
            ``pos_weight`` is dropped from it unless the label is binary or
            multilabel, the label types it applies to.
        metric: Validation metric to optimise. Defaults by label type
            (:data:`DEFAULT_METRIC`).
        report_metrics: Extra validation metrics to record.
        n_agents: Number of agents.
        n_cycles: Most cycles; each agent proposes once per cycle.
        patience: Cycles without progress before stopping. A cycle makes
            progress if the selected trial changes or some trial beats the
            best score so far. Counting new best scores too keeps a search
            going while it finds better configs whose gains are still too
            noisy to be selected.
        max_trials: Most trials in total (the baseline included).
        time_budget: Most seconds for the whole search.
        seed: Training seed, the same for every trial so scores compare fairly.
            It also fixes the ``cv_folds`` folds.
        cv_folds: Score each trial by patient-grouped cross-validation with
            this many folds of ``train_dataset`` instead of on
            ``val_dataset``. Much less noisy when the validation split is
            small, at the cost of ``cv_folds`` trainings per trial.
        device: Torch device for training.
        run_dir: Directory for ``trials.jsonl``, ``lessons.json`` and
            ``meta.json``. A run with the same directory resumes where the
            last one stopped. ``meta.json`` records the settings the trials
            depend on (metric, ``cv_folds``, ``promote_se``, seed, baseline,
            search space and data sizes); resuming with any of them changed
            raises ``ValueError`` rather than mixing old and new trials.
            Budgets (``n_cycles``, ``max_trials``, ...) may change.
        lessons_path: A JSON file of lessons to reuse across runs. Lessons
            are stored per search space and task (inputs, label, models and
            their arguments), so a run only reads lessons learned on the same
            kind of search.
        promote_se: With ``cv_folds``, every trial is compared with the
            baseline on the shared folds. A trial is a candidate if its gain
            has a lower bound above 0, the mean minus ``promote_se``
            standard errors; the baseline is always a candidate. Without a
            baseline (or if the baseline trial failed) there is nothing to
            compare with, and every finished trial is a candidate. ``None``
            selects the best mean.
        one_se: With ``promote_se``, select the most interpretable
            candidate (lowest rung, :meth:`SearchSpace.rung`) whose score is
            within one paired standard error of the best candidate's, the
            one-SE rule (:func:`select_simplest`). ``False`` selects the
            candidate with the highest lower bound (:func:`select_trial`).
        check: Before searching, give every model in the space one training
            step on a few training samples (:func:`check_models`) and remove
            the ones that fail, e.g. a model that cannot read one of the
            task's input types, so no trial is spent on them. They are listed
            in ``ImproveResult.dropped_models``.
        task_card: The task's :class:`~pyhealth.tasks.TaskCard` (e.g.
            ``task.task_card()``): the design choices with their rationale and
            the feature dictionary. Reports use it to explain the model in
            clinical terms and to check the directions of its effects.
        features: A :class:`~pyhealth.autoimprove.FeatureSpace` to search
            input features as well as settings (needs ``cv_folds``). Cycles
            alternate: feature cycles hold the model at the current selection
            and propose groups of specs, each accepted only if its paired gain
            over the accepted set clears the lower bound (forward selection);
            model cycles hold the accepted features and propose configs.
        feature_cycles: Feature cycles and model cycles in each round,
            e.g. ``(2, 1)``.
        confirm: After the search, re-score the reference and the top
            ``confirm_top`` candidates on new folds with ``confirm_seeds``
            training seeds, and select on those scores (:meth:`confirm`).
            This removes part of the winner's curse before the test.
        confirm_top: Candidates confirmed.
        confirm_seeds: Training seeds per confirmed config.
        confirm_se: The bound candidates must clear on the confirmation
            scores, in SEs. Stricter than ``promote_se`` by default, because
            the confirmed candidates were already chosen as the best of many.
        prune: After confirmation, drop each accepted feature group whose
            removal costs less than one SE (:meth:`prune`).
        early_stop: Stop a trial after 2 folds when every fold so far is
            below the reference's by more than this margin. ``None`` runs all
            folds.
        suspicion_auc: A new feature whose ROC-AUC alone exceeds this (or
            is below 1 minus it) is flagged as possible leakage and not
            accepted.
        knowledge: A :class:`~pyhealth.autoimprove.KnowledgeBase`, already
            built and mapped. Feature cycles then also follow the "evidence"
            strategy, which shows the LLM the untried published predictors
            with their citations, and the report gets a literature section.

    Examples:
        >>> from pyhealth.autoimprove import (AutoImprover, ClaudeLLM,
        ...     LLMProposer, PipelineConfig)
        >>> improver = AutoImprover(
        ...     train_ds, val_ds, LLMProposer(ClaudeLLM()),
        ...     baseline=PipelineConfig("RNN", {"hidden_dim": 128}, {"epochs": 10}),
        ...     n_agents=3, n_cycles=5, run_dir="runs/mortality",
        ... )  # doctest: +SKIP
        >>> result = improver.run()  # doctest: +SKIP
        >>> print(result.summary())  # doctest: +SKIP
        >>> improver.attribute()  # doctest: +SKIP
        >>> improver.evaluate_on_test(test_ds)  # doctest: +SKIP
        >>> improver.report("runs/mortality/report.html")  # doctest: +SKIP
    """

    def __init__(
        self,
        train_dataset: SampleDataset,
        val_dataset: SampleDataset,
        proposer: Proposer,
        baseline: PipelineConfig | None = None,
        space: SearchSpace | None = None,
        metric: str | None = None,
        report_metrics: Sequence[str] = (),
        n_agents: int = 3,
        n_cycles: int = 5,
        patience: int = 2,
        max_trials: int | None = None,
        time_budget: float | None = None,
        seed: int = 0,
        device: str | None = None,
        run_dir: str | None = None,
        lessons_path: str | None = None,
        cv_folds: int | None = None,
        promote_se: float | None = 1.0,
        one_se: bool = True,
        check: bool = True,
        task_card: TaskCard | None = None,
        features: FeatureSpace | None = None,
        feature_cycles: tuple[int, int] = (2, 1),
        confirm: bool = False,
        confirm_top: int = 3,
        confirm_seeds: int = 3,
        confirm_se: float = 2.0,
        prune: bool = True,
        early_stop: float | None = None,
        suspicion_auc: float = 0.95,
        knowledge: KnowledgeBase | None = None,
    ):
        if features is not None and not cv_folds:
            raise ValueError("feature search needs cv_folds")
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.proposer = proposer
        self.baseline = baseline
        self.space = space or SearchSpace()
        self.mode = _label_mode(train_dataset)
        if self.mode not in ("binary", "multilabel") and "pos_weight" in self.space.train_params:
            self.space = SearchSpace(
                self.space.models,
                {k: v for k, v in self.space.train_params.items() if k != "pos_weight"},
            )
        self.metric = metric or DEFAULT_METRIC.get(self.mode, "loss")
        self.maximize = greater_is_better(self.metric)
        self.report_metrics = list(report_metrics)
        self.n_agents = n_agents
        self.n_cycles = n_cycles
        self.patience = patience
        self.max_trials = max_trials
        self.time_budget = time_budget
        self.seed = seed
        self.device = device
        self.run_dir = run_dir
        self.lessons_path = lessons_path
        self.cv_folds = cv_folds
        self.promote_se = promote_se if cv_folds else None
        self.one_se = one_se
        if self.promote_se is not None and baseline is None:
            logger.warning(
                "promote_se compares every trial with the baseline, and no "
                "baseline was given: every finished trial is a candidate, and %s",
                "the one-SE rule picks among them" if one_se
                else "the best mean score is selected",
            )
        self.folds = (
            patient_folds(train_dataset, cv_folds, seed) if cv_folds else None
        )
        self.check = check
        self.task_card = task_card
        self.features = features
        self.feature_cycles = tuple(feature_cycles)
        self.confirm_enabled = confirm
        self.confirm_top = confirm_top
        self.confirm_seeds = confirm_seeds
        self.confirm_se = confirm_se
        self.prune_enabled = prune
        self.early_stop = early_stop
        self.suspicion_auc = suspicion_auc
        self.knowledge = knowledge
        self.confirmed: dict[int, list[float]] = {}
        self.pruned_id: int | None = None
        self._raw_cache: dict[tuple[str, str], np.ndarray] = {}
        self._samples_cache: dict[str, list] = {}
        self._exhausted = False
        self.dropped_models: dict[str, str] = {}
        self._checked = False
        self.trials: list[Trial] = []
        self.lessons: list[str] = []
        self._task = describe_task(train_dataset, val_dataset)
        if cv_folds:
            self._task += (
                f"\nScores are the mean over {cv_folds} patient-grouped "
                "cross-validation folds of the training samples; "
                f"{self.metric}_sd is the spread between folds."
            )
        self._fingerprint = space_fingerprint(self.space, train_dataset)
        if features is not None:
            self._fingerprint += "-features"
        self._settings = self._run_settings()
        self._proposer_failures = 0

    # ------------------------------------------------------------------ run
    def run(self) -> ImproveResult:
        """Runs the search and returns its trials and lessons."""
        if self.check and not self._checked:
            self._check_models()
        self._load()
        self._save_settings()
        self._warn_if_small_validation()
        start = time.perf_counter()
        if self.baseline is not None and not any(
            t.agent == "baseline" for t in self.trials
        ):
            config, notes = self.space.validate(self.baseline)
            self._evaluate(config, -1, "baseline", "baseline", notes)
        first_cycle = max((t.cycle for t in self.trials), default=-1) + 1
        best, top = self._selected_id(), self._top_score()
        stale = 0
        for cycle in range(first_cycle, self.n_cycles):
            kind = self._cycle_kind(cycle)
            strategies = list(STRATEGIES) if kind == "model" else [
                s for s in FEATURE_STRATEGIES if s != "evidence" or self.knowledge]
            first = "evidence" if kind == "features" and self.knowledge else "explore"
            for i in range(self.n_agents):
                if self._out_of_budget(start) or self._exhausted:
                    break
                strategy = first if cycle == 0 else strategies[
                    (cycle + i) % len(strategies)
                ]
                if kind == "features":
                    self._feature_turn(cycle, f"agent{i}", strategy)
                else:
                    self._turn(cycle, f"agent{i}", strategy)
            try:
                self.lessons = self.proposer.reflect(
                    self.trials, self.lessons, self.metric, space=self.space
                )
            except Exception as e:  # noqa: BLE001 - keep the lessons we have
                logger.warning("lessons not updated in cycle %d: %s", cycle,
                               describe_error(e))
            self._save_lessons()
            new_best, new_top = self._selected_id(), self._top_score()
            if (new_best is not None and new_best != best) or _better(
                new_top, top, self.maximize
            ):
                stale = 0
            else:
                stale += 1
            best, top = new_best, new_top
            logger.info(
                "cycle %d: selected #%s, %s=%s", cycle, best, self.metric,
                self._best_score(),
            )
            if stale >= self.patience or self._out_of_budget(start) or self._exhausted:
                break
        if self.confirm_enabled and self.folds is not None and not self.confirmed:
            self.confirm()
            if self.prune_enabled and self.features is not None:
                self.prune()
        return self.result()

    def _cycle_kind(self, cycle: int) -> str:
        if self.features is None:
            return "model"
        if not self.proposer.proposes_models:
            return "features"
        n_features, n_models = self.feature_cycles
        return "features" if cycle % (n_features + n_models) < n_features else "model"

    def _check_models(self) -> None:
        """Removes models that do not run on the training data from the space.

        Raises:
            ValueError: If the baseline's model, or every model, fails.
        """
        self._checked = True
        failed = {
            name: error
            for name, error in check_models(
                self.space, self.train_dataset, seed=self.seed, device=self.device
            ).items()
            if error is not None
        }
        if not failed:
            return
        if self.baseline is not None and self.baseline.model in failed:
            raise ValueError(
                f"The baseline model {self.baseline.model} does not run on this "
                f"data: {failed[self.baseline.model]}"
            )
        if len(failed) == len(self.space.models):
            raise ValueError(
                "No model in the search space runs on this data: "
                + "; ".join(f"{k}: {v}" for k, v in failed.items())
            )
        for name, error in failed.items():
            logger.warning("%s removed from the search space: %s", name, error)
        self.dropped_models = failed
        self.space = SearchSpace(
            {k: v for k, v in self.space.models.items() if k not in failed},
            self.space.train_params,
        )
        self._fingerprint = space_fingerprint(self.space, self.train_dataset)

    def _warn_if_small_validation(self, min_positives: int = 30) -> None:
        if self.mode != "binary" or self.cv_folds:
            return
        positives = _positive_count(self.val_dataset)
        if positives is not None and positives < min_positives:
            logger.warning(
                "The validation split has %d positive samples. Validation "
                "scores will be noisy and the search may pick configs that "
                "only fit this split; trust evaluate_on_test, not the "
                "leaderboard.",
                positives,
            )

    def result(self) -> ImproveResult:
        """The current state as an :class:`ImproveResult`.

        After :meth:`confirm`, the selection is made among the confirmed
        trials on their confirmation scores.
        """
        # a feature group rejected during the search never becomes the result
        trials = [t for t in self.trials if t.accepted is not False]
        bound = self.promote_se
        if self.confirmed:
            by_id = {t.trial_id: t for t in trials}
            trials = [
                replace(by_id[i], fold_scores=list(scores),
                        score=float(np.mean(scores)))
                for i, scores in self.confirmed.items() if i in by_id
            ]
            bound = max(self.promote_se or 0.0, self.confirm_se)
        if self.promote_se is not None and self.one_se:
            best, top = select_simplest(trials, self.maximize, bound, self.space.rung)
        else:
            best = top = select_trial(trials, self.maximize, bound)
        if self.confirmed:
            by_id = {t.trial_id: t for t in self.trials}
            best = by_id.get(best.trial_id) if best is not None else None
            top = by_id.get(top.trial_id) if top is not None else None
            if self.pruned_id is not None and self.pruned_id in by_id:
                best = by_id[self.pruned_id]  # parsimony: the pruned selection
        baseline = next((t for t in self.trials if t.agent == "baseline"), None)
        return ImproveResult(
            list(self.trials), best, baseline, list(self.lessons), self.maximize,
            dict(self.dropped_models), top,
        )

    def selection_rule(self) -> str:
        """How the selection is made, in plain words (for prompts and reports)."""
        if self.promote_se is None:
            return "The selection is the trial with the best score."
        simplest = (
            "the most interpretable model (lowest rung) whose score is within one "
            "paired standard error of the best"
        )
        baseline = next((t for t in self.trials if t.agent == "baseline"), None)
        if self.baseline is None or (baseline is not None and baseline.error):
            if not self.one_se:
                return ("There is no starting pipeline to compare with, so the "
                        "selection is the trial with the best mean score.")
            return ("There is no starting pipeline to compare with, so every trial "
                    f"is a candidate. The selection is {simplest}: simpler models "
                    "win ties.")
        bound = (
            f"Every trial is compared with the starting pipeline fold by fold. "
            f"A trial is a candidate if its gain is above 0 by more than "
            f"{self.promote_se:g} standard error(s), so a gain must be "
            f"consistent across folds, not only high on average."
        )
        if not self.one_se:
            return bound + " The selection is the candidate with the highest bound."
        return bound + (
            " Among the candidates, the selection is the most interpretable "
            "model (lowest rung) whose score is within one paired standard error "
            "of the best candidate's: simpler models win ties."
        )

    def _turn(self, cycle: int, agent: str, strategy: str) -> None:
        feedback = ""
        accepted = self._accepted()
        for _ in range(2):  # one retry when the proposal is invalid or repeated
            context = self._context(agent, strategy, feedback)
            try:
                proposal = self.proposer.propose(context)
            except ValueError as e:
                feedback = f"your last proposal was rejected: {e}"
                continue
            except Exception as e:  # noqa: BLE001 - an API error or refusal
                self._skip_turn(agent, cycle, e)
                return
            self._proposer_failures = 0
            try:
                config, notes = self.space.validate(proposal)
            except ValueError as e:
                feedback = f"your last proposal was rejected: {e}"
                continue
            if _key(config, accepted) in context.tried:
                feedback = "your last proposal repeated an earlier trial"
                if notes:
                    feedback += (
                        " once it was changed to fit the search space ("
                        + "; ".join(notes)
                        + "). Stay inside the ranges listed above"
                    )
                continue
            self._evaluate(config, cycle, agent, strategy, notes, features=accepted)
            return
        logger.warning(
            "%s made no usable proposal in cycle %d: %s", agent, cycle, feedback
        )

    def _skip_turn(self, agent: str, cycle: int, error: Exception) -> None:
        """Skips a turn whose proposer failed, e.g. an API error or a refusal.

        Raises:
            RuntimeError: After :data:`MAX_PROPOSER_FAILURES` failures in a
                row (e.g. missing credentials), so a broken backend does not
                spend the whole budget.
        """
        self._proposer_failures += 1
        text = describe_error(error)
        if self._proposer_failures >= MAX_PROPOSER_FAILURES:
            raise RuntimeError(
                f"the proposer failed {self._proposer_failures} turns in a row; "
                f"last error: {text}"
            ) from error
        logger.warning("%s's proposer failed in cycle %d, turn skipped: %s",
                       agent, cycle, text)

    # --------------------------------------------------------- feature search
    def _feature_turn(self, cycle: int, agent: str, strategy: str) -> None:
        """One feature proposal: validate, test for leaks, score, accept or not."""
        base = self._incumbent(cycle)
        if base is None:
            return
        accepted = self._accepted()
        tried = {name for t in self.trials for name in t.added}
        have = {s["name"] for s in accepted}
        feedback = ""
        for _ in range(2):  # one retry when the proposal is invalid
            context = self._context(agent, strategy, feedback, kind="features")
            try:
                proposal = self.proposer.propose_features(context)
            except ProposerExhausted:
                self._exhausted = True
                return
            except ValueError as e:
                feedback = f"your last proposal was rejected: {e}"
                continue
            except Exception as e:  # noqa: BLE001 - an API error or refusal
                self._skip_turn(agent, cycle, e)
                return
            self._proposer_failures = 0
            try:
                specs = self.features.validate_many(list(proposal.get("features") or []))
            except ValueError as e:
                feedback = f"your last proposal was rejected: {e}"
                continue
            new = [s for s in specs if s.name not in have and s.name not in tried]
            if not new:
                feedback = "every feature in your last proposal was already tried"
                continue
            leaks = {k: v for k, v in self.features.future_blind(
                new, self.train_dataset).items() if v}
            if leaks:
                feedback = "the future-blind test rejected " + "; ".join(
                    f"{k}: {v}" for k, v in leaks.items())
                continue
            hypothesis = {k: proposal[k] for k in ("expected", "kind") if proposal.get(k)}
            config = PipelineConfig(
                base.config.model, dict(base.config.model_params),
                dict(base.config.train_params), str(proposal.get("rationale", "")),
                hypothesis,
            )
            trial = self._evaluate(
                config, cycle, agent, strategy, features=accepted + [
                    s.to_dict() for s in new], added=[s.name for s in new],
                parent=base, save=False,
            )
            self._judge(trial, base, new)
            self._save_trial(trial)
            return
        logger.warning(
            "%s made no usable feature proposal in cycle %d: %s", agent, cycle, feedback
        )

    def _judge(self, trial: Trial, base: Trial, new: Sequence[FeatureSpec]) -> None:
        """Accepts a feature group if its paired gain over ``base`` clears the bound."""
        if trial.error is not None or trial.stopped_early:
            trial.accepted = False
            return
        labels = self._labels()
        for spec in new:
            column = self._raw(spec, "train")
            filled = np.where(np.isnan(column), np.nanmedian(column)
                              if np.isfinite(column).any() else 0.0, column)
            if len(np.unique(labels)) > 1 and np.ptp(filled) > 0:
                auc = roc_auc_score(labels, filled)
                if max(auc, 1 - auc) > self.suspicion_auc:
                    trial.suspicious.append(spec.name)
                    trial.notes.append(
                        f"{spec.name} alone has ROC-AUC {max(auc, 1 - auc):.3f}: "
                        "possible leakage; held for review"
                    )
        gain, se = paired_gain(trial.fold_scores, base.fold_scores, self.maximize)
        bound = gain - (self.promote_se or 1.0) * se
        trial.accepted = bound > 0 and not trial.suspicious
        trial.notes.append(
            f"vs #{base.trial_id} without the new features: {gain:+.4f} ± {se:.4f}; "
            + ("accepted" if trial.accepted else "not accepted")
        )

    def _accepted(self) -> list[dict[str, Any]]:
        """The accepted feature specs: those of the last accepted feature trial."""
        accepted = [t for t in self.trials if t.accepted]
        return [dict(s) for s in accepted[-1].features] if accepted else []

    def _incumbent(self, cycle: int) -> Trial | None:
        """The trial with the selected config and the accepted features."""
        selected = self.result().best or next(
            (t for t in self.trials if t.agent == "baseline"), None)
        if selected is None:
            return None
        accepted = self._accepted()
        key = _key(selected.config, accepted)
        for t in self.trials:
            if (_key(t.config, t.features) == key and t.error is None
                    and not t.stopped_early):
                return t
        trial = self._evaluate(
            PipelineConfig(selected.config.model, dict(selected.config.model_params),
                           dict(selected.config.train_params),
                           "the selected config with the accepted features"),
            cycle, "incumbent", "incumbent", features=accepted,
        )
        return trial if trial.error is None else None

    def _labels(self) -> np.ndarray:
        if "labels" not in self._samples_cache:
            key = next(iter(self.train_dataset.output_schema))
            self._samples_cache["labels"] = np.array([
                float(np.ravel(self._samples("train")[i][key])[0])
                for i in range(len(self.train_dataset))
            ])
        return self._samples_cache["labels"]

    def _samples(self, split: str) -> list:
        if split not in self._samples_cache:
            dataset = self.train_dataset if split == "train" else self.val_dataset
            self._samples_cache[split] = [dataset[i] for i in range(len(dataset))]
        return self._samples_cache[split]

    def _raw(self, spec: FeatureSpec, split: str,
             dataset: SampleDataset | None = None) -> np.ndarray:
        key = (spec.key(), split)
        if key not in self._raw_cache:
            if dataset is None:
                dataset = self.train_dataset if split == "train" else self.val_dataset
            self._raw_cache[key] = self.features.compute([spec], dataset)[:, 0]
        return self._raw_cache[key]

    def _datasets(self, specs: Sequence[dict[str, Any]]) -> tuple[Any, Any]:
        """Training and validation sets with a trial's features added."""
        if not specs:
            return self.train_dataset, self.val_dataset
        valid = [FeatureSpec.from_dict(s) for s in specs]
        names = [v.name for v in valid]
        train = EngineeredDataset(
            self.train_dataset, np.column_stack([self._raw(v, "train") for v in valid]),
            names, base_samples=self._samples("train"))
        val = EngineeredDataset(
            self.val_dataset, np.column_stack([self._raw(v, "val") for v in valid]),
            names, stats=train.stats, base_samples=self._samples("val"))
        return train, val

    def _test_dataset(self, specs, test_dataset, train) -> Any:
        if not specs:
            return test_dataset
        valid = [FeatureSpec.from_dict(s) for s in specs]
        raw = np.column_stack([self._raw(v, f"test{id(test_dataset)}", test_dataset)
                               for v in valid])
        return EngineeredDataset(test_dataset, raw, [v.name for v in valid],
                                 stats=train.stats)

    def confirm(self) -> dict[str, Any]:
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
        result = self.result()
        baseline = result.baseline
        reference = baseline if baseline is not None and baseline.error is None else None
        scored = [t for t in self.trials if t.error is None and t.fold_scores]
        if not scored or (reference is not None and not reference.fold_scores):
            return {}
        n = len((reference or scored[0]).fold_scores)
        candidates = [t for t in scored if not t.stopped_early
                      and len(t.fold_scores) == n and t is not reference
                      and t.accepted is not False]
        sign = 1 if self.maximize else -1

        def clears(t: Trial) -> bool:
            if reference is None:  # nothing to compare with: every trial
                return True
            gain, se = paired_gain(t.fold_scores, reference.fold_scores, self.maximize)
            return gain - (self.promote_se or 1.0) * se > 0

        bound = [t for t in candidates if clears(t)]
        chosen = sorted(bound, key=lambda t: -sign * t.score)[: self.confirm_top]
        if result.best is not None and result.best not in [*chosen, reference]:
            chosen.append(result.best)
        self._confirm_folds = patient_folds(self.train_dataset, n, self.seed + 1)
        for t in [reference, *chosen] if reference is not None else chosen:
            self.confirmed[t.trial_id] = self._confirm_scores(t.config, t.features)
        out = {"seeds": [self.seed + k for k in range(self.confirm_seeds)],
               "fold_seed": self.seed + 1,
               "scores": {str(k): v for k, v in self.confirmed.items()}}
        if self.run_dir:
            with open(os.path.join(self.run_dir, "confirm.json"), "w") as f:
                json.dump(out, f, indent=1)
        return out

    def _confirm_scores(self, config, specs) -> list[float]:
        train, val = self._datasets(specs)
        runs = [
            run_config_cv(config, self.space, train, val, self.metric,
                          self._confirm_folds, (), self.seed + k, self.device)["fold_scores"]
            for k in range(self.confirm_seeds)
        ]
        return [float(v) for v in np.mean(np.array(runs), axis=0)]

    def prune(self) -> dict[str, Any]:
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
        best = self.result().best
        if best is None or not best.features or best.trial_id not in self.confirmed:
            return {}
        groups = _feature_groups(best, self.trials)
        current = list(best.features)
        scores = self.confirmed[best.trial_id]
        ablation, dropped = [], []
        while groups:
            rows = []
            for trial_id, names in groups:
                without = [s for s in current if s["name"] not in names]
                lost = self._confirm_scores(best.config, without)
                gain, se = paired_gain(scores, lost, self.maximize)
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
            pruned = self._evaluate(
                PipelineConfig(best.config.model, dict(best.config.model_params),
                               dict(best.config.train_params),
                               "the selection without features it does not need"),
                max((t.cycle for t in self.trials), default=0), "pruned", "prune",
                features=current,
            )
            if pruned.error is None:
                self.confirmed[pruned.trial_id] = scores
                self.pruned_id = pruned.trial_id
                out["pruned_trial"] = pruned.trial_id
        if self.run_dir:
            with open(os.path.join(self.run_dir, "prune.json"), "w") as f:
                json.dump(out, f, indent=1)
            with open(os.path.join(self.run_dir, "confirm.json")) as f:
                saved = json.load(f)
            saved["scores"] = {str(k): v for k, v in self.confirmed.items()}
            with open(os.path.join(self.run_dir, "confirm.json"), "w") as f:
                json.dump(saved, f, indent=1)
        return out

    def _evaluate(self, config, cycle, agent, strategy, notes=None, features=None,
                  added=None, parent=None, save=True) -> Trial:
        trial = Trial(len(self.trials), cycle, agent, strategy, config,
                      metric_name=self.metric, notes=list(notes or []),
                      features=[dict(s) for s in features or []],
                      added=list(added or []))
        if parent is None and agent != "baseline":
            parent = _parent_of(config, self.trials)
        if parent is not None:
            trial.parent, trial.change = parent.trial_id, config_diff(
                parent.config, config
            )
        try:
            train, val = self._datasets(trial.features)
            if self.folds is not None:
                out = run_config_cv(
                    config, self.space, train, val, self.metric, self.folds,
                    self.report_metrics, self.seed, self.device,
                    stop_if=self._early_stop_rule(),
                )
            else:
                out = run_config(
                    config, self.space, train, val, self.metric,
                    self.report_metrics, self.seed, self.device,
                )
            trial.metrics = out["metrics"]
            trial.score = out["metrics"][self.metric]
            trial.seconds = out["seconds"]
            trial.n_params = out["n_params"]
            trial.fold_scores = out.get("fold_scores", [])
            trial.stopped_early = bool(out.get("stopped_early"))
            if trial.stopped_early:
                trial.notes.append("stopped after "
                                   f"{len(trial.fold_scores)} folds: clearly worse")
            if trial.score is None or math.isnan(trial.score):
                where = ("a cross-validation fold" if self.folds is not None
                         else "the validation split")
                trial.error = trial.error_brief = (
                    f"{self.metric} is undefined on {where}"
                )
        except Exception as e:  # noqa: BLE001 - a failed trial is feedback
            trial.error = describe_error(e)
            trial.error_brief = describe_error(e, message=False)
        self.trials.append(trial)
        if save:
            self._save_trial(trial)
        return trial

    def _early_stop_rule(self):
        reference = next((t for t in self.trials if t.agent == "baseline"
                          and t.fold_scores), None)
        if self.early_stop is None or reference is None:
            return None
        sign = 1 if self.maximize else -1

        def stop(scores: list[float]) -> bool:
            if len(scores) < 2:
                return False
            return all(sign * (r - x) > self.early_stop
                       for r, x in zip(reference.fold_scores, scores))

        return stop

    def _context(self, agent, strategy, feedback, kind="model") -> ProposalContext:
        result = self.result()
        done = result.leaderboard(k=len(self.trials))
        card = ""
        if self.task_card is not None:
            card = "\n".join(f"- {name}: {item['text']}"
                              for name, item in self.task_card.design())
        return ProposalContext(
            agent=agent,
            strategy=strategy,
            space=self.space,
            task=self._task,
            metric=self.metric,
            maximize=self.maximize,
            own_trials=[t for t in self.trials if t.agent == agent][-8:],
            leaderboard=done[:8],
            worst=list(reversed(done[8:]))[:3] if len(done) > 8 else [],
            tried={_key(t.config, t.features) for t in self.trials},
            lessons=self.lessons,
            feedback=feedback,
            selected=result.best,
            selection_rule=self.selection_rule() if self.promote_se is not None else "",
            kind=kind,
            features=self.features,
            accepted=self._accepted(),
            feature_trials=[t for t in self.trials if t.added][-12:],
            card=card,
            evidence=self._evidence_lines() if kind == "features" else (),
        )

    def _evidence_lines(self, k: int = 8) -> list[str]:
        if self.knowledge is None:
            return []
        tried = {name for t in self.trials for name in t.added}
        lines = []
        for spec in self.knowledge.ranked_specs():
            if spec["name"] in tried:
                continue
            cites = ", ".join(spec.get("citations", []))
            lines.append(f"{spec['name']}: {spec['definition']} (direction "
                         f"{spec.get('direction') or 'ambiguous'}; prior "
                         f"{spec.get('prior', 0):.2f}; {cites}). Spec: "
                         + json.dumps({k: v for k, v in spec.items() if k not in (
                             "name", "definition", "prior")}))
            if len(lines) >= k:
                break
        return lines

    def _best_score(self) -> float | None:
        best = self.result().best
        return None if best is None else best.score

    def _top_score(self) -> float | None:
        top = self.result().leaderboard(k=1)
        return top[0].score if top else None

    def _selected_id(self) -> int | None:
        best = self.result().best
        return None if best is None else best.trial_id

    def _out_of_budget(self, start: float) -> bool:
        if self.max_trials is not None and len(self.trials) >= self.max_trials:
            return True
        if self.time_budget is not None:
            return time.perf_counter() - start >= self.time_budget
        return False

    # ----------------------------------------------------------- test split
    def evaluate_on_test(
        self,
        test_dataset: SampleDataset,
        n_boot: int = 1000,
        subgroups: Mapping[str, Sequence[Any]] | None = None,
        return_predictions: bool = False,
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
            subgroups: Attribute name to one label per test sample, in the
                dataset's order, e.g. ``{"sex": [...], "age band": [...]}``.
            return_predictions: Also return the test predictions, e.g. to
                compute other metrics without training again. They are not
                written to ``test.json``.

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
        result = self.result()
        if result.best is None:
            raise ValueError("no successful trial to evaluate")
        configs = {"best": result.best}
        if (result.baseline is not None and result.baseline.error is None
                and result.baseline is not result.best):
            configs["baseline"] = result.baseline
        if result.top is not None and result.top is not result.best:
            configs["top"] = result.top
        held = self._held_back(result.best)
        if held:
            configs["without_flagged"] = replace(result.best, features=[
                s for s in result.best.features if s["name"] not in held])
        probs, out = {}, {"metric": self.metric}
        if held:
            out["held_back"] = sorted(held)
        for name, trial in configs.items():
            config = trial.config
            train, val = self._datasets(trial.features)
            model = train_pipeline(
                config, self.space, train, val, self.metric, self.seed, self.device,
            )
            test_ds = self._test_dataset(trial.features, test_dataset, train)
            y_true, y_prob, patient_ids = predict(model, test_ds)
            probs[name] = y_prob
            groups = np.asarray(patient_ids) if patient_ids else None
            if self.mode == "binary":
                out[name] = bootstrap_ci(
                    y_true, y_prob, self.metric, groups=groups,
                    n_boot=n_boot, seed=self.seed,
                )
            else:
                value = get_metrics_fn(self.mode)(y_true, y_prob, metrics=[self.metric])
                out[name] = {"estimate": float(value[self.metric])}
            out[name]["config"] = config.to_dict()
            out[name]["trial_id"] = trial.trial_id
            if self.mode == "binary":
                out[name]["clinical"] = clinical_summary(
                    np.ravel(y_true), np.ravel(y_prob), groups=groups,
                    subgroups=subgroups,
                )
        pairs = {"difference": ("best", "baseline"), "top_minus_best": ("top", "best")}
        for key, (a, b) in pairs.items():
            if a in probs and b in probs and self.mode == "binary":
                out[key] = paired_bootstrap_diff(
                    y_true, probs[a], probs[b], self.metric,
                    groups=groups, n_boot=n_boot, seed=self.seed,
                )
        if self.run_dir:
            os.makedirs(self.run_dir, exist_ok=True)
            with open(os.path.join(self.run_dir, "test.json"), "w") as f:
                json.dump(out, f, indent=1, default=float)
        if return_predictions:
            out["predictions"] = {"y_true": y_true, "patient_ids": patient_ids, **probs}
        return out

    # ---------------------------------------------------------- explaining
    def attribute(self, trial_id: int | None = None) -> dict[str, Any]:
        """Measures what each changed setting of a trial contributes.

        Each setting that differs between the trial (by default the
        selected one) and its reference is reverted, one at a time, and the
        reverted config is scored on the same folds. A setting's
        contribution is the trial's score minus the reverted config's, with
        its paired standard error. The reference is the baseline when the
        trial uses the baseline's model, and otherwise the default settings
        of the trial's own model (arguments of different models cannot be
        compared). With a ``run_dir`` the result is saved to
        ``attribution.json``.

        Returns:
            ``{"trial_id", "reference", "rows": [{"setting", "from", "to",
            "contribution", "se", "folds_better", "score_without"}], ...}``.

        Examples:
            >>> improver.attribute()  # doctest: +SKIP
            {'trial_id': 17, 'reference': 'the starting pipeline (#0)', 'rows': [...]}
        """
        result = self.result()
        target = result.best if trial_id is None else self.trials[trial_id]
        if target is None or target.error is not None:
            raise ValueError("no successful trial to attribute")
        base = result.baseline
        if base is not None and base.error is None and base.config.model == (
            target.config.model
        ):
            reference = base.config
            label = f"the starting pipeline (#{base.trial_id})"
        else:
            reference, _ = self.space.validate(PipelineConfig(target.config.model))
            label = f"the default settings of {target.config.model}"
        out: dict[str, Any] = {
            "trial_id": target.trial_id, "reference": label, "metric": self.metric,
            "score": target.score, "rows": [],
        }
        for group in ("model_params", "train_params"):
            ref_params = getattr(reference, group)
            for name, value in sorted(getattr(target.config, group).items()):
                if name not in ref_params or ref_params[name] == value:
                    continue
                reverted = PipelineConfig(
                    target.config.model, dict(target.config.model_params),
                    dict(target.config.train_params),
                )
                getattr(reverted, group)[name] = ref_params[name]
                scored = self._score(reverted, target.features)
                row = {"setting": name, "from": ref_params[name], "to": value,
                       "score_without": scored["score"],
                       "fold_scores_without": scored["fold_scores"]}
                if scored["fold_scores"] and target.fold_scores:
                    gain, se = paired_gain(
                        target.fold_scores, scored["fold_scores"], self.maximize
                    )
                    sign = 1 if self.maximize else -1
                    better = sum(
                        sign * (a - b) > FOLD_TIE
                        for a, b in zip(target.fold_scores, scored["fold_scores"])
                    )
                    row.update(contribution=gain, se=se, folds_better=better)
                else:
                    sign = 1 if self.maximize else -1
                    row.update(contribution=sign * (target.score - scored["score"]),
                               se=None, folds_better=None)
                out["rows"].append(row)
        if self.run_dir:
            os.makedirs(self.run_dir, exist_ok=True)
            with open(os.path.join(self.run_dir, "attribution.json"), "w") as f:
                json.dump(out, f, indent=1, default=str)
        return out

    def _score(self, config: PipelineConfig, features=()) -> dict[str, Any]:
        config, _ = self.space.validate(config)
        train, val = self._datasets(list(features))
        if self.folds is not None:
            out = run_config_cv(
                config, self.space, train, val, self.metric, self.folds, (),
                self.seed, self.device,
            )
        else:
            out = run_config(
                config, self.space, train, val, self.metric, (), self.seed,
                self.device,
            )
        return {"score": out["metrics"][self.metric],
                "fold_scores": out.get("fold_scores", [])}

    def explain(
        self,
        top_k: int = 10,
        stability: bool = True,
        max_samples: int = 2000,
    ) -> dict[str, Any]:
        """Explains how the selected model decides, in clinical terms.

        The selected config is trained on the training split and its
        predictions on training samples are attributed to the inputs
        (:func:`~pyhealth.autoimprove.explain.attributions`: exact TreeSHAP
        for trees, integrated gradients for torch models). The summary
        (:func:`~pyhealth.autoimprove.explain.summarize`) gives:

        - importance by the task card's clinical factor groups;
        - the top features, each with its meaning and the shape of its
          effect, in bins of at least 11 patients;
        - each feature's learned direction, compared with the direction the
          card expects.

        With ``stability`` and ``cv_folds``, the config is also trained on
        each fold's training part, to show how many of the top features each
        fold model shares. With a ``run_dir`` the result is saved to
        ``explain.json``; it holds aggregates only.

        Examples:
            >>> improver.explain()["groups"]  # doctest: +SKIP
            [{'group': 'labs', 'share': 0.52}, {'group': 'vital signs', 'share': 0.31}, ...]
        """
        from .explain import attributions, summarize, top_overlap

        result = self.result()
        if result.best is None:
            raise ValueError("no successful trial to explain")
        config = result.best.config
        train, val = self._datasets(result.best.features)
        extra = {
            s["name"]: FeatureInfo(
                s.get("definition", s["name"]), "engineered",
                s.get("direction", "") if s.get("direction", "") in DIRECTIONS else "",
                rationale=s.get("rationale", ""))
            for s in result.best.features
        }

        def ranked(model, data):
            attrib = attributions(model, data, max_samples=max_samples)
            names = getattr(model, "feature_names", None)
            fields = getattr(model, "feature_layout", None)
            by_field = {"engineered": list(extra)} if extra else None
            if names and fields:
                by_field, start = {}, 0
                for entry in fields:
                    width = entry.get("width") or 0
                    by_field[entry["key"]] = list(names[start:start + width])
                    start += width
                if "engineered" in by_field and extra:
                    by_field["engineered"] = list(extra)
            return summarize(attrib, self.task_card, by_field, top_k=top_k,
                             extra=extra)

        model = train_pipeline(config, self.space, train, val, self.metric,
                               self.seed, self.device)
        out = ranked(model, train)
        out["trial_id"] = result.best.trial_id
        if stability and self.folds is not None and out["features"]:
            tops = []
            everything = np.arange(len(train))
            for held_out in self.folds:
                part, _, part_val = fold_parts(
                    train, val, np.setdiff1d(everything, held_out).tolist(),
                    held_out.tolist())
                fold_model = train_pipeline(config, self.space, part, part_val,
                                            self.metric, self.seed, self.device)
                tops.append([f["name"] for f in ranked(fold_model, part)["features"]])
            out["stability"] = top_overlap([f["name"] for f in out["features"]], tops)
        if self.run_dir:
            os.makedirs(self.run_dir, exist_ok=True)
            with open(os.path.join(self.run_dir, "explain.json"), "w") as f:
                json.dump(out, f, indent=1, default=float)
        return out

    def flags(self) -> list[dict[str, Any]]:
        """Items a clinician should decide on before the model is used.

        Raised from the task card, :meth:`explain` and
        :meth:`evaluate_on_test`: a missing task card, a top feature with no
        entry in the card's dictionary, a learned effect opposite to the
        expected direction, poor calibration, and a large gap between
        subgroups. Each flag carries the decision recorded with
        :meth:`approve`, if any.

        Examples:
            >>> for flag in improver.flags():  # doctest: +SKIP
            ...     print(flag["id"], flag["detail"])
            direction:spo2_min learned +, expected -
        """
        from .report import flags_for

        review = self._read_json("review.json") or {}
        best = self.result().best
        suspicious = sorted({name for t in self.trials for name in t.suspicious
                             if best is not None and name in {
                                 s["name"] for s in best.features}})
        return [
            {**flag, "review": review.get(flag["id"])}
            for flag in flags_for(self.task_card, self._read_json("explain.json"),
                                  self._read_json("test.json"), suspicious)
        ]

    def _held_back(self, best: Trial | None) -> set[str]:
        """Engineered features of the selection with an open review flag."""
        if best is None or not best.features:
            return set()
        names = {s["name"] for s in best.features}
        held = set()
        for flag in self.flags():
            kind, _, name = flag["id"].partition(":")
            if (kind in ("direction", "no_rationale", "suspicion") and name in names
                    and (flag.get("review") or {}).get("decision") != "approved"):
                held.add(name)
        return held

    def approve(
        self,
        flag_id: str,
        reviewer: str,
        note: str = "",
        decision: str = "approved",
    ) -> dict[str, Any]:
        """Records a clinician's decision on a flag in ``review.json``.

        Args:
            flag_id: The ``id`` of a flag from :meth:`flags`.
            reviewer: Who decided.
            note: Their reasoning.
            decision: ``"approved"`` (keep as is) or ``"rejected"`` (change
                the model or the task before use).

        Examples:
            >>> improver.approve("direction:spo2_min", "Dr. A",  # doctest: +SKIP
            ...                  "SpO2 is charted only when low; known artifact")
        """
        if not self.run_dir:
            raise ValueError("approve needs a run_dir to record the decision")
        if decision not in ("approved", "rejected"):
            raise ValueError("decision must be 'approved' or 'rejected'")
        known = {flag["id"] for flag in self.flags()}
        if flag_id not in known:
            raise ValueError(f"no flag {flag_id!r}; current flags: {sorted(known)}")
        review = self._read_json("review.json") or {}
        review[flag_id] = {"decision": decision, "reviewer": reviewer, "note": note,
                           "date": time.strftime("%Y-%m-%d")}
        with open(os.path.join(self.run_dir, "review.json"), "w") as f:
            json.dump(review, f, indent=1)
        return review[flag_id]

    def report(self, path: str | None = None):
        """Writes the run's report: HTML, Markdown and JSON.

        The report explains the selected model and how the search got there:
        the model and its settings in plain words, why it was selected (the
        selection rule, its rung on the interpretability ladder and the
        price of choosing it over the highest-scoring candidate), what each
        changed setting contributed (if :meth:`attribute` was run), the test
        comparison (if :meth:`evaluate_on_test` was run), and the reasoning
        trace: every trial with its hypothesis, change, result and verdict.
        Statements are labelled by their evidence: measured, tested or
        hypothesis. It holds definitions, configs and aggregate scores only.

        Args:
            path: Where to write the HTML; ``.md`` and ``.json`` files are
                written next to it. Defaults to ``report.html`` in
                ``run_dir``; with neither, nothing is written.

        Returns:
            The :class:`~pyhealth.autoimprove.report.Report`.

        Examples:
            >>> improver.report("runs/mortality/report.html")  # doctest: +SKIP
        """
        from .report import build_report

        report = build_report(
            self.trials, metric=self.metric, maximize=self.maximize,
            promote_se=self.promote_se, one_se=self.one_se, space=self.space,
            task=self._task, lessons=self.lessons,
            test=self._read_json("test.json"),
            attribution=self._read_json("attribution.json"),
            dropped_models=self.dropped_models, rule=self.selection_rule(),
            card=self.task_card, explanation=self._read_json("explain.json"),
            review=self._read_json("review.json") or {},
            confirmed=self.confirmed, pruning=self._read_json("prune.json"),
            confirm_se=self.confirm_se, literature=self._literature(),
        )
        if path is None and self.run_dir:
            path = os.path.join(self.run_dir, "report.html")
        if path is not None:
            report.save(path)
        return report

    def _literature(self) -> dict[str, Any] | None:
        """The knowledge base's gap report and benchmark range (saved for load_run)."""
        if self.knowledge is None:
            return self._read_json("literature.json")
        out = {"gap": self.knowledge.gap_report(),
               "benchmark": self.knowledge.benchmark_range(),
               "rejected": sum(1 for e in self.knowledge.evidence
                               if e.verified is False),
               "verified": len(self.knowledge.records())}
        if self.run_dir:
            os.makedirs(self.run_dir, exist_ok=True)
            with open(os.path.join(self.run_dir, "literature.json"), "w") as f:
                json.dump(out, f, indent=1, default=str)
        return out

    def _read_json(self, name: str) -> Any:
        if not self.run_dir or not os.path.exists(os.path.join(self.run_dir, name)):
            return None
        with open(os.path.join(self.run_dir, name)) as f:
            return json.load(f)

    # ---------------------------------------------------------- persistence
    def _load(self) -> None:
        stored = self._read_lessons_file()
        if self._fingerprint in stored:
            self.lessons = list(stored[self._fingerprint]["lessons"])
        if not self.run_dir:
            return
        self._check_settings()
        trials_path = os.path.join(self.run_dir, "trials.jsonl")
        if os.path.exists(trials_path) and not self.trials:
            with open(trials_path) as f:
                self.trials = [Trial.from_dict(json.loads(line)) for line in f]
        lessons_file = os.path.join(self.run_dir, "lessons.json")
        if os.path.exists(lessons_file):
            with open(lessons_file) as f:
                self.lessons = list(json.load(f))
        confirmed = self._read_json("confirm.json")
        if confirmed and not self.confirmed:
            self.confirmed = {int(k): v for k, v in confirmed["scores"].items()}
            self.pruned_id = (self._read_json("prune.json") or {}).get("pruned_trial")

    def _run_settings(self) -> dict[str, Any]:
        """The settings a resumed run must share with the trials it continues."""
        settings = {
            "metric": self.metric,
            "cv_folds": self.cv_folds,
            "promote_se": self.promote_se,
            "seed": self.seed,
            "baseline": None if self.baseline is None else self.baseline.to_dict(),
            "space": self.space.describe(),
            "data": (
                f"inputs: {_schema_names(self.train_dataset.input_schema)} | "
                f"label: {_schema_names(self.train_dataset.output_schema)} | "
                f"{len(self.train_dataset)} training and "
                f"{len(self.val_dataset)} validation samples"
            ),
        }
        return json.loads(json.dumps(settings, default=str))

    def _check_settings(self) -> None:
        """Records the run's settings in ``meta.json``, or checks them on resume.

        Raises:
            ValueError: If ``run_dir`` holds a run with different settings.
        """
        path = os.path.join(self.run_dir, "meta.json")
        if os.path.exists(path):
            with open(path) as f:
                stored = json.load(f)
            changed = [k for k, v in self._settings.items()
                       if k in stored and stored[k] != v]
            if changed:
                raise ValueError(
                    f"{self.run_dir} holds a run with a different "
                    f"{', '.join(changed)}; resuming would mix its trials with "
                    "this run's. Use a new run_dir, or delete this one to start "
                    "over."
                )
            return
        if os.path.exists(os.path.join(self.run_dir, "trials.jsonl")):
            logger.warning(
                "%s has no meta.json, so its settings cannot be checked; the "
                "current ones are recorded", self.run_dir,
            )
        os.makedirs(self.run_dir, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._settings, f, indent=1)

    def _save_settings(self) -> None:
        """Writes run.json: what a report needs besides the trials."""
        if not self.run_dir:
            return
        os.makedirs(self.run_dir, exist_ok=True)
        settings = {
            "metric": self.metric, "maximize": self.maximize, "task": self._task,
            "cv_folds": self.cv_folds, "promote_se": self.promote_se,
            "one_se": self.one_se, "seed": self.seed,
            "models": sorted(self.space.models), "rule": self.selection_rule(),
            "dropped_models": self.dropped_models, "confirm_se": self.confirm_se,
        }
        with open(os.path.join(self.run_dir, "run.json"), "w") as f:
            json.dump(settings, f, indent=1)
        if self.task_card is not None:
            with open(os.path.join(self.run_dir, "card.json"), "w") as f:
                json.dump(self.task_card.to_dict(), f, indent=1)

    def _save_trial(self, trial: Trial) -> None:
        if not self.run_dir:
            return
        os.makedirs(self.run_dir, exist_ok=True)
        with open(os.path.join(self.run_dir, "trials.jsonl"), "a") as f:
            f.write(json.dumps(trial.to_dict(), default=str) + "\n")

    def _save_lessons(self) -> None:
        if self.run_dir:
            os.makedirs(self.run_dir, exist_ok=True)
            with open(os.path.join(self.run_dir, "lessons.json"), "w") as f:
                json.dump(self.lessons, f, indent=1)
        if self.lessons_path:
            stored = self._read_lessons_file()
            stored[self._fingerprint] = {
                "search": describe_search(self.space, self.train_dataset),
                "lessons": self.lessons,
            }
            with open(self.lessons_path, "w") as f:
                json.dump(stored, f, indent=1)

    def _read_lessons_file(self) -> dict[str, Any]:
        """Lessons per search fingerprint; an older plain list is set aside."""
        if not self.lessons_path or not os.path.exists(self.lessons_path):
            return {}
        with open(self.lessons_path) as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
        logger.warning(
            "%s holds lessons without a search fingerprint; they are not "
            "reused and will be replaced",
            self.lessons_path,
        )
        return {}


def describe_search(space: SearchSpace, dataset: SampleDataset) -> str:
    """The search a set of lessons belongs to: inputs, label and models.

    Examples:
        >>> describe_search(SearchSpace(), train_ds)  # doctest: +SKIP
        'inputs: conditions (sequence) | label: label (binary) | models: CNN(...), ...'
    """
    models = ", ".join(
        f"{name}({','.join(sorted(spec.params))})"
        for name, spec in sorted(space.models.items())
    )
    return (
        f"inputs: {_schema_names(dataset.input_schema)} | "
        f"label: {_schema_names(dataset.output_schema)} | models: {models}"
    )


def space_fingerprint(space: SearchSpace, dataset: SampleDataset) -> str:
    """A short stable id for :func:`describe_search`.

    Examples:
        >>> space_fingerprint(SearchSpace(), train_ds)  # doctest: +SKIP
        '3f9a1c0be27d'
    """
    return hashlib.sha256(describe_search(space, dataset).encode()).hexdigest()[:12]


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
        best = max(done, key=lambda t: sign * t.score)
        return best, best
    top = max(candidates, key=lambda t: sign * t.score)
    rung = rung or (lambda config: 0)

    def within(t: Trial) -> bool:
        if t is top:
            return True
        if len(t.fold_scores) != len(top.fold_scores):
            return False
        gap, se = paired_gain(top.fold_scores, t.fold_scores, maximize)
        return gap <= se

    near = [t for t in candidates if within(t)]
    # simpler wins: the lower rung, then fewer engineered features
    selected = min(near, key=lambda t: (rung(t.config), len(t.features),
                                        -sign * t.score))
    return selected, top


def _key(config: PipelineConfig, features: Sequence[dict[str, Any]]) -> str:
    """A trial's identity: its config and the names of its features."""
    names = sorted(str(s.get("name", "")) for s in features)
    return config.key() + ("|" + ",".join(names) if names else "")


def _feature_groups(best: Trial, trials: Sequence[Trial]) -> list[tuple[int, list[str]]]:
    """The selected trial's features, grouped by the accepted trial that added them."""
    groups: dict[int, list[str]] = {}
    for spec in best.features:
        name = spec["name"]
        owner = next((t.trial_id for t in trials if name in t.added and t.accepted),
                     -1 - len(groups))
        groups.setdefault(owner, []).append(name)
    return list(groups.items())


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


def describe_task(train_dataset: SampleDataset, val_dataset: SampleDataset) -> str:
    """A plain-text task summary for prompts: schema, sizes and label counts.

    Only aggregate numbers are included, never individual samples.

    Examples:
        >>> print(describe_task(train_ds, val_ds))  # doctest: +SKIP
    """
    lines = [
        f"Inputs: {_schema_names(train_dataset.input_schema)}",
        (
            f"Label: {_schema_names(train_dataset.output_schema)} "
            f"(type: {_label_mode(train_dataset)})"
        ),
        (
            f"Training samples: {len(train_dataset)}; validation samples: "
            f"{len(val_dataset)}"
        ),
    ]
    for key, proc in getattr(train_dataset, "input_processors", {}).items():
        try:
            lines.append(f"- {key}: {type(proc).__name__}, size {proc.size()}")
        except Exception:  # noqa: BLE001 - size() is optional on processors
            lines.append(f"- {key}: {type(proc).__name__}")
    if _label_mode(train_dataset) == "binary":
        positives = _positive_count(train_dataset)
        if positives is not None:
            lines.append(
                f"Training positives: {positives} of {len(train_dataset)} "
                f"({positives / max(len(train_dataset), 1):.1%})"
            )
    return "\n".join(lines)


def _label_mode(dataset: SampleDataset) -> str:
    """The label type, resolved from the output schema as models resolve it."""
    entry = next(iter(dataset.output_schema.values()))
    if isinstance(entry, tuple):  # (processor, kwargs)
        entry = entry[0]
    if isinstance(entry, str):
        return entry.lower()
    cls = entry if inspect.isclass(entry) else type(entry)
    for name, registered in PROCESSOR_REGISTRY.items():
        if issubclass(cls, registered):
            return name.lower()
    return "unknown"


def _schema_names(schema: dict) -> str:
    return ", ".join(
        f"{k} ({v if isinstance(v, str) else getattr(v, '__name__', type(v).__name__)})"
        for k, v in schema.items()
    )


def _positive_count(dataset: SampleDataset) -> int | None:
    key = next(iter(dataset.output_schema))
    if len(dataset) > MAX_LABEL_SCAN:
        return None
    try:
        return int(sum(float(dataset[i][key]) > 0.5 for i in range(len(dataset))))
    except Exception:  # noqa: BLE001 - labels that are not 0/1 scalars
        return None


def _better(new: float | None, old: float | None, maximize: bool) -> bool:
    if new is None:
        return False
    if old is None:
        return True
    return new > old if maximize else new < old
