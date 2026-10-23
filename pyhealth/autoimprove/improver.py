"""The improvement loop: propose, train, score, learn, repeat."""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import math
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pyhealth.datasets import SampleDataset
from pyhealth.metrics import bootstrap_ci, paired_bootstrap_diff
from pyhealth.processors import PROCESSOR_REGISTRY
from pyhealth.trainer import get_metrics_fn

from .evaluate import (
    MAX_LABEL_SCAN,
    check_models,
    describe_error,
    greater_is_better,
    patient_folds,
    predict,
    run_config,
    run_config_cv,
    train_pipeline,
)
from .proposers import STRATEGIES, ProposalContext, Proposer
from .space import PipelineConfig, SearchSpace

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


@dataclass
class Trial:
    """One proposed configuration and how it did on validation.

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

    def to_dict(self) -> dict[str, Any]:
        """The trial as JSON-ready types."""
        data = dict(self.__dict__)
        data["config"] = {**self.config.to_dict(), "rationale": self.config.rationale}
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Trial:
        """Rebuilds a trial saved with :meth:`to_dict`."""
        data = dict(data)
        data["config"] = PipelineConfig.from_dict(data["config"])
        return cls(**data)


@dataclass
class ImproveResult:
    """The outcome of :meth:`AutoImprover.run`.

    Args:
        trials: Every trial, in the order run.
        best: The selected trial: the best score, or with paired selection
            (``cv_folds``, ``promote_se`` and a baseline) the trial with the
            highest lower bound on its fold-by-fold gain over the baseline.
        baseline: The trial of the starting config, if one was given.
        lessons: The lessons at the end of the run.
        maximize: Whether larger scores are better.
        dropped_models: Models removed from the space before the search
            because they did not run on the training data, with the error.

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

    def leaderboard(self, k: int = 10) -> list[Trial]:
        """The ``k`` best finished trials, best first."""
        done = [t for t in self.trials if t.error is None]
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
            baseline on the shared folds, and the selection is the trial
            whose gain has the highest lower bound, mean minus
            ``promote_se`` standard errors. The baseline stays selected
            unless some bound is above 0. ``None`` selects the best mean, as
            does a run without a baseline (or whose baseline trial failed),
            since there is nothing to compare with.
        check: Before searching, give every model in the space one training
            step on a few training samples (:func:`check_models`) and remove
            the ones that fail, e.g. a model that cannot read one of the
            task's input types, so no trial is spent on them. They are listed
            in ``ImproveResult.dropped_models``.

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
        >>> improver.evaluate_on_test(test_ds)  # doctest: +SKIP
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
        check: bool = True,
    ):
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
        if self.promote_se is not None and baseline is None:
            logger.warning(
                "promote_se compares every trial with the baseline, and no "
                "baseline was given: the best mean score is selected instead"
            )
        self.folds = (
            patient_folds(train_dataset, cv_folds, seed) if cv_folds else None
        )
        self.check = check
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
        self._settings = self._run_settings()
        self._proposer_failures = 0

    # ------------------------------------------------------------------ run
    def run(self) -> ImproveResult:
        """Runs the search and returns its trials and lessons."""
        if self.check and not self._checked:
            self._check_models()
        self._load()
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
        strategies = list(STRATEGIES)
        for cycle in range(first_cycle, self.n_cycles):
            for i in range(self.n_agents):
                if self._out_of_budget(start):
                    break
                strategy = "explore" if cycle == 0 else strategies[
                    (cycle + i) % len(strategies)
                ]
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
            if stale >= self.patience or self._out_of_budget(start):
                break
        return self.result()

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
        """The current state as an :class:`ImproveResult`."""
        best = select_trial(self.trials, self.maximize, self.promote_se)
        baseline = next((t for t in self.trials if t.agent == "baseline"), None)
        return ImproveResult(
            list(self.trials), best, baseline, list(self.lessons), self.maximize,
            dict(self.dropped_models),
        )

    def _turn(self, cycle: int, agent: str, strategy: str) -> None:
        feedback = ""
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
            if config.key() in context.tried:
                feedback = "your last proposal repeated an earlier trial"
                if notes:
                    feedback += (
                        " once it was changed to fit the search space ("
                        + "; ".join(notes)
                        + "). Stay inside the ranges listed above"
                    )
                continue
            self._evaluate(config, cycle, agent, strategy, notes)
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

    def _evaluate(self, config, cycle, agent, strategy, notes=None) -> Trial:
        trial = Trial(len(self.trials), cycle, agent, strategy, config,
                      metric_name=self.metric, notes=list(notes or []))
        try:
            if self.folds is not None:
                out = run_config_cv(
                    config, self.space, self.train_dataset, self.val_dataset,
                    self.metric, self.folds, self.report_metrics, self.seed,
                    self.device,
                )
            else:
                out = run_config(
                    config, self.space, self.train_dataset, self.val_dataset,
                    self.metric, self.report_metrics, self.seed, self.device,
                )
            trial.metrics = out["metrics"]
            trial.score = out["metrics"][self.metric]
            trial.seconds = out["seconds"]
            trial.n_params = out["n_params"]
            trial.fold_scores = out.get("fold_scores", [])
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
        self._save_trial(trial)
        return trial

    def _context(self, agent, strategy, feedback) -> ProposalContext:
        result = self.result()
        done = result.leaderboard(k=len(self.trials))
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
            tried={t.config.key() for t in self.trials},
            lessons=self.lessons,
            feedback=feedback,
            selected=result.best,
            selection_rule=self._selection_rule(result.baseline),
        )

    def _selection_rule(self, baseline: Trial | None) -> str:
        if self.promote_se is None:
            return ""
        if baseline is None or baseline.error is not None:
            return ("There is no baseline to compare with, so the selection is "
                    "the config with the best mean score.")
        return (
            "The selection is the config whose fold-by-fold gain over the "
            f"baseline has the highest lower bound (mean minus "
            f"{self.promote_se:g} standard errors), so a gain must be "
            "consistent across folds, not only high on average."
        )

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
        return_predictions: bool = False,
    ) -> dict[str, Any]:
        """Retrains the best and the baseline configs and scores them on test.

        Run this once, after the search. For binary labels the result holds
        patient-level bootstrap intervals (``bootstrap_ci``) and a paired
        bootstrap interval for best minus baseline (``paired_bootstrap_diff``).
        Other label types get the metric's value on test.

        Args:
            test_dataset: Held-out samples, never seen by the search.
            n_boot: Bootstrap resamples.
            return_predictions: Also return the test predictions, e.g. to
                compute other metrics without training again.

        Returns:
            ``{"best": {...}, "baseline": {...}, "difference": {...}}``;
            ``baseline`` and ``difference`` are present when a baseline was
            given, and ``difference`` only for binary labels. With
            ``return_predictions``, ``"predictions"`` holds ``y_true``,
            ``patient_ids`` and each config's ``y_prob`` under its name.

        Examples:
            >>> improver.evaluate_on_test(test_ds)  # doctest: +SKIP
        """
        result = self.result()
        if result.best is None:
            raise ValueError("no successful trial to evaluate")
        configs = {"best": result.best.config}
        if result.baseline is not None and result.baseline.error is None:
            configs["baseline"] = result.baseline.config
        probs, out = {}, {}
        for name, config in configs.items():
            model = train_pipeline(
                config, self.space, self.train_dataset, self.val_dataset,
                self.metric, self.seed, self.device,
            )
            y_true, y_prob, patient_ids = predict(model, test_dataset)
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
        if "baseline" in probs and self.mode == "binary":
            out["difference"] = paired_bootstrap_diff(
                y_true, probs["best"], probs["baseline"], self.metric,
                groups=groups, n_boot=n_boot, seed=self.seed,
            )
        if return_predictions:
            out["predictions"] = {"y_true": y_true, "patient_ids": patient_ids, **probs}
        return out

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
