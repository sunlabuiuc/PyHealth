"""The improvement loop: propose, train, score, learn, repeat."""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import numpy as np

from pyhealth.datasets import SampleDataset
from pyhealth.tasks.task_card import TaskCard

from .evaluation import confirm, explain, guardrails, holdout, report
from .evaluation.cv import patient_folds, run_config, run_config_cv
from .evaluation.records import ImproveResult, Trial, _parent_of
from .evaluation.scoring import greater_is_better
from .evaluation.selection import _better, _rule_text, select_simplest, select_trial
from .evaluation.training import check_models
from .search import turns
from .search.context import ProposalContext
from .search.features import EngineeredDataset, FeatureSpace, FeatureSpec
from .search.prompts import STRATEGIES
from .search.proposers import Proposer
from .search.space import (
    PipelineConfig,
    SearchSpace,
    _derived,
    _key,
    config_diff,
    describe_search,
    space_fingerprint,
)
from .search.turns import GENERATED_KINDS, TURN_KINDS
from .utils.data import _label_mode, _positive_count, _schema_names, describe_task
from .utils.errors import describe_error
from .utils.sandbox import SandboxError
from .utils.store import RunStore

logger = logging.getLogger(__name__)

DEFAULT_METRIC = {
    "binary": "pr_auc",
    "multiclass": "f1_macro",
    "multilabel": "pr_auc_samples",
    "regression": "mse",
}
"""Metric optimised by default for each label type."""


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

    By default proposals are data, never code: nothing a proposer returns is
    executed. A ``cycle_plan`` with ``"code"``, ``"evaluation"`` or
    ``"analysis"`` turns opts in to generated code: a model with its training
    (:mod:`~pyhealth.autoimprove.search.codegen`), a new evaluation that can become
    a guardrail (:mod:`~pyhealth.autoimprove.evaluation.guardrails`), or an analysis of
    the training data (:mod:`~pyhealth.autoimprove.search.analysis`). That code is
    checked statically and runs in its own process
    (:mod:`~pyhealth.autoimprove.utils.sandbox`); only predictions, numbers and
    suppressed tables come back. Scripts that use these turns need an
    ``if __name__ == "__main__":`` guard.

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
        task: The task that built the samples. Code, evaluation and analysis
            prompts include its source code; ``task_card`` defaults to
            ``task.task_card()``.
        cycle_plan: The kind of each cycle, repeated in order: any of
            :data:`TURN_KINDS`, e.g. ``("analysis", "model", "code",
            "evaluation")``. Defaults to model cycles, alternating with
            feature cycles by ``feature_cycles`` when ``features`` is set.
            ``"features"`` needs ``features``; ``"evaluation"`` needs
            ``cv_folds``. Evaluation and analysis cycles add no trials, so
            they don't count towards ``patience``.
        code_timeout: Seconds one training of generated code may take (per
            fold).
        eval_timeout: Seconds one evaluation or analysis may take.
        max_guardrails: Most evaluations active as guardrails at once; more
            are kept for the report only.
        guardrail_se: Standard errors by which a trial must be worse than the
            baseline on a guardrail to be removed from the candidates.
        max_findings: Most analysis findings kept for prompts.

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
        task: Any = None,
        cycle_plan: Sequence[str] | None = None,
        code_timeout: float = 1800.0,
        eval_timeout: float = 300.0,
        max_guardrails: int = 3,
        guardrail_se: float = 1.0,
        max_findings: int = 12,
    ):
        if features is not None and not cv_folds:
            raise ValueError("feature search needs cv_folds")
        if task_card is None and task is not None and hasattr(task, "task_card"):
            task_card = task.task_card()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.proposer = proposer
        self.baseline = baseline
        self.space = space or SearchSpace()
        self.mode = _label_mode(train_dataset)
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
        self.store = RunStore(run_dir, lessons_path)
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
        self.task = task
        self.cycle_plan = self._plan(cycle_plan)
        self.code_timeout = code_timeout
        self.eval_timeout = eval_timeout
        self.max_guardrails = max_guardrails
        self.guardrail_se = guardrail_se
        self.max_findings = max_findings
        self.evaluations: list[dict[str, Any]] = []
        self.findings: list[str] = []
        self.analyses: list[dict[str, Any]] = []
        self._oof: dict[int, dict[str, Any] | None] = {}
        self._columns_cache: dict[str, np.ndarray] | None = None
        self._task_source: str | None = None
        self._fingerprint = space_fingerprint(self.space, train_dataset)
        if features is not None:
            self._fingerprint += "-features"
        generated = sorted(set(self.cycle_plan) & set(GENERATED_KINDS))
        if generated:
            self._fingerprint += "-" + "-".join(generated)
        self._settings = self._run_settings()
        self._proposer_failures = 0

    def _plan(self, cycle_plan: Sequence[str] | None) -> tuple[str, ...]:
        """The cycle plan, checked; the default reproduces feature_cycles."""
        if cycle_plan is None:
            if self.features is None:
                return ("model",)
            if not self.proposer.proposes_models:
                return ("features",)
            n_features, n_models = self.feature_cycles
            return ("features",) * n_features + ("model",) * n_models
        plan = tuple(cycle_plan)
        unknown = [k for k in plan if k not in TURN_KINDS]
        if not plan or unknown:
            raise ValueError(f"cycle_plan needs kinds from {list(TURN_KINDS)}; "
                             f"got {list(plan)}")
        if "features" in plan and self.features is None:
            raise ValueError('a "features" cycle needs features=FeatureSpace(...)')
        if "evaluation" in plan and not self.cv_folds:
            raise ValueError('an "evaluation" cycle needs cv_folds: guardrails '
                             "compare trials fold by fold")
        return plan

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
            # the selection rule depends on whether the baseline ran
            self._save_settings()
        first_cycle = max((t.cycle for t in self.trials), default=-1) + 1
        best, top = self._selected_id(), self._top_score()
        stale = 0
        strategies = list(STRATEGIES)
        for cycle in range(first_cycle, self.n_cycles):
            kind = self._cycle_kind(cycle)
            for i in range(self.n_agents):
                if self._out_of_budget(start) or self._exhausted:
                    break
                strategy = "explore" if cycle == 0 else strategies[
                    (cycle + i) % len(strategies)
                ]
                if kind == "features":
                    turns.feature_turn(self, cycle, f"agent{i}", strategy)
                elif kind == "evaluation":
                    turns.evaluation_turn(self, cycle, f"agent{i}")
                elif kind == "analysis":
                    turns.analysis_turn(self, cycle, f"agent{i}")
                else:
                    turns.model_turn(self, cycle, f"agent{i}", strategy, kind=kind)
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
            elif kind not in ("evaluation", "analysis"):  # these add no trials
                stale += 1
            best, top = new_best, new_top
            logger.info(
                "cycle %d: selected #%s, %s=%s", cycle, best, self.metric,
                self._best_score(),
            )
            if stale >= self.patience or self._out_of_budget(start) or self._exhausted:
                break
        if self.confirm_enabled and self.folds is not None and not self.confirmed:
            confirm.confirm(self)
            if self.prune_enabled and self.features is not None:
                confirm.prune(self)
        return self.result()

    def _cycle_kind(self, cycle: int) -> str:
        return self.cycle_plan[cycle % len(self.cycle_plan)]

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
        trials on their confirmation scores. Trials that fail an active
        guardrail (:meth:`guardrail_failures`) are not candidates.
        """
        # a feature group rejected during the search never becomes the result
        fails = self._guardrail_fails()
        trials = [t for t in self.trials if t.accepted is not False and not fails(t)]
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
        baseline = next((t for t in self.trials if t.agent == "baseline"), None)
        compared = self.baseline is not None and not (
            baseline is not None and baseline.error
        )
        return _rule_text(self.promote_se, self.one_se, compared)

    # ------------------------------------------------- stored predictions
    def _store_oof(self, trial_id: int, oof: dict[str, Any]) -> None:
        self._oof[trial_id] = oof
        self.store.save_arrays(f"preds/trial_{trial_id}.npz", index=oof["index"],
                               fold=oof["fold"], y_true=oof["y_true"],
                               y_prob=oof["y_prob"], split=np.array(oof["split"]))

    def oof_predictions(self, trial_id: int) -> dict[str, Any] | None:
        """A trial's out-of-fold predictions on the training split, if stored.

        Returns:
            ``{"index", "fold", "y_true", "y_prob", "split"}``
            (:func:`~pyhealth.autoimprove.evaluation.scoring.out_of_fold`), read from
            ``run_dir/preds`` for a resumed run, or ``None``.

        Examples:
            >>> improver.oof_predictions(3)["y_prob"].shape  # doctest: +SKIP
            (800, 1)
        """
        if trial_id not in self._oof:
            data = self.store.load_arrays(f"preds/trial_{trial_id}.npz")
            if data is not None:
                self._oof[trial_id] = {**data, "split": str(data["split"])}
        return self._oof.get(trial_id)

    def _columns(self) -> dict[str, np.ndarray]:
        """Sample columns of the training split, for evaluations and analyses."""
        if self._columns_cache is None:
            from .search.analysis import sample_columns

            self._columns_cache = sample_columns(self.train_dataset,
                                                 self._samples("train"))
        return self._columns_cache

    def _analysis_columns(self) -> dict[str, np.ndarray]:
        """The columns an analysis sees: sample columns, fold, the selection's
        out-of-fold prediction."""
        columns = dict(self._columns())
        if self.folds is not None:
            fold = np.zeros(len(self.train_dataset), dtype=np.int16)
            for k, held in enumerate(self.folds):
                fold[held] = k
            columns["fold"] = fold
        best = self.result().best
        oof = self.oof_predictions(best.trial_id) if best is not None else None
        if oof is not None and oof["split"] == "train":
            prob = np.asarray(oof["y_prob"], dtype=float)
            if prob.ndim == 1 or prob.shape[1] == 1:
                full = np.full(len(self.train_dataset), np.nan)
                full[oof["index"]] = prob.reshape(len(prob))
                columns["oof_prob"] = full
        return columns

    # ---------------------------------------------------------- guardrails
    def _baseline_id(self) -> int | None:
        base = next((t for t in self.trials if t.agent == "baseline"
                     and t.error is None), None)
        return None if base is None else base.trial_id

    def _guardrail_fails(self):
        from .evaluation.guardrails import admissible_filter

        if not any(e["status"] == "active" for e in self.evaluations):
            return lambda trial: []
        return admissible_filter(self.evaluations, self._baseline_id(),
                                 self.guardrail_se)

    def guardrail_failures(self) -> dict[int, list[str]]:
        """Trial id to the active guardrails it fails (``"<name>.<metric>"``).

        Examples:
            >>> improver.guardrail_failures()  # doctest: +SKIP
            {7: ['calibration.ece']}
        """
        fails = self._guardrail_fails()
        return {t.trial_id: fails(t) for t in self.trials
                if t.error is None and fails(t)}

    def _scorable(self) -> list[Trial]:
        """Finished trials with out-of-fold predictions on the training split."""
        out = []
        for t in self.trials:
            if t.error is not None or t.stopped_early:
                continue
            oof = self.oof_predictions(t.trial_id)
            if oof is not None and oof["split"] == "train":
                out.append(t)
        return out

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
            _derived(selected.config, "the selected config with the accepted features"),
            cycle, "incumbent", "incumbent", features=accepted, early_stop=False,
        )
        # A reference for paired gains: it must have every fold.
        return trial if trial.error is None and not trial.stopped_early else None

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
        return confirm.confirm(self)

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
        return confirm.prune(self)

    def _evaluate(self, config, cycle, agent, strategy, notes=None, features=None,
                  added=None, parent=None, save=True, early_stop=True) -> Trial:
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
        if config.code and self.run_dir:
            self.store.write_text(f"code/trial_{trial.trial_id}.py", config.code)
        try:
            out = self._run_config(
                config, self.report_metrics, trial.features,
                # references (incumbent, pruned) always run every fold
                stop_if=self._early_stop_rule() if early_stop else None,
                label=f"trial_{trial.trial_id}",
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
            elif out.get("oof") is not None:
                self._store_oof(trial.trial_id, out["oof"])
                guardrails.score_with_evaluations(self, trial)
        except SandboxError as e:  # generated code failed in its process
            trial.error, trial.error_brief = e.full, e.brief
        except Exception as e:  # noqa: BLE001 - a failed trial is feedback
            trial.error = describe_error(e)
            trial.error_brief = describe_error(e, message=False)
        self.trials.append(trial)
        if save:
            self._save_trial(trial)
        if self.evaluations:
            guardrails.save_evaluations(self)
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
        return turns.build_context(self, agent, strategy, feedback, kind=kind)

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
        return holdout.evaluate_on_test(self, test_dataset, n_boot=n_boot,
                                        return_predictions=return_predictions,
                                        subgroups=subgroups)

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
        return explain.attribute(self, trial_id=trial_id)

    def _run_config(
        self, config: PipelineConfig, report_metrics, features=(), stop_if=None,
        label: str = "trial",
    ) -> dict[str, Any]:
        """Scores a config, with these engineered features, on this run's
        folds or on the validation split. ``stop_if`` applies to folds only."""
        train, val = self._datasets(list(features))
        if self.folds is not None:
            return run_config_cv(
                config, self.space, train, val, self.metric, self.folds,
                report_metrics, self.seed, self.device, stop_if=stop_if,
                timeout=self.code_timeout, log_dir=self.store.log_dir(), label=label,
            )
        return run_config(
            config, self.space, train, val, self.metric, report_metrics,
            self.seed, self.device, timeout=self.code_timeout,
            log_dir=self.store.log_dir(), label=label,
        )

    def explain(
        self,
        top_k: int = 10,
        stability: bool = True,
        max_samples: int = 2000,
    ) -> dict[str, Any]:
        """Explains how the selected model decides, in clinical terms.

        The selected config is trained on the training split and its
        predictions on training samples are attributed to the inputs
        (:func:`~pyhealth.autoimprove.evaluation.explain.attributions`: exact TreeSHAP
        for trees, integrated gradients for torch models). The summary
        (:func:`~pyhealth.autoimprove.evaluation.explain.summarize`) gives:

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
        return explain.explain_selection(self, top_k=top_k, stability=stability,
                                         max_samples=max_samples)

    def flags(self) -> list[dict[str, Any]]:
        """Items a clinician should decide on before the model is used.

        Raised from the task card and the latest :meth:`explain` and
        :meth:`evaluate_on_test` results (kept in memory, or read from
        ``run_dir`` when resuming a saved run): a missing task card, a top feature with no
        entry in the card's dictionary, a learned effect opposite to the
        expected direction, poor calibration, and a large gap between
        subgroups. Each flag carries the decision recorded with
        :meth:`approve`, if any.

        Examples:
            >>> for flag in improver.flags():  # doctest: +SKIP
            ...     print(flag["id"], flag["detail"])
            direction:spo2_min learned +, expected -
        """
        return report.flags(self)

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
        return report.approve(self, flag_id, reviewer, note=note, decision=decision)

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
            The :class:`~pyhealth.autoimprove.evaluation.report.Report`.

        Examples:
            >>> improver.report("runs/mortality/report.html")  # doctest: +SKIP
        """
        return report.write_report(self, path=path)

    # ---------------------------------------------------------- persistence
    def _load(self) -> None:
        """Restores a run from its store: trials, lessons and later results."""
        shared = self.store.read_shared_lessons()
        if self._fingerprint in shared:
            self.lessons = list(shared[self._fingerprint]["lessons"])
        if not self.run_dir:
            return
        self.store.check_settings(self._settings)
        if not self.trials:
            self.trials = [Trial.from_dict(r) for r in self.store.read_trials()]
        lessons = self.store.read_json("lessons.json")
        if lessons is not None:
            self.lessons = list(lessons)
        confirmed = self.store.read_json("confirm.json")
        if confirmed and not self.confirmed:
            self.confirmed = {int(k): v for k, v in confirmed["scores"].items()}
            pruning = self.store.read_json("prune.json") or {}
            self.pruned_id = pruning.get("pruned_trial")
        if not self.evaluations:
            guardrails.load_evaluations(self)
        if not self.findings:
            self.findings = list(self.store.read_json("findings.json") or [])

    def _run_settings(self) -> dict[str, Any]:
        """The settings a resumed run must share with the trials it continues."""
        settings = {
            "metric": self.metric,
            "cv_folds": self.cv_folds,
            "promote_se": self.promote_se,
            "seed": self.seed,
            "baseline": None if self.baseline is None else self.baseline.to_dict(),
            "space": self.space.describe(rungs=False),
            "data": (
                f"inputs: {_schema_names(self.train_dataset.input_schema)} | "
                f"label: {_schema_names(self.train_dataset.output_schema)} | "
                f"{len(self.train_dataset)} training and "
                f"{len(self.val_dataset)} validation samples"
            ),
        }
        return json.loads(json.dumps(settings, default=str))

    def _save_settings(self) -> None:
        """Writes run.json: what a report needs besides the trials."""
        self.store.write_json("run.json", {
            "metric": self.metric, "maximize": self.maximize, "task": self._task,
            "cv_folds": self.cv_folds, "promote_se": self.promote_se,
            "one_se": self.one_se, "seed": self.seed,
            "models": sorted(self.space.models), "rule": self.selection_rule(),
            "dropped_models": self.dropped_models, "confirm_se": self.confirm_se,
        })
        if self.task_card is not None:
            self.store.write_json("card.json", self.task_card.to_dict())

    def _save_trial(self, trial: Trial) -> None:
        self.store.append_trial(trial.to_dict())

    def _save_lessons(self) -> None:
        self.store.save_lessons(self.lessons, self._fingerprint,
                                describe_search(self.space, self.train_dataset))
