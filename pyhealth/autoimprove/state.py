"""What one run holds: settings, data, trials, lessons, and its store.

:class:`RunState` is created by :class:`~pyhealth.autoimprove.AutoImprover`
and passed to the search and evaluation functions, so none of them needs the
improver itself. Its methods are the steps that read or change the run as a
whole: scoring a config into a trial, the datasets with accepted features,
resuming from and saving to ``run_dir``, and the loop's start, review and
finish.
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Any

import numpy as np

from pyhealth.datasets import SampleDataset
from pyhealth.tasks.task_card import TaskCard

from .config import RunConfig
from .evaluation import guardrails, selection
from .evaluation.guardrails import admissible_filter
from .evaluation.records import ImproveResult, Trial, _parent_of
from .evaluation.selection import (
    _better,
    _rule_text,
    select_simplest,
    select_trial,
    signal_check,
)
from .evaluation.training import (
    EarlyStop,
    chance_fold_scores,
    check_models,
    greater_is_better,
    patient_folds,
    run_config,
    run_config_cv,
)
from .search import turns
from .search.features import EngineeredDataset, FeatureSpace, FeatureSpec
from .search.prompts import STRATEGIES
from .search.proposers import ProposalContext, Proposer
from .search.space import (
    PipelineConfig,
    SearchSpace,
    _derived,
    _key,
    config_diff,
    describe_search,
    space_fingerprint,
)
from .search.turns import PLAN_KINDS
from .utils import data
from .utils.data import (
    _label_mode,
    _positive_count,
    _schema_names,
    describe_error,
    describe_task,
)
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


class RunState:
    """Everything one run holds, and the steps that change it.

    :class:`~pyhealth.autoimprove.AutoImprover` creates one and hands it to
    the search and evaluation functions, which read and update it: the
    settings, the datasets and folds, the trials, lessons, guardrails and
    findings, and the run's store. Its arguments are AutoImprover's.
    """

    def __init__(
        self,
        train_dataset: SampleDataset,
        val_dataset: SampleDataset,
        config: RunConfig | str | dict | None = None,
        proposer: Proposer | None = None,
        space: SearchSpace | None = None,
        features: FeatureSpace | None = None,
        task: Any = None,
    ):
        self.config = cfg = RunConfig.load(config)
        search, budget, ev = cfg.search, cfg.budget, cfg.evaluation
        cv_folds = ev.cv_folds
        if features is not None and not cv_folds:
            raise ValueError("feature search needs evaluation.cv_folds")
        if isinstance(task, TaskCard):
            task_card, task = task, None
        else:
            task_card = task.task_card() if hasattr(task, "task_card") else None
        baseline = cfg.baseline.to_pipeline() if cfg.baseline is not None else None
        if space is None:
            space = cfg.space.build() if cfg.space is not None else SearchSpace()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.proposer = proposer if proposer is not None else cfg.proposer.build()
        self.baseline = baseline
        self.space = space
        self.mode = _label_mode(train_dataset)
        self.metric = search.metric or DEFAULT_METRIC.get(self.mode, "loss")
        self.maximize = greater_is_better(self.metric)
        self.report_metrics = list(search.report_metrics)
        self.n_agents = search.proposals_per_round
        self.batched = search.mode == "batched"
        self.workers = search.workers
        self.instructions = search.instructions
        self.strategies = list(search.strategies or STRATEGIES)
        self.n_cycles = budget.rounds
        self.patience = budget.patience
        self.max_trials = budget.max_trials
        self.time_budget = budget.time_budget
        self.seed = search.seed
        self.device = search.device
        self.run_dir = cfg.output.run_dir
        self.lessons_path = cfg.output.lessons_path
        self.store = RunStore(self.run_dir, self.lessons_path)
        self.cv_folds = cv_folds
        self.promote_se = ev.promote_se if cv_folds else None
        self.one_se = ev.one_se
        if self.promote_se is not None and baseline is None:
            logger.warning(
                "promote_se compares every trial with the baseline, and no "
                "baseline was given: every finished trial is a candidate, and %s",
                "the one-SE rule picks among them" if ev.one_se
                else "the best mean score is selected",
            )
        self.folds = (
            patient_folds(train_dataset, cv_folds, self.seed) if cv_folds else None
        )
        self.check = ev.check
        self.task_card = task_card
        self.features = features
        self.confirm_enabled = cfg.confirm.enabled
        self.confirm_top = cfg.confirm.top
        self.confirm_seeds = cfg.confirm.seeds
        self.confirm_se = cfg.confirm.se
        self.prune_enabled = cfg.confirm.prune
        self.early_stop = budget.early_stop
        self.suspicion_auc = ev.suspicion_auc
        self.min_events_per_fold = ev.min_events_per_fold
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
        self.guardrails_enabled = cfg.guardrails.enabled
        self.guardrail_every = cfg.guardrails.every
        self.guardrail_turns = cfg.guardrails.turns
        self.analysis_enabled = cfg.analysis.enabled
        self.analysis_every = cfg.analysis.every
        self.analysis_turns = cfg.analysis.turns
        if self.guardrails_enabled and not cv_folds:
            raise ValueError("guardrails need evaluation.cv_folds: they compare "
                             "trials fold by fold")
        self.cycle_plan = self._plan(search.plan)
        self.code_timeout = cfg.code.timeout
        self.eval_timeout = cfg.guardrails.timeout
        self.max_guardrails = cfg.guardrails.max
        self.guardrail_se = cfg.guardrails.se
        self.max_findings = cfg.analysis.max_findings
        self.evaluations: list[dict[str, Any]] = []
        self.findings: list[str] = []
        self.analyses: list[dict[str, Any]] = []
        self._oof: dict[int, dict[str, Any] | None] = {}
        self._columns_cache: dict[str, np.ndarray] | None = None
        self._task_source: str | None = None
        self._fingerprint = space_fingerprint(self.space, train_dataset)
        if features is not None:
            self._fingerprint += "-features"
        generated = sorted({"code"} & set(self.cycle_plan)
                           | ({"evaluation"} if self.guardrails_enabled else set())
                           | ({"analysis"} if self.analysis_enabled else set()))
        if generated:
            self._fingerprint += "-" + "-".join(generated)
        self._settings = self._run_settings()
        self._proposer_failures = 0

    def _plan(self, plan: Sequence[str] | None) -> tuple[str, ...]:
        """The plan, checked: two feature rounds then a model round by default."""
        if plan is None:
            if self.features is None:
                return ("model",)
            if not self.proposer.proposes_models:
                return ("features",)
            return ("features", "features", "model")
        plan = tuple(plan)
        tools = [k for k in plan if k in ("evaluation", "analysis")]
        if tools:
            raise ValueError(
                f"{tools} are not kinds of round: enable them with "
                "guardrails.enabled / analysis.enabled instead")
        unknown = [k for k in plan if k not in PLAN_KINDS]
        if not plan or unknown:
            raise ValueError(f"search.plan needs kinds from {list(PLAN_KINDS)}; "
                             f"got {list(plan)}")
        if "features" in plan and self.features is None:
            raise ValueError('a "features" round needs features=FeatureSpace(...)')
        return plan

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
        if self.mode != "binary":
            return
        if self.cv_folds:
            events = self._fold_events()
            if events and min(events) < self.min_events_per_fold:
                logger.warning(
                    "The %d folds have %s positive samples. With fewer than %d "
                    "per fold, one lucky fold can carry a trial's whole gain: "
                    "treat the search's gains as unconfirmed until "
                    "evaluate_on_test (or more events) supports them.",
                    len(events), ", ".join(map(str, events)),
                    self.min_events_per_fold,
                )
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

    def _fold_events(self) -> list[int] | None:
        """Positive samples in each CV fold (binary tasks with CV only)."""
        if (self.mode != "binary" or not self.folds
                or len(self.train_dataset) > data.MAX_LABEL_SCAN):
            return None
        labels = self._labels()
        return [int((labels[fold] > 0.5).sum()) for fold in self.folds]

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
            (:func:`~pyhealth.autoimprove.evaluation.training.out_of_fold`), read from
            ``run_dir/preds`` for a resumed run, or ``None``.

        Examples:
            >>> improver.oof_predictions(3)["y_prob"].shape  # doctest: +SKIP
            (800, 1)
        """
        if trial_id not in self._oof:
            arrays = self.store.load_arrays(f"preds/trial_{trial_id}.npz")
            if arrays is not None:
                self._oof[trial_id] = {**arrays, "split": str(arrays["split"])}
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

    def _evaluate(self, config, cycle, agent, strategy, notes=None, features=None,
                  added=None, parent=None, save=True, early_stop=True) -> Trial:
        """Scores one config, with these engineered features, as a new trial."""
        trial = self._new_trial(config, cycle, agent, strategy, notes, features, added,
                                parent)
        return self._record(trial, self._score(trial, early_stop=early_stop), save)

    def evaluate_batch(self, proposals, cycle: int, features=None) -> list[Trial]:
        """Scores a round's proposals, ``workers`` at a time, as new trials.

        Trial ids follow the proposals' order, and each trial's parent and
        reference are those before the round, so the result does not depend
        on which training finishes first.

        Args:
            proposals: ``(agent, strategy, config, notes)`` tuples.
        """
        trials = []
        for k, (agent, strategy, config, notes) in enumerate(proposals):
            trial = self._new_trial(config, cycle, agent, strategy, notes, features,
                                    trial_id=len(self.trials) + k)
            trials.append(trial)
        workers = min(self.workers, len(trials)) or 1
        stop_if = self._early_stop_rule()
        if workers == 1:
            outcomes = [self._score(t, stop_if=stop_if) for t in trials]
        else:
            with ThreadPoolExecutor(workers) as pool:
                outcomes = list(pool.map(
                    lambda t: self._score(t, stop_if=stop_if, isolate=True), trials))
        return [self._record(t, out) for t, out in zip(trials, outcomes)]

    def _new_trial(self, config, cycle, agent, strategy, notes=None, features=None,
                   added=None, parent=None, trial_id=None) -> Trial:
        trial = Trial(len(self.trials) if trial_id is None else trial_id, cycle, agent,
                      strategy, config, metric_name=self.metric,
                      notes=list(notes or []),
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
        return trial

    def _score(self, trial, early_stop=True, stop_if=None, isolate=False):
        """Trains and scores a trial's config: the result, or the exception."""
        if stop_if is None and early_stop:
            stop_if = self._early_stop_rule()
        try:
            return self._run_config(
                trial.config, self.report_metrics, trial.features,
                # references (incumbent, pruned) always run every fold
                stop_if=stop_if if early_stop else None,
                label=f"trial_{trial.trial_id}", isolate=isolate,
            )
        except Exception as e:  # noqa: BLE001 - a failed trial is feedback
            return e

    def _record(self, trial, out, save=True) -> Trial:
        """Fills a trial in from its result (or its error) and keeps it."""
        try:
            if isinstance(out, Exception):
                raise out
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
        return EarlyStop(reference.fold_scores, self.early_stop, self.maximize)

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

    # ------------------------------------------------------- the run's steps
    def start(self) -> None:
        """Checks the models, resumes from run_dir, and scores the baseline."""
        if self.check and not self._checked:
            self._check_models()
        self._load()
        self._save_settings()
        self._warn_if_small_validation()
        self._started = time.perf_counter()
        self._progress = (self._selected_id(), self._top_score())
        if self.baseline is not None and not any(
            t.agent == "baseline" for t in self.trials
        ):
            config, notes = self.space.validate(self.baseline)
            self._evaluate(config, -1, "baseline", "baseline", notes)
            # the selection rule depends on whether the baseline ran
            self._save_settings()
            self._progress = (self._selected_id(), self._top_score())

    def rounds(self) -> range:
        """The rounds still to run: a resumed run continues after its last."""
        first = max((t.cycle for t in self.trials), default=-1) + 1
        return range(first, self.n_cycles)

    def round_kind(self, cycle: int) -> str:
        return self.cycle_plan[cycle % len(self.cycle_plan)]

    def agents(self, cycle: int) -> list[tuple[str, str]]:
        """Each agent's name and strategy this round: all explore at first,
        then they rotate through the strategies."""
        strategies = self.strategies
        return [(f"agent{i}", "explore" if cycle == 0
                 else strategies[(cycle + i) % len(strategies)])
                for i in range(self.n_agents)]

    def stopped(self) -> bool:
        """The trial or time budget is spent, or the proposer has run out."""
        return self._out_of_budget(self._started) or self._exhausted

    def review(self, cycle: int, kind: str) -> bool:
        """Updates the lessons after a round; returns whether it made progress:
        a new selection, or a better best score."""
        try:
            self.lessons = self.proposer.reflect(
                self.trials, self.lessons, self.metric, space=self.space
            )
        except Exception as e:  # noqa: BLE001 - keep the lessons we have
            logger.warning("lessons not updated in cycle %d: %s", cycle,
                           describe_error(e))
        self._save_lessons()
        best, top = self._progress
        new_best, new_top = self._selected_id(), self._top_score()
        progressed = (new_best is not None and new_best != best) or _better(
            new_top, top, self.maximize)
        self._progress = (new_best, new_top)
        logger.info("cycle %d: selected #%s, %s=%s", cycle, new_best, self.metric,
                    self._best_score())
        return progressed

    def finish(self) -> None:
        """Confirms the leaders on new folds and seeds, and prunes features,
        when the config asks for it; then compares the selection and the
        baseline with chance (``signal.json``)."""
        if self.confirm_enabled and self.folds is not None and not self.confirmed:
            selection.confirm(self)
            if self.prune_enabled and self.features is not None:
                selection.prune(self)
        self._check_signal()

    def _check_signal(self) -> None:
        """Compares the selection and baseline with chance (binary CV only).

        Saves ``signal.json``, read by :meth:`flags` and :meth:`report`, and
        warns when the gains may not reflect any signal in the inputs.
        """
        # A signal.json from an earlier run in this run_dir no longer applies.
        self.store.remove("signal.json")
        best = self.result().best
        if self.mode != "binary" or self.folds is None or best is None:
            return
        if len(self.train_dataset) > data.MAX_LABEL_SCAN:
            # Reading every label would load every training sample.
            logger.info("Skipped the chance check: more than %d training samples.",
                        data.MAX_LABEL_SCAN)
            return
        base = next((t for t in self.trials
                     if t.agent == "baseline" and t.fold_scores), None)
        folds, selected = self.folds, best.fold_scores
        base_scores = base.fold_scores if base else None
        if best.trial_id in self.confirmed and (
            base is None or base.trial_id in self.confirmed
        ):
            # Compare on the confirmation folds: fresh scores for both, and
            # chance on the same partition.
            folds = patient_folds(self.train_dataset, len(self.folds), self.seed + 1)
            selected = self.confirmed[best.trial_id]
            base_scores = self.confirmed[base.trial_id] if base else None
        chance = chance_fold_scores(self._labels(), folds, self.metric)
        if chance is None or not self.maximize:
            return
        check = signal_check(chance, selected, base_scores, self.metric)
        if check is None:
            return
        self.store.save_result("signal.json", check, str)
        if check["no_signal"]:
            logger.warning("No signal check: %s", check["detail"])

    def _run_config(
        self, config: PipelineConfig, report_metrics, features=(), stop_if=None,
        label: str = "trial", isolate: bool = False,
    ) -> dict[str, Any]:
        """Scores a config, with these engineered features, on this run's
        folds or on the validation split. ``stop_if`` applies to folds only."""
        train, val = self._datasets(list(features))
        if self.folds is not None:
            return run_config_cv(
                config, self.space, train, val, self.metric, self.folds,
                report_metrics, self.seed, self.device, stop_if=stop_if,
                timeout=self.code_timeout, log_dir=self.store.log_dir(), label=label,
                isolate=isolate,
            )
        return run_config(
            config, self.space, train, val, self.metric, report_metrics,
            self.seed, self.device, timeout=self.code_timeout,
            log_dir=self.store.log_dir(), label=label, isolate=isolate,
        )





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
        self.store.write_text("config.yaml", self.config.to_yaml())
        self.store.write_json("run.json", {
            "metric": self.metric, "maximize": self.maximize, "task": self._task,
            "cv_folds": self.cv_folds, "promote_se": self.promote_se,
            "one_se": self.one_se, "seed": self.seed,
            "models": sorted(self.space.models), "rule": self.selection_rule(),
            "dropped_models": self.dropped_models, "confirm_se": self.confirm_se,
            "fold_events": self._fold_events(),
            "min_events_per_fold": self.min_events_per_fold,
        })
        if self.task_card is not None:
            self.store.write_json("card.json", self.task_card.to_dict())

    def _save_trial(self, trial: Trial) -> None:
        self.store.append_trial(trial.to_dict())

    def _save_lessons(self) -> None:
        self.store.save_lessons(self.lessons, self._fingerprint,
                                describe_search(self.space, self.train_dataset))
