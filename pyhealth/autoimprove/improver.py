"""The improvement loop: propose, train, score, learn, repeat.

:class:`AutoImprover` is the one entry point. It holds the run's
:class:`~pyhealth.autoimprove.state.RunState` and runs the loop; what each step
does lives in :mod:`~pyhealth.autoimprove.search` (what to try next) and
:mod:`~pyhealth.autoimprove.evaluation` (how good it is).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pyhealth.datasets import SampleDataset

from .config import RunConfig
from .evaluation import explain, holdout, report, selection
from .evaluation.records import ImproveResult, Trial
from .search import turns
from .search.features import FeatureSpace
from .search.proposers import Proposer
from .search.space import SearchSpace
from .state import DEFAULT_METRIC, RunState

__all__ = ["AutoImprover", "DEFAULT_METRIC"]


class AutoImprover:
    """Improves a PyHealth pipeline automatically, by guided search.

    Each round, a few agents propose attempts: a config from the search space,
    engineered input features, or a model they write themselves. Every
    attempt is trained and scored on the same patient-grouped folds (or the
    validation split), the proposer rewrites its lessons from the results, and
    the next round builds on them. The search stops after ``budget.rounds``,
    after ``budget.patience`` rounds without progress, or when the trial or
    time budget runs out. Among the attempts that beat the baseline, the most
    interpretable one within noise of the best is selected.

    Every setting is in a :class:`~pyhealth.autoimprove.RunConfig`, usually
    read from YAML; see that class for what each one does.

    The test split is never used during the search. Call
    :meth:`evaluate_on_test` once afterwards to compare the selected config
    with the baseline on held-out patients.

    By default proposals are data, never code. A plan with ``code`` rounds,
    or evaluation and analysis turns, opts in to generated code, which is
    checked statically and runs in its own process
    (:mod:`~pyhealth.autoimprove.utils.sandbox`). Scripts that use them need
    an ``if __name__ == "__main__":`` guard.

    Args:
        train_dataset: Training samples, e.g. from
            ``set_task(task, split=PatientSplit(...))``.
        val_dataset: Validation samples, used to score every trial (or, with
            ``evaluation.cv_folds``, only for early stopping).
        config: A :class:`~pyhealth.autoimprove.RunConfig`, a path to its
            YAML, a dict, or ``None`` for the defaults.
        proposer: Where proposals come from. Defaults to the config's
            ``proposer`` section (random search unless it says ``llm``).
        space: Allowed models and ranges. Defaults to the config's ``space``
            section, or ``SearchSpace()``.
        features: A :class:`~pyhealth.autoimprove.FeatureSpace` to search
            input features as well (needs ``evaluation.cv_folds``).
        task: The task that built the samples, or its
            :class:`~pyhealth.tasks.TaskCard`. Its card explains the model in
            clinical terms in reports; code prompts include its source.

    Examples:
        >>> from pyhealth.autoimprove import AutoImprover
        >>> improver = AutoImprover(train_ds, val_ds, "runs/mortality.yaml")  # doctest: +SKIP
        >>> result = improver.run()  # doctest: +SKIP
        >>> print(result.summary())  # doctest: +SKIP
        >>> improver.evaluate_on_test(test_ds)  # doctest: +SKIP
        >>> improver.report()  # doctest: +SKIP
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
        self.state = RunState(train_dataset, val_dataset, config, proposer=proposer,
                              space=space, features=features, task=task)
        self.config = self.state.config

    # ------------------------------------------------------------- the loop
    def run(self) -> ImproveResult:
        """Runs the search and returns its trials and lessons.

        Each round, every agent takes a turn of the round's kind (from
        ``search.plan``); then the proposer rewrites its lessons. The search
        stops after ``budget.rounds``, after ``budget.patience`` rounds without
        progress, or when the trial or time budget runs out.
        """
        s = self.state
        s.start()  # check models, resume, score the baseline
        stale = 0
        for cycle in s.rounds():
            kind = s.round_kind(cycle)
            for agent, strategy in s.agents(cycle):
                if s.stopped():
                    break
                turns.take_turn(s, cycle, kind, agent, strategy)
            if s.review(cycle, kind):
                stale = 0
            elif kind not in turns.ADDS_NO_TRIALS:
                stale += 1
            if stale >= s.patience or s.stopped():
                break
        s.finish()  # confirm and prune, if the config asks
        return s.result()

    def result(self) -> ImproveResult:
        """The run's trials, selection and lessons so far."""
        return self.state.result()

    # ------------------------------------------------------ what a run holds
    @property
    def trials(self) -> list[Trial]:
        return self.state.trials

    @property
    def lessons(self) -> list[str]:
        return self.state.lessons

    @property
    def findings(self) -> list[str]:
        """What analysis turns found in the training data."""
        return self.state.findings

    @property
    def evaluations(self) -> list[dict[str, Any]]:
        """Evaluations proposed so far, with their status and scores."""
        return self.state.evaluations

    @property
    def space(self) -> SearchSpace:
        return self.state.space

    @property
    def proposer(self) -> Proposer:
        return self.state.proposer

    @property
    def metric(self) -> str:
        return self.state.metric

    def selection_rule(self) -> str:
        """The rule the selection follows, in words."""
        return self.state.selection_rule()

    def oof_predictions(self, trial_id: int) -> dict[str, Any] | None:
        """A trial's out-of-fold predictions on the training split, if stored."""
        return self.state.oof_predictions(trial_id)

    def guardrail_failures(self) -> dict[int, list[str]]:
        """Trials removed from the candidates, with the guardrails they fail."""
        return self.state.guardrail_failures()

    # --------------------------------------------------- after the search
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
        return selection.confirm(self.state)

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
        return selection.prune(self.state)

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
        return holdout.evaluate_on_test(self.state, test_dataset, n_boot=n_boot,
                                        return_predictions=return_predictions,
                                        subgroups=subgroups)

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
        return explain.attribute(self.state, trial_id=trial_id)

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
        return explain.explain_selection(self.state, top_k=top_k, stability=stability,
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
        return report.flags(self.state)

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
        return report.approve(self.state, flag_id, reviewer, note=note, decision=decision)

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
        return report.write_report(self.state, path=path)
