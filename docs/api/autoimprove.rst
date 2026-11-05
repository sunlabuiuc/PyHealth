AutoImprove
===========

:mod:`pyhealth.autoimprove` improves a pipeline for you. You give it training
and validation splits and your current configuration. A few agents then take
turns proposing changes: the model, its arguments and the training settings.
Each proposal is trained on the training split and scored on the validation
split. The agents see each other's results and work through four strategies
in turn: explore, refine, crossover and contrast. The design follows
`codegen-classroom <https://github.com/google-research/codegen-classroom>`_.

Proposals come from an LLM (:class:`~pyhealth.autoimprove.LLMProposer`, with
:class:`~pyhealth.autoimprove.ClaudeLLM` or any ``llm(system, prompt) -> str``
callable), or from random search
(:class:`~pyhealth.autoimprove.RandomProposer`), which needs no API key.

After each cycle, the LLM proposer rewrites a short list of *lessons* from the
results. Later prompts build on those lessons, and so do later runs that pass
``lessons_path``. The search therefore improves the guidance it runs on, as
well as the pipeline.

.. code-block:: python

    from pyhealth.autoimprove import (
        AutoImprover, ClaudeLLM, LLMProposer, PipelineConfig,
    )
    from pyhealth.datasets import PatientSplit

    train, val, test = dataset.set_task(task, split=PatientSplit((0.7, 0.1, 0.2)))
    improver = AutoImprover(
        train, val, LLMProposer(ClaudeLLM()),
        baseline=PipelineConfig("RNN", {"hidden_dim": 128}, {"epochs": 10}),
        n_agents=3, n_cycles=5, run_dir="runs/mortality",
        cv_folds=5,                  # score by patient-grouped CV
    )
    result = improver.run()          # resumes if runs/mortality has trials
    print(result.summary())          # validation leaderboard
    improver.attribute()             # what each changed setting contributed
    print(improver.evaluate_on_test(test))  # selected vs baseline, once, on test
    improver.report()                # runs/mortality/report.html, .md, .json

Choosing a config you can trust
-------------------------------

A search ranks many configs on the same data, so the top score is partly
luck. On a small validation split this luck can dominate: the winner looks
best on validation and does worse on test. Three settings guard against it.

- ``cv_folds=5`` scores every trial by patient-grouped cross-validation over
  the training split (:func:`~pyhealth.autoimprove.patient_folds`). The
  validation split is then used only for early stopping. Each trial trains
  ``cv_folds`` models.
- ``promote_se=1.0`` (the default with ``cv_folds``) decides which trials
  are *candidates*. All trials share the same folds, so each is compared
  with the baseline fold by fold. A trial is a candidate if its gain has a
  lower bound above 0: the mean minus one standard error of the per-fold
  differences. A config with a higher mean but an inconsistent gain is not
  a candidate, and the baseline stays selected unless some trial qualifies.
- ``one_se=True`` (the default) picks among the candidates by the one-SE
  rule (:func:`~pyhealth.autoimprove.select_simplest`). The selection is the
  most interpretable candidate whose score is within one paired standard
  error of the best candidate's. Interpretability is a rung on a ladder
  (:meth:`~pyhealth.autoimprove.SearchSpace.rung`): 1 for logistic
  regression, whose coefficients read as odds ratios; 2 for depth-1 XGBoost,
  an additive model with one effect curve per input; 3 for shallow trees and
  MLPs; 4 for deeper trees and sequence models. When two models are within
  noise of each other, the one a clinician can check wins, and the report
  states what that choice cost. ``one_se=False`` selects the candidate with
  the highest lower bound instead (:func:`~pyhealth.autoimprove.select_trial`).

The run's report
----------------

Every trial is a record in the run's decision log, ``trials.jsonl``. It holds
the config with the proposer's rationale and expectation, both written
before the trial ran; the trial it was changed from (``parent``); what
changed (``change``); and the scores. LLM proposers are asked for the
expectation and the parent with every proposal.

- :meth:`~pyhealth.autoimprove.AutoImprover.attribute` measures what each
  changed setting of the selected trial contributed. It reverts the settings
  one at a time and scores each reverted config on the same folds. This
  answers "which of the five changes helped?" with a measurement, not a
  guess.
- :meth:`~pyhealth.autoimprove.AutoImprover.report` writes ``report.html``,
  ``report.md`` and ``report.json``. They contain:

  - the selected model and its settings in plain words;
  - why it was selected: the candidates, the ladder rung, and the price of
    interpretability against the best candidate;
  - the attribution;
  - the test comparison;
  - the reasoning trace: a timeline, every trial with its hypothesis, change,
    result and verdict, what worked and what didn't (from pairs of trials
    that differ in one setting), and the lessons.

  Every statement is labelled by its evidence. *Measured* means a score from
  the log. *Tested* means a controlled comparison. *Hypothesis* means a
  proposer's rationale or lesson. Numbers in the lessons that match nothing
  in the log are flagged
  (:func:`~pyhealth.autoimprove.unverified_numbers`).
- :func:`~pyhealth.autoimprove.load_run` rebuilds the report from a run
  directory alone, with no data or model. It also works on runs made before
  this version: the parent of each trial is then inferred and marked as
  such.

Lessons saved to ``lessons_path`` are stored per search, keyed by the inputs,
the label and the models in the search space. A run reads only the lessons
learned on the same kind of search, and the lessons may mention only models
that the space allows.

Models that cannot read your inputs
-----------------------------------

Not every model reads every input type. XGBoost, for example, needs
fixed-width columns and rejects code sequences. Before searching, the
improver gives each model in the space one training step on a few training
samples (:func:`~pyhealth.autoimprove.check_models`). Models that fail are
removed from the space, logged with the error, and listed in
``ImproveResult.dropped_models``, so no trial is spent on them. If the
baseline's model fails, ``run()`` stops with the error instead. Pass
``check=False`` to skip the check.

Safety and data
---------------

- **Proposals are data, never code.** Every proposal is a JSON config,
  checked against a :class:`~pyhealth.autoimprove.SearchSpace` before
  anything runs. Unknown models are rejected, unknown arguments are dropped,
  and out-of-range numbers are clipped. Nothing the LLM writes is executed.
- **Prompts hold aggregates only.** They contain the search space, the input
  and label schema, split sizes, the training positive count and validation
  scores. They never contain patient records. Your data use agreement still
  decides whether you may send even these to an external service.
- **The test split is never used during the search.** Validation scores are
  selected on many times, so they are optimistic, especially with few
  positive samples (the improver warns below 30).
  :meth:`~pyhealth.autoimprove.AutoImprover.evaluate_on_test` retrains the
  best and the baseline configs and compares them on test. For binary labels
  it reports patient-level bootstrap intervals and a paired difference.
  ``return_predictions=True`` also returns the test predictions, for other
  metrics without training again.

API Reference
-------------

.. automodule:: pyhealth.autoimprove
    :members: AutoImprover, ImproveResult, Trial, PipelineConfig, SearchSpace,
        ModelSpec, Param, Proposer, LLMProposer, RandomProposer, ClaudeLLM,
        with_xgboost, patient_folds, run_config_cv, select_trial, paired_gain,
        check_models, describe_error, select_simplest, config_diff, Report,
        build_report, load_run, unverified_numbers
    :show-inheritance:
