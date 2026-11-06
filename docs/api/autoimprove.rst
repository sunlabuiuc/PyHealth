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

Explaining the model in clinical terms
--------------------------------------

A model a clinician can't check won't be used. Give the improver the task's
card (:class:`~pyhealth.tasks.TaskCard`, from ``task.task_card()``). The card
records the design choices with their rationale and has a dictionary of the
features: meaning, clinical group and expected direction of effect. Then:

.. code-block:: python

    improver = AutoImprover(train, val, proposer, ..., task_card=task.task_card())
    result = improver.run()
    improver.attribute()       # what each changed setting contributed
    improver.explain()         # how the selected model decides
    improver.evaluate_on_test(test, subgroups={"sex": sex, "age band": band})
    for flag in improver.flags():       # items for a clinician
        print(flag["id"], flag["detail"])
    improver.approve("direction:spo2_n", reviewer="...", note="...")
    improver.report()

- :meth:`~pyhealth.autoimprove.AutoImprover.explain` attributes the selected
  model's predictions to its inputs. Trees use exact TreeSHAP; torch models
  use integrated gradients, which are exact for logistic regression. It
  reports:

  - importance by clinical factor group;
  - the top features with their meaning;
  - the shape of each one's effect, in bins of at least 11 patients;
  - each feature's learned direction, checked against the card's expected
    direction;
  - how many top features the fold models share.
- :meth:`~pyhealth.autoimprove.AutoImprover.evaluate_on_test` adds clinical
  performance (:mod:`pyhealth.metrics.clinical`):

  - calibration;
  - what flagging the top 5, 10 or 20% of risk would catch;
  - net benefit;
  - results within each subgroup.
- :meth:`~pyhealth.autoimprove.AutoImprover.flags` lists what a clinician
  should decide on before the model is used:

  - a missing task card;
  - a top feature the card doesn't describe;
  - a learned effect opposite to the expected one, which can come from a
    wrong expectation, confounding by indication, overlap with a related
    feature, a coding artifact, a data error or leakage;
  - poor calibration;
  - a large gap between subgroups.

  :meth:`~pyhealth.autoimprove.AutoImprover.approve` records each decision
  in ``review.json``, and the report shows it.
- The report then also has *How it decides*, *Performance in clinical terms*,
  *Clinician review*, and a checklist of where it covers the main items of
  the TRIPOD+AI reporting guideline (Collins et al., BMJ 2024).

Reports and saved files hold aggregates only. Every bin and subgroup count
below 11 patients is suppressed, and effect-bin cut points are midpoints, not
patient values.

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

Features as data
----------------

New inputs are written as specs (JSON), never as code
(:class:`~pyhealth.autoimprove.FeatureSpace`). A spec combines a fixed set of
operators over PyHealth's event log:

- ``count``, ``exists``, ``days_since``, ``stat`` (min, max, mean, last,
  slope), ``n_distinct``, ``trend`` and ``blocks_with``, each over events
  matching a selector in a window that ends strictly before the prediction
  time t;
- ``ratio``, ``diff``, ``band`` and ``ge`` thresholds, ``any``, ``cross``,
  and ``column`` for an existing task feature.

Every spec must carry its clinical ``rationale`` and expected ``direction``.
Its column name and plain-language definition are generated from the spec
itself, so the sentence a clinician reviews is exactly what was computed:

.. code-block:: python

    fs = FeatureSpace(base_dataset, time_key="landmark")   # or FeatureSpace.from_task
    specs = [{"op": "ge", "x": 3,
              "of": {"op": "count", "window": 365,
                     "select": {"event_type": "med_orders", "group": "SABA"}},
              "rationale": "frequent reliever use signals poor control",
              "direction": "+"}]
    for row in fs.check(specs, train):    # definition, coverage, future-blind
        print(row["definition"])          # 1 if there were 3 or more SABA orders
                                          # in the 365 days before the landmark, else 0
    specs = fs.validate_many(specs)
    train_fe = fs.augment(train, specs)   # adds an "engineered" tensor field
    val_fe = fs.augment(val, specs, stats=train_fe.stats)

- **Refusals.** An unknown operator, group or attribute, a window outside the
  menu, a ``where`` value that fewer than 11 patients have, a missing
  rationale or a ninth feature is refused with a reason.
- **The future-blind test.** :meth:`~pyhealth.autoimprove.FeatureSpace.future_blind`
  computes every spec twice: on the full record, and on the record as known
  at t. Rows dated at or after t are removed. The values of rows known only
  later, such as a stay's length known at discharge (``known_at``), are
  blanked. Derived events are rebuilt from the truncated record. Any
  difference rejects the spec.
- **Per-fold processing.** Engineered columns are imputed and scaled with
  statistics from each fold's training part
  (:func:`~pyhealth.autoimprove.fold_parts`), so a held-out fold never informs
  its own medians and scales.
- **Task hooks**, all optional, on :class:`~pyhealth.tasks.BaseTask`:

  - ``time_key``: the sample field holding t;
  - ``feature_vocabulary()``: named code groups such as SABA;
  - ``derived_events(patient, t)`` with ``derived_event_types``: clinical
    definitions, such as exacerbation episodes, built from events before t.

  Tasks without them work as before.

See ``examples/autoimprove_features_mimic3.py``. It checks one hand-written
feature and measures its gain by paired cross-validation, with no search and
no LLM.

Searching features
------------------

Give the improver a FeatureSpace to search inputs as well as settings:

.. code-block:: python

    improver = AutoImprover(
        train, val, LLMProposer(ClaudeLLM()), baseline=..., space=...,
        features=fs, cv_folds=5, confirm=True, task_card=task.task_card(),
    )
    result = improver.run()

- **Alternating cycles** (``feature_cycles=(2, 1)``). Feature cycles hold
  the model at the current selection, and each agent proposes a group of
  specs with its rationale and expected direction. Model cycles hold the
  accepted features and propose configs. With one side fixed, every accepted
  change has a measured gain of its own.
- **Forward selection.** A group joins the accepted set only if its paired
  gain over the accepted set clears the lower bound. Specs that fail the
  future-blind test are refused before anything trains. A group that was
  rejected never becomes the result.
- **Proposers.**

  - :class:`~pyhealth.autoimprove.LLMProposer` sees the operator menu, the
    task card, the accepted features and the earlier groups.
  - :class:`~pyhealth.autoimprove.RandomProposer` proposes random valid
    specs. They are labelled as having no rationale, so they need a
    clinician's approval to stay.
  - :class:`~pyhealth.autoimprove.QueueProposer` tries a given ranked list
    in order, with no LLM.
- **Confirmation** (``confirm=True``). After the search, the leading
  candidates are scored again on new folds with several training seeds
  (:meth:`~pyhealth.autoimprove.AutoImprover.confirm`), and the selection must
  clear a stricter bound (``confirm_se=2``). Candidates picked as the best of
  many on the same folds are partly lucky; this removes part of that luck
  before the test.
- **Pruning** (:meth:`~pyhealth.autoimprove.AutoImprover.prune`). Each
  accepted group whose removal costs less than one SE is dropped, so a
  clinician reviews the shortest list. The ablation shows what each group
  added.
- **Leakage and review.** A new feature that alone predicts the outcome
  almost perfectly (``suspicion_auc``) is held as possible leakage. Features
  with no rationale, or whose learned effect is opposite to their stated
  direction, are held back until :meth:`~pyhealth.autoimprove.AutoImprover.approve`.
  The test then also scores the model without them.
- ``early_stop`` ends a trial after two folds when it is clearly worse than
  the starting pipeline.

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
        ModelSpec, Param, Proposer, LLMProposer, RandomProposer, QueueProposer,
        ProposerExhausted, ClaudeLLM,
        with_xgboost, patient_folds, run_config_cv, select_trial, paired_gain,
        check_models, describe_error, select_simplest, config_diff, Report,
        build_report, load_run, unverified_numbers, fold_parts
    :show-inheritance:

.. automodule:: pyhealth.autoimprove.explain
    :members: attributions, summarize, top_overlap

.. automodule:: pyhealth.autoimprove.features
    :members: FeatureSpace, FeatureSpec, EngineeredDataset, WINDOWS, BLOCKS, AGGS
