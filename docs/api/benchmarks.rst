Benchmarks
==========

:mod:`pyhealth.benchmarks` provides the base abstractions for benchmarks. A
benchmark defines and executes evaluations. It knows nothing about runners,
storage or leaderboards.

An :class:`~pyhealth.benchmarks.Evaluation` is one unit of work, identified by
a hash of its parameters. A :class:`~pyhealth.benchmarks.Benchmark` runs a
batch of evaluations and returns one :class:`~pyhealth.benchmarks.Result` per
evaluation.

.. autoclass:: pyhealth.benchmarks.Evaluation
    :members:

.. autoclass:: pyhealth.benchmarks.Result
    :members:

.. autoclass:: pyhealth.benchmarks.Benchmark
    :members:
    :private-members: _evaluate

CPBench (experimental)
----------------------

CPBench uses the generic Evaluation and Result classes. Every Evaluation has
exactly six blocks: ``dataset``, ``task``, ``model``, ``method``, ``metrics``,
``protocol``. CPBench validates these before adapter dispatch, scopes randomness
and returns one Result per Evaluation, retaining successful results when another
fails. There is no CP-specific configuration class.

For example::

    Evaluation({
        "dataset": {"class": SyntheticDataset, "params": {"seed": 5}},
        "task": {"class": SyntheticTask, "params": {"mode": "multiclass"}},
        "model": {
            "class": MLP,
            "params": {"embedding_dim": 8, "hidden_dim": 8},
            "training": {"epochs": 1},
        },
        "method": {
            "class": BaseConformal,
            "params": {"alpha": 0.1, "random_state": 17},
            "options": {"calibration_kwargs": {}},
        },
        "metrics": ["accuracy", "miscoverage_overall_ps", "set_size"],
        "protocol": {
            "split": {
                "class": PatientSplit,
                "params": {"ratios": [0.4, 0.4, 0.2], "seed": 5},
                "roles": ["train", "calibration", "test"],
            },
            "seed": 17,
            "batch_size": 16,
            "device": "cpu",
            "order": "dataset",
            "feedback": "none",
        },
    })

``dataset.class(**params)`` provides the BaseDataset ``set_task`` interface;
``task.class(**params)`` supplies a task and its schemas. Existing PyHealth Split
classes determine the parts, and ``roles`` names them in order. Training requires
``train`` first so preprocessing is fitted only on training. Constructor params
and method options default to empty mappings. Protocol defaults are seed 0,
batch size 32, CPU, dataset order and no feedback. One seed is one evaluation;
cross-evaluation aggregation belongs to the caller.

Model sources
~~~~~~~~~~~~~

The model block must contain exactly one of ``training``, ``checkpoint`` or
``predictions``. ``training`` holds native Trainer arguments. Alternatively::

    "model": {
        "class": MLP,
        "params": {"embedding_dim": 8, "hidden_dim": 8},
        "checkpoint": "/path/to/checkpoint-directory",
    }

A checkpoint directory contains native ``Trainer.save_ckpt`` weights in
``model.ckpt`` and fitted ``SampleBuilder.save`` metadata in ``schema.pkl``.
Loading validates the task schema and reuses saved processors without fitting
or training. It requires no training split; calibration/test subjects can be
selected explicitly. The caller must use data held out from checkpoint training.
The model class/constructor params must match the saved weights.

``predictions`` is recognized as a path-string source, but the current adapters
reject cached predictions before data loading because their native calibration
APIs require a model. No replay-model workaround is supplied.

Adapters and metrics
~~~~~~~~~~~~~~~~~~~~

The explicit ADAPTERS mapping in ``adapters/__init__.py`` selects by
``method["class"]``. BaseConformal, LABEL, RC3P, SCRIB and NeighborhoodLabel use
fixed calibration; FavMac uses multilabel cost control. There is no inference
from inheritance. Adding method support changes the adapter mapping and,
if needed, an adapter; the common contract stays the same.

Both adapters accept ``calibration_kwargs`` in method ``options``, require
calibration/test parts and implement dataset order without test feedback.
Unsupported protocol rules fail explicitly. A test adapter demonstrates immediate
feedback using the same contract and shared preparation/collection functions.

Metrics must be requested explicitly using native metric names. Unknown names
fail instead of disappearing. Multiclass scoring returns None for conditional
metrics without observations. FavMac uses existing multilabel metrics; tp/fp are
mean unweighted counts, even with custom cost/value weights. Results contain
metrics and class labels. Labels are excluded from test prediction calls; shapes
and class encodings are validated before scoring.

Prediction collection
~~~~~~~~~~~~~~~~~~~~~

``collect_predictions(dataset, batch_size, predict_batch)`` calls an adapter's
callback on each full batch. The callback selects native inputs, controls gradients
and converts outputs to named NumPy arrays. Collection copies those arrays, checks
row alignment and consistent output names, and retains dataset order. It imposes
no label key, probability, prediction-set or PyTorch output format.

The current adapters share explicitly classification-specific PyTorch calls and
validation in ``helpers/classification.py``. These exclude labels from native
calls and collect ``y_true``, ``y_prob`` and ``y_predset``. An interval callback can
instead return bounds, without changing the collector. Interval metrics are not
implemented. Online adapters control their own loop and use ``copy_predictions``
before feedback when storing arrays that may share a method's buffers.

.. autofunction:: pyhealth.benchmarks.cpbench.helpers.prediction.collect_predictions

.. autofunction:: pyhealth.benchmarks.cpbench.helpers.prediction.copy_predictions

No benchmark artifacts or timing reports are written. Dataset APIs may use their
configured caches. Run the synthetic example from the repository root with::

    python -m examples.cpbench.synthetic

.. autoclass:: pyhealth.benchmarks.cpbench.CPBench
    :members:

.. autofunction:: pyhealth.benchmarks.cpbench.evaluation.validate_evaluation

.. autofunction:: pyhealth.benchmarks.cpbench.adapters.fixed_calibration.run

.. autofunction:: pyhealth.benchmarks.cpbench.adapters.favmac.run
