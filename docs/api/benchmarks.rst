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
``protocol``. ``validate_evaluations`` checks these blocks, duplicate evaluations
and supported methods before any evaluation runs. Invalid specifications raise.
During execution, CPBench scopes randomness and converts runtime exceptions to
failed Results, retaining other evaluations' results. ``validate_results`` checks
that every evaluation has exactly one Result with a valid status and a metric
mapping. Invalid adapter return values raise. Metric-specific checks belong to
adapters and scorers. There is no CP-specific configuration class.

Complete example, run from the repository root::

    from examples.cpbench.synthetic_data import SyntheticDataset, SyntheticTask
    from pyhealth.benchmarks import Evaluation
    from pyhealth.benchmarks.cpbench import CPBench
    from pyhealth.calib.predictionset import BaseConformal
    from pyhealth.datasets import PatientSplit
    from pyhealth.models import MLP

    evaluation = Evaluation({
        # seed controls feature noise; mode generates one integer label per sample.
        "dataset": {
            "class": SyntheticDataset,
            "params": {"seed": 5, "mode": "multiclass"},
        },
        # The task chooses the label processor, without changing the raw labels.
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

    result = CPBench().evaluate([evaluation])[evaluation]
    assert result.status == "success", result.metadata

Every ``params`` mapping contains arguments for its own ``class`` constructor.
``SyntheticDataset.mode`` changes the generated samples: one integer label for
multiclass, or a list of labels for multilabel. ``SyntheticTask.mode`` chooses
how to encode those labels; it does not change their meaning. A real dataset and
task use their own constructor arguments. CPBench checks processed label/output
formats and model compatibility; the caller chooses the task appropriate to the data.

The data seed controls feature noise, the split seed controls patient assignment,
and ``protocol.seed`` scopes global Python/NumPy/PyTorch randomness. Native
``random_state`` arguments control the method's own generators. ``roles`` names
split outputs in order; it does not select samples by patient-ID prefixes.
``alpha=0.1`` requests a miscoverage level of 0.1, corresponding to 90% coverage;
it does not assert that this coverage was achieved on the test data.

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
``predictions``. ``training`` holds native Trainer arguments. To use a checkpoint,
replace the model block with the following. The path is a placeholder: the directory
must already contain compatible weights and fitted preprocessing::

    "model": {
        "class": MLP,
        "params": {"embedding_dim": 8, "hidden_dim": 8},
        "checkpoint": "/path/to/checkpoint-directory",
    }

A checkpoint directory contains native ``Trainer.save_ckpt`` weights in
``model.ckpt`` and fitted ``SampleBuilder.save`` metadata in ``schema.pkl``.
Loading validates the task schema and reuses saved processors without fitting
or training. Fixed calibration, FavMac and CovariateLabel need no training split
in this mode;
ClusterLabel still requires a train role for reference embeddings. Calibration/test
subjects can be selected explicitly. The caller must use data held out from checkpoint training.
The model class/constructor params must match the saved weights.

``predictions`` is recognized as a path-string source, but the current adapters
reject cached predictions before data loading because their native calibration
APIs require a model. No replay-model workaround is supplied.

Adapters and metrics
~~~~~~~~~~~~~~~~~~~~

The explicit ADAPTERS mapping in ``adapters/__init__.py`` selects by
``method["class"]``. BaseConformal, LABEL, RC3P, SCRIB, NeighborhoodLabel and
LabelShift use fixed calibration; FavMac uses multilabel cost control. ClusterLabel adds
reference embedding extraction from a declared train split. CovariateLabel uses
native covariate-shift weighting. There is no inference
from inheritance. Adding method support changes the adapter mapping and,
if needed, an adapter; the common contract stays the same.

All four adapters accept ``calibration_kwargs`` in method ``options`` and
implement dataset order without test feedback. Calibration/test parts are required;
ClusterLabel additionally requires the train reference part and an embedding-capable
model.
Unsupported protocol rules fail explicitly. A test adapter demonstrates immediate
feedback using the same contract and shared preparation/collection functions.

CovariateLabel defaults to native KDE fitting on calibration/test embeddings.
It uses **all test inputs before prediction**, excluding their labels from every
embedding and prediction call. It requires a multiclass model supporting
``embed=True``. With paired importable ``kde_cal``/``kde_test`` constructor
callables, only calibration embeddings are extracted. Alternatively, pass
``cal_weights`` in ``method.options.calibration_kwargs``: one importance weight
per calibration row, in dataset order. This path needs no embeddings.
``cal_embeddings``/``test_embeddings`` are computed from declared splits and cannot
be supplied in options. A complete runnable example is in the adapter docstring.

LabelShift requires a supplied target label distribution in class-encoding order.
Starting from the complete example above, use equal target probabilities for this
balanced synthetic population (this example does not simulate a shift)::

    from copy import deepcopy
    from pyhealth.calib.predictionset import LabelShift

    params = deepcopy(evaluation.params)
    params["method"] = {
        "class": LabelShift,
        "params": {"alpha": 0.1},
        "options": {
            "calibration_kwargs": {
                "target_label_dist": [1 / 3, 1 / 3, 1 / 3],
                # source_label_dist is omitted: estimate it from calibration labels.
            },
        },
    }

    label_shift_evaluation = Evaluation(params)
    result = CPBench().evaluate([label_shift_evaluation])[label_shift_evaluation]
    assert result.status == "success", result.metadata

For real data, supply the target population's known or separately estimated
probabilities; these numbers are inputs, not a declaration that the data matches them.
The adapter passes these distributions to native calibration; it does not estimate
the target distribution from test labels.

Metrics must be requested explicitly using native metric names. Unknown names
fail instead of disappearing. Multiclass scoring returns None for conditional
metrics without observations. FavMac uses existing multilabel metrics; tp/fp are
mean unweighted counts, even with custom cost/value weights. Results contain
metrics and class labels. Labels are excluded from test prediction calls; shapes
and class encodings are validated before scoring.

Shared adapter code
~~~~~~~~~~~~~~~~~~~

``shared/preparation.py`` prepares datasets and splits, trains or loads models,
and manages model checkpoints and caching. ``shared/prediction.py`` collects,
copies, joins and saves prediction arrays. ``shared/classification.py`` contains
classification calls, label/output checks and multiclass scoring. Adapters can
reuse these functions or supply their own procedure. Evaluation validation and
per-evaluation random seeding live with CPBench in ``benchmark.py``.

Prediction collection
~~~~~~~~~~~~~~~~~~~~~

``collect_predictions(dataset, batch_size, predict_batch)`` calls an adapter's
callback on each full batch. The callback selects native inputs, controls gradients
and converts outputs to named NumPy arrays. Collection copies those arrays, checks
row alignment and consistent output names, and retains dataset order. It imposes
no label key, probability, prediction-set or PyTorch output format.

The current adapters share explicitly classification-specific PyTorch calls and
validation in ``shared/classification.py``. These exclude labels from native
calls and collect ``y_true``, ``y_prob`` and ``y_predset``. An interval callback can
instead return bounds, without changing the collector. Interval metrics are not
implemented. Online adapters control their own loop and use ``copy_predictions``
before feedback when storing arrays that may share a method's buffers.

.. autofunction:: pyhealth.benchmarks.cpbench.shared.prediction.collect_predictions

.. autofunction:: pyhealth.benchmarks.cpbench.shared.prediction.copy_predictions

Cache and outputs
~~~~~~~~~~~~~~~~~

The directory arguments are optional. Continuing the complete example::

    from pathlib import Path

    results = CPBench().evaluate(
        [evaluation],
        cache_dir=Path("cache"),
        output_dir=Path("run"),
    )

``cache_dir`` stores fitted base-model checkpoints under
``cpbench/models/<model-input-hash>/``. The key includes dataset, task, model,
split, run seed, batch size and device, excluding CP methods and requested metrics.
Changing alpha, selecting another supported CP method or changing metrics can
therefore reuse the same model. Calibration still runs for each evaluation.
Cache hits load saved fitted processors and fresh model instances, and skip
training. An existing checkpoint input is loaded directly without duplicating it
in the cache. Dataset APIs retain their own configured cache locations.

Cache entries are protected by a file lock and published by atomic directory
rename. Preparation preserves global RNG state so cache hits and misses do not advance
those streams differently before method execution. Entries assume the declared dataset,
code and model inputs are unchanged; clear the cache when changing referenced
files or implementations without changing the evaluation specification.

``output_dir`` stores one ``<evaluation.id>.npz`` containing the successful
calibrated test prediction arrays. ``Result.artifact`` points to it. Current
adapters save ``y_true``, ``y_prob`` and ``y_predset``, in dataset order; class
ordering is in Result metadata. Writes use a temporary file and atomic replace.
Metrics remain in the returned Result. These scored outputs support later
analysis; they do not enable the unsupported ``model.predictions`` input mode.

With neither directory supplied, no CPBench cache/output artifacts are written
and Result.artifact remains None. No timing reports or cross-run aggregation are
added. Run the synthetic example from the repository root with::

    python -m examples.cpbench.synthetic

.. autoclass:: pyhealth.benchmarks.cpbench.CPBench
    :members:

.. autofunction:: pyhealth.benchmarks.cpbench.benchmark.validate_evaluation

.. autofunction:: pyhealth.benchmarks.cpbench.adapters.fixed_calibration.run

.. autofunction:: pyhealth.benchmarks.cpbench.adapters.favmac.run

.. autofunction:: pyhealth.benchmarks.cpbench.adapters.cluster_label.run

.. autofunction:: pyhealth.benchmarks.cpbench.adapters.covariate_label.run
