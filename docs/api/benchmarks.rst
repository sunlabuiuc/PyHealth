Benchmarks
==========

:mod:`pyhealth.benchmarks` provides the base abstractions for benchmarks. A
benchmark validates and executes evaluations, then optionally saves the batch
results. Runners and leaderboards remain separate concerns.

An :class:`~pyhealth.benchmarks.Evaluation` is one unit of work, identified by
a hash of its parameters. A :class:`~pyhealth.benchmarks.Benchmark` runs a
batch of evaluations and returns one :class:`~pyhealth.benchmarks.Result` per
evaluation.

.. autoclass:: pyhealth.benchmarks.Evaluation
    :members:

.. autoclass:: pyhealth.benchmarks.Result
    :members:

.. autofunction:: pyhealth.benchmarks.save_results

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
``random_state`` arguments control the method's own generators.
``alpha=0.1`` requests a miscoverage level of 0.1, corresponding to 90% coverage;
it does not assert that this coverage was achieved on the test data.

``dataset.class(**params)`` provides the BaseDataset ``set_task`` interface;
``task.class(**params)`` supplies a task and its schemas. Existing PyHealth Split
classes determine the parts. Three outputs mean train/calibration/test; four
mean train/validation/calibration/test. Preprocessing is fitted only on training.
Constructor params and method options default to empty mappings. Protocol
defaults are seed 0,
batch size 32, CPU, dataset order and no feedback. One seed is one evaluation;
cross-evaluation aggregation belongs to the caller.

Model preparation
~~~~~~~~~~~~~~~~~

The model block supplies its constructor arguments and native Trainer settings
in ``training``. These settings describe how to produce a missing model. With
``cache_dir``, CPBench reuses matching checkpoints and base-model outputs; without
it, a fresh model is trained. Generated artifact paths are not evaluation inputs.

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
fail instead of disappearing. Undefined prediction-set rates return NaN.
FavMac forwards its configured weights and target cost to weighted metrics;
ordinary tp/fp retain their unweighted meaning. Results contain metrics and
class labels. Labels are excluded from test prediction calls; shapes
and class encodings are validated before scoring.

Shared adapter code
~~~~~~~~~~~~~~~~~~~

``shared/evaluation_preparation.py`` prepares datasets, models and required
outputs, and manages cache locations and reuse. ``shared/model_outputs.py``
collects base-model outputs, handles saved-output model calls and writes output
files. ``shared/collection.py`` collects, copies and joins batches of arrays.
``shared/classification.py`` contains classification calls, label/output checks
and multiclass scoring. Evaluation validation and per-evaluation random seeding
live with CPBench in ``benchmark.py``.

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

.. autofunction:: pyhealth.benchmarks.cpbench.shared.collection.collect_predictions

.. autofunction:: pyhealth.benchmarks.cpbench.shared.collection.copy_predictions

Cache and result files
~~~~~~~~~~~~~~~~~~~~~~

The directory arguments are independent. Continuing the complete example::

    from pathlib import Path

    results = CPBench().evaluate(
        [evaluation],
        cache_dir=Path("cache"),
        output_dir=Path("run"),
    )

``cache_dir`` holds reusable datasets, model checkpoints and prediction arrays.
Adapters declare the outputs they need; shared evaluation preparation manages reuse.
``output_dir`` holds one ``results.json`` containing every evaluation in the
batch, including failures. The base Benchmark writes it after result validation.
Results are always returned; neither directory is required.

The layout is::

    cache/
      cpbench/
        checkpoints/<model-key>/
          model.ckpt
          schema.pkl
        predictions/
          base/<model-key>/<split>.npz
          conformal/<prediction-key>.npz
        datasets/
    run/
      results.json

Each JSON record is keyed by evaluation ID and contains ``evaluation`` (serialized
parameters), ``status``, ``metrics``, ``metadata`` and ``artifact``. NumPy values
become ordinary JSON values, paths become strings, and non-finite numbers become
null during serialization. Returned results are not modified. Reusing an output
directory atomically replaces the complete batch file; it does not append records.
Write failures propagate to the caller.

Model cache keys include dataset, task, model, split, seed, batch size and device;
they exclude the CP method and requested metrics. Different CP methods can reuse
the same fitted base model. Complete base-output caches skip dataset loading,
model construction and inference. Missing outputs are collected from the saved
checkpoint. Sample and class ordering are checked before combining outputs.
Checkpoint creation and output-cache extension are protected by a file lock.
Preparation preserves global RNG state across cache hits and misses.

Prediction keys include dataset, task, model, method and protocol, excluding only
requested metrics. Adapters write ``y_true``, ``y_prob`` and ``y_predset`` in test
order and set ``Result.artifact`` to the cache file. Class ordering is retained in
result metadata. Base outputs contain encoded labels and the probabilities,
logits or embeddings required by each adapter. Each CP method runs its own
calibration using those outputs.

Cache entries assume that referenced data/code have not changed in place; clear
the cache when changing them without updating evaluation parameters. Native dataset
caches use the managed datasets directory for BaseDataset subclasses. With no
cache_dir, no CPBench model or prediction files are written. With no output_dir,
no results.json is written.

Manual persistence
~~~~~~~~~~~~~~~~~~

The automatic workflow uses the same public writers available to user code::

    from pyhealth.benchmarks import save_results
    from pyhealth.benchmarks.cpbench import save_checkpoint, save_predictions

``save_results(results, path)`` writes one mapping of Evaluation keys to Result
values. It accepts an explicit file path, so results from multiple calls can be
combined before saving. Continuing the complete evaluation example::

    from copy import deepcopy
    from pathlib import Path
    from pyhealth.benchmarks import Evaluation, save_results

    first = CPBench().evaluate([evaluation])  # No automatic persistence.
    params = deepcopy(evaluation.params)
    params["protocol"]["seed"] = 18
    second_evaluation = Evaluation(params)
    second = CPBench().evaluate([second_evaluation])
    save_results({**first, **second}, Path("manual/results.json"))

``save_predictions(predictions, path)`` writes only the supplied named arrays to
an NPZ file. It does not take an Evaluation or Result. ``save_checkpoint(model,
dataset, destination)`` writes native model weights and the dataset's fitted
preprocessing to a new directory. The dataset must carry the model's fitted
processors; an existing destination is rejected with FileExistsError.

All three writers return None and write complete files/directories atomically.
They do not choose cache keys, acquire cache locks, decide reuse, or construct
Results. Shared evaluation preparation manages caches; automatic result
persistence belongs to Benchmark.evaluate. Run the synthetic example from the
repository root with::

    python -m examples.cpbench.synthetic

.. autofunction:: pyhealth.benchmarks.cpbench.save_predictions

.. autofunction:: pyhealth.benchmarks.cpbench.save_checkpoint

.. autoclass:: pyhealth.benchmarks.cpbench.CPBench
    :members:

.. autofunction:: pyhealth.benchmarks.cpbench.benchmark.validate_evaluation

.. autofunction:: pyhealth.benchmarks.cpbench.adapters.fixed_calibration.run

.. autofunction:: pyhealth.benchmarks.cpbench.adapters.favmac.run

.. autofunction:: pyhealth.benchmarks.cpbench.adapters.cluster_label.run

.. autofunction:: pyhealth.benchmarks.cpbench.adapters.covariate_label.run
