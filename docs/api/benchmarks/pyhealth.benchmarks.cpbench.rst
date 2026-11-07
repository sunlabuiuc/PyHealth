.. _cpbench-guide:

pyhealth.benchmarks.cpbench
===========================

Overview
--------

CPBench uses the generic Evaluation and Result classes. Every Evaluation has
exactly six blocks: ``dataset``, ``task``, ``model``, ``method``, ``metrics``,
``protocol``. ``validate_evaluations`` checks these blocks, duplicate evaluations
and supported methods before any evaluation runs. Invalid specifications raise.
During execution, CPBench scopes randomness and converts runtime exceptions to
failed Results, retaining other evaluations' results. ``validate_results`` checks
that every evaluation has exactly one Result with a valid status and a metric
mapping. Invalid adapter return values raise. Metric-specific checks belong to
adapters and scorers.

Complete Evaluation
-------------------

Run this example from the repository root:

.. code-block:: python

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
how to encode those labels; it does not change their meaning. A real dataset
and task use their own constructor arguments. CPBench checks processed
label/output formats and model compatibility; the caller chooses the task
appropriate to the data.

The data seed controls feature noise, the split seed controls patient
assignment, and ``protocol.seed`` scopes global Python/NumPy/PyTorch randomness. Native
``random_state`` arguments control the method's own generators.
``alpha=0.1`` requests a miscoverage level of 0.1, corresponding to 90%
coverage; it does not assert that this coverage was achieved on the test data.

``dataset.class(**params)`` provides the BaseDataset ``set_task`` interface;
``task.class(**params)`` supplies a task and its schemas. Existing PyHealth Split
classes determine the parts. Three outputs mean train/calibration/test; four
mean train/validation/calibration/test. Preprocessing is fitted only on training.
Constructor params and method options default to empty mappings. Protocol
defaults are seed 0, batch size 32, CPU, dataset order and no feedback. One
seed is one evaluation; cross-evaluation aggregation belongs to the caller.

Model Preparation
-----------------

The model block supplies its constructor arguments and native Trainer settings
in ``training``. These settings describe how to produce a missing model. With
``cache_dir``, CPBench reuses matching checkpoints and base-model outputs; without
it, a fresh model is trained. Generated artifact paths are not evaluation inputs.

``model.training.monitor`` requires a validation partition. Starting from the
complete evaluation, use four ratios in train/validation/calibration/test order:

.. code-block:: python

    from copy import deepcopy

    params = deepcopy(evaluation.params)
    params["protocol"]["split"]["params"]["ratios"] = [0.4, 0.2, 0.2, 0.2]
    params["model"]["training"] = {"epochs": 1, "monitor": "accuracy"}
    validation_evaluation = Evaluation(params)
    result = CPBench().evaluate([validation_evaluation])[validation_evaluation]
    assert result.status == "success", result.metadata

Adapters
--------

``adapters/ADAPTERS`` selects by the exact ``method["class"]``. To support a new
method, add its mapping and an adapter if needed. Current adapters accept native
constructor arguments in ``method.params`` and only ``calibration_kwargs`` in
``method.options``. Unknown options fail explicitly.

The current adapters require dataset order and no test feedback. They calibrate
on the calibration partition and score the test partition. Each adapter example
below includes its complete evaluation and imports.

.. _cpbench-fixed-calibration:

Fixed Calibration
~~~~~~~~~~~~~~~~~

BaseConformal, LABEL, RC3P, SCRIB, NeighborhoodLabel and LabelShift use this
adapter. Data, task and model must share one multiclass label encoding. The
native method defines the meaning of ``alpha``, ``risk`` and other constructor
arguments. NeighborhoodLabel additionally requires a model supporting
``embed=True``; calibration embeddings can be passed through its native
``calibration_kwargs``.

LabelShift requires ``target_label_dist`` in encoded class order.
``source_label_dist`` is optional; native calibration estimates it from the
calibration labels when omitted. For the balanced synthetic population:

.. code-block:: python

    from examples.cpbench.synthetic_data import SyntheticDataset, SyntheticTask
    from pyhealth.benchmarks import Evaluation
    from pyhealth.benchmarks.cpbench import CPBench
    from pyhealth.calib.predictionset import LabelShift
    from pyhealth.datasets import PatientSplit
    from pyhealth.models import MLP

    label_shift_evaluation = Evaluation({
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
            "class": LabelShift,
            "params": {"alpha": 0.1, "random_state": 17},
            "options": {
                "calibration_kwargs": {"target_label_dist": [1 / 3, 1 / 3, 1 / 3]},
            },
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

    result = CPBench().evaluate([label_shift_evaluation])[label_shift_evaluation]
    assert result.status == "success", result.metadata

For real data, supply the target population's known or separately estimated
probabilities. The adapter does not estimate them from test labels. This example
demonstrates execution, without simulating a label shift.

.. _cpbench-cluster-label:

ClusterLabel
~~~~~~~~~~~~

The model must be multiclass and support ``embed=True``. The train partition
supplies reference embeddings. Omit ``train_embeddings`` from calibration
arguments: CPBench computes them. Native calibration extracts calibration
embeddings unless ``cal_embeddings`` is supplied. Calibration batch size defaults
to ``protocol.batch_size``:

.. code-block:: python

    from examples.cpbench.synthetic_data import SyntheticDataset, SyntheticTask
    from pyhealth.benchmarks import Evaluation
    from pyhealth.benchmarks.cpbench import CPBench
    from pyhealth.calib.predictionset import ClusterLabel
    from pyhealth.datasets import PatientSplit
    from pyhealth.models import MLP

    cluster_evaluation = Evaluation({
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
            "class": ClusterLabel,
            "params": {"alpha": 0.1, "n_clusters": 2, "random_state": 17},
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

    result = CPBench().evaluate([cluster_evaluation])[cluster_evaluation]
    assert result.status == "success", result.metadata

.. _cpbench-covariate-label:

CovariateLabel
~~~~~~~~~~~~~~

By default, native density estimation uses **all test inputs before prediction**.
Test labels are excluded from embedding, calibration and prediction calls. The
model must be multiclass and support ``embed=True``. Embeddings follow dataset
order and use ``protocol.batch_size``.

Paired ``kde_cal``/``kde_test`` constructor arguments may supply already-fitted,
importable density callables; then only calibration embeddings are extracted.
Alternatively, pass ``cal_weights`` in ``calibration_kwargs``: one importance
weight per calibration row, in dataset order. These are likelihood ratios, not
class costs. Supplied weights remove the need for embeddings. Omit
``cal_embeddings`` and ``test_embeddings``: CPBench computes them when needed.
Other calibration arguments go directly to native calibration:

.. code-block:: python

    from examples.cpbench.synthetic_data import SyntheticDataset, SyntheticTask
    from pyhealth.benchmarks import Evaluation
    from pyhealth.benchmarks.cpbench import CPBench
    from pyhealth.calib.predictionset import CovariateLabel
    from pyhealth.datasets import PatientSplit
    from pyhealth.models import MLP

    covariate_evaluation = Evaluation({
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
            "class": CovariateLabel,
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

    result = CPBench().evaluate([covariate_evaluation])[covariate_evaluation]
    assert result.status == "success", result.metadata

This synthetic example demonstrates execution, without simulating covariate shift.

.. _cpbench-favmac:

FavMac
~~~~~~

The dataset, task and model must be multilabel and share a class encoding.
The model must return calibration/test logits and test probabilities.
``cost_weights`` and ``value_weights`` may be lists in encoded class order;
the adapter converts them to arrays. Calibration arguments are passed directly
to native calibration. This example uses multilabel samples and their matching
processor:

.. code-block:: python

    from examples.cpbench.synthetic_data import SyntheticDataset, SyntheticTask
    from pyhealth.benchmarks import Evaluation
    from pyhealth.benchmarks.cpbench import CPBench
    from pyhealth.calib.predictionset import FavMac
    from pyhealth.datasets import PatientSplit
    from pyhealth.models import MLP

    favmac_evaluation = Evaluation({
        # Generate multilabel samples, such as [0, 2], rather than class indices.
        "dataset": {
            "class": SyntheticDataset,
            "params": {"seed": 5, "mode": "multilabel"},
        },
        # The task chooses the label processor, without changing the raw labels.
        "task": {"class": SyntheticTask, "params": {"mode": "multilabel"}},
        "model": {
            "class": MLP,
            "params": {"embedding_dim": 8, "hidden_dim": 8},
            "training": {"epochs": 1},
        },
        "method": {
            "class": FavMac,
            "params": {
                "target_cost": 0.8,
                "cost_weights": [1, 2, 1],
                "value_weights": [2, 1, 3],
            },
            "options": {"calibration_kwargs": {}},
        },
        "metrics": ["tp", "fp", "fp_cost", "tp_value", "set_size",
            "miscoverage_overall_ps", "fp_cost_violation_rate"],
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

    result = CPBench().evaluate([favmac_evaluation])[favmac_evaluation]
    assert result.status == "success", result.metadata

Metrics
-------

Request native metric names explicitly. Unknown names fail instead of
disappearing. Undefined prediction-set rates return NaN. Multiclass adapters
use multiclass metrics; FavMac uses multilabel metrics and forwards its weights
and target cost to weighted measurements. Ordinary tp/fp remain unweighted.
Arrays and NaNs remain in ``Result.metrics``; JSON conversion belongs to the
result writer. Results include class labels in encoded column order.

See the :doc:`multiclass <../metrics/pyhealth.metrics.multiclass>`,
:doc:`multilabel <../metrics/pyhealth.metrics.multilabel>` and
:doc:`prediction-set <../metrics/pyhealth.metrics.prediction_set>` metric references
for names and definitions.

Shared Adapter Code
-------------------

``shared/evaluation_preparation.py`` prepares datasets, models and required
outputs, and manages cache locations and reuse. ``shared/model_outputs.py``
collects base-model outputs, handles saved-output model calls and writes output
files. ``shared/collection.py`` collects, copies and joins batches of arrays.
``shared/classification.py`` contains classification calls, label/output checks
and multiclass scoring. Evaluation validation and per-evaluation random seeding
live with CPBench in ``benchmark.py``.

Adapters expose ``run(evaluation: Evaluation, cache_dir: Path | None) -> Result``.
The adapter performs the method's evaluation and returns its metrics in a
Result. The adapter defines its own method options; the ``calibration_kwargs``
restriction above applies to the current adapters. Register the module, rather
than its run function. For example, this existing mapping is:

.. code-block:: python

    from pyhealth.benchmarks.cpbench.adapters import ADAPTERS, fixed_calibration
    from pyhealth.calib.predictionset import BaseConformal

    ADAPTERS[BaseConformal] = fixed_calibration

Prediction Collection
---------------------

``collect_predictions(dataset, batch_size, predict_batch)`` calls an adapter's
callback on each full batch. The callback selects native inputs, controls
gradients and converts outputs to named NumPy arrays. Collection copies those arrays, checks
row alignment and consistent output names, and retains dataset order. It imposes
no label key, probability, prediction-set or PyTorch output format.

The current adapters share explicitly classification-specific PyTorch calls and
validation in ``shared/classification.py``. These exclude labels from native
calls and collect ``y_true``, ``y_prob`` and ``y_predset``. An interval callback
can instead return bounds, without changing the collector. Interval metrics are not
implemented. Online adapters control their own loop and use ``copy_predictions``
before feedback when storing arrays that may share a method's buffers.

Cache and Result Files
----------------------

The directory arguments are independent. Using the complete BaseConformal and
ClusterLabel evaluations above:

.. code-block:: python

    from pathlib import Path

    results = CPBench().evaluate(
        [evaluation, cluster_evaluation],
        cache_dir=Path("cache"),
        output_dir=Path("run"),
    )
    for evaluated, result in results.items():
        if result.status == "failed":
            print(evaluated.id, result.metadata["error_type"])
        else:
            print(evaluated.id, result.metrics)

``cache_dir`` holds reusable datasets, model checkpoints and prediction
arrays. Adapters declare the outputs they need; shared evaluation preparation
manages reuse. CP calibration and scoring still run for each evaluation,
even when all required base outputs are cached.
``output_dir`` holds one ``results.json`` containing every evaluation in the
batch, including failures. The base Benchmark writes it after result validation.
Results are always returned; neither directory is required.

The layout is:

.. code-block:: text

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

Each JSON record is keyed by evaluation ID and contains ``evaluation``
(serialized parameters), ``status``, ``metrics``, ``metadata`` and ``artifact``. NumPy values
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
the cache when changing them without updating evaluation parameters. Native
dataset caches use the managed datasets directory for BaseDataset subclasses. With no
cache_dir, no CPBench model or prediction files are written. With no output_dir,
no results.json is written.

Manual Persistence
------------------

The automatic workflow uses the same public writers available to user code:

.. code-block:: python

    from pyhealth.benchmarks import save_results
    from pyhealth.benchmarks.cpbench import save_checkpoint, save_predictions

``save_results(results, path)`` writes one mapping of Evaluation keys to Result
values. It accepts an explicit file path, so results from multiple calls can be
combined before saving. Continuing the complete evaluation example:

.. code-block:: python

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
persistence belongs to Benchmark.evaluate.

API Reference
-------------

Base Classes
~~~~~~~~~~~~

A complete evaluation using the generic Benchmark:

.. code-block:: python

    from pyhealth.benchmarks import Benchmark, Evaluation, Result

    class Squares(Benchmark):
        @property
        def name(self):
            return "squares"

        def _evaluate(self, evaluations, cache_dir, output_dir):
            return {
                ev: Result("success", {"square": ev.params["x"] ** 2})
                for ev in evaluations
            }

    square_evaluation = Evaluation({"x": 3})
    square_result = Squares().evaluate([square_evaluation])[square_evaluation]
    assert square_result.metrics == {"square": 9}

.. autoclass:: pyhealth.benchmarks.Evaluation
    :members:
    :undoc-members:
    :show-inheritance:
    :exclude-members: params, id

.. autoclass:: pyhealth.benchmarks.Result
    :members:
    :undoc-members:
    :show-inheritance:
    :exclude-members: status, metrics, metadata, artifact

.. autoclass:: pyhealth.benchmarks.Benchmark
    :members:
    :undoc-members:
    :show-inheritance:
    :private-members: _evaluate

.. autofunction:: pyhealth.benchmarks.save_results

CPBench
~~~~~~~

.. autoclass:: pyhealth.benchmarks.cpbench.CPBench
    :members:
    :undoc-members:
    :show-inheritance:

.. autofunction:: pyhealth.benchmarks.cpbench.benchmark.validate_evaluation

Model Outputs and Persistence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: pyhealth.benchmarks.cpbench.required_model_outputs

.. autofunction:: pyhealth.benchmarks.cpbench.load_model_predictions

.. autofunction:: pyhealth.benchmarks.cpbench.save_model_predictions

.. autofunction:: pyhealth.benchmarks.cpbench.save_predictions

.. autofunction:: pyhealth.benchmarks.cpbench.save_checkpoint

Prediction Collection
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: pyhealth.benchmarks.cpbench.shared.collection.collect_predictions

.. autofunction:: pyhealth.benchmarks.cpbench.shared.collection.copy_predictions

Adapters
~~~~~~~~

.. autofunction:: pyhealth.benchmarks.cpbench.adapters.fixed_calibration.run

.. autofunction:: pyhealth.benchmarks.cpbench.adapters.cluster_label.run

.. autofunction:: pyhealth.benchmarks.cpbench.adapters.covariate_label.run

.. autofunction:: pyhealth.benchmarks.cpbench.adapters.favmac.run
