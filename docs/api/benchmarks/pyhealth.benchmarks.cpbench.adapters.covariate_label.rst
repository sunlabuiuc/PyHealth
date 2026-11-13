.. _cpbench-covariate-label:

pyhealth.benchmarks.cpbench.adapters.covariate_label
====================================================

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

Shared evaluation and cache settings are documented in
:doc:`pyhealth.benchmarks.cpbench`.

API Reference
-------------

.. autofunction:: pyhealth.benchmarks.cpbench.adapters.covariate_label.run
