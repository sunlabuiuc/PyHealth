.. _cpbench-cluster-label:

pyhealth.benchmarks.cpbench.adapters.cluster_label
==================================================

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

Shared evaluation and cache settings are documented in
:doc:`pyhealth.benchmarks.cpbench`.

API Reference
-------------

.. autofunction:: pyhealth.benchmarks.cpbench.adapters.cluster_label.run
