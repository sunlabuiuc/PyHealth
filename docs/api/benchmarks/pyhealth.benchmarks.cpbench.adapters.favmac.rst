.. _cpbench-favmac:

pyhealth.benchmarks.cpbench.adapters.favmac
===========================================

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

Shared evaluation and cache settings are documented in
:doc:`pyhealth.benchmarks.cpbench`.

API Reference
-------------

.. autofunction:: pyhealth.benchmarks.cpbench.adapters.favmac.run
