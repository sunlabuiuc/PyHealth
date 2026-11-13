.. _cpbench-fixed-calibration:

pyhealth.benchmarks.cpbench.adapters.fixed_calibration
======================================================

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

Shared evaluation and cache settings are documented in
:doc:`pyhealth.benchmarks.cpbench`.

API Reference
-------------

.. autofunction:: pyhealth.benchmarks.cpbench.adapters.fixed_calibration.run
