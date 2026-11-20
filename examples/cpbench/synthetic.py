"""Run from the repository root: python -m examples.cpbench.synthetic."""

import json

from examples.cpbench.synthetic_data import SyntheticDataset, SyntheticTask
from pyhealth.benchmarks import Evaluation
from pyhealth.benchmarks.cpbench import CPBench
from pyhealth.calib.predictionset import (
    LABEL,
    RC3P,
    SCRIB,
    BaseConformal,
    FavMac,
    NeighborhoodLabel,
)
from pyhealth.datasets import PatientSplit
from pyhealth.models import MLP


def main():
    evaluations = []
    # Arguments follow each native method's API. alpha is miscoverage;
    # LABEL's alpha list uses encoded class order, SCRIB's risk controls its
    # error criterion, and FavMac's target_cost uses native cost units.
    for method, arguments, calibration, mode in (
        (BaseConformal, {"alpha": 0.1, "random_state": 17}, {}, "multiclass"),
        (BaseConformal, {"alpha": 0.2, "random_state": 17}, {}, "multiclass"),
        (
            BaseConformal,
            {"alpha": 0.1, "score_type": "aps", "random_state": 17},
            {},
            "multiclass",
        ),
        (LABEL, {"alpha": [0.1, 0.2, 0.3], "random_state": 17}, {}, "multiclass"),
        (RC3P, {"alpha": 0.2, "random_state": 17}, {}, "multiclass"),
        (SCRIB, {"risk": 0.2}, {}, "multiclass"),
        (
            NeighborhoodLabel,
            {"alpha": 0.2, "k_neighbors": 5, "random_state": 17},
            {"batch_size": 16},
            "multiclass",
        ),
        (FavMac, {"target_cost": 0.8}, {}, "multilabel"),
    ):
        evaluations.append(
            Evaluation(
                {
                    # Each params mapping is passed to that class's constructor.
                    "dataset": {
                        "class": SyntheticDataset,
                        # mode changes raw labels: integer vs list of labels.
                        # This seed controls feature noise, not patient splitting.
                        "params": {"seed": 5, "mode": mode},
                    },
                    # The task encodes the generated labels; it does not change them.
                    "task": {"class": SyntheticTask, "params": {"mode": mode}},
                    "model": {
                        "class": MLP,
                        "params": {"embedding_dim": 8, "hidden_dim": 8},
                        "training": {"epochs": 1},  # Train on the train split only.
                    },
                    "method": {
                        "class": method,
                        "params": arguments,
                        "options": {"calibration_kwargs": calibration},
                    },
                    "metrics": ["tp", "fp", "roc_auc_micro"]
                    if mode == "multilabel"
                    else ["accuracy", "set_size", "miscoverage_overall_ps"],
                    "protocol": {
                        "split": {
                            "class": PatientSplit,
                            # Fractions in train/calibration/test order.
                            # The shuffle seed is independent of the evaluation seed.
                            "params": {"ratios": [0.4, 0.4, 0.2], "seed": 5},
                        },
                        # Seeds global RNGs; does not replace native random_state.
                        "seed": 17,
                        "batch_size": 16,
                        "device": "cpu",
                        "order": "dataset",  # Iterate the test split in dataset order.
                        "feedback": "none",  # Do not update from test labels.
                    },
                }
            )
        )
    results = CPBench().evaluate(evaluations)
    for evaluation, result in results.items():
        print(
            json.dumps(
                {
                    "method": evaluation.params["method"]["class"].__name__,
                    "parameters": evaluation.params["method"]["params"],
                    "status": result.status,
                    "metrics": result.metrics,
                    "error": result.metadata.get("error"),
                }
            )
        )
    if any(result.status != "success" for result in results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
