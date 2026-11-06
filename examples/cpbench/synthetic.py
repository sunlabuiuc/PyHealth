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
                    "dataset": {
                        "class": SyntheticDataset,
                        "params": {"seed": 5, "mode": mode},
                    },
                    "task": {"class": SyntheticTask, "params": {"mode": mode}},
                    "model": {
                        "class": MLP,
                        "params": {"embedding_dim": 8, "hidden_dim": 8},
                        "training": {"epochs": 1},
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
                            "params": {"ratios": [0.4, 0.4, 0.2], "seed": 5},
                            "roles": ["train", "calibration", "test"],
                        },
                        "seed": 17,
                        "batch_size": 16,
                        "device": "cpu",
                        "order": "dataset",
                        "feedback": "none",
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
