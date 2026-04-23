"""Example: temporal vs random evaluation on synthetic clinical data.

This example demonstrates the core EMDOT-style idea:
train on earlier years and test on later years, then compare against
a random split baseline.

It is intentionally synthetic and lightweight so it runs anywhere.
"""

from pyhealth.tasks.temporal_evaluation import (
    run_ablation,
    run_random_experiment,
)


SYNTHETIC_DATA = [
    {"patient_id": 1, "year": 2010, "features": [0.20, 0.10], "label": 0},
    {"patient_id": 2, "year": 2011, "features": [0.25, 0.15], "label": 0},
    {"patient_id": 3, "year": 2012, "features": [0.70, 0.60], "label": 1},
    {"patient_id": 4, "year": 2013, "features": [0.40, 0.35], "label": 0},
    {"patient_id": 5, "year": 2014, "features": [0.55, 0.45], "label": 1},
    {"patient_id": 6, "year": 2015, "features": [0.60, 0.50], "label": 1},
    {"patient_id": 7, "year": 2016, "features": [0.65, 0.55], "label": 1},
    {"patient_id": 8, "year": 2017, "features": [0.35, 0.25], "label": 0},
    {"patient_id": 9, "year": 2018, "features": [0.75, 0.60], "label": 1},
    {"patient_id": 10, "year": 2019, "features": [0.15, 0.10], "label": 0},
    {"patient_id": 11, "year": 2020, "features": [0.82, 0.70], "label": 1},
    {"patient_id": 12, "year": 2021, "features": [0.18, 0.12], "label": 0},
]


def safe_metric(value):
    if value is None:
        return "None"
    return f"{value:.3f}"


def main() -> None:
    print("=== Temporal ablation ===")
    for result in run_ablation(SYNTHETIC_DATA, split_years=[2013, 2015, 2017]):
        print(
            f"split_year={result.split_year} | "
            f"train={result.train_size} | "
            f"test={result.test_size} | "
            f"accuracy={result.accuracy:.3f} | "
            f"auroc={safe_metric(result.auroc)} | "
            f"auprc={safe_metric(result.auprc)} | "
            f"brier={safe_metric(result.brier)} | "
            f"f1={result.f1:.3f}"
        )

    print("\n=== Random baseline ===")
    random_result = run_random_experiment(SYNTHETIC_DATA, random_state=42)
    print(
        f"train={random_result.train_size} | "
        f"test={random_result.test_size} | "
        f"accuracy={random_result.accuracy:.3f} | "
        f"auroc={safe_metric(random_result.auroc)} | "
        f"auprc={safe_metric(random_result.auprc)} | "
        f"brier={safe_metric(random_result.brier)} | "
        f"f1={random_result.f1:.3f}"
    )


if __name__ == "__main__":
    main()