"""Demo: temporal evaluation under realistic synthetic clinical shift.

This example generates synthetic clinical data with temporal drift and compares:
1. multiple temporal cutoffs
2. a random split baseline

The goal is to simulate the EMDOT idea more realistically than a tiny hand-made
toy dataset, while still remaining lightweight and reproducible.
"""

from pyhealth.tasks.temporal_evaluation import (
    generate_synthetic_temporal_shift_data,
    run_ablation,
    run_random_experiment,
)


def format_result(prefix: str, result) -> str:
    return (
        f"{prefix} | "
        f"train={result.train_size} | "
        f"test={result.test_size} | "
        f"accuracy={result.accuracy:.3f} | "
        f"auroc={result.auroc:.3f if result.auroc is not None else 'None'} | "
        f"auprc={result.auprc:.3f if result.auprc is not None else 'None'} | "
        f"brier={result.brier:.3f if result.brier is not None else 'None'} | "
        f"f1={result.f1:.3f}"
    )


def safe_metric(value):
    if value is None:
        return "None"
    return f"{value:.3f}"


def main() -> None:
    dataset = generate_synthetic_temporal_shift_data(
        n_patients_per_year=40,
        start_year=2010,
        end_year=2021,
        seed=42,
    )

    print("=== Synthetic temporal shift dataset ===")
    print(f"Total records: {len(dataset)}")
    print(f"Years: {dataset[0]['year']} to {dataset[-1]['year']}")
    print()

    print("=== Temporal ablation ===")
    temporal_results = run_ablation(dataset, split_years=[2013, 2015, 2017, 2019])
    for result in temporal_results:
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

    print()
    print("=== Random baseline ===")
    random_result = run_random_experiment(dataset, random_state=42)
    print(
        f"train={random_result.train_size} | "
        f"test={random_result.test_size} | "
        f"accuracy={random_result.accuracy:.3f} | "
        f"auroc={safe_metric(random_result.auroc)} | "
        f"auprc={safe_metric(random_result.auprc)} | "
        f"brier={safe_metric(random_result.brier)} | "
        f"f1={random_result.f1:.3f}"
    )

    print()
    print("Interpretation:")
    print(
        "This synthetic dataset introduces temporal drift, so temporal splits "
        "better reflect future deployment conditions than random splits."
    )


if __name__ == "__main__":
    main()