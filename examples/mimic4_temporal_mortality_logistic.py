"""Example: temporal vs random evaluation on MIMIC-IV task samples."""

from pyhealth.datasets import MIMIC4EHRDataset
from pyhealth.tasks.temporal_evaluation import (
    run_random_experiment,
    run_temporal_experiment,
    sample_dataset_to_temporal_records,
)
from pyhealth.tasks.temporal_risk_prediction import TemporalMortalityMIMIC4


def safe_metric(value):
    if value is None:
        return "None"
    return f"{value:.3f}"


def main() -> None:
    dataset = MIMIC4EHRDataset(
        root="YOUR_DATA_ROOT",
        tables=["admissions", "diagnoses_icd", "procedures_icd", "prescriptions"],
        dev=True,
    )

    sample_dataset = dataset.set_task(TemporalMortalityMIMIC4())
    records = sample_dataset_to_temporal_records(sample_dataset)

    temporal_result = run_temporal_experiment(records, split_year=2017)
    random_result = run_random_experiment(records, random_state=42)

    print("=== Temporal split ===")
    print(
        f"train={temporal_result.train_size} | "
        f"test={temporal_result.test_size} | "
        f"accuracy={temporal_result.accuracy:.3f} | "
        f"auroc={safe_metric(temporal_result.auroc)} | "
        f"auprc={safe_metric(temporal_result.auprc)} | "
        f"brier={safe_metric(temporal_result.brier)} | "
        f"f1={temporal_result.f1:.3f}"
    )

    print("\n=== Random split ===")
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