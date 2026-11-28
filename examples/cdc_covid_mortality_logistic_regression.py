# Contributor: Arun Vishnu (anv5@illinois.edu)
# Paper: Evaluating Model Performance in Medical Datasets Over Time
# https://proceedings.mlr.press/v209/zhou23a.html
# Example: Temporal and feature-set mortality ablation.

"""Temporal CDC COVID-19 mortality ablation with PyHealth LogisticRegression.

The experiment compares expanding-history and six-month sliding training on
the exact same future-month test cases, for demographic-only and expanded
pre-outcome feature sets. The default is fully synthetic, needs no download,
and runs on CPU. Point ``--data-root`` at a directory containing the official
``n8mc-b4w4.csv`` (or a bounded export with all 19 columns) for a real-data run.

This is a case-level surveillance analysis, not longitudinal patient modeling.
``hosp_yn`` and ``icu_yn`` are excluded to reduce care-trajectory leakage.
Fixed epochs avoid selecting hyperparameters on future months. AUROC is not a
calibration or prevalence estimate, especially for a label-stratified subset.

Examples:
    python examples/cdc_covid_mortality_logistic_regression.py
    python examples/cdc_covid_mortality_logistic_regression.py \
        --data-root /data/cdc --epochs 3 --results results.csv
"""

from __future__ import annotations

import argparse
import csv
import random
import tempfile
from pathlib import Path
from typing import Any

import torch

from pyhealth.datasets import CDCCOVIDDataset, get_dataloader
from pyhealth.models import LogisticRegression
from pyhealth.tasks import CDCCOVIDMortalityPrediction
from pyhealth.trainer import Trainer

FIELDS = (
    "case_month",
    "res_state",
    "state_fips_code",
    "res_county",
    "county_fips_code",
    "age_group",
    "sex",
    "race",
    "ethnicity",
    "case_positive_specimen",
    "case_onset_interval",
    "process",
    "exposure_yn",
    "current_status",
    "symptom_status",
    "hosp_yn",
    "icu_yn",
    "death_yn",
    "underlying_conditions_yn",
)


def month_range(start_year: int, start_month: int, count: int) -> list[str]:
    """Build consecutive YYYY-MM strings."""
    return [
        f"{start_year + (start_month - 1 + offset) // 12:04d}-"
        f"{(start_month - 1 + offset) % 12 + 1:02d}"
        for offset in range(count)
    ]


def synthetic_rows(seed: int = 7) -> list[dict[str, Any]]:
    """Create a small synthetic cohort with temporal signal drift."""
    rng = random.Random(seed)
    rows = []
    for month_index, month in enumerate(month_range(2020, 1, 18)):
        for case_index in range(60):
            old = rng.random() < 0.30
            symptomatic = rng.random() < 0.60
            underlying = rng.random() < 0.35
            # Risk drops over time and the association with symptoms weakens.
            score = (
                -2.4
                + 1.7 * old
                + (1.1 - 0.05 * month_index) * symptomatic
                + 0.9 * underlying
                - 0.035 * month_index
            )
            probability = 1.0 / (1.0 + pow(2.718281828, -score))
            death = rng.random() < probability
            rows.append(
                {
                    "case_month": month,
                    "res_state": "ZZ",
                    "state_fips_code": "00",
                    "res_county": "SYNTHETIC",
                    "county_fips_code": "00000",
                    "age_group": "65+ years" if old else "18 to 49 years",
                    "sex": "Female" if case_index % 2 else "Male",
                    "race": "White" if case_index % 3 else "Black",
                    "ethnicity": "Non-Hispanic/Latino",
                    "case_positive_specimen": str(rng.choice((0, 1, 2))),
                    "case_onset_interval": str(rng.choice((0, 1, 2))),
                    "process": "Routine surveillance",
                    "exposure_yn": "Yes" if case_index % 4 else "Missing",
                    "current_status": "Laboratory-confirmed case",
                    "symptom_status": "Symptomatic" if symptomatic else "Asymptomatic",
                    "hosp_yn": "Yes" if death else "No",
                    "icu_yn": "Yes" if death else "No",
                    "death_yn": "Yes" if death else "No",
                    "underlying_conditions_yn": "Yes" if underlying else "No",
                }
            )
    return rows


def write_synthetic(root: Path) -> None:
    """Write the synthetic cohort using the official CSV column order."""
    with (root / "n8mc-b4w4.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(synthetic_rows())


def train_evaluate(
    samples,
    train_indices: list[int],
    test_indices: list[int],
    epochs: int,
    seed: int,
) -> float:
    """Fit the existing PyHealth model and return future-month AUROC."""
    torch.manual_seed(seed)
    train = samples.subset(train_indices)
    test = samples.subset(test_indices)
    model = LogisticRegression(dataset=samples, embedding_dim=32)
    trainer = Trainer(
        model=model,
        metrics=["roc_auc"],
        device="cpu",
        enable_logging=False,
    )
    trainer.train(
        get_dataloader(train, batch_size=128, shuffle=True),
        epochs=epochs,
        optimizer_params={"lr": 0.01},
    )
    metrics = trainer.evaluate(get_dataloader(test, batch_size=256))
    return float(metrics["roc_auc"])


def run_ablation(root: Path, cache: Path, epochs: int, seed: int) -> list[dict]:
    """Run both feature and temporal-window ablations."""
    dataset = CDCCOVIDDataset(str(root), cache_dir=cache, num_workers=1)
    results = []
    for feature_set in ("demographics", "preoutcome"):
        samples = dataset.set_task(
            CDCCOVIDMortalityPrediction(feature_set), num_workers=1
        )
        months: dict[str, list[int]] = {}
        labels: dict[str, list[int]] = {}
        for index in range(len(samples)):
            sample = samples[index]
            month = sample["case_month"]
            months.setdefault(month, []).append(index)
            labels.setdefault(month, []).append(int(sample["mortality"].item()))
        ordered = sorted(months)
        eligible = [month for month in ordered if len(set(labels[month])) == 2]
        test_months = eligible[-4:]
        if not test_months:
            raise ValueError("No evaluation month contains both mortality classes")

        for test_month in test_months:
            prior = [month for month in ordered if month < test_month]
            for strategy, train_months in (
                ("expanding", prior),
                ("sliding_6m", prior[-6:]),
            ):
                train_indices = [
                    index for month in train_months for index in months[month]
                ]
                train_labels = {
                    label for month in train_months for label in labels[month]
                }
                if len(train_labels) < 2:
                    continue
                auroc = train_evaluate(
                    samples,
                    train_indices,
                    months[test_month],
                    epochs,
                    seed,
                )
                row = {
                    "feature_set": feature_set,
                    "strategy": strategy,
                    "test_month": test_month,
                    "train_months": len(train_months),
                    "train_n": len(train_indices),
                    "test_n": len(months[test_month]),
                    "test_deaths": sum(labels[test_month]),
                    "auroc": round(auroc, 4),
                }
                results.append(row)
                print(row)
        samples.close()
    return results


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--results", type=Path)
    return parser.parse_args()


def main() -> None:
    """Run on a supplied CDC export or the generated synthetic cohort."""
    args = parse_args()
    with tempfile.TemporaryDirectory() as temporary:
        temp = Path(temporary)
        root = args.data_root or temp / "synthetic"
        root.mkdir(parents=True, exist_ok=True)
        if args.data_root is None:
            write_synthetic(root)
        cache = args.cache_dir or temp / "cache"
        results = run_ablation(root, cache, args.epochs, args.seed)
        if args.results:
            args.results.parent.mkdir(parents=True, exist_ok=True)
            with args.results.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(results[0]))
                writer.writeheader()
                writer.writerows(results)


if __name__ == "__main__":
    main()
