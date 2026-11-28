"""Compare CLIPStress tabular dropout on the native SUPPORT2 dataset.

Contributor: Siqi Zhu (siqizhu4).
Model paper: https://proceedings.mlr.press/v287/yang25a.html
Data: https://hbiostat.org/data/repo/supportdesc

Offline demo: python examples/support2_mortality_clip_stress.py --output-dir runs/demo
Real data: add --csv /path/to/support2.csv --epochs 50 --limit 1000 --seed 0.
The default is synthetic data loaded through Support2Dataset, not clinical evidence.

This example compares dropout 0, 0.1, 0.3 using only age, sex, comorbidity count,
diabetes, dementia and cancer status. The target is hospital death (hospdead).
Outcomes, follow-up times and prognostic scores never enter the predictors.
This is a limited tabular-only API example, not stress detection or a validation
of multimodal pretraining. No claim of deployment-ready prediction is intended.
"""

import argparse
import importlib.metadata
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pyhealth.datasets import Support2Dataset, create_sample_dataset
from pyhealth.models import CLIPStress

if __package__:
    from .clip_stress_experiment_utils import (
        file_hash,
        fit_classifier,
        seed_training,
        source_hashes,
        write_json,
    )
else:
    from clip_stress_experiment_utils import (
        file_hash,
        fit_classifier,
        seed_training,
        source_hashes,
        write_json,
    )


FEATURES = ("age", "sex", "num.co", "diabetes", "dementia", "ca")
CATEGORIES = {"sex": ("female", "male"), "ca": ("no", "yes", "metastatic")}


def synthetic_table():
    """Generate fixed pseudo records, independent of the training seed."""
    rng = np.random.default_rng(7)
    n = 100
    age = rng.uniform(30, 90, n)
    return pd.DataFrame(
        {
            "sno": np.arange(1, n + 1),
            "age": age,
            "sex": rng.choice(["female", "male"], n),
            "num.co": rng.integers(0, 5, n),
            "diabetes": rng.integers(0, 2, n),
            "dementia": rng.integers(0, 2, n),
            "ca": rng.choice(["no", "yes", "metastatic"], n),
            "hospdead": (age + rng.normal(0, 20, n) > 65).astype(int),
        }
    )


def load_records(csv_path, limit):
    """Load a deterministic ID subset using the existing Support2Dataset class."""
    table = pd.read_csv(csv_path) if csv_path else synthetic_table()
    # The original HBiostat CSV has a leading unnamed row ID absent from its
    # header; pandas preserves that explicit field as the dataframe index.
    if "sno" not in table and not isinstance(table.index, pd.RangeIndex):
        table = table.rename_axis("sno").reset_index()
    if "sno" not in table:
        id_column = next((x for x in ("id", "Unnamed: 0") if x in table), None)
        if id_column is None:
            raise ValueError("SUPPORT2 requires an explicit sno/id column.")
        table = table.rename(columns={id_column: "sno"})
    required = {"sno", "hospdead", *FEATURES}
    if not required.issubset(table.columns):
        raise ValueError(f"Missing columns: {required - set(table.columns)}")
    if table.sno.isna().any() or table.sno.duplicated().any():
        raise ValueError("Each patient must have one unique nonmissing identifier.")
    # IDs are numeric in the source, so cohort selection precedes string conversion.
    table = table.sort_values("sno", kind="stable").head(limit).copy()
    if not table.hospdead.isin([0, 1]).all():
        raise ValueError("Every selected hospital-death label must be binary.")
    rows = []
    with tempfile.TemporaryDirectory(prefix="support2-example-") as temporary:
        root = Path(temporary)
        # Keep only the allowlist and target even in the native dataset cache.
        table.loc[:, ["sno", *FEATURES, "hospdead"]].to_csv(
            root / "support2.csv",
            index=False,
        )
        # JSON is valid YAML. Restrict the native loader's attribute schema too.
        config = root / "config.yaml"
        write_json(
            config,
            {
                "version": "1.0",
                "tables": {
                    "support2": {
                        "file_path": "support2.csv",
                        "patient_id": "sno",
                        "timestamp": None,
                        "attributes": [*FEATURES, "hospdead"],
                    }
                },
            },
        )
        dataset = Support2Dataset(
            root=str(root),
            tables=["support2"],
            cache_dir=root / "cache",
            config_path=str(config),
        )
        for patient in dataset.iter_patients():
            events = patient.get_events(event_type="support2")
            if len(events) != 1:
                raise ValueError("Expected exactly one record per patient.")
            event = events[0]
            rows.append(
                {
                    "patient_id": str(patient.patient_id),
                    "label": int(event.hospdead),
                    **{key: getattr(event, key, None) for key in FEATURES},
                }
            )
    return sorted(rows, key=lambda row: int(row["patient_id"]))


def prepare_records(records, split_seed=7):
    """Encode a fixed feature allowlist and fit imputation/scaling on train only."""
    ids = np.array([row["patient_id"] for row in records])
    if len(ids) < 15 or len(np.unique(ids)) != len(ids):
        raise ValueError("Need at least 15 distinct patients.")
    order = np.random.default_rng(split_seed).permutation(len(ids))
    n = len(ids) // 5
    groups = {
        "train": order[2 * n :],
        "validation": order[:n],
        "test": order[n : 2 * n],
    }
    labels = np.array([row["label"] for row in records], dtype=np.int64)
    values, columns = [], []
    for feature in FEATURES:
        if feature in CATEGORIES:
            categories = CATEGORIES[feature]
            raw = [str(row[feature]).lower() for row in records]
            for category in (*categories, "unknown"):
                values.append(
                    [
                        float(
                            value == category
                            if category != "unknown"
                            else value not in categories
                        )
                        for value in raw
                    ]
                )
                columns.append(f"{feature}={category}")
        else:
            values.append(
                pd.to_numeric(
                    pd.Series([row[feature] for row in records]),
                    errors="coerce",
                ).to_numpy()
            )
            columns.append(feature)
    x = np.asarray(values, dtype=np.float32).T
    x[~np.isfinite(x)] = np.nan
    train_values = x[groups["train"]]
    medians = np.array(
        [
            np.nanmedian(col) if np.isfinite(col).any() else 0.0
            for col in train_values.T
        ],
        dtype=np.float32,
    )
    x = np.where(np.isnan(x), medians, x)
    mean, std = x[groups["train"]].mean(0), x[groups["train"]].std(0)
    x = ((x - mean) / np.maximum(std, 1e-6)).astype(np.float32)
    datasets = {}
    for name, indices in groups.items():
        if set(labels[indices]) != {0, 1}:
            raise ValueError(
                f"Both classes are required in {name}; do not tune splits."
            )
        samples = [
            {"patient_id": ids[i], "tabular": x[i], "label": int(labels[i])}
            for i in indices
        ]
        kwargs = (
            {}
            if name == "train"
            else {
                "input_processors": datasets["train"].input_processors,
                "output_processors": datasets["train"].output_processors,
            }
        )
        datasets[name] = create_sample_dataset(
            samples,
            {"tabular": "tensor"},
            {"label": "binary"},
            **kwargs,
        )
    metadata = {
        "feature_allowlist": list(FEATURES),
        "encoded_columns": columns,
        "train_imputation_median": medians.tolist(),
        "train_mean": mean.tolist(),
        "train_std": std.tolist(),
        "subjects": {name: ids[idx].tolist() for name, idx in groups.items()},
        "class_counts": {
            name: {str(y): int((labels[idx] == y).sum()) for y in (0, 1)}
            for name, idx in groups.items()
        },
    }
    return datasets, metadata


def execute(args):
    """Run the three fixed dropout configurations with validation-only selection."""
    if args.limit < 15 or min(args.epochs, args.threads) < 1 or args.seed < 0:
        raise ValueError("Invalid sample limit, seed, epochs or threads.")
    torch.set_num_threads(args.threads)
    records = load_records(args.csv, args.limit)
    datasets, metadata = prepare_records(records)
    report = {
        "complete": False,
        "synthetic": args.csv is None,
        "data_sha256": file_hash(args.csv) if args.csv else None,
        "source_sha256": source_hashes(),
        "preprocessing": metadata,
        "configuration": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "environment": {
            key: importlib.metadata.version(key)
            for key in ("torch", "numpy", "scipy", "pyhealth")
        },
        "results": {},
        "details": {},
    }
    for dropout in (0.0, 0.1, 0.3):
        seed_training(args.seed)
        model = CLIPStress(
            datasets["train"],
            signal_key=None,
            text_key=None,
            hidden_dim=64,
            projection_dim=32,
            dropout=dropout,
            residual_dropout=dropout,
            projection_dropout=dropout,
            stage="finetune",
        )
        name = f"dropout-{dropout:g}"
        metrics, details = fit_classifier(
            model,
            datasets["train"],
            datasets["validation"],
            datasets["test"],
            args,
            args.output_dir / name,
        )
        report["results"][name] = metrics
        report["details"][name] = details
        report["complete"] = len(report["results"]) == 3
        write_json(args.output_dir / "results.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    execute(parser.parse_args())
