"""
Author: Mohsin Shah
NetID: mohsins2
Paper: A Case Study Exploring the Current Landscape of Synthetic Medical
       Record Generation with Commercial LLMs
Paper link: https://proceedings.mlr.press/v287/lin25a.html

Description:
    This PyHealth example reproduces/extends a small tabular mortality prediction
    experiment from the above paper using the eICU dataset.

    I use 10 hand-crafted ICU features and compare three training regimes:

    1) real_train_real_test:
        - Train MLP classifier on real eICU data
        - Evaluate on a held-out real test set

    2) synth_baseline_train_real_test:
        - Train on GPT-generated baseline synthetic patients
        - Evaluate on the same real test set

    3) synth_privacy_train_real_test:
        - Train on GPT-generated privacy-aware synthetic patients
        - Evaluate on the same real test set

    The goal is to demonstrate how PyHealth can work with small custom
    tabular EHR-style datasets and LLM-generated synthetic cohorts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd

from pyhealth.datasets import SampleEHRDataset, split_by_patient, get_dataloader
from pyhealth.models import MLP
from pyhealth.trainer import Trainer


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Raw GitHub URLs to your CSVs in:
#   https://github.com/mohsinposts/CS598-DLH-LLM-eICU
REAL_URL = (
    "https://raw.githubusercontent.com/"
    "mohsinposts/CS598-DLH-LLM-eICU/main/real_icu_10feat.csv"
)
SYNTH_BASELINE_URL = (
    "https://raw.githubusercontent.com/"
    "mohsinposts/CS598-DLH-LLM-eICU/main/synthetic_baseline_10feat_clean.csv"
)
SYNTH_PRIVACY_URL = (
    "https://raw.githubusercontent.com/"
    "mohsinposts/CS598-DLH-LLM-eICU/main/synthetic_privacy_10feat_clean.csv"
)

FEATURE_COLS = [
    "age",
    "gender",
    "diabetes",
    "heartrate",
    "meanbp",
    "respiratoryrate",
    "temperature",
    "wbc",
    "creatinine",
    "bun",
]
LABEL_COL = "icu_mortality"


@dataclass
class DatasetSplit:
    """Container for train/val/test splits for a given data source."""

    train: SampleEHRDataset
    val: SampleEHRDataset
    test: SampleEHRDataset


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def load_icu_table(url: str, source_name: str) -> pd.DataFrame:
    """Loads one of the 10-feature ICU CSVs and tags it with a source column.

    Args:
        url: HTTPS URL to the CSV file.
        source_name: String identifier for this source
            (e.g., "real", "synthetic_baseline", "synthetic_privacy").

    Returns:
        Pandas DataFrame with an extra "source" column.
    """
    df = pd.read_csv(url)
    df["source"] = source_name

    # Basic sanity checks so users see issues early if schema drifts.
    missing = set(FEATURE_COLS + [LABEL_COL]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns {missing} in {source_name} CSV")

    return df


def df_to_sample_ehr_dataset(
    df: pd.DataFrame,
    dataset_name: str,
    task_name: str = "icu_mortality_prediction",
) -> SampleEHRDataset:
    """Wraps a tabular ICU dataframe into a PyHealth SampleEHRDataset.

    Each row is treated as a single ICU visit. All features are stored as
    length-1 sequences to make them compatible with sequence models as well.

    Args:
        df: DataFrame with FEATURE_COLS and LABEL_COL.
        dataset_name: Unique name for the dataset (used by PyHealth).
        task_name: Name for the task (used by PyHealth).

    Returns:
        A SampleEHRDataset instance.
    """
    samples: List[Dict] = []

    for idx, row in df.reset_index(drop=True).iterrows():
        patient_id = f"{dataset_name}_p{idx}"
        visit_id = f"{dataset_name}_v{idx}"

        sample: Dict = {
            "patient_id": patient_id,
            "visit_id": visit_id,
            **{
                col: [float(row[col])]
                for col in FEATURE_COLS
            },
            "label": int(row[LABEL_COL]),
        }
        samples.append(sample)

    dataset = SampleEHRDataset(
        samples=samples,
        dataset_name=dataset_name,
        task_name=task_name,
    )
    dataset.stat()
    return dataset


def make_splits(
    dataset: SampleEHRDataset,
    split_ratio: Tuple[float, float, float] = (0.7, 0.1, 0.2),
) -> DatasetSplit:
    """Splits a SampleEHRDataset into patient-level train/val/test sets.

    Args:
        dataset: The full dataset to split.
        split_ratio: Tuple of (train, val, test) fractions. Defaults to
            (0.7, 0.1, 0.2).

    Returns:
        DatasetSplit object with train, val, and test SampleEHRDatasets.
    """
    train_ds, val_ds, test_ds = split_by_patient(dataset, list(split_ratio))
    return DatasetSplit(train=train_ds, val=val_ds, test=test_ds)


# ---------------------------------------------------------------------------
# Model / training helpers
# ---------------------------------------------------------------------------


def build_mlp_model(dataset: SampleEHRDataset) -> MLP:
    """Constructs a simple MLP baseline for tabular ICU features."""
    model = MLP(
        dataset=dataset,
        feature_keys=FEATURE_COLS,
        label_key="label",
        mode="binary",
        hidden_dim=1,
    )
    return model


def train_and_eval(
    dataset_split: DatasetSplit,
    full_dataset: SampleEHRDataset,
    device: str = "cpu",
    max_epochs: int = 50,
) -> Dict[str, float]:
    """Trains MLP on the given split and evaluates on test set.

    Args:
        dataset_split: DatasetSplit containing train/val/test sets.
        full_dataset: Full dataset before splitting (needed for model initialization).
        device: Device for training (e.g., "cpu", "cuda:0").
        max_epochs: Number of training epochs.

    Returns:
        Dict of evaluation metrics (ROC-AUC, PR-AUC, accuracy, F1).
    """
    train_loader = get_dataloader(dataset_split.train, batch_size=64, shuffle=True)
    val_loader = get_dataloader(dataset_split.val, batch_size=128, shuffle=False)
    test_loader = get_dataloader(dataset_split.test, batch_size=128, shuffle=False)

    model = build_mlp_model(full_dataset)

    trainer = Trainer(
        model=model,
        device=device,
        metrics=["roc_auc", "pr_auc", "accuracy", "f1"],
    )

    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=max_epochs,
        monitor="pr_auc",
    )

    metrics = trainer.evaluate(test_loader)
    return metrics


# ---------------------------------------------------------------------------
# Main experiment: real vs synthetic training sources
# ---------------------------------------------------------------------------


def run_eicu_llm_experiments(device: str = "cpu") -> pd.DataFrame:
    """Runs three training setups and returns a summary results table.

    1) real_train_real_test
    2) synth_baseline_train_real_test
    3) synth_privacy_train_real_test

    All setups evaluate on the same held-out real test set to simulate the
    paper’s “downstream utility on real patients” comparison.

    Returns:
        Pandas DataFrame with one row per setting and columns:
        ["setting", "train_source", "test_source", "roc_auc",
         "pr_auc", "accuracy", "f1"].
    """
    real_df = load_icu_table(REAL_URL, "real")
    synth_base_df = load_icu_table(SYNTH_BASELINE_URL, "synthetic_baseline")
    synth_priv_df = load_icu_table(SYNTH_PRIVACY_URL, "synthetic_privacy")

    # Build a real dataset and split once — I reuse the same real test set
    real_dataset = df_to_sample_ehr_dataset(real_df, dataset_name="eICU10F_real")
    real_split = make_splits(real_dataset)

    results_rows = []

    # ------------------------------------------------------------------
    # 1) real_train_real_test
    # ------------------------------------------------------------------
    real_metrics = train_and_eval(real_split, real_dataset, device=device, max_epochs=50)
    results_rows.append(
        {
            "setting": "real_train_real_test",
            "train_source": "real",
            "test_source": "real",
            **real_metrics,
        }
    )

    # ------------------------------------------------------------------
    # 2) synth_baseline_train_real_test
    #    Train on GPT baseline synthetic, test on the same real test set.
    # ------------------------------------------------------------------
    synth_base_dataset = df_to_sample_ehr_dataset(
        synth_base_df, dataset_name="eICU10F_synth_baseline"
    )
    # For simplicity I split synthetic data but always evaluate on real test
    synth_base_split = make_splits(synth_base_dataset)
    synth_base_metrics = train_and_eval(
        DatasetSplit(
            train=synth_base_split.train,
            val=synth_base_split.val,
            test=real_split.test,
        ),
        synth_base_dataset,
        device=device,
        max_epochs=50,
    )
    results_rows.append(
        {
            "setting": "synth_baseline_train_real_test",
            "train_source": "synthetic_baseline",
            "test_source": "real",
            **synth_base_metrics,
        }
    )

    # ------------------------------------------------------------------
    # 3) synth_privacy_train_real_test
    #    Train on GPT privacy-aware synthetic, test on real patients.
    synth_priv_dataset = df_to_sample_ehr_dataset(
        synth_priv_df, dataset_name="eICU10F_synth_privacy"
    )
    synth_priv_split = make_splits(synth_priv_dataset)
    synth_priv_metrics = train_and_eval(
        DatasetSplit(
            train=synth_priv_split.train,
            val=synth_priv_split.val,
            test=real_split.test,
        ),
        synth_priv_dataset,
        device=device,
        max_epochs=50,
    )
    results_rows.append(
        {
            "setting": "synth_privacy_train_real_test",
            "train_source": "synthetic_privacy",
            "test_source": "real",
            **synth_priv_metrics,
        }
    )

    results_df = pd.DataFrame(results_rows)
    return results_df


if __name__ == "__main__":
    results = run_eicu_llm_experiments(device="cpu")
    print("\n=== eICU LLM synthetic mortality experiment (PyHealth) ===")
    print(results.to_string(index=False))
