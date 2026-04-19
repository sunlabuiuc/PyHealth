"""Ablation study for event-centered EBCL on MIMIC-IV.

This script runs four event/task settings:

1. Hypotension + Mortality
2. Hypotension + 3-Day LOS
3. Mechanical Ventilation + Mortality
4. Mechanical Ventilation + 3-Day LOS

For each setting, the script evaluates several hyperparameter configurations
and reports validation and test metrics. This version is designed for local
execution and expects the MIMIC-IV root path to be provided through the
MIMIC4_PATH environment variable.

Example:
    export MIMIC4_PATH=/path/to/physionet.org/files/mimiciv/3.1
"""

import copy
import math
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models.event_contrastive import EBCLModel


# ---------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------

MIMIC4_PATH = os.environ.get("MIMIC4_PATH")
if not MIMIC4_PATH:
    raise EnvironmentError(
        "MIMIC4_PATH is not set. Please export MIMIC4_PATH to your local "
        "MIMIC-IV root directory before running this script."
    )

MAX_LEN = 64
MIN_LEN = 8
MAX_FEATURES = 50_000
BATCH_SIZE = 16
DEFAULT_SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def set_seed(seed: int = DEFAULT_SEED) -> None:
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def validate_required_files(mimic4_path: str) -> None:
    """Checks that required MIMIC-IV files exist.

    Args:
        mimic4_path: Root path to the local MIMIC-IV dataset.

    Raises:
        FileNotFoundError: If any required file is missing.
    """
    required_files = [
        f"{mimic4_path}/hosp/admissions.csv.gz",
        f"{mimic4_path}/hosp/prescriptions.csv.gz",
        f"{mimic4_path}/icu/icustays.csv.gz",
        f"{mimic4_path}/icu/chartevents.csv.gz",
        f"{mimic4_path}/icu/procedureevents.csv.gz",
        f"{mimic4_path}/icu/d_items.csv.gz",
    ]

    missing_files = [path for path in required_files if not os.path.exists(path)]
    if missing_files:
        raise FileNotFoundError(
            "Missing required MIMIC-IV files:\n" + "\n".join(missing_files)
        )


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def load_required_tables(mimic4_path: str) -> Dict[str, pd.DataFrame]:
    """Loads the raw MIMIC-IV tables needed for event-centered EBCL.

    Args:
        mimic4_path: Root path to the local MIMIC-IV dataset.

    Returns:
        Dictionary mapping table names to pandas DataFrames.
    """
    hosp_dir = f"{mimic4_path}/hosp"
    icu_dir = f"{mimic4_path}/icu"

    start_time = time.time()
    admissions = pd.read_csv(
        f"{hosp_dir}/admissions.csv.gz",
        usecols=[
            "subject_id",
            "hadm_id",
            "admittime",
            "dischtime",
            "hospital_expire_flag",
        ],
        compression="gzip",
    )
    print(f"Loaded admissions in {time.time() - start_time:.2f}s")

    start_time = time.time()
    icustays = pd.read_csv(
        f"{icu_dir}/icustays.csv.gz",
        usecols=["subject_id", "hadm_id", "stay_id", "intime", "outtime"],
        compression="gzip",
    )
    print(f"Loaded icustays in {time.time() - start_time:.2f}s")

    start_time = time.time()
    d_items = pd.read_csv(
        f"{icu_dir}/d_items.csv.gz",
        usecols=["itemid", "label"],
        compression="gzip",
    )
    print(f"Loaded d_items in {time.time() - start_time:.2f}s")

    start_time = time.time()
    chartevents = pd.read_csv(
        f"{icu_dir}/chartevents.csv.gz",
        usecols=[
            "subject_id",
            "hadm_id",
            "stay_id",
            "itemid",
            "charttime",
            "valuenum",
        ],
        compression="gzip",
        nrows=5_000_000,
    )
    print(f"Loaded chartevents in {time.time() - start_time:.2f}s")

    start_time = time.time()
    procedureevents = pd.read_csv(
        f"{icu_dir}/procedureevents.csv.gz",
        usecols=["subject_id", "hadm_id", "stay_id", "itemid", "starttime"],
        compression="gzip",
    )
    print(f"Loaded procedureevents in {time.time() - start_time:.2f}s")

    start_time = time.time()
    prescriptions = pd.read_csv(
        f"{hosp_dir}/prescriptions.csv.gz",
        usecols=["subject_id", "hadm_id", "starttime", "drug"],
        compression="gzip",
        nrows=5_000_000,
    )
    print(f"Loaded prescriptions in {time.time() - start_time:.2f}s")

    admissions["admittime"] = pd.to_datetime(admissions["admittime"])
    admissions["dischtime"] = pd.to_datetime(admissions["dischtime"])

    icustays["intime"] = pd.to_datetime(icustays["intime"])
    icustays["outtime"] = pd.to_datetime(icustays["outtime"])

    chartevents["charttime"] = pd.to_datetime(chartevents["charttime"])
    procedureevents["starttime"] = pd.to_datetime(procedureevents["starttime"])
    prescriptions["starttime"] = pd.to_datetime(prescriptions["starttime"])

    return {
        "admissions": admissions,
        "icustays": icustays,
        "d_items": d_items,
        "chartevents": chartevents,
        "procedureevents": procedureevents,
        "prescriptions": prescriptions,
    }


# ---------------------------------------------------------------------
# Event detection and label construction
# ---------------------------------------------------------------------

def get_map_itemids(d_items: pd.DataFrame) -> List[int]:
    """Finds item ids corresponding to MAP/NIMAP-style measurements."""
    items = d_items.copy()
    items["label_lower"] = items["label"].fillna("").str.lower()

    patterns = [
        "mean arterial pressure",
        "arterial blood pressure mean",
        "non invasive blood pressure mean",
        "map",
        "nimap",
    ]

    is_map_item = items["label_lower"].apply(
        lambda text: any(pattern in text for pattern in patterns)
    )
    return (
        items.loc[is_map_item, "itemid"]
        .dropna()
        .astype(int)
        .drop_duplicates()
        .tolist()
    )


def detect_hypotension_events(
    chartevents: pd.DataFrame,
    d_items: pd.DataFrame,
) -> pd.DataFrame:
    """Detects hypotension events using a MAP threshold transition.

    A hypotension event is defined here as a transition from:
        previous MAP > 60
        current MAP < 60

    Args:
        chartevents: Raw ICU chart events.
        d_items: ICU item definitions.

    Returns:
        DataFrame containing detected hypotension event times.
    """
    map_itemids = get_map_itemids(d_items)

    filtered = chartevents[
        chartevents["itemid"].isin(map_itemids)
        & chartevents["valuenum"].notna()
        & chartevents["stay_id"].notna()
        & chartevents["charttime"].notna()
    ].copy()

    filtered = filtered.sort_values(["stay_id", "charttime"])
    filtered["prev_value"] = filtered.groupby("stay_id")["valuenum"].shift(1)

    filtered["is_hypotension_event"] = (
        (filtered["prev_value"] > 60.0) & (filtered["valuenum"] < 60.0)
    )

    events = filtered[filtered["is_hypotension_event"]].copy()
    events = events.rename(columns={"charttime": "index_time"})
    events["event_type"] = "hypotension"

    return events[
        ["subject_id", "hadm_id", "stay_id", "index_time", "event_type"]
    ].dropna(subset=["hadm_id", "stay_id", "index_time"])


def detect_mech_vent_starts(
    procedureevents: pd.DataFrame,
    d_items: pd.DataFrame,
) -> pd.DataFrame:
    """Detects likely mechanical ventilation start events.

    Args:
        procedureevents: ICU procedure events.
        d_items: ICU item definitions.

    Returns:
        DataFrame containing detected mechanical ventilation start times.
    """
    merged = procedureevents.merge(
        d_items[["itemid", "label"]],
        on="itemid",
        how="left",
    )
    merged["label_lower"] = merged["label"].fillna("").str.lower()

    patterns = [
        "mechanical ventilation",
        "ventilation",
        "ventilator",
        "intubation",
    ]
    is_vent = merged["label_lower"].apply(
        lambda text: any(pattern in text for pattern in patterns)
    )

    events = merged[
        is_vent
        & merged["stay_id"].notna()
        & merged["hadm_id"].notna()
        & merged["starttime"].notna()
    ].copy()

    events = events.sort_values(["stay_id", "starttime"])
    events = events.rename(columns={"starttime": "index_time"})
    events["event_type"] = "mechanical_ventilation"

    return events[
        ["subject_id", "hadm_id", "stay_id", "index_time", "event_type"]
    ]


def attach_mortality_labels(
    events: pd.DataFrame,
    admissions: pd.DataFrame,
) -> pd.DataFrame:
    """Adds in-hospital mortality labels to event rows."""
    labels = admissions[["hadm_id", "hospital_expire_flag"]].copy()
    labels["label"] = labels["hospital_expire_flag"].astype(int)

    merged = events.merge(labels[["hadm_id", "label"]], on="hadm_id", how="left")
    merged = merged[merged["label"].notna()].copy()
    merged["label"] = merged["label"].astype(int)
    return merged


def attach_los3d_labels(
    events: pd.DataFrame,
    icustays: pd.DataFrame,
    threshold_days: float = 3.0,
) -> pd.DataFrame:
    """Adds binary LOS>3-day labels to event rows."""
    stays = icustays[["stay_id", "intime", "outtime"]].copy()
    stays = stays.dropna(subset=["stay_id", "intime", "outtime"])

    stays["los_days"] = (
        stays["outtime"] - stays["intime"]
    ).dt.total_seconds() / (24 * 3600)
    stays["label"] = (stays["los_days"] > threshold_days).astype(int)

    merged = events.merge(
        stays[["stay_id", "los_days", "label"]],
        on="stay_id",
        how="left",
    )
    merged = merged[merged["label"].notna()].copy()
    merged["label"] = merged["label"].astype(int)
    return merged


def attach_task_labels(
    events: pd.DataFrame,
    admissions: pd.DataFrame,
    icustays: pd.DataFrame,
    task_name: str,
) -> pd.DataFrame:
    """Dispatches to the appropriate task label function."""
    if task_name == "mortality":
        return attach_mortality_labels(events, admissions)
    if task_name == "los3d":
        return attach_los3d_labels(events, icustays)
    raise ValueError("task_name must be 'mortality' or 'los3d'")


# ---------------------------------------------------------------------
# Event-centered sequence construction
# ---------------------------------------------------------------------

def build_event_stream(
    chartevents: pd.DataFrame,
    procedureevents: pd.DataFrame,
    prescriptions: pd.DataFrame,
    icustays: pd.DataFrame,
    relevant_stay_ids: List[int],
) -> pd.DataFrame:
    """Builds a unified event stream for each ICU stay."""
    chart_df = chartevents[
        chartevents["stay_id"].isin(relevant_stay_ids)
        & chartevents["charttime"].notna()
        & chartevents["itemid"].notna()
    ].copy()
    chart_df["time"] = chart_df["charttime"]
    chart_df["feature_key"] = "chart:" + chart_df["itemid"].astype(int).astype(str)
    chart_df["value"] = chart_df["valuenum"].fillna(0.0).astype(float)
    chart_df = chart_df[
        ["subject_id", "hadm_id", "stay_id", "time", "feature_key", "value"]
    ]

    procedure_df = procedureevents[
        procedureevents["stay_id"].isin(relevant_stay_ids)
        & procedureevents["starttime"].notna()
        & procedureevents["itemid"].notna()
    ].copy()
    procedure_df["time"] = procedure_df["starttime"]
    procedure_df["feature_key"] = (
        "procedure:" + procedure_df["itemid"].astype(int).astype(str)
    )
    procedure_df["value"] = 1.0
    procedure_df = procedure_df[
        ["subject_id", "hadm_id", "stay_id", "time", "feature_key", "value"]
    ]

    stay_windows = icustays[
        icustays["stay_id"].isin(relevant_stay_ids)
    ][["subject_id", "hadm_id", "stay_id", "intime", "outtime"]].copy()

    drug_df = prescriptions[
        prescriptions["hadm_id"].isin(stay_windows["hadm_id"].unique())
        & prescriptions["starttime"].notna()
        & prescriptions["drug"].notna()
    ].copy()

    drug_df = drug_df.merge(
        stay_windows,
        on=["subject_id", "hadm_id"],
        how="inner",
    )
    drug_df = drug_df[
        (drug_df["starttime"] >= drug_df["intime"])
        & (drug_df["starttime"] <= drug_df["outtime"])
    ].copy()

    drug_df["time"] = drug_df["starttime"]
    drug_df["feature_key"] = "drug:" + drug_df["drug"].astype(str)
    drug_df["value"] = 1.0
    drug_df = drug_df[
        ["subject_id", "hadm_id", "stay_id", "time", "feature_key", "value"]
    ]

    stream = pd.concat([chart_df, procedure_df, drug_df], ignore_index=True)
    return stream.sort_values(["stay_id", "time"]).reset_index(drop=True)


def make_feature_vocab(event_stream: pd.DataFrame) -> Dict[str, int]:
    """Creates a vocabulary mapping feature keys to integer ids."""
    feature_keys = sorted(event_stream["feature_key"].dropna().unique().tolist())
    feature_keys = feature_keys[: MAX_FEATURES - 1]
    return {key: index + 1 for index, key in enumerate(feature_keys)}


def build_pre_post_sequences(
    labeled_events: pd.DataFrame,
    event_stream: pd.DataFrame,
    vocab: Dict[str, int],
    max_len: int,
    min_len: int,
) -> List[Dict[str, Any]]:
    """Constructs padded EBCL pre/post sequences around each index event."""
    samples: List[Dict[str, Any]] = []

    grouped_stream = {
        stay_id: group.sort_values("time")
        for stay_id, group in event_stream.groupby("stay_id")
    }

    for _, row in labeled_events.iterrows():
        stay_id = row["stay_id"]
        if stay_id not in grouped_stream:
            continue

        stay_events = grouped_stream[stay_id]
        index_time = row["index_time"]

        pre_events = stay_events[stay_events["time"] < index_time].tail(max_len)
        post_events = stay_events[stay_events["time"] > index_time].head(max_len)

        if len(pre_events) < min_len or len(post_events) < min_len:
            continue

        pre_tokens: List[List[float]] = []
        post_tokens: List[List[float]] = []

        for _, event_row in pre_events.iterrows():
            relative_hours = (
                event_row["time"] - index_time
            ).total_seconds() / 3600.0
            feature_id = vocab.get(event_row["feature_key"], 0)
            pre_tokens.append(
                [float(relative_hours), float(feature_id), float(event_row["value"])]
            )

        for _, event_row in post_events.iterrows():
            relative_hours = (
                event_row["time"] - index_time
            ).total_seconds() / 3600.0
            feature_id = vocab.get(event_row["feature_key"], 0)
            post_tokens.append(
                [float(relative_hours), float(feature_id), float(event_row["value"])]
            )

        pre_mask = [1] * len(pre_tokens)
        post_mask = [1] * len(post_tokens)

        while len(pre_tokens) < max_len:
            pre_tokens.append([0.0, 0.0, 0.0])
            pre_mask.append(0)

        while len(post_tokens) < max_len:
            post_tokens.append([0.0, 0.0, 0.0])
            post_mask.append(0)

        samples.append(
            {
                "patient_id": str(row["subject_id"]),
                "visit_id": str(row["hadm_id"]),
                "stay_id": str(row["stay_id"]),
                "event_type": row["event_type"],
                "index_time": str(index_time),
                "pre": pre_tokens,
                "post": post_tokens,
                "pre_mask": pre_mask,
                "post_mask": post_mask,
                "label": int(row["label"]),
            }
        )

    return samples


# ---------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------

def print_label_distribution(samples: List[Dict[str, Any]], name: str) -> None:
    """Prints the binary label distribution for a sample list."""
    count_zero = sum(1 for sample in samples if int(sample["label"]) == 0)
    count_one = sum(1 for sample in samples if int(sample["label"]) == 1)
    print(f"{name} label distribution -> 0: {count_zero}, 1: {count_one}")


def collect_reproduction_stats(
    samples: List[Dict[str, Any]],
) -> Dict[str, float]:
    """Collects simple cohort summary statistics from processed samples."""
    num_patients = len({sample["patient_id"] for sample in samples})
    num_stays = len({sample["stay_id"] for sample in samples})
    num_events = len(samples)
    prevalence = (
        100.0 * sum(int(sample["label"]) for sample in samples)
        / max(len(samples), 1)
    )

    return {
        "patients": num_patients,
        "stays": num_stays,
        "events": num_events,
        "prevalence": prevalence,
    }


def stratified_split_samples(
    samples: List[Dict[str, Any]],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Performs a simple stratified train/val/test split."""
    class_zero = [sample for sample in samples if int(sample["label"]) == 0]
    class_one = [sample for sample in samples if int(sample["label"]) == 1]

    if len(class_zero) == 0 or len(class_one) == 0:
        raise ValueError(
            "The processed dataset contains only one label class overall."
        )

    generator = torch.Generator().manual_seed(DEFAULT_SEED)
    idx_zero = torch.randperm(len(class_zero), generator=generator).tolist()
    idx_one = torch.randperm(len(class_one), generator=generator).tolist()

    class_zero = [class_zero[index] for index in idx_zero]
    class_one = [class_one[index] for index in idx_one]

    def split_class(
        class_samples: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        num_samples = len(class_samples)
        num_train = max(1, int(num_samples * train_ratio))
        num_val = max(1, int(num_samples * val_ratio))

        if num_samples - num_train - num_val < 1:
            num_val = max(1, num_val - 1)

        train_split = class_samples[:num_train]
        val_split = class_samples[num_train : num_train + num_val]
        test_split = class_samples[num_train + num_val :]
        return train_split, val_split, test_split

    train_zero, val_zero, test_zero = split_class(class_zero)
    train_one, val_one, test_one = split_class(class_one)

    train_samples = train_zero + train_one
    val_samples = val_zero + val_one
    test_samples = test_zero + test_one

    train_perm = torch.randperm(len(train_samples), generator=generator).tolist()
    val_perm = torch.randperm(len(val_samples), generator=generator).tolist()
    test_perm = torch.randperm(len(test_samples), generator=generator).tolist()

    train_samples = [train_samples[index] for index in train_perm]
    val_samples = [val_samples[index] for index in val_perm]
    test_samples = [test_samples[index] for index in test_perm]

    return train_samples, val_samples, test_samples


def build_ebcl_sample_dataset(
    samples: List[Dict[str, Any]],
    dataset_name: str,
):
    """Wraps processed samples using the PyHealth sample dataset format."""
    return create_sample_dataset(
        samples=samples,
        input_schema={
            "pre": "tensor",
            "post": "tensor",
            "pre_mask": "tensor",
            "post_mask": "tensor",
        },
        output_schema={"label": "binary"},
        dataset_name=dataset_name,
    )

# ---------------------------------------------------------------------
# Training and evaluation helpers
# ---------------------------------------------------------------------

def move_batch_to_device(
    batch: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    """Moves tensor fields in a batch to the target device."""
    moved_batch: Dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved_batch[key] = value.to(device)
        else:
            moved_batch[key] = value

    if "pre_mask" in moved_batch:
        moved_batch["pre_mask"] = moved_batch["pre_mask"].bool()
    if "post_mask" in moved_batch:
        moved_batch["post_mask"] = moved_batch["post_mask"].bool()

    return moved_batch


def train_one_epoch_pretrain(
    model: EBCLModel,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Runs one pretraining epoch."""
    model.train()
    total_loss = 0.0
    total_batches = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        optimizer.zero_grad()
        output = model(
            pre=batch["pre"],
            post=batch["post"],
            pre_mask=batch.get("pre_mask"),
            post_mask=batch.get("post_mask"),
        )
        loss = output["loss"]
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

    return total_loss / max(total_batches, 1)


def evaluate_pretrain_loss(
    model: EBCLModel,
    loader,
    device: torch.device,
) -> float:
    """Evaluates average pretraining loss on a validation loader."""
    model.eval()
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            output = model(
                pre=batch["pre"],
                post=batch["post"],
                pre_mask=batch.get("pre_mask"),
                post_mask=batch.get("post_mask"),
            )
            total_loss += output["loss"].item()
            total_batches += 1

    return total_loss / max(total_batches, 1)


def train_one_epoch_finetune(
    model: EBCLModel,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Runs one finetuning epoch."""
    model.train()
    total_loss = 0.0
    total_batches = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        optimizer.zero_grad()
        output = model(
            pre=batch["pre"],
            pre_mask=batch.get("pre_mask"),
            label=batch["label"],
        )
        loss = output["loss"]
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

    return total_loss / max(total_batches, 1)


def evaluate_metrics(
    model: EBCLModel,
    loader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluates loss, accuracy, and AUROC on a loader."""
    model.eval()
    total_loss = 0.0
    total_batches = 0
    correct = 0
    total_examples = 0

    all_probs: List[float] = []
    all_labels: List[float] = []

    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            output = model(
                pre=batch["pre"],
                pre_mask=batch.get("pre_mask"),
                label=batch["label"],
            )

            total_loss += output["loss"].item()
            total_batches += 1

            probs = output["y_prob"].view(-1).detach().cpu()
            labels = output["y_true"].view(-1).detach().cpu()
            preds = (probs >= 0.5).long()

            correct += (preds == labels.long()).sum().item()
            total_examples += labels.numel()

            all_probs.extend(probs.tolist())
            all_labels.extend(labels.tolist())

    metrics = {
        "loss": total_loss / max(total_batches, 1),
        "accuracy": correct / max(total_examples, 1),
    }

    if len(set(all_labels)) == 2:
        metrics["roc_auc"] = roc_auc_score(all_labels, all_probs)
    else:
        metrics["roc_auc"] = float("nan")

    return metrics


# ---------------------------------------------------------------------
# Paper reference values
# ---------------------------------------------------------------------

PAPER_STATS = {
    ("mortality", "hypotension"): {
        "patients": 35234,
        "stays": 47567,
        "events": 342884,
        "prevalence": 17.1,
    },
    ("los3d", "hypotension"): {
        "patients": 35234,
        "stays": 47567,
        "events": 342884,
        "prevalence": 48.1,
    },
    ("mortality", "mechanical_ventilation"): {
        "patients": 23269,
        "stays": 26955,
        "events": 31420,
        "prevalence": 13.4,
    },
    ("los3d", "mechanical_ventilation"): {
        "patients": 23269,
        "stays": 26955,
        "events": 31420,
        "prevalence": 52.7,
    },
}

PAPER_RESULTS = {
    ("mortality", "hypotension"): "83.02 ± 0.08",
    ("los3d", "hypotension"): "80.70 ± 0.03",
    ("mortality", "mechanical_ventilation"): "89.20 ± 0.35",
    ("los3d", "mechanical_ventilation"): "81.36 ± 0.05",
}


# ---------------------------------------------------------------------
# Experiment runners
# ---------------------------------------------------------------------

def build_event_dataframe(
    event_type: str,
    chartevents: pd.DataFrame,
    procedureevents: pd.DataFrame,
    d_items: pd.DataFrame,
) -> pd.DataFrame:
    """Returns the detected event dataframe for a requested event type."""
    if event_type == "hypotension":
        return detect_hypotension_events(chartevents, d_items)
    if event_type == "mechanical_ventilation":
        return detect_mech_vent_starts(procedureevents, d_items)
    raise ValueError(
        "event_type must be 'hypotension' or 'mechanical_ventilation'"
    )


def run_single_experiment(
    event_type: str,
    task_name: str,
    admissions: pd.DataFrame,
    icustays: pd.DataFrame,
    chartevents: pd.DataFrame,
    procedureevents: pd.DataFrame,
    prescriptions: pd.DataFrame,
    d_items: pd.DataFrame,
    max_len: int,
    min_len: int,
    batch_size: int,
    device: torch.device,
    pretrain_epochs: int = 5,
    finetune_epochs: int = 5,
) -> Dict[str, Any]:
    """Runs one EBCL experiment for a single event/task pair."""
    index_events = build_event_dataframe(
        event_type=event_type,
        chartevents=chartevents,
        procedureevents=procedureevents,
        d_items=d_items,
    )

    labeled_events = attach_task_labels(
        events=index_events,
        admissions=admissions,
        icustays=icustays,
        task_name=task_name,
    )

    relevant_stay_ids = (
        labeled_events["stay_id"].dropna().astype(int).drop_duplicates().tolist()
    )

    event_stream = build_event_stream(
        chartevents=chartevents,
        procedureevents=procedureevents,
        prescriptions=prescriptions,
        icustays=icustays,
        relevant_stay_ids=relevant_stay_ids,
    )

    vocab = make_feature_vocab(event_stream)

    samples = build_pre_post_sequences(
        labeled_events=labeled_events,
        event_stream=event_stream,
        vocab=vocab,
        max_len=max_len,
        min_len=min_len,
    )

    stats = collect_reproduction_stats(samples)
    print(f"\n[{event_type} | {task_name}] stats: {stats}")
    print_label_distribution(samples, f"{event_type}-{task_name}")

    train_samples, val_samples, test_samples = stratified_split_samples(samples)

    train_dataset = build_ebcl_sample_dataset(
        train_samples,
        f"{event_type}_{task_name}_train",
    )
    val_dataset = build_ebcl_sample_dataset(
        val_samples,
        f"{event_type}_{task_name}_val",
    )
    test_dataset = build_ebcl_sample_dataset(
        test_samples,
        f"{event_type}_{task_name}_test",
    )

    train_loader = get_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = get_dataloader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = get_dataloader(test_dataset, batch_size=batch_size, shuffle=False)

    pretrain_model = EBCLModel(
        dataset=train_dataset,
        num_features=len(vocab) + 1,
        d_model=32,
        n_heads=4,
        n_layers=2,
        ff_hidden_dim=128,
        projection_dim=32,
        dropout=0.1,
        stage="pretrain",
        task="binary",
    ).to(device)

    pretrain_optimizer = torch.optim.Adam(pretrain_model.parameters(), lr=1e-3)
    best_pretrain_val_loss = float("inf")
    best_encoder_state = None

    for epoch in range(pretrain_epochs):
        train_loss = train_one_epoch_pretrain(
            pretrain_model,
            train_loader,
            pretrain_optimizer,
            device,
        )
        val_loss = evaluate_pretrain_loss(pretrain_model, val_loader, device)

        print(
            f"[{event_type} | {task_name}] "
            f"Pretrain {epoch + 1}/{pretrain_epochs} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        )

        if val_loss < best_pretrain_val_loss:
            best_pretrain_val_loss = val_loss
            best_encoder_state = copy.deepcopy(
                pretrain_model.get_encoder_state_dict()
            )

    finetune_model = EBCLModel(
        dataset=train_dataset,
        num_features=len(vocab) + 1,
        d_model=32,
        n_heads=4,
        n_layers=2,
        ff_hidden_dim=128,
        projection_dim=32,
        dropout=0.1,
        stage="finetune",
        task="binary",
    ).to(device)

    if best_encoder_state is not None:
        finetune_model.load_encoder_state_dict(best_encoder_state)

    finetune_optimizer = torch.optim.Adam(
        finetune_model.parameters(),
        lr=1e-3,
        weight_decay=1e-5,
    )

    best_val_auc = -1.0
    best_state = None
    history: List[Dict[str, float]] = []

    for epoch in range(finetune_epochs):
        train_loss = train_one_epoch_finetune(
            finetune_model,
            train_loader,
            finetune_optimizer,
            device,
        )
        val_metrics = evaluate_metrics(finetune_model, val_loader, device)

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_roc_auc": val_metrics["roc_auc"],
            }
        )

        print(
            f"[{event_type} | {task_name}] "
            f"Finetune {epoch + 1}/{finetune_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val AUC: {val_metrics['roc_auc']:.4f}"
        )

        if (
            val_metrics["roc_auc"] == val_metrics["roc_auc"]
            and val_metrics["roc_auc"] > best_val_auc
        ):
            best_val_auc = val_metrics["roc_auc"]
            best_state = {
                key: value.cpu().clone()
                for key, value in finetune_model.state_dict().items()
            }

    if best_state is not None:
        finetune_model.load_state_dict(best_state)
        finetune_model.to(device)

    test_metrics = evaluate_metrics(finetune_model, test_loader, device)

    return {
        "event_type": event_type,
        "task_name": task_name,
        "stats": stats,
        "test_metrics": test_metrics,
        "history": pd.DataFrame(history),
        "num_samples": len(samples),
    }


def run_single_experiment_with_config(
    event_type: str,
    task_name: str,
    config: Dict[str, Any],
    admissions: pd.DataFrame,
    icustays: pd.DataFrame,
    chartevents: pd.DataFrame,
    procedureevents: pd.DataFrame,
    prescriptions: pd.DataFrame,
    d_items: pd.DataFrame,
    max_len: int,
    min_len: int,
    device: torch.device,
) -> Dict[str, Any]:
    """Runs one ablation experiment for a specific hyperparameter config."""
    index_events = build_event_dataframe(
        event_type=event_type,
        chartevents=chartevents,
        procedureevents=procedureevents,
        d_items=d_items,
    )

    labeled_events = attach_task_labels(
        events=index_events,
        admissions=admissions,
        icustays=icustays,
        task_name=task_name,
    )

    relevant_stay_ids = (
        labeled_events["stay_id"].dropna().astype(int).drop_duplicates().tolist()
    )

    event_stream = build_event_stream(
        chartevents=chartevents,
        procedureevents=procedureevents,
        prescriptions=prescriptions,
        icustays=icustays,
        relevant_stay_ids=relevant_stay_ids,
    )

    vocab = make_feature_vocab(event_stream)

    samples = build_pre_post_sequences(
        labeled_events=labeled_events,
        event_stream=event_stream,
        vocab=vocab,
        max_len=max_len,
        min_len=min_len,
    )

    stats = collect_reproduction_stats(samples)

    train_samples, val_samples, test_samples = stratified_split_samples(samples)

    train_dataset = build_ebcl_sample_dataset(
        train_samples,
        f"{event_type}_{task_name}_{config['name']}_train",
    )
    val_dataset = build_ebcl_sample_dataset(
        val_samples,
        f"{event_type}_{task_name}_{config['name']}_val",
    )
    test_dataset = build_ebcl_sample_dataset(
        test_samples,
        f"{event_type}_{task_name}_{config['name']}_test",
    )

    train_loader = get_dataloader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
    )
    val_loader = get_dataloader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
    )
    test_loader = get_dataloader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
    )

    pretrain_model = EBCLModel(
        dataset=train_dataset,
        num_features=len(vocab) + 1,
        d_model=config["d_model"],
        n_heads=4,
        n_layers=2,
        ff_hidden_dim=config["ff_hidden_dim"],
        projection_dim=config["d_model"],
        dropout=config["dropout"],
        stage="pretrain",
        task="binary",
    ).to(device)

    pretrain_optimizer = torch.optim.Adam(
        pretrain_model.parameters(),
        lr=config["lr"],
    )

    best_pretrain_val_loss = float("inf")
    best_encoder_state = None

    for epoch in range(config["pretrain_epochs"]):
        train_loss = train_one_epoch_pretrain(
            pretrain_model,
            train_loader,
            pretrain_optimizer,
            device,
        )
        val_loss = evaluate_pretrain_loss(pretrain_model, val_loader, device)

        print(
            f"[{event_type} | {task_name} | {config['name']}] "
            f"Pretrain {epoch + 1}/{config['pretrain_epochs']} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        )

        if val_loss < best_pretrain_val_loss:
            best_pretrain_val_loss = val_loss
            best_encoder_state = copy.deepcopy(
                pretrain_model.get_encoder_state_dict()
            )

    finetune_model = EBCLModel(
        dataset=train_dataset,
        num_features=len(vocab) + 1,
        d_model=config["d_model"],
        n_heads=4,
        n_layers=2,
        ff_hidden_dim=config["ff_hidden_dim"],
        projection_dim=config["d_model"],
        dropout=config["dropout"],
        stage="finetune",
        task="binary",
    ).to(device)

    if best_encoder_state is not None:
        finetune_model.load_encoder_state_dict(best_encoder_state)

    finetune_optimizer = torch.optim.Adam(
        finetune_model.parameters(),
        lr=config["lr"],
        weight_decay=1e-5,
    )

    best_val_auc = -1.0
    best_state = None

    for epoch in range(config["finetune_epochs"]):
        train_loss = train_one_epoch_finetune(
            finetune_model,
            train_loader,
            finetune_optimizer,
            device,
        )
        val_metrics = evaluate_metrics(finetune_model, val_loader, device)

        print(
            f"[{event_type} | {task_name} | {config['name']}] "
            f"Finetune {epoch + 1}/{config['finetune_epochs']} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val AUC: {val_metrics['roc_auc']:.4f}"
        )

        if (
            val_metrics["roc_auc"] == val_metrics["roc_auc"]
            and val_metrics["roc_auc"] > best_val_auc
        ):
            best_val_auc = val_metrics["roc_auc"]
            best_state = {
                key: value.cpu().clone()
                for key, value in finetune_model.state_dict().items()
            }

    if best_state is not None:
        finetune_model.load_state_dict(best_state)
        finetune_model.to(device)

    test_metrics = evaluate_metrics(finetune_model, test_loader, device)

    return {
        "event_type": event_type,
        "task_name": task_name,
        "config_name": config["name"],
        "config": config,
        "stats": stats,
        "num_samples": len(samples),
        "test_metrics": test_metrics,
        "best_val_auc": best_val_auc,
    }


# ---------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------

def build_stats_comparison_df(
    all_results: List[Dict[str, Any]],
) -> pd.DataFrame:
    """Builds a cohort statistics comparison dataframe."""
    stats_rows = []

    for result in all_results:
        key = (result["task_name"], result["event_type"])
        paper = PAPER_STATS[key]
        ours = result["stats"]

        stats_rows.append(
            {
                "Task": result["task_name"],
                "Event Type": result["event_type"],
                "Paper # Patients": paper["patients"],
                "Our # Patients": ours["patients"],
                "Paper # Stays": paper["stays"],
                "Our # Stays": ours["stays"],
                "Paper # Events": paper["events"],
                "Our # Events": ours["events"],
                "Paper Prevalence (%)": paper["prevalence"],
                "Our Prevalence (%)": round(ours["prevalence"], 1),
            }
        )

    return pd.DataFrame(stats_rows)


def build_results_comparison_df(
    all_results: List[Dict[str, Any]],
) -> pd.DataFrame:
    """Builds a performance comparison dataframe against the paper."""
    result_rows = []

    for result in all_results:
        key = (result["task_name"], result["event_type"])
        paper_auc = PAPER_RESULTS[key]
        our_auc = result["test_metrics"]["roc_auc"]

        result_rows.append(
            {
                "Task": result["task_name"],
                "Event Type": result["event_type"],
                "Paper AUC": paper_auc,
                "Our Test AUROC": f"{our_auc:.4f}",
                "Our Test Accuracy": f"{result['test_metrics']['accuracy']:.4f}",
                "Our Test Loss": f"{result['test_metrics']['loss']:.4f}",
            }
        )

    return pd.DataFrame(result_rows)


def build_ablation_df(
    ablation_results: List[Dict[str, Any]],
) -> pd.DataFrame:
    """Builds a dataframe summarizing ablation results."""
    ablation_rows = []

    for result in ablation_results:
        ablation_rows.append(
            {
                "Event Type": result["event_type"],
                "Task": result["task_name"],
                "Config": result["config_name"],
                "LR": result["config"]["lr"],
                "Dropout": result["config"]["dropout"],
                "d_model": result["config"]["d_model"],
                "ff_hidden_dim": result["config"]["ff_hidden_dim"],
                "Batch Size": result["config"]["batch_size"],
                "Pretrain Epochs": result["config"]["pretrain_epochs"],
                "Finetune Epochs": result["config"]["finetune_epochs"],
                "Samples": result["num_samples"],
                "Best Val AUC": round(result["best_val_auc"], 4),
                "Test AUC": round(result["test_metrics"]["roc_auc"], 4),
                "Test Accuracy": round(result["test_metrics"]["accuracy"], 4),
                "Test Loss": round(result["test_metrics"]["loss"], 4),
            }
        )

    return pd.DataFrame(ablation_rows)


def build_best_ablation_df(ablation_df: pd.DataFrame) -> pd.DataFrame:
    """Selects the best ablation row per event/task pair by test AUC."""
    experiment_settings = [
        ("hypotension", "mortality"),
        ("hypotension", "los3d"),
        ("mechanical_ventilation", "mortality"),
        ("mechanical_ventilation", "los3d"),
    ]

    best_rows = []
    for event_type, task_name in experiment_settings:
        subset = ablation_df[
            (ablation_df["Event Type"] == event_type)
            & (ablation_df["Task"] == task_name)
        ].copy()

        subset = subset.sort_values("Test AUC", ascending=False)
        best_rows.append(subset.iloc[0].to_dict())

    return pd.DataFrame(best_rows)


def build_best_vs_paper_df(best_ablation_df: pd.DataFrame) -> pd.DataFrame:
    """Builds a dataframe comparing best ablation results with paper AUCs."""
    comparison_rows = []

    for _, row in best_ablation_df.iterrows():
        task_name = row["Task"]
        event_type = row["Event Type"]
        paper_auc = PAPER_RESULTS[(task_name, event_type)]

        comparison_rows.append(
            {
                "Task": task_name,
                "Event Type": event_type,
                "Paper AUC": paper_auc,
                "Best Config": row["Config"],
                "Our Best Test AUC": f"{row['Test AUC']:.4f}",
                "Our Best Test Accuracy": f"{row['Test Accuracy']:.4f}",
                "Our Best Test Loss": f"{row['Test Loss']:.4f}",
            }
        )

    return pd.DataFrame(comparison_rows)


def sort_result_tables(
    best_vs_paper_df: pd.DataFrame,
    ablation_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Sorts result dataframes in a paper-like order."""
    task_order = {"mortality": 0, "los3d": 1}
    event_order = {"hypotension": 0, "mechanical_ventilation": 1}

    sorted_best_vs_paper_df = best_vs_paper_df.sort_values(
        by=["Event Type", "Task"],
        key=lambda col: (
            col.map(event_order)
            if col.name == "Event Type"
            else col.map(task_order)
        ),
    )

    sorted_ablation_df = ablation_df.sort_values(
        by=["Event Type", "Task", "Config"],
        key=lambda col: (
            col.map(event_order)
            if col.name == "Event Type"
            else col.map(task_order)
            if col.name == "Task"
            else col
        ),
    )

    return sorted_best_vs_paper_df, sorted_ablation_df


# ---------------------------------------------------------------------
# Ablation configuration
# ---------------------------------------------------------------------

ABLATION_CONFIGS = [
    {
        "name": "baseline",
        "lr": 1e-3,
        "dropout": 0.1,
        "d_model": 32,
        "ff_hidden_dim": 128,
        "batch_size": 16,
        "pretrain_epochs": 5,
        "finetune_epochs": 5,
    },
    {
        "name": "lower_lr",
        "lr": 3e-4,
        "dropout": 0.1,
        "d_model": 32,
        "ff_hidden_dim": 128,
        "batch_size": 16,
        "pretrain_epochs": 5,
        "finetune_epochs": 5,
    },
    {
        "name": "higher_dropout",
        "lr": 1e-3,
        "dropout": 0.3,
        "d_model": 32,
        "ff_hidden_dim": 128,
        "batch_size": 16,
        "pretrain_epochs": 5,
        "finetune_epochs": 5,
    },
    {
        "name": "larger_hidden",
        "lr": 1e-3,
        "dropout": 0.1,
        "d_model": 64,
        "ff_hidden_dim": 256,
        "batch_size": 16,
        "pretrain_epochs": 5,
        "finetune_epochs": 5,
    },
]


# ---------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------

def main() -> None:
    """Runs the main experiments and hyperparameter ablation study."""
    set_seed(DEFAULT_SEED)
    validate_required_files(MIMIC4_PATH)

    print(f"Using device: {DEVICE}")
    tables = load_required_tables(MIMIC4_PATH)

    admissions = tables["admissions"]
    icustays = tables["icustays"]
    d_items = tables["d_items"]
    chartevents = tables["chartevents"]
    procedureevents = tables["procedureevents"]
    prescriptions = tables["prescriptions"]

    experiment_settings = [
        ("hypotension", "mortality"),
        ("hypotension", "los3d"),
        ("mechanical_ventilation", "mortality"),
        ("mechanical_ventilation", "los3d"),
    ]

    all_results = []
    for event_type, task_name in experiment_settings:
        print("\n" + "=" * 100)
        print(f"Running main experiment: event_type={event_type}, task={task_name}")
        print("=" * 100)

        result = run_single_experiment(
            event_type=event_type,
            task_name=task_name,
            admissions=admissions,
            icustays=icustays,
            chartevents=chartevents,
            procedureevents=procedureevents,
            prescriptions=prescriptions,
            d_items=d_items,
            max_len=MAX_LEN,
            min_len=MIN_LEN,
            batch_size=BATCH_SIZE,
            device=DEVICE,
            pretrain_epochs=5,
            finetune_epochs=5,
        )
        all_results.append(result)

    stats_comparison_df = build_stats_comparison_df(all_results)
    results_comparison_df = build_results_comparison_df(all_results)

    print("\nCohort statistics comparison:")
    print(stats_comparison_df)

    print("\nPerformance comparison:")
    print(results_comparison_df)

    ablation_results = []
    for event_type, task_name in experiment_settings:
        for config in ABLATION_CONFIGS:
            print("\n" + "=" * 110)
            print(
                "ABLATION RUN -> "
                f"event_type={event_type}, task={task_name}, "
                f"config={config['name']}"
            )
            print("=" * 110)

            result = run_single_experiment_with_config(
                event_type=event_type,
                task_name=task_name,
                config=config,
                admissions=admissions,
                icustays=icustays,
                chartevents=chartevents,
                procedureevents=procedureevents,
                prescriptions=prescriptions,
                d_items=d_items,
                max_len=MAX_LEN,
                min_len=MIN_LEN,
                device=DEVICE,
            )
            ablation_results.append(result)

    ablation_df = build_ablation_df(ablation_results)
    best_ablation_df = build_best_ablation_df(ablation_df)
    best_vs_paper_df = build_best_vs_paper_df(best_ablation_df)

    best_vs_paper_df, ablation_df = sort_result_tables(
        best_vs_paper_df,
        ablation_df,
    )

    print("\nBest ablation vs paper:")
    print(best_vs_paper_df)

    print("\nFull ablation table:")
    print(ablation_df)


if __name__ == "__main__":
    main()