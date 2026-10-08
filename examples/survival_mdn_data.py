# Contributor: Neil Hajela (nhajela2@illinois.edu)
"""SUPPORT and synthetic inputs for the Survival MDN example.

The raw route uses PyHealth's existing Support2Dataset. The benchmark route
uses the verified feature table and exact memberships supplied with the project.
All scaling is fitted on training patients only. No data are downloaded.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pyhealth.datasets import SampleDataset, Support2Dataset, create_sample_dataset

FEATURES = [
    "age",
    "meanbp",
    "hrt",
    "resp",
    "temp",
    "sod",
    "wblc",
    "crea",
    "sex_female",
    *[f"num.co_{i}" for i in range(1, 10)],
    "race_white",
    "race_black",
    "race_hispanic",
    "race_asian",
    "race_missing",
    "diabetes_1",
    "dementia_1",
    "ca_no",
    "ca_yes",
]


@dataclass
class Split:
    """Numeric data for one disjoint patient subset."""

    x: np.ndarray
    duration: np.ndarray
    event: np.ndarray
    patient_id: np.ndarray

    def dataset(self) -> SampleDataset:
        """Create an in-memory sample dataset using stateless tensor processors."""
        return create_sample_dataset(
            samples=[
                {
                    "patient_id": str(pid),
                    "features": x.tolist(),
                    "duration": [float(t)],
                    "event": [float(e)],
                }
                for pid, x, t, e in zip(
                    self.patient_id, self.x, self.duration, self.event
                )
            ],
            input_schema={"features": "tensor"},
            output_schema={"duration": "tensor", "event": "tensor"},
            in_memory=True,
        )


@dataclass
class Bundle:
    """Training, validation and test subsets."""

    train: Split
    valid: Split
    test: Split


def encode(frame: pd.DataFrame) -> pd.DataFrame:
    """Encode raw SUPPORT into the recovered 27-feature representation.

    Args:
        frame: Records with explicit patient_id, d.time, death and covariates.

    Returns:
        Unstandardized benchmark-compatible feature and label table.
    """
    frame = frame.copy()
    for key in FEATURES[:8] + ["num.co", "diabetes", "dementia", "d.time", "death"]:
        frame[key] = pd.to_numeric(frame[key], errors="raise")
    df = frame.dropna(subset=["wblc", "crea"]).copy()
    required = FEATURES[:8] + ["sex", "num.co", "diabetes", "dementia", "ca"]
    if df[required].isna().any().any():
        raise ValueError("Unexpected missing covariates after cohort selection")
    columns = [df[k].astype(float) for k in FEATURES[:8]]
    columns += [(df.sex == "female").astype(float)]
    columns += [(df["num.co"] == i).astype(float) for i in range(1, 10)]
    columns += [
        (df.race == r).astype(float) for r in ["white", "black", "hispanic", "asian"]
    ]
    columns += [
        df.race.isna().astype(float),
        (df.diabetes == 1).astype(float),
        (df.dementia == 1).astype(float),
        (df.ca == "no").astype(float),
        (df.ca == "yes").astype(float),
    ]
    out = pd.DataFrame(np.column_stack(columns).astype(np.float32), columns=FEATURES)
    out["patient_id"] = pd.to_numeric(df.patient_id).to_numpy(dtype=np.int64)
    out["duration_years_soden"] = df["d.time"].to_numpy(float) / 365.25
    out["event"] = df.death.to_numpy(float)
    return out.sort_values("patient_id").reset_index(drop=True)


def read_support2(csv_path: Path) -> pd.DataFrame:
    """Load records through the existing PyHealth Support2Dataset.

    Args:
        csv_path: Local raw SUPPORT2 CSV with sno or explicit patient_id.

    Returns:
        Encoded records extracted from the dataset's parsed event table.
    """
    raw = pd.read_csv(csv_path)
    if "sno" not in raw:
        if "patient_id" not in raw:
            raise ValueError("CSV must have an explicit sno or patient_id column")
        raw = raw.rename(columns={"patient_id": "sno"})
    with TemporaryDirectory(prefix="survival_mdn_support_") as tmp:
        root = Path(tmp)
        raw.to_csv(root / "support2.csv", index=False)
        dataset = Support2Dataset(
            root=str(root),
            tables=["support2"],
            cache_dir=root / "cache",
            num_workers=1,
        )
        # Collect the small parsed baseline table once instead of scanning the
        # same parquet file separately for every patient.
        frame = dataset.global_event_df.collect().to_pandas()
        frame = frame.rename(
            columns={c: c.removeprefix("support2/") for c in frame.columns}
        )
        if frame.patient_id.duplicated().any():
            raise ValueError("Expected one baseline SUPPORT event per patient")
        return encode(frame)


def split_frame(
    frame: pd.DataFrame, membership: Path | None, split: int, seed: int
) -> Bundle:
    """Split patients and fit standardization on training features only.

    Args:
        frame: Encoded unstandardized records.
        membership: Exact split matrix or None for random 70/15/15 subsets.
        split: Benchmark split from 1 to 10.
        seed: Random partition seed when exact memberships are absent.

    Returns:
        Three disjoint standardized subsets.
    """
    if frame.patient_id.duplicated().any():
        raise ValueError("Duplicate patient_id")
    if membership is not None:
        key = f"split_{split}"
        matrix = pd.read_csv(membership)[["patient_id", key]]
        if set(frame.patient_id) != set(matrix.patient_id):
            raise ValueError("Feature and membership IDs must match exactly")
        frame = frame.merge(matrix, on="patient_id", validate="one_to_one")
        labels = frame[key].to_numpy()
    else:
        order = np.random.RandomState(seed).permutation(len(frame))
        labels = np.full(len(frame), "test", dtype="U5")
        labels[order[: int(0.7 * len(frame))]] = "train"
        labels[order[int(0.7 * len(frame)) : int(0.85 * len(frame))]] = "valid"
    if not set(labels) <= {"train", "valid", "test"}:
        raise ValueError("Invalid membership label")
    x = frame[FEATURES].to_numpy(np.float32)
    t = frame.duration_years_soden.to_numpy(float) + 0.001
    e = frame.event.to_numpy(np.float32)
    if not np.isfinite(x).all() or not np.isfinite(t).all():
        raise ValueError("Features and durations must be finite")
    if (t <= 0).any() or not np.isin(e, [0, 1]).all():
        raise ValueError("Invalid duration or event")
    if any(np.sum(labels == k) < 2 for k in ["train", "valid", "test"]):
        raise ValueError("Every subset requires at least two patients")
    scaler = StandardScaler().fit(x[labels == "train"])
    x = scaler.transform(x).astype(np.float32)
    parts = []
    for name in ["train", "valid", "test"]:
        mask = labels == name
        parts.append(
            Split(x[mask], t[mask], e[mask], frame.patient_id.to_numpy()[mask])
        )
    return Bundle(*parts)


def synthetic(n: int, seed: int) -> Bundle:
    """Create a reproducible offline survival example with independent censoring."""
    if n < 24:
        raise ValueError("Use at least 24 synthetic samples")
    rng = np.random.RandomState(seed)
    x = rng.normal(size=(n, 27)).astype(np.float32)
    event_time = rng.lognormal(0.5 - 0.4 * x[:, 0] + 0.3 * x[:, 1], 0.55)
    censor_time = rng.lognormal(0.9, 0.65, n)
    frame = pd.DataFrame(x, columns=FEATURES)
    frame["patient_id"] = np.arange(n)
    frame["duration_years_soden"] = np.minimum(event_time, censor_time)
    frame["event"] = (event_time <= censor_time).astype(float)
    return split_frame(frame, None, 1, seed)
