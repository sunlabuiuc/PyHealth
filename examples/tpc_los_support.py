"""Load and pad prepared data for the TPC example.

Contributor: Audrey Labrie.
Paper: Temporal Pointwise Convolutional Networks for Length of Stay Prediction
in the Intensive Care Unit (Rocheteau et al., 2021).
Paper link: https://arxiv.org/abs/2007.09483
Usage: examples/eicu_remaining_los_tpc.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from pyhealth.datasets import create_sample_dataset
from torch.utils.data import Dataset


class PreparedICUStays:
    """Load the train or val files.

    Args:
        root: Folder with prepared data and saved encoders.
        split: Either train or val.
    """

    def __init__(self, root: Path, split: str) -> None:
        if split not in {"train", "val"}:
            raise ValueError("This example only reads training and validation")
        root = Path(root)
        folder = root / split

        def names(filename: str) -> list[str]:
            return json.loads((root / filename).read_text())["features"]

        measurements = names("timeseries_scaler.json")
        static_names = names("flat_encoder.json")
        diagnosis_names = names("diagnosis_encoder.json")
        columns = (["time"] + measurements
                   + [name + "_mask" for name in measurements] + ["hour"])
        self.features = len(measurements)
        self.static_dim = len(static_names)
        self.diagnosis_dim = len(diagnosis_names)
        labels = pd.read_csv(
            folder / "labels.csv", dtype={"patient": "int64", "uniquepid": str}
        )
        if (labels.patient.duplicated().any() or labels.uniquepid.isna().any()
                or labels.uniquepid.str.strip().eq("").any()):
            raise ValueError(f"Invalid label identifiers in {split}")
        labels = labels.set_index("patient")
        self.people = frozenset(labels.uniquepid.str.strip())
        los = pd.to_numeric(labels.actualiculos, errors="raise")
        if not np.isfinite(los.to_numpy()).all() or (los <= 0).any():
            raise ValueError("Length-of-stay targets must be finite and positive")

        def aligned(filename: str, expected: list[str]) -> pd.DataFrame:
            table = pd.read_csv(folder / filename, dtype={"patient": "int64"})
            if (list(table.columns) != ["patient"] + expected
                    or table.patient.duplicated().any()
                    or set(table.patient) != set(labels.index)):
                raise ValueError(f"Schema or stay alignment differs: {filename}")
            return table.set_index("patient").loc[labels.index, expected].astype(
                "float32"
            )

        static = aligned("flat.csv", static_names)
        diagnoses = aligned("diagnoses.csv", diagnosis_names)
        series = pd.read_csv(
            folder / "timeseries.csv",
            dtype={"patient": "int64", **{c: "float32" for c in columns}},
        )
        if list(series.columns) != ["patient"] + columns or series.empty:
            raise ValueError("Unexpected or empty time-series schema")
        ids = series.patient.to_numpy()
        elapsed = series.time.to_numpy(dtype="float64", copy=True)
        x = series[columns].to_numpy(dtype="float32", copy=True)
        if not all(np.isfinite(v).all()
                   for v in (x, static.to_numpy(), diagnoses.to_numpy())):
            raise ValueError("Model inputs must be finite")
        if ((elapsed < 1) | (elapsed > 335) | (elapsed % 1 != 0)).any():
            raise ValueError("Elapsed times must be integer hours 1 through 335")
        masks = x[:, 1 + self.features:1 + 2 * self.features]
        if ((masks < 0) | (masks > 1)).any():
            raise ValueError("Invalid observation-recency indicators")
        boundaries = np.r_[0, 1 + np.flatnonzero(ids[1:] != ids[:-1]), len(ids)]
        run_ids = ids[boundaries[:-1]]
        if (len(run_ids) != len(set(run_ids))
                or set(run_ids) != set(labels.index)):
            raise ValueError("Time-series rows must be contiguous by stay")
        x[:, 0] /= 24.0
        self.samples: list[dict[str, Any]] = []
        for stay, start, end in zip(run_ids, boundaries[:-1], boundaries[1:]):
            hours = elapsed[start:end]
            valid = hours > 5.0
            if (np.diff(hours) != 1).any() or not valid.any():
                raise ValueError("Each stay needs an hourly grid and valid hours")
            self.samples.append({
                "patient_id": str(labels.loc[stay, "uniquepid"]).strip(),
                "record_id": str(int(stay)),
                "x": x[start:end].T,
                "static": static.loc[stay].to_numpy(dtype="float32"),
                "diagnoses": diagnoses.loc[stay].to_numpy(dtype="float32"),
                "targets": np.maximum(
                    float(los.loc[stay]) - hours / 24, 1 / 48
                ).astype("float32"),
                "valid_mask": valid,
            })

    def __len__(self) -> int:
        """Return the number of ICU stays."""
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return one stay with copied tensors."""
        return {
            key: torch.from_numpy(np.array(value, copy=True, order="C"))
            if isinstance(value, np.ndarray) else value
            for key, value in self.samples[index].items()
        }


def as_pyhealth(samples: list[dict[str, Any]], training: Any = None) -> Any:
    """Put each ICU stay in a PyHealth dataset.

    Args:
        samples: One dictionary per stay.
        training: Training dataset to reuse its processors, if provided.

    Returns:
        A PyHealth dataset containing the same samples.
    """
    if not samples:
        raise ValueError("PyHealth samples must be nonempty")
    processors = {} if training is None else {
        "input_processors": training.input_processors,
        "output_processors": training.output_processors,
    }
    dataset = create_sample_dataset(
        samples=samples,
        input_schema={k: "raw" for k in ("x", "static", "diagnoses", "valid_mask")},
        output_schema={"targets": "raw"},
        dataset_name="TPC-LoS-example",
        task_name="hourly_remaining_icu_los",
        in_memory=True,
        **processors,
    )
    if len(dataset) != len(samples):
        raise RuntimeError("PyHealth changed the number of stays")
    for index in (0, len(samples) - 1):
        for key, value in samples[index].items():
            converted = dataset[index][key]
            if isinstance(value, torch.Tensor):
                if (value.dtype != converted.dtype
                        or not torch.equal(value, converted)):
                    raise RuntimeError(f"PyHealth changed tensor field {key}")
            elif value != converted:
                raise RuntimeError(f"PyHealth changed identifier field {key}")
    return dataset


class IndexedDataset(Dataset):
    """Let a DataLoader read PyHealth samples."""

    def __init__(self, dataset: Any) -> None:
        """Store the dataset.

        Args:
            dataset: PyHealth dataset to read from.
        """
        self.dataset = dataset

    def __len__(self) -> int:
        """Return the number of stays."""
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return one stay.

        Args:
            index: Sample position.

        Returns:
            The stay's sample dictionary.
        """
        return self.dataset[index]


def collate_stays(samples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """Add padding so stays in a batch have the same length.

    Args:
        samples: Stays before padding.

    Returns:
        Input tensors, targets, and a mask that excludes padding.
    """
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    lengths = [s["x"].shape[1] for s in samples]
    batch, longest = len(samples), max(lengths)
    x = torch.zeros(
        batch, samples[0]["x"].shape[0], longest, dtype=torch.float32
    )
    targets = torch.ones(batch, longest, dtype=torch.float32)
    valid = torch.zeros(batch, longest, dtype=torch.bool)
    for index, (sample, length) in enumerate(zip(samples, lengths)):
        if (sample["targets"].shape != (length,)
                or sample["valid_mask"].shape != (length,)):
            raise ValueError("Sequence, target, and mask lengths differ")
        x[index, :, :length] = sample["x"]
        targets[index, :length] = sample["targets"]
        valid[index, :length] = sample["valid_mask"]
    return {
        "x": x, "targets": targets, "valid_mask": valid,
        "static": torch.stack([s["static"] for s in samples]),
        "diagnoses": torch.stack([s["diagnoses"] for s in samples]),
    }
