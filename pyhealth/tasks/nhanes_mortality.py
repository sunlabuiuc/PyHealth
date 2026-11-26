"""Mortality prediction tasks for the NHANES dataset.

Contributors: Inoh Pak (inohpak2), Yejin Kim (yejink6)

Paper: Song et al., MotionAge: A Deep Learning Framework for Biological Age
    Prediction from Wearable Activity, MLHC 2026 (PMLR 340).
    https://proceedings.mlr.press/v340/song26a.html

Description: Five-year mortality prediction from 3.5-day activity windows,
    with an optional MotionAge input format (intensity, hour, day, wear mask).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from .base_task import BaseTask


def get_mortality_label(
    eligstat: float | None,
    mortstat: float | None,
    followup_months: float | None,
    prediction_months: int = 60,
) -> int | None:
    """Determines mortality within a specified prediction horizon.

    Args:
        eligstat: Mortality linkage eligibility status.
        mortstat: Mortality status.
        followup_months: Follow-up duration in months.
        prediction_months: Prediction horizon in months.

    Returns:
        1 for death within the prediction horizon,
        0 for survival beyond the prediction horizon,
        or None when the outcome cannot be determined.
    """
    if prediction_months <= 0:
        raise ValueError("prediction_months must be positive")

    if pd.isna(eligstat) or eligstat != 1:
        return None

    if pd.isna(mortstat) or mortstat not in (0, 1):
        return None

    if pd.isna(followup_months):
        return None

    if not np.isfinite(followup_months) or followup_months < 0:
        return None

    if mortstat == 1 and followup_months <= prediction_months:
        return 1

    if followup_months > prediction_months:
        return 0

    return None


def build_window_inputs(
    epochs: pd.DataFrame,
    window_size: int = 1008,
    min_wear_minutes: int = 1,
    min_valid_fraction: float = 0.3,
    include_wear_fraction: bool = True,
    wear_representation: str = "fraction",
) -> tuple[np.ndarray, list[dict[str, int | float]]]:
    """Creates valid activity windows for mortality prediction.

    Args:
        epochs: Five-minute activity epochs for one participant.
        window_size: Number of epochs per nonoverlapping window.
        min_wear_minutes: Minimum wear minutes per epoch.
        min_valid_fraction: Minimum valid epoch fraction per window.
        include_wear_fraction: Whether to include wear as a feature.
        wear_representation: Representation of wear coverage.
            "fraction" preserves continuous wear fractions between
            zero and one. "binary" indicates whether each epoch
            meets the minimum wear threshold. This option does not
            affect the output when include_wear_fraction is False.

    Returns:
        Activity windows and their corresponding metadata.

    Raises:
        ValueError: If configuration parameters or epoch indices
            are invalid.
    """
    if window_size <= 0:
        raise ValueError("window_size must be positive")

    if not 1 <= min_wear_minutes <= 5:
        raise ValueError("min_wear_minutes must be between 1 and 5")

    if not 0 <= min_valid_fraction <= 1:
        raise ValueError("min_valid_fraction must be between 0 and 1")

    if wear_representation not in ("fraction", "binary"):
        raise ValueError("wear_representation must be 'fraction' or 'binary'")

    epochs = epochs.sort_values("epoch")

    if not np.array_equal(
        epochs["epoch"].to_numpy(),
        np.arange(1, len(epochs) + 1),
    ):
        raise ValueError("Epoch indices must be consecutive")

    windows = []
    window_info = []

    for start in range(0, len(epochs), window_size):
        window = epochs.iloc[start : start + window_size]

        if len(window) != window_size:
            continue

        valid = window["wear_minutes"].to_numpy() >= min_wear_minutes

        valid_fraction = float(valid.mean())

        if valid_fraction < min_valid_fraction:
            continue

        intensity = window["intensity"].to_numpy(dtype=np.float32)

        intensity = np.where(valid, intensity, 0)

        if include_wear_fraction:
            if wear_representation == "binary":
                wear_feature = valid.astype(np.float32)
            else:
                wear_feature = window["wear_fraction"].to_numpy(dtype=np.float32)

            features = np.column_stack([intensity, wear_feature])
        else:
            features = intensity.reshape(-1, 1)

        windows.append(features)

        window_info.append(
            {
                "window_id": start // window_size + 1,
                "valid_fraction": valid_fraction,
            }
        )

    num_features = 2 if include_wear_fraction else 1

    if windows:
        activity = np.stack(windows).astype(np.float32)
    else:
        activity = np.empty(
            (0, window_size, num_features),
            dtype=np.float32,
        )

    return activity, window_info


class NHANESMortalityPrediction(BaseTask):
    """Predicts mortality from NHANES accelerometer activity.

    Works with NHANESDataset for the 2003-2004 and 2005-2006
    survey cycles.

    Each participant's seven-day activity recording is divided
    into nonoverlapping windows of five-minute epochs.
    Windows with insufficient wear coverage are excluded.

    The binary outcome indicates whether death occurred within
    the specified prediction horizon after the interview.

    Participants with insufficient mortality follow-up or
    unknown mortality status are excluded.

    Activity inputs can include intensity alone, intensity with
    binary wear coverage, or intensity with continuous wear
    fractions. These representations support controlled feature
    ablation studies inspired by MotionAge.

    Args:
        root: Root directory containing the NHANES data.
        prediction_months: Mortality prediction horizon in months.
        window_size: Number of five-minute epochs per window.
        min_wear_minutes: Minimum wear minutes per valid epoch.
        min_valid_fraction: Minimum valid epoch fraction per window.
        include_wear_fraction: Whether to include wear as a feature.
        wear_representation: Wear coverage representation.
            "fraction" uses continuous wear fractions.
            "binary" uses a threshold-based coverage indicator.
        feature_format: "activity" returns a combined activity tensor.
            "motionage" returns separate intensity, hour, relative day,
            and wear mask tensors. Fractional wear can be used as
            continuous pooling weights.
        include_static: Includes age, gender, BMI, systolic and diastolic
            blood pressure when using the MotionAge format.
        pos_weight: Positive-class weight for binary cross-entropy.
            Use a fixed training-derived weight, "balanced" with a
            training-only processor fit, or None for unweighted loss.

    Examples:
        >>> from pyhealth.datasets import NHANESDataset
        >>> from pyhealth.tasks import NHANESMortalityPrediction
        >>> dataset = NHANESDataset(root="data/nhanes")
        >>> task = NHANESMortalityPrediction(
        ...     root="data/nhanes",
        ...     wear_representation="binary",
        ... )
        >>> samples = dataset.set_task(task)
    """

    task_name: str = "NHANESMortalityPrediction"

    input_schema: ClassVar[dict[str, str]] = {
        "activity": "tensor",
    }

    output_schema: ClassVar[dict[str, str]] = {
        "label": "binary",
    }

    def __init__(
        self,
        root: str | Path,
        prediction_months: int = 60,
        window_size: int = 1008,
        min_wear_minutes: int = 1,
        min_valid_fraction: float = 0.3,
        include_wear_fraction: bool = True,
        wear_representation: str = "fraction",
        feature_format: str = "activity",
        include_static: bool = False,
        pos_weight: float | str | None = None,
    ) -> None:
        """Initializes the NHANES mortality prediction task."""
        if prediction_months <= 0:
            raise ValueError("prediction_months must be positive")

        if not 1 <= window_size <= 2016:
            raise ValueError("window_size must be between 1 and 2016")

        if not 1 <= min_wear_minutes <= 5:
            raise ValueError("min_wear_minutes must be between 1 and 5")

        if not 0 <= min_valid_fraction <= 1:
            raise ValueError("min_valid_fraction must be between 0 and 1")

        if wear_representation not in ("fraction", "binary"):
            raise ValueError("wear_representation must be 'fraction' or 'binary'")

        self.root = str(Path(root).resolve())
        self.prediction_months = prediction_months
        self.window_size = window_size
        self.min_wear_minutes = min_wear_minutes
        self.min_valid_fraction = min_valid_fraction
        self.include_wear_fraction = include_wear_fraction
        self.wear_representation = wear_representation

        if feature_format not in ("activity", "motionage"):
            raise ValueError("feature_format must be 'activity' or 'motionage'")

        if feature_format == "motionage" and not include_wear_fraction:
            raise ValueError("MotionAge requires wear information")

        if include_static and feature_format != "motionage":
            raise ValueError("Static features require motionage format")

        self.feature_format = feature_format
        self.include_static = include_static
        self.pos_weight = pos_weight

        self.input_schema = {"activity": "tensor"}
        self.output_schema = {"label": "binary"}

        if feature_format == "motionage":
            self.input_schema = {
                "intensity": "tensor",
                "hour": "tensor",
                "day": "tensor",
                "wear_mask": "tensor",
            }

        if include_static:
            self.input_schema["static"] = (
                "tensor",
                {
                    "normalization": "zscore",
                    "missing_indicators": True,
                    "impute_strategy": "zero",
                },
            )

        if pos_weight is not None:
            self.output_schema["label"] = (
                "binary",
                {"pos_weight": pos_weight},
            )

        super().__init__()

    def __call__(
        self,
        patient: Any,
    ) -> list[dict[str, Any]]:
        """Generates mortality prediction samples for one participant.

        Args:
            patient: A PyHealth patient containing NHANES metadata.

        Returns:
            A list of valid activity windows and binary mortality
            labels. Returns an empty list for participants without
            eligible mortality outcomes or valid activity windows.

        Raises:
            FileNotFoundError: If the activity file is missing.
            ValueError: If participant metadata or activity is invalid.
        """
        events = patient.get_events(event_type="participants")

        if len(events) == 0:
            return []

        if len(events) != 1:
            raise ValueError(f"Expected one event for {patient.patient_id}")

        event = events[0]

        label = get_mortality_label(
            eligstat=event.eligstat,
            mortstat=event.mortstat,
            followup_months=event.permth_int,
            prediction_months=self.prediction_months,
        )

        if label is None:
            return []

        activity_path = Path(self.root) / event.activity_file

        if not activity_path.is_file():
            raise FileNotFoundError(activity_path)

        columns = [
            "epoch",
            "intensity",
            "wear_minutes",
            "wear_fraction",
        ]

        if self.feature_format == "motionage":
            columns.extend(["hour", "day"])

        epochs = pd.read_parquet(
            activity_path,
            columns=columns,
        ).sort_values("epoch")

        if len(epochs) != 2016:
            raise ValueError(f"Expected 2016 epochs for {patient.patient_id}")

        activity_windows, window_info = build_window_inputs(
            epochs=epochs,
            window_size=self.window_size,
            min_wear_minutes=self.min_wear_minutes,
            min_valid_fraction=self.min_valid_fraction,
            include_wear_fraction=self.include_wear_fraction,
            wear_representation=self.wear_representation,
        )

        samples = []

        for activity, info in zip(activity_windows, window_info):
            sample = {
                "patient_id": patient.patient_id,
                "window_id": info["window_id"],
                "valid_fraction": info["valid_fraction"],
                "label": label,
            }

            if self.feature_format == "activity":
                sample["activity"] = activity

            else:
                start = (info["window_id"] - 1) * self.window_size
                window = epochs.iloc[start : start + self.window_size]

                valid = window["wear_minutes"].to_numpy() >= self.min_wear_minutes

                sample.update(
                    {
                        "intensity": activity[:, 0].copy(),
                        "hour": window["hour"].to_numpy(dtype=np.float32),
                        "day": window["day"].to_numpy(dtype=np.float32) - 1,
                        "wear_mask": np.where(valid, activity[:, 1], 0).astype(
                            np.float32
                        ),
                    }
                )

                if self.include_static:
                    fields = (
                        "age",
                        "gender",
                        "bmi",
                        "systolic_bp",
                        "diastolic_bp",
                    )

                    values = [getattr(event, field) for field in fields]

                    sample["static"] = np.array(
                        [
                            float(value) if pd.notna(value) else np.nan
                            for value in values
                        ],
                        dtype=np.float32,
                    )

            samples.append(sample)

        return samples
