"""5-year all-cause mortality prediction from NHANES wearable activity.

Paper: Song, Yin, et al., "MotionAge: A Deep Learning Framework for Biological
Age Prediction from Wearable Activity." https://openreview.net/pdf?id=xGw1o9acWa

Cuts each :class:`~pyhealth.datasets.NHANESDataset` participant's wear-aware
5-minute epochs into fixed-length windows, each labeled with whether the
participant died within the follow-up horizon. The defaults follow the paper:
3.5-day windows, no overlap, at least 30% of epochs worn, and a 60-month
horizon measured from the NHANES interview.
"""

import math
from typing import Any

import numpy as np
import pandas as pd
import torch

from .base_task import BaseTask

# Covariates per ``covariate_set``, in the order they appear in the tensors.
# "clinical" is the paper's routine clinical set (MotionAge-FRC), "age_sex"
# its age-adjusted variant (MotionAge-FA); ``None`` is wearable only (-F).
NUMERIC_COVARIATES = {
    "clinical": ["age", "bmi", "waist", "dbp", "sbp"],
    "age_sex": ["age"],
}
CATEGORICAL_COVARIATES = {
    "clinical": ["sex", "income"],
    "age_sex": ["sex"],
}
# Number of codes per categorical covariate, including code 0 for missing:
# sex 1 = male, 2 = female; income = NHANES INDHHINC codes 1-13.
CATEGORY_SIZES = {"sex": 3, "income": 14}


class NHANESMortalityPrediction(BaseTask):
    """Predicts death within ``followup_months`` from a window of activity.

    Each participant contributes one sample per window of ``seq_len`` epochs
    (starting every ``stride`` epochs) in which at least ``min_coverage`` of
    the epochs were worn. All windows of a participant share their label, so
    split by patient (e.g. ``PatientSplit``) and average a participant's
    window probabilities to get one prediction per participant.

    Label: 1 if ``mortstat == 1`` and ``permth_int <= followup_months``,
    else 0. Survivors followed for less than ``followup_months`` have an
    unknown label and are skipped, as are participants without follow-up.

    Sample fields:

    * ``intensity``: float ``[seq_len, 1]``, mean counts per worn minute,
      z-scored with statistics fitted on the worn epochs of the training
      split. Epochs that were not worn become 0 (the training mean), so
      models must use ``wear_mask`` to ignore them.
    * ``hour``: long ``[seq_len]``, hour of day 0-23.
    * ``day``: long ``[seq_len]``, day of week 0-6 (0 = Sunday).
    * ``wear_mask``: float ``[seq_len]``, 1 where the epoch was worn.
    * ``covariates`` (unless ``covariate_set`` is ``None``): float, the
      numeric covariates in ``numeric_covariates`` order, z-scored on the
      training split, missing values set to 0, followed by one missing-value
      indicator per covariate.
    * ``covariate_codes`` (unless ``covariate_set`` is ``None``): long, one
      code per entry of ``categorical_covariates``, 0 when missing. Sizes for
      embedding tables are in ``category_sizes``.
    * ``mortality``: binary label.
    * ``age``, ``sex`` and ``window_start``: raw values passed through
      unprocessed, for mapping risk to MotionAge and for bookkeeping.

    Args:
        seq_len: Window length in 5-minute epochs. 1008 is 3.5 days; 288 is
            one day.
        stride: Epochs between window starts. Defaults to ``seq_len``
            (no overlap).
        min_coverage: Minimum fraction of worn epochs for a window to be kept.
        followup_months: Mortality horizon in months.
        covariate_set: ``"clinical"``, ``"age_sex"`` or ``None`` (wearable
            only, no covariate fields).
        wear_aware: If False, non-wear epochs are treated as real zero
            activity: ``intensity`` keeps their 0 counts and ``wear_mask`` is
            all ones. The same windows are kept either way, so the two
            settings can be compared directly.

    Raises:
        ValueError: If an argument is out of range.

    Examples:
        >>> import tempfile
        >>> from pyhealth.datasets.nhanes import NHANESDataset
        >>> from pyhealth.datasets.nhanes_synthetic import generate_synthetic_nhanes
        >>> from pyhealth.tasks.nhanes_mortality_prediction import (
        ...     NHANESMortalityPrediction,
        ... )
        >>> root = tempfile.mkdtemp()
        >>> _ = generate_synthetic_nhanes(  # doctest: +SKIP
        ...     root, n_participants=10, n_days=7
        ... )
        >>> dataset = NHANESDataset(root=root)  # doctest: +SKIP
        >>> samples = dataset.set_task(NHANESMortalityPrediction())  # doctest: +SKIP
        >>> samples[0]["intensity"].shape  # doctest: +SKIP
        torch.Size([1008, 1])
    """

    task_name: str = "NHANESMortalityPrediction"

    def __init__(
        self,
        seq_len: int = 1008,
        stride: int | None = None,
        min_coverage: float = 0.3,
        followup_months: int = 60,
        covariate_set: str | None = "clinical",
        wear_aware: bool = True,
    ) -> None:
        if seq_len < 1:
            raise ValueError(f"seq_len must be at least 1, got {seq_len}")
        if stride is not None and stride < 1:
            raise ValueError(f"stride must be at least 1, got {stride}")
        if not 0.0 <= min_coverage <= 1.0:
            raise ValueError(f"min_coverage must be in [0, 1], got {min_coverage}")
        if followup_months < 1:
            raise ValueError(
                f"followup_months must be at least 1, got {followup_months}"
            )
        if covariate_set is not None and covariate_set not in NUMERIC_COVARIATES:
            raise ValueError(
                "covariate_set must be None or one of "
                f"{sorted(NUMERIC_COVARIATES)}, got {covariate_set!r}"
            )
        super().__init__()
        self.seq_len = seq_len
        self.stride = seq_len if stride is None else stride
        self.min_coverage = min_coverage
        self.followup_months = followup_months
        self.covariate_set = covariate_set
        self.wear_aware = wear_aware
        self.numeric_covariates = list(NUMERIC_COVARIATES.get(covariate_set, []))
        self.categorical_covariates = list(
            CATEGORICAL_COVARIATES.get(covariate_set, [])
        )
        self.category_sizes = [CATEGORY_SIZES[c] for c in self.categorical_covariates]

        schema: dict[str, Any] = {
            # One feature channel, so z-scoring learns a single mean and std;
            # non-wear epochs are NaN here, so the fit uses worn epochs only.
            "intensity": (
                "tensor",
                {"normalization": "zscore", "impute_strategy": "zero"},
            ),
            "hour": ("tensor", {"dtype": torch.long}),
            "day": ("tensor", {"dtype": torch.long}),
            "wear_mask": "tensor",
        }
        if covariate_set is not None:
            schema["covariates"] = (
                "tensor",
                {
                    "normalization": "zscore",
                    "missing_indicators": True,
                    "impute_strategy": "zero",
                },
            )
            schema["covariate_codes"] = ("tensor", {"dtype": torch.long})
        self.input_schema = schema
        self.output_schema = {"mortality": "binary"}

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Builds the labeled windows for one participant.

        Args:
            patient: A patient from :class:`~pyhealth.datasets.NHANESDataset`.

        Returns:
            One sample per kept window; empty if the label is unknown or no
            window has enough wear.
        """
        events = patient.get_events(event_type="nhanes")
        if not events:
            return []
        event = events[0]
        label = self._label(event)
        if label is None:
            return []
        epochs = pd.read_parquet(event.epoch_file)
        fixed = {
            "patient_id": patient.patient_id,
            "mortality": label,
            "age": _number(event.age),
            "sex": _code(event.sex, CATEGORY_SIZES["sex"]),
        }
        if self.covariate_set is not None:
            fixed["covariates"] = np.array(
                [_number(getattr(event, name)) for name in self.numeric_covariates],
                dtype=np.float32,
            )
            fixed["covariate_codes"] = np.array(
                [
                    _code(getattr(event, name), CATEGORY_SIZES[name])
                    for name in self.categorical_covariates
                ],
                dtype=np.int64,
            )
        return [{**fixed, **window} for window in self._windows(epochs)]

    def _label(self, event: Any) -> int | None:
        """Returns the horizon label, or None when it cannot be known."""
        mortstat = _number(event.mortstat)
        months = _number(event.permth_int)
        if math.isnan(months) or mortstat not in (0.0, 1.0):
            return None
        if mortstat == 1.0:
            return int(months <= self.followup_months)
        if months < self.followup_months:
            return None  # Alive when follow-up ended, before the horizon.
        return 0

    def _windows(self, epochs: pd.DataFrame) -> list[dict[str, Any]]:
        """Cuts one participant's epochs into windows with enough wear."""
        wear = epochs["wear_mask"].to_numpy(dtype=np.float32)
        intensity = epochs["intensity"].to_numpy(dtype=np.float32)
        if self.wear_aware:
            intensity = np.where(wear > 0, intensity, np.nan).astype(np.float32)
            mask = wear
        else:
            mask = np.ones_like(wear)
        hour = epochs["hour"].to_numpy(dtype=np.int64)
        day = epochs["day"].to_numpy(dtype=np.int64)

        windows = []
        for start in range(0, len(epochs) - self.seq_len + 1, self.stride):
            end = start + self.seq_len
            if wear[start:end].mean() < self.min_coverage:
                continue
            windows.append(
                {
                    "window_start": start,
                    "intensity": intensity[start:end, None],
                    "hour": hour[start:end],
                    "day": day[start:end],
                    "wear_mask": mask[start:end],
                }
            )
        return windows


def _number(value: Any) -> float:
    """Converts an attribute to float, with NaN for missing values."""
    if value is None:
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _code(value: Any, size: int) -> int:
    """Converts a categorical code to an int in ``1..size-1``, or 0 if missing."""
    number = _number(value)
    if math.isnan(number) or number != int(number) or not 1 <= number < size:
        return 0
    return int(number)
