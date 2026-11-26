"""
Contributors: Manling Tan, Yining Jin, Xiao Luo
NetIDs: manling4, yining34, xiaoluo6

Paper Title:
Temporal Pointwise Convolutional Networks for
Length of Stay Prediction in the Intensive Care Unit

Paper Authors:
Emma Rocheteau, Pietro Liò, and Stephanie Hyland (2021)

Paper URL:
https://arxiv.org/abs/2007.09483

Hourly remaining ICU length-of-stay task with missingness experiments.
"""

import hashlib
from copy import deepcopy
from typing import Any, ClassVar

import numpy as np
import polars as pl

from .yaib_eicu import _DYNAMIC_FEATURES, YAIBLengthOfStayEICU


class TPCLengthOfStayEICU(YAIBLengthOfStayEICU):
    """Generate hourly remaining-LOS labels for eICU stays.

    This is a standalone task, not a TPC model implementation.

    Static features are inherited from YAIB. Labels are in hours,
    starting at hour 5 and continuing through min(floor(LOS), 336).

    Dynamic timestamps represent completed hourly bins:
    bin [h-1, h) is available at hour h.

    Dynamic measurements remain raw with NaNs at task creation.
    Inherited sequence_v2 processors handle normalization and
    forward-fill downstream.

    Missingness is applied to observed values before processing.
    Decay is computed from the post-masking, pre-forward-fill
    observation matrix.

    Args:
        missing_rate: Fraction of observed feature cells to mask,
            between 0 and 1.
        seed: Seed combined with stay ID for reproducible masking.
        include_decay: Whether to add per-feature decay sequences.

    Examples:
        >>> from pyhealth.datasets import eICUDataset
        >>> from pyhealth.tasks import TPCLengthOfStayEICU
        >>> task = TPCLengthOfStayEICU(missing_rate=0.1, include_decay=True)
        >>> task.task_name
        'TPCLengthOfStayEICU'
        >>> dataset = eICUDataset(  # doctest: +SKIP
        ...     root="/path/to/eicu-crd/2.0",
        ...     tables=[
        ...         "lab",
        ...         "vitalperiodic",
        ...         "vitalaperiodic",
        ...         "intakeoutput",
        ...         "respiratorycharting",
        ...         "nursecharting",
        ...     ],
        ... )
        >>> samples = dataset.set_task(task)  # doctest: +SKIP
    """

    task_name: str = "TPCLengthOfStayEICU"

    MAX_HOURS: ClassVar[int] = 14 * 24
    START_HOUR: ClassVar[int] = 5

    def __init__(
        self,
        missing_rate: float = 0.0,
        seed: int = 42,
        include_decay: bool = False,
    ) -> None:
        """Configure masking and optional decay features."""

        if not 0.0 <= missing_rate <= 1.0:
            raise ValueError("missing_rate must be between 0 and 1.")

        self.missing_rate = missing_rate
        self.seed = seed
        self.include_decay = include_decay

        self.input_schema = deepcopy(type(self).input_schema)

        if include_decay:
            self.input_schema["decay"] = (
                "sequence_v2",
                {
                    "levels": [
                        {
                            "ordering": "series",
                            "values": [len(self.DYNAMIC_FEATURES)],
                        }
                    ]
                },
            )

    def _base_cohort(
        self,
        patient: Any,
        stays: pl.DataFrame,
    ) -> tuple[pl.DataFrame, pl.DataFrame, dict[int, list[float]]]:
        """Select eligible adult ICU stays.

        Args:
            patient: PyHealth patient with eICU events.
            stays: Parsed ICU stay records.

        Returns:
            Eligible stays, hourly measurements, and static features.
        """

        static = self._static(patient, stays)

        keep = [
            stay_id
            for stay_id, hours in stays.select("stay_id", "los_hours").iter_rows()
            if np.isfinite(hours)
            and hours >= self.START_HOUR
            and np.isfinite(static[stay_id][0])
            and static[stay_id][0] >= 18
        ]

        stays = stays.filter(pl.col("stay_id").is_in(keep))

        # Extract the 48 hourly YAIB dynamic features for eligible adult stays.
        hourly = self._hourly(
            self._measurements(
                patient,
                stays,
                _DYNAMIC_FEATURES,
            )
        )

        return stays, hourly, {key: static[key] for key in keep}

    @staticmethod
    def _decay(values: np.ndarray) -> np.ndarray:
        """Compute exponentially decaying observation indicators.

        An observed value resets its decay indicator to 1.
        Each subsequent missing hour multiplies the previous
        indicator by 0.75.

        Features with no previous observations remain at 0.

        Decay is computed before forward-fill so that artificially
        imputed values do not count as new observations.

        Args:
            values: Two-dimensional hourly feature matrix
                containing NaNs for missing observations.

        Returns:
            Float array of the same shape with decay indicators.
        """

        if values.ndim != 2:
            raise ValueError("values must be a 2D (hours, features) array")

        n_hours, n_features = values.shape

        decay = np.zeros(
            (n_hours, n_features),
            dtype=float,
        )

        for hour in range(n_hours):
            observed = ~np.isnan(values[hour])

            if hour > 0:
                decay[hour] = 0.75 * decay[hour - 1]

            decay[hour, observed] = 1.0

        return decay

    def _apply_missingness(
        self,
        values: np.ndarray,
        stay_id: Any,
    ) -> np.ndarray:
        """Randomly mask observed cells reproducibly for each stay.

        Args:
            values: Hourly dynamic feature matrix.
            stay_id: ICU stay identifier for deterministic masking.

        Returns:
            Copy of values with selected cells replaced by NaN.
        """

        values = values.copy()

        if self.missing_rate == 0.0:
            return values

        observed = np.argwhere(~np.isnan(values))

        n_hide = round(len(observed) * self.missing_rate)

        if n_hide == 0:
            return values

        key = f"{self.seed}:{stay_id}".encode()

        stay_seed = int.from_bytes(
            hashlib.sha256(key).digest()[:8],
            "little",
        )

        rng = np.random.default_rng(stay_seed)

        selected = rng.choice(
            len(observed),
            size=n_hide,
            replace=False,
        )

        rows, cols = observed[selected].T

        values[rows, cols] = np.nan

        return values

    def __call__(
        self,
        patient: Any,
    ) -> list[dict[str, Any]]:
        """Create hourly remaining-LOS samples for eligible stays.

        Args:
            patient: PyHealth patient with eICU records.

        Returns:
            Per-stay samples containing patient_id, stay_id,
            dynamic, static, los, and optionally decay.
        """

        all_stays = self._stays(patient)

        if all_stays.height == 0:
            return []

        stays, hourly, static = self._base_cohort(
            patient,
            all_stays,
        )

        samples = []

        for stay_id, los_hours, end in stays.select(
            "stay_id",
            "los_hours",
            "end",
        ).iter_rows():
            if end < self.START_HOUR:
                continue

            last_hour = min(
                end,
                self.MAX_HOURS,
            )

            # Each hourly bin becomes available only after
            # the bin is complete, preventing future leakage.
            dynamic = self._dynamic(
                hourly,
                stay_id,
                last_hour - 1,
            )

            dynamic["time"] = [float(h) for h in range(1, last_hour + 1)]

            dynamic["values"] = self._apply_missingness(
                dynamic["values"],
                stay_id,
            )

            prediction_hours = [
                float(hour)
                for hour in range(
                    self.START_HOUR,
                    last_hour + 1,
                )
            ]

            remaining = [
                max(
                    0.0,
                    float(los_hours) - hour,
                )
                for hour in prediction_hours
            ]

            sample = {
                "patient_id": patient.patient_id,
                "stay_id": stay_id,
                "dynamic": dynamic,
                "static": self._static_series(static[stay_id]),
                "los": self._labels(
                    prediction_hours,
                    remaining,
                ),
            }

            if self.include_decay:
                sample["decay"] = {
                    "time": list(dynamic["time"]),
                    "values": self._decay(dynamic["values"]),
                }

            samples.append(sample)

        return samples
