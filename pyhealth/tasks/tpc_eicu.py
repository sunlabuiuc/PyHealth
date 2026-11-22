"""Hourly remaining-LOS task, contributed by Xiao Luo.

UIUC NetID: xiaoluo6.
Paper: Temporal Pointwise Convolutional Networks for Length of Stay
Prediction in the Intensive Care Unit, https://arxiv.org/abs/2007.09483.
Implements five-hour cohort and labels; exposes preprocessing extension points.
"""

from copy import deepcopy
from typing import Any, ClassVar

import numpy as np
import polars as pl

from .yaib_eicu import _DYNAMIC_FEATURES, YAIBLengthOfStayEICU


class TPCLengthOfStayEICU(YAIBLengthOfStayEICU):
    """Hourly remaining ICU stay prediction inspired by Rocheteau et al.

    Paper: https://arxiv.org/abs/2007.09483.
    This is a standalone task, not a TPC model implementation. Static features
    are age, sex (male=1, female=0), admission height and weight, inherited
    from YAIB. Unknown sex and invalid height/weight remain NaN.
    Labels are HOURS, from hour 5 through min(floor(LOS), 336), inclusive.
    Dynamic timestamps mark completed bins: bin [h-1,h) is available at h.
    Remaining LOS is not clipped to 14 days. Dynamic values remain raw with
    NaNs; sequence_v2 processors fit normalization on training patients and
    forward-fill using inherited YAIB processors. Nonzero missingness and decay
    are explicit placeholders pending preprocessing implementation.

    Args:
        missing_rate: Planned removal fraction in [0, 1]. Only 0 works until
            the preprocessing extension point is implemented.
        seed: Seed combined with each stay ID for repeatable masking.
        include_decay: Planned separate 48-channel decay series. Leave False
            until the preprocessing extension point is implemented.
    Examples:
        >>> from pyhealth.tasks import TPCLengthOfStayEICU
        >>> task = TPCLengthOfStayEICU()
        >>> task.START_HOUR, task.MAX_HOURS
        (5, 336)
        >>> task.missing_rate
        0.0

    """

    task_name: str = "TPCLengthOfStayEICU"
    MAX_HOURS: ClassVar[int] = 14 * 24
    START_HOUR: ClassVar[int] = 5

    def __init__(
        self, missing_rate: float = 0.0, seed: int = 42, include_decay: bool = False
    ) -> None:
        """Configure random observation masking."""
        if not 0.0 <= missing_rate <= 1.0:
            raise ValueError("missing_rate must be between 0 and 1.")

        self.missing_rate = missing_rate
        self.seed = seed
        self.input_schema = deepcopy(type(self).input_schema)
        self.include_decay = include_decay
        if include_decay:
            self.input_schema["decay"] = (
                "sequence_v2",
                {
                    "levels": [
                        {"ordering": "series", "values": [len(self.DYNAMIC_FEATURES)]}
                    ]
                },
            )

    def _base_cohort(
        self, patient: Any, stays: pl.DataFrame
    ) -> tuple[pl.DataFrame, pl.DataFrame, dict[int, list[float]]]:
        """Keep known adults with at least five hours; allow sparse records.

        YAIB's six-hour and measurement-density exclusions do not implement
        this project's cohort. Missing age is excluded conservatively.

        Args:
            patient: PyHealth patient containing eICU events.
            stays: Parsed stays with stay_id and los_hours columns.

        Returns:
            Eligible stays, aggregated measurements, and static features
            indexed by stay ID.
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
        # TODO(B): review feature coverage and aggregation. This
        # existing YAIB path is a provisional baseline, not a new implementation.
        hourly = self._hourly(self._measurements(patient, stays, _DYNAMIC_FEATURES))
        return stays, hourly, {key: static[key] for key in keep}

    @staticmethod
    def _decay(values: np.ndarray) -> np.ndarray:
        """Extension point for observation-recency features.

        TODO(B): implement after agreeing on the decay formula.
        Expected shape equals values.shape; compute AFTER observation removal,
        separately per stay. Do not read future values.
        """
        raise NotImplementedError(
            "TODO(B): implement _decay before include_decay=True."
        )

    def _apply_missingness(self, values: np.ndarray, stay_id: Any) -> np.ndarray:
        """Extension point for reproducible observation removal.

        TODO(A/B): integrate the existing _apply_missingness implementation
        from the repository-root tpc_eicu.py prototype. Do not reimplement
        working team code. Confirm its removal unit and seed behavior before
        integration; run the pending masking tests afterwards. The zero-removal
        baseline returns an independent unchanged copy.
        """
        if self.missing_rate == 0.0:
            return values.copy()
        raise NotImplementedError(
            "TODO(A/B): integrate existing prototype masking before missing_rate > 0."
        )

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Generate hourly ICU samples with uncapped remaining LOS labels.

        Args:
            patient: PyHealth patient with eICU admission and measurement events.

        Returns:
            One dictionary per eligible stay containing patient_id, stay_id,
            dynamic, static, los, and optional decay series. Returns an empty
            list when no eligible stays exist. Dynamic/decay time marks bin
            completion; los time marks the prediction hour. Labels are hours.
        """
        all_stays = self._stays(patient)

        if all_stays.height == 0:
            return []

        stays, hourly, static = self._base_cohort(patient, all_stays)

        samples = []

        for stay_id, los_hours, end in stays.select(
            "stay_id", "los_hours", "end"
        ).iter_rows():
            if end < self.START_HOUR:
                continue

            last_hour = min(end, self.MAX_HOURS)

            # Bin b contains offsets [60*b, 60*(b+1)); it is only
            # available at b+1. Never expose an unfinished bin at label h.
            dynamic = self._dynamic(hourly, stay_id, last_hour - 1)
            dynamic["time"] = [float(h) for h in range(1, last_hour + 1)]

            # Keep all observations from hour 0 onward.
            dynamic["values"] = self._apply_missingness(dynamic["values"], stay_id)

            prediction_hours = [
                float(hour) for hour in range(self.START_HOUR, last_hour + 1)
            ]

            remaining = [max(0.0, float(los_hours) - hour) for hour in prediction_hours]

            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "stay_id": stay_id,
                    "dynamic": dynamic,
                    "static": self._static_series(static[stay_id]),
                    "los": self._labels(prediction_hours, remaining),
                }
            )

            if self.include_decay:
                samples[-1]["decay"] = {
                    "time": list(dynamic["time"]),
                    "values": self._decay(dynamic["values"]),
                }

        return samples
