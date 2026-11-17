from copy import deepcopy
import hashlib
from typing import Any, ClassVar

import numpy as np
import polars as pl

from .yaib_eicu import YAIBLengthOfStayEICU, _DYNAMIC_FEATURES


class TPCLengthOfStayEICU(YAIBLengthOfStayEICU):
    """Hourly remaining ICU stay prediction inspired by Rocheteau et al.

    Paper: https://arxiv.org/abs/2007.09483.
    This is a standalone task, not a TPC model implementation. Static features
    are age, sex (male=1, female=0), admission height and weight, inherited
    from YAIB. Unknown sex and invalid height/weight remain NaN.
    Labels are HOURS, from hour 5 through min(floor(LOS), 336), inclusive.
    Remaining LOS is not clipped to 14 days. Dynamic values remain raw with
    NaNs; sequence_v2 processors fit normalization on training patients and
    forward-fill after observation masking. Optional decay is a separate input.

    Args:
        missing_rate: Fraction of observed hourly cells hidden, in [0, 1].
        seed: Seed combined with each stay ID for repeatable masking.
        include_decay: Add a separate 48-channel decay series. Defaults to
            False for compatibility with the team prototype.
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
        hourly = self._hourly(self._measurements(patient, stays, _DYNAMIC_FEATURES))
        return stays, hourly, {key: static[key] for key in keep}

    @staticmethod
    def _decay(values: np.ndarray) -> np.ndarray:
        """Compute causal observation recency after masking.

        Reimplements the original TPC preprocessing formula: 0 before the
        first observation, 1 when observed, 1 / (gap * 4/3) afterwards.
        Source: EmmaRocheteau/TPC-LoS-prediction/eICU_preprocessing/timeseries.py.
        No forward-looking information or cross-stay state is used.
        """
        result = np.zeros_like(values, dtype=float)
        last = np.full(values.shape[1], -1, dtype=int)
        for hour, row in enumerate(values):
            observed = np.isfinite(row)
            last[observed] = hour
            seen = last >= 0
            gap = hour - last[seen]
            result[hour, seen] = 1.0 / np.maximum(1.0, gap * (4.0 / 3.0))
        return result

    def _apply_missingness(self, values: np.ndarray, stay_id: Any) -> np.ndarray:
        """Randomly hide a fraction of originally observed values."""
        values = values.copy()

        if self.missing_rate == 0.0:
            return values

        observed = np.argwhere(~np.isnan(values))
        n_hide = round(len(observed) * self.missing_rate)

        if n_hide == 0:
            return values

        key = f"{self.seed}:{stay_id}".encode("utf-8")
        stay_seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "little")
        rng = np.random.default_rng(stay_seed)

        selected = rng.choice(len(observed), size=n_hide, replace=False)

        rows, cols = observed[selected].T
        values[rows, cols] = np.nan

        return values

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Generate hourly ICU samples with uncapped remaining LOS labels."""
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

            dynamic = self._dynamic(hourly, stay_id, last_hour)

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
