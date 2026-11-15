
import hashlib
from typing import Any, ClassVar

import numpy as np

from .yaib_eicu import YAIBLengthOfStayEICU


class TPCLengthOfStayEICU(YAIBLengthOfStayEICU):
    """Hourly ICU length-of-stay prediction with missingness experiments."""

    task_name: str = "TPCLengthOfStayEICU"
    MAX_HOURS: ClassVar[int] = 14 * 24
    START_HOUR: ClassVar[int] = 5

    def __init__(self, missing_rate: float = 0.0, seed: int = 42):
        """Configure random observation masking."""
        if not 0.0 <= missing_rate <= 1.0:
            raise ValueError("missing_rate must be between 0 and 1.")

        self.missing_rate = missing_rate
        self.seed = seed

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
        stay_seed = int.from_bytes(
            hashlib.sha256(key).digest()[:8], "little"
        )
        rng = np.random.default_rng(stay_seed)

        selected = rng.choice(
            len(observed), size=n_hide, replace=False
        )

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
                float(hour)
                for hour in range(self.START_HOUR, last_hour + 1)
            ]

            remaining = [
                max(0.0, float(los_hours) - hour)
                for hour in prediction_hours
            ]

            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "stay_id": stay_id,
                    "dynamic": dynamic,
                    "static": self._static_series(static[stay_id]),
                    "los": self._labels(prediction_hours, remaining),
                }
            )

        return samples
