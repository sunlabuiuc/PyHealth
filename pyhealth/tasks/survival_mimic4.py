"""In-hospital survival prediction from the first ICU hours on MIMIC-IV.

The survival setting of TrajSurv (Zeng et al., MLHC 2025,
https://arxiv.org/abs/2508.00657): hourly labs and vitals over the first 36
hours of an ICU stay predict the time to in-hospital death, censored at
hospital discharge. TrajSurv used MIMIC-III (MetaVision stays) and eICU;
this task builds the same setting on MIMIC-IV with the YAIB feature set.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, ClassVar

import numpy as np
import polars as pl

from .yaib_mimic4 import _parse_time, _YAIBMIMIC4Task


class InHospitalSurvivalMIMIC4(_YAIBMIMIC4Task):
    """Time to in-hospital death after the first 36 ICU hours, on MIMIC-IV.

    Works with :class:`~pyhealth.datasets.MIMIC4EHRDataset`.

    Cohort: the YAIB base cohort (adults, ICU stays of 6 hours or more with
    enough measurements; see :class:`~pyhealth.tasks.YAIBMortalityMIMIC4`),
    restricted to ICU stays of at least ``observation_hours`` hours and to
    the first such stay of each hospital admission. Stays whose patient died
    or left the hospital before the end of the observation window are
    dropped.

    Each sample is one stay:

    - ``dynamic``: the 48 YAIB hourly variables (labs, vitals, urine output)
      for hours ``0 .. observation_hours - 1`` after ICU admission, a
      ``jagged`` series that the processor forward-fills, standardizes
      and masks (36 hourly bins by default, as in TrajSurv).
    - ``static``: age, sex, height and weight, a flat vector standardized
      with missing-value indicators, which
      :class:`~pyhealth.models.TrajSurv` concatenates to its initial state
      (unlike the YAIB tasks, where it is a ``jagged`` series at
      ``t = 0``). As in YAIB, height and weight are the medians over the
      whole ICU stay, so they can include measurements charted after the
      observation window.
    - ``summary``: the last measured value of each hourly variable in the
      window followed by the static variables, for models that need one flat
      vector, such as :class:`~pyhealth.models.SurvivalMDN` (missing values
      get indicators and are filled after standardization).
    - ``duration`` and ``event``: the time from the end of the observation
      window to death in the hospital (``event`` 1) or to hospital discharge
      (``event`` 0), in ``time_unit``.

    Fit the processors on the training patients only, with
    ``set_task(task, split=...)``.

    Requires the ``labevents``, ``chartevents``, ``outputevents`` and
    ``admissions`` tables.

    Args:
        observation_hours: Length of the observation window (36 in TrajSurv).
        time_unit: ``"days"`` (default) or ``"hours"`` for ``duration``.

    Examples:
        >>> from pyhealth.datasets import MIMIC4EHRDataset
        >>> from pyhealth.tasks import InHospitalSurvivalMIMIC4
        >>> dataset = MIMIC4EHRDataset(
        ...     root="/path/to/mimic-iv/2.2",
        ...     tables=["labevents", "chartevents", "outputevents", "admissions"],
        ... )
        >>> samples = dataset.set_task(InHospitalSurvivalMIMIC4())  # doctest: +SKIP
    """

    task_name: str = "InHospitalSurvivalMIMIC4"
    input_schema: ClassVar[dict[str, Any]] = {
        "dynamic": _YAIBMIMIC4Task.input_schema["dynamic"],
        "static": (
            "tensor",
            {
                "normalization": "zscore",
                "missing_indicators": True,
                "impute_strategy": "zero",
            },
        ),
        "summary": (
            "tensor",
            {
                "normalization": "zscore",
                "missing_indicators": True,
                "impute_strategy": "zero",
            },
        ),
    }
    output_schema: ClassVar[dict[str, Any]] = {"duration": "tensor", "event": "tensor"}
    _EXTRA_TABLES: ClassVar[tuple[str, ...]] = ("admissions",)

    def __init__(self, observation_hours: int = 36, time_unit: str = "days") -> None:
        super().__init__()
        if observation_hours < 2:
            raise ValueError("observation_hours must be at least 2")
        if time_unit not in ("days", "hours"):
            raise ValueError("time_unit must be 'days' or 'hours'")
        self.observation_hours = int(observation_hours)
        self.time_unit = time_unit

    def _outcomes(self, patient: Any) -> dict[int, tuple[datetime, bool]]:
        """Per admission: the end of follow-up and whether it is a death."""
        admissions = patient.get_events(event_type="admissions", return_df=True)
        if admissions.height == 0:
            return {}
        rows = admissions.select(
            pl.col("admissions/hadm_id").cast(pl.Int64).alias("hadm_id"),
            _parse_time("admissions/dischtime").alias("dischtime"),
            _parse_time("admissions/deathtime").alias("deathtime"),
            pl.col("admissions/hospital_expire_flag")
            .cast(pl.Int64, strict=False)
            .fill_null(0)
            .alias("expired"),
        )
        outcomes = {}
        for hadm_id, dischtime, deathtime, expired in rows.iter_rows():
            if hadm_id is None:
                continue
            died = bool(expired) and (deathtime is not None or dischtime is not None)
            end = (deathtime or dischtime) if died else dischtime
            if end is not None:
                outcomes[hadm_id] = (end, died)
        return outcomes

    def _summary(self, dynamic: dict[str, Any], static: list[float]) -> list[float]:
        values = dynamic["values"]
        last = np.full(values.shape[1], np.nan)
        for feature in range(values.shape[1]):
            observed = values[~np.isnan(values[:, feature]), feature]
            if observed.size:
                last[feature] = observed[-1]
        return [float(v) for v in last] + [float(v) for v in static]

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        all_stays = self._stays(patient)
        if all_stays.height == 0:
            return []
        stays, hourly, static = self._base_cohort(patient, all_stays)
        outcomes = self._outcomes(patient)
        window = timedelta(hours=self.observation_hours)
        scale = 24.0 if self.time_unit == "days" else 1.0

        samples, seen = [], set()
        for stay_id, hadm_id, intime, los_hours in stays.sort("intime").select(
            "stay_id", "hadm_id", "intime", "los_hours"
        ).iter_rows():
            if hadm_id in seen or los_hours < self.observation_hours:
                continue
            outcome = outcomes.get(hadm_id)
            if outcome is None:
                continue
            end, died = outcome
            duration = (end - (intime + window)).total_seconds() / 3600.0 / scale
            if duration <= 0:
                continue
            seen.add(hadm_id)
            dynamic = self._dynamic(hourly, stay_id, self.observation_hours - 1)
            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "stay_id": stay_id,
                    "dynamic": dynamic,
                    "static": static[stay_id],
                    "summary": self._summary(dynamic, static[stay_id]),
                    "duration": [duration],
                    "event": [float(died)],
                }
            )
        return samples
