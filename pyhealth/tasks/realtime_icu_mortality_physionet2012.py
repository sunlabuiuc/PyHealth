# Authors: Hanyang Chen (hc81), Siqi Song (siqis3)
# Paper: Frost, Li & Harris. Robust Real-Time Mortality Prediction in the Intensive
#   Care Unit using Temporal Difference Learning. ML4H 2024, PMLR 259:350-363.
# Paper link: https://proceedings.mlr.press/v259/frost25a.html
# Description: Open-access version of the real-time ICU mortality task on the
#   PhysioNet/CinC Challenge 2012 data (12,000 ICU stays, first 48 hours), sharing the
#   feature vocabulary of RealTimeICUMortalityMIMIC4 for its laboratory features.
"""Real-time ICU mortality prediction on PhysioNet 2012 (open access)."""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Any, ClassVar

import polars as pl

from .realtime_icu_mortality_mimic4 import (
    _LAB,
    RealTimeICUMortalityMIMIC4,
    _to_float,
)

#: Origin of the elapsed-time axis used by ``PhysioNet2012Dataset``.
_ORIGIN = datetime(2012, 1, 1)


class RealTimeICUMortalityPhysioNet2012(RealTimeICUMortalityMIMIC4):
    """Real-time ICU mortality with next-state observations (PhysioNet 2012).

    The same sample format as
    :class:`~pyhealth.tasks.RealTimeICUMortalityMIMIC4` on the open-access
    PhysioNet/Computing in Cardiology Challenge 2012 data, which can be
    downloaded without credentials. It makes the temporal-difference
    framework of Frost et al. (ML4H 2024) usable on open data:

    * **Features.** 20 laboratory parameters map onto the MIMIC-IV task's lab
      features, with the same ids and units, so models can be validated
      across datasets with ``feature_set="labs"``. With ``feature_set="all"``
      (default) the vital signs and other parameters (heart rate, blood
      pressures, temperature, GCS, respiratory rate, urine output, FiO2,
      mechanical ventilation, magnesium, cholesterol, troponin-I) are added
      as extra features whose ids follow the MIMIC-IV vocabulary. The data
      has no drug infusions.
    * **Timeline.** Each record is one ICU stay with observations from its
      first 48 hours; elapsed times are placed after a fixed origin.
      Markers whose next-state window lies beyond the observed 48 hours are
      terminal, so the TD target bootstraps within the observation period.
    * **Labels.** The challenge's ``Survival`` (days from ICU admission to a
      recorded death, in or out of hospital; -1 if none) gives the time of
      death, so all horizons include deaths after discharge.
    * **Demographics.** Admission age, gender (0 = female in the challenge
      data) and weight (admission weight, then charted weights).

    Works with :class:`~pyhealth.datasets.PhysioNet2012Dataset`.

    Args:
        **kwargs: Any argument of
            :class:`~pyhealth.tasks.RealTimeICUMortalityMIMIC4`;
            ``feature_set`` is ``"all"`` (default) or ``"labs"``.

    Examples:
        >>> from pyhealth.datasets import PhysioNet2012Dataset
        >>> from pyhealth.tasks import RealTimeICUMortalityPhysioNet2012
        >>> dataset = PhysioNet2012Dataset(  # doctest: +SKIP
        ...     root="data/physionet2012", download=True
        ... )
        >>> task = RealTimeICUMortalityPhysioNet2012(next_state_delay_hours=12)
        >>> samples = dataset.set_task(task)  # doctest: +SKIP
    """

    task_name: str = "RealTimeICUMortalityPhysioNet2012"

    #: Challenge laboratory parameters -> lab features of the MIMIC-IV task.
    LAB_PARAMETERS: ClassVar[dict[str, str]] = {
        "Albumin": "Albumin",
        "ALP": "Alkaline phosphatase",
        "ALT": "Alanine transaminase",
        "AST": "Aspartate transferase",
        "Bilirubin": "Total bilirubin",
        "BUN": "Urea",
        "Creatinine": "Creatinine",
        "Glucose": "Glucose (serum)",
        "HCO3": "Bicarbonate",
        "HCT": "Haematocrit",
        "K": "Potassium",
        "Lactate": "Lactate",
        "Na": "Sodium",
        "PaCO2": "Blood CO2 pressure",
        "PaO2": "Blood O2 pressure",
        "pH": "Blood pH",
        "Platelets": "Platelet count",
        "SaO2": "Blood O2 saturation",
        "TroponinT": "Troponin-T",
        "WBC": "White blood cell count",
    }

    #: Other challenge parameters, added as features after the MIMIC-IV ones.
    EXTRA_PARAMETERS: ClassVar[dict[str, str]] = {
        "HR": "Heart rate",
        "SysABP": "Systolic BP (invasive)",
        "DiasABP": "Diastolic BP (invasive)",
        "MAP": "Mean BP (invasive)",
        "NISysABP": "Systolic BP (non-invasive)",
        "NIDiasABP": "Diastolic BP (non-invasive)",
        "NIMAP": "Mean BP (non-invasive)",
        "Temp": "Temperature",
        "GCS": "Glasgow coma scale",
        "RespRate": "Respiratory rate",
        "Urine": "Urine output",
        "FiO2": "FiO2",
        "MechVent": "Mechanical ventilation",
        "Mg": "Magnesium",
        "Cholesterol": "Cholesterol",
        "TroponinI": "Troponin-I",
    }

    def __init__(self, **kwargs: Any) -> None:
        if kwargs.get("feature_set", "all") not in ("all", "labs"):
            raise ValueError("feature_set must be 'all' or 'labs' for PhysioNet 2012")
        super().__init__(**kwargs)

    @classmethod
    def build_feature_names(cls) -> list[str]:
        """MIMIC-IV vocabulary followed by the challenge-only features.

        Returns:
            The feature names; the first ones (and their ids) are identical to
            :meth:`RealTimeICUMortalityMIMIC4.build_feature_names`.
        """
        return super().build_feature_names() + list(cls.EXTRA_PARAMETERS.values())

    def pre_filter(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Keeps the ``patients`` rows and the task's ``events`` rows.

        Args:
            df: The dataset's global event frame.

        Returns:
            The filtered frame.
        """
        columns = set(df.collect_schema().names())
        keep = pl.col("event_type") == "patients"
        if "events/parameter" in columns:
            names = list(self.LAB_PARAMETERS) + list(self.EXTRA_PARAMETERS) + ["Weight"]
            keep = keep | (
                (pl.col("event_type") == "events")
                & pl.col("events/parameter").is_in(names)
            )
        return df.filter(keep)

    def _parameter_map(self) -> dict[str, str]:
        """Challenge parameter -> feature name for the configured feature set."""
        mapping = dict(self.LAB_PARAMETERS)
        if self.feature_set == "all":
            mapping.update(self.EXTRA_PARAMETERS)
        return mapping

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Generates one sample per (subsampled) state marker of an ICU stay.

        Args:
            patient: A :class:`~pyhealth.data.Patient` from
                :class:`~pyhealth.datasets.PhysioNet2012Dataset` (one record).

        Returns:
            A list of sample dictionaries (same format as the MIMIC-IV task).
        """
        info = patient.get_events(event_type="patients")
        if not info:
            return []
        info = info[0]
        age = _to_float(getattr(info, "age", None))
        if not math.isfinite(age):
            return []
        gender = 1.0 if _to_float(getattr(info, "gender", None)) == 0 else 0.0
        survival = _to_float(getattr(info, "survival", None))
        death = (
            _ORIGIN + timedelta(days=survival)
            if math.isfinite(survival) and survival >= 0
            else None
        )
        low_w, high_w = self.VALID_WEIGHT_KG
        weights: list[tuple[Any, float]] = []
        admission_weight = _to_float(getattr(info, "weight", None))
        if low_w <= admission_weight <= high_w:
            weights.append((_ORIGIN - timedelta(days=1), admission_weight))

        df = patient.get_events(event_type="events", return_df=True)
        mapping = self._parameter_map()
        events = []
        if df.height:
            for time, name, value in zip(
                df["timestamp"].to_list(),
                df["events/parameter"].to_list(),
                df["events/value"].to_list(),
            ):
                value = _to_float(value)
                if time is None or not math.isfinite(value) or value < 0:
                    continue
                if name == "Weight":
                    if low_w <= value <= high_w:
                        weights.append((time, value))
                    continue
                feature = mapping.get(str(name))
                if feature is None:
                    continue
                value *= self.LAB_UNIT_FACTORS.get(feature, 1.0)
                events.append((time, feature, value, _LAB))
        events.sort(key=lambda e: (e[0], e[1]))
        return self._timeline_samples(
            patient.patient_id,
            events,
            weights,
            death,
            gender,
            lambda _when: age,
        )
