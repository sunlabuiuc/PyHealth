"""SUPPORT2 time-to-death task in the Deep Cox Mixtures protocol.

Contributor: <AUTHOR NAME> (NetID: liu323)

Paper: Chirag Nagpal, Steve Yadlowsky, Negar Rostamzadeh and Katherine Heller.
Deep Cox Mixtures for Survival Regression. MLHC 2021.
https://proceedings.mlr.press/v149/nagpal21a.html

The covariates, one-hot coding and race grouping follow the paper's released
preprocessing for SUPPORT (all 9,105 patients, 44 features). Imputation and
scaling are left to the caller so they can be fit on training patients only.
"""

import math
from collections.abc import Mapping
from typing import Any, ClassVar

from pyhealth.data import Patient

from .base_task import BaseTask

#: Numeric covariates, in the paper's order. Missing values become NaN.
SUPPORT2_DCM_NUMERIC: tuple[str, ...] = (
    "age",
    "num.co",
    "meanbp",
    "wblc",
    "hrt",
    "resp",
    "temp",
    "pafi",
    "alb",
    "bili",
    "crea",
    "sod",
    "ph",
    "glucose",
    "bun",
    "urine",
    "adlp",
    "adls",
)
#: Categorical covariates and their levels. Every level gets a column (no
#: reference level is dropped) and a missing value gets all zeros, which is
#: what ``pandas.get_dummies`` produces in the paper's code.
SUPPORT2_DCM_CATEGORIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("sex", ("female", "male")),
    (
        "dzgroup",
        (
            "ARF/MOSF w/Sepsis",
            "CHF",
            "COPD",
            "Cirrhosis",
            "Colon Cancer",
            "Coma",
            "Lung Cancer",
            "MOSF w/Malig",
        ),
    ),
    ("dzclass", ("ARF/MOSF", "COPD/CHF/Cirrhosis", "Cancer", "Coma")),
    ("income", ("$11-$25k", "$25-$50k", ">$50k", "under $11k")),
    ("race", ("asian", "black", "hispanic", "other", "white")),
    ("ca", ("metastatic", "no", "yes")),
)
#: Names of the 44 entries of the ``features`` vector, in order.
SUPPORT2_DCM_FEATURES: tuple[str, ...] = SUPPORT2_DCM_NUMERIC + tuple(
    f"{name}={level}" for name, levels in SUPPORT2_DCM_CATEGORIES for level in levels
)


def _blank(value: Any) -> bool:
    """Whether a raw CSV value is missing."""
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    return isinstance(value, str) and value.strip().lower() in ("", "nan", "none")


class SurvivalTimeToEventSupport2DCM(BaseTask):
    """Time to death on SUPPORT2 with the Deep Cox Mixtures covariates.

    Use with :class:`~pyhealth.datasets.Support2Dataset`. Unlike
    :class:`~pyhealth.tasks.SurvivalTimeToEventSupport2` (the 8,873-patient
    complete-case DeepSurv benchmark with 14 covariates), this task keeps
    all 9,105 patients and the 24 covariates of the Deep Cox Mixtures paper:
    18 numeric ones, in ``SUPPORT2_DCM_NUMERIC`` order, followed by one-hot
    codes of sex, disease group, disease class, income, race and cancer
    status (``SUPPORT2_DCM_CATEGORIES``), giving ``features: [44]`` named by
    ``SUPPORT2_DCM_FEATURES``. A missing numeric value is NaN and a missing
    category is all zeros. The paper imputes missing values with the mean
    and standardizes every column; fit both on training patients only before
    training a model, since the samples are not imputed here.

    Labels are ``duration: [d.time]`` in days and ``event: [death]`` (1 for
    death, 0 for censoring). Each sample also has ``race``: the recorded
    race, or ``"missing"``. The paper's minority group is every patient whose
    race is not ``"white"``, missing race included.

    Examples:
        >>> from pyhealth.tasks import SurvivalTimeToEventSupport2DCM
        >>> task = SurvivalTimeToEventSupport2DCM()
        >>> task.output_schema
        {'duration': 'tensor', 'event': 'tensor'}
        >>> from pyhealth.tasks.survival_time_to_event_support2_dcm import (
        ...     SUPPORT2_DCM_FEATURES,
        ... )
        >>> len(SUPPORT2_DCM_FEATURES), SUPPORT2_DCM_FEATURES[18]
        (44, 'sex=female')
    """

    task_name: str = "SurvivalTimeToEventSupport2DCM"
    input_schema: ClassVar[dict[str, str]] = {"features": "tensor"}
    output_schema: ClassVar[dict[str, str]] = {
        "duration": "tensor",
        "event": "tensor",
    }

    @staticmethod
    def encode(record: Mapping[str, Any], source_id: str) -> dict[str, Any]:
        """Encode one SUPPORT2 row.

        Args:
            record: The row's raw values by column name.
            source_id: Identifier used in error messages.

        Returns:
            ``features`` (44 floats, NaN where a numeric value is missing),
            ``duration``, ``event`` and ``race``.

        Raises:
            ValueError: For non-numeric values, unknown categories, a
                missing or non-positive ``d.time``, or ``death`` not 0/1.

        Examples:
            >>> row = {name: "1" for name in SUPPORT2_DCM_NUMERIC}
            >>> row.update({"sex": "male", "dzgroup": "Coma", "dzclass": "Coma",
            ...             "income": "", "race": "white", "ca": "no",
            ...             "d.time": "30", "death": "1"})
            >>> sample = SurvivalTimeToEventSupport2DCM.encode(row, "1")
            >>> len(sample["features"]), sample["duration"], sample["race"]
            (44, [30.0], 'white')
        """
        features: list[float] = []
        for name in SUPPORT2_DCM_NUMERIC:
            raw = record.get(name)
            if _blank(raw):
                features.append(math.nan)
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"SUPPORT2 record {source_id}: non-numeric {name} {raw!r}"
                ) from exc
            if not math.isfinite(value):
                raise ValueError(f"SUPPORT2 record {source_id}: {name} not finite")
            features.append(value)
        race = "missing"
        for name, levels in SUPPORT2_DCM_CATEGORIES:
            raw = record.get(name)
            level = None if _blank(raw) else str(raw).strip()
            if level is not None and level not in levels:
                raise ValueError(f"SUPPORT2 record {source_id}: unknown {name} {raw!r}")
            features.extend(1.0 if level == option else 0.0 for option in levels)
            if name == "race" and level is not None:
                race = level
        try:
            time = float(record["d.time"])
            event = float(record["death"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"SUPPORT2 record {source_id}: missing or non-numeric d.time/death"
            ) from exc
        if not math.isfinite(time) or time <= 0:
            raise ValueError(f"SUPPORT2 record {source_id}: d.time must be > 0")
        if event not in (0.0, 1.0):
            raise ValueError(f"SUPPORT2 record {source_id}: death must be 0 or 1")
        return {
            "features": features,
            "duration": [time],
            "event": [event],
            "race": race,
        }

    def __call__(self, patient: Patient) -> list[dict[str, Any]]:
        """Return the patient's single sample.

        Args:
            patient: Patient with exactly one ``support2`` event; the patient
                ID is SUPPORT2's patient number (``sno``).

        Returns:
            One sample with ``patient_id``, ``features``, ``duration``,
            ``event`` and ``race``.

        Raises:
            ValueError: If the patient does not have exactly one record or
                the record is invalid (see :meth:`encode`).
        """
        events = patient.get_events("support2")
        if len(events) != 1:
            raise ValueError(
                f"SUPPORT2 record {patient.patient_id}: expected exactly one "
                f"support2 event, got {len(events)}"
            )
        sample = self.encode(events[0].attr_dict, patient.patient_id)
        return [{"patient_id": patient.patient_id, **sample}]
