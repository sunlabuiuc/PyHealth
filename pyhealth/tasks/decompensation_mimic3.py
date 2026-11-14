"""MIMIC-III in-hospital decompensation task.

Authors: Sarah Husted, Allyson Moran (Georgia Tech CSE/OAN 6250 project).

Paper reproduced (task and data pipeline): Yeche et al., "Dynamic Survival
Analysis for Early Event Prediction", CHIL 2024,
https://proceedings.mlr.press/v248/yeche24a.html

The cohort, label and preprocessing rules are ported from the MIMIC-III
benchmark of Harutyunyan et al. (2019),
https://github.com/YerevaNN/mimic3-benchmarks (``mimic3benchmark`` package
and ``Discretizer``). Predicts, at every hour of the ICU stay, death within
a horizon (24 hours by default).

Everything except :class:`DecompensationMIMIC3` is plain Python over simple
inputs, so it can be tested with a few hand-made stays.

Deviations from the benchmark (also listed on the task class):
    * Hour ``h`` holds events in ``(h - 1, h]``, so a label at hour ``t``
      only sees events up to ``t``.
    * Categorical Glasgow coma scale values are ordinal integers, not one-hot.
    * Missing values are left as NaN for the processors to fill (forward
      fill, then the training mean) instead of the benchmark's normal values.
    * FiO2 follows the intended rule (divide by 100 when above 1 and not in
      torr); the benchmark's Python 2 code path may divide every value.
    * Outlier ranges are optional (``apply_ranges``); the benchmark's episode
      script does not apply them.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from datetime import datetime
from typing import Any, ClassVar

import numpy as np

from .base_task import BaseTask

EPS = 1e-6

FEATURES: list[str] = [
    "Capillary refill rate",
    "Diastolic blood pressure",
    "Fraction inspired oxygen",
    "Glascow coma scale eye opening",
    "Glascow coma scale motor response",
    "Glascow coma scale total",
    "Glascow coma scale verbal response",
    "Glucose",
    "Heart Rate",
    "Height",
    "Mean blood pressure",
    "Oxygen saturation",
    "Respiratory rate",
    "Systolic blood pressure",
    "Temperature",
    "Weight",
    "pH",
]

# variable -> table -> item IDs (STATUS == "ready" rows of the benchmark map).
ITEMS: dict[str, dict[str, list[int]]] = {
    "Capillary refill rate": {"chartevents": [115, 3348, 8377]},
    "Diastolic blood pressure": {
        "chartevents": [8368, 8440, 8441, 8502, 8503, 8504, 8506, 8507, 8555,
                        220051, 220180, 224643, 225310]
    },
    "Fraction inspired oxygen": {"chartevents": [189, 727, 3420, 3422, 223835]},
    "Glascow coma scale eye opening": {"chartevents": [184, 220739]},
    "Glascow coma scale motor response": {"chartevents": [454, 223901]},
    "Glascow coma scale total": {"chartevents": [198]},
    "Glascow coma scale verbal response": {"chartevents": [723, 223900]},
    "Glucose": {
        "chartevents": [807, 811, 1529, 3745, 220621, 225664, 226537],
        "labevents": [50809, 50931, 51478],
    },
    "Heart Rate": {"chartevents": [211, 220045]},
    "Height": {"chartevents": [1394, 226707, 226730]},
    "Mean blood pressure": {
        "chartevents": [52, 224, 456, 3312, 3314, 3316, 3320, 3322, 6702,
                        220052, 220181, 224322, 225312]
    },
    "Oxygen saturation": {
        "chartevents": [646, 834, 8498, 220227, 220277],
        "labevents": [50817],
    },
    "Respiratory rate": {
        "chartevents": [614, 615, 618, 651, 3603, 220210, 224422, 224689,
                        224690]
    },
    "Systolic blood pressure": {
        "chartevents": [51, 442, 455, 3313, 3315, 3317, 3321, 3323, 6701,
                        220050, 220179, 224167, 225309, 227243]
    },
    "Temperature": {
        "chartevents": [676, 677, 678, 679, 3654, 3655, 223761, 223762]
    },
    "Weight": {
        "chartevents": [763, 3580, 3581, 3582, 3693, 224639, 226512, 226531]
    },
    "pH": {
        "chartevents": [780, 860, 1126, 1673, 3839, 4202, 4753, 220274, 223830],
        "labevents": [50820, 50831, 51094, 51491],
    },
}

# Item labels the unit rules of Temperature, Weight and Height look at.
ITEM_LABELS: dict[int, str] = {
    676: "Temperature C", 677: "Temperature C (calc)", 678: "Temperature F",
    679: "Temperature F (calc)", 763: "Daily Weight", 1394: "Height Inches",
    3580: "Present Weight  (kg)", 3581: "Present Weight  (lb)",
    3582: "Present Weight  (oz)", 3654: "Temp Rectal [F]",
    3655: "Temp Skin [C]", 3693: "Weight Kg", 223761: "Temperature Fahrenheit",
    223762: "Temperature Celsius", 224639: "Daily Weight",
    226512: "Admission Weight (Kg)", 226531: "Admission Weight (lbs.)",
    226707: "Height", 226730: "Height (cm)",
}

ITEM_TO_VARIABLE: dict[tuple[str, int], str] = {
    (table, itemid): variable
    for variable, tables in ITEMS.items()
    for table, ids in tables.items()
    for itemid in ids
}

# variable -> (outlier low, valid low, valid high, outlier high)
RANGES: dict[str, tuple[float, float, float, float]] = {
    "Capillary refill rate": (0, 0, 1, 1),
    "Diastolic blood pressure": (0, 0, 375, 375),
    "Fraction inspired oxygen": (0.2, 0.21, 1, 1.1),
    "Glascow coma scale eye opening": (1, 1, 4, 4),
    "Glascow coma scale motor response": (1, 1, 6, 6),
    "Glascow coma scale total": (3, 3, 15, 15),
    "Glascow coma scale verbal response": (1, 1, 5, 5),
    "Glucose": (0, 33, 2000, 2200),
    "Heart Rate": (0, 0, 350, 390),
    "Height": (0, 0, 240, 275),
    "Mean blood pressure": (0, 14, 330, 375),
    "Oxygen saturation": (0, 0, 100, 150),
    "pH": (6.3, 6.3, 8.4, 10),
    "Respiratory rate": (0, 0, 300, 330),
    "Systolic blood pressure": (0, 0, 375, 375),
    "Temperature": (14.2, 26, 45, 47),
    "Weight": (0, 0, 250, 250),
}

# Ordinal codes for the text values of the discretizer config.
GCS_TEXT: dict[str, dict[str, int]] = {
    "Glascow coma scale eye opening": {
        "1 No Response": 1, "None": 1, "To Pain": 2, "2 To pain": 2,
        "To Speech": 3, "3 To speech": 3, "Spontaneously": 4,
        "4 Spontaneously": 4,
    },
    "Glascow coma scale motor response": {
        "1 No Response": 1, "No response": 1, "Abnormal extension": 2,
        "2 Abnorm extensn": 2, "Abnormal Flexion": 3, "3 Abnorm flexion": 3,
        "Flex-withdraws": 4, "4 Flex-withdraws": 4, "Localizes Pain": 5,
        "5 Localizes Pain": 5, "Obeys Commands": 6, "6 Obeys Commands": 6,
    },
    "Glascow coma scale verbal response": {
        "1 No Response": 1, "No Response": 1, "No Response-ETT": 1,
        "1.0 ET/Trach": 1, "Incomprehensible sounds": 2,
        "2 Incomp sounds": 2, "Inappropriate Words": 3,
        "3 Inapprop words": 3, "Confused": 4, "4 Confused": 4,
        "Oriented": 5, "5 Oriented": 5,
    },
}

_NUMERIC = re.compile(r"^(\d+(\.\d*)?|\.\d+)$")
_BP = re.compile(r"^(\d+)/(\d+)$")
_CRR = {"Normal <3 secs": 0.0, "Brisk": 0.0, "Abnormal >3 secs": 1.0,
        "Delayed": 1.0}


def _to_float(raw: Any) -> float | None:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _strict_float(raw: Any) -> float | None:
    """Parses plain numbers only; text such as ``ERROR`` or ``<5`` is None."""
    text = str(raw).strip()
    return float(text) if _NUMERIC.match(text) else None


def _mentions(text: str, label: str, needle: str) -> bool:
    return needle in text.lower() or needle in label.lower()


def clean_value(
    variable: str, raw: Any, unit: str = "", label: str = ""
) -> float | None:
    """Cleans one raw event value following the benchmark's rules.

    Args:
        variable: One of :data:`FEATURES`.
        raw: The raw ``value`` column (usually a string).
        unit: The ``valueuom`` column.
        label: The item label (see :data:`ITEM_LABELS`).

    Returns:
        The cleaned number, or None if the value is unusable.

    Examples:
        >>> clean_value("Systolic blood pressure", "120/80")
        120.0
        >>> clean_value("Fraction inspired oxygen", "40")
        0.4
        >>> clean_value("Glucose", "ERROR") is None
        True
        >>> round(clean_value("Temperature", "98.6", "", "Temperature F"), 1)
        37.0
    """
    unit = "" if unit is None else str(unit)
    label = "" if label is None else str(label)
    if variable in GCS_TEXT:
        codes = GCS_TEXT[variable]
        return float(codes[raw]) if raw in codes else None
    if variable in ("Systolic blood pressure", "Diastolic blood pressure"):
        text = str(raw).strip()
        if "/" in text:
            match = _BP.match(text)
            if match is None:
                return None
            group = 1 if variable.startswith("Systolic") else 2
            return float(match.group(group))
        return _to_float(text)
    if variable == "Capillary refill rate":
        return _CRR.get(str(raw).strip())
    if variable in ("Glucose", "pH"):
        return _strict_float(raw)
    if variable == "Oxygen saturation":
        value = _strict_float(raw)
        return None if value is None else (value * 100.0 if value <= 1 else value)
    value = _to_float(raw)
    if value is None:
        return None
    if variable == "Fraction inspired oxygen":
        return value / 100.0 if "torr" not in unit.lower() and value > 1.0 else value
    if variable == "Temperature":
        # The benchmark tests for an upper-case "F" in a lower-cased string,
        # which never matches, so only the value >= 79 rule converts.
        return (value - 32.0) * 5.0 / 9.0 if value >= 79 else value
    if variable == "Weight":
        if _mentions(unit, label, "oz"):
            return value / 16.0 * 0.453592
        if _mentions(unit, label, "lb"):
            return value * 0.453592
        return value
    if variable == "Height":
        return float(round(value * 2.54)) if _mentions(unit, label, "in") else value
    return value


def apply_range(variable: str, value: float) -> float | None:
    """Applies the benchmark's reference range to a cleaned value.

    Values outside the outlier bounds are dropped, and values between an
    outlier bound and the valid bound are clipped to the valid bound.

    Examples:
        >>> apply_range("Temperature", 20.0)
        26
        >>> apply_range("Temperature", 50.0) is None
        True
    """
    if variable not in RANGES:
        return value
    out_low, low, high, out_high = RANGES[variable]
    if value < out_low or value > out_high:
        return None
    return min(max(value, low), high)


def clean_events(
    rows: Iterable[tuple[float, str, int, Any, str]], apply_ranges: bool = False
) -> list[tuple[float, str, float]]:
    """Maps raw rows to cleaned ``(hours, variable, value)`` events.

    Args:
        rows: ``(hours since ICU admission, table, itemid, raw value, unit)``.
        apply_ranges: Whether to apply :data:`RANGES` after cleaning.

    Returns:
        Events for the 17 variables; unmapped items and unusable values are
        dropped.

    Examples:
        >>> rows = [(0.5, "chartevents", 211, "80", ""),
        ...         (0.5, "chartevents", 999999, "1", "")]
        >>> clean_events(rows)
        [(0.5, 'Heart Rate', 80.0)]
    """
    events = []
    for hours, table, itemid, raw, unit in rows:
        variable = ITEM_TO_VARIABLE.get((table, int(itemid)))
        if variable is None:
            continue
        value = clean_value(variable, raw, unit, ITEM_LABELS.get(int(itemid), ""))
        if value is not None and apply_ranges:
            value = apply_range(variable, value)
        if value is not None:
            events.append((float(hours), variable, float(value)))
    return events


def hourly_matrix(
    events: Iterable[tuple[float, str, float]], n_hours: int
) -> np.ndarray:
    """Bins events into hours ``0..n_hours`` as a ``[n_hours + 1, 17]`` array.

    Hour ``h`` holds events with ``h - 1 < time <= h`` (events at time 0 go to
    hour 0). Of several events of one variable in an hour the latest wins, and
    of several at the same instant the largest wins, as in the benchmark.
    Empty cells are NaN.

    Examples:
        >>> events = [(0.5, "Heart Rate", 80.0), (1.0, "Heart Rate", 82.0)]
        >>> hourly_matrix(events, 2)[:, FEATURES.index("Heart Rate")].tolist()
        [nan, 82.0, nan]
    """
    column = {name: index for index, name in enumerate(FEATURES)}
    values = np.full((n_hours + 1, len(FEATURES)), np.nan)
    largest: dict[tuple[float, int], float] = {}
    for time, variable, value in events:
        key = (time, column[variable])
        largest[key] = max(largest.get(key, value), value)
    for (time, index), value in sorted(largest.items()):
        hour = max(0, math.ceil(time - EPS))
        if hour <= n_hours:
            values[hour, index] = value
    return values


def age_years(intime: datetime, dob: datetime) -> float:
    """Age at ICU admission as in the benchmark (365-day years).

    Examples:
        >>> round(age_years(datetime(2020, 1, 1), datetime(2000, 1, 1)), 1)
        20.0
    """
    age = (intime - dob).total_seconds() / 3600.0 / 24.0 / 365.0
    return 90.0 if age < 0 else age


def in_hospital_mortality(
    admittime: datetime,
    dischtime: datetime,
    deathtime: datetime | None,
    dod: datetime | None,
) -> int:
    """1 if the date of death or the death time falls inside the admission.

    Examples:
        >>> admit, disch = datetime(2100, 1, 1), datetime(2100, 1, 10)
        >>> in_hospital_mortality(admit, disch, datetime(2100, 1, 5), None)
        1
        >>> in_hospital_mortality(admit, disch, None, datetime(2101, 1, 1))
        0
    """
    for moment in (dod, deathtime):
        if moment is not None and admittime <= moment <= dischtime:
            return 1
    return 0


def select_stays(stays: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Applies the benchmark's stay filters in its order.

    Stays with a ward or care-unit transfer go first, then admissions that
    still have more than one stay, then ages below 18.

    Args:
        stays: Dicts with ``hadm_id``, ``first_wardid``, ``last_wardid``,
            ``first_careunit``, ``last_careunit`` and ``age``.

    Examples:
        >>> stay = dict(hadm_id=1, first_wardid=5, last_wardid=5,
        ...             first_careunit="MICU", last_careunit="MICU", age=60.0)
        >>> len(select_stays([stay]))
        1
    """
    kept = [
        s for s in stays
        if s["first_wardid"] == s["last_wardid"]
        and s["first_careunit"] == s["last_careunit"]
    ]
    counts: dict[Any, int] = {}
    for stay in kept:
        counts[stay["hadm_id"]] = counts.get(stay["hadm_id"], 0) + 1
    return [s for s in kept if counts[s["hadm_id"]] == 1 and s["age"] >= 18]


def decompensation_steps(
    los_hours: float,
    lived_hours: float | None,
    mortality: int,
    first_event_hours: float,
    horizon: float = 24.0,
    shortest_length: float = 4.0,
) -> list[dict[str, float]]:
    """Hourly prediction steps of one stay, as ``create_decompensation.py``.

    Args:
        los_hours: ICU length of stay in hours.
        lived_hours: Death time minus ICU admission in hours, or None.
        mortality: In-hospital mortality flag of the stay.
        first_event_hours: Hours to the first valid measurement.
        horizon: The prediction horizon in hours.
        shortest_length: Steps at or before this hour are dropped.

    Returns:
        Dicts with ``t``, ``label`` (death within ``horizon``), ``event``
        (death observed) and ``duration`` (hours to death or to stay end).

    Examples:
        >>> steps = decompensation_steps(30.0, 40.0, 1, 0.5)
        >>> steps[0]["t"], steps[-1]["t"], sum(s["label"] for s in steps)
        (5.0, 30.0, 14)
    """
    lived = 1e18 if lived_hours is None else lived_hours
    event = int(mortality == 1 and lived_hours is not None)
    steps = []
    for t in np.arange(0.0, min(los_hours, lived) + EPS, 1.0):
        if t <= shortest_length or t <= first_event_hours:
            continue
        label = 0 if mortality == 0 else int(lived - t < horizon)
        duration = (lived - t) if event else (los_hours - t)
        steps.append(
            {"t": float(t), "label": label, "event": event,
             "duration": float(duration)}
        )
    return steps


def build_sample(
    patient_id: str,
    events: list[tuple[float, str, float]],
    los_hours: float,
    lived_hours: float | None,
    mortality: int,
    horizon: float = 24.0,
) -> dict[str, Any] | None:
    """Builds one per-stay sample, or None if the stay yields no steps.

    Events outside ``[0, los_hours]`` are dropped, and nothing after the last
    prediction step (so nothing after death) enters the features.

    Returns:
        A dict with ``patient_id``, ``dynamic`` (``time`` and ``values``) and
        ``label`` (``time`` and nested ``values``), the shape the per-step
        label processors expect.

    Examples:
        >>> events = [(0.5, "Heart Rate", 80.0), (9.2, "Heart Rate", 90.0)]
        >>> sample = build_sample("p1", events, 12.0, None, 0)
        >>> sample["dynamic"]["values"].shape, sample["label"]["time"][0]
        ((13, 17), 5.0)
    """
    inside = [e for e in events if -EPS < e[0] < los_hours + EPS]
    if not inside:
        return None
    first = min(e[0] for e in inside)
    steps = decompensation_steps(
        los_hours, lived_hours, mortality, first, horizon=horizon
    )
    if not steps:
        return None
    last_hour = round(steps[-1]["t"])
    return {
        "patient_id": patient_id,
        "dynamic": {
            "time": [float(h) for h in range(last_hour + 1)],
            "values": hourly_matrix(inside, last_hour),
        },
        "label": {
            "time": [s["t"] for s in steps],
            "values": [[float(s["label"])] for s in steps],
        },
    }


# --- PyHealth task ----------------------------------------------------------

_REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
    "patients": ("dob", "dod"),
    "admissions": ("hadm_id", "dischtime", "deathtime"),
    "icustays": (
        "hadm_id", "icustay_id", "outtime", "first_wardid", "last_wardid",
        "first_careunit", "last_careunit", "los",
    ),
    "chartevents": ("hadm_id", "icustay_id", "itemid", "value", "valueuom"),
    "labevents": ("hadm_id", "itemid", "value", "valueuom"),
}


def _dt(value: Any) -> datetime | None:
    """Parses a timestamp that may be a datetime, a string or missing."""
    if value is None or isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value).strip())
    except ValueError:
        return None


def _key(value: Any) -> str | None:
    """Normalizes an ID (``1``, ``1.0`` or ``"1"``) to a string key."""
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return None


def _hours(later: datetime, earlier: datetime) -> float:
    return (later - earlier).total_seconds() / 3600.0


def _rows(patient: Any, table: str) -> list[dict[str, Any]]:
    frame = patient.get_events(event_type=table, return_df=True)
    return [] if frame is None else frame.to_dicts()


class DecompensationMIMIC3(BaseTask):
    """In-hospital decompensation prediction on MIMIC-III.

    Predicts, at every hour of the ICU stay, whether the patient dies within
    ``horizon`` hours (24 by default), following the MIMIC-III benchmark of
    Harutyunyan et al. (2019). Each stay is one sample with a series of 17
    hourly clinical variables and a label for every hour after the fourth,
    once the first measurement has been made.

    Cohort (as the benchmark): stays without a ward or care-unit transfer,
    admissions with exactly one such stay, adults (age 18 or over, in
    365-day years; a negative age counts as 90). In-hospital mortality is a
    date of death or death time inside the admission. Steps run hourly from
    hour 0 to the end of the stay or the death, whichever is first, and the
    series stops at the last step, so nothing after death is seen.

    Deviations from the benchmark:

    - Hour ``h`` holds events in ``(h - 1, h]`` (the latest value of a
      variable wins), so a label at hour ``t`` only sees events up to ``t``.
    - Glasgow coma scale text values are ordinal integers, not one-hot.
    - Missing values are left to the processors: forward fill, then the
      training mean, then standardization with a missing-value mask. The
      benchmark fills with fixed normal values.
    - FiO2 is divided by 100 when above 1 and not in torr.
    - Outlier ranges are applied only with ``apply_ranges=True``; the
      benchmark's episode script does not apply them.

    Training needs a model that reads the ``sequence_v2`` input and the
    per-hour label records, such as :class:`~pyhealth.models.RNN_V2`.

    Requires the ``chartevents`` and ``labevents`` tables and a MIMIC-III
    config with the columns of ``_REQUIRED_COLUMNS`` (ward and care-unit
    columns of ``icustays``, ``deathtime`` of ``admissions``, ``hadm_id`` and
    ``icustay_id`` on the events).

    Args:
        horizon: Prediction horizon in hours; a step is positive if death
            follows within this many hours.
        apply_ranges: Whether to clip or drop values by the benchmark's
            plausibility ranges.

    Examples:
        >>> from pyhealth.datasets import MIMIC3Dataset
        >>> from pyhealth.tasks import DecompensationMIMIC3
        >>> dataset = MIMIC3Dataset(
        ...     root="/path/to/mimic-iii",
        ...     tables=["chartevents", "labevents"],
        ...     config_path="pyhealth/datasets/configs/mimic3_icu.yaml",
        ... )
        >>> samples = dataset.set_task(DecompensationMIMIC3(horizon=24))
    """

    task_name: str = "DecompensationMIMIC3"
    input_schema: ClassVar[dict[str, Any]] = {
        "dynamic": (
            "sequence_v2",
            {
                "levels": [
                    {
                        "ordering": "series",
                        "values": [len(FEATURES)],
                        "impute": [
                            {"method": "forward_fill"},
                            {"method": "mean", "by_dims": ()},
                            {"method": "zero"},
                        ],
                        "normalize": {"method": "zscore"},
                        "missing_indicator": True,
                    }
                ]
            },
        ),
    }
    output_schema: ClassVar[dict[str, Any]] = {
        "decompensation": "timeseries_binary"
    }

    def __init__(self, horizon: float = 24.0, apply_ranges: bool = False) -> None:
        super().__init__()
        self.horizon = float(horizon)
        self.apply_ranges = bool(apply_ranges)
        # Samples differ by these options, so keep their caches apart.
        if self.horizon != 24.0:
            self.task_name = f"{self.task_name}_h{self.horizon:g}"
        if self.apply_ranges:
            self.task_name = f"{self.task_name}_ranges"

    def pre_filter(self, df: Any) -> Any:
        """Drops measurement rows of items the task does not use.

        Args:
            df: The global event ``polars.LazyFrame``.

        Returns:
            The filtered frame.

        Raises:
            ValueError: If a required table or column is missing.

        Examples:
            >>> import polars as pl
            >>> df = pl.LazyFrame({
            ...     "event_type": ["chartevents", "chartevents"],
            ...     "chartevents/itemid": ["211", "999"],
            ... })
            >>> task = DecompensationMIMIC3()
            >>> missing = task._missing(df.collect_schema().names())
            >>> "patients" in missing
            True
        """
        import polars as pl

        missing = self._missing(df.collect_schema().names())
        if missing:
            raise ValueError(
                f"{self.task_name} needs these MIMIC-III columns: "
                f"{missing}; add the tables and use a config that has them."
            )
        ids: dict[str, set[int]] = {}
        for (table, itemid) in ITEM_TO_VARIABLE:
            ids.setdefault(table, set()).add(itemid)
        keep = ~pl.col("event_type").is_in(list(ids))
        for table, itemids in ids.items():
            keep = keep | (
                (pl.col("event_type") == table)
                & pl.col(f"{table}/itemid")
                .cast(pl.Int64, strict=False)
                .is_in(sorted(itemids))
            )
        return df.filter(keep)

    @staticmethod
    def _missing(columns: list[str]) -> list[str]:
        have = set(columns)
        return [
            f"{table}/{name}"
            for table, names in _REQUIRED_COLUMNS.items()
            for name in names
            if f"{table}/{name}" not in have
        ]

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Builds one sample per eligible ICU stay of a patient.

        Args:
            patient: A PyHealth patient with the MIMIC-III tables.

        Returns:
            Samples with ``patient_id``, ``stay_id``, ``dynamic`` (hourly
            series) and ``decompensation`` (per-hour labels).
        """
        demographics = _rows(patient, "patients")
        stay_rows = _rows(patient, "icustays")
        if not demographics or not stay_rows:
            return []
        dob = _dt(demographics[0].get("patients/dob"))
        dod = _dt(demographics[0].get("patients/dod"))
        admissions = {}
        for row in _rows(patient, "admissions"):
            admissions[_key(row.get("admissions/hadm_id"))] = row

        candidates = []
        for row in stay_rows:
            intime = _dt(row.get("timestamp"))
            outtime = _dt(row.get("icustays/outtime"))
            hadm = _key(row.get("icustays/hadm_id"))
            if intime is None or outtime is None or hadm not in admissions:
                continue
            los = row.get("icustays/los")
            try:
                los_hours = float(los) * 24.0
            except (TypeError, ValueError):
                los_hours = _hours(outtime, intime)
            if not math.isfinite(los_hours):
                los_hours = _hours(outtime, intime)
            candidates.append(
                {
                    "hadm_id": hadm,
                    "first_wardid": row.get("icustays/first_wardid"),
                    "last_wardid": row.get("icustays/last_wardid"),
                    "first_careunit": row.get("icustays/first_careunit"),
                    "last_careunit": row.get("icustays/last_careunit"),
                    "age": 90.0 if dob is None else age_years(intime, dob),
                    "stay_id": _key(row.get("icustays/icustay_id")),
                    "intime": intime,
                    "outtime": outtime,
                    "los_hours": los_hours,
                }
            )
        stays = select_stays(candidates)
        if not stays:
            return []

        raw = []
        for table in ("chartevents", "labevents"):
            for row in _rows(patient, table):
                time = _dt(row.get("timestamp"))
                value = row.get(f"{table}/value")
                if value is None:
                    value = row.get(f"{table}/valuenum")
                itemid = _key(row.get(f"{table}/itemid"))
                if time is None or value is None or itemid is None:
                    continue
                raw.append(
                    (
                        time, table, int(itemid), value,
                        row.get(f"{table}/valueuom") or "",
                        _key(row.get(f"{table}/icustay_id")),
                    )
                )

        samples = []
        for stay in stays:
            rows = [
                (_hours(time, stay["intime"]), table, itemid, value, unit)
                for time, table, itemid, value, unit, stay_id in raw
                if stay_id == stay["stay_id"]
                or (stay_id is None and stay["intime"] <= time <= stay["outtime"])
            ]
            adm = admissions[stay["hadm_id"]]
            deathtime = _dt(adm.get("admissions/deathtime"))
            mortality = in_hospital_mortality(
                _dt(adm.get("timestamp")),
                _dt(adm.get("admissions/dischtime")),
                deathtime,
                dod,
            )
            lived = None if deathtime is None else _hours(deathtime, stay["intime"])
            sample = build_sample(
                patient.patient_id,
                clean_events(rows, self.apply_ranges),
                stay["los_hours"],
                lived,
                mortality,
                horizon=self.horizon,
            )
            if sample is None:
                continue
            sample["stay_id"] = stay["stay_id"]
            sample["decompensation"] = sample.pop("label")
            samples.append(sample)
        return samples
