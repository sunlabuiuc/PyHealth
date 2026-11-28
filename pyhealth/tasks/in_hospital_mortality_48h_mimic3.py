"""48-hour in-hospital mortality task on MIMIC-III, as summary statistics.

Reproduces the feature extraction of Hegselmann et al. (2020), "An
evaluation of the doctor-interpretability of generalized additive models
with interactions" (adapted from the Harutyunyan et al. 2019 in-hospital
mortality benchmark): the mean and standard deviation of 17 clinical
variables over the first 48 hours of an ICU stay, giving 34 raw-unit
features per sample (missing values are not z-scored away).

The itemid-to-variable mapping, GCS/capillary-refill categorical value
mapping, and outlier/valid-range thresholds below are adapted from the
public resources of the original Harutyunyan/Hegselmann benchmark code
(``YerevaNN/mimic3-benchmarks``, ``mimic3benchmark/resources/*`` and
``mimic3models/resources/channel_info.json``), used as a reference per the
course's allowed use of original authors' code; the official cohort-level
reproduction (exact row counts, outlier counts, EBM baseline) is run by the
team locally, outside this PyHealth contribution.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any, ClassVar

import polars as pl

from .base_task import BaseTask

#: Sentinel for an unmeasured variable, or a std that cannot be computed
#: (fewer than 2 readings). Kept in raw units, never z-scored.
MISSING = -1.0

#: The 17 Harutyunyan/Hegselmann clinical variables, in a fixed (alphabetical)
#: order. ``FEATURE_NAMES`` is ``[f"{v}_mean" for v in VARIABLES] +
#: [f"{v}_std" for v in VARIABLES]`` (34 features total).
VARIABLES: ClassVar[list[str]] = [
    "capillary_refill_rate",
    "diastolic_blood_pressure",
    "fraction_inspired_oxygen",
    "glascow_coma_scale_eye_opening",
    "glascow_coma_scale_motor_response",
    "glascow_coma_scale_total",
    "glascow_coma_scale_verbal_response",
    "glucose",
    "heart_rate",
    "height",
    "mean_blood_pressure",
    "oxygen_saturation",
    "respiratory_rate",
    "systolic_blood_pressure",
    "temperature",
    "weight",
    "ph",
]

FEATURE_NAMES: ClassVar[list[str]] = [f"{v}_mean" for v in VARIABLES] + [
    f"{v}_std" for v in VARIABLES
]


def _f_to_c(value: float) -> float:
    return (value - 32.0) * 5.0 / 9.0


def _lb_to_kg(value: float) -> float:
    return value * 0.45359237


def _oz_to_kg(value: float) -> float:
    return value * 0.0283495


def _in_to_cm(value: float) -> float:
    return value * 2.54


def _IDENTITY(value: float) -> float:
    return value


def _rescale_fio2(value: float) -> float:
    """FiO2 is sometimes charted as a percentage (21-100) instead of the
    canonical fraction (0.21-1.0). Adapted (simplified) from
    mimic3benchmark/preprocessing.py's ``clean_fio2``, which also excludes
    readings whose unit contains "torr"; this task does not carry enough
    per-reading unit context to replicate that exclusion, so it rescales
    every value greater than 1.0."""
    return value / 100.0 if value > 1.0 else value


def _rescale_oxygen_saturation(value: float) -> float:
    """SpO2/SaO2 is sometimes charted as a fraction (0-1) instead of the
    canonical percentage (0-100). Adapted from mimic3benchmark/
    preprocessing.py's ``clean_o2sat``."""
    return value * 100.0 if value <= 1.0 else value


#: Per-variable rescaling applied after unit conversion and before outlier
#: filtering, for variables whose raw scale (fraction vs. percentage) is
#: inconsistent across MIMIC-III sources.
_RESCALE_FNS: dict[str, Callable[[float], float]] = {
    "fraction_inspired_oxygen": _rescale_fio2,
    "oxygen_saturation": _rescale_oxygen_saturation,
}

# Categorical value -> ordinal, adapted from mimic3models/resources/
# channel_info.json (GCS components) and mimic3benchmark/preprocessing.py's
# clean_crr (capillary refill rate). Keys are lower-cased/stripped before
# lookup, to be robust to the casing variants MIMIC-III chartevents uses.
_CRR_MAP: dict[str, float] = {
    "normal <3 secs": 0.0,
    "brisk": 0.0,
    "abnormal >3 secs": 1.0,
    "delayed": 1.0,
}
_GCS_EYE_MAP: dict[str, float] = {
    "none": 0.0,
    "1 no response": 1.0,
    "2 to pain": 2.0,
    "to pain": 2.0,
    "3 to speech": 3.0,
    "to speech": 3.0,
    "4 spontaneously": 4.0,
    "spontaneously": 4.0,
}
_GCS_MOTOR_MAP: dict[str, float] = {
    "1 no response": 1.0,
    "no response": 1.0,
    "2 abnorm extensn": 2.0,
    "abnormal extension": 2.0,
    "3 abnorm flexion": 3.0,
    "abnormal flexion": 3.0,
    "4 flex-withdraws": 4.0,
    "flex-withdraws": 4.0,
    "5 localizes pain": 5.0,
    "localizes pain": 5.0,
    "6 obeys commands": 6.0,
    "obeys commands": 6.0,
}
_GCS_VERBAL_MAP: dict[str, float] = {
    "no response-ett": 1.0,
    "no response": 1.0,
    "1 no response": 1.0,
    "1.0 et/trach": 1.0,
    "2 incomp sounds": 2.0,
    "incomprehensible sounds": 2.0,
    "3 inapprop words": 3.0,
    "inappropriate words": 3.0,
    "4 confused": 4.0,
    "confused": 4.0,
    "5 oriented": 5.0,
    "oriented": 5.0,
}

# Variable -> itemids (as strings) and how to read/convert them. An itemid
# present in more than one of {chart,}/{lab,} dicts below is read from each
# matching event independently. "categorical" variables are read from the
# raw string `value` column and mapped via the table; everything else is
# read from the numeric `valuenum` column and passed through `unit`.
# Itemids and units adapted from mimic3benchmark/resources/
# itemid_to_variable_map.csv (STATUS=ready rows), CareVue + MetaVision +
# labevents sources.
_VARIABLE_SPECS: dict[str, dict[str, Any]] = {
    "capillary_refill_rate": {
        "chart_itemids": {"3348", "115", "8377"},
        "categorical": _CRR_MAP,
    },
    "diastolic_blood_pressure": {
        "chart_itemids": {
            "8368", "220051", "225310", "8555", "8441", "220180", "8502",
            "8440", "8503", "8504", "8507", "8506", "224643",
        },
    },
    "fraction_inspired_oxygen": {
        "chart_itemids": {"3420", "223835", "3422", "189", "727"},
    },
    "glascow_coma_scale_eye_opening": {
        "chart_itemids": {"184", "220739"},
        "categorical": _GCS_EYE_MAP,
    },
    "glascow_coma_scale_motor_response": {
        "chart_itemids": {"454", "223901"},
        "categorical": _GCS_MOTOR_MAP,
    },
    # CareVue-only by design: MetaVision never charts a single "GCS Total"
    # item (nor does the official mimic3-benchmarks itemid map or pipeline
    # derive one from eye+motor+verbal), so this variable is intentionally
    # sparser for MetaVision-era stays, matching the reference benchmark.
    "glascow_coma_scale_total": {
        "chart_itemids": {"198"},
    },
    "glascow_coma_scale_verbal_response": {
        "chart_itemids": {"723", "223900"},
        "categorical": _GCS_VERBAL_MAP,
    },
    "glucose": {
        "chart_itemids": {
            "807", "811", "1529", "3745", "225664", "220621", "226537",
        },
        "lab_itemids": {"50931", "50809", "51478"},
    },
    "heart_rate": {
        "chart_itemids": {"211", "220045"},
    },
    "height": {
        "chart_itemids": {"226730"},
        "chart_itemids_in": {"226707", "1394"},
    },
    "mean_blood_pressure": {
        "chart_itemids": {
            "52", "220052", "225312", "224", "6702", "224322", "456",
            "220181", "3312", "3314", "3316", "3322", "3320",
        },
    },
    "oxygen_saturation": {
        "chart_itemids": {"834", "8498", "220227", "646", "220277"},
        "lab_itemids": {"50817"},
    },
    "respiratory_rate": {
        "chart_itemids": {
            "618", "220210", "3603", "224689", "614", "651", "224422",
            "615", "224690",
        },
    },
    "systolic_blood_pressure": {
        "chart_itemids": {
            "51", "220050", "225309", "6701", "455", "220179", "3313",
            "3315", "442", "3317", "3323", "3321", "224167", "227243",
        },
    },
    "temperature": {
        "chart_itemids": {"3655", "677", "676", "223762"},
        "chart_itemids_f": {"3654", "678", "223761", "679"},
    },
    "weight": {
        "chart_itemids": {"763", "224639", "3693", "3580", "226512"},
        "chart_itemids_lb": {"3581", "226531"},
        "chart_itemids_oz": {"3582"},
    },
    "ph": {
        "chart_itemids": {
            "3839", "1673", "780", "1126", "4753", "4202", "860", "223830",
            "220274",
        },
        "lab_itemids": {"50820", "51491", "50831", "51094"},
    },
}

# Outlier/valid thresholds adapted from mimic3benchmark/resources/
# variable_ranges.csv. A reading outside [OUTLIER_LOW, OUTLIER_HIGH] is
# dropped before computing mean/std for that variable's window, approximating
# the paper's implausible-measurement filtering (the exact count of dropped
# rows depends on the real cohort, reproduced locally, not here).
VARIABLE_RANGES: ClassVar[dict[str, tuple]] = {
    "capillary_refill_rate": (0, 1),
    "diastolic_blood_pressure": (0, 375),
    "fraction_inspired_oxygen": (0.2, 1.1),
    "glascow_coma_scale_eye_opening": (1, 4),
    "glascow_coma_scale_motor_response": (1, 6),
    "glascow_coma_scale_total": (3, 15),
    "glascow_coma_scale_verbal_response": (1, 5),
    "glucose": (0, 2200),
    "heart_rate": (0, 390),
    "height": (0, 275),
    "mean_blood_pressure": (0, 375),
    "oxygen_saturation": (0, 150),
    "respiratory_rate": (0, 330),
    "systolic_blood_pressure": (0, 375),
    "temperature": (14.2, 47),
    "weight": (0, 250),
    "ph": (6.3, 10),
}


def _build_itemid_index():
    chart_variable: dict[str, str] = {}
    chart_unit: dict[str, Callable[[float], float]] = {}
    lab_variable: dict[str, str] = {}
    for variable, spec in _VARIABLE_SPECS.items():
        for itemid in spec.get("chart_itemids", ()):
            chart_variable[itemid] = variable
            chart_unit[itemid] = _IDENTITY
        for itemid in spec.get("chart_itemids_f", ()):
            chart_variable[itemid] = variable
            chart_unit[itemid] = _f_to_c
        for itemid in spec.get("chart_itemids_in", ()):
            chart_variable[itemid] = variable
            chart_unit[itemid] = _in_to_cm
        for itemid in spec.get("chart_itemids_lb", ()):
            chart_variable[itemid] = variable
            chart_unit[itemid] = _lb_to_kg
        for itemid in spec.get("chart_itemids_oz", ()):
            chart_variable[itemid] = variable
            chart_unit[itemid] = _oz_to_kg
        for itemid in spec.get("lab_itemids", ()):
            lab_variable[itemid] = variable
    return chart_variable, chart_unit, lab_variable


_CHART_ITEMID_VARIABLE, _CHART_ITEMID_UNIT, _LAB_ITEMID_VARIABLE = (
    _build_itemid_index()
)
_CATEGORICAL_VARIABLES: dict[str, dict[str, float]] = {
    name: spec["categorical"]
    for name, spec in _VARIABLE_SPECS.items()
    if "categorical" in spec
}


def _hadm_key(value: Any) -> int | None:
    """Normalizes a hadm_id to a plain int for joining across tables.

    ADMISSIONS and ICUSTAYS are scanned as independent dask/polars tables,
    so their hadm_id columns are not guaranteed to be inferred as the same
    dtype (e.g. Int64 vs Float64) on real data; a plain dict keyed by the
    raw attribute values could then silently miss matches (``100 == 100.0``
    but ``"100" != "100.0"``). hadm_id is always an integer id in MIMIC-III,
    so round-tripping through float first normalizes int, float and
    numeric-string representations alike.

    Returns None for a missing hadm_id (rather than raising), so a record
    with no hadm_id simply never matches the join -- the same outcome a
    plain ``dict.get`` gave before this coercion was added.
    """
    if value is None:
        return None
    return int(float(value))


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def _mean_std(values: list[float]) -> tuple:
    if not values:
        return MISSING, MISSING
    if len(values) == 1:
        return values[0], MISSING
    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, variance**0.5


class InHospitalMortality48hMIMIC3(BaseTask):
    """48h summary-statistic in-hospital mortality task for MIMIC-III.

    Works with :class:`~pyhealth.datasets.MIMIC3Dataset` loaded with at
    least the ``chartevents`` table (``labevents`` is also used, for
    glucose and pH readings that are more commonly charted as labs).

    For each ICU stay with a known outcome, age >= 18 at admission, and a
    length of stay of at least 48 hours, extracts the first 48 hours of
    :data:`VARIABLES` (17 clinical variables; categorical ones mapped to an
    ordinal scale, implausible readings dropped per :data:`VARIABLE_RANGES`,
    units harmonized to the variable's canonical unit) as a flat 34-value
    ``[mean_1..mean_17, std_1..std_17]`` feature vector. An unmeasured
    variable, or a variable with only one reading (undefined std), is
    :data:`MISSING` (``-1.0``); values are kept in raw clinical units
    (no z-score normalization), matching the GA2M model's expectation of
    an explicit "missing" bin rather than an imputed/normalized value.

    Attributes:
        task_name (str): The name of the task.
        input_schema (Dict[str, str]): ``{"features": "tensor"}``.
        output_schema (Dict[str, str]): ``{"mortality": "binary"}``.

    Examples:
        >>> from pyhealth.datasets import MIMIC3Dataset
        >>> from pyhealth.tasks import InHospitalMortality48hMIMIC3
        >>> dataset = MIMIC3Dataset(
        ...     root="/path/to/mimic-iii/1.4",
        ...     tables=["chartevents", "labevents"],
        ... )
        >>> task = InHospitalMortality48hMIMIC3()
        >>> samples = dataset.set_task(task)
    """

    task_name: str = "InHospitalMortality48hMIMIC3"
    input_schema: ClassVar[dict[str, str]] = {"features": "tensor"}
    output_schema: ClassVar[dict[str, str]] = {"mortality": "binary"}

    #: Hours of data observed from ICU admission.
    observation_window_hours: float = 48.0
    #: Minimum age (years) at ICU admission to be included.
    min_age_years: float = 18.0

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        demographics = patient.get_events(event_type="patients")
        if len(demographics) != 1:
            return []
        dob = _parse_datetime(demographics[0].dob)
        if dob is None:
            return []

        admissions = {
            _hadm_key(a.hadm_id): a for a in patient.get_events(event_type="admissions")
        }
        icustays = patient.get_events(event_type="icustays")

        samples = []
        for icustay in icustays:
            admission = admissions.get(_hadm_key(icustay.hadm_id))
            if admission is None or admission.hospital_expire_flag is None:
                continue
            intime = icustay.timestamp
            outtime = _parse_datetime(icustay.outtime)
            if intime is None or outtime is None:
                continue
            los_hours = (outtime - intime).total_seconds() / 3600.0
            if los_hours < self.observation_window_hours:
                continue
            age_years = (intime - dob).total_seconds() / (3600.0 * 24 * 365.25)
            if age_years < self.min_age_years:
                continue

            window_end = intime + timedelta(hours=self.observation_window_hours)
            values_by_variable: dict[str, list[float]] = {v: [] for v in VARIABLES}
            self._collect_chart_values(patient, intime, window_end, values_by_variable)
            self._collect_lab_values(patient, intime, window_end, values_by_variable)

            features: list[float] = []
            means: list[float] = []
            stds: list[float] = []
            for variable in VARIABLES:
                mean, std = _mean_std(values_by_variable[variable])
                means.append(mean)
                stds.append(std)
            features = means + stds

            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "icustay_id": icustay.icustay_id,
                    "hadm_id": icustay.hadm_id,
                    "features": features,
                    "mortality": int(admission.hospital_expire_flag),
                }
            )
        return samples

    @staticmethod
    def _filter_outliers(variable: str, value: float) -> float | None:
        bounds = VARIABLE_RANGES.get(variable)
        if bounds is None:
            return value
        low, high = bounds
        if value < low or value > high:
            return None
        return value

    def _collect_chart_values(
        self,
        patient: Any,
        start: datetime,
        end: datetime,
        values_by_variable: dict[str, list[float]],
    ) -> None:
        df = patient.get_events(
            event_type="chartevents", start=start, end=end, return_df=True
        )
        if df.height == 0:
            return
        df = df.with_columns(pl.col("chartevents/itemid").cast(pl.Utf8))
        df = df.filter(pl.col("chartevents/itemid").is_in(list(_CHART_ITEMID_VARIABLE)))
        if df.height == 0:
            return
        for row in df.to_dicts():
            itemid = row["chartevents/itemid"]
            variable = _CHART_ITEMID_VARIABLE[itemid]
            categorical = _CATEGORICAL_VARIABLES.get(variable)
            if categorical is not None:
                raw = row.get("chartevents/value")
                if raw is None:
                    continue
                value = categorical.get(str(raw).strip().lower())
                if value is None:
                    continue
            else:
                raw = row.get("chartevents/valuenum")
                if raw is None:
                    continue
                value = _CHART_ITEMID_UNIT[itemid](float(raw))
                rescale = _RESCALE_FNS.get(variable)
                if rescale is not None:
                    value = rescale(value)
            value = self._filter_outliers(variable, value)
            if value is not None:
                values_by_variable[variable].append(value)

    def _collect_lab_values(
        self,
        patient: Any,
        start: datetime,
        end: datetime,
        values_by_variable: dict[str, list[float]],
    ) -> None:
        if not _LAB_ITEMID_VARIABLE:
            return
        df = patient.get_events(
            event_type="labevents", start=start, end=end, return_df=True
        )
        if df.height == 0:
            return
        df = df.with_columns(pl.col("labevents/itemid").cast(pl.Utf8))
        df = df.filter(pl.col("labevents/itemid").is_in(list(_LAB_ITEMID_VARIABLE)))
        if df.height == 0:
            return
        for row in df.to_dicts():
            itemid = row["labevents/itemid"]
            variable = _LAB_ITEMID_VARIABLE[itemid]
            raw = row.get("labevents/valuenum")
            if raw is None:
                continue
            value = float(raw)
            rescale = _RESCALE_FNS.get(variable)
            if rescale is not None:
                value = rescale(value)
            value = self._filter_outliers(variable, value)
            if value is not None:
                values_by_variable[variable].append(value)
