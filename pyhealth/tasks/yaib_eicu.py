"""YAIB benchmark tasks for eICU-CRD.

Re-implements the eICU-CRD cohorts of YAIB (Yet Another ICU Benchmark,
van de Water et al., ICLR 2024, https://github.com/rvandewater/YAIB) on top of
the PyHealth task API. YAIB builds its cohorts with the ``ricu`` R package
(https://github.com/eth-mds/ricu) and the YAIB-cohorts scripts
(https://github.com/rvandewater/YAIB-cohorts); the item names, unit
conversions, plausibility bounds and exclusion rules below are ported from
those sources.

All tasks share the YAIB feature set: 4 static variables (age, sex, height,
weight) and 48 dynamic variables binned hourly from ICU admission.

eICU-CRD is centered on the ICU stay: every table carries the ICU stay ID
(``patientunitstayid``) and times are offsets in minutes from ICU admission.
"""

import itertools
from typing import Any, ClassVar, NamedTuple

import numpy as np
import polars as pl

from ..data import Patient
from ..processors import Records
from .base_task import BaseTask


class _Item(NamedTuple):
    """Rows of one eICU table that feed a feature (a ricu concept item).

    ``selector`` names the column that selects rows (``None`` for a whole
    value column); ``values`` are matched exactly and ``regex`` is matched
    case-insensitively. ``scale``/``offset`` convert a value as
    ``value * scale + offset``; for a lab item this is applied only to rows
    whose unit matches ``unit_regex``, and rows whose unit matches
    ``drop_unit_regex`` are discarded. ``strip_percent`` removes a ``%`` sign
    before the value is parsed.
    """

    table: str
    selector: str | None = None
    values: tuple[str, ...] = ()
    regex: str | None = None
    value_column: str = ""
    unit_column: str | None = None
    unit_regex: str | None = None
    scale: float = 1.0
    offset: float = 0.0
    drop_unit_regex: str | None = None
    strip_percent: bool = False


def _lab(
    *names: str,
    unit_regex: str | None = None,
    scale: float = 1.0,
    offset: float = 0.0,
    drop_unit_regex: str | None = None,
) -> _Item:
    return _Item(
        "lab",
        "labname",
        tuple(names),
        None,
        "labresult",
        "labmeasurenameinterface",
        unit_regex,
        scale,
        offset,
        drop_unit_regex,
    )


def _selects(item: _Item, column: str) -> pl.Expr:
    """Whether a row of ``item.table`` is one of the item's rows.

    ``column`` is the selector column, for example ``"lab/labname"``.
    """
    selected = pl.lit(False)
    if item.values:
        selected = selected | pl.col(column).is_in(list(item.values))
    if item.regex:
        selected = selected | pl.col(column).str.contains("(?i)" + item.regex).fill_null(
            False
        )
    return selected


def _vital(table: str, column: str) -> _Item:
    return _Item(table, None, (), None, column)


def _intake(*labels: str, regex: str | None = None) -> _Item:
    return _Item("intakeoutput", "celllabel", tuple(labels), regex, "cellvaluenumeric")


def _resp(*labels: str) -> _Item:
    return _Item(
        "respiratorycharting",
        "respchartvaluelabel",
        tuple(labels),
        None,
        "respchartvalue",
        strip_percent=True,
    )


def _nurse(*labels: str, scale: float = 1.0, offset: float = 0.0) -> _Item:
    return _Item(
        "nursecharting",
        "nursingchartcelltypevalname",
        tuple(labels),
        None,
        "nursingchartvalue",
        scale=scale,
        offset=offset,
    )


# feature: (min, max, items). Out-of-range values are dropped, as in ricu.
_DYNAMIC_FEATURES: dict[str, tuple[float | None, float | None, list[_Item]]] = {
    "alb": (0, 6, [_lab("albumin")]),
    "alp": (0, None, [_lab("alkaline phos.")]),
    "alt": (0, None, [_lab("ALT (SGPT)")]),
    "ast": (0, None, [_lab("AST (SGOT)")]),
    "be": (-25, 25, [_lab("Base Excess")]),
    "bicar": (5, 50, [_lab("bicarbonate")]),
    "bili": (0, 100, [_lab("total bilirubin")]),
    "bili_dir": (0, 50, [_lab("direct bilirubin")]),
    "bnd": (None, None, [_lab("-bands")]),
    "bun": (0, 200, [_lab("BUN")]),
    "ca": (4, 20, [_lab("calcium", unit_regex="mmol/l", scale=4.0)]),
    "cai": (0.5, 2, [_lab("ionized calcium")]),
    "ck": (0, None, [_lab("CPK")]),
    "ckmb": (0, None, [_lab("CPK-MB")]),
    "cl": (80, 130, [_lab("chloride")]),
    "crea": (0, 15, [_lab("creatinine")]),
    "crp": (0, None, [_lab("CRP", unit_regex="mg/dl", scale=10.0)]),
    "dbp": (
        0,
        200,
        [
            _vital("vitalperiodic", "systemicdiastolic"),
            _vital("vitalaperiodic", "noninvasivediastolic"),
        ],
    ),
    "fgn": (0, 1500, [_lab("fibrinogen")]),
    "fio2": (
        21,
        100,
        [
            _resp("FiO2"),
            _lab("FiO2", drop_unit_regex=r"mm\(hg\)"),
        ],
    ),
    "glu": (0, 1000, [_lab("glucose"), _lab("bedside glucose")]),
    "hgb": (4, 18, [_lab("Hgb")]),
    "hr": (0, 300, [_vital("vitalperiodic", "heartrate")]),
    "inr_pt": (None, None, [_lab("PT - INR")]),
    "k": (0, 10, [_lab("potassium")]),
    "lact": (0, 50, [_lab("lactate")]),
    "lymph": (0, 100, [_lab("-lymphs")]),
    "map": (
        0,
        250,
        [
            _vital("vitalperiodic", "systemicmean"),
            _vital("vitalaperiodic", "noninvasivemean"),
        ],
    ),
    "mch": (0, None, [_lab("MCH")]),
    "mchc": (20, 50, [_lab("MCHC")]),
    "mcv": (50, 150, [_lab("MCV")]),
    "methb": (0, 100, [_lab("Methemoglobin")]),
    "mg": (0.5, 5, [_lab("magnesium", unit_regex="mEq/L", scale=1 / 1.215)]),
    "na": (110, 165, [_lab("sodium")]),
    "neut": (0, 100, [_lab("-polys")]),
    "o2sat": (
        50,
        100,
        [
            _vital("vitalperiodic", "sao2"),
            _lab("O2 Sat (%)"),
        ],
    ),
    "pco2": (10, 150, [_lab("paCO2")]),
    "ph": (6.8, 8, [_lab("pH")]),
    "phos": (0, 40, [_lab("phosphate")]),
    "plt": (5, 1200, [_lab("platelets x 1000")]),
    "po2": (40, 600, [_lab("paO2")]),
    "ptt": (0, None, [_lab("PTT")]),
    "resp": (
        0,
        120,
        [
            _vital("vitalperiodic", "respiration"),
            _nurse("Respiratory Rate"),
        ],
    ),
    "sbp": (
        0,
        300,
        [
            _vital("vitalperiodic", "systemicsystolic"),
            _vital("vitalaperiodic", "noninvasivesystolic"),
        ],
    ),
    "temp": (
        32,
        42,
        [
            _vital("vitalperiodic", "temperature"),
            _nurse("Temperature (C)"),
            _nurse("Temperature (F)", scale=5 / 9, offset=-160 / 9),
        ],
    ),
    "tnt": (0, None, [_lab("troponin - T")]),
    "urine": (
        0,
        2000,
        [
            _intake("Urine", "URINE CATHETER"),
            _intake(regex=r"catheter.+output|output.+catheter"),
        ],
    ),
    "wbc": (0, None, [_lab("WBC x 1000")]),
}

_STATIC_FEATURES = ("age", "sex", "height", "weight")

# Features summed (instead of median-aggregated) within an hourly bin.
_SUM_FEATURES = ("urine",)

_TABLE_OFFSETS = {
    "lab": "labresultoffset",
    "vitalperiodic": "observationoffset",
    "vitalaperiodic": "observationoffset",
    "intakeoutput": "intakeoutputoffset",
    "respiratorycharting": "respchartoffset",
    "nursecharting": "nursingchartoffset",
}
_MEASUREMENT_TABLES = tuple(_TABLE_OFFSETS)

_MEASUREMENT_SCHEMA = {
    "stay_id": pl.Int64,
    "hour": pl.Int64,
    "feature": pl.Int64,
    "value": pl.Float64,
    "is_sum": pl.Boolean,
}


def _parse_number(value: Any) -> float:
    """as.numeric for a scalar, NaN when it is missing or not a number."""
    if value is None:
        return float("nan")
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return float("nan")


def _eicu_age(value: Any) -> float:
    """ricu eicu_age: "> 89" becomes 90, other ages are numeric."""
    if isinstance(value, str) and value.strip() == "> 89":
        return 90.0
    age = _parse_number(value)
    return age if 0 <= age <= 100 else float("nan")


def _longest_false_run(values: np.ndarray) -> int:
    longest = current = 0
    for value in values:
        current = 0 if value else current + 1
        longest = max(longest, current)
    return longest


class _YAIBEICUTask(BaseTask):
    """Shared YAIB base cohort and feature extraction for eICU-CRD.

    Base cohort (YAIB-cohorts base_cohort.R), applied per ICU stay:

    1. ICU length of stay is negative or missing.
    2. ICU length of stay is shorter than 6 hours.
    3. Fewer than 4 hourly bins with any dynamic measurement (stays with no
       measurement at all are not removed by this rule, matching YAIB).
    4. A gap of more than 12 consecutive hourly bins without any dynamic
       measurement within the first 7 days of the stay.
    5. Age below 18.

    Age comes from the ``patient`` table, where eICU reports ages above 89 as
    ``"> 89"``; as in ricu, these become 90. Height (10 to 230 cm) and weight
    (1 to 500 kg) are the admission height and weight.

    Dynamic features are binned at ``floor(offset / 60)``, the hours since
    ICU admission. Every eICU table carries the ICU stay ID, so measurements
    need no attribution to stays. Samples are keyed by ``stay_id``, the eICU
    ``patientunitstayid``.

    Each sample holds ``dynamic``, a ``sequence_v2`` series of the hourly
    bins (``{"time": [0, 1, ..., T], "values": [T + 1, 48]}``, time in hours
    since ICU admission, ``NaN`` for bins without measurement), and
    ``static``, a ``sequence_v2`` series with one record at hour 0, known
    before any hourly bin (``{"time": [0], "values": [[age, sex, height,
    weight]]}``, sex encoded as 1 for male and 0 for female, ``NaN`` when
    unavailable). Per-hour labels are ``timeseries_*`` labels on the same
    hours (``{"time": [...], "values": [...]}``), so a model predicts at hour
    ``h`` from the bins up to ``h``. Missing values are left to the
    processors, which apply YAIB's preprocessing: the dynamic features are
    forward-filled, with values before the first measurement set to the
    training mean (0 once standardized), standardized, and get a
    missing-value mask; the static features get a missing-value mask, are
    standardized and set to the training mean (0 once standardized) where
    missing. The statistics are learned when the processors are fitted, so
    use ``set_task(task, split=...)`` to learn them on the training patients
    only. Unlike YAIB, which label-encodes sex as a category, sex is
    standardized like the other static features.

    Training needs a model that reads the ``sequence_v2`` input and the
    per-hour label records, such as :class:`~pyhealth.models.RNN_V2`;
    :class:`~pyhealth.models.RNN` and :class:`~pyhealth.models.MultimodalRNN`
    do not.

    Unlike on MIMIC, ricu's eICU definition of ``o2sat`` already combines
    pulse oximetry (``vitalperiodic``) with arterial saturation (``lab``), so
    no deviation is needed there. YAIB adds the nurse-charted respiratory
    rate to ``resp`` for the full eICU only, not for the eICU demo; this task
    follows the full-eICU definition.

    Requires the ``lab``, ``vitalperiodic``, ``vitalaperiodic``,
    ``intakeoutput``, ``respiratorycharting`` and ``nursecharting`` tables.
    """

    input_schema: ClassVar[dict[str, Any]] = {
        # Forward fill, then the training mean (0 once standardized) before
        # the first measurement, then 0 for a feature never seen in training
        # (normalization leaves it as is): YAIB's preprocessing, with the
        # missing-value indicators as the mask.
        "dynamic": (
            "sequence_v2",
            {
                "levels": [
                    {
                        "ordering": "series",
                        "values": [len(_DYNAMIC_FEATURES)],
                        "impute": [
                            {"method": "forward_fill"},
                            {"method": "mean", "by_dims": ()},  # per feature
                            {"method": "zero"},
                        ],
                        "normalize": {"method": "zscore"},
                        "missing_indicator": True,
                    }
                ]
            },
        ),
        # The training mean (0 once standardized), then 0 for a feature never
        # seen in training: YAIB's standardize-then-zero-fill, with the
        # missing-value indicators as the mask.
        "static": (
            "sequence_v2",
            {
                "levels": [
                    {
                        "ordering": "series",
                        "values": [len(_STATIC_FEATURES)],
                        "impute": [
                            {"method": "mean", "by_dims": ()},  # per feature
                            {"method": "zero"},
                        ],
                        "normalize": {"method": "zscore"},
                        "missing_indicator": True,
                    }
                ]
            },
        ),
    }

    DYNAMIC_FEATURES: ClassVar[list[str]] = list(_DYNAMIC_FEATURES)
    STATIC_FEATURES: ClassVar[list[str]] = list(_STATIC_FEATURES)

    # Tables a task needs beyond the measurement tables.
    _EXTRA_TABLES: ClassVar[tuple[str, ...]] = ()
    # Tables read to find the stays with the outcome, for the eICU hospital
    # prevalence exclusion of YAIB-cohorts; empty for tasks without it.
    _PREVALENCE_TABLES: ClassVar[tuple[str, ...]] = ()
    # Patients read at once when finding the stays with the outcome.
    _PREVALENCE_BATCH_SIZE: ClassVar[int] = 1024

    def _features_read(
        self,
    ) -> list[dict[str, tuple[float | None, float | None, list[_Item]]]]:
        """The feature definitions whose rows the task reads."""
        return [_DYNAMIC_FEATURES]

    def _extra_rows(self) -> dict[str, pl.Expr]:
        """Rows a task reads beyond its features, by table."""
        return {}

    def pre_filter(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Checks the tables, drops unused rows and low-prevalence hospitals.

        Rows of a measurement table that no feature selects (for example
        unused lab names) are dropped. For tasks with a hospital prevalence
        exclusion, the stays of hospitals where no stay has the outcome are
        dropped too.
        """
        columns = df.collect_schema().names()
        missing = [
            table
            for table in (*_MEASUREMENT_TABLES, *self._EXTRA_TABLES)
            if not any(column.startswith(f"{table}/") for column in columns)
        ]
        if missing:
            raise ValueError(
                f"{self.task_name} requires the eICU tables {missing}; "
                "add them to the dataset's `tables`."
            )

        selected: dict[str, list[pl.Expr]] = {}
        read_all: set[str] = set()  # tables with a whole value column read
        for features in self._features_read():
            for _, _, items in features.values():
                for item in items:
                    if item.selector is None:
                        read_all.add(item.table)
                    else:
                        selected.setdefault(item.table, []).append(
                            _selects(item, f"{item.table}/{item.selector}")
                        )
        for table, rows in self._extra_rows().items():
            selected.setdefault(table, []).append(rows)
        filters = {
            table: pl.any_horizontal(masks)
            for table, masks in selected.items()
            if table not in read_all
        }
        keep = ~pl.col("event_type").is_in(list(filters))
        for table, rows in filters.items():
            keep = keep | ((pl.col("event_type") == table) & rows)
        df = df.filter(keep)

        if self._PREVALENCE_TABLES:
            df = self._drop_stays(df, self._low_prevalence_stays(df))
        return df

    def _has_outcome(self, patient: Any, stays: pl.DataFrame) -> dict[int, bool]:
        """Whether each stay has the outcome at any time, for prevalence."""
        raise NotImplementedError

    def _low_prevalence_stays(self, df: pl.LazyFrame) -> set[int]:
        """Stays of eICU hospitals where no stay has the outcome.

        As in YAIB-cohorts ``aki.R`` and ``sepsis.R`` (eICU only), the
        prevalence counts every ICU stay of the dataset, before any
        exclusion. Patients are read in batches to bound memory.
        """
        events = df.filter(
            pl.col("event_type").is_in(["patient", *self._PREVALENCE_TABLES])
        )
        patient_ids = (
            events.filter(pl.col("event_type") == "patient")
            .select("patient_id")
            .unique()
            .collect(engine="streaming")
            .to_series()
            .sort()
        )
        hospitals: dict[int, int] = {}
        with_outcome: set[int] = set()
        for batch in itertools.batched(patient_ids, self._PREVALENCE_BATCH_SIZE):
            patients = (
                events.filter(pl.col("patient_id").is_in(batch))
                .collect(engine="streaming")
                .partition_by("patient_id", as_dict=True)
            )
            for (patient_id,), patient_df in patients.items():
                patient = Patient(patient_id=patient_id, data_source=patient_df)
                stays = self._stays(patient)
                outcome = self._has_outcome(patient, stays)
                for stay_id, hospital_id in stays.select(
                    "stay_id", "hospital_id"
                ).iter_rows():
                    if hospital_id is None:
                        continue
                    hospitals[stay_id] = hospital_id
                    if outcome.get(stay_id, False):
                        with_outcome.add(hospital_id)
        return {
            stay_id
            for stay_id, hospital_id in hospitals.items()
            if hospital_id not in with_outcome
        }

    @staticmethod
    def _drop_stays(df: pl.LazyFrame, excluded: set[int]) -> pl.LazyFrame:
        """Drops the events of the excluded ICU stays."""
        if not excluded:
            return df
        stay_columns = [
            column
            for column in df.collect_schema().names()
            if column.endswith("/patientunitstayid")
        ]
        stay = pl.coalesce(
            [pl.col(column).cast(pl.Int64, strict=False) for column in stay_columns]
        )
        return df.filter(stay.is_null() | ~stay.is_in(sorted(excluded)))

    def _stays(self, patient: Any) -> pl.DataFrame:
        stays = patient.get_events(event_type="patient", return_df=True)
        if stays.height == 0:
            return pl.DataFrame(
                schema={
                    "stay_id": pl.Int64,
                    "los_minutes": pl.Float64,
                    "los_hours": pl.Float64,
                    "end": pl.Int64,
                    "hospital_id": pl.Int64,
                    "status": pl.String,
                }
            )
        return (
            stays.select(
                pl.col("patient/patientunitstayid").cast(pl.Int64).alias("stay_id"),
                pl.col("patient/unitdischargeoffset")
                .cast(pl.Float64, strict=False)
                .alias("los_minutes"),
                pl.col("patient/hospitalid").cast(pl.Int64).alias("hospital_id"),
                pl.col("patient/unitdischargestatus").alias("status"),
            )
            .drop_nulls(["stay_id", "los_minutes"])
            .with_columns((pl.col("los_minutes") / 60).alias("los_hours"))
            .with_columns(pl.col("los_hours").floor().cast(pl.Int64).alias("end"))
        )

    def _static(self, patient: Any, stays: pl.DataFrame) -> dict[int, list[float]]:
        events = patient.get_events(event_type="patient", return_df=True)
        static = {stay_id: [np.nan, np.nan, np.nan, np.nan] for stay_id in stays["stay_id"]}
        if events.height == 0:
            return static
        for stay_id, gender, age, height, weight in events.select(
            pl.col("patient/patientunitstayid").cast(pl.Int64),
            pl.col("patient/gender"),
            pl.col("patient/age"),
            pl.col("patient/admissionheight").cast(pl.Float64, strict=False),
            pl.col("patient/admissionweight").cast(pl.Float64, strict=False),
        ).iter_rows():
            if stay_id not in static:
                continue
            sex = {"Male": 1.0, "Female": 0.0}.get(str(gender).strip(), np.nan)
            static[stay_id] = [
                _eicu_age(age),
                sex,
                height if height is not None and 10 <= height <= 230 else np.nan,
                weight if weight is not None and 1 <= weight <= 500 else np.nan,
            ]
        return static

    @staticmethod
    def _lab_frame(events: pl.DataFrame, rows: list[dict[str, Any]]) -> pl.DataFrame:
        if not rows:
            return pl.DataFrame()
        lookup = pl.DataFrame(
            rows,
            schema={
                "selector": pl.String,
                "feature": pl.Int64,
                "lower": pl.Float64,
                "upper": pl.Float64,
                "is_sum": pl.Boolean,
                "unit_regex": pl.String,
                "conv_scale": pl.Float64,
                "conv_offset": pl.Float64,
                "drop_unit_regex": pl.String,
            },
        )
        return (
            events.select(
                pl.col("lab/patientunitstayid").cast(pl.Int64).alias("stay_id"),
                pl.col("lab/labresultoffset").cast(pl.Float64).alias("offset"),
                pl.col("lab/labname").alias("selector"),
                pl.col("lab/labresult").cast(pl.Float64, strict=False).alias("value"),
                pl.col("lab/labmeasurenameinterface").alias("unit"),
            )
            .join(lookup, on="selector", how="inner")
            .with_columns(
                pl.when(
                    pl.col("drop_unit_regex").is_not_null()
                    & pl.col("unit")
                    .str.contains(pl.col("drop_unit_regex"), literal=False)
                    .fill_null(False)
                )
                .then(pl.lit(None, dtype=pl.Float64))
                .when(
                    pl.col("unit_regex").is_not_null()
                    & pl.col("unit")
                    .str.contains(pl.col("unit_regex"), literal=False)
                    .fill_null(False)
                )
                .then(pl.col("value") * pl.col("conv_scale") + pl.col("conv_offset"))
                .otherwise(pl.col("value"))
                .alias("value")
            )
        )

    def _measurements(
        self,
        patient: Any,
        stays: pl.DataFrame,
        features: dict[str, tuple[float | None, float | None, list[_Item]]],
    ) -> pl.DataFrame:
        """Returns valid measurements as (stay_id, hour, feature, value, is_sum)."""
        names = list(features)
        frames: list[pl.DataFrame] = []

        for table in _MEASUREMENT_TABLES:
            table_items = [
                (index, name, item)
                for index, name in enumerate(names)
                for item in features[name][2]
                if item.table == table
            ]
            if not table_items:
                continue
            events = patient.get_events(event_type=table, return_df=True)
            if events.height == 0:
                continue

            stay_expr = pl.col(f"{table}/patientunitstayid").cast(pl.Int64).alias("stay_id")
            offset_expr = pl.col(f"{table}/{_TABLE_OFFSETS[table]}").cast(
                pl.Float64, strict=False
            ).alias("offset")

            if table == "lab":
                rows = [
                    {
                        "selector": value,
                        "feature": index,
                        "lower": features[name][0],
                        "upper": features[name][1],
                        "is_sum": name in _SUM_FEATURES,
                        "unit_regex": None if item.unit_regex is None else "(?i)" + item.unit_regex,
                        "conv_scale": float(item.scale),
                        "conv_offset": float(item.offset),
                        "drop_unit_regex": (
                            None if item.drop_unit_regex is None else "(?i)" + item.drop_unit_regex
                        ),
                    }
                    for index, name, item in table_items
                    for value in item.values
                ]
                frame = self._lab_frame(events, rows)
                if frame.height:
                    frames.append(frame)
                continue

            for index, name, item in table_items:
                lower, upper, _ = features[name]
                value = pl.col(f"{table}/{item.value_column}")
                if item.strip_percent:
                    value = value.cast(pl.String).str.replace("%", "", literal=True)
                value = value.cast(pl.Float64, strict=False) * item.scale + item.offset
                rows = events
                if item.selector is not None:
                    rows = rows.filter(_selects(item, f"{table}/{item.selector}"))
                frame = rows.select(
                    stay_expr,
                    offset_expr,
                    value.alias("value"),
                    pl.lit(index, dtype=pl.Int64).alias("feature"),
                    pl.lit(lower, dtype=pl.Float64).alias("lower"),
                    pl.lit(upper, dtype=pl.Float64).alias("upper"),
                    pl.lit(name in _SUM_FEATURES).alias("is_sum"),
                )
                if frame.height:
                    frames.append(frame)

        if not frames:
            return pl.DataFrame(schema=_MEASUREMENT_SCHEMA)

        return (
            pl.concat(frames, how="diagonal_relaxed")
            .join(stays.select("stay_id"), on="stay_id", how="inner")
            .with_columns((pl.col("offset") / 60).floor().cast(pl.Int64).alias("hour"))
            .filter(
                pl.col("value").is_not_null()
                & ~pl.col("value").is_nan()
                & (pl.col("lower").is_null() | (pl.col("value") >= pl.col("lower")))
                & (pl.col("upper").is_null() | (pl.col("value") <= pl.col("upper")))
            )
            .select(list(_MEASUREMENT_SCHEMA))
        )

    @staticmethod
    def _hourly(measurements: pl.DataFrame) -> pl.DataFrame:
        """Aggregates measurements per (stay_id, hour, feature)."""
        return measurements.group_by("stay_id", "hour", "feature").agg(
            pl.when(pl.col("is_sum").first())
            .then(pl.col("value").sum())
            .otherwise(pl.col("value").median())
            .alias("value")
        )

    def _base_cohort(
        self, patient: Any, stays: pl.DataFrame
    ) -> tuple[pl.DataFrame, pl.DataFrame, dict[int, list[float]]]:
        """Applies the YAIB base cohort exclusions."""
        static = self._static(patient, stays)
        # Measurements are attributed using all stays, before any exclusion.
        hourly = self._hourly(self._measurements(patient, stays, _DYNAMIC_FEATURES))
        observed = hourly.select("stay_id", "hour").unique()

        keep = []
        for stay_id, los_hours, end in stays.select(
            "stay_id", "los_hours", "end"
        ).iter_rows():
            # Exclusions 1, 2 and 5.
            if end < 0 or los_hours < 6 or static[stay_id][0] < 18:
                continue
            hours = observed.filter(pl.col("stay_id") == stay_id)["hour"].to_numpy()
            # Exclusion 3: counted over all bins, including those outside the stay.
            if 0 < len(hours) < 4:
                continue
            # Exclusion 4: gaps are measured on the first 7 days of the stay.
            grid = np.zeros(min(end, 7 * 24) + 1, dtype=bool)
            in_grid = hours[(hours >= 0) & (hours < len(grid))]
            grid[in_grid] = True
            if _longest_false_run(grid) > 12:
                continue
            keep.append(stay_id)

        stays = stays.filter(pl.col("stay_id").is_in(keep))
        hourly = hourly.filter(pl.col("stay_id").is_in(keep))
        return stays, hourly, {stay_id: static[stay_id] for stay_id in keep}

    def _dynamic(self, hourly: pl.DataFrame, stay_id: int, end: int) -> dict[str, Any]:
        """Builds the series of hourly bins 0..end: [end + 1, 48] values."""
        values = np.full((end + 1, len(self.DYNAMIC_FEATURES)), np.nan)
        rows = hourly.filter(
            (pl.col("stay_id") == stay_id)
            & (pl.col("hour") >= 0)
            & (pl.col("hour") <= end)
        )
        values[rows["hour"].to_numpy(), rows["feature"].to_numpy()] = rows["value"].to_numpy()
        return {"time": [float(hour) for hour in range(end + 1)], "values": values}

    @staticmethod
    def _static_series(values: list[float]) -> Records:
        """The static features as one record at hour 0, before any bin."""
        return {"time": [0.0], "values": [values]}

    @staticmethod
    def _labels(time: list[float], labels: list[float]) -> dict[str, Any]:
        """A per-hour label series on the hours of dynamic."""
        return {"time": time, "values": [[label] for label in labels]}

    @staticmethod
    def _icu_deaths(stays: pl.DataFrame) -> dict[int, float]:
        """Returns the ICU death time in hours since ICU admission per stay.

        eICU records the outcome at unit discharge: a stay whose
        ``unitdischargestatus`` is ``"Expired"`` is an ICU death at
        ``unitdischargeoffset``.
        """
        return {
            stay_id: los_hours
            for stay_id, status, los_hours in stays.select(
                "stay_id", "status", "los_hours"
            ).iter_rows()
            if status is not None and str(status).strip() == "Expired"
        }

    @staticmethod
    def _series(
        hourly: pl.DataFrame, stay_id: int, index: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns the measured hours and values of one feature, by hour.

        ``index`` is the feature's position in the features ``hourly`` was
        built from. Unlike ``_dynamic``, this includes hours outside the stay.
        """
        rows = hourly.filter(
            (pl.col("stay_id") == stay_id) & (pl.col("feature") == index)
        ).sort("hour")
        return rows["hour"].to_numpy(), rows["value"].to_numpy()

    @staticmethod
    def _window_starts(hours: np.ndarray, duration: int) -> np.ndarray:
        """Index of the first row in the window (hour - duration, hour].

        As ricu's slide(..., left_closed = FALSE) over measured hours.
        """
        return np.searchsorted(hours, hours - duration, side="right")

    @classmethod
    def _window_min(cls, hours: np.ndarray, values: np.ndarray, duration: int) -> np.ndarray:
        starts = cls._window_starts(hours, duration)
        return np.array([values[start : i + 1].min() for i, start in enumerate(starts)])

    @classmethod
    def _window_sum(cls, hours: np.ndarray, values: np.ndarray, duration: int) -> np.ndarray:
        cumulative = np.concatenate([[0.0], np.cumsum(values)])
        starts = cls._window_starts(hours, duration)
        return cumulative[np.arange(1, len(values) + 1)] - cumulative[starts]

    @staticmethod
    def _outcome_window(onset: int | None, steps: int, before: int, after: int) -> list[float]:
        """Labels 1 for steps onset - before to onset + after, else 0.

        As YAIB-cohorts outcome_window; an onset outside the steps observed
        steps labels nothing.
        """
        labels = [0.0] * steps
        if onset is not None and 0 <= onset < steps:
            for step in range(max(0, onset - before), min(steps, onset + after + 1)):
                labels[step] = 1.0
        return labels

class YAIBMortalityEICU(_YAIBEICUTask):
    """YAIB ICU mortality prediction after 24 hours on eICU-CRD.

    Works with :class:`~pyhealth.datasets.eICUDataset`.

    Predicts death in the ICU from the first 24 hours of the stay (hourly bins
    0 to 24 inclusive, as in YAIB). On top of the YAIB base cohort, stays are
    excluded if the patient died within 30 hours of ICU admission or the ICU
    stay is shorter than 30 hours.

    A stay whose unitdischargestatus is "Expired" is an ICU death at
    unitdischargeoffset.

    Requires the lab, vitalperiodic, vitalaperiodic, intakeoutput,
    respiratorycharting and nursecharting tables.

    Examples:
        >>> from pyhealth.datasets import eICUDataset
        >>> from pyhealth.tasks import YAIBMortalityEICU
        >>> dataset = eICUDataset(
        ...     root="/path/to/eicu-crd/2.0",
        ...     tables=["lab", "vitalperiodic", "vitalaperiodic",
        ...             "intakeoutput", "respiratorycharting",
        ...             "nursecharting"],
        ... )
        >>> samples = dataset.set_task(YAIBMortalityEICU())
    """

    task_name: str = "YAIBMortalityEICU"
    output_schema: ClassVar[dict[str, Any]] = {"mortality": "binary"}

    OBSERVATION_HOURS: ClassVar[int] = 24
    MIN_STAY_HOURS: ClassVar[int] = 30

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        all_stays = self._stays(patient)
        if all_stays.height == 0:
            return []
        stays, hourly, static = self._base_cohort(patient, all_stays)
        deaths = self._icu_deaths(all_stays)

        samples = []
        for stay_id, los_hours, end in stays.select(
            "stay_id", "los_hours", "end"
        ).iter_rows():
            if los_hours < self.MIN_STAY_HOURS:
                continue
            death = deaths.get(stay_id)
            if death is not None and death < self.MIN_STAY_HOURS:
                continue
            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "stay_id": stay_id,
                    "dynamic": self._dynamic(hourly, stay_id, min(end, self.OBSERVATION_HOURS)),
                    "static": self._static_series(static[stay_id]),
                    "mortality": int(death is not None),
                }
            )
        return samples


class YAIBKidneyFunctionEICU(_YAIBEICUTask):
    """YAIB kidney function prediction on eICU-CRD.

    Works with :class:`~pyhealth.datasets.eICUDataset`.

    Predicts the median creatinine (mg/dL) on the second day of the ICU stay
    from the first 24 hours (hourly bins 0 to 24 inclusive, as in YAIB). As in
    YAIB, the label is the median of the hourly creatinine bins after hour 24
    up to and including hour 48 (bins 25 to 48), where each bin is itself the
    median of its measurements. On top of the YAIB base cohort, stays are
    excluded if the ICU stay is shorter than 48 hours or has no creatinine in
    that window.

    As in YAIB, the label is min-max scaled to 0..1 on the range of the
    training labels (learned when the processors are fitted, so use
    set_task(task, split=...); other splits are not clipped). The loss is
    on the scaled label, and predictions and metrics are in mg/dL.

    Requires the lab, vitalperiodic, vitalaperiodic, intakeoutput,
    respiratorycharting and nursecharting tables.

    Examples:
        >>> from pyhealth.datasets import eICUDataset
        >>> from pyhealth.tasks import YAIBKidneyFunctionEICU
        >>> dataset = eICUDataset(
        ...     root="/path/to/eicu-crd/2.0",
        ...     tables=["lab", "vitalperiodic", "vitalaperiodic",
        ...             "intakeoutput", "respiratorycharting",
        ...             "nursecharting"],
        ... )
        >>> samples = dataset.set_task(YAIBKidneyFunctionEICU())
    """

    task_name: str = "YAIBKidneyFunctionEICU"
    output_schema: ClassVar[dict[str, Any]] = {
        "creatinine": ("regression", {"normalization": "minmax"})
    }

    OBSERVATION_HOURS: ClassVar[int] = 24
    MIN_STAY_HOURS: ClassVar[int] = 48

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        all_stays = self._stays(patient)
        if all_stays.height == 0:
            return []
        stays, hourly, static = self._base_cohort(patient, all_stays)
        labels = dict(
            hourly.filter(
                (pl.col("feature") == self.DYNAMIC_FEATURES.index("crea"))
                & (pl.col("hour") > self.OBSERVATION_HOURS)
                & (pl.col("hour") <= self.MIN_STAY_HOURS)
            )
            .group_by("stay_id")
            .agg(pl.col("value").median())
            .iter_rows()
        )

        samples = []
        for stay_id, los_hours, end in stays.select(
            "stay_id", "los_hours", "end"
        ).iter_rows():
            if los_hours < self.MIN_STAY_HOURS or stay_id not in labels:
                continue
            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "stay_id": stay_id,
                    "dynamic": self._dynamic(hourly, stay_id, min(end, self.OBSERVATION_HOURS)),
                    "static": self._static_series(static[stay_id]),
                    "creatinine": labels[stay_id],
                }
            )
        return samples


class YAIBLengthOfStayEICU(_YAIBEICUTask):
    """YAIB remaining length of stay prediction on eICU-CRD.

    Works with :class:`~pyhealth.datasets.eICUDataset`.

    Predicts, at every hour of the ICU stay, the remaining hours until ICU
    discharge, capped at 7 days. As in YAIB, the stay is observed for at most
    7 days (hourly bins 0 to min(end, 168), where end is the ICU length of
    stay in whole hours), and the label at hour h is min(168, end - h). The
    cohort is the YAIB base cohort without further exclusions.

    The label is a per-hour series on the hours of dynamic
    (timeseries_regression), so models predict at every hour. As in YAIB, it
    is min-max scaled to 0..1 on the range of all hourly training labels
    (learned when the processors are fitted, so use set_task(task,
    split=...); other splits are not clipped). The loss is on the scaled
    label, and predictions and metrics are in hours.

    Requires the lab, vitalperiodic, vitalaperiodic, intakeoutput,
    respiratorycharting and nursecharting tables.

    Examples:
        >>> from pyhealth.datasets import eICUDataset
        >>> from pyhealth.tasks import YAIBLengthOfStayEICU
        >>> dataset = eICUDataset(
        ...     root="/path/to/eicu-crd/2.0",
        ...     tables=["lab", "vitalperiodic", "vitalaperiodic",
        ...             "intakeoutput", "respiratorycharting",
        ...             "nursecharting"],
        ... )
        >>> samples = dataset.set_task(YAIBLengthOfStayEICU())
    """

    task_name: str = "YAIBLengthOfStayEICU"
    output_schema: ClassVar[dict[str, Any]] = {
        "los": ("timeseries_regression", {"normalization": "minmax"})
    }

    MAX_HOURS: ClassVar[int] = 7 * 24

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        all_stays = self._stays(patient)
        if all_stays.height == 0:
            return []
        stays, hourly, static = self._base_cohort(patient, all_stays)

        samples = []
        for stay_id, end in stays.select("stay_id", "end").iter_rows():
            dynamic = self._dynamic(hourly, stay_id, min(end, self.MAX_HOURS))
            remaining = [
                float(min(self.MAX_HOURS, end - hour)) for hour in range(len(dynamic["time"]))
            ]
            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "stay_id": stay_id,
                    "dynamic": dynamic,
                    "static": self._static_series(static[stay_id]),
                    "los": self._labels(dynamic["time"], remaining),
                }
            )
        return samples


class YAIBAKIEICU(_YAIBEICUTask):
    """YAIB acute kidney injury (AKI) prediction on eICU-CRD.

    Works with :class:`~pyhealth.datasets.eICUDataset`.

    Predicts, at every hour of the ICU stay, AKI (KDIGO stage 1 or higher)
    within the next 6 hours, following YAIB-cohorts aki.R and its KDIGO
    callbacks:

    - Creatinine stage, at every hourly creatinine bin (including bins before
      ICU admission), from the minimum over the past 48 hours and 7 days
      (windows (t - 48h, t] and (t - 168h, t]): stage 3 for at least 3 times
      the 7-day minimum, or at least 4 mg/dL with a rise of 0.3 over 48 hours
      or 1.5 times the 7-day minimum; stage 2 for 2 times; stage 1 for a rise
      of 0.3 over 48 hours or 1.5 times.
    - Urine stage, at every hourly urine bin: the hourly urine output is
      divided by the hours since the previous urine bin plus one (one if that
      exceeds 24), these rates are summed over the past 6, 12 and 24 hours
      and divided by the patient's weight. Stage 3 for a 24-hour value below
      0.3 or a 12-hour value of 0, stage 2 for a 12-hour value below 0.5,
      stage 1 for a 6-hour value below 0.5, each only once that many hours
      have passed since ICU admission. As in YAIB, stays without a valid
      weight are not divided by a weight (YAIB's 75 kg default never applies,
      since ricu drops missing weights before the join).

    AKI onset is the first hour with either stage at 1 or higher. On top of
    the YAIB base cohort, stays are excluded if the onset is before hour 6 of
    the ICU stay, including onsets before ICU admission (as in
    YAIB-cohorts), if the baseline creatinine (the running minimum up to the
    first creatinine in the ICU) exceeds 4 mg/dL, or if the stay belongs to
    an eICU hospital where no ICU stay has AKI at any time (YAIB-cohorts
    ``aki.R``, eICU only). The stay is observed until 6 hours after the
    onset, for at most 7 days, and every hour from 6 hours before to 6 hours
    after the onset is labeled 1.

    Requires the ``lab``, ``vitalperiodic``, ``vitalaperiodic``,
    ``intakeoutput``, ``respiratorycharting`` and ``nursecharting`` tables.

    Examples:
        >>> from pyhealth.datasets import eICUDataset
        >>> from pyhealth.tasks import YAIBAKIEICU
        >>> dataset = eICUDataset(
        ...     root="/path/to/eicu-crd/2.0",
        ...     tables=["lab", "vitalperiodic", "vitalaperiodic",
        ...             "intakeoutput", "respiratorycharting",
        ...             "nursecharting"],
        ... )
        >>> samples = dataset.set_task(YAIBAKIEICU())
    """

    task_name: str = "YAIBAKIEICU"
    output_schema: ClassVar[dict[str, Any]] = {"aki": "timeseries_binary"}

    MAX_HOURS: ClassVar[int] = 7 * 24
    HORIZON: ClassVar[int] = 6
    MIN_ONSET_HOURS: ClassVar[int] = 6
    MAX_BASELINE_CREA: ClassVar[float] = 4.0

    _PREVALENCE_TABLES: ClassVar[tuple[str, ...]] = ("lab", "intakeoutput")
    _CREA: ClassVar[dict[str, tuple[float | None, float | None, list[_Item]]]] = {
        "crea": _DYNAMIC_FEATURES["crea"],
    }

    @classmethod
    def _kdigo_crea(cls, hours: np.ndarray, crea: np.ndarray) -> np.ndarray:
        min_48 = cls._window_min(hours, crea, 48)
        min_168 = cls._window_min(hours, crea, 168)
        return np.select(
            [
                crea >= 3 * min_168,
                (crea >= 4) & ((crea - min_48 >= 0.3) | (crea >= 1.5 * min_168)),
                crea >= 2 * min_168,
                crea >= min_48 + 0.3,
                crea >= 1.5 * min_168,
            ],
            [3, 3, 2, 1, 1],
            default=0,
        )

    def _urine_rates(self, patient: Any, stays: pl.DataFrame) -> pl.DataFrame:
        """Hourly maximum urine rates (YAIB ``urine_rate``).

        As in YAIB, the rate is computed on the unaggregated urine rows
        (after flooring their times to hours): each row is divided by the
        hours since the previous row plus one, so within an hour only the
        first row is spread over the gap, and the maximum per hour is kept.
        The order of rows within an hour therefore matters. ricu reads the
        rows item by item (the ``"Urine"`` and ``"URINE CATHETER"`` labels,
        then the catheter output pattern), in table order within an item;
        rows are ordered here by item, then by time.
        """
        empty = pl.DataFrame(
            schema={"stay_id": pl.Int64, "hour": pl.Int64, "rate": pl.Float64}
        )
        events = patient.get_events(event_type="intakeoutput", return_df=True)
        if events.height == 0:
            return empty
        urine = pl.concat(
            [
                events.filter(_selects(item, f"intakeoutput/{item.selector}")).with_columns(
                    pl.lit(index).alias("item")
                )
                for index, item in enumerate(_DYNAMIC_FEATURES["urine"][2])
            ]
        )
        frame = (
            urine.select(
                pl.col("intakeoutput/patientunitstayid").cast(pl.Int64).alias("stay_id"),
                pl.col("intakeoutput/intakeoutputoffset")
                .cast(pl.Float64, strict=False)
                .alias("offset"),
                pl.col("intakeoutput/cellvaluenumeric")
                .cast(pl.Float64, strict=False)
                .alias("value"),
                "item",
            )
            .join(stays.select("stay_id"), on="stay_id", how="inner")
            .filter(
                pl.col("value").is_not_null()
                & ~pl.col("value").is_nan()
                & (pl.col("value") >= 0)
                & (pl.col("value") <= 2000)
            )
            .with_columns((pl.col("offset") / 60).floor().cast(pl.Int64).alias("hour"))
            .sort("stay_id", "hour", "item", "offset")
            .with_columns(
                (
                    pl.col("hour") - pl.col("hour").shift(1).over("stay_id") + 1
                ).alias("gap")
            )
            .with_columns(
                pl.when(pl.col("gap").is_null() | (pl.col("gap") > 24))
                .then(1)
                .otherwise(pl.col("gap"))
                .alias("gap")
            )
            .with_columns((pl.col("value") / pl.col("gap")).alias("rate"))
            .filter(
                pl.col("rate").is_not_null()
                & ~pl.col("rate").is_nan()
                & (pl.col("rate") >= 0)
                & (pl.col("rate") <= 2000)
            )
        )
        return frame.group_by("stay_id", "hour").agg(pl.col("rate").max())

    @classmethod
    def _kdigo_urine(
        cls, hours: np.ndarray, rate: np.ndarray, weight: float
    ) -> tuple[np.ndarray, np.ndarray]:
        divisor = 1.0 if np.isnan(weight) else weight
        per_kg = {d: cls._window_sum(hours, rate, d) / divisor for d in (6, 12, 24)}
        stage = np.select(
            [
                (hours >= 24) & (per_kg[24] < 0.3),
                (hours >= 12) & (per_kg[12] == 0),
                (hours >= 12) & (per_kg[12] < 0.5),
                (hours >= 6) & (per_kg[6] < 0.5),
            ],
            [3, 3, 2, 1],
            default=0,
        )
        return hours, stage

    def _aki_onset(
        self,
        crea_hours: np.ndarray,
        crea: np.ndarray,
        urine_rates: pl.DataFrame,
        stay_id: int,
        weight: float,
    ) -> tuple[int | None, float | None]:
        """Returns the first AKI hour and the baseline creatinine of a stay."""
        onsets = []
        baseline = None
        if len(crea):
            stage = self._kdigo_crea(crea_hours, crea)
            onsets.extend(crea_hours[stage >= 1].tolist())
            running_min = np.minimum.accumulate(crea)
            in_icu = np.nonzero(crea_hours >= 0)[0]
            last = in_icu[0] if len(in_icu) else len(crea) - 1
            baseline = float(running_min[last])
        urine = urine_rates.filter(pl.col("stay_id") == stay_id).sort("hour")
        if urine.height:
            urine_hours = urine["hour"].to_numpy()
            rates = urine["rate"].to_numpy()
            urine_hours, stage = self._kdigo_urine(urine_hours, rates, weight)
            onsets.extend(urine_hours[stage >= 1].tolist())
        return (int(min(onsets)) if onsets else None), baseline

    def _has_outcome(self, patient: Any, stays: pl.DataFrame) -> dict[int, bool]:
        static = self._static(patient, stays)
        hourly = self._hourly(self._measurements(patient, stays, self._CREA))
        urine_rates = self._urine_rates(patient, stays)
        return {
            stay_id: self._aki_onset(
                *self._series(hourly, stay_id, 0),
                urine_rates,
                stay_id,
                static[stay_id][3],
            )[0]
            is not None
            for stay_id in stays["stay_id"]
        }

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        all_stays = self._stays(patient)
        if all_stays.height == 0:
            return []
        stays, hourly, static = self._base_cohort(patient, all_stays)
        urine_rates = self._urine_rates(patient, stays)
        crea_index = self.DYNAMIC_FEATURES.index("crea")

        samples = []
        for stay_id, end in stays.select("stay_id", "end").iter_rows():
            onset, baseline = self._aki_onset(
                *self._series(hourly, stay_id, crea_index),
                urine_rates,
                stay_id,
                static[stay_id][3],
            )
            if onset is not None and onset < self.MIN_ONSET_HOURS:
                continue
            if baseline is not None and baseline > self.MAX_BASELINE_CREA:
                continue
            last = min(end, self.MAX_HOURS)
            if onset is not None:
                last = min(last, onset + self.HORIZON)
            dynamic = self._dynamic(hourly, stay_id, last)
            labels = self._outcome_window(onset, len(dynamic["time"]), self.HORIZON, self.HORIZON)
            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "stay_id": stay_id,
                    "dynamic": dynamic,
                    "static": self._static_series(static[stay_id]),
                    "aki": self._labels(dynamic["time"], labels),
                }
            )
        return samples

class YAIBSepsisEICU(_YAIBEICUTask):
    """YAIB sepsis prediction on eICU-CRD.

    Works with :class:`~pyhealth.datasets.eICUDataset`.

    Predicts, at every hour of the ICU stay, Sepsis-3 onset within the next 6
    hours, following YAIB-cohorts sepsis.R with its sep3_alt definition and
    the SOFA score of the ricu version YAIB pins (0.6.0):

    - Suspected infection: on eICU, YAIB uses si_mode="abx", so every start
      of continuous antibiotic treatment is a suspected infection (body
      fluid samples are not required). Only antibiotics given continuously
      count: from each antibiotic start, no gap of more than 24 hours
      without antibiotic over the next 72 hours (or until ICU death). The
      first suspected infection of the stay is used.
    - SOFA: the worst value of each component over the past 24 hours
      ([t - 24h, t]), summed. Components: respiration (PaO2/FiO2 from
      values up to 2 hours apart, capped at 200 when not ventilated),
      coagulation (platelets), liver (bilirubin), cardiovascular (mean
      arterial pressure and dopamine, norepinephrine, dobutamine and
      epinephrine infusions running for at least an hour), central nervous
      system (GCS, carried forward for 6 hours, 15 when sedated) and renal
      (creatinine and 24-hour urine output).
    - Onset: the first hour from 48 hours before to 24 hours after the
      suspected infection at which SOFA has risen by 2 or more over its
      running minimum since the start of that window.

    eICU specifics, as in ricu: antibiotics come from ``infusiondrug`` and
    ``medication``, vasopressor rates from ``infusiondrug`` (converted to
    mcg/kg/min with the row's ``patientweight``), and intubation from a
    RASS of -3 or lower (carried forward for 6 hours). Ventilation starts at
    the ventilation start times of ``respiratorycare`` and at charted
    ventilator settings in ``respiratorycharting``, and ends at the
    ventilation end times and at charted ``off`` or ``Suspended`` values;
    each start opens a window that ends at the next end within 6 hours, or
    else after 6 hours, kept if at least 30 minutes long.

    Note on the SOFA grid (deliberate deviation from YAIB): the hourly SOFA
    score is computed from the first to the last hour of any SOFA component.
    In ricu 0.6.0, which YAIB pins, ``fill_gaps`` takes the duration of a
    stay's time range as its end, so the grid runs from the first hour to
    ``last - first``: past the data when it starts before ICU admission, and
    too short otherwise, dropping later SOFA values (or all of them when the
    first hour exceeds ``last - first``). This is a ricu bug that YAIB
    inherits, not a design choice, and is fixed as in
    :class:`~pyhealth.tasks.YAIBSepsisMIMIC4`.

    Note on ventilation windows (deliberate deviation from YAIB): ricu's
    ``vent_ind`` pairs each ventilation start with the next end within
    ``match_win`` (6 hours) and stores the window's length in its duration
    column; a start without a matching end should get a window of
    ``match_win``. In ricu 0.6.0, which YAIB pins, the fallback is
    ``calc_dur <- function(x, y) fifelse(is.na(y), x + match_win, y - x)``
    with ``x`` the start time, so such a start gets the duration
    ``start + 6 hours``: a time point used as a length. This is a bug, not a
    design choice: the matched branch of the same expression returns a
    length (``end - start``), ricu documents the windows as start and end
    events "separated by at most ``match_win``", and the result depends on
    when the start falls relative to ICU admission. A ventilator setting
    charted on day 4 counts as ventilation for more than 4 days, and starts
    more than 5.5 hours before ICU admission get a negative or sub-30-minute
    length and are dropped. YAIB-cohorts sets no ``vent_ind`` option and
    does not mention it, and the bug is still in ricu's main branch, so YAIB
    inherits it unknowingly. This task gives such windows 6 hours, as
    :class:`~pyhealth.tasks.YAIBSepsisMIMIC3` does; MIMIC-IV records
    ventilation with end times, so it is not affected.

    With ricu's SOFA grid and ventilation windows instead, the cohort, labels
    and features were verified to match YAIB-cohorts exactly on the eICU
    demo (875 stays). With the fixes, 8 of those stays have an onset before
    hour 6 and are excluded; the others are unchanged.

    On top of the YAIB base cohort, stays with onset before hour 6 of the
    ICU stay are excluded, including onsets before ICU admission (as in
    YAIB-cohorts), as are stays of an eICU hospital where no ICU stay has
    sepsis at any time (YAIB-cohorts ``sepsis.R``, eICU only). The stay is
    observed until 6 hours after the onset, for at most 7 days, and every
    hour from 6 hours before to 6 hours after the onset is labeled 1.

    Requires the ``lab``, ``vitalperiodic``, ``vitalaperiodic``,
    ``intakeoutput``, ``respiratorycharting``, ``nursecharting``,
    ``infusiondrug``, ``medication`` and ``respiratorycare`` tables.

    Examples:
        >>> from pyhealth.datasets import eICUDataset
        >>> from pyhealth.tasks import YAIBSepsisEICU
        >>> dataset = eICUDataset(
        ...     root="/path/to/eicu-crd/2.0",
        ...     tables=["lab", "vitalperiodic", "vitalaperiodic",
        ...             "intakeoutput", "respiratorycharting",
        ...             "nursecharting", "infusiondrug", "medication",
        ...             "respiratorycare"],
        ... )
        >>> samples = dataset.set_task(YAIBSepsisEICU())
    """

    task_name: str = "YAIBSepsisEICU"
    output_schema: ClassVar[dict[str, Any]] = {"sepsis": "timeseries_binary"}

    MAX_HOURS: ClassVar[int] = 7 * 24
    HORIZON: ClassVar[int] = 6
    MIN_ONSET_HOURS: ClassVar[int] = 6

    # SOFA inputs, aggregated per hour as in ricu's SOFA concepts.
    _SOFA_INPUTS: ClassVar[dict[str, tuple[float | None, float | None, list[_Item]]]] = {
        "po2": _DYNAMIC_FEATURES["po2"],
        "fio2": _DYNAMIC_FEATURES["fio2"],
        "plt": _DYNAMIC_FEATURES["plt"],
        "bili": _DYNAMIC_FEATURES["bili"],
        "map": _DYNAMIC_FEATURES["map"],
        "crea": _DYNAMIC_FEATURES["crea"],
        "urine": _DYNAMIC_FEATURES["urine"],
        "egcs": (1, 4, [_nurse("Eyes")]),
        "mgcs": (1, 6, [_nurse("Motor")]),
        "vgcs": (1, 5, [_nurse("Verbal")]),
        "tgcs": (3, 15, [_nurse("GCS Total")]),
    }
    _SOFA_AGGREGATES: ClassVar[dict[str, str]] = {
        "po2": "min",
        "fio2": "max",
        "plt": "min",
        "bili": "max",
        "map": "min",
        "crea": "max",
        "urine": "sum",
        "egcs": "min",
        "mgcs": "min",
        "vgcs": "min",
        "tgcs": "min",
    }

    # drug: (rate regex, ml_to_mcg, minimum rate, maximum rate, duration regex)
    _VASOPRESSORS: ClassVar[dict[str, tuple[str, float, float, float, str]]] = {
        "dopa": (r"^dopa.*\(.+\)$", 1600.0, 0.0, 50.0, r"^dopa"),
        "norepi": (r"^norepi.*\(.+\)$", 32.0, 0.0, 3.0, r"^norepi"),
        "dobu": (r"^dobu.*\(.+\)$", 2000.0, 0.0, 50.0, r"^dobu"),
        "epi": (r"^epi( |n).*\(.+\)$", 40.0, 0.0, 1.5, r"^epi( |n)"),
    }
    _ANTIBIOTIC_INFUSION: ClassVar[str] = (
        r"bactrim|cipro|flagyl|metronidazole|zithromax|zosyn|"
        r"(((amika|cleo|ofloxa)|(azithro|clinda|tobra|vanco)my)c|"
        r"(ampi|oxa|peni|pipera)cill|cefazol|levaqu|rifamp)in"
    )
    _ANTIBIOTIC_MEDICATION: ClassVar[str] = (
        r"cipro|flagyl|maxipime|metronidazole|tazobactam|zosyn|"
        r"cef(azolin|epime)|(((azithro|clinda|vanco)my|ofloxa|vanco)c|"
        r"levaqu|piperacill|roceph)in"
    )

    # respiratorycharting rows that start (respcharttypecat) or end
    # (respchartvalue) ventilation.
    _VENT_START_TYPES: ClassVar[tuple[str, ...]] = (
        "Start",
        "Continued",
        "respFlowPtVentData",
    )
    _VENT_END_VALUES: ClassVar[tuple[str, ...]] = ("off", "Off", "Suspended")
    _VENT_MATCH_MINUTES: ClassVar[int] = 6 * 60
    _VENT_MIN_MINUTES: ClassVar[int] = 30
    # nursecharting label of the RASS, read for intubation.
    _SEDATION_LABEL: ClassVar[str] = "Sedation Score"

    _EXTRA_TABLES: ClassVar[tuple[str, ...]] = (
        "infusiondrug",
        "medication",
        "respiratorycare",
    )
    _PREVALENCE_TABLES: ClassVar[tuple[str, ...]] = (
        *_MEASUREMENT_TABLES,
        *_EXTRA_TABLES,
    )

    def _features_read(
        self,
    ) -> list[dict[str, tuple[float | None, float | None, list[_Item]]]]:
        return [_DYNAMIC_FEATURES, self._SOFA_INPUTS]

    def _extra_rows(self) -> dict[str, pl.Expr]:
        drugs = "|".join(
            [
                *(duration for *_, duration in self._VASOPRESSORS.values()),
                self._ANTIBIOTIC_INFUSION,
            ]
        )
        return {
            "respiratorycharting": pl.col(
                "respiratorycharting/respcharttypecat"
            ).is_in(list(self._VENT_START_TYPES))
            | pl.col("respiratorycharting/respchartvalue").is_in(
                list(self._VENT_END_VALUES)
            ),
            "nursecharting": pl.col("nursecharting/nursingchartcelltypevalname")
            == self._SEDATION_LABEL,
            "infusiondrug": pl.col("infusiondrug/drugname")
            .str.contains(f"(?i)({drugs})")
            .fill_null(False),
            "medication": pl.col("medication/drugname")
            .str.contains("(?i)" + self._ANTIBIOTIC_MEDICATION)
            .fill_null(False),
        }

    # Event helpers ----------------------------------------------------------

    def _table_events(
        self,
        patient: Any,
        table: str,
        columns: dict[str, pl.Expr],
    ) -> pl.DataFrame:
        """Events of an eICU table, with the offset in minutes."""
        events = patient.get_events(event_type=table, return_df=True)
        if events.height == 0:
            return pl.DataFrame()
        offset = {
            **_TABLE_OFFSETS,
            "infusiondrug": "infusionoffset",
            "medication": "drugstartoffset",
            "respiratorycare": "respcarestatusoffset",
        }[table]
        return events.select(
            pl.col(f"{table}/patientunitstayid").cast(pl.Int64).alias("stay_id"),
            pl.col(f"{table}/{offset}").cast(pl.Float64, strict=False).alias("offset"),
            **columns,
        )

    @staticmethod
    def _merge_ranges(
        starts: np.ndarray, ends: np.ndarray, gap: float
    ) -> list[tuple[float, float]]:
        """Merges ranges as ricu's merge_ranges (foverlaps, first match)."""
        ends = ends + gap
        order = np.lexsort((ends, starts))
        starts, ends = starts[order], ends[order]
        merged: dict[float, float] = {}
        for i in range(len(starts)):
            first = next(
                j
                for j in range(len(starts))
                if starts[j] <= ends[i] and ends[j] >= starts[i]
            )
            key = float(starts[first])
            merged[key] = max(merged.get(key, float(ends[i])), float(ends[i]))
        return [(start, end - gap) for start, end in merged.items()]

    # Vasopressors -----------------------------------------------------------

    def _rate_frame(
        self,
        patient: Any,
        static: dict[int, list[float]],
        regex: str,
        ml_to_mcg: float,
        lower: float,
        upper: float,
    ) -> pl.DataFrame:
        """Hourly vasopressor rate in mcg/kg/min (ricu eicu_rate_kg)."""
        events = self._table_events(
            patient,
            "infusiondrug",
            {
                "drugname": pl.col("infusiondrug/drugname"),
                "rate": pl.col("infusiondrug/drugrate").cast(pl.Float64, strict=False),
                "patientweight": pl.col("infusiondrug/patientweight").cast(
                    pl.Float64, strict=False
                ),
            },
        )
        if events.height == 0:
            return pl.DataFrame()
        weights = pl.DataFrame(
            {
                "stay_id": list(static),
                "static_weight": [static[stay_id][3] for stay_id in static],
            },
            schema={"stay_id": pl.Int64, "static_weight": pl.Float64},
        )
        events = (
            events.filter(pl.col("drugname").str.contains("(?i)" + regex, literal=False))
            .join(weights, on="stay_id", how="left")
            .with_columns(
                pl.col("patientweight").fill_null(pl.col("static_weight")).alias("weight"),
                pl.col("drugname").str.extract(r"\(([^()]*)\)$", 1).alias("unit"),
            )
        )
        if events.height == 0:
            return pl.DataFrame()
        events = events.with_columns(
            pl.when(~pl.col("unit").str.contains("(?i)/kg/").fill_null(False))
            .then(pl.col("rate") / pl.col("weight"))
            .otherwise(pl.col("rate"))
            .alias("rate"),
            pl.when(~pl.col("unit").str.contains("(?i)/kg/").fill_null(False))
            .then(pl.col("unit").str.replace(r"^([^/]*)/", "$1/kg/"))
            .otherwise(pl.col("unit"))
            .alias("unit"),
        )
        events = events.with_columns(
            pl.when(pl.col("unit").str.contains("(?i)/hr$").fill_null(False))
            .then(pl.col("rate") / 60)
            .otherwise(pl.col("rate"))
            .alias("rate"),
            pl.when(pl.col("unit").str.contains("(?i)/hr$").fill_null(False))
            .then(pl.col("unit").str.replace(r"(?i)/hr$", "/min"))
            .otherwise(pl.col("unit"))
            .alias("unit"),
        )
        events = events.with_columns(
            pl.when(pl.col("unit").str.contains(r"(?i)^mg/").fill_null(False))
            .then(pl.col("rate") * 1000)
            .otherwise(pl.col("rate"))
            .alias("rate"),
            pl.when(pl.col("unit").str.contains(r"(?i)^mg/").fill_null(False))
            .then(pl.col("unit").str.replace(r"(?i)^mg/", "mcg/"))
            .otherwise(pl.col("unit"))
            .alias("unit"),
        )
        events = events.with_columns(
            pl.when(pl.col("unit").str.contains(r"(?i)^units/").fill_null(False))
            .then(pl.lit(None, dtype=pl.Float64))
            .otherwise(pl.col("rate"))
            .alias("rate"),
        )
        events = events.with_columns(
            pl.when(pl.col("unit").str.contains(r"(?i)^ml/").fill_null(False))
            .then(pl.col("rate") * ml_to_mcg)
            .otherwise(pl.col("rate"))
            .alias("rate"),
            pl.when(pl.col("unit").str.contains(r"(?i)^ml/").fill_null(False))
            .then(pl.col("unit").str.replace(r"(?i)^ml/", "mcg/"))
            .otherwise(pl.col("unit"))
            .alias("unit"),
        )
        events = events.with_columns(
            pl.when(pl.col("unit").str.contains(r"(?i)^nanograms/").fill_null(False))
            .then(pl.col("rate") / 1000)
            .otherwise(pl.col("rate"))
            .alias("rate"),
            pl.when(pl.col("unit").str.contains(r"(?i)^nanograms/").fill_null(False))
            .then(pl.col("unit").str.replace(r"(?i)^nanograms/", "mcg/"))
            .otherwise(pl.col("unit"))
            .alias("unit"),
        )
        events = events.with_columns(
            pl.when(
                pl.col("unit").str.contains(r"(?i)^unknown$").fill_null(False)
                | pl.col("unit").str.contains(r"(?i)^ml$").fill_null(False)
            )
            .then(pl.lit(None, dtype=pl.Float64))
            .otherwise(pl.col("rate"))
            .alias("rate")
        )
        return (
            events.filter(
                pl.col("rate").is_not_null()
                & ~pl.col("rate").is_nan()
                & (pl.col("rate") >= lower)
                & (pl.col("rate") <= upper)
            )
            .with_columns((pl.col("offset") / 60).floor().cast(pl.Int64).alias("hour"))
            .select("stay_id", "offset", "hour", "rate")
        )

    def _rate_durations(
        self, patient: Any, regex: str
    ) -> dict[int, list[tuple[int, float]]]:
        """Grouped infusion starts and durations in hours (ricu eicu_duration)."""
        events = self._table_events(
            patient, "infusiondrug", {"drugname": pl.col("infusiondrug/drugname")}
        )
        out: dict[int, list[tuple[int, float]]] = {}
        if events.height == 0:
            return out
        events = (
            events.filter(pl.col("drugname").str.contains("(?i)" + regex, literal=False))
            .with_columns((pl.col("offset") / 60).floor().cast(pl.Int64).alias("hour"))
            .select("stay_id", "hour")
            .unique()
        )
        for (stay_id,), rows in events.group_by("stay_id"):
            hours = np.sort(rows["hour"].to_numpy())
            if len(hours) == 0:
                continue
            group = np.cumsum(np.r_[True, np.diff(hours) > 5]) - 1
            starts: dict[int, float] = {}
            for group_id in np.unique(group):
                in_group = hours[group == group_id]
                start = int(in_group.min())
                duration = float(in_group.max() - in_group.min())
                starts[start] = max(starts.get(start, 0.0), duration)
            out[stay_id] = sorted(starts.items())
        return out

    def _vasopressors(
        self, patient: Any, static: dict[int, list[float]]
    ) -> dict[int, dict[str, dict[int, float]]]:
        """Hourly maximum rates of infusions lasting >= 1 hour (ricu vaso60)."""
        out: dict[int, dict[str, dict[int, float]]] = {}
        for name, (regex, ml_to_mcg, lower, upper, duration_regex) in self._VASOPRESSORS.items():
            rates = self._rate_frame(patient, static, regex, ml_to_mcg, lower, upper)
            durations = self._rate_durations(patient, duration_regex)
            if rates.height == 0:
                continue
            for (stay_id,), rows in rates.group_by("stay_id"):
                stay_durations = durations.get(stay_id, [])
                if not stay_durations:
                    continue
                periods = [
                    (start, end)
                    for start, end in self._merge_ranges(
                        np.array([start for start, _ in stay_durations], dtype=float),
                        np.array(
                            [start + duration for start, duration in stay_durations],
                            dtype=float,
                        ),
                        gap=5 / 60,
                    )
                    if end - start >= 1.0 - 1e-9
                ]
                if not periods:
                    continue
                rows = rows.sort("offset")
                offsets = rows["offset"].to_numpy()
                values = rows["rate"].to_numpy()
                # ricu expands each rate at the minute resolution used by
                # vaso60, then floors the expanded minutes to hours. A rate
                # near the end of an hour therefore also contributes to the
                # next hour, and the hourly maximum is taken.
                per_hour: dict[int, float] = {}
                for i, offset in enumerate(offsets):
                    gap = offsets[i + 1] - offset if i + 1 < len(offsets) else 60
                    gap = min(max(float(gap), 0.0), 360.0)
                    end = offset + gap - 1
                    for hour in range(int(offset // 60), int(end // 60) + 1):
                        per_hour[hour] = max(per_hour.get(hour, -np.inf), float(values[i]))
                hourly: dict[int, float] = {}
                for hour, value in per_hour.items():
                    if any(start - 1e-9 <= hour <= end + 1e-9 for start, end in periods):
                        hourly[hour] = value
                if hourly:
                    out.setdefault(stay_id, {})[name] = hourly
        return out

    # Ventilation and intubation --------------------------------------------

    def _ventilated_hours(self, patient: Any) -> dict[int, set[int]]:
        """Hours with mechanical ventilation (ricu ``vent_ind`` on eICU).

        Ventilation starts at every nonzero ``ventstartoffset`` and
        ``priorventstartoffset`` of ``respiratorycare`` (the offset is the
        start time) and at every ``respiratorycharting`` row of type
        ``Start``, ``Continued`` or ``respFlowPtVentData``; it ends at every
        nonzero ``ventendoffset`` and ``priorventendoffset`` and at every
        charted value ``off``, ``Off`` or ``Suspended``. Each start opens a
        window that ends at the first end whose minute, minus one, is not
        before the start and at most 6 hours after it (a window of 6 hours if
        there is none; see the class docstring). Windows shorter than 30
        minutes are dropped; per starting hour the longest window, in whole
        hours, is kept and expanded to the hours it covers, extended to hour
        0 when it ends before ICU admission.
        """
        care = self._table_events(
            patient,
            "respiratorycare",
            {
                "ventstart": pl.col("respiratorycare/ventstartoffset").cast(
                    pl.Float64, strict=False
                ),
                "priorventstart": pl.col("respiratorycare/priorventstartoffset").cast(
                    pl.Float64, strict=False
                ),
                "ventend": pl.col("respiratorycare/ventendoffset").cast(
                    pl.Float64, strict=False
                ),
                "priorventend": pl.col("respiratorycare/priorventendoffset").cast(
                    pl.Float64, strict=False
                ),
            },
        )
        chart = self._table_events(
            patient,
            "respiratorycharting",
            {
                "typecat": pl.col("respiratorycharting/respcharttypecat"),
                "value": pl.col("respiratorycharting/respchartvalue"),
            },
        )
        # (stay_id, minute) of each start and end.
        starts: list[tuple[int, float]] = []
        ends: list[tuple[int, float]] = []
        if care.height:
            for column, found in (
                ("ventstart", starts),
                ("priorventstart", starts),
                ("ventend", ends),
                ("priorventend", ends),
            ):
                found.extend(
                    care.filter(pl.col(column).is_not_null() & (pl.col(column) != 0))
                    .select("stay_id", column)
                    .iter_rows()
                )
        if chart.height:
            chart = chart.drop_nulls("offset")
            starts.extend(
                chart.filter(pl.col("typecat").is_in(list(self._VENT_START_TYPES)))
                .select("stay_id", "offset")
                .iter_rows()
            )
            ends.extend(
                chart.filter(pl.col("value").is_in(list(self._VENT_END_VALUES)))
                .select("stay_id", "offset")
                .iter_rows()
            )
        ends_by_stay: dict[int, list[float]] = {}
        for stay_id, minute in ends:
            ends_by_stay.setdefault(stay_id, []).append(minute)
        # Ends shifted by one minute, as in ricu, so an end at the start
        # minute does not close the window.
        shifted_ends = {
            stay_id: np.sort(np.array(minutes)) - 1
            for stay_id, minutes in ends_by_stay.items()
        }
        longest: dict[tuple[int, int], int] = {}
        for stay_id, minute in starts:
            duration = float(self._VENT_MATCH_MINUTES)
            shifted = shifted_ends.get(stay_id)
            if shifted is not None:
                i = np.searchsorted(shifted, minute, side="left")
                if i < len(shifted) and shifted[i] - minute <= self._VENT_MATCH_MINUTES:
                    duration = float(shifted[i] + 1 - minute)
            if duration < self._VENT_MIN_MINUTES:
                continue
            key = (stay_id, int(minute // 60))
            longest[key] = max(longest.get(key, 0), int(duration // 60))
        out: dict[int, set[int]] = {}
        for (stay_id, hour), hours in longest.items():
            out.setdefault(stay_id, set()).update(range(hour, max(hour + hours, 0) + 1))
        return out

    def _intubated_hours(self, patient: Any) -> dict[int, set[int]]:
        """Hours with RASS -3 or lower, carried forward for 6 hours (ett_gcs)."""
        events = self._table_events(
            patient,
            "nursecharting",
            {
                "label": pl.col("nursecharting/nursingchartcelltypevalname"),
                "value": pl.col("nursecharting/nursingchartvalue").cast(
                    pl.Float64, strict=False
                ),
            },
        )
        out: dict[int, set[int]] = {}
        if events.height == 0:
            return out
        events = events.filter(
            (pl.col("label") == self._SEDATION_LABEL)
            & (pl.col("value") >= -5)
            & (pl.col("value") <= 4)
        ).with_columns((pl.col("offset") / 60).floor().cast(pl.Int64).alias("hour"))
        for stay_id, hour in events.filter(pl.col("value") <= -3).select(
            "stay_id", "hour"
        ).iter_rows():
            out.setdefault(stay_id, set()).update(range(hour, hour + 7))
        return out


    # SOFA inputs ------------------------------------------------------------

    def _sofa_inputs(
        self, patient: Any, stays: pl.DataFrame
    ) -> dict[int, dict[str, dict[int, float]]]:
        """Hourly lab and chart SOFA inputs: stay -> input -> hour -> value."""
        names = list(self._SOFA_INPUTS)
        rows = self._measurements(patient, stays, self._SOFA_INPUTS)
        out: dict[int, dict[str, dict[int, float]]] = {}
        for (stay_id, feature), group in rows.group_by("stay_id", "feature"):
            name = names[feature]
            aggregate = getattr(pl.col("value"), self._SOFA_AGGREGATES[name])()
            hourly = group.group_by("hour").agg(aggregate)
            out.setdefault(stay_id, {})[name] = dict(hourly.iter_rows())
        return out

    # SOFA -------------------------------------------------------------------

    @staticmethod
    def _pafi(po2: dict[int, float], fio2: dict[int, float]) -> dict[int, float]:
        """PaO2/FiO2 from values up to 2 hours apart (ricu pafi)."""

        def last_within(series: dict[int, float], hour: int) -> float:
            for previous in range(hour, hour - 3, -1):
                if previous in series:
                    return series[previous]
            return np.nan

        pairs = {(h, v, last_within(fio2, h)) for h, v in po2.items()}
        pairs |= {(h, last_within(po2, h), v) for h, v in fio2.items()}
        pafi: dict[int, float] = {}
        for hour, pao2, fraction in pairs:
            fraction = 21.0 if np.isnan(fraction) else fraction
            if not np.isnan(pao2) and fraction != 0:
                pafi[hour] = 100 * pao2 / fraction
        return pafi

    @staticmethod
    def _gcs(inputs: dict[str, dict[int, float]], intubated: set[int]) -> dict[int, float]:
        """GCS per hour with a component or total (ricu gcs)."""
        components = [inputs.get(name, {}) for name in ("egcs", "vgcs", "mgcs")]
        total_gcs = inputs.get("tgcs", {})
        hours = sorted(set(total_gcs).union(*components))
        gcs: dict[int, float] = {}
        for hour in hours:
            if hour in intubated:
                gcs[hour] = 15.0
                continue
            window = [h for h in hours if hour - 6 <= h <= hour]
            charted = [total_gcs[h] for h in window if h in total_gcs]
            if charted:
                gcs[hour] = charted[-1]
                continue
            total = 0.0
            for component, normal in zip(components, (4.0, 5.0, 6.0)):
                observed = [component[h] for h in window if h in component]
                total += observed[-1] if observed else normal
            gcs[hour] = total
        return gcs

    @staticmethod
    def _urine24(urine: dict[int, float]) -> dict[int, float]:
        """24-hour urine output (ricu urine24)."""
        if not urine:
            return {}
        first, last = min(urine), max(urine)
        out = {}
        for hour in range(first, last + 1):
            window = range(max(first, hour - 24), hour + 1)
            if len(window) < 12:
                out[hour] = np.nan
            else:
                total = sum(urine.get(h, 0.0) for h in window)
                out[hour] = total * 24 / len(window)
        return out

    def _sofa(
        self,
        inputs: dict[str, dict[int, float]],
        vasopressors: dict[str, dict[int, float]],
        ventilated: set[int],
        intubated: set[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Hourly SOFA score from the first to the last hour of any component."""
        eps = np.finfo(float).eps ** 0.5
        nan = np.nan
        components: dict[str, dict[int, float]] = {}

        pafi = self._pafi(inputs.get("po2", {}), inputs.get("fio2", {}))
        respiration = {}
        for hour in set(pafi) | ventilated:
            value = pafi.get(hour, nan)
            if value < 200 and hour not in ventilated:
                value = 200.0
            respiration[hour] = (
                0 if np.isnan(value) else 4 - np.searchsorted([100, 200, 300, 400], value, side="right")
            )
        components["resp"] = respiration
        components["coag"] = {
            h: 4 - np.searchsorted([20, 50, 100, 150], v, side="right")
            for h, v in inputs.get("plt", {}).items()
        }
        components["liver"] = {
            h: np.searchsorted([1.2, 2, 6, 12], v, side="right")
            for h, v in inputs.get("bili", {}).items()
        }

        mean_ap = inputs.get("map", {})
        cardio = {}
        for hour in set(mean_ap).union(*vasopressors.values()):
            dopa, norepi, dobu, epi = (
                vasopressors.get(drug, {}).get(hour, nan)
                for drug in ("dopa", "norepi", "dobu", "epi")
            )
            if dopa > 15 + eps or epi > 0.1 + eps or norepi > 0.1 + eps:
                cardio[hour] = 4
            elif dopa > 5 + eps or eps < epi <= 0.1 + eps or eps < norepi <= 0.1 + eps:
                cardio[hour] = 3
            elif eps < dopa <= 5 + eps or dobu > eps:
                cardio[hour] = 2
            else:
                cardio[hour] = 1 if mean_ap.get(hour, nan) < 70 else 0
        components["cardio"] = cardio

        components["cns"] = {
            h: 4 - np.searchsorted([6, 10, 13, 15], v, side="right")
            for h, v in self._gcs(inputs, intubated).items()
        }

        crea = inputs.get("crea", {})
        urine = self._urine24(inputs.get("urine", {}))
        renal = {}
        for hour in set(crea) | set(urine):
            creatinine, output = crea.get(hour, nan), urine.get(hour, nan)
            if creatinine >= 5 or output < 200:
                renal[hour] = 4
            elif 3.5 <= creatinine < 5 or output < 500:
                renal[hour] = 3
            elif 2 <= creatinine < 3.5:
                renal[hour] = 2
            elif 1.2 <= creatinine < 2:
                renal[hour] = 1
            else:
                renal[hour] = 0
        components["renal"] = renal

        hours = set().union(*components.values())
        if not hours:
            return np.array([], dtype=int), np.array([])
        # Deviates from ricu 0.6.0, whose grid ends at (last - first); see the
        # class docstring.
        grid = np.arange(min(hours), max(hours) + 1)
        sofa = np.zeros(len(grid))
        for scores in components.values():
            values = np.array([scores.get(h, nan) for h in grid], dtype=float)
            for i in range(len(grid)):
                window = values[max(0, i - 24) : i + 1]
                if not np.isnan(window).all():
                    sofa[i] += np.nanmax(window)
        return grid, sofa


    # Suspected infection and onset -------------------------------------------

    def _continuous_antibiotics(
        self, patient: Any, deaths: dict[int, float]
    ) -> dict[int, list[int]]:
        """Start hours of continuous antibiotic treatment (YAIB abx_cont)."""
        infusion = self._table_events(
            patient,
            "infusiondrug",
            {"drugname": pl.col("infusiondrug/drugname")},
        )
        medication = self._table_events(
            patient,
            "medication",
            {
                "drugname": pl.col("medication/drugname"),
                "stop": pl.col("medication/drugstopoffset").cast(pl.Float64, strict=False),
            },
        )
        frames = []
        if infusion.height:
            frames.append(
                infusion.filter(
                    pl.col("drugname").str.contains(
                        "(?i)" + self._ANTIBIOTIC_INFUSION, literal=False
                    )
                )
                .with_columns(
                    (pl.col("offset") / 60).floor().cast(pl.Int64).alias("start"),
                    pl.lit(0.0, dtype=pl.Float64).alias("duration"),
                )
                .select("stay_id", "start", "duration")
            )
        if medication.height:
            frames.append(
                medication.filter(
                    pl.col("drugname").str.contains(
                        "(?i)" + self._ANTIBIOTIC_MEDICATION, literal=False
                    )
                )
                .with_columns(
                    (pl.col("offset") / 60).floor().cast(pl.Int64).alias("start"),
                    ((pl.col("stop") - pl.col("offset")) / 60).floor().alias("duration"),
                )
                .drop_nulls("duration")
                .select("stay_id", "start", "duration")
            )
        if not frames:
            return {}
        rows = pl.concat(frames, how="diagonal_relaxed")
        out: dict[int, list[int]] = {}
        for (stay_id,), group in rows.group_by("stay_id"):
            per_start = group.group_by("start").agg(pl.col("duration").max()).sort("start")
            starts = per_start["start"].to_numpy()
            durations = per_start["duration"].to_numpy()
            death_hour = None
            if stay_id in deaths:
                death_hour = int(np.floor(deaths[stay_id]))
                keep = starts <= death_hour
                starts, durations = starts[keep], durations[keep]
            for start in starts:
                window = (starts >= start) & (starts <= start + 72)
                last = start + 72
                if death_hour is not None:
                    last = min(last, death_hour)
                following = np.append(starts[window][1:], last)
                covered = np.maximum.accumulate(starts[window] + durations[window])
                if (following - covered).max() <= 24:
                    out.setdefault(stay_id, []).append(int(start))
        return out

    @staticmethod
    def _suspected_infection(antibiotics: list[int]) -> int | None:
        """First suspected infection hour (ricu susp_inf, mode "abx")."""
        return min(antibiotics) if antibiotics else None

    @staticmethod
    def _sepsis_onset(
        hours: np.ndarray, sofa: np.ndarray, infection: int | None
    ) -> int | None:
        """First hour of a SOFA rise >= 2 around the infection (ricu sep3)."""
        if infection is None:
            return None
        window = (hours >= infection - 48) & (hours <= infection + 24)
        hours, sofa = hours[window], sofa[window]
        if len(sofa) == 0:
            return None
        rising = np.nonzero(sofa - np.minimum.accumulate(sofa) >= 2)[0]
        return int(hours[rising[0]]) if len(rising) else None

    def _onsets(self, patient: Any, stays: pl.DataFrame) -> dict[int, int | None]:
        """Returns the sepsis onset hour of each stay (None without sepsis)."""
        static = self._static(patient, stays)
        inputs = self._sofa_inputs(patient, stays)
        vasopressors = self._vasopressors(patient, static)
        ventilated = self._ventilated_hours(patient)
        intubated = self._intubated_hours(patient)
        antibiotics = self._continuous_antibiotics(patient, self._icu_deaths(stays))
        onsets: dict[int, int | None] = {}
        for stay_id in stays["stay_id"]:
            sofa_hours, sofa = self._sofa(
                inputs.get(stay_id, {}),
                vasopressors.get(stay_id, {}),
                ventilated.get(stay_id, set()),
                intubated.get(stay_id, set()),
            )
            infection = self._suspected_infection(antibiotics.get(stay_id, []))
            onsets[stay_id] = self._sepsis_onset(sofa_hours, sofa, infection)
        return onsets

    def _has_outcome(self, patient: Any, stays: pl.DataFrame) -> dict[int, bool]:
        return {
            stay_id: onset is not None
            for stay_id, onset in self._onsets(patient, stays).items()
        }

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        all_stays = self._stays(patient)
        if all_stays.height == 0:
            return []
        stays, hourly, static = self._base_cohort(patient, all_stays)
        onsets = self._onsets(patient, stays)

        samples = []
        for stay_id, end in stays.select("stay_id", "end").iter_rows():
            onset = onsets[stay_id]
            if onset is not None and onset < self.MIN_ONSET_HOURS:
                continue
            last = min(end, self.MAX_HOURS)
            if onset is not None:
                last = min(last, onset + self.HORIZON)
            dynamic = self._dynamic(hourly, stay_id, last)
            labels = self._outcome_window(onset, len(dynamic["time"]), self.HORIZON, self.HORIZON)
            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "stay_id": stay_id,
                    "dynamic": dynamic,
                    "static": self._static_series(static[stay_id]),
                    "sepsis": self._labels(dynamic["time"], labels),
                }
            )
        return samples

