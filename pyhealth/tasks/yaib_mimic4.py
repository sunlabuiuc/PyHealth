"""YAIB benchmark tasks for MIMIC-IV.

Re-implements the MIMIC-IV cohorts of YAIB (Yet Another ICU Benchmark,
van de Water et al., ICLR 2024, https://github.com/rvandewater/YAIB) on top of
the PyHealth task API. YAIB builds its cohorts with the ``ricu`` R package
(https://github.com/eth-mds/ricu) and the YAIB-cohorts scripts
(https://github.com/rvandewater/YAIB-cohorts); the item IDs, unit conversions,
plausibility bounds and exclusion rules below are ported from those sources.

All tasks share the YAIB feature set: 4 static variables (age, sex, height,
weight) and 48 dynamic variables binned hourly from ICU admission.
"""

from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Any, ClassVar, NamedTuple

import numpy as np
import polars as pl

from .base_task import BaseTask


class _Item(NamedTuple):
    """Rows of one MIMIC-IV table that feed a feature (a ricu concept item).

    ``scale``/``offset`` convert a value as ``value * scale + offset``, applied
    only to rows whose unit matches ``unit_regex`` (case-insensitive).
    """

    table: str
    itemids: tuple[int, ...]
    unit_regex: str | None = None
    scale: float = 1.0
    offset: float = 0.0


def _lab(*itemids: int, **kwargs: Any) -> _Item:
    return _Item("labevents", itemids, **kwargs)


def _chart(*itemids: int, **kwargs: Any) -> _Item:
    return _Item("chartevents", itemids, **kwargs)


# feature: (min, max, items). Out-of-range values are dropped, as in ricu.
_DYNAMIC_FEATURES: dict[str, tuple[float | None, float | None, list[_Item]]] = {
    "alb": (0, 6, [_lab(50862, 53085)]),
    "alp": (0, None, [_lab(50863, 53086)]),
    "alt": (0, None, [_lab(50861)]),
    "ast": (0, None, [_lab(50878)]),
    "be": (-25, 25, [_lab(50802)]),
    "bicar": (5, 50, [_lab(50882)]),
    "bili": (0, 100, [_lab(50885, 53089)]),
    "bili_dir": (0, 50, [_lab(50883)]),
    "bnd": (None, None, [_lab(51144)]),
    "bun": (0, 200, [_lab(51006)]),
    "ca": (4, 20, [_lab(50893)]),
    "cai": (0.5, 2, [_lab(50808)]),
    "ck": (0, None, [_lab(50910)]),
    "ckmb": (0, None, [_lab(50911)]),
    "cl": (80, 130, [_lab(50902)]),
    "crea": (0, 15, [_lab(50912)]),
    "crp": (0, None, [_lab(50889, unit_regex="mg/dl", scale=10.0)]),
    "dbp": (0, 200, [_chart(220051, 220180)]),
    "fgn": (0, 1500, [_lab(51214)]),
    "fio2": (21, 100, [_chart(223835), _lab(50816)]),
    "glu": (0, 1000, [_lab(50809, 50931)]),
    "hgb": (4, 18, [_lab(51222)]),
    "hr": (0, 300, [_chart(220045)]),
    "inr_pt": (None, None, [_lab(51237)]),
    "k": (0, 10, [_lab(50971)]),
    "lact": (0, 50, [_lab(50813)]),
    "lymph": (0, 100, [_lab(51244)]),
    "map": (0, 250, [_chart(220052, 220181, 225312)]),
    "mch": (0, None, [_lab(51248)]),
    "mchc": (20, 50, [_lab(51249)]),
    "mcv": (50, 150, [_lab(51250)]),
    "methb": (0, 100, [_lab(50814)]),
    "mg": (0.5, 5, [_lab(50960)]),
    "na": (110, 165, [_lab(50983)]),
    "neut": (0, 100, [_lab(51256)]),
    # Deviates from YAIB, see the o2sat note in _YAIBMIMIC4Task.
    "o2sat": (50, 100, [_chart(220277, 226253, 220227), _lab(50817)]),
    "pco2": (10, 150, [_lab(50818)]),
    "ph": (6.8, 8, [_lab(50820)]),
    "phos": (0, 40, [_lab(50970)]),
    "plt": (5, 1200, [_lab(51265)]),
    "po2": (40, 600, [_lab(50821)]),
    "ptt": (0, None, [_lab(51275)]),
    "resp": (0, 120, [_chart(220210, 224688, 224689, 224690)]),
    "sbp": (0, 300, [_chart(220050, 220179)]),
    "temp": (
        32,
        42,
        [
            _chart(223762),
            _chart(223761, 224027, unit_regex="f", scale=5 / 9, offset=-160 / 9),
        ],
    ),
    "tnt": (0, None, [_lab(51003)]),
    "urine": (
        0,
        2000,
        [
            _Item(
                "outputevents",
                (226557, 226558, 226559, 226560, 226561, 226563, 226564,
                 226565, 226566, 226567, 226584, 227510),
            )
        ],
    ),
    "wbc": (0, None, [_lab(51301)]),
}

# Chart-based static features, aggregated by median over the whole stay.
_STATIC_CHART_FEATURES: dict[str, tuple[float | None, float | None, list[_Item]]] = {
    "height": (10, 230, [_chart(226707, unit_regex="^in", scale=2.54)]),
    "weight": (1, 500, [_chart(226512)]),
}

# Features summed (instead of median-aggregated) within an hourly bin.
_SUM_FEATURES = ("urine",)

# Value column per measurement table.
_VALUE_COLUMNS = {
    "labevents": "valuenum",
    "chartevents": "valuenum",
    "outputevents": "value",
}
_UNIT_COLUMNS = {
    "labevents": "valueuom",
    "chartevents": "valueuom",
    "outputevents": "valueuom",
}

_MEASUREMENT_SCHEMA = {
    "stay_id": pl.Int64,
    "hour": pl.Int64,
    "feature": pl.Int64,
    "value": pl.Float64,
    "is_sum": pl.Boolean,
}

_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def _item_lookup(
    features: dict[str, tuple[float | None, float | None, list[_Item]]],
) -> pl.DataFrame:
    """Flattens feature definitions into one row per (table, itemid)."""
    rows = []
    for index, (name, (lower, upper, items)) in enumerate(features.items()):
        for item in items:
            for itemid in item.itemids:
                rows.append(
                    {
                        "table": item.table,
                        "itemid": itemid,
                        "feature": index,
                        "unit_regex": (
                            None if item.unit_regex is None
                            else "(?i)" + item.unit_regex
                        ),
                        "scale": item.scale,
                        "offset": item.offset,
                        "lower": None if lower is None else float(lower),
                        "upper": None if upper is None else float(upper),
                        "is_sum": name in _SUM_FEATURES,
                    }
                )
    return pl.DataFrame(
        rows,
        schema={
            "table": pl.String,
            "itemid": pl.Int64,
            "feature": pl.Int64,
            "unit_regex": pl.String,
            "scale": pl.Float64,
            "offset": pl.Float64,
            "lower": pl.Float64,
            "upper": pl.Float64,
            "is_sum": pl.Boolean,
        },
    )


def _parse_time(column: str) -> pl.Expr:
    return pl.col(column).str.strptime(pl.Datetime("ms"), _TIME_FORMAT, strict=False)


def _hours_since(start: str) -> pl.Expr:
    """Whole hours from ``start`` to the event timestamp, floored as in ricu."""
    return (
        (pl.col("timestamp") - pl.col(start)).dt.total_seconds() / 3600
    ).floor().cast(pl.Int64)


def _longest_false_run(values: np.ndarray) -> int:
    longest = current = 0
    for value in values:
        current = 0 if value else current + 1
        longest = max(longest, current)
    return longest


class _YAIBMIMIC4Task(BaseTask):
    """Shared YAIB base cohort and feature extraction for MIMIC-IV.

    Base cohort (YAIB-cohorts ``base_cohort.R``), applied per ICU stay:

    1. ICU length of stay is negative or missing.
    2. ICU length of stay is shorter than 6 hours.
    3. Fewer than 4 hourly bins with any dynamic measurement (stays with no
       measurement at all are not removed by this rule, matching YAIB).
    4. A gap of more than 12 consecutive hourly bins without any dynamic
       measurement within the first 7 days of the stay.
    5. Age below 18.

    Dynamic features are binned at ``floor(hours since ICU admission)``.
    Laboratory events carry only a hospital admission ID and are attributed to
    the first ICU stay of that admission whose discharge is not before the
    measurement (or the last stay, for measurements after all stays), which is
    how ricu maps admission-level data onto ICU stays.

    Each sample holds ``dynamic`` (hourly timestamps from ICU admission and a
    ``[T, 48]`` array with ``NaN`` for bins without measurement) and
    ``static`` (``[age, sex, height, weight]``, sex encoded as 1 for male and
    0 for female, ``NaN`` when unavailable). Missing values are left to the
    processors, which apply YAIB's preprocessing: the dynamic features are
    standardized, get missing-value indicators and are forward-filled, with
    values before the first measurement set to 0 (the training mean); the
    static features get missing-value indicators, are standardized and
    zero-filled. The statistics are learned when the processors are fitted,
    so use ``set_task(task, split=...)`` to learn them on the training
    patients only. Unlike YAIB, which label-encodes sex as a category, sex is
    standardized like the other static features.

    Note on ``o2sat`` (deliberate deviation from YAIB): YAIB pins ricu 0.6.0,
    whose MIMIC-IV definition of ``o2sat`` lists the blood gas item 50817
    under ``chartevents``. 50817 is a ``labevents`` item, so that entry never
    matches and arterial oxygen saturation is silently dropped; the charted
    arterial saturation (220227) is missing as well. Both items exist in
    MIMIC-IV 2.2, and ricu's ``o2sat`` for other databases (eICU, AUMCdb)
    already combined pulse oximetry with arterial saturation, so the omission
    is a bug that YAIB inherits rather than a design choice. ricu fixed it in
    July 2024 (commit 0624af5). This task follows the fixed definition: items
    220277, 226253 and 220227 from ``chartevents`` plus 50817 from
    ``labevents``. With ricu 0.6.0's definition instead, the cohort, labels
    and all features were verified to match YAIB-cohorts exactly on the
    MIMIC-IV demo. Both ricu versions also include 226253 ("SpO2 Desat
    Limit"), which is a monitor alarm threshold rather than a measurement; it
    is kept for consistency with ricu.

    Requires the ``labevents``, ``chartevents`` and ``outputevents`` tables.
    """

    input_schema: ClassVar[dict[str, Any]] = {
        "dynamic": (
            "timeseries",
            {"normalization": "zscore", "missing_indicators": True},
        ),
        "static": (
            "tensor",
            {
                "normalization": "zscore",
                "missing_indicators": True,
                "impute_strategy": "zero",
            },
        ),
    }

    DYNAMIC_FEATURES: ClassVar[list[str]] = list(_DYNAMIC_FEATURES)
    STATIC_FEATURES: ClassVar[list[str]] = ["age", "sex", "height", "weight"]

    _DYNAMIC_ITEMS: ClassVar[pl.DataFrame] = _item_lookup(_DYNAMIC_FEATURES)
    _STATIC_ITEMS: ClassVar[pl.DataFrame] = _item_lookup(_STATIC_CHART_FEATURES)
    # Item IDs a task needs beyond the features, by table.
    _EXTRA_ITEMIDS: ClassVar[dict[str, tuple[int, ...]]] = {}
    # Tables a task needs beyond the measurement tables.
    _EXTRA_TABLES: ClassVar[tuple[str, ...]] = ()

    ICU_WARDS: ClassVar[tuple[str, ...]] = (
        "Cardiac Vascular Intensive Care Unit (CVICU)",
        "Coronary Care Unit (CCU)",
        "Medical Intensive Care Unit (MICU)",
        "Medical/Surgical Intensive Care Unit (MICU/SICU)",
        "Neuro Intermediate",
        "Neuro Stepdown",
        "Neuro Surgical Intensive Care Unit (Neuro SICU)",
        "Surgical Intensive Care Unit (SICU)",
        "Trauma SICU (TSICU)",
    )

    def pre_filter(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Drops measurement rows whose item ID is not used by the task."""
        columns = df.collect_schema().names()
        missing = [
            table
            for table in (*_VALUE_COLUMNS, *self._EXTRA_TABLES)
            if not any(column.startswith(f"{table}/") for column in columns)
        ]
        if missing:
            raise ValueError(
                f"{self.task_name} requires the MIMIC-IV tables {missing}; "
                "add them to the dataset's `tables`."
            )

        items = pl.concat([self._DYNAMIC_ITEMS, self._STATIC_ITEMS])
        itemids = {
            table: set(items.filter(pl.col("table") == table)["itemid"].to_list())
            for table in _VALUE_COLUMNS
        }
        for table, extra in self._EXTRA_ITEMIDS.items():
            itemids.setdefault(table, set()).update(extra)
        keep = ~pl.col("event_type").is_in(list(itemids))
        for table, ids in itemids.items():
            keep = keep | (
                (pl.col("event_type") == table)
                & pl.col(f"{table}/itemid").is_in([str(i) for i in sorted(ids)])
            )
        return df.filter(keep)

    def _stays(self, patient: Any) -> pl.DataFrame:
        stays = patient.get_events(event_type="icustays", return_df=True)
        return (
            stays.select(
                pl.col("icustays/stay_id").cast(pl.Int64).alias("stay_id"),
                pl.col("icustays/hadm_id").cast(pl.Int64).alias("hadm_id"),
                pl.col("timestamp").alias("intime"),
                _parse_time("icustays/outtime").alias("outtime"),
            )
            .drop_nulls(["intime", "outtime"])
            .sort("intime")
        )

    def _measurements(
        self, patient: Any, stays: pl.DataFrame, items: pl.DataFrame
    ) -> pl.DataFrame:
        """Returns valid measurements as (stay_id, hour, feature, value, is_sum)."""
        frames = []
        for table, value_column in _VALUE_COLUMNS.items():
            table_items = items.filter(pl.col("table") == table)
            if table_items.height == 0:
                continue
            events = patient.get_events(event_type=table, return_df=True)
            if events.height == 0:
                continue
            events = events.select(
                pl.col("timestamp"),
                pl.col(f"{table}/hadm_id").cast(pl.Int64).alias("hadm_id"),
                pl.col(f"{table}/itemid").cast(pl.Int64).alias("itemid"),
                pl.col(f"{table}/{value_column}")
                .cast(pl.Float64, strict=False)
                .alias("value"),
                pl.col(f"{table}/{_UNIT_COLUMNS[table]}").alias("unit"),
                *(
                    [pl.col(f"{table}/stay_id").cast(pl.Int64).alias("stay_id")]
                    if table != "labevents"
                    else []
                ),
            ).join(table_items, on="itemid", how="inner")

            if table == "labevents":
                events = self._assign_stays(events, stays)
            frames.append(events.drop("hadm_id"))

        if not frames:
            return pl.DataFrame(schema=_MEASUREMENT_SCHEMA)

        converted = pl.col("unit").str.contains(pl.col("unit_regex")).fill_null(False)
        return (
            pl.concat(frames, how="diagonal_relaxed")
            .join(stays.select("stay_id", "intime"), on="stay_id", how="inner")
            .with_columns(
                pl.when(converted)
                .then(pl.col("value") * pl.col("scale") + pl.col("offset"))
                .otherwise(pl.col("value"))
                .alias("value"),
                _hours_since("intime").alias("hour"),
            )
            .filter(
                pl.col("value").is_not_null()
                & pl.col("value").is_not_nan()
                & (pl.col("lower").is_null() | (pl.col("value") >= pl.col("lower")))
                & (pl.col("upper").is_null() | (pl.col("value") <= pl.col("upper")))
            )
            .select(list(_MEASUREMENT_SCHEMA))
        )

    @staticmethod
    def _assign_stays(events: pl.DataFrame, stays: pl.DataFrame) -> pl.DataFrame:
        """Maps admission-level events onto ICU stays (ricu ``upgrade_id``)."""
        by_outtime = stays.select("hadm_id", "stay_id", "outtime").sort("outtime")
        last_stay = (
            by_outtime.group_by("hadm_id")
            .agg(pl.col("stay_id").last().alias("last_stay_id"))
        )
        return (
            events.filter(pl.col("hadm_id").is_not_null())
            .with_columns(pl.col("timestamp").cast(by_outtime.schema["outtime"]))
            .sort("timestamp")
            .join_asof(
                by_outtime,
                left_on="timestamp",
                right_on="outtime",
                by="hadm_id",
                strategy="forward",
                check_sortedness=False,
            )
            .join(last_stay, on="hadm_id", how="inner")
            .with_columns(pl.coalesce("stay_id", "last_stay_id").alias("stay_id"))
            .drop("outtime", "last_stay_id")
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

    def _static(self, patient: Any, stays: pl.DataFrame) -> dict[int, list[float]]:
        demographics = patient.get_events(event_type="patients")
        age, sex = float("nan"), float("nan")
        if demographics:
            anchor_age = demographics[0].anchor_age
            if anchor_age is not None:
                age = float(anchor_age)
                if not 0 <= age <= 100:
                    age = float("nan")
            sex = {"M": 1.0, "F": 0.0}.get(demographics[0].gender, float("nan"))

        chart = (
            self._measurements(patient, stays, self._STATIC_ITEMS)
            .group_by("stay_id", "feature")
            .agg(pl.col("value").median())
        )
        static = {stay_id: [age, sex, np.nan, np.nan] for stay_id in stays["stay_id"]}
        for stay_id, feature, value in chart.iter_rows():
            static[stay_id][2 + feature] = value
        return static

    def _base_cohort(
        self, patient: Any, stays: pl.DataFrame
    ) -> tuple[pl.DataFrame, pl.DataFrame, dict[int, list[float]]]:
        """Applies the YAIB base cohort exclusions.

        Args:
            patient: The patient.
            stays: All ICU stays of the patient, from ``_stays``.

        Returns:
            The included stays (with ``los_hours`` and ``end`` = last hourly
            bin), the hourly dynamic features of those stays, and the static
            features per stay.
        """
        stays = stays.with_columns(
            ((pl.col("outtime") - pl.col("intime")).dt.total_seconds() / 3600)
            .alias("los_hours")
        ).with_columns(pl.col("los_hours").floor().cast(pl.Int64).alias("end"))
        static = self._static(patient, stays)
        # Measurements are attributed using all stays, before any exclusion.
        hourly = self._hourly(
            self._measurements(patient, stays, self._DYNAMIC_ITEMS)
        )
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

    def _dynamic(
        self, hourly: pl.DataFrame, stay_id: int, intime: datetime, end: int
    ) -> tuple[list[datetime], np.ndarray]:
        """Builds the ``[end + 1, 48]`` hourly grid for bins ``0..end``."""
        values = np.full((end + 1, len(self.DYNAMIC_FEATURES)), np.nan)
        rows = hourly.filter(
            (pl.col("stay_id") == stay_id)
            & (pl.col("hour") >= 0)
            & (pl.col("hour") <= end)
        )
        values[rows["hour"].to_numpy(), rows["feature"].to_numpy()] = (
            rows["value"].to_numpy()
        )
        timestamps = [intime + timedelta(hours=hour) for hour in range(end + 1)]
        return timestamps, values

    def _icu_deaths(self, patient: Any, stays: pl.DataFrame) -> dict[int, datetime]:
        """Returns the ICU death time per ICU stay.

        A hospital death (``hospital_expire_flag``) counts as an ICU death
        unless the last ward before hospital discharge (from ``transfers``) is
        known and is not one of ``icu_wards``. The death is attributed to the
        first ICU stay of the admission discharged at or after the time of
        death, or else the last one (ricu ``death_icu``).
        """
        admissions = patient.get_events(event_type="admissions", return_df=True)
        deaths = (
            admissions.filter(
                pl.col("admissions/hospital_expire_flag").cast(pl.Int64, strict=False)
                == 1
            )
            .select(
                pl.col("admissions/hadm_id").cast(pl.Int64).alias("hadm_id"),
                _parse_time("admissions/deathtime").alias("timestamp"),
            )
            .drop_nulls("timestamp")
        )
        if deaths.height == 0:
            return {}

        transfers = (
            patient.get_events(event_type="transfers", return_df=True)
            .select(
                pl.col("transfers/hadm_id").cast(pl.Int64).alias("hadm_id"),
                pl.col("timestamp"),
                pl.col("transfers/eventtype").alias("eventtype"),
                pl.col("transfers/careunit").alias("careunit"),
            )
            .filter(pl.col("hadm_id").is_not_null())
            .sort("hadm_id", "timestamp")
        )
        icu_wards = list(getattr(self, "icu_wards", self.ICU_WARDS))
        not_icu = (
            transfers.filter(
                pl.col("eventtype").shift(-1).over("hadm_id") == "discharge"
            )
            .filter(~pl.col("careunit").is_in(icu_wards).fill_null(False))
            ["hadm_id"]
        )
        deaths = deaths.filter(~pl.col("hadm_id").is_in(not_icu.to_list()))
        if deaths.height == 0:
            return {}

        deaths = self._assign_stays(deaths.sort("timestamp"), stays)
        return dict(deaths.select("stay_id", "timestamp").iter_rows())

    def _series(
        self, hourly: pl.DataFrame, stay_id: int, feature: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns the measured hours and values of one feature, by hour.

        Unlike :meth:`_dynamic`, this includes hours outside the stay.
        """
        rows = hourly.filter(
            (pl.col("stay_id") == stay_id)
            & (pl.col("feature") == self.DYNAMIC_FEATURES.index(feature))
        ).sort("hour")
        return rows["hour"].to_numpy(), rows["value"].to_numpy()

    @staticmethod
    def _window_starts(hours: np.ndarray, duration: int) -> np.ndarray:
        """Index of the first row in the window ``(hour - duration, hour]``.

        As ricu's ``slide(..., left_closed = FALSE)`` over measured hours.
        """
        return np.searchsorted(hours, hours - duration, side="right")

    @classmethod
    def _window_min(
        cls, hours: np.ndarray, values: np.ndarray, duration: int
    ) -> np.ndarray:
        starts = cls._window_starts(hours, duration)
        return np.array([values[start : i + 1].min() for i, start in enumerate(starts)])

    @classmethod
    def _window_sum(
        cls, hours: np.ndarray, values: np.ndarray, duration: int
    ) -> np.ndarray:
        cumulative = np.concatenate([[0.0], np.cumsum(values)])
        starts = cls._window_starts(hours, duration)
        return cumulative[np.arange(1, len(values) + 1)] - cumulative[starts]

    @staticmethod
    def _outcome_window(
        onset: int | None, steps: int, before: int, after: int
    ) -> list[float]:
        """Labels 1 for steps ``onset - before`` to ``onset + after``, else 0.

        As YAIB-cohorts ``outcome_window``; an onset outside the ``steps``
        observed steps labels nothing.
        """
        labels = [0.0] * steps
        if onset is not None and 0 <= onset < steps:
            for step in range(max(0, onset - before), min(steps, onset + after + 1)):
                labels[step] = 1.0
        return labels


class YAIBMortalityMIMIC4(_YAIBMIMIC4Task):
    """YAIB ICU mortality prediction after 24 hours on MIMIC-IV.

    Predicts death in the ICU from the first 24 hours of the stay (hourly bins
    0 to 24 inclusive, as in YAIB). On top of the YAIB base cohort, stays are
    excluded if the patient died within 30 hours of ICU admission or the ICU
    stay is shorter than 30 hours.

    A hospital death (``hospital_expire_flag``) counts as an ICU death unless
    the last ward before hospital discharge (from ``transfers``) is known and
    is not an ICU ward. The death is attributed to the first ICU stay of the
    admission discharged at or after the time of death, or else the last one.

    Requires the ``labevents``, ``chartevents``, ``outputevents`` and
    ``transfers`` tables.

    Args:
        icu_wards: Care units counted as ICU wards. ricu uses the distinct
            ``icustays.first_careunit`` values of the dataset; the default is
            that set for MIMIC-IV 2.2, which YAIB was built on.

    Examples:
        >>> from pyhealth.datasets import MIMIC4EHRDataset
        >>> from pyhealth.tasks import YAIBMortalityMIMIC4
        >>> dataset = MIMIC4EHRDataset(
        ...     root="/path/to/mimic-iv/2.2",
        ...     tables=["labevents", "chartevents", "outputevents", "transfers"],
        ... )
        >>> samples = dataset.set_task(YAIBMortalityMIMIC4())
    """

    task_name: str = "YAIBMortalityMIMIC4"
    output_schema: ClassVar[dict[str, Any]] = {"mortality": "binary"}

    OBSERVATION_HOURS: ClassVar[int] = 24
    MIN_STAY_HOURS: ClassVar[int] = 30

    def __init__(self, icu_wards: Sequence[str] | None = None) -> None:
        super().__init__()
        self.icu_wards = list(self.ICU_WARDS if icu_wards is None else icu_wards)

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        all_stays = self._stays(patient)
        if all_stays.height == 0:
            return []
        stays, hourly, static = self._base_cohort(patient, all_stays)
        deaths = self._icu_deaths(patient, all_stays)

        samples = []
        for stay_id, intime, los_hours, end in stays.select(
            "stay_id", "intime", "los_hours", "end"
        ).iter_rows():
            if los_hours < self.MIN_STAY_HOURS:
                continue
            death = deaths.get(stay_id)
            if death is not None and death - intime < timedelta(
                hours=self.MIN_STAY_HOURS
            ):
                continue
            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "stay_id": stay_id,
                    "dynamic": self._dynamic(
                        hourly, stay_id, intime, min(end, self.OBSERVATION_HOURS)
                    ),
                    "static": static[stay_id],
                    "mortality": int(death is not None),
                }
            )
        return samples


class YAIBKidneyFunctionMIMIC4(_YAIBMIMIC4Task):
    """YAIB kidney function prediction on MIMIC-IV.

    Predicts the median creatinine (mg/dL) on the second day of the ICU stay
    from the first 24 hours (hourly bins 0 to 24 inclusive, as in YAIB). As in
    YAIB, the label is the median of the hourly creatinine bins after hour 24
    up to and including hour 48 (bins 25 to 48), where each bin is itself the
    median of its measurements. On top of the YAIB base cohort, stays are
    excluded if the ICU stay is shorter than 48 hours or has no creatinine in
    that window.

    Requires the ``labevents``, ``chartevents`` and ``outputevents`` tables.

    Examples:
        >>> from pyhealth.datasets import MIMIC4EHRDataset
        >>> from pyhealth.tasks import YAIBKidneyFunctionMIMIC4
        >>> dataset = MIMIC4EHRDataset(
        ...     root="/path/to/mimic-iv/2.2",
        ...     tables=["labevents", "chartevents", "outputevents"],
        ... )
        >>> samples = dataset.set_task(YAIBKidneyFunctionMIMIC4())
    """

    task_name: str = "YAIBKidneyFunctionMIMIC4"
    output_schema: ClassVar[dict[str, Any]] = {"creatinine": "regression"}

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
        for stay_id, intime, los_hours, end in stays.select(
            "stay_id", "intime", "los_hours", "end"
        ).iter_rows():
            if los_hours < self.MIN_STAY_HOURS or stay_id not in labels:
                continue
            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "stay_id": stay_id,
                    "dynamic": self._dynamic(
                        hourly, stay_id, intime, min(end, self.OBSERVATION_HOURS)
                    ),
                    "static": static[stay_id],
                    "creatinine": labels[stay_id],
                }
            )
        return samples


class YAIBLengthOfStayMIMIC4(_YAIBMIMIC4Task):
    """YAIB remaining length of stay prediction on MIMIC-IV.

    Predicts, at every hour of the ICU stay, the remaining hours until ICU
    discharge, capped at 7 days. As in YAIB, the stay is observed for at most
    7 days (hourly bins 0 to ``min(end, 168)``, where ``end`` is the ICU
    length of stay in whole hours), and the label at hour ``h`` is
    ``min(168, end - h)``. The cohort is the YAIB base cohort without further
    exclusions.

    The label is a per-hour time series on the same grid as ``dynamic``
    (``timeseries_regression``), so models predict at every hour.

    Requires the ``labevents``, ``chartevents`` and ``outputevents`` tables.

    Examples:
        >>> from pyhealth.datasets import MIMIC4EHRDataset
        >>> from pyhealth.tasks import YAIBLengthOfStayMIMIC4
        >>> dataset = MIMIC4EHRDataset(
        ...     root="/path/to/mimic-iv/2.2",
        ...     tables=["labevents", "chartevents", "outputevents"],
        ... )
        >>> samples = dataset.set_task(YAIBLengthOfStayMIMIC4())
    """

    task_name: str = "YAIBLengthOfStayMIMIC4"
    output_schema: ClassVar[dict[str, Any]] = {"los": "timeseries_regression"}

    MAX_HOURS: ClassVar[int] = 7 * 24

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        all_stays = self._stays(patient)
        if all_stays.height == 0:
            return []
        stays, hourly, static = self._base_cohort(patient, all_stays)

        samples = []
        for stay_id, intime, end in stays.select("stay_id", "intime", "end").iter_rows():
            timestamps, values = self._dynamic(
                hourly, stay_id, intime, min(end, self.MAX_HOURS)
            )
            remaining = [
                float(min(self.MAX_HOURS, end - hour)) for hour in range(len(timestamps))
            ]
            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "stay_id": stay_id,
                    "dynamic": (timestamps, values),
                    "static": static[stay_id],
                    "los": (timestamps, remaining),
                }
            )
        return samples


class YAIBAKIMIMIC4(_YAIBMIMIC4Task):
    """YAIB acute kidney injury (AKI) prediction on MIMIC-IV.

    Predicts, at every hour of the ICU stay, AKI (KDIGO stage 1 or higher)
    within the next 6 hours, following YAIB-cohorts ``aki.R`` and its KDIGO
    callbacks:

    - Creatinine stage, at every hourly creatinine bin (including bins before
      ICU admission), from the minimum over the past 48 hours and 7 days
      (windows ``(t - 48h, t]`` and ``(t - 168h, t]``): stage 3 for at least
      3 times the 7-day minimum, or at least 4 mg/dL with a rise of 0.3 over
      48 hours or 1.5 times the 7-day minimum; stage 2 for 2 times; stage 1
      for a rise of 0.3 over 48 hours or 1.5 times.
    - Urine stage, at every hourly urine bin: the hourly urine output is
      divided by the hours since the previous urine bin plus one (one if
      that exceeds 24), these rates are summed over the past 6, 12 and 24
      hours and divided by the patient's weight. Stage 3 for a 24-hour value
      below 0.3 or a 12-hour value of 0, stage 2 for a 12-hour value below
      0.5, stage 1 for a 6-hour value below 0.5, each only once that many
      hours have passed since ICU admission. As in YAIB, stays without any
      weight are not divided by a weight (YAIB's 75 kg default only applies
      to stays whose weight record is missing a value).

    AKI onset is the first hour with either stage at 1 or higher. On top of
    the YAIB base cohort, stays are excluded if the onset is within the
    first 6 hours of the stay, or if the baseline creatinine (the running
    minimum up to the first creatinine in the ICU) exceeds 4 mg/dL. The stay
    is observed until 6 hours after the onset, for at most 7 days, and every
    hour from 6 hours before to 6 hours after the onset is labeled 1.

    Requires the ``labevents``, ``chartevents`` and ``outputevents`` tables.

    Examples:
        >>> from pyhealth.datasets import MIMIC4EHRDataset
        >>> from pyhealth.tasks import YAIBAKIMIMIC4
        >>> dataset = MIMIC4EHRDataset(
        ...     root="/path/to/mimic-iv/2.2",
        ...     tables=["labevents", "chartevents", "outputevents"],
        ... )
        >>> samples = dataset.set_task(YAIBAKIMIMIC4())
    """

    task_name: str = "YAIBAKIMIMIC4"
    output_schema: ClassVar[dict[str, Any]] = {"aki": "timeseries_binary"}

    MAX_HOURS: ClassVar[int] = 7 * 24
    HORIZON: ClassVar[int] = 6
    MIN_ONSET_HOURS: ClassVar[int] = 6
    MAX_BASELINE_CREA: ClassVar[float] = 4.0

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

    @classmethod
    def _kdigo_urine(
        cls, hours: np.ndarray, urine: np.ndarray, weight: float
    ) -> tuple[np.ndarray, np.ndarray]:
        gap = np.diff(hours, prepend=hours[:1]) + 1.0
        gap[0] = 1.0
        gap[gap > 24] = 1.0
        rate = urine / gap
        keep = (rate >= 0) & (rate <= 2000)  # ricu bounds of urine_rate
        hours, rate = hours[keep], rate[keep]
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
        self, hourly: pl.DataFrame, stay_id: int, weight: float
    ) -> tuple[int | None, float | None]:
        """Returns the first AKI hour and the baseline creatinine of a stay."""
        onsets = []
        baseline = None
        crea_hours, crea = self._series(hourly, stay_id, "crea")
        if len(crea):
            stage = self._kdigo_crea(crea_hours, crea)
            onsets.extend(crea_hours[stage >= 1].tolist())
            running_min = np.minimum.accumulate(crea)
            in_icu = np.nonzero(crea_hours >= 0)[0]
            last = in_icu[0] if len(in_icu) else len(crea) - 1
            baseline = float(running_min[last])
        urine_hours, urine = self._series(hourly, stay_id, "urine")
        if len(urine):
            urine_hours, stage = self._kdigo_urine(urine_hours, urine, weight)
            onsets.extend(urine_hours[stage >= 1].tolist())
        return (int(min(onsets)) if onsets else None), baseline

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        all_stays = self._stays(patient)
        if all_stays.height == 0:
            return []
        stays, hourly, static = self._base_cohort(patient, all_stays)

        samples = []
        for stay_id, intime, end in stays.select("stay_id", "intime", "end").iter_rows():
            onset, baseline = self._aki_onset(hourly, stay_id, static[stay_id][3])
            if onset is not None and onset < self.MIN_ONSET_HOURS:
                continue
            if baseline is not None and baseline > self.MAX_BASELINE_CREA:
                continue
            last = min(end, self.MAX_HOURS)
            if onset is not None:
                last = min(last, onset + self.HORIZON)
            timestamps, values = self._dynamic(hourly, stay_id, intime, last)
            labels = self._outcome_window(
                onset, len(timestamps), self.HORIZON, self.HORIZON
            )
            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "stay_id": stay_id,
                    "dynamic": (timestamps, values),
                    "static": static[stay_id],
                    "aki": (timestamps, labels),
                }
            )
        return samples


class YAIBSepsisMIMIC4(_YAIBMIMIC4Task):
    """YAIB sepsis prediction on MIMIC-IV.

    Predicts, at every hour of the ICU stay, Sepsis-3 onset within the next 6
    hours, following YAIB-cohorts ``sepsis.R`` with its ``sep3_alt``
    definition and the SOFA score of the ricu version YAIB pins (0.6.0):

    - Suspected infection: an antibiotic followed by a body fluid sample
      within 24 hours (at the antibiotic time), or a sample followed by an
      antibiotic within 72 hours (at the sample time). Only antibiotics
      given continuously count: from each antibiotic start, no gap of more
      than 24 hours without antibiotic over the next 72 hours (or until ICU
      death). The first suspected infection of the stay is used.
    - SOFA: the worst value of each component over the past 24 hours
      (``[t - 24h, t]``), summed. Components: respiration (PaO2/FiO2 from
      values up to 2 hours apart, capped at 200 when not ventilated),
      coagulation (platelets), liver (bilirubin), cardiovascular (mean
      arterial pressure and dopamine, norepinephrine, dobutamine and
      epinephrine infusions running for at least an hour), central nervous
      system (GCS, carried forward for 6 hours, 15 when intubated) and renal
      (creatinine and 24-hour urine output).
    - Onset: the first hour from 48 hours before to 24 hours after the
      suspected infection at which SOFA has risen by 2 or more over its
      running minimum since the start of that window.

    Note on the SOFA grid (deliberate deviation from YAIB): the hourly SOFA
    score is computed from the first to the last hour of any SOFA component.
    In ricu 0.6.0, which YAIB pins, ``fill_gaps`` takes the duration of a
    stay's time range as its end, so the grid runs from the first hour to
    ``last - first``: past the data when it starts before ICU admission, and
    too short otherwise, dropping later SOFA values (or all of them when the
    first hour exceeds ``last - first``). This is a ricu bug that YAIB
    inherits, not a design choice. With ricu's grid instead, all SOFA values,
    suspected infections, onsets and the cohort and labels were verified to
    match ricu and YAIB-cohorts exactly on the MIMIC-IV demo.

    On top of the YAIB base cohort, stays with onset within the first 6
    hours of the stay are excluded. The stay is observed until 6 hours after
    the onset, for at most 7 days, and every hour from 6 hours before to 6
    hours after the onset is labeled 1.

    Requires the ``labevents``, ``chartevents``, ``outputevents``,
    ``inputevents``, ``procedureevents``, ``microbiologyevents`` and
    ``transfers`` tables.

    Args:
        icu_wards: Care units counted as ICU wards for ICU death, which ends
            the antibiotic window; see :class:`YAIBMortalityMIMIC4`.

    Examples:
        >>> from pyhealth.datasets import MIMIC4EHRDataset
        >>> from pyhealth.tasks import YAIBSepsisMIMIC4
        >>> dataset = MIMIC4EHRDataset(
        ...     root="/path/to/mimic-iv/2.2",
        ...     tables=["labevents", "chartevents", "outputevents", "inputevents",
        ...             "procedureevents", "microbiologyevents", "transfers"],
        ... )
        >>> samples = dataset.set_task(YAIBSepsisMIMIC4())
    """

    task_name: str = "YAIBSepsisMIMIC4"
    output_schema: ClassVar[dict[str, Any]] = {"sepsis": "timeseries_binary"}

    MAX_HOURS: ClassVar[int] = 7 * 24
    HORIZON: ClassVar[int] = 6
    MIN_ONSET_HOURS: ClassVar[int] = 6

    # SOFA inputs, aggregated per hour as in ricu's SOFA concepts.
    _SOFA_INPUTS: ClassVar[
        dict[str, tuple[float | None, float | None, list[_Item]]]
    ] = {
        "po2": _DYNAMIC_FEATURES["po2"],
        "fio2": _DYNAMIC_FEATURES["fio2"],
        "plt": _DYNAMIC_FEATURES["plt"],
        "bili": _DYNAMIC_FEATURES["bili"],
        "map": _DYNAMIC_FEATURES["map"],
        "crea": _DYNAMIC_FEATURES["crea"],
        "urine": _DYNAMIC_FEATURES["urine"],
        "egcs": (1, 4, [_chart(220739)]),
        "mgcs": (1, 6, [_chart(223901)]),
        "vgcs": (1, 5, [_chart(223900)]),
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
    }
    _SOFA_ITEMS: ClassVar[pl.DataFrame] = _item_lookup(_SOFA_INPUTS)
    # Vasopressor: (inputevents item ID, minimum rate, maximum rate).
    _VASOPRESSORS: ClassVar[dict[str, tuple[int, float, float]]] = {
        "dopa": (221662, 0, 50),
        "norepi": (221906, 0, 3),
        "dobu": (221653, 0, 50),
        "epi": (221289, 0, 1.5),
    }
    _ANTIBIOTICS: ClassVar[tuple[int, ...]] = (
        225798, 225837, 225838, 225840, 225842, 225843, 225844, 225845, 225847,
        225848, 225850, 225851, 225853, 225855, 225857, 225859, 225860, 225862,
        225863, 225865, 225866, 225868, 225869, 225871, 225873, 225875, 225876,
        225877, 225879, 225881, 225882, 225883, 225884, 225885, 225886, 225888,
        225889, 225890, 225892, 225893, 225895, 225896, 225897, 225898, 225899,
        225900, 225902, 225903, 225905, 227691, 228003,
    )
    # Invasive and non-invasive ventilation (procedureevents).
    _VENTILATION: ClassVar[tuple[int, ...]] = (225792, 225794)
    _INTUBATED_GCS: ClassVar[str] = "No Response-ETT"

    _EXTRA_ITEMIDS: ClassVar[dict[str, tuple[int, ...]]] = {
        "chartevents": (220739, 223901, 223900),
        "inputevents": (221662, 221906, 221653, 221289, *_ANTIBIOTICS),
        "procedureevents": _VENTILATION,
    }
    _EXTRA_TABLES: ClassVar[tuple[str, ...]] = (
        "inputevents",
        "procedureevents",
        "microbiologyevents",
        "transfers",
    )

    def __init__(self, icu_wards: Sequence[str] | None = None) -> None:
        super().__init__()
        self.icu_wards = list(self.ICU_WARDS if icu_wards is None else icu_wards)

    # Event times ------------------------------------------------------------

    def _stay_events(
        self,
        patient: Any,
        table: str,
        stays: pl.DataFrame,
        columns: dict[str, pl.Expr],
    ) -> pl.DataFrame:
        """Events of a stay-level table, in whole minutes since ICU admission.

        ``minute`` is the event time and ``end_minute`` the ``endtime``, if
        the table has one, both floored as ricu does.
        """
        events = patient.get_events(event_type=table, return_df=True)
        if events.height == 0:
            return pl.DataFrame()
        has_end = f"{table}/endtime" in events.columns
        time_type = events.schema["timestamp"]
        events = events.select(
            pl.col("timestamp"),
            pl.col(f"{table}/stay_id").cast(pl.Int64).alias("stay_id"),
            pl.col(f"{table}/itemid").cast(pl.Int64).alias("itemid"),
            *([_parse_time(f"{table}/endtime").alias("end")] if has_end else []),
            **columns,
        ).join(stays.select("stay_id", "intime"), on="stay_id", how="inner")

        def minutes(column: str) -> pl.Expr:
            delta = pl.col(column).cast(time_type) - pl.col("intime")
            return delta.dt.total_seconds() // 60

        events = events.with_columns(minutes("timestamp").alias("minute"))
        if has_end:
            events = events.with_columns(minutes("end").alias("end_minute"))
        return events

    @staticmethod
    def _expand_hours(start_hour: int, duration_minutes: float) -> range:
        """Hours covered by a window, as ricu's ``expand`` of a ``win_tbl``."""
        end = int((start_hour * 60 + duration_minutes) // 60)
        return range(start_hour, max(end, 0) + 1)

    # SOFA inputs ------------------------------------------------------------

    def _sofa_inputs(
        self, patient: Any, stays: pl.DataFrame
    ) -> dict[int, dict[str, dict[int, float]]]:
        """Hourly lab and chart SOFA inputs: stay -> input -> hour -> value."""
        names = list(self._SOFA_INPUTS)
        rows = self._measurements(patient, stays, self._SOFA_ITEMS)
        out: dict[int, dict[str, dict[int, float]]] = {}
        for (stay_id, feature), group in rows.group_by("stay_id", "feature"):
            name = names[feature]
            aggregate = getattr(pl.col("value"), self._SOFA_AGGREGATES[name])()
            hourly = group.group_by("hour").agg(aggregate)
            out.setdefault(stay_id, {})[name] = dict(hourly.iter_rows())
        return out

    @staticmethod
    def _merge_ranges(
        starts: np.ndarray, ends: np.ndarray, gap: int
    ) -> list[tuple[int, int]]:
        """Merges ranges as ricu's ``merge_ranges`` (``foverlaps``, first match).

        Each range is grouped under the first range, in start order, that
        overlaps it, after extending all ends by ``gap``.
        """
        ends = ends + gap
        order = np.lexsort((ends, starts))
        starts, ends = starts[order], ends[order]
        merged: dict[int, int] = {}
        for i in range(len(starts)):
            first = next(
                j
                for j in range(len(starts))
                if starts[j] <= ends[i] and ends[j] >= starts[i]
            )
            key = int(starts[first])
            merged[key] = max(merged.get(key, int(ends[i])), int(ends[i]))
        return [(start, end - gap) for start, end in merged.items()]

    def _vasopressors(
        self, patient: Any, stays: pl.DataFrame
    ) -> dict[int, dict[str, dict[int, float]]]:
        """Hourly maximum rate of infusions lasting >= 1 hour (ricu ``vaso60``).

        Rates are spread over every minute of their infusion (median per
        minute), and kept within the infusion periods (per ``linkorderid``,
        merged when less than 5 minutes apart) that last at least an hour.
        """
        events = self._stay_events(
            patient,
            "inputevents",
            stays,
            {
                "rate": pl.col("inputevents/rate").cast(pl.Float64, strict=False),
                "link": pl.col("inputevents/linkorderid").cast(pl.Int64, strict=False),
            },
        )
        out: dict[int, dict[str, dict[int, float]]] = {}
        if events.height == 0:
            return out
        events = events.drop_nulls(["minute", "end_minute"])
        for name, (itemid, lower, upper) in self._VASOPRESSORS.items():
            drug = events.filter(pl.col("itemid") == itemid)
            for (stay_id,), rows in drug.group_by("stay_id"):
                rates = rows.filter(
                    pl.col("rate").is_between(lower, upper)
                    & (pl.col("end_minute") >= pl.col("minute"))
                )
                durations = (
                    rows.group_by("link")
                    .agg(pl.col("minute").min(), pl.col("end_minute").max())
                    .with_columns(
                        (pl.col("end_minute") - pl.col("minute")).alias("dur")
                    )
                    .group_by("minute")
                    .agg(pl.col("dur").max())
                    .filter(pl.col("dur") > 0)
                )
                if rates.height == 0 or durations.height == 0:
                    continue
                periods = [
                    (start, end)
                    for start, end in self._merge_ranges(
                        durations["minute"].to_numpy(),
                        (durations["minute"] + durations["dur"]).to_numpy(),
                        gap=5,
                    )
                    if end - start >= 60
                ]
                if not periods:
                    continue
                lengths = (rates["end_minute"] - rates["minute"] + 1).to_numpy()
                minutes = np.concatenate(
                    [np.arange(s, s + n) for s, n in zip(rates["minute"], lengths)]
                )
                per_minute = (
                    pl.DataFrame(
                        {"minute": minutes, "rate": np.repeat(rates["rate"], lengths)}
                    )
                    .group_by("minute")
                    .agg(pl.col("rate").median())
                )
                inside = pl.lit(False)
                for start, end in periods:
                    inside = inside | pl.col("minute").is_between(start, end)
                hourly = (
                    per_minute.filter(inside)
                    .group_by((pl.col("minute") // 60).alias("hour"))
                    .agg(pl.col("rate").max())
                )
                if hourly.height:
                    out.setdefault(stay_id, {})[name] = dict(hourly.iter_rows())
        return out

    def _ventilated_hours(
        self, patient: Any, stays: pl.DataFrame
    ) -> dict[int, set[int]]:
        """Hours with invasive or non-invasive ventilation (ricu ``vent_ind``)."""
        events = self._stay_events(patient, "procedureevents", stays, {})
        out: dict[int, set[int]] = {}
        if events.height == 0:
            return out
        events = events.filter(
            pl.col("itemid").is_in(list(self._VENTILATION))
        ).drop_nulls(["minute", "end_minute"])
        for stay_id, minute, end_minute in events.select(
            "stay_id", "minute", "end_minute"
        ).iter_rows():
            out.setdefault(stay_id, set()).update(
                self._expand_hours(minute // 60, end_minute - minute)
            )
        return out

    def _intubated_hours(
        self, patient: Any, stays: pl.DataFrame
    ) -> dict[int, set[int]]:
        """Hours with a verbal GCS of "No Response-ETT" (ricu ``ett_gcs``)."""
        events = self._stay_events(
            patient, "chartevents", stays, {"value": pl.col("chartevents/value")}
        )
        out: dict[int, set[int]] = {}
        if events.height == 0:
            return out
        intubated = events.filter(pl.col("value") == self._INTUBATED_GCS)
        for stay_id, minute in intubated.select("stay_id", "minute").iter_rows():
            out.setdefault(stay_id, set()).update(self._expand_hours(minute // 60, 1))
        return out

    # SOFA -------------------------------------------------------------------

    @staticmethod
    def _pafi(po2: dict[int, float], fio2: dict[int, float]) -> dict[int, float]:
        """PaO2/FiO2 from values up to 2 hours apart (ricu ``pafi``).

        Each PaO2 is paired with the last FiO2 of the past 2 hours and each
        FiO2 with the last PaO2; a missing FiO2 counts as 21%.
        """

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
    def _gcs(
        inputs: dict[str, dict[int, float]], intubated: set[int]
    ) -> dict[int, float]:
        """GCS per hour with a component (ricu ``gcs``).

        Each component is carried forward over the past 6 hours of GCS
        measurements; missing components count as normal; intubated hours
        count as 15.
        """
        components = [inputs.get(name, {}) for name in ("egcs", "vgcs", "mgcs")]
        hours = sorted(set().union(*components))
        gcs: dict[int, float] = {}
        for hour in hours:
            if hour in intubated:
                gcs[hour] = 15.0
                continue
            window = [h for h in hours if hour - 6 <= h <= hour]
            total = 0.0
            for component, normal in zip(components, (4.0, 5.0, 6.0)):
                observed = [component[h] for h in window if h in component]
                total += observed[-1] if observed else normal
            gcs[hour] = total
        return gcs

    @staticmethod
    def _urine24(urine: dict[int, float]) -> dict[int, float]:
        """24-hour urine output (ricu ``urine24``).

        Over the hours from the first to the last urine measurement, the sum
        over ``[t - 24h, t]`` scaled to 24 hours, or NaN with fewer than 12
        hours in the window.
        """
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
                0 if np.isnan(value)
                else 4 - np.searchsorted([100, 200, 300, 400], value, side="right")
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
            elif (
                dopa > 5 + eps
                or eps < epi <= 0.1 + eps
                or eps < norepi <= 0.1 + eps
            ):
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
        self, patient: Any, stays: pl.DataFrame, deaths: dict[int, datetime]
    ) -> dict[int, list[int]]:
        """Start hours of continuous antibiotic treatment (YAIB ``abx_cont``).

        From each antibiotic start, the gaps between the end of the
        antibiotics so far and the next start, over the next 72 hours (or
        until ICU death), must all be at most 24 hours. Times and durations
        are floored to hours, as when ricu loads them hourly.
        """
        events = self._stay_events(patient, "inputevents", stays, {})
        out: dict[int, list[int]] = {}
        if events.height == 0:
            return out
        events = events.filter(
            pl.col("itemid").is_in(list(self._ANTIBIOTICS))
        ).drop_nulls(["minute", "end_minute"])
        intime = dict(stays.select("stay_id", "intime").iter_rows())
        for (stay_id,), rows in events.group_by("stay_id"):
            per_start = (
                rows.group_by((pl.col("minute") // 60).alias("start"))
                .agg(((pl.col("end_minute") - pl.col("minute")) // 60).max().alias("dur"))
                .sort("start")
            )
            starts = per_start["start"].to_numpy()
            durations = per_start["dur"].to_numpy()
            death_hour = None
            if stay_id in deaths:
                death = deaths[stay_id] - intime[stay_id]
                death_hour = int(death.total_seconds() // 3600)
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

    def _sample_hours(self, patient: Any, stays: pl.DataFrame) -> dict[int, list[int]]:
        """Hours with a body fluid sample (ricu ``samp``).

        As in ricu, a sample is attributed to an ICU stay by its
        ``chartdate`` (midnight), and timed by its ``charttime``, or noon of
        ``chartdate`` if missing.
        """
        events = patient.get_events(event_type="microbiologyevents", return_df=True)
        out: dict[int, list[int]] = {}
        if events.height == 0:
            return out
        events = events.select(
            pl.col("timestamp"),
            pl.coalesce(
                _parse_time("microbiologyevents/charttime").cast(
                    events.schema["timestamp"]
                ),
                pl.col("timestamp") + timedelta(hours=12),
            ).alias("sample_time"),
            pl.col("microbiologyevents/hadm_id").cast(pl.Int64).alias("hadm_id"),
        )
        events = (
            self._assign_stays(events, stays)
            .join(stays.select("stay_id", "intime"), on="stay_id", how="inner")
            .with_columns(pl.col("sample_time").alias("timestamp"))
            .with_columns(_hours_since("intime").alias("hour"))
        )
        for (stay_id,), rows in events.group_by("stay_id"):
            out[stay_id] = sorted(set(rows["hour"].to_list()))
        return out

    @staticmethod
    def _suspected_infection(
        antibiotics: list[int], samples: list[int]
    ) -> int | None:
        """First suspected infection hour (ricu ``susp_inf``, mode ``"and"``)."""
        times = [a for a in antibiotics if any(a <= s <= a + 24 for s in samples)]
        times += [s for s in samples if any(s <= a <= s + 72 for a in antibiotics)]
        return min(times) if times else None

    @staticmethod
    def _sepsis_onset(
        hours: np.ndarray, sofa: np.ndarray, infection: int | None
    ) -> int | None:
        """First hour of a SOFA rise >= 2 around the infection (ricu ``sep3``)."""
        if infection is None:
            return None
        window = (hours >= infection - 48) & (hours <= infection + 24)
        hours, sofa = hours[window], sofa[window]
        if len(sofa) == 0:
            return None
        rising = np.nonzero(sofa - np.minimum.accumulate(sofa) >= 2)[0]
        return int(hours[rising[0]]) if len(rising) else None

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        all_stays = self._stays(patient)
        if all_stays.height == 0:
            return []
        stays, hourly, static = self._base_cohort(patient, all_stays)
        if stays.height == 0:
            return []

        inputs = self._sofa_inputs(patient, all_stays)
        vasopressors = self._vasopressors(patient, all_stays)
        ventilated = self._ventilated_hours(patient, all_stays)
        intubated = self._intubated_hours(patient, all_stays)
        antibiotics = self._continuous_antibiotics(
            patient, all_stays, self._icu_deaths(patient, all_stays)
        )
        sample_hours = self._sample_hours(patient, all_stays)

        samples = []
        for stay_id, intime, end in stays.select("stay_id", "intime", "end").iter_rows():
            sofa_hours, sofa = self._sofa(
                inputs.get(stay_id, {}),
                vasopressors.get(stay_id, {}),
                ventilated.get(stay_id, set()),
                intubated.get(stay_id, set()),
            )
            infection = self._suspected_infection(
                antibiotics.get(stay_id, []), sample_hours.get(stay_id, [])
            )
            onset = self._sepsis_onset(sofa_hours, sofa, infection)
            if onset is not None and onset < self.MIN_ONSET_HOURS:
                continue
            last = min(end, self.MAX_HOURS)
            if onset is not None:
                last = min(last, onset + self.HORIZON)
            timestamps, values = self._dynamic(hourly, stay_id, intime, last)
            labels = self._outcome_window(
                onset, len(timestamps), self.HORIZON, self.HORIZON
            )
            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "stay_id": stay_id,
                    "dynamic": (timestamps, values),
                    "static": static[stay_id],
                    "sepsis": (timestamps, labels),
                }
            )
        return samples
