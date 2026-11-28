"""Vasopressor-onset prediction task on MIMIC-III (Wu et al. 2022).

Reproduces the cohort of "Learning Optimal Summaries of Clinical
Time-series with Concept Bottleneck Models" (MLHC 2022,
https://proceedings.mlr.press/v182/wu22a.html), original code at
https://github.com/dtak/optimal-summaries-public (``vasopressor/``):
for each patient's first ICU stay, the first 6 hourly measurements of 28
vitals/labs, their measurement indicators and 10 static variables,
labelled by whether a vasopressor is started later in the stay.

What is taken from the original notebooks (cohort rules, variable order,
the 10 statics) and what is a labelled assumption (the ``itemid`` map and
the vasopressor item set, which live in files the original repository
does not ship) is spelled out on :class:`VasopressorOnsetMIMIC3`.
"""

from collections.abc import Iterable, Sequence
from datetime import datetime
from typing import Any, ClassVar

import numpy as np
import polars as pl

from .base_task import BaseTask

# The 28 time-varying vitals/labs, in the original column order
# (``preprocess_helpers.py: changing_vars``).
CHANGING_VARS: list[str] = [
    "dbp",
    "fio2",
    "GCS",
    "hr",
    "map",
    "sbp",
    "spontaneousrr",
    "spo2",
    "temp",
    "urine",
    "bun",
    "magnesium",
    "platelets",
    "sodium",
    "alt",
    "hct",
    "po2",
    "ast",
    "potassium",
    "wbc",
    "bicarbonate",
    "creatinine",
    "lactate",
    "pco2",
    "glucose",
    "inr",
    "hgb",
    "bilirubin_total",
]

# The 10 static variables, in the original column order (they follow the
# 28 variables + 28 indicators in each patient array). ``weight`` is the
# carried-forward weight at the last input hour; ``Weight`` is the
# first-measured weight of the stay.
STATIC_VARS: list[str] = [
    "Age",
    "is_F",
    "is_urgent",
    "is_emergency",
    "is_sicu",
    "is_tsicu",
    "is_micu",
    "is_csru",
    "weight",
    "Weight",
]

# CANDIDATE itemid map (not the original's): the original map lives in a
# package the repository does not ship, so these are the primary itemids
# of each variable in the public MIMIC-Extract resource
# https://github.com/MLforHealth/MIMIC_Extract (itemid_to_variable_map).
# Pass ``itemid_map=`` to the task to replace it.
DEFAULT_ITEMIDS: dict[str, tuple[int, ...]] = {
    "dbp": (8368, 220051, 8441, 220180),
    "fio2": (3420, 223835, 3422),
    "GCS_eye": (184, 220739),
    "GCS_motor": (454, 223901),
    "GCS_verbal": (723, 223900),
    "hr": (211, 220045),
    "map": (52, 220052, 456, 220181),
    "sbp": (51, 220050, 455, 220179),
    "spontaneousrr": (614, 224689),
    "spo2": (646, 220277),
    "temp": (676, 677, 223762, 678, 679, 223761),
    "urine": (40055, 226559, 40069, 226560),
    "bun": (51006, 225624, 781, 1162),
    "magnesium": (50960, 220635, 821, 1532),
    "platelets": (51265, 227457, 828),
    "sodium": (50983, 220645, 837, 1536),
    "alt": (50861, 220644, 769),
    "hct": (51221, 220545, 813),
    "po2": (779,),
    "ast": (50878, 220587, 770),
    "potassium": (50971, 227464, 829, 1535),
    "wbc": (51301, 220546, 861, 1127, 1542),
    "bicarbonate": (50882, 227443),
    "creatinine": (50912, 220615, 791, 1525),
    "lactate": (50813, 225668, 818, 1531),
    "pco2": (50818, 778, 220235),
    "glucose": (50931, 220621, 807, 811, 1529),
    "inr": (51237, 227467, 815, 1530),
    "hgb": (51222, 220228, 814),
    "bilirubin_total": (50885, 225690, 1538, 848),
    "weight": (3580, 3693, 224639, 226512),
}

# Temperatures charted in Fahrenheit, converted to Celsius.
FAHRENHEIT_ITEMIDS: tuple[int, ...] = (678, 679, 223761)

# The three GCS sub-scores, summed into ``GCS`` as in the original.
GCS_PARTS: tuple[str, ...] = ("GCS_eye", "GCS_motor", "GCS_verbal")

# CANDIDATE vasopressor set (not the original's): the label file
# ``vasopressors_mv.csv`` is not in the original repository. These are
# the MetaVision ``inputevents_mv`` itemids of the mimic-code vasoactive
# agents: norepinephrine, epinephrine, dopamine, phenylephrine,
# vasopressin, dobutamine, milrinone. Pass ``vasopressor_itemids=`` to
# the task to replace it.
DEFAULT_VASOPRESSOR_ITEMIDS: tuple[int, ...] = (
    221906,
    221289,
    221662,
    221749,
    222315,
    221653,
    221986,
)

TIME_LEN = 6

_TIME_FORMATS = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d")


class _Reading:
    """A unit-converted measurement, in the shape ``_hourly_series`` reads.

    Args:
        timestamp: when the measurement was charted.
        itemid: its MIMIC-III itemid.
        valuenum: its numeric value.
    """

    __slots__ = ("itemid", "timestamp", "valuenum")

    def __init__(self, timestamp: datetime, itemid: int, valuenum: float) -> None:
        self.timestamp = timestamp
        self.itemid = itemid
        self.valuenum = valuenum


def _to_float(value: Any) -> float | None:
    """Returns ``value`` as a finite float, or None if it is not one."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _to_int(value: Any) -> int | None:
    """Returns ``value`` (``5``, ``"5"``, ``5.0``) as an int, or None."""
    number = _to_float(value)
    return None if number is None else int(number)


def _to_datetime(value: Any) -> datetime | None:
    """Returns ``value`` as a datetime (parsing MIMIC strings), or None."""
    if isinstance(value, datetime):
        return value
    if value is None:
        return None
    for time_format in _TIME_FORMATS:
        try:
            return datetime.strptime(str(value), time_format)
        except ValueError:
            continue
    return None


def _hourly_series(
    events: Iterable[Any],
    itemid_map: dict[int, int],
    start: datetime,
    num_vars: int,
    time_len: int = TIME_LEN,
    value_attr: str = "valuenum",
    aggregate: str = "mean",
) -> tuple[np.ndarray, np.ndarray]:
    """Bins timestamped events into hourly values + indicators.

    An event falls in hour ``h`` when ``h <= (timestamp - start) < h + 1``
    hours, for ``0 <= h < time_len``. Several events of one variable in
    one hour are averaged (or summed, with ``aggregate="sum"``).
    Unmeasured entries are NaN with indicator 0.

    Args:
        events: event objects with ``timestamp``, ``itemid`` and a numeric
            ``value_attr`` attribute.
        itemid_map: maps ``itemid`` to a variable index.
        start: hour-0 timestamp (ICU admission time).
        num_vars: number of variables.
        time_len: number of hourly bins. Default is 6.
        value_attr: attribute holding the number (``"valuenum"`` in
            chartevents / labevents, ``"value"`` in outputevents).
        aggregate: ``"mean"`` or ``"sum"`` within an hour.

    Returns:
        Tuple of ``(values, indicators)``, each of shape
        ``(time_len, num_vars)``.

    Examples:
        >>> from datetime import datetime, timedelta
        >>> from types import SimpleNamespace
        >>> start = datetime(2020, 1, 1, 0, 0, 0)
        >>> events = [
        ...     SimpleNamespace(
        ...         timestamp=start + timedelta(hours=1, minutes=30),
        ...         itemid=211,
        ...         valuenum=80.0,
        ...     )
        ... ]
        >>> values, indicators = _hourly_series(
        ...     events, {211: 0}, start, num_vars=1
        ... )
        >>> values.shape
        (6, 1)
        >>> float(values[1, 0])
        80.0
        >>> float(indicators[1, 0])
        1.0
    """
    totals = np.zeros((time_len, num_vars), dtype=float)
    counts = np.zeros((time_len, num_vars), dtype=float)
    for event in events:
        var_idx = itemid_map.get(_to_int(getattr(event, "itemid", None)))
        timestamp = getattr(event, "timestamp", None)
        value = _to_float(getattr(event, value_attr, None))
        if var_idx is None or timestamp is None or value is None:
            continue
        hour = int((timestamp - start).total_seconds() // 3600)
        if not 0 <= hour < time_len:
            continue
        totals[hour, var_idx] += value
        counts[hour, var_idx] += 1.0
    measured = counts > 0
    values = np.full((time_len, num_vars), np.nan, dtype=float)
    if aggregate == "sum":
        values[measured] = totals[measured]
    else:
        values[measured] = totals[measured] / counts[measured]
    return values, measured.astype(float)


def _forward_fill(values: np.ndarray) -> np.ndarray:
    """Carries the last measured value of each column forward in time.

    Args:
        values: array of shape ``(time_len, num_vars)`` with NaN where
            nothing was measured.

    Returns:
        A copy in which each NaN that follows a measurement holds that
        measurement; NaN before the first measurement stays NaN.

    Examples:
        >>> import numpy as np
        >>> _forward_fill(np.array([[np.nan], [2.0], [np.nan]])).tolist()
        [[nan], [2.0], [2.0]]
    """
    filled = values.copy()
    for t in range(1, filled.shape[0]):
        missing = np.isnan(filled[t])
        filled[t, missing] = filled[t - 1, missing]
    return filled


def fit_standardizer(
    samples: Sequence[dict[str, Any]],
    keys: Sequence[str] = ("values", "statics"),
    method: str = "zscore",
) -> dict[str, dict[str, np.ndarray]]:
    """Fits per-variable standardization statistics on raw task samples.

    Use with ``VasopressorOnsetMIMIC3(normalization=None)`` samples to
    control the scaling yourself, e.g. to compare the two scalings:

    - ``"zscore"``: ``(x - mean) / std``, PyHealth's ``"zscore"``.
    - ``"variance"``: ``(x - mean) / variance``, which is what the
      original preprocessing notebook does (it never takes the square
      root). Not available in PyHealth's processors.

    Statistics come from the observed (non-NaN) entries of ``samples``
    only, so pass the training samples.

    Args:
        samples: raw samples (training part).
        keys: fields to standardize; statistics are per entry of the
            last axis. Default is ``("values", "statics")``.
        method: ``"zscore"`` or ``"variance"``.

    Returns:
        Dict mapping each key to ``{"center": ..., "scale": ...}`` arrays.
        A variable with zero spread gets scale 1.

    Raises:
        ValueError: if ``method`` is unknown.

    Examples:
        >>> samples = [{"values": [[0.0], [4.0]], "statics": [1.0]}]
        >>> stats = fit_standardizer(samples, method="zscore")
        >>> float(stats["values"]["center"][0])
        2.0
        >>> float(stats["values"]["scale"][0])
        2.0
        >>> stats = fit_standardizer(samples, method="variance")
        >>> float(stats["values"]["scale"][0])
        4.0
    """
    if method not in ("zscore", "variance"):
        raise ValueError(f"method must be 'zscore' or 'variance', got {method!r}")
    stats: dict[str, dict[str, np.ndarray]] = {}
    for key in keys:
        stacked = np.stack(
            [np.asarray(sample[key], dtype=float) for sample in samples]
        )
        flat = stacked.reshape(-1, stacked.shape[-1])
        observed = ~np.isnan(flat)
        count = np.maximum(observed.sum(axis=0), 1)
        center = np.where(observed, flat, 0.0).sum(axis=0) / count
        variance = (
            np.where(observed, (flat - center) ** 2, 0.0).sum(axis=0) / count
        )
        scale = variance if method == "variance" else np.sqrt(variance)
        stats[key] = {"center": center, "scale": np.where(scale > 0, scale, 1.0)}
    return stats


def apply_standardizer(
    samples: Sequence[dict[str, Any]],
    stats: dict[str, dict[str, np.ndarray]],
) -> list[dict[str, Any]]:
    """Standardizes raw task samples with fitted statistics.

    Missing (NaN) entries become 0, i.e. the training mean.

    Args:
        samples: raw samples (any part).
        stats: statistics from :func:`fit_standardizer`.

    Returns:
        New sample dicts whose standardized fields are nested lists; all
        other fields are passed through unchanged.

    Examples:
        >>> samples = [{"values": [[0.0], [float("nan")]], "label": 1}]
        >>> stats = fit_standardizer(
        ...     [{"values": [[0.0], [4.0]]}], keys=("values",)
        ... )
        >>> apply_standardizer(samples, stats)
        [{'values': [[-1.0], [0.0]], 'label': 1}]
    """
    standardized = []
    for sample in samples:
        new_sample = dict(sample)
        for key, stat in stats.items():
            array = np.asarray(sample[key], dtype=float)
            array = (array - stat["center"]) / stat["scale"]
            new_sample[key] = np.nan_to_num(array, nan=0.0).tolist()
        standardized.append(new_sample)
    return standardized


_ZSCORE = {"normalization": "zscore", "impute_strategy": "zero"}


class _VasopressorOnsetTask(BaseTask):
    """Shared logic of the vasopressor-onset tasks (Wu et al. 2022).

    Holds everything that does not depend on the database version: the
    cohort rules, the hourly binning, the label and the layout of the
    sample. :class:`VasopressorOnsetMIMIC3` and
    :class:`~pyhealth.tasks.VasopressorOnsetMIMIC4` set the class
    attributes below and override the four hooks ``_stay_allowed``,
    ``_age``, ``_admission_flags`` and ``_unit_flags``.

    Attributes:
        _INPUT_TABLE: event type holding the vasopressor infusions.
        _STAY_ID_KEY: ICU stay identifier (attribute and sample key).
        _DEFAULT_ITEMIDS: default variable name to itemids map.
        _DEFAULT_VASOPRESSOR_ITEMIDS: default vasopressor itemids.
        _FAHRENHEIT_ITEMIDS: temperature itemids charted in Fahrenheit.
        _POUND_ITEMIDS: weight itemids charted in pounds.
    """

    input_schema: ClassVar[dict[str, Any]] = {
        "values": ("tensor", dict(_ZSCORE)),
        "indicators": "tensor",
        "statics": ("tensor", dict(_ZSCORE)),
    }
    output_schema: ClassVar[dict[str, str]] = {"vasopressor_onset": "binary"}

    _INPUT_TABLE: ClassVar[str] = "inputevents_mv"
    _STAY_ID_KEY: ClassVar[str] = "icustay_id"
    _DEFAULT_ITEMIDS: ClassVar[dict[str, tuple[int, ...]]] = DEFAULT_ITEMIDS
    _DEFAULT_VASOPRESSOR_ITEMIDS: ClassVar[tuple[int, ...]] = (
        DEFAULT_VASOPRESSOR_ITEMIDS
    )
    _FAHRENHEIT_ITEMIDS: ClassVar[tuple[int, ...]] = FAHRENHEIT_ITEMIDS
    _POUND_ITEMIDS: ClassVar[tuple[int, ...]] = ()

    def __init__(
        self,
        normalization: str | None = "zscore",
        time_len: int = TIME_LEN,
        min_los_hours: float = 6.0,
        max_los_hours: float = 600.0,
        forward_fill: bool = True,
        require_weight: bool = True,
        itemid_map: dict[str, Sequence[int]] | None = None,
        vasopressor_itemids: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        if normalization not in ("zscore", None):
            raise ValueError("normalization must be 'zscore' or None")
        if time_len < 1:
            raise ValueError("time_len must be at least 1")
        self.normalization = normalization
        self.time_len = time_len
        self.min_los_hours = min_los_hours
        self.max_los_hours = max_los_hours
        self.forward_fill = forward_fill
        self.require_weight = require_weight
        self.itemid_map = {
            name: tuple(int(i) for i in itemids)
            for name, itemids in (itemid_map or self._DEFAULT_ITEMIDS).items()
        }
        self.vasopressor_itemids: set[int] = {
            int(i)
            for i in (vasopressor_itemids or self._DEFAULT_VASOPRESSOR_ITEMIDS)
        }
        if normalization is None:
            self.input_schema = {
                "values": "tensor",
                "indicators": "tensor",
                "statics": "tensor",
            }
        # itemid -> column, for the 27 directly measured variables, the
        # three GCS sub-scores and the weight.
        self._columns = (
            [name for name in CHANGING_VARS if name != "GCS"]
            + list(GCS_PARTS)
            + ["weight"]
        )
        self._chart_map = {
            itemid: self._columns.index(name)
            for name in self._columns
            if name != "urine"
            for itemid in self.itemid_map.get(name, ())
        }
        self._urine_map = {
            itemid: self._columns.index("urine")
            for itemid in self.itemid_map.get("urine", ())
        }

    def pre_filter(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Drops measurement rows whose itemid the task does not read.

        Keeps the patient frames small on the full databases, where
        ``chartevents`` alone has hundreds of millions of rows. Tables
        that are not loaded are left alone.

        Args:
            df: the dataset's global event frame.

        Returns:
            The frame without the unused chart, lab, output and input
            rows.
        """
        columns = set(df.collect_schema().names())
        wanted = {
            "chartevents": set(self._chart_map),
            "labevents": set(self._chart_map),
            "outputevents": set(self._urine_map),
            self._INPUT_TABLE: set(self.vasopressor_itemids),
        }
        keep = pl.lit(True)
        for table, itemids in wanted.items():
            column = f"{table}/itemid"
            if column not in columns:
                continue
            used = pl.col(column).cast(pl.Float64, strict=False).is_in(
                [float(i) for i in sorted(itemids)]
            )
            keep = keep & ((pl.col("event_type") != table) | used)
        return df.filter(keep)

    # ------------------------------------------------- version hooks
    def _stay_allowed(self, stay: Any) -> bool:
        """Whether the first ICU stay may enter the cohort.

        Args:
            stay: the ICU stay event.

        Returns:
            True; subclasses add version-specific rules.
        """
        return True

    def _age(self, demographics: Any, stay: Any) -> float:
        """Age in years at ICU admission, capped at 90 (NaN if unknown).

        Args:
            demographics: the patient's ``patients`` event, or None.
            stay: the ICU stay event.

        Returns:
            The age, from the date of birth.
        """
        dob = _to_datetime(getattr(demographics, "dob", None))
        if dob is None:
            return float("nan")
        return min((stay.timestamp - dob).days / 365.25, 90.0)

    def _admission_flags(self, admission_type: str) -> tuple[float, float]:
        """``(is_urgent, is_emergency)`` of an admission type.

        Args:
            admission_type: the admission's ``admission_type``.

        Returns:
            The two 0/1 flags.
        """
        kind = admission_type.upper()
        return float(kind == "URGENT"), float(kind == "EMERGENCY")

    def _unit_flags(self, unit: str) -> tuple[float, float, float, float]:
        """``(is_sicu, is_tsicu, is_micu, is_csru)`` of a care unit.

        Args:
            unit: the stay's ``first_careunit``.

        Returns:
            The four 0/1 flags.
        """
        unit = unit.upper()
        return (
            float(unit == "SICU"),
            float(unit == "TSICU"),
            float(unit == "MICU"),
            float(unit == "CSRU"),
        )

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Turns one patient into zero or one vasopressor-onset sample.

        Args:
            patient: a :class:`~pyhealth.data.Patient` with ``icustays``,
                ``patients``, ``admissions``, ``chartevents``,
                ``labevents``, ``outputevents`` and vasopressor input
                events.

        Returns:
            A list with one sample dict, or an empty list when the
            patient is outside the cohort.
        """
        stay = self._first_stay(patient)
        if stay is None:
            return []
        label = self._label_onset(patient, stay)
        if label is None:
            return []
        measured, weights = self._measurements(patient, stay)
        statics = self._static_features(patient, stay, weights)
        if statics is None:
            return []
        values, indicators = self._assemble(measured)
        if self.forward_fill:
            values = _forward_fill(values)
        return [
            {
                "patient_id": patient.patient_id,
                self._STAY_ID_KEY: str(getattr(stay, self._STAY_ID_KEY, "")),
                "values": values.tolist(),
                "indicators": indicators.tolist(),
                "statics": statics.tolist(),
                "vasopressor_onset": label,
            }
        ]

    def _stay_status(self, patient: Any) -> tuple[Any | None, str | None]:
        """Finds the first ICU stay and why it is excluded, if it is.

        Args:
            patient: the patient object.

        Returns:
            Tuple of the first ICU stay event (None without one) and the
            exclusion reason: ``"no_icu_stay"``, ``"stay_not_allowed"``
            (see ``_stay_allowed``), ``"los_out_of_range"`` (outside
            ``[max(min_los_hours, time_len), max_los_hours]`` or no
            discharge time), or None when the stay passes.
        """
        stays = [
            stay
            for stay in patient.get_events(event_type="icustays")
            if stay.timestamp is not None
        ]
        if not stays:
            return None, "no_icu_stay"
        stay = min(stays, key=lambda s: s.timestamp)
        if not self._stay_allowed(stay):
            return stay, "stay_not_allowed"
        outtime = _to_datetime(getattr(stay, "outtime", None))
        if outtime is None:
            return stay, "los_out_of_range"
        los_hours = (outtime - stay.timestamp).total_seconds() / 3600.0
        if los_hours < max(self.min_los_hours, self.time_len):
            return stay, "los_out_of_range"
        if los_hours > self.max_los_hours:
            return stay, "los_out_of_range"
        return stay, None

    def _first_stay(self, patient: Any) -> Any | None:
        """Returns the first ICU stay if it meets the cohort rules.

        Args:
            patient: the patient object.

        Returns:
            The first ICU stay event, or None when
            :meth:`_stay_status` gives an exclusion reason.
        """
        stay, reason = self._stay_status(patient)
        return None if reason else stay

    def exclusion_reason(self, patient: Any) -> str | None:
        """Names the cohort rule that excludes a patient, if any.

        The rules are checked in the order the task applies them, so
        counting the reasons over a dataset gives the cohort flow.

        Args:
            patient: the patient object.

        Returns:
            None for a patient in the cohort, otherwise one of
            ``"no_icu_stay"``, ``"stay_not_allowed"``,
            ``"los_out_of_range"``, ``"early_vasopressor"`` (started in
            the first ``time_len`` hours) and ``"no_weight"``.
        """
        stay, reason = self._stay_status(patient)
        if reason:
            return reason
        if self._label_onset(patient, stay) is None:
            return "early_vasopressor"
        _, weights = self._measurements(patient, stay)
        if self.require_weight and len(weights) == 0:
            return "no_weight"
        return None

    def _label_onset(self, patient: Any, stay: Any) -> int | None:
        """Labels the stay by its first vasopressor start.

        Args:
            patient: the patient object.
            stay: the ICU stay event.

        Returns:
            1 if the first vasopressor of the stay starts between
            ``time_len`` and ``max_los_hours`` hours after ICU admission,
            0 if there is none in the stay, and None (excluded) if it
            starts within the first ``time_len`` hours.
        """
        outtime = _to_datetime(getattr(stay, "outtime", None))
        onset_hours = None
        for event in patient.get_events(
            event_type=self._INPUT_TABLE, start=stay.timestamp, end=outtime
        ):
            itemid = _to_int(getattr(event, "itemid", None))
            if itemid not in self.vasopressor_itemids:
                continue
            hours = (event.timestamp - stay.timestamp).total_seconds() / 3600.0
            if onset_hours is None or hours < onset_hours:
                onset_hours = hours
        if onset_hours is None or onset_hours > self.max_los_hours:
            return 0
        if onset_hours < self.time_len:
            return None
        return 1

    def _measurements(
        self, patient: Any, stay: Any
    ) -> tuple[np.ndarray, np.ndarray]:
        """Bins the stay's chart, lab and output events by hour.

        Args:
            patient: the patient object.
            stay: the ICU stay event.

        Returns:
            Hours are counted from the admission minute (chart times
            have no seconds). Tuple of the hourly values of shape
            ``(time_len, columns)``
            (27 direct variables, 3 GCS sub-scores, weight; NaN where
            unmeasured) and the ``(n, 2)`` array of
            ``(hours since admission, kg)`` weight measurements of the
            stay, sorted by time.
        """
        # Chart times are recorded to the minute while the ICU admission
        # time has seconds, so events charted in the admission minute
        # (the admission weight, typically) carry a time just before it.
        # The window therefore starts at the admission minute.
        start = stay.timestamp.replace(second=0, microsecond=0)
        outtime = _to_datetime(getattr(stay, "outtime", None))
        num_columns = len(self._columns)
        temp_col = self._columns.index("temp")
        fio2_col = self._columns.index("fio2")
        weight_col = self._columns.index("weight")
        readings = []
        weights = []
        for table in ("chartevents", "labevents"):
            for event in patient.get_events(
                event_type=table, start=start, end=outtime
            ):
                itemid = _to_int(getattr(event, "itemid", None))
                column = self._chart_map.get(itemid)
                value = _to_float(getattr(event, "valuenum", None))
                if column is None or value is None:
                    continue
                if column == temp_col and itemid in self._FAHRENHEIT_ITEMIDS:
                    value = (value - 32.0) * 5.0 / 9.0
                if column == fio2_col and value <= 1.0:
                    value = value * 100.0
                if column == weight_col:
                    if itemid in self._POUND_ITEMIDS:
                        value = value * 0.45359237
                    if value <= 0:
                        continue
                    hours = (event.timestamp - start).total_seconds() / 3600.0
                    weights.append((hours, value))
                readings.append(_Reading(event.timestamp, itemid, value))
        values, _ = _hourly_series(
            readings, self._chart_map, start, num_columns, self.time_len
        )
        urine, urine_measured = _hourly_series(
            patient.get_events(event_type="outputevents", start=start, end=outtime),
            self._urine_map,
            start,
            num_columns,
            self.time_len,
            value_attr="value",
            aggregate="sum",
        )
        values[urine_measured > 0] = urine[urine_measured > 0]
        return values, np.array(sorted(weights), dtype=float).reshape(-1, 2)

    def _assemble(self, measured: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Arranges the binned columns into the 28 variables.

        GCS is the sum of the sub-scores measured in that hour; its
        indicator is 1 if any sub-score was measured.

        Args:
            measured: hourly values from :meth:`_measurements`.

        Returns:
            Tuple of ``(values, indicators)``, each of shape
            ``(time_len, 28)``, values NaN where unmeasured.
        """
        values = np.full((self.time_len, len(CHANGING_VARS)), np.nan)
        for index, name in enumerate(CHANGING_VARS):
            if name == "GCS":
                parts = measured[
                    :, [self._columns.index(part) for part in GCS_PARTS]
                ]
                any_part = ~np.isnan(parts).all(axis=1)
                values[any_part, index] = np.nansum(parts[any_part], axis=1)
            else:
                values[:, index] = measured[:, self._columns.index(name)]
        return values, (~np.isnan(values)).astype(float)

    def _static_features(
        self, patient: Any, stay: Any, weights: np.ndarray
    ) -> np.ndarray | None:
        """Returns the 10 static variables of ``STATIC_VARS`` for a stay.

        Args:
            patient: the patient object.
            stay: the ICU stay event.
            weights: ``(n, 2)`` array of ``(hours, kg)`` weight
                measurements of the stay, sorted by time.

        Returns:
            Array of shape ``(10,)``, NaN where a value is unknown, or
            None when ``require_weight`` is set and the stay has no
            weight measurement.
        """
        if len(weights) == 0 and self.require_weight:
            return None
        demographics = patient.get_events(event_type="patients")
        gender = getattr(demographics[0], "gender", None) if demographics else None
        age = self._age(demographics[0] if demographics else None, stay)
        hadm_id = str(getattr(stay, "hadm_id", ""))
        admission_type = ""
        for admission in patient.get_events(event_type="admissions"):
            if str(getattr(admission, "hadm_id", "")) == hadm_id:
                admission_type = str(getattr(admission, "admission_type", ""))
                break
        unit = str(getattr(stay, "first_careunit", "") or "")
        first_weight = np.nan
        last_weight = np.nan
        if len(weights):
            first_weight = weights[0, 1]
            in_window = weights[weights[:, 0] < self.time_len]
            # Before the first measurement the series has no value to
            # carry forward; the first measurement stands in for it.
            last_weight = in_window[-1, 1] if len(in_window) else first_weight
        return np.array(
            [
                age,
                float(str(gender).upper() == "F") if gender is not None else np.nan,
                *self._admission_flags(admission_type),
                *self._unit_flags(unit),
                last_weight,
                first_weight,
            ],
            dtype=float,
        )


class VasopressorOnsetMIMIC3(_VasopressorOnsetTask):
    """Vasopressor-onset prediction from the first 6 ICU hours (MIMIC-III).

    Cohort and inputs of Wu et al., "Learning Optimal Summaries of
    Clinical Time-series with Concept Bottleneck Models" (MLHC 2022,
    https://proceedings.mlr.press/v182/wu22a.html). Works with
    :class:`~pyhealth.datasets.MIMIC3Dataset` loaded with the tables
    ``["chartevents", "labevents", "outputevents", "inputevents_mv"]``.

    One sample per patient, from the first ICU stay:

    - ``values``: ``(6, 28)`` hourly vitals/labs in the order of
      ``CHANGING_VARS``.
    - ``indicators``: ``(6, 28)``, 1 where the variable was measured in
      that hour.
    - ``statics``: the 10 variables of ``STATIC_VARS``.
    - ``vasopressor_onset``: 1 if a vasopressor is started between
      ``time_len`` and ``max_los_hours`` hours after ICU admission.

    From the original notebooks: first ICU stay only; ICU length of stay
    between 6 and 600 hours; patients given a vasopressor in the first 6
    hours are excluded; patients with no weight measurement are excluded;
    age is capped at 90; GCS is the sum of its three sub-scores.

    Assumptions, because the original's query files are not public (each
    is an argument, so it can be changed):

    - the ``itemid`` map and the vasopressor items are public candidates
      (``DEFAULT_ITEMIDS``, ``DEFAULT_VASOPRESSOR_ITEMIDS``);
    - only MetaVision stays are used (``metavision_only``), since the
      label comes from ``inputevents_mv``;
    - values are carried forward within the window (``forward_fill``);
      urine output is summed within an hour, everything else averaged;
    - Fahrenheit temperatures are converted to Celsius and FiO2 fractions
      to percent.

    Standardization is PyHealth's ``"zscore"`` by default, with missing
    entries set to the mean; fit it on the training patients only by
    passing a split to ``set_task``. The original instead divides by the
    variance and uses full-cohort statistics. To reproduce that scaling,
    use ``normalization=None`` with :func:`fit_standardizer`.

    Args:
        normalization: ``"zscore"`` or ``None`` (raw values, NaN where
            unmeasured). Default is ``"zscore"``.
        time_len: number of hourly input steps. Default is 6.
        min_los_hours: minimum ICU length of stay. Default is 6.
        max_los_hours: maximum ICU length of stay, and the latest onset
            counted as positive. Default is 600.
        forward_fill: carry values forward within the window.
            Default is True.
        metavision_only: keep only stays recorded in MetaVision.
            Default is True.
        require_weight: drop patients with no weight measurement.
            Default is True.
        itemid_map: variable name to itemids, replacing
            ``DEFAULT_ITEMIDS``. Default is None.
        vasopressor_itemids: ``inputevents_mv`` itemids that count as a
            vasopressor, replacing ``DEFAULT_VASOPRESSOR_ITEMIDS``.
            Default is None.

    Examples:
        >>> from pyhealth.datasets import MIMIC3Dataset, PatientSplit
        >>> from pyhealth.tasks import VasopressorOnsetMIMIC3
        >>> dataset = MIMIC3Dataset(  # doctest: +SKIP
        ...     root="/path/to/mimic-iii/1.4",
        ...     tables=["chartevents", "labevents", "outputevents",
        ...             "inputevents_mv"],
        ... )
        >>> task = VasopressorOnsetMIMIC3()
        >>> task.output_schema
        {'vasopressor_onset': 'binary'}
        >>> train, val, test = dataset.set_task(  # doctest: +SKIP
        ...     task, split=PatientSplit(ratios=(0.68, 0.17, 0.15), seed=1)
        ... )
    """

    task_name: str = "VasopressorOnsetMIMIC3"

    def __init__(
        self,
        normalization: str | None = "zscore",
        time_len: int = TIME_LEN,
        min_los_hours: float = 6.0,
        max_los_hours: float = 600.0,
        forward_fill: bool = True,
        metavision_only: bool = True,
        require_weight: bool = True,
        itemid_map: dict[str, Sequence[int]] | None = None,
        vasopressor_itemids: Sequence[int] | None = None,
    ) -> None:
        super().__init__(
            normalization=normalization,
            time_len=time_len,
            min_los_hours=min_los_hours,
            max_los_hours=max_los_hours,
            forward_fill=forward_fill,
            require_weight=require_weight,
            itemid_map=itemid_map,
            vasopressor_itemids=vasopressor_itemids,
        )
        self.metavision_only = metavision_only

    def _stay_allowed(self, stay: Any) -> bool:
        """Keeps MetaVision stays only when ``metavision_only`` is set.

        Args:
            stay: the ICU stay event.

        Returns:
            False for a non-MetaVision stay under ``metavision_only``.
        """
        dbsource = str(getattr(stay, "dbsource", "") or "").lower()
        return not self.metavision_only or dbsource == "metavision"
