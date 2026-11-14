"""Real-time ICU acuity and life-sustaining therapy prediction on MIMIC-IV.

The prediction setting of APRICOT-M (Contreras et al., "Real-time prediction
of intensive care unit patient acuity and therapy requirements using
state-space modelling", Nature Communications 16, 7315 (2025),
https://doi.org/10.1038/s41467-025-62121-1). Every 4 hours of an ICU stay,
the clinical events of the past 4 hours and the patient's static data predict
the acuity state of the next 4 hours (discharge, stable, unstable or
deceased), transitions between stable and unstable, and the onset or stop of
mechanical ventilation, vasopressors and continuous renal replacement
therapy. The acuity states follow the computable phenotype of Ren et al.
(2022): a 4-hour interval is unstable when the patient receives at least one
life-sustaining therapy in it.

This is a clean reimplementation from the paper's Methods. The authors'
code (https://github.com/iheallab/apricotM) is GPL-3.0, so no code was
copied; it was only read to confirm details the paper leaves open (the
MIMIC-IV item IDs of the therapies and the order of the 12 outputs).
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Any, ClassVar

import polars as pl

from .yaib_mimic4 import _parse_time, _YAIBMIMIC4Task

# MIMIC-IV item IDs of the life-sustaining therapies (MetaVision items).
# Kept local to this task on purpose: YAIBSepsisMIMIC4 holds its own copies
# of the shared IDs, so extending these lists cannot move the sepsis labels.
MECHANICAL_VENTILATION_ITEMIDS = (225792,)  # procedureevents: Invasive Ventilation
CRRT_ITEMIDS = (225802, 225803, 225809)  # procedureevents: CRRT, CVVHD, CVVHDF
MASSIVE_TRANSFUSION_ITEMIDS = (229620,)  # procedureevents: Massive Transfusion
VASOPRESSOR_ITEMIDS = (  # inputevents
    221906,  # norepinephrine
    221289,  # epinephrine
    229617,  # epinephrine (second item)
    221662,  # dopamine
    222315,  # vasopressin
    221749,  # phenylephrine
)

# The six routine vital signs every included stay must have (paper Methods).
_REQUIRED_VITALS = ("hr", "resp", "sbp", "dbp", "temp", "o2sat")

_STATES = ("discharge", "stable", "unstable", "deceased")


class ICUAcuityMIMIC4(_YAIBMIMIC4Task):
    """Next-4-hour ICU acuity, acuity transitions and therapy changes (MIMIC-IV).

    Works with :class:`~pyhealth.datasets.MIMIC4EHRDataset`.

    Cohort (paper Methods): every ICU stay of an adult (age 18 or more) that
    lasts between ``min_stay_hours`` (8) hours and ``max_stay_days`` (30)
    days and has at least one measurement of each of the six routine vital
    signs (heart rate, respiratory rate, systolic and diastolic blood
    pressure, temperature and oxygen saturation).

    Each stay is cut into 4-hour intervals from ICU admission. Interval
    ``j`` covers hours ``[4j, 4j + 4)``. A therapy is present in an interval
    when its start-end period overlaps it. The interval is ``unstable`` when
    mechanical ventilation, a vasopressor, CRRT or (if
    ``include_massive_transfusion``) a massive transfusion is present, and
    ``stable`` otherwise. The last interval of the stay is ``deceased`` when
    the patient died in the ICU, and ``discharge`` when the patient left the
    ICU alive (with ``discharge_home_only``, only when the hospital
    discharged the patient home within 24 hours of leaving the ICU;
    otherwise it keeps its stable or unstable state). A death up to
    ``death_tolerance_hours`` after the recorded ICU discharge counts as an
    ICU death. When the stay's admission has no ``admissions`` row, the
    outcome is unknown, so the stay's last interval gets no label: the
    sample predicting it is dropped, and the earlier windows are kept.

    One sample per stay and interval ``j >= 1`` (the prediction window) whose
    observation window, hours ``[4(j - 1), 4j)``, has at least one event:

    - ``events``: the clinical events of the observation window as a
      ``sequence_v2`` series of (time, variable, value) triplets: time in
      hours since ICU admission, the variable name as the token, and the
      value. The processor scales each variable to [0, 1] (min-max per
      variable, fitted on the training patients). The variables are the 48
      YAIB labs and vitals (see
      :class:`~pyhealth.tasks.YAIBMortalityMIMIC4`), with their plausibility
      bounds and unit conversions. Events are not resampled or imputed.
    - ``static``: age, sex, height and weight (missing values get
      indicators, then standardized), as in the YAIB tasks.
    - ``acuity``: 12 binary labels, in the order of :attr:`LABELS`: the
      next interval's state (one-hot: discharge, stable, unstable,
      deceased), the transitions unstable to stable and stable to unstable,
      and for mechanical ventilation, vasopressors and CRRT the stop (on in
      the observation interval, off in the prediction interval) and the
      onset (off, then on).
    - ``stay_id`` and ``window`` (``j``), for episode-level evaluation.

    Fit the processors on the training patients only, with
    ``set_task(task, split=...)``.

    Requires the ``labevents``, ``chartevents``, ``outputevents``,
    ``procedureevents``, ``inputevents`` and ``admissions`` tables.

    Args:
        window_hours: Length of the observation and prediction windows (4 in
            the paper).
        min_stay_hours: Shortest ICU stay kept.
        max_stay_days: Longest ICU stay kept.
        include_massive_transfusion: Count a massive transfusion as a
            life-sustaining therapy for the stable/unstable state (as in the
            paper; it gets no label of its own).
        discharge_home_only: Label the last interval ``discharge`` only when
            the patient went home within 24 hours of leaving the ICU, as in
            the authors' MIMIC-IV code. By default any alive ICU discharge
            counts, as the paper's text describes.
        death_tolerance_hours: A hospital death up to this many hours after
            ICU discharge still counts as an ICU death (the recorded ICU
            discharge can precede the death time slightly).

    Examples:
        >>> from pyhealth.datasets import MIMIC4EHRDataset
        >>> from pyhealth.tasks import ICUAcuityMIMIC4
        >>> dataset = MIMIC4EHRDataset(
        ...     root="/path/to/mimic-iv/2.2",
        ...     tables=["labevents", "chartevents", "outputevents",
        ...             "procedureevents", "inputevents", "admissions"],
        ... )  # doctest: +SKIP
        >>> samples = dataset.set_task(ICUAcuityMIMIC4())  # doctest: +SKIP
        >>> ICUAcuityMIMIC4.LABELS[:4]
        ['discharge', 'stable', 'unstable', 'deceased']
    """

    task_name: str = "ICUAcuityMIMIC4"
    input_schema: ClassVar[dict[str, Any]] = {
        "events": (
            "sequence_v2",
            {
                "levels": [
                    {
                        "ordering": "series",
                        "tokens": 1,
                        "values": [1],
                        "normalize": {"method": "minmax", "by_tokens": [0]},
                    }
                ]
            },
        ),
        # A flat vector, which APRICOT-M fuses with the pooled events (unlike
        # the YAIB tasks, where static is a sequence_v2 series at t = 0).
        "static": (
            "tensor",
            {
                "normalization": "zscore",
                "missing_indicators": True,
                "impute_strategy": "zero",
            },
        ),
    }
    output_schema: ClassVar[dict[str, Any]] = {"acuity": "tensor"}

    LABELS: ClassVar[list[str]] = [
        *_STATES,
        "unstable_to_stable",
        "stable_to_unstable",
        "mv_stop",
        "mv_onset",
        "vp_stop",
        "vp_onset",
        "crrt_stop",
        "crrt_onset",
    ]

    _EXTRA_ITEMIDS: ClassVar[dict[str, tuple[int, ...]]] = {
        "procedureevents": (
            *MECHANICAL_VENTILATION_ITEMIDS,
            *CRRT_ITEMIDS,
            *MASSIVE_TRANSFUSION_ITEMIDS,
        ),
        "inputevents": VASOPRESSOR_ITEMIDS,
    }
    _EXTRA_TABLES: ClassVar[tuple[str, ...]] = (
        "procedureevents",
        "inputevents",
        "admissions",
    )

    def __init__(
        self,
        window_hours: float = 4.0,
        min_stay_hours: float = 8.0,
        max_stay_days: float = 30.0,
        include_massive_transfusion: bool = True,
        discharge_home_only: bool = False,
        death_tolerance_hours: float = 1.0,
    ) -> None:
        super().__init__()
        if window_hours <= 0:
            raise ValueError("window_hours must be positive")
        if min_stay_hours < 2 * window_hours:
            raise ValueError(
                "min_stay_hours must cover one observation and one prediction "
                "window (2 * window_hours)"
            )
        self.window_hours = float(window_hours)
        self.min_stay_hours = float(min_stay_hours)
        self.max_stay_days = float(max_stay_days)
        self.include_massive_transfusion = include_massive_transfusion
        self.discharge_home_only = discharge_home_only
        self.death_tolerance_hours = float(death_tolerance_hours)

    # ------------------------------------------------------------ therapies
    def _therapies(
        self, patient: Any, stays: pl.DataFrame
    ) -> dict[int, list[tuple[str, float, float]]]:
        """Per stay: (therapy, start hour, end hour) since ICU admission."""
        kinds = {
            **{i: "mv" for i in MECHANICAL_VENTILATION_ITEMIDS},
            **{i: "crrt" for i in CRRT_ITEMIDS},
            **{i: "vp" for i in VASOPRESSOR_ITEMIDS},
        }
        if self.include_massive_transfusion:
            kinds.update({i: "bt" for i in MASSIVE_TRANSFUSION_ITEMIDS})
        intimes = dict(stays.select("stay_id", "intime").iter_rows())
        therapies: dict[int, list[tuple[str, float, float]]] = {}
        for table in ("procedureevents", "inputevents"):
            events = patient.get_events(event_type=table, return_df=True)
            if events.height == 0:
                continue
            rows = events.select(
                pl.col(f"{table}/stay_id").cast(pl.Int64, strict=False),
                pl.col(f"{table}/itemid").cast(pl.Int64, strict=False),
                pl.col("timestamp"),
                _parse_time(f"{table}/endtime"),
            )
            for stay_id, itemid, start, end in rows.iter_rows():
                kind = kinds.get(itemid)
                intime = intimes.get(stay_id)
                if kind is None or intime is None or start is None:
                    continue
                end = start if end is None or end < start else end
                therapies.setdefault(stay_id, []).append(
                    (kind, _hours(start, intime), _hours(end, intime))
                )
        return therapies

    def _interval_flags(
        self, therapies: list[tuple[str, float, float]], n_intervals: int
    ) -> list[set[str]]:
        """The therapies present in each interval."""
        flags: list[set[str]] = [set() for _ in range(n_intervals)]
        w = self.window_hours
        for kind, start, end in therapies:
            if end < 0 or (end == start and start < 0):
                continue  # entirely before ICU admission
            first = max(0, math.floor(start / w))
            # A period overlaps [lo, hi) when start < hi and end > lo; a
            # point event (start == end) belongs to the interval holding it.
            last = math.ceil(end / w) - 1 if end > start else first
            for j in range(first, min(last, n_intervals - 1) + 1):
                flags[j].add(kind)
        return flags

    # ------------------------------------------------------------ outcomes
    def _ends(
        self, patient: Any
    ) -> dict[int, tuple[datetime | None, bool, datetime | None]]:
        """Per admission: death time (or None), discharged home, discharge time."""
        admissions = patient.get_events(event_type="admissions", return_df=True)
        if admissions.height == 0:
            return {}
        rows = admissions.select(
            pl.col("admissions/hadm_id").cast(pl.Int64, strict=False),
            _parse_time("admissions/dischtime"),
            _parse_time("admissions/deathtime"),
            pl.col("admissions/hospital_expire_flag").cast(pl.Int64, strict=False),
            pl.col("admissions/discharge_location")
            if "admissions/discharge_location" in admissions.columns
            else pl.lit(None, dtype=pl.String),
        )
        ends = {}
        for hadm_id, dischtime, deathtime, expired, location in rows.iter_rows():
            if hadm_id is None:
                continue
            death = None
            if expired == 1:
                death = deathtime or dischtime
            ends[hadm_id] = (death, location == "HOME", dischtime)
        return ends

    def _final_state(
        self,
        intime: datetime,
        outtime: datetime,
        end: tuple[datetime | None, bool, datetime | None],
    ) -> str | None:
        """``deceased``, ``discharge`` or None (keep the therapy state)."""
        death, home, dischtime = end
        tolerance = timedelta(hours=self.death_tolerance_hours)
        if death is not None and intime < death <= outtime + tolerance:
            return "deceased"
        if death is not None and death <= intime:
            return None
        if not self.discharge_home_only:
            return "discharge"
        near = (
            dischtime is not None
            and abs((dischtime - outtime).total_seconds()) <= 24 * 3600
        )
        return "discharge" if home and near else None

    # ------------------------------------------------------------ samples
    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        stays = self._stays(patient)
        if stays.height == 0:
            return []
        stays = stays.with_columns(
            ((pl.col("outtime") - pl.col("intime")).dt.total_seconds() / 3600)
            .alias("los_hours")
        )
        static = self._static(patient, stays)
        events = self._measurements(
            patient, stays, self._DYNAMIC_ITEMS, exact_hours=True
        )
        therapies = self._therapies(patient, stays)
        ends = self._ends(patient)
        required = [self.DYNAMIC_FEATURES.index(name) for name in _REQUIRED_VITALS]
        w = self.window_hours

        samples = []
        for stay_id, hadm_id, intime, outtime, los in stays.select(
            "stay_id", "hadm_id", "intime", "outtime", "los_hours"
        ).iter_rows():
            if not self.min_stay_hours <= los <= self.max_stay_days * 24:
                continue
            age = static[stay_id][0]
            if not age >= 18:  # also drops a missing (NaN) age
                continue
            stay_events = events.filter(
                (pl.col("stay_id") == stay_id)
                & (pl.col("hours") >= 0)
                & (pl.col("hours") < los)
            ).sort("hours")
            seen = set(stay_events["feature"].unique().to_list())
            if not all(feature in seen for feature in required):
                continue

            n_intervals = math.ceil(los / w)
            flags = self._interval_flags(therapies.get(stay_id, []), n_intervals)
            states = ["unstable" if f else "stable" for f in flags]
            end = ends.get(hadm_id)
            # Without an admissions row the outcome is unknown: no sample
            # predicts the last interval.
            n_windows = n_intervals if end is not None else n_intervals - 1
            final = None if end is None else self._final_state(intime, outtime, end)
            if final is not None:
                states[-1] = final

            hours = stay_events["hours"].to_list()
            features = stay_events["feature"].to_list()
            values = stay_events["value"].to_list()
            start = 0
            for j in range(1, n_windows):
                lo, hi = (j - 1) * w, j * w
                while start < len(hours) and hours[start] < lo:
                    start += 1
                stop = start
                while stop < len(hours) and hours[stop] < hi:
                    stop += 1
                if stop == start:
                    continue
                samples.append(
                    {
                        "patient_id": patient.patient_id,
                        "stay_id": stay_id,
                        "window": j,
                        "events": {
                            "time": hours[start:stop],
                            "tokens": [
                                self.DYNAMIC_FEATURES[f] for f in features[start:stop]
                            ],
                            "values": [[v] for v in values[start:stop]],
                        },
                        "static": static[stay_id],
                        "acuity": self._labels_for(flags, states, j),
                    }
                )
        return samples

    @staticmethod
    def _labels_for(flags: list[set[str]], states: list[str], j: int) -> list[float]:
        """The 12 labels of prediction interval ``j`` (see :attr:`LABELS`)."""
        state, previous = states[j], states[j - 1]
        labels = [float(state == name) for name in _STATES]
        labels.append(float(previous == "unstable" and state == "stable"))
        labels.append(float(previous == "stable" and state == "unstable"))
        for therapy in ("mv", "vp", "crrt"):
            before, after = therapy in flags[j - 1], therapy in flags[j]
            labels.append(float(before and not after))
            labels.append(float(not before and after))
        return labels


def _hours(time: datetime, origin: datetime) -> float:
    return (time - origin).total_seconds() / 3600.0
