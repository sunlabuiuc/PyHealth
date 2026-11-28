# Authors: Christopher Perez (cjperez3), Kai Cheung (kaicc2), Sarah Lee (slee537)
# Paper: MedMod: Multimodal Benchmark for Medical Prediction Tasks with Electronic
#   Health Records and Chest X-Ray Scans (Elsharief et al., CHIL 2025)
# Paper link: https://proceedings.mlr.press/v287/elsharief25a.html
# Description: MedMod decompensation task (death within 24 h) on MIMIC-IV + MIMIC-CXR.

"""Decompensation from ICU time series and chest X-rays on MIMIC-IV + MIMIC-CXR.

The decompensation task of MedMod: at regular times during an ICU stay, predict
whether the patient dies within the next 24 hours, from the EHR time series so
far and the most recent chest X-ray.

Paper:
    Elsharief, S., et al. (2025). MedMod: Multimodal benchmark for medical
    prediction tasks with electronic health records and chest X-ray scans.
    CHIL 2025. https://proceedings.mlr.press/v287/elsharief25a.html

Sampling and labels follow ``create_decompensation.py`` of mimic3-benchmarks
(Harutyunyan et al., 2019), which MedMod uses. Stay exclusions, EHR variables
and image pairing reuse :mod:`pyhealth.tasks.ehr_cxr_mimic4`.
"""

from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Any, ClassVar

import numpy as np
import polars as pl

from .base_task import BaseTask
from .ehr_cxr_mimic4 import (
    _EVENT_TABLES,
    _MEASURED_NAMES,
    _TIME_FORMAT,
    EHRCXRMortalityMIMIC4,
    _eligible_stays,
    _input_schema,
    _last_cxr,
    _measurements,
    _pre_filter,
    _time_series,
)


def _parse(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value
    return datetime.strptime(str(value), _TIME_FORMAT)


def _window_series(
    rows: pl.DataFrame,
    stay_id: int,
    intime: datetime,
    start: datetime,
    end: datetime,
    bin_hours: int,
) -> np.ndarray | None:
    """Time series of a stay's ICU rows over ``[start, end)``.

    While the window starts at or before ICU admission this is ``_time_series``
    as in the mortality task. Once it starts later, the last value of each
    feature before the window is carried into the first bin as unmeasured
    (mask 0), so that labs and charts measured before the window are treated
    the same way and never count as measured in it.
    """
    if start <= intime:
        return _time_series(rows, stay_id, start, end, bin_hours)
    rows = rows.filter(pl.col("timestamp") < end)
    if rows.height == 0:
        return None
    before = rows.filter(pl.col("timestamp") < start)
    inside = rows.filter(pl.col("timestamp") >= start)
    carried = (
        before.sort("timestamp")
        .group_by("feature")
        .agg(pl.all().last())
        .with_columns(
            pl.lit(start - timedelta(seconds=1))
            .cast(rows.schema["timestamp"])
            .alias("timestamp"),
            pl.lit(stay_id).cast(rows.schema["stay_id"]).alias("stay_id"),
        )
        .select(rows.columns)
    )
    series = _time_series(pl.concat([carried, inside]), stay_id, start, end, bin_hours)
    first_bin = inside.filter(pl.col("timestamp") < start + timedelta(hours=bin_hours))
    measured = set(first_bin["feature"].to_list())
    n = len(_MEASURED_NAMES) + 1
    for feature in range(len(_MEASURED_NAMES)):
        if feature not in measured:
            series[0, n + feature] = 0.0
    scales = [_MEASURED_NAMES.index(f"gcs_{s}") for s in ("eye", "motor", "verbal")]
    if not all(i in measured for i in scales):
        series[0, 2 * n - 1] = 0.0
    return series


class MedModDecompensationMIMIC4(BaseTask):
    """Death within the next 24 hours, predicted during an ICU stay.

    Works with :class:`~pyhealth.datasets.MIMIC4Dataset`.

    Samples: for every eligible ICU stay, one sample at each prediction time
    ``t`` (hours since ICU admission) that is a multiple of
    ``sample_every_hours``, greater than ``min_hours`` and not after the end
    of the stay or the time of death. Stays without any measurement before
    ``t`` give no sample at ``t``. Stay exclusions are those of
    :class:`~pyhealth.tasks.EHRCXRMortalityMIMIC4`.

    ``decompensation``: 1 if the patient dies in hospital less than
    ``horizon_hours`` after ``t``, else 0 (``hospital_expire_flag`` and
    ``deathtime`` of the stay's admission; a death without ``deathtime``
    counts as 0, as in mimic3-benchmarks).

    ``ehr``: the 17 MedFuse variables of
    :class:`~pyhealth.tasks.EHRCXRMortalityMIMIC4` and their masks, in
    ``bin_hours`` bins over the ``window_hours`` before ``t``:
    ``[window_hours / bin_hours, 34]``. Only ICU data of the stay is used:
    labs from before ICU admission are never used. The one exception is chart
    values recorded under the stay's ``stay_id`` before admission, however
    early (MIMIC-IV charts admission height and weight just before
    ``intime``), as in the mortality task:

    - While ``t <= window_hours``, the window starts at or before admission.
      These chart values count as measured, in the bin of their own time, or
      in the first bin if they are older than the window. Bins start at
      ``t - window_hours``, so when ``t`` is a multiple of ``bin_hours`` (as
      with the defaults) a bin edge falls on admission and these values sit
      in bins before it; otherwise the bin that spans admission can hold such
      a value together with the first values after admission. All other
      bins before admission are empty.
    - Once ``t > window_hours``, the window starts after admission. The last
      value of each variable before the window, chart or lab, is carried into
      the first bin with mask 0.

    ``cxr``: the last image with a view in ``view_positions`` taken between
    ICU admission and ``t`` (inclusive). Images taken after ``t`` never enter
    the sample. Same-time images go to the highest ``dicom_id``.

    ``cxr_mask``: 1 with such an image, else 0 with a zero ``cxr``. With
    ``paired_only``, prediction times without an image are dropped.

    Requires the ``patients``, ``admissions``, ``icustays``, ``chartevents``
    and ``labevents`` EHR tables and the ``metadata`` CXR table.

    Args:
        image_size: Side of the square image.
        view_positions: Accepted ``ViewPosition`` values.
        bin_hours: Width of a time bin in hours.
        window_hours: Length of the EHR window before ``t``; a multiple of
            ``bin_hours``.
        sample_every_hours: Hours between prediction times (MedMod uses 1).
        min_hours: Prediction times must be later than this (4 in MedMod).
        horizon_hours: Prediction horizon (24 in MedMod).
        paired_only: Whether to drop prediction times without an image.

    Examples:
        >>> from pyhealth.tasks import MedModDecompensationMIMIC4
        >>> task = MedModDecompensationMIMIC4(sample_every_hours=4)
        >>> sorted(task.input_schema), task.output_schema
        (['cxr', 'cxr_mask', 'ehr'], {'decompensation': 'binary'})
        >>> from pyhealth.datasets import MIMIC4Dataset, PatientSplit
        >>> dataset = MIMIC4Dataset(  # doctest: +SKIP
        ...     ehr_root="/path/to/mimiciv/3.1",
        ...     cxr_root="/path/to/mimic-cxr-jpg/2.1.0",
        ...     ehr_tables=["chartevents", "labevents"],
        ...     cxr_tables=["metadata"],
        ... )
        >>> train, val, test = dataset.set_task(  # doctest: +SKIP
        ...     task, split=PatientSplit(ratios=(0.7, 0.1, 0.2), seed=0)
        ... )
    """

    task_name: str = "MedModDecompensationMIMIC4"
    output_schema: ClassVar[dict[str, Any]] = {"decompensation": "binary"}

    FEATURES: ClassVar[list[str]] = EHRCXRMortalityMIMIC4.FEATURES

    def __init__(
        self,
        image_size: int = 224,
        view_positions: Sequence[str] = ("AP",),
        bin_hours: int = 2,
        window_hours: int = 48,
        sample_every_hours: int = 4,
        min_hours: float = 4.0,
        horizon_hours: float = 24.0,
        paired_only: bool = False,
    ) -> None:
        if bin_hours <= 0 or window_hours <= 0 or sample_every_hours <= 0:
            raise ValueError(
                "bin_hours, window_hours and sample_every_hours must be positive."
            )
        if window_hours % bin_hours != 0:
            raise ValueError("window_hours must be a multiple of bin_hours.")
        if horizon_hours <= 0 or min_hours < 0:
            raise ValueError("horizon_hours must be positive and min_hours >= 0.")
        self.input_schema = _input_schema(image_size)
        super().__init__()
        self.view_positions = list(view_positions)
        self.bin_hours = bin_hours
        self.window_hours = window_hours
        self.sample_every_hours = sample_every_hours
        self.min_hours = min_hours
        self.horizon_hours = horizon_hours
        self.paired_only = paired_only

    def pre_filter(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Drops measurement rows whose item ID is not used by the task."""
        return _pre_filter(df, self.task_name, [*_EVENT_TABLES, "metadata"])

    def prediction_hours(self, last_hour: float) -> list[int]:
        """Prediction times (hours since ICU admission) for a stay.

        Args:
            last_hour: Hours from ICU admission to the end of the stay or to
                death, whichever comes first.

        Returns:
            The multiples of ``sample_every_hours`` in
            ``(min_hours, last_hour]``.
        """
        step = self.sample_every_hours
        return [
            t
            for t in range(step, int(last_hour + 1e-6) + 1, step)
            if t > self.min_hours and t <= last_hour + 1e-6
        ]

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Builds the samples of one patient.

        Args:
            patient: The patient.

        Returns:
            One sample per eligible stay and prediction time, with
            ``patient_id``, ``visit_id``, ``stay_id``, ``hour``, ``ehr``,
            ``cxr`` (a path or None), ``cxr_mask`` and ``decompensation``.
        """
        stays = _eligible_stays(patient)
        if stays.height == 0:
            return []
        measurements = _measurements(patient)
        cxrs = patient.get_events(event_type="metadata", return_df=True)
        admissions = {
            int(event.hadm_id): (
                int(event.hospital_expire_flag),
                _parse(getattr(event, "deathtime", None)),
            )
            for event in patient.get_events(event_type="admissions")
            if event.hospital_expire_flag is not None
        }

        window = timedelta(hours=self.window_hours)
        samples = []
        for stay_id, hadm_id, intime, outtime in stays.iter_rows():
            if hadm_id not in admissions:
                continue
            expired, deathtime = admissions[hadm_id]
            stay_hours = (outtime - intime).total_seconds() / 3600
            lived_hours = float("inf")
            if expired and deathtime is not None:
                lived_hours = (deathtime - intime).total_seconds() / 3600
            icu_rows = measurements.filter(
                (pl.col("stay_id") == stay_id) | (pl.col("timestamp") >= intime)
            )
            for hour in self.prediction_hours(min(stay_hours, lived_hours)):
                end = intime + timedelta(hours=hour)
                ehr = _window_series(
                    icu_rows, stay_id, intime, end - window, end, self.bin_hours
                )
                if ehr is None:
                    continue
                cxr = _last_cxr(cxrs, intime, end, self.view_positions)
                if self.paired_only and cxr is None:
                    continue
                label = int(expired == 1 and lived_hours - hour < self.horizon_hours)
                samples.append(
                    {
                        "patient_id": patient.patient_id,
                        "visit_id": str(hadm_id),
                        "stay_id": stay_id,
                        "hour": hour,
                        "ehr": ehr,
                        "cxr": cxr,
                        "cxr_mask": float(cxr is not None),
                        "decompensation": label,
                    }
                )
        return samples
