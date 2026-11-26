"""In-hospital mortality prediction on the PhysioNet 2012 ICU data.

The irregular time-series classification setup of Raindrop (Zhang et al.,
ICLR 2022, "Graph-Guided Network for Irregularly Sampled Multivariate Time
Series"), also used by ViTST (Li et al., NeurIPS 2023) and the joint
sequence-image model of Chen et al. (AAAI 2025): from the 36 time series of
the first 48 hours of an ICU stay, kept at their own observation times,
predict in-hospital death.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import polars as pl

from .base_task import BaseTask
from .physionet2012_forecasting import TIME_SERIES_VARIABLES

if TYPE_CHECKING:
    from ..datasets import BaseDataset, SampleDataset

MAX_OBSERVATION_TIMES = 215
"""Most distinct observation times of a PhysioNet 2012 stay (Raindrop's 215)."""


class PhysioNet2012Mortality(BaseTask):
    """Predict in-hospital death from the irregular ICU time series.

    Works with :class:`~pyhealth.datasets.PhysioNet2012Dataset`.

    Each stay gives one sample. Its observations within the first
    ``total_hours`` hours are kept at their own times (no binning): the
    distinct observation times, in hours since ICU admission, form
    ``x_time``, and ``x`` holds the value of each variable at each of those
    times (the mean of repeated measurements at the same minute), NaN where
    the variable was not measured. Both are padded to ``max_length`` times
    with NaN, so every sample has the same shape; models find the padding
    from the non-finite times. Stays with more times keep the first
    ``max_length``.

    Values can be z-scored with statistics from
    :func:`physionet2012_mortality_statistics`, and values more than
    ``outlier_sd`` standard deviations from the mean (e.g. a pH of 735)
    dropped.

    Stays without any time-series observation (the 12 records Raindrop and
    SeFT remove) or without an outcome (set c has none in some releases)
    are skipped, which leaves Raindrop's 11,988 stays.

    Samples have ``x`` (``[max_length, V]``), ``x_time`` (``[max_length]``)
    and the binary label ``mortality``.

    Args:
        mean: Per-variable mean used for z-scoring. ``None`` keeps raw values.
        std: Per-variable standard deviation; required with ``mean``.
        variables: Time series to use, in the order of the feature axis.
            Defaults to the 36 of Raindrop (Weight, also recorded on
            admission, is left out).
        max_length: Number of observation times per sample.
        total_hours: Observations at or after this many hours are dropped.
        outlier_sd: Drop z-scores at or beyond this magnitude; ``None``
            keeps all values. Only applies with ``mean`` and ``std``.

    Examples:
        >>> from pyhealth.datasets import PhysioNet2012Dataset
        >>> from pyhealth.tasks import (
        ...     PhysioNet2012Mortality, physionet2012_mortality_statistics
        ... )
        >>> dataset = PhysioNet2012Dataset(root="data/physionet2012")  # doctest: +SKIP
        >>> mean, std = physionet2012_mortality_statistics(dataset)  # doctest: +SKIP
        >>> samples = dataset.set_task(
        ...     PhysioNet2012Mortality(mean=mean, std=std)
        ... )  # doctest: +SKIP
    """

    task_name: str = "PhysioNet2012Mortality"
    input_schema: ClassVar[dict[str, str]] = {"x": "tensor", "x_time": "tensor"}
    output_schema: ClassVar[dict[str, str]] = {"mortality": "binary"}

    def __init__(
        self,
        mean: Mapping[str, float] | None = None,
        std: Mapping[str, float] | None = None,
        variables: Sequence[str] = TIME_SERIES_VARIABLES,
        max_length: int = MAX_OBSERVATION_TIMES,
        total_hours: float = 48.0,
        outlier_sd: float | None = None,
    ) -> None:
        super().__init__()
        if (mean is None) != (std is None):
            raise ValueError("Give both mean and std, or neither.")
        self.variables = list(variables)
        self.mean = None if mean is None else {v: float(mean[v]) for v in self.variables}
        self.std = None if std is None else {v: float(std[v]) for v in self.variables}
        self.max_length = int(max_length)
        self.total_hours = float(total_hours)
        self.outlier_sd = outlier_sd

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        info = patient.get_events(event_type="patients")
        if not info:
            return []
        death = info[0].attr_dict.get("in_hospital_death")
        if death is None or not np.isfinite(float(death)):
            return []
        events = patient.get_events(event_type="events", return_df=True)
        if events.height == 0:
            return []
        observed = _observations(events, self.variables, self.total_hours)
        if observed.height == 0:
            return []

        index = {v: i for i, v in enumerate(self.variables)}
        minutes = observed["minutes"].to_numpy()
        columns = np.array([index[p] for p in observed["parameter"].to_list()])
        values = observed["value"].to_numpy().astype(np.float64)
        if self.mean is not None:
            center = np.array([self.mean[v] for v in self.variables])
            scale = np.array([self.std[v] for v in self.variables])
            values = (values - center[columns]) / scale[columns]
            if self.outlier_sd is not None:
                keep = np.abs(values) < self.outlier_sd
                minutes, columns, values = minutes[keep], columns[keep], values[keep]
            if len(values) == 0:
                return []

        times, rows = np.unique(minutes, return_inverse=True)
        times, keep = times[: self.max_length], rows < self.max_length
        x = np.full((self.max_length, len(self.variables)), np.nan, dtype=np.float32)
        x[rows[keep], columns[keep]] = values[keep]
        x_time = np.full(self.max_length, np.nan, dtype=np.float32)
        x_time[: len(times)] = times / 60.0
        return [{
            "patient_id": patient.patient_id,
            "x": x,
            "x_time": x_time,
            "mortality": int(float(death)),
        }]


def _observations(
    events: pl.DataFrame | pl.LazyFrame,
    variables: Sequence[str],
    total_hours: float,
    by: Sequence[str] = (),
) -> pl.DataFrame | pl.LazyFrame:
    """Mean value per (``by`` columns, minute since admission, variable).

    Shared by the task and :func:`physionet2012_mortality_statistics` so both
    see the same observations.
    """
    keys = [*by, "minutes", "parameter"]
    return (
        events.select(
            *by,
            pl.col("events/minutes").cast(pl.Int64).alias("minutes"),
            pl.col("events/parameter").alias("parameter"),
            pl.col("events/value").cast(pl.Float64).alias("value"),
        )
        .filter(
            pl.col("parameter").is_in(list(variables))
            & (pl.col("minutes") < total_hours * 60)
        )
        .group_by(keys)
        .agg(pl.col("value").mean())
        .sort(keys)
    )


def physionet2012_mortality_statistics(
    dataset: BaseDataset,
    patient_ids: Sequence[str] | None = None,
    variables: Sequence[str] = TIME_SERIES_VARIABLES,
    total_hours: float = 48.0,
) -> tuple[dict[str, float], dict[str, float]]:
    """Per-variable mean and standard deviation of the observed values.

    Pass the training ``patient_ids`` so the test stays don't inform the
    normalization.

    Args:
        dataset: A :class:`~pyhealth.datasets.PhysioNet2012Dataset`.
        patient_ids: Stays to compute the statistics on; all if ``None``.
        variables: The time series to describe.
        total_hours: Observations at or after this many hours are ignored.

    Returns:
        ``(mean, std)``, each a dict keyed by variable (population standard
        deviation, ``ddof=0``; 1 where it is 0), rounded to 6 decimals.

    Examples:
        >>> mean, std = physionet2012_mortality_statistics(dataset)  # doctest: +SKIP
        >>> round(mean["HR"], 1)  # doctest: +SKIP
        87.4
    """
    events = dataset.global_event_df.filter(pl.col("event_type") == "events")
    if patient_ids is not None:
        events = events.filter(pl.col("patient_id").is_in(list(patient_ids)))
    stats = (
        _observations(events, variables, total_hours, by=["patient_id"])
        .group_by("parameter")
        .agg(
            pl.col("value").mean().alias("mean"),
            pl.col("value").std(ddof=0).alias("std"),
        )
        .collect()
    )
    stats = {row["parameter"]: row for row in stats.iter_rows(named=True)}
    missing = [v for v in variables if v not in stats]
    if missing:
        raise ValueError(f"No observations of {missing}")
    # Rounded so repeated runs give the same task (and reuse its cache).
    mean = {v: round(float(stats[v]["mean"]), 6) for v in variables}
    std = {v: round(float(stats[v]["std"]), 6) or 1.0 for v in variables}
    return mean, std


def split_by_record_ids(
    dataset: SampleDataset,
    train_ids: Sequence[str],
    val_ids: Sequence[str],
    test_ids: Sequence[str],
) -> tuple[SampleDataset, SampleDataset, SampleDataset]:
    """Splits samples by given record IDs, e.g. a published split.

    IDs without a sample are ignored, so a split of all stays also works on
    a subset of them.

    Args:
        dataset: Samples keyed by ``patient_id`` (the PhysioNet RecordID).
        train_ids: Records of the training set.
        val_ids: Records of the validation set.
        test_ids: Records of the test set.

    Returns:
        ``(train, val, test)`` views of ``dataset``.

    Raises:
        ValueError: If a record is in more than one part.

    Examples:
        >>> train, val, test = split_by_record_ids(
        ...     samples, ["132539"], ["132540"], ["132541"]
        ... )  # doctest: +SKIP
    """
    parts = [{str(i) for i in ids} for ids in (train_ids, val_ids, test_ids)]
    if parts[0] & parts[1] or parts[0] & parts[2] or parts[1] & parts[2]:
        raise ValueError("A record is in more than one part of the split.")
    index = dataset.patient_to_index

    def subset(ids: set[str]) -> SampleDataset:
        return dataset.subset(
            [i for pid in sorted(ids, key=str) if pid in index for i in index[pid]]
        )

    return subset(parts[0]), subset(parts[1]), subset(parts[2])
