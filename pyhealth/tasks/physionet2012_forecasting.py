"""Irregular time-series forecasting on the PhysioNet 2012 ICU data.

The setup of the irregular multivariate time-series forecasting benchmarks
(e.g. GraFITi, Warpformer and APN, Liu et al., AAAI 2026, "Rethinking
Irregular Time Series Forecasting: A Simple yet Effective Baseline"): from
the first 36 hours of an ICU stay, forecast the values observed at the next
three observation times.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import polars as pl

from .base_task import BaseTask

if TYPE_CHECKING:
    from ..datasets import BaseDataset, SampleDataset

TIME_SERIES_VARIABLES = (
    "Albumin", "ALP", "ALT", "AST", "Bilirubin", "BUN", "Cholesterol",
    "Creatinine", "DiasABP", "FiO2", "GCS", "Glucose", "HCO3", "HCT", "HR", "K",
    "Lactate", "Mg", "MAP", "MechVent", "Na", "NIDiasABP", "NIMAP", "NISysABP",
    "PaCO2", "PaO2", "pH", "Platelets", "RespRate", "SaO2", "SysABP", "Temp",
    "TroponinI", "TroponinT", "Urine", "WBC",
)
"""The 36 time series used by the irregular-forecasting benchmarks.

Weight is also recorded as a time series, but the benchmarks (and
:class:`PhysioNet2012Forecasting`) leave it out.
"""


class PhysioNet2012Forecasting(BaseTask):
    """Forecast the next observed values of 36 ICU time series.

    Works with :class:`~pyhealth.datasets.PhysioNet2012Dataset`.

    The observations of each variable are averaged within each hour since
    ICU admission, z-scored with the given statistics, and values more than
    ``outlier_sd`` standard deviations from the mean are dropped. The hours
    before ``observation_hours`` form the input, a grid of
    ``observation_hours x len(variables)`` values with NaN where nothing was
    observed. The targets are the next ``prediction_steps`` hours, at or after
    ``observation_hours``, in which any variable was observed, again with NaN
    for the unobserved variables. Times are given in units of
    ``total_hours`` (hour / 48), as in the benchmarks.

    This follows the PhysioNet 2012 forecasting setup of the benchmarks
    (observation window of 36 hours, 3 forecast steps, hourly bins, z-scored
    values with outliers beyond 5 SD removed; see
    :func:`physionet2012_statistics` and :func:`physionet2012_benchmark_split`
    for the normalization and the split).

    Samples have ``x`` (``[observation_hours, V]``), ``x_time``
    (``[observation_hours]``), ``y_time`` (``[prediction_steps]``) and the
    label ``y`` (``[prediction_steps, V]``); a missing step is all NaN with
    time 0. Stays without an observed input value or without a target are
    skipped. Models compute the loss over the observed (finite) targets
    only, e.g. :class:`~pyhealth.models.APN`.

    Args:
        mean: Per-variable mean used for z-scoring, e.g. from
            :func:`physionet2012_statistics`. ``None`` keeps raw values.
        std: Per-variable standard deviation; required with ``mean``.
        observation_hours: Hours of the stay used as input.
        prediction_steps: Number of future observation times to forecast.
        variables: Time series to use, in the order of the feature axis.
        outlier_sd: Drop z-scores at or beyond this magnitude; ``None``
            keeps all values. Only applies with ``mean`` and ``std``.
        total_hours: Scale of the times (``hour / total_hours``).

    Examples:
        >>> from pyhealth.datasets import PhysioNet2012Dataset
        >>> from pyhealth.tasks import (
        ...     PhysioNet2012Forecasting, physionet2012_statistics
        ... )
        >>> dataset = PhysioNet2012Dataset(root="data/physionet2012")  # doctest: +SKIP
        >>> mean, std = physionet2012_statistics(dataset)  # doctest: +SKIP
        >>> samples = dataset.set_task(
        ...     PhysioNet2012Forecasting(mean=mean, std=std)
        ... )  # doctest: +SKIP
    """

    task_name: str = "PhysioNet2012Forecasting"
    input_schema: ClassVar[dict[str, str]] = {
        "x": "tensor", "x_time": "tensor", "y_time": "tensor"
    }
    output_schema: ClassVar[dict[str, str]] = {"y": "tensor"}

    def __init__(
        self,
        mean: Mapping[str, float] | None = None,
        std: Mapping[str, float] | None = None,
        observation_hours: int = 36,
        prediction_steps: int = 3,
        variables: Sequence[str] = TIME_SERIES_VARIABLES,
        outlier_sd: float | None = 5.0,
        total_hours: float = 48.0,
    ) -> None:
        super().__init__()
        if (mean is None) != (std is None):
            raise ValueError("Give both mean and std, or neither.")
        self.variables = list(variables)
        self.mean = None if mean is None else {v: float(mean[v]) for v in self.variables}
        self.std = None if std is None else {v: float(std[v]) for v in self.variables}
        self.observation_hours = int(observation_hours)
        self.prediction_steps = int(prediction_steps)
        self.outlier_sd = outlier_sd
        self.total_hours = float(total_hours)

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        events = patient.get_events(event_type="events", return_df=True)
        if events.height == 0:
            return []
        hourly = _hourly_means(events, self.variables).sort("hour", "parameter")
        if hourly.height == 0:
            return []
        index = {v: i for i, v in enumerate(self.variables)}
        hours = hourly["hour"].to_numpy()
        columns = np.array([index[p] for p in hourly["parameter"].to_list()])
        values = hourly["value"].to_numpy().astype(np.float64)
        if self.mean is not None:
            center = np.array([self.mean[v] for v in self.variables])
            scale = np.array([self.std[v] for v in self.variables])
            values = (values - center[columns]) / scale[columns]
            if self.outlier_sd is not None:
                keep = np.abs(values) < self.outlier_sd
                hours, columns, values = hours[keep], columns[keep], values[keep]

        n_vars = len(self.variables)
        x = np.full((self.observation_hours, n_vars), np.nan, dtype=np.float32)
        observed = hours < self.observation_hours
        x[hours[observed], columns[observed]] = values[observed]
        if not np.isfinite(x).any():
            return []

        future_hours = np.unique(hours[~observed])[: self.prediction_steps]
        if len(future_hours) == 0:
            return []
        y = np.full((self.prediction_steps, n_vars), np.nan, dtype=np.float32)
        y_time = np.zeros(self.prediction_steps, dtype=np.float32)
        for step, hour in enumerate(future_hours):
            at = hours == hour
            y[step, columns[at]] = values[at]
            y_time[step] = hour / self.total_hours
        x_time = np.arange(self.observation_hours, dtype=np.float32) / self.total_hours
        return [{
            "patient_id": patient.patient_id,
            "x": x,
            "x_time": x_time,
            "y_time": y_time,
            "y": y,
        }]


def _hourly_means(
    events: pl.DataFrame | pl.LazyFrame,
    variables: Sequence[str],
    by: Sequence[str] = (),
) -> pl.DataFrame | pl.LazyFrame:
    """Mean value per (``by`` columns, hour since admission, variable).

    Shared by the task and :func:`physionet2012_statistics` so both bin the
    same way. Works on an eager or a lazy frame of ``events`` rows.
    """
    keys = [*by, "hour", "parameter"]
    return (
        events.select(
            *by,
            (pl.col("events/minutes").cast(pl.Int64) // 60).alias("hour"),
            pl.col("events/parameter").alias("parameter"),
            pl.col("events/value").cast(pl.Float64).alias("value"),
        )
        .filter(pl.col("parameter").is_in(list(variables)))
        .group_by(keys)
        .agg(pl.col("value").mean())
    )


def physionet2012_statistics(
    dataset: BaseDataset,
    patient_ids: Sequence[str] | None = None,
    variables: Sequence[str] = TIME_SERIES_VARIABLES,
) -> tuple[dict[str, float], dict[str, float]]:
    """Per-variable mean and standard deviation of the hourly means.

    The forecasting benchmarks standardize with statistics over all records,
    including the test records; pass the training ``patient_ids`` to use the
    training records only.

    Args:
        dataset: A :class:`~pyhealth.datasets.PhysioNet2012Dataset`.
        patient_ids: Records to compute the statistics on; all if ``None``.
        variables: The time series to describe.

    Returns:
        ``(mean, std)``, each a dict keyed by variable (population standard
        deviation, ``ddof=0``), rounded to 6 decimals.

    Examples:
        >>> mean, std = physionet2012_statistics(dataset)  # doctest: +SKIP
        >>> round(mean["HR"], 1)  # doctest: +SKIP
        86.8
    """
    events = dataset.global_event_df.filter(pl.col("event_type") == "events")
    if patient_ids is not None:
        events = events.filter(pl.col("patient_id").is_in(list(patient_ids)))
    hourly = (
        _hourly_means(events, variables, by=["patient_id"])
        .group_by("parameter")
        .agg(
            pl.col("value").mean().alias("mean"),
            pl.col("value").std(ddof=0).alias("std"),
        )
        .collect()
    )
    stats = {row["parameter"]: row for row in hourly.iter_rows(named=True)}
    missing = [v for v in variables if v not in stats]
    if missing:
        raise ValueError(f"No observations of {missing}")
    # Rounded so repeated runs give the same task (and reuse its cache): the
    # parallel aggregation can change the last digits.
    mean = {v: round(float(stats[v]["mean"]), 6) for v in variables}
    std = {v: round(float(stats[v]["std"]), 6) or 1.0 for v in variables}
    return mean, std


def physionet2012_benchmark_split(
    dataset: SampleDataset, test_size: float = 0.1, val_size: float = 0.1
) -> tuple[SampleDataset, SampleDataset, SampleDataset]:
    """The deterministic split of the forecasting benchmarks.

    Records are ordered by ``RecordID`` (sets a, b, c); the last
    ``test_size`` of them are the test set, and the last ``val_size`` of the
    rest the validation set, as ``sklearn.model_selection.train_test_split``
    with ``shuffle=False`` does in the benchmark code.

    Args:
        dataset: Samples from :class:`PhysioNet2012Forecasting`.
        test_size: Fraction of records for testing.
        val_size: Fraction of the remaining records for validation.

    Returns:
        ``(train, val, test)`` views of ``dataset``.

    Examples:
        >>> train, val, test = physionet2012_benchmark_split(samples)  # doctest: +SKIP
    """
    ids = sorted(dataset.patient_to_index, key=int)
    n_test = math.ceil(test_size * len(ids))
    rest, test_ids = ids[: len(ids) - n_test], ids[len(ids) - n_test :]
    n_val = math.ceil(val_size * len(rest))
    train_ids, val_ids = rest[: len(rest) - n_val], rest[len(rest) - n_val :]

    def subset(part: list[str]) -> SampleDataset:
        return dataset.subset(
            [i for pid in part for i in dataset.patient_to_index[pid]]
        )

    return subset(train_ids), subset(val_ids), subset(test_ids)
