"""MIMIC-III irregular time-series forecasting used by APN and GRU-ODE-Bayes.

Contributor: Arslan Shafique
Paper: "Rethinking Irregular Time Series Forecasting: A Simple yet Effective
Baseline" (Liu et al., AAAI 2026), https://arxiv.org/abs/2505.11250

The preprocessing follows the MIMIC-III benchmark introduced with
GRU-ODE-Bayes: 96 longitudinal variables are aggregated into 30-minute bins
during the first 48 hours, the first 36 hours are observed, and the next three
times containing a measurement are forecast.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import polars as pl

from .base_task import BaseTask

if TYPE_CHECKING:
    from ..datasets import BaseDataset

logger = logging.getLogger(__name__)

LAB_VARIABLES = (
    "Albumin",
    "Alanine Aminotransferase (ALT)",
    "Alkaline Phosphatase",
    "Anion Gap",
    "Asparate Aminotransferase (AST)",
    "Base Excess",
    "Basophils",
    "Bicarbonate",
    "Bilirubin, Total",
    "Calcium, Total",
    "Calculated Total CO2",
    "Chloride",
    "Creatinine",
    "Eosinophils",
    "Glucose",
    "Hematocrit",
    "Hemoglobin",
    "Lactate",
    "Lymphocytes",
    "MCH",
    "MCHC",
    "MCV",
    "Magnesium",
    "Monocytes",
    "Neutrophils",
    "PT",
    "PTT",
    "Phosphate",
    "Platelet Count",
    "Potassium",
    "RDW",
    "Red Blood Cells",
    "Sodium",
    "Specific Gravity",
    "Urea Nitrogen",
    "White Blood Cells",
    "pCO2",
    "pH",
    "pO2",
)

INPUT_VARIABLES = (
    "Albumin 5%",
    "Dextrose 5%",
    "Lorazepam (Ativan)",
    "Calcium Gluconate",
    "Midazolam (Versed)",
    "Phenylephrine",
    "Furosemide (Lasix)",
    "Hydralazine",
    "Norepinephrine",
    "Magnesium Sulfate",
    "Nitroglycerin",
    "Insulin - Glargine",
    "Insulin - Humalog",
    "Insulin - Regular",
    "Heparin Sodium",
    "Morphine Sulfate",
    "Potassium Chloride",
    "Packed Red Blood Cells",
    "Gastric Meds",
    "D5 1/2NS",
    "LR",
    "K Phos",
    "Solution",
    "Sterile Water",
    "Metoprolol",
    "Piggyback",
    "OR Crystalloid Intake",
    "OR Cell Saver Intake",
    "PO Intake",
    "GT Flush",
    "KCL (Bolus)",
    "Magnesium Sulfate (Bolus)",
)

OUTPUT_VARIABLES = (
    "Gastric Gastric Tube",
    "Stool Out Stool",
    "Urine Out Incontinent",
    "Ultrafiltrate Ultrafiltrate",
    "Foley",
    "Void",
    "Condom Cath",
    "Fecal Bag",
    "Ostomy (output)",
    "Chest Tube #1",
    "Chest Tube #2",
    "Jackson Pratt #1",
    "OR EBL",
    "Pre-Admission",
    "TF Residual",
)

PRESCRIPTION_VARIABLES = (
    "Aspirin",
    "Bisacodyl",
    "Docusate Sodium",
    "D5W",
    "Humulin-R Insulin",
    "Potassium Chloride",
    "Magnesium Sulfate",
    "Metoprolol Tartrate",
    "Sodium Chloride 0.9%  Flush",
    "Pantoprazole",
)

MIMIC3_FORECASTING_VARIABLES = tuple(
    [*(f"lab:{name}" for name in LAB_VARIABLES)]
    + [*(f"input:{name}" for name in INPUT_VARIABLES)]
    + [*(f"output:{name}" for name in OUTPUT_VARIABLES)]
    + [*(f"prescription:{name}" for name in PRESCRIPTION_VARIABLES)]
)
"""The 96 variables in the MIMIC-III forecasting benchmark."""

_TABLE_SPECS = {
    "labevents": ("lab", "label", "valuenum", "mean"),
    "inputevents_mv": ("input", "label", "amount", "sum"),
    "outputevents": ("output", "label", "value", "sum"),
    "prescriptions": ("prescription", "drug", "dose_val_rx", "sum"),
}


def _parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if value is None:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


def _number(value: Any) -> float | None:
    """Converts a scalar or prescription range such as ``"10-20"``."""
    if value is None:
        return None
    text = str(value).strip()
    if "-" in text and not text.startswith("-"):
        parts = text.split("-", 1)
        try:
            left = float(parts[0])
            right = float(parts[1] or parts[0])
            return (left + right) / 2.0
        except ValueError:
            return None
    try:
        number = float(text)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


class MIMIC3Forecasting(BaseTask):
    """Forecast 96 MIMIC-III variables after a 36-hour observation window.

    Works with :class:`~pyhealth.datasets.MIMIC3Dataset` loaded with
    ``labevents``, ``inputevents_mv``, ``outputevents`` and ``prescriptions``.
    It follows the cohort and forecasting protocol of De Brouwer et al.'s
    GRU-ODE-Bayes benchmark, which is also the MIMIC-III benchmark used by
    APN. Measurements are aggregated into half-hour bins: laboratory values
    are averaged, while treatments, outputs and prescriptions are summed.

    The feature axis contains the 39 laboratory, 32 treatment, 15 output and
    10 prescription variables in :data:`MIMIC3_FORECASTING_VARIABLES`.
    ``x`` has shape ``[72, V]`` and covers the first 36 hours. ``y`` has
    shape ``[prediction_steps, V]`` and contains the next observed time bins.
    Missing values remain NaN, and times are divided by 96 (the number of
    half-hour bins in 48 hours), matching the reference benchmark.

    Args:
        mean: Per-variable mean for z-scoring. ``None`` keeps raw values.
        std: Per-variable standard deviation; required with ``mean``.
        variables: Feature names to include. The default is all 96 benchmark
            variables; a subset is useful for focused experiments and tests.
        observation_hours: Length of the observed window in hours.
        total_hours: Total extraction window and time-normalization scale.
        prediction_steps: Number of future observed time bins to forecast.
        bin_minutes: Width of one time bin.
        outlier_sd: Drop normalized values at or beyond this magnitude.
        min_observations: Minimum number of finite cells in the 48-hour grid.
        benchmark_cohort: Apply the paper's single-admission, age, stay-length,
            chart-data and non-newborn cohort filters.

    Examples:
        >>> from pyhealth.datasets import MIMIC3Dataset
        >>> from pyhealth.tasks import MIMIC3Forecasting
        >>> dataset = MIMIC3Dataset(  # doctest: +SKIP
        ...     root="/path/to/mimic-iii/1.4",
        ...     tables=["labevents", "inputevents_mv", "outputevents",
        ...             "prescriptions"],
        ... )
        >>> samples = dataset.set_task(  # doctest: +SKIP
        ...     MIMIC3Forecasting(mean=mean, std=std)
        ... )
    """

    task_name: str = "MIMIC3Forecasting"
    input_schema: ClassVar[dict[str, str]] = {
        "x": "tensor",
        "x_time": "tensor",
        "y_time": "tensor",
    }
    output_schema: ClassVar[dict[str, str]] = {"y": "tensor"}

    def __init__(
        self,
        mean: Mapping[str, float] | None = None,
        std: Mapping[str, float] | None = None,
        variables: Sequence[str] = MIMIC3_FORECASTING_VARIABLES,
        observation_hours: float = 36.0,
        total_hours: float = 48.0,
        prediction_steps: int = 3,
        bin_minutes: int = 30,
        outlier_sd: float | None = 5.0,
        min_observations: int = 50,
        benchmark_cohort: bool = True,
    ) -> None:
        super().__init__()
        if (mean is None) != (std is None):
            raise ValueError("Give both mean and std, or neither.")
        unknown = sorted(set(variables) - set(MIMIC3_FORECASTING_VARIABLES))
        if unknown:
            raise ValueError(f"Unknown MIMIC-III forecasting variables: {unknown}")
        if bin_minutes <= 0 or 60 % bin_minutes:
            raise ValueError("bin_minutes must be a positive divisor of 60.")
        self.variables = list(variables)
        self.mean = (
            None
            if mean is None
            else {name: float(mean[name]) for name in self.variables}
        )
        self.std = (
            None if std is None else {name: float(std[name]) for name in self.variables}
        )
        if self.std is not None and any(value <= 0 for value in self.std.values()):
            raise ValueError("Every standard deviation must be positive.")
        self.observation_hours = float(observation_hours)
        self.total_hours = float(total_hours)
        self.prediction_steps = int(prediction_steps)
        self.bin_minutes = int(bin_minutes)
        self.outlier_sd = outlier_sd
        self.min_observations = int(min_observations)
        self.benchmark_cohort = bool(benchmark_cohort)
        self._variable_index = {name: i for i, name in enumerate(self.variables)}

    @property
    def bins_per_hour(self) -> int:
        return 60 // self.bin_minutes

    @property
    def observation_bins(self) -> int:
        return round(self.observation_hours * self.bins_per_hour)

    @property
    def total_bins(self) -> int:
        return round(self.total_hours * self.bins_per_hour)

    def pre_filter(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Warns once when a required source table was not loaded."""
        columns = df.collect_schema().names()
        missing = [
            table
            for table in _TABLE_SPECS
            if not any(name.startswith(f"{table}/") for name in columns)
        ]
        if missing:
            logger.warning(
                "%s: source table(s) %s are not loaded and will add no values.",
                self.task_name,
                missing,
            )
        return df

    def _eligible_admissions(self, patient: Any) -> list[Any]:
        admissions = patient.get_events(event_type="admissions")
        if not self.benchmark_cohort:
            return admissions
        if len(admissions) != 1:
            return []
        admission = admissions[0]
        discharge = _parse_datetime(getattr(admission, "dischtime", None))
        if admission.timestamp is None or discharge is None:
            return []
        stay_days = (discharge - admission.timestamp).total_seconds() / 86400.0
        if not (3.0 <= stay_days < 30.0):
            return []
        if str(getattr(admission, "admission_type", "")).upper() == "NEWBORN":
            return []
        chart_data = getattr(admission, "has_chartevents_data", None)
        if chart_data is not None and str(chart_data) not in {"1", "1.0"}:
            return []
        demographics = patient.get_events(event_type="patients")
        if demographics:
            birth = _parse_datetime(getattr(demographics[0], "dob", None))
            if birth is not None:
                age = (admission.timestamp - birth).total_seconds() / 31557600.0
                if age <= 15.0:
                    return []
        return admissions

    def _table_points(
        self, patient: Any, table: str, hadm_id: str
    ) -> list[tuple[datetime, int, float, str]]:
        origin, label_attr, value_attr, aggregation = _TABLE_SPECS[table]
        frame = patient.get_events(event_type=table, return_df=True)
        if frame.height == 0:
            return []
        hadm_column = f"{table}/hadm_id"
        if hadm_column in frame.columns:
            frame = frame.filter(pl.col(hadm_column).cast(pl.String) == hadm_id)
        points = []
        for row in frame.iter_rows(named=True):
            timestamp = row.get("timestamp")
            label = row.get(f"{table}/{label_attr}")
            value = _number(row.get(f"{table}/{value_attr}"))
            key = f"{origin}:{label}"
            if (
                not isinstance(timestamp, datetime)
                or value is None
                or key not in self._variable_index
            ):
                continue
            times = [timestamp]
            values = [value]
            if table == "inputevents_mv":
                end = _parse_datetime(row.get("inputevents_mv/endtime"))
                if value < 0:
                    value = abs(value)
                    values = [value]
                if end is not None and end < timestamp:
                    timestamp, end = end, timestamp
                    times = [timestamp]
                if end is not None and end > timestamp:
                    duration = (end - timestamp).total_seconds() / 60.0
                    repeats = max(1, math.ceil(duration / self.bin_minutes))
                    times = [
                        timestamp + timedelta(minutes=i * self.bin_minutes)
                        for i in range(repeats)
                    ]
                    values = [value / repeats] * repeats
            points.extend(
                (time, self._variable_index[key], val, aggregation)
                for time, val in zip(times, values)
            )
        return points

    def _grid(self, patient: Any, admission: Any) -> np.ndarray | None:
        hadm_id = str(admission.hadm_id)
        points = [
            point
            for table in _TABLE_SPECS
            for point in self._table_points(patient, table, hadm_id)
        ]
        if not points:
            return None
        reference = min(point[0] for point in points)
        grouped: dict[tuple[int, int, str], list[float]] = defaultdict(list)
        for timestamp, variable, value, aggregation in points:
            elapsed = (timestamp - reference).total_seconds() / 60.0
            time_bin = round(elapsed / self.bin_minutes)
            if 0 <= time_bin < self.total_bins:
                grouped[(time_bin, variable, aggregation)].append(value)
        grid = np.full((self.total_bins, len(self.variables)), np.nan, dtype=np.float64)
        for (time_bin, variable, aggregation), values in grouped.items():
            grid[time_bin, variable] = (
                float(np.mean(values)) if aggregation == "mean" else float(sum(values))
            )
        if int(np.isfinite(grid).sum()) < self.min_observations:
            return None
        return grid

    def _normalize(self, grid: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            return grid.astype(np.float32)
        center = np.array([self.mean[name] for name in self.variables])
        scale = np.array([self.std[name] for name in self.variables])
        normalized = (grid - center) / scale
        if self.outlier_sd is not None:
            normalized[np.abs(normalized) >= self.outlier_sd] = np.nan
        return normalized.astype(np.float32)

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        samples = []
        for admission in self._eligible_admissions(patient):
            raw_grid = self._grid(patient, admission)
            if raw_grid is None:
                continue
            grid = self._normalize(raw_grid)
            x = grid[: self.observation_bins]
            if not np.isfinite(x).any():
                continue
            future = np.flatnonzero(
                np.isfinite(grid[self.observation_bins :]).any(axis=1)
            )[: self.prediction_steps]
            if len(future) == 0:
                continue
            future = future + self.observation_bins
            y = np.full(
                (self.prediction_steps, len(self.variables)),
                np.nan,
                dtype=np.float32,
            )
            y_time = np.zeros(self.prediction_steps, dtype=np.float32)
            for step, time_bin in enumerate(future):
                y[step] = grid[time_bin]
                y_time[step] = time_bin / self.total_bins
            x_time = (
                np.arange(self.observation_bins, dtype=np.float32) / self.total_bins
            )
            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "visit_id": str(admission.hadm_id),
                    "x": x,
                    "x_time": x_time,
                    "y_time": y_time,
                    "y": y,
                }
            )
        return samples


def mimic3_forecasting_statistics(
    dataset: BaseDataset,
    patient_ids: Sequence[str] | None = None,
    variables: Sequence[str] = MIMIC3_FORECASTING_VARIABLES,
    min_observations: int = 50,
) -> tuple[dict[str, float], dict[str, float]]:
    """Computes benchmark means and standard deviations after binning.

    The reference benchmark normalizes each variable over the complete
    cohort. Pass training ``patient_ids`` instead to avoid test-set leakage.
    Values are aggregated exactly as :class:`MIMIC3Forecasting` aggregates
    them, before z-scoring and outlier removal.

    Args:
        dataset: A loaded :class:`~pyhealth.datasets.MIMIC3Dataset`.
        patient_ids: Patients to include; all patients when ``None``.
        variables: Variables for which to compute statistics.
        min_observations: Cohort threshold passed to the task.

    Returns:
        ``(mean, std)`` dictionaries keyed by variable.

    Examples:
        >>> mean, std = mimic3_forecasting_statistics(dataset)  # doctest: +SKIP
        >>> task = MIMIC3Forecasting(mean=mean, std=std)  # doctest: +SKIP
    """
    task = MIMIC3Forecasting(
        variables=variables,
        min_observations=min_observations,
        outlier_sd=None,
    )
    allowed = None if patient_ids is None else set(patient_ids)
    sums = np.zeros(len(task.variables), dtype=np.float64)
    sums_squared = np.zeros(len(task.variables), dtype=np.float64)
    counts = np.zeros(len(task.variables), dtype=np.int64)
    for patient in dataset.iter_patients():
        if allowed is not None and patient.patient_id not in allowed:
            continue
        for admission in task._eligible_admissions(patient):
            grid = task._grid(patient, admission)
            if grid is None:
                continue
            finite = np.isfinite(grid)
            values = np.where(finite, grid, 0.0)
            sums += values.sum(axis=0)
            sums_squared += np.square(values).sum(axis=0)
            counts += finite.sum(axis=0)
    missing = [name for name, count in zip(task.variables, counts) if count == 0]
    if missing:
        raise ValueError(f"No observations of {missing}")
    means = sums / counts
    variances = np.maximum(sums_squared / counts - np.square(means), 0.0)
    stds = np.sqrt(variances)
    stds[stds == 0] = 1.0
    return (
        {name: round(float(value), 6) for name, value in zip(task.variables, means)},
        {name: round(float(value), 6) for name, value in zip(task.variables, stds)},
    )
