"""In-hospital mortality prediction on the PhysioNet 2012 ICU data.

Authors: Sadegh Nassiri, Kaushal Seelam (Georgia Tech, CSE 6250 project).

The input layout follows TransEHR (Xu et al., "TransEHR: Self-Supervised
Transformer for Clinical Time Series Data", ML4H 2023,
https://proceedings.mlr.press/v225/xu23a.html): the first 48 hours of an ICU
stay become an hourly grid of measurements plus a separate, irregular stream of
lab-event times, and the label is in-hospital death.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import polars as pl
import torch

from .base_task import BaseTask

# Intentionally shared with the forecasting task (and physionet2012_normalization)
# so both bin hourly means identically; one implementation avoids drift.
from .physionet2012_forecasting import _hourly_means

if TYPE_CHECKING:
    from ..datasets import BaseDataset

MORTALITY_VARIABLES = (
    "Weight", "ALP", "ALT", "AST", "Albumin", "BUN", "Bilirubin", "Cholesterol",
    "Creatinine", "DiasABP", "FiO2", "GCS", "Glucose", "HCO3", "HCT", "HR", "K",
    "Lactate", "MAP", "Mg", "NIDiasABP", "NIMAP", "NISysABP", "Na", "PaCO2",
    "PaO2", "Platelets", "RespRate", "SaO2", "SysABP", "Temp", "TroponinI",
    "TroponinT", "Urine", "WBC", "pH",
)
"""The 36 time series of the TransEHR experiments: the 37 PhysioNet 2012
series without the mechanical-ventilation flag (``MechVent``). Unlike
:data:`~pyhealth.tasks.physionet2012_forecasting.TIME_SERIES_VARIABLES`, this
includes ``Weight``. The ``Weight`` column holds in-stay measurements only: the
00:00 admission weight is a descriptor stored in the ``patients`` table, not in
the events the grid reads."""

VITAL_VARIABLES = (
    "DiasABP", "Glucose", "HR", "MAP", "RespRate", "SysABP", "Temp",
)
"""Regularly sampled measurements. The authors' ``feature_meta.json`` (read by
their ``data/dataset.py``) is not public, so this list is taken from the
commented-out ``p_vitals`` line in their ``preprocess/preprocess_event.py``;
every other variable is treated as a lab."""

LAB_VARIABLES = tuple(v for v in MORTALITY_VARIABLES if v not in VITAL_VARIABLES)
"""Irregularly sampled measurements, whose observation times form the event
stream. This gives 29 variables; the paper reports 28, and which one it leaves
out is not documented, so pass ``lab_variables`` to override."""

FEATURE_GROUPS = ("vitals", "labs")
STATIC_FEATURES = (
    "age", "gender=0", "gender=1", "icutype=1", "icutype=2", "icutype=3",
    "icutype=4", "height",
)


def build_mortality_sample(
    minutes: Sequence[int] | np.ndarray,
    parameters: Sequence[str] | np.ndarray,
    values: Sequence[float] | np.ndarray,
    variables: Sequence[str] = MORTALITY_VARIABLES,
    lab_variables: Sequence[str] = LAB_VARIABLES,
    center: Mapping[str, float] | None = None,
    scale: Mapping[str, float] | None = None,
    observation_hours: int = 48,
    max_events: int = 400,
    max_events_per_variable: int = 150,
    clip: float | None = None,
) -> dict[str, np.ndarray] | None:
    """Turn the observations of one ICU stay into model inputs.

    Observations at or after ``observation_hours``, and missing values, are
    dropped. The remaining ones are averaged per hour and variable into a grid
    (optionally normalized as ``(value - center) / scale``), and the times of
    the ``lab_variables`` observations form an event stream.

    Args:
        minutes: Minutes since ICU admission of each observation.
        parameters: Variable name of each observation.
        values: Value of each observation (NaN is missing).
        variables: Variables of the measurement grid, in column order.
        lab_variables: Variables whose observation times form the event
            stream; event type ``k + 1`` is ``lab_variables[k]``.
        center: Per-variable value subtracted before scaling; ``None`` keeps
            raw values.
        scale: Per-variable divisor; a zero divisor sets the variable to 0.
        observation_hours: Length of the observation window.
        max_events: Maximum length of the event stream (earliest are kept).
        max_events_per_variable: Maximum events kept per lab variable
            (earliest are kept).
        clip: Clip normalized values to ``[-clip, clip]``; ``None`` (the
            default, as in the TransEHR preprocessing) keeps them as is. The
            raw data has entry errors (e.g. a pH of 727) that normalize to
            values in the thousands. Needs ``center`` and ``scale``.

    Returns:
        A dict with ``measurements`` (``[T, V]``, 0 where unobserved),
        ``observed`` (``[T, V]``, 1 where observed), ``padding_mask`` (``[T]``,
        1 up to the last hour with an observation), ``event_times``
        (``[max_events]``, hours since admission, 0 for padding) and
        ``event_types`` (``[max_events]``, 0 for padding), or ``None`` if
        nothing was observed in the window.

    Examples:
        >>> sample = build_mortality_sample(
        ...     [10, 70], ["HR", "HR"], [60.0, 80.0],
        ...     variables=["HR"], lab_variables=[], observation_hours=3,
        ... )
        >>> sample["measurements"].ravel().tolist()
        [60.0, 80.0, 0.0]
    """
    if (center is None) != (scale is None):
        raise ValueError("Give both center and scale, or neither.")
    if clip is not None and (center is None or clip <= 0):
        raise ValueError("clip must be positive and needs center and scale.")
    minutes = np.asarray(minutes, dtype=np.int64)
    parameters = np.asarray(parameters, dtype=object)
    values = np.asarray(values, dtype=np.float64)
    column = {v: i for i, v in enumerate(variables)}

    keep = np.isfinite(values) & (minutes >= 0) & (minutes < observation_hours * 60)
    minutes, parameters, values = minutes[keep], parameters[keep], values[keep]

    n_vars = len(variables)
    totals = np.zeros((observation_hours, n_vars))
    counts = np.zeros((observation_hours, n_vars))
    in_grid = np.array([p in column for p in parameters], dtype=bool)
    if in_grid.any():
        rows = minutes[in_grid] // 60
        cols = np.array([column[p] for p in parameters[in_grid]], dtype=np.int64)
        np.add.at(totals, (rows, cols), values[in_grid])
        np.add.at(counts, (rows, cols), 1.0)
    observed = counts > 0
    if not observed.any():
        return None

    grid = np.zeros((observation_hours, n_vars))
    grid[observed] = totals[observed] / counts[observed]
    if center is not None:
        shift = np.array([center[v] for v in variables], dtype=np.float64)
        divisor = np.array([scale[v] for v in variables], dtype=np.float64)
        safe = np.where(divisor > 0, divisor, 1.0)
        grid = np.where(divisor > 0, (grid - shift) / safe, 0.0)
        grid[~observed] = 0.0
        if clip is not None:
            grid = np.clip(grid, -clip, clip)

    padding_mask = np.zeros(observation_hours)
    padding_mask[: int(np.nonzero(observed.any(axis=1))[0].max()) + 1] = 1.0

    times: list[np.ndarray] = []
    types: list[np.ndarray] = []
    for type_id, variable in enumerate(lab_variables, start=1):
        at = np.sort(minutes[parameters == variable])[:max_events_per_variable]
        times.append(at / 60.0)
        types.append(np.full(len(at), type_id, dtype=np.int64))
    event_times = np.zeros(max_events, dtype=np.float32)
    event_types = np.zeros(max_events, dtype=np.int64)
    if times:
        all_times, all_types = np.concatenate(times), np.concatenate(types)
        order = np.lexsort((all_types, all_times))[:max_events]
        event_times[: len(order)] = all_times[order]
        event_types[: len(order)] = all_types[order]

    return {
        "measurements": grid.astype(np.float32),
        "observed": observed.astype(np.float32),
        "padding_mask": padding_mask.astype(np.float32),
        "event_times": event_times,
        "event_types": event_types,
    }


class PhysioNet2012InHospitalMortality(BaseTask):
    """Predict in-hospital death from the first hours of a PhysioNet 2012 stay.

    Works with :class:`~pyhealth.datasets.PhysioNet2012Dataset`. Each ICU stay
    gives one sample, as in the TransEHR experiments (Xu et al., 2023):

    * ``measurements`` / ``observed`` / ``padding_mask``: the hourly grid of
      the selected variables over the first ``observation_hours`` hours (see
      :func:`build_mortality_sample`).
    * ``event_times`` / ``event_types``: the irregular stream of lab
      observation times (types are 1-based indices into ``lab_variables``,
      0 is padding). Empty when ``"labs"`` is not in ``feature_groups``.
    * ``static``: age, one-hot gender, one-hot ICU type and height (see
      :data:`STATIC_FEATURES`). Missing height is set to ``height_fill``.
    * label ``mortality``: in-hospital death.

    ``feature_groups`` selects which variables enter the grid, so vitals-only
    and vitals-plus-labs variants of the task can be compared:
    ``("vitals",)`` uses the vitals only; ``("labs",)`` the labs only (with the
    event stream); ``("vitals", "labs")`` all (the default).

    Normalize with :func:`physionet2012_normalization`, computed on the
    training records only.

    Args:
        center: Per-variable median for normalization; ``None`` keeps raw
            values. Required together with ``scale``.
        scale: Per-variable ``p95 - p5``; a zero scale zeroes the variable.
        observation_hours: Hours of the stay used as input.
        feature_groups: Variable groups to use: ``"vitals"``, ``"labs"``.
        icu_types: Keep only stays of these ICU types (1-4); ``None`` keeps all.
        vital_variables: Variables of the ``"vitals"`` group.
        lab_variables: Variables of the ``"labs"`` group and of the event stream.
        max_events: Length of the padded event stream.
        max_events_per_variable: Maximum events per lab variable.
        height_fill: Value used for missing heights.
        clip: Clip normalized values to ``[-clip, clip]``; ``None`` keeps them
            (see :func:`build_mortality_sample`).

    Examples:
        >>> from pyhealth.datasets import PhysioNet2012Dataset
        >>> from pyhealth.tasks import (
        ...     PhysioNet2012InHospitalMortality, physionet2012_normalization
        ... )
        >>> dataset = PhysioNet2012Dataset(root="data/physionet2012")  # doctest: +SKIP
        >>> center, scale = physionet2012_normalization(dataset)  # doctest: +SKIP
        >>> task = PhysioNet2012InHospitalMortality(
        ...     center=center, scale=scale, feature_groups=("vitals",)
        ... )  # doctest: +SKIP
        >>> samples = dataset.set_task(task)  # doctest: +SKIP
    """

    task_name: str = "PhysioNet2012InHospitalMortality"
    input_schema: ClassVar[dict[str, Any]] = {
        "measurements": "tensor",
        "observed": "tensor",
        "padding_mask": "tensor",
        "event_times": "tensor",
        "event_types": ("tensor", {"dtype": torch.long}),
        "static": "tensor",
    }
    output_schema: ClassVar[dict[str, str]] = {"mortality": "binary"}

    def __init__(
        self,
        center: Mapping[str, float] | None = None,
        scale: Mapping[str, float] | None = None,
        observation_hours: int = 48,
        feature_groups: Sequence[str] = FEATURE_GROUPS,
        icu_types: Sequence[int] | None = None,
        vital_variables: Sequence[str] = VITAL_VARIABLES,
        lab_variables: Sequence[str] = LAB_VARIABLES,
        max_events: int = 400,
        max_events_per_variable: int = 150,
        height_fill: float = 170.0,
        clip: float | None = None,
    ) -> None:
        super().__init__()
        if (center is None) != (scale is None):
            raise ValueError("Give both center and scale, or neither.")
        if clip is not None and (center is None or clip <= 0):
            raise ValueError("clip must be positive and needs center and scale.")
        groups = tuple(feature_groups)
        unknown = set(groups) - set(FEATURE_GROUPS)
        if unknown or not groups:
            raise ValueError(
                f"feature_groups must be a non-empty subset of {FEATURE_GROUPS}, "
                f"got {tuple(feature_groups)}"
            )
        if observation_hours < 1:
            raise ValueError("observation_hours must be at least 1.")
        self.feature_groups = groups
        self.use_labs = "labs" in groups
        self.lab_variables = list(lab_variables)
        self.variables = [
            v
            for v in MORTALITY_VARIABLES
            if (v in vital_variables and "vitals" in groups)
            or (v in self.lab_variables and self.use_labs)
        ]
        if not self.variables:
            raise ValueError("The selected feature groups contain no variables.")
        self.center = None if center is None else dict(center)
        self.scale = None if scale is None else dict(scale)
        if self.center is not None:
            missing = [v for v in self.variables if v not in self.center
                       or v not in self.scale]
            if missing:
                raise ValueError(f"center/scale lack variables {missing}")
        self.observation_hours = int(observation_hours)
        self.icu_types = None if icu_types is None else {int(t) for t in icu_types}
        self.max_events = int(max_events)
        self.max_events_per_variable = int(max_events_per_variable)
        self.height_fill = float(height_fill)
        self.clip = None if clip is None else float(clip)

    def _static(self, info: Any) -> np.ndarray:
        """Static vector (:data:`STATIC_FEATURES`) of a ``patients`` row."""

        def number(name: str) -> float:
            try:
                return float(getattr(info, name))
            except (TypeError, ValueError):
                return float("nan")

        age, gender = number("age"), number("gender")
        icu, height = number("icutype"), number("height")
        return np.array(
            [
                0.0 if np.isnan(age) else age,
                float(gender == 0),
                float(gender == 1),
                *(float(icu == k) for k in (1, 2, 3, 4)),
                self.height_fill if np.isnan(height) else height,
            ],
            dtype=np.float32,
        )

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        infos = patient.get_events(event_type="patients")
        if not infos:
            return []
        info = infos[0]
        try:
            label = int(float(info.in_hospital_death))
        except (TypeError, ValueError):
            return []  # outcome unknown
        if label not in (0, 1):
            return []
        if self.icu_types is not None:
            try:
                icu_type = int(float(info.icutype))
            except (TypeError, ValueError):
                return []
            if icu_type not in self.icu_types:
                return []

        events = patient.get_events(event_type="events", return_df=True)
        if events.height == 0:
            return []
        sample = build_mortality_sample(
            events["events/minutes"].cast(pl.Int64).to_numpy(),
            events["events/parameter"].to_list(),
            events["events/value"].cast(pl.Float64).to_numpy(),
            variables=self.variables,
            lab_variables=self.lab_variables if self.use_labs else [],
            center=self.center,
            scale=self.scale,
            observation_hours=self.observation_hours,
            max_events=self.max_events,
            max_events_per_variable=self.max_events_per_variable,
            clip=self.clip,
        )
        if sample is None:
            return []
        return [
            {
                "patient_id": patient.patient_id,
                **sample,
                "static": self._static(info),
                "mortality": label,
            }
        ]


def physionet2012_normalization(
    dataset: BaseDataset,
    patient_ids: Sequence[str] | None = None,
    variables: Sequence[str] = MORTALITY_VARIABLES,
) -> tuple[dict[str, float], dict[str, float]]:
    """Per-variable median and ``p95 - p5`` of the hourly means.

    These are the statistics TransEHR's preprocessing normalizes with
    (``(x - median) / (p95 - p5)``, computed over non-zero values). Pass the
    training ``patient_ids`` so that no information leaks from validation or
    test stays.

    Args:
        dataset: A :class:`~pyhealth.datasets.PhysioNet2012Dataset`.
        patient_ids: Records to compute the statistics on; all if ``None``.
        variables: The time series to describe.

    Returns:
        ``(center, scale)``, dicts keyed by variable, rounded to 6 decimals. A
        variable that is constant (``p95 == p5``) has scale 0.

    Raises:
        ValueError: If a variable has no non-zero observation (this includes
            ``Weight`` when no stay has an in-stay weight measurement).

    Examples:
        >>> center, scale = physionet2012_normalization(dataset)  # doctest: +SKIP
        >>> sorted(center) == sorted(MORTALITY_VARIABLES)  # doctest: +SKIP
        True
    """
    events = dataset.global_event_df.filter(pl.col("event_type") == "events")
    if patient_ids is not None:
        events = events.filter(pl.col("patient_id").is_in(list(patient_ids)))
    hourly = _hourly_means(events, variables, by=["patient_id"])
    stats = (
        hourly.filter(pl.col("value") != 0)
        .group_by("parameter")
        .agg(
            pl.col("value").quantile(0.5, "linear").alias("median"),
            pl.col("value").quantile(0.05, "linear").alias("p5"),
            pl.col("value").quantile(0.95, "linear").alias("p95"),
        )
        .collect()
    )
    rows = {row["parameter"]: row for row in stats.iter_rows(named=True)}
    missing = [v for v in variables if v not in rows]
    if missing:
        raise ValueError(f"No non-zero observations of {missing}")
    center = {v: round(float(rows[v]["median"]), 6) for v in variables}
    scale = {v: round(float(rows[v]["p95"] - rows[v]["p5"]), 6) for v in variables}
    return center, scale
