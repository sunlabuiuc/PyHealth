"""Longitudinal liver survival samples for time-varying survival models.

Reproduces the sample construction of the Model for Allograft Survival (MAS):

    Gao, X., Cooper, M., Naghibzadeh, M., Azhie, A., Bhat, M., and
    Krishnan, R. G. "Predicting Long-Term Allograft Survival in Liver
    Transplant Recipients." Machine Learning for Healthcare, PMLR 252, 2024.
    https://proceedings.mlr.press/v252/gao24a.html

Each patient has repeated lab visits and one outcome. Two sample layouts are
built from a patient's visits, both using only labs measured at or before the
visit (missing labs are forward-filled from earlier visits):

- **Counting process** (for time-varying Cox models such as MAS): visit ``k``
  covers the interval ``(visit_k, visit_{k+1}]`` (the last visit covers
  ``(visit_K, end]``) with the labs measured *at* visit ``k``. The event flag
  is 1 only on the last interval of a patient who had the event.
- **Landmark** (for models such as DeepSurv): visit ``k`` is a sample whose
  label is the time from that visit to the event or censoring.

Input times follow the Mayo PBC ``pbcseq`` layout: days since enrollment, and a
status code of 0 (censored), 1 (transplant), or 2 (death). Output times are in
years.
"""

import math
import zlib
from collections.abc import Mapping, Sequence
from functools import partial
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from .base_task import BaseTask

DAYS_PER_YEAR = 365.25
STATUS_CENSORED = 0
STATUS_TRANSPLANT = 1
STATUS_DEATH = 2
EVENT_DEFINITIONS = ("death", "death_or_transplant")


def forward_fill(features: Sequence[Sequence[float]]) -> np.ndarray:
    """Carries each feature's last observed value forward in time.

    Values are filled from earlier visits only. A feature that has not been
    observed yet stays NaN.

    Args:
        features: Visit-by-feature values with shape ``(K, F)``; NaN marks a
            missing value.

    Returns:
        A float64 array of shape ``(K, F)``.

    Examples:
        >>> forward_fill([[1.0, float("nan")], [float("nan"), 2.0]]).tolist()
        [[1.0, nan], [1.0, 2.0]]
    """
    filled = np.array(features, dtype=np.float64)
    if filled.ndim != 2:
        raise ValueError(f"features must have shape (K, F), got {filled.shape}.")
    for k in range(1, filled.shape[0]):
        missing = np.isnan(filled[k])
        filled[k, missing] = filled[k - 1, missing]
    return filled


def event_indicator(status: int, event: str = "death") -> int:
    """Maps a ``pbcseq`` status code to an event indicator.

    Args:
        status: 0 (censored), 1 (transplant), or 2 (death).
        event: ``"death"`` counts only deaths, so a transplant is censored at
            the transplant date. ``"death_or_transplant"`` counts both.

    Returns:
        1 if the status counts as an event, otherwise 0.

    Raises:
        ValueError: If ``status`` or ``event`` is not recognised.
    """
    if event not in EVENT_DEFINITIONS:
        raise ValueError(f"event must be one of {EVENT_DEFINITIONS}, got {event!r}.")
    if status not in (STATUS_CENSORED, STATUS_TRANSPLANT, STATUS_DEATH):
        raise ValueError(f"status must be 0, 1, or 2, got {status!r}.")
    if status == STATUS_DEATH:
        return 1
    if status == STATUS_TRANSPLANT and event == "death_or_transplant":
        return 1
    return 0


def _validated_visits(
    patient_id: str,
    visit_days: Sequence[float],
    features: Sequence[Sequence[float]],
    futime: float,
    status: int,
    event: str,
    ffill: bool = True,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Checks one patient's visits and forward-fills their labs.

    Returns:
        The visit days, the ``(K, F)`` labs (forward-filled if ``ffill``), and
        the patient's event indicator.

    Raises:
        ValueError: If the visits are empty, non-finite, unsorted, negative,
            at or after ``futime``, or do not match ``features``.
    """
    days = np.asarray(visit_days, dtype=np.float64)
    if days.ndim != 1 or days.size == 0:
        raise ValueError(f"Patient {patient_id}: needs at least one visit.")
    if not math.isfinite(futime):
        raise ValueError(f"Patient {patient_id}: futime must be finite.")
    if not np.all(np.isfinite(days)):
        raise ValueError(f"Patient {patient_id}: visit days must be finite.")
    if days[0] < 0 or np.any(np.diff(days) <= 0):
        raise ValueError(
            f"Patient {patient_id}: visit days must be >= 0 and strictly increasing."
        )
    if days[-1] >= futime:
        raise ValueError(
            f"Patient {patient_id}: last visit (day {days[-1]}) is not before "
            f"futime (day {futime})."
        )
    filled = forward_fill(features)
    if not ffill:
        filled = np.array(features, dtype=np.float64)
    if filled.shape[0] != days.size:
        raise ValueError(
            f"Patient {patient_id}: {days.size} visits but {filled.shape[0]} "
            "feature rows."
        )
    final_event = event_indicator(status, event)
    return days, filled, final_event


def build_counting_process_samples(
    patient_id: str,
    visit_days: Sequence[float],
    features: Sequence[Sequence[float]],
    futime: float,
    status: int,
    event: str = "death",
    ffill: bool = True,
) -> list[dict[str, Any]]:
    """Builds one ``[start, stop, event]`` sample per visit of a patient.

    Args:
        patient_id: The patient identifier.
        visit_days: Visit times in days since enrollment, strictly increasing,
            starting at 0 or later and ending before ``futime``.
        features: Lab values per visit, shape ``(K, F)``. Missing values (NaN)
            are forward-filled from earlier visits; values never observed stay
            NaN for the feature processor to impute.
        futime: Days from enrollment to the event or censoring.
        status: 0 (censored), 1 (transplant), or 2 (death).
        event: Event definition passed to :func:`event_indicator`.
        ffill: Forward-fill missing labs from earlier visits (default). If
            False, missing labs stay NaN for the feature processor to impute.

    Returns:
        A list of ``K`` sample dicts with keys ``patient_id``, ``visit_id``
        (``"{patient_id}_{k}"``), ``landmark_time`` (visit time in years),
        ``features`` (list of ``F`` floats), and ``survival_interval``
        (``[start, stop, event]`` in years).

    Raises:
        ValueError: If the visits are empty, non-finite, unsorted, negative,
            at or after ``futime``, or do not match ``features``.

    Examples:
        >>> rows = build_counting_process_samples(
        ...     "1", [0, 365.25], [[1.0], [2.0]], futime=730.5, status=2
        ... )
        >>> [row["survival_interval"] for row in rows]
        [[0.0, 1.0, 0], [1.0, 2.0, 1]]
    """
    days, filled, final_event = _validated_visits(
        patient_id, visit_days, features, futime, status, event, ffill
    )
    stops = np.append(days[1:], futime)
    samples = []
    for k, (start, stop) in enumerate(zip(days, stops)):
        is_last = k == days.size - 1
        samples.append(
            {
                "patient_id": str(patient_id),
                "visit_id": f"{patient_id}_{k}",
                "landmark_time": float(start / DAYS_PER_YEAR),
                "features": filled[k].tolist(),
                "survival_interval": [
                    float(start / DAYS_PER_YEAR),
                    float(stop / DAYS_PER_YEAR),
                    final_event if is_last else 0,
                ],
            }
        )
    return samples


def build_landmark_samples(
    patient_id: str,
    visit_days: Sequence[float],
    features: Sequence[Sequence[float]],
    futime: float,
    status: int,
    event: str = "death",
    ffill: bool = True,
) -> list[dict[str, Any]]:
    """Builds one ``[duration, event]`` landmark sample per visit of a patient.

    Each visit is a prediction time: its features are the labs known at that
    visit, and its label is the time from the visit to the patient's event or
    censoring.

    Args:
        patient_id: The patient identifier.
        visit_days: Visit times in days since enrollment, strictly increasing,
            starting at 0 or later and ending before ``futime``.
        features: Lab values per visit, shape ``(K, F)``. Missing values (NaN)
            are forward-filled from earlier visits; values never observed stay
            NaN for the feature processor to impute.
        futime: Days from enrollment to the event or censoring.
        status: 0 (censored), 1 (transplant), or 2 (death).
        event: Event definition passed to :func:`event_indicator`.
        ffill: Forward-fill missing labs from earlier visits (default). If
            False, missing labs stay NaN for the feature processor to impute.

    Returns:
        A list of ``K`` sample dicts with keys ``patient_id``, ``visit_id``,
        ``landmark_time`` (visit time in years), ``features``, and
        ``survival`` (``[duration, event]``, duration in years).

    Raises:
        ValueError: If the visits are empty, non-finite, unsorted, negative,
            at or after ``futime``, or do not match ``features``.

    Examples:
        >>> rows = build_landmark_samples(
        ...     "1", [0, 365.25], [[1.0], [2.0]], futime=730.5, status=2
        ... )
        >>> [row["survival"] for row in rows]
        [[2.0, 1], [1.0, 1]]
    """
    days, filled, final_event = _validated_visits(
        patient_id, visit_days, features, futime, status, event, ffill
    )
    return [
        {
            "patient_id": str(patient_id),
            "visit_id": f"{patient_id}_{k}",
            "landmark_time": float(day / DAYS_PER_YEAR),
            "features": filled[k].tolist(),
            "survival": [float((futime - day) / DAYS_PER_YEAR), final_event],
        }
        for k, day in enumerate(days)
    ]


def build_binary_horizon_samples(
    patient_id: str,
    visit_days: Sequence[float],
    features: Sequence[Sequence[float]],
    futime: float,
    status: int,
    horizon: float,
    event: str = "death",
    ffill: bool = True,
) -> list[dict[str, Any]]:
    """Builds fixed-horizon binary samples: an event within ``horizon`` years?

    Each visit gets ``label`` 1 if the patient has the event within
    ``horizon`` years after it, and 0 if they are followed for longer than
    ``horizon`` years. Visits whose follow-up ends sooner without the event
    (censored, or a competing transplant) are **dropped**, because their label
    is unknown; counting them as 0 would bias the label.

    Args:
        patient_id: The patient identifier.
        visit_days: Visit times in days since enrollment, strictly increasing,
            starting at 0 or later and ending before ``futime``.
        features: Lab values per visit, shape ``(K, F)``.
        futime: Days from enrollment to the event or censoring.
        status: 0 (censored), 1 (transplant), or 2 (death).
        horizon: Prediction horizon in years (positive).
        event: Event definition passed to :func:`event_indicator`.
        ffill: Forward-fill missing labs from earlier visits.

    Returns:
        Sample dicts with keys ``patient_id``, ``visit_id``,
        ``landmark_time``, ``features``, and ``label`` (0 or 1).

    Raises:
        ValueError: If ``horizon`` is not positive, or the visits are invalid
            (see :func:`build_counting_process_samples`).

    Examples:
        >>> rows = build_binary_horizon_samples(
        ...     "1", [0, 365.25], [[1.0], [2.0]], futime=730.5, status=2,
        ...     horizon=1.5)
        >>> [(row["visit_id"], row["label"]) for row in rows]
        [('1_0', 0), ('1_1', 1)]
    """
    if not horizon > 0:
        raise ValueError(f"horizon must be positive, got {horizon}.")
    days, filled, final_event = _validated_visits(
        patient_id, visit_days, features, futime, status, event, ffill
    )
    samples = []
    for k, day in enumerate(days):
        remaining = (futime - day) / DAYS_PER_YEAR
        if remaining > horizon:
            label = 0
        elif final_event == 1:
            label = 1
        else:
            continue
        samples.append(
            {
                "patient_id": str(patient_id),
                "visit_id": f"{patient_id}_{k}",
                "landmark_time": float(day / DAYS_PER_YEAR),
                "features": filled[k].tolist(),
                "label": label,
            }
        )
    return samples


def counting_process_samples_from_frame(
    visits: pd.DataFrame,
    feature_names: Sequence[str],
    event: str = "death",
) -> list[dict[str, Any]]:
    """Builds counting-process samples for every patient in a visit table.

    Args:
        visits: One row per visit with the ``pbcseq`` columns ``id``,
            ``day``, ``futime``, and ``status``, plus one column per feature.
            ``futime`` and ``status`` must be constant within a patient.
        feature_names: Feature columns to use, in order.
        event: Event definition passed to :func:`event_indicator`.

    Returns:
        Samples from :func:`build_counting_process_samples`, ordered by
        patient id and visit day.

    Raises:
        ValueError: If a required column is missing, ``futime`` or
            ``status`` varies within a patient, or ``status`` is not an integer.
    """
    required = ["id", "day", "futime", "status", *feature_names]
    missing = [column for column in required if column not in visits.columns]
    if missing:
        raise ValueError(f"Visit table is missing columns {missing}.")
    samples: list[dict[str, Any]] = []
    ordered = visits.sort_values(["id", "day"], kind="stable")
    for patient_id, rows in ordered.groupby("id", sort=True):
        if rows["futime"].nunique() != 1 or rows["status"].nunique() != 1:
            raise ValueError(
                f"Patient {patient_id}: futime and status must be constant."
            )
        status = float(rows["status"].iloc[0])
        if not status.is_integer():
            raise ValueError(
                f"Patient {patient_id}: status must be 0, 1, or 2, got {status}."
            )
        samples.extend(
            build_counting_process_samples(
                patient_id=str(patient_id),
                visit_days=rows["day"].to_numpy(dtype=np.float64),
                features=rows[list(feature_names)].to_numpy(dtype=np.float64),
                futime=float(rows["futime"].iloc[0]),
                status=int(status),
                event=event,
            )
        )
    return samples


FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "mas4": ("bili", "albumin", "ast", "protime"),
    "albi": ("bili", "albumin"),
    "mayo": ("age", "bili", "albumin", "protime", "edema"),
    "full": (
        "bili",
        "albumin",
        "ast",
        "protime",
        "alk_phos",
        "chol",
        "platelet",
        "ascites",
        "hepato",
        "spiders",
        "edema",
        "age",
        "sex",
        "trt",
    ),
}
"""Named ``pbcseq`` feature sets.

- ``mas4``: the four MAS labs in PBC (prothrombin time stands in for INR).
- ``albi``: the inputs of the ALBI score.
- ``mayo``: the inputs of the Mayo PBC risk score.
- ``full``: every visit and patient variable except ``stage``, which changes
  over time and is used to define the distribution-shift split.
"""

MODES = ("counting_process", "landmark", "binary_horizon")

LOG_SCALE_FEATURES = frozenset(
    {"bili", "albumin", "ast", "protime", "alk_phos", "chol", "platelet"}
)
"""Positive, right-skewed ``pbcseq`` labs that ``log_transform=True`` logs."""


class LongitudinalLiverSurvival(BaseTask):
    """Time-to-event samples from repeated liver-lab visits.

    Builds the samples used to reproduce the Model for Allograft Survival
    (MAS; Gao et al., MLHC 2024) on a :class:`~pyhealth.datasets.PBCSeqDataset`.
    Each visit becomes one sample whose features are the labs known at that
    visit (forward-filled from earlier visits, never from later ones).

    - ``mode="counting_process"``: label ``survival_interval`` =
      ``[start, stop, event]`` in years, for time-varying Cox models (MAS).
    - ``mode="landmark"``: label ``survival`` = ``[duration, event]`` in years
      from the visit, for models such as DeepSurv (DeepMAS).
    - ``mode="binary_horizon"``: binary ``label`` = event within ``horizon``
      years of the visit, for standard classifiers. Visits censored before the
      horizon are dropped (see :func:`build_binary_horizon_samples`).

    Features are z-scored by ``TensorProcessor`` with statistics learned in
    ``fit()``, and missing values become the training mean. Fit on training
    patients only with ``dataset.set_task(task, split=PatientSplit(...))``.

    Every sample also carries, unprocessed: ``patient_id``, ``visit_id``,
    ``landmark_time`` (visit time in years, for time-dependent evaluation), and
    ``baseline_stage`` (histologic stage at the first visit, for the
    stage-shift experiment).

    Args:
        mode: ``"counting_process"`` (default), ``"landmark"``, or
            ``"binary_horizon"``.
        feature_set: A name in :data:`FEATURE_SETS` (default ``"mas4"``) or a
            sequence of ``pbcseq`` column names.
        event: ``"death"`` (default; transplant is censored at the transplant
            date) or ``"death_or_transplant"``.
        log_transform: ``False`` (default; raw values, as in the MAS paper),
            ``True`` (natural log of every selected lab in
            :data:`LOG_SCALE_FEATURES`, as in the Mayo PBC score), or a
            sequence of feature names to log. Non-positive values become
            missing. Logging is applied before
            z-scoring and tames the heavy right tails of bilirubin, AST, and
            prothrombin time.
        horizon: Years for ``mode="binary_horizon"`` (required there).
        forward_fill: Forward-fill missing labs from earlier visits (default;
            the MAS paper's imputation). If False, missing labs are filled
            with the training mean by the feature processor.
        inject_missing: Missing-data ablation: ``{feature: rate}`` sets each
            follow-up value (not day 0) of that feature to missing with
            probability ``rate``, independently, before imputation.
        inject_seed: Seed for ``inject_missing``. The same seed and patient
            always give the same mask.
        missing_indicators: Append a 0/1 column per feature marking values that
            were missing before imputation (for missingness ablations).

    Attributes:
        feature_names: The feature columns, in the order of ``features``.
        log_features: The features that are log-transformed.

    Raises:
        ValueError: If ``mode``, ``feature_set``, or ``event`` is not
            recognised.

    Examples:
        >>> from pyhealth.datasets import PatientSplit, PBCSeqDataset
        >>> dataset = PBCSeqDataset(root="/path/to/pbcseq")  # doctest: +SKIP
        >>> task = LongitudinalLiverSurvival(mode="counting_process")
        >>> task.feature_names
        ('bili', 'albumin', 'ast', 'protime')
        >>> train, val, test = dataset.set_task(  # doctest: +SKIP
        ...     task, split=PatientSplit(ratios=(0.7, 0.15, 0.15), seed=0)
        ... )
    """

    task_name: str = "LongitudinalLiverSurvival"
    event_type: str = "pbcseq"

    def __init__(
        self,
        mode: str = "counting_process",
        feature_set: str | Sequence[str] = "mas4",
        event: str = "death",
        log_transform: bool | Sequence[str] = False,
        missing_indicators: bool = False,
        horizon: float | None = None,
        forward_fill: bool = True,
        inject_missing: Mapping[str, float] | None = None,
        inject_seed: int = 0,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}.")
        if event not in EVENT_DEFINITIONS:
            raise ValueError(
                f"event must be one of {EVENT_DEFINITIONS}, got {event!r}."
            )
        if isinstance(feature_set, str):
            if feature_set not in FEATURE_SETS:
                raise ValueError(
                    f"feature_set must be one of {list(FEATURE_SETS)} or a list "
                    f"of columns, got {feature_set!r}."
                )
            feature_names = FEATURE_SETS[feature_set]
        else:
            feature_names = tuple(feature_set)
            if not feature_names:
                raise ValueError("feature_set must not be empty.")
        self.mode = mode
        self.event = event
        self.feature_names = feature_names
        self.missing_indicators = missing_indicators
        if log_transform is True:
            logged = [name for name in feature_names if name in LOG_SCALE_FEATURES]
        elif log_transform is False:
            logged = []
        else:
            logged = list(log_transform)
            unknown = [name for name in logged if name not in feature_names]
            if unknown:
                raise ValueError(f"log_transform names unknown features {unknown}.")
        self.log_features = tuple(logged)
        self.input_schema = {
            "features": (
                "tensor",
                {
                    "normalization": "zscore",
                    "impute_strategy": "zero",
                    "missing_indicators": missing_indicators,
                },
            )
        }
        if mode == "binary_horizon":
            if horizon is None or not horizon > 0:
                raise ValueError("mode='binary_horizon' needs a positive horizon.")
        elif horizon is not None:
            raise ValueError("horizon is only used with mode='binary_horizon'.")
        self.horizon = horizon
        self.forward_fill = forward_fill
        injection = dict(inject_missing or {})
        unknown = [name for name in injection if name not in feature_names]
        if unknown:
            raise ValueError(f"inject_missing names unknown features {unknown}.")
        if any(not 0 <= rate < 1 for rate in injection.values()):
            raise ValueError("inject_missing rates must be in [0, 1).")
        self.inject_missing = injection
        self.inject_seed = inject_seed
        if mode == "counting_process":
            self.output_schema = {"survival_interval": "survival_interval"}
        elif mode == "landmark":
            self.output_schema = {"survival": "survival"}
        else:
            self.output_schema = {"label": "binary"}
        super().__init__()

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Builds the samples of one patient.

        Args:
            patient: A :class:`~pyhealth.data.Patient` with ``pbcseq`` events.

        Returns:
            One sample per visit, or an empty list if the patient has no
            visits.

        Raises:
            ValueError: If a feature column is missing, the outcome is not
                constant across visits, or the visits are invalid (see
                :func:`build_counting_process_samples`).
        """
        visits = patient.get_events(event_type=self.event_type, return_df=True)
        if visits.height == 0:
            return []
        columns = ["day", "futime", "status", "stage", *self.feature_names]
        missing = [
            name
            for name in columns
            if f"{self.event_type}/{name}" not in visits.columns
        ]
        if missing:
            raise ValueError(f"pbcseq events are missing attributes {missing}.")
        values = visits.select(
            [_numeric(f"{self.event_type}/{name}").alias(name) for name in columns]
        ).sort("day")
        futimes = values["futime"].unique()
        statuses = values["status"].unique()
        if futimes.null_count() or statuses.null_count():
            raise ValueError(
                f"Patient {patient.patient_id}: futime and status must be present."
            )
        if futimes.len() != 1 or statuses.len() != 1:
            raise ValueError(
                f"Patient {patient.patient_id}: futime and status must be constant."
            )
        status = float(statuses[0])
        if not status.is_integer():
            raise ValueError(
                f"Patient {patient.patient_id}: status must be 0, 1, or 2, "
                f"got {status}."
            )
        features = values.select(
            [
                _log_positive(name) if name in self.log_features else pl.col(name)
                for name in self.feature_names
            ]
        ).to_numpy()
        if self.inject_missing:
            features = _inject_missing(
                features,
                values["day"].to_numpy(),
                [self.inject_missing.get(n, 0.0) for n in self.feature_names],
                self.inject_seed,
                str(patient.patient_id),
            )
        if self.mode == "counting_process":
            build = build_counting_process_samples
        elif self.mode == "landmark":
            build = build_landmark_samples
        else:
            build = partial(build_binary_horizon_samples, horizon=self.horizon)
        samples = build(
            patient_id=str(patient.patient_id),
            visit_days=values["day"].to_numpy(),
            features=features,
            futime=float(futimes[0]),
            status=int(status),
            event=self.event,
            ffill=self.forward_fill,
        )
        stage = values["stage"][0]
        baseline_stage = None if stage is None or math.isnan(stage) else int(stage)
        for sample in samples:
            sample["baseline_stage"] = baseline_stage
        return samples


def _inject_missing(
    features: np.ndarray,
    days: np.ndarray,
    rates: Sequence[float],
    seed: int,
    patient_id: str,
) -> np.ndarray:
    """Sets follow-up values to NaN at random, reproducibly per patient.

    Day-0 values are kept. Each other value of feature ``f`` is removed with
    probability ``rates[f]``, using a generator seeded by ``seed`` and the
    patient id, so a patient's mask does not depend on the other patients.
    """
    rng = np.random.default_rng([seed, zlib.crc32(patient_id.encode())])
    drop = rng.random(features.shape) < np.asarray(rates, dtype=float)
    drop &= (np.asarray(days) > 0)[:, None]
    features = features.copy()
    features[drop] = np.nan
    return features


def _log_positive(column: str) -> pl.Expr:
    """Natural log of positive values; zero, negative, or null become null."""
    return pl.when(pl.col(column) > 0).then(pl.col(column).log()).alias(column)


def _numeric(column: str) -> pl.Expr:
    """Parses a ``pbcseq`` attribute as a float column.

    Attributes are loaded as strings. ``sex`` is coded ``f``/``m`` in the R
    data and becomes 1.0 for female, 0.0 for male. Empty or unparsable values
    become null.
    """
    text = pl.col(column).cast(pl.Utf8).str.strip_chars().str.to_lowercase()
    if column.endswith("/sex"):
        return (
            pl.when(text == "f")
            .then(1.0)
            .when(text == "m")
            .then(0.0)
            .otherwise(text.cast(pl.Float64, strict=False))
        )
    return text.cast(pl.Float64, strict=False)
