"""Time-to-event survival prediction on METABRIC."""

from __future__ import annotations

import logging
import math
from typing import Any, ClassVar, Literal

from .base_task import BaseTask

logger = logging.getLogger(__name__)

#: Feature columns of ``metabric.csv``, in the order of the source ``.npz``
#: arrays. Four gene-expression values, then age and four binary clinical
#: indicators. :mod:`pyhealth.datasets.metabric` keeps its own copy, so the
#: task and the dataset do not import from each other; a test checks that the
#: two agree.
METABRIC_FEATURES: tuple[str, ...] = (
    "mki67",
    "egfr",
    "pgr",
    "erbb2",
    "age_at_diagnosis",
    "hormone_treatment",
    "radiotherapy",
    "chemotherapy",
    "er_positive",
)
METABRIC_GENE_FEATURES: tuple[str, ...] = METABRIC_FEATURES[:4]
METABRIC_CLINICAL_FEATURES: tuple[str, ...] = METABRIC_FEATURES[4:]

FeatureSet = Literal["all", "clinical", "genes"]
TimeUnit = Literal["months", "years", "paper"]


class MetabricSurvivalPrediction(BaseTask):
    """Predict the time to death from the METABRIC baseline features.

    Works with :class:`~pyhealth.datasets.MetabricDataset`.

    One sample per patient, for continuous-time survival models such as
    :class:`~pyhealth.models.SurvivalMDN`:

    - ``features``: the chosen baseline features as a flat vector. With
      ``standardize=True`` the ``tensor`` processor z-scores each feature
      with statistics fitted on the training patients, as SODEN and Survival
      MDN do (pass ``split=`` to ``set_task``).
    - ``duration``: follow-up time in ``time_unit``.
    - ``event``: 1.0 if the death was observed, 0.0 if censored.

    The task options support ablations on the task itself:

    - ``feature_set`` keeps all nine features, only the five clinical ones
      (age, hormone treatment, radiotherapy, chemotherapy, ER status) or
      only the four gene-expression values (MKI67, EGFR, PGR, ERBB2).
    - ``horizon_months`` applies administrative censoring: follow-up beyond
      the horizon is cut at the horizon and counted as censored, which
      mimics a study with shorter follow-up.

    Args:
        feature_set: ``"all"`` (default), ``"clinical"`` or ``"genes"``.
        time_unit: Unit of ``duration``: ``"months"`` (default), ``"years"``
            or ``"paper"``. ``"paper"`` is months / 365.25, the scale of the
            SODEN and Survival MDN files, so results can be compared with
            their tables directly.
        horizon_months: Administrative censoring time in months, or ``None``
            (default) for the full follow-up.
        min_duration_months: Durations below this are raised to it, since
            continuous-time models need strictly positive times. The
            released data has one patient with zero follow-up. Defaults to
            0.01 months (about 7 hours).
        standardize: Z-score the features with training statistics.
            Defaults to True.

    Raises:
        ValueError: If an option is out of range, or if ``min_duration_months``
            is not smaller than ``horizon_months``.

    Examples:
        >>> from pyhealth.datasets import MetabricDataset
        >>> from pyhealth.tasks import MetabricSurvivalPrediction
        >>> dataset = MetabricDataset(root="/data/metabric")  # doctest: +SKIP
        >>> task = MetabricSurvivalPrediction(feature_set="clinical")
        >>> train, valid, test = dataset.set_task(  # doctest: +SKIP
        ...     task, split=dataset.paper_split(1)
        ... )
        >>> train[0]["features"].shape  # doctest: +SKIP
        torch.Size([5])
    """

    task_name: str = "MetabricSurvivalPrediction"
    input_schema: dict[str, Any]
    output_schema: ClassVar[dict[str, Any]] = {
        "duration": "tensor",
        "event": "tensor",
    }

    _FEATURE_SETS: ClassVar[dict[str, tuple[str, ...]]] = {
        "all": METABRIC_FEATURES,
        "clinical": METABRIC_CLINICAL_FEATURES,
        "genes": METABRIC_GENE_FEATURES,
    }
    _TIME_SCALES: ClassVar[dict[str, float]] = {
        "months": 1.0,
        "years": 1.0 / 12.0,
        "paper": 1.0 / 365.25,
    }

    def __init__(
        self,
        feature_set: FeatureSet = "all",
        time_unit: TimeUnit = "months",
        horizon_months: float | None = None,
        min_duration_months: float = 0.01,
        standardize: bool = True,
    ) -> None:
        if feature_set not in self._FEATURE_SETS:
            raise ValueError(
                f"feature_set must be one of {sorted(self._FEATURE_SETS)}, "
                f"got {feature_set!r}"
            )
        if time_unit not in self._TIME_SCALES:
            raise ValueError(
                f"time_unit must be one of {sorted(self._TIME_SCALES)}, "
                f"got {time_unit!r}"
            )
        if horizon_months is not None and not 0 < horizon_months < math.inf:
            raise ValueError("horizon_months must be positive and finite, or None")
        if not 0 < min_duration_months < math.inf:
            raise ValueError("min_duration_months must be positive and finite")
        if horizon_months is not None and min_duration_months >= horizon_months:
            raise ValueError(
                "min_duration_months must be smaller than horizon_months, or "
                "raised durations would fall beyond the censoring horizon"
            )
        self.feature_set = feature_set
        self.time_unit = time_unit
        self.horizon_months = None if horizon_months is None else float(horizon_months)
        self.min_duration_months = float(min_duration_months)
        self.standardize = bool(standardize)
        self.feature_names: tuple[str, ...] = self._FEATURE_SETS[feature_set]
        self.input_schema = {
            "features": (
                ("tensor", {"normalization": "zscore"}) if standardize else "tensor"
            )
        }
        super().__init__()

    @staticmethod
    def _as_float(value: Any) -> float | None:
        """Parses a CSV value, returning None when it is missing or not finite."""
        if value is None:
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    def _label(self, duration_months: float, event: float) -> tuple[float, float]:
        """Applies the horizon, the minimum duration and the time unit."""
        if self.horizon_months is not None and duration_months > self.horizon_months:
            duration_months, event = self.horizon_months, 0.0
        duration_months = max(duration_months, self.min_duration_months)
        return duration_months * self._TIME_SCALES[self.time_unit], event

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Builds the survival sample of one patient.

        Args:
            patient: A METABRIC patient with one ``metabric`` event.

        Returns:
            A list with one sample, or an empty list if a feature or label is
            missing or invalid (a warning is logged).
        """
        events = patient.get_events(event_type="metabric")
        if len(events) == 0:
            return []
        if len(events) > 1:
            logger.warning(
                "Patient %s has %d metabric rows, expected 1; using the first",
                patient.patient_id,
                len(events),
            )
        row = events[0]

        features = [self._as_float(row[name]) for name in self.feature_names]
        duration = self._as_float(row["duration_months"])
        event = self._as_float(row["event"])
        if any(value is None for value in features):
            logger.warning("Patient %s has a missing feature", patient.patient_id)
            return []
        if duration is None or duration < 0 or event not in (0.0, 1.0):
            logger.warning("Patient %s has an invalid label", patient.patient_id)
            return []

        duration, event = self._label(duration, event)
        return [
            {
                "patient_id": patient.patient_id,
                "features": features,
                "duration": [duration],
                "event": [event],
            }
        ]
