"""Authors: Murphy Crosby and Chidozie Opara.

Time-to-event task for SUPPORT2 in the representation used by Deep Cox Mixtures.
Paper: Deep Cox Mixtures for Survival Regression, Nagpal et al., MLHC 2021.
https://proceedings.mlr.press/v149/nagpal21a.html
"""

import math
from collections.abc import Mapping, Sequence
from typing import ClassVar

import numpy as np

from pyhealth.data import Patient

from .base_task import BaseTask

# SUPPORT2 in the representation used by Deep Cox Mixtures (Nagpal et al., 2021):
# all 9,105 patients, 18 numeric covariates and 6 one-hot coded categorical.
DCM_SUPPORT2_NUMERIC = (
    "age",
    "num.co",
    "meanbp",
    "wblc",
    "hrt",
    "resp",
    "temp",
    "pafi",
    "alb",
    "bili",
    "crea",
    "sod",
    "ph",
    "glucose",
    "bun",
    "urine",
    "adlp",
    "adls",
)

# Every level gets its own column (no reference level is dropped). A missing
# value sets all of that covariate's columns to 0.
DCM_SUPPORT2_CATEGORIES = (
    ("sex", ("female", "male")),
    (
        "dzgroup",
        (
            "ARF/MOSF w/Sepsis",
            "CHF",
            "COPD",
            "Cirrhosis",
            "Colon Cancer",
            "Coma",
            "Lung Cancer",
            "MOSF w/Malig",
        ),
    ),
    ("dzclass", ("ARF/MOSF", "COPD/CHF/Cirrhosis", "Cancer", "Coma")),
    ("income", ("$11-$25k", "$25-$50k", ">$50k", "under $11k")),
    ("race", ("asian", "black", "hispanic", "other", "white")),
    ("ca", ("metastatic", "no", "yes")),
)

# Feature names in output order: 18 numeric, then 26 one-hot columns.
DCM_SUPPORT2_FEATURES = DCM_SUPPORT2_NUMERIC + tuple(
    f"{name}={level}" for name, levels in DCM_SUPPORT2_CATEGORIES for level in levels
)

_MISSING_TOKENS = frozenset({"", "nan", "na"})


def _is_missing(value: object) -> bool:
    """Whether a raw SUPPORT2 field holds no value."""
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    return str(value).strip().lower() in _MISSING_TOKENS


def _dcm_support2_values(
    record: Mapping[str, object], source_id: str
) -> tuple[list[float], float, float]:
    """Encode one SUPPORT2 record as 44 raw features and its labels.

    Args:
        record: Column name to raw value for one patient.
        source_id: Patient identifier, used in error messages.

    Returns:
        The 44 features in the order of ``DCM_SUPPORT2_FEATURES`` (NaN for a
        missing numeric, all zeros for a missing categorical), the follow-up
        time in days, and the event indicator (1 death, 0 censored).

    Raises:
        ValueError: If a value is malformed or a label is missing.
    """
    features: list[float] = []
    for name in DCM_SUPPORT2_NUMERIC:
        raw = record.get(name)
        if _is_missing(raw):
            features.append(math.nan)
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"SUPPORT2 record {source_id}: non-numeric {name} {raw!r}"
            ) from exc
        if not math.isfinite(value):
            raise ValueError(f"SUPPORT2 record {source_id}: {name} must be finite")
        features.append(value)

    for name, levels in DCM_SUPPORT2_CATEGORIES:
        raw = record.get(name)
        level = None if _is_missing(raw) else str(raw).strip()
        if level is not None and level not in levels:
            raise ValueError(f"SUPPORT2 record {source_id}: unknown {name} {raw!r}")
        features.extend(1.0 if level == candidate else 0.0 for candidate in levels)

    labels = []
    for name in ("d.time", "death"):
        raw = record.get(name)
        try:
            labels.append(float(raw))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"SUPPORT2 record {source_id}: missing or non-numeric {name}"
            ) from exc
    time, event = labels
    if not math.isfinite(time) or time <= 0:
        raise ValueError(f"SUPPORT2 record {source_id}: d.time must be positive")
    if event not in (0.0, 1.0):
        raise ValueError(f"SUPPORT2 record {source_id}: death must be 0 or 1")
    return features, time, event


class SurvivalDCMSupport2(BaseTask):
    """Time to death for SUPPORT2, in the Deep Cox Mixtures representation.

    Use with :class:`~pyhealth.datasets.Support2Dataset`. Following Nagpal et
    al. (2021), this keeps all 9,105 patients and encodes 18 numeric covariates
    and 6 one-hot coded categoricals as ``features: [44]``, in the order of
    ``DCM_SUPPORT2_FEATURES``. Labels are ``duration: [days]`` and
    ``event: [indicator]``, with event 1 for death and 0 for censoring.

    This differs from :class:`~pyhealth.tasks.SurvivalTimeToEventSupport2`,
    which uses the DeepSurv benchmark representation (8,873 complete cases,
    27 features).

    Without ``feature_mean`` and ``feature_scale`` the features are raw, and a
    missing numeric is NaN. Raw output is only for fitting the statistics; do
    not train on it. With both supplied, every feature is standardized and a
    missing numeric becomes 0, which is mean imputation.

    To match the paper's preprocessing, fit the statistics on raw features:
    ``feature_mean`` is each column's mean over observed values, and
    ``feature_scale`` is each column's standard deviation (population
    standard deviation, ``ddof=0``, as in sklearn's ``StandardScaler``) after
    its missing values are replaced by that mean. This is smaller than the
    standard deviation of the observed values alone for columns with many
    missing values.

    Note:
        The original authors fit the imputer and scaler on all patients before
        cross-validation. Fit the statistics on training patients only to keep
        test-fold information out of training. This is a deliberate departure
        from the released code.

    Args:
        feature_mean: Optional 44 means fitted on training patients only.
        feature_scale: Matching 44 positive standard deviations. Supply both
            arguments or neither. Use a scale of 1 for a constant feature.

    Examples:
        >>> from pyhealth.tasks import SurvivalDCMSupport2
        >>> task = SurvivalDCMSupport2()
        >>> task.output_schema
        {'duration': 'tensor', 'event': 'tensor'}
        >>> from pyhealth.tasks.survival_dcm_support2 import DCM_SUPPORT2_FEATURES
        >>> len(DCM_SUPPORT2_FEATURES), DCM_SUPPORT2_FEATURES[18]
        (44, 'sex=female')
    """

    task_name = "SurvivalDCMSupport2"
    input_schema: ClassVar[dict[str, str]] = {"features": "tensor"}
    output_schema: ClassVar[dict[str, str]] = {"duration": "tensor", "event": "tensor"}

    def __init__(
        self,
        feature_mean: Sequence[float] | None = None,
        feature_scale: Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        if (feature_mean is None) != (feature_scale is None):
            raise ValueError("Supply both feature_mean and feature_scale, or neither")
        self.feature_mean: list[float] | None = None
        self.feature_scale: list[float] | None = None
        if feature_mean is not None:
            mean = np.asarray(feature_mean, dtype=float)
            scale = np.asarray(feature_scale, dtype=float)
            n = len(DCM_SUPPORT2_FEATURES)
            if mean.shape != (n,) or scale.shape != (n,):
                raise ValueError(f"Expected {n} means and {n} scales")
            if not np.isfinite(mean).all() or not np.isfinite(scale).all():
                raise ValueError("feature_mean and feature_scale must be finite")
            if (scale <= 0).any():
                raise ValueError("feature_scale must be positive")
            self.feature_mean, self.feature_scale = mean.tolist(), scale.tolist()

    def __call__(self, patient: Patient) -> list[dict]:
        """Return one sample for the patient.

        Args:
            patient: Patient with exactly one ``support2`` event.

        Returns:
            One sample with ``patient_id``, ``features``, ``duration`` and
            ``event``.

        Raises:
            ValueError: For invalid values or not exactly one record.
        """
        events = patient.get_events("support2")
        if len(events) != 1:
            raise ValueError(
                f"SUPPORT2 record {patient.patient_id}: expected exactly one "
                f"support2 event, got {len(events)}"
            )
        features, time, event = _dcm_support2_values(
            events[0].attr_dict, patient.patient_id
        )
        if self.feature_mean is not None:
            values = (np.asarray(features) - self.feature_mean) / self.feature_scale
            features = np.where(np.isnan(values), 0.0, values).tolist()
        return [
            {
                "patient_id": patient.patient_id,
                "features": features,
                "duration": [time],
                "event": [event],
            }
        ]
