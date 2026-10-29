"""Authors: Aritra Saha and Ugochukwu Njoku.

Duration/event tasks for GBSG and SUPPORT2, compatible with PyHealth SurvivalMDN.
Paper: Survival Mixture Density Networks, Han et al., MLHC 2022.
https://proceedings.mlr.press/v182/han22a.html
"""

import math
from collections.abc import Mapping, Sequence
from typing import ClassVar

import numpy as np

from pyhealth.data import Patient

from .base_task import BaseTask

GBSG_FEATURES = tuple(f"x{i}" for i in range(7))
_FLOAT32_MAX = float(np.finfo(np.float32).max)


def _gbsg_values(record: Mapping, source_id: str) -> tuple[list[float], float, float]:
    """Parse one source record without imputation or label repair."""
    values = []
    for field in (*GBSG_FEATURES, "time", "event"):
        try:
            value = float(record[field])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"GBSG record {source_id}: missing/non-numeric {field}"
            ) from exc
        if not math.isfinite(value) or abs(value) > _FLOAT32_MAX:
            raise ValueError(f"GBSG record {source_id}: {field} must be finite float32")
        values.append(value)
    features, time, event = values[:7], values[7], values[8]
    if time <= 0 or np.float32(time) <= 0:
        raise ValueError(f"GBSG record {source_id}: time must be strictly positive")
    if event not in (0, 1):
        raise ValueError(
            f"GBSG record {source_id}: event must be 1 observed or 0 censored"
        )
    return features, time, event


class SurvivalTimeToEventGBSG(BaseTask):
    """Extract seven ordered features and GBSG duration/event labels.

    By default, times remain in source months with no normalization or offset.
    The benchmark endpoint is recurrence-free survival. The task emits
    ``features: [7]``, ``duration: [time]`` and ``event: [indicator]``, with 1
    for an observed event and 0 for censoring. These outputs match PyHealth's
    accepted SurvivalMDN, which sets ``mode=None`` and supplies its own loss
    to the standard Trainer. No model-side time scaling is assumed.

    Args:
        feature_mean: Optional seven means fitted on training records only.
        feature_scale: Matching seven positive training standard deviations.
            Supply both arguments or neither. Constant features should use
            a scale of 1, as in sklearn's StandardScaler. Without these
            arguments, features are returned in their source representation.
        time_divisor: Positive divisor applied to every time. Default 1.0
            (source months).
        time_offset: Non-negative value added after dividing. Default 0.0.
            The emitted time is ``time / time_divisor + time_offset``. The
            Survival MDN authors' protocol uses the largest time in the full
            2,232-record cohort as the divisor and an offset of 0.002; see
            :func:`~pyhealth.datasets.gbsg_paper_time_transform`.

    Examples:
        >>> from pyhealth.tasks import SurvivalTimeToEventGBSG
        >>> task = SurvivalTimeToEventGBSG()
        >>> task.output_schema
        {'duration': 'tensor', 'event': 'tensor'}
        >>> paper = SurvivalTimeToEventGBSG(time_divisor=87.36, time_offset=0.002)
        >>> paper.time_offset
        0.002
    """

    task_name = "SurvivalTimeToEventGBSG"
    input_schema: ClassVar[dict[str, str]] = {"features": "tensor"}
    output_schema: ClassVar[dict[str, str]] = {"duration": "tensor", "event": "tensor"}

    def __init__(
        self,
        feature_mean: Sequence[float] | None = None,
        feature_scale: Sequence[float] | None = None,
        time_divisor: float = 1.0,
        time_offset: float = 0.0,
    ) -> None:
        super().__init__()
        time_divisor, time_offset = float(time_divisor), float(time_offset)
        if not math.isfinite(time_divisor) or time_divisor <= 0:
            raise ValueError("time_divisor must be finite and positive")
        if not math.isfinite(time_offset) or time_offset < 0:
            raise ValueError("time_offset must be finite and non-negative")
        self.time_divisor, self.time_offset = time_divisor, time_offset
        if (feature_mean is None) != (feature_scale is None):
            raise ValueError("Supply both feature_mean and feature_scale, or neither")
        self.feature_mean = None if feature_mean is None else list(feature_mean)
        self.feature_scale = None if feature_scale is None else list(feature_scale)
        if self.feature_mean is not None:
            mean = np.asarray(self.feature_mean, dtype=float)
            scale = np.asarray(self.feature_scale, dtype=float)
            if (
                mean.shape != (7,)
                or scale.shape != (7,)
                or not np.isfinite(mean).all()
                or not np.isfinite(scale).all()
                or (scale <= 0).any()
            ):
                raise ValueError(
                    "Expected seven finite means and seven positive scales"
                )
            self.feature_mean = mean.tolist()
            self.feature_scale = scale.tolist()

    def __call__(self, patient: Patient) -> list[dict]:
        """Return one sample, rejecting missing or multiple source records.

        Args:
            patient: Patient containing exactly one ``gbsg`` event. Its ID
                is a surrogate source-row identifier, not an original ID.

        Returns:
            One sample with the source ID, features, duration and event.

        Raises:
            ValueError: For invalid features/labels or ambiguous source rows.
        """
        events = patient.get_events("gbsg")
        if len(events) != 1:
            raise ValueError(
                f"GBSG record {patient.patient_id}: expected exactly one gbsg event, "
                f"got {len(events)}"
            )
        features, time, event = _gbsg_values(events[0].attr_dict, patient.patient_id)
        if self.feature_mean is not None:
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                features = (
                    (np.asarray(features) - self.feature_mean) / self.feature_scale
                ).tolist()
            if not all(math.isfinite(v) and abs(v) <= _FLOAT32_MAX for v in features):
                raise ValueError("Standardized GBSG features must be finite float32")
        time = time / self.time_divisor + self.time_offset
        if not (time > 0 and np.float32(time) > 0):
            raise ValueError(
                f"GBSG record {patient.patient_id}: converted time must be positive"
            )
        return [
            {
                "patient_id": patient.patient_id,
                "source_id": patient.patient_id,
                "features": features,
                "duration": [time],
                "event": [event],
            }
        ]


# SUPPORT2: the DeepSurv benchmark representation used by Survival MDN.
SUPPORT2_NUMERIC = ("age", "meanbp", "hrt", "resp", "temp", "sod", "wblc", "crea")
# Categorical levels in the benchmark's code order. The first level of each
# is the reference category and gets no column (drop-first one-hot coding).
# Missing race is its own level, as in the benchmark.
SUPPORT2_CATEGORIES = (
    ("sex", ("male", "female")),
    ("num.co", tuple(str(i) for i in range(10))),
    ("race", ("other", "white", "black", "hispanic", "asian", "missing")),
    ("diabetes", ("0", "1")),
    ("dementia", ("0", "1")),
    ("ca", ("metastatic", "no", "yes")),
)
SUPPORT2_FEATURES = SUPPORT2_NUMERIC + tuple(
    f"{name}={level}" for name, levels in SUPPORT2_CATEGORIES for level in levels[1:]
)


def _blank(value) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _support2_values(
    record: Mapping, source_id: str
) -> tuple[list[float], float, float] | None:
    """Encode one SUPPORT2 row, or return None outside the complete-case cohort.

    The benchmark keeps the 8,873 of 9,105 patients with all eight numeric
    covariates observed; other rows return None. Unknown categories, and a
    missing value in a categorical other than race, raise instead.
    """
    if any(_blank(record.get(name)) for name in SUPPORT2_NUMERIC):
        return None
    features = []
    for name in SUPPORT2_NUMERIC:
        try:
            value = float(record[name])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"SUPPORT2 record {source_id}: non-numeric {name}"
            ) from exc
        if not math.isfinite(value) or abs(value) > _FLOAT32_MAX:
            raise ValueError(
                f"SUPPORT2 record {source_id}: {name} must be finite float32"
            )
        features.append(value)
    for name, levels in SUPPORT2_CATEGORIES:
        raw = record.get(name)
        if _blank(raw):
            if name != "race":
                raise ValueError(f"SUPPORT2 record {source_id}: missing {name}")
            level = "missing"
        else:
            level = str(raw).strip()
            if name in ("num.co", "diabetes", "dementia"):
                try:
                    level = str(int(float(level)))
                except ValueError:
                    pass
        if level not in levels:
            raise ValueError(f"SUPPORT2 record {source_id}: unknown {name} {raw!r}")
        features.extend(1.0 if level == other else 0.0 for other in levels[1:])
    try:
        time, event = float(record["d.time"]), float(record["death"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"SUPPORT2 record {source_id}: missing d.time/death") from exc
    if not math.isfinite(time) or time <= 0:
        raise ValueError(f"SUPPORT2 record {source_id}: d.time must be positive")
    if event not in (0, 1):
        raise ValueError(f"SUPPORT2 record {source_id}: death must be 0 or 1")
    return features, time, event


class SurvivalTimeToEventSupport2(BaseTask):
    """Time to death for SUPPORT2, in the DeepSurv benchmark representation.

    Use with :class:`~pyhealth.datasets.Support2Dataset`. Like the DeepSurv
    benchmark and the Survival MDN paper, this keeps the 8,873 patients
    with all eight numeric covariates observed (age, mean blood pressure,
    heart rate, respiratory rate, temperature, sodium, white blood cell
    count, creatinine); other patients produce no sample. Six categorical
    covariates (sex, number of comorbidities, race, diabetes, dementia,
    cancer status) are one-hot coded with the first level dropped, giving
    ``features: [27]`` in the order of ``SUPPORT2_FEATURES``. Missing race
    is its own category. Labels are ``duration: [time]`` and ``event: [indicator]``,
    with event 1 for death and 0 for censoring.

    Args:
        feature_mean: Optional 27 means fitted on training patients only.
        feature_scale: Matching 27 positive standard deviations. Supply both
            arguments or neither. Use a scale of 1 for a constant feature,
            as sklearn's StandardScaler does.
        time_divisor: Positive divisor for ``d.time`` (days). Default 1.0.
        time_offset: Non-negative value added after dividing. Default 0.0.
            The Survival MDN authors use 365.25 and 0.001 (years plus their
            loader's offset); see
            :func:`~pyhealth.datasets.support2_paper_time_transform`.

    Examples:
        >>> from pyhealth.tasks import SurvivalTimeToEventSupport2
        >>> task = SurvivalTimeToEventSupport2(time_divisor=365.25, time_offset=0.001)
        >>> task.output_schema
        {'duration': 'tensor', 'event': 'tensor'}
        >>> from pyhealth.tasks.survival_time_to_event import SUPPORT2_FEATURES
        >>> len(SUPPORT2_FEATURES), SUPPORT2_FEATURES[8]
        (27, 'sex=female')
    """

    task_name = "SurvivalTimeToEventSupport2"
    input_schema: ClassVar[dict[str, str]] = {"features": "tensor"}
    output_schema: ClassVar[dict[str, str]] = {"duration": "tensor", "event": "tensor"}

    def __init__(
        self,
        feature_mean: Sequence[float] | None = None,
        feature_scale: Sequence[float] | None = None,
        time_divisor: float = 1.0,
        time_offset: float = 0.0,
    ) -> None:
        super().__init__()
        time_divisor, time_offset = float(time_divisor), float(time_offset)
        if not math.isfinite(time_divisor) or time_divisor <= 0:
            raise ValueError("time_divisor must be finite and positive")
        if not math.isfinite(time_offset) or time_offset < 0:
            raise ValueError("time_offset must be finite and non-negative")
        self.time_divisor, self.time_offset = time_divisor, time_offset
        if (feature_mean is None) != (feature_scale is None):
            raise ValueError("Supply both feature_mean and feature_scale, or neither")
        self.feature_mean = self.feature_scale = None
        if feature_mean is not None:
            mean = np.asarray(feature_mean, dtype=float)
            scale = np.asarray(feature_scale, dtype=float)
            n = len(SUPPORT2_FEATURES)
            if (
                mean.shape != (n,)
                or scale.shape != (n,)
                or not np.isfinite(mean).all()
                or not np.isfinite(scale).all()
                or (scale <= 0).any()
            ):
                raise ValueError(f"Expected {n} finite means and {n} positive scales")
            self.feature_mean, self.feature_scale = mean.tolist(), scale.tolist()

    def __call__(self, patient: Patient) -> list[dict]:
        """Return one sample, or none for a patient outside the cohort.

        Args:
            patient: Patient with exactly one ``support2`` event; the patient
                ID is SUPPORT2's patient number (``sno``).

        Returns:
            ``[]`` if a numeric covariate is missing, else one sample with
            ``patient_id``, ``features``, ``duration`` and ``event``.

        Raises:
            ValueError: For invalid values or not exactly one record.
        """
        events = patient.get_events("support2")
        if len(events) != 1:
            raise ValueError(
                f"SUPPORT2 record {patient.patient_id}: expected exactly one "
                f"support2 event, got {len(events)}"
            )
        parsed = _support2_values(events[0].attr_dict, patient.patient_id)
        if parsed is None:
            return []
        features, time, event = parsed
        if self.feature_mean is not None:
            features = (
                (np.asarray(features) - self.feature_mean) / self.feature_scale
            ).tolist()
            if not all(math.isfinite(v) and abs(v) <= _FLOAT32_MAX for v in features):
                raise ValueError(
                    "Standardized SUPPORT2 features must be finite float32"
                )
        time = time / self.time_divisor + self.time_offset
        if not (time > 0 and np.float32(time) > 0):
            raise ValueError(
                f"SUPPORT2 record {patient.patient_id}: converted time must be positive"
            )
        return [
            {
                "patient_id": patient.patient_id,
                "features": features,
                "duration": [time],
                "event": [event],
            }
        ]
