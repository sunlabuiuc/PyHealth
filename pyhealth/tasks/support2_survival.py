# Contributor: Xingyao Wang (xingyao6)
# Paper: Deep Cox Mixtures for Survival Regression
# https://proceedings.mlr.press/v149/nagpal21a.html
"""SUPPORT2 task that keeps censored follow-up time and the death indicator."""

import math
from typing import Any, ClassVar, Optional

from pyhealth.data import Patient

from .base_task import BaseTask

#: Categorical SUPPORT2 covariates used by Deep Cox Mixtures (Nagpal et al.).
SUPPORT2_DCM_CATEGORICAL = ("sex", "dzgroup", "dzclass", "income", "race", "ca")
#: Numeric SUPPORT2 covariates used by Deep Cox Mixtures (Nagpal et al.).
SUPPORT2_DCM_NUMERIC = (
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
#: All 24 covariates, in the order the task and the examples use them.
SUPPORT2_DCM_FEATURES = SUPPORT2_DCM_CATEGORICAL + SUPPORT2_DCM_NUMERIC


def _as_float(value: Any, name: str) -> float:
    """Convert a label value to float, rejecting missing or non-numeric values.

    Args:
        value: Raw attribute value.
        name: Field name for the error message.

    Returns:
        The value as a finite or nonfinite float (callers check finiteness).

    Raises:
        ValueError: The value is missing or cannot be converted.
    """
    if value is None:
        raise ValueError(f"SUPPORT2 {name} is missing")
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"SUPPORT2 {name} is not numeric: {value!r}") from error


class Support2SurvivalTask(BaseTask):
    """Censored time-to-death samples with 24 raw baseline SUPPORT2 covariates.

    Works with :class:`~pyhealth.datasets.Support2Dataset`. Each patient has
    one SUPPORT2 row and yields one sample:

    - ``covariates``: dictionary of the 24 covariates used by Deep Cox
      Mixtures (6 categorical, 18 numeric; see ``SUPPORT2_DCM_FEATURES``).
      Values are raw; a missing value is ``None``. No imputation, encoding or
      scaling is fitted here, so each training split can fit its own.
    - ``time``: follow-up in days (``d.time``), positive.
    - ``event``: 1 if the patient died during follow-up (``death``), 0 if
      censored.

    The dataset's derived survival estimates (``surv2m``, ``surv6m``,
    ``prg2m``, ...), the outcome fields and hospital-death fields are never
    used as inputs. Most measurements are from study day 3, so this is not an
    admission-time prediction task.

    How it differs from the existing SUPPORT2 tasks:
    :class:`~pyhealth.tasks.SurvivalPreprocessSupport2` predicts the
    dataset's model-derived 2- or 6-month survival probability (a regression
    label without censoring), and
    :class:`~pyhealth.tasks.SurvivalTimeToEventSupport2` follows the
    DeepSurv/Survival MDN protocol with 27 encoded values and drops rows with
    missing numeric values. This task keeps every row, the raw covariates and
    the censoring indicator.

    Examples:
        >>> from pyhealth.datasets import Support2Dataset
        >>> from pyhealth.tasks import Support2SurvivalTask
        >>> task = Support2SurvivalTask()
        >>> task.output_schema
        {'time': 'raw', 'event': 'raw'}
        >>> # dataset = Support2Dataset(root="/path/to/support2")
        >>> # samples = dataset.set_task(task)
    """

    #: Task identifier used in cache paths.
    task_name: str = "support2_censored_survival"
    #: One raw dictionary of covariates; fit preprocessing per training split.
    input_schema: ClassVar[dict[str, str]] = {"covariates": "raw"}
    #: Raw follow-up time (days) and event indicator (1 death, 0 censored).
    output_schema: ClassVar[dict[str, str]] = {"time": "raw", "event": "raw"}

    def __call__(self, patient: Patient) -> list[dict[str, Any]]:
        """Convert one patient into at most one censored survival sample.

        Args:
            patient: Patient with at most one ``support2`` event.

        Returns:
            ``[{"patient_id", "covariates", "time", "event"}]``, or ``[]`` when
            the patient has no SUPPORT2 row.

        Raises:
            ValueError: More than one SUPPORT2 row, or ``d.time`` is missing,
                nonfinite or not positive, or ``death`` is not 0/1.
        """
        events = patient.get_events(event_type="support2")
        if not events:
            return []
        if len(events) != 1:
            raise ValueError("SUPPORT2 requires exactly one baseline row per patient")
        row = events[0]
        time = _as_float(getattr(row, "d.time", None), "d.time")
        death = _as_float(getattr(row, "death", None), "death")
        if not math.isfinite(time) or time <= 0:
            raise ValueError(f"SUPPORT2 d.time must be positive and finite: {time}")
        if death not in (0.0, 1.0):
            raise ValueError(f"SUPPORT2 death must be 0 or 1: {death}")
        covariates: dict[str, Optional[Any]] = {
            key: getattr(row, key, None) for key in SUPPORT2_DCM_FEATURES
        }
        return [
            {
                "patient_id": str(patient.patient_id),
                "covariates": covariates,
                "time": time,
                "event": int(death),
            }
        ]
