"""MEDS timelines for the ETHOS generative model, and its zero-shot outcomes.

ETHOS is trained without labels, on whole patient timelines, and then predicts
an outcome by sampling future timelines from the prediction time:

* :class:`EthosTimelineMEDS` gives the training timelines.
* :class:`EthosOutcomeMEDS` runs an outcome task from
  :mod:`~pyhealth.tasks.meds_outcome_tasks`, keeps its cohort and labels, and
  swaps its code list for the same rows in the form ETHOS tokenizes, cut at the
  same prediction time.
* :class:`EthosRolloutRule` says when a sampled timeline stops and how it counts
  toward the risk. ``EthosOutcomeMEDS`` builds the matching one.

Rollout rules follow the ETHOS-ARES code that produced the paper's Table S1
(``src/ethos/datasets/{mimic_icu,readmission,hospital_mortality}.py`` and
``notebooks/{figures,hosp_readmission}.ipynb``, commit 2d54383). MIT License.
Copyright (c) 2024 Paweł Renc.

References:
    Renc P., Grzeszczyk M. K., Oufattole N., et al. Foundation Model of
    Electronic Medical Records for Adaptive Risk Estimation. arXiv:2502.06124.
    https://arxiv.org/abs/2502.06124

    Source code: https://github.com/ipolharvard/ethos-ares
"""

from dataclasses import dataclass
from typing import Any, ClassVar

import polars as pl

from ..processors.ethos_processor import STATIC_CODE_PREFIXES, TIMELINE_END_TOKEN
from .base_task import BaseTask
from .meds_outcome_tasks import (
    ICU_ADMISSION_PREFIX,
    HospitalReadmissionMEDS,
    ICUAdmissionMEDS,
    ProlongedLengthOfStayMEDS,
    _completed_stays,
    _load_events,
    _rows_at_admission,
    _rows_at_discharge,
)

# MEDS codes, the same as in in_hospital_mortality_meds.py and
# meds_outcome_tasks.py, kept here so this module does not import them.
ADMISSION_PREFIX = "HOSPITAL_ADMISSION"
DISCHARGE_PREFIX = "HOSPITAL_DISCHARGE"
DIED_CODE = "HOSPITAL_DISCHARGE//DIED"
DEATH_CODE = "MEDS_DEATH"

#: Why a rollout stopped. The first three are ETHOS's.
STOP_TOKEN = "token_of_interest"
TIME_LIMIT = "time_limit"
INVALID_TOKEN = "key_error"
TOKEN_LIMIT = "token_limit"

#: ETHOS's default limit on simulated time.
DEFAULT_TIME_LIMIT_DAYS = 365.25 * 2


@dataclass(frozen=True)
class EthosRolloutRule:
    """When a sampled timeline stops, and how it counts toward a risk.

    The risk of a sample is the share of positive rollouts among those not
    dropped, as in ETHOS's ``preprocess_inference_results``. Rollouts stopped by
    an invalid token or by ``max_new_tokens`` are always dropped.

    Args:
        stop_prefixes: Code prefixes that end a rollout.
        positive_prefixes: A rollout stopped by one of these is positive.
        time_limit_days: Simulated time at which a rollout gives up.
        min_days: A rollout stopped at least this long after the prediction
            time is positive, whatever stopped it.
        drop_prefixes: Rollouts stopped by one of these are dropped.
        keep_time_limited: Count rollouts that reach the time limit as
            negative. By default they are dropped.

    Examples:
        >>> from pyhealth.tasks import EthosRolloutRule
        >>> rule = EthosRolloutRule(
        ...     stop_prefixes=("HOSPITAL_DISCHARGE", "MEDS_DEATH", "TIMELINE_END"),
        ...     positive_prefixes=("HOSPITAL_DISCHARGE//DIED", "MEDS_DEATH"),
        ...     drop_prefixes=("TIMELINE_END",),
        ... )
        >>> rule.score("MEDS_DEATH", "token_of_interest", days=3.0)
        True
        >>> rule.score("HOSPITAL_DISCHARGE//HOME", "token_of_interest", days=3.0)
        False
        >>> rule.score("TIMELINE_END", "token_of_interest", days=3.0) is None
        True
    """

    stop_prefixes: tuple[str, ...]
    positive_prefixes: tuple[str, ...] = ()
    time_limit_days: float = DEFAULT_TIME_LIMIT_DAYS
    min_days: float | None = None
    drop_prefixes: tuple[str, ...] = ()
    keep_time_limited: bool = False

    def score(self, code: str | None, reason: str, days: float) -> bool | None:
        """Scores one rollout.

        Args:
            code: The last generated token.
            reason: Why the rollout stopped.
            days: Simulated days since the prediction time.

        Returns:
            Whether the rollout is positive, or ``None`` if it is dropped.
        """
        if reason == TIME_LIMIT:
            return False if self.keep_time_limited else None
        if reason != STOP_TOKEN or code.startswith(self.drop_prefixes):
            return None
        if code.startswith(self.positive_prefixes):
            return True
        return self.min_days is not None and days >= self.min_days


def _rollout_rule(task: BaseTask) -> EthosRolloutRule:
    """Returns the rollout rule that matches an outcome task's label.

    Args:
        task: An outcome task from :mod:`.meds_outcome_tasks`.

    Returns:
        The rule, using the task's ``death_as_positive``, ``window_days`` or
        ``threshold_days``.
    """
    if isinstance(task, ICUAdmissionMEDS):
        deaths = (DEATH_CODE, DIED_CODE) if task.death_as_positive else ()
        return EthosRolloutRule(
            stop_prefixes=(
                ICU_ADMISSION_PREFIX,
                DISCHARGE_PREFIX,
                DEATH_CODE,
                TIMELINE_END_TOKEN,
            ),
            positive_prefixes=(ICU_ADMISSION_PREFIX, *deaths),
            drop_prefixes=(TIMELINE_END_TOKEN,),
        )
    if isinstance(task, HospitalReadmissionMEDS):
        return EthosRolloutRule(
            stop_prefixes=(ADMISSION_PREFIX, DEATH_CODE, TIMELINE_END_TOKEN),
            positive_prefixes=(ADMISSION_PREFIX, DEATH_CODE),
            time_limit_days=task.window_days,
            keep_time_limited=True,
        )
    return EthosRolloutRule(
        stop_prefixes=(DISCHARGE_PREFIX, DEATH_CODE, TIMELINE_END_TOKEN),
        min_days=task.threshold_days,
        drop_prefixes=(TIMELINE_END_TOKEN,),
    )


def _timeline_fields(
    rows: pl.DataFrame,
    prefixes: tuple[str, ...] = STATIC_CODE_PREFIXES,
) -> dict[str, dict[str, Any]] | None:
    """Splits MEDS rows into the ``timeline`` and ``context`` sample fields.

    Args:
        rows: MEDS events, with ``timestamp``, ``meds/code`` and
            ``meds/numeric_value``.
        prefixes: Static code prefixes.

    Returns:
        The two fields, or ``None`` if no row is left for the timeline.
    """
    code = pl.col("meds/code")
    is_static = code.str.starts_with(prefixes[0])
    for prefix in prefixes[1:]:
        is_static = is_static | code.str.starts_with(prefix)
    static = rows.filter(is_static)
    timeline = rows.filter(~is_static & pl.col("timestamp").is_not_null())
    if timeline.height == 0:
        return None
    return {
        "timeline": {
            "time": timeline["timestamp"].to_list(),
            "code": timeline["meds/code"].to_list(),
            "numeric_value": timeline["meds/numeric_value"].to_list(),
        },
        "context": {
            "time": static["timestamp"].to_list(),
            "code": static["meds/code"].to_list(),
            "start": timeline["timestamp"].min(),
        },
    }


class EthosTimelineMEDS(BaseTask):
    """One whole MEDS timeline per patient, for training ETHOS.

    There is no label: :class:`~pyhealth.models.ETHOS` learns to predict the
    next token. Fit on the training patients only, with
    ``set_task(EthosTimelineMEDS(), split=PatientSplit(...))``.

    Attributes:
        task_name (str): The name of the task.
        input_schema (Dict[str, str]): ``timeline``, the tokenized events, and
            ``context``, the static patient context.
        output_schema (Dict[str, str]): Empty.

    Examples:
        >>> from pyhealth.datasets import MEDSDataset, PatientSplit
        >>> from pyhealth.tasks import EthosTimelineMEDS
        >>> dataset = MEDSDataset(root="/path/to/meds")  # doctest: +SKIP
        >>> train, test = dataset.set_task(  # doctest: +SKIP
        ...     EthosTimelineMEDS(), split=PatientSplit(ratios=(0.9, 0.1), seed=0)
        ... )
    """

    task_name: str = "EthosTimelineMEDS"
    input_schema: ClassVar[dict[str, str]] = {
        "timeline": "ethos",
        "context": "ethos_context",
    }
    output_schema: ClassVar[dict[str, str]] = {}

    def pre_filter(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Restricts the global scan to MEDS events."""
        return df.filter(pl.col("event_type") == "meds")

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        fields = _timeline_fields(patient.get_events(event_type="meds", return_df=True))
        if fields is None:
            return []
        return [{"patient_id": patient.patient_id, **fields}]


class EthosOutcomeMEDS(BaseTask):
    """An outcome task's cohort and labels, as ETHOS timelines.

    Runs ``outcome_task(**options)`` and replaces each sample's ``codes`` with
    the same rows, so that the cohort, the label and every leakage exclusion
    stay the outcome task's own. The timeline ends at the prediction time, and
    :attr:`rule` says how rollouts from there become a risk:

    * :class:`~pyhealth.tasks.ICUAdmissionMEDS`, at admission: stop at ICU
      admission, discharge, death or the end of the timeline. An ICU admission
      is positive, and so is a death if ``death_as_positive`` is on, including
      a discharge coded ``HOSPITAL_DISCHARGE//DIED``. ETHOS's Table S1 uses the
      ICU-only label, ``death_as_positive=False``. The two-year time limit and
      ``TIMELINE_END`` drop a rollout.
    * :class:`~pyhealth.tasks.ProlongedLengthOfStayMEDS`, at admission: stop at
      discharge, death or the end of the timeline. A rollout is positive if it
      stops at least ``threshold_days`` after admission, inclusive. The
      two-year time limit and ``TIMELINE_END`` drop a rollout.
    * :class:`~pyhealth.tasks.HospitalReadmissionMEDS`, at discharge: stop at
      admission, death or the end of the timeline. An admission or a death is
      positive. A rollout that passes ``window_days``, or reaches
      ``TIMELINE_END``, is negative. ``require_full_followup`` only changes the
      cohort, not the rollout.

    Pass the fitted processors of :class:`EthosTimelineMEDS` as
    ``input_processors``, so that both tasks share one vocabulary.

    Args:
        outcome_task (type[BaseTask]): One of the three task classes above.
        **options: Keyword arguments for ``outcome_task``.

    Attributes:
        task_name (str): ``Ethos`` and the outcome task's name.
        input_schema (Dict[str, str]): Same as :class:`EthosTimelineMEDS`.
        output_schema (Dict[str, str]): The outcome task's.
        rule (EthosRolloutRule): How rollouts become a risk.

    Raises:
        ValueError: If ``outcome_task`` is not one of the three.

    Examples:
        >>> from pyhealth.tasks import EthosOutcomeMEDS, ICUAdmissionMEDS
        >>> task = EthosOutcomeMEDS(ICUAdmissionMEDS, death_as_positive=False)
        >>> task.task_name
        'EthosICUAdmissionMEDS'
        >>> task.output_schema
        {'icu_admission': 'binary'}
        >>> task.rule.positive_prefixes
        ('ICU_ADMISSION',)
    """

    input_schema: ClassVar[dict[str, str]] = EthosTimelineMEDS.input_schema

    def __init__(self, outcome_task: type[BaseTask], **options: Any) -> None:
        if outcome_task not in (
            ICUAdmissionMEDS,
            HospitalReadmissionMEDS,
            ProlongedLengthOfStayMEDS,
        ):
            raise ValueError(
                "outcome_task must be ICUAdmissionMEDS, HospitalReadmissionMEDS "
                f"or ProlongedLengthOfStayMEDS, got {outcome_task}."
            )
        super().__init__()
        # The class and options rather than an instance: dataset caches hash
        # this object's attributes, and an instance's text form changes per run.
        self.outcome_task = outcome_task
        self.options = options
        self.task_name = f"Ethos{outcome_task.task_name}"
        self.output_schema = dict(outcome_task.output_schema)
        self.rule = _rollout_rule(outcome_task(**options))

    def pre_filter(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Restricts the global scan to MEDS events."""
        return df.filter(pl.col("event_type") == "meds")

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        outcomes = self.outcome_task(**self.options)(patient)
        if not outcomes:
            return []

        events = _load_events(patient)
        stays = {stay["hadm_id"]: stay for stay in _completed_stays(events)}
        cut = (
            _rows_at_discharge
            if self.outcome_task is HospitalReadmissionMEDS
            else _rows_at_admission
        )
        samples = []
        for outcome in outcomes:
            fields = _timeline_fields(cut(events, stays[outcome["hadm_id"]]))
            if fields is not None:
                kept = {key: val for key, val in outcome.items() if key != "codes"}
                samples.append({**kept, **fields})
        return samples
