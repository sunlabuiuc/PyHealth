"""Hospital outcome prediction tasks for the Medical Event Data Standard (MEDS).

This module provides three MEDS-native outcome tasks that work with any
PyHealth model taking a code sequence:

* :class:`ICUAdmissionMEDS`: ICU admission during a hospital stay.
* :class:`HospitalReadmissionMEDS`: readmission or death within 30 days of
  discharge.
* :class:`ProlongedLengthOfStayMEDS`: a stay of at least 10 days.

The cohorts and labels follow the ETHOS-ARES reference code. In-hospital
mortality already exists as
:class:`~pyhealth.tasks.in_hospital_mortality_meds.InHospitalMortalityMEDS`,
which observes the stay itself rather than predicting at admission.

Task definitions
----------------
As in ``InHospitalMortalityMEDS``, a *stay* is the set of events sharing one
``hadm_id``, with admission time ``t_a`` (earliest ``HOSPITAL_ADMISSION//*``)
and discharge time ``t_d`` (latest ``HOSPITAL_DISCHARGE//*``). Stays without
both events, or with ``t_d <= t_a``, produce no sample. A stay *ended in death*
if its discharge code is ``HOSPITAL_DISCHARGE//DIED`` or a ``MEDS_DEATH`` event
falls in ``[t_a, t_d]``. The discharge code is the one stamped ``t_d``; if a
stay has two discharge events at exactly ``t_d``, which one is taken is not
guaranteed to be stable.

ICU admission and prolonged stay predict **at hospital admission**:

* **Features.** Static events (null timestamp), every event strictly before
  ``t_a`` (including earlier stays), then this stay's admission event. Other
  events stamped exactly ``t_a`` are dropped because their order relative to
  the admission is unknown: a direct ICU admission shares that timestamp.
* **ICU label.** 1 if an ``ICU_ADMISSION//*`` event carries this ``hadm_id`` or
  falls in ``[t_a, t_d]``. By default a death in the stay is also positive
  (``death_as_positive``), on the reasoning that the patient should have been
  admitted to the ICU.
* **Prolonged stay label.** 1 if the time from ``t_a`` to the end of the stay
  is at least ``threshold_days``. The stay ends at ``t_d`` or at an earlier
  ``MEDS_DEATH``. Stays that ended in death are kept.

Readmission predicts **at discharge**:

* **Cohort.** Stays that did not end in death.
* **Features.** Static events, every event strictly before ``t_d``, and the
  events stamped ``t_d`` that carry this ``hadm_id`` (the discharge,
  diagnoses and DRG codes). Events stamped ``t_d`` from any other stay are
  dropped, so a same-instant readmission cannot reach the input, and so are
  all events of a stay admitted at or after ``t_d``: a same-day
  readmission's date-resolution codes (e.g. HCPCS at midnight) sort before
  ``t_d`` but belong to the next stay.
* **Label.** 1 if the next ``HOSPITAL_ADMISSION//*`` of another stay or the next
  ``MEDS_DEATH`` falls within ``window_days`` of ``t_d`` (inclusive).
  ``require_full_followup=True`` also drops negatives whose timeline ends
  less than ``window_days`` after discharge (the "reduced" cohort).

``MEDS_DEATH`` never enters the input of any task, and neither do events after
the prediction time. ``ICU_ADMISSION//*`` and ``ICU_DISCHARGE//*`` events of the
current stay are also dropped from admission-time input: MIMIC-IV can stamp an
ICU entry before the hospital admission it belongs to, which would otherwise
place the label in the history. ICU events of earlier stays remain.

Admission-time input also drops events of the current stay stamped at exactly
midnight before ``t_a``. These are date-resolution records such as HCPCS billing
codes, which sort before the admission on the same day but describe care given
during the stay. Timed events of the current stay before ``t_a``, such as
emergency department labs and transfers, are kept. This is a heuristic: it also
drops a timed event of the current stay that really happened at midnight.

Configuration
-------------
These tasks read ``hadm_id``, which is not part of the core MEDS schema, so
pass the bundled ``configs/meds_with_hadm.yaml`` (or your own config listing
it). ``ICUAdmissionMEDS`` also needs ``ICU_ADMISSION//*`` codes, as produced by
the MIMIC-IV MEDS event configs. On a dataset without them every label is
driven by deaths alone.

The ``ICU_ADMISSION``, ``HOSPITAL_ADMISSION`` and ``HOSPITAL_DISCHARGE`` tokens,
the prediction times and the labels are adapted from ETHOS-ARES
(``src/ethos/datasets/{mimic_icu,readmission,hospital_mortality}.py`` and
``notebooks/{icu_admission,hosp_readmission,all_task_label_dump}.ipynb``,
commit 2d54383). Copyright (c) 2024 Pawel Renc, MIT License.

References:
    Renc P., Grzeszczyk M. K., Oufattole N., et al. Foundation Model of
    Electronic Medical Records for Adaptive Risk Estimation. arXiv:2502.06124.
    https://arxiv.org/abs/2502.06124

    Source code: https://github.com/ipolharvard/ethos-ares
"""

from datetime import datetime, timedelta
from typing import Any, ClassVar

import polars as pl

from .base_task import BaseTask

# The same MEDS codes as in_hospital_mortality_meds.py, kept here so these
# tasks do not import another task's module.
ADMISSION_PREFIX = "HOSPITAL_ADMISSION"
DISCHARGE_PREFIX = "HOSPITAL_DISCHARGE"
DIED_CODE = "HOSPITAL_DISCHARGE//DIED"
DEATH_CODE = "MEDS_DEATH"

ICU_ADMISSION_PREFIX = "ICU_ADMISSION"
ICU_DISCHARGE_PREFIX = "ICU_DISCHARGE"


def _load_events(patient: Any) -> pl.DataFrame | None:
    """Returns a patient's MEDS events with an integer ``_hadm`` column.

    Args:
        patient (Any): A PyHealth ``Patient``.

    Returns:
        pl.DataFrame | None: The events, or ``None`` if there are none.
    """
    events = patient.get_events(event_type="meds", return_df=True)
    if events.height == 0:
        return None
    # Nulls promote hadm_id to float through the Dask/pandas pipeline; cast it
    # back so stays join cleanly and emitted ids stay integral.
    return events.with_columns(
        pl.col("meds/hadm_id").cast(pl.Int64, strict=False).alias("_hadm")
    )


def _completed_stays(events: pl.DataFrame) -> list[dict[str, Any]]:
    """Builds one record per completed stay, in admission order.

    Args:
        events (pl.DataFrame): Output of :func:`_load_events`.

    Returns:
        list[dict[str, Any]]: Records with ``hadm_id``, ``admit``,
        ``discharge``, ``died`` (the stay ended in death) and ``end`` (the
        earliest in-stay ``MEDS_DEATH``, otherwise ``discharge``).
    """
    code = pl.col("meds/code")
    has_hadm = pl.col("_hadm").is_not_null()
    admissions = (
        events.filter(code.str.starts_with(ADMISSION_PREFIX) & has_hadm)
        .group_by("_hadm")
        .agg(pl.col("timestamp").min().alias("admit"))
    )
    discharges = (
        events.filter(code.str.starts_with(DISCHARGE_PREFIX) & has_hadm)
        .group_by("_hadm")
        .agg(
            pl.col("timestamp").max().alias("discharge"),
            code.sort_by("timestamp").last().alias("discharge_code"),
        )
    )
    stays = (
        admissions.join(discharges, on="_hadm", how="inner")
        .filter(pl.col("discharge") > pl.col("admit"))
        .sort("admit")
    )
    deaths = events.filter(code == DEATH_CODE)["timestamp"].drop_nulls().to_list()

    records = []
    for stay in stays.iter_rows(named=True):
        admit, discharge = stay["admit"], stay["discharge"]
        in_stay_deaths = [t for t in deaths if admit <= t <= discharge]
        records.append(
            {
                "hadm_id": stay["_hadm"],
                "admit": admit,
                "discharge": discharge,
                "died": stay["discharge_code"] == DIED_CODE or bool(in_stay_deaths),
                "end": min(in_stay_deaths, default=discharge),
            }
        )
    return records


def _is_this_stay(hadm_id: int) -> pl.Expr:
    """Selects events carrying ``hadm_id``; null ids count as other stays."""
    return (pl.col("_hadm") == hadm_id).fill_null(False)


def _rows_at_admission(events: pl.DataFrame, stay: dict[str, Any]) -> pl.DataFrame:
    """Returns the input rows of an admission-time prediction.

    Args:
        events (pl.DataFrame): Output of :func:`_load_events`.
        stay (dict[str, Any]): A record from :func:`_completed_stays`.

    Returns:
        pl.DataFrame: Static rows, history before admission, then the admission
        event itself. See the module docstring for the exclusions.
    """
    code = pl.col("meds/code")
    time = pl.col("timestamp")
    this_stay = _is_this_stay(stay["hadm_id"])
    this_stay_icu = this_stay & (
        code.str.starts_with(ICU_ADMISSION_PREFIX)
        | code.str.starts_with(ICU_DISCHARGE_PREFIX)
    )
    this_stay_dated = this_stay & (time.dt.time() == pl.time(0))
    admission_event = (
        (time == stay["admit"]) & this_stay & code.str.starts_with(ADMISSION_PREFIX)
    )
    history = (time < stay["admit"]) & ~this_stay_icu & ~this_stay_dated
    window = time.is_null() | history
    return events.filter((window | admission_event) & (code != DEATH_CODE))


def _codes_at_admission(events: pl.DataFrame, stay: dict[str, Any]) -> list[str]:
    """Returns the codes of :func:`_rows_at_admission`."""
    return _rows_at_admission(events, stay)["meds/code"].to_list()


def _rows_at_discharge(events: pl.DataFrame, stay: dict[str, Any]) -> pl.DataFrame:
    """Returns the input rows of a discharge-time prediction.

    Args:
        events (pl.DataFrame): Output of :func:`_load_events`.
        stay (dict[str, Any]): A record from :func:`_completed_stays`.

    Returns:
        pl.DataFrame: Static rows, everything before discharge, then the events
        of this stay stamped at discharge. Events of a stay admitted at or
        after this discharge are dropped even when stamped earlier: a
        same-day readmission's date-resolution codes (e.g. HCPCS at
        midnight) sort before this discharge and would reveal the label.
    """
    code = pl.col("meds/code")
    time = pl.col("timestamp")
    this_stay = _is_this_stay(stay["hadm_id"])
    later_stays = (
        events.filter(
            code.str.starts_with(ADMISSION_PREFIX)
            & (time >= stay["discharge"])
            & ~this_stay
        )["_hadm"]
        .drop_nulls()
        .unique()
        .to_list()
    )
    at_discharge = (time == stay["discharge"]) & this_stay
    window = time.is_null() | (time < stay["discharge"]) | at_discharge
    later = pl.col("_hadm").is_in(later_stays).fill_null(False)
    return events.filter(window & ~later & (code != DEATH_CODE))


def _codes_at_discharge(events: pl.DataFrame, stay: dict[str, Any]) -> list[str]:
    """Returns the codes of :func:`_rows_at_discharge`."""
    return _rows_at_discharge(events, stay)["meds/code"].to_list()


def _next_outcome_time(
    events: pl.DataFrame, stay: dict[str, Any]
) -> datetime | None:
    """Returns the time of the next admission or death after a discharge.

    Args:
        events (pl.DataFrame): Output of :func:`_load_events`.
        stay (dict[str, Any]): A record from :func:`_completed_stays`.

    Returns:
        datetime | None: The earliest outcome time, or ``None`` if neither
        event follows the discharge.
    """
    code = pl.col("meds/code")
    time = pl.col("timestamp")
    readmission = (
        code.str.starts_with(ADMISSION_PREFIX)
        & (time >= stay["discharge"])
        & ~_is_this_stay(stay["hadm_id"])
    )
    death = (code == DEATH_CODE) & (time > stay["discharge"])
    return events.filter(readmission | death)["timestamp"].min()


class ICUAdmissionMEDS(BaseTask):
    """ICU admission prediction at hospital admission for MEDS datasets.

    One sample per completed stay. The input is the patient's history up to
    and including the admission event, and the binary label is whether the
    patient reached the ICU during the stay. By default a death during the
    stay also counts as positive, following the ETHOS-ARES reference code. See
    the module docstring for the full definition.

    Args:
        death_as_positive (bool): Treat a death during the stay as positive
            even without an ICU admission. Defaults to ``True``, matching
            ``notebooks/icu_admission.ipynb``. ``False`` gives the ICU-only
            label of ``notebooks/all_task_label_dump.ipynb``.
        code_mapping (Optional[Dict[str, Tuple[str, str]]]): Optional vocab
            mapping forwarded to :class:`BaseTask`.

    Attributes:
        task_name (str): The name of the task.
        input_schema (Dict[str, str]): ``codes``, the observed MEDS codes.
        output_schema (Dict[str, str]): ``icu_admission``, a binary label.

    Examples:
        >>> from pathlib import Path
        >>> import pyhealth.datasets.configs as meds_configs
        >>> from pyhealth.datasets import MEDSDataset
        >>> from pyhealth.tasks import ICUAdmissionMEDS
        >>> cfg = Path(meds_configs.__file__).parent / "meds_with_hadm.yaml"
        >>> dataset = MEDSDataset(
        ...     root="/path/to/mimic-iv-demo-meds/0.0.1",
        ...     config_path=str(cfg),
        ... )
        >>> samples = dataset.set_task(ICUAdmissionMEDS())
        >>> icu_only = ICUAdmissionMEDS(death_as_positive=False)
    """

    task_name: str = "ICUAdmissionMEDS"
    input_schema: ClassVar[dict[str, str]] = {"codes": "sequence"}
    output_schema: ClassVar[dict[str, str]] = {"icu_admission": "binary"}

    def __init__(
        self,
        death_as_positive: bool = True,
        code_mapping: dict[str, tuple[str, str]] | None = None,
    ) -> None:
        super().__init__(code_mapping=code_mapping)
        self.death_as_positive = death_as_positive

    def pre_filter(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Restricts the global scan to MEDS events."""
        return df.filter(pl.col("event_type") == "meds")

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        events = _load_events(patient)
        if events is None:
            return []

        icu_events = events.filter(
            pl.col("meds/code").str.starts_with(ICU_ADMISSION_PREFIX)
        )
        samples: list[dict[str, Any]] = []
        for stay in _completed_stays(events):
            reached_icu = (
                icu_events.filter(
                    _is_this_stay(stay["hadm_id"])
                    | pl.col("timestamp").is_between(stay["admit"], stay["discharge"])
                ).height
                > 0
            )
            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "hadm_id": stay["hadm_id"],
                    "codes": _codes_at_admission(events, stay),
                    "icu_admission": int(
                        reached_icu or (self.death_as_positive and stay["died"])
                    ),
                }
            )
        return samples


class HospitalReadmissionMEDS(BaseTask):
    """Readmission prediction at hospital discharge for MEDS datasets.

    One sample per completed stay that did not end in death. The input is the
    patient's history up to and including this stay's discharge events, and
    the binary label is whether the patient was readmitted or died within
    ``window_days``. See the module docstring for the full definition.

    Differences from
    :class:`~pyhealth.tasks.readmission_prediction.ReadmissionPredictionMIMIC4`,
    which also labels readmission within a window after discharge:

    * **Data.** This task reads any MEDS dataset through
      :class:`~pyhealth.datasets.MEDSDataset`, whose events are all in one
      ``meds`` table. ``ReadmissionPredictionMIMIC4`` reads the MIMIC-IV
      ``patients``, ``admissions``, ``diagnoses_icd``, ``procedures_icd`` and
      ``prescriptions`` tables of ``MIMIC4EHRDataset``.
    * **Input.** One code sequence with the whole history up to discharge,
      instead of the current admission's diagnoses, procedures and drugs as
      three sequences.
    * **Death.** A death within the window counts as positive, and stays that
      ended in death are excluded. ``ReadmissionPredictionMIMIC4`` does not
      look at deaths.
    * **Cohort.** Every completed stay that did not end in death is a sample,
      including a patient's last stay (negative unless the patient dies within
      the window), and ``require_full_followup`` gives the ETHOS-ARES
      "Reduced" cohort. ``ReadmissionPredictionMIMIC4`` only makes samples for
      admissions followed by another admission, skips visits missing any of
      the three code types, and drops patients under 18.
    * **Window.** 30 days by default, inclusive, from discharge to the next
      admission of another stay at or after discharge, as in ETHOS-ARES.
      ``ReadmissionPredictionMIMIC4`` uses 15 days by default, exclusive.

    Args:
        window_days (float): Outcome window after discharge, inclusive.
            Defaults to ``30.0``, matching ETHOS-ARES.
        require_full_followup (bool): Drop negatives whose timeline ends less
            than ``window_days`` after discharge, since their outcome is
            unobserved. Defaults to ``False`` (the "Full" cohort of
            ``notebooks/hosp_readmission.ipynb``); ``True`` gives the
            "Reduced" cohort.
        code_mapping (Optional[Dict[str, Tuple[str, str]]]): Optional vocab
            mapping forwarded to :class:`BaseTask`.

    Attributes:
        task_name (str): The name of the task.
        input_schema (Dict[str, str]): ``codes``, the observed MEDS codes.
        output_schema (Dict[str, str]): ``readmission``, a binary label.

    Raises:
        ValueError: If ``window_days`` is not positive.

    Examples:
        >>> from pathlib import Path
        >>> import pyhealth.datasets.configs as meds_configs
        >>> from pyhealth.datasets import MEDSDataset
        >>> from pyhealth.tasks import HospitalReadmissionMEDS
        >>> cfg = Path(meds_configs.__file__).parent / "meds_with_hadm.yaml"
        >>> dataset = MEDSDataset(
        ...     root="/path/to/mimic-iv-demo-meds/0.0.1",
        ...     config_path=str(cfg),
        ... )
        >>> samples = dataset.set_task(HospitalReadmissionMEDS())
        >>> reduced = HospitalReadmissionMEDS(require_full_followup=True)
    """

    task_name: str = "HospitalReadmissionMEDS"
    input_schema: ClassVar[dict[str, str]] = {"codes": "sequence"}
    output_schema: ClassVar[dict[str, str]] = {"readmission": "binary"}

    def __init__(
        self,
        window_days: float = 30.0,
        require_full_followup: bool = False,
        code_mapping: dict[str, tuple[str, str]] | None = None,
    ) -> None:
        if window_days <= 0:
            raise ValueError(f"window_days must be positive, got {window_days}.")
        super().__init__(code_mapping=code_mapping)
        self.window_days = float(window_days)
        self.require_full_followup = require_full_followup

    def pre_filter(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Restricts the global scan to MEDS events."""
        return df.filter(pl.col("event_type") == "meds")

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        events = _load_events(patient)
        if events is None:
            return []

        window = timedelta(days=self.window_days)
        timeline_end = events["timestamp"].max()
        samples: list[dict[str, Any]] = []
        for stay in _completed_stays(events):
            if stay["died"]:
                continue

            discharge = stay["discharge"]
            outcome_time = _next_outcome_time(events, stay)
            label = outcome_time is not None and outcome_time - discharge <= window
            censored = timeline_end - discharge < window
            if self.require_full_followup and not label and censored:
                continue

            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "hadm_id": stay["hadm_id"],
                    "codes": _codes_at_discharge(events, stay),
                    "readmission": int(label),
                }
            )
        return samples


class ProlongedLengthOfStayMEDS(BaseTask):
    """Prolonged length of stay prediction at admission for MEDS datasets.

    One sample per completed stay, including stays that ended in death. The
    input is the patient's history up to and including the admission event,
    and the binary label is whether the stay lasted at least
    ``threshold_days``. See the module docstring for the full definition.

    Args:
        threshold_days (float): Minimum length of a prolonged stay, inclusive.
            Defaults to ``10.0``, matching
            ``notebooks/all_task_label_dump.ipynb``.
        code_mapping (Optional[Dict[str, Tuple[str, str]]]): Optional vocab
            mapping forwarded to :class:`BaseTask`.

    Attributes:
        task_name (str): The name of the task.
        input_schema (Dict[str, str]): ``codes``, the observed MEDS codes.
        output_schema (Dict[str, str]): ``prolonged_los``, a binary label.

    Raises:
        ValueError: If ``threshold_days`` is not positive.

    Examples:
        >>> from pathlib import Path
        >>> import pyhealth.datasets.configs as meds_configs
        >>> from pyhealth.datasets import MEDSDataset
        >>> from pyhealth.tasks import ProlongedLengthOfStayMEDS
        >>> cfg = Path(meds_configs.__file__).parent / "meds_with_hadm.yaml"
        >>> dataset = MEDSDataset(
        ...     root="/path/to/mimic-iv-demo-meds/0.0.1",
        ...     config_path=str(cfg),
        ... )
        >>> samples = dataset.set_task(ProlongedLengthOfStayMEDS())
        >>> week = ProlongedLengthOfStayMEDS(threshold_days=7.0)
    """

    task_name: str = "ProlongedLengthOfStayMEDS"
    input_schema: ClassVar[dict[str, str]] = {"codes": "sequence"}
    output_schema: ClassVar[dict[str, str]] = {"prolonged_los": "binary"}

    def __init__(
        self,
        threshold_days: float = 10.0,
        code_mapping: dict[str, tuple[str, str]] | None = None,
    ) -> None:
        if threshold_days <= 0:
            raise ValueError(f"threshold_days must be positive, got {threshold_days}.")
        super().__init__(code_mapping=code_mapping)
        self.threshold_days = float(threshold_days)

    def pre_filter(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Restricts the global scan to MEDS events."""
        return df.filter(pl.col("event_type") == "meds")

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        events = _load_events(patient)
        if events is None:
            return []

        threshold = timedelta(days=self.threshold_days)
        return [
            {
                "patient_id": patient.patient_id,
                "hadm_id": stay["hadm_id"],
                "codes": _codes_at_admission(events, stay),
                "prolonged_los": int(stay["end"] - stay["admit"] >= threshold),
            }
            for stay in _completed_stays(events)
        ]
