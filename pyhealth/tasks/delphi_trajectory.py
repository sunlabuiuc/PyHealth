"""Disease-trajectory tasks for Delphi-2M.

Contributor: Evgeniy Tikhonov (GitHub: etikhonov-dev)

Paper: Shmatko et al., Learning the natural history of human disease with
generative transformers, Nature 2025.
https://doi.org/10.1038/s41586-025-09529-3

Each task turns one patient's events into prefix-to-next-event samples
(``codes``, ``ages``, ``next_code``, ``time_to_next``) for
:class:`~pyhealth.models.Delphi`. Ages are days since birth. Nothing here
downloads UK Biobank or PhysioNet data.
:class:`DelphiTrajectoryUKB` subclasses :class:`DelphiTrajectoryMEDS`
because a pre-extracted UK Biobank table has the same timestamp-and-code
shape as MEDS; it only changes the event type and the birth codes.
"""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta
from typing import Any, ClassVar

import polars as pl

from .base_task import BaseTask

_DAY_SECONDS = 86400.0
MEDS_BIRTH_CODES: tuple[str, ...] = (
    "MEDS_BIRTH",
    "SNOMED/3950001",
    "SNOMED/184099003",
)
UKB_BIRTH_CODES: tuple[str, ...] = MEDS_BIRTH_CODES + ("UKB_BIRTH",)


def _coerce_timestamp(value: Any) -> datetime | None:
    """Return a naive datetime, or None when ``value`` has no time."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    if isinstance(value, str):
        text = value.strip()
        if text == "" or text.lower() in {"nat", "nan", "none", "null"}:
            return None
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(text, fmt)
            except ValueError:
                continue
        if "." in text:
            head = text.split(".", 1)[0]
            try:
                return datetime.strptime(head, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                return None
    return None


def _coerce_float(value: Any) -> float | None:
    """Return a finite float, or None for missing values."""
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if text == "" or text.lower() in {"nan", "none", "null", "nat"}:
            return None
        value = text
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def _age_days(when: datetime, birth: datetime) -> float:
    return max(0.0, (when - birth).total_seconds() / _DAY_SECONDS)


def _trajectory_samples(
    patient_id: str,
    codes: list[str],
    ages: list[float],
    min_events: int,
    max_seq_len: int,
) -> list[dict[str, Any]]:
    """Emit one sample per next-event, keeping at most ``max_seq_len`` tokens."""
    samples: list[dict[str, Any]] = []
    for end in range(min_events, len(codes)):
        start = max(0, end - max_seq_len)
        samples.append(
            {
                "patient_id": patient_id,
                "codes": list(codes[start:end]),
                "ages": [float(age) for age in ages[start:end]],
                "next_code": codes[end],
                "time_to_next": float(max(0.0, ages[end] - ages[end - 1])),
            }
        )
    return samples


class DelphiTrajectoryMEDS(BaseTask):
    """Next-event samples from a MEDS patient for Delphi-2M.

    Birth is the earliest ``MEDS_BIRTH``, ``SNOMED/3950001`` or
    ``SNOMED/184099003`` event (MIMIC-IV-MEDS and EHRSHOT-MEDS). Without a
    birth event the earliest timestamp is age 0, unless ``require_birth``
    is set. Static rows (``timestamp is None``) and the birth event itself
    are dropped. Ages are days since that anchor.

    Args:
        min_events: Shortest history that yields a sample. Default 1.
        max_seq_len: Tokens kept from the end of each prefix. Default 128.
        require_birth: Drop patients with no birth event. Default False.
        code_prefix: Keep only codes that start with this string.
        exclude_prefixes: Drop codes that start with any of these strings.
        event_type: Event table to read. Default ``"meds"``.
        birth_codes: Codes treated as the birth anchor.
        code_mapping: Optional vocabulary map forwarded to
            :class:`~pyhealth.tasks.BaseTask`.

    Examples:
        >>> from datetime import datetime
        >>> import polars as pl
        >>> from pyhealth.data import Patient
        >>> from pyhealth.tasks import DelphiTrajectoryMEDS
        >>> t = datetime(2020, 1, 1)
        >>> frame = pl.DataFrame({
        ...     "patient_id": ["p", "p", "p"],
        ...     "event_type": ["meds", "meds", "meds"],
        ...     "timestamp": [t, t, t],
        ...     "meds/code": ["MEDS_BIRTH", "A", "B"],
        ... })
        >>> DelphiTrajectoryMEDS()(Patient("p", frame))[0]["next_code"]
        'B'
    """

    task_name: str = "DelphiTrajectoryMEDS"
    input_schema: ClassVar[dict[str, str]] = {"codes": "sequence", "ages": "tensor"}
    output_schema: ClassVar[dict[str, str]] = {
        "next_code": "multiclass",
        "time_to_next": "tensor",
    }

    def __init__(
        self,
        min_events: int = 1,
        max_seq_len: int = 128,
        require_birth: bool = False,
        code_prefix: str | None = None,
        exclude_prefixes: tuple[str, ...] | list[str] = (),
        event_type: str = "meds",
        birth_codes: tuple[str, ...] = MEDS_BIRTH_CODES,
        code_mapping: dict[str, tuple[str, str]] | None = None,
    ) -> None:
        if min_events < 1:
            raise ValueError(f"min_events must be >= 1, got {min_events}.")
        if max_seq_len < 1:
            raise ValueError(f"max_seq_len must be >= 1, got {max_seq_len}.")
        super().__init__(code_mapping=code_mapping)
        self.min_events = min_events
        self.max_seq_len = max_seq_len
        self.require_birth = require_birth
        self.code_prefix = code_prefix
        self.exclude_prefixes = tuple(exclude_prefixes)
        self.event_type = event_type
        self.birth_codes = frozenset(birth_codes)

    def pre_filter(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Keep this task's event table in the dataset scan."""
        return df.filter(pl.col("event_type") == self.event_type)

    def _kept_events(
        self, patient: Any
    ) -> tuple[datetime | None, list[tuple[datetime, str]]]:
        raw: list[tuple[datetime | None, str]] = []
        for event in patient.get_events(event_type=self.event_type):
            code = event.attr_dict.get("code")
            if code is None or str(code).strip() == "":
                continue
            raw.append((_coerce_timestamp(event.timestamp), str(code).strip()))
        births = [
            when for when, code in raw if code in self.birth_codes and when is not None
        ]
        if births:
            birth = min(births)
        elif self.require_birth:
            return None, []
        else:
            stamped = [when for when, _ in raw if when is not None]
            if not stamped:
                return None, []
            birth = min(stamped)
        kept: list[tuple[datetime, str]] = []
        for when, code in raw:
            if when is None or code in self.birth_codes:
                continue
            if self.code_prefix is not None and not code.startswith(self.code_prefix):
                continue
            if any(code.startswith(prefix) for prefix in self.exclude_prefixes):
                continue
            kept.append((when, code))
        kept.sort(key=lambda row: (row[0], row[1]))
        return birth, kept

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        birth, kept = self._kept_events(patient)
        if birth is None or len(kept) <= self.min_events:
            return []
        codes = [code for _, code in kept]
        ages = [_age_days(when, birth) for when, _ in kept]
        return _trajectory_samples(
            patient.patient_id, codes, ages, self.min_events, self.max_seq_len
        )


class DelphiTrajectoryMIMIC4(BaseTask):
    """Next-event ICD samples from a MIMIC-IV patient for Delphi-2M.

    Works with :class:`~pyhealth.datasets.MIMIC4EHRDataset` loaded with
    ``tables=["diagnoses_icd"]``. Birth is ``anchor_age * 365.25`` days
    before 1 January of ``anchor_year``. Diagnoses in one admission share
    that admission's timestamp and are ordered by ``seq_num``, so
    ``mask_ties=True`` hides co-occurring codes from each other. An
    in-hospital ``deathtime``, or else ``dod``, adds a terminal ``DEATH``
    token. A death time earlier than the last diagnosis, whether
    ``deathtime`` or ``dod``, is moved forward to that diagnosis.
    ``icd_code`` is used as the token and ``icd_version`` is not, so an
    ICD-9 code and an ICD-10 code for the same condition stay different
    tokens.

    Args:
        min_events: Shortest history that yields a sample. Default 1.
        max_seq_len: Tokens kept from the end of each prefix. Default 128.
        code_mapping: Optional vocabulary map forwarded to
            :class:`~pyhealth.tasks.BaseTask`.

    Examples:
        >>> from pyhealth.tasks import DelphiTrajectoryMIMIC4
        >>> DelphiTrajectoryMIMIC4(min_events=1).task_name
        'DelphiTrajectoryMIMIC4'
    """

    task_name: str = "DelphiTrajectoryMIMIC4"
    input_schema: ClassVar[dict[str, str]] = {"codes": "sequence", "ages": "tensor"}
    output_schema: ClassVar[dict[str, str]] = {
        "next_code": "multiclass",
        "time_to_next": "tensor",
    }

    def __init__(
        self,
        min_events: int = 1,
        max_seq_len: int = 128,
        code_mapping: dict[str, tuple[str, str]] | None = None,
    ) -> None:
        if min_events < 1:
            raise ValueError(f"min_events must be >= 1, got {min_events}.")
        if max_seq_len < 1:
            raise ValueError(f"max_seq_len must be >= 1, got {max_seq_len}.")
        super().__init__(code_mapping=code_mapping)
        self.min_events = min_events
        self.max_seq_len = max_seq_len

    def pre_filter(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Keep the patients, admissions and diagnoses tables."""
        return df.filter(
            pl.col("event_type").is_in(["patients", "admissions", "diagnoses_icd"])
        )

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        people = patient.get_events(event_type="patients")
        if not people:
            return []
        person = people[0]
        year = _coerce_float(person.attr_dict.get("anchor_year"))
        age_years = _coerce_float(person.attr_dict.get("anchor_age"))
        if year is None or age_years is None:
            return []
        birth = datetime(int(year), 1, 1) - timedelta(days=age_years * 365.25)

        admit_at: dict[str, datetime] = {}
        death_times: list[datetime] = []
        for admission in patient.get_events(event_type="admissions"):
            hadm = admission.attr_dict.get("hadm_id")
            when = _coerce_timestamp(admission.timestamp)
            if hadm is not None and when is not None:
                admit_at[str(hadm)] = when
            death = _coerce_timestamp(admission.attr_dict.get("deathtime"))
            if death is not None:
                death_times.append(death)

        records: list[tuple[datetime, int, str]] = []
        for event in patient.get_events(event_type="diagnoses_icd"):
            code = event.attr_dict.get("icd_code")
            if code is None or str(code).strip() == "":
                continue
            hadm = event.attr_dict.get("hadm_id")
            when = admit_at.get(str(hadm)) if hadm is not None else None
            if when is None:
                when = _coerce_timestamp(event.timestamp)
            if when is None:
                continue
            seq = _coerce_float(event.attr_dict.get("seq_num"))
            records.append((when, int(seq) if seq is not None else 0, str(code).strip()))

        if death_times:
            death = max(death_times)
        else:
            death = _coerce_timestamp(person.attr_dict.get("dod"))
        if death is not None and records:
            last = max(when for when, _, _ in records)
            death = max(death, last)
            records.append((death, 10**9, "DEATH"))
        if len(records) <= self.min_events:
            return []
        records.sort(key=lambda row: (row[0], row[1], row[2]))
        codes = [code for _, _, code in records]
        ages = [_age_days(when, birth) for when, _, _ in records]
        return _trajectory_samples(
            patient.patient_id, codes, ages, self.min_events, self.max_seq_len
        )


class DelphiTrajectoryUKB(DelphiTrajectoryMEDS):
    """Next-event samples from pre-extracted UK Biobank events.

    Subclasses :class:`DelphiTrajectoryMEDS` because a pre-extracted UK
    Biobank table has the same shape as MEDS. Build one event per code with
    ``event_type="ukb"``, a timestamp, and a ``code`` attribute (stored as
    ``ukb/code``). This class only sets that event type and adds
    ``UKB_BIRTH`` to the MEDS and SNOMED birth codes. No dataset loader
    produces these rows. UK Biobank access is credentialed and is not
    downloaded here.

    Examples:
        >>> from pyhealth.tasks import DelphiTrajectoryUKB
        >>> DelphiTrajectoryUKB().task_name
        'DelphiTrajectoryUKB'
    """

    task_name: str = "DelphiTrajectoryUKB"

    def __init__(
        self,
        min_events: int = 1,
        max_seq_len: int = 128,
        require_birth: bool = False,
        code_prefix: str | None = None,
        exclude_prefixes: tuple[str, ...] | list[str] = (),
        code_mapping: dict[str, tuple[str, str]] | None = None,
    ) -> None:
        super().__init__(
            min_events=min_events,
            max_seq_len=max_seq_len,
            require_birth=require_birth,
            code_prefix=code_prefix,
            exclude_prefixes=exclude_prefixes,
            event_type="ukb",
            birth_codes=UKB_BIRTH_CODES,
            code_mapping=code_mapping,
        )
