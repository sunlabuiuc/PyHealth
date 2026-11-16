"""Emergency department disposition from MIMIC-IV-ED "pseudo-notes".

The tasks of MEME, the Multiple Embedding Model for EHR (Lee et al.,
"Clinical decision support using pseudo-notes from multiple streams of EHR
data", npj Digital Medicine 8, 394, 2025,
https://doi.org/10.1038/s41746-025-01777-x; preprint "Emergency Department
Decision Support using Clinical Pseudo-notes", arXiv:2402.00160). Each ED stay's structured data is written as one short
templated text per domain: arrival, triage, medication reconciliation, vital
signs, medications dispensed (Pyxis) and, optionally, ED diagnosis codes.
:class:`~pyhealth.models.MEME` embeds each text with a frozen encoder.

The note templates are adapted from the reference preprocessing
(``preprocessing/mimiciv_ed_convert.py`` in https://github.com/Simonlee711/MEME),
which is distributed under the MIT License:

    Copyright (c) 2023 Simon Lee

    Permission is hereby granted, free of charge, to any person obtaining a
    copy of this software and associated documentation files (the
    "Software"), to deal in the Software without restriction, including
    without limitation the rights to use, copy, modify, merge, publish,
    distribute, sublicense, and/or sell copies of the Software, and to permit
    persons to whom the Software is furnished to do so, subject to the
    following conditions:

    The above copyright notice and this permission notice shall be included
    in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
    OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
    MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
    NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
    DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
    OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
    USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from datetime import datetime, timedelta
from typing import Any, ClassVar

import polars as pl

from .base_task import BaseTask
from .ehr_text_serialization import mimic4_age_at

logger = logging.getLogger(__name__)

#: The pseudo-note domains, in the paper's order. ``codes`` is optional.
ED_NOTE_DOMAINS = ("arrival", "triage", "medrecon", "vitals", "pyxis", "codes")

#: Outcome names of :class:`EDOutcomesMIMIC4`.
ED_OUTCOMES = ("discharge_not_home", "mortality", "icu")

_VITALS = (
    ("temperature", "temperature"),
    ("heartrate", "pulse"),
    ("resprate", "respirations"),
    ("o2sat", "o2 saturation"),
    ("sbp", "systolic blood pressure"),
    ("dbp", "diastolic blood pressure"),
    ("pain", "pain"),
)

#: The ED table behind each optional note domain.
_DOMAIN_TABLES = {
    "triage": "triage",
    "medrecon": "medrecon",
    "vitals": "vitalsign",
    "pyxis": "pyxis",
    "codes": "diagnosis",
}


def _event_types(df: pl.LazyFrame) -> set[str]:
    """Returns the event types present in a global event dataframe."""
    return set(
        df.select(pl.col("event_type").unique()).collect()["event_type"].to_list()
    )


def _value(value: Any) -> str:
    """A recorded value as lower-case text, or ``"not recorded"``."""
    if value is None or str(value).strip().lower() in ("", "nan", "none"):
        return "not recorded"
    return str(value).strip().lower()


def _time(value: Any) -> str:
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    return str(value)


def arrival_note(
    gender: str | None,
    race: str | None,
    arrival_transport: str | None,
    intime: Any,
    age: int | None = None,
) -> str:
    """Writes the arrival pseudo-note of an ED stay.

    Args:
        gender: ``"F"`` or ``"M"``.
        race: Race as recorded in ``edstays``.
        arrival_transport: E.g. ``"AMBULANCE"`` or ``"WALK IN"``.
        intime: Arrival time.
        age: Age at arrival, if known.

    Returns:
        The note.

    Examples:
        >>> arrival_note("F", "WHITE", "AMBULANCE", "2180-05-06 19:17:00", 52)
        'A 52 year old white female, arrived via ambulance at 2180-05-06 19:17:00.'
    """
    sex = {"F": "female", "M": "male"}.get(str(gender).upper(), "patient")
    race = _value(race)
    who = sex if race == "not recorded" else f"{race} {sex}"
    head = f"A {age} year old {who}," if age is not None else f"A {who},"
    return f"{head} arrived via {_value(arrival_transport)} at {_time(intime)}."


def triage_note(triage: Any | None) -> str:
    """Writes the triage pseudo-note from a ``triage`` event.

    Args:
        triage: An object with the ``triage`` columns as attributes, or
            ``None`` when the stay has no triage record.

    Returns:
        The note.

    Examples:
        >>> from types import SimpleNamespace
        >>> t = SimpleNamespace(temperature="98.4", heartrate="70", resprate="16",
        ...                     o2sat="97", sbp="106", dbp="63", pain="0",
        ...                     acuity="3", chiefcomplaint="Abd pain")
        >>> triage_note(t)[:46]
        'At triage: temperature was 98.4, pulse was 70,'
    """
    if triage is None:
        return "Triage measurements were not taken."
    parts = ["At triage:"]
    parts += [f"{name} was {_value(getattr(triage, col, None))}," for col, name in _VITALS]
    parts.append(f"chief complaint was {_value(getattr(triage, 'chiefcomplaint', None))}.")
    parts.append(f"Acuity score was {_value(getattr(triage, 'acuity', None))}.")
    return " ".join(parts)


def vitals_note(vitals: Iterable[Any]) -> str:
    """Writes the vital-signs pseudo-note from ``vitalsign`` events.

    Args:
        vitals: Events with a ``timestamp`` and the ``vitalsign`` columns, in
            time order.

    Returns:
        The note.

    Examples:
        >>> vitals_note([])
        'The patient had no vitals recorded.'
    """
    vitals = list(vitals)
    if not vitals:
        return "The patient had no vitals recorded."
    parts = ["The patient had the following vitals:"]
    for v in vitals:
        values = [f"{name} was {_value(getattr(v, col, None))}" for col, name in _VITALS]
        parts.append(f"At {_time(v.timestamp)}, " + ", ".join(values) + ".")
    return " ".join(parts)


def pyxis_note(dispensed: Iterable[Any]) -> str:
    """Writes the pseudo-note of medications dispensed in the ED (Pyxis).

    Medications dispensed at the same time are listed together.

    Args:
        dispensed: ``pyxis`` events with a ``timestamp`` and a ``name``, in
            time order.

    Returns:
        The note.

    Examples:
        >>> from types import SimpleNamespace
        >>> pyxis_note([SimpleNamespace(timestamp="2180-05-06 20:00:00",
        ...                             name="Morphine")])
        'The patient received the following medications: At 2180-05-06 20:00:00, morphine were administered.'
    """
    groups: dict[str, list[str]] = {}
    for d in dispensed:
        groups.setdefault(_time(d.timestamp), []).append(_value(d.name))
    if not groups:
        return "The patient did not receive any medications."
    parts = ["The patient received the following medications:"]
    for when, names in groups.items():
        parts.append(f"At {when}, {', '.join(names)} were administered.")
    return " ".join(parts)


def medrecon_note(medications: Iterable[Any]) -> str:
    """Writes the medication-reconciliation pseudo-note (home medications).

    Args:
        medications: ``medrecon`` events with ``name`` and
            ``etcdescription``.

    Returns:
        The note.

    Examples:
        >>> medrecon_note([])
        'The patient was not on any previous medications.'
    """
    medications = list(medications)
    if not medications:
        return "The patient was not on any previous medications."
    parts = ["The patient was previously taking the following medications:"]
    for m in medications:
        parts.append(f"{_value(m.name)}, {_value(getattr(m, 'etcdescription', None))}.")
    return " ".join(parts)


def codes_note(diagnoses: Iterable[Any]) -> str:
    """Writes the ED diagnosis-code pseudo-note.

    Args:
        diagnoses: ``diagnosis`` events with ``icd_version``, ``icd_code``
            and ``icd_title``, in ``seq_num`` order.

    Returns:
        The note.

    Examples:
        >>> from types import SimpleNamespace
        >>> codes_note([SimpleNamespace(icd_version="9", icd_code="431",
        ...                             icd_title="INTRACEREBRAL HEMORRHAGE")])
        'The patient received the following diagnostic codes: ICD-9 code: [431], intracerebral hemorrhage.'
    """
    diagnoses = list(diagnoses)
    if not diagnoses:
        return "The patient had no diagnostic codes recorded."
    parts = ["The patient received the following diagnostic codes:"]
    for d in diagnoses:
        code = str(d.icd_code).strip().replace(".", "").lower()
        parts.append(
            f"ICD-{_value(d.icd_version)} code: [{code}], {_value(d.icd_title)}."
        )
    return " ".join(parts)


def _seq(event: Any) -> int:
    try:
        return int(event.seq_num)
    except (TypeError, ValueError):
        return 0


class EDDispositionMIMIC4(BaseTask):
    """Predicts whether an ED visit ends in admission, from pseudo-notes.

    The ED disposition task of MEME (Lee et al., npj Digital Medicine 2025).
    One sample per ED stay in ``edstays`` whose disposition is ``HOME``
    (label 0), ``ADMITTED`` or ``EXPIRED`` (label 1), as in the reference
    code; transfers, elopements and other dispositions are left out.

    Each sample has one text field per domain (see :data:`ED_NOTE_DOMAINS`):

    - ``arrival``: age, race, sex, arrival transport and time.
    - ``triage``: triage vitals, pain, chief complaint and acuity.
    - ``medrecon``: medications the patient was taking before the visit.
    - ``vitals``: every vital-sign measurement of the stay, with its time.
    - ``pyxis``: medications dispensed in the ED, grouped by time.
    - ``codes`` (only with ``include_codes=True``): the ED diagnosis codes.

    **Leakage.** ED diagnoses in ``diagnosis`` are coded at the end of the
    stay, so they partly encode the decision being predicted. The paper and
    its code use them; they are left out by default here. Set
    ``include_codes=True`` to reproduce the paper's input. Vital signs and
    Pyxis records also span the whole stay, so a longer stay (more common
    for admitted patients) writes longer notes; ``observation_hours`` keeps
    only events in the first hours after arrival. The reference code also
    appended insurance, language and marital status from the hospital
    ``admissions`` table, which only exists for admitted patients; those are
    never used here.

    Uses ``edstays`` plus whichever of ``triage``, ``vitalsign``,
    ``medrecon``, ``pyxis`` and ``diagnosis`` are loaded, and ``patients``
    for the age when an EHR root is given (see
    :class:`~pyhealth.datasets.MIMIC4Dataset`).

    Args:
        include_codes: Add the ``codes`` note (ED diagnoses; see above).
        observation_hours: Keep only vitals, Pyxis and medication
            reconciliation events within this many hours of arrival. ``None``
            (default, as in the paper) keeps the whole stay.

    Examples:
        >>> from pyhealth.datasets import MIMIC4Dataset
        >>> from pyhealth.tasks import EDDispositionMIMIC4
        >>> dataset = MIMIC4Dataset(  # doctest: +SKIP
        ...     ehr_root="/path/to/mimic-iv/2.2",
        ...     ed_root="/path/to/mimic-iv-ed/2.2",
        ...     ed_tables=["triage", "vitalsign", "medrecon", "pyxis"],
        ... )
        >>> samples = dataset.set_task(EDDispositionMIMIC4())  # doctest: +SKIP
        >>> sorted(EDDispositionMIMIC4().input_schema)
        ['arrival', 'medrecon', 'pyxis', 'triage', 'vitals']
    """

    task_name: str = "EDDispositionMIMIC4"
    output_schema: ClassVar[dict[str, str]] = {"disposition": "binary"}
    positive_dispositions = ("ADMITTED", "EXPIRED")
    negative_dispositions = ("HOME",)

    def __init__(
        self, include_codes: bool = False, observation_hours: float | None = None
    ):
        self.include_codes = include_codes
        self.observation_hours = observation_hours
        domains = [d for d in ED_NOTE_DOMAINS if include_codes or d != "codes"]
        self.input_schema = {d: "text" for d in domains}

    def pre_filter(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Warns about ED tables the notes need but the dataset did not load.

        A missing table would make the matching note the "not recorded" text
        (e.g. "The patient had no vitals recorded.") for every stay, which
        reads like a clinical fact. One warning names every such table.

        Args:
            df: The dataset's global event dataframe.

        Returns:
            ``df``, unchanged.

        Examples:
            >>> import polars as pl
            >>> df = pl.LazyFrame({"event_type": ["edstays", "triage"]})
            >>> EDDispositionMIMIC4().pre_filter(df) is df  # logs a warning
            True
        """
        self._warn_missing_tables(_event_types(df))
        return df

    def _warn_missing_tables(self, present: set[str]) -> None:
        missing = [
            (domain, table)
            for domain, table in _DOMAIN_TABLES.items()
            if domain in self.input_schema and table not in present
        ]
        if missing:
            logger.warning(
                "%s found no events for ED table(s) %s, so the %s note(s) will "
                "be the 'not recorded' text for every stay: load them with "
                "ed_tables=[...] in MIMIC4Dataset",
                self.task_name,
                ", ".join(repr(t) for _, t in missing),
                ", ".join(repr(d) for d, _ in missing),
            )

    def _age(self, patient: Any, intime: datetime | None) -> int | None:
        people = patient.get_events(event_type="patients")
        return mimic4_age_at(people[0], intime) if people else None

    def _stay_events(self, patient: Any, table: str, stay_id: str, intime: Any):
        events = [
            e
            for e in patient.get_events(event_type=table)
            if str(getattr(e, "stay_id", None)) == stay_id
        ]
        if self.observation_hours is not None and intime is not None:
            end = intime + timedelta(hours=self.observation_hours)
            events = [e for e in events if e.timestamp is None or e.timestamp <= end]
        return events

    def notes(self, patient: Any, stay: Any) -> dict[str, str]:
        """Writes the pseudo-notes of one ED stay.

        Args:
            patient: The :class:`~pyhealth.data.Patient`.
            stay: Its ``edstays`` event.

        Returns:
            A ``{domain: text}`` dict with the keys of ``input_schema``.

        Examples:
            >>> task = EDDispositionMIMIC4()
            >>> notes = task.notes(patient, stay)  # doctest: +SKIP
        """
        stay_id = str(stay.stay_id)
        intime = stay.timestamp

        def events(table):
            return self._stay_events(patient, table, stay_id, intime)

        triage = events("triage")
        notes = {
            "arrival": arrival_note(
                stay.gender, stay.race, stay.arrival_transport, intime,
                self._age(patient, intime),
            ),
            "triage": triage_note(triage[0] if triage else None),
            "medrecon": medrecon_note(events("medrecon")),
            "vitals": vitals_note(events("vitalsign")),
            "pyxis": pyxis_note(events("pyxis")),
        }
        if self.include_codes:
            # coded at departure, so never cut by observation_hours
            diagnoses = [
                e
                for e in patient.get_events(event_type="diagnosis")
                if str(getattr(e, "stay_id", None)) == stay_id
            ]
            notes["codes"] = codes_note(sorted(diagnoses, key=_seq))
        return notes

    def _label(self, patient: Any, stay: Any) -> Any:
        disposition = str(stay.disposition).upper()
        if disposition in self.positive_dispositions:
            return 1
        if disposition in self.negative_dispositions:
            return 0
        return None

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        samples = []
        for stay in patient.get_events(event_type="edstays"):
            label = self._label(patient, stay)
            if label is None:
                continue
            sample = {
                "patient_id": patient.patient_id,
                "visit_id": str(stay.stay_id),
                **self.notes(patient, stay),
            }
            sample[next(iter(self.output_schema))] = label
            samples.append(sample)
        return samples


class EDOutcomesMIMIC4(EDDispositionMIMIC4):
    """Hospital outcomes of admitted ED patients, from ED pseudo-notes.

    The multitask setting of MEME (Lee et al., npj Digital Medicine 2025):
    for ED stays that led to a hospital admission, predict three outcomes
    together as a multilabel target ``outcomes``:

    - ``discharge_not_home``: the hospital discharge location is not
      ``HOME`` (a missing location counts as home, as in the reference
      code);
    - ``mortality``: died in the hospital (``hospital_expire_flag``);
    - ``icu``: an ICU stay during the admission.

    The inputs are the same pseudo-notes as :class:`EDDispositionMIMIC4`.
    Needs an EHR root with ``admissions`` and ``icustays`` (both loaded by
    default by :class:`~pyhealth.datasets.MIMIC4Dataset`). A stay is used
    when its ``hadm_id`` matches an admission.

    The label vocabulary is built from the data, so an outcome that never
    occurs (e.g. no deaths in a small demo) has no output; read the order
    from ``samples.output_processors["outcomes"].label_vocab``.

    Args:
        include_codes: Add the ``codes`` note (ED diagnoses).
        observation_hours: See :class:`EDDispositionMIMIC4`.

    Examples:
        >>> from pyhealth.tasks import EDOutcomesMIMIC4
        >>> EDOutcomesMIMIC4().output_schema
        {'outcomes': 'multilabel'}
    """

    task_name: str = "EDOutcomesMIMIC4"
    output_schema: ClassVar[dict[str, str]] = {"outcomes": "multilabel"}

    def pre_filter(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Warns when the dataset has no ``admissions``: no stay gets a label.

        Also warns about missing ED tables, as
        :meth:`EDDispositionMIMIC4.pre_filter` does.

        Args:
            df: The dataset's global event dataframe.

        Returns:
            ``df``, unchanged.

        Examples:
            >>> import polars as pl
            >>> df = pl.LazyFrame({"event_type": ["edstays"]})
            >>> EDOutcomesMIMIC4().pre_filter(df) is df  # logs a warning
            True
        """
        present = _event_types(df)
        if "admissions" not in present:
            logger.warning(
                "EDOutcomesMIMIC4 found no 'admissions' events, so it yields no "
                "samples: load MIMIC4Dataset with an ehr_root as well as ed_root"
            )
        self._warn_missing_tables(present)
        return df

    def _label(self, patient: Any, stay: Any) -> Any:
        hadm_id = getattr(stay, "hadm_id", None)
        if hadm_id in (None, "", "nan"):
            return None
        hadm_id = str(hadm_id)
        admissions = [
            a
            for a in patient.get_events(event_type="admissions")
            if str(a.hadm_id) == hadm_id
        ]
        if not admissions:
            return None
        admission = admissions[0]
        outcomes = []
        location = getattr(admission, "discharge_location", None)
        if location not in (None, "", "nan") and str(location).upper() != "HOME":
            outcomes.append("discharge_not_home")
        if str(getattr(admission, "hospital_expire_flag", "0")).strip() in ("1", "1.0"):
            outcomes.append("mortality")
        if any(
            str(getattr(i, "hadm_id", None)) == hadm_id
            for i in patient.get_events(event_type="icustays")
        ):
            outcomes.append("icu")
        return outcomes
