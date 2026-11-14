"""Turn a task's coded EHR samples into Markdown text for an LLM encoder.

This implements the serialization step of Hegselmann et al., "Large language
models are powerful electronic health record encoders" (npj Digital Medicine,
2025; https://arxiv.org/abs/2502.17403). There, each patient's record before
the prediction time is written as Markdown: demographics first, then the
unique codes of each category, most recent first, with every code replaced
by its description. A frozen LLM embedding model then encodes the text (see
:class:`~pyhealth.models.LLMEHREncoder`).

The reference code is https://github.com/stefanhgm/ehrshot-benchmark
(Apache-2.0); this is a reimplementation on PyHealth's task API.

:class:`EHRTextSerializationTask` wraps any existing task. It serializes the
codes the wrapped task already put in each sample, so the text holds exactly
the information the task allows before its prediction time: the wrapper
adds no events and cannot leak the label.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .base_task import BaseTask

logger = logging.getLogger(__name__)

# Fields of common PyHealth tasks and the vocabularies their codes come from,
# tried in order. MIMIC-IV mixes ICD-9 and ICD-10 without a version prefix in
# the samples; the wrapper takes each code's version from the patient's
# ``icd_version`` (see ICD_EVENT_TYPES), so only that version's vocabulary is
# tried. Without a version, ICD-10 is tried first, except for codes shaped
# like ICD-9 E/V codes (see CodeDescriber).
DEFAULT_SECTIONS: dict[str, tuple[str, tuple[str, ...]]] = {
    "conditions": ("Conditions", ("ICD10CM", "ICD9CM")),
    "procedures": ("Procedures", ("ICD10PROC", "ICD9PROC")),
    "drugs": ("Medications", ("NDC", "ATC")),
    "drugs_hist": ("Medications", ("NDC", "ATC")),
    "labs": ("Lab tests", ()),
}

# Input processors whose fields hold codes, the only fields serialized. Other
# inputs (time series, tensors, numbers, images) are not codes and are skipped.
# StageNet inputs are ``(times, codes)`` pairs; their codes are serialized.
# Not supported: "cehr" (concept records, not plain codes) and the numeric
# "stagenet_tensor", "timeseries", "tensor" and similar inputs.
CODE_PROCESSORS: frozenset[str] = frozenset({
    "sequence", "sequence_v2", "jagged", "nested_sequence", "deep_nested_sequence",
    "multi_hot", "nested_multihot", "stagenet",
})
TIMED_CODE_PROCESSORS: frozenset[str] = frozenset({"stagenet"})

# Event types whose ``icd_code``/``icd_version`` attributes give the ICD
# version of a sample's unprefixed codes (MIMIC-IV).
ICD_EVENT_TYPES: tuple[str, ...] = ("diagnoses_icd", "procedures_icd")

# For tasks whose single field holds vocabulary-prefixed codes such as
# "ICD10CM/E11.9" (EHRSHOT), the section a code goes to, by prefix. Other
# prefixes go to "Conditions", as in the paper's serializer.
PREFIX_SECTIONS: dict[str, str] = {
    "RxNorm": "Medications",
    "RxNorm Extension": "Medications",
    "CVX": "Medications",
    "ATC": "Medications",
    "NDC": "Medications",
    "CPT4": "Procedures",
    "HCPCS": "Procedures",
    "ICD9Proc": "Procedures",
    "ICD9PROC": "Procedures",
    "ICD10PCS": "Procedures",
    "ICD10PROC": "Procedures",
    "LOINC": "Lab tests",
    "Visit": "Visits",
}

# Vocabulary prefixes as written in OMOP/EHRSHOT, mapped to medcode names.
_MEDCODE_NAMES = {
    "ICD9CM": "ICD9CM",
    "ICD10CM": "ICD10CM",
    "ICD9Proc": "ICD9PROC",
    "ICD9PROC": "ICD9PROC",
    "ICD10PCS": "ICD10PROC",
    "ICD10PROC": "ICD10PROC",
    "ATC": "ATC",
    "NDC": "NDC",
    "RxNorm": "RxNorm",
}


class _Keyed:
    """An object with a stable ``str``, for the task's cache key.

    ``set_task`` keys its cache on ``json.dumps(vars(task), default=str)``;
    a plain object or function would print with its memory address and miss
    the cache on every run.
    """

    def __init__(self, obj: Any, key: str) -> None:
        self.obj = obj
        self.key = key

    def __str__(self) -> str:
        return self.key


def _name(obj: Any) -> str:
    """A stable name for a function, distinct for distinct lambdas.

    Every lambda in a module shares the ``__qualname__`` ``<lambda>`` (and
    nested functions share ``f.<locals>.g``), so these also get their line
    number and a hash of their code. Values captured from an enclosing scope
    are not part of the name.
    """
    if obj is None:
        return "None"
    name = f"{getattr(obj, '__module__', '')}.{getattr(obj, '__qualname__', obj)}"
    code = getattr(obj, "__code__", None)
    if code is not None and ("<lambda>" in name or "<locals>" in name):
        name += f"@{code.co_firstlineno}#{_code_hash(code)}"
    return name


def _code_hash(code: Any) -> str:
    """A hash of a code object's bytecode, constants and names."""
    digest = hashlib.sha256()
    pending = [code]
    while pending:
        current = pending.pop()
        digest.update(current.co_code)
        digest.update(repr(current.co_names).encode())
        for const in current.co_consts:
            if hasattr(const, "co_code"):  # a nested function's code
                pending.append(const)
            else:
                digest.update(repr(const).encode())
    return digest.hexdigest()[:12]


def _processor_name(spec: Any) -> str | None:
    """The registered name of an input schema entry's processor."""
    if isinstance(spec, tuple):
        spec = spec[0]
    if isinstance(spec, str):
        return spec
    from ..processors import PROCESSOR_REGISTRY

    cls = spec if isinstance(spec, type) else type(spec)
    for name, registered in PROCESSOR_REGISTRY.items():
        if registered is cls:
            return name
    return None


class CodeDescriber:
    """Looks up a code's description in PyHealth's medical code vocabularies.

    A code with a vocabulary prefix (``"ICD10CM/E11.9"``) is looked up in
    that vocabulary. A code with an ICD version prefix (``"9_4019"``, as
    some MIMIC-IV tasks write them) is tried only in the ``vocabularies``
    of that ICD version. Any other code is tried in ``vocabularies`` in
    order.
    A code without a version prefix shaped like an ICD-9 E or V code
    (``"E8490"``, ``"V1582"``) is tried in ICD-9 vocabularies first: over 700
    such strings are also ICD-10-CM codes with another meaning (ICD-9
    ``E000`` "External cause status", ICD-10 ``E00.0`` "Congenital
    iodine-deficiency syndrome"). Prefix the version to be exact.
    Unknown codes return ``None``. Vocabularies are loaded on first use
    (:class:`~pyhealth.medcode.InnerMap`, downloaded once and cached) and
    lookups are memoized. A vocabulary that fails to load raises
    ``RuntimeError`` unless ``raw_fallback=True``: otherwise ``set_task``
    would cache raw-code text under the same key as described text.

    Args:
        lookup: Optional ``lookup(code, vocabularies) -> str | None`` that
            replaces the medcode lookup, e.g. for a custom vocabulary or to
            work offline.
        name: The lookup's name in the task's cache key. Defaults to the
            function's module, name, line and code hash; pass one when two
            lookups share their code but not their data, e.g. closures
            over different tables.
        raw_fallback: If a vocabulary cannot be loaded, warn and keep its
            codes raw instead of raising. This is part of the cache key, so
            such text is never reused by a run without it.

    Examples:
        >>> describe = CodeDescriber(
        ...     lookup=lambda code, vocabularies: {"4019": "Hypertension"}.get(code)
        ... )
        >>> describe("4019", ("ICD9CM",))
        'Hypertension'
        >>> describe("ZZZ", ("ICD9CM",)) is None
        True
        >>> describe("9_4019", ("ICD10CM", "ICD9CM"))  # ICD-9, version-prefixed
        'Hypertension'
        >>> CodeDescriber(lookup=describe, name="my-table-v2")
        CodeDescriber(lookup=my-table-v2)
        >>> CodeDescriber(raw_fallback=True)
        CodeDescriber(raw_fallback=True)
    """

    def __init__(
        self,
        lookup: Callable[[str, Sequence[str]], str | None] | None = None,
        name: str | None = None,
        raw_fallback: bool = False,
    ) -> None:
        self._lookup = lookup
        self._name = name
        self.raw_fallback = raw_fallback
        self._maps: dict[str, Any] = {}
        self._memo: dict[tuple[str, tuple[str, ...]], str | None] = {}

    def __repr__(self) -> str:
        args = []
        if self._lookup is not None:
            args.append(f"lookup={self._name or _name(self._lookup)}")
        if self.raw_fallback:
            args.append("raw_fallback=True")
        return f"CodeDescriber({', '.join(args)})"

    def __call__(self, code: str, vocabularies: Sequence[str] = ()) -> str | None:
        key = (code, tuple(vocabularies))
        if key not in self._memo:
            self._memo[key] = self._describe(code, tuple(vocabularies))
        return self._memo[key]

    def _describe(self, code: str, vocabularies: tuple[str, ...]) -> str | None:
        prefix, sep, bare = code.partition("/")
        if sep and prefix in _MEDCODE_NAMES:
            code, vocabularies = bare, (_MEDCODE_NAMES[prefix],)
        version, sep, bare = code.partition("_")
        if sep and version in ("9", "10") and bare:
            code = bare
            vocabularies = tuple(
                v for v in vocabularies if re.fullmatch(rf"ICD{version}\D.*", v)
            )
        elif re.fullmatch(r"[EV]\d{2,4}", code):
            # an ICD-9 E/V shape: ICD-9 first (a stable sort keeps the rest)
            vocabularies = tuple(
                sorted(vocabularies, key=lambda v: not v.startswith("ICD9"))
            )
        if self._lookup is not None:
            return self._lookup(code, vocabularies)
        for vocabulary in vocabularies:
            inner_map = self._load(vocabulary)
            if inner_map is None:
                continue
            try:
                if code in inner_map:
                    return str(inner_map.lookup(code))
            except Exception:  # noqa: BLE001 - a malformed code is just unknown
                logger.debug("cannot look up %r in %s", code, vocabulary)
        return None

    def _load(self, vocabulary: str) -> Any:
        if vocabulary not in self._maps:
            try:
                from pyhealth.medcode import InnerMap

                self._maps[vocabulary] = InnerMap.load(vocabulary)
            except Exception as e:
                if not self.raw_fallback:
                    raise RuntimeError(
                        f"vocabulary {vocabulary} could not be loaded "
                        f"({type(e).__name__}: {e}); pass "
                        "CodeDescriber(raw_fallback=True) to keep raw codes"
                    ) from e
                logger.warning("vocabulary %s unavailable (%s); raw codes are "
                               "used", vocabulary, type(e).__name__)
                self._maps[vocabulary] = None
        return self._maps[vocabulary]


def serialize_ehr(
    sections: Mapping[str, Sequence[str]],
    demographics: Mapping[str, Any] | None = None,
    max_codes_per_section: int | None = None,
    max_chars: int | None = None,
) -> str:
    """Writes a patient's record as the paper's Markdown text.

    Each section lists its entries once, in the order given (pass them most
    recent first). Empty sections are left out.

    Args:
        sections: Heading to entries (code descriptions), most recent first.
        demographics: Attribute to value, e.g. ``{"Gender": "F", "Age": 63}``.
        max_codes_per_section: Keep at most this many entries per section.
        max_chars: Cut the text to this many characters, at a line break.
            The encoder's ``max_length`` also truncates, in tokens.

    Returns:
        The Markdown text.

    Examples:
        >>> print(serialize_ehr(
        ...     {"Conditions": ["Sepsis", "Hypertension", "Sepsis"]},
        ...     demographics={"Gender": "F", "Age": 63},
        ... ))
        # Electronic Health Record
        <BLANKLINE>
        ## Patient Demographics
        - Gender: F
        - Age: 63
        <BLANKLINE>
        ## Conditions
        - Sepsis
        - Hypertension
    """
    parts = ["# Electronic Health Record"]
    shown = {k: v for k, v in (demographics or {}).items() if v not in (None, "")}
    if shown:
        lines = [f"- {k}: {v}" for k, v in shown.items()]
        parts.append("## Patient Demographics\n" + "\n".join(lines))
    for heading, entries in sections.items():
        unique = list(dict.fromkeys(e for e in entries if e))
        if max_codes_per_section is not None:
            unique = unique[:max_codes_per_section]
        if unique:
            parts.append(f"## {heading}\n" + "\n".join(f"- {e}" for e in unique))
    text = "\n\n".join(parts)
    if max_chars is not None and len(text) > max_chars:
        cut = text.rfind("\n", 0, max_chars)
        text = text[: cut if cut > 0 else max_chars]
    return text


def _flatten(value: Any) -> list[str]:
    """Codes of a sample field: a list, or a list of visits (lists)."""
    if value is None:
        return []
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        return [str(value)]
    out: list[str] = []
    for item in value:
        if isinstance(item, (list, tuple)):
            out.extend(_flatten(item))
        elif item is not None and str(item).strip():
            out.append(str(item).strip())
    return out


def _events(patient: Any, event_type: str) -> list[Any]:
    try:
        return list(patient.get_events(event_type=event_type) or [])
    except Exception:  # noqa: BLE001 - datasets without the table
        return []


def mimic4_age_at(person: Any, when: Any) -> int | None:
    """MIMIC-IV age of a patient at a given time, or None if it can't be told.

    MIMIC-IV gives ``anchor_age``, the age in the shifted ``anchor_year``, so
    the age at a time is ``anchor_age + (year of that time - anchor_year)``.

    Args:
        person: The patient's ``patients`` event, with ``anchor_age`` and
            ``anchor_year``.
        when: A datetime, e.g. an admission or ED arrival time.

    Returns:
        The age in whole years, or None when ``anchor_age``, ``anchor_year``
        or ``when`` is missing or not a number.

    Examples:
        >>> from datetime import datetime
        >>> from types import SimpleNamespace
        >>> person = SimpleNamespace(anchor_age="63", anchor_year="2150")
        >>> mimic4_age_at(person, datetime(2155, 3, 1))
        68
        >>> mimic4_age_at(person, None) is None
        True
    """
    try:
        anchor_age = int(float(person.anchor_age))
        anchor_year = int(float(person.anchor_year))
        return anchor_age + when.year - anchor_year
    except (AttributeError, TypeError, ValueError):
        return None


def _age_at_sample(person: Any, patient: Any, sample: Mapping[str, Any]) -> int | None:
    """MIMIC-IV age at the sample's admission, or None if it can't be told.

    ``anchor_age`` is the age in ``anchor_year``, so the age at an admission
    is ``anchor_age + (admission year - anchor_year)``. The admission is the
    one whose ``hadm_id`` is the sample's ``hadm_id`` or ``visit_id``.
    """
    visit = sample.get("hadm_id", sample.get("visit_id"))
    if visit is None:
        return None
    for admission in _events(patient, "admissions"):
        timestamp = getattr(admission, "timestamp", None)
        if str(getattr(admission, "hadm_id", None)) == str(visit) and timestamp:
            return mimic4_age_at(person, timestamp)
    return None


def mimic_demographics(
    patient: Any, sample: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Gender and age at the sample's admission for a MIMIC-IV patient.

    MIMIC-IV gives ``anchor_age``, the age in the shifted ``anchor_year``;
    the age at the sample's admission (its ``hadm_id`` or ``visit_id``) is
    ``anchor_age`` plus the years from ``anchor_year`` to that admission.
    Age is left out when this can't be computed: no ``anchor_year``, no
    admission id in the sample (e.g. patient-level samples), or no matching
    admission. Other datasets give gender only. Return an empty dict to
    leave demographics out.

    Args:
        patient: The patient.
        sample: The wrapped task's sample the header is for.

    Examples:
        >>> from datetime import datetime
        >>> from types import SimpleNamespace
        >>> person = SimpleNamespace(gender="F", anchor_age="63",
        ...                          anchor_year="2150")
        >>> admission = SimpleNamespace(hadm_id="7",
        ...                             timestamp=datetime(2155, 3, 1))
        >>> events = {"patients": [person], "admissions": [admission]}
        >>> patient = SimpleNamespace(get_events=lambda event_type: events[event_type])
        >>> mimic_demographics(patient, {"hadm_id": "7"})
        {'Gender': 'F', 'Age': 68}
        >>> mimic_demographics(patient)  # no admission: no age
        {'Gender': 'F'}
    """
    persons = _events(patient, "patients")
    if not persons:
        return {}
    person = persons[0]
    out = {}
    gender = getattr(person, "gender", None)
    if gender:
        out["Gender"] = gender
    age = _age_at_sample(person, patient, sample or {})
    if age is not None:
        out["Age"] = age
    return out


def icd_versions(patient: Any) -> dict[str, str]:
    """The ICD version of each of a patient's ICD codes, where known.

    Reads ``icd_code`` and ``icd_version`` from the patient's
    :data:`ICD_EVENT_TYPES` events (MIMIC-IV). A code seen with both
    versions is left out, as is every code of a dataset without versions.

    Examples:
        >>> from types import SimpleNamespace
        >>> events = {"diagnoses_icd": [
        ...     SimpleNamespace(icd_code="E8490", icd_version="9"),
        ...     SimpleNamespace(icd_code="E119", icd_version=10),
        ... ]}
        >>> patient = SimpleNamespace(
        ...     get_events=lambda event_type: events.get(event_type, []))
        >>> icd_versions(patient)
        {'E8490': '9', 'E119': '10'}
    """
    seen: dict[str, set[str]] = {}
    for event_type in ICD_EVENT_TYPES:
        for event in _events(patient, event_type):
            code = getattr(event, "icd_code", None)
            version = getattr(event, "icd_version", None)
            if code in (None, "") or version in (None, ""):
                continue
            version = str(version).split(".")[0].strip()
            seen.setdefault(str(code).strip(), set()).add(version)
    return {code: next(iter(versions)) for code, versions in seen.items()
            if len(versions) == 1 and next(iter(versions)) in ("9", "10")}


class EHRTextSerializationTask(BaseTask):
    """Wraps a task so each sample carries its record as Markdown text.

    Works with the dataset of the wrapped task, e.g.
    :class:`~pyhealth.datasets.MIMIC4EHRDataset`,
    :class:`~pyhealth.datasets.MIMIC3Dataset` or
    :class:`~pyhealth.datasets.EHRShotDataset`.

    Every sample of ``base_task`` keeps its label; its code fields become
    one ``text`` field (:func:`serialize_ehr`): the unique codes of each
    field, most recent first, with descriptions from PyHealth's vocabularies
    and the raw code where none is found. Because only the codes already in
    the sample are used, the text covers the same time window as the
    wrapped task.

    A field of vocabulary-prefixed codes (EHRSHOT's ``"ICD10CM/E11.9"``) is
    split into sections by prefix (:data:`PREFIX_SECTIONS`).

    Only code fields (inputs with a processor in :data:`CODE_PROCESSORS`,
    such as ``"sequence"``) are serialized; for a ``"stagenet"`` input,
    whose value is a ``(times, codes)`` pair, the codes are. Other inputs,
    e.g. a ``"timeseries"`` or ``"stagenet_tensor"`` of lab values or a
    ``"cehr"`` input, are left out of the text with a warning; a task with
    no code field raises ``ValueError``.

    Args:
        base_task: The task to wrap, e.g. ``MortalityPredictionMIMIC4()``.
        sections: Sample field to ``(heading, vocabularies)``. Defaults to
            :data:`DEFAULT_SECTIONS` for the fields the task has; any other
            code field is shown under its own name with raw codes.
        demographics: ``demographics(patient, sample) -> dict`` for the
            header of each sample, or ``None`` for no demographics. Defaults to
            :func:`mimic_demographics`. A lambda or nested function is
            named in the cache key by its code, not by captured values.
        describer: Code lookup; defaults to :class:`CodeDescriber`. Pass
            ``CodeDescriber(lookup=lambda code, vocabularies: None)`` to keep
            raw codes.
        max_codes_per_section: Keep at most this many codes per section.
        max_chars: Cut the text to this many characters.
        keep_inputs: Also keep the wrapped task's input fields, e.g. for a
            count-based baseline on the same samples.
        drop_codes: Codes that stand for "missing" and are left out.
            Default ``("0",)``: MIMIC writes a missing NDC as ``"0"``.

    Examples:
        >>> from pyhealth.tasks import MortalityPredictionMIMIC4
        >>> task = EHRTextSerializationTask(MortalityPredictionMIMIC4())
        >>> task.input_schema, task.output_schema
        ({'text': 'text'}, {'mortality': 'binary'})
    """

    def __init__(
        self,
        base_task: BaseTask,
        sections: Mapping[str, tuple[str, Sequence[str]]] | None = None,
        demographics: Callable[[Any, Mapping[str, Any]], Mapping[str, Any]]
        | None = mimic_demographics,
        describer: CodeDescriber | None = None,
        max_codes_per_section: int | None = None,
        max_chars: int | None = None,
        keep_inputs: bool = False,
        drop_codes: Sequence[str] = ("0",),
    ) -> None:
        params = json.dumps(vars(base_task), sort_keys=True, default=str)
        self._base = _Keyed(base_task, f"{_name(type(base_task))}{params}")
        self.task_name = f"{base_task.task_name}/text"
        inputs = dict(base_task.input_schema)
        known = dict(DEFAULT_SECTIONS, **(sections or {}))
        codes = [f for f, spec in inputs.items()
                 if _processor_name(spec) in CODE_PROCESSORS]
        skipped = [f for f in inputs if f not in codes]
        if sections and set(sections) & set(skipped):
            raise ValueError(
                f"sections names non-code inputs {sorted(set(sections) & set(skipped))}"
                f"; only {sorted(CODE_PROCESSORS)} inputs can be serialized"
            )
        if not codes:
            raise ValueError(
                f"{base_task.task_name} has no code inputs to serialize; its "
                f"inputs are {inputs}"
            )
        if skipped:
            logger.warning("%s: inputs %s are not codes and are left out of "
                           "the text", base_task.task_name, skipped)
        self.sections = {
            field: known.get(field, (field.replace("_", " ").capitalize(), ()))
            for field in codes
        }
        self.input_fields = sorted(inputs)
        self.timed_fields = sorted(
            f for f in codes
            if _processor_name(inputs[f]) in TIMED_CODE_PROCESSORS
        )
        self._demographics = _Keyed(demographics, _name(demographics))
        describer = describer or CodeDescriber()
        self._describer = _Keyed(describer, repr(describer))
        self.max_codes_per_section = max_codes_per_section
        self.max_chars = max_chars
        self.keep_inputs = keep_inputs
        self.drop_codes = sorted(set(drop_codes))
        self.input_schema = {**(inputs if keep_inputs else {}), "text": "text"}
        self.output_schema = dict(base_task.output_schema)

    @property
    def base_task(self) -> BaseTask:
        """The wrapped task."""
        return self._base.obj

    @property
    def demographics(
        self,
    ) -> Callable[[Any, Mapping[str, Any]], Mapping[str, Any]] | None:
        """The function giving the demographics header."""
        return self._demographics.obj

    @property
    def describer(self) -> CodeDescriber:
        """The code lookup."""
        return self._describer.obj

    def pre_filter(self, df: Any) -> Any:
        return self.base_task.pre_filter(df)

    def serialize_sample(
        self,
        sample: Mapping[str, Any],
        demographics: Mapping[str, Any] | None = None,
        icd_versions: Mapping[str, str] | None = None,
    ) -> str:
        """The Markdown text of one sample of the wrapped task.

        Args:
            sample: A sample of the wrapped task.
            demographics: The header's attributes.
            icd_versions: Code to ICD version (``"9"`` or ``"10"``) for
                unprefixed codes; such a code is looked up only in that
                version's vocabularies. See :func:`icd_versions`.

        Examples:
            >>> from pyhealth.tasks import MortalityPredictionMIMIC4
            >>> task = EHRTextSerializationTask(
            ...     MortalityPredictionMIMIC4(),
            ...     describer=CodeDescriber(lookup=lambda c, v: None),
            ... )
            >>> text = task.serialize_sample(
            ...     {"conditions": ["4019", "25000"], "procedures": [],
            ...      "drugs": ["00002"]}
            ... )
            >>> text.splitlines()[2:4]
            ['## Conditions', '- 25000']
        """
        grouped: dict[str, list[str]] = {}
        for field, (heading, vocabularies) in self.sections.items():
            # most recent first: tasks list codes in time order
            value = sample.get(field)
            if field in self.timed_fields and isinstance(value, (list, tuple)):
                value = value[1] if len(value) == 2 else None  # (times, codes)
            for code in reversed(_flatten(value)):
                if code in self.drop_codes:
                    continue
                prefix, sep, _ = code.partition("/")
                section = heading
                if sep and len(self.sections) == 1:
                    section = PREFIX_SECTIONS.get(prefix, "Conditions")
                lookup = code
                if (icd_versions and code in icd_versions
                        and any(v.startswith("ICD") for v in vocabularies)):
                    lookup = f"{icd_versions[code]}_{code}"
                text = self.describer(lookup, vocabularies) or code
                grouped.setdefault(section, []).append(text)
        return serialize_ehr(
            grouped, demographics,
            max_codes_per_section=self.max_codes_per_section,
            max_chars=self.max_chars,
        )

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        samples = self.base_task(patient)
        if not samples:
            return []
        versions = icd_versions(patient)
        out = []
        for sample in samples:
            demographics = (self.demographics(patient, sample)
                            if self.demographics else None)
            new = {k: v for k, v in sample.items()
                   if k not in self.input_fields or self.keep_inputs}
            new["text"] = self.serialize_sample(sample, demographics, versions)
            out.append(new)
        return out
