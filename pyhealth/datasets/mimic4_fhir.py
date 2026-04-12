"""MIMIC-IV FHIR ingestion using flattened resource tables.

The maintainer-requested architecture for FHIR in PyHealth is:

1. Stream NDJSON/NDJSON.GZ FHIR resources from disk.
2. Normalize each resource type into a 2D table (Patient, Encounter, Condition,
   Observation, MedicationRequest, Procedure).
3. Feed those tables through the standard YAML-driven
   :class:`~pyhealth.datasets.BaseDataset` pipeline so downstream task processing
   operates on :class:`~pyhealth.data.Patient` and ``global_event_df`` rows rather
   than custom nested FHIR objects.

This module implements that flow. The dataset builds normalized resource tables
under its cache directory, then loads them through a regular ``tables:`` config in
``configs/mimic4_fhir.yaml``.
"""

from __future__ import annotations

import functools
import gzip
import hashlib
import itertools
import logging
import operator
import os
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import dask.dataframe as dd
import narwhals as nw
import orjson
import pandas as pd
import platformdirs
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from litdata.processing.data_processor import in_notebook
from yaml import safe_load

from ..data import Patient
from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)

DEFAULT_PAD = 0
DEFAULT_UNK = 1
FHIR_SCHEMA_VERSION = 3

FHIR_TABLES: List[str] = [
    "patient",
    "encounter",
    "condition",
    "observation",
    "medication_request",
    "procedure",
]

# Tables that carry ``patient_id`` for cohort discovery when ``patient.parquet`` is absent.
FHIR_TABLES_FOR_PATIENT_IDS: List[str] = [t for t in FHIR_TABLES if t != "patient"]

FHIR_TABLE_FILE_NAMES: Dict[str, str] = {
    table_name: f"{table_name}.parquet" for table_name in FHIR_TABLES
}

FHIR_TABLE_COLUMNS: Dict[str, List[str]] = {
    "patient": [
        "patient_id",
        "patient_fhir_id",
        "birth_date",
        "gender",
        "deceased_boolean",
        "deceased_datetime",
    ],
    "encounter": [
        "patient_id",
        "resource_id",
        "encounter_id",
        "event_time",
        "encounter_class",
        "encounter_end",
    ],
    "condition": [
        "patient_id",
        "resource_id",
        "encounter_id",
        "event_time",
        "concept_key",
    ],
    "observation": [
        "patient_id",
        "resource_id",
        "encounter_id",
        "event_time",
        "concept_key",
    ],
    "medication_request": [
        "patient_id",
        "resource_id",
        "encounter_id",
        "event_time",
        "concept_key",
    ],
    "procedure": [
        "patient_id",
        "resource_id",
        "encounter_id",
        "event_time",
        "concept_key",
    ],
}

EVENT_TYPE_TO_TOKEN_TYPE = {
    "encounter": 1,
    "condition": 2,
    "medication_request": 3,
    "observation": 4,
    "procedure": 5,
}


def _fhir_json_loads_ndjson_line(line: str) -> Any:
    return orjson.loads(line.encode("utf-8"))


def _parse_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        dt = None
    if dt is None and len(s) >= 10:
        try:
            dt = datetime.strptime(s[:10], "%Y-%m-%d")
        except ValueError:
            return None
    if dt is None:
        return None
    if dt.tzinfo is not None:
        return dt.replace(tzinfo=None)
    return dt


def _as_naive(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is None:
        return None
    if dt.tzinfo is not None:
        return dt.replace(tzinfo=None)
    return dt


def _coding_key(coding: Dict[str, Any]) -> str:
    system = coding.get("system") or "unknown"
    code = coding.get("code") or "unknown"
    return f"{system}|{code}"


def _first_coding(obj: Optional[Dict[str, Any]]) -> Optional[str]:
    if not obj:
        return None
    codings = obj.get("coding") or []
    if not codings and "concept" in obj:
        codings = (obj.get("concept") or {}).get("coding") or []
    if not codings:
        return None
    return _coding_key(codings[0])


def _ref_id(ref: Optional[str]) -> Optional[str]:
    if not ref:
        return None
    if "/" in ref:
        return ref.rsplit("/", 1)[-1]
    return ref


def _unwrap_resource_dict(raw: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    resource = raw.get("resource") if "resource" in raw else raw
    return resource if isinstance(resource, dict) else None


def iter_resources_from_ndjson_obj(obj: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """Yield resource dictionaries from one parsed NDJSON object."""

    if "entry" in obj:
        for entry in obj.get("entry") or []:
            resource = entry.get("resource")
            if isinstance(resource, dict):
                yield resource
        return

    resource = _unwrap_resource_dict(obj)
    if resource is not None:
        yield resource


def iter_ndjson_objects(path: Path) -> Iterator[Dict[str, Any]]:
    if path.suffix == ".gz":
        opener = gzip.open(path, "rt", encoding="utf-8", errors="replace")
    else:
        opener = open(path, encoding="utf-8", errors="replace")
    with opener as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            parsed = _fhir_json_loads_ndjson_line(line)
            if isinstance(parsed, dict):
                yield parsed


def _clinical_concept_key(res: Dict[str, Any]) -> Optional[str]:
    """Resolve a stable token key from a flattened FHIR resource."""

    resource_type = res.get("resourceType")
    if resource_type == "MedicationRequest":
        medication_cc = res.get("medicationCodeableConcept")
        if isinstance(medication_cc, dict):
            concept_key = _first_coding(medication_cc)
            if concept_key:
                return concept_key
        medication_ref = res.get("medicationReference")
        if isinstance(medication_ref, dict):
            reference = medication_ref.get("reference")
            if reference:
                ref_id = _ref_id(reference)
                return f"MedicationRequest/reference|{ref_id or reference}"
        return None

    code = res.get("code")
    if isinstance(code, dict):
        return _first_coding(code)
    return None


def patient_id_for_resource(
    resource: Dict[str, Any],
    resource_type: Optional[str] = None,
) -> Optional[str]:
    resource_type = resource_type or resource.get("resourceType")
    if resource_type == "Patient":
        patient_id = resource.get("id")
        return str(patient_id) if patient_id is not None else None
    if resource_type == "Encounter":
        return _ref_id((resource.get("subject") or {}).get("reference"))
    if resource_type in {"Condition", "Observation", "MedicationRequest", "Procedure"}:
        return _ref_id((resource.get("subject") or {}).get("reference"))
    return None


def _resource_time_string(
    resource: Dict[str, Any],
    resource_type: Optional[str] = None,
) -> Optional[str]:
    resource_type = resource_type or resource.get("resourceType")
    if resource_type == "Patient":
        return resource.get("birthDate")
    if resource_type == "Encounter":
        return (resource.get("period") or {}).get("start")
    if resource_type == "Condition":
        return resource.get("onsetDateTime") or resource.get("recordedDate")
    if resource_type == "Observation":
        return resource.get("effectiveDateTime") or resource.get("issued")
    if resource_type == "MedicationRequest":
        return resource.get("authoredOn")
    if resource_type == "Procedure":
        return resource.get("performedDateTime") or resource.get("recordedDate")
    return None


@dataclass
class ConceptVocab:
    """Maps concept keys to dense ids."""

    token_to_id: Dict[str, int] = field(default_factory=dict)
    pad_id: int = DEFAULT_PAD
    unk_id: int = DEFAULT_UNK
    _next_id: int = 2

    def __post_init__(self) -> None:
        if not self.token_to_id:
            self.token_to_id = {"<pad>": self.pad_id, "<unk>": self.unk_id}
            self._next_id = 2

    def add_token(self, key: str) -> int:
        if key in self.token_to_id:
            return self.token_to_id[key]
        token_id = self._next_id
        self._next_id += 1
        self.token_to_id[key] = token_id
        return token_id

    def __getitem__(self, key: str) -> int:
        return self.token_to_id.get(key, self.unk_id)

    @property
    def vocab_size(self) -> int:
        return self._next_id

    def to_json(self) -> Dict[str, Any]:
        return {"token_to_id": self.token_to_id, "next_id": self._next_id}

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "ConceptVocab":
        vocab = cls()
        loaded = dict(data.get("token_to_id") or {})
        if not loaded:
            vocab._next_id = int(data.get("next_id", 2))
            return vocab
        vocab.token_to_id = loaded
        vocab._next_id = int(data.get("next_id", max(loaded.values()) + 1))
        return vocab

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(
            orjson.dumps(self.to_json(), option=orjson.OPT_SORT_KEYS)
        )

    @classmethod
    def load(cls, path: str) -> "ConceptVocab":
        return cls.from_json(orjson.loads(Path(path).read_bytes()))


def ensure_special_tokens(vocab: ConceptVocab) -> Dict[str, int]:
    specials: Dict[str, int] = {}
    for name in ("<cls>", "<reg>", "<mor>", "<readm>"):
        specials[name] = vocab.add_token(name)
    return specials


def synthetic_mpf_one_patient_resources() -> List[Dict[str, Any]]:
    patient: Dict[str, Any] = {
        "resourceType": "Patient",
        "id": "p-synth-1",
        "birthDate": "1950-01-01",
        "gender": "female",
    }
    encounter: Dict[str, Any] = {
        "resourceType": "Encounter",
        "id": "e1",
        "subject": {"reference": "Patient/p-synth-1"},
        "period": {"start": "2020-06-01T10:00:00Z"},
        "class": {"code": "IMP"},
    }
    condition: Dict[str, Any] = {
        "resourceType": "Condition",
        "id": "c1",
        "subject": {"reference": "Patient/p-synth-1"},
        "encounter": {"reference": "Encounter/e1"},
        "code": {
            "coding": [{"system": "http://hl7.org/fhir/sid/icd-10-cm", "code": "I10"}]
        },
        "onsetDateTime": "2020-06-01T11:00:00Z",
    }
    return [patient, encounter, condition]


def synthetic_mpf_two_patient_resources() -> List[Dict[str, Any]]:
    dead_patient: Dict[str, Any] = {
        "resourceType": "Patient",
        "id": "p-synth-2",
        "birthDate": "1940-05-05",
        "deceasedBoolean": True,
    }
    dead_encounter: Dict[str, Any] = {
        "resourceType": "Encounter",
        "id": "e-dead",
        "subject": {"reference": "Patient/p-synth-2"},
        "period": {"start": "2020-07-01T10:00:00Z"},
        "class": {"code": "IMP"},
    }
    dead_observation: Dict[str, Any] = {
        "resourceType": "Observation",
        "id": "o-dead",
        "subject": {"reference": "Patient/p-synth-2"},
        "encounter": {"reference": "Encounter/e-dead"},
        "effectiveDateTime": "2020-07-01T12:00:00Z",
        "code": {"coding": [{"system": "http://loinc.org", "code": "789-0"}]},
    }
    return [
        *synthetic_mpf_one_patient_resources(),
        dead_patient,
        dead_encounter,
        dead_observation,
    ]


def synthetic_mpf_one_patient_ndjson_text() -> str:
    return (
        "\n".join(
            orjson.dumps(resource).decode("utf-8")
            for resource in synthetic_mpf_one_patient_resources()
        )
        + "\n"
    )


def synthetic_mpf_two_patient_ndjson_text() -> str:
    return (
        "\n".join(
            orjson.dumps(resource).decode("utf-8")
            for resource in synthetic_mpf_two_patient_resources()
        )
        + "\n"
    )


def read_fhir_settings_yaml(path: Optional[str] = None) -> Dict[str, Any]:
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "configs", "mimic4_fhir.yaml")
    with open(path, encoding="utf-8") as stream:
        data = safe_load(stream)
    return data if isinstance(data, dict) else {}


def _table_schema(table_name: str) -> pa.Schema:
    return pa.schema([(column, pa.string()) for column in FHIR_TABLE_COLUMNS[table_name]])


class _BufferedParquetWriter:
    def __init__(self, path: Path, schema: pa.Schema, batch_size: int = 50_000) -> None:
        self.path = path
        self.schema = schema
        self.batch_size = batch_size
        self.rows: List[Dict[str, Any]] = []
        self.writer: Optional[pq.ParquetWriter] = None
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, row: Dict[str, Any]) -> None:
        self.rows.append(row)
        if len(self.rows) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        table = pa.Table.from_pylist(self.rows, schema=self.schema)
        if self.writer is None:
            self.writer = pq.ParquetWriter(str(self.path), self.schema)
        self.writer.write_table(table)
        self.rows.clear()

    def close(self) -> None:
        self.flush()
        if self.writer is None:
            pq.write_table(pa.Table.from_pylist([], schema=self.schema), str(self.path))
            return
        self.writer.close()


def _normalize_deceased_boolean_for_storage(value: Any) -> Optional[str]:
    """Map ``Patient.deceasedBoolean`` to stored ``\"true\"`` / ``\"false\"`` / ``None``.

    FHIR JSON uses real booleans; some exports incorrectly use strings. Python's
    ``bool(\"false\")`` is ``True``, so we must not coerce unknown values with
    ``bool()`` or non-living patients can be written as ``deceased_boolean=\"true\"``.
    """
    if value is None:
        return None
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, str):
        key = value.strip().lower()
        if key in ("true", "1", "yes", "y", "t"):
            return "true"
        if key in ("false", "0", "no", "n", "f", ""):
            return "false"
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value == 0:
            return "false"
        if value == 1:
            return "true"
        return None
    return None


def _flatten_resource_to_table_row(
    resource: Dict[str, Any],
) -> Optional[Tuple[str, Dict[str, Optional[str]]]]:
    resource_type = resource.get("resourceType")
    patient_id = patient_id_for_resource(resource, resource_type)
    if not patient_id:
        return None

    if resource_type == "Patient":
        return "patient", {
            "patient_id": patient_id,
            "patient_fhir_id": str(resource.get("id") or patient_id),
            "birth_date": resource.get("birthDate"),
            "gender": resource.get("gender"),
            "deceased_boolean": _normalize_deceased_boolean_for_storage(
                resource.get("deceasedBoolean")
            ),
            "deceased_datetime": resource.get("deceasedDateTime"),
        }

    resource_id = str(resource.get("id")) if resource.get("id") is not None else None
    event_time = _resource_time_string(resource, resource_type)

    if resource_type == "Encounter":
        return "encounter", {
            "patient_id": patient_id,
            "resource_id": resource_id,
            "encounter_id": resource_id,
            "event_time": event_time,
            "encounter_class": (resource.get("class") or {}).get("code"),
            "encounter_end": (resource.get("period") or {}).get("end"),
        }

    linked_encounter_id = _ref_id((resource.get("encounter") or {}).get("reference"))
    concept_key = _clinical_concept_key(resource)
    row = {
        "patient_id": patient_id,
        "resource_id": resource_id,
        "encounter_id": linked_encounter_id,
        "event_time": event_time,
        "concept_key": concept_key,
    }
    if resource_type == "Condition":
        return "condition", row
    if resource_type == "Observation":
        return "observation", row
    if resource_type == "MedicationRequest":
        return "medication_request", row
    if resource_type == "Procedure":
        return "procedure", row
    return None


GlobPatternArg = str | Sequence[str]
"""Type alias for glob pattern argument: single string or sequence of strings."""


def sorted_ndjson_files(root: Path, glob_pattern: GlobPatternArg) -> List[Path]:
    """Return sorted unique file paths under ``root`` matching glob pattern(s).

    Args:
        root (Path): Root directory to search under.
        glob_pattern (GlobPatternArg): Single glob string (e.g., ``"*.ndjson.gz"``)
            or sequence of glob strings. Patterns are applied to ``root.glob()``;
            results are deduplicated and sorted lexicographically by string path.

    Returns:
        List[Path]: Sorted list of matching files. Empty if no matches.

    Example:
        >>> from pathlib import Path
        >>> root = Path("/data/fhir")
        >>> # Single pattern:
        >>> files = sorted_ndjson_files(root, "**/*.ndjson.gz")
        >>> # Multiple patterns (deduplicated):
        >>> files = sorted_ndjson_files(root, [
        ...     "**/MimicPatient*.ndjson.gz",
        ...     "**/MimicEncounter*.ndjson.gz",
        ... ])
    """

    patterns = [glob_pattern] if isinstance(glob_pattern, str) else list(glob_pattern)
    files: set[Path] = set()
    for pat in patterns:
        files.update(p for p in root.glob(pat) if p.is_file())
    return sorted(files, key=lambda p: str(p))


def stream_fhir_ndjson_to_flat_tables(
    root: Path,
    glob_pattern: GlobPatternArg,
    out_dir: Path,
) -> None:
    """Stream NDJSON resources into normalized per-resource Parquet tables.

    Reads all NDJSON/NDJSON.GZ files matching ``glob_pattern`` under ``root``,
    parses each line as FHIR JSON, normalizes each resource via
    ``_flatten_resource_to_table_row``, and writes rows to per-resource-type
    Parquet tables under ``out_dir``. Resources are skipped if their type
    is not in ``FHIR_TABLES`` (e.g., Medication, Specimen, Organization).

    Args:
        root (Path): Root directory containing NDJSON/NDJSON.GZ files.
        glob_pattern (GlobPatternArg): Single glob string or sequence of glob strings
            to match NDJSON files. E.g., ``"**/*.ndjson.gz"`` or
            ``["**/MimicPatient*.ndjson.gz", "**/MimicEncounter*.ndjson.gz"]``.
        out_dir (Path): Output directory for per-resource-type Parquet tables.
            Created if absent. Writes:
            - ``patient.parquet``
            - ``encounter.parquet``
            - ``condition.parquet``
            - ``observation.parquet``
            - ``medication_request.parquet``
            - ``procedure.parquet``

    Raises:
        IOError: If files cannot be read or written.

    Notes:
        - All matching files are decompressed and fully parsed (no early exit
          for unsupported resource types).
        - Rows are buffered in memory (batch size 50k) before writing.
        - Empty output tables are still created.
        - Writers are always closed in a ``finally`` block (including on errors).
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    writers = {
        table_name: _BufferedParquetWriter(
            path=out_dir / FHIR_TABLE_FILE_NAMES[table_name],
            schema=_table_schema(table_name),
        )
        for table_name in FHIR_TABLES
    }

    try:
        files = sorted_ndjson_files(root, glob_pattern)
        if not files:
            return

        for file_path in files:
            for ndjson_obj in iter_ndjson_objects(file_path):
                for resource in iter_resources_from_ndjson_obj(ndjson_obj):
                    flattened = _flatten_resource_to_table_row(resource)
                    if flattened is None:
                        continue
                    table_name, row = flattened
                    writers[table_name].add(row)
    finally:
        for writer in writers.values():
            writer.close()


def _sorted_patient_ids_from_flat_tables(table_dir: Path) -> List[str]:
    patient_table = table_dir / FHIR_TABLE_FILE_NAMES["patient"]
    if patient_table.exists():
        return (
            pl.scan_parquet(str(patient_table))
            .select("patient_id")
            .unique()
            .sort("patient_id")
            .collect(engine="streaming")["patient_id"]
            .to_list()
        )

    frames = [
        pl.scan_parquet(str(table_dir / FHIR_TABLE_FILE_NAMES[table_name])).select("patient_id")
        for table_name in FHIR_TABLES_FOR_PATIENT_IDS
    ]
    return (
        pl.concat(frames)
        .unique()
        .sort("patient_id")
        .collect(engine="streaming")["patient_id"]
        .to_list()
    )


def filter_flat_tables_by_patient_ids(
    source_dir: Path,
    out_dir: Path,
    keep_ids: Sequence[str],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    keep_ids_set = set(keep_ids)
    for table_name in FHIR_TABLES:
        src = source_dir / FHIR_TABLE_FILE_NAMES[table_name]
        dst = out_dir / FHIR_TABLE_FILE_NAMES[table_name]
        pl.scan_parquet(str(src)).filter(pl.col("patient_id").is_in(keep_ids_set)).sink_parquet(
            str(dst)
        )


def _clean_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        return value or None
    return str(value)


def _deceased_boolean_column_means_dead(value: Any) -> bool:
    """True only for an explicit affirmative stored flag (not Python truthiness)."""
    s = _clean_string(value)
    return s is not None and s.lower() == "true"


def _row_datetime(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return _as_naive(value)
    try:
        return _parse_dt(str(value))
    except Exception:
        return None


def _concept_key_from_row(row: Dict[str, Any]) -> str:
    event_type = row.get("event_type")
    if event_type == "condition":
        return _clean_string(row.get("condition/concept_key")) or "condition|unknown"
    if event_type == "observation":
        return _clean_string(row.get("observation/concept_key")) or "observation|unknown"
    if event_type == "medication_request":
        return (
            _clean_string(row.get("medication_request/concept_key"))
            or "medication_request|unknown"
        )
    if event_type == "procedure":
        return _clean_string(row.get("procedure/concept_key")) or "procedure|unknown"
    if event_type == "encounter":
        encounter_class = _clean_string(row.get("encounter/encounter_class"))
        return f"encounter|{encounter_class}" if encounter_class else "encounter|unknown"
    return f"{event_type or 'event'}|unknown"


def _linked_encounter_id_from_row(row: Dict[str, Any]) -> Optional[str]:
    event_type = row.get("event_type")
    if event_type == "condition":
        return _clean_string(row.get("condition/encounter_id"))
    if event_type == "observation":
        return _clean_string(row.get("observation/encounter_id"))
    if event_type == "medication_request":
        return _clean_string(row.get("medication_request/encounter_id"))
    if event_type == "procedure":
        return _clean_string(row.get("procedure/encounter_id"))
    if event_type == "encounter":
        return _clean_string(row.get("encounter/encounter_id"))
    return None


def _birth_datetime_from_patient(patient: Patient) -> Optional[datetime]:
    for row in patient.data_source.iter_rows(named=True):
        if row.get("event_type") != "patient":
            continue
        birth = _row_datetime(row.get("timestamp"))
        if birth is not None:
            return birth
        birth_raw = _clean_string(row.get("patient/birth_date"))
        if birth_raw:
            return _parse_dt(birth_raw)
    return None


def _sequential_visit_idx_for_time(
    event_time: Optional[datetime],
    visit_encounters: List[Tuple[datetime, int]],
) -> int:
    if not visit_encounters:
        return 0
    if event_time is None:
        return visit_encounters[-1][1]
    event_time = _as_naive(event_time)
    chosen = visit_encounters[0][1]
    for encounter_start, visit_idx in visit_encounters:
        if encounter_start <= event_time:
            chosen = visit_idx
        else:
            break
    return chosen


def collect_cehr_timeline_events(
    patient: Patient,
) -> List[Tuple[datetime, str, str, int]]:
    """Collect CEHR timeline events directly from flattened patient rows."""

    rows = list(
        patient.data_source.sort(["timestamp", "event_type"], nulls_last=True).iter_rows(
            named=True
        )
    )

    encounter_rows: List[Tuple[datetime, str]] = []
    for row in rows:
        if row.get("event_type") != "encounter":
            continue
        encounter_id = _linked_encounter_id_from_row(row)
        encounter_start = _row_datetime(row.get("timestamp"))
        if encounter_id is None or encounter_start is None:
            continue
        encounter_rows.append((encounter_start, encounter_id))

    encounter_rows.sort(key=lambda pair: pair[0])
    encounter_visit_idx = {
        encounter_id: visit_idx
        for visit_idx, (_, encounter_id) in enumerate(encounter_rows)
    }
    encounter_start_by_id = {
        encounter_id: encounter_start for encounter_start, encounter_id in encounter_rows
    }
    visit_encounters = [
        (encounter_start, visit_idx)
        for visit_idx, (encounter_start, _) in enumerate(encounter_rows)
    ]

    events: List[Tuple[datetime, str, str, int]] = []
    unlinked: List[Tuple[Optional[datetime], str, str]] = []

    for row in rows:
        event_type = row.get("event_type")
        if event_type not in EVENT_TYPE_TO_TOKEN_TYPE:
            continue

        event_time = _row_datetime(row.get("timestamp"))
        concept_key = _concept_key_from_row(row)

        if event_type == "encounter":
            encounter_id = _linked_encounter_id_from_row(row)
            if encounter_id is None or event_time is None:
                continue
            visit_idx = encounter_visit_idx.get(encounter_id)
            if visit_idx is None:
                continue
            events.append((event_time, concept_key, event_type, visit_idx))
            continue

        encounter_id = _linked_encounter_id_from_row(row)
        if encounter_id and encounter_id in encounter_visit_idx:
            visit_idx = encounter_visit_idx[encounter_id]
            if event_time is None:
                event_time = encounter_start_by_id.get(encounter_id)
            if event_time is None:
                continue
            events.append((event_time, concept_key, event_type, visit_idx))
        else:
            unlinked.append((event_time, concept_key, event_type))

    for event_time, concept_key, event_type in unlinked:
        visit_idx = _sequential_visit_idx_for_time(event_time, visit_encounters)
        if event_time is None:
            if not visit_encounters:
                continue
            for encounter_start, encounter_visit_idx_value in visit_encounters:
                if encounter_visit_idx_value == visit_idx:
                    event_time = encounter_start
                    break
            else:
                event_time = visit_encounters[-1][0]
        if event_time is None:
            continue
        events.append((event_time, concept_key, event_type, visit_idx))

    events.sort(key=lambda item: item[0])
    return events


def warm_mpf_vocab_from_patient(
    vocab: ConceptVocab,
    patient: Patient,
    clinical_cap: int,
) -> None:
    tail = collect_cehr_timeline_events(patient)[-clinical_cap:] if clinical_cap > 0 else []
    for _, concept_key, _, _ in tail:
        vocab.add_token(concept_key)


def build_cehr_sequences(
    patient: Patient,
    vocab: ConceptVocab,
    max_len: int,
    *,
    base_time: Optional[datetime] = None,
    grow_vocab: bool = True,
) -> Tuple[List[int], List[int], List[float], List[float], List[int], List[int]]:
    """Flatten a patient's tabular FHIR rows into CEHR-aligned feature lists."""

    events = collect_cehr_timeline_events(patient)
    birth = _birth_datetime_from_patient(patient)

    if base_time is None and events:
        base_time = events[0][0]
    elif base_time is None:
        base_time = datetime.now()

    concept_ids: List[int] = []
    token_types: List[int] = []
    time_stamps: List[float] = []
    ages: List[float] = []
    visit_orders: List[int] = []
    visit_segments: List[int] = []

    base_time = _as_naive(base_time)
    birth = _as_naive(birth)
    tail = events[-max_len:] if max_len > 0 else []

    for event_time, concept_key, event_type, visit_idx in tail:
        event_time = _as_naive(event_time)
        concept_id = vocab.add_token(concept_key) if grow_vocab else vocab[concept_key]
        token_type = EVENT_TYPE_TO_TOKEN_TYPE.get(event_type, 0)
        time_delta = (
            float((event_time - base_time).total_seconds())
            if base_time is not None and event_time is not None
            else 0.0
        )
        age_years = 0.0
        if birth is not None and event_time is not None:
            age_years = (event_time - birth).days / 365.25

        concept_ids.append(concept_id)
        token_types.append(token_type)
        time_stamps.append(time_delta)
        ages.append(age_years)
        visit_orders.append(min(visit_idx, 511))
        visit_segments.append(visit_idx % 2)

    return concept_ids, token_types, time_stamps, ages, visit_orders, visit_segments


def infer_mortality_label(patient: Patient) -> int:
    """Heuristic binary label from flattened patient rows."""

    for row in patient.data_source.iter_rows(named=True):
        if row.get("event_type") == "patient":
            if _deceased_boolean_column_means_dead(row.get("patient/deceased_boolean")):
                return 1
            if _clean_string(row.get("patient/deceased_datetime")):
                return 1

    for row in patient.data_source.iter_rows(named=True):
        if row.get("event_type") != "condition":
            continue
        concept_key = (_clean_string(row.get("condition/concept_key")) or "").lower()
        if any(token in concept_key for token in ("death", "deceased", "mortality")):
            return 1
    return 0


class MIMIC4FHIRDataset(BaseDataset):
    """MIMIC-IV on FHIR with flattened resource tables and standard task flow.

    This dataset normalizes raw MIMIC-IV FHIR NDJSON/NDJSON.GZ exports into
    six flattened Parquet tables (Patient, Encounter, Condition, Observation,
    MedicationRequest, Procedure), then pipelines them through
    :class:`~pyhealth.datasets.BaseDataset` for standard downstream task
    processing (global event dataframe, patient iteration, task sampling).

    **Ingest flow (out-of-core):**
    1. Scan NDJSON files matching ``glob_patterns`` (defaults to six Mimic* families).
    2. Parse and flatten each FHIR resource into a row in the appropriate table.
    3. Cache normalized tables as Parquet under ``cache_dir / {uuid} / flattened_tables/``.
    4. Load and compose tables into ``global_event_df`` via YAML config.

    **Data model:**
    - Resource types outside ``FHIR_TABLES`` (Medication, Specimen, …) are skipped.
    - Timestamps are coerced from heterogeneous FHIR ISO 8601 strings (with/without
      timezone, or date-only). Coercion keeps downstream Polars/Dask pipelines robust.
    - Concept keys are derived from the first FHIR coding or synthesized from references.

    **Cache fingerprinting:**
    Cache invalidation includes ``glob_patterns`` and YAML digest, so changes to either
    create a new independent cache.
    """

    def __init__(
        self,
        root: str,
        config_path: Optional[str] = None,
        glob_pattern: Optional[str] = None,
        glob_patterns: Optional[Sequence[str]] = None,
        max_patients: Optional[int] = None,
        ingest_num_shards: Optional[int] = None,
        vocab_path: Optional[str] = None,
        cache_dir: Optional[str | Path] = None,
        num_workers: int = 1,
        dev: bool = False,
    ) -> None:
        """Initialize a MIMIC-IV FHIR dataset.

        Args:
            root (str): Path to the NDJSON/NDJSON.GZ export directory.
            config_path (Optional[str]): Path to a custom YAML config file. Defaults to
                ``pyhealth/datasets/configs/mimic4_fhir.yaml``.
            glob_pattern (Optional[str]): Single glob pattern for NDJSON files
                (e.g., ``"*.ndjson.gz"``). Mutually exclusive with ``glob_patterns``.
                Overrides YAML setting.
            glob_patterns (Optional[Sequence[str]]): Multiple glob patterns as a list.
                Patterns are deduplicated and sorted. Mutually exclusive with ``glob_pattern``.
                Overrides YAML setting.
            max_patients (Optional[int]): If set, ingest is limited to the first *N*
                unique patient IDs (sorted). Ingest still parses all matching NDJSON
                unless you narrow ``glob_patterns`` / ``glob_pattern``. For faster
                prototyping on a laptop, combine with narrow globs.
            ingest_num_shards (Optional[int]): Ignored; retained for API compatibility.
            vocab_path (Optional[str]): Path to a pre-built ConceptVocab JSON file.
                If provided and exists, it is loaded; otherwise a new vocab is created.
            cache_dir (Optional[str | Path]): Cache directory root. Behavior:

                - **None** (default): Auto-generated under ``platformdirs.user_cache_dir()``.
                - **str** or **Path**: Used as root; a UUID is appended per configuration.

            num_workers (int): Number of worker processes for task sampling. Defaults to 1.
            dev (bool): Development mode: limits to 1000 patients if ``max_patients`` is None.

        Raises:
            ValueError: If both ``glob_pattern`` and ``glob_patterns`` are provided.
            TypeError: If ``glob_patterns`` in YAML is not a list.
            FileNotFoundError: If ``root`` or ``config_path`` does not exist.

        Notes:
            - **Glob resolution order:** ``glob_patterns`` kwarg → ``glob_pattern`` kwarg
              → YAML ``glob_patterns`` → YAML ``glob_pattern`` → ``"**/*.ndjson.gz"`` (fallback).
            - **Default YAML globs** match only the six MIMIC shard families that map to
              flattened tables, skipping ~10% of PhysioNet exports (Medication, Specimen, …).
            - **Cache fingerprinting** includes ``glob_patterns`` and config YAML digest,
              so changes invalidate the cache.

        Example:
            >>> from pyhealth.datasets import MIMIC4FHIRDataset
            >>> # Using default YAML globs (PhysioNet-compatible):
            >>> ds = MIMIC4FHIRDataset(root="/data/mimic4_fhir")
            >>> print(ds.glob_patterns)  # Shows: [MimicPatient*, ..., MimicProcedure*]
            >>> # Using a custom glob for non-standard NDJSON naming:
            >>> ds = MIMIC4FHIRDataset(
            ...     root="/data/ndjson",
            ...     glob_pattern="*.ndjson",
            ...     max_patients=100,
            ... )
            >>> # Using a narrowed set of patterns for faster testing:
            >>> ds = MIMIC4FHIRDataset(
            ...     root="/data/mimic4_fhir",
            ...     glob_patterns=["**/MimicPatient*.ndjson.gz", "**/MimicObservation*.ndjson.gz"],
            ... )
        """
        del ingest_num_shards

        default_cfg = os.path.join(
            os.path.dirname(__file__), "configs", "mimic4_fhir.yaml"
        )
        self._fhir_config_path = str(Path(config_path or default_cfg).resolve())
        self._fhir_settings = read_fhir_settings_yaml(self._fhir_config_path)
        if glob_pattern is not None and glob_patterns is not None:
            raise ValueError("Pass at most one of glob_pattern and glob_patterns.")
        if glob_patterns is not None:
            self.glob_patterns = list(glob_patterns)
        elif glob_pattern is not None:
            self.glob_patterns = [glob_pattern]
        else:
            raw_list = self._fhir_settings.get("glob_patterns")
            if raw_list:
                if not isinstance(raw_list, list):
                    raise TypeError(
                        "mimic4_fhir.yaml glob_patterns must be a list of strings."
                    )
                self.glob_patterns = [str(x) for x in raw_list]
            elif self._fhir_settings.get("glob_pattern") is not None:
                self.glob_patterns = [str(self._fhir_settings["glob_pattern"])]
            else:
                self.glob_patterns = ["**/*.ndjson.gz"]
        self.glob_pattern = (
            self.glob_patterns[0]
            if len(self.glob_patterns) == 1
            else "; ".join(self.glob_patterns)
        )
        self.max_patients = 1000 if dev and max_patients is None else max_patients
        self.source_root = str(Path(root).expanduser().resolve())
        self.vocab = (
            ConceptVocab.load(vocab_path)
            if vocab_path and os.path.isfile(vocab_path)
            else ConceptVocab()
        )
        super().__init__(
            root=self.source_root,
            tables=FHIR_TABLES,
            dataset_name="mimic4_fhir",
            config_path=self._fhir_config_path,
            cache_dir=cache_dir,
            num_workers=num_workers,
            dev=dev,
        )

    def _init_cache_dir(self, cache_dir: str | Path | None) -> Path:
        try:
            yaml_digest = hashlib.sha256(
                Path(self._fhir_config_path).read_bytes()
            ).hexdigest()[:16]
        except OSError:
            yaml_digest = "missing"
        identity = orjson.dumps(
            {
                "source_root": self.source_root,
                "tables": sorted(self.tables),
                "dataset_name": self.dataset_name,
                "dev": self.dev,
                "glob_patterns": self.glob_patterns,
                "max_patients": self.max_patients,
                "fhir_schema_version": FHIR_SCHEMA_VERSION,
                "fhir_yaml_digest16": yaml_digest,
            },
            option=orjson.OPT_SORT_KEYS,
        ).decode("utf-8")
        cache_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, identity))
        if cache_dir is None:
            out = Path(platformdirs.user_cache_dir(appname="pyhealth")) / cache_id
            out.mkdir(parents=True, exist_ok=True)
            logger.info("No cache_dir provided. Using default cache dir: %s", out)
        else:
            out = Path(cache_dir) / cache_id
            out.mkdir(parents=True, exist_ok=True)
            logger.info("Using provided cache_dir: %s", out)
        return out

    @property
    def prepared_tables_dir(self) -> Path:
        return self.cache_dir / "flattened_tables"

    def _ensure_prepared_tables(self) -> None:
        root = Path(self.source_root)
        if not root.is_dir():
            raise FileNotFoundError(f"MIMIC4 FHIR root not found: {root}")

        expected_files = [
            self.prepared_tables_dir / FHIR_TABLE_FILE_NAMES[table_name]
            for table_name in FHIR_TABLES
        ]
        if all(path.is_file() for path in expected_files):
            return

        if self.prepared_tables_dir.exists():
            shutil.rmtree(self.prepared_tables_dir)

        try:
            if self.max_patients is None:
                staging_root = self.create_tmpdir()
                staging = staging_root / "flattened_fhir_tables"
                staging.mkdir(parents=True, exist_ok=True)
                stream_fhir_ndjson_to_flat_tables(root, self.glob_patterns, staging)
                shutil.move(str(staging), str(self.prepared_tables_dir))
                return

            staging_root = self.create_tmpdir()
            staging = staging_root / "flattened_fhir_tables"
            staging.mkdir(parents=True, exist_ok=True)
            stream_fhir_ndjson_to_flat_tables(root, self.glob_patterns, staging)

            filtered_root = self.create_tmpdir()
            filtered = filtered_root / "flattened_fhir_tables_filtered"
            patient_ids = _sorted_patient_ids_from_flat_tables(staging)
            keep_ids = patient_ids[: self.max_patients]
            filter_flat_tables_by_patient_ids(staging, filtered, keep_ids)
            shutil.move(str(filtered), str(self.prepared_tables_dir))
        finally:
            self.clean_tmpdir()

    def _event_transform(self, output_dir: Path) -> None:
        self._ensure_prepared_tables()
        super()._event_transform(output_dir)

    def load_table(self, table_name: str) -> dd.DataFrame:
        """Load one flattened Parquet table, mirroring BaseDataset.load_table's contract.

        Differences from the base CSV path that are intentional and FHIR-specific:
        - Source is a pre-built Parquet file under ``prepared_tables_dir``, not CSV.
        - Timestamps use ``errors="coerce"`` (FHIR ISO strings include timezone ``Z`` suffix
          or are partial dates; ``errors="raise"`` would break).
        - After timestamp parsing, any tz-aware column is stripped to naive UTC
          (Dask's ``to_parquet`` / ``sort_values`` path cannot handle tz-aware datetimes).
        - Rows with null ``patient_id`` are dropped before returning so the caller's
          ``sort_values("patient_id")`` in ``_event_transform`` never sees null keys.
        Everything else (column lowercasing, preprocess hook, join, attribute renaming)
        matches BaseDataset.load_table exactly.

        NOTE: This method mirrors BaseDataset.load_table (base_dataset.py).
        The ONLY deviations are:
          1. dd.read_parquet() instead of _scan_csv_tsv_gz()
          2. errors="coerce" + utc=True in dd.to_datetime
          3. map_partitions(tz_localize(None)) for tz-aware dates
          4. dropna(subset=["patient_id"])
        If BaseDataset.load_table changes, audit those 4 points here.
        """

        assert self.config is not None, "Config must be provided to load tables"
        if table_name not in self.config.tables:
            raise ValueError(f"Table {table_name} not found in config")

        table_cfg = self.config.tables[table_name]
        path = self.prepared_tables_dir / table_cfg.file_path
        if not path.exists():
            raise FileNotFoundError(f"Flattened table not found: {path}")

        logger.info("Scanning FHIR flattened table: %s from %s", table_name, path)
        df: dd.DataFrame = dd.read_parquet(
            str(path),
            split_row_groups=True,  # type: ignore[arg-type]
            blocksize="64MB",
        ).replace("", pd.NA)

        # Mirror BaseDataset.load_table: lowercase columns before preprocess hook.
        df = df.rename(columns=str.lower)

        # Mirror BaseDataset.load_table: optional preprocess_{table_name} hook.
        preprocess_func = getattr(self, f"preprocess_{table_name}", None)
        if preprocess_func is not None:
            logger.info(
                "Preprocessing FHIR table: %s with %s", table_name, preprocess_func.__name__
            )
            df = preprocess_func(nw.from_native(df)).to_native()  # type: ignore[union-attr]

        # Mirror BaseDataset.load_table: handle joins (resolved against prepared_tables_dir).
        for join_cfg in table_cfg.join:
            other_path = self.prepared_tables_dir / Path(join_cfg.file_path).name
            if not other_path.exists():
                raise FileNotFoundError(f"FHIR join table not found: {other_path}")
            logger.info("Joining FHIR table %s with %s", table_name, other_path)
            join_df: dd.DataFrame = dd.read_parquet(
                str(other_path),
                split_row_groups=True,  # type: ignore[arg-type]
                blocksize="64MB",
            ).replace("", pd.NA)
            join_df = join_df.rename(columns=str.lower)
            join_key = join_cfg.on.lower()
            columns = [c.lower() for c in join_cfg.columns]
            df = df.merge(join_df[[join_key] + columns], on=join_key, how=join_cfg.how)

        patient_id_col = table_cfg.patient_id
        timestamp_col = table_cfg.timestamp
        timestamp_format = table_cfg.timestamp_format
        attribute_cols = table_cfg.attributes

        # Timestamp parsing: coerce rather than raise for FHIR heterogeneous strings.
        if timestamp_col:
            if isinstance(timestamp_col, list):
                timestamp_series: dd.Series = functools.reduce(
                    operator.add, (df[col].astype("string") for col in timestamp_col)
                )
            else:
                timestamp_series = df[timestamp_col].astype("string")

            # utc=True avoids mixed-offset parse errors; we strip tz after.
            timestamp_series = dd.to_datetime(
                timestamp_series,
                format=timestamp_format,
                errors="coerce",
                utc=True,
            )

            def _strip_tz_to_naive_ms(part: pd.Series) -> pd.Series:
                if getattr(part.dtype, "tz", None) is not None:
                    part = part.dt.tz_localize(None)
                return part.astype("datetime64[ms]")

            timestamp_series = timestamp_series.map_partitions(_strip_tz_to_naive_ms)
            df = df.assign(timestamp=timestamp_series)
        else:
            df = df.assign(timestamp=pd.NaT)

        # Mirror BaseDataset.load_table: patient_id from config column or row index.
        if patient_id_col:
            df = df.assign(patient_id=df[patient_id_col].astype("string"))
        else:
            df = df.reset_index(drop=True)
            df = df.assign(patient_id=df.index.astype("string"))

        # Drop rows without a patient key; BaseDataset._event_transform's sort_values
        # on "patient_id" fails on null keys with Dask's division-calculation logic.
        df = df.dropna(subset=["patient_id"])

        df = df.assign(event_type=table_name)

        rename_attr = {attr.lower(): f"{table_name}/{attr}" for attr in attribute_cols}
        df = df.rename(columns=rename_attr)
        attr_cols = [rename_attr[attr.lower()] for attr in attribute_cols]
        final_cols = ["patient_id", "event_type", "timestamp"] + attr_cols
        return df[final_cols]

    @property
    def unique_patient_ids(self) -> List[str]:
        if self._unique_patient_ids is None:
            self._unique_patient_ids = (
                self.global_event_df.select("patient_id")
                .unique()
                .sort("patient_id")
                .collect(engine="streaming")["patient_id"]
                .to_list()
            )
            logger.info("Found %d unique patient IDs", len(self._unique_patient_ids))
        return self._unique_patient_ids

    def set_task(
        self,
        task: Any = None,
        num_workers: Optional[int] = None,
        input_processors: Optional[Any] = None,
        output_processors: Optional[Any] = None,
    ) -> Any:
        self._main_guard(self.set_task.__name__)
        if task is None:
            raise ValueError(
                "Pass a task instance, e.g. MPFClinicalPredictionTask(max_len=512)."
            )

        from pyhealth.tasks.mpf_clinical_prediction import MPFClinicalPredictionTask

        if isinstance(task, MPFClinicalPredictionTask):
            worker_count = (
                1
                if in_notebook()
                else (num_workers if num_workers is not None else self.num_workers)
            )
            warmup_pids = self._mpf_patient_ids_for_task(task)
            patient_count = len(warmup_pids)
            effective_workers = min(worker_count, patient_count) if patient_count else 1
            ensure_special_tokens(self.vocab)
            self._warm_mpf_vocabulary(task, warmup_pids)
            task.frozen_vocab = effective_workers > 1
            task.vocab = self.vocab
            task._specials = ensure_special_tokens(self.vocab)

        return super().set_task(
            task,
            num_workers,
            input_processors,
            output_processors,
        )

    def _mpf_patient_ids_for_task(self, task: Any) -> List[str]:
        filtered = task.pre_filter(self.global_event_df)
        return (
            filtered.select("patient_id")
            .unique()
            .collect(engine="streaming")
            .to_series()
            .sort()
            .to_list()
        )

    def _warm_mpf_vocabulary(self, task: Any, patient_ids: List[str]) -> None:
        clinical_cap = max(0, task.max_len - 2)
        base = self.global_event_df
        for batch in itertools.batched(patient_ids, 128):
            patients = (
                base.filter(pl.col("patient_id").is_in(batch))
                .collect(engine="streaming")
                .partition_by("patient_id", as_dict=True)
            )
            for patient_key, patient_df in patients.items():
                patient_id = patient_key[0]
                patient = Patient(patient_id=patient_id, data_source=patient_df)
                warm_mpf_vocab_from_patient(self.vocab, patient, clinical_cap)

    def gather_samples(self, task: Any) -> List[Dict[str, Any]]:
        task.vocab = self.vocab
        task._specials = None
        task.frozen_vocab = False
        samples: List[Dict[str, Any]] = []
        for patient in self.iter_patients():
            samples.extend(task(patient))
        return samples
