"""FHIR NDJSON parsing, flattening, and Parquet table writing.

Key public API
--------------
stream_fhir_ndjson_to_flat_tables(root, glob_pattern, out_dir)
    Stream all matching NDJSON/NDJSON.GZ resources into six per-type Parquet tables.

sorted_ndjson_files(root, glob_pattern)
    List matching NDJSON files under root (deduplicated, sorted).

filter_flat_tables_by_patient_ids(source_dir, out_dir, keep_ids)
    Subset existing flattened tables to a specific patient cohort.
"""

from __future__ import annotations

import gzip
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import orjson
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

GlobPatternArg = str | Sequence[str]
"""Single glob string or sequence of strings for NDJSON file discovery."""

__all__ = [
    # Types
    "GlobPatternArg",
    # Constants
    "FHIR_SCHEMA_VERSION",
    "FHIR_TABLES",
    "FHIR_TABLES_FOR_PATIENT_IDS",
    "FHIR_TABLE_FILE_NAMES",
    "FHIR_TABLE_COLUMNS",
    # Datetime helpers
    "parse_dt",
    "as_naive",
    # FHIR iteration
    "iter_ndjson_objects",
    "iter_resources_from_ndjson_obj",
    # Resource extraction
    "patient_id_for_resource",
    # Pipeline
    "sorted_ndjson_files",
    "stream_fhir_ndjson_to_flat_tables",
    "filter_flat_tables_by_patient_ids",
    "sorted_patient_ids_from_flat_tables",
]

FHIR_SCHEMA_VERSION = 3

FHIR_TABLES: List[str] = [
    "patient",
    "encounter",
    "condition",
    "observation",
    "medication_request",
    "procedure",
]

FHIR_TABLES_FOR_PATIENT_IDS: List[str] = [t for t in FHIR_TABLES if t != "patient"]

FHIR_TABLE_FILE_NAMES: Dict[str, str] = {t: f"{t}.parquet" for t in FHIR_TABLES}

FHIR_TABLE_COLUMNS: Dict[str, List[str]] = {
    "patient": ["patient_id", "patient_fhir_id", "birth_date", "gender", "deceased_boolean", "deceased_datetime"],
    "encounter": ["patient_id", "resource_id", "encounter_id", "event_time", "encounter_class", "encounter_end"],
    "condition": ["patient_id", "resource_id", "encounter_id", "event_time", "concept_key"],
    "observation": ["patient_id", "resource_id", "encounter_id", "event_time", "concept_key"],
    "medication_request": ["patient_id", "resource_id", "encounter_id", "event_time", "concept_key"],
    "procedure": ["patient_id", "resource_id", "encounter_id", "event_time", "concept_key"],
}

# ---------------------------------------------------------------------------
# Datetime helpers (also imported by cehr_processor)
# ---------------------------------------------------------------------------


def parse_dt(s: Optional[str]) -> Optional[datetime]:
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
    return dt.replace(tzinfo=None) if dt.tzinfo is not None else dt


def as_naive(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is None:
        return None
    return dt.replace(tzinfo=None) if dt.tzinfo is not None else dt


# ---------------------------------------------------------------------------
# FHIR JSON helpers
# ---------------------------------------------------------------------------


def _coding_key(coding: Dict[str, Any]) -> str:
    return f"{coding.get('system') or 'unknown'}|{coding.get('code') or 'unknown'}"


def _first_coding(obj: Optional[Dict[str, Any]]) -> Optional[str]:
    if not obj:
        return None
    codings = obj.get("coding") or []
    if not codings and "concept" in obj:
        codings = (obj.get("concept") or {}).get("coding") or []
    return _coding_key(codings[0]) if codings else None


def _ref_id(ref: Optional[str]) -> Optional[str]:
    if not ref:
        return None
    return ref.rsplit("/", 1)[-1] if "/" in ref else ref


def _unwrap_resource_dict(raw: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    resource = raw.get("resource") if "resource" in raw else raw
    return resource if isinstance(resource, dict) else None


def iter_resources_from_ndjson_obj(obj: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """Yield resource dicts from one parsed NDJSON object (Bundle or bare resource)."""
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
    """Yield parsed JSON objects from a plain or gzip-compressed NDJSON file."""
    opener = (
        gzip.open(path, "rt", encoding="utf-8", errors="replace")
        if path.suffix == ".gz"
        else open(path, encoding="utf-8", errors="replace")
    )
    with opener as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            parsed = orjson.loads(line)
            if isinstance(parsed, dict):
                yield parsed


# ---------------------------------------------------------------------------
# Resource field extraction
# ---------------------------------------------------------------------------


def _clinical_concept_key(res: Dict[str, Any]) -> Optional[str]:
    """Resolve a stable token key from a FHIR resource."""
    resource_type = res.get("resourceType")
    if resource_type == "MedicationRequest":
        med_cc = res.get("medicationCodeableConcept")
        if isinstance(med_cc, dict):
            key = _first_coding(med_cc)
            if key:
                return key
        med_ref = res.get("medicationReference")
        if isinstance(med_ref, dict):
            ref = med_ref.get("reference")
            if ref:
                return f"MedicationRequest/reference|{_ref_id(ref) or ref}"
        return None
    code = res.get("code")
    return _first_coding(code) if isinstance(code, dict) else None


def patient_id_for_resource(
    resource: Dict[str, Any],
    resource_type: Optional[str] = None,
) -> Optional[str]:
    resource_type = resource_type or resource.get("resourceType")
    if resource_type == "Patient":
        pid = resource.get("id")
        return str(pid) if pid is not None else None
    if resource_type in {"Encounter", "Condition", "Observation", "MedicationRequest", "Procedure"}:
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


# ---------------------------------------------------------------------------
# Flattening
# ---------------------------------------------------------------------------


def _normalize_deceased_boolean_for_storage(value: Any) -> Optional[str]:
    """Map Patient.deceasedBoolean to stored "true"/"false"/None.

    FHIR JSON uses real booleans; some exports use strings. Python's
    bool("false") is True, so we must not coerce with bool().
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


_RESOURCE_TYPE_TO_TABLE: Dict[str, str] = {
    "Condition": "condition",
    "Observation": "observation",
    "MedicationRequest": "medication_request",
    "Procedure": "procedure",
}


def _flatten_resource_to_table_row(
    resource: Dict[str, Any],
) -> Optional[Tuple[str, Dict[str, Optional[str]]]]:
    """Map one FHIR resource dict to (table_name, row_dict), or None if unsupported."""
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
            "deceased_boolean": _normalize_deceased_boolean_for_storage(resource.get("deceasedBoolean")),
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

    table_name = _RESOURCE_TYPE_TO_TABLE.get(resource_type)
    if table_name is None:
        return None
    return table_name, {
        "patient_id": patient_id,
        "resource_id": resource_id,
        "encounter_id": _ref_id((resource.get("encounter") or {}).get("reference")),
        "event_time": event_time,
        "concept_key": _clinical_concept_key(resource),
    }


# ---------------------------------------------------------------------------
# Parquet writer
# ---------------------------------------------------------------------------


def _table_schema(table_name: str) -> pa.Schema:
    return pa.schema([(col, pa.string()) for col in FHIR_TABLE_COLUMNS[table_name]])


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


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def sorted_ndjson_files(root: Path, glob_pattern: GlobPatternArg) -> List[Path]:
    """Return sorted unique file paths under root matching glob pattern(s).

    Args:
        root: Root directory to search under.
        glob_pattern: Single glob string or sequence of glob strings.

    Returns:
        Sorted list of matching files. Empty if no matches.
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
    """Stream NDJSON resources into normalized per-resource Parquet tables under out_dir.

    Args:
        root: Root directory containing NDJSON/NDJSON.GZ files.
        glob_pattern: Single glob string or sequence of glob strings.
        out_dir: Output directory for per-resource-type Parquet tables.
            Creates patient.parquet, encounter.parquet, condition.parquet,
            observation.parquet, medication_request.parquet, procedure.parquet.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    writers = {
        name: _BufferedParquetWriter(path=out_dir / FHIR_TABLE_FILE_NAMES[name], schema=_table_schema(name))
        for name in FHIR_TABLES
    }
    try:
        for file_path in sorted_ndjson_files(root, glob_pattern):
            for ndjson_obj in iter_ndjson_objects(file_path):
                for resource in iter_resources_from_ndjson_obj(ndjson_obj):
                    result = _flatten_resource_to_table_row(resource)
                    if result is not None:
                        writers[result[0]].add(result[1])
    finally:
        for writer in writers.values():
            writer.close()


def sorted_patient_ids_from_flat_tables(table_dir: Path) -> List[str]:
    """Return sorted unique patient IDs from a directory of flattened Parquet tables."""
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
        pl.scan_parquet(str(table_dir / FHIR_TABLE_FILE_NAMES[t])).select("patient_id")
        for t in FHIR_TABLES_FOR_PATIENT_IDS
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
    """Filter all flattened tables to only include rows for the given patient IDs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    keep_set = set(keep_ids)
    for name in FHIR_TABLES:
        src = source_dir / FHIR_TABLE_FILE_NAMES[name]
        dst = out_dir / FHIR_TABLE_FILE_NAMES[name]
        pl.scan_parquet(str(src)).filter(pl.col("patient_id").is_in(keep_set)).sink_parquet(str(dst))
