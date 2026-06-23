"""FHIR NDJSON parsing, generic flattening, and tabular table writing.

This module is the **stateless engine** behind FHIR-to-tabular conversion. It
knows nothing about any specific FHIR source or resource type: the per-resource
projection is supplied as a declarative registry of :class:`ResourceSpec` objects
(see ``MIMIC4FHIR.RESOURCE_SPECS`` for an example) and applied generically by
:func:`flatten_resource`.

Key public API
--------------
flatten_resource(resource, specs)
    Project one FHIR resource dict into ``(table_name, row_dict)`` using a spec
    registry, or ``None`` if the resource is unconfigured / missing a required
    field.

stream_fhir_ndjson_to_flat_tables(root, glob_pattern, out_dir, specs, output_format)
    Stream all matching NDJSON/NDJSON.GZ resources into per-type flat tables
    (parquet/csv/tsv), validating + counting drops along the way.

sorted_ndjson_files(root, glob_pattern)
    List matching NDJSON files under root (deduplicated, sorted).

filter_flat_tables_by_patient_ids(source_dir, out_dir, keep_ids, tables, output_format)
    Subset existing flattened tables to a specific patient cohort.

Authors:
    John Wu and Evan Febrianto
"""

from __future__ import annotations

import gzip
import logging
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import orjson
import polars as pl
import pyarrow as pa
import pyarrow.csv as pa_csv
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

GlobPatternArg = str | Sequence[str]
"""Single glob string or sequence of strings for NDJSON file discovery."""

__all__ = [
    # Types
    "GlobPatternArg",
    "Col",
    "ResourceSpec",
    # Constants
    "FHIR_SCHEMA_VERSION",
    "SUPPORTED_OUTPUT_FORMATS",
    "TRANSFORMS",
    # Spec helpers
    "tables_from_specs",
    "columns_from_specs",
    "table_file_name",
    "load_resource_specs_from_yaml",
    # Datetime helpers
    "parse_dt",
    "as_naive",
    # FHIR iteration
    "iter_ndjson_objects",
    "iter_resources_from_ndjson_obj",
    # Extraction
    "flatten_resource",
    # Pipeline
    "sorted_ndjson_files",
    "stream_fhir_ndjson_to_flat_tables",
    "filter_flat_tables_by_patient_ids",
    "sorted_patient_ids_from_flat_tables",
]

# Bump when the flattening engine or its output layout changes; folded into the
# dataset cache identity so stale caches rebuild automatically.
FHIR_SCHEMA_VERSION = 4

SUPPORTED_OUTPUT_FORMATS = ("parquet", "csv", "tsv")


# ---------------------------------------------------------------------------
# Declarative extraction spec (the registry's value type)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Col:
    """How to project one flat column out of a FHIR resource.

    Typically constructed indirectly by :meth:`ResourceSpec.from_dict` while
    loading the dataset's YAML config; direct instantiation is supported for
    programmatic use.

    Attributes:
        locate: Ordered dotted paths into the resource; the first that resolves
            to a non-null value wins. This is how FHIR choice-types (``onset[x]``,
            ``effective[x]``, …) are handled — list every variant explicitly.
        transform: Name of a value transform in :data:`TRANSFORMS` that converts
            the located leaf into a flat scalar string.
        required: When ``True``, a resource whose ``locate`` cannot be resolved is
            dropped (and counted) rather than emitted with a null.
    """

    locate: Tuple[str, ...]
    transform: str = "identity"
    required: bool = False

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, ctx: str = "") -> "Col":
        """Build a :class:`Col` from a YAML-style dict.

        Expected shape::

            { locate: ["path.a", "path.b"], transform: "ref_id", required: false }

        ``transform`` defaults to ``"identity"`` and must name an entry in
        :data:`TRANSFORMS`. ``required`` defaults to ``False``. A missing or
        empty ``locate`` field raises ``ValueError``.

        Args:
            data: Mapping containing ``locate`` (required) and the optional
                ``transform`` / ``required`` keys.
            ctx: Optional context string used in error messages
                (e.g. ``"Patient.patient_id"``).
        """
        if not isinstance(data, Mapping):
            raise ValueError(
                f"{ctx or 'Col'}: expected a mapping, got {type(data).__name__}."
            )
        raw_locate = data.get("locate")
        if not raw_locate:
            raise ValueError(
                f"{ctx or 'Col'}: missing required field 'locate'."
            )
        if isinstance(raw_locate, str):
            locate: Tuple[str, ...] = (raw_locate,)
        else:
            locate = tuple(str(p) for p in raw_locate)
            if not locate:
                raise ValueError(
                    f"{ctx or 'Col'}: 'locate' must list at least one path."
                )
        transform = str(data.get("transform", "identity"))
        if transform not in TRANSFORMS:
            allowed = ", ".join(sorted(TRANSFORMS.keys()))
            raise ValueError(
                f"{ctx or 'Col'}: unknown transform {transform!r}. "
                f"Allowed: {allowed}."
            )
        required = bool(data.get("required", False))
        return cls(locate=locate, transform=transform, required=required)


@dataclass(frozen=True)
class ResourceSpec:
    """How to project one FHIR resource type into a flat table.

    Typically constructed indirectly by :func:`load_resource_specs_from_yaml`
    while loading the dataset's YAML config; direct instantiation is supported
    for programmatic use.

    Attributes:
        table: Output table name (also the per-type file stem).
        columns: Mapping of output column name -> :class:`Col`. Insertion order
            defines the table's column order.
    """

    table: str
    columns: Mapping[str, Col]

    @classmethod
    def from_dict(
        cls, resource_type: str, data: Mapping[str, Any]
    ) -> "ResourceSpec":
        """Build a :class:`ResourceSpec` from a YAML-style dict.

        Expected shape::

            {
                table: "patient",
                columns: {
                    patient_id: { locate: ["id"], required: true },
                    birth_date: { locate: ["birthDate"] },
                    ...
                },
            }

        Args:
            resource_type: FHIR resourceType string this spec describes
                (e.g. ``"Patient"``). Used only for error messages.
            data: Mapping containing ``table`` (required, str) and
                ``columns`` (required, mapping of column name -> Col-shaped
                mapping).
        """
        if not isinstance(data, Mapping):
            raise ValueError(
                f"resource_specs.{resource_type}: expected a mapping, "
                f"got {type(data).__name__}."
            )
        table = data.get("table")
        if not isinstance(table, str) or not table:
            raise ValueError(
                f"resource_specs.{resource_type}: missing required field "
                f"'table' (string)."
            )
        raw_columns = data.get("columns")
        if not isinstance(raw_columns, Mapping) or not raw_columns:
            raise ValueError(
                f"resource_specs.{resource_type}: missing required field "
                f"'columns' (non-empty mapping)."
            )
        columns: Dict[str, Col] = {}
        for col_name, col_data in raw_columns.items():
            columns[str(col_name)] = Col.from_dict(
                col_data,
                ctx=f"resource_specs.{resource_type}.columns.{col_name}",
            )
        return cls(table=str(table), columns=columns)


def load_resource_specs_from_yaml(
    raw: Mapping[str, Any],
) -> Dict[str, ResourceSpec]:
    """Build the spec registry from a parsed YAML's ``resource_specs:`` block.

    Args:
        raw: The full parsed YAML mapping (top-level dict). The
            ``resource_specs`` key, if present, must be a mapping of FHIR
            resourceType -> ResourceSpec-shaped dict.

    Returns:
        Insertion-ordered mapping of resourceType to :class:`ResourceSpec`.

    Raises:
        ValueError: If the ``resource_specs`` block is missing, empty, or
            contains a malformed entry.
    """
    block = raw.get("resource_specs")
    if not isinstance(block, Mapping) or not block:
        raise ValueError(
            "config: missing or empty top-level 'resource_specs:' block. "
            "Declare at least one FHIR resourceType -> spec mapping."
        )
    specs: Dict[str, ResourceSpec] = {}
    for resource_type, data in block.items():
        specs[str(resource_type)] = ResourceSpec.from_dict(
            str(resource_type), data
        )
    return specs


def tables_from_specs(specs: Mapping[str, ResourceSpec]) -> List[str]:
    """Ordered, de-duplicated list of output table names declared by *specs*."""
    seen: Dict[str, None] = {}
    for spec in specs.values():
        seen.setdefault(spec.table, None)
    return list(seen.keys())


def columns_from_specs(specs: Mapping[str, ResourceSpec]) -> Dict[str, List[str]]:
    """Map each output table name to its ordered column names."""
    return {spec.table: list(spec.columns.keys()) for spec in specs.values()}


def table_file_name(table_name: str, output_format: str = "parquet") -> str:
    """Filename for a flattened table given the output format."""
    ext = "parquet" if output_format == "parquet" else output_format
    return f"{table_name}.{ext}"


# ---------------------------------------------------------------------------
# Datetime helpers (kept for external callers)
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
    """CodeableConcept -> ``"system|code"`` for its first coding (or None)."""
    if not isinstance(obj, dict):
        return None
    codings = obj.get("coding") or []
    if not codings and "concept" in obj:
        codings = (obj.get("concept") or {}).get("coding") or []
    return _coding_key(codings[0]) if codings else None


def _ref_id(ref: Optional[Any]) -> Optional[str]:
    """``{"reference": "Patient/p1"}`` or ``"Patient/p1"`` -> ``"p1"``."""
    if isinstance(ref, dict):
        ref = ref.get("reference")
    if not ref:
        return None
    return ref.rsplit("/", 1)[-1] if "/" in ref else ref


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


def _medication_concept_key(value: Any) -> Optional[str]:
    """MedicationRequest medication[x] -> a stable concept key.

    Accepts either a ``medicationCodeableConcept`` (-> ``"system|code"``) or a
    ``medicationReference`` (-> ``"MedicationRequest/reference|<id>"``).
    """
    if not isinstance(value, dict):
        return None
    if "coding" in value or "concept" in value:
        key = _first_coding(value)
        if key:
            return key
    ref = value.get("reference")
    if ref:
        return f"MedicationRequest/reference|{_ref_id(ref) or ref}"
    return None


def _identity(value: Any) -> Optional[str]:
    if value is None or isinstance(value, str):
        return value
    return str(value)


# Transform registry: how a located leaf becomes a flat scalar string.
TRANSFORMS = {
    "identity": _identity,
    "ref_id": _ref_id,
    "coding_key": _first_coding,
    "bool_norm": _normalize_deceased_boolean_for_storage,
    "med_concept": _medication_concept_key,
}


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
            try:
                parsed = orjson.loads(line)
            except orjson.JSONDecodeError as e:
                logger.warning("Skipping invalid JSON line in %s: %s", path, e)
                continue
            if isinstance(parsed, dict):
                yield parsed


# ---------------------------------------------------------------------------
# Generic extraction engine
# ---------------------------------------------------------------------------


def _get_path(obj: Any, path: str) -> Any:
    """Walk a dotted path (e.g. ``"encounter.reference"``) safely; None if absent."""
    cur = obj
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def _first_located(resource: Dict[str, Any], paths: Tuple[str, ...]) -> Any:
    """First non-null value among the ordered ``paths`` (choice-type resolution)."""
    for path in paths:
        value = _get_path(resource, path)
        if value is not None:
            return value
    return None


def flatten_resource(
    resource: Dict[str, Any],
    specs: Mapping[str, ResourceSpec],
) -> Optional[Tuple[str, Dict[str, Optional[str]]]]:
    """Project one FHIR resource into ``(table_name, row)`` via *specs*.

    Returns ``None`` if the resource type is not configured in *specs*, or if a
    column marked ``required`` cannot be resolved (a dropped/corrupted resource).
    """
    spec = specs.get(resource.get("resourceType"))
    if spec is None:
        return None
    row: Dict[str, Optional[str]] = {}
    for name, col in spec.columns.items():
        raw = _first_located(resource, col.locate)
        if raw is None and col.required:
            return None
        row[name] = TRANSFORMS[col.transform](raw)
    return spec.table, row


# ---------------------------------------------------------------------------
# Tabular writer (parquet / csv / tsv)
# ---------------------------------------------------------------------------


def _table_schema(columns: Sequence[str]) -> pa.Schema:
    return pa.schema([(col, pa.string()) for col in columns])


class _BufferedTableWriter:
    """Buffered, streaming writer for one flat table in parquet/csv/tsv."""

    def __init__(
        self,
        path: Path,
        schema: pa.Schema,
        output_format: str = "parquet",
        batch_size: int = 50_000,
    ) -> None:
        self.path = path
        self.schema = schema
        self.output_format = output_format
        self.batch_size = batch_size
        self.rows: List[Dict[str, Any]] = []
        self._pq_writer: Optional[pq.ParquetWriter] = None
        self._fh = None
        self._csv_header_written = False
        self._delimiter = "\t" if output_format == "tsv" else ","
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, row: Dict[str, Any]) -> None:
        self.rows.append(row)
        if len(self.rows) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        table = pa.Table.from_pylist(self.rows, schema=self.schema)
        if self.output_format == "parquet":
            if self._pq_writer is None:
                self._pq_writer = pq.ParquetWriter(str(self.path), self.schema)
            self._pq_writer.write_table(table)
        else:
            if self._fh is None:
                self._fh = open(self.path, "wb")
            pa_csv.write_csv(
                table,
                self._fh,
                write_options=pa_csv.WriteOptions(
                    include_header=not self._csv_header_written,
                    delimiter=self._delimiter,
                ),
            )
            self._csv_header_written = True
        self.rows.clear()

    def close(self) -> None:
        self.flush()
        if self.output_format == "parquet":
            if self._pq_writer is None:
                pq.write_table(
                    pa.Table.from_pylist([], schema=self.schema), str(self.path)
                )
            else:
                self._pq_writer.close()
            return
        if self._fh is None:
            # Empty table: still write a header-only file for a stable schema.
            self._fh = open(self.path, "wb")
            pa_csv.write_csv(
                pa.Table.from_pylist([], schema=self.schema),
                self._fh,
                write_options=pa_csv.WriteOptions(
                    include_header=True, delimiter=self._delimiter
                ),
            )
        self._fh.close()


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
    specs: Mapping[str, ResourceSpec],
    output_format: str = "parquet",
) -> None:
    """Stream NDJSON resources into normalized per-resource flat tables.

    Resources are validated as they stream: anything whose type is not in
    *specs*, or which is missing a ``required`` field, is dropped and counted; a
    summary is logged at the end so corruption is visible rather than silent.

    Args:
        root: Root directory containing NDJSON/NDJSON.GZ files.
        glob_pattern: Single glob string or sequence of glob strings.
        out_dir: Output directory for per-resource-type tables.
        specs: Registry mapping FHIR resourceType -> :class:`ResourceSpec`.
        output_format: One of :data:`SUPPORTED_OUTPUT_FORMATS`.
    """
    if output_format not in SUPPORTED_OUTPUT_FORMATS:
        raise ValueError(
            f"Unsupported output_format {output_format!r}; "
            f"expected one of {SUPPORTED_OUTPUT_FORMATS}."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = tables_from_specs(specs)
    columns = columns_from_specs(specs)
    writers = {
        name: _BufferedTableWriter(
            path=out_dir / table_file_name(name, output_format),
            schema=_table_schema(columns[name]),
            output_format=output_format,
        )
        for name in tables
    }

    ingested: Counter = Counter()
    dropped: Counter = Counter()
    skipped_unconfigured: Counter = Counter()
    try:
        for file_path in sorted_ndjson_files(root, glob_pattern):
            for ndjson_obj in iter_ndjson_objects(file_path):
                for resource in iter_resources_from_ndjson_obj(ndjson_obj):
                    resource_type = resource.get("resourceType")
                    result = flatten_resource(resource, specs)
                    if result is None:
                        if resource_type in specs:
                            dropped[resource_type] += 1
                        else:
                            skipped_unconfigured[resource_type] += 1
                        continue
                    table_name, row = result
                    writers[table_name].add(row)
                    ingested[table_name] += 1
    finally:
        for writer in writers.values():
            writer.close()

    logger.info(
        "FHIR flatten complete (%s): %s",
        output_format,
        {name: ingested[name] for name in tables},
    )
    for resource_type, count in dropped.items():
        logger.warning(
            "FHIR flatten: dropped %d %s resource(s) missing a required field "
            "(e.g. patient reference).",
            count,
            resource_type,
        )
    if skipped_unconfigured:
        total = sum(skipped_unconfigured.values())
        logger.info(
            "FHIR flatten: skipped %d resource(s) of %d unconfigured type(s): %s",
            total,
            len(skipped_unconfigured),
            dict(skipped_unconfigured),
        )


def _scan_flat_table(path: Path, output_format: str) -> pl.LazyFrame:
    if output_format == "parquet":
        return pl.scan_parquet(str(path))
    sep = "\t" if output_format == "tsv" else ","
    # infer_schema_length=0 keeps every column as Utf8 (flat tables are all strings).
    return pl.scan_csv(str(path), separator=sep, infer_schema_length=0)


def sorted_patient_ids_from_flat_tables(
    table_dir: Path,
    tables: Sequence[str],
    output_format: str = "parquet",
) -> List[str]:
    """Return sorted unique patient IDs from a directory of flattened tables."""
    patient_path = table_dir / table_file_name("patient", output_format)
    if "patient" in tables and patient_path.exists():
        return (
            _scan_flat_table(patient_path, output_format)
            .select("patient_id")
            .unique()
            .sort("patient_id")
            .collect(engine="streaming")["patient_id"]
            .to_list()
        )
    frames = [
        _scan_flat_table(
            table_dir / table_file_name(t, output_format), output_format
        ).select("patient_id")
        for t in tables
        if t != "patient"
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
    tables: Sequence[str],
    output_format: str = "parquet",
) -> None:
    """Filter all flattened tables to only include rows for the given patient IDs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    keep_set = set(keep_ids)
    for name in tables:
        src = source_dir / table_file_name(name, output_format)
        dst = out_dir / table_file_name(name, output_format)
        lf = _scan_flat_table(src, output_format).filter(
            pl.col("patient_id").is_in(keep_set)
        )
        if output_format == "parquet":
            lf.sink_parquet(str(dst))
        else:
            sep = "\t" if output_format == "tsv" else ","
            lf.sink_csv(str(dst), separator=sep)
