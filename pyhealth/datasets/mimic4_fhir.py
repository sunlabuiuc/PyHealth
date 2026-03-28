"""MIMIC-IV FHIR (NDJSON) ingestion for CEHR-style sequences.

Loads newline-delimited JSON (plain ``*.ndjson`` or gzip ``*.ndjson.gz``, as on
PhysioNet), or Bundle ``entry`` resources, groups by Patient id, and builds
token timelines for MPF / EHRMambaCEHR.

:class:`MIMIC4FHIRDataset` materializes a PyHealth-standard **global event
table** as Parquet (``patient_id``, ``timestamp``, ``event_type``,
``fhir/resource_json``) under the dataset cache. Ingest **hash-partitions** rows
by ``patient_id`` into multiple shard files (bounded memory, no full-table sort);
ingest schedules one process pool task per NDJSON/GZ file (worker count capped by
``os.cpu_count()``). ``global_event_df`` may scan several ``part-*.parquet`` files
like other multi-part caches. Per-patient time order still comes from
:class:`~pyhealth.data.Patient` (``data_source.sort("timestamp")``). The same
``global_event_df`` / :class:`~pyhealth.data.Patient` / :meth:`set_task` path
as CSV-backed datasets applies downstream.

Settings such as ``glob_pattern`` live in ``configs/mimic4_fhir.yaml`` and are
read by :func:`read_fhir_settings_yaml`. For PhysioNet MIMIC-IV on FHIR, set
``root`` to the ``fhir/`` directory that contains ``Mimic*.ndjson.gz`` shards
(e.g. ``MimicPatient.ndjson.gz``, ``MimicEncounter.ndjson.gz``); the default
``glob_pattern`` is ``**/*.ndjson.gz``. For tests, write small
``*.ndjson`` / ``*.ndjson.gz`` files and point ``root`` / ``glob_pattern`` at them.

**JSON / ingest.** NDJSON lines use ``orjson``. Row reading, batching, and Parquet
shard hashing follow a **fixed implementation** in this module (not configurable on
:class:`MIMIC4FHIRDataset`). :data:`FHIR_SCHEMA_VERSION` is part of the dataset cache
fingerprint so ingest or schema changes invalidate cached Parquet.

Plain single-resource lines (not Bundle / ``{"resource":...}`` wrappers) store the
**original line text** in ``fhir/resource_json`` where safe (skipping
``orjson.dumps``). Otherwise the resource is serialized with ``orjson.dumps``.
Column buffers build Arrow tables without per-row ``from_pylist`` dict materialization.
"""

from __future__ import annotations

import concurrent.futures
import gzip
import itertools
import hashlib
import logging
import multiprocessing
import os
import zlib
import platformdirs
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, Iterator, List, Optional, Set, Tuple

import orjson
import polars as pl
from litdata.processing.data_processor import in_notebook
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
from yaml import safe_load

from ..data import Patient
from .base_dataset import BaseDataset

# Normalized event table (BaseDataset / Patient contract)
FHIR_EVENT_TYPE: str = "fhir"
FHIR_RESOURCE_JSON_COL: str = "fhir/resource_json"
FHIR_SCHEMA_VERSION: int = 2

_FHIR_NDJSON_USE_RAW_LINE: bool = True
_FHIR_NDJSON_USE_COLUMN_BUFFERS: bool = True

logger = logging.getLogger(__name__)

DEFAULT_PAD = 0
DEFAULT_UNK = 1


def _fhir_json_loads_ndjson_line(line: str) -> Any:
    return orjson.loads(line.encode("utf-8"))


def _fhir_json_dumps_resource(res: Dict[str, Any]) -> str:
    return orjson.dumps(res).decode("utf-8")


def _fhir_json_loads_resource_column(raw: str | bytes) -> Any:
    b = raw.encode("utf-8") if isinstance(raw, str) else raw
    return orjson.loads(b)


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


def _clinical_concept_key(res: Dict[str, Any]) -> Optional[str]:
    """Resolve a stable vocabulary key; resource-type-specific per FHIR R4."""

    rt = res.get("resourceType")
    if rt == "MedicationRequest":
        mcc = res.get("medicationCodeableConcept")
        if isinstance(mcc, dict):
            ck = _first_coding(mcc)
            if ck:
                return ck
        mref = res.get("medicationReference")
        if isinstance(mref, dict):
            ref = mref.get("reference")
            if ref:
                rid = _ref_id(ref)
                return f"MedicationRequest/reference|{rid or ref}"
        return None
    code = res.get("code")
    if isinstance(code, dict):
        return _first_coding(code)
    return None


@dataclass
class ConceptVocab:
    """Maps FHIR coding keys to dense ids. Supports save/load for streaming builds."""

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
        tid = self._next_id
        self._next_id += 1
        self.token_to_id[key] = tid
        return tid

    def __getitem__(self, key: str) -> int:
        return self.token_to_id.get(key, self.unk_id)

    @property
    def vocab_size(self) -> int:
        return self._next_id

    def to_json(self) -> Dict[str, Any]:
        return {"token_to_id": self.token_to_id, "next_id": self._next_id}

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> ConceptVocab:
        v = cls()
        loaded = dict(data.get("token_to_id") or {})
        if not loaded:
            v._next_id = int(data.get("next_id", 2))
            return v
        v.token_to_id = loaded
        v._next_id = int(data.get("next_id", max(loaded.values()) + 1))
        return v

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(
            orjson.dumps(self.to_json(), option=orjson.OPT_SORT_KEYS)
        )

    @classmethod
    def load(cls, path: str) -> ConceptVocab:
        return cls.from_json(orjson.loads(Path(path).read_bytes()))


def ensure_special_tokens(vocab: ConceptVocab) -> Dict[str, int]:
    """Reserve special tokens for MPF / readout."""

    out: Dict[str, int] = {}
    for name in ("<cls>", "<reg>", "<mor>", "<readm>"):
        out[name] = vocab.add_token(name)
    return out


@dataclass
class FHIRPatient:
    """Minimal patient container for FHIR resources (not pyhealth.data.Patient)."""

    patient_id: str
    resources: List[Dict[str, Any]]
    birth_date: Optional[datetime] = None

    def get_patient_resource(self) -> Optional[Dict[str, Any]]:
        for r in self.resources:
            if r.get("resourceType") == "Patient":
                return r
        return None


def parse_ndjson_line(line: str) -> Any:
    line = line.strip()
    if not line:
        return None
    return _fhir_json_loads_ndjson_line(line)


def iter_ndjson_file(path: Path) -> Generator[Dict[str, Any], None, None]:
    if path.suffix == ".gz":
        opener = gzip.open(path, "rt", encoding="utf-8", errors="replace")
    else:
        opener = open(path, encoding="utf-8", errors="replace")
    with opener as f:
        for line in f:
            obj = parse_ndjson_line(line)
            if obj is not None:
                yield obj


def iter_ndjson_raw_line_and_obj(
    path: Path,
) -> Generator[Tuple[str, Dict[str, Any]], None, None]:
    """Yield ``(line_text, parsed_obj)`` for each non-empty NDJSON object line."""


    if path.suffix == ".gz":
        opener = gzip.open(path, "rt", encoding="utf-8", errors="replace")
    else:
        opener = open(path, encoding="utf-8", errors="replace")
    with opener as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            obj: Any = _fhir_json_loads_ndjson_line(raw)
            if isinstance(obj, dict):
                yield raw, obj


def _ndjson_root_has_top_level_bundle_or_resource_wrapper(obj: Dict[str, Any]) -> bool:
    """True if root object is a Bundle or a ``{"resource": ...}`` wrapper line."""

    return "entry" in obj or "resource" in obj


def _ref_id(ref: Optional[str]) -> Optional[str]:
    if not ref:
        return None
    if "/" in ref:
        return ref.rsplit("/", 1)[-1]
    return ref


def _unwrap_resource_dict(raw: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    r = raw.get("resource") if "resource" in raw else raw
    return r if isinstance(r, dict) else None


def iter_resources_from_ndjson_obj(obj: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """Yield FHIR resource dicts from one parsed NDJSON object.

    Expands ``Bundle`` ``entry`` resources; otherwise yields a single resource.
    """

    if isinstance(obj, dict) and "entry" in obj:
        for ent in obj.get("entry") or []:
            res = ent.get("resource")
            if isinstance(res, dict):
                yield res
    else:
        r = _unwrap_resource_dict(obj)
        if r is not None:
            yield r


def patient_id_for_resource(
    res: Dict[str, Any],
    resource_type: Optional[str] = None,
) -> Optional[str]:
    """Logical patient id for sharding and tabular ``patient_id`` (FHIR subject refs)."""

    rid: Optional[str] = None
    rt = resource_type if resource_type is not None else res.get("resourceType")
    if rt == "Patient":
        pid = res.get("id")
        rid = str(pid) if pid is not None else None
    elif rt == "Encounter":
        rid = _ref_id((res.get("subject") or {}).get("reference"))
    elif rt in ("Condition", "Observation", "MedicationRequest", "Procedure"):
        rid = _ref_id((res.get("subject") or {}).get("reference"))
    return rid


RESOURCE_TYPE_TO_TOKEN_TYPE = {
    "Encounter": 1,
    "Condition": 2,
    "MedicationRequest": 3,
    "Observation": 4,
    "Procedure": 5,
}


def _event_time(
    res: Dict[str, Any],
    resource_type: Optional[str] = None,
) -> Optional[datetime]:
    rt = resource_type if resource_type is not None else res.get("resourceType")
    if rt == "Encounter":
        return _parse_dt((res.get("period") or {}).get("start"))
    if rt == "Condition":
        return _parse_dt(res.get("onsetDateTime") or res.get("recordedDate"))
    if rt == "Observation":
        return _parse_dt(res.get("effectiveDateTime") or res.get("issued"))
    if rt == "MedicationRequest":
        return _parse_dt(res.get("authoredOn"))
    if rt == "Procedure":
        return _parse_dt(res.get("performedDateTime") or res.get("recordedDate"))
    return None


def resource_row_timestamp(
    res: Dict[str, Any],
    resource_type: Optional[str] = None,
) -> Optional[datetime]:
    """Timestamp for ``Patient.data_source`` sort order and Parquet ``timestamp``."""

    t = _event_time(res, resource_type)
    if t is not None:
        return t
    rt = resource_type if resource_type is not None else res.get("resourceType")
    if rt == "Patient":
        return _parse_dt(res.get("birthDate"))
    return None


def _sequential_visit_idx_for_time(
    t: Optional[datetime], visit_encounters: List[Tuple[datetime, int]]
) -> int:
    """Map event time to the sequential ``visit_idx`` used in the main encounter loop.

    ``visit_encounters`` lists ``(encounter_start, visit_idx)`` only for encounters
    with a valid ``period.start``, in the same order as :func:`build_cehr_sequences`
    assigns ``visit_idx`` (sorted ``encounters``, skipping those without start). This
    must not use raw indices into the full ``encounters`` list, or indices diverge
    when some encounters lack a start time.
    """

    if not visit_encounters:
        return 0
    if t is None:
        return visit_encounters[-1][1]
    t = _as_naive(t)
    chosen = visit_encounters[0][1]
    for es, vidx in visit_encounters:
        if es <= t:
            chosen = vidx
        else:
            break
    return chosen


def _concept_vocab_key_for_cehr(res: Dict[str, Any]) -> str:
    """Dense-vocabulary string for one resource (aligned with MPF CEHR tokens)."""

    rt = res.get("resourceType")
    ck = _clinical_concept_key(res)
    if rt == "Observation":
        ck = ck or "obs|unknown"
    if ck is None:
        ck = f"{(rt or 'res').lower()}|unknown"
    return ck


def collect_cehr_timeline_events(
    patient: FHIRPatient,
) -> List[Tuple[datetime, Dict[str, Any], int]]:
    """Clinical events in CEHR timeline order (time, resource, visit_idx).

    Matches encounter grouping and visit indexing used by
    :func:`build_cehr_sequences` / MPF so vocabulary warmup stays consistent
    without materializing per-token feature lists.
    """

    events: List[Tuple[datetime, Dict[str, Any], int]] = []
    encounters = [r for r in patient.resources if r.get("resourceType") == "Encounter"]
    encounters.sort(key=lambda e: _event_time(e) or datetime.min)

    visit_encounters: List[Tuple[datetime, int]] = []
    _v = 0
    for enc in encounters:
        _es = _event_time(enc)
        if _es is None:
            continue
        visit_encounters.append((_as_naive(_es), _v))
        _v += 1

    visit_idx = 0
    for enc in encounters:
        eid = enc.get("id")
        enc_start = _event_time(enc)
        if enc_start is None:
            continue
        for r in patient.resources:
            if r.get("resourceType") == "Patient":
                continue
            rt = r.get("resourceType")
            if rt not in RESOURCE_TYPE_TO_TOKEN_TYPE:
                continue
            if rt == "Encounter" and r.get("id") != eid:
                continue
            if rt != "Encounter":
                enc_ref = (r.get("encounter") or {}).get("reference")
                if enc_ref:
                    ref_eid = _ref_id(enc_ref)
                    if ref_eid is None or str(eid) != str(ref_eid):
                        continue
                else:
                    continue
            t = _event_time(r)
            if t is None:
                t = enc_start
            events.append((t, r, visit_idx))
        visit_idx += 1

    for r in patient.resources:
        if r.get("resourceType") == "Patient":
            continue
        rt = r.get("resourceType")
        if rt not in RESOURCE_TYPE_TO_TOKEN_TYPE:
            continue
        if rt == "Encounter":
            continue
        enc_ref = (r.get("encounter") or {}).get("reference")
        if enc_ref:
            continue
        t_evt = _event_time(r)
        v_idx = _sequential_visit_idx_for_time(t_evt, visit_encounters)
        t = t_evt
        if t is None:
            if visit_encounters:
                for es, v in visit_encounters:
                    if v == v_idx:
                        t = es
                        break
                else:
                    t = visit_encounters[-1][0]
            if t is None:
                continue
        events.append((t, r, v_idx))

    events.sort(key=lambda x: x[0])
    return events


def warm_mpf_vocab_from_fhir_patient(
    vocab: ConceptVocab,
    patient: FHIRPatient,
    clinical_cap: int,
) -> None:
    """Register concept keys for the tail clinical window (MPF parallel path).

    Lighter than :func:`build_cehr_sequences`: no per-position feature lists.
    """

    tail = (
        collect_cehr_timeline_events(patient)[-clinical_cap:]
        if clinical_cap > 0
        else []
    )
    for _, res, _ in tail:
        vocab.add_token(_concept_vocab_key_for_cehr(res))


def build_cehr_sequences(
    patient: FHIRPatient,
    vocab: ConceptVocab,
    max_len: int,
    *,
    base_time: Optional[datetime] = None,
    grow_vocab: bool = True,
) -> Tuple[
    List[int],
    List[int],
    List[float],
    List[float],
    List[int],
    List[int],
]:
    """Flatten patient resources into CEHR-aligned lists (pre-padding).

    Args:
        max_len: Maximum number of **clinical** tokens emitted (after time sort and
            tail slice). Use ``0`` to emit no clinical tokens (empty lists; avoids
            Python's ``events[-0:]`` which would incorrectly take the full timeline).
            Downstream MPF tasks reserve two slots for ``<mor>``/``<cls>`` and
            ``<reg>``, so pass ``max_len - 2`` there when the final tensor length
            is fixed.
        grow_vocab: If True (default), assign new dense ids via ``add_token``. If
            False, use only existing ids (``<unk>`` for unknown codes)—for parallel
            ``set_task`` workers after a main-process vocabulary warmup.
    """

    birth = patient.birth_date
    if birth is None:
        pr = patient.get_patient_resource()
        if pr:
            birth = _parse_dt(pr.get("birthDate"))

    events = collect_cehr_timeline_events(patient)

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
    for t, res, v_idx in tail:
        t = _as_naive(t)
        rt = res.get("resourceType")
        ck = _concept_vocab_key_for_cehr(res)
        if grow_vocab:
            cid = vocab.add_token(ck)
        else:
            cid = vocab[ck]
        tt = RESOURCE_TYPE_TO_TOKEN_TYPE.get(rt, 0)
        ts = float((t - base_time).total_seconds()) if base_time and t else 0.0
        age_y = 0.0
        if birth and t:
            age_y = (t - birth).days / 365.25
        seg = v_idx % 2
        concept_ids.append(cid)
        token_types.append(tt)
        time_stamps.append(ts)
        ages.append(age_y)
        visit_orders.append(min(v_idx, 511))
        visit_segments.append(seg)

    return concept_ids, token_types, time_stamps, ages, visit_orders, visit_segments


def fhir_patient_from_patient(patient: Patient) -> FHIRPatient:
    """Rebuild :class:`FHIRPatient` from a tabular :class:`~pyhealth.data.Patient`."""

    resources: List[Dict[str, Any]] = []
    for row in patient.data_source.iter_rows(named=True):
        raw = row.get(FHIR_RESOURCE_JSON_COL)
        if not raw:
            continue
        resources.append(_fhir_json_loads_resource_column(raw))
    birth: Optional[datetime] = None
    for r in resources:
        if r.get("resourceType") == "Patient":
            birth = _parse_dt(r.get("birthDate"))
            break
    return FHIRPatient(
        patient_id=patient.patient_id, resources=resources, birth_date=birth
    )


def infer_mortality_label(patient: FHIRPatient) -> int:
    """Heuristic binary label: 1 if deceased or explicit death condition."""

    pr = patient.get_patient_resource()
    if pr and pr.get("deceasedBoolean") is True:
        return 1
    if pr and pr.get("deceasedDateTime"):
        return 1
    for r in patient.resources:
        if r.get("resourceType") != "Condition":
            continue
        ck = (_first_coding(r.get("code") or {}) or "").lower()
        if any(x in ck for x in ("death", "deceased", "mortality")):
            return 1
    return 0


def synthetic_mpf_one_patient_resources() -> List[Dict[str, Any]]:
    """Minimal FHIR NDJSON rows for one patient (tests, quick-start examples)."""

    patient: Dict[str, Any] = {
        "resourceType": "Patient",
        "id": "p-synth-1",
        "birthDate": "1950-01-01",
        "gender": "female",
    }
    enc: Dict[str, Any] = {
        "resourceType": "Encounter",
        "id": "e1",
        "subject": {"reference": "Patient/p-synth-1"},
        "period": {"start": "2020-06-01T10:00:00Z"},
        "class": {"code": "IMP"},
    }
    cond: Dict[str, Any] = {
        "resourceType": "Condition",
        "id": "c1",
        "subject": {"reference": "Patient/p-synth-1"},
        "encounter": {"reference": "Encounter/e1"},
        "code": {
            "coding": [{"system": "http://hl7.org/fhir/sid/icd-10-cm", "code": "I10"}]
        },
        "onsetDateTime": "2020-06-01T11:00:00Z",
    }
    return [patient, enc, cond]


def synthetic_mpf_two_patient_resources() -> List[Dict[str, Any]]:
    """Two-patient fixture including a deceased patient (binary label smoke tests)."""

    dead_p: Dict[str, Any] = {
        "resourceType": "Patient",
        "id": "p-synth-2",
        "birthDate": "1940-05-05",
        "deceasedBoolean": True,
    }
    dead_enc: Dict[str, Any] = {
        "resourceType": "Encounter",
        "id": "e-dead",
        "subject": {"reference": "Patient/p-synth-2"},
        "period": {"start": "2020-07-01T10:00:00Z"},
        "class": {"code": "IMP"},
    }
    dead_obs: Dict[str, Any] = {
        "resourceType": "Observation",
        "id": "o-dead",
        "subject": {"reference": "Patient/p-synth-2"},
        "encounter": {"reference": "Encounter/e-dead"},
        "effectiveDateTime": "2020-07-01T12:00:00Z",
        "code": {"coding": [{"system": "http://loinc.org", "code": "789-0"}]},
    }
    return [*synthetic_mpf_one_patient_resources(), dead_p, dead_enc, dead_obs]


def synthetic_mpf_one_patient_ndjson_text() -> str:
    return (
        "\n".join(
            orjson.dumps(r).decode("utf-8") for r in synthetic_mpf_one_patient_resources()
        )
        + "\n"
    )


def synthetic_mpf_two_patient_ndjson_text() -> str:
    return (
        "\n".join(
            orjson.dumps(r).decode("utf-8")
            for r in synthetic_mpf_two_patient_resources()
        )
        + "\n"
    )


def read_fhir_settings_yaml(path: Optional[str] = None) -> Dict[str, Any]:
    """Load FHIR YAML (glob pattern, version); not a CSV ``DatasetConfig`` schema.

    Args:
        path: Defaults to ``configs/mimic4_fhir.yaml`` beside this module.

    Returns:
        Parsed mapping.
    """
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "configs", "mimic4_fhir.yaml")
    with open(path, encoding="utf-8") as f:
        data = safe_load(f)
    return data if isinstance(data, dict) else {}


def fhir_events_arrow_schema() -> pa.Schema:
    """Arrow schema for normalized FHIR event rows."""

    return pa.schema(
        [
            ("patient_id", pa.string()),
            ("event_type", pa.string()),
            ("timestamp", pa.timestamp("ms")),
            (FHIR_RESOURCE_JSON_COL, pa.string()),
        ]
    )


def _crc32_shard_index(key: str, num_shards: int) -> int:
    """Stable shard index in ``[0, num_shards)`` (portable ``crc32``)."""

    u = zlib.crc32(key.encode("utf-8")) & 0xFFFFFFFF
    return int(u % max(1, num_shards))


def _normalize_fhir_chunk_work(args: Any) -> Dict[str, Any]:
    """Pickle-friendly fields for :func:`_process_fhir_file_chunk`."""

    if isinstance(args, dict):
        w = {**args}
        w["file_path"] = Path(w["file_path"])
        w["out_dir"] = Path(w["out_dir"])
    elif len(args) == 5:
        file_idx, file_path, out_dir, num_shards, batch_size = args  # type: ignore[misc]
        w = {
            "file_idx": file_idx,
            "file_path": Path(file_path),
            "out_dir": Path(out_dir),
            "num_shards": num_shards,
            "batch_size": batch_size,
        }
    else:
        raise TypeError(
            "expected a dict or 5-tuple "
            "(file_idx, file_path, out_dir, num_shards, batch_size) for "
            f"_process_fhir_file_chunk, got {type(args)!r}"
        )
    w["ingest_use_raw_ndjson_line"] = _FHIR_NDJSON_USE_RAW_LINE
    w["ingest_use_column_buffers"] = _FHIR_NDJSON_USE_COLUMN_BUFFERS
    return w


def _process_fhir_file_chunk(
    args: Any,
) -> int:
    """Read one NDJSON/NDJSON.GZ file and write hash-sharded Parquet rows.

    Output files are ``shard-{file_idx:04d}-{shard_idx:04d}.parquet`` (``file_idx`` is
    the file's index in the sorted ingest list) so tasks never share a path. Multiple
    flushes for the same shard use one :class:`~pyarrow.parquet.ParquetWriter`.

    A single compressed ``.ndjson.gz`` is still consumed sequentially in one process
    (standard gzip stream); scheduling **one process per input file** lets large
    files (e.g. Chartevents vs Labevents) run on different CPUs instead of being
    stuck together in a static path batch.

    Args:
        args: A ``dict`` with ``file_idx``, ``file_path``, ``out_dir``,
            ``num_shards``, and ``batch_size``, or a 5-tuple of those values in that
            order. Ingest behavior matches :data:`FHIR_SCHEMA_VERSION`.

    Returns:
        Row count (FHIR resources with a resolvable ``patient_id``) for this file.
    """

    wk = _normalize_fhir_chunk_work(args)
    file_idx = int(wk["file_idx"])
    fp = wk["file_path"]
    out_dir = wk["out_dir"]
    num_shards = max(1, int(wk["num_shards"]))
    batch_size = int(wk["batch_size"])
    ingest_use_raw_ndjson_line = bool(wk["ingest_use_raw_ndjson_line"])
    ingest_use_column_buffers = bool(wk["ingest_use_column_buffers"])

    schema = fhir_events_arrow_schema()
    out_dir.mkdir(parents=True, exist_ok=True)
    writers: List[Optional[pq.ParquetWriter]] = [None] * num_shards
    n_rows = 0

    if ingest_use_column_buffers:
        buf_pid: List[List[str]] = [[] for _ in range(num_shards)]
        buf_et: List[List[str]] = [[] for _ in range(num_shards)]
        buf_ts: List[List[Optional[datetime]]] = [[] for _ in range(num_shards)]
        buf_rj: List[List[str]] = [[] for _ in range(num_shards)]
    else:
        batches: List[List[Dict[str, Any]]] = [[] for _ in range(num_shards)]

    def flush(shard: int) -> None:
        nonlocal n_rows
        if ingest_use_column_buffers:
            if not buf_pid[shard]:
                return
            count = len(buf_pid[shard])
            table = pa.table(
                {
                    "patient_id": buf_pid[shard],
                    "event_type": buf_et[shard],
                    "timestamp": buf_ts[shard],
                    FHIR_RESOURCE_JSON_COL: buf_rj[shard],
                },
                schema=schema,
            )
            buf_pid[shard].clear()
            buf_et[shard].clear()
            buf_ts[shard].clear()
            buf_rj[shard].clear()
        else:
            if not batches[shard]:
                return
            table = pa.Table.from_pylist(batches[shard], schema=schema)
            count = len(batches[shard])
            batches[shard].clear()
        if writers[shard] is None:
            writers[shard] = pq.ParquetWriter(
                str(out_dir / f"shard-{file_idx:04d}-{shard:04d}.parquet"),
                schema,
            )
        writers[shard].write_table(table)
        n_rows += count

    def append_row(
        shard: int, pid: str, ts: Optional[datetime], resource_json: str
    ) -> None:
        if ingest_use_column_buffers:
            buf_pid[shard].append(pid)
            buf_et[shard].append(FHIR_EVENT_TYPE)
            buf_ts[shard].append(ts)
            buf_rj[shard].append(resource_json)
        else:
            batches[shard].append(
                {
                    "patient_id": pid,
                    "event_type": FHIR_EVENT_TYPE,
                    "timestamp": ts,
                    FHIR_RESOURCE_JSON_COL: resource_json,
                }
            )

    if fp.is_file():
        for raw_line, obj in iter_ndjson_raw_line_and_obj(fp):
            use_raw_for_line = ingest_use_raw_ndjson_line and (
                not _ndjson_root_has_top_level_bundle_or_resource_wrapper(obj)
            )
            for res in iter_resources_from_ndjson_obj(obj):
                rt = res.get("resourceType")
                pid = patient_id_for_resource(res, rt)
                if not pid:
                    continue
                ts = resource_row_timestamp(res, rt)
                resource_json = (
                    raw_line if use_raw_for_line else _fhir_json_dumps_resource(res)
                )
                shard = _crc32_shard_index(pid, num_shards)
                append_row(shard, pid, ts, resource_json)
                cur_len = (
                    len(buf_pid[shard])
                    if ingest_use_column_buffers
                    else len(batches[shard])
                )
                if cur_len >= batch_size:
                    flush(shard)

    for s in range(num_shards):
        flush(s)
    for s in range(num_shards):
        if writers[s] is not None:
            writers[s].close()

    return n_rows


def stream_fhir_ndjson_root_to_sharded_parquet(
    root: Path,
    glob_pattern: str,
    out_dir: Path,
    *,
    num_shards: int = 16,
    batch_size: int = 50_000,
) -> int:
    """Stream matching NDJSON / NDJSON.GZ files into hash-sharded Parquet files.

    Files under ``root`` matching ``glob_pattern`` are read in parallel: **one
    process pool task per file**, up to ``min(os.cpu_count(), N)`` workers, so the
    pool load-balances across uneven file sizes (MIMIC-IV FHIR has a few very
    large ``*.ndjson.gz`` shards). Each task writes
    ``shard-{file_index}-{hash_bucket}.parquet``; the downstream cache globs
    ``shard-*.parquet`` and scans them with Polars.

    All rows for a given ``patient_id`` share one hash bucket (same ``num_shards``);
    output paths are disjoint per input file. Shards with no rows for a file
    produce no file. If no input files match, or all rows lack a ``patient_id``,
    writes a single empty ``shard-0000.parquet``.

    Returns:
        Number of rows written (FHIR resources with a resolvable ``patient_id``).
    """

    schema = fhir_events_arrow_schema()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    num_shards = max(1, int(num_shards))

    all_files = sorted(
        p for p in root.glob(glob_pattern) if p.is_file()
    )
    if not all_files:
        pq.write_table(
            pa.Table.from_pylist([], schema=schema),
            str(out_dir / "shard-0000.parquet"),
        )
        return 0

    cpu = os.cpu_count() or 1
    max_workers = min(cpu, len(all_files))
    work_args = [
        {
            "file_idx": i,
            "file_path": all_files[i],
            "out_dir": out_dir,
            "num_shards": num_shards,
            "batch_size": batch_size,
        }
        for i in range(len(all_files))
    ]

    if len(work_args) == 1:
        n_rows = _process_fhir_file_chunk(work_args[0])  # type: ignore[arg-type]
    else:
        # ``spawn`` matches :meth:`BaseDataset._task_transform` — avoid ``fork`` with
        # a Polars-import-loaded parent (see Polars multiprocessing docs).
        ctx = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=ctx,
        ) as executor:
            counts = list(
                tqdm(
                    executor.map(_process_fhir_file_chunk, work_args),
                    total=len(work_args),
                    desc="FHIR NDJSON ingest",
                    unit="file",
                )
            )
        n_rows = sum(counts)

    if n_rows == 0:
        pq.write_table(
            pa.Table.from_pylist([], schema=schema),
            str(out_dir / "shard-0000.parquet"),
        )
    return n_rows


class MIMIC4FHIRDataset(BaseDataset):
    """MIMIC-IV on FHIR (NDJSON / NDJSON.GZ / Bundle) with PyHealth's tabular cache.

    Streams resources to ``global_event_df`` Parquet
    (``patient_id``, ``timestamp``, ``event_type``, ``fhir/resource_json``), then
    uses :class:`~pyhealth.data.Patient` and standard :meth:`set_task` like other
    datasets. MPF uses :class:`~pyhealth.tasks.mpf_clinical_prediction.MPFClinicalPredictionTask`.

    YAML defaults (e.g. ``glob_pattern``) live in
    ``pyhealth/datasets/configs/mimic4_fhir.yaml``. NDJSON→Parquet ingest is
    implemented in this module and is not tunable via the constructor.

    Args:
        root: Root directory scanned for NDJSON/NDJSON.GZ (PhysioNet: the ``fhir/``
            folder with ``Mimic*.ndjson.gz`` files).
        config_path: Optional path to the FHIR YAML settings file.
        glob_pattern: If set, overrides the YAML ``glob_pattern`` (default
            ``**/*.ndjson.gz`` for credentialled exports).
        max_patients: After streaming, keep only the first N patient ids (sorted).
        ingest_num_shards: Number of hash shards for the NDJSON→Parquet pass;
            defaults from YAML ``ingest_num_shards`` or CPU-based heuristics.
        vocab_path: Optional path to a saved :class:`ConceptVocab` JSON.
        cache_dir: Forwarded to :class:`~pyhealth.datasets.BaseDataset`.
        num_workers: Forwarded to :class:`~pyhealth.datasets.BaseDataset`.
        dev: If True and ``max_patients`` is None, caps at 1000 patients.

    Example:
        >>> from pyhealth.datasets import MIMIC4FHIRDataset
        >>> from pyhealth.tasks.mpf_clinical_prediction import (
        ...     MPFClinicalPredictionTask,
        ... )
        >>> ds = MIMIC4FHIRDataset(root="/path/to/fhir", max_patients=50)
        >>> task = MPFClinicalPredictionTask(max_len=256)
        >>> sample_ds = ds.set_task(task)  # doctest: +SKIP
    """

    def __init__(
        self,
        root: str,
        config_path: Optional[str] = None,
        glob_pattern: Optional[str] = None,
        max_patients: Optional[int] = None,
        ingest_num_shards: Optional[int] = None,
        vocab_path: Optional[str] = None,
        cache_dir: Optional[str | Path] = None,
        num_workers: int = 1,
        dev: bool = False,
    ) -> None:
        default_cfg = os.path.join(
            os.path.dirname(__file__), "configs", "mimic4_fhir.yaml"
        )
        self._fhir_config_path = str(Path(config_path or default_cfg).resolve())
        self._fhir_settings = read_fhir_settings_yaml(self._fhir_config_path)
        self.glob_pattern = (
            glob_pattern
            if glob_pattern is not None
            else str(self._fhir_settings.get("glob_pattern", "**/*.ndjson.gz"))
        )
        mp = max_patients
        if dev and mp is None:
            mp = 1000
        self.max_patients = mp
        if ingest_num_shards is not None:
            self.ingest_num_shards = max(1, int(ingest_num_shards))
        else:
            raw_shards = self._fhir_settings.get("ingest_num_shards")
            if raw_shards is not None:
                self.ingest_num_shards = max(1, int(raw_shards))
            else:
                self.ingest_num_shards = max(4, min(32, (os.cpu_count() or 4) * 2))
        if vocab_path and os.path.isfile(vocab_path):
            self.vocab = ConceptVocab.load(vocab_path)
        else:
            self.vocab = ConceptVocab()
        super().__init__(
            root=root,
            tables=["fhir_ndjson"],
            dataset_name="mimic4_fhir",
            config_path=None,
            cache_dir=cache_dir,
            num_workers=num_workers,
            dev=dev,
        )

    def _init_cache_dir(self, cache_dir: str | Path | None) -> Path:
        try:
            y_digest = hashlib.sha256(
                Path(self._fhir_config_path).read_bytes()
            ).hexdigest()[:16]
        except OSError:
            y_digest = "missing"
        id_str = orjson.dumps(
            {
                "root": str(self.root),
                "tables": sorted(self.tables),
                "dataset_name": self.dataset_name,
                "dev": self.dev,
                "glob_pattern": self.glob_pattern,
                "max_patients": self.max_patients,
                "ingest_num_shards": self.ingest_num_shards,
                "fhir_schema_version": FHIR_SCHEMA_VERSION,
                "fhir_yaml_digest16": y_digest,
            },
            option=orjson.OPT_SORT_KEYS,
        ).decode("utf-8")
        cid = str(uuid.uuid5(uuid.NAMESPACE_DNS, id_str))
        if cache_dir is None:
            out = Path(platformdirs.user_cache_dir(appname="pyhealth")) / cid
            out.mkdir(parents=True, exist_ok=True)
            logger.info("No cache_dir provided. Using default cache dir: %s", out)
        else:
            out = Path(cache_dir) / cid
            out.mkdir(parents=True, exist_ok=True)
            logger.info("Using provided cache_dir: %s", out)
        return out

    def _event_transform(self, output_dir: Path) -> None:
        root = Path(self.root).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"MIMIC4 FHIR root not found: {root}")
        try:
            staging = self.create_tmpdir() / "fhir_event_shards"
            staging.mkdir(parents=True, exist_ok=True)
            stream_fhir_ndjson_root_to_sharded_parquet(
                root,
                self.glob_pattern,
                staging,
                num_shards=self.ingest_num_shards,
                batch_size=50_000,
            )
            staged_files = sorted(staging.glob("shard-*.parquet"))
            if output_dir.exists():
                shutil.rmtree(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            keep: Optional[Set[str]] = None
            if self.max_patients is not None:
                lf_all = pl.concat(
                    [pl.scan_parquet(str(p)) for p in staged_files]
                )
                pids = (
                    lf_all.select("patient_id")
                    .unique()
                    .sort("patient_id")
                    .collect(engine="streaming")["patient_id"]
                    .to_list()
                )
                keep = set(pids[: self.max_patients])

            if keep is None:
                for i, p in enumerate(staged_files):
                    shutil.move(str(p), str(output_dir / f"part-{i:05d}.parquet"))
            else:
                for i, p in enumerate(staged_files):
                    pl.scan_parquet(str(p)).filter(
                        pl.col("patient_id").is_in(keep)
                    ).sink_parquet(str(output_dir / f"part-{i:05d}.parquet"))
        except Exception as e:
            if output_dir.exists():
                logger.error(
                    "Error during FHIR event caching, removing incomplete dir %s",
                    output_dir,
                )
                shutil.rmtree(output_dir)
            raise e
        finally:
            self.clean_tmpdir()

    @property
    def unique_patient_ids(self) -> List[str]:
        """Sorted unique patient ids (stable across multi-part Parquet caches)."""

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

    def iter_patients(self, df: Optional[pl.LazyFrame] = None) -> Iterator[Patient]:
        if df is not None:
            yield from super().iter_patients(df)
            return
        base = self.global_event_df
        for patient_id in self.unique_patient_ids:
            patient_df = base.filter(pl.col("patient_id") == patient_id).collect(
                engine="streaming"
            )
            yield Patient(patient_id=patient_id, data_source=patient_df)

    def stats(self) -> None:
        super().stats()

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
            nw = (
                1
                if in_notebook()
                else (num_workers if num_workers is not None else self.num_workers)
            )
            # Match :meth:`BaseDataset._task_transform`: unique ids after ``pre_filter``,
            # not the full cached cohort (e.g. task subclasses that cap patients).
            warmup_pids = self._mpf_patient_ids_for_task(task)
            pid_n = len(warmup_pids)
            effective_workers = min(nw, pid_n) if pid_n else 1
            ensure_special_tokens(self.vocab)
            # Always warm in the main process: single-worker ``set_task`` can skip
            # ``BaseDataset._task_transform`` on LitData cache hit, leaving a fresh
            # vocab at specials-only without this pass.
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
        """Sorted unique patient ids that ``task`` will see (same as ``_task_transform``)."""

        lf_filtered = task.pre_filter(self.global_event_df)
        return (
            lf_filtered.select("patient_id")
            .unique()
            .collect(engine="streaming")
            .to_series()
            .sort()
            .to_list()
        )

    def _warm_mpf_vocabulary(self, task: Any, patient_ids: List[str]) -> None:
        """Main-process vocab keys only (parallel ``set_task`` workers use frozen vocab).

        Args:
            task: MPF task (uses ``max_len`` for clinical token cap).
            patient_ids: Cohort to warm (post-``pre_filter``), not necessarily
                :attr:`unique_patient_ids`.
        """

        clinical_cap = max(0, task.max_len - 2)
        # Same batching as :func:`_task_transform_fn` — one collect per batch, not
        # one full scan per patient.
        batch_size = 128
        base = self.global_event_df
        for batch in itertools.batched(patient_ids, batch_size):
            patients = (
                base.filter(pl.col("patient_id").is_in(batch))
                .collect(engine="streaming")
                .partition_by("patient_id", as_dict=True)
            )
            for patient_key, patient_df in patients.items():
                patient_id = patient_key[0]
                py_patient = Patient(patient_id=patient_id, data_source=patient_df)
                fp = fhir_patient_from_patient(py_patient)
                warm_mpf_vocab_from_fhir_patient(self.vocab, fp, clinical_cap)

    def gather_samples(self, task: Any) -> List[Dict[str, Any]]:
        """Run ``task`` on each :class:`~pyhealth.data.Patient` (tabular path)."""

        task.vocab = self.vocab
        task._specials = None
        task.frozen_vocab = False
        samples: List[Dict[str, Any]] = []
        for p in self.iter_patients():
            samples.extend(task(p))
        return samples
