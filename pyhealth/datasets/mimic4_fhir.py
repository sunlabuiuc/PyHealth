"""MIMIC-IV FHIR (NDJSON) ingestion for CEHR-style sequences.

Loads NDJSON lines (one JSON object per line) or Bundle ``entry`` resources,
groups by Patient id, and builds token timelines for MPF / EHRMambaCEHR.

Use ``MIMIC4_FHIR_ROOT`` or pass ``root`` explicitly. Unit tests use
:func:`synthetic_ndjson_lines` only.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Sequence, Tuple

from yaml import safe_load

logger = logging.getLogger(__name__)

DEFAULT_PAD = 0
DEFAULT_UNK = 1


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
        v.token_to_id = dict(data["token_to_id"])
        v._next_id = int(data.get("next_id", max(v.token_to_id.values()) + 1))
        return v

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_json(), f)

    @classmethod
    def load(cls, path: str) -> ConceptVocab:
        with open(path, encoding="utf-8") as f:
            return cls.from_json(json.load(f))


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


def parse_ndjson_line(line: str) -> Optional[Dict[str, Any]]:
    line = line.strip()
    if not line:
        return None
    return json.loads(line)


def iter_ndjson_file(path: Path) -> Generator[Dict[str, Any], None, None]:
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            obj = parse_ndjson_line(line)
            if obj is not None:
                yield obj


def _ref_id(ref: Optional[str]) -> Optional[str]:
    if not ref:
        return None
    if "/" in ref:
        return ref.rsplit("/", 1)[-1]
    return ref


def group_resources_by_patient(
    resources: Sequence[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    by_patient: Dict[str, List[Dict[str, Any]]] = {}
    for raw in resources:
        r = raw.get("resource") if "resource" in raw else raw
        if not isinstance(r, dict):
            continue
        rid: Optional[str] = None
        rt = r.get("resourceType")
        if rt == "Patient":
            rid = r.get("id")
        elif rt == "Encounter":
            rid = _ref_id((r.get("subject") or {}).get("reference"))
        elif rt in ("Condition", "Observation", "MedicationRequest", "Procedure"):
            rid = _ref_id((r.get("subject") or {}).get("reference"))
        if rid:
            by_patient.setdefault(rid, []).append(r)
    return by_patient


RESOURCE_TYPE_TO_TOKEN_TYPE = {
    "Encounter": 1,
    "Condition": 2,
    "MedicationRequest": 3,
    "Observation": 4,
    "Procedure": 5,
}


def _event_time(res: Dict[str, Any]) -> Optional[datetime]:
    rt = res.get("resourceType")
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


def build_cehr_sequences(
    patient: FHIRPatient,
    vocab: ConceptVocab,
    max_len: int,
    *,
    base_time: Optional[datetime] = None,
) -> Tuple[
    List[int],
    List[int],
    List[float],
    List[float],
    List[int],
    List[int],
]:
    """Flatten patient resources into CEHR-aligned lists (pre-padding)."""

    birth = patient.birth_date
    if birth is None:
        pr = patient.get_patient_resource()
        if pr:
            birth = _parse_dt(pr.get("birthDate"))

    events: List[Tuple[datetime, Dict[str, Any], int]] = []
    encounters = [r for r in patient.resources if r.get("resourceType") == "Encounter"]
    encounters.sort(key=lambda e: _event_time(e) or datetime.min)

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
                if enc_ref and eid and eid not in enc_ref:
                    continue
            t = _event_time(r)
            if t is None:
                t = enc_start
            events.append((t, r, visit_idx))
        visit_idx += 1

    events.sort(key=lambda x: x[0])

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

    prev_visit = -1
    base_time = _as_naive(base_time)
    birth = _as_naive(birth)
    for t, res, v_idx in events[-max_len:]:
        t = _as_naive(t)
        rt = res.get("resourceType")
        code_obj = res.get("code") or {}
        ck = _first_coding(code_obj)
        if rt == "Observation":
            ck = ck or "obs|unknown"
        if ck is None:
            ck = f"{(rt or 'res').lower()}|unknown"
        cid = vocab.add_token(ck)
        tt = RESOURCE_TYPE_TO_TOKEN_TYPE.get(rt, 0)
        ts = float((t - base_time).total_seconds()) if base_time and t else 0.0
        age_y = 0.0
        if birth and t:
            age_y = (t - birth).days / 365.25
        seg = 0 if v_idx != prev_visit else 1
        prev_visit = v_idx
        concept_ids.append(cid)
        token_types.append(tt)
        time_stamps.append(ts)
        ages.append(age_y)
        visit_orders.append(min(v_idx, 511))
        visit_segments.append(seg)

    return concept_ids, token_types, time_stamps, ages, visit_orders, visit_segments


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


def load_yaml_config(path: Optional[str]) -> Dict[str, Any]:
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "configs", "mimic4_fhir.yaml")
    with open(path, encoding="utf-8") as f:
        data = safe_load(f)
    return data if isinstance(data, dict) else {}


def collect_resources_from_root(root: Path, glob_pattern: str) -> List[Dict[str, Any]]:
    """Read all NDJSON / Bundle lines under root."""

    all_res: List[Dict[str, Any]] = []
    for fp in sorted(root.glob(glob_pattern)):
        if not fp.is_file():
            continue
        for obj in iter_ndjson_file(fp):
            if isinstance(obj, dict) and "entry" in obj:
                for ent in obj.get("entry") or []:
                    res = ent.get("resource")
                    if isinstance(res, dict):
                        all_res.append(res)
            else:
                all_res.append(obj)
    return all_res


@dataclass
class MIMIC4FHIRDataset:
    """FHIR NDJSON dataset: scan disk, group by patient, expose :meth:`gather_samples`."""

    root: str
    config_path: Optional[str] = None
    glob_pattern: str = "**/*.ndjson"
    max_patients: Optional[int] = None
    vocab_path: Optional[str] = None

    def __post_init__(self) -> None:
        self._cfg = load_yaml_config(self.config_path)
        self.glob_pattern = self._cfg.get("glob_pattern", self.glob_pattern)
        if self.vocab_path and os.path.isfile(self.vocab_path):
            self.vocab = ConceptVocab.load(self.vocab_path)
        else:
            self.vocab = ConceptVocab()
        self._patients: Optional[List[FHIRPatient]] = None

    def load_patients(self) -> List[FHIRPatient]:
        if self._patients is not None:
            return self._patients
        root = Path(self.root).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"MIMIC4 FHIR root not found: {root}")
        all_res = collect_resources_from_root(root, self.glob_pattern)
        grouped = group_resources_by_patient(all_res)
        patients: List[FHIRPatient] = []
        for pid, res_list in sorted(grouped.items()):
            pr = FHIRPatient(patient_id=pid, resources=res_list)
            for r in res_list:
                if r.get("resourceType") == "Patient":
                    pr.birth_date = _parse_dt(r.get("birthDate"))
                    break
            patients.append(pr)
            if self.max_patients is not None and len(patients) >= self.max_patients:
                break
        self._patients = patients
        logger.info("Loaded %d FHIR patients from %s", len(patients), root)
        return patients

    def gather_samples(self, task: Any) -> List[Dict[str, Any]]:
        """Apply an MPF clinical task; task receives :class:`FHIRPatient`."""

        task.vocab = self.vocab
        task._specials = None
        patients = self.load_patients()
        samples: List[Dict[str, Any]] = []
        for p in patients:
            samples.extend(task(p))
        return samples


def synthetic_ndjson_lines() -> List[str]:
    """Minimal synthetic FHIR lines for unit tests (no PHI)."""

    patient = {
        "resourceType": "Patient",
        "id": "p-synth-1",
        "birthDate": "1950-01-01",
        "gender": "female",
    }
    enc = {
        "resourceType": "Encounter",
        "id": "e1",
        "subject": {"reference": "Patient/p-synth-1"},
        "period": {"start": "2020-06-01T10:00:00Z"},
        "class": {"code": "IMP"},
    }
    cond = {
        "resourceType": "Condition",
        "id": "c1",
        "subject": {"reference": "Patient/p-synth-1"},
        "encounter": {"reference": "Encounter/e1"},
        "code": {
            "coding": [{"system": "http://hl7.org/fhir/sid/icd-10-cm", "code": "I10"}]
        },
        "onsetDateTime": "2020-06-01T11:00:00Z",
    }
    return [json.dumps(patient), json.dumps(enc), json.dumps(cond)]


def synthetic_ndjson_lines_two_class() -> List[str]:
    """Two patients (alive + deceased) so binary label fit succeeds."""

    base = synthetic_ndjson_lines()
    dead_p = {
        "resourceType": "Patient",
        "id": "p-synth-2",
        "birthDate": "1940-05-05",
        "deceasedBoolean": True,
    }
    dead_enc = {
        "resourceType": "Encounter",
        "id": "e-dead",
        "subject": {"reference": "Patient/p-synth-2"},
        "period": {"start": "2020-07-01T10:00:00Z"},
        "class": {"code": "IMP"},
    }
    dead_obs = {
        "resourceType": "Observation",
        "id": "o-dead",
        "subject": {"reference": "Patient/p-synth-2"},
        "encounter": {"reference": "Encounter/e-dead"},
        "effectiveDateTime": "2020-07-01T12:00:00Z",
        "code": {"coding": [{"system": "http://loinc.org", "code": "789-0"}]},
    }
    return base + [json.dumps(dead_p), json.dumps(dead_enc), json.dumps(dead_obs)]


def build_fhir_sample_dataset_from_lines(
    lines: Sequence[str],
    task: Any,
    *,
    vocab_path: Optional[str] = None,
) -> Tuple[List[FHIRPatient], ConceptVocab, List[Dict[str, Any]]]:
    """Parse in-memory NDJSON lines into patients and task samples (for tests)."""

    resources: List[Dict[str, Any]] = []
    for line in lines:
        o = parse_ndjson_line(line)
        if o:
            resources.append(o)
    grouped = group_resources_by_patient(resources)
    vocab = (
        ConceptVocab.load(vocab_path)
        if vocab_path and os.path.isfile(vocab_path)
        else ConceptVocab()
    )
    patients: List[FHIRPatient] = []
    for pid, res_list in grouped.items():
        pr = FHIRPatient(patient_id=pid, resources=res_list)
        for r in res_list:
            if r.get("resourceType") == "Patient":
                pr.birth_date = _parse_dt(r.get("birthDate"))
                break
        patients.append(pr)
    task.vocab = vocab
    task._specials = None
    samples: List[Dict[str, Any]] = []
    for p in patients:
        samples.extend(task(p))
    return patients, vocab, samples
