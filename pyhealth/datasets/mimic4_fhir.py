"""MIMIC-IV FHIR (NDJSON) ingestion for CEHR-style sequences.

Loads newline-delimited JSON (plain ``*.ndjson`` or gzip ``*.ndjson.gz``, as on
PhysioNet), or Bundle ``entry`` resources, groups by Patient id, and builds
token timelines for MPF / EHRMambaCEHR.

Settings such as ``glob_pattern`` live in ``configs/mimic4_fhir.yaml`` and are
read by :func:`read_fhir_settings_yaml`. For disk data, point
:class:`MIMIC4FHIRDataset` at your PhysioNet export (``MIMIC4_FHIR_ROOT``); for
tests, use :func:`synthetic_ndjson_lines` / :func:`synthetic_ndjson_lines_two_class`
or a temporary ``*.ndjson`` file tree.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, Iterator, List, Optional, Sequence, Tuple

from yaml import safe_load

from .base_dataset import BaseDataset

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
    if path.suffix == ".gz":
        opener = gzip.open(path, "rt", encoding="utf-8", errors="replace")
    else:
        opener = open(path, encoding="utf-8", errors="replace")
    with opener as f:
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
    """Flatten patient resources into CEHR-aligned lists (pre-padding).

    Args:
        max_len: Maximum number of **clinical** tokens emitted (after time sort and
            tail slice). Use ``0`` to emit no clinical tokens (empty lists; avoids
            Python's ``events[-0:]`` which would incorrectly take the full timeline).
            Downstream MPF tasks reserve two slots for ``<mor>``/``<cls>`` and
            ``<reg>``, so pass ``max_len - 2`` there when the final tensor length
            is fixed.
    """

    birth = patient.birth_date
    if birth is None:
        pr = patient.get_patient_resource()
        if pr:
            birth = _parse_dt(pr.get("birthDate"))

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
        seg = v_idx % 2
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


def collect_resources_from_root(root: Path, glob_pattern: str) -> List[Dict[str, Any]]:
    """Read all NDJSON / NDJSON.GZ / Bundle lines under root matching ``glob_pattern``."""

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


class MIMIC4FHIRDataset(BaseDataset):
    """MIMIC-IV on FHIR (NDJSON / Bundle) for CEHR token timelines.

    Mirrors the *root + YAML config + task* workflow of
    :class:`~pyhealth.datasets.MIMIC4Dataset`, but parses FHIR R4 resources instead
    of MIMIC CSV tables. This class does **not** materialize a Parquet
    ``global_event_df``; use :meth:`gather_samples` or :meth:`set_task` with
    :class:`~pyhealth.tasks.mpf_clinical_prediction.MPFClinicalPredictionTask`.

    Configuration defaults live in ``pyhealth/datasets/configs/mimic4_fhir.yaml``
    (``glob_pattern``, ``version``).

    Args:
        root: Directory tree containing NDJSON files.
        config_path: Optional path to the FHIR YAML settings file.
        glob_pattern: If set, overrides the YAML ``glob_pattern``.
        max_patients: Stop after this many patients (sorted by id).
        vocab_path: Optional path to a saved :class:`ConceptVocab` JSON.
        cache_dir: Forwarded to :class:`~pyhealth.datasets.BaseDataset`.
        num_workers: Forwarded to :class:`~pyhealth.datasets.BaseDataset`.
        dev: If True and ``max_patients`` is None, caps loading at 1000 patients.

    Example:
        >>> from pyhealth.datasets import MIMIC4FHIRDataset
        >>> from pyhealth.tasks.mpf_clinical_prediction import (
        ...     MPFClinicalPredictionTask,
        ... )
        >>> ds = MIMIC4FHIRDataset(root="/path/to/ndjson", max_patients=50)
        >>> task = MPFClinicalPredictionTask(max_len=256)
        >>> sample_ds = ds.set_task(task)  # doctest: +SKIP

    Raises:
        FileNotFoundError: If ``root`` is not a directory when loading patients.
    """

    def __init__(
        self,
        root: str,
        config_path: Optional[str] = None,
        glob_pattern: Optional[str] = None,
        max_patients: Optional[int] = None,
        vocab_path: Optional[str] = None,
        cache_dir: Optional[str | Path] = None,
        num_workers: int = 1,
        dev: bool = False,
    ) -> None:
        super().__init__(
            root=root,
            tables=["fhir_ndjson"],
            dataset_name="mimic4_fhir",
            config_path=None,
            cache_dir=cache_dir,
            num_workers=num_workers,
            dev=dev,
        )
        cfg_path = config_path or os.path.join(
            os.path.dirname(__file__), "configs", "mimic4_fhir.yaml"
        )
        self._fhir_settings = read_fhir_settings_yaml(cfg_path)
        self.glob_pattern = (
            glob_pattern
            if glob_pattern is not None
            else str(self._fhir_settings.get("glob_pattern", "**/*.ndjson"))
        )
        self.max_patients = max_patients
        if self.dev and self.max_patients is None:
            self.max_patients = 1000
        if vocab_path and os.path.isfile(vocab_path):
            self.vocab = ConceptVocab.load(vocab_path)
        else:
            self.vocab = ConceptVocab()
        self._patients: Optional[List[FHIRPatient]] = None

    @property
    def global_event_df(self) -> Any:
        raise NotImplementedError(
            "MIMIC4FHIRDataset does not build global_event_df. "
            "Use gather_samples(task) or set_task(task) with "
            "MPFClinicalPredictionTask."
        )

    @property
    def unique_patient_ids(self) -> List[str]:  # type: ignore[override]
        return [p.patient_id for p in self.load_patients()]

    def get_patient(self, patient_id: str) -> Any:
        raise NotImplementedError(
            "MIMIC4FHIRDataset does not map to pyhealth.data.Patient; "
            "use load_patients() and FHIRPatient."
        )

    def iter_patients(self, df: Optional[Any] = None) -> Iterator[Any]:
        raise NotImplementedError(
            "Use load_patients(); FHIR path does not stream Polars Patient rows."
        )

    def stats(self) -> None:
        n = len(self.load_patients())
        print(f"Dataset: {self.dataset_name}")
        print(f"Dev mode: {self.dev}")
        print(f"FHIR patients: {n}")

    def set_task(
        self,
        task: Any = None,
        num_workers: Optional[int] = None,  # unused; FHIR path is in-process
        input_processors: Optional[Any] = None,
        output_processors: Optional[Any] = None,
    ) -> Any:
        """Build a :class:`~pyhealth.datasets.SampleDataset` from FHIR + task."""
        self._main_guard(self.set_task.__name__)
        if task is None:
            raise ValueError(
                "Pass a task instance, e.g. MPFClinicalPredictionTask(max_len=512)."
            )
        from .sample_dataset import create_sample_dataset

        samples = self.gather_samples(task)
        return create_sample_dataset(
            samples=samples,
            input_schema=task.input_schema,
            output_schema=task.output_schema,
            dataset_name=self.dataset_name,
            task_name=task.task_name,
            input_processors=input_processors,
            output_processors=output_processors,
        )

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
        """Run ``task`` on each :class:`FHIRPatient` (sets ``task.vocab``)."""

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
