"""NDJSON file bodies for :mod:`tests.core.test_mimic4_fhir_dataset` (disk-only ingest)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import orjson


# ---------------------------------------------------------------------------
# Synthetic in-memory FHIR resources
# ---------------------------------------------------------------------------


def _one_patient_resources() -> List[Dict[str, Any]]:
    return [
        {"resourceType": "Patient", "id": "p-synth-1", "birthDate": "1950-01-01", "gender": "female"},
        {
            "resourceType": "Encounter",
            "id": "e1",
            "subject": {"reference": "Patient/p-synth-1"},
            "period": {"start": "2020-06-01T10:00:00Z"},
            "class": {"code": "IMP"},
        },
        {
            "resourceType": "Condition",
            "id": "c1",
            "subject": {"reference": "Patient/p-synth-1"},
            "encounter": {"reference": "Encounter/e1"},
            "code": {"coding": [{"system": "http://hl7.org/fhir/sid/icd-10-cm", "code": "I10"}]},
            "onsetDateTime": "2020-06-01T11:00:00Z",
        },
    ]


def _two_patient_resources() -> List[Dict[str, Any]]:
    return [
        *_one_patient_resources(),
        {"resourceType": "Patient", "id": "p-synth-2", "birthDate": "1940-05-05", "deceasedBoolean": True},
        {
            "resourceType": "Encounter",
            "id": "e-dead",
            "subject": {"reference": "Patient/p-synth-2"},
            "period": {"start": "2020-07-01T10:00:00Z"},
            "class": {"code": "IMP"},
        },
        {
            "resourceType": "Observation",
            "id": "o-dead",
            "subject": {"reference": "Patient/p-synth-2"},
            "encounter": {"reference": "Encounter/e-dead"},
            "effectiveDateTime": "2020-07-01T12:00:00Z",
            "code": {"coding": [{"system": "http://loinc.org", "code": "789-0"}]},
        },
    ]


# ---------------------------------------------------------------------------
# Text serialisers
# ---------------------------------------------------------------------------


def ndjson_one_patient_text() -> str:
    return "\n".join(orjson.dumps(r).decode("utf-8") for r in _one_patient_resources()) + "\n"


def ndjson_two_class_text() -> str:
    return "\n".join(orjson.dumps(r).decode("utf-8") for r in _two_patient_resources()) + "\n"


# ---------------------------------------------------------------------------
# Disk writers
# ---------------------------------------------------------------------------


def write_two_class_ndjson(directory: Path, *, name: str = "fixture.ndjson") -> Path:
    path = directory / name
    path.write_text(ndjson_two_class_text(), encoding="utf-8")
    return path


def write_one_patient_ndjson(directory: Path, *, name: str = "fixture.ndjson") -> Path:
    path = directory / name
    path.write_text(ndjson_one_patient_text(), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Shared test helper
# ---------------------------------------------------------------------------


def run_task(ds: Any, task: Any) -> List[Dict[str, Any]]:
    """Run *task* over every patient in *ds* without the LitData caching pipeline.

    This helper mirrors the direct-iteration path that the old
    ``MIMIC4FHIRDataset.gather_samples`` provided.  It is intentionally kept
    here (the shared fixture module) so all FHIR test files can import it
    rather than each maintaining their own copy.

    Args:
        ds: A :class:`~pyhealth.datasets.MIMIC4FHIRDataset` instance whose
            ``global_event_df`` has already been built.
        task: A :class:`~pyhealth.tasks.MPFClinicalPredictionTask` instance.

    Returns:
        Flat list of sample dicts, one per patient.
    """
    task.vocab = ds.vocab
    task._specials = None
    task.frozen_vocab = False
    return [s for patient in ds.iter_patients() for s in task(patient)]
