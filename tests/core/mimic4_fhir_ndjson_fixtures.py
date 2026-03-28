"""NDJSON file bodies for :mod:`tests.core.test_mimic4_fhir_dataset` (disk-only ingest)."""

from __future__ import annotations

from pathlib import Path

from pyhealth.datasets.mimic4_fhir import (
    synthetic_mpf_one_patient_ndjson_text,
    synthetic_mpf_two_patient_ndjson_text,
)


def ndjson_one_patient_text() -> str:
    return synthetic_mpf_one_patient_ndjson_text()


def ndjson_two_class_text() -> str:
    return synthetic_mpf_two_patient_ndjson_text()


def write_two_class_ndjson(directory: Path, *, name: str = "fixture.ndjson") -> Path:
    path = directory / name
    path.write_text(ndjson_two_class_text(), encoding="utf-8")
    return path


def write_one_patient_ndjson(directory: Path, *, name: str = "fixture.ndjson") -> Path:
    path = directory / name
    path.write_text(ndjson_one_patient_text(), encoding="utf-8")
    return path
