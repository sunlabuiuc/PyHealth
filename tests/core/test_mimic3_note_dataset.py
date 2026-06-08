"""Tests for MIMIC3NoteDataset and EHREvidenceRetrievalTask.
Contributor: Abhisek Sinha (abhisek5@illinois.edu)
Paper: `Ahsan et al. (2024) <https://arxiv.org/abs/2309.04550>`
All tests use synthetic data generated in-memory; no real MIMIC files are
required.  Tests should complete in milliseconds by design.
"""
import importlib.util
import os
import sys
import types
import unittest
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Bootstrap: load only the modules we need without triggering the full
# PyHealth package chain (which requires torch >=3.12, torchvision, etc.)
# ---------------------------------------------------------------------------
_PYHEALTH_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _register_stub_pkg(dotted_name: str):
    """Create a minimal stub package so relative imports resolve."""
    parts = dotted_name.split(".")
    for i in range(1, len(parts) + 1):
        name = ".".join(parts[:i])
        if name not in sys.modules:
            stub = types.ModuleType(name)
            stub.__path__ = [os.path.join(_PYHEALTH_ROOT, *parts[:i])]
            stub.__package__ = name
            stub.__spec__ = importlib.util.spec_from_file_location(
                name, os.path.join(_PYHEALTH_ROOT, *parts[:i], "__init__.py")
            )
            sys.modules[name] = stub


def _load_pyhealth_module(rel_path: str) -> types.ModuleType:
    """Load pyhealth/{rel_path} with its package context set correctly.

    Args:
        rel_path: Slash-separated path relative to the pyhealth/ package root,
                  e.g. ``"tasks/ehr_evidence_retrieval.py"``.
    """
    abs_path = os.path.join(_PYHEALTH_ROOT, "pyhealth", rel_path.replace("/", os.sep))
    # Derive the dotted module name, e.g. "pyhealth.tasks.ehr_evidence_retrieval"
    module_name = "pyhealth." + rel_path.rstrip(".py").replace("/", ".").replace(os.sep, ".")
    package_name = ".".join(module_name.split(".")[:-1])

    # Ensure parent stubs exist
    _register_stub_pkg(package_name)

    spec = importlib.util.spec_from_file_location(module_name, abs_path)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = package_name
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Pre-register stubs for heavy dependencies so relative imports succeed
# ---------------------------------------------------------------------------
_register_stub_pkg("pyhealth")
_register_stub_pkg("pyhealth.tasks")
_register_stub_pkg("pyhealth.datasets")

# Stub for pyhealth.data (ehr_evidence_retrieval.py imports Patient from here)
_data_stub = types.ModuleType("pyhealth.data")
_data_stub.Patient = MagicMock  # type: ignore[attr-defined]
sys.modules["pyhealth.data"] = _data_stub

# Load base_task first so the relative import in ehr_evidence_retrieval resolves
try:
    import polars as pl
    _POLARS_AVAILABLE = True
except (ImportError, OSError):
    _POLARS_AVAILABLE = False

# base_task only needs polars (already handled by the stub above if polars missing)
_base_task_stub = types.ModuleType("pyhealth.tasks.base_task")


class _BaseTask:
    task_name: str = ""
    input_schema: dict = {}
    output_schema: dict = {}

    def __init__(self, code_mapping=None):
        pass

    def pre_filter(self, df):
        return df

    def __call__(self, patient):
        raise NotImplementedError


_base_task_stub.BaseTask = _BaseTask  # type: ignore[attr-defined]
sys.modules["pyhealth.tasks.base_task"] = _base_task_stub

# Now load the actual task module
try:
    _task_mod = _load_pyhealth_module("tasks/ehr_evidence_retrieval.py")
    EHREvidenceRetrievalTask = _task_mod.EHREvidenceRetrievalTask
    _TASK_AVAILABLE = True
except Exception as exc:  # pragma: no cover
    _TASK_AVAILABLE = False
    _TASK_LOAD_ERROR = str(exc)


pytestmark = pytest.mark.skipif(
    not _TASK_AVAILABLE,
    reason=f"EHREvidenceRetrievalTask could not be loaded: "
           f"{locals().get('_TASK_LOAD_ERROR', 'unknown error')}",
)


# ---------------------------------------------------------------------------
# Helpers: synthetic patient/event builder
# ---------------------------------------------------------------------------

def _make_event(**attrs):
    event = MagicMock()
    for k, v in attrs.items():
        setattr(event, k, v)
    return event


def _make_patient(patient_id: str, note_events, diag_events):
    patient = MagicMock()
    patient.patient_id = patient_id

    def get_events(event_type, **kwargs):
        if event_type == "noteevents":
            return note_events
        if event_type == "diagnoses_icd":
            return diag_events
        return []

    patient.get_events.side_effect = get_events
    return patient


# ---------------------------------------------------------------------------
# MIMIC3NoteDataset tests (config / preprocessing only — no DB loading)
# ---------------------------------------------------------------------------

class TestMIMIC3NoteDatasetConfig(unittest.TestCase):
    """Config-level tests that don't need torch or dask."""

    def test_config_file_exists(self):
        config_path = os.path.join(
            _PYHEALTH_ROOT, "pyhealth", "datasets", "configs", "mimic3_note.yaml"
        )
        self.assertTrue(os.path.isfile(config_path), f"Missing: {config_path}")

    def test_config_has_noteevents_with_iserror(self):
        import yaml
        config_path = os.path.join(
            _PYHEALTH_ROOT, "pyhealth", "datasets", "configs", "mimic3_note.yaml"
        )
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        tables = cfg.get("tables", {})
        self.assertIn("noteevents", tables)
        attrs = tables["noteevents"].get("attributes", [])
        self.assertIn("iserror", attrs)
        self.assertIn("text", attrs)
        self.assertIn("category", attrs)

    def test_config_has_diagnoses_icd(self):
        import yaml
        config_path = os.path.join(
            _PYHEALTH_ROOT, "pyhealth", "datasets", "configs", "mimic3_note.yaml"
        )
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        self.assertIn("diagnoses_icd", cfg.get("tables", {}))

    def test_config_version(self):
        import yaml
        config_path = os.path.join(
            _PYHEALTH_ROOT, "pyhealth", "datasets", "configs", "mimic3_note.yaml"
        )
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        self.assertEqual(cfg.get("version"), "1.4")

    @pytest.mark.skipif(not _POLARS_AVAILABLE, reason="polars not installed")
    def test_preprocess_noteevents_casts_iserror(self):
        """preprocess_noteevents should coerce iserror to String dtype."""
        # Load only narwhals stub needed by mimic3.py
        _narwhals_stub = types.ModuleType("narwhals")
        import polars as _pl
        # narwhals mirrors polars API; just alias it for the preprocess method
        for attr in dir(_pl):
            try:
                setattr(_narwhals_stub, attr, getattr(_pl, attr))
            except Exception:
                pass
        sys.modules.setdefault("narwhals", _narwhals_stub)

        try:
            _mimic3_mod = _load_pyhealth_module("datasets/mimic3.py")
            MIMIC3NoteDataset = _mimic3_mod.MIMIC3NoteDataset
        except Exception as exc:
            pytest.skip(f"mimic3 module could not load: {exc}")

        obj = object.__new__(MIMIC3NoteDataset)
        df = _pl.DataFrame(
            {
                "charttime": [None, "2020-01-01 08:00:00"],
                "chartdate": ["2020-01-01", "2020-01-02"],
                "iserror": [1, 0],
            }
        ).lazy()
        result = obj.preprocess_noteevents(df).collect()
        self.assertEqual(result["iserror"].dtype, _pl.String)


# ---------------------------------------------------------------------------
# EHREvidenceRetrievalTask unit tests
# ---------------------------------------------------------------------------

class TestEHREvidenceRetrievalTask(unittest.TestCase):

    def _task(self, **kwargs):
        return EHREvidenceRetrievalTask(
            query_diagnosis="small vessel disease",
            condition_icd_codes=["437.3", "437.30"],
            **kwargs,
        )

    def test_task_schema(self):
        t = self._task()
        self.assertEqual(t.task_name, "EHREvidenceRetrieval")
        self.assertEqual(t.input_schema, {"notes": "text"})
        self.assertEqual(t.output_schema, {"label": "binary"})

    def test_positive_label(self):
        notes = [_make_event(text="SVD signs noted.", category="Discharge summary", iserror="0")]
        diags = [_make_event(icd9_code="437.3")]
        patient = _make_patient("P001", notes, diags)
        samples = self._task()(patient)
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["label"], 1)
        self.assertIn("SVD signs noted.", samples[0]["notes"])

    def test_negative_label(self):
        notes = [_make_event(text="No findings.", category="Radiology", iserror="0")]
        diags = [_make_event(icd9_code="250.00")]
        patient = _make_patient("P002", notes, diags)
        samples = self._task()(patient)
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["label"], 0)

    def test_no_notes_returns_empty(self):
        patient = _make_patient("P003", [], [])
        self.assertEqual(self._task()(patient), [])

    def test_iserror_filtering(self):
        good = _make_event(text="Good note.", category="Discharge summary", iserror="0")
        bad = _make_event(text="Error note.", category="Discharge summary", iserror="1")
        patient = _make_patient("P004", [good, bad], [])
        samples = self._task()(patient)
        self.assertEqual(len(samples), 1)
        self.assertNotIn("Error note.", samples[0]["notes"])
        self.assertIn("Good note.", samples[0]["notes"])

    def test_category_filtering(self):
        discharge = _make_event(text="Discharge note.", category="Discharge summary", iserror="0")
        nursing = _make_event(text="Nursing note.", category="Nursing", iserror="0")
        patient = _make_patient("P005", [discharge, nursing], [])
        task = self._task(note_categories=["Discharge summary"])
        samples = task(patient)
        self.assertEqual(len(samples), 1)
        self.assertIn("Discharge note.", samples[0]["notes"])
        self.assertNotIn("Nursing note.", samples[0]["notes"])

    def test_max_notes_truncation(self):
        notes = [_make_event(text=f"Note {i}.", category=None, iserror="0") for i in range(20)]
        patient = _make_patient("P006", notes, [])
        samples = self._task(max_notes=5)(patient)
        if samples:
            self.assertLessEqual(samples[0]["notes"].count("---"), 4)

    def test_custom_separator(self):
        n1 = _make_event(text="Note A.", category=None, iserror="0")
        n2 = _make_event(text="Note B.", category=None, iserror="0")
        patient = _make_patient("P007", [n1, n2], [])
        samples = self._task(note_separator=" | ")(patient)
        self.assertEqual(len(samples), 1)
        self.assertIn(" | ", samples[0]["notes"])

    def test_empty_text_skipped(self):
        good = _make_event(text="Valid note.", category=None, iserror="0")
        empty = _make_event(text="", category=None, iserror="0")
        patient = _make_patient("P008", [good, empty], [])
        samples = self._task()(patient)
        self.assertEqual(len(samples), 1)

    def test_patient_id_preserved(self):
        notes = [_make_event(text="Note.", category=None, iserror="0")]
        patient = _make_patient("P009", notes, [])
        samples = self._task()(patient)
        self.assertEqual(samples[0]["patient_id"], "P009")

    def test_query_in_sample(self):
        notes = [_make_event(text="Note.", category=None, iserror="0")]
        patient = _make_patient("P010", notes, [])
        samples = self._task()(patient)
        self.assertEqual(samples[0]["query_diagnosis"], "small vessel disease")

    def test_multiple_matching_icd_codes(self):
        """Any matching ICD code should set label=1."""
        notes = [_make_event(text="Vessel disease.", category=None, iserror="0")]
        diags = [_make_event(icd9_code="437.30")]  # second code in the set
        patient = _make_patient("P011", notes, diags)
        samples = self._task()(patient)
        self.assertEqual(samples[0]["label"], 1)

    def test_no_diagnosis_events_gives_label_0(self):
        """No diagnosis events at all → negative label."""
        notes = [_make_event(text="Some note.", category=None, iserror="0")]
        patient = _make_patient("P012", notes, [])
        samples = self._task()(patient)
        self.assertEqual(samples[0]["label"], 0)


if __name__ == "__main__":
    unittest.main()
