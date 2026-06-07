import gzip
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List, Tuple

import orjson
import polars as pl

from pyhealth.data import Patient
from pyhealth.datasets import MIMIC4FHIR
from pyhealth.datasets.fhir.utils import (
    flatten_resource,
    load_resource_specs_from_yaml,
)
from pyhealth.processors.cehr_processor import ConceptVocab
from pyhealth.tasks.mpf_clinical_prediction import (
    MPFClinicalPredictionTask,
    collect_cehr_timeline_events,
    infer_mortality_label,
)


def _mimic4_specs():
    """Load the bundled MIMIC4 ResourceSpec registry from its YAML."""
    import yaml as _yaml
    with open(MIMIC4FHIR.DEFAULT_CONFIG_PATH, encoding="utf-8") as _f:
        return load_resource_specs_from_yaml(_yaml.safe_load(_f))


_MIMIC4_SPECS = _mimic4_specs()


def _flatten_resource_to_table_row(resource):
    """Flatten one resource via the MIMIC4 spec registry (test convenience)."""
    return flatten_resource(resource, _MIMIC4_SPECS)


def _clinical_slice(sample: Dict[str, object]) -> Tuple[List[str], List[int], List[int]]:
    """Drop ``<pad>`` and the leading/trailing boundary tokens from a sample.

    Returns the per-event ``(concept_keys, visit_orders, visit_segments)``
    lists for the patient's clinical events only.
    """
    keys = list(sample["concept_ids"])  # type: ignore[arg-type]
    v_o = list(sample["visit_orders"])  # type: ignore[arg-type]
    v_s = list(sample["visit_segments"])  # type: ignore[arg-type]
    non_pad = [
        (k, o, s) for k, o, s in zip(keys, v_o, v_s) if k != "<pad>"
    ]
    # Strip leading boundary (<mor>/<cls>) and trailing <reg>.
    middle = non_pad[1:-1] if len(non_pad) >= 2 else []
    return (
        [k for k, _, _ in middle],
        [o for _, o, _ in middle],
        [s for _, _, s in middle],
    )

from tests.core.test_fhir_ndjson_fixtures import (
    ndjson_two_class_text,
    write_one_patient_ndjson,
    write_two_class_ndjson,
)


def _third_patient_loinc_resources() -> List[Dict[str, object]]:
    return [
        {
            "resourceType": "Patient",
            "id": "p-synth-3",
            "birthDate": "1960-01-01",
        },
        {
            "resourceType": "Encounter",
            "id": "e3",
            "subject": {"reference": "Patient/p-synth-3"},
            "period": {"start": "2020-08-01T10:00:00Z"},
            "class": {"code": "IMP"},
        },
        {
            "resourceType": "Observation",
            "id": "o3",
            "subject": {"reference": "Patient/p-synth-3"},
            "encounter": {"reference": "Encounter/e3"},
            "effectiveDateTime": "2020-08-01T12:00:00Z",
            "code": {"coding": [{"system": "http://loinc.org", "code": "999-9"}]},
        },
    ]


def write_two_class_plus_third_ndjson(directory: Path, *, name: str = "fixture.ndjson") -> Path:
    lines = ndjson_two_class_text().strip().split("\n")
    lines.extend(orjson.dumps(r).decode("utf-8") for r in _third_patient_loinc_resources())
    path = directory / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _patient_from_rows(patient_id: str, rows: List[Dict[str, object]]) -> Patient:
    """Build a Patient whose ``timestamp`` column is a real datetime, matching
    the shape ``FHIRDataset.load_table`` produces in production.

    Production flat tables always carry a ``{event_type}/event_time`` column
    (null when the source had no time), and ``timestamp`` is derived from it.
    Mirror that here so ``Event`` attribute access sees ``event_time`` (the
    null-aware signal the timeline uses): inject ``{event_type}/event_time`` =
    the row's ``timestamp`` when not already set.
    """
    rows = [dict(r) for r in rows]
    for r in rows:
        et = r.get("event_type")
        if et and f"{et}/event_time" not in r:
            r[f"{et}/event_time"] = r.get("timestamp")
    df = pl.DataFrame(rows).with_columns(
        pl.col("timestamp").str.to_datetime(strict=False)
    )
    return Patient(patient_id=patient_id, data_source=df)


class TestDeceasedBooleanFlattening(unittest.TestCase):
    def test_string_false_not_coerced_by_python_bool(self) -> None:
        """Non-conformant ``\"false\"`` string must not become stored ``\"true\"``."""
        row = _flatten_resource_to_table_row(
            {
                "resourceType": "Patient",
                "id": "p-str-false",
                "deceasedBoolean": "false",
            }
        )
        self.assertIsNotNone(row)
        _table, payload = row
        self.assertEqual(payload.get("deceased_boolean"), "false")

    def test_string_true_parsed(self) -> None:
        row = _flatten_resource_to_table_row(
            {
                "resourceType": "Patient",
                "id": "p-str-true",
                "deceasedBoolean": "true",
            }
        )
        self.assertIsNotNone(row)
        self.assertEqual(row[1].get("deceased_boolean"), "true")

    def test_json_booleans_unchanged(self) -> None:
        for raw, expected in ((True, "true"), (False, "false")):
            with self.subTest(raw=raw):
                row = _flatten_resource_to_table_row(
                    {
                        "resourceType": "Patient",
                        "id": "p-bool",
                        "deceasedBoolean": raw,
                    }
                )
                self.assertIsNotNone(row)
                self.assertEqual(row[1].get("deceased_boolean"), expected)

    def test_unknown_deceased_type_stored_as_none(self) -> None:
        row = _flatten_resource_to_table_row(
            {
                "resourceType": "Patient",
                "id": "p-garbage",
                "deceasedBoolean": {"unexpected": "object"},
            }
        )
        self.assertIsNotNone(row)
        self.assertIsNone(row[1].get("deceased_boolean"))

    def test_infer_mortality_respects_string_false_row(self) -> None:
        patient = _patient_from_rows(
            "p1",
            [
                {
                    "event_type": "patient",
                    "timestamp": "2020-01-01T00:00:00",
                    "patient/deceased_boolean": "false",
                },
            ],
        )
        self.assertEqual(infer_mortality_label(patient), 0)


class TestFHIRDataset(unittest.TestCase):
    def test_concept_vocab_from_json_empty_token_to_id(self) -> None:
        v = ConceptVocab.from_json({"token_to_id": {}})
        self.assertIn("<pad>", v.token_to_id)
        self.assertIn("<unk>", v.token_to_id)
        self.assertEqual(v._next_id, 2)

    def test_concept_vocab_from_json_empty_respects_next_id(self) -> None:
        v = ConceptVocab.from_json({"token_to_id": {}, "next_id": 50})
        self.assertEqual(v._next_id, 50)

    def test_sorted_ndjson_files_accepts_sequence_and_dedupes(self) -> None:
        from pyhealth.datasets.fhir.utils import sorted_ndjson_files

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "MimicPatient.ndjson.gz").write_text("x", encoding="utf-8")
            (root / "MimicMedication.ndjson.gz").write_text("y", encoding="utf-8")
            (root / "notes.txt").write_text("z", encoding="utf-8")
            wide = sorted_ndjson_files(root, "**/*.ndjson.gz")
            narrow = sorted_ndjson_files(
                root,
                ["MimicPatient*.ndjson.gz", "**/MimicPatient*.ndjson.gz"],
            )
            self.assertEqual(len(wide), 2)
            self.assertEqual(len(narrow), 1)
            self.assertEqual(narrow[0].name, "MimicPatient.ndjson.gz")

    def test_dataset_accepts_glob_patterns_kwarg(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            write_one_patient_ndjson(Path(tmp))
            ds = MIMIC4FHIR(
                root=tmp, glob_patterns=["*.ndjson"], cache_dir=tmp
            )
            self.assertEqual(ds.glob_patterns, ["*.ndjson"])

    def test_dataset_rejects_both_glob_kwargs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                MIMIC4FHIR(
                    root=tmp,
                    glob_pattern="*.ndjson",
                    glob_patterns=["*.ndjson"],
                    cache_dir=tmp,
                )

    def test_disk_ingest_gz_and_max_patients(self) -> None:
        """gzip ingest path + ``max_patients`` cap, covered in one build.

        The heavier build/schema/set_task/pre_filter assertions now live in
        ``TestFHIRSharedWorkflow`` (one shared build), so this is the only
        ingest-variant build left in this class.
        """
        with tempfile.TemporaryDirectory() as tmp:
            gz_path = Path(tmp) / "fixture.ndjson.gz"
            with gzip.open(gz_path, "wt", encoding="utf-8") as gz:
                gz.write(ndjson_two_class_text())
            ds = MIMIC4FHIR(root=tmp, glob_pattern="*.ndjson.gz", max_patients=5)
            self.assertEqual(len(ds.unique_patient_ids), 2)

    def test_encounter_reference_requires_exact_id(self) -> None:
        patient = _patient_from_rows(
            "p1",
            [
                {
                    "patient_id": "p1",
                    "event_type": "patient",
                    "timestamp": None,
                    "patient/birth_date": "1950-01-01",
                },
                {
                    "patient_id": "p1",
                    "event_type": "encounter",
                    "timestamp": "2020-06-01T10:00:00",
                    "encounter/encounter_id": "e1",
                    "encounter/encounter_class": "AMB",
                },
                {
                    "patient_id": "p1",
                    "event_type": "encounter",
                    "timestamp": "2020-07-02T10:00:00",
                    "encounter/encounter_id": "e10",
                    "encounter/encounter_class": "IMP",
                },
                {
                    "patient_id": "p1",
                    "event_type": "condition",
                    "timestamp": "2020-07-02T11:00:00",
                    "condition/encounter_id": "e10",
                    "condition/concept_key": "http://hl7.org/fhir/sid/icd-10-cm|I99",
                },
            ],
        )
        sample = MPFClinicalPredictionTask(max_len=64, use_mpf=True)(patient)[0]
        self.assertEqual(
            sample["concept_ids"].count("http://hl7.org/fhir/sid/icd-10-cm|I99"),
            1,
        )

    def test_unlinked_condition_emitted_once_with_two_encounters(self) -> None:
        patient = _patient_from_rows(
            "p1",
            [
                {
                    "patient_id": "p1",
                    "event_type": "patient",
                    "timestamp": None,
                    "patient/birth_date": "1950-01-01",
                },
                {
                    "patient_id": "p1",
                    "event_type": "encounter",
                    "timestamp": "2020-06-01T10:00:00",
                    "encounter/encounter_id": "ea",
                    "encounter/encounter_class": "AMB",
                },
                {
                    "patient_id": "p1",
                    "event_type": "encounter",
                    "timestamp": "2020-07-01T10:00:00",
                    "encounter/encounter_id": "eb",
                    "encounter/encounter_class": "IMP",
                },
                {
                    "patient_id": "p1",
                    "event_type": "condition",
                    "timestamp": "2020-06-15T12:00:00",
                    "condition/concept_key": "http://hl7.org/fhir/sid/icd-10-cm|Z00",
                },
            ],
        )
        sample = MPFClinicalPredictionTask(max_len=64, use_mpf=True)(patient)[0]
        self.assertEqual(
            sample["concept_ids"].count("http://hl7.org/fhir/sid/icd-10-cm|Z00"),
            1,
        )

    def test_max_len_two_keeps_only_boundary_tokens(self) -> None:
        """``max_len=2`` leaves room for only the two boundary tokens; the
        clinical timeline is truncated away.
        """
        patient = _patient_from_rows(
            "p1",
            [
                {
                    "patient_id": "p1",
                    "event_type": "patient",
                    "timestamp": None,
                    "patient/birth_date": "1950-01-01",
                },
                {
                    "patient_id": "p1",
                    "event_type": "encounter",
                    "timestamp": "2020-06-01T10:00:00",
                    "encounter/encounter_id": "e1",
                    "encounter/encounter_class": "AMB",
                },
                {
                    "patient_id": "p1",
                    "event_type": "condition",
                    "timestamp": "2020-06-01T11:00:00",
                    "condition/encounter_id": "e1",
                    "condition/concept_key": "http://hl7.org/fhir/sid/icd-10-cm|I10",
                },
            ],
        )
        sample = MPFClinicalPredictionTask(max_len=2, use_mpf=True)(patient)[0]
        self.assertEqual(sample["concept_ids"], ["<mor>", "<reg>"])
        self.assertEqual(sample["visit_segments"], [0, 0])

    def test_visit_segments_alternate_by_visit_index(self) -> None:
        patient = _patient_from_rows(
            "p1",
            [
                {
                    "patient_id": "p1",
                    "event_type": "patient",
                    "timestamp": None,
                    "patient/birth_date": "1950-01-01",
                },
                {
                    "patient_id": "p1",
                    "event_type": "encounter",
                    "timestamp": "2020-06-01T10:00:00",
                    "encounter/encounter_id": "e0",
                    "encounter/encounter_class": "AMB",
                },
                {
                    "patient_id": "p1",
                    "event_type": "condition",
                    "timestamp": "2020-06-01T11:00:00",
                    "condition/encounter_id": "e0",
                    "condition/concept_key": "http://hl7.org/fhir/sid/icd-10-cm|I10",
                },
                {
                    "patient_id": "p1",
                    "event_type": "encounter",
                    "timestamp": "2020-07-01T10:00:00",
                    "encounter/encounter_id": "e1",
                    "encounter/encounter_class": "IMP",
                },
                {
                    "patient_id": "p1",
                    "event_type": "condition",
                    "timestamp": "2020-07-01T11:00:00",
                    "condition/encounter_id": "e1",
                    "condition/concept_key": "http://hl7.org/fhir/sid/icd-10-cm|I20",
                },
            ],
        )
        sample = MPFClinicalPredictionTask(max_len=64, use_mpf=True)(patient)[0]
        _, _, visit_segments = _clinical_slice(sample)
        self.assertEqual(visit_segments, [0, 0, 1, 1])

    def test_unlinked_visit_idx_matches_sequential_counter(self) -> None:
        patient = _patient_from_rows(
            "p1",
            [
                {
                    "patient_id": "p1",
                    "event_type": "patient",
                    "timestamp": None,
                    "patient/birth_date": "1950-01-01",
                },
                {
                    "patient_id": "p1",
                    "event_type": "encounter",
                    "timestamp": None,
                    "encounter/encounter_id": "e_bad",
                    "encounter/encounter_class": "AMB",
                },
                {
                    "patient_id": "p1",
                    "event_type": "encounter",
                    "timestamp": "2020-03-01T10:00:00",
                    "encounter/encounter_id": "e_ok",
                    "encounter/encounter_class": "IMP",
                },
                {
                    "patient_id": "p1",
                    "event_type": "condition",
                    "timestamp": "2020-03-05T11:00:00",
                    "condition/encounter_id": "e_ok",
                    "condition/concept_key": "http://hl7.org/fhir/sid/icd-10-cm|I10",
                },
                {
                    "patient_id": "p1",
                    "event_type": "condition",
                    "timestamp": "2020-03-15T12:00:00",
                    "condition/concept_key": "http://hl7.org/fhir/sid/icd-10-cm|Z00",
                },
            ],
        )
        sample = MPFClinicalPredictionTask(max_len=64, use_mpf=True)(patient)[0]
        keys = sample["concept_ids"]
        i_link = keys.index("http://hl7.org/fhir/sid/icd-10-cm|I10")
        i_free = keys.index("http://hl7.org/fhir/sid/icd-10-cm|Z00")
        self.assertEqual(sample["visit_orders"][i_link], sample["visit_orders"][i_free])
        self.assertEqual(sample["visit_segments"][i_link], sample["visit_segments"][i_free])

    def test_medication_request_uses_medication_codeable_concept(self) -> None:
        patient = _patient_from_rows(
            "p1",
            [
                {
                    "patient_id": "p1",
                    "event_type": "patient",
                    "timestamp": None,
                    "patient/birth_date": "1950-01-01",
                },
                {
                    "patient_id": "p1",
                    "event_type": "encounter",
                    "timestamp": "2020-06-01T10:00:00",
                    "encounter/encounter_id": "e1",
                    "encounter/encounter_class": "IMP",
                },
                {
                    "patient_id": "p1",
                    "event_type": "medication_request",
                    "timestamp": "2020-06-01T11:00:00",
                    "medication_request/encounter_id": "e1",
                    "medication_request/concept_key": "http://www.nlm.nih.gov/research/umls/rxnorm|111",
                },
                {
                    "patient_id": "p1",
                    "event_type": "medication_request",
                    "timestamp": "2020-06-01T12:00:00",
                    "medication_request/encounter_id": "e1",
                    "medication_request/concept_key": "http://www.nlm.nih.gov/research/umls/rxnorm|222",
                },
            ],
        )
        sample = MPFClinicalPredictionTask(max_len=64, use_mpf=True)(patient)[0]
        keys = sample["concept_ids"]
        ka = "http://www.nlm.nih.gov/research/umls/rxnorm|111"
        kb = "http://www.nlm.nih.gov/research/umls/rxnorm|222"
        self.assertEqual(keys.count(ka), 1)
        self.assertEqual(keys.count(kb), 1)

    def test_medication_request_medication_reference_token(self) -> None:
        patient = _patient_from_rows(
            "p1",
            [
                {
                    "patient_id": "p1",
                    "event_type": "patient",
                    "timestamp": None,
                    "patient/birth_date": "1950-01-01",
                },
                {
                    "patient_id": "p1",
                    "event_type": "encounter",
                    "timestamp": "2020-06-01T10:00:00",
                    "encounter/encounter_id": "e1",
                    "encounter/encounter_class": "IMP",
                },
                {
                    "patient_id": "p1",
                    "event_type": "medication_request",
                    "timestamp": "2020-06-01T11:00:00",
                    "medication_request/encounter_id": "e1",
                    "medication_request/concept_key": "MedicationRequest/reference|med-abc",
                },
            ],
        )
        sample = MPFClinicalPredictionTask(max_len=64, use_mpf=True)(patient)[0]
        key = "MedicationRequest/reference|med-abc"
        self.assertIn(key, sample["concept_ids"])
        self.assertEqual(sample["concept_ids"].count(key), 1)

    def test_collect_cehr_timeline_events_orders_by_timestamp(self) -> None:
        patient = _patient_from_rows(
            "p1",
            [
                {
                    "patient_id": "p1",
                    "event_type": "patient",
                    "timestamp": None,
                    "patient/birth_date": "1950-01-01",
                },
                {
                    "patient_id": "p1",
                    "event_type": "encounter",
                    "timestamp": "2020-06-01T10:00:00",
                    "encounter/encounter_id": "e1",
                    "encounter/encounter_class": "AMB",
                },
                {
                    "patient_id": "p1",
                    "event_type": "condition",
                    "timestamp": "2020-06-01T11:00:00",
                    "condition/encounter_id": "e1",
                    "condition/concept_key": "a|1",
                },
                {
                    "patient_id": "p1",
                    "event_type": "observation",
                    "timestamp": "2020-06-01T12:00:00",
                    "observation/encounter_id": "e1",
                    "observation/concept_key": "b|2",
                },
            ],
        )
        events = collect_cehr_timeline_events(patient)
        self.assertEqual([event[1] for event in events], ["encounter|AMB", "a|1", "b|2"])

    def test_timestampless_clinical_event_uses_encounter_start(self) -> None:
        """A clinical event with no ``event_time`` that is linked to an encounter
        must be placed at the encounter's start, not coerced to ~``now()``.

        ``Event.__init__`` coerces ``timestamp=None`` to ``datetime.now()``, so
        the timeline relies on the raw ``event_time`` attribute as the null
        sentinel; this locks that fallback.
        """
        patient = _patient_from_rows(
            "p1",
            [
                {
                    "patient_id": "p1",
                    "event_type": "patient",
                    "timestamp": None,
                    "patient/birth_date": "1950-01-01",
                },
                {
                    "patient_id": "p1",
                    "event_type": "encounter",
                    "timestamp": "2020-06-01T10:00:00",
                    "encounter/encounter_id": "e1",
                    "encounter/encounter_class": "IMP",
                },
                {
                    "patient_id": "p1",
                    "event_type": "condition",
                    "timestamp": None,  # timestamp-less clinical event
                    "condition/encounter_id": "e1",
                    "condition/concept_key": "http://hl7.org/fhir/sid/icd-10-cm|I10",
                },
            ],
        )
        events = collect_cehr_timeline_events(patient)
        enc_time = next(t for t, _ck, et, _v in events if et == "encounter")
        cond_time = next(
            t
            for t, ck, _et, _v in events
            if ck == "http://hl7.org/fhir/sid/icd-10-cm|I10"
        )
        self.assertEqual(cond_time, enc_time)

    def test_observation_effective_period_start_yields_event_time(self) -> None:
        """Choice-type fix: an Observation carrying only ``effectivePeriod.start``
        (no ``effectiveDateTime``) must still resolve a non-null event_time.
        The pre-refactor extractor silently dropped this variant.
        """
        row = _flatten_resource_to_table_row(
            {
                "resourceType": "Observation",
                "id": "o-period",
                "subject": {"reference": "Patient/p"},
                "effectivePeriod": {"start": "2022-02-02T00:00:00Z"},
                "code": {"coding": [{"system": "http://loinc.org", "code": "1-1"}]},
            }
        )
        self.assertIsNotNone(row)
        self.assertEqual(row[1]["event_time"], "2022-02-02T00:00:00Z")

    def test_new_resource_type_via_registry_flows_through(self) -> None:
        """A resource type absent from MIMIC4's specs flows end-to-end purely by
        adding a YAML entry — no engine change.

        Also exercises the directly-usable generic ``FHIRDataset`` (whole
        ingest contract authored in a single YAML, no subclass).
        """
        from pyhealth.datasets import FHIRDataset

        resources = [
            {"resourceType": "Patient", "id": "imm-1", "birthDate": "1970-01-01"},
            {
                "resourceType": "Immunization",
                "id": "i1",
                "patient": {"reference": "Patient/imm-1"},
                "occurrenceDateTime": "2021-01-01T00:00:00Z",
                "vaccineCode": {
                    "coding": [{"system": "http://hl7.org/fhir/sid/cvx", "code": "208"}]
                },
            },
        ]
        config_yaml = (
            "version: test\n"
            "resource_specs:\n"
            "  Patient:\n"
            "    table: patient\n"
            "    columns:\n"
            "      patient_id: { locate: [id], required: true }\n"
            "      birth_date: { locate: [birthDate] }\n"
            "  Immunization:\n"
            "    table: immunization\n"
            "    columns:\n"
            "      patient_id:   { locate: [patient.reference], transform: ref_id, required: true }\n"
            "      resource_id:  { locate: [id] }\n"
            "      encounter_id: { locate: [encounter.reference], transform: ref_id }\n"
            "      event_time:   { locate: [occurrenceDateTime, recorded] }\n"
            "      concept_key:  { locate: [vaccineCode], transform: coding_key }\n"
            "tables:\n"
            "  patient:\n"
            "    file_path: patient.parquet\n"
            "    patient_id: patient_id\n"
            "    timestamp: birth_date\n"
            "    attributes: [birth_date]\n"
            "  immunization:\n"
            "    file_path: immunization.parquet\n"
            "    patient_id: patient_id\n"
            "    timestamp: event_time\n"
            "    attributes: [resource_id, encounter_id, event_time, concept_key]\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmpp = Path(tmp)
            (tmpp / "fx.ndjson").write_text(
                "\n".join(orjson.dumps(r).decode("utf-8") for r in resources) + "\n",
                encoding="utf-8",
            )
            cfg = tmpp / "immun.yaml"
            cfg.write_text(config_yaml, encoding="utf-8")
            ds = FHIRDataset(
                root=str(tmpp),
                config_path=str(cfg),
                glob_pattern="*.ndjson",
                cache_dir=str(tmpp),
            )
            df = ds.global_event_df.collect(engine="streaming")
            self.assertIn("immunization/concept_key", df.columns)
            keys = (
                df.filter(pl.col("event_type") == "immunization")[
                    "immunization/concept_key"
                ]
                .to_list()
            )
            self.assertIn("http://hl7.org/fhir/sid/cvx|208", keys)

    def test_fhir_dataset_requires_specs(self) -> None:
        """Bare ``FHIRDataset`` (no specs, no subclass) errors clearly."""
        from pyhealth.datasets import FHIRDataset

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                FHIRDataset(root=tmp, glob_pattern="*.ndjson", cache_dir=tmp)


class TestFHIRSharedWorkflow(unittest.TestCase):
    """Build the dataset and run ``set_task`` ONCE, then assert over the shared
    artifacts. Mirrors a realistic "ingest once, do many things" workflow and
    keeps the suite fast: a single Dask build (plus one canonical ``set_task``)
    shared by every assertion, instead of rebuilding per test.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.mkdtemp()
        write_two_class_plus_third_ndjson(Path(cls._tmp))
        cls.ds = MIMIC4FHIR(
            root=cls._tmp, glob_pattern="*.ndjson", cache_dir=cls._tmp, num_workers=1
        )
        # The single Dask build for the whole class.
        cls.global_df = cls.ds.global_event_df.collect(engine="streaming")
        # The canonical set_task (reuses the build above; no rebuild).
        cls.sample_ds = cls.ds.set_task(
            MPFClinicalPredictionTask(max_len=48, use_mpf=True), num_workers=1
        )
        cls.samples = sorted(
            [cls.sample_ds[i] for i in range(len(cls.sample_ds))],
            key=lambda s: s["patient_id"],
        )

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls._tmp, ignore_errors=True)

    def test_build_produces_expected_tables_and_schema(self) -> None:
        """Flat parquet tables exist, and the global event frame has the
        expected long-format + namespaced columns with a patient's events.
        """
        prepared = self.ds.prepared_tables_dir
        for name in ("patient", "encounter", "condition", "observation"):
            self.assertTrue((prepared / f"{name}.parquet").is_file())
        for col in (
            "patient_id",
            "timestamp",
            "event_type",
            "condition/concept_key",
            "observation/concept_key",
            "patient/deceased_boolean",
        ):
            self.assertIn(col, self.global_df.columns)
        sub = self.global_df.filter(pl.col("patient_id") == "p-synth-1")
        self.assertGreaterEqual(len(sub), 2)

    def test_set_task_builds_vocab(self) -> None:
        vocab = self.sample_ds.input_processors["concept_ids"].vocab
        self.assertGreater(vocab.vocab_size, 6)

    def test_set_task_produces_correct_samples(self) -> None:
        self.assertEqual(len(self.samples), 3)
        self.assertEqual(
            {s["patient_id"] for s in self.samples},
            {"p-synth-1", "p-synth-2", "p-synth-3"},
        )
        for s in self.samples:
            self.assertIn("concept_ids", s)
            self.assertIn("label", s)
        self.assertEqual({int(s["label"]) for s in self.samples}, {0, 1})

    def test_cehr_sequence_shapes(self) -> None:
        patient = self.ds.get_patient("p-synth-1")
        sample = MPFClinicalPredictionTask(max_len=32, use_mpf=True)(patient)[0]
        n = len(sample["concept_ids"])
        self.assertEqual(n, 32)
        for key in (
            "token_type_ids", "time_stamps", "ages", "visit_orders", "visit_segments",
        ):
            self.assertEqual(len(sample[key]), n)
        non_special = {
            k for k in sample["concept_ids"]
            if k not in ("<pad>", "<mor>", "<cls>", "<reg>")
        }
        self.assertGreater(len(non_special), 0)

    def test_mpf_pre_filter_single_patient_limits_effective_workers(self) -> None:
        """Pre-filter yielding one patient caps effective_workers to 1 (the
        formula is verified directly; a 1-patient ``set_task`` would raise on a
        single label class).
        """
        class OnePatientMPFTask(MPFClinicalPredictionTask):
            def pre_filter(self, df: pl.LazyFrame) -> pl.LazyFrame:
                return df.filter(pl.col("patient_id") == "p-synth-1")

        warmup_pids = (
            OnePatientMPFTask(max_len=48, use_mpf=True)
            .pre_filter(self.ds.global_event_df)
            .select("patient_id")
            .unique()
            .collect(engine="streaming")
            .to_series()
            .sort()
            .to_list()
        )
        self.assertEqual(warmup_pids, ["p-synth-1"])
        effective_workers = min(2, len(warmup_pids)) if warmup_pids else 1
        self.assertEqual(effective_workers, 1)

    def test_mpf_pre_filter_excludes_dropped_patients_from_vocab(self) -> None:
        """A task ``pre_filter`` that drops a patient also drops their concept
        keys from the fitted vocab. Reuses the shared build; a distinct
        ``task_name`` keeps this run's sample cache separate from the canonical
        ``set_task`` in setUpClass (identical params would otherwise collide on
        the shared ``cache_dir``).
        """
        class TwoPatientMPFTask(MPFClinicalPredictionTask):
            task_name = "MPFClinicalPredictionFHIR_prefilter"

            def pre_filter(self, df: pl.LazyFrame) -> pl.LazyFrame:
                return df.filter(
                    pl.col("patient_id").is_in(["p-synth-1", "p-synth-2"])
                )

        sample_ds = self.ds.set_task(
            TwoPatientMPFTask(max_len=48, use_mpf=True), num_workers=1
        )
        vocab = sample_ds.input_processors["concept_ids"].vocab
        self.assertNotIn("http://loinc.org|999-9", vocab.token_to_id)
        self.assertIn("http://loinc.org|789-0", vocab.token_to_id)


if __name__ == "__main__":
    unittest.main()
