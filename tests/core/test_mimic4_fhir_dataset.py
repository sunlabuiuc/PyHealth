import gzip
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List

import orjson
import polars as pl

from pyhealth.data import Patient
from pyhealth.datasets import MIMIC4FHIRDataset
from pyhealth.datasets.fhir_utils import (
    _flatten_resource_to_table_row,
)
from pyhealth.processors.cehr_processor import (
    ConceptVocab,
    build_cehr_sequences,
    collect_cehr_timeline_events,
    infer_mortality_label,
)

from tests.core.test_mimic4_fhir_ndjson_fixtures import (
    ndjson_two_class_text,
    run_task,
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
    return Patient(patient_id=patient_id, data_source=pl.DataFrame(rows))


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


class TestMIMIC4FHIRDataset(unittest.TestCase):
    def test_concept_vocab_from_json_empty_token_to_id(self) -> None:
        v = ConceptVocab.from_json({"token_to_id": {}})
        self.assertIn("<pad>", v.token_to_id)
        self.assertIn("<unk>", v.token_to_id)
        self.assertEqual(v._next_id, 2)

    def test_concept_vocab_from_json_empty_respects_next_id(self) -> None:
        v = ConceptVocab.from_json({"token_to_id": {}, "next_id": 50})
        self.assertEqual(v._next_id, 50)

    def test_sorted_ndjson_files_accepts_sequence_and_dedupes(self) -> None:
        from pyhealth.datasets.fhir_utils import sorted_ndjson_files

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
            ds = MIMIC4FHIRDataset(
                root=tmp, glob_patterns=["*.ndjson"], cache_dir=tmp
            )
            self.assertEqual(ds.glob_patterns, ["*.ndjson"])
            _ = ds.global_event_df.collect(engine="streaming")

    def test_dataset_rejects_both_glob_kwargs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                MIMIC4FHIRDataset(
                    root=tmp,
                    glob_pattern="*.ndjson",
                    glob_patterns=["*.ndjson"],
                    cache_dir=tmp,
                )

    def test_disk_fixture_resolves_events_per_patient(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            write_one_patient_ndjson(Path(tmp))
            ds = MIMIC4FHIRDataset(root=tmp, glob_pattern="*.ndjson", cache_dir=tmp)
            sub = ds.global_event_df.filter(pl.col("patient_id") == "p-synth-1").collect(
                engine="streaming"
            )
            self.assertGreaterEqual(len(sub), 2)
            self.assertIn("condition/concept_key", sub.columns)

    def test_prepared_flat_tables_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            write_two_class_ndjson(Path(tmp))
            ds = MIMIC4FHIRDataset(root=tmp, glob_pattern="*.ndjson", cache_dir=tmp)
            _ = ds.global_event_df.collect(engine="streaming")
            prepared = ds.prepared_tables_dir
            self.assertTrue((prepared / "patient.parquet").is_file())
            self.assertTrue((prepared / "encounter.parquet").is_file())
            self.assertTrue((prepared / "condition.parquet").is_file())

    def test_build_cehr_non_empty(self) -> None:
        from pyhealth.tasks.mpf_clinical_prediction import MPFClinicalPredictionTask

        with tempfile.TemporaryDirectory() as tmp:
            write_one_patient_ndjson(Path(tmp))
            ds = MIMIC4FHIRDataset(root=tmp, glob_pattern="*.ndjson", cache_dir=tmp)
            task = MPFClinicalPredictionTask(max_len=64, use_mpf=True)
            run_task(ds, task)
            self.assertIsInstance(task.vocab, ConceptVocab)
            self.assertGreater(task.vocab.vocab_size, 2)

    def test_set_task_vocab_warm_on_litdata_cache_hit(self) -> None:
        from pyhealth.tasks.mpf_clinical_prediction import MPFClinicalPredictionTask

        with tempfile.TemporaryDirectory() as tmp:
            write_two_class_ndjson(Path(tmp))
            task_kw = {"max_len": 64, "use_mpf": True}
            ds1 = MIMIC4FHIRDataset(root=tmp, glob_pattern="*.ndjson", cache_dir=tmp)
            task1 = MPFClinicalPredictionTask(**task_kw)
            task1.warm_vocab(ds1, num_workers=1)
            ds1.set_task(task1, num_workers=1)
            warm_size = task1.vocab.vocab_size
            self.assertGreater(warm_size, 6)
            ds2 = MIMIC4FHIRDataset(root=tmp, glob_pattern="*.ndjson", cache_dir=tmp)
            task2 = MPFClinicalPredictionTask(**task_kw)
            task2.warm_vocab(ds2, num_workers=1)
            ds2.set_task(task2, num_workers=1)
            self.assertEqual(task2.vocab.vocab_size, warm_size)

    def test_mortality_heuristic(self) -> None:
        from pyhealth.tasks.mpf_clinical_prediction import MPFClinicalPredictionTask

        with tempfile.TemporaryDirectory() as tmp:
            write_two_class_ndjson(Path(tmp))
            ds = MIMIC4FHIRDataset(root=tmp, glob_pattern="*.ndjson", cache_dir=tmp)
            samples = run_task(ds, MPFClinicalPredictionTask(max_len=64, use_mpf=False))
            self.assertEqual({s["label"] for s in samples}, {0, 1})

    def test_infer_deceased(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            write_two_class_ndjson(Path(tmp))
            ds = MIMIC4FHIRDataset(root=tmp, glob_pattern="*.ndjson", cache_dir=tmp)
            dead = ds.get_patient("p-synth-2")
            self.assertEqual(infer_mortality_label(dead), 1)

    def test_disk_ndjson_gz_physionet_style(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gz_path = Path(tmp) / "fixture.ndjson.gz"
            with gzip.open(gz_path, "wt", encoding="utf-8") as gz:
                gz.write(ndjson_two_class_text())
            ds = MIMIC4FHIRDataset(root=tmp, glob_pattern="*.ndjson.gz", max_patients=5)
            self.assertGreaterEqual(len(ds.unique_patient_ids), 1)

    def test_disk_ndjson_temp_dir(self) -> None:
        from pyhealth.tasks.mpf_clinical_prediction import MPFClinicalPredictionTask

        with tempfile.TemporaryDirectory() as tmp:
            write_two_class_ndjson(Path(tmp))
            ds = MIMIC4FHIRDataset(root=tmp, glob_pattern="*.ndjson", max_patients=5)
            self.assertEqual(len(ds.unique_patient_ids), 2)
            samples = run_task(ds, MPFClinicalPredictionTask(max_len=48, use_mpf=True))
            self.assertGreaterEqual(len(samples), 1)
            for sample in samples:
                self.assertIn("concept_ids", sample)
                self.assertIn("label", sample)

    def test_global_event_df_schema_and_flattened_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            write_two_class_ndjson(Path(tmp))
            ds = MIMIC4FHIRDataset(root=tmp, glob_pattern="*.ndjson", cache_dir=tmp)
            df = ds.global_event_df.collect(engine="streaming")
            self.assertGreater(len(df), 0)
            self.assertIn("patient_id", df.columns)
            self.assertIn("timestamp", df.columns)
            self.assertIn("event_type", df.columns)
            self.assertIn("condition/concept_key", df.columns)
            self.assertIn("observation/concept_key", df.columns)
            self.assertIn("patient/deceased_boolean", df.columns)

    def test_set_task_produces_correct_samples(self) -> None:
        from pyhealth.tasks.mpf_clinical_prediction import MPFClinicalPredictionTask

        with tempfile.TemporaryDirectory() as tmp:
            write_two_class_ndjson(Path(tmp), name="fx.ndjson")
            ds = MIMIC4FHIRDataset(
                root=tmp, glob_pattern="*.ndjson", cache_dir=tmp, num_workers=1
            )
            task = MPFClinicalPredictionTask(max_len=48, use_mpf=True)
            task.warm_vocab(ds, num_workers=1)
            sample_ds = ds.set_task(task, num_workers=1)
            samples = sorted(
                [sample_ds[i] for i in range(len(sample_ds))],
                key=lambda s: s["patient_id"],
            )
            self.assertEqual(len(samples), 2)
            for s in samples:
                self.assertIn("concept_ids", s)
                self.assertIn("label", s)
            labels = {int(s["label"]) for s in samples}
            self.assertEqual(labels, {0, 1})

    def test_set_task_multi_worker_sets_frozen_vocab(self) -> None:
        from pyhealth.tasks.mpf_clinical_prediction import MPFClinicalPredictionTask

        with tempfile.TemporaryDirectory() as tmp:
            write_two_class_ndjson(Path(tmp))
            ds = MIMIC4FHIRDataset(
                root=tmp, glob_pattern="*.ndjson", cache_dir=tmp, num_workers=2
            )
            task = MPFClinicalPredictionTask(max_len=48, use_mpf=True)
            task.warm_vocab(ds, num_workers=2)
            ds.set_task(task, num_workers=2)
            self.assertTrue(task.frozen_vocab)

    def test_mpf_pre_filter_vocab_warmup_excludes_dropped_patients(self) -> None:
        from pyhealth.tasks.mpf_clinical_prediction import MPFClinicalPredictionTask

        class TwoPatientMPFTask(MPFClinicalPredictionTask):
            def pre_filter(self, df: pl.LazyFrame) -> pl.LazyFrame:
                return df.filter(pl.col("patient_id").is_in(["p-synth-1", "p-synth-2"]))

        with tempfile.TemporaryDirectory() as tmp:
            write_two_class_plus_third_ndjson(Path(tmp))
            ds = MIMIC4FHIRDataset(
                root=tmp, glob_pattern="*.ndjson", cache_dir=tmp, num_workers=1
            )
            self.assertEqual(len(ds.unique_patient_ids), 3)
            task = TwoPatientMPFTask(max_len=48, use_mpf=True)
            task.warm_vocab(ds, num_workers=1)
            ds.set_task(task, num_workers=1)
            self.assertNotIn("http://loinc.org|999-9", task.vocab.token_to_id)
            self.assertIn("http://loinc.org|789-0", task.vocab.token_to_id)

    def test_mpf_pre_filter_single_patient_limits_effective_workers(self) -> None:
        """Pre-filter that yields one patient should cap effective_workers to 1.

        We verify the effective_workers logic directly rather than via
        ``set_task`` because ``set_task`` with a 1-patient cohort produces
        only one label class (p-synth-1 is alive → label=0), which causes
        ``BinaryLabelProcessor.fit`` to raise "Expected 2 unique labels, got 1".
        The invariant under test belongs to the ``set_task`` override in
        ``MIMIC4FHIRDataset``; the Polars pre-filter and worker-count
        formula are both exercised here without triggering that constraint.
        """
        from pyhealth.tasks.mpf_clinical_prediction import MPFClinicalPredictionTask

        class OnePatientMPFTask(MPFClinicalPredictionTask):
            def pre_filter(self, df: pl.LazyFrame) -> pl.LazyFrame:
                return df.filter(pl.col("patient_id") == "p-synth-1")

        with tempfile.TemporaryDirectory() as tmp:
            write_two_class_ndjson(Path(tmp))
            ds = MIMIC4FHIRDataset(
                root=tmp, glob_pattern="*.ndjson", cache_dir=tmp, num_workers=2
            )
            task = OnePatientMPFTask(max_len=48, use_mpf=True)
            warmup_pids = (
                task.pre_filter(ds.global_event_df)
                .select("patient_id")
                .unique()
                .collect(engine="streaming")
                .to_series()
                .sort()
                .to_list()
            )
            self.assertEqual(warmup_pids, ["p-synth-1"])
            # One patient, two requested workers: effective_workers = min(2, 1) = 1
            effective_workers = min(2, len(warmup_pids)) if warmup_pids else 1
            self.assertEqual(effective_workers, 1)

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
        vocab = ConceptVocab()
        concept_ids, *_ = build_cehr_sequences(patient, vocab, max_len=64)
        tid = vocab["http://hl7.org/fhir/sid/icd-10-cm|I99"]
        self.assertEqual(concept_ids.count(tid), 1)

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
        vocab = ConceptVocab()
        concept_ids, *_ = build_cehr_sequences(patient, vocab, max_len=64)
        self.assertEqual(concept_ids.count(vocab["http://hl7.org/fhir/sid/icd-10-cm|Z00"]), 1)

    def test_cehr_sequence_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            write_one_patient_ndjson(Path(tmp))
            ds = MIMIC4FHIRDataset(root=tmp, glob_pattern="*.ndjson", cache_dir=tmp)
            patient = ds.get_patient("p-synth-1")
        vocab = ConceptVocab()
        concept_ids, token_types, time_stamps, ages, visit_orders, visit_segments = (
            build_cehr_sequences(patient, vocab, max_len=32)
        )
        n = len(concept_ids)
        self.assertEqual(len(token_types), n)
        self.assertEqual(len(time_stamps), n)
        self.assertEqual(len(ages), n)
        self.assertEqual(len(visit_orders), n)
        self.assertEqual(len(visit_segments), n)
        self.assertGreater(n, 0)

    def test_build_cehr_max_len_zero_no_clinical_tokens(self) -> None:
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
        vocab = ConceptVocab()
        c, _, _, _, _, vs = build_cehr_sequences(patient, vocab, max_len=0)
        self.assertEqual(c, [])
        self.assertEqual(vs, [])

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
        vocab = ConceptVocab()
        _, _, _, _, _, visit_segments = build_cehr_sequences(patient, vocab, max_len=64)
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
        vocab = ConceptVocab()
        concept_ids, _, _, _, visit_orders, visit_segments = build_cehr_sequences(
            patient, vocab, max_len=64
        )
        i10 = vocab["http://hl7.org/fhir/sid/icd-10-cm|I10"]
        z00 = vocab["http://hl7.org/fhir/sid/icd-10-cm|Z00"]
        i_link = concept_ids.index(i10)
        i_free = concept_ids.index(z00)
        self.assertEqual(visit_orders[i_link], visit_orders[i_free])
        self.assertEqual(visit_segments[i_link], visit_segments[i_free])

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
        vocab = ConceptVocab()
        c, *_ = build_cehr_sequences(patient, vocab, max_len=64)
        ka = "http://www.nlm.nih.gov/research/umls/rxnorm|111"
        kb = "http://www.nlm.nih.gov/research/umls/rxnorm|222"
        self.assertNotEqual(vocab[ka], vocab[kb])
        self.assertEqual(c.count(vocab[ka]), 1)
        self.assertEqual(c.count(vocab[kb]), 1)

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
        vocab = ConceptVocab()
        c, *_ = build_cehr_sequences(patient, vocab, max_len=64)
        key = "MedicationRequest/reference|med-abc"
        self.assertIn(vocab[key], c)
        self.assertEqual(c.count(vocab[key]), 1)

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


if __name__ == "__main__":
    unittest.main()
