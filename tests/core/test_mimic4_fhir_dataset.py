import gzip
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List

import orjson
import polars as pl

from pyhealth.datasets import MIMIC4FHIRDataset
from pyhealth.datasets.mimic4_fhir import (
    ConceptVocab,
    FHIR_RESOURCE_JSON_COL,
    FHIR_EVENT_TYPE,
    build_cehr_sequences,
    fhir_patient_from_patient,
    infer_mortality_label,
    synthetic_mpf_two_patient_ndjson_text,
)

from tests.core.mimic4_fhir_ndjson_fixtures import (
    ndjson_two_class_text,
    write_one_patient_ndjson,
    write_two_class_ndjson,
)


def _third_patient_loinc_resources() -> List[Dict[str, Any]]:
    """Third synthetic patient with a LOINC code not present on p-synth-1/2."""

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
    """Two-class PhysioNet-style fixture plus an extra patient (LOINC 999-9)."""

    lines = synthetic_mpf_two_patient_ndjson_text().strip().split("\n")
    lines.extend(
        orjson.dumps(r).decode("utf-8") for r in _third_patient_loinc_resources()
    )
    path = directory / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class TestMIMIC4FHIRDataset(unittest.TestCase):
    def test_concept_vocab_from_json_empty_token_to_id(self) -> None:
        """Corrupted save with empty ``token_to_id`` must not call ``max()`` on []."""

        v = ConceptVocab.from_json({"token_to_id": {}})
        self.assertIn("<pad>", v.token_to_id)
        self.assertIn("<unk>", v.token_to_id)
        self.assertEqual(v._next_id, 2)

    def test_concept_vocab_from_json_empty_respects_next_id(self) -> None:
        v = ConceptVocab.from_json({"token_to_id": {}, "next_id": 50})
        self.assertEqual(v._next_id, 50)

    def test_disk_fixture_resolves_events_per_patient(self) -> None:
        """NDJSON on disk → Parquet cache carries multiple rows for ``p-synth-1``."""

        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            write_one_patient_ndjson(tdir)
            ds = MIMIC4FHIRDataset(root=tmp, glob_pattern="*.ndjson", cache_dir=tmp)
            sub = (
                ds.global_event_df.filter(pl.col("patient_id") == "p-synth-1")
                .collect(engine="streaming")
            )
            self.assertGreaterEqual(len(sub), 2)

    def test_build_cehr_non_empty(self) -> None:
        from pyhealth.tasks.mpf_clinical_prediction import MPFClinicalPredictionTask

        with tempfile.TemporaryDirectory() as tmp:
            write_one_patient_ndjson(Path(tmp))
            ds = MIMIC4FHIRDataset(root=tmp, glob_pattern="*.ndjson", cache_dir=tmp)
            task = MPFClinicalPredictionTask(max_len=64, use_mpf=True)
            ds.gather_samples(task)
            self.assertIsInstance(ds.vocab, ConceptVocab)
            self.assertGreater(ds.vocab.vocab_size, 2)

    def test_set_task_vocab_warm_on_litdata_cache_hit(self) -> None:
        """MPF ``set_task`` must fill ``ds.vocab`` even when ``_task_transform`` skips."""

        from pyhealth.tasks.mpf_clinical_prediction import MPFClinicalPredictionTask

        with tempfile.TemporaryDirectory() as tmp:
            write_two_class_ndjson(Path(tmp))
            task_kw = {"max_len": 64, "use_mpf": True}
            ds1 = MIMIC4FHIRDataset(root=tmp, glob_pattern="*.ndjson", cache_dir=tmp)
            ds1.set_task(MPFClinicalPredictionTask(**task_kw), num_workers=1)
            warm_size = ds1.vocab.vocab_size
            self.assertGreater(
                warm_size, 6, "fixture plus MPF specials should exceed pad/unk only"
            )
            ds2 = MIMIC4FHIRDataset(root=tmp, glob_pattern="*.ndjson", cache_dir=tmp)
            ds2.set_task(MPFClinicalPredictionTask(**task_kw), num_workers=1)
            self.assertEqual(ds2.vocab.vocab_size, warm_size)

    def test_mortality_heuristic(self) -> None:
        from pyhealth.tasks.mpf_clinical_prediction import MPFClinicalPredictionTask

        with tempfile.TemporaryDirectory() as tmp:
            write_two_class_ndjson(Path(tmp))
            ds = MIMIC4FHIRDataset(root=tmp, glob_pattern="*.ndjson", cache_dir=tmp)
            task = MPFClinicalPredictionTask(max_len=64, use_mpf=False)
            samples = ds.gather_samples(task)
            labels = {s["label"] for s in samples}
            self.assertEqual(labels, {0, 1})

    def test_infer_deceased(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            write_two_class_ndjson(Path(tmp))
            ds = MIMIC4FHIRDataset(root=tmp, glob_pattern="*.ndjson", cache_dir=tmp)
            dead = fhir_patient_from_patient(ds.get_patient("p-synth-2"))
            self.assertEqual(infer_mortality_label(dead), 1)

    def test_disk_ndjson_gz_physionet_style(self) -> None:
        """Gzip NDJSON (PhysioNet ``*.ndjson.gz``) matches default glob when set."""

        with tempfile.TemporaryDirectory() as tmp:
            gz_path = Path(tmp) / "fixture.ndjson.gz"
            with gzip.open(gz_path, "wt", encoding="utf-8") as gz:
                gz.write(ndjson_two_class_text())
            ds = MIMIC4FHIRDataset(
                root=tmp, glob_pattern="*.ndjson.gz", max_patients=5
            )
            self.assertGreaterEqual(len(ds.unique_patient_ids), 1)

    def test_disk_ndjson_temp_dir(self) -> None:
        """Load from a temp directory (cleanup via context manager)."""

        with tempfile.TemporaryDirectory() as tmp:
            write_two_class_ndjson(Path(tmp))
            ds = MIMIC4FHIRDataset(
                root=tmp, glob_pattern="*.ndjson", max_patients=5
            )
            self.assertEqual(len(ds.unique_patient_ids), 2)
            from pyhealth.tasks.mpf_clinical_prediction import (
                MPFClinicalPredictionTask,
            )

            task = MPFClinicalPredictionTask(max_len=48, use_mpf=True)
            samples = ds.gather_samples(task)
            self.assertGreaterEqual(len(samples), 1)
            for s in samples:
                self.assertIn("concept_ids", s)
                self.assertIn("label", s)

    def test_sharded_ingest_sorted_patient_ids_multi_part_cache(self) -> None:
        """Hash shards → ``part-*.parquet``; patient ids exposed in sorted order."""

        with tempfile.TemporaryDirectory() as tmp:
            write_two_class_ndjson(Path(tmp))
            ds = MIMIC4FHIRDataset(
                root=tmp,
                glob_pattern="*.ndjson",
                cache_dir=tmp,
                ingest_num_shards=8,
            )
            ids = ds.unique_patient_ids
            self.assertEqual(ids, sorted(ids))
            self.assertEqual(set(ids), {"p-synth-1", "p-synth-2"})
            part_dir = ds.cache_dir / "global_event_df.parquet"
            parts = sorted(part_dir.glob("part-*.parquet"))
            self.assertGreaterEqual(len(parts), 1)
            # ``p-synth-1`` / ``p-synth-2`` crc32 to different slots for 8 shards.
            self.assertGreaterEqual(len(parts), 2)

    def test_global_event_df_schema_and_streaming_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            write_two_class_ndjson(Path(tmp))
            ds = MIMIC4FHIRDataset(
                root=tmp,
                glob_pattern="*.ndjson",
                cache_dir=tmp,
                max_patients=5,
            )
            lf = ds.global_event_df
            df = lf.collect(engine="streaming")
            self.assertGreater(len(df), 0)
            self.assertIn("patient_id", df.columns)
            self.assertIn("timestamp", df.columns)
            self.assertIn("event_type", df.columns)
            self.assertIn(FHIR_RESOURCE_JSON_COL, df.columns)
            self.assertTrue((df["event_type"] == FHIR_EVENT_TYPE).all())

    def test_set_task_parity_with_gather_samples_ndjson(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            write_two_class_ndjson(Path(tmp), name="fx.ndjson")
            from pyhealth.tasks.mpf_clinical_prediction import (
                MPFClinicalPredictionTask,
            )

            ds = MIMIC4FHIRDataset(
                root=tmp, glob_pattern="*.ndjson", cache_dir=tmp, num_workers=1
            )
            task = MPFClinicalPredictionTask(max_len=48, use_mpf=True)
            ref = sorted(ds.gather_samples(task), key=lambda s: s["patient_id"])
            task2 = MPFClinicalPredictionTask(max_len=48, use_mpf=True)
            sample_ds = ds.set_task(task2, num_workers=1)
            got = sorted(
                [sample_ds[i] for i in range(len(sample_ds))],
                key=lambda s: s["patient_id"],
            )
            self.assertEqual(len(got), len(ref))
            for a, b in zip(ref, got):
                self.assertEqual(a["label"], int(b["label"]))
                ac = a["concept_ids"]
                bc = b["concept_ids"]
                if hasattr(bc, "tolist"):
                    bc = bc.tolist()
                self.assertEqual(ac, bc)

    def test_gather_samples_resets_frozen_vocab_after_set_task(self) -> None:
        """Reusing the same task after ``set_task`` must grow a new dataset's vocab."""

        with tempfile.TemporaryDirectory() as tmp_a, tempfile.TemporaryDirectory() as tmp_b:
            write_two_class_ndjson(Path(tmp_a), name="a.ndjson")
            write_two_class_ndjson(Path(tmp_b), name="b.ndjson")
            from pyhealth.tasks.mpf_clinical_prediction import (
                MPFClinicalPredictionTask,
            )

            ds_a = MIMIC4FHIRDataset(
                root=tmp_a, glob_pattern="*.ndjson", cache_dir=tmp_a, num_workers=1
            )
            ds_b = MIMIC4FHIRDataset(
                root=tmp_b, glob_pattern="*.ndjson", cache_dir=tmp_b, num_workers=1
            )
            task = MPFClinicalPredictionTask(max_len=48, use_mpf=True)
            ds_a.set_task(task, num_workers=1)
            # Single-process transform: vocab grows during caching; no pre-warm pass.
            self.assertFalse(task.frozen_vocab)

            ref = sorted(
                ds_b.gather_samples(MPFClinicalPredictionTask(max_len=48, use_mpf=True)),
                key=lambda s: s["patient_id"],
            )
            got = sorted(ds_b.gather_samples(task), key=lambda s: s["patient_id"])
            self.assertEqual(len(got), len(ref))
            for a, b in zip(ref, got):
                self.assertEqual(a["label"], b["label"])
                ac = a["concept_ids"]
                bc = b["concept_ids"]
                if hasattr(bc, "tolist"):
                    bc = bc.tolist()
                self.assertEqual(ac, bc)

    def test_set_task_multi_worker_sets_frozen_vocab(self) -> None:
        """``effective_workers > 1`` requires main-process warmup and frozen ids."""

        with tempfile.TemporaryDirectory() as tmp:
            write_two_class_ndjson(Path(tmp))
            from pyhealth.tasks.mpf_clinical_prediction import (
                MPFClinicalPredictionTask,
            )

            ds = MIMIC4FHIRDataset(
                root=tmp, glob_pattern="*.ndjson", cache_dir=tmp, num_workers=2
            )
            task = MPFClinicalPredictionTask(max_len=48, use_mpf=True)
            ds.set_task(task, num_workers=2)
            self.assertTrue(task.frozen_vocab)

    def test_mpf_pre_filter_vocab_warmup_excludes_dropped_patients(self) -> None:
        """Warmup must not deserialize patients omitted by ``pre_filter``."""

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
            ds.set_task(task, num_workers=1)
            self.assertNotIn("http://loinc.org|999-9", ds.vocab.token_to_id)
            self.assertIn("http://loinc.org|789-0", ds.vocab.token_to_id)

    def test_mpf_pre_filter_patient_ids_drive_effective_workers(self) -> None:
        """``len(_mpf_patient_ids_for_task)`` must match ``_task_transform`` slicing."""

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
            warmup_pids = ds._mpf_patient_ids_for_task(task)
            self.assertEqual(warmup_pids, ["p-synth-1"])
            nw = 2
            pid_n = len(warmup_pids)
            effective_workers = min(nw, pid_n) if pid_n else 1
            self.assertEqual(effective_workers, 1)
            self.assertFalse(effective_workers > 1)

    def test_encounter_reference_requires_exact_id(self) -> None:
        """``e1`` must not match reference ``Encounter/e10`` (substring bug)."""

        from pyhealth.datasets.mimic4_fhir import FHIRPatient

        patient_r = {
            "resourceType": "Patient",
            "id": "p1",
            "birthDate": "1950-01-01",
        }
        enc1 = {
            "resourceType": "Encounter",
            "id": "e1",
            "subject": {"reference": "Patient/p1"},
            "period": {"start": "2020-06-01T10:00:00Z"},
            "class": {"code": "AMB"},
        }
        enc10 = {
            "resourceType": "Encounter",
            "id": "e10",
            "subject": {"reference": "Patient/p1"},
            "period": {"start": "2020-07-02T10:00:00Z"},
            "class": {"code": "IMP"},
        }
        cond_e10 = {
            "resourceType": "Condition",
            "id": "c99",
            "subject": {"reference": "Patient/p1"},
            "encounter": {"reference": "Encounter/e10"},
            "code": {
                "coding": [{"system": "http://hl7.org/fhir/sid/icd-10-cm", "code": "I99"}]
            },
            "onsetDateTime": "2020-07-02T11:00:00Z",
        }
        pr = FHIRPatient(
            patient_id="p1",
            resources=[patient_r, enc1, enc10, cond_e10],
        )
        vocab = ConceptVocab()
        concept_ids, *_ = build_cehr_sequences(pr, vocab, max_len=64)
        tid = vocab["http://hl7.org/fhir/sid/icd-10-cm|I99"]
        self.assertEqual(concept_ids.count(tid), 1)

    def test_unlinked_condition_emitted_once_with_two_encounters(self) -> None:
        """No encounter.reference: must not duplicate once per encounter loop."""

        from pyhealth.datasets.mimic4_fhir import FHIRPatient

        patient_r = {
            "resourceType": "Patient",
            "id": "p1",
            "birthDate": "1950-01-01",
        }
        enc_a = {
            "resourceType": "Encounter",
            "id": "ea",
            "subject": {"reference": "Patient/p1"},
            "period": {"start": "2020-06-01T10:00:00Z"},
            "class": {"code": "AMB"},
        }
        enc_b = {
            "resourceType": "Encounter",
            "id": "eb",
            "subject": {"reference": "Patient/p1"},
            "period": {"start": "2020-07-01T10:00:00Z"},
            "class": {"code": "IMP"},
        }
        cond = {
            "resourceType": "Condition",
            "id": "cx",
            "subject": {"reference": "Patient/p1"},
            "code": {
                "coding": [{"system": "http://hl7.org/fhir/sid/icd-10-cm", "code": "Z00"}]
            },
            "onsetDateTime": "2020-06-15T12:00:00Z",
        }
        pr = FHIRPatient(
            patient_id="p1",
            resources=[patient_r, enc_a, enc_b, cond],
        )
        vocab = ConceptVocab()
        concept_ids, *_ = build_cehr_sequences(pr, vocab, max_len=64)
        z00 = vocab["http://hl7.org/fhir/sid/icd-10-cm|Z00"]
        self.assertEqual(concept_ids.count(z00), 1)

    def test_cehr_sequence_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            write_one_patient_ndjson(Path(tmp))
            ds = MIMIC4FHIRDataset(root=tmp, glob_pattern="*.ndjson", cache_dir=tmp)
            p = fhir_patient_from_patient(ds.get_patient("p-synth-1"))
        v = ConceptVocab()
        c, tt, ts, ag, vo, vs = build_cehr_sequences(p, v, max_len=32)
        n = len(c)
        self.assertEqual(len(tt), n)
        self.assertEqual(len(ts), n)
        self.assertGreater(n, 0)

    def test_build_cehr_max_len_zero_no_clinical_tokens(self) -> None:
        """``max_len=0`` must not use ``events[-0:]`` (full list); emit nothing."""

        from pyhealth.datasets.mimic4_fhir import FHIRPatient

        patient_r = {
            "resourceType": "Patient",
            "id": "p1",
            "birthDate": "1950-01-01",
        }
        enc = {
            "resourceType": "Encounter",
            "id": "e1",
            "subject": {"reference": "Patient/p1"},
            "period": {"start": "2020-06-01T10:00:00Z"},
            "class": {"code": "AMB"},
        }
        cond = {
            "resourceType": "Condition",
            "id": "c1",
            "subject": {"reference": "Patient/p1"},
            "encounter": {"reference": "Encounter/e1"},
            "code": {
                "coding": [{"system": "http://hl7.org/fhir/sid/icd-10-cm", "code": "I10"}]
            },
            "onsetDateTime": "2020-06-01T11:00:00Z",
        }
        pr = FHIRPatient(patient_id="p1", resources=[patient_r, enc, cond])
        vocab = ConceptVocab()
        c, tt, ts, ag, vo, vs = build_cehr_sequences(pr, vocab, max_len=0)
        self.assertEqual(c, [])
        self.assertEqual(vs, [])

    def test_visit_segments_alternate_by_visit_index(self) -> None:
        """CEHR-style segments: all tokens in visit ``k`` share ``k % 2``."""

        from pyhealth.datasets.mimic4_fhir import FHIRPatient

        patient_r = {
            "resourceType": "Patient",
            "id": "p1",
            "birthDate": "1950-01-01",
        }
        enc0 = {
            "resourceType": "Encounter",
            "id": "e0",
            "subject": {"reference": "Patient/p1"},
            "period": {"start": "2020-06-01T10:00:00Z"},
            "class": {"code": "AMB"},
        }
        enc1 = {
            "resourceType": "Encounter",
            "id": "e1",
            "subject": {"reference": "Patient/p1"},
            "period": {"start": "2020-07-01T10:00:00Z"},
            "class": {"code": "IMP"},
        }
        c0 = {
            "resourceType": "Condition",
            "id": "c0",
            "subject": {"reference": "Patient/p1"},
            "encounter": {"reference": "Encounter/e0"},
            "code": {
                "coding": [{"system": "http://hl7.org/fhir/sid/icd-10-cm", "code": "I10"}]
            },
            "onsetDateTime": "2020-06-01T11:00:00Z",
        }
        c1 = {
            "resourceType": "Condition",
            "id": "c1",
            "subject": {"reference": "Patient/p1"},
            "encounter": {"reference": "Encounter/e1"},
            "code": {
                "coding": [{"system": "http://hl7.org/fhir/sid/icd-10-cm", "code": "I20"}]
            },
            "onsetDateTime": "2020-07-01T11:00:00Z",
        }
        pr = FHIRPatient(
            patient_id="p1",
            resources=[patient_r, enc0, enc1, c0, c1],
        )
        vocab = ConceptVocab()
        _, _, _, _, _, vs = build_cehr_sequences(pr, vocab, max_len=64)
        self.assertEqual(len(vs), 4)
        self.assertEqual(vs, [0, 0, 1, 1])

    def test_unlinked_visit_idx_matches_sequential_counter(self) -> None:
        """Skipped encounters (no ``period.start``) must not shift unlinked ``visit_idx``."""

        from pyhealth.datasets.mimic4_fhir import FHIRPatient

        patient_r = {
            "resourceType": "Patient",
            "id": "p1",
            "birthDate": "1950-01-01",
        }
        enc_no_start = {
            "resourceType": "Encounter",
            "id": "e_bad",
            "subject": {"reference": "Patient/p1"},
            "class": {"code": "AMB"},
        }
        enc_ok = {
            "resourceType": "Encounter",
            "id": "e_ok",
            "subject": {"reference": "Patient/p1"},
            "period": {"start": "2020-03-01T10:00:00Z"},
            "class": {"code": "IMP"},
        }
        cond_linked = {
            "resourceType": "Condition",
            "id": "c_link",
            "subject": {"reference": "Patient/p1"},
            "encounter": {"reference": "Encounter/e_ok"},
            "code": {
                "coding": [{"system": "http://hl7.org/fhir/sid/icd-10-cm", "code": "I10"}]
            },
            "onsetDateTime": "2020-03-05T11:00:00Z",
        }
        cond_unlinked = {
            "resourceType": "Condition",
            "id": "c_free",
            "subject": {"reference": "Patient/p1"},
            "code": {
                "coding": [{"system": "http://hl7.org/fhir/sid/icd-10-cm", "code": "Z00"}]
            },
            "onsetDateTime": "2020-03-15T12:00:00Z",
        }
        pr = FHIRPatient(
            patient_id="p1",
            resources=[patient_r, enc_no_start, enc_ok, cond_linked, cond_unlinked],
        )
        vocab = ConceptVocab()
        c, _, _, _, vo, vs = build_cehr_sequences(pr, vocab, max_len=64)
        i10 = vocab["http://hl7.org/fhir/sid/icd-10-cm|I10"]
        z00 = vocab["http://hl7.org/fhir/sid/icd-10-cm|Z00"]
        i_link = c.index(i10)
        i_free = c.index(z00)
        self.assertEqual(vo[i_link], vo[i_free])
        self.assertEqual(vs[i_link], vs[i_free])

    def test_medication_request_uses_medication_codeable_concept(self) -> None:
        """FHIR R4 MedicationRequest carries Rx in ``medicationCodeableConcept``, not ``code``."""

        from pyhealth.datasets.mimic4_fhir import FHIRPatient

        patient_r = {
            "resourceType": "Patient",
            "id": "p1",
            "birthDate": "1950-01-01",
        }
        enc = {
            "resourceType": "Encounter",
            "id": "e1",
            "subject": {"reference": "Patient/p1"},
            "period": {"start": "2020-06-01T10:00:00Z"},
            "class": {"code": "IMP"},
        }
        mr_a = {
            "resourceType": "MedicationRequest",
            "id": "m1",
            "subject": {"reference": "Patient/p1"},
            "encounter": {"reference": "Encounter/e1"},
            "authoredOn": "2020-06-01T11:00:00Z",
            "medicationCodeableConcept": {
                "coding": [
                    {
                        "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                        "code": "111",
                    }
                ]
            },
        }
        mr_b = {
            "resourceType": "MedicationRequest",
            "id": "m2",
            "subject": {"reference": "Patient/p1"},
            "encounter": {"reference": "Encounter/e1"},
            "authoredOn": "2020-06-01T12:00:00Z",
            "medicationCodeableConcept": {
                "coding": [
                    {
                        "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                        "code": "222",
                    }
                ]
            },
        }
        pr = FHIRPatient(
            patient_id="p1",
            resources=[patient_r, enc, mr_a, mr_b],
        )
        vocab = ConceptVocab()
        c, _, _, _, _, _ = build_cehr_sequences(pr, vocab, max_len=64)
        ka = "http://www.nlm.nih.gov/research/umls/rxnorm|111"
        kb = "http://www.nlm.nih.gov/research/umls/rxnorm|222"
        self.assertNotEqual(vocab[ka], vocab[kb])
        self.assertEqual(c.count(vocab[ka]), 1)
        self.assertEqual(c.count(vocab[kb]), 1)

    def test_medication_request_medication_reference_token(self) -> None:
        """When only ``medicationReference`` is present, use a stable ref-based key."""

        from pyhealth.datasets.mimic4_fhir import FHIRPatient

        patient_r = {
            "resourceType": "Patient",
            "id": "p1",
            "birthDate": "1950-01-01",
        }
        enc = {
            "resourceType": "Encounter",
            "id": "e1",
            "subject": {"reference": "Patient/p1"},
            "period": {"start": "2020-06-01T10:00:00Z"},
            "class": {"code": "IMP"},
        }
        mr = {
            "resourceType": "MedicationRequest",
            "id": "m1",
            "subject": {"reference": "Patient/p1"},
            "encounter": {"reference": "Encounter/e1"},
            "authoredOn": "2020-06-01T11:00:00Z",
            "medicationReference": {"reference": "Medication/med-abc"},
        }
        pr = FHIRPatient(patient_id="p1", resources=[patient_r, enc, mr])
        vocab = ConceptVocab()
        c, _, _, _, _, _ = build_cehr_sequences(pr, vocab, max_len=64)
        key = "MedicationRequest/reference|med-abc"
        self.assertIn(vocab[key], c)
        self.assertEqual(c.count(vocab[key]), 1)


if __name__ == "__main__":
    unittest.main()
