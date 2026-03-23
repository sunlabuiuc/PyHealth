import gzip
import tempfile
import unittest
from pathlib import Path

from pyhealth.datasets import MIMIC4FHIRDataset
from pyhealth.datasets.mimic4_fhir import (
    ConceptVocab,
    build_cehr_sequences,
    build_fhir_sample_dataset_from_lines,
    group_resources_by_patient,
    infer_mortality_label,
    synthetic_ndjson_lines,
    synthetic_ndjson_lines_two_class,
)


class TestMIMIC4FHIRDataset(unittest.TestCase):
    def test_group_resources(self) -> None:
        lines = synthetic_ndjson_lines()
        resources = []
        for line in lines:
            import json

            resources.append(json.loads(line))
        g = group_resources_by_patient(resources)
        self.assertIn("p-synth-1", g)
        self.assertGreaterEqual(len(g["p-synth-1"]), 2)

    def test_build_cehr_non_empty(self) -> None:
        from pyhealth.tasks.mpf_clinical_prediction import MPFClinicalPredictionTask

        lines = synthetic_ndjson_lines()
        _, vocab, _ = build_fhir_sample_dataset_from_lines(
            lines,
            MPFClinicalPredictionTask(max_len=64, use_mpf=True),
        )
        self.assertIsInstance(vocab, ConceptVocab)
        self.assertGreater(vocab.vocab_size, 2)

    def test_mortality_heuristic(self) -> None:
        lines = synthetic_ndjson_lines_two_class()
        from pyhealth.tasks.mpf_clinical_prediction import MPFClinicalPredictionTask

        task = MPFClinicalPredictionTask(max_len=64, use_mpf=False)
        _, _, samples = build_fhir_sample_dataset_from_lines(lines, task)
        labels = {s["label"] for s in samples}
        self.assertEqual(labels, {0, 1})

    def test_infer_deceased(self) -> None:
        from pyhealth.datasets.mimic4_fhir import FHIRPatient, parse_ndjson_line

        lines = synthetic_ndjson_lines_two_class()
        resources = [parse_ndjson_line(x) for x in lines if parse_ndjson_line(x)]
        g = group_resources_by_patient(resources)
        dead = FHIRPatient(patient_id="p-synth-2", resources=g["p-synth-2"])
        self.assertEqual(infer_mortality_label(dead), 1)

    def test_disk_ndjson_gz_physionet_style(self) -> None:
        """Gzip NDJSON (PhysioNet ``*.ndjson.gz``) matches default glob when set."""

        lines = synthetic_ndjson_lines_two_class()
        with tempfile.TemporaryDirectory() as tmp:
            gz_path = Path(tmp) / "fixture.ndjson.gz"
            with gzip.open(gz_path, "wt", encoding="utf-8") as gz:
                gz.write("\n".join(lines) + "\n")
            ds = MIMIC4FHIRDataset(
                root=tmp, glob_pattern="*.ndjson.gz", max_patients=5
            )
            self.assertGreaterEqual(len(ds.unique_patient_ids), 1)

    def test_disk_ndjson_temp_dir(self) -> None:
        """Load from a temp directory (cleanup via context manager)."""

        lines = synthetic_ndjson_lines_two_class()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixture.ndjson"
            path.write_text("\n".join(lines), encoding="utf-8")
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

    def test_global_event_df_not_supported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ds = MIMIC4FHIRDataset(root=tmp, max_patients=2)
            with self.assertRaises(NotImplementedError):
                _ = ds.global_event_df

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
        lines = synthetic_ndjson_lines()
        from pyhealth.datasets.mimic4_fhir import FHIRPatient, parse_ndjson_line

        resources = [parse_ndjson_line(x) for x in lines if parse_ndjson_line(x)]
        g = group_resources_by_patient(resources)
        p = FHIRPatient(patient_id="p-synth-1", resources=g["p-synth-1"])
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


if __name__ == "__main__":
    unittest.main()
