import unittest

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


if __name__ == "__main__":
    unittest.main()
