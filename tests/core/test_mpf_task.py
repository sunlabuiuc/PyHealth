import unittest

from pyhealth.datasets import (
    build_fhir_sample_dataset_from_lines,
    synthetic_ndjson_lines_two_class,
)
from pyhealth.tasks.mpf_clinical_prediction import MPFClinicalPredictionTask


class TestMPFClinicalPredictionTask(unittest.TestCase):
    def test_mpf_sets_boundary_tokens(self) -> None:
        task = MPFClinicalPredictionTask(max_len=32, use_mpf=True)
        _, vocab, samples = build_fhir_sample_dataset_from_lines(
            synthetic_ndjson_lines_two_class(), task
        )
        self.assertGreater(len(samples), 0)
        s0 = samples[0]
        mor = vocab["<mor>"]
        reg = vocab["<reg>"]
        self.assertEqual(s0["concept_ids"][0], mor)
        self.assertEqual(s0["concept_ids"][-1], reg)

    def test_no_mpf_uses_cls_reg(self) -> None:
        task = MPFClinicalPredictionTask(max_len=32, use_mpf=False)
        _, vocab, samples = build_fhir_sample_dataset_from_lines(
            synthetic_ndjson_lines_two_class(), task
        )
        s0 = samples[0]
        cls_id = vocab["<cls>"]
        reg_id = vocab["<reg>"]
        self.assertEqual(s0["concept_ids"][0], cls_id)
        self.assertEqual(s0["concept_ids"][-1], reg_id)

    def test_schema_keys(self) -> None:
        task = MPFClinicalPredictionTask(max_len=16, use_mpf=True)
        _, _, samples = build_fhir_sample_dataset_from_lines(
            synthetic_ndjson_lines_two_class(), task
        )
        for k in task.input_schema:
            self.assertIn(k, samples[0])
        self.assertIn("label", samples[0])


if __name__ == "__main__":
    unittest.main()
