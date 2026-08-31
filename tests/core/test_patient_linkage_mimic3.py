import unittest
from pathlib import Path

from pyhealth.datasets import MIMIC3Dataset
from pyhealth.models.medlink import convert_to_ir_format
from pyhealth.tasks import PatientLinkageMIMIC3Task


class TestPatientLinkageMIMIC3Task(unittest.TestCase):
    """Regression test: PatientLinkageMIMIC3Task's input_schema previously
    used processor-type strings ("integer", "string", "datetime") that
    aren't registered in pyhealth.processors, so dataset.set_task() failed
    immediately with ValueError: Unknown processor. Fixed to use "raw"
    (pass-through), the correct processor for these metadata fields.
    """

    @classmethod
    def setUpClass(cls):
        test_dir = Path(__file__).parent.parent.parent
        root = str(test_dir / "test-resources" / "core" / "mimic3demo")
        cls.dataset = MIMIC3Dataset(
            root=root, tables=["diagnoses_icd", "admissions", "patients"]
        )

    def test_set_task_runs_without_error(self):
        # This is the direct regression check: set_task() used to raise
        # ValueError: Unknown processor: integer before reaching any patient.
        sample_dataset = self.dataset.set_task(PatientLinkageMIMIC3Task())
        self.assertGreater(len(sample_dataset), 0)

    def test_raw_call_produces_query_and_history_pair(self):
        # Patient 44083 has 3 admissions; the query is the last (198330),
        # and the history side concatenates the two earlier ones
        # (125157, then 131048) with a single [SEP] separator.
        patient = self.dataset.get_patient("44083")
        samples = PatientLinkageMIMIC3Task()(patient)
        self.assertEqual(len(samples), 1)
        sample = samples[0]

        self.assertEqual(sample["visit_id"], "198330")
        self.assertEqual(sample["d_visit_id"], "131048")
        self.assertEqual(sample["d_visit_ids"], "125157|131048")
        self.assertIn("[SEP]", sample["d_conditions"])
        self.assertGreater(len(sample["conditions"]), 0)
        self.assertGreater(len(sample["d_conditions"]), 0)

    def test_single_prior_admission_has_no_separator(self):
        # Patient 10094 has exactly one prior admission (168074) before the
        # query, so d_conditions should have no [SEP] token and a single
        # entry in d_visit_ids.
        patient = self.dataset.get_patient("10094")
        samples = PatientLinkageMIMIC3Task()(patient)
        self.assertEqual(len(samples), 1)
        sample = samples[0]

        self.assertEqual(sample["d_visit_ids"], "168074")
        self.assertNotIn("[SEP]", sample["d_conditions"])

    def test_convert_to_ir_format_accepts_sample_dataset_directly(self):
        # Regression check for the patient_linkage_mimic3_medlink.py example:
        # it used to call convert_to_ir_format(sample_dataset.samples), but
        # SampleDataset has no .samples attribute (only __iter__/__getitem__/
        # __len__), so that line raised AttributeError even after the
        # processor-name fix above. convert_to_ir_format only iterates its
        # argument, so the SampleDataset itself must work directly.
        sample_dataset = self.dataset.set_task(PatientLinkageMIMIC3Task())
        corpus, queries, qrels, corpus_meta, queries_meta = convert_to_ir_format(
            sample_dataset
        )
        self.assertEqual(len(queries), len(sample_dataset))
        self.assertEqual(set(queries.keys()), set(qrels.keys()))
        self.assertEqual(set(queries.keys()), set(queries_meta.keys()))
        self.assertTrue(corpus)
        self.assertEqual(set(corpus.keys()), set(corpus_meta.keys()))


if __name__ == "__main__":
    unittest.main()
