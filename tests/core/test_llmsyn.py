import json
import os
import unittest
from unittest.mock import patch

import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import LLMSYNModel
from pyhealth.models.llmsyn import (
    _MockLLMBackend,
    _LLMSYNPipeline,
    _normalize_icd9,
    _normalize_cpt,
    _parse_codes,
    _parse_output,
    _DEFAULT_STATS,
)
from pyhealth.tasks import SyntheticEHRGenerationTask


def _make_dataset():
    samples = [
        {
            "patient_id": "p0",
            "visit_id": "v0",
            "conditions": ["4019", "41401"],
            "mortality": 0,
        },
        {
            "patient_id": "p1",
            "visit_id": "v1",
            "conditions": ["4280", "42731"],
            "mortality": 1,
        },
    ]
    return create_sample_dataset(
        samples=samples,
        input_schema={"conditions": "sequence"},
        output_schema={"mortality": "binary"},
        dataset_name="llmsyn_test",
    )


class TestCodeNormalization(unittest.TestCase):
    def test_normalize_icd9_already_prefixed(self):
        self.assertEqual(_normalize_icd9("ICD9:4019"), "ICD9:4019")

    def test_normalize_icd9_bare(self):
        self.assertEqual(_normalize_icd9("4019"), "ICD9:4019")

    def test_normalize_cpt_already_prefixed(self):
        self.assertEqual(_normalize_cpt("CPT:93000"), "CPT:93000")

    def test_normalize_cpt_bare(self):
        self.assertEqual(_normalize_cpt("93000"), "CPT:93000")

    def test_parse_codes_icd9(self):
        codes = _parse_codes("ICD9:4280,ICD9:42731", "ICD9")
        self.assertEqual(codes, ["ICD9:4280", "ICD9:42731"])

    def test_parse_codes_cpt(self):
        codes = _parse_codes("CPT:93000,CPT:36415", "CPT")
        self.assertEqual(codes, ["CPT:93000", "CPT:36415"])


class TestParseOutput(unittest.TestCase):
    def test_full_output(self):
        text = (
            "Age: 67\nGender: Female\nEthnicity: WHITE\n"
            "Insurance: Medicare\nSurvived: Yes\n"
            "MainDiagnosis: ICD9:41401\n"
            "Complications: ICD9:4280,ICD9:42731\n"
            "Procedures: CPT:93000,CPT:36415"
        )
        r = _parse_output(text)
        self.assertEqual(r["Age"], 67)
        self.assertEqual(r["Gender"], "Female")
        self.assertEqual(r["Survived"], "Yes")
        self.assertEqual(r["MainDiagnosis"], "ICD9:41401")
        self.assertIn("ICD9:4280", r["Complications"])
        self.assertIn("CPT:93000", r["Procedures"])

    def test_none_complications(self):
        r = _parse_output("Complications: None\nProcedures: None")
        self.assertEqual(r["Complications"], [])
        self.assertEqual(r["Procedures"], [])

    def test_bare_code_normalized(self):
        r = _parse_output("MainDiagnosis: 4019")
        self.assertEqual(r["MainDiagnosis"], "ICD9:4019")


class TestMockBackend(unittest.TestCase):
    def setUp(self):
        self.backend = _MockLLMBackend(seed=42)

    def test_demographics_step(self):
        out = self.backend.generate("Generate a synthetic EHR patient demographic profile")
        self.assertIn("Age:", out)
        self.assertIn("Gender:", out)
        self.assertIn("Survived:", out)

    def test_diagnosis_step(self):
        out = self.backend.generate("Generate the main ICD-9 diagnosis")
        self.assertIn("MainDiagnosis:", out)

    def test_complications_step(self):
        out = self.backend.generate("Generate complication ICD-9 codes")
        self.assertIn("Complications:", out)

    def test_procedures_step(self):
        out = self.backend.generate("Generate CPT procedure codes")
        self.assertIn("Procedures:", out)


class TestPipeline(unittest.TestCase):
    def setUp(self):
        backend = _MockLLMBackend(seed=0)
        self.pipeline = _LLMSYNPipeline(
            backend=backend,
            stats=_DEFAULT_STATS,
            noise_scale=0.0,
            prior_mode="full",
            enable_rag=False,
        )

    def test_generate_record_keys(self):
        record = self.pipeline.generate_record()
        for key in ["Age", "Gender", "Ethnicity", "Insurance",
                    "Survived", "MainDiagnosis", "Complications",
                    "Procedures", "Raw"]:
            self.assertIn(key, record)

    def test_generate_n(self):
        records = self.pipeline.generate_n(3)
        self.assertEqual(len(records), 3)

    def test_raw_contains_step_outputs(self):
        record = self.pipeline.generate_record()
        for key in ["step_0_raw", "step_1_raw", "step_2_raw", "step_3_raw"]:
            self.assertIn(key, record["Raw"])


class TestLLMSYNModel(unittest.TestCase):
    def setUp(self):
        self.dataset = _make_dataset()
        self.model = LLMSYNModel(
            dataset=self.dataset,
            llm_provider="mock",
            seed=42,
        )

    def test_forward_keys(self):
        loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        with torch.no_grad():
            out = self.model(**batch)
        self.assertIn("loss", out)
        self.assertIn("y_prob", out)
        self.assertIn("y_true", out)
        self.assertIn("logit", out)

    def test_forward_shapes(self):
        loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        with torch.no_grad():
            out = self.model(**batch)
        self.assertEqual(out["y_prob"].shape[0], 2)
        self.assertEqual(out["y_true"].shape[0], 2)
        self.assertEqual(out["loss"].dim(), 0)

    def test_generate(self):
        records = self.model.generate(n=3)
        self.assertEqual(len(records), 3)
        self.assertIn("MainDiagnosis", records[0])

    def test_stats_path(self):
        stats_path = os.path.join(
            os.path.dirname(__file__),
            "..", "..", "test-resources", "llmsyn", "stats.json",
        )
        stats_path = os.path.normpath(stats_path)
        if not os.path.exists(stats_path):
            self.skipTest("test-resources/llmsyn/stats.json not found")
        model = LLMSYNModel(
            dataset=self.dataset,
            llm_provider="mock",
            stats_path=stats_path,
            seed=0,
        )
        records = model.generate(n=1)
        self.assertEqual(len(records), 1)


class TestAblationVariants(unittest.TestCase):
    def setUp(self):
        self.dataset = _make_dataset()

    def _run_variant(self, prior_mode, enable_rag):
        model = LLMSYNModel(
            dataset=self.dataset,
            llm_provider="mock",
            prior_mode=prior_mode,
            enable_rag=enable_rag,
            seed=0,
        )
        return model.generate(n=1)

    def test_llmsyn_full(self):
        records = self._run_variant(prior_mode="full", enable_rag=True)
        self.assertEqual(len(records), 1)

    def test_llmsyn_prior(self):
        records = self._run_variant(prior_mode="full", enable_rag=False)
        self.assertEqual(len(records), 1)

    def test_llmsyn_base(self):
        records = self._run_variant(prior_mode="sampled", enable_rag=False)
        self.assertEqual(len(records), 1)


class TestRAG(unittest.TestCase):
    def setUp(self):
        backend = _MockLLMBackend(seed=42)
        self.pipeline = _LLMSYNPipeline(
            backend=backend,
            stats=_DEFAULT_STATS,
            noise_scale=0.0,
            prior_mode="full",
            enable_rag=True,
        )

    @patch(
        "pyhealth.datasets.llmsyn_utils.get_medical_knowledge",
        return_value="TEST_RAG_TEXT",
    )
    def test_rag_is_called(self, mock_rag):
        self.pipeline.generate_record()
        mock_rag.assert_called_once()

    @patch(
        "pyhealth.datasets.llmsyn_utils.get_medical_knowledge",
        return_value="TEST_RAG_TEXT",
    )
    def test_rag_text_in_prompt(self, mock_rag):
        captured = []
        original = self.pipeline._backend.generate

        def capturing(prompt, max_tokens=256):
            captured.append(prompt)
            return original(prompt, max_tokens)

        self.pipeline._backend.generate = capturing
        self.pipeline.generate_record()
        mock_rag.assert_called_once()
        self.assertTrue(any("TEST_RAG_TEXT" in p for p in captured))

    def test_rag_not_called_when_disabled(self):
        pipeline = _LLMSYNPipeline(
            backend=_MockLLMBackend(seed=0),
            stats=_DEFAULT_STATS,
            noise_scale=0.0,
            prior_mode="full",
            enable_rag=False,
        )
        with patch(
            "pyhealth.datasets.llmsyn_utils.get_medical_knowledge"
        ) as mock_rag:
            pipeline.generate_record()
            mock_rag.assert_not_called()


class TestSyntheticEHRGenerationTask(unittest.TestCase):
    def test_task_name(self):
        task = SyntheticEHRGenerationTask()
        self.assertEqual(task.task_name, "SyntheticEHRGeneration")

    def test_schemas(self):
        task = SyntheticEHRGenerationTask()
        self.assertIn("conditions", task.input_schema)
        self.assertEqual(task.input_schema["conditions"], "sequence")
        self.assertEqual(task.output_schema["mortality"], "binary")


if __name__ == "__main__":
    unittest.main()
