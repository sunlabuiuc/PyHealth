"""Unit tests for LLM-based evidence retrieval components.

Covers:

- :class:`pyhealth.datasets.SyntheticEHRNotesDataset`
- :class:`pyhealth.tasks.EvidenceRetrievalMIMIC3`
- :class:`pyhealth.models.LLMEvidenceRetriever`
  (sequential and single-prompt prompting)
- :class:`pyhealth.models.CBERTLiteRetriever`

Tests use synthetic data only and complete in milliseconds.

Author:
    Arnab Karmakar (arnabk3@illinois.edu)
"""
import json
import os
import tempfile
import unittest
from unittest import mock

import torch

from pyhealth.datasets import SyntheticEHRNotesDataset
from pyhealth.models import (
    CBERTLiteRetriever,
    HashingEncoder,
    LLMEvidenceRetriever,
    LLMRetrieverConfig,
    OpenAIBackend,
    StubLLMBackend,
)
from pyhealth.tasks import EvidenceRetrievalMIMIC3
from pyhealth.tasks.evidence_retrieval_mimic3 import split_sentences


class _ScriptedBackend:
    """Tiny backend whose responses are scripted by prompt substring."""

    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def __call__(self, prompt: str) -> str:
        self.calls.append(prompt)
        for substring, payload in self.responses:
            if substring in prompt:
                return json.dumps(payload)
        return json.dumps({"decision": "no", "role": "sign", "explanation": ""})


class TestSyntheticEHRNotesDataset(unittest.TestCase):
    """Dataset-level tests: generation, verification, task integration."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.cache = tempfile.TemporaryDirectory()
        cls.dataset = SyntheticEHRNotesDataset(
            root=cls.tmp.name, cache_dir=cls.cache.name
        )

    @classmethod
    def tearDownClass(cls):
        cls.cache.cleanup()
        cls.tmp.cleanup()

    def test_dataset_materialises_csv(self):
        self.dataset.stats()  # smoke test - should not raise

    def test_dataset_has_expected_patients(self):
        self.assertEqual(len(self.dataset.unique_patient_ids), 5)

    def test_dataset_conditions_attribute(self):
        self.assertIn("stroke", self.dataset.conditions)
        self.assertIn("pneumonia", self.dataset.conditions)
        self.assertIn("intracranial hemorrhage", self.dataset.conditions)

    def test_task_schema_is_text_to_binary(self):
        task = EvidenceRetrievalMIMIC3()
        self.assertEqual(task.input_schema["note_text"], "text")
        self.assertEqual(task.output_schema["is_positive"], "binary")

    def test_task_generates_samples_condition_agnostic(self):
        task = EvidenceRetrievalMIMIC3()
        samples = self.dataset.set_task(task)
        try:
            self.assertGreater(len(samples), 0)
            sample = samples[0]
            self.assertIn("note_text", sample)
            self.assertIn("condition", sample)
            self.assertIn("is_positive", sample)
        finally:
            samples.close()

    def test_task_explicit_conditions_expand_samples(self):
        task = EvidenceRetrievalMIMIC3(conditions=["stroke", "pneumonia"])
        samples = self.dataset.set_task(task)
        try:
            conditions_seen = {sample["condition"] for sample in samples}
            self.assertEqual(conditions_seen, {"stroke", "pneumonia"})
        finally:
            samples.close()

    def test_task_rejects_empty_conditions(self):
        with self.assertRaises(ValueError):
            EvidenceRetrievalMIMIC3(conditions=[])


class TestLLMEvidenceRetriever(unittest.TestCase):
    """LLM retriever tests: sequential vs single-prompt and forward()."""

    def test_sequential_prompt_style_uses_two_calls_on_yes(self):
        backend = _ScriptedBackend(
            [
                (
                    "Is the patient at risk",
                    {
                        "decision": "yes",
                        "role": "risk",
                        "explanation": "",
                        "confidence": 0.92,
                    },
                ),
                (
                    "natural-language explanation",
                    {
                        "decision": "yes",
                        "role": "risk",
                        "explanation": "patient on warfarin following craniotomy",
                        "confidence": 0.88,
                    },
                ),
            ]
        )
        model = LLMEvidenceRetriever(
            backend=backend,
            config=LLMRetrieverConfig(prompt_style="sequential"),
        )
        snippet = model.retrieve_evidence(
            note_text="Patient on warfarin post craniotomy with acute hemorrhage.",
            condition="intracranial hemorrhage",
            note_id="n0001",
        )
        self.assertEqual(snippet.decision, "yes")
        self.assertIn("warfarin", snippet.explanation.lower())
        self.assertEqual(len(backend.calls), 2)

    def test_sequential_skips_second_pass_on_no(self):
        backend = _ScriptedBackend(
            [
                (
                    "Is the patient at risk",
                    {
                        "decision": "no",
                        "role": "sign",
                        "explanation": "",
                        "confidence": 0.95,
                    },
                )
            ]
        )
        model = LLMEvidenceRetriever(
            backend=backend,
            config=LLMRetrieverConfig(prompt_style="sequential"),
        )
        snippet = model.retrieve_evidence(
            note_text="Routine follow-up. No neurologic symptoms.",
            condition="intracranial hemorrhage",
            note_id="n0002",
        )
        self.assertEqual(snippet.decision, "no")
        self.assertEqual(snippet.explanation, "")
        self.assertEqual(len(backend.calls), 1)

    def test_single_prompt_style_runs_one_call(self):
        backend = _ScriptedBackend(
            [
                (
                    "one response",
                    {
                        "decision": "yes",
                        "role": "sign",
                        "explanation": "right lower lobe consolidation on chest X-ray",
                        "confidence": 0.81,
                    },
                )
            ]
        )
        model = LLMEvidenceRetriever(
            backend=backend,
            config=LLMRetrieverConfig(prompt_style="single"),
        )
        snippet = model.retrieve_evidence(
            note_text="Fever and cough; CXR shows RLL consolidation.",
            condition="pneumonia",
            note_id="n0007",
        )
        self.assertEqual(snippet.decision, "yes")
        self.assertIn("consolidation", snippet.explanation.lower())
        self.assertEqual(len(backend.calls), 1)

    def test_prompt_cache_reuses_backend_response(self):
        backend = _ScriptedBackend(
            [
                (
                    "Is the patient at risk",
                    {
                        "decision": "no",
                        "role": "sign",
                        "explanation": "",
                        "confidence": 0.9,
                    },
                )
            ]
        )
        model = LLMEvidenceRetriever(
            backend=backend,
            config=LLMRetrieverConfig(prompt_style="sequential", cache_size=8),
        )
        for _ in range(3):
            model.retrieve_evidence(
                note_text="Routine follow up, no acute issues.",
                condition="intracranial hemorrhage",
                note_id="n0002",
            )
        self.assertEqual(
            len(backend.calls), 1,
            "Identical prompts must be served from the cache after the first call.",
        )

    def test_cache_disabled_repeats_backend_calls(self):
        backend = _ScriptedBackend(
            [
                (
                    "Is the patient at risk",
                    {
                        "decision": "no",
                        "role": "sign",
                        "explanation": "",
                        "confidence": 0.9,
                    },
                )
            ]
        )
        model = LLMEvidenceRetriever(
            backend=backend,
            config=LLMRetrieverConfig(prompt_style="sequential", cache_size=0),
        )
        for _ in range(3):
            model.retrieve_evidence(
                note_text="Routine follow up, no acute issues.",
                condition="intracranial hemorrhage",
                note_id="n0002",
            )
        self.assertEqual(len(backend.calls), 3)

    def test_stub_backend_default_runs_offline(self):
        model = LLMEvidenceRetriever(backend=StubLLMBackend())
        snippet = model.retrieve_evidence(
            note_text="Restricted diffusion in right MCA territory.",
            condition="stroke",
            note_id="n0005",
        )
        self.assertEqual(snippet.decision, "yes")
        self.assertIsNotNone(snippet.source_sentence)

    def test_forward_returns_binary_logit_and_loss(self):
        model = LLMEvidenceRetriever(backend=StubLLMBackend())
        outputs = model(
            note_text=[
                "Patient on warfarin post craniotomy.",
                "Routine follow up, no acute issues.",
            ],
            condition=["intracranial hemorrhage", "intracranial hemorrhage"],
            is_positive=[1, 0],
        )
        self.assertEqual(outputs["logit"].shape, (2, 1))
        self.assertEqual(outputs["y_prob"].shape, (2, 1))
        self.assertIn("loss", outputs)
        self.assertTrue(torch.is_tensor(outputs["loss"]))
        self.assertEqual(outputs["y_prob"][0].item() > 0.5, True)
        self.assertEqual(outputs["y_prob"][1].item() < 0.5, True)


class TestCBERTLiteRetriever(unittest.TestCase):
    """IR-baseline tests: ranking, top-k, forward()."""

    def test_ranks_condition_relevant_sentence_first(self):
        model = CBERTLiteRetriever(top_k=2)
        ranked = model.retrieve_evidence(
            note_text=(
                "Patient admitted for elective knee arthroplasty. "
                "CT head shows hyperdense intraparenchymal hemorrhage in "
                "right frontal lobe. Hemoglobin stable."
            ),
            condition="intracranial hemorrhage",
            note_id="n0002",
        )
        self.assertGreater(len(ranked), 0)
        top = ranked[0]
        self.assertIn(
            "hemorrhage", top.sentence.lower(),
            "Highest ranked sentence should reference the condition.",
        )
        self.assertGreaterEqual(top.score, 0.0)
        self.assertLessEqual(top.score, 1.0)

    def test_top_k_limit_is_respected(self):
        model = CBERTLiteRetriever(top_k=2)
        ranked = model.retrieve_evidence(
            note_text=(
                "Fever. Cough. Right lower lobe consolidation. Started on "
                "ceftriaxone. Discharge pending."
            ),
            condition="pneumonia",
        )
        self.assertLessEqual(len(ranked), 2)

    def test_rejects_invalid_top_k(self):
        with self.assertRaises(ValueError):
            CBERTLiteRetriever(top_k=0)

    def test_rejects_invalid_threshold(self):
        with self.assertRaises(ValueError):
            CBERTLiteRetriever(decision_threshold=1.5)

    def test_hashing_encoder_dims_and_normalization(self):
        encoder = HashingEncoder(dim=16)
        embeddings = encoder(["hemorrhage after craniotomy"])
        self.assertEqual(len(embeddings), 1)
        self.assertEqual(len(embeddings[0]), 16)
        norm_sq = sum(v * v for v in embeddings[0])
        self.assertAlmostEqual(norm_sq, 1.0, places=5)

    def test_forward_returns_batch_outputs(self):
        model = CBERTLiteRetriever(top_k=2)
        outputs = model(
            note_text=[
                "Right lower lobe consolidation noted on chest X-ray.",
                "Routine lipid panel, no acute issues.",
            ],
            condition=["pneumonia", "pneumonia"],
            is_positive=[1, 0],
        )
        self.assertEqual(outputs["logit"].shape, (2, 1))
        self.assertIn("loss", outputs)
        self.assertIn("snippets", outputs)


class TestOpenAIBackend(unittest.TestCase):
    """OpenAI backend tests — no real API calls, only contract checks."""

    def test_missing_api_key_raises(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                OpenAIBackend()

    def test_explicit_api_key_bypasses_env(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            # This should succeed in constructing; we don't call __call__,
            # so no network I/O occurs.
            backend = OpenAIBackend(api_key="sk-test-placeholder")
            self.assertEqual(backend.model, "gpt-4o-mini")
            self.assertEqual(backend.temperature, 0.0)

    def test_call_uses_configured_model_and_json_format(self):
        # Stub the OpenAI client to avoid any real API traffic.
        fake_choice = mock.Mock()
        fake_choice.message.content = '{"decision": "yes", "role": "sign"}'
        fake_response = mock.Mock(choices=[fake_choice])
        fake_client = mock.Mock()
        fake_client.chat.completions.create.return_value = fake_response

        backend = OpenAIBackend(api_key="sk-test", model="gpt-4o-mini")
        backend.client = fake_client

        payload = backend("Condition: stroke\nNote:\nMCA infarct.\n")
        self.assertEqual(payload, '{"decision": "yes", "role": "sign"}')

        # Verify the request was shaped correctly.
        fake_client.chat.completions.create.assert_called_once()
        call_kwargs = fake_client.chat.completions.create.call_args.kwargs
        self.assertEqual(call_kwargs["model"], "gpt-4o-mini")
        self.assertEqual(
            call_kwargs["response_format"], {"type": "json_object"}
        )
        self.assertEqual(call_kwargs["temperature"], 0.0)


class TestSentenceSplitter(unittest.TestCase):
    def test_splits_on_terminal_punctuation(self):
        text = "Fever. Cough and chills! Any fatigue?"
        self.assertEqual(
            split_sentences(text),
            ["Fever.", "Cough and chills!", "Any fatigue?"],
        )

    def test_empty_input_returns_empty_list(self):
        self.assertEqual(split_sentences(""), [])
        self.assertEqual(split_sentences(None), [])  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
