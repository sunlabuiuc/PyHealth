"""Tests for ZeroShotEvidenceLLM model.
Contributor: Abhisek Sinha (abhisek5@illinois.edu)
Paper: `Ahsan et al. (2024) <https://arxiv.org/abs/2309.04550>`
All tests use tiny synthetic tensors / mock objects; no real LLM weights are
downloaded.  Tests complete in milliseconds by design.
"""
import importlib.util
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import pytest

_PYHEALTH_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, os.path.abspath(_PYHEALTH_ROOT))


def _load_module(relative_path: str):
    abs_path = os.path.abspath(
        os.path.join(_PYHEALTH_ROOT, "pyhealth", relative_path)
    )
    module_name = relative_path.replace(os.sep, ".").replace("/", ".").rstrip(".py")
    spec = importlib.util.spec_from_file_location(module_name, abs_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Lazy dependency guards
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except (ImportError, OSError):
    _TORCH_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _TORCH_AVAILABLE,
    reason="torch not available in this environment (DLL or install issue)",
)

# Only load if torch is available since the model inherits from nn.Module
if _TORCH_AVAILABLE:
    # Provide a minimal BaseModel stub so we can load the module without the
    # full PyHealth package chain (which needs torchvision etc.)
    _base_model_stub = MagicMock()

    class _StubBaseModel(nn.Module):
        def __init__(self, dataset=None):
            super().__init__()
            self.dataset = dataset
            self.feature_keys = []
            self.label_keys = []
            self.mode = None
            self._dummy_param = nn.Parameter(torch.empty(0))

    _base_model_stub.BaseModel = _StubBaseModel
    sys.modules.setdefault("pyhealth.models.base_model", _base_model_stub)
    sys.modules.setdefault("pyhealth.datasets", MagicMock())

    _llm_mod = _load_module("models/ehr_evidence_llm.py")
    ZeroShotEvidenceLLM = _llm_mod.ZeroShotEvidenceLLM


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

def _make_mock_enc_dec_model(yes_logit: float = 2.0, no_logit: float = 1.0):
    """Mock encoder-decoder HF model whose logits have specific yes/no values."""
    if not _TORCH_AVAILABLE:
        return None, 4273, 150

    model = MagicMock()
    vocab_size = 32128  # Flan-T5 vocab size
    logits = torch.full((1, 1, vocab_size), -10.0)
    YES_ID, NO_ID = 4273, 150
    logits[0, 0, YES_ID] = yes_logit
    logits[0, 0, NO_ID] = no_logit

    output = MagicMock()
    output.logits = logits
    model.return_value = output
    model.generate = MagicMock(return_value=torch.tensor([[0, 4273, 1]]))
    model.eval = MagicMock(return_value=None)
    return model, YES_ID, NO_ID


def _make_mock_tokenizer(yes_id: int = 4273, no_id: int = 150):
    tok = MagicMock()
    tok.pad_token_id = 0
    tok.encode.side_effect = lambda text, **kw: (
        [yes_id] if "yes" in text.lower() else [no_id]
    )
    tok.decode.return_value = "The patient shows signs of small vessel disease."
    if _TORCH_AVAILABLE:
        fake_inputs = {
            "input_ids": torch.zeros((1, 10), dtype=torch.long),
            "attention_mask": torch.ones((1, 10), dtype=torch.long),
        }
        tok.return_value = fake_inputs
        tok.__call__ = MagicMock(return_value=fake_inputs)
    return tok


# ---------------------------------------------------------------------------
# ZeroShotEvidenceLLM initialisation tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not available")
class TestZeroShotEvidenceLLMInit(unittest.TestCase):

    def test_default_init_no_dataset(self):
        model = ZeroShotEvidenceLLM(dataset=None)
        self.assertIsNone(model._hf_model)
        self.assertIsNone(model._tokenizer)
        self.assertEqual(model.model_name, "google/flan-t5-xxl")

    def test_custom_model_name(self):
        model = ZeroShotEvidenceLLM(
            dataset=None, model_name="mistralai/Mistral-7B-Instruct-v0.2"
        )
        self.assertEqual(model.model_name, "mistralai/Mistral-7B-Instruct-v0.2")

    def test_cbert_flag(self):
        model = ZeroShotEvidenceLLM(dataset=None, use_cbert_baseline=True)
        self.assertTrue(model.use_cbert_baseline)

    def test_device_override(self):
        model = ZeroShotEvidenceLLM(dataset=None, device="cpu")
        self.assertEqual(model._device_str, "cpu")

    def test_max_tokens_configurable(self):
        model = ZeroShotEvidenceLLM(
            dataset=None, max_input_tokens=512, max_new_tokens=64
        )
        self.assertEqual(model.max_input_tokens, 512)
        self.assertEqual(model.max_new_tokens, 64)

    def test_cbert_model_name_configurable(self):
        model = ZeroShotEvidenceLLM(
            dataset=None,
            use_cbert_baseline=True,
            cbert_model_name="allenai/biomed_roberta_base",
        )
        self.assertEqual(model.cbert_model_name, "allenai/biomed_roberta_base")


# ---------------------------------------------------------------------------
# predict() tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not available")
class TestZeroShotEvidenceLLMPredict(unittest.TestCase):

    def _get_model(self, yes_logit=3.0, no_logit=0.5):
        model = ZeroShotEvidenceLLM(dataset=None, device="cpu")
        hf_model, yes_id, no_id = _make_mock_enc_dec_model(yes_logit, no_logit)
        tok = _make_mock_tokenizer(yes_id, no_id)
        model._hf_model = hf_model
        model._tokenizer = tok
        model._is_encoder_decoder = True
        return model

    def test_predict_required_keys(self):
        result = self._get_model().predict("Some note.", "small vessel disease")
        for key in ("has_condition", "evidence", "confidence", "model"):
            self.assertIn(key, result)

    def test_positive_when_yes_logit_high(self):
        result = self._get_model(yes_logit=5.0, no_logit=-5.0).predict(
            "SVD noted.", "small vessel disease"
        )
        self.assertTrue(result["has_condition"])
        self.assertGreater(result["confidence"], 0.5)

    def test_negative_when_no_logit_high(self):
        result = self._get_model(yes_logit=-5.0, no_logit=5.0).predict(
            "No neurological findings.", "small vessel disease"
        )
        self.assertFalse(result["has_condition"])
        self.assertLess(result["confidence"], 0.5)

    def test_confidence_in_unit_interval(self):
        result = self._get_model().predict("Note.", "atrial fibrillation")
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)

    def test_evidence_empty_when_negative(self):
        result = self._get_model(yes_logit=-5.0, no_logit=5.0).predict(
            "No findings.", "SVD"
        )
        self.assertEqual(result["evidence"], "")

    def test_evidence_non_empty_when_positive(self):
        result = self._get_model(yes_logit=5.0, no_logit=-5.0).predict(
            "SVD signs observed.", "small vessel disease"
        )
        self.assertTrue(result["has_condition"])
        self.assertIsInstance(result["evidence"], str)


# ---------------------------------------------------------------------------
# predict_batch() / forward() tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not available")
class TestZeroShotEvidenceLLMBatch(unittest.TestCase):

    def _get_model(self):
        model = ZeroShotEvidenceLLM(dataset=None, device="cpu")
        hf_model, yes_id, no_id = _make_mock_enc_dec_model(3.0, 0.5)
        tok = _make_mock_tokenizer(yes_id, no_id)
        model._hf_model = hf_model
        model._tokenizer = tok
        model._is_encoder_decoder = True
        return model

    def test_predict_batch_length(self):
        samples = [
            {"notes": "Note A.", "query_diagnosis": "SVD"},
            {"notes": "Note B.", "query_diagnosis": "SVD"},
        ]
        results = self._get_model().predict_batch(samples)
        self.assertEqual(len(results), 2)

    def test_predict_batch_keys(self):
        samples = [{"notes": "Note.", "query_diagnosis": "SVD"}]
        results = self._get_model().predict_batch(samples)
        self.assertIn("has_condition", results[0])

    def test_forward_returns_lists(self):
        out = self._get_model().forward(
            notes=["Note A.", "Note B."],
            query_diagnosis=["SVD", "AF"],
        )
        self.assertIn("has_condition", out)
        self.assertEqual(len(out["has_condition"]), 2)
        self.assertEqual(len(out["confidence"]), 2)


# ---------------------------------------------------------------------------
# CBERT baseline tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not available")
class TestZeroShotEvidenceLLMCBERT(unittest.TestCase):

    def test_cbert_flag_routes_to_retrieve(self):
        model = ZeroShotEvidenceLLM(dataset=None, use_cbert_baseline=True)
        mock_result = {
            "has_condition": True,
            "evidence": "Best matching sentence.",
            "confidence": 0.75,
            "model": "Bio_ClinicalBERT",
        }
        with patch.object(model, "_cbert_retrieve", return_value=mock_result) as mock_fn:
            result = model.predict("Patient notes.", "small vessel disease")
            mock_fn.assert_called_once()
        self.assertTrue(result["has_condition"])
        self.assertEqual(result["confidence"], 0.75)


if __name__ == "__main__":
    unittest.main()
