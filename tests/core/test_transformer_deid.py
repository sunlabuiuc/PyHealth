"""
Unit tests for the TransformerDeID model.

Author:
    Matt McKenna (mtm16@illinois.edu)
"""

import unittest

import torch

from pyhealth.datasets import create_sample_dataset
from pyhealth.models.transformer_deid import (
    IGNORE_INDEX,
    LABEL_VOCAB,
    TransformerDeID,
    align_labels,
)
from pyhealth.processors.text_processor import TextProcessor


def _make_dataset():
    """Create a minimal in-memory dataset matching DeIDNERTask output."""
    samples = [
        {
            "patient_id": "p1",
            "text": "Patient John Smith was seen",
            "labels": "O B-NAME I-NAME O O",
        },
        {
            "patient_id": "p2",
            "text": "Admitted on 01/15/2024 to clinic",
            "labels": "O O B-DATE O O",
        },
    ]
    return create_sample_dataset(
        samples=samples,
        input_schema={"text": TextProcessor},
        output_schema={"labels": TextProcessor},
        dataset_name="test_deid",
        task_name="DeIDNER",
        in_memory=True,
    )


class TestLabelVocab(unittest.TestCase):
    def test_vocab_size(self):
        """O + 7 categories * 2 (B/I) = 15."""
        self.assertEqual(len(LABEL_VOCAB), 15)

    def test_o_is_zero(self):
        self.assertEqual(LABEL_VOCAB["O"], 0)

    def test_all_categories_present(self):
        for cat in ("AGE", "CONTACT", "DATE", "ID", "LOCATION", "NAME", "PROFESSION"):
            self.assertIn(f"B-{cat}", LABEL_VOCAB)
            self.assertIn(f"I-{cat}", LABEL_VOCAB)


class TestAlignLabels(unittest.TestCase):
    def test_no_subword_splits(self):
        """When every word is a single token, labels pass through."""
        # word_ids: None=CLS, 0, 1, 2, None=SEP
        word_ids = [None, 0, 1, 2, None]
        word_labels = [0, 11, 12]  # O, B-NAME, I-NAME
        result = align_labels(word_ids, word_labels)
        self.assertEqual(result, [IGNORE_INDEX, 0, 11, 12, IGNORE_INDEX])

    def test_subword_split(self):
        """Non-first subtokens should get IGNORE_INDEX."""
        # "Smith" split into 2 subtokens (word_id=1 twice)
        word_ids = [None, 0, 1, 1, 2, None]
        word_labels = [0, 12, 0]  # O, I-NAME, O
        result = align_labels(word_ids, word_labels)
        self.assertEqual(
            result,
            [IGNORE_INDEX, 0, 12, IGNORE_INDEX, 0, IGNORE_INDEX],
        )

    def test_all_special_tokens(self):
        """All-None word_ids should produce all IGNORE_INDEX."""
        word_ids = [None, None]
        result = align_labels(word_ids, [])
        self.assertEqual(result, [IGNORE_INDEX, IGNORE_INDEX])


class TestTransformerDeIDInit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = _make_dataset()
        cls.model = TransformerDeID(dataset=cls.dataset)

    @classmethod
    def tearDownClass(cls):
        cls.dataset.close()

    def test_feature_key(self):
        self.assertEqual(self.model.feature_key, "text")

    def test_label_key(self):
        self.assertEqual(self.model.label_key, "labels")

    def test_num_labels(self):
        self.assertEqual(self.model.num_labels, 15)

    def test_classifier_output_dim(self):
        self.assertEqual(self.model.classifier.out_features, 15)

    def test_encoder_hidden_size(self):
        """BERT-base has hidden_size=768."""
        self.assertEqual(self.model.encoder.config.hidden_size, 768)


class TestTransformerDeIDForward(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = _make_dataset()
        cls.model = TransformerDeID(dataset=cls.dataset)
        cls.model.eval()
        # Run a forward pass with raw strings (same format as task output).
        with torch.no_grad():
            cls.result = cls.model(
                text=[
                    "Patient John Smith was seen",
                    "Admitted on 01/15/2024 to clinic",
                ],
                labels=[
                    "O B-NAME I-NAME O O",
                    "O O B-DATE O O",
                ],
            )

    @classmethod
    def tearDownClass(cls):
        cls.dataset.close()

    def test_output_has_required_keys(self):
        for key in ("loss", "logit", "y_prob", "y_true"):
            self.assertIn(key, self.result)

    def test_loss_is_scalar(self):
        self.assertEqual(self.result["loss"].dim(), 0)

    def test_logit_shape(self):
        """logit should be (batch, seq_len, num_labels)."""
        logit = self.result["logit"]
        self.assertEqual(logit.shape[0], 2)  # batch size
        self.assertEqual(logit.shape[2], 15)  # num labels

    def test_y_prob_shape_matches_logit(self):
        self.assertEqual(
            self.result["y_prob"].shape, self.result["logit"].shape
        )

    def test_y_prob_sums_to_one(self):
        """Softmax probabilities should sum to ~1 along label dim."""
        sums = self.result["y_prob"].sum(dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-5))

    def test_backward(self):
        """Loss backward should produce gradients."""
        # Need train mode and fresh forward pass for gradients.
        self.model.train()
        result = self.model(
            text=["Patient John Smith was seen"],
            labels=["O B-NAME I-NAME O O"],
        )
        result["loss"].backward()
        has_grad = any(
            p.requires_grad and p.grad is not None
            for p in self.model.parameters()
        )
        self.assertTrue(has_grad)
        self.model.eval()
        self.model.zero_grad()

    def test_deidentify_returns_string(self):
        result = self.model.deidentify("Patient John Smith was seen")
        self.assertIsInstance(result, str)

    def test_deidentify_same_word_count(self):
        """Output should have same number of words (redacted or not)."""
        text = "Patient John Smith was seen"
        result = self.model.deidentify(text)
        self.assertEqual(len(result.split()), len(text.split()))

    def test_deidentify_custom_redact_marker(self):
        result = self.model.deidentify("Patient John", redact="[PHI]")
        self.assertNotIn("[REDACTED]", result)
        # Every word should be either an original word or the custom marker.
        for word in result.split():
            self.assertTrue(
                word in ("Patient", "John") or word == "[PHI]",
                f"Unexpected word in output: {word}",
            )


if __name__ == "__main__":
    unittest.main()
