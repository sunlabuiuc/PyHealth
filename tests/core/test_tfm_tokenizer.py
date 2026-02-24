import unittest
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import TFMTokenizer, get_tfm_tokenizer_2x2x8, get_tfm_token_classifier_64x4


class TestTFMTokenizer(unittest.TestCase):
    """Test cases for the TFMTokenizer model."""

    def setUp(self):
        """Set up test data and model."""
        # Create dummy EEG-style samples with STFT and signal inputs
        self.samples = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "signal": torch.randn((5,6100)).numpy().tolist(),  # (n_samples,)
                "label": 1,
            },
            {
                "patient_id": "patient-1",
                "visit_id": "visit-0",
                "signal": torch.randn((5,6100)).numpy().tolist(),
                "label": 0,
            },
        ]

        self.input_schema = {
            "signal": "tensor",
        }
        self.output_schema = {"label": "binary"}

        self.dataset = create_sample_dataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            dataset_name="test",
        )

        self.model = TFMTokenizer(
            dataset=self.dataset,
            emb_size=64,
            code_book_size=128,  # Small for testing
            use_classifier=True,
        )

    def test_model_initialization(self):
        """Test that the TFMTokenizer model initializes correctly."""
        self.assertIsInstance(self.model, TFMTokenizer)
        self.assertEqual(self.model.emb_size, 64)
        self.assertEqual(self.model.code_book_size, 128)
        self.assertTrue(self.model.use_classifier)
        self.assertEqual(len(self.model.feature_keys), 1)
        self.assertIn("signal", self.model.feature_keys)

    def test_model_forward(self):
        """Test that the TFMTokenizer forward pass works correctly."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            ret = self.model(**data_batch)

        self.assertIn("loss", ret)
        self.assertIn("y_prob", ret)
        self.assertIn("y_true", ret)
        self.assertIn("logit", ret)
        self.assertIn("tokens", ret)
        self.assertIn("embeddings", ret)

        self.assertEqual(ret["y_prob"].shape[0], 2)
        self.assertEqual(ret["y_true"].shape[0], 2)
        self.assertEqual(ret["logit"].shape[0], 2)
        self.assertEqual(ret["loss"].dim(), 0)
        self.assertEqual(ret["tokens"].shape[0], 2)
        self.assertEqual(ret["embeddings"].shape[0], 2)

    def test_model_backward(self):
        """Test that the TFMTokenizer backward pass works correctly."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        ret = self.model(**data_batch)
        ret["loss"].backward()

        has_gradient = any(
            param.requires_grad and param.grad is not None
            for param in self.model.parameters()
        )
        self.assertTrue(has_gradient, "No parameters have gradients after backward pass")

    def test_tokenizer_only(self):
        """Test TFMTokenizer without classifier."""
        model_no_classifier = TFMTokenizer(
            dataset=self.dataset,
            emb_size=64,
            code_book_size=128,
            use_classifier=False,
        )

        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            ret = model_no_classifier(**data_batch)

        self.assertIn("loss", ret)
        self.assertIn("tokens", ret)
        self.assertIn("embeddings", ret)
        self.assertNotIn("y_prob", ret)
        self.assertNotIn("logit", ret)

    def test_get_embeddings(self):
        """Test extraction of embeddings from dataloader."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)

        embeddings = self.model.get_embeddings(train_loader)

        self.assertEqual(embeddings.shape[0], 2)  # 2 samples
        self.assertEqual(embeddings.shape[1], 5)  # 5 channels
        self.assertEqual(embeddings.shape[3], 64)  # emb_size

    def test_get_tokens(self):
        """Test extraction of tokens from dataloader."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)

        tokens = self.model.get_tokens(train_loader)

        self.assertEqual(tokens.shape[0], 2)  # 2 samples
        self.assertTrue(torch.all(tokens >= 0))
        self.assertTrue(torch.all(tokens < 128))  # Within codebook size


class TestTFMTokenizerFactories(unittest.TestCase):
    """Test factory functions for TFM-Tokenizer."""

    def test_get_tfm_tokenizer_2x2x8(self):
        """Test factory function for tokenizer."""
        tokenizer = get_tfm_tokenizer_2x2x8(code_book_size=512, emb_size=64)

        self.assertIsNotNone(tokenizer)
        self.assertEqual(tokenizer.code_book_size, 512)
        self.assertEqual(tokenizer.emb_size, 64)

    def test_get_tfm_token_classifier_64x4(self):
        """Test factory function for classifier."""
        classifier = get_tfm_token_classifier_64x4(n_classes=5, code_book_size=512)

        self.assertIsNotNone(classifier)
        self.assertEqual(classifier.classification_head.out_features, 5)


if __name__ == "__main__":
    unittest.main()
