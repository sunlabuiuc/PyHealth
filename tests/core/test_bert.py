"""Unit tests for the BERT model.

This module contains comprehensive tests for the BERT and BERTLayer classes,
covering initialization, forward pass, backward pass, and various configurations.
"""

import unittest
from typing import Dict, Type, Union

import torch

from pyhealth.datasets import SampleDataset, get_dataloader
from pyhealth.models import BERT, BERTLayer, BIOMEDICAL_MODELS
from pyhealth.processors.base_processor import FeatureProcessor


class TestBERTLayer(unittest.TestCase):
    """Test cases for the BERTLayer class."""

    def setUp(self):
        """Set up test fixtures."""
        # Use a smaller model for faster tests
        self.model_name = "bert-base-uncased"
        self.sample_texts = [
            "Patient presents with chest pain.",
            "Annual wellness checkup completed.",
        ]

    def test_layer_initialization(self):
        """Test BERTLayer initialization with default parameters."""
        layer = BERTLayer(model_name=self.model_name)
        
        self.assertEqual(layer.pooling, "cls")
        self.assertEqual(layer.max_length, 512)
        self.assertFalse(layer.freeze_encoder)
        self.assertEqual(layer.freeze_layers, 0)
        self.assertEqual(layer.hidden_size, 768)

    def test_layer_cls_pooling(self):
        """Test BERTLayer with CLS pooling strategy."""
        layer = BERTLayer(model_name=self.model_name, pooling="cls")
        
        with torch.no_grad():
            embeddings = layer(self.sample_texts)
        
        self.assertEqual(embeddings.shape, (2, 768))

    def test_layer_mean_pooling(self):
        """Test BERTLayer with mean pooling strategy."""
        layer = BERTLayer(model_name=self.model_name, pooling="mean")
        
        with torch.no_grad():
            embeddings = layer(self.sample_texts)
        
        self.assertEqual(embeddings.shape, (2, 768))

    def test_layer_max_pooling(self):
        """Test BERTLayer with max pooling strategy."""
        layer = BERTLayer(model_name=self.model_name, pooling="max")
        
        with torch.no_grad():
            embeddings = layer(self.sample_texts)
        
        self.assertEqual(embeddings.shape, (2, 768))

    def test_layer_single_text_input(self):
        """Test BERTLayer with single text string input."""
        layer = BERTLayer(model_name=self.model_name)
        
        with torch.no_grad():
            embeddings = layer("Patient presents with fever.")
        
        self.assertEqual(embeddings.shape, (1, 768))

    def test_layer_return_attention(self):
        """Test BERTLayer returning attention weights."""
        layer = BERTLayer(model_name=self.model_name)
        
        with torch.no_grad():
            embeddings, attentions = layer(self.sample_texts, return_attention=True)
        
        self.assertEqual(embeddings.shape, (2, 768))
        self.assertIsNotNone(attentions)
        # BERT-base has 12 layers
        self.assertEqual(len(attentions), 12)

    def test_layer_max_length_truncation(self):
        """Test BERTLayer truncates long sequences."""
        layer = BERTLayer(model_name=self.model_name, max_length=32)
        
        # Create a very long text
        long_text = "word " * 100
        
        with torch.no_grad():
            embeddings = layer(long_text)
        
        self.assertEqual(embeddings.shape, (1, 768))

    def test_layer_output_size(self):
        """Test get_output_size method."""
        layer = BERTLayer(model_name=self.model_name)
        
        self.assertEqual(layer.get_output_size(), 768)


class TestBERT(unittest.TestCase):
    """Test cases for the BERT model."""

    def setUp(self):
        """Set up test data and model."""
        self.samples = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "clinical_note": "Patient presents with acute chest pain.",
                "label": 1,
            },
            {
                "patient_id": "patient-1",
                "visit_id": "visit-1",
                "clinical_note": "Annual wellness visit. No complaints.",
                "label": 0,
            },
            {
                "patient_id": "patient-2",
                "visit_id": "visit-2",
                "clinical_note": "Follow-up for hypertension management.",
                "label": 0,
            },
        ]

        self.input_schema: Dict[str, Union[str, Type[FeatureProcessor]]] = {
            "clinical_note": "text"
        }
        self.output_schema: Dict[str, Union[str, Type[FeatureProcessor]]] = {
            "label": "binary"
        }

        self.dataset = SampleDataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            dataset_name="test_clinical_notes",
        )

    def test_model_initialization(self):
        """Test BERT model initialization with default parameters."""
        model = BERT(dataset=self.dataset)
        
        self.assertIsInstance(model, BERT)
        self.assertEqual(model.feature_key, "clinical_note")
        self.assertEqual(model.label_key, "label")
        self.assertEqual(model.pooling, "cls")
        self.assertEqual(model.max_length, 512)

    def test_model_initialization_with_biobert_alias(self):
        """Test BERT model initialization with BioBERT alias."""
        # Just test that alias resolution works without downloading
        resolved_name = BIOMEDICAL_MODELS.get("biobert")
        self.assertEqual(resolved_name, "dmis-lab/biobert-v1.1")
        
        resolved_name = BIOMEDICAL_MODELS.get("bert-base-uncased")
        self.assertEqual(resolved_name, "bert-base-uncased")

    def test_model_forward_pass(self):
        """Test BERT forward pass produces correct output structure."""
        model = BERT(dataset=self.dataset, model_name="bert-base-uncased")
        
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        data_batch = next(iter(train_loader))
        
        with torch.no_grad():
            output = model(**data_batch)
        
        # Check output keys
        self.assertIn("loss", output)
        self.assertIn("y_prob", output)
        self.assertIn("y_true", output)
        self.assertIn("logit", output)
        
        # Check shapes
        self.assertEqual(output["y_prob"].shape[0], 2)
        self.assertEqual(output["y_true"].shape[0], 2)
        self.assertEqual(output["logit"].shape[0], 2)
        # Binary classification: output size is 1
        self.assertEqual(output["y_prob"].shape[1], 1)
        
        # Check loss is scalar
        self.assertEqual(output["loss"].dim(), 0)

    def test_model_backward_pass(self):
        """Test BERT backward pass computes gradients."""
        model = BERT(dataset=self.dataset, model_name="bert-base-uncased")
        
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        data_batch = next(iter(train_loader))
        
        output = model(**data_batch)
        output["loss"].backward()
        
        # Check that at least some parameters have gradients
        has_gradient = any(
            param.requires_grad and param.grad is not None
            for param in model.parameters()
        )
        self.assertTrue(has_gradient, "No parameters have gradients after backward pass")

    def test_model_with_embeddings(self):
        """Test BERT returns embeddings when requested."""
        model = BERT(dataset=self.dataset, model_name="bert-base-uncased")
        
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        data_batch = next(iter(train_loader))
        data_batch["embed"] = True
        
        with torch.no_grad():
            output = model(**data_batch)
        
        self.assertIn("embed", output)
        self.assertEqual(output["embed"].shape[0], 2)
        self.assertEqual(output["embed"].shape[1], model.encoder.hidden_size)

    def test_model_encode_method(self):
        """Test BERT encode method for getting embeddings."""
        model = BERT(dataset=self.dataset, model_name="bert-base-uncased")
        
        texts = ["Test sentence one.", "Test sentence two."]
        embeddings = model.encode(texts)
        
        self.assertEqual(embeddings.shape, (2, 768))

    def test_model_with_classifier_hidden_dim(self):
        """Test BERT with hidden layer in classifier head."""
        model = BERT(
            dataset=self.dataset,
            model_name="bert-base-uncased",
            classifier_hidden_dim=256,
        )
        
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        data_batch = next(iter(train_loader))
        
        with torch.no_grad():
            output = model(**data_batch)
        
        self.assertIn("loss", output)
        self.assertEqual(output["y_prob"].shape[0], 2)

    def test_model_with_different_pooling(self):
        """Test BERT with different pooling strategies."""
        for pooling in ["cls", "mean", "max"]:
            with self.subTest(pooling=pooling):
                model = BERT(
                    dataset=self.dataset,
                    model_name="bert-base-uncased",
                    pooling=pooling,
                )
                
                train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
                data_batch = next(iter(train_loader))
                
                with torch.no_grad():
                    output = model(**data_batch)
                
                self.assertIn("loss", output)

    def test_model_repr(self):
        """Test BERT __repr__ method."""
        model = BERT(dataset=self.dataset, model_name="bert-base-uncased")
        repr_str = repr(model)
        
        self.assertIn("BERT", repr_str)
        self.assertIn("bert-base-uncased", repr_str)


class TestBERTMulticlass(unittest.TestCase):
    """Test BERT model with multiclass classification."""

    def setUp(self):
        """Set up multiclass test data."""
        self.samples = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "transcription": "Cardiovascular examination reveals regular rhythm.",
                "specialty": "Cardiology",
            },
            {
                "patient_id": "patient-1",
                "visit_id": "visit-1",
                "transcription": "Patient presents with skin rash.",
                "specialty": "Dermatology",
            },
            {
                "patient_id": "patient-2",
                "visit_id": "visit-2",
                "transcription": "Neurological exam shows normal reflexes.",
                "specialty": "Neurology",
            },
        ]

        self.input_schema = {"transcription": "text"}
        self.output_schema = {"specialty": "multiclass"}

        self.dataset = SampleDataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            dataset_name="test_specialties",
        )

    def test_multiclass_forward_pass(self):
        """Test BERT multiclass forward pass."""
        model = BERT(dataset=self.dataset, model_name="bert-base-uncased")
        
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        data_batch = next(iter(train_loader))
        
        with torch.no_grad():
            output = model(**data_batch)
        
        # Check output shape matches number of classes
        num_classes = model.get_output_size()
        self.assertEqual(output["y_prob"].shape[1], num_classes)
        
        # Check probabilities sum to 1 (softmax for multiclass)
        prob_sums = output["y_prob"].sum(dim=1)
        torch.testing.assert_close(
            prob_sums,
            torch.ones_like(prob_sums),
            atol=1e-5,
            rtol=1e-5,
        )


class TestBERTMultilabel(unittest.TestCase):
    """Test BERT model with multilabel classification."""

    def setUp(self):
        """Set up multilabel test data."""
        # Multilabel expects lists of label names/indices
        self.samples = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "note": "Patient has diabetes and hypertension.",
                "conditions": ["diabetes", "hypertension"],
            },
            {
                "patient_id": "patient-1",
                "visit_id": "visit-1",
                "note": "Patient is obese with diabetes.",
                "conditions": ["diabetes", "obesity"],
            },
            {
                "patient_id": "patient-2",
                "visit_id": "visit-2",
                "note": "Patient has all three conditions.",
                "conditions": ["diabetes", "hypertension", "obesity"],
            },
        ]

        self.input_schema = {"note": "text"}
        self.output_schema = {"conditions": "multilabel"}

        self.dataset = SampleDataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            dataset_name="test_conditions",
        )

    def test_multilabel_forward_pass(self):
        """Test BERT multilabel forward pass."""
        model = BERT(dataset=self.dataset, model_name="bert-base-uncased")
        
        # Get the number of unique labels
        num_labels = model.get_output_size()
        
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        data_batch = next(iter(train_loader))
        
        with torch.no_grad():
            output = model(**data_batch)
        
        # Check output shape matches number of labels
        self.assertEqual(output["y_prob"].shape[1], num_labels)
        
        # Check probabilities are between 0 and 1 (sigmoid for multilabel)
        self.assertTrue(torch.all(output["y_prob"] >= 0))
        self.assertTrue(torch.all(output["y_prob"] <= 1))


class TestBERTParameterGroups(unittest.TestCase):
    """Test BERT parameter grouping for different learning rates."""

    def setUp(self):
        """Set up test fixtures."""
        self.samples = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "text": "Test text.",
                "label": 0,
            },
            {
                "patient_id": "patient-1",
                "visit_id": "visit-1",
                "text": "Another test.",
                "label": 1,
            },
        ]

        self.dataset = SampleDataset(
            samples=self.samples,
            input_schema={"text": "text"},
            output_schema={"label": "binary"},
            dataset_name="test",
        )

    def test_get_encoder_parameters(self):
        """Test getting encoder parameters separately."""
        model = BERT(dataset=self.dataset, model_name="bert-base-uncased")
        
        encoder_params = list(model.get_encoder_parameters())
        self.assertGreater(len(encoder_params), 0)

    def test_get_classifier_parameters(self):
        """Test getting classifier parameters separately."""
        model = BERT(dataset=self.dataset, model_name="bert-base-uncased")
        
        classifier_params = list(model.get_classifier_parameters())
        self.assertGreater(len(classifier_params), 0)

    def test_differential_learning_rates(self):
        """Test setting different learning rates for encoder and classifier."""
        model = BERT(dataset=self.dataset, model_name="bert-base-uncased")
        
        # Create optimizer with different learning rates
        optimizer = torch.optim.Adam([
            {"params": model.get_encoder_parameters(), "lr": 2e-5},
            {"params": model.get_classifier_parameters(), "lr": 1e-3},
        ])
        
        # Verify optimizer has two parameter groups
        self.assertEqual(len(optimizer.param_groups), 2)
        self.assertEqual(optimizer.param_groups[0]["lr"], 2e-5)
        self.assertEqual(optimizer.param_groups[1]["lr"], 1e-3)


class TestBiomedicalModelAliases(unittest.TestCase):
    """Test biomedical model alias resolution."""

    def test_biobert_alias(self):
        """Test BioBERT alias resolution."""
        self.assertEqual(
            BIOMEDICAL_MODELS["biobert"],
            "dmis-lab/biobert-v1.1"
        )

    def test_bert_base_uncased_alias(self):
        """Test bert-base-uncased alias resolution."""
        self.assertEqual(
            BIOMEDICAL_MODELS["bert-base-uncased"],
            "bert-base-uncased"
        )

    def test_bert_base_cased_alias(self):
        """Test bert-base-cased alias resolution."""
        self.assertEqual(
            BIOMEDICAL_MODELS["bert-base-cased"],
            "bert-base-cased"
        )

    def test_all_aliases_are_strings(self):
        """Test all aliases map to string model names."""
        for alias, model_name in BIOMEDICAL_MODELS.items():
            with self.subTest(alias=alias):
                self.assertIsInstance(alias, str)
                self.assertIsInstance(model_name, str)


if __name__ == "__main__":
    unittest.main()

