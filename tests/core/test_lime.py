"""
Test suite for LIME explainer implementation.
"""
import unittest
from typing import Dict
import tempfile
import pickle
import shutil

import torch
import torch.nn as nn
import litdata

from pyhealth.datasets import SampleDataset, get_dataloader
from pyhealth.datasets.sample_dataset import SampleBuilder
from pyhealth.models import MLP, StageNet, BaseModel
from pyhealth.interpret.methods import LimeExplainer
from pyhealth.interpret.methods.base_interpreter import BaseInterpreter


# ---------------------------------------------------------------------------
# Mock helpers to satisfy the LIME API's dataset/processor requirements
# ---------------------------------------------------------------------------

class _MockProcessor:
    """Mock feature processor with configurable schema."""

    def __init__(self, schema_tuple=("value",)):
        self._schema = schema_tuple

    def schema(self):
        return self._schema

    def is_token(self):
        return False


class _MockDataset:
    """Lightweight stand-in for SampleDataset in unit tests."""

    def __init__(self, input_schema, output_schema, processors=None):
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.input_processors = processors or {
            k: _MockProcessor() for k in input_schema
        }


# ---------------------------------------------------------------------------
# Test model helpers
# ---------------------------------------------------------------------------

class _SimpleLimeModel(BaseModel):
    """Minimal model for testing LIME with continuous inputs."""

    def __init__(self):
        dataset = _MockDataset(
            input_schema={"x": "tensor"},
            output_schema={"y": "binary"},
        )
        super().__init__(dataset=dataset)

        self.linear1 = nn.Linear(3, 4, bias=True)
        self.linear2 = nn.Linear(4, 1, bias=True)

    def forward(self, **kwargs) -> dict:
        x = kwargs["x"]
        if isinstance(x, tuple):
            x = x[0]
        y = kwargs.get("y", None)

        hidden = torch.relu(self.linear1(x))
        logit = self.linear2(hidden)
        y_prob = torch.sigmoid(logit)

        result = {
            "logit": logit,
            "y_prob": y_prob,
            "loss": torch.zeros((), device=y_prob.device),
        }
        if y is not None:
            result["y_true"] = y.to(y_prob.device)
        return result

    def forward_from_embedding(self, **kwargs) -> dict:
        return self.forward(**kwargs)

    def get_embedding_model(self):
        return None


class _SimpleEmbeddingModel(nn.Module):
    """Simple embedding module mapping integer tokens to vectors."""

    def __init__(self, vocab_size: int = 20, embedding_dim: int = 4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {key: self.embedding(value.long()) for key, value in inputs.items()}


class _EmbeddingForwardModel(BaseModel):
    """Toy model exposing forward_from_embedding for discrete features."""

    def __init__(self, schema=("value",)):
        dataset = _MockDataset(
            input_schema={"seq": "sequence"},
            output_schema={"label": "binary"},
            processors={"seq": _MockProcessor(schema)},
        )
        super().__init__(dataset=dataset)

        self.embedding_model = _SimpleEmbeddingModel()
        self.linear = nn.Linear(4, 1, bias=True)

    def forward(self, **kwargs) -> dict:
        seq = kwargs["seq"]
        if isinstance(seq, tuple):
            # Find value position via schema
            schema = self.dataset.input_processors["seq"].schema()
            seq_val = seq[schema.index("value")]
        else:
            seq_val = seq

        embedded = self.embedding_model({"seq": seq_val})["seq"]
        pooled = embedded.mean(dim=1)
        logits = self.linear(pooled)
        y_prob = torch.sigmoid(logits)

        return {
            "logit": logits,
            "y_prob": y_prob,
            "loss": torch.zeros((), device=logits.device),
        }

    def forward_from_embedding(self, **kwargs) -> dict:
        seq = kwargs["seq"]
        if isinstance(seq, tuple):
            schema = self.dataset.input_processors["seq"].schema()
            seq_emb = seq[schema.index("value")]
        else:
            seq_emb = seq

        pooled = seq_emb.mean(dim=1)
        logits = self.linear(pooled)
        y_prob = torch.sigmoid(logits)

        return {
            "logit": logits,
            "y_prob": y_prob,
            "loss": torch.zeros((), device=logits.device),
        }

    def get_embedding_model(self):
        return self.embedding_model


class _MultiFeatureModel(BaseModel):
    """Model with multiple feature inputs for testing multi-feature LIME."""

    def __init__(self):
        dataset = _MockDataset(
            input_schema={"x1": "tensor", "x2": "tensor"},
            output_schema={"y": "binary"},
        )
        super().__init__(dataset=dataset)

        self.linear1 = nn.Linear(2, 3, bias=True)
        self.linear2 = nn.Linear(2, 3, bias=True)
        self.linear_out = nn.Linear(6, 1, bias=True)

    def forward(self, **kwargs) -> dict:
        x1 = kwargs["x1"]
        x2 = kwargs["x2"]
        if isinstance(x1, tuple):
            x1 = x1[0]
        if isinstance(x2, tuple):
            x2 = x2[0]
        y = kwargs.get("y", None)

        h1 = torch.relu(self.linear1(x1))
        h2 = torch.relu(self.linear2(x2))
        combined = torch.cat([h1, h2], dim=-1)
        logit = self.linear_out(combined)
        y_prob = torch.sigmoid(logit)

        result = {
            "logit": logit,
            "y_prob": y_prob,
            "loss": torch.zeros((), device=y_prob.device),
        }
        if y is not None:
            result["y_true"] = y.to(y_prob.device)
        return result

    def forward_from_embedding(self, **kwargs) -> dict:
        return self.forward(**kwargs)

    def get_embedding_model(self):
        return None


class TestLimeExplainerBasic(unittest.TestCase):
    """Basic tests for LimeExplainer functionality."""

    def setUp(self):
        self.model = _SimpleLimeModel()
        self.model.eval()

        # Set deterministic weights
        with torch.no_grad():
            self.model.linear1.weight.copy_(
                torch.tensor([
                    [0.5, -0.3, 0.2],
                    [0.1, 0.4, -0.1],
                    [-0.2, 0.3, 0.5],
                    [0.3, -0.1, 0.2],
                ])
            )
            self.model.linear1.bias.copy_(torch.tensor([0.1, -0.1, 0.2, 0.0]))
            self.model.linear2.weight.copy_(torch.tensor([[0.4, -0.3, 0.2, 0.1]]))
            self.model.linear2.bias.copy_(torch.tensor([0.05]))

        self.labels = torch.zeros((1, 1))
        self.explainer = LimeExplainer(
            self.model, 
            use_embeddings=False,
            n_samples=100,
            kernel_width=0.25,
            random_seed=42,
        )

    def test_inheritance(self):
        """LimeExplainer should inherit from BaseInterpreter."""
        self.assertIsInstance(self.explainer, BaseInterpreter)

    def test_lime_initialization(self):
        """Test that LimeExplainer initializes correctly."""
        explainer = LimeExplainer(self.model, use_embeddings=False)
        self.assertIsInstance(explainer, LimeExplainer)
        self.assertEqual(explainer.model, self.model)
        self.assertFalse(explainer.use_embeddings)
        self.assertEqual(explainer.n_samples, 1000)
        self.assertIsNotNone(explainer.kernel_width)

    def test_attribute_returns_dict(self):
        """Attribute method should return dictionary of LIME attributions."""
        inputs = torch.tensor([[1.0, 0.5, -0.3]])
        
        attributions = self.explainer.attribute(
            x=inputs,
            y=self.labels,
        )

        self.assertIsInstance(attributions, dict)
        self.assertIn("x", attributions)
        self.assertEqual(attributions["x"].shape, inputs.shape)

    def test_lime_values_are_tensors(self):
        """LIME values should be PyTorch tensors."""
        inputs = torch.tensor([[0.8, -0.2, 0.5]])
        
        attributions = self.explainer.attribute(
            x=inputs,
            y=self.labels,
        )

        self.assertIsInstance(attributions["x"], torch.Tensor)
        self.assertFalse(attributions["x"].requires_grad)

    def test_baseline_generation(self):
        """Should generate baseline automatically if not provided."""
        inputs = torch.tensor([[1.0, 0.5, -0.3], [0.5, 1.0, 0.2]])
        
        attributions = self.explainer.attribute(
            x=inputs,
            y=torch.zeros((2, 1)),
        )

        self.assertEqual(attributions["x"].shape, inputs.shape)

    def test_custom_baseline(self):
        """Should accept custom baseline dictionary."""
        inputs = torch.tensor([[1.0, 0.5, -0.3]])
        baseline = {"x": torch.zeros_like(inputs)}
        
        attributions = self.explainer.attribute(
            baseline=baseline,
            x=inputs,
            y=self.labels,
        )

        self.assertEqual(attributions["x"].shape, inputs.shape)

    def test_target_class_idx_none(self):
        """Should handle None target class index (max prediction)."""
        inputs = torch.tensor([[1.0, 0.5, -0.3]])
        
        attributions = self.explainer.attribute(
            x=inputs,
            y=self.labels,
            target_class_idx=None,
        )

        self.assertIn("x", attributions)
        self.assertEqual(attributions["x"].shape, inputs.shape)

    def test_target_class_idx_specified(self):
        """Should handle specific target class index."""
        inputs = torch.tensor([[1.0, 0.5, -0.3]])
        
        attr_class_0 = self.explainer.attribute(
            x=inputs,
            y=self.labels,
            target_class_idx=0,
        )
        
        attr_class_1 = self.explainer.attribute(
            x=inputs,
            y=self.labels,
            target_class_idx=1,
        )

        # Attributions should differ for different classes
        self.assertFalse(torch.allclose(attr_class_0["x"], attr_class_1["x"], atol=0.01))

    def test_attribution_values_are_finite(self):
        """Test that attribution values are finite (no NaN or Inf)."""
        inputs = torch.tensor([[1.0, 0.5, -0.3]])
        
        attributions = self.explainer.attribute(
            x=inputs,
            y=self.labels,
        )

        self.assertTrue(torch.isfinite(attributions["x"]).all())

    def test_multiple_samples(self):
        """Test attribution on batch with multiple samples."""
        inputs = torch.tensor([[1.0, 0.5, -0.3], [0.5, 1.0, 0.2], [-0.5, 0.3, 0.8]])
        
        attributions = self.explainer.attribute(
            x=inputs,
            y=torch.zeros((3, 1)),
        )

        # Check batch dimension
        self.assertEqual(attributions["x"].shape[0], 3)
        self.assertEqual(attributions["x"].shape, inputs.shape)

    def test_callable_interface(self):
        """LimeExplainer instances should be callable via BaseInterpreter.__call__."""
        inputs = torch.tensor([[0.3, -0.4, 0.5]])
        kwargs = {"x": inputs, "y": self.labels}

        from_attribute = self.explainer.attribute(**kwargs)
        from_call = self.explainer(**kwargs)

        torch.testing.assert_close(
            from_call["x"], 
            from_attribute["x"],
            rtol=1e-3,
            atol=1e-4
        )

    def test_different_n_samples(self):
        """Test with different numbers of perturbation samples."""
        inputs = torch.tensor([[1.0, 0.5, -0.3]])
        
        # Few samples
        explainer_few = LimeExplainer(
            self.model, 
            use_embeddings=False,
            n_samples=50,
            random_seed=42,
        )
        attr_few = explainer_few.attribute(x=inputs, y=self.labels)

        # More samples
        explainer_many = LimeExplainer(
            self.model, 
            use_embeddings=False,
            n_samples=200,
            random_seed=42,
        )
        attr_many = explainer_many.attribute(x=inputs, y=self.labels)

        # Both should produce valid output
        self.assertEqual(attr_few["x"].shape, inputs.shape)
        self.assertEqual(attr_many["x"].shape, inputs.shape)
        self.assertTrue(torch.isfinite(attr_few["x"]).all())
        self.assertTrue(torch.isfinite(attr_many["x"]).all())

    def test_different_regularization_methods(self):
        """Test LIME with different regularization methods."""
        inputs = torch.tensor([[1.0, 0.5, -0.3]])
        
        # Lasso
        explainer_lasso = LimeExplainer(
            self.model,
            use_embeddings=False,
            n_samples=100,
            feature_selection="lasso",
            alpha=0.01,
            random_seed=42,
        )
        attr_lasso = explainer_lasso.attribute(x=inputs, y=self.labels)
        
        # Ridge
        explainer_ridge = LimeExplainer(
            self.model,
            use_embeddings=False,
            n_samples=100,
            feature_selection="ridge",
            alpha=0.01,
            random_seed=42,
        )
        attr_ridge = explainer_ridge.attribute(x=inputs, y=self.labels)
        
        # None (OLS)
        explainer_none = LimeExplainer(
            self.model,
            use_embeddings=False,
            n_samples=100,
            feature_selection="none",
            random_seed=42,
        )
        attr_none = explainer_none.attribute(x=inputs, y=self.labels)
        
        # All should produce valid finite output
        self.assertTrue(torch.isfinite(attr_lasso["x"]).all())
        self.assertTrue(torch.isfinite(attr_ridge["x"]).all())
        self.assertTrue(torch.isfinite(attr_none["x"]).all())

    def test_different_distance_modes(self):
        """Test LIME with different distance modes."""
        inputs = torch.tensor([[1.0, 0.5, -0.3]])
        
        # Cosine distance
        explainer_cosine = LimeExplainer(
            self.model,
            use_embeddings=False,
            n_samples=100,
            distance_mode="cosine",
            random_seed=42,
        )
        attr_cosine = explainer_cosine.attribute(x=inputs, y=self.labels)
        
        # Euclidean distance
        explainer_euclidean = LimeExplainer(
            self.model,
            use_embeddings=False,
            n_samples=100,
            distance_mode="euclidean",
            random_seed=42,
        )
        attr_euclidean = explainer_euclidean.attribute(x=inputs, y=self.labels)
        
        # Both should produce valid output
        self.assertTrue(torch.isfinite(attr_cosine["x"]).all())
        self.assertTrue(torch.isfinite(attr_euclidean["x"]).all())

    def test_reproducibility_with_seed(self):
        """Test that results are reproducible with same random seed."""
        inputs = torch.tensor([[1.0, 0.5, -0.3]])
        
        explainer1 = LimeExplainer(
            self.model,
            use_embeddings=False,
            n_samples=100,
            random_seed=42,
        )
        explainer2 = LimeExplainer(
            self.model,
            use_embeddings=False,
            n_samples=100,
            random_seed=42,
        )
        
        attr1 = explainer1.attribute(x=inputs, y=self.labels)
        attr2 = explainer2.attribute(x=inputs, y=self.labels)
        
        # Results should be identical with same seed
        torch.testing.assert_close(attr1["x"], attr2["x"], rtol=1e-5, atol=1e-6)


class TestLimeExplainerEmbedding(unittest.TestCase):
    """Tests for LimeExplainer with embedding-based models."""

    def setUp(self):
        self.model = _EmbeddingForwardModel()
        self.model.eval()

        # Set deterministic weights
        with torch.no_grad():
            self.model.linear.weight.copy_(torch.tensor([[0.4, -0.3, 0.2, 0.1]]))
            self.model.linear.bias.copy_(torch.tensor([0.05]))

        self.labels = torch.zeros((1, 1))
        self.explainer = LimeExplainer(
            self.model,
            use_embeddings=True,
            n_samples=100,
            random_seed=42,
        )

    def test_embedding_initialization(self):
        """Test that LimeExplainer initializes with embedding mode."""
        self.assertTrue(self.explainer.use_embeddings)
        self.assertTrue(hasattr(self.model, "forward_from_embedding"))

    def test_attribute_with_embeddings(self):
        """Test attribution computation in embedding mode."""
        seq_inputs = torch.tensor([[1, 2, 3]])
        
        attributions = self.explainer.attribute(
            seq=seq_inputs,
            label=self.labels,
        )

        self.assertIn("seq", attributions)
        self.assertEqual(attributions["seq"].shape, seq_inputs.shape)

    def test_embedding_attributions_are_finite(self):
        """Test that embedding-based attributions are finite."""
        seq_inputs = torch.tensor([[5, 10, 15]])
        
        attributions = self.explainer.attribute(
            seq=seq_inputs,
            label=self.labels,
        )

        self.assertTrue(torch.isfinite(attributions["seq"]).all())

    def test_embedding_with_time_info(self):
        """Test attribution with time information (temporal data)."""
        # Create a model with time-aware schema ("time", "value")
        model = _EmbeddingForwardModel(schema=("time", "value"))
        model.eval()
        with torch.no_grad():
            model.linear.weight.copy_(torch.tensor([[0.4, -0.3, 0.2, 0.1]]))
            model.linear.bias.copy_(torch.tensor([0.05]))

        explainer = LimeExplainer(
            model, use_embeddings=True, n_samples=100, random_seed=42,
        )

        time_tensor = torch.tensor([[0.0, 1.5, 3.0]])
        seq_tensor = torch.tensor([[1, 2, 3]])

        attributions = explainer.attribute(
            seq=(time_tensor, seq_tensor),
            label=self.labels,
        )

        self.assertIn("seq", attributions)
        self.assertEqual(attributions["seq"].shape, seq_tensor.shape)

    def test_embedding_with_custom_baseline(self):
        """Test embedding-based LIME with custom baseline."""
        seq_inputs = torch.tensor([[1, 2, 3]])
        # Custom baseline must be in embedding space (post-embedding) because
        # the LIME code does not embed user-supplied baselines.
        embedded_baseline = self.model.embedding_model({"seq": torch.zeros_like(seq_inputs)})
        baseline = {"seq": embedded_baseline["seq"]}
        
        attributions = self.explainer.attribute(
            baseline=baseline,
            seq=seq_inputs,
            label=self.labels,
        )

        self.assertEqual(attributions["seq"].shape, seq_inputs.shape)

    def test_embedding_model_without_forward_from_embedding_fails(self):
        """Test that using embeddings without forward_from_embedding raises error."""
        # Use a plain nn.Module that does NOT inherit from BaseModel and
        # therefore does not have forward_from_embedding.
        class _BareModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.feature_keys = ["x"]
                self.label_keys = ["y"]
                self.linear = nn.Linear(3, 1)

        model = _BareModel()
        with self.assertRaises(AssertionError):
            LimeExplainer(model, use_embeddings=True)


class TestLimeExplainerMultiFeature(unittest.TestCase):
    """Tests for LimeExplainer with multiple feature inputs."""

    def setUp(self):
        self.model = _MultiFeatureModel()
        self.model.eval()

        # Set deterministic weights
        with torch.no_grad():
            self.model.linear1.weight.copy_(
                torch.tensor([[0.5, -0.3], [0.1, 0.4], [-0.2, 0.3]])
            )
            self.model.linear2.weight.copy_(
                torch.tensor([[0.3, -0.1], [0.2, 0.5], [0.4, -0.2]])
            )
            self.model.linear_out.weight.copy_(
                torch.tensor([[0.1, 0.2, -0.1, 0.3, -0.2, 0.15]])
            )

        self.labels = torch.zeros((1, 1))
        self.explainer = LimeExplainer(
            self.model,
            use_embeddings=False,
            n_samples=100,
            random_seed=42,
        )

    def test_multi_feature_attribution(self):
        """Test attribution with multiple feature inputs."""
        x1 = torch.tensor([[1.0, 0.5]])
        x2 = torch.tensor([[-0.3, 0.8]])

        attributions = self.explainer.attribute(
            x1=x1,
            x2=x2,
            y=self.labels,
        )

        self.assertIn("x1", attributions)
        self.assertIn("x2", attributions)
        self.assertEqual(attributions["x1"].shape, x1.shape)
        self.assertEqual(attributions["x2"].shape, x2.shape)

    def test_multi_feature_with_custom_baselines(self):
        """Test multi-feature attribution with custom baselines."""
        x1 = torch.tensor([[1.0, 0.5]])
        x2 = torch.tensor([[-0.3, 0.8]])
        baseline = {
            "x1": torch.zeros_like(x1),
            "x2": torch.ones_like(x2) * 0.5,
        }

        attributions = self.explainer.attribute(
            baseline=baseline,
            x1=x1,
            x2=x2,
            y=self.labels,
        )

        self.assertEqual(attributions["x1"].shape, x1.shape)
        self.assertEqual(attributions["x2"].shape, x2.shape)

    def test_multi_feature_finite_values(self):
        """Test that multi-feature attributions are finite."""
        # Test with single sample to avoid baseline batch size issues
        x1 = torch.tensor([[1.0, 0.5]])
        x2 = torch.tensor([[-0.3, 0.8]])

        attributions = self.explainer.attribute(
            x1=x1,
            x2=x2,
            y=torch.zeros((1, 1)),
        )

        self.assertTrue(torch.isfinite(attributions["x1"]).all())
        self.assertTrue(torch.isfinite(attributions["x2"]).all())


@unittest.skip("MLP does not yet support the LIME interpreter API")
class TestLimeExplainerMLP(unittest.TestCase):
    """Test cases for LIME with MLP model on real dataset."""

    def setUp(self):
        """Set up test data and model."""
        self.samples = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "conditions": ["cond-33", "cond-86", "cond-80", "cond-12"],
                "procedures": [1.0, 2.0, 3.5, 4],
                "label": 0,
            },
            {
                "patient_id": "patient-1",
                "visit_id": "visit-1",
                "conditions": ["cond-33", "cond-86", "cond-80"],
                "procedures": [5.0, 2.0, 3.5, 4],
                "label": 1,
            },
            {
                "patient_id": "patient-2",
                "visit_id": "visit-2",
                "conditions": ["cond-55", "cond-12"],
                "procedures": [2.0, 3.0, 1.5, 5],
                "label": 1,
            },
        ]

        # Define input and output schemas
        self.input_schema = {
            "conditions": "sequence",
            "procedures": "tensor",
        }
        self.output_schema = {"label": "binary"}

        # Create temporary directory for dataset
        self.temp_dir = tempfile.mkdtemp()
        
        # Create dataset using SampleBuilder
        builder = SampleBuilder(self.input_schema, self.output_schema)
        builder.fit(self.samples)
        builder.save(f"{self.temp_dir}/schema.pkl")
        
        # Optimize samples into dataset format
        def sample_generator():
            for sample in self.samples:
                yield {"sample": pickle.dumps(sample)}
        
        litdata.optimize(
            fn=builder.transform,
            inputs=list(sample_generator()),
            output_dir=self.temp_dir,
            num_workers=1,
            chunk_bytes="64MB",
        )
        
        # Create dataset
        self.dataset = SampleDataset(
            path=self.temp_dir,
            dataset_name="test_lime",
        )

        # Create model
        self.model = MLP(
            dataset=self.dataset,
            embedding_dim=32,
            hidden_dim=32,
            n_layers=2,
        )
        self.model.eval()

        # Create dataloader
        self.test_loader = get_dataloader(self.dataset, batch_size=1, shuffle=False)

    def tearDown(self):
        """Clean up temporary directory."""
        if hasattr(self, 'temp_dir'):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_lime_mlp_basic_attribution(self):
        """Test basic LIME attribution computation with MLP."""
        explainer = LimeExplainer(
            self.model,
            use_embeddings=True,
            n_samples=50,
            random_seed=42,
        )
        data_batch = next(iter(self.test_loader))

        # Compute attributions
        attributions = explainer.attribute(**data_batch)

        # Check output structure
        self.assertIn("conditions", attributions)
        self.assertIn("procedures", attributions)

        # Check shapes match input shapes
        self.assertEqual(
            attributions["conditions"].shape, data_batch["conditions"].shape
        )
        self.assertEqual(
            attributions["procedures"].shape, data_batch["procedures"].shape
        )

        # Check that attributions are tensors
        self.assertIsInstance(attributions["conditions"], torch.Tensor)
        self.assertIsInstance(attributions["procedures"], torch.Tensor)

    def test_lime_mlp_with_target_class(self):
        """Test LIME attribution with specific target class."""
        explainer = LimeExplainer(
            self.model,
            use_embeddings=True,
            n_samples=50,
            random_seed=42,
        )
        data_batch = next(iter(self.test_loader))

        # Compute attributions for class 0
        attr_class_0 = explainer.attribute(**data_batch, target_class_idx=0)

        # Compute attributions for class 1
        attr_class_1 = explainer.attribute(**data_batch, target_class_idx=1)

        # Check that attributions are different for different classes
        self.assertFalse(
            torch.allclose(attr_class_0["conditions"], attr_class_1["conditions"], atol=0.01)
        )

    def test_lime_mlp_values_finite(self):
        """Test that LIME values are finite (no NaN or Inf)."""
        explainer = LimeExplainer(
            self.model,
            use_embeddings=True,
            n_samples=50,
            random_seed=42,
        )
        data_batch = next(iter(self.test_loader))

        attributions = explainer.attribute(**data_batch)

        # Check no NaN or Inf
        self.assertTrue(torch.isfinite(attributions["conditions"]).all())
        self.assertTrue(torch.isfinite(attributions["procedures"]).all())

    def test_lime_mlp_multiple_samples(self):
        """Test LIME on batch with multiple samples."""
        explainer = LimeExplainer(
            self.model,
            use_embeddings=True,
            n_samples=50,
            random_seed=42,
        )

        # Use batch size > 1
        test_loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        data_batch = next(iter(test_loader))

        attributions = explainer.attribute(**data_batch)

        # Check batch dimension
        self.assertEqual(attributions["conditions"].shape[0], 2)
        self.assertEqual(attributions["procedures"].shape[0], 2)

    def test_lime_mlp_different_n_samples(self):
        """Test LIME with different numbers of perturbation samples."""
        data_batch = next(iter(self.test_loader))

        # Few samples
        explainer_few = LimeExplainer(
            self.model,
            use_embeddings=True,
            n_samples=30,
            random_seed=42,
        )
        attr_few = explainer_few.attribute(**data_batch)

        # More samples
        explainer_many = LimeExplainer(
            self.model,
            use_embeddings=True,
            n_samples=100,
            random_seed=42,
        )
        attr_many = explainer_many.attribute(**data_batch)

        # Both should produce valid output
        self.assertIn("conditions", attr_few)
        self.assertIn("conditions", attr_many)
        self.assertEqual(attr_few["conditions"].shape, attr_many["conditions"].shape)

    def test_lime_mlp_different_regularization(self):
        """Test LIME with different regularization methods on MLP."""
        data_batch = next(iter(self.test_loader))

        # Lasso
        explainer_lasso = LimeExplainer(
            self.model,
            use_embeddings=True,
            n_samples=50,
            feature_selection="lasso",
            alpha=0.01,
            random_seed=42,
        )
        attr_lasso = explainer_lasso.attribute(**data_batch)

        # Ridge
        explainer_ridge = LimeExplainer(
            self.model,
            use_embeddings=True,
            n_samples=50,
            feature_selection="ridge",
            alpha=0.01,
            random_seed=42,
        )
        attr_ridge = explainer_ridge.attribute(**data_batch)

        # Both should produce valid finite output
        self.assertTrue(torch.isfinite(attr_lasso["conditions"]).all())
        self.assertTrue(torch.isfinite(attr_ridge["conditions"]).all())


class TestLimeExplainerStageNet(unittest.TestCase):
    """Test cases for LIME with StageNet model.

    Note: StageNet tests demonstrate LIME working with temporal/sequential data.
    """

    def setUp(self):
        """Set up test data and StageNet model."""
        self.samples = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "codes": ([0.0, 2.0, 1.3], ["505800458", "50580045810", "50580045811"]),
                "procedures": (
                    [0.0, 1.5],
                    [["A05B", "A05C", "A06A"], ["A11D", "A11E"]],
                ),
                "lab_values": (None, [[1.0, 2.55, 3.4], [4.1, 5.5, 6.0]]),
                "label": 1,
            },
            {
                "patient_id": "patient-1",
                "visit_id": "visit-1",
                "codes": (
                    [0.0, 2.0, 1.3, 1.0, 2.0],
                    [
                        "55154191800",
                        "551541928",
                        "55154192800",
                        "705182798",
                        "70518279800",
                    ],
                ),
                "procedures": ([0.0], [["A04A", "B035", "C129"]]),
                "lab_values": (
                    None,
                    [
                        [1.4, 3.2, 3.5],
                        [4.1, 5.9, 1.7],
                        [4.5, 5.9, 1.7],
                    ],
                ),
                "label": 0,
            },
        ]

        # Define input and output schemas
        self.input_schema = {
            "codes": "stagenet",
            "procedures": "stagenet",
            "lab_values": "stagenet_tensor",
        }
        self.output_schema = {"label": "binary"}

        # Create temporary directory for dataset
        self.temp_dir = tempfile.mkdtemp()
        
        # Create dataset using SampleBuilder
        builder = SampleBuilder(self.input_schema, self.output_schema)
        builder.fit(self.samples)
        builder.save(f"{self.temp_dir}/schema.pkl")
        
        # Optimize samples into dataset format
        def sample_generator():
            for sample in self.samples:
                yield {"sample": pickle.dumps(sample)}
        
        litdata.optimize(
            fn=builder.transform,
            inputs=list(sample_generator()),
            output_dir=self.temp_dir,
            num_workers=1,
            chunk_bytes="64MB",
        )
        
        # Create dataset
        self.dataset = SampleDataset(
            path=self.temp_dir,
            dataset_name="test_stagenet_lime",
        )

        # Create StageNet model
        self.model = StageNet(
            dataset=self.dataset,
            embedding_dim=32,
            chunk_size=16,
            levels=2,
        )
        self.model.eval()

        # Create dataloader
        self.test_loader = get_dataloader(self.dataset, batch_size=1, shuffle=False)

    def tearDown(self):
        """Clean up temporary directory."""
        if hasattr(self, 'temp_dir'):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_lime_initialization_stagenet(self):
        """Test that LimeExplainer works with StageNet."""
        explainer = LimeExplainer(
            self.model,
            use_embeddings=True,
            n_samples=50,
            random_seed=42,
        )
        self.assertIsInstance(explainer, LimeExplainer)
        self.assertEqual(explainer.model, self.model)

    def test_lime_basic_attribution_stagenet(self):
        """Test basic LIME attribution computation with StageNet."""
        explainer = LimeExplainer(
            self.model,
            use_embeddings=True,
            n_samples=50,
            random_seed=42,
        )
        data_batch = next(iter(self.test_loader))

        # Compute attributions
        attributions = explainer.attribute(**data_batch)

        # Check output structure
        self.assertIn("codes", attributions)
        self.assertIn("procedures", attributions)
        self.assertIn("lab_values", attributions)

        # Check that attributions are tensors
        self.assertIsInstance(attributions["codes"], torch.Tensor)
        self.assertIsInstance(attributions["procedures"], torch.Tensor)
        self.assertIsInstance(attributions["lab_values"], torch.Tensor)

    def test_lime_stagenet_values_finite(self):
        """Test that LIME values on StageNet are finite."""
        explainer = LimeExplainer(
            self.model,
            use_embeddings=True,
            n_samples=50,
            random_seed=42,
        )
        data_batch = next(iter(self.test_loader))

        attributions = explainer.attribute(**data_batch)

        # Check no NaN or Inf
        self.assertTrue(torch.isfinite(attributions["codes"]).all())
        self.assertTrue(torch.isfinite(attributions["procedures"]).all())
        self.assertTrue(torch.isfinite(attributions["lab_values"]).all())

    def test_lime_stagenet_target_class(self):
        """Test LIME with specific target class on StageNet."""
        explainer = LimeExplainer(
            self.model,
            use_embeddings=True,
            n_samples=50,
            random_seed=42,
        )
        data_batch = next(iter(self.test_loader))

        # Compute attributions for class 1
        attributions = explainer.attribute(**data_batch, target_class_idx=1)

        # Check output structure
        self.assertIn("codes", attributions)
        self.assertTrue(torch.isfinite(attributions["codes"]).all())
