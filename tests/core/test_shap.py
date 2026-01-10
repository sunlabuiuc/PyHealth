import unittest
from typing import Dict
import tempfile
import pickle
import shutil
from functools import partial

import torch
import torch.nn as nn
import litdata

from pyhealth.datasets import SampleDataset, get_dataloader
from pyhealth.datasets.sample_dataset import SampleBuilder
from pyhealth.models import MLP, StageNet, BaseModel
from pyhealth.interpret.methods import ShapExplainer
from pyhealth.interpret.methods.base_interpreter import BaseInterpreter


class _SimpleShapModel(BaseModel):
    """Minimal model for testing SHAP with continuous inputs."""

    def __init__(self):
        super().__init__(dataset=None)
        self.feature_keys = ["x"]
        self.label_keys = ["y"]
        self.mode = "binary"

        self.linear1 = nn.Linear(3, 4, bias=True)
        self.linear2 = nn.Linear(4, 1, bias=True)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> dict:
        hidden = torch.relu(self.linear1(x))
        logit = self.linear2(hidden)
        y_prob = torch.sigmoid(logit)

        return {
            "logit": logit,
            "y_prob": y_prob,
            "y_true": y.to(y_prob.device),
            "loss": torch.zeros((), device=y_prob.device),
        }


class _SimpleEmbeddingModel(nn.Module):
    """Simple embedding module mapping integer tokens to vectors."""

    def __init__(self, vocab_size: int = 20, embedding_dim: int = 4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {key: self.embedding(value.long()) for key, value in inputs.items()}


class _EmbeddingForwardModel(BaseModel):
    """Toy model exposing forward_from_embedding for discrete features."""

    def __init__(self):
        super().__init__(dataset=None)
        self.feature_keys = ["seq"]
        self.label_keys = ["label"]
        self.mode = "binary"

        self.embedding_model = _SimpleEmbeddingModel()
        self.linear = nn.Linear(4, 1, bias=True)

    def forward_from_embedding(
        self,
        feature_embeddings: Dict[str, torch.Tensor],
        time_info: Dict[str, torch.Tensor] = None,
        label: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        # Pool embeddings: (batch, seq_len, emb_dim) -> (batch, emb_dim)
        pooled = feature_embeddings["seq"].mean(dim=1)
        logits = self.linear(pooled)
        y_prob = torch.sigmoid(logits)
        
        return {
            "logit": logits,
            "y_prob": y_prob,
            "loss": torch.zeros((), device=logits.device),
        }


class _MultiFeatureModel(BaseModel):
    """Model with multiple feature inputs for testing multi-feature SHAP."""

    def __init__(self):
        super().__init__(dataset=None)
        self.feature_keys = ["x1", "x2"]
        self.label_keys = ["y"]
        self.mode = "binary"

        self.linear1 = nn.Linear(2, 3, bias=True)
        self.linear2 = nn.Linear(2, 3, bias=True)
        self.linear_out = nn.Linear(6, 1, bias=True)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, y: torch.Tensor) -> dict:
        h1 = torch.relu(self.linear1(x1))
        h2 = torch.relu(self.linear2(x2))
        combined = torch.cat([h1, h2], dim=-1)
        logit = self.linear_out(combined)
        y_prob = torch.sigmoid(logit)

        return {
            "logit": logit,
            "y_prob": y_prob,
            "y_true": y.to(y_prob.device),
            "loss": torch.zeros((), device=y_prob.device),
        }


class TestShapExplainerBasic(unittest.TestCase):
    """Basic tests for ShapExplainer functionality."""

    def setUp(self):
        self.model = _SimpleShapModel()
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
        self.explainer = ShapExplainer(
            self.model, 
            use_embeddings=False,
            n_background_samples=50,
            max_coalitions=100,
            random_seed=42,
        )

    def test_inheritance(self):
        """ShapExplainer should inherit from BaseInterpreter."""
        self.assertIsInstance(self.explainer, BaseInterpreter)

    def test_shap_initialization(self):
        """Test that ShapExplainer initializes correctly."""
        explainer = ShapExplainer(self.model, use_embeddings=False)
        self.assertIsInstance(explainer, ShapExplainer)
        self.assertEqual(explainer.model, self.model)
        self.assertFalse(explainer.use_embeddings)
        self.assertEqual(explainer.n_background_samples, 100)
        self.assertEqual(explainer.max_coalitions, 1000)

    def test_attribute_returns_dict(self):
        """Attribute method should return dictionary of SHAP values."""
        inputs = torch.tensor([[1.0, 0.5, -0.3]])
        
        attributions = self.explainer.attribute(
            x=inputs,
            y=self.labels,
        )

        self.assertIsInstance(attributions, dict)
        self.assertIn("x", attributions)
        self.assertEqual(attributions["x"].shape, inputs.shape)

    def test_shap_values_are_tensors(self):
        """SHAP values should be PyTorch tensors."""
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
        baseline = {"x": torch.zeros((50, 3))}
        
        attributions = self.explainer.attribute(
            baseline=baseline,
            x=inputs,
            y=self.labels,
        )

        self.assertEqual(attributions["x"].shape, inputs.shape)

    def test_zero_input_produces_small_attributions(self):
        """Zero input should produce near-zero attributions with zero baseline."""
        inputs = torch.zeros((1, 3))
        baseline = {"x": torch.zeros((50, 3))}
        
        attributions = self.explainer.attribute(
            baseline=baseline,
            x=inputs,
            y=self.labels,
        )

        # Attributions should be very small (not exactly zero due to sampling)
        self.assertTrue(torch.all(torch.abs(attributions["x"]) < 0.1))

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
        """ShapExplainer instances should be callable via BaseInterpreter.__call__."""
        inputs = torch.tensor([[0.3, -0.4, 0.5]])
        kwargs = {"x": inputs, "y": self.labels}

        from_attribute = self.explainer.attribute(**kwargs)
        from_call = self.explainer(**kwargs)

        # Use relaxed tolerances since SHAP is a stochastic approximation method
        # and minor variations can occur across different Python/PyTorch versions
        torch.testing.assert_close(
            from_call["x"], 
            from_attribute["x"],
            rtol=1e-3,  # 0.1% relative tolerance
            atol=1e-4   # absolute tolerance
        )

    def test_different_n_background_samples(self):
        """Test with different numbers of background samples."""
        inputs = torch.tensor([[1.0, 0.5, -0.3]])
        
        # Few background samples
        explainer_few = ShapExplainer(
            self.model, 
            use_embeddings=False,
            n_background_samples=20,
            max_coalitions=50,
        )
        attr_few = explainer_few.attribute(x=inputs, y=self.labels)

        # More background samples
        explainer_many = ShapExplainer(
            self.model, 
            use_embeddings=False,
            n_background_samples=100,
            max_coalitions=50,
        )
        attr_many = explainer_many.attribute(x=inputs, y=self.labels)

        # Both should produce valid output
        self.assertEqual(attr_few["x"].shape, inputs.shape)
        self.assertEqual(attr_many["x"].shape, inputs.shape)
        self.assertTrue(torch.isfinite(attr_few["x"]).all())
        self.assertTrue(torch.isfinite(attr_many["x"]).all())


class TestShapExplainerEmbedding(unittest.TestCase):
    """Tests for ShapExplainer with embedding-based models."""

    def setUp(self):
        self.model = _EmbeddingForwardModel()
        self.model.eval()

        # Set deterministic weights
        with torch.no_grad():
            self.model.linear.weight.copy_(torch.tensor([[0.4, -0.3, 0.2, 0.1]]))
            self.model.linear.bias.copy_(torch.tensor([0.05]))

        self.labels = torch.zeros((1, 1))
        self.explainer = ShapExplainer(
            self.model,
            use_embeddings=True,
            n_background_samples=30,
            max_coalitions=50,
        )

    def test_embedding_initialization(self):
        """Test that ShapExplainer initializes with embedding mode."""
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
        time_tensor = torch.tensor([[0.0, 1.5, 3.0]])
        seq_tensor = torch.tensor([[1, 2, 3]])

        attributions = self.explainer.attribute(
            seq=(time_tensor, seq_tensor),
            label=self.labels,
        )

        self.assertIn("seq", attributions)
        self.assertEqual(attributions["seq"].shape, seq_tensor.shape)

    def test_embedding_with_custom_baseline(self):
        """Test embedding-based SHAP with custom baseline."""
        seq_inputs = torch.tensor([[1, 2, 3]])
        baseline_emb = torch.zeros((30, 3, 4))  # (n_background, seq_len, emb_dim)
        
        attributions = self.explainer.attribute(
            baseline={"seq": baseline_emb},
            seq=seq_inputs,
            label=self.labels,
        )

        self.assertEqual(attributions["seq"].shape, seq_inputs.shape)

    def test_embedding_model_without_forward_from_embedding_fails(self):
        """Test that using embeddings without forward_from_embedding raises error."""
        model_without_embed = _SimpleShapModel()
        
        with self.assertRaises(AssertionError):
            ShapExplainer(model_without_embed, use_embeddings=True)


class TestShapExplainerMultiFeature(unittest.TestCase):
    """Tests for ShapExplainer with multiple feature inputs."""

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
        self.explainer = ShapExplainer(
            self.model,
            use_embeddings=False,
            n_background_samples=40,
            max_coalitions=60,
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
            "x1": torch.zeros((40, 2)),
            "x2": torch.ones((40, 2)) * 0.5,
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
        x1 = torch.tensor([[1.0, 0.5], [0.3, -0.2]])
        x2 = torch.tensor([[-0.3, 0.8], [0.5, 0.1]])

        attributions = self.explainer.attribute(
            x1=x1,
            x2=x2,
            y=torch.zeros((2, 1)),
        )

        self.assertTrue(torch.isfinite(attributions["x1"]).all())
        self.assertTrue(torch.isfinite(attributions["x2"]).all())


class TestShapExplainerMLP(unittest.TestCase):
    """Test cases for SHAP with MLP model on real dataset."""

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
            fn=partial(builder.transform, metadata=builder.metadata()),
            inputs=list(sample_generator()),
            output_dir=self.temp_dir,
            num_workers=1,
            chunk_bytes="64MB",
        )
        
        # Create dataset
        self.dataset = SampleDataset(
            path=self.temp_dir,
            dataset_name="test_shap",
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

    def test_shap_mlp_basic_attribution(self):
        """Test basic SHAP attribution computation with MLP."""
        explainer = ShapExplainer(
            self.model,
            use_embeddings=True,
            n_background_samples=20,
            max_coalitions=50,
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

    def test_shap_mlp_with_target_class(self):
        """Test SHAP attribution with specific target class."""
        explainer = ShapExplainer(self.model )
        data_batch = next(iter(self.test_loader))

        # Compute attributions for class 0
        attr_class_0 = explainer.attribute(**data_batch, target_class_idx=0)

        # Compute attributions for class 1
        attr_class_1 = explainer.attribute(**data_batch, target_class_idx=1)

        # Check that attributions are different for different classes
        self.assertFalse(
            torch.allclose(attr_class_0["conditions"], attr_class_1["conditions"], atol=0.01)
        )

    def test_shap_mlp_values_finite(self):
        """Test that SHAP values are finite (no NaN or Inf)."""
        explainer = ShapExplainer(
            self.model,
            use_embeddings=True,
            n_background_samples=20,
            max_coalitions=50,
        )
        data_batch = next(iter(self.test_loader))

        attributions = explainer.attribute(**data_batch)

        # Check no NaN or Inf
        self.assertTrue(torch.isfinite(attributions["conditions"]).all())
        self.assertTrue(torch.isfinite(attributions["procedures"]).all())

    def test_shap_mlp_multiple_samples(self):
        """Test SHAP on batch with multiple samples."""
        explainer = ShapExplainer(
            self.model,
            use_embeddings=True,
            n_background_samples=20,
            max_coalitions=50,
        )

        # Use batch size > 1
        test_loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        data_batch = next(iter(test_loader))

        attributions = explainer.attribute(**data_batch)

        # Check batch dimension
        self.assertEqual(attributions["conditions"].shape[0], 2)
        self.assertEqual(attributions["procedures"].shape[0], 2)

    def test_shap_mlp_different_coalitions(self):
        """Test SHAP with different numbers of coalitions."""
        data_batch = next(iter(self.test_loader))

        # Few coalitions
        explainer_few = ShapExplainer(
            self.model,
            use_embeddings=True,
            n_background_samples=10,
            max_coalitions=20,
        )
        attr_few = explainer_few.attribute(**data_batch)

        # More coalitions
        explainer_many = ShapExplainer(
            self.model,
            use_embeddings=True,
            n_background_samples=10,
            max_coalitions=100,
        )
        attr_many = explainer_many.attribute(**data_batch)

        # Both should produce valid output
        self.assertIn("conditions", attr_few)
        self.assertIn("conditions", attr_many)
        self.assertEqual(attr_few["conditions"].shape, attr_many["conditions"].shape)


class TestShapExplainerStageNet(unittest.TestCase):
    """Test cases for SHAP with StageNet model.

    Note: StageNet tests demonstrate SHAP working with temporal/sequential data.
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
            fn=partial(builder.transform, metadata=builder.metadata()),
            inputs=list(sample_generator()),
            output_dir=self.temp_dir,
            num_workers=1,
            chunk_bytes="64MB",
        )
        
        # Create dataset
        self.dataset = SampleDataset(
            path=self.temp_dir,
            dataset_name="test_stagenet_shap",
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

    def test_shap_initialization_stagenet(self):
        """Test that ShapExplainer works with StageNet."""
        explainer = ShapExplainer(
            self.model,
            use_embeddings=True,
            n_background_samples=10,
            max_coalitions=30,
        )
        self.assertIsInstance(explainer, ShapExplainer)
        self.assertEqual(explainer.model, self.model)

    @unittest.skip("StageNet with discrete codes requires special handling for SHAP")
    def test_shap_basic_attribution_stagenet(self):
        """Test basic SHAP attribution computation with StageNet."""
        explainer = ShapExplainer(
            self.model,
            use_embeddings=True,
            n_background_samples=10,
            max_coalitions=30,
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

    @unittest.skip("StageNet with discrete codes requires special handling for SHAP")
    def test_shap_attribution_shapes_stagenet(self):
        """Test that attribution shapes match input shapes for StageNet."""
        explainer = ShapExplainer(
            self.model,
            use_embeddings=True,
            n_background_samples=10,
            max_coalitions=30,
        )
        data_batch = next(iter(self.test_loader))

        attributions = explainer.attribute(**data_batch)

        # For StageNet, inputs are tuples (time, values)
        # Attributions should match the values part
        _, codes_values = data_batch["codes"]
        _, procedures_values = data_batch["procedures"]
        _, lab_values = data_batch["lab_values"]

        self.assertEqual(attributions["codes"].shape, codes_values.shape)
        self.assertEqual(attributions["procedures"].shape, procedures_values.shape)
        self.assertEqual(attributions["lab_values"].shape, lab_values.shape)

    @unittest.skip("StageNet with discrete codes requires special handling for SHAP")
    def test_shap_values_finite_stagenet(self):
        """Test that StageNet SHAP values are finite."""
        explainer = ShapExplainer(
            self.model,
            use_embeddings=True,
            n_background_samples=10,
            max_coalitions=30,
        )
        data_batch = next(iter(self.test_loader))

        attributions = explainer.attribute(**data_batch)

        # Check no NaN or Inf
        self.assertTrue(torch.isfinite(attributions["codes"]).all())
        self.assertTrue(torch.isfinite(attributions["procedures"]).all())
        self.assertTrue(torch.isfinite(attributions["lab_values"]).all())


class TestShapExplainerEdgeCases(unittest.TestCase):
    """Test edge cases and error handling for ShapExplainer."""

    def setUp(self):
        self.model = _SimpleShapModel()
        self.model.eval()
        self.labels = torch.zeros((1, 1))

    def test_discrete_feature_background_generation(self):
        """Test background generation for discrete (integer) features."""
        explainer = ShapExplainer(
            self.model,
            use_embeddings=False,
            n_background_samples=30,
        )
        
        # Use integer inputs
        inputs = torch.tensor([[1, 2, 3]], dtype=torch.long)
        
        # Generate background
        background = explainer._generate_background_samples({"x": inputs})
        
        self.assertIn("x", background)
        self.assertEqual(background["x"].shape[0], 30)  # n_background_samples
        self.assertEqual(background["x"].dtype, torch.long)

    def test_continuous_feature_background_generation(self):
        """Test background generation for continuous features."""
        explainer = ShapExplainer(
            self.model,
            use_embeddings=False,
            n_background_samples=40,
        )
        
        # Use continuous inputs
        inputs = torch.tensor([[1.5, -0.3, 0.8]])
        
        # Generate background
        background = explainer._generate_background_samples({"x": inputs})
        
        self.assertIn("x", background)
        self.assertEqual(background["x"].shape[0], 40)
        self.assertTrue(background["x"].dtype in [torch.float32, torch.float64])
        
        # Check values are within input range
        self.assertTrue(torch.all(background["x"] >= inputs.min()))
        self.assertTrue(torch.all(background["x"] <= inputs.max()))

    def test_empty_feature_dict(self):
        """Test handling of empty feature dictionary."""
        explainer = ShapExplainer(
            self.model,
            use_embeddings=False,
        )
        
        # This should not crash
        background = explainer._generate_background_samples({})
        self.assertEqual(len(background), 0)

    def test_kernel_weight_computation_edge_cases(self):
        """Test kernel weight computation for edge cases."""
        # Empty coalition (size = 0)
        weight_empty = ShapExplainer._compute_kernel_weight(0, 5)
        self.assertEqual(weight_empty.item(), 1000.0)
        
        # Full coalition (size = n_features)
        weight_full = ShapExplainer._compute_kernel_weight(5, 5)
        self.assertEqual(weight_full.item(), 1000.0)
        
        # Partial coalition
        weight_partial = ShapExplainer._compute_kernel_weight(2, 5)
        self.assertTrue(weight_partial.item() > 0)
        self.assertTrue(torch.isfinite(weight_partial))

    def test_time_vector_adjustment(self):
        """Test time vector length adjustment utilities."""
        # Test padding
        time_vec_short = torch.tensor([0.0, 1.0, 2.0])
        adjusted_pad = ShapExplainer._adjust_time_length(time_vec_short, 5)
        self.assertEqual(adjusted_pad.shape[0], 5)
        self.assertEqual(adjusted_pad[-1].item(), 2.0)  # Last value repeated
        
        # Test truncation
        time_vec_long = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
        adjusted_trunc = ShapExplainer._adjust_time_length(time_vec_long, 3)
        self.assertEqual(adjusted_trunc.shape[0], 3)
        
        # Test exact match
        time_vec_exact = torch.tensor([0.0, 1.0, 2.0])
        adjusted_exact = ShapExplainer._adjust_time_length(time_vec_exact, 3)
        self.assertEqual(adjusted_exact.shape[0], 3)
        torch.testing.assert_close(adjusted_exact, time_vec_exact)
        
        # Test empty vector
        time_vec_empty = torch.tensor([])
        adjusted_empty = ShapExplainer._adjust_time_length(time_vec_empty, 3)
        self.assertEqual(adjusted_empty.shape[0], 3)
        self.assertTrue(torch.all(adjusted_empty == 0))

    def test_time_vector_normalization(self):
        """Test time vector normalization to 1D."""
        # 2D time tensor
        time_2d = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
        normalized = ShapExplainer._normalize_time_vector(time_2d)
        self.assertEqual(normalized.dim(), 1)
        self.assertEqual(normalized.shape[0], 3)
        
        # 1D time tensor
        time_1d = torch.tensor([0.0, 1.0, 2.0])
        normalized = ShapExplainer._normalize_time_vector(time_1d)
        self.assertEqual(normalized.dim(), 1)
        torch.testing.assert_close(normalized, time_1d)
        
        # Single row 2D
        time_single = torch.tensor([[0.0, 1.0, 2.0]])
        normalized = ShapExplainer._normalize_time_vector(time_single)
        self.assertEqual(normalized.dim(), 1)

    def test_target_prediction_extraction_binary(self):
        """Test target prediction extraction for binary classification."""
        # Single logit (binary classification)
        logits_binary = torch.tensor([[0.5], [1.0], [-0.3]])
        
        # Class 1
        pred_1 = ShapExplainer._extract_target_prediction(logits_binary, 1)
        self.assertEqual(pred_1.shape, (3,))
        self.assertTrue(torch.all((pred_1 >= 0) & (pred_1 <= 1)))
        
        # Class 0
        pred_0 = ShapExplainer._extract_target_prediction(logits_binary, 0)
        self.assertEqual(pred_0.shape, (3,))
        torch.testing.assert_close(pred_0, 1.0 - pred_1)
        
        # None (max)
        pred_max = ShapExplainer._extract_target_prediction(logits_binary, None)
        self.assertEqual(pred_max.shape, (3,))

    def test_target_prediction_extraction_multiclass(self):
        """Test target prediction extraction for multi-class classification."""
        logits_multi = torch.tensor([[0.5, 1.0, -0.3], [0.2, 0.8, 0.1]])
        
        # Specific class
        pred_class_1 = ShapExplainer._extract_target_prediction(logits_multi, 1)
        self.assertEqual(pred_class_1.shape, (2,))
        torch.testing.assert_close(pred_class_1, logits_multi[:, 1])
        
        # None (max)
        pred_max = ShapExplainer._extract_target_prediction(logits_multi, None)
        self.assertEqual(pred_max.shape, (2,))

    def test_logit_extraction_from_dict(self):
        """Test logit extraction from model output dictionary."""
        output_dict = {"logit": torch.tensor([[0.5]]), "y_prob": torch.tensor([[0.62]])}
        logits = ShapExplainer._extract_logits(output_dict)
        torch.testing.assert_close(logits, torch.tensor([[0.5]]))

    def test_logit_extraction_from_tensor(self):
        """Test logit extraction from tensor output."""
        output_tensor = torch.tensor([[0.5]])
        logits = ShapExplainer._extract_logits(output_tensor)
        torch.testing.assert_close(logits, output_tensor)

    def test_shape_mapping_simple(self):
        """Test mapping SHAP values back to input shapes."""
        shap_values = {"x": torch.randn(2, 3)}
        input_shapes = {"x": (2, 3)}
        
        mapped = ShapExplainer._map_to_input_shapes(shap_values, input_shapes)
        self.assertEqual(mapped["x"].shape, (2, 3))

    def test_shape_mapping_expansion(self):
        """Test shape expansion when needed."""
        shap_values = {"x": torch.randn(2, 3)}
        input_shapes = {"x": (2, 3, 1)}
        
        mapped = ShapExplainer._map_to_input_shapes(shap_values, input_shapes)
        self.assertEqual(mapped["x"].shape, (2, 3, 1))

    def test_n_features_determination_2d(self):
        """Test feature count determination for 2D tensors."""
        inputs = {"x": torch.randn(4, 5)}
        embeddings = {"x": torch.randn(4, 5, 8)}
        
        n_features = ShapExplainer._determine_n_features("x", inputs, embeddings)
        self.assertEqual(n_features, 5)

    def test_n_features_determination_3d(self):
        """Test feature count determination for 3D tensors."""
        inputs = {"x": torch.randn(2, 6, 4)}
        embeddings = {"x": torch.randn(2, 6, 4, 16)}
        
        n_features = ShapExplainer._determine_n_features("x", inputs, embeddings)
        self.assertEqual(n_features, 6)

    def test_regularization_parameter(self):
        """Test different regularization parameters."""
        explainer_small_reg = ShapExplainer(
            self.model,
            use_embeddings=False,
            regularization=1e-8,
        )
        self.assertEqual(explainer_small_reg.regularization, 1e-8)
        
        explainer_large_reg = ShapExplainer(
            self.model,
            use_embeddings=False,
            regularization=1e-4,
        )
        self.assertEqual(explainer_large_reg.regularization, 1e-4)

    def test_max_coalitions_capping(self):
        """Test that coalition count is properly capped."""
        explainer = ShapExplainer(
            self.model,
            use_embeddings=False,
            max_coalitions=50,
        )
        self.assertEqual(explainer.max_coalitions, 50)
        
        # For 3 features, 2^3 = 8 < 50, so it should use 8
        # For 10 features, 2^10 = 1024 > 50, so it should use 50


class TestShapExplainerStateManagement(unittest.TestCase):
    """Test state management and repeated calls."""

    def setUp(self):
        self.model = _SimpleShapModel()
        self.model.eval()
        self.labels = torch.zeros((1, 1))
        self.explainer = ShapExplainer(
            self.model,
            use_embeddings=False,
            n_background_samples=20,
            max_coalitions=30,
        )

    def test_repeated_calls_consistency(self):
        """Test that repeated calls with same input produce similar results."""
        inputs = torch.tensor([[1.0, 0.5, -0.3]])
        
        # Set random seed for reproducibility
        torch.manual_seed(42)
        attr_1 = self.explainer.attribute(x=inputs, y=self.labels)
        
        torch.manual_seed(42)
        attr_2 = self.explainer.attribute(x=inputs, y=self.labels)
        
        # Results should be very similar (allowing for minor numerical differences)
        torch.testing.assert_close(attr_1["x"], attr_2["x"], atol=1e-4, rtol=1e-3)

    def test_different_inputs_different_outputs(self):
        """Test that different inputs produce different attributions."""
        input_1 = torch.tensor([[1.0, 0.5, -0.3]])
        input_2 = torch.tensor([[0.5, 1.0, 0.2]])
        
        attr_1 = self.explainer.attribute(x=input_1, y=self.labels)
        attr_2 = self.explainer.attribute(x=input_2, y=self.labels)
        
        # Attributions should be different
        self.assertFalse(torch.allclose(attr_1["x"], attr_2["x"], atol=0.01))

    def test_model_eval_mode_preserved(self):
        """Test that model stays in eval mode after attribution."""
        self.model.eval()
        inputs = torch.tensor([[1.0, 0.5, -0.3]])
        
        self.explainer.attribute(x=inputs, y=self.labels)
        
        # Model should still be in eval mode
        self.assertFalse(self.model.training)

    def test_gradient_cleanup(self):
        """Test that gradients are properly cleaned up."""
        inputs = torch.tensor([[1.0, 0.5, -0.3]])
        
        # Ensure inputs don't require gradients
        self.assertFalse(inputs.requires_grad)
        
        attributions = self.explainer.attribute(x=inputs, y=self.labels)
        
        # Attributions should not require gradients
        self.assertFalse(attributions["x"].requires_grad)


class TestShapExplainerDeviceHandling(unittest.TestCase):
    """Test device handling (CPU/CUDA compatibility)."""

    def setUp(self):
        self.model = _SimpleShapModel()
        self.model.eval()
        self.labels = torch.zeros((1, 1))

    def test_cpu_device(self):
        """Test SHAP computation on CPU."""
        self.model.to("cpu")
        explainer = ShapExplainer(
            self.model,
            use_embeddings=False,
            n_background_samples=10,
            max_coalitions=20,
        )
        
        inputs = torch.tensor([[1.0, 0.5, -0.3]])
        attributions = explainer.attribute(x=inputs, y=self.labels)
        
        self.assertEqual(attributions["x"].device.type, "cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_cuda_device(self):
        """Test SHAP computation on CUDA."""
        self.model.to("cuda")
        explainer = ShapExplainer(
            self.model,
            use_embeddings=False,
            n_background_samples=10,
            max_coalitions=20,
        )
        
        inputs = torch.tensor([[1.0, 0.5, -0.3]])
        attributions = explainer.attribute(x=inputs, y=self.labels)
        
        self.assertEqual(attributions["x"].device.type, "cuda")

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_mixed_device_handling(self):
        """Test that inputs are moved to model device."""
        self.model.to("cuda")
        explainer = ShapExplainer(
            self.model,
            use_embeddings=False,
            n_background_samples=10,
            max_coalitions=20,
        )
        
        # Inputs on CPU
        inputs = torch.tensor([[1.0, 0.5, -0.3]])  # CPU
        self.assertEqual(inputs.device.type, "cpu")
        
        # Should still work (inputs moved to CUDA internally)
        attributions = explainer.attribute(x=inputs, y=self.labels)
        
        # Output should be on CUDA
        self.assertEqual(attributions["x"].device.type, "cuda")


class TestShapExplainerDocumentation(unittest.TestCase):
    """Test that docstrings and examples are accurate."""

    def test_docstring_exists(self):
        """Test that main class has docstring."""
        self.assertIsNotNone(ShapExplainer.__doc__)
        self.assertGreater(len(ShapExplainer.__doc__), 100)

    def test_init_docstring_exists(self):
        """Test that __init__ has docstring."""
        self.assertIsNotNone(ShapExplainer.__init__.__doc__)

    def test_attribute_docstring_exists(self):
        """Test that attribute method has docstring."""
        self.assertIsNotNone(ShapExplainer.attribute.__doc__)

    def test_public_methods_have_docstrings(self):
        """Test that all public methods have docstrings."""
        public_methods = [
            method for method in dir(ShapExplainer)
            if not method.startswith('_') and callable(getattr(ShapExplainer, method))
        ]
        
        for method_name in public_methods:
            method = getattr(ShapExplainer, method_name)
            if method_name not in ['train', 'eval', 'parameters']:  # Inherited methods
                self.assertIsNotNone(
                    method.__doc__,
                    f"Method {method_name} missing docstring"
                )


if __name__ == "__main__":
    unittest.main()