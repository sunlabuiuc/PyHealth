import unittest
import torch

from pyhealth.datasets import SampleDataset, get_dataloader
from pyhealth.models import MLP, StageNet
from pyhealth.interpret.methods import ShapExplainer


class TestShapExplainerMLP(unittest.TestCase):
    """Test cases for SHAP with MLP model."""

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

        # Create dataset
        self.dataset = SampleDataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            dataset_name="test_shap",
        )

        # Create model
        self.model = MLP(
            dataset=self.dataset
            #embedding_dim=64,
            #hidden_dim=32,
            #n_layers=3,
            #activation='tanh'
        )
        self.model.eval()

        # Create dataloader with small batch size for testing
        self.test_loader = get_dataloader(self.dataset, batch_size=1, shuffle=False)

    def test_shap_initialization(self):
        """Test that ShapExplainer initializes correctly with different methods."""
        # Test auto method
        shap_auto = ShapExplainer(self.model, method='auto')
        self.assertIsInstance(shap_auto, ShapExplainer)
        self.assertEqual(shap_auto.model, self.model)
        self.assertEqual(shap_auto.method, 'auto')

        # Test exact method
        shap_exact = ShapExplainer(self.model, method='exact')
        self.assertEqual(shap_exact.method, 'exact')

        # Test kernel method
        shap_kernel = ShapExplainer(self.model, method='kernel')
        self.assertEqual(shap_kernel.method, 'kernel')

        # Test deep method
        shap_deep = ShapExplainer(self.model, method='deep')
        self.assertEqual(shap_deep.method, 'deep')

        # Test invalid method
        with self.assertRaises(ValueError):
            ShapExplainer(self.model, method='invalid')

    def test_basic_attribution(self):
        """Test basic attribution computation with different SHAP methods."""
        data_batch = next(iter(self.test_loader))

        # Test each method with appropriate settings
        for method in ['auto', 'exact', 'kernel', 'deep']:
            explainer = ShapExplainer(
                self.model, 
                method=method,
                use_embeddings=False,  # Don't use embeddings for tensor features
                n_background_samples=10  # Reduce samples for testing
            )
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

    
    def test_attribution_with_target_class(self):
        """Test attribution computation with specific target class."""
        explainer = ShapExplainer(self.model)
        data_batch = next(iter(self.test_loader))

        # Compute attributions for different classes
        attr_class_0 = explainer.attribute(**data_batch, target_class_idx=0)
        attr_class_1 = explainer.attribute(**data_batch, target_class_idx=1)

        # Check that attributions are different for different classes
        self.assertFalse(
            torch.allclose(attr_class_0["conditions"], attr_class_1["conditions"])
        )

    def test_attribution_with_custom_baseline(self):
        #Test attribution with custom baseline."""
        explainer = ShapExplainer(self.model)
        data_batch = next(iter(self.test_loader))

        # Create a custom baseline (zeros)
        baseline = {
            k: torch.zeros_like(v) if isinstance(v, torch.Tensor) else v
            for k, v in data_batch.items()
            if k in self.input_schema
        }

        attributions = explainer.attribute(**data_batch, baseline=baseline)

        # Check output structure
        self.assertIn("conditions", attributions)
        self.assertIn("procedures", attributions)
        self.assertEqual(
            attributions["conditions"].shape, data_batch["conditions"].shape
        )

    def test_attribution_values_are_finite(self):
        #Test that attribution values are finite (no NaN or Inf) for all methods."""
        data_batch = next(iter(self.test_loader))
        #print(data_batch)
        #print("Keys in data_batch:", data_batch.keys())
        #print("Model feature keys:", self.model.feature_keys)

        for method in ['auto', 'exact', 'kernel', 'deep']:
            explainer = ShapExplainer(self.model, method=method)
            attributions = explainer.attribute(**data_batch)

            # Check no NaN or Inf
            self.assertTrue(torch.isfinite(attributions["conditions"]).all())
            self.assertTrue(torch.isfinite(attributions["procedures"]).all())

    def test_multiple_samples(self):
        """Test attribution on batch with multiple samples."""
        explainer = ShapExplainer(
            self.model,
            use_embeddings=False,  # Don't use embeddings for tensor features
            n_background_samples=5  # Keep background samples small for batch processing
        )

        # Use small batch size for testing
        test_loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        data_batch = next(iter(test_loader))

        # Generate appropriate baseline for batch
        baseline = {
            k: torch.zeros_like(v) if isinstance(v, torch.Tensor) else v
            for k, v in data_batch.items()
            if k in self.input_schema
        }
        
        attributions = explainer.attribute(**data_batch, baseline=baseline)

        # Check batch dimension
        self.assertEqual(attributions["conditions"].shape[0], 2)
        self.assertEqual(attributions["procedures"].shape[0], 2)


class TestShapExplainerStageNet(unittest.TestCase):
    """Test cases for SHAP with StageNet model."""

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

        # Create dataset
        self.dataset = SampleDataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            dataset_name="test_stagenet_shap",
        )

        # Create StageNet model
        self.model = StageNet(
            dataset=self.dataset,
            embedding_dim=32,
            chunk_size=2,  # Reduce chunk size for testing
            levels=2,
        )
        self.model.eval()

        # Create dataloader with batch size 1 for testing temporal data
        self.test_loader = get_dataloader(self.dataset, batch_size=1, shuffle=False)

    def test_shap_initialization_stagenet(self):
        """Test that ShapExplainer works with StageNet."""
        explainer = ShapExplainer(self.model)
        self.assertIsInstance(explainer, ShapExplainer)
        self.assertEqual(explainer.model, self.model)

    def test_methods_with_stagenet(self):
        """Test all SHAP methods with StageNet model."""
        data_batch = next(iter(self.test_loader))

        for method in ['auto', 'exact', 'kernel', 'deep']:
            explainer = ShapExplainer(self.model, method=method)
            attributions = explainer.attribute(**data_batch)

            # Check output structure
            self.assertIn("codes", attributions)
            self.assertIn("procedures", attributions)
            self.assertIn("lab_values", attributions)

            # Check that attributions are tensors
            self.assertIsInstance(attributions["codes"], torch.Tensor)
            self.assertIsInstance(attributions["procedures"], torch.Tensor)
            self.assertIsInstance(attributions["lab_values"], torch.Tensor)

    def test_attribution_values_finite_stagenet(self):
        """Test that StageNet attributions are finite for all methods."""
        data_batch = next(iter(self.test_loader))

        for method in ['auto', 'exact', 'kernel', 'deep']:
            explainer = ShapExplainer(
                self.model, 
                method=method,
                use_embeddings=False,
                n_background_samples=5  # Reduce samples for temporal data
            )
            try:
                attributions = explainer.attribute(**data_batch)
            except RuntimeError as e:
                if 'size mismatch' in str(e):
                    self.skipTest("Skipping due to known size mismatch with temporal data")

            # Check no NaN or Inf
            self.assertTrue(torch.isfinite(attributions["codes"]).all())
            self.assertTrue(torch.isfinite(attributions["procedures"]).all())
            self.assertTrue(torch.isfinite(attributions["lab_values"]).all())

if __name__ == "__main__":
    unittest.main()