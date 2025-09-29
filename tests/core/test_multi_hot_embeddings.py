import unittest

import torch

from pyhealth.datasets import SampleDataset, get_dataloader
from pyhealth.models.embedding import EmbeddingModel
from pyhealth.models.mlp import MLP


class TestMultiHotEmbeddings(unittest.TestCase):
    """Test suite for MultiHotProcessor integration with EmbeddingModel and MLP.
    
    Tests the end-to-end flow of using multi-hot encoded categorical features
    (e.g., patient ethnicity) as inputs to neural network models.
    """
    
    def setUp(self) -> None:
        self.samples = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "ethnicity": ["asian", "non_hispanic"],
                "label": 0,
            },
            {
                "patient_id": "patient-1",
                "visit_id": "visit-0",
                "ethnicity": ["white", "hispanic"],
                "label": 1,
            },
            {
                "patient_id": "patient-2",
                "visit_id": "visit-0",
                "ethnicity": ["black"],
                "label": 1,
            },
        ]

        self.input_schema = {"ethnicity": "multi_hot"}
        self.output_schema = {"label": "binary"}

        self.dataset = SampleDataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            dataset_name="test-multihot",
        )

    def test_embedding_model_linear_projection(self):
        """Test that MultiHotProcessor outputs are linearly projected to embedding_dim.
        
        Verifies that:
        1. Multi-hot encoded vectors (batch, num_categories) are transformed
        2. Output has the correct embedding dimension (batch, embedding_dim)
        3. The transformation is non-trivial (not a passthrough)
        """
        print("\n[TEST] Running: test_embedding_model_linear_projection")
        embedding_model = EmbeddingModel(self.dataset, embedding_dim=16)
        dataloader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        batch = next(iter(dataloader))
        ethnicity = batch["ethnicity"]
        embedded = embedding_model({"ethnicity": ethnicity})
        self.assertEqual(embedded["ethnicity"].shape[-1], 16)
        self.assertFalse(torch.equal(embedded["ethnicity"], ethnicity.to(embedding_model.device)))
        print("[TEST] ✓ test_embedding_model_linear_projection passed")

    def test_mlp_forward_with_multihot(self):
        """Test full MLP forward pass with multi-hot categorical features.
        
        Verifies that:
        1. MLP can process multi-hot encoded inputs through embedding layer
        2. Forward pass produces all expected outputs (loss, y_prob, y_true, logit)
        3. End-to-end training pipeline works with categorical demographics
        
        Example flow:
        - ethnicity: ["asian", "non_hispanic"] → multi-hot [1,0,0,1,0] 
          → linear projection → (embedding_dim) → MLP → prediction
        """
        print("\n[TEST] Running: test_mlp_forward_with_multihot")
        model = MLP(dataset=self.dataset, embedding_dim=16)
        dataloader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        batch = next(iter(dataloader))
        with torch.no_grad():
            ret = model(**batch)
        self.assertIn("loss", ret)
        self.assertIn("y_prob", ret)
        self.assertIn("y_true", ret)
        self.assertIn("logit", ret)
        print("[TEST] ✓ test_mlp_forward_with_multihot passed")


if __name__ == "__main__":
    unittest.main()

