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

    def test_multihot_encoding_correctness(self):
        """Test that multi-hot encoding produces correct binary vectors.
        
        Verifies that:
        1. Vocabulary is built correctly from all unique categories
        2. Each processed sample has 1s at indices corresponding to its categories
        3. All other indices are 0
        4. Vector size matches vocabulary size
        
        Example:
            If vocabulary = {"asian": 0, "black": 1, "hispanic": 2, "non_hispanic": 3, "white": 4}
            Then ["asian", "non_hispanic"] → [1.0, 0.0, 0.0, 1.0, 0.0]
        """
        print("\n[TEST] Running: test_multihot_encoding_correctness")
        
        # Get the processor
        processor = self.dataset.input_processors["ethnicity"]
        
        # Check vocabulary was built correctly
        # Expected categories: asian, black, hispanic, non_hispanic, white (sorted alphabetically)
        expected_categories = {"asian", "black", "hispanic", "non_hispanic", "white"}
        self.assertEqual(set(processor.label_vocab.keys()), expected_categories)
        
        # Verify vocabulary size
        vocab_size = processor.size()
        self.assertEqual(vocab_size, 5)
        print(f"[TEST] Vocabulary: {processor.label_vocab}")
        
        # Test each sample's encoding - iterate through dataset directly
        # to access samples in order
        
        # Sample 0: ["asian", "non_hispanic"]
        ethnicity_0 = self.dataset[0]["ethnicity"]
        
        asian_idx = processor.label_vocab["asian"]
        non_hispanic_idx = processor.label_vocab["non_hispanic"]
        
        print(f"[TEST] Sample 0 original: ['asian', 'non_hispanic']")
        print(f"[TEST] Sample 0 encoded: {ethnicity_0.tolist()}")
        print(f"[TEST] Checking: asian_idx={asian_idx}, non_hispanic_idx={non_hispanic_idx}")
        
        # Check that asian and non_hispanic positions are 1
        self.assertEqual(ethnicity_0[asian_idx].item(), 1.0)
        self.assertEqual(ethnicity_0[non_hispanic_idx].item(), 1.0)
        
        # Check that all other positions are 0
        for i in range(vocab_size):
            if i not in [asian_idx, non_hispanic_idx]:
                self.assertEqual(ethnicity_0[i].item(), 0.0)
        
        print(f"[TEST] ✓ Sample 0 encoding verified")
        
        # Sample 1: ["white", "hispanic"]
        ethnicity_1 = self.dataset[1]["ethnicity"]
        
        white_idx = processor.label_vocab["white"]
        hispanic_idx = processor.label_vocab["hispanic"]
        
        print(f"[TEST] Sample 1 original: ['white', 'hispanic']")
        print(f"[TEST] Sample 1 encoded: {ethnicity_1.tolist()}")
        print(f"[TEST] Checking: white_idx={white_idx}, hispanic_idx={hispanic_idx}")
        
        self.assertEqual(ethnicity_1[white_idx].item(), 1.0)
        self.assertEqual(ethnicity_1[hispanic_idx].item(), 1.0)
        
        for i in range(vocab_size):
            if i not in [white_idx, hispanic_idx]:
                self.assertEqual(ethnicity_1[i].item(), 0.0)
        
        print(f"[TEST] ✓ Sample 1 encoding verified")
        
        # Sample 2: ["black"]
        ethnicity_2 = self.dataset[2]["ethnicity"]
        
        black_idx = processor.label_vocab["black"]
        
        print(f"[TEST] Sample 2 original: ['black']")
        print(f"[TEST] Sample 2 encoded: {ethnicity_2.tolist()}")
        print(f"[TEST] Checking: black_idx={black_idx}")
        
        self.assertEqual(ethnicity_2[black_idx].item(), 1.0)
        
        for i in range(vocab_size):
            if i != black_idx:
                self.assertEqual(ethnicity_2[i].item(), 0.0)
        
        print(f"[TEST] ✓ Sample 2 encoding verified")
        print("[TEST] ✓ test_multihot_encoding_correctness passed")

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

