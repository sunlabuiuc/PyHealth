"""
test_dinov2.py

Unit tests for the DINOv2 foundation model integration in PyHealth.
Uses a mock dataset to isolate model architecture, forward pass, and 
gradient flow validation without requiring heavy data loading.

Reference: 
- Oquab, M., et al. (2024). DINOv2: Learning Robust Visual Features without Supervision. TMLR.

Author:
    Mumme, Raymond Paul rmumme2@illinois.edu
"""

import unittest
import torch
from pyhealth.models import DINOv2

class DummyDermoscopyDataset:
    """A minimal mock of the PyHealth DermoscopyDataset to bypass data loading."""
    def __init__(self):
        self.samples = [
            {"image": torch.randn(3, 224, 224), "melanoma": 0},
            {"image": torch.randn(3, 224, 224), "melanoma": 1}
        ]
        self.input_schema = {"image": "image"}
        self.output_schema = {"melanoma": "binary"}
        
        # Create a mock processor that tells get_output_size() the label dim is 1
        class MockProcessor:
            def size(self): return 1
            
        self.output_processors = {"melanoma": MockProcessor()}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]

class TestDINOv2(unittest.TestCase):
    """Test suite for the DINOv2 model architecture and freezing logic."""

    @classmethod
    def setUpClass(cls):
        """Runs once for the entire test class to prevent reloading weights."""
        cls.dataset = DummyDermoscopyDataset()
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Add the required BaseModel schema arguments
        cls.model = DINOv2(
            dataset=cls.dataset,
            feature_keys=["image"],
            label_key="melanoma",
            mode="binary",
            model_size="vits14"
        )
        cls.model.to(cls.device)

    def test_initialization(self):
        """Verifies that the model initializes correctly with PyHealth kwargs."""
        self.assertIsNotNone(self.__class__.model)
        self.assertEqual(self.__class__.model.mode, "binary")
        self.assertIn("image", self.__class__.model.feature_keys)

    def test_forward_pass_and_gradients(self):
        """Verifies forward pass, shapes, and gradient flow (required by rubric)."""
        batch_size = 4
        mock_images = torch.randn(batch_size, 3, 224, 224).to(self.__class__.device)
        mock_labels = torch.randint(0, 2, (batch_size,)).to(self.__class__.device)
        
        batch = {
            "image": mock_images,
            "melanoma": mock_labels
        }

        # Forward Pass
        self.__class__.model.train() # Set to train to enable gradients
        self.__class__.model.zero_grad() # Clear gradients from any prior tests
        outputs = self.__class__.model(**batch)

        self.assertIn("loss", outputs)
        self.assertIn("y_prob", outputs)
        self.assertEqual(outputs["y_prob"].shape, (batch_size, 1))
        
        # Gradient Computation
        loss = outputs["loss"]
        loss.backward()
        
        # Check if gradients propagated to the linear classifier head
        self.assertIsNotNone(self.__class__.model.fc.weight.grad)
        self.assertTrue(torch.any(self.__class__.model.fc.weight.grad != 0))

        # Verify the backbone is actually frozen (required for Linear Probing)
        # The first parameter of the backbone should have NO gradient
        backbone_param = next(self.__class__.model.backbone.parameters())
        self.assertIsNone(backbone_param.grad)
    
    def test_embedding_extraction(self):
        """Verifies the embed=True mode and forward_from_embedding."""
        batch_size = 2
        mock_images = torch.randn(batch_size, 3, 224, 224).to(self.__class__.device)
        batch = {"image": mock_images}

        # Extract embeddings
        self.__class__.model.eval()
        embed_outputs = self.__class__.model(embed=True, **batch)
        self.assertIn("embed", embed_outputs)
        
        # vits14 should output 384 dimensions
        embeddings = embed_outputs["embed"]
        self.assertEqual(embeddings.shape, (batch_size, 384))

        # Test forward_from_embedding
        final_outputs = self.__class__.model.forward_from_embedding(embeddings)
        self.assertIn("y_prob", final_outputs)

if __name__ == '__main__':
    unittest.main()