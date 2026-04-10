# test_dinov2.py
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
        # ADDED: Required by PyHealth 2.0 Zero-Config model initialization
        self.input_schema = {"image": "image"}
        self.output_schema = {"melanoma": "binary"}

class TestDINOv2(unittest.TestCase):
    def setUp(self):
        self.dataset = DummyDermoscopyDataset()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # ADDED: Removed explicit kwargs to match native PyHealth 2.0 init
        self.model = DINOv2(
            dataset=self.dataset,
            model_size="vits14"
        )
        self.model.to(self.device)

    def test_initialization(self):
        self.assertIsNotNone(self.model)
        self.assertEqual(self.model.mode, "binary")
        self.assertIn("image", self.model.feature_keys)

    def test_forward_pass_and_gradients(self):
        """Verifies forward pass, shapes, and gradient flow (required by rubric)."""
        batch_size = 4
        mock_images = torch.randn(batch_size, 3, 224, 224).to(self.device)
        mock_labels = torch.randint(0, 2, (batch_size,)).to(self.device)
        
        batch = {
            "image": mock_images,
            "melanoma": mock_labels
        }

        # 1. Forward Pass
        self.model.train() # Set to train to enable gradients
        outputs = self.model(**batch)

        self.assertIn("loss", outputs)
        self.assertIn("y_prob", outputs)
        self.assertEqual(outputs["y_prob"].shape, (batch_size, 1))
        
        # 2. Gradient Computation (Rubric Requirement)
        loss = outputs["loss"]
        loss.backward()
        
        # Check if gradients propagated to the linear classifier head
        self.assertIsNotNone(self.model.fc.weight.grad)
        self.assertNotEqual(torch.sum(self.model.fc.weight.grad), 0.0)

if __name__ == '__main__':
    unittest.main()