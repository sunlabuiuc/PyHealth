"""
test_convnext_tiny.py

Unit tests for the custom ConvNeXt Tiny architecture integration within 
PyHealth's TorchvisionModel wrapper. Validates nested classifier 
replacement and hidden embedding extraction.

Reference:
- Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, Saining Xie. (2022).
"A ConvNet for the 2020s." IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

Author:
    Mumme, Raymond Paul rmumme2@illinois.edu
"""

import unittest
import torch
from pyhealth.models import TorchvisionModel

class DummyVisionDataset:
    """A minimal mock of a PyHealth Dataset to bypass data loading."""
    
    def __init__(self):
        """Initializes the mock dataset with dummy images and schemas."""
        self.samples = [
            {"image": torch.randn(3, 224, 224), "label": 0},
            {"image": torch.randn(3, 224, 224), "label": 1}
        ]
        self.input_schema = {"image": "image"}
        self.output_schema = {"label": "binary"}
        
        # Mock processor to satisfy BaseModel's output size calculations
        class MockProcessor:
            """Mock feature processor for label dimension calculation."""
            def size(self): 
                """Returns the expected output dimension (1 for binary)."""
                return 1
            
        self.output_processors = {"label": MockProcessor()}

    def __len__(self):
        """Returns the total number of samples in the mock dataset."""
        return len(self.samples)

    def __getitem__(self, index):
        """Retrieves a single sample from the mock dataset by index."""
        return self.samples[index]

class TestConvNeXt(unittest.TestCase):
    """Test suite specifically for the ConvNeXt additions to TorchvisionModel."""
    
    def setUp(self):
        """Initializes the mock dataset and device."""
        self.dataset = DummyVisionDataset()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_convnext_tiny_architecture(self):
        """Verifies the nested classifier replacement and forward pass for ConvNeXt."""
        model = TorchvisionModel(
            dataset=self.dataset,
            model_name="convnext_tiny",
            model_config={"weights": None}
        )
        model.to(self.device)
        model.train() # Enable gradients
        
        batch_size = 2
        mock_images = torch.randn(batch_size, 3, 224, 224).to(self.device)
        mock_labels = torch.randint(0, 2, (batch_size, 1), dtype=torch.float32).to(self.device)
        
        # Test Forward Pass
        outputs = model(image=mock_images, label=mock_labels)
        self.assertIn("loss", outputs)
        self.assertIn("y_prob", outputs)
        self.assertEqual(outputs["y_prob"].shape, (batch_size, 1))
        
        # Verify Gradient Flow through the specifically replaced classifier.2
        loss = outputs["loss"]
        loss.backward()
        self.assertIsNotNone(model.model.classifier[2].weight.grad)
        self.assertTrue(torch.any(model.model.classifier[2].weight.grad != 0))

    def test_convnext_embedding_extraction(self):
        """Verifies that ConvNeXt correctly extracts hidden embeddings."""
        model = TorchvisionModel(
            dataset=self.dataset,
            model_name="convnext_tiny",
            model_config={"weights": None}
        )
        model.to(self.device)
        model.eval()

        batch_size = 2
        mock_images = torch.randn(batch_size, 3, 224, 224).to(self.device)
        
        # Test embed=True
        embed_outputs = model(image=mock_images, embed=True)
        self.assertIn("embed", embed_outputs)
        
        # ConvNeXt tiny outputs a 768-dimensional feature vector before the classifier
        self.assertEqual(embed_outputs["embed"].shape, (batch_size, 768))

if __name__ == '__main__':
    unittest.main()