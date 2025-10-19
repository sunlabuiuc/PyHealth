import unittest
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pyhealth.interpret.methods import GradientSaliencyMapping

class SimpleDataset(Dataset):
    def __init__(self, num_samples=10):
        self.images = torch.randn(num_samples, 3, 32, 32)  # Simple RGB images
        self.labels = torch.randint(0, 2, (num_samples,))  # Binary labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return {
            'image': self.images[idx],
            'disease': self.labels[idx]
        }

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, kernel_size=3, padding=1)
        self.fc = nn.Linear(1024, 2)  # 32x32 -> 1024

    def forward(self, **kwargs):
        x = kwargs['image']
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return {'y_prob': torch.softmax(x, dim=1)}

class TestSaliencyMethods(unittest.TestCase):
    """Test saliency methods on a simple dataset and model."""

    def setUp(self):
        """Set up a sample dataset and model for testing."""
        self.dataset = SimpleDataset()
        self.dataloader = DataLoader(self.dataset, batch_size=2, shuffle=False)
        self.model = SimpleCNN()
        
    def test_gradient_saliency_map(self):
        """Test saliency map computation."""
        saliency_maps = GradientSaliencyMapping(self.model, self.dataloader, batches=1)
        
        self.assertEqual(len(saliency_maps), 1)  # Requested 1 batch
        batch_result = saliency_maps[0]
        
        # Check if all expected keys are present
        self.assertIn('saliency', batch_result)
        self.assertIn('image', batch_result)
        self.assertIn('label', batch_result)
        
        # Check shapes
        saliency = batch_result['saliency']
        self.assertEqual(saliency.shape[0], 2)  # Batch size
        self.assertEqual(len(saliency.shape), 3)  # [batch_size, height, width]
        self.assertEqual(saliency.shape[1:], (32, 32))  # Height and width should match input
        
        # Check that saliency values are non-negative (due to abs())
        self.assertTrue(torch.all(saliency >= 0))
        
        # Check that gradients were computed (non-zero saliency)
        self.assertTrue(torch.any(saliency > 0))
    
if __name__ == "__main__":
    unittest.main()