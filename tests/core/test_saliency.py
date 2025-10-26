import unittest
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pyhealth.interpret.methods.saliency import SaliencyMaps

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

        self.model = SimpleCNN()
        self.model.eval()  # Set model to evaluation mode

    def test_gradient_saliency_map_basic(self):
        """Test basic saliency map computation."""
        dataloader = DataLoader(self.dataset, batch_size=2, shuffle=False)
        saliency = SaliencyMaps(self.model, dataloader, batches=1)
        
        # Get saliency maps
        saliency_maps = saliency.get_gradient_saliency_maps()
   
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

    def test_multiple_batches(self):
        """Test saliency map computation with multiple batches."""
        dataloader = DataLoader(self.dataset, batch_size=2, shuffle=False)
        saliency = SaliencyMaps(self.model, dataloader, batches=2)
        saliency_maps = saliency.get_gradient_saliency_maps()
        
        self.assertEqual(len(saliency_maps), 2)  # Requested 2 batches
        for batch_result in saliency_maps:
            self.assertEqual(batch_result['saliency'].shape[0], 2)  # Batch size
            self.assertTrue(torch.all(batch_result['saliency'] >= 0))

    def test_custom_keys(self):
        """Test saliency map computation with custom key names."""
        # Create dataset with different key names
        class CustomDataset(Dataset):
            def __init__(self, num_samples=10):
                self.images = torch.randn(num_samples, 3, 32, 32)
                self.labels = torch.randint(0, 2, (num_samples,))

            def __len__(self):
                return len(self.images)

            def __getitem__(self, idx):
                return {
                    'x_ray': self.images[idx],
                    'condition': self.labels[idx]
                }
        
        custom_dataset = CustomDataset()
        dataloader = DataLoader(custom_dataset, batch_size=2, shuffle=False)
        saliency = SaliencyMaps(self.model, dataloader, batches=1, 
                             image_key='x_ray', label_key='condition')
        saliency_maps = saliency.get_gradient_saliency_maps()
        
        self.assertEqual(len(saliency_maps), 1)
        batch_result = saliency_maps[0]
        self.assertEqual(batch_result['saliency'].shape[0], 2)

    def test_visualization_methods(self):
        """Test visualization method initialization (without actual plotting)."""
        import matplotlib.pyplot as plt
        
        dataloader = DataLoader(self.dataset, batch_size=2, shuffle=False)
        saliency = SaliencyMaps(self.model, dataloader, batches=1)
        
        # Just test that the method exists and can be called without errors
        # We don't test the actual plotting since it's visual output
        try:
            fig = plt.figure()  # Create figure but don't display
            saliency.imshowSaliencyCompFromDict(
                plt, 0, 0, "Test", {0: "Class 0", 1: "Class 1"}
            )
            plt.close(fig)  # Clean up
        except Exception as e:
            self.fail(f"Visualization method raised an exception: {e}")
    
if __name__ == "__main__":
    unittest.main()