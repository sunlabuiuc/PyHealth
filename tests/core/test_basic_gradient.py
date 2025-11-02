import unittest
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pyhealth.interpret.methods.basic_gradient import BasicGradientSaliencyMaps
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

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

class TestBasicGradient(unittest.TestCase):
    """Test basic gradient saliency methods on a simple dataset and model."""

    def setUp(self):
        """Set up a sample dataset and model for testing."""
        self.dataset = SimpleDataset()
        self.model = SimpleCNN()
        self.model.eval()  # Set model to evaluation mode

    def test_gradient_saliency_map_basic(self):
        """Test basic saliency map computation with dataloader."""
        dataloader = DataLoader(self.dataset, batch_size=2, shuffle=False)
        saliency = BasicGradientSaliencyMaps(self.model, dataloader=dataloader, batches=1)
        saliency.init_gradient_saliency_maps()
        
        # Get saliency maps - now returns (batch_maps, dataloader_maps)
        batch_maps, dataloader_maps = saliency.get_gradient_saliency_maps()
        
        # Should have dataloader maps but no batch maps
        self.assertIsNone(batch_maps)
        self.assertIsNotNone(dataloader_maps)
        self.assertEqual(len(dataloader_maps), 1)  # Requested 1 batch
        
        batch_result = dataloader_maps[0]
        
        # Check if all expected keys are present
        self.assertIn('saliency', batch_result)
        self.assertIn('image', batch_result)
        self.assertIn('label', batch_result)
        
        # Check shapes
        saliency_map = batch_result['saliency']
        self.assertEqual(saliency_map.shape[0], 2)  # Batch size
        self.assertEqual(len(saliency_map.shape), 3)  # [batch_size, height, width] after max over channels
        self.assertEqual(saliency_map.shape[1:], (32, 32))  # Height and width should match input
        
        # Check that saliency values are non-negative (due to abs())
        self.assertTrue(torch.all(saliency_map >= 0))
        
        # Check that gradients were computed (non-zero saliency)
        self.assertTrue(torch.any(saliency_map > 0))

    def test_multiple_batches(self):
        """Test saliency map computation with multiple batches."""
        dataloader = DataLoader(self.dataset, batch_size=2, shuffle=False)
        saliency = BasicGradientSaliencyMaps(self.model, dataloader=dataloader, batches=2)
        saliency.init_gradient_saliency_maps()
        batch_maps, dataloader_maps = saliency.get_gradient_saliency_maps()
        
        self.assertIsNone(batch_maps)
        self.assertEqual(len(dataloader_maps), 2)  # Requested 2 batches
        for batch_result in dataloader_maps:
            self.assertEqual(batch_result['saliency'].shape[0], 2)  # Batch size
            self.assertTrue(torch.all(batch_result['saliency'] >= 0))
    
    def test_input_batch(self):
        """Test saliency map computation with direct input batch."""
        # Create a single batch
        batch = {
            'image': torch.randn(2, 3, 32, 32),
            'disease': torch.randint(0, 2, (2,))
        }
        
        saliency = BasicGradientSaliencyMaps(self.model, input_batch=batch)
        saliency.init_gradient_saliency_maps()
        batch_maps, dataloader_maps = saliency.get_gradient_saliency_maps()
        
        # Should have batch maps but no dataloader maps
        self.assertIsNotNone(batch_maps)
        self.assertIsNone(dataloader_maps)
        self.assertEqual(len(batch_maps), 1)
        
        batch_result = batch_maps[0]
        self.assertIn('saliency', batch_result)
        self.assertEqual(batch_result['saliency'].shape[0], 2)
    
    def test_both_inputs(self):
        """Test saliency map computation with both dataloader and input batch."""
        dataloader = DataLoader(self.dataset, batch_size=2, shuffle=False)
        batch = {
            'image': torch.randn(2, 3, 32, 32),
            'disease': torch.randint(0, 2, (2,))
        }
        
        saliency = BasicGradientSaliencyMaps(
            self.model, 
            dataloader=dataloader, 
            input_batch=batch,
            batches=1
        )
        saliency.init_gradient_saliency_maps()
        batch_maps, dataloader_maps = saliency.get_gradient_saliency_maps()
        
        # Should have both
        self.assertIsNotNone(batch_maps)
        self.assertIsNotNone(dataloader_maps)
        self.assertEqual(len(batch_maps), 1)
        self.assertEqual(len(dataloader_maps), 1)

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
        saliency = BasicGradientSaliencyMaps(
            self.model, 
            dataloader=dataloader, 
            batches=1, 
            image_key='x_ray', 
            label_key='condition'
        )
        saliency.init_gradient_saliency_maps()
        batch_maps, dataloader_maps = saliency.get_gradient_saliency_maps()
        
        self.assertIsNone(batch_maps)
        self.assertEqual(len(dataloader_maps), 1)
        batch_result = dataloader_maps[0]
        self.assertEqual(batch_result['saliency'].shape[0], 2)

    def test_visualization_with_batch_index(self):
        """Test visualization method with batch_index (using dataloader)."""
        dataloader = DataLoader(self.dataset, batch_size=2, shuffle=False)
        saliency = BasicGradientSaliencyMaps(self.model, dataloader=dataloader, batches=1)
        saliency.init_gradient_saliency_maps()
        
        # Test that the method can be called without errors
        try:
            fig = plt.figure()
            saliency.visualize_saliency_map(
                plt, 
                image_index=0, 
                batch_index=0,
                title="Test", 
                id2label={0: "Class 0", 1: "Class 1"}
            )
            plt.close(fig)
        except Exception as e:
            self.fail(f"Visualization method raised an exception: {e}")
    
    def test_visualization_without_batch_index(self):
        """Test visualization method without batch_index (using input_batch)."""
        batch = {
            'image': torch.randn(2, 3, 32, 32),
            'disease': torch.randint(0, 2, (2,))
        }
        
        saliency = BasicGradientSaliencyMaps(self.model, input_batch=batch)
        saliency.init_gradient_saliency_maps()
        
        # Test that the method can be called without errors
        try:
            fig = plt.figure()
            saliency.visualize_saliency_map(
                plt, 
                image_index=0,
                title="Test", 
                id2label={0: "Class 0", 1: "Class 1"}
            )
            plt.close(fig)
        except Exception as e:
            self.fail(f"Visualization method raised an exception: {e}")
    
    def test_no_inputs_raises_error(self):
        """Test that initialization without dataloader or input_batch raises error."""
        with self.assertRaises(ValueError):
            BasicGradientSaliencyMaps(self.model)
    
    def test_invalid_batch_index_raises_error(self):
        """Test that invalid batch_index raises error."""
        dataloader = DataLoader(self.dataset, batch_size=2, shuffle=False)
        saliency = BasicGradientSaliencyMaps(self.model, dataloader=dataloader, batches=1)
        saliency.init_gradient_saliency_maps()
        
        with self.assertRaises(ValueError):
            saliency.visualize_saliency_map(
                plt, 
                image_index=0, 
                batch_index=5,  # Out of range
                title="Test"
            )
    
if __name__ == "__main__":
    unittest.main()
