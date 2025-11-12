import unittest
import torch
import torch.nn as nn
from pyhealth.interpret.methods.basic_gradient import BasicGradientSaliencyMaps
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

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
    """Test basic gradient saliency methods with batch inputs."""

    def setUp(self):
        """Set up a sample model for testing."""
        self.model = SimpleCNN()
        self.model.eval()  # Set model to evaluation mode

    def test_gradient_saliency_map_basic(self):
        """Test basic saliency map computation with batch input."""
        # Create a single batch
        batch = {
            'image': torch.randn(2, 3, 32, 32),
            'disease': torch.randint(0, 2, (2,))
        }
        
        saliency = BasicGradientSaliencyMaps(self.model, input_batch=batch)
        saliency.init_gradient_saliency_maps()
        
        # Get saliency maps
        batch_maps = saliency.get_gradient_saliency_maps()
        
        # Should have batch maps
        self.assertIsNotNone(batch_maps)
        self.assertEqual(len(batch_maps), 1)
        
        batch_result = batch_maps[0]
        
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

    def test_input_batch_with_list(self):
        """Test saliency map computation with list input."""
        # Create batch as list
        images = torch.randn(2, 3, 32, 32)
        labels = torch.randint(0, 2, (2,))
        batch = [images, labels]
        
        saliency = BasicGradientSaliencyMaps(self.model, input_batch=batch)
        saliency.init_gradient_saliency_maps()
        batch_maps = saliency.get_gradient_saliency_maps()
        
        self.assertIsNotNone(batch_maps)
        self.assertEqual(len(batch_maps), 1)
        batch_result = batch_maps[0]
        self.assertIn('saliency', batch_result)
        self.assertEqual(batch_result['saliency'].shape[0], 2)

    def test_input_batch_with_tensor(self):
        """Test saliency map computation with tensor input."""
        # Create batch as single tensor
        batch = torch.randn(2, 3, 32, 32)
        
        saliency = BasicGradientSaliencyMaps(self.model, input_batch=batch)
        saliency.init_gradient_saliency_maps()
        batch_maps = saliency.get_gradient_saliency_maps()
        
        self.assertIsNotNone(batch_maps)
        self.assertEqual(len(batch_maps), 1)
        batch_result = batch_maps[0]
        self.assertIn('saliency', batch_result)
        self.assertEqual(batch_result['saliency'].shape[0], 2)

    def test_custom_keys(self):
        """Test saliency map computation with custom key names."""
        # Create batch with different key names
        batch = {
            'x_ray': torch.randn(2, 3, 32, 32),
            'condition': torch.randint(0, 2, (2,))
        }
        
        saliency = BasicGradientSaliencyMaps(
            self.model, 
            input_batch=batch,
            image_key='x_ray', 
            label_key='condition'
        )
        saliency.init_gradient_saliency_maps()
        batch_maps = saliency.get_gradient_saliency_maps()
        
        self.assertIsNotNone(batch_maps)
        self.assertEqual(len(batch_maps), 1)
        batch_result = batch_maps[0]
        self.assertEqual(batch_result['saliency'].shape[0], 2)

    def test_visualization(self):
        """Test visualization method."""
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
    
    def test_visualization_with_different_indices(self):
        """Test visualization with different image indices."""
        batch = {
            'image': torch.randn(4, 3, 32, 32),
            'disease': torch.randint(0, 2, (4,))
        }
        
        saliency = BasicGradientSaliencyMaps(self.model, input_batch=batch)
        saliency.init_gradient_saliency_maps()
        
        # Test visualization of different images
        for idx in [0, 1, 2]:
            try:
                fig = plt.figure()
                saliency.visualize_saliency_map(
                    plt, 
                    image_index=idx,
                    title=f"Test Image {idx}"
                )
                plt.close(fig)
            except Exception as e:
                self.fail(f"Visualization for index {idx} raised an exception: {e}")
    
    def test_invalid_input_type_raises_error(self):
        """Test that invalid input type raises error."""
        with self.assertRaises(ValueError):
            BasicGradientSaliencyMaps(self.model, input_batch="invalid")
    
    def test_invalid_image_index_raises_error(self):
        """Test that invalid image index raises error."""
        batch = {
            'image': torch.randn(2, 3, 32, 32),
            'disease': torch.randint(0, 2, (2,))
        }
        
        saliency = BasicGradientSaliencyMaps(self.model, input_batch=batch)
        
        with self.assertRaises((IndexError, ValueError)):
            saliency.visualize_saliency_map(plt, image_index=5)  # Out of range
    
if __name__ == "__main__":
    unittest.main()
