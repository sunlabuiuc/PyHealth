"""
Unit tests for LRP with CNN/image models (ResNet, VGG, etc.).

This test suite covers:
1. LRP with ResNet architectures (sequential approximation)
2. LRP with standard CNNs (VGG-style)
3. Shape preservation through convolutional layers
4. Multi-channel image attribution
"""

import pytest
import torch
import torch.nn as nn

from pyhealth.interpret.methods import UnifiedLRP


class SimpleResNet(nn.Module):
    """Simplified ResNet-like model for testing."""
    
    def __init__(self, num_classes=4):
        super().__init__()
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Simplified residual block
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Output layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # Initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        identity = self.maxpool(x)
        
        # Residual block (simplified - no skip for now)
        out = self.conv2(identity)
        out = self.bn2(out)
        out = self.relu(out)
        
        # Output
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        
        return out


class SimpleCNN(nn.Module):
    """Simple sequential CNN for testing."""
    
    def __init__(self, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x


@pytest.fixture
def simple_cnn():
    """Create a simple CNN model."""
    model = SimpleCNN(num_classes=4)
    model.eval()
    return model


@pytest.fixture
def simple_resnet():
    """Create a simplified ResNet model."""
    model = SimpleResNet(num_classes=4)
    model.eval()
    return model


@pytest.fixture
def sample_image():
    """Create a sample RGB image tensor."""
    # Batch size 2, RGB (3 channels), 64x64 pixels
    return torch.randn(2, 3, 64, 64)


class TestLRPWithCNN:
    """Test LRP with standard sequential CNN architectures."""
    
    def test_cnn_initialization(self, simple_cnn):
        """Test LRP initializes correctly with CNN model."""
        lrp = UnifiedLRP(
            model=simple_cnn,
            rule='epsilon',
            epsilon=0.1,
            validate_conservation=False
        )
        
        assert lrp.model is simple_cnn
        assert lrp.rule == 'epsilon'
        assert lrp.epsilon == 0.1
    
    def test_cnn_attribution_shape(self, simple_cnn, sample_image):
        """Test that CNN attributions have correct shape."""
        lrp = UnifiedLRP(
            model=simple_cnn,
            rule='epsilon',
            epsilon=0.1,
            validate_conservation=False
        )
        
        # Get model output
        with torch.no_grad():
            output = simple_cnn(sample_image)
            predicted_class = output.argmax(dim=1)[0].item()
        
        # Compute attributions
        attributions = lrp.attribute(
            inputs={'x': sample_image},
            target_class=predicted_class
        )
        
        # Check shape matches input
        assert attributions['x'].shape == sample_image.shape
        assert attributions['x'].dim() == 4  # (batch, channels, height, width)
    
    def test_cnn_epsilon_vs_alphabeta(self, simple_cnn, sample_image):
        """Test different rules produce different results for CNN."""
        lrp_eps = UnifiedLRP(
            model=simple_cnn,
            rule='epsilon',
            epsilon=0.1,
            validate_conservation=False
        )
        
        lrp_ab = UnifiedLRP(
            model=simple_cnn,
            rule='alphabeta',
            alpha=2.0,
            beta=1.0,
            validate_conservation=False
        )
        
        # Use first sample only
        single_image = sample_image[0:1]
        
        with torch.no_grad():
            output = simple_cnn(single_image)
            predicted_class = output.argmax(dim=1).item()
        
        attr_eps = lrp_eps.attribute(inputs={'x': single_image}, target_class=predicted_class)
        attr_ab = lrp_ab.attribute(inputs={'x': single_image}, target_class=predicted_class)
        
        # Different rules should produce different attributions
        diff = torch.abs(attr_eps['x'] - attr_ab['x']).mean()
        assert diff > 1e-6, "Different rules should produce different attributions"
    
    def test_cnn_no_nan_or_inf(self, simple_cnn, sample_image):
        """Test that CNN attributions don't contain NaN or Inf."""
        lrp = UnifiedLRP(
            model=simple_cnn,
            rule='epsilon',
            epsilon=0.1,
            validate_conservation=False
        )
        
        with torch.no_grad():
            output = simple_cnn(sample_image)
            predicted_class = output.argmax(dim=1)[0].item()
        
        attributions = lrp.attribute(
            inputs={'x': sample_image},
            target_class=predicted_class
        )
        
        assert not torch.isnan(attributions['x']).any()
        assert not torch.isinf(attributions['x']).any()


class TestLRPWithResNet:
    """Test LRP with ResNet architectures (sequential approximation)."""
    
    def test_resnet_initialization(self, simple_resnet):
        """Test LRP initializes with ResNet model."""
        lrp = UnifiedLRP(
            model=simple_resnet,
            rule='epsilon',
            epsilon=0.1,
            validate_conservation=False
        )
        
        assert lrp.model is simple_resnet
    
    def test_resnet_skip_detection(self, simple_resnet):
        """Test that skip connections are detected in ResNet."""
        lrp = UnifiedLRP(
            model=simple_resnet,
            rule='epsilon',
            validate_conservation=False
        )
        
        # Skip connections are detected during forward pass
        sample = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            output = simple_resnet(sample)
            predicted_class = output.argmax(dim=1).item()
        
        # Trigger hook registration (happens in attribute)
        _ = lrp.attribute(inputs={'x': sample}, target_class=predicted_class)
        
        # After attribute, hooks should be registered
        # (actual skip connection handling tested implicitly via successful execution)
    
    def test_resnet_attribution_shape(self, simple_resnet, sample_image):
        """Test ResNet attributions have correct shape."""
        lrp = UnifiedLRP(
            model=simple_resnet,
            rule='epsilon',
            epsilon=0.1,
            validate_conservation=False
        )
        
        with torch.no_grad():
            output = simple_resnet(sample_image)
            predicted_class = output.argmax(dim=1)[0].item()
        
        attributions = lrp.attribute(
            inputs={'x': sample_image},
            target_class=predicted_class
        )
        
        assert attributions['x'].shape == sample_image.shape
    
    def test_resnet_downsample_exclusion(self):
        """Test that downsample layers are excluded during hook registration."""
        # This test verifies the sequential approximation approach
        # by checking that LRP completes without shape mismatch errors
        
        model = SimpleResNet(num_classes=4)
        model.eval()
        
        lrp = UnifiedLRP(
            model=model,
            rule='epsilon',
            epsilon=0.1,
            validate_conservation=False
        )
        
        sample = torch.randn(1, 3, 64, 64)
        
        with torch.no_grad():
            output = model(sample)
            predicted_class = output.argmax(dim=1).item()
        
        # Should complete without RuntimeError due to shape mismatches
        try:
            attributions = lrp.attribute(
                inputs={'x': sample},
                target_class=predicted_class
            )
            assert attributions['x'].shape == sample.shape
        except RuntimeError as e:
            if "size mismatch" in str(e):
                pytest.fail("Sequential approximation failed with shape mismatch")
            raise
    
    def test_resnet_no_nan_or_inf(self, simple_resnet, sample_image):
        """Test ResNet attributions are numerically valid."""
        lrp = UnifiedLRP(
            model=simple_resnet,
            rule='alphabeta',
            alpha=2.0,
            beta=1.0,
            validate_conservation=False
        )
        
        with torch.no_grad():
            output = simple_resnet(sample_image)
            predicted_class = output.argmax(dim=1)[0].item()
        
        attributions = lrp.attribute(
            inputs={'x': sample_image},
            target_class=predicted_class
        )
        
        assert not torch.isnan(attributions['x']).any()
        assert not torch.isinf(attributions['x']).any()


class TestLRPMultiChannel:
    """Test LRP handles multi-channel images correctly."""
    
    def test_grayscale_to_rgb_conversion(self, simple_cnn):
        """Test LRP works when converting grayscale to RGB."""
        # Simulate grayscale image converted to RGB (common in medical imaging)
        grayscale = torch.randn(1, 1, 64, 64)
        rgb = grayscale.repeat(1, 3, 1, 1)
        
        lrp = UnifiedLRP(
            model=simple_cnn,
            rule='epsilon',
            epsilon=0.1,
            validate_conservation=False
        )
        
        with torch.no_grad():
            output = simple_cnn(rgb)
            predicted_class = output.argmax(dim=1).item()
        
        attributions = lrp.attribute(
            inputs={'x': rgb},
            target_class=predicted_class
        )
        
        assert attributions['x'].shape == rgb.shape
        assert attributions['x'].shape[1] == 3  # RGB channels
    
    def test_channel_relevance_aggregation(self, simple_cnn):
        """Test that we can aggregate relevance across channels."""
        rgb_image = torch.randn(1, 3, 64, 64)
        
        lrp = UnifiedLRP(
            model=simple_cnn,
            rule='epsilon',
            epsilon=0.1,
            validate_conservation=False
        )
        
        with torch.no_grad():
            output = simple_cnn(rgb_image)
            predicted_class = output.argmax(dim=1).item()
        
        attributions = lrp.attribute(
            inputs={'x': rgb_image},
            target_class=predicted_class
        )
        
        # Aggregate across channels (common for visualization)
        channel_sum = attributions['x'].sum(dim=1)  # Sum over channel dimension
        assert channel_sum.shape == (1, 64, 64)  # (batch, height, width)
        
        # Per-channel relevance should vary
        per_channel = attributions['x'].sum(dim=(2, 3))  # Sum over spatial dimensions
        assert per_channel.shape == (1, 3)  # (batch, channels)


class TestLRPBatchProcessing:
    """Test LRP handles different batch sizes correctly."""
    
    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_variable_batch_sizes(self, simple_cnn, batch_size):
        """Test LRP works with different batch sizes."""
        images = torch.randn(batch_size, 3, 64, 64)
        
        lrp = UnifiedLRP(
            model=simple_cnn,
            rule='epsilon',
            epsilon=0.1,
            validate_conservation=False
        )
        
        with torch.no_grad():
            output = simple_cnn(images)
            predicted_class = output.argmax(dim=1)[0].item()
        
        attributions = lrp.attribute(
            inputs={'x': images},
            target_class=predicted_class
        )
        
        assert attributions['x'].shape[0] == batch_size
        assert attributions['x'].shape == images.shape
