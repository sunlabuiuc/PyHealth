"""
Quick test script to verify LRP CNN support.
"""
import torch
import torch.nn as nn
from pyhealth.interpret.methods import LayerWiseRelevancePropagation

# Create a simple CNN model for testing
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc = nn.Linear(32 * 7 * 7, 10)
        
    def forward(self, image, **kwargs):
        x = self.conv1(image)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        logit = self.fc(x)
        return {'logit': logit, 'y_prob': torch.softmax(logit, dim=1)}

# Test the LRP implementation
print("Testing LRP with CNN...")
model = SimpleCNN()
model.eval()

# Create dummy input
dummy_image = torch.randn(1, 1, 28, 28)

# Initialize LRP
lrp = LayerWiseRelevancePropagation(
    model=model,
    rule="epsilon",
    epsilon=0.01,
    use_embeddings=False  # CNN mode
)

# Compute attributions
print("Computing attributions...")
try:
    attributions = lrp.attribute(image=dummy_image)
    print(f"✓ Success!")
    print(f"  Attribution keys: {list(attributions.keys())}")
    print(f"  Shape: {attributions['image'].shape}")
    print(f"  Total relevance: {attributions['image'].sum().item():.4f}")
    print("\nCNN support is working! 🎉")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
