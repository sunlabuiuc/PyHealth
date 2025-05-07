"""
Test case for EnhancedResNet50 model.
"""

import torch
from pyhealth.models import EnhancedResNet50

def test_enhanced_resnet50():
    """
    Test the EnhancedResNet50 model with dummy input.
    """
    model = EnhancedResNet50(num_classes=3)
    dummy_input = torch.randn(2, 3, 224, 224)
    logits, features, attn_weights = model(dummy_input)
    assert logits.shape == (2, 3), f"Expected logits shape (2, 3), got {logits.shape}"
    assert features.shape == (2, 2048), f"Expected features shape (2, 2048), got {features.shape}"
    assert attn_weights.shape == (2, 1, 7, 7), f"Expected attention weights shape (2, 1, 7, 7), got {attn_weights.shape}"
    print("EnhancedResNet50 test passed.")

if __name__ == "__main__":
    test_enhanced_resnet50()
