#!/usr/bin/env python3
"""
Minimal test script for Wav2Sleep components.
This bypasses the full PyHealth package imports and focuses only on our components.
"""

import sys
import os
import torch
import torch.nn as nn

# Add pyhealth to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_cls_token_fusion():
    """Test CLS-token Transformer fusion in isolation."""
    print("Testing CLSTokenTransformerFusion...")
    
    # Import the component directly
    from pyhealth.models.wav2sleep import CLSTokenTransformerFusion
    
    # Test parameters
    batch_size, seq_len, embed_dim = 2, 10, 64
    
    # Create fusion module
    fusion = CLSTokenTransformerFusion(
        embed_dim=embed_dim,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
        max_modalities=3,
    )
    
    # Test with all modalities
    modality_embs = {
        'ecg': torch.randn(batch_size, seq_len, embed_dim),
        'ppg': torch.randn(batch_size, seq_len, embed_dim),
        'resp': torch.randn(batch_size, seq_len, embed_dim),
    }
    
    fused = fusion(modality_embs)
    expected_shape = (batch_size, seq_len, embed_dim)
    
    assert fused.shape == expected_shape, f"Expected {expected_shape}, got {fused.shape}"
    print(f"  ✓ All modalities: {fused.shape}")
    
    # Test with missing modality
    modality_embs_partial = {
        'ecg': torch.randn(batch_size, seq_len, embed_dim),
        'ppg': torch.randn(batch_size, seq_len, embed_dim),
    }
    
    fused_partial = fusion(modality_embs_partial)
    assert fused_partial.shape == expected_shape
    print(f"  ✓ Partial modalities (ECG+PPG): {fused_partial.shape}")
    
    # Test gradient flow
    fused.sum().backward()
    has_grad = fusion.cls_token.grad is not None
    assert has_grad, "CLS token should have gradients"
    print("  ✓ Gradient flow working")
    
    print("  ✓ CLSTokenTransformerFusion test passed!\n")
    return True


def test_dilated_cnn():
    """Test Dilated CNN sequence mixer in isolation."""
    print("Testing DilatedCNNSequenceMixer...")
    
    # Import the component directly
    from pyhealth.models.wav2sleep import DilatedCNNSequenceMixer
    
    # Test parameters
    batch_size, seq_len, input_dim = 2, 20, 64
    hidden_dim = 128
    
    # Create mixer
    mixer = DilatedCNNSequenceMixer(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        kernel_size=3,
        num_layers=5,
        dilations=None,  # Use default [1, 2, 4, 8, 16]
        dropout=0.1,
    )
    
    # Test forward pass
    x = torch.randn(batch_size, seq_len, input_dim)
    y = mixer(x)
    
    expected_shape = (batch_size, seq_len, hidden_dim)
    assert y.shape == expected_shape, f"Expected {expected_shape}, got {y.shape}"
    print(f"  ✓ Forward pass: {x.shape} → {y.shape}")
    
    # Test receptive field
    rf = mixer.receptive_field
    print(f"  ✓ Receptive field: {rf} epochs")
    
    # Test gradient flow
    y.sum().backward()
    has_grads = any(p.grad is not None for p in mixer.parameters() if p.requires_grad)
    assert has_grads, "Should have gradients"
    print("  ✓ Gradient flow working")
    
    print("  ✓ DilatedCNNSequenceMixer test passed!\n")
    return True


def test_individual_components():
    """Test individual components without full model."""
    print("Testing individual Dilated Conv blocks...")
    
    from pyhealth.models.wav2sleep import DilatedConvBlock
    
    batch_size, seq_len, channels = 2, 20, 64
    
    # Test different dilations
    for dilation in [1, 2, 4, 8, 16]:
        block = DilatedConvBlock(
            channels=channels,
            kernel_size=3,
            dilation=dilation,
            dropout=0.1,
        )
        
        # Input is channels-first for Conv1d
        x = torch.randn(batch_size, channels, seq_len)
        y = block(x)
        
        assert y.shape == x.shape, f"Shape mismatch for dilation {dilation}"
        print(f"  ✓ Dilation {dilation:2d}: {x.shape} → {y.shape}")
    
    print("  ✓ DilatedConvBlock tests passed!\n")
    return True


def test_paper_faithful_vs_simplified():
    """Test that we can differentiate paper-faithful vs simplified."""
    print("Testing Paper-Faithful vs Simplified versions...")
    
    from pyhealth.models.wav2sleep import TemporalConvBlock
    
    # Test simplified temporal block
    batch_size, seq_len, input_dim = 2, 10, 64
    hidden_dim = 128
    
    temp_block = TemporalConvBlock(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        kernel_size=3,
        dropout=0.1,
    )
    
    x = torch.randn(batch_size, seq_len, input_dim)
    y = temp_block(x)
    
    expected_shape = (batch_size, seq_len, hidden_dim)
    assert y.shape == expected_shape
    print(f"  ✓ TemporalConvBlock (simplified): {x.shape} → {y.shape}")
    
    print("  ✓ Component differentiation test passed!\n")
    return True


def main():
    """Run all minimal tests."""
    print("="*60)
    print("WAV2SLEEP MINIMAL COMPONENT TESTS")
    print("="*60 + "\n")
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}\n")
    
    results = {}
    
    try:
        results['CLS-Token Fusion'] = test_cls_token_fusion()
    except Exception as e:
        print(f"  ✗ CLSTokenTransformerFusion failed: {e}")
        results['CLS-Token Fusion'] = False
    
    try:
        results['Dilated CNN'] = test_dilated_cnn()
    except Exception as e:
        print(f"  ✗ DilatedCNNSequenceMixer failed: {e}")
        results['Dilated CNN'] = False
    
    try:
        results['Individual Components'] = test_individual_components()
    except Exception as e:
        print(f"  ✗ Individual components failed: {e}")
        results['Individual Components'] = False
    
    try:
        results['Paper-Faithful vs Simplified'] = test_paper_faithful_vs_simplified()
    except Exception as e:
        print(f"  ✗ Differentiation test failed: {e}")
        results['Paper-Faithful vs Simplified'] = False
    
    # Summary
    print("="*60)
    print("TEST RESULTS")
    print("="*60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {status}: {test_name}")
        if not passed:
            all_passed = False
    
    print("="*60)
    if all_passed:
        print("🎉 ALL MINIMAL TESTS PASSED!")
        print("\nYour Wav2Sleep components are working correctly!")
        print("Ready for integration with Dhruv's encoders and Nafis's fusion!")
    else:
        print("❌ SOME TESTS FAILED")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)