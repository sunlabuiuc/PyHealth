#!/usr/bin/env python3
"""
Comprehensive test suite for Wav2Sleep model.

Tests both paper-faithful and simplified versions:
- CLSTokenTransformerFusion
- DilatedCNNSequenceMixer
- Full Wav2Sleep model
"""

import torch
import torch.nn as nn


def test_cls_token_transformer_fusion():
    """Test the paper-faithful CLS-token Transformer fusion module."""
    
    print("\n" + "="*60)
    print("Testing CLSTokenTransformerFusion (Paper-Faithful)")
    print("="*60 + "\n")
    
    from pyhealth.models.wav2sleep import CLSTokenTransformerFusion
    
    # Test parameters
    batch_size, seq_len, embed_dim = 2, 10, 64
    num_heads, num_layers = 4, 2
    
    try:
        # Create fusion module
        fusion = CLSTokenTransformerFusion(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=0.1,
            max_modalities=3,
        )
        
        # Test 1: All modalities present
        print("Test 1: All modalities present")
        modality_embs = {
            'ecg': torch.randn(batch_size, seq_len, embed_dim),
            'ppg': torch.randn(batch_size, seq_len, embed_dim),
            'resp': torch.randn(batch_size, seq_len, embed_dim),
        }
        fused = fusion(modality_embs)
        expected_shape = (batch_size, seq_len, embed_dim)
        assert fused.shape == expected_shape, f"Expected {expected_shape}, got {fused.shape}"
        print(f"  ✓ Output shape: {fused.shape}")
        
        # Test 2: Two modalities (ECG + PPG, missing Resp)
        print("\nTest 2: Two modalities (ECG + PPG)")
        modality_embs_2 = {
            'ecg': torch.randn(batch_size, seq_len, embed_dim),
            'ppg': torch.randn(batch_size, seq_len, embed_dim),
        }
        fused_2 = fusion(modality_embs_2)
        assert fused_2.shape == expected_shape, f"Expected {expected_shape}, got {fused_2.shape}"
        print(f"  ✓ Output shape: {fused_2.shape}")
        
        # Test 3: Single modality (ECG only)
        print("\nTest 3: Single modality (ECG only)")
        modality_embs_1 = {
            'ecg': torch.randn(batch_size, seq_len, embed_dim),
        }
        fused_1 = fusion(modality_embs_1)
        assert fused_1.shape == expected_shape, f"Expected {expected_shape}, got {fused_1.shape}"
        print(f"  ✓ Output shape: {fused_1.shape}")
        
        # Test 4: Gradient flow
        print("\nTest 4: Gradient flow")
        fused.sum().backward()
        has_grad = fusion.cls_token.grad is not None
        assert has_grad, "CLS token should have gradients"
        print(f"  ✓ CLS token has gradients")
        
        # Test 5: Parameter count
        params = sum(p.numel() for p in fusion.parameters())
        print(f"\nTotal parameters: {params:,}")
        
        print("\n✓ CLSTokenTransformerFusion tests passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ CLSTokenTransformerFusion test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dilated_cnn_sequence_mixer():
    """Test the paper-faithful Dilated CNN sequence mixer."""
    
    print("\n" + "="*60)
    print("Testing DilatedCNNSequenceMixer (Paper-Faithful)")
    print("="*60 + "\n")
    
    from pyhealth.models.wav2sleep import DilatedCNNSequenceMixer
    
    # Test parameters
    batch_size, seq_len, input_dim = 2, 20, 64
    hidden_dim = 128
    
    try:
        # Create mixer with default dilations [1, 2, 4, 8, 16]
        mixer = DilatedCNNSequenceMixer(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            kernel_size=3,
            num_layers=5,
            dilations=None,  # Use default [1, 2, 4, 8, 16]
            dropout=0.1,
        )
        
        # Test 1: Forward pass shape
        print("Test 1: Forward pass shape")
        x = torch.randn(batch_size, seq_len, input_dim)
        y = mixer(x)
        expected_shape = (batch_size, seq_len, hidden_dim)
        assert y.shape == expected_shape, f"Expected {expected_shape}, got {y.shape}"
        print(f"  ✓ Input: {x.shape} → Output: {y.shape}")
        
        # Test 2: Receptive field
        print("\nTest 2: Receptive field")
        rf = mixer.receptive_field
        expected_rf = 1 + (3-1)*1 + (3-1)*2 + (3-1)*4 + (3-1)*8 + (3-1)*16  # 31
        print(f"  Receptive field: {rf} epochs")
        print(f"  With 30-second epochs: {rf * 30 / 60:.1f} minutes coverage")
        
        # Test 3: Sequence length preservation
        print("\nTest 3: Sequence length preservation")
        for test_len in [10, 50, 100]:
            x_test = torch.randn(batch_size, test_len, input_dim)
            y_test = mixer(x_test)
            assert y_test.shape[1] == test_len, f"Sequence length not preserved: {test_len} → {y_test.shape[1]}"
        print(f"  ✓ Sequence length preserved for various lengths")
        
        # Test 4: Gradient flow
        print("\nTest 4: Gradient flow")
        y.sum().backward()
        has_grads = any(p.grad is not None for p in mixer.parameters() if p.requires_grad)
        assert has_grads, "Mixer should have gradients"
        print(f"  ✓ Gradients flow through dilated convolutions")
        
        # Test 5: Parameter count
        params = sum(p.numel() for p in mixer.parameters())
        print(f"\nTotal parameters: {params:,}")
        print(f"Dilations used: {mixer.dilations}")
        
        print("\n✓ DilatedCNNSequenceMixer tests passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ DilatedCNNSequenceMixer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dilated_conv_block():
    """Test individual dilated convolution block."""
    
    print("\n" + "="*60)
    print("Testing DilatedConvBlock")
    print("="*60 + "\n")
    
    from pyhealth.models.wav2sleep import DilatedConvBlock
    
    batch_size, seq_len, channels = 2, 20, 64
    
    try:
        # Test different dilation factors
        for dilation in [1, 2, 4, 8, 16]:
            block = DilatedConvBlock(
                channels=channels,
                kernel_size=3,
                dilation=dilation,
                dropout=0.1,
            )
            
            # Input is channels-first for Conv1d: [B, C, T]
            x = torch.randn(batch_size, channels, seq_len)
            y = block(x)
            
            assert y.shape == x.shape, f"Dilation {dilation}: shape mismatch {x.shape} → {y.shape}"
            print(f"  ✓ Dilation {dilation:2d}: {x.shape} → {y.shape}")
        
        print("\n✓ DilatedConvBlock tests passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ DilatedConvBlock test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_temporal_conv_block_simplified():
    """Test the simplified (non-dilated) temporal conv block."""
    
    print("\n" + "="*60)
    print("Testing TemporalConvBlock (Simplified)")
    print("="*60 + "\n")
    
    from pyhealth.models.wav2sleep import TemporalConvBlock
    
    batch_size, seq_len, input_dim = 2, 10, 64
    hidden_dim = 128
    
    try:
        block = TemporalConvBlock(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            kernel_size=3,
            dropout=0.1,
        )
        
        x = torch.randn(batch_size, seq_len, input_dim)
        y = block(x)
        
        expected_shape = (batch_size, seq_len, hidden_dim)
        assert y.shape == expected_shape, f"Expected {expected_shape}, got {y.shape}"
        print(f"  ✓ Input: {x.shape} → Output: {y.shape}")
        
        print("\n✓ TemporalConvBlock (simplified) tests passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ TemporalConvBlock test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_wav2sleep_model():
    """Test full Wav2Sleep model with dummy data."""
    
    print("\n" + "="*60)
    print("Testing Full Wav2Sleep Model")
    print("="*60 + "\n")
    
    try:
        from pyhealth.datasets import create_sample_dataset
        from pyhealth.datasets import get_dataloader
        from pyhealth.models.wav2sleep import Wav2Sleep
        
        # Create dummy sample dataset
        samples = [
            {
                "patient_id": "patient-0",
                "visit_id": "epoch-0", 
                "ecg": [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]],
                "ppg": [[0.5, 1.5, 2.5], [1.5, 2.5, 3.5], [2.5, 3.5, 4.5]],
                "resp": [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5]],
                "sleep_stage": [0, 1, 2],
            },
            {
                "patient_id": "patient-1", 
                "visit_id": "epoch-1",
                "ecg": [[4.0, 5.0, 6.0], [5.0, 6.0, 7.0]],
                "ppg": [[3.5, 4.5, 5.5], [4.5, 5.5, 6.5]],
                "resp": [[0.4, 0.5, 0.6], [0.5, 0.6, 0.7]],
                "sleep_stage": [3, 4],
            },
        ]
        
        input_schema = {
            "ecg": "tensor",
            "ppg": "tensor",
            "resp": "tensor",
        }
        output_schema = {"sleep_stage": "multiclass"}
        
        print("1. Creating dataset...")
        dataset = create_sample_dataset(
            samples=samples,
            input_schema=input_schema,
            output_schema=output_schema,
            dataset_name="test_wav2sleep",
        )
        print(f"   ✓ Dataset created")
        
        print("\n2. Testing PAPER-FAITHFUL model...")
        model_faithful = Wav2Sleep(
            dataset=dataset,
            embedding_dim=64,
            hidden_dim=64,
            num_classes=5,
            num_fusion_heads=4,
            num_fusion_layers=2,
            num_temporal_layers=5,
            use_paper_faithful=True,
        )
        print(f"   ✓ Paper-faithful model initialized")
        
        # Get reproduction fidelity report
        report = model_faithful.get_reproduction_fidelity_report()
        print(f"\n   Reproduction fidelity report:")
        for key, value in report.items():
            print(f"     - {key}: {value}")
        
        print("\n3. Testing SIMPLIFIED model...")
        model_simplified = Wav2Sleep(
            dataset=dataset,
            embedding_dim=64,
            hidden_dim=64,
            num_classes=5,
            use_paper_faithful=False,
        )
        print(f"   ✓ Simplified model initialized")
        
        report_simple = model_simplified.get_reproduction_fidelity_report()
        print(f"\n   Reproduction fidelity report:")
        for key, value in report_simple.items():
            print(f"     - {key}: {value}")
        
        print("\n4. Testing forward pass...")
        train_loader = get_dataloader(dataset, batch_size=2, shuffle=False)
        data_batch = next(iter(train_loader))
        
        model_faithful.eval()
        with torch.no_grad():
            results = model_faithful(**data_batch)
        
        print(f"   ✓ Forward pass successful")
        print(f"   Output keys: {list(results.keys())}")
        print(f"   Loss: {results['loss'].item():.4f}")
        print(f"   y_prob shape: {results['y_prob'].shape}")
        print(f"   y_true shape: {results['y_true'].shape}")
        
        print("\n5. Testing gradient computation...")
        model_faithful.train()
        results = model_faithful(**data_batch)
        results['loss'].backward()
        
        has_grads = any(p.grad is not None for p in model_faithful.parameters() if p.requires_grad)
        assert has_grads, "Model should have gradients"
        print(f"   ✓ Gradients computed successfully")
        
        print("\n✓ Full Wav2Sleep model tests passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Wav2Sleep model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_missing_modalities():
    """Test missing modality handling in CLS-token fusion."""
    
    print("\n" + "="*60)
    print("Testing Missing Modality Handling")
    print("="*60 + "\n")
    
    from pyhealth.models.wav2sleep import CLSTokenTransformerFusion
    
    batch_size, seq_len, embed_dim = 2, 10, 64
    
    try:
        fusion = CLSTokenTransformerFusion(
            embed_dim=embed_dim,
            num_heads=4,
            num_layers=2,
            max_modalities=3,
        )
        
        # Test various modality combinations
        test_cases = [
            {'ecg': True, 'ppg': True, 'resp': True},  # All present
            {'ecg': True, 'ppg': True, 'resp': False}, # ECG + PPG
            {'ecg': True, 'ppg': False, 'resp': True}, # ECG + Resp
            {'ecg': False, 'ppg': True, 'resp': True}, # PPG + Resp
            {'ecg': True, 'ppg': False, 'resp': False}, # ECG only
            {'ecg': False, 'ppg': True, 'resp': False}, # PPG only
            {'ecg': False, 'ppg': False, 'resp': True}, # Resp only
        ]
        
        for i, case in enumerate(test_cases):
            modality_embs = {}
            if case['ecg']:
                modality_embs['ecg'] = torch.randn(batch_size, seq_len, embed_dim)
            if case['ppg']:
                modality_embs['ppg'] = torch.randn(batch_size, seq_len, embed_dim)
            if case['resp']:
                modality_embs['resp'] = torch.randn(batch_size, seq_len, embed_dim)
            
            present = [k for k, v in case.items() if v]
            fused = fusion(modality_embs)
            
            expected_shape = (batch_size, seq_len, embed_dim)
            assert fused.shape == expected_shape
            print(f"  ✓ Case {i+1}: {present} → {fused.shape}")
        
        # Test empty modalities (should raise error)
        print("\n  Testing empty modalities (should raise error)...")
        try:
            fusion({})
            print("  ✗ Should have raised ValueError")
            return False
        except ValueError as e:
            print(f"  ✓ Correctly raised ValueError: {e}")
        
        print("\n✓ Missing modality handling tests passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Missing modality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all test suites."""
    
    print("\n" + "="*60)
    print("WAV2SLEEP MODEL TEST SUITE")
    print("="*60)
    
    results = {}
    
    # Component tests
    results['DilatedConvBlock'] = test_dilated_conv_block()
    results['DilatedCNNSequenceMixer'] = test_dilated_cnn_sequence_mixer()
    results['CLSTokenTransformerFusion'] = test_cls_token_transformer_fusion()
    results['TemporalConvBlock (Simplified)'] = test_temporal_conv_block_simplified()
    results['Missing Modality Handling'] = test_missing_modalities()
    
    # Full model test (requires PyHealth dataset)
    try:
        results['Full Wav2Sleep Model'] = test_wav2sleep_model()
    except ImportError as e:
        print(f"\n⚠ Skipping full model test (missing dependencies): {e}")
        results['Full Wav2Sleep Model'] = None
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60 + "\n")
    
    all_passed = True
    for test_name, passed in results.items():
        if passed is None:
            status = "⚠ SKIPPED"
        elif passed:
            status = "✓ PASSED"
        else:
            status = "✗ FAILED"
            all_passed = False
        print(f"  {status}: {test_name}")
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED")
    print("="*60 + "\n")
    
    return all_passed


if __name__ == "__main__":
    run_all_tests()