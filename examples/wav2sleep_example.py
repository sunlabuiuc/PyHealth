"""
Example usage of wav2sleep model for sleep stage classification.

This script demonstrates how to use the wav2sleep model with different
modality combinations and synthetic data for testing.

Author: Meredith McClain (mmcclan2)
"""

import torch
from wav2sleep_pyhealth import Wav2Sleep

def example_basic_usage():
    """Basic example with all modalities."""
    print("\n" + "="*50)
    print("Example 1: Training with all modalities")
    print("="*50)
    
    # Define modalities (signal name -> samples per epoch)
    modalities = {
        "ecg": 1024,  # 34 Hz * 30 seconds
        "ppg": 1024,
        "abd": 256,   # 8 Hz * 30 seconds
        "thx": 256
    }
    
    # Create model
    model = Wav2Sleep(
        modalities=modalities,
        num_classes=5,
        feature_dim=128,
        dropout=0.1
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {num_params:,} parameters")
    
    # Generate synthetic data for testing
    # Simulate 10 hours of sleep (1200 epochs of 30 seconds each)
    batch_size = 4
    T = 1200  # number of epochs
    
    inputs = {
        "ecg": torch.randn(batch_size, 1, T * 1024),
        "ppg": torch.randn(batch_size, 1, T * 1024),
        "abd": torch.randn(batch_size, 1, T * 256),
        "thx": torch.randn(batch_size, 1, T * 256)
    }
    
    # Generate random labels (0=Wake, 1=N1, 2=N2, 3=N3, 4=REM)
    labels = torch.randint(0, 5, (batch_size, T))
    
    # Forward pass with all modalities
    output = model(inputs, labels)
    
    print(f"\nLogits shape: {output['logits'].shape}")
    print(f"Loss: {output['loss'].item():.4f}")
    print(f"Predictions shape: {output['predictions'].shape}")
    
    return model


def example_subset_modalities():
    """Example with subset of modalities (ECG only)."""
    print("\n" + "="*50)
    print("Example 2: Inference with ECG only")
    print("="*50)
    
    # Model with potential for multiple modalities
    modalities = {
        "ecg": 1024,
        "ppg": 1024,
        "thx": 256
    }
    
    model = Wav2Sleep(modalities=modalities, num_classes=5)
    
    # Inference with only ECG (e.g., if PPG sensor fails)
    batch_size = 4
    T = 1200
    
    inputs_ecg_only = {
        "ecg": torch.randn(batch_size, 1, T * 1024)
    }
    
    # Get predictions without labels (inference mode)
    probs = model.predict_proba(inputs_ecg_only)
    
    print(f"Probabilities shape: {probs.shape}")
    print(f"Example probabilities for first epoch:")
    print(probs[0, 0])
    print(f"Sum of probabilities: {probs[0, 0].sum().item():.4f} (should be ~1.0)")
    

def example_variable_combinations():
    """Example testing different modality combinations."""
    print("\n" + "="*50)
    print("Example 3: Testing variable modality combinations")
    print("="*50)
    
    modalities = {
        "ecg": 1024,
        "ppg": 1024,
        "abd": 256,
        "thx": 256
    }
    
    model = Wav2Sleep(modalities=modalities, num_classes=5)
    
    batch_size = 2
    T = 100  # Shorter sequence for quick testing
    
    # Test different combinations
    test_cases = [
        {"ecg": torch.randn(batch_size, 1, T * 1024)},
        {"ecg": torch.randn(batch_size, 1, T * 1024), 
         "thx": torch.randn(batch_size, 1, T * 256)},
        {"ppg": torch.randn(batch_size, 1, T * 1024), 
         "abd": torch.randn(batch_size, 1, T * 256)},
        {"ecg": torch.randn(batch_size, 1, T * 1024),
         "ppg": torch.randn(batch_size, 1, T * 1024),
         "abd": torch.randn(batch_size, 1, T * 256),
         "thx": torch.randn(batch_size, 1, T * 256)}
    ]
    
    for i, inputs in enumerate(test_cases, 1):
        probs = model.predict_proba(inputs)
        modality_names = ", ".join(inputs.keys())
        print(f"Test {i} ({modality_names}): Output shape = {probs.shape} ✓")


def main():
    """Run all examples."""
    print("\nWav2Sleep Model Example")
    print("="*50)
    
    # Run examples
    model = example_basic_usage()
    example_subset_modalities()
    example_variable_combinations()
    
    print("\n" + "="*50)
    print("Example completed successfully!")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
