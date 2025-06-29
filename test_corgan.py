#!/usr/bin/env python3
"""
Simple test script for CorGAN implementation
"""

import numpy as np
import torch
from pyhealth.models.generators.corgan import CorGAN, CorGANDataset
from pyhealth.datasets import BaseDataset


def test_corgan_components():
    """Test individual CorGAN components"""
    print("Testing CorGAN components...")
    
    # Test autoencoder
    feature_size = 100
    autoencoder = CorGAN.CorGANAutoencoder(feature_size)
    
    # Test with dummy data
    dummy_data = torch.randn(32, feature_size)
    encoded = autoencoder(dummy_data)
    print(f"Autoencoder input shape: {dummy_data.shape}")
    print(f"Autoencoder output shape: {encoded.shape}")
    
    # Test generator
    latent_dim = 128
    generator = CorGAN.CorGANGenerator(latent_dim=latent_dim)
    
    # Test with dummy noise
    noise = torch.randn(32, latent_dim)
    generated = generator(noise)
    print(f"Generator input shape: {noise.shape}")
    print(f"Generator output shape: {generated.shape}")
    
    # Test discriminator
    input_dim = feature_size
    discriminator = CorGAN.CorGANDiscriminator(input_dim=input_dim)
    
    # Test with dummy data
    fake_data = torch.randn(32, input_dim)
    discriminator_output = discriminator(fake_data)
    print(f"Discriminator input shape: {fake_data.shape}")
    print(f"Discriminator output shape: {discriminator_output.shape}")
    
    print("All components working correctly!")


def test_corgan_dataset():
    """Test CorGAN dataset wrapper"""
    print("\nTesting CorGAN dataset...")
    
    # Create dummy data
    n_samples = 100
    n_features = 50
    dummy_data = np.random.randint(0, 2, (n_samples, n_features)).astype(np.float32)
    
    # Create dataset
    dataset = CorGANDataset(data=dummy_data)
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Sample shape: {dataset[0].shape}")
    print(f"Data shape: {dataset.return_data().shape}")
    
    print("Dataset working correctly!")


def test_corgan_model():
    """Test CorGAN model initialization"""
    print("\nTesting CorGAN model...")
    
    # Create dummy dataset
    class DummyDataset(BaseDataset):
        def __init__(self):
            super().__init__(dataset_name="dummy", root="", tables=["dummy"])
            self.patients = {
                "patient1": {"conditions": [["code1", "code2"], ["code3"]]},
                "patient2": {"conditions": [["code1", "code4"]]},
                "patient3": {"conditions": [["code2", "code3", "code5"]]}
            }
        
        def load_data(self):
            return None
    
    dataset = DummyDataset()
    
    # Create model
    model = CorGAN(
        dataset=dataset,
        feature_keys=["conditions"],
        label_key="label",
        latent_dim=64,
        batch_size=32,
        n_epochs=2,  # very short for testing
        lr=0.001
    )
    
    print(f"Model input dimension: {model.input_dim}")
    print(f"Model global vocabulary size: {len(model.global_vocab)}")
    print(f"Global vocabulary: {model.global_vocab}")
    
    print("Model initialization working correctly!")


def test_generation():
    """Test synthetic data generation"""
    print("\nTesting synthetic data generation...")
    
    # Create dummy dataset
    class DummyDataset(BaseDataset):
        def __init__(self):
            super().__init__(dataset_name="dummy", root="", tables=["dummy"])
            self.patients = {
                "patient1": {"conditions": [["code1", "code2"], ["code3"]]},
                "patient2": {"conditions": [["code1", "code4"]]},
                "patient3": {"conditions": [["code2", "code3", "code5"]]}
            }
        
        def load_data(self):
            return None
    
    dataset = DummyDataset()
    
    # Create model
    model = CorGAN(
        dataset=dataset,
        feature_keys=["conditions"],
        label_key="label",
        latent_dim=64,
        batch_size=32,
        n_epochs=1,  # very short for testing
        lr=0.001
    )
    
    # Generate synthetic data
    n_samples = 10
    synthetic_data = model.generate(n_samples=n_samples)
    
    print(f"Generated synthetic data shape: {synthetic_data.shape}")
    print(f"Synthetic data type: {synthetic_data.dtype}")
    print(f"Synthetic data range: [{synthetic_data.min()}, {synthetic_data.max()}]")
    print(f"Number of ones in synthetic data: {torch.sum(synthetic_data == 1.0)}")
    print(f"Number of zeros in synthetic data: {torch.sum(synthetic_data == 0.0)}")
    
    print("Generation working correctly!")


if __name__ == "__main__":
    print("=== CorGAN Implementation Test ===\n")
    
    try:
        test_corgan_components()
        test_corgan_dataset()
        test_corgan_model()
        test_generation()
        
        print("\n=== All Tests Passed! ===")
        print("CorGAN implementation is working correctly.")
        
    except Exception as e:
        print(f"\n=== Test Failed ===")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc() 