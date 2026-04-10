"""
Quick test of the training pipeline with dummy data.

This script:
1. Generates dummy dataset
2. Creates data loaders
3. Runs 1 epoch of training to verify everything works
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from generate_dummy_data import generate_dummy_dataset
from train import train_model, get_device


def test_training_pipeline():
    """Test the complete training pipeline with minimal data."""
    
    print("=" * 80)
    print("Testing Retina U-Net Training Pipeline")
    print("=" * 80)
    
    # Setup
    device = get_device()
    dummy_data_dir = './dummy_data_test'
    checkpoint_dir = './checkpoints_test'
    log_dir = './logs_test'
    
    # Step 1: Generate dummy dataset
    print("\n[1/4] Generating dummy dataset...")
    generate_dummy_dataset(
        output_dir=dummy_data_dir,
        num_train_samples=2,
        num_val_samples=1,
        img_size=(320, 320),
        objects_per_image=1
    )
    print("✓ Dummy dataset generated")
    
    # Step 2: Test data loading
    print("\n[2/4] Testing data loading...")
    try:
        from data_loader import LIDCDataLoader
        
        dataloaders = LIDCDataLoader.create_dataloaders(
            dummy_data_dir,
            batch_size=2,
            num_workers=0,
            target_size=(320, 320)
        )
        
        train_loader = dataloaders['train']
        val_loader = dataloaders['val']
        
        print(f"  Training batches: {len(train_loader)}")
        print(f"  Validation batches: {len(val_loader)}")
        
        # Check first batch
        batch = next(iter(train_loader))
        print(f"  Batch image shape: {batch['image'].shape}")
        print(f"  Batch mask shape: {batch['mask'].shape}")
        print("✓ Data loading works")
        
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 3: Test model forward pass
    print("\n[3/4] Testing model forward pass...")
    try:
        from model import RetinaUNet
        from train import transform_batch_to_retina_format
        
        model = RetinaUNet(
            in_channels=1,
            num_classes=2,
            dim=2,
            fpn_base_channels=32,  # Smaller for testing
            fpn_out_channels=96,
            rpn_hidden_channels=128
        )
        model.to(device)
        model.eval()
        
        with torch.no_grad():
            batch_data = transform_batch_to_retina_format(batch, device, dim=2)
            images = batch_data['images']
            
            print(f"  Input shape: {images.shape}")
            
            outputs = model(images)
            
            print(f"  Class logits shape: {outputs['class_logits'].shape}")
            print(f"  BBox deltas shape: {outputs['bbox_deltas'].shape}")
            print(f"  Segmentation shape: {outputs['segmentation'].shape}")
            print("✓ Model forward pass works")
            
    except Exception as e:
        print(f"✗ Model forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 4: Test training loop
    print("\n[4/4] Testing training loop (1 epoch)...")
    try:
        # Train for 1 epoch
        model_trained, trainer = train_model(
            data_dir=dummy_data_dir,
            checkpoint_dir=checkpoint_dir,
            log_dir=log_dir,
            num_epochs=1,
            batch_size=2,
            lr=1e-4,
            num_workers=0,
            dim=2,
            target_size=(320, 320)
        )
        
        print("✓ Training loop completed successfully")
        
    except Exception as e:
        print(f"✗ Training loop failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 80)
    print("✓ All tests passed! Training pipeline is working correctly.")
    print("=" * 80)
    
    return True


if __name__ == '__main__':
    success = test_training_pipeline()
    sys.exit(0 if success else 1)
