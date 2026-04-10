"""
Generate dummy dataset for testing Retina U-Net training pipeline.

Creates simple synthetic images with segmentation masks suitable for testing.
Adapted from medicaldetectiontoolkit's generate_toys.py.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List
import pickle


def create_dummy_image_and_mask(
    img_size: Tuple[int, int] = (320, 320),
    num_objects: int = 1,
    foreground_margin: int = 50
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a synthetic medical image with segmentation mask.
    
    Generates random background with circular objects of random sizes.
    Can create multiple objects in a single image.
    
    Args:
        img_size: Image size (H, W)
        num_objects: Number of objects to create in the image
        foreground_margin: Minimum distance from object center to image edge
    
    Returns:
        image: (H, W) float array with values in [0, 1]
        mask: (H, W) uint8 array with binary segmentation
    """
    H, W = img_size
    
    # Create random background
    image = np.random.rand(H, W).astype(np.float32)
    mask = np.zeros((H, W), dtype=np.uint8)
    
    # Add random objects
    for _ in range(num_objects):
        # Random center within margins
        center_y = np.random.randint(foreground_margin, H - foreground_margin)
        center_x = np.random.randint(foreground_margin, W - foreground_margin)
        
        # Random radius
        radius = np.random.randint(10, min(30, foreground_margin // 2))
        
        # Create circular object
        for y in range(H):
            for x in range(W):
                dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                if dist <= radius:
                    image[y, x] += 0.3  # Brighten object
                    mask[y, x] = 1
    
    # Clip image to valid range
    image = np.clip(image, 0, 1)
    
    return image, mask


def generate_dummy_dataset(
    output_dir: str,
    num_train_samples: int = 2,
    num_val_samples: int = 1,
    img_size: Tuple[int, int] = (320, 320),
    objects_per_image: int = 1
):
    """
    Generate dummy dataset for testing.
    
    Creates synthetic images with segmentation masks and saves them in
    the format expected by LIDCDataLoader.
    
    Args:
        output_dir: Directory to save data
        num_train_samples: Number of training samples
        num_val_samples: Number of validation samples
        img_size: Image size (H, W)
        objects_per_image: Number of objects per image
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    records = []
    
    # Generate training data
    print(f"Generating {num_train_samples} training samples...")
    for idx in range(num_train_samples):
        image, mask = create_dummy_image_and_mask(
            img_size=img_size,
            num_objects=objects_per_image
        )
        
        # Save as .npy files
        patient_id = f"DUMMY_{idx:04d}"
        slice_idx = 0
        
        img_path = output_path / f"{patient_id}_slice_{slice_idx}_img.npy"
        mask_path = output_path / f"{patient_id}_slice_{slice_idx}_mask.npy"
        
        np.save(img_path, image)
        np.save(mask_path, mask)
        
        has_nodule = 1 if mask.max() > 0 else 0
        records.append({
            'patient_id': patient_id,
            'slice': slice_idx,
            'image_path': str(img_path),
            'mask_path': str(mask_path),
            'has_nodule': has_nodule,
            'split': 'train'
        })
        
        print(f"  Created {patient_id}: {has_nodule} object(s)")
    
    # Generate validation data
    print(f"Generating {num_val_samples} validation samples...")
    for idx in range(num_val_samples):
        image, mask = create_dummy_image_and_mask(
            img_size=img_size,
            num_objects=objects_per_image
        )
        
        patient_id = f"DUMMY_VAL_{idx:04d}"
        slice_idx = 0
        
        img_path = output_path / f"{patient_id}_slice_{slice_idx}_img.npy"
        mask_path = output_path / f"{patient_id}_slice_{slice_idx}_mask.npy"
        
        np.save(img_path, image)
        np.save(mask_path, mask)
        
        has_nodule = 1 if mask.max() > 0 else 0
        records.append({
            'patient_id': patient_id,
            'slice': slice_idx,
            'image_path': str(img_path),
            'mask_path': str(mask_path),
            'has_nodule': has_nodule,
            'split': 'val'
        })
        
        print(f"  Created {patient_id}: {has_nodule} object(s)")
    
    # Save metadata
    df = pd.DataFrame(records)
    metadata_path = output_path / 'metadata.csv'
    df.to_csv(metadata_path, index=False)
    
    print(f"\nDataset created at: {output_dir}")
    print(f"  Total samples: {len(records)}")
    print(f"  Train: {num_train_samples}, Val: {num_val_samples}")
    print(f"  Metadata saved: {metadata_path}")
    
    return output_dir


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate dummy dataset for testing')
    parser.add_argument('--output_dir', type=str, default='./dummy_data',
                        help='Output directory for generated data')
    parser.add_argument('--num_train', type=int, default=2,
                        help='Number of training samples')
    parser.add_argument('--num_val', type=int, default=1,
                        help='Number of validation samples')
    parser.add_argument('--img_size', type=int, nargs=2, default=[320, 320],
                        help='Image size [H W]')
    parser.add_argument('--objects_per_image', type=int, default=1,
                        help='Number of objects per image')
    
    args = parser.parse_args()
    
    generate_dummy_dataset(
        args.output_dir,
        num_train_samples=args.num_train,
        num_val_samples=args.num_val,
        img_size=tuple(args.img_size),
        objects_per_image=args.objects_per_image
    )
