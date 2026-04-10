"""
Data loading and preprocessing for LIDC-IDRI dataset using PyHealth-compatible format.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import pickle
from typing import Dict, List, Tuple, Optional
import pandas as pd
from pathlib import Path


class LIDCDataset(Dataset):
    """
    LIDC-IDRI dataset loader compatible with PyHealth format.
    
    Expects preprocessed data structure:
    data_dir/
    ├── patient_id_slice_0_img.npy
    ├── patient_id_slice_0_mask.npy
    ├── patient_id_slice_1_img.npy
    ├── patient_id_slice_1_mask.npy
    ...
    └── metadata.csv
    """
    
    def __init__(
        self,
        data_dir: str,
        metadata_file: str = 'metadata.csv',
        split: str = 'train',
        normalize: bool = True,
        target_size: Optional[Tuple[int, int]] = None,
        augmentation: bool = False
    ):
        """
        Initialize LIDC dataset.
        
        Args:
            data_dir: Path to preprocessed data directory
            metadata_file: Name of metadata CSV file
            split: 'train', 'val', or 'test'
            normalize: Whether to normalize images
            target_size: Target image size (H, W). If None, use original size
            augmentation: Whether to apply data augmentation
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.normalize = normalize
        self.target_size = target_size
        self.augmentation = augmentation
        
        # Load metadata
        metadata_path = self.data_dir / metadata_file
        if metadata_path.exists():
            self.metadata = pd.read_csv(metadata_path)
        else:
            # Generate metadata from files
            self.metadata = self._generate_metadata()
        
        # Filter by split if available
        if 'split' in self.metadata.columns:
            self.metadata = self.metadata[self.metadata['split'] == split].reset_index(drop=True)
        
        self.samples = self.metadata.to_dict('records')
    
    def _generate_metadata(self) -> pd.DataFrame:
        """Generate metadata from available .npy files."""
        records = []
        for npy_file in self.data_dir.glob('*_img.npy'):
            # Extract patient_id and slice from filename
            # Format: patient_id_slice_N_img.npy
            parts = npy_file.stem.split('_')
            slice_idx = parts[-2]
            patient_id = '_'.join(parts[:-3])
            
            mask_file = npy_file.parent / npy_file.name.replace('_img.npy', '_mask.npy')
            if mask_file.exists():
                records.append({
                    'patient_id': patient_id,
                    'slice': int(slice_idx),
                    'image_path': str(npy_file),
                    'mask_path': str(mask_file),
                    'has_nodule': 1 if np.load(mask_file).max() > 0 else 0
                })
        
        return pd.DataFrame(records)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample.
        
        Returns:
            Dictionary with keys:
                - 'image': (1, H, W) tensor
                - 'mask': (1, H, W) tensor
                - 'patient_id': str
                - 'slice_idx': int
        """
        sample = self.samples[idx]
        
        # Load image and mask
        image = np.load(sample['image_path']).astype(np.float32)
        mask = np.load(sample['mask_path']).astype(np.float32)
        
        # Ensure 2D (sometimes saved with extra dimensions)
        if image.ndim == 3:
            image = image[0] if image.shape[0] == 1 else image[:, :, 0]
        if mask.ndim == 3:
            mask = mask[0] if mask.shape[0] == 1 else mask[:, :, 0]
        
        # Normalize image
        if self.normalize:
            image = self._normalize(image)
        
        # Resize if needed
        if self.target_size is not None:
            image = self._resize(image, self.target_size, order=1)
            mask = self._resize(mask, self.target_size, order=0)
        
        # Data augmentation (if training)
        if self.augmentation and self.split == 'train':
            image, mask = self._augment(image, mask)
        
        # Make arrays C-contiguous to avoid negative strides
        image = np.ascontiguousarray(image)
        mask = np.ascontiguousarray(mask)
        
        # Convert to tensors
        image = torch.from_numpy(image).unsqueeze(0)  # (1, H, W)
        mask = torch.from_numpy(mask).unsqueeze(0)    # (1, H, W)
        
        return {
            'image': image,
            'mask': mask,
            'patient_id': sample['patient_id'],
            'slice_idx': sample['slice']
        }
    
    @staticmethod
    def _normalize(image: np.ndarray) -> np.ndarray:
        """Normalize image to [-1, 1] range."""
        # Clip to reasonable HU range for lung CT
        image = np.clip(image, -1200, 600)
        
        # Standardize
        mean = image.mean()
        std = image.std()
        if std > 0:
            image = (image - mean) / std
        
        return image
    
    @staticmethod
    def _resize(image: np.ndarray, target_size: Tuple[int, int], order: int = 1) -> np.ndarray:
        """Resize image to target size using specified interpolation order."""
        from scipy.ndimage import zoom
        h, w = image.shape
        th, tw = target_size
        zoom_h = th / h
        zoom_w = tw / w
        return zoom(image, (zoom_h, zoom_w), order=order)
    
    @staticmethod
    def _augment(image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply data augmentation (random flip, rotation, etc.)."""
        # Random horizontal flip
        if np.random.rand() > 0.5:
            image = np.fliplr(image)
            mask = np.fliplr(mask)
        
        # Random vertical flip
        if np.random.rand() > 0.5:
            image = np.flipud(image)
            mask = np.flipud(mask)
        
        # Random rotation
        if np.random.rand() > 0.7:
            k = np.random.randint(1, 4)
            image = np.rot90(image, k)
            mask = np.rot90(mask, k)
        
        return image, mask


class LIDCDataLoader:
    """Factory for creating train/val/test dataloaders."""
    
    @staticmethod
    def create_dataloaders(
        data_dir: str,
        batch_size: int = 4,
        num_workers: int = 4,
        target_size: Optional[Tuple[int, int]] = None,
        train_split: float = 0.7,
        val_split: float = 0.15,
        seed: int = 42
    ) -> Dict[str, DataLoader]:
        """
        Create train, val, and test dataloaders.
        
        Args:
            data_dir: Path to data directory
            batch_size: Batch size
            num_workers: Number of workers for data loading
            target_size: Target image size
            train_split: Proportion of data for training (0-1)
            val_split: Proportion of data for validation (0-1)
            seed: Random seed
        
        Returns:
            Dictionary with 'train', 'val', 'test' dataloaders
        """
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Create datasets for each split
        train_dataset = LIDCDataset(
            data_dir, split='train', normalize=True,
            target_size=target_size, augmentation=True
        )
        val_dataset = LIDCDataset(
            data_dir, split='val', normalize=True,
            target_size=target_size, augmentation=False
        )
        test_dataset = LIDCDataset(
            data_dir, split='test', normalize=True,
            target_size=target_size, augmentation=False
        )
        
        # If splits not in metadata, split randomly
        if len(train_dataset) == 0:
            all_dataset = LIDCDataset(data_dir, normalize=True, target_size=target_size)
            n_train = int(len(all_dataset) * train_split)
            n_val = int(len(all_dataset) * val_split)
            
            indices = np.random.permutation(len(all_dataset))
            train_indices = indices[:n_train]
            val_indices = indices[n_train:n_train+n_val]
            test_indices = indices[n_train+n_val:]
            
            train_dataset = torch.utils.data.Subset(all_dataset, train_indices)
            val_dataset = torch.utils.data.Subset(all_dataset, val_indices)
            test_dataset = torch.utils.data.Subset(all_dataset, test_indices)
        
        # Create dataloaders
        dataloaders = {
            'train': DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True,
                num_workers=num_workers, pin_memory=True
            ),
            'val': DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=True
            ),
            'test': DataLoader(
                test_dataset, batch_size=batch_size, shuffle=False,
                num_workers=0, pin_memory=True
            )
        }
        
        return dataloaders


class PyHealthLIDCWrapper:
    """Wrapper to integrate with PyHealth framework."""
    
    def __init__(self, dataset_dir: str):
        """
        Initialize PyHealth wrapper.
        
        Args:
            dataset_dir: Path to LIDC dataset directory
        """
        self.dataset_dir = dataset_dir
        self.dataset = LIDCDataset(dataset_dir)
    
    def get_pyhealth_dataset(self):
        """Convert to PyHealth Dataset format."""
        from pyhealth.data import Patient, Visit, Event
        
        patients = {}
        for sample in self.dataset.samples:
            patient_id = sample['patient_id']
            
            if patient_id not in patients:
                patients[patient_id] = Patient(patient_id=patient_id)
            
            # Create visit for each slice
            visit_id = f"slice_{sample['slice']}"
            visit = Visit(visit_id=visit_id, patient_id=patient_id)
            
            # Add events (image and mask)
            image_event = Event(
                code=sample['image_path'],
                table='CT_SCANS',
                vocabulary='LOCAL',
                visit_id=visit_id,
                patient_id=patient_id
            )
            mask_event = Event(
                code=sample['mask_path'],
                table='ANNOTATIONS',
                vocabulary='LOCAL',
                visit_id=visit_id,
                patient_id=patient_id,
                label=sample['has_nodule']
            )
            
            visit.add_event(image_event)
            visit.add_event(mask_event)
            patients[patient_id].add_visit(visit)
        
        return patients
