"""
DataProcess_nifti.py
Loads 3D NIfTI volumes from the OASIS cross-sectional dataset,
maps CDR scores to 4 classes, performs a stratified 80/20 split,
and serialises the result to .pt tensor files.
"""

import os
import random
import shutil
import time

import numpy as np
import pandas as pd
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ── reproducibility ──────────────────────────────────────────────────────────
_START_RUNTIME = time.time()

seed = 24
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)

# ── paths ────────────────────────────────────────────────────────────────────
ARCHIVE_PATH = os.path.join(os.getcwd(), 'archive')
NII_DIR = os.path.join(ARCHIVE_PATH, 'oasis', 'OASIS')
CSV_PATH = os.path.join(ARCHIVE_PATH, 'oasis_cross-sectional.csv')
print('Archive path:', ARCHIVE_PATH)
print('NIfTI dir:   ', NII_DIR)
print('CSV path:    ', CSV_PATH)

# ── load CSV and map CDR → class ─────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)

CDR_TO_CLASS = {
    0.0: 'none',
    0.5: 'verymild',
    1.0: 'mild',
    2.0: 'moderate'
}

CLASS_TO_IDX = {
    'none': 0,
    'verymild': 1,
    'mild': 2,
    'moderate': 3
}

df['class_name'] = df['CDR'].map(CDR_TO_CLASS)

# Build mapping from subject ID to NIfTI filename
nii_files = os.listdir(NII_DIR)
id_to_nii = {}
for f in nii_files:
    subject_id = f.split('_mpr_')[0]
    id_to_nii[subject_id] = f

df['nii_filename'] = df['ID'].map(id_to_nii)

print('Class distribution:')
print(df['class_name'].value_counts())
print(f'\nTotal: {len(df)} subjects')
print(f'Missing NIfTI files: {df["nii_filename"].isna().sum()}')

# ── stratified 80/20 split ───────────────────────────────────────────────────
train_df, val_df = train_test_split(
    df, test_size=0.2, random_state=seed, stratify=df['class_name']
)

print(f'\nTrain: {len(train_df)}, Val: {len(val_df)}')
print('\nTrain class distribution:')
print(train_df['class_name'].value_counts())
print('\nVal class distribution:')
print(val_df['class_name'].value_counts())

# ── create directory structure and copy NIfTI files ──────────────────────────
# Uncomment the block below on first run to organise files into class folders.

# classes = ['none', 'verymild', 'mild', 'moderate']
#
# for split in ['train', 'val']:
#     for cls in classes:
#         os.makedirs(os.path.join(ARCHIVE_PATH, split, cls), exist_ok=True)
#
# def copy_files(dataframe, split_name):
#     for _, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc=f'Copying {split_name}'):
#         src = os.path.join(NII_DIR, row['nii_filename'])
#         dst = os.path.join(ARCHIVE_PATH, split_name, row['class_name'], row['nii_filename'])
#         if not os.path.exists(dst):
#             shutil.copy2(src, dst)
#
# copy_files(train_df, 'train')
# copy_files(val_df, 'val')
#
# print('\n--- Files copied ---')
# for split in ['train', 'val']:
#     print(f'\n{split}:')
#     for cls in classes:
#         d = os.path.join(ARCHIVE_PATH, split, cls)
#         n = len(os.listdir(d))
#         print(f'  {cls}: {n}')

# ── NIfTI dataset class ─────────────────────────────────────────────────────
CLASS_ORDER = ['none', 'verymild', 'mild', 'moderate']


class NiftiDataset(Dataset):
    """Custom Dataset for loading NIfTI volumes from class-organized folders.
    Works like torchvision.datasets.ImageFolder but for .nii files.
    """
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.classes = CLASS_ORDER
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.samples = []   # list of (file_path, label)
        self.targets = []   # list of labels

        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            for fname in sorted(os.listdir(cls_dir)):
                if fname.endswith('.nii') or fname.endswith('.nii.gz'):
                    self.samples.append((os.path.join(cls_dir, fname), self.class_to_idx[cls]))
                    self.targets.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = nib.load(path)
        volume = img.get_fdata().astype(np.float32)
        # Add channel dimension: (D, H, W) -> (1, D, H, W)
        volume = torch.from_numpy(volume).unsqueeze(0)
        return volume, label


# Quick test
ds = NiftiDataset(os.path.join(ARCHIVE_PATH, 'train'))
print(f'\nTrain dataset: {len(ds)} samples')
print(f'Classes: {ds.classes}')
print(f'class_to_idx: {ds.class_to_idx}')
vol, lbl = ds[0]
print(f'Sample shape: {vol.shape}, label: {lbl}')

# ── data loading ─────────────────────────────────────────────────────────────

def load_data(batch_size=2):
    """Load 3D NIfTI data from archive/train and archive/val."""
    train_dataset = NiftiDataset(os.path.join(ARCHIVE_PATH, 'train'))
    val_dataset = NiftiDataset(os.path.join(ARCHIVE_PATH, 'val'))

    print(f'Train: {len(train_dataset)} volumes')
    print(f'Val:   {len(val_dataset)} volumes')
    print(f'Classes: {train_dataset.classes}')

    for split_name, ds in [('Train', train_dataset), ('Val', val_dataset)]:
        counts = {}
        for cls_name, cls_idx in ds.class_to_idx.items():
            counts[cls_name] = ds.targets.count(cls_idx)
        print(f'{split_name} distribution: {counts}')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


train_loader, val_loader = load_data()

# ── save / load .pt helpers ──────────────────────────────────────────────────

def save_processed_dataset(loader, filename):
    """Pack all 3D volumes from a DataLoader into a single .pt file."""
    all_imgs = []
    all_labels = []

    print(f'Packing {len(loader.dataset)} volumes into {filename}...')
    for imgs, labels in tqdm(loader):
        all_imgs.append(imgs)
        all_labels.append(labels)

    data_dict = {
        'imgs': torch.cat(all_imgs),
        'labels': torch.cat(all_labels)
    }
    print(f'Tensor shape: {data_dict["imgs"].shape}')
    torch.save(data_dict, filename)
    print(f'Saved to {filename}')


def load_from_pt(train_path, val_path, batch_size=2):
    """Load pre-saved .pt files and return DataLoaders for 3D volumes."""
    train_data = torch.load(train_path)
    val_data = torch.load(val_path)

    train_ds = TensorDataset(train_data['imgs'], train_data['labels'])
    val_ds = TensorDataset(val_data['imgs'], val_data['labels'])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f'Loaded train: {len(train_ds)} volumes, val: {len(val_ds)} volumes')
    return train_loader, val_loader


# ── serialise (uncomment on first run) ───────────────────────────────────────
# save_processed_dataset(train_loader, os.path.join(ARCHIVE_PATH, 'nifti_train.pt'))
# save_processed_dataset(val_loader,   os.path.join(ARCHIVE_PATH, 'nifti_val.pt'))

# ── verify reload ────────────────────────────────────────────────────────────
train_loader_pt, val_loader_pt = load_from_pt(
    os.path.join(os.getcwd(), 'nifti_train.pt'),
    os.path.join(os.getcwd(), 'nifti_val.pt')
)

imgs, labels = next(iter(train_loader_pt))
print(f'\nBatch shape: {imgs.shape}')   # (batch, 1, 91, 109, 91)
print(f'Labels: {labels}')
print(f'Dtype: {imgs.dtype}')

elapsed = time.time() - _START_RUNTIME
print(f'\nTotal runtime: {elapsed/60:.1f} minutes')
