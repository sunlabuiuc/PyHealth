'''
train_pyhealth.py

Training script for Alzheimer's disease classification using a 3D convolutional neural network (AD3DCNN)
on preprocessed ADNI MRI scans. This script leverages PyHealth's Trainer framework for streamlined model
training, evaluation, and logging.
'''
import os
import yaml
import time
import random

import torch
import torch.nn.functional as F
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from pyhealth.trainer import Trainer as _Trainer

from src.dataset import NiftiDataset
from src.model import AD3DCNN
from src.transforms import Normalize, ToTensor, Compose

# Fixed target shape for padding/cropping
TARGET_SHAPE = (160, 160, 160)


def load_config(path=None):
    '''
    Load YAML configuration from the configs directory by default.
    '''
    if path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(script_dir, '..', 'configs', 'config.yaml')
    with open(path) as f:
        return yaml.safe_load(f)


def pad_or_crop_to(x: torch.Tensor, target_shape: tuple):
    '''
    Pad or crop a 4D tensor [C,D,H,W] to `target_shape` (D,H,W).
    '''
    C, D, H, W = x.shape
    tD, tH, tW = target_shape
    # Compute padding widths
    pad_d = max(0, tD - D)
    pad_h = max(0, tH - H)
    pad_w = max(0, tW - W)
    # F.pad expects (W_left, W_right, H_left, H_right, D_left, D_right)
    x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))
    # Crop extras if dimension is too big
    return x[:, :tD, :tH, :tW]


def pad_collate(batch):
    '''
    Collate function to pad/crop all images in `batch` to the maximum shape.
    '''
    images = [b['image'] for b in batch]
    labels = torch.stack([b['label'] for b in batch], dim=0).long()
    ages = None
    if 'age' in batch[0]:
        ages = torch.tensor([b['age'] for b in batch], dtype=torch.float32)
    # Determine max D,H,W across batch
    Ds, Hs, Ws = zip(*(img.shape[-3:] for img in images))
    maxD, maxH, maxW = max(Ds), max(Hs), max(Ws)
    # Pad/crop each image
    padded = [pad_or_crop_to(img, (maxD, maxH, maxW)) for img in images]
    images = torch.stack(padded, dim=0).float()
    batch_dict = {'image': images, 'label': labels}
    if ages is not None:
        batch_dict['age'] = ages
    return batch_dict


class WrappedModel(nn.Module):
    '''
    Wraps AD3DCNN + loss_fn for PyHealth Trainer compatibility.
    '''
    def __init__(self, base_model: nn.Module, loss_fn: nn.Module):
        super().__init__()
        self.base = base_model
        self.loss_fn = loss_fn
        self.mode = None  # set by Trainer.evaluate

    def forward(self, image, label=None, age=None, **kwargs):
        device = next(self.base.parameters()).device
        image = image.to(device)
        # Run base model with or without age
        logits = self.base(image, age) if age is not None else self.base(image)
        out = {'logits': logits}
        if label is not None:
            out['loss'] = self.loss_fn(logits, label.to(device))
        return out


class Trainer(_Trainer):
    '''
    Extends PyHealth Trainer to add batch-level TensorBoard logging.
    '''
    def __init__(self, *args, writer: SummaryWriter = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.writer = writer

    def _train_epoch(self, train_loader, epoch: int):
        # Run base train logic
        stats = super()._train_epoch(train_loader, epoch)
        # Per-batch logging
        for batch_idx, batch in enumerate(train_loader):
            inputs, targets = self._unpack_batch(batch)
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs['logits'], targets)
            gs = epoch * len(train_loader) + batch_idx + 1
            self.writer.add_scalar('Loss/train_batch', loss.item(), gs)
            acc = (outputs['logits'].argmax(1)==targets).float().mean().item()
            self.writer.add_scalar('Acc/train_batch', acc, gs)
        # Epoch-level logging
        self.writer.add_scalar('Loss/train', stats['train_loss'], epoch+1)
        self.writer.add_scalar('Loss/val', stats['val_loss'], epoch+1)
        self.writer.add_scalar('Acc/train', stats['train_accuracy'], epoch+1)
        self.writer.add_scalar('Acc/val', stats['val_accuracy'], epoch+1)
        # Print summary
        print(f"Epoch {epoch+1:02d}: Train {stats['train_loss']:.3f}/{stats['train_accuracy']:.3f} | Val {stats['val_loss']:.3f}/{stats['val_accuracy']:.3f}")
        return stats


def make_nifti_dataset(split: str, labels_path, preproc_dir,
                       train_idx, val_idx, test_idx,
                       train_transforms, val_transforms):
    '''
    Prepare Subset for given split with appropriate transforms.
    '''
    idxs = {'train': train_idx, 'val': val_idx, 'test': test_idx}[split]
    tf = train_transforms if split=='train' else val_transforms
    full = NiftiDataset(labels_path, preproc_dir, transform=tf)
    return Subset(full, idxs)


def main():
    '''
    Orchestrates data loading, model setup, and calls Trainer.train().
    '''
    cfg = load_config()
    # Resolve paths
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg['data']['preproc_dir'] = os.path.join(root, cfg['data']['preproc_dir'])
    cfg['data']['labels'] = os.path.join(root, cfg['data']['labels'])
    dev = torch.device(cfg['training']['device'])
    # TensorBoard writer
    writer = SummaryWriter(log_dir=cfg['paths']['log_dir'])

    # Define transforms
    val_tf = Compose([Normalize(), ToTensor()])
    train_tf = Compose([Normalize(), ToTensor()])

    # Load full dataset and split indices
    full_raw = NiftiDataset(cfg['data']['labels'], cfg['data']['preproc_dir'], transform=val_tf)
    total = len(full_raw)
    n_train = int(total * cfg['data']['train_split'])
    n_val = int(total * cfg['data']['val_split'])
    indices = list(range(total))
    if 'seed' in cfg['data']: random.seed(cfg['data']['seed'])
    random.shuffle(indices)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train+n_val]
    test_idx = indices[n_train+n_val:]

    # Sampler for class balancing
    labels = full_raw.df['diagnosis'].map({'CN':0,'MCI':1,'AD':2}).values
    class_w = 1.0 / np.bincount(labels)
    weights = class_w[labels][train_idx]
    sampler = WeightedRandomSampler(weights, num_samples=len(train_idx), replacement=True)

    # DataLoaders
    train_ds = make_nifti_dataset('train', cfg['data']['labels'], cfg['data']['preproc_dir'], train_idx, val_idx, test_idx, train_tf, val_tf)
    val_ds   = make_nifti_dataset('val',   cfg['data']['labels'], cfg['data']['preproc_dir'], train_idx, val_idx, test_idx, train_tf, val_tf)
    test_ds  = make_nifti_dataset('test',  cfg['data']['labels'], cfg['data']['preproc_dir'], train_idx, val_idx, test_idx, train_tf, val_tf)

    train_loader = DataLoader(train_ds, batch_size=cfg['training']['batch_size'], sampler=sampler,
                              num_workers=cfg['training']['num_workers'], collate_fn=pad_collate)
    val_loader   = DataLoader(val_ds,   batch_size=cfg['training']['batch_size'], shuffle=False,
                              num_workers=cfg['training']['num_workers'], collate_fn=pad_collate)
    test_loader  = DataLoader(test_ds,  batch_size=cfg['training']['batch_size'], shuffle=False,
                              num_workers=cfg['training']['num_workers'], collate_fn=pad_collate)

    # Model & loss
    base = AD3DCNN(in_channels=1, num_classes=cfg['model']['num_classes'], include_age=cfg['data']['include_age']).to(dev)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_w, device=dev, dtype=torch.float32))
    wrapped = WrappedModel(base, loss_fn)

    # Optimizer & scheduler
    lr = float(cfg['training']['lr'])
    opt = optim.Adam(wrapped.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3)

    # PyHealth Trainer
    os.makedirs(cfg['paths']['ckpt_dir'], exist_ok=True)
    trainer = Trainer(wrapped, metrics=['accuracy','balanced_accuracy'], device=dev,
                      output_path=cfg['paths']['ckpt_dir'], exp_name='cnn_ad', writer=writer)
    trainer.optimizer = opt
    trainer.scheduler = scheduler

    # Train & evaluate
    metrics = trainer.train(train_loader, val_loader, test_loader, cfg['training']['epochs'])
    print('Final test metrics:', metrics)
    writer.close()


if __name__ == '__main__':
    main()