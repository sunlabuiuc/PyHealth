"""
LUDB ECG Delineation — Ablation Study
======================================
This script demonstrates an end-to-end pipeline for ECG wave delineation using
the LUDBDataset and ecg_delineation_ludb_fn task, then ablates key design choices
using a lightweight 1-D U-Net model.

Ablations performed
-------------------
1. Input mode       : pulse-aligned windows  vs.  raw 10-second windows
2. Pulse-window size: 125 / 250 / 375 samples  (pulse-aligned mode only)
3. U-Net depth      : 2 / 3 encoder stages

Run
---
    python examples/ludb_ecg_delineation_unet1d.py

Requirements
------------
    pip install wfdb torch pyhealth

Set DATA_ROOT to the ``data/`` folder from the LUDB PhysioNet download.
"""

import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ── PyHealth imports ────────────────────────────────────────────────────────
from pyhealth.datasets.ludb import LUDBDataset
from pyhealth.tasks.ecg_delineation import ecg_delineation_ludb_fn

# ── Configuration ────────────────────────────────────────────────────────────
DATA_ROOT = "/Users/delin/Documents/DL4H/physionet.org/files/ludb/1.0.1/data"
NUM_CLASSES = 4          # 0=background, 1=P, 2=QRS, 3=T
BATCH_SIZE = 16
EPOCHS = 5               # keep short for demonstration
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Lightweight 1-D U-Net ────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Two Conv1d → BN → ReLU layers."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet1D(nn.Module):
    """Encoder–decoder U-Net for 1-D sequence segmentation.

    Args:
        in_channels: number of input channels (1 for single-lead ECG).
        num_classes: number of output segmentation classes.
        base_filters: number of filters in the first encoder block.
        depth: number of encoder / decoder stages.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = NUM_CLASSES,
        base_filters: int = 32,
        depth: int = 3,
    ):
        super().__init__()
        self.depth = depth

        enc_channels = [in_channels] + [base_filters * (2 ** i) for i in range(depth)]
        self.encoders = nn.ModuleList(
            [ConvBlock(enc_channels[i], enc_channels[i + 1]) for i in range(depth)]
        )
        self.pools = nn.ModuleList(
            [nn.MaxPool1d(2) for _ in range(depth)]
        )

        self.bottleneck = ConvBlock(enc_channels[-1], enc_channels[-1] * 2)

        dec_channels = [enc_channels[-1] * 2] + list(reversed(enc_channels[1:]))
        self.upconvs = nn.ModuleList(
            [nn.ConvTranspose1d(dec_channels[i], dec_channels[i + 1], kernel_size=2, stride=2)
             for i in range(depth)]
        )
        self.decoders = nn.ModuleList(
            [ConvBlock(dec_channels[i + 1] * 2, dec_channels[i + 1]) for i in range(depth)]
        )

        self.head = nn.Conv1d(dec_channels[-1], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)
        x = self.bottleneck(x)
        for upconv, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = upconv(x)
            # handle odd-length mismatches
            if x.shape[-1] != skip.shape[-1]:
                x = nn.functional.pad(x, (0, skip.shape[-1] - x.shape[-1]))
            x = dec(torch.cat([skip, x], dim=1))
        return self.head(x)


# ── Torch dataset wrapper ────────────────────────────────────────────────────

class EpochDataset(Dataset):
    """Wraps SampleSignalDataset samples for direct use with DataLoader."""

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        epoch = pickle.load(open(self.samples[idx]["epoch_path"], "rb"))
        signal = torch.tensor(epoch["signal"], dtype=torch.float32)  # (1, T)
        label = torch.tensor(epoch["label"], dtype=torch.long)       # (T,)
        return signal, label


def pad_collate(batch):
    """Pad signals in a batch to the same length."""
    signals, labels = zip(*batch)
    max_len = max(s.shape[-1] for s in signals)
    signals = torch.stack([
        nn.functional.pad(s, (0, max_len - s.shape[-1])) for s in signals
    ])
    labels = torch.stack([
        nn.functional.pad(l, (0, max_len - l.shape[0])) for l in labels
    ])
    return signals, labels


# ── Training / evaluation helpers ───────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for signals, labels in loader:
        signals, labels = signals.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(signals)          # (B, C, T)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * signals.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    for signals, labels in loader:
        signals, labels = signals.to(DEVICE), labels.to(DEVICE)
        preds = model(signals).argmax(dim=1)   # (B, T)
        correct += (preds == labels).sum().item()
        total += labels.numel()
    return correct / total


def run_ablation(name: str, task_fn_kwargs: dict, model_kwargs: dict):
    """Build dataset, train for EPOCHS, and return final val accuracy."""
    print(f"\n{'=' * 60}")
    print(f"Ablation: {name}")
    print(f"  task kwargs : {task_fn_kwargs}")
    print(f"  model kwargs: {model_kwargs}")

    # Build PyHealth SampleSignalDataset
    ds = LUDBDataset(root=DATA_ROOT, dev=False, refresh_cache=True)
    sample_ds = ds.set_task(
        lambda rec: ecg_delineation_ludb_fn(rec, **task_fn_kwargs),
        task_name="ecg_delineation",
    )
    all_samples = sample_ds.samples

    # 8:1:1 patient split (records 1-160 / 161-180 / 181-200)
    train_s = [s for s in all_samples if int(s["patient_id"]) <= 160]
    val_s   = [s for s in all_samples if 160 < int(s["patient_id"]) <= 180]

    train_loader = DataLoader(EpochDataset(train_s), batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=pad_collate)
    val_loader   = DataLoader(EpochDataset(val_s),   batch_size=BATCH_SIZE,
                              shuffle=False, collate_fn=pad_collate)

    model = UNet1D(**model_kwargs).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion)
        acc  = evaluate(model, val_loader)
        print(f"  Epoch {epoch:2d}/{EPOCHS} — loss: {loss:.4f}  val_acc: {acc:.4f}")

    val_acc = evaluate(model, val_loader)
    print(f"  Final val accuracy: {val_acc:.4f}")
    return val_acc


# ── Ablation configurations ──────────────────────────────────────────────────

ABLATIONS = [
    # (name, task_fn_kwargs, model_kwargs)
    (
        "pulse-aligned / window=250 / depth=3",
        {"use_pulse_aligned": True,  "pulse_window": 250},
        {"depth": 3},
    ),
    (
        "pulse-aligned / window=125 / depth=3",
        {"use_pulse_aligned": True,  "pulse_window": 125},
        {"depth": 3},
    ),
    (
        "pulse-aligned / window=375 / depth=3",
        {"use_pulse_aligned": True,  "pulse_window": 375},
        {"depth": 3},
    ),
    (
        "pulse-aligned / window=250 / depth=2",
        {"use_pulse_aligned": True,  "pulse_window": 250},
        {"depth": 2},
    ),
    (
        "raw 10-sec window / depth=3",
        {"use_pulse_aligned": False},
        {"depth": 3},
    ),
]


if __name__ == "__main__":
    results = {}
    for name, task_kw, model_kw in ABLATIONS:
        results[name] = run_ablation(name, task_kw, model_kw)

    print("\n\n" + "=" * 60)
    print("Ablation Summary")
    print("=" * 60)
    for name, acc in results.items():
        print(f"  {name:<50s}  val_acc = {acc:.4f}")
