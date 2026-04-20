"""
ICBHI 2017 Respiratory Sound Classification — end-to-end example.

This script demonstrates:
  1. Loading the ICBHI dataset with ICBHIDataset
  2. Applying ICBHIRespiratoryTask to produce cycle-level samples
  3. Splitting into train / test sets
  4. Building a minimal 1-D CNN baseline and training for a few epochs

Requirements:
  - ICBHI 2017 dataset downloaded and extracted to ICBHI_ROOT
  - pip install pyhealth torch scipy

Dataset:
  https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge

Usage:
  python examples/icbhi_respiratory_classification.py --root /data/ICBHI_final_database
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pyhealth.datasets import ICBHIDataset, split_by_patient
from pyhealth.tasks import ICBHIRespiratoryTask
from pyhealth.tasks.icbhi_respiratory_classification import LABEL_NAMES


# ---------------------------------------------------------------------------
# Minimal 1-D CNN baseline
# ---------------------------------------------------------------------------

class RespiratoryConv1D(nn.Module):
    """Lightweight 1-D CNN for respiratory sound cycle classification."""

    def __init__(self, n_classes: int = 4, target_samples: int = 8000) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=64, stride=8, padding=28),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(16, 32, kernel_size=32, stride=4, padding=12),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
        )
        self.classifier = nn.Linear(32 * 16, n_classes)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        x = self.encoder(signal)          # (B, 32, 16)
        x = x.flatten(start_dim=1)       # (B, 512)
        return self.classifier(x)         # (B, n_classes)


# ---------------------------------------------------------------------------
# Collate: stack variable-length-free tensors from SampleDataset
# ---------------------------------------------------------------------------

def collate_fn(batch):
    signals = torch.stack([item["signal"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    return signals, labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    print(f"Loading ICBHI dataset from: {args.root}")
    dataset = ICBHIDataset(
        root=args.root,
        subset="both",
        dev=args.dev,
        cache_dir=args.cache_dir,
    )
    dataset.stat()

    task = ICBHIRespiratoryTask(
        resample_rate=args.resample_rate,
        target_length=args.target_length,
    )
    samples = dataset.set_task(task)
    print(f"Total cycles (samples): {len(samples)}")

    # Use the official train/test split embedded in sample["split"]
    train_samples = [s for s in samples if s["split"] == "train"]
    test_samples  = [s for s in samples if s["split"] == "test"]
    print(f"  Train cycles: {len(train_samples)}")
    print(f"  Test  cycles: {len(test_samples)}")

    train_loader = DataLoader(
        train_samples, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0,
    )
    test_loader = DataLoader(
        test_samples, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    target_samples = int(args.target_length * args.resample_rate)
    model = RespiratoryConv1D(n_classes=4, target_samples=target_samples)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for signals, labels in train_loader:
            signals, labels = signals.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(signals)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
            total += len(labels)

        train_acc = correct / total if total else 0.0
        print(f"Epoch {epoch}/{args.epochs} — loss: {total_loss/total:.4f}  train acc: {train_acc:.3f}")

    # Evaluate on test set
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for signals, labels in test_loader:
            signals, labels = signals.to(device), labels.to(device)
            correct += (model(signals).argmax(1) == labels).sum().item()
            total += len(labels)

    test_acc = correct / total if total else 0.0
    print(f"\nTest accuracy: {test_acc:.3f}  ({correct}/{total})")
    print("Label names:", LABEL_NAMES)

    samples.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ICBHI respiratory classification example")
    parser.add_argument("--root", required=True, help="Path to ICBHI_final_database directory")
    parser.add_argument("--cache_dir", default=".cache/icbhi", help="PyHealth cache directory")
    parser.add_argument("--resample_rate", type=int, default=4000)
    parser.add_argument("--target_length", type=float, default=5.0,
                        help="Fixed cycle duration in seconds (pad/trim)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--dev", action="store_true",
                        help="Use dev mode (1000-patient cap) for quick iteration")
    args = parser.parse_args()
    main(args)
