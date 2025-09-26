# ------------------------------------------------------------------------------
# Author:      Subin Pradeep & Utkarsh Prasad
# NetID:       subinpp2 & uprasad3
# Description: Seizure detection classification task using EEGSeizureDataset.
# ------------------------------------------------------------------------------

from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from pyhealth.tasks.task_template import TaskTemplate
from pyhealth.datasets.eeg_seizure_dataset import EEGSeizureDataset
from pyhealth.models.seizure_crnn import SeizureCRNN


class SeizureDetectionTask(TaskTemplate):
    """
    Task: Binary classification of seizure vs. background EEG windows.

    Splits by subject: last K subjects held out as test set.
    """

    def __init__(
        self,
        preproc_file: str,
        holdout_subjects: int = 4,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 5e-5,
        epochs: int = 30,
        device: Optional[str] = None,
    ):
        super().__init__()

        # Load dataset
        self.dataset = EEGSeizureDataset(preproc_file)
        subs = np.unique(self.dataset.subj)
        test_subs = subs[-holdout_subjects :]

        # Train/test split by subject
        all_idx = np.arange(len(self.dataset))
        train_idx = [i for i in all_idx if self.dataset.subj[i] not in test_subs]
        test_idx = [i for i in all_idx if self.dataset.subj[i] in test_subs]

        self.train_loader = DataLoader(
            Subset(self.dataset, train_idx), batch_size=batch_size, shuffle=True
        )
        self.test_loader = DataLoader(
            Subset(self.dataset, test_idx), batch_size=batch_size
        )

        # Device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Model, loss, optimizer
        self.model = SeizureCRNN().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

        self.epochs = epochs

    def train(self) -> Dict[str, Any]:
        """Train for the configured number of epochs; return training history."""
        self.model.train()
        history = {"loss": [], "acc": []}

        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0.0
            correct = 0

            for X, y, _ in self.train_loader:
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()

                logits = self.model(X)
                loss = self.criterion(logits, y)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item() * X.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()

            avg_loss = epoch_loss / len(self.train_loader.dataset)
            acc = correct / len(self.train_loader.dataset)

            history["loss"].append(avg_loss)
            history["acc"].append(acc)
            print(f"[Train] Epoch {epoch}/{self.epochs} — loss={avg_loss:.4f} acc={acc:.4f}")

        return history

    def evaluate(self) -> Dict[str, Any]:
        """Evaluate on held‑out subjects; return test metrics."""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for X, y, _ in self.test_loader:
                X, y = X.to(self.device), y.to(self.device)
                preds = self.model(X).argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        return {"test_acc": correct / total}
