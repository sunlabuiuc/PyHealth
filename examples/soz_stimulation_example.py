# examples/soz_stimulation_example.py

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

from pyhealth.datasets import SOZStimulationDataset


class SimpleSOZCNN(nn.Module):
    """Simple 1D CNN for SOZ classification from stimulation-locked EEG."""
    def __init__(self, in_channels: int, n_classes: int = 2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(32, n_classes)

    def forward(self, x_stim):
        # x_stim: [B, C, T]
        x = self.conv1(x_stim)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)  # [B, 32, 1]
        x = x.squeeze(-1) # [B, 32]
        return self.fc(x)


def main():
    root = "data/soz_spes_processed"
    train_ds = SOZStimulationDataset(root=root, split="train")
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)

    first_sample, _ = train_ds[0]
    in_channels = first_sample["X_stim"].shape[0]

    model = SimpleSOZCNN(in_channels=in_channels, n_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(2):  # toy run
        model.train()
        total_loss = 0.0
        for features, labels in train_loader:
            x_stim = features["X_stim"]         # [B, C, T]
            labels = torch.as_tensor(labels, dtype=torch.long)

            optimizer.zero_grad()
            logits = model(x_stim)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x_stim.size(0)

        avg_loss = total_loss / len(train_ds)
        print(f"Epoch {epoch+1}: loss = {avg_loss:.4f}")


if __name__ == "__main__":
    main()
