import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
from torch.utils.data import DataLoader, Dataset

from pyhealth.datasets import eICUDataset
from pyhealth.models.tpc import TPC
from pyhealth.tasks.hourly_los import HourlyLOSEICU


class SimpleLoSDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "time_series": s["time_series"].float(),
            "static": s["static"].float(),
            "target": s["target_los_hours"].float().reshape(-1)[0],
        }


def collate_fn(batch):
    max_t = max(item["time_series"].shape[0] for item in batch)
    feat_dim = batch[0]["time_series"].shape[1]

    padded_ts = []
    statics = []
    targets = []

    for item in batch:
        ts = item["time_series"]
        pad_len = max_t - ts.shape[0]
        if pad_len > 0:
            pad = torch.zeros(pad_len, feat_dim)
            ts = torch.cat([ts, pad], dim=0)
        padded_ts.append(ts)
        statics.append(item["static"])
        targets.append(item["target"])

    return {
        "time_series": torch.stack(padded_ts),
        "static": torch.stack(statics),
        "target": torch.stack(targets),
    }


def msle_loss(pred, target, eps=1e-6):
    return torch.mean((torch.log(pred + 1.0 + eps) - torch.log(target + 1.0 + eps)) ** 2)


def main():
    root = "/home/medukonis/Documents/Illinois/Spring_2026/CS598_Deep_Learning_For_Healthcare/Project/Datasets/eicu-collaborative-research-database-2.0"

    base_dataset = eICUDataset(
        root=root,
        tables=["patient", "lab"],
        dev=True,
    )

    task_dataset = base_dataset.set_task(
        HourlyLOSEICU(
            time_series_tables=["lab"],
            time_series_features={
                "lab": ["-basos"],
            },
            static_features=[],
            min_history_hours=5,
            max_hours=48,
        ),
        num_workers=1,
    )

    samples = [task_dataset[i] for i in range(min(len(task_dataset), 128))]

    print("num task samples:", len(samples))

    if len(samples) == 0:
        print("No samples were generated.")
        return

    print("first sample keys:", samples[0].keys())
    print("time_series shape:", len(samples[0]["time_series"]), "x", len(samples[0]["time_series"][0]))
    print("first sample full:", samples[0])

    #raise SystemExit

    train_dataset = SimpleLoSDataset(samples)
    loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

    model = TPC(
        input_dim=len(samples[0]["time_series"][0]),
        static_dim=len(samples[0]["static"]),
        temporal_channels=4,
        pointwise_channels=4,
        num_layers=2,
        kernel_size=3,
        fc_dim=16,
        dropout=0.1,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(5):
        model.train()
        epoch_loss = 0.0

        for batch in loader:
            pred = model(batch["time_series"], batch["static"])
            loss = msle_loss(pred, batch["target"])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= max(len(loader), 1)
        print(f"epoch={epoch} loss={epoch_loss:.4f}")


if __name__ == "__main__":
    main()
