import os
import sys
import argparse
import random

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from pyhealth.datasets import eICUDataset
from pyhealth.models.tpc import TPC
from pyhealth.tasks.hourly_los import HourlyLOSEICU


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_categorical_vocab(samples, categorical_feature_names):
    """
    Build vocab from training samples only.

    Returns:
        {
            "gender": {"Female": 0, "Male": 1, "__MISSING__": 2},
            "ethnicity": {...},
        }
    """
    vocab = {}

    for feature_name in categorical_feature_names:
        values = set()

        for s in samples:
            raw = s.get("categorical_static_raw", {}).get(feature_name)
            if raw is None:
                raw = "__MISSING__"
            values.add(str(raw))

        sorted_values = sorted(values)
        vocab[feature_name] = {val: i for i, val in enumerate(sorted_values)}

    return vocab


def encode_categorical_one_hot(raw_dict, categorical_feature_names, vocab):
    encoded = []

    for feature_name in categorical_feature_names:
        feature_vocab = vocab[feature_name]
        raw = raw_dict.get(feature_name)
        if raw is None:
            raw = "__MISSING__"
        raw = str(raw)

        if raw not in feature_vocab:
            raw = "__MISSING__"

        one_hot = [0.0] * len(feature_vocab)
        one_hot[feature_vocab[raw]] = 1.0
        encoded.extend(one_hot)

    return encoded


class SimpleLoSDataset(Dataset):
    def __init__(
        self,
        samples,
        channel_mode: str = "full",
        include_categorical_statics: bool = False,
        categorical_feature_names=None,
        categorical_vocab=None,
    ):
        """
        channel_mode:
            - full:     keep [value, mask, decay]
            - no_decay: keep [value, mask]
            - no_mask:  keep [value, decay]
        """
        self.samples = samples
        self.channel_mode = channel_mode
        self.include_categorical_statics = include_categorical_statics
        self.categorical_feature_names = categorical_feature_names or []
        self.categorical_vocab = categorical_vocab or {}

    def __len__(self):
        return len(self.samples)

    def _select_channels(self, ts: torch.Tensor) -> torch.Tensor:
        feat_dim = ts.shape[1]
        if feat_dim % 3 != 0:
            raise ValueError(
                f"Expected feature dimension divisible by 3 for [value, mask, decay], got {feat_dim}"
            )

        if self.channel_mode == "full":
            return ts

        num_raw_features = feat_dim // 3
        kept_cols = []

        for i in range(num_raw_features):
            base = i * 3
            if self.channel_mode == "no_decay":
                kept_cols.extend([base, base + 1])      # value, mask
            elif self.channel_mode == "no_mask":
                kept_cols.extend([base, base + 2])      # value, decay
            else:
                raise ValueError(f"Unknown channel_mode: {self.channel_mode}")

        return ts[:, kept_cols]

    def __getitem__(self, idx):
        s = self.samples[idx]

        ts = s["time_series"]
        if not isinstance(ts, torch.Tensor):
            ts = torch.tensor(ts, dtype=torch.float32)
        else:
            ts = ts.float()

        static = s["static"]
        if not isinstance(static, torch.Tensor):
            static = torch.tensor(static, dtype=torch.float32)
        else:
            static = static.float()

        if self.include_categorical_statics:
            raw_cats = s.get("categorical_static_raw", {})
            cat_vec = encode_categorical_one_hot(
                raw_dict=raw_cats,
                categorical_feature_names=self.categorical_feature_names,
                vocab=self.categorical_vocab,
            )
            cat_tensor = torch.tensor(cat_vec, dtype=torch.float32)
            static = torch.cat([static, cat_tensor], dim=0)

        target = s["target_los_hours"]
        if not isinstance(target, torch.Tensor):
            target = torch.tensor(target, dtype=torch.float32)
        else:
            target = target.float().reshape(-1)[0]

        ts = self._select_channels(ts)

        return {
            "time_series": ts,
            "static": static,
            "target": target,
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
            pad = torch.zeros(pad_len, feat_dim, dtype=ts.dtype)
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
    return torch.mean(
        (torch.log(pred + 1.0 + eps) - torch.log(target + 1.0 + eps)) ** 2
    )


def mse_loss(pred, target):
    return torch.mean((pred - target) ** 2)


def evaluate(model, loader, loss_fn):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            pred = model(batch["time_series"], batch["static"])
            loss = loss_fn(pred, batch["target"])
            total_loss += loss.item()

    return total_loss / max(len(loader), 1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run eICU hourly LoS prediction with TPC and ablations."
    )
    parser.add_argument(
        "--root",
        type=str,
        default="/home/medukonis/Documents/Illinois/Spring_2026/CS598_Deep_Learning_For_Healthcare/Project/Datasets/eicu-collaborative-research-database-2.0",
        help="Path to eICU dataset root",
    )
    parser.add_argument("--dev", action="store_true", help="Use dev mode dataset")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--temporal_channels", type=int, default=4)
    parser.add_argument("--pointwise_channels", type=int, default=4)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--fc_dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--loss",
        type=str,
        choices=["msle", "mse"],
        default="msle",
    )
    parser.add_argument(
        "--channel_mode",
        type=str,
        choices=["full", "no_decay", "no_mask"],
        default="no_decay",
        help="full=[value,mask,decay], no_decay=[value,mask], no_mask=[value,decay]",
    )
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument(
        "--include_categorical_statics",
        action="store_true",
        help="Append one-hot categorical statics to numeric static vector",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    categorical_feature_names = ["gender", "ethnicity"]

    print("=" * 80)
    print("TPC eICU Hourly LoS Run Configuration")
    print("=" * 80)
    print(f"root: {args.root}")
    print(f"dev: {args.dev}")
    print(f"epochs: {args.epochs}")
    print(f"batch_size: {args.batch_size}")
    print(f"max_samples: {args.max_samples}")
    print(f"num_layers: {args.num_layers}")
    print(f"temporal_channels: {args.temporal_channels}")
    print(f"pointwise_channels: {args.pointwise_channels}")
    print(f"kernel_size: {args.kernel_size}")
    print(f"fc_dim: {args.fc_dim}")
    print(f"dropout: {args.dropout}")
    print(f"lr: {args.lr}")
    print(f"loss: {args.loss}")
    print(f"channel_mode: {args.channel_mode}")
    print(f"train_ratio: {args.train_ratio}")
    print(f"include_categorical_statics: {args.include_categorical_statics}")
    print(f"categorical_feature_names: {categorical_feature_names}")
    print(f"seed: {args.seed}")
    print("=" * 80)

    base_dataset = eICUDataset(
        root=args.root,
        tables=["patient", "lab"],
        dev=args.dev,
    )

    task_dataset = base_dataset.set_task(
        HourlyLOSEICU(
            time_series_tables=["lab"],
            time_series_features={
                "lab": [
                    "potassium",
                    "sodium",
                    "creatinine",
                    "glucose",
                    "BUN",
                ],
            },
            numeric_static_features=["admissionheight", "admissionweight"],
            categorical_static_features=[],
            min_history_hours=5,
            max_hours=48,
        ),
        num_workers=1,
    )

    samples = [task_dataset[i] for i in range(min(len(task_dataset), args.max_samples))]

    print("num task samples:", len(samples))
    if len(samples) == 0:
        print("No samples were generated.")
        return

    print("first sample keys:", samples[0].keys())
    print(
        "raw time_series shape:",
        len(samples[0]["time_series"]),
        "x",
        len(samples[0]["time_series"][0]),
    )
    print("raw static dim:", len(samples[0]["static"]))

    train_size = int(len(samples) * args.train_ratio)
    val_size = len(samples) - train_size

    if train_size == 0 or val_size == 0:
        raise ValueError(
            f"Invalid split with train_size={train_size}, val_size={val_size}. "
            f"Increase max_samples or adjust train_ratio."
        )

    rng = random.Random(args.seed)
    shuffled_indices = list(range(len(samples)))
    rng.shuffle(shuffled_indices)

    train_indices = shuffled_indices[:train_size]
    val_indices = shuffled_indices[train_size:]

    train_samples = [samples[i] for i in train_indices]
    val_samples = [samples[i] for i in val_indices]

    print(f"train samples: {len(train_samples)}")
    print(f"val samples: {len(val_samples)}")

    categorical_vocab = {}
    if args.include_categorical_statics:
        categorical_vocab = build_categorical_vocab(
            samples=train_samples,
            categorical_feature_names=categorical_feature_names,
        )
        print("categorical vocab sizes:")
        for feat, vocab in categorical_vocab.items():
            print(f"  {feat}: {len(vocab)}")

    train_dataset = SimpleLoSDataset(
        train_samples,
        channel_mode=args.channel_mode,
        include_categorical_statics=args.include_categorical_statics,
        categorical_feature_names=categorical_feature_names,
        categorical_vocab=categorical_vocab,
    )

    val_dataset = SimpleLoSDataset(
        val_samples,
        channel_mode=args.channel_mode,
        include_categorical_statics=args.include_categorical_statics,
        categorical_feature_names=categorical_feature_names,
        categorical_vocab=categorical_vocab,
    )

    first_item = train_dataset[0]
    input_dim = first_item["time_series"].shape[1]
    static_dim = first_item["static"].shape[0]

    print("model input_dim:", input_dim)
    print("model static_dim:", static_dim)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    model = TPC(
        input_dim=input_dim,
        static_dim=static_dim,
        temporal_channels=args.temporal_channels,
        pointwise_channels=args.pointwise_channels,
        num_layers=args.num_layers,
        kernel_size=args.kernel_size,
        fc_dim=args.fc_dim,
        dropout=args.dropout,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.loss == "msle":
        loss_fn = msle_loss
    else:
        loss_fn = mse_loss

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        epoch_train_loss = 0.0

        for batch in train_loader:
            pred = model(batch["time_series"], batch["static"])
            loss = loss_fn(pred, batch["target"])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

        epoch_train_loss /= max(len(train_loader), 1)
        epoch_val_loss = evaluate(model, val_loader, loss_fn)

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        best_val_loss = min(best_val_loss, epoch_val_loss)

        print(
            f"epoch={epoch} train_loss={epoch_train_loss:.4f} val_loss={epoch_val_loss:.4f}"
        )

    print("=" * 80)
    print("Run complete")
    print(f"final_train_loss: {train_losses[-1]:.4f}")
    print(f"final_val_loss: {val_losses[-1]:.4f}")
    print(f"best_val_loss: {best_val_loss:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
