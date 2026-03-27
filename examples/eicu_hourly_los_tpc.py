import os
import sys
import argparse
import random
from typing import Dict, List, Optional

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from pyhealth.datasets import eICUDataset
from pyhealth.models.tpc import TPC
from pyhealth.tasks.hourly_los import HourlyLOSEICU


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_categorical_vocab(
    samples: List[dict],
    categorical_feature_names: List[str],
) -> Dict[str, Dict[str, int]]:
    vocab: Dict[str, Dict[str, int]] = {}

    for feature_name in categorical_feature_names:
        values = set()

        for sample in samples:
            raw = sample.get("categorical_static_raw", {}).get(feature_name)
            if raw is None:
                raw = "__MISSING__"
            values.add(str(raw))

        sorted_values = sorted(values)
        if "__MISSING__" not in sorted_values:
            sorted_values.append("__MISSING__")

        vocab[feature_name] = {val: i for i, val in enumerate(sorted_values)}

    return vocab


def encode_categorical_one_hot(
    raw_dict: Dict[str, object],
    categorical_feature_names: List[str],
    vocab: Dict[str, Dict[str, int]],
) -> List[float]:
    encoded: List[float] = []

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


def run_model_smoke_test() -> None:
    print("=" * 80)
    print("Running TPC smoke test")
    print("=" * 80)

    batch_size, seq_len, num_features, static_dim = 2, 6, 5, 3

    x_values = torch.randn(batch_size, seq_len, num_features)
    x_decay = torch.rand(batch_size, seq_len, num_features)
    x_static = torch.randn(batch_size, static_dim)

    model = TPC(
        input_dim=num_features,
        static_dim=static_dim,
        temporal_channels=4,
        pointwise_channels=4,
        num_layers=2,
        kernel_size=3,
        fc_dim=16,
        dropout=0.1,
        return_sequence=False,
    )

    y = model(x_values=x_values, x_decay=x_decay, static=x_static)

    print("Smoke test passed.")
    print("x_values shape:", x_values.shape)
    print("x_decay shape:", x_decay.shape)
    print("x_static shape:", x_static.shape)
    print("output shape:", y.shape)
    print("=" * 80)


class SimpleLoSDataset(Dataset):
    """
    Converts task samples into tensors expected by TPC.

    Expected incoming time_series layout from the task:
        [value_1, mask_1, decay_1, value_2, mask_2, decay_2, ...]

    This wrapper feeds:
        - x_values
        - x_decay
        - x_mask
        - static
        - target
    """

    def __init__(
        self,
        samples: List[dict],
        channel_mode: str = "full",
        include_categorical_statics: bool = False,
        categorical_feature_names: Optional[List[str]] = None,
        categorical_vocab: Optional[Dict[str, Dict[str, int]]] = None,
    ) -> None:
        self.samples = samples
        self.channel_mode = channel_mode
        self.include_categorical_statics = include_categorical_statics
        self.categorical_feature_names = categorical_feature_names or []
        self.categorical_vocab = categorical_vocab or {}

    def __len__(self) -> int:
        return len(self.samples)

    def _split_value_mask_decay(
        self,
        ts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Splits [T, 3F] into:
            values: [T, F]
            masks:  [T, F]
            decay:  [T, F]
        """
        if ts.ndim != 2:
            raise ValueError(f"Expected time_series shape [T, D], got {tuple(ts.shape)}")

        feat_dim = ts.shape[1]
        if feat_dim % 3 != 0:
            raise ValueError(
                "Expected feature dimension divisible by 3 for "
                f"[value, mask, decay], got {feat_dim}"
            )

        num_raw_features = feat_dim // 3

        values = []
        masks = []
        decay = []

        for i in range(num_raw_features):
            base = i * 3
            values.append(ts[:, base].unsqueeze(1))
            masks.append(ts[:, base + 1].unsqueeze(1))
            decay.append(ts[:, base + 2].unsqueeze(1))

        values_tensor = torch.cat(values, dim=1)
        masks_tensor = torch.cat(masks, dim=1)
        decay_tensor = torch.cat(decay, dim=1)

        return values_tensor, masks_tensor, decay_tensor

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        ts = sample["time_series"]
        if not isinstance(ts, torch.Tensor):
            ts = torch.tensor(ts, dtype=torch.float32)
        else:
            ts = ts.float()

        static = sample["static"]
        if not isinstance(static, torch.Tensor):
            static = torch.tensor(static, dtype=torch.float32)
        else:
            static = static.float()

        if self.include_categorical_statics:
            raw_cats = sample.get("categorical_static_raw", {})
            cat_vec = encode_categorical_one_hot(
                raw_dict=raw_cats,
                categorical_feature_names=self.categorical_feature_names,
                vocab=self.categorical_vocab,
            )
            cat_tensor = torch.tensor(cat_vec, dtype=torch.float32)
            static = torch.cat([static, cat_tensor], dim=0)

        x_values, x_mask, x_decay = self._split_value_mask_decay(ts)

        target = sample["target_los_sequence"]
        if not isinstance(target, torch.Tensor):
            target = torch.tensor(target, dtype=torch.float32)
        else:
            target = target.float()

        if self.channel_mode == "no_decay":
            x_decay = torch.zeros_like(x_decay)
        elif self.channel_mode in {"full", "no_mask"}:
            pass
        else:
            raise ValueError(f"Unknown channel_mode: {self.channel_mode}")

        return {
            "x_values": x_values,
            "x_decay": x_decay,
            "x_mask": x_mask,
            "static": static,
            "target": target,
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    max_t = max(item["x_values"].shape[0] for item in batch)
    feat_dim = batch[0]["x_values"].shape[1]

    padded_values = []
    padded_decay = []
    padded_mask = []
    padded_targets = []
    padded_target_masks = []
    statics = []

    for item in batch:
        x_values = item["x_values"]
        x_decay = item["x_decay"]
        x_mask = item["x_mask"]
        target = item["target"]
        target_mask = torch.ones(target.shape[0], dtype=torch.float32)

        pad_len = max_t - x_values.shape[0]
        if pad_len > 0:
            value_pad = torch.zeros(pad_len, feat_dim, dtype=x_values.dtype)
            decay_pad = torch.zeros(pad_len, feat_dim, dtype=x_decay.dtype)
            mask_pad = torch.zeros(pad_len, feat_dim, dtype=x_mask.dtype)
            target_pad = torch.zeros(pad_len, dtype=target.dtype)
            target_mask_pad = torch.zeros(pad_len, dtype=torch.float32)

            x_values = torch.cat([x_values, value_pad], dim=0)
            x_decay = torch.cat([x_decay, decay_pad], dim=0)
            x_mask = torch.cat([x_mask, mask_pad], dim=0)
            target = torch.cat([target, target_pad], dim=0)
            target_mask = torch.cat([target_mask, target_mask_pad], dim=0)

        padded_values.append(x_values)
        padded_decay.append(x_decay)
        padded_mask.append(x_mask)
        padded_targets.append(target)
        padded_target_masks.append(target_mask)
        statics.append(item["static"])

    return {
        "x_values": torch.stack(padded_values),           # [B, T, F]
        "x_decay": torch.stack(padded_decay),             # [B, T, F]
        "x_mask": torch.stack(padded_mask),               # [B, T, F]
        "static": torch.stack(statics),                   # [B, S]
        "target": torch.stack(padded_targets),            # [B, T]
        "target_mask": torch.stack(padded_target_masks),  # [B, T]
    }


def msle_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.mean(
        (torch.log(pred + 1.0 + eps) - torch.log(target + 1.0 + eps)) ** 2
    )


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - target) ** 2)


def masked_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    target_mask: torch.Tensor,
    loss_name: str,
    eps: float = 1e-6,
) -> torch.Tensor:
    valid = target_mask > 0

    pred_valid = pred[valid]
    target_valid = target[valid]

    if loss_name == "msle":
        return torch.mean(
            (torch.log(pred_valid + 1.0 + eps) - torch.log(target_valid + 1.0 + eps)) ** 2
        )
    elif loss_name == "mse":
        return torch.mean((pred_valid - target_valid) ** 2)
    else:
        raise ValueError(f"Unknown loss_name: {loss_name}")


def evaluate(model: TPC, loader: DataLoader, loss_name: str) -> Dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_mae_num = 0.0
    total_mse_num = 0.0
    total_count = 0.0

    with torch.no_grad():
        for batch in loader:
            pred = model(
                x_values=batch["x_values"],
                x_decay=batch["x_decay"],
                static=batch["static"],
            )
            target = batch["target"]
            target_mask = batch["target_mask"]

            loss = masked_loss(pred, target, target_mask, loss_name)
            total_loss += loss.item()

            valid = target_mask > 0
            pred_valid = pred[valid]
            target_valid = target[valid]

            total_mae_num += torch.sum(torch.abs(pred_valid - target_valid)).item()
            total_mse_num += torch.sum((pred_valid - target_valid) ** 2).item()
            total_count += float(pred_valid.numel())

    mean_loss = total_loss / max(len(loader), 1)
    mae = total_mae_num / max(total_count, 1.0)
    rmse = (total_mse_num / max(total_count, 1.0)) ** 0.5

    return {
        "loss": mean_loss,
        "mae": mae,
        "rmse": rmse,
    }


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
        help="full=value+decay, no_decay=value+zero_decay, no_mask=value+decay",
    )
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument(
        "--include_categorical_statics",
        action="store_true",
        help="Append one-hot categorical statics to numeric static vector",
    )
    parser.add_argument(
        "--smoke_model_only",
        action="store_true",
        help="Run a tiny synthetic forward pass through TPC and exit.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.smoke_model_only:
        run_model_smoke_test()
        return

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

    sample_to_inspect = None
    for s in samples:
        ts = s["time_series"]
        has_observation = False
        for row in ts:
            row_list = row.tolist() if hasattr(row, "tolist") else row
            masks = row_list[1::3]
            if any(m > 0 for m in masks):
                has_observation = True
                break
        if has_observation:
            sample_to_inspect = s
            break

    if sample_to_inspect is None:
        print("\nNo sample with observed measurements found in current subset.\n")

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

    debug_item = train_dataset[0]
    # print("\n--- WRAPPER OUTPUT INSPECTION ---")
    # print("x_values shape:", debug_item["x_values"].shape)
    # print("x_decay shape:", debug_item["x_decay"].shape)
    # print("x_mask shape:", debug_item["x_mask"].shape)
    # print("static shape:", debug_item["static"].shape)
    # print("target shape:", debug_item["target"].shape)
    # print("target:", debug_item["target"])
    # print("x_values:")
    # print(debug_item["x_values"])
    # print("x_decay:")
    # print(debug_item["x_decay"])
    # print("x_mask:")
    # print(debug_item["x_mask"])
    # print("--- END WRAPPER OUTPUT INSPECTION ---\n")

    val_dataset = SimpleLoSDataset(
        val_samples,
        channel_mode=args.channel_mode,
        include_categorical_statics=args.include_categorical_statics,
        categorical_feature_names=categorical_feature_names,
        categorical_vocab=categorical_vocab,
    )

    first_item = train_dataset[0]
    input_dim = first_item["x_values"].shape[1]
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
        return_sequence=True,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        epoch_train_loss = 0.0

        for batch in train_loader:
            pred = model(
                x_values=batch["x_values"],
                x_decay=batch["x_decay"],
                static=batch["static"],
            )

            loss = masked_loss(pred, batch["target"], batch["target_mask"], args.loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

        epoch_train_loss /= max(len(train_loader), 1)

        val_results = evaluate(model, val_loader, args.loss)
        epoch_val_loss = val_results["loss"]

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        best_val_loss = min(best_val_loss, float(epoch_val_loss))

        print(
            f"epoch={epoch} train_loss={epoch_train_loss:.4f} "
            f"val_loss={epoch_val_loss:.4f}"
        )

    final_val_results = evaluate(model, val_loader, args.loss)

    print("=" * 80)
    print("Run complete")
    print(f"channel_mode: {args.channel_mode}")
    print(f"include_categorical_statics: {args.include_categorical_statics}")
    print(f"final_train_loss: {train_losses[-1]:.4f}")
    print(f"final_val_loss: {final_val_results['loss']:.4f}")
    print(f"best_val_loss: {best_val_loss:.4f}")
    print(f"final_val_mae: {final_val_results['mae']:.4f}")
    print(f"final_val_rmse: {final_val_results['rmse']:.4f}")
    print(
        "ABLATION_SUMMARY "
        f"channel_mode={args.channel_mode} "
        f"include_categorical_statics={args.include_categorical_statics} "
        f"val_loss={final_val_results['loss']:.4f} "
        f"mae={final_val_results['mae']:.4f} "
        f"rmse={final_val_results['rmse']:.4f}"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
