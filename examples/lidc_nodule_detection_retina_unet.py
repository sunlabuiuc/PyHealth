"""Synthetic ablation example for RetinaUNet.

Contributor: Tuan Nguyen
NetID: tuanmn2
Paper: Retina U-Net: Embarrassingly Simple Exploitation of Segmentation
    Supervision for Medical Object Detection
Paper link: https://proceedings.mlr.press/v116/jaeger20a/jaeger20a.pdf
Description: Lightweight synthetic ablation example for the RetinaUNet model
    in PyHealth.

This example is intentionally lightweight so it can be used in a PyHealth pull
request without depending on any real dataset. It demonstrates the intended
task contract for RetinaUNet:

- image tensor input
- bounding boxes + box labels for detection
- segmentation mask as auxiliary supervision

It also doubles as a minimal ablation example by comparing multiple model
configurations on the same synthetic dataset. The comparison is intentionally
small and fast; its purpose is to show how hyperparameter variations can be
tested, not to reproduce the original Retina U-Net paper benchmark.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Dataset

from pyhealth.models import RetinaUNet


class _DummyOutputProcessor:
    def size(self):
        return 1


class _RetinaConfigDataset:
    def __init__(self):
        self.input_schema = {"image": "tensor"}
        self.output_schema = {"label": "binary"}
        self.output_processors = {"label": _DummyOutputProcessor()}


class SyntheticLIDCDataset(Dataset):
    """Small synthetic dataset with positive and negative slices."""

    def __init__(self, num_samples: int, image_size: int):
        self.samples: List[Dict[str, torch.Tensor]] = []
        for idx in range(num_samples):
            image = torch.zeros(1, image_size, image_size)
            seg_target = torch.zeros(image_size, image_size, dtype=torch.long)

            if idx % 2 == 0:
                x1 = 8 + (idx % 6)
                y1 = 10 + (idx % 5)
                x2 = x1 + 10
                y2 = y1 + 8
                image[:, y1:y2, x1:x2] = 1.0
                class_label = 1 if (idx % 4 == 0) else 2
                seg_target[y1:y2, x1:x2] = class_label
                boxes = torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32)
                labels = torch.tensor([class_label], dtype=torch.long)
            else:
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                labels = torch.zeros((0,), dtype=torch.long)

            self.samples.append(
                {
                    "image": image,
                    "boxes": boxes,
                    "labels": labels,
                    "seg_target": seg_target,
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return self.samples[index]


def collate_detection_batch(
    batch: List[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor | list]:
    images = torch.stack([sample["image"] for sample in batch], dim=0)
    seg_target = torch.stack([sample["seg_target"] for sample in batch], dim=0)
    boxes = [sample["boxes"] for sample in batch]
    labels = [sample["labels"] for sample in batch]
    return {
        "image": images,
        "boxes": boxes,
        "labels": labels,
        "seg_target": seg_target,
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


@dataclass(frozen=True)
class AblationConfig:
    """Small configuration record for the synthetic comparison."""

    name: str
    base_channels: int
    learning_rate: float


def build_ablation_configs() -> List[AblationConfig]:
    """Returns a small set of fast, PR-friendly ablation configurations."""
    return [
        AblationConfig(name="small_width", base_channels=8, learning_rate=1e-3),
        AblationConfig(name="default_width", base_channels=16, learning_rate=1e-3),
        AblationConfig(name="lower_lr", base_channels=16, learning_rate=5e-4),
    ]


def run_epoch(
    model: RetinaUNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    """Runs one epoch and returns average training losses."""
    model.train()
    totals = {
        "loss": 0.0,
        "cls_loss": 0.0,
        "bbox_loss": 0.0,
        "seg_loss": 0.0,
    }
    num_steps = 0

    for step, batch in enumerate(loader, start=1):
        batch["image"] = batch["image"].to(device)
        batch["seg_target"] = batch["seg_target"].to(device)
        batch["boxes"] = [box.to(device) for box in batch["boxes"]]
        batch["labels"] = [label.to(device) for label in batch["labels"]]

        optimizer.zero_grad(set_to_none=True)
        output = model(**batch)
        output["loss"].backward()
        optimizer.step()

        num_steps += 1
        totals["loss"] += output["loss"].item()
        totals["cls_loss"] += output["cls_loss"].item()
        totals["bbox_loss"] += output["bbox_loss"].item()
        totals["seg_loss"] += output["seg_loss"].item()

        print(
            "[train] "
            f"step={step} "
            f"loss={output['loss'].item():.5f} "
            f"cls={output['cls_loss'].item():.5f} "
            f"bbox={output['bbox_loss'].item():.5f} "
            f"seg={output['seg_loss'].item():.5f}"
        )

    return {key: value / max(num_steps, 1) for key, value in totals.items()}


@torch.no_grad()
def run_eval(
    model: RetinaUNet,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Runs a tiny evaluation pass and summarizes prediction volume."""
    model.eval()
    batch = next(iter(loader))
    batch["image"] = batch["image"].to(device)
    output = model(image=batch["image"])

    num_detections = [
        int(detection["boxes"].shape[0]) for detection in output["detections"]
    ]
    avg_detections = sum(num_detections) / max(len(num_detections), 1)

    print(f"[eval] batch_detections={len(output['detections'])}")
    print(
        f"[eval] first_boxes_shape={tuple(output['detections'][0]['boxes'].shape)} "
        f"first_scores_shape={tuple(output['detections'][0]['scores'].shape)}"
    )

    return {
        "avg_detections_per_sample": avg_detections,
        "max_detections_in_batch": float(max(num_detections, default=0)),
    }


def run_ablation(
    config: AblationConfig,
    dataset: SyntheticLIDCDataset,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, float | int | str]:
    """Trains and evaluates one ablation configuration."""
    set_seed(args.seed)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_detection_batch,
    )

    config_dataset = _RetinaConfigDataset()
    model = RetinaUNet(
        dataset=config_dataset,
        in_channels=1,
        num_classes=2,
        base_channels=config.base_channels,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    print("\n" + "=" * 80)
    print(
        f"Running config={config.name} "
        f"base_channels={config.base_channels} "
        f"lr={config.learning_rate}"
    )

    train_metrics: Dict[str, float] = {}
    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}")
        train_metrics = run_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            device=device,
        )

    eval_metrics = run_eval(model=model, loader=loader, device=device)

    return {
        "config": config.name,
        "base_channels": config.base_channels,
        "learning_rate": config.learning_rate,
        "train_loss": train_metrics["loss"],
        "cls_loss": train_metrics["cls_loss"],
        "bbox_loss": train_metrics["bbox_loss"],
        "seg_loss": train_metrics["seg_loss"],
        "avg_detections": eval_metrics["avg_detections_per_sample"],
    }


def print_summary_table(results: List[Dict[str, float | int | str]]) -> None:
    """Prints a compact comparison table for the ablation study."""
    print("\n" + "=" * 80)
    print("Synthetic ablation summary")
    print(
        "config           base_channels   lr         "
        "train_loss   cls_loss   bbox_loss   seg_loss   avg_dets"
    )
    for result in results:
        print(
            f"{result['config']:<16} "
            f"{int(result['base_channels']):>13}   "
            f"{float(result['learning_rate']):<10.5f} "
            f"{float(result['train_loss']):>10.5f} "
            f"{float(result['cls_loss']):>10.5f} "
            f"{float(result['bbox_loss']):>11.5f} "
            f"{float(result['seg_loss']):>10.5f} "
            f"{float(result['avg_detections']):>9.2f}"
        )

    best_result = min(results, key=lambda item: float(item["train_loss"]))
    print("\nObservation:")
    print(
        "- Lower train loss is better in this toy setup. "
        f"The best synthetic config here is {best_result['config']}."
    )
    print(
        "- This comparison is only a lightweight PR-facing ablation and should "
        "not be interpreted as a paper benchmark."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    dataset = SyntheticLIDCDataset(
        num_samples=args.num_samples,
        image_size=args.image_size,
    )
    configs = build_ablation_configs()

    print("Running synthetic RetinaUNet ablation example")
    results = [
        run_ablation(config=config, dataset=dataset, args=args, device=device)
        for config in configs
    ]
    print_summary_table(results)


if __name__ == "__main__":
    main()
