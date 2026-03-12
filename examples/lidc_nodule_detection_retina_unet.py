"""Synthetic ablation example for RetinaUNet.

This script follows the course naming convention:
`examples/{dataset}_{task_name}_{model}.py`

The dataset is a tiny synthetic stand-in to validate:
- model wiring
- training/evaluation loop
- ablation workflow

For full LIDC experiments, replace `build_synthetic_dataset` with a real loader.
"""

from __future__ import annotations

import argparse
import random

import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader, split_by_visit
from pyhealth.models import RetinaUNet
from pyhealth.trainer import Trainer


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def make_circle_image(size: int, radius: int, cx: int, cy: int) -> list[list[float]]:
    image = []
    for y in range(size):
        row = []
        for x in range(size):
            dist = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
            row.append(1.0 if dist <= radius else 0.0)
        image.append(row)
    return image


def build_synthetic_dataset(num_samples: int, image_size: int):
    samples = []
    for idx in range(num_samples):
        has_nodule = idx % 2
        radius = 4 if has_nodule else 2
        cx = 10 + (idx % 8)
        cy = 12 + (idx % 10)
        image = make_circle_image(image_size, radius, cx, cy)
        samples.append(
            {
                "patient_id": f"p-{idx}",
                "visit_id": f"v-{idx}",
                "image": image,
                "label": has_nodule,
            }
        )
    dataset = create_sample_dataset(
        samples=samples,
        input_schema={"image": "tensor"},
        output_schema={"label": "binary"},
        dataset_name="lidc_synthetic",
    )
    return dataset


def run_ablation(
    dataset,
    batch_size: int,
    epochs: int,
    device: str,
    base_channels: int,
    seg_loss_weight: float,
) -> dict:
    train_data, val_data, test_data = split_by_visit(dataset, [0.6, 0.2, 0.2], seed=7)
    train_loader = get_dataloader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = get_dataloader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = get_dataloader(test_data, batch_size=batch_size, shuffle=False)

    model = RetinaUNet(
        dataset=dataset,
        in_channels=1,
        base_channels=base_channels,
        seg_loss_weight=seg_loss_weight,
    )
    trainer = Trainer(model=model, device=device)
    trainer.train(train_dataloader=train_loader, val_dataloader=val_loader, epochs=epochs)
    return trainer.evaluate(test_loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-samples", type=int, default=80)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    dataset = build_synthetic_dataset(args.num_samples, args.image_size)

    experiments = [
        {"name": "cls_only", "base_channels": 16, "seg_loss_weight": 0.0},
        {"name": "cls_plus_seg", "base_channels": 16, "seg_loss_weight": 0.1},
    ]

    print("Running RetinaUNet ablation on synthetic LIDC-style data")
    for exp in experiments:
        result = run_ablation(
            dataset=dataset,
            batch_size=args.batch_size,
            epochs=args.epochs,
            device=args.device,
            base_channels=exp["base_channels"],
            seg_loss_weight=exp["seg_loss_weight"],
        )
        print("-" * 80)
        print(f"Experiment: {exp['name']}")
        print(f"base_channels={exp['base_channels']}, seg_loss_weight={exp['seg_loss_weight']}")
        print(f"Metrics: {result}")


if __name__ == "__main__":
    main()
