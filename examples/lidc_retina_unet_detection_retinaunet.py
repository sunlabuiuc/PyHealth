"""Ablation example: RetinaUNetCTDataset + RetinaUNetDetectionTask.

This script demonstrates the dataset + task pipeline on a small
synthetic CT-like corpus (no real LIDC data required) and sweeps two
dataset knobs that directly affect the detection targets fed to a
downstream Retina U-Net model:

1. ``skip_empty_slices`` — keeps only slices that contain lesions. This
   is the common training regime for medical detection: most CT slices
   are background-only, so including them dilutes positive supervision.

2. ``hu_window`` — clip-and-normalize the intensity range. Different
   anatomical targets need different windows (lung vs. abdomen), and
   the chosen window changes the numerical distribution of the input.

For each configuration we report how many slices survive, how many
bounding boxes the Retina U-Net task extracts in total, and the mean
box area — all stats a downstream detector's training loop would be
sensitive to.

Run::

    python examples/lidc_retina_unet_detection_retinaunet.py
"""

from typing import Dict, Tuple

import numpy as np

from pyhealth.datasets.retina_unet_ct_dataset import RetinaUNetCTDataset
from pyhealth.tasks.retina_unet_detection import RetinaUNetDetectionTask


def build_synthetic_corpus(
    num_patients: int = 3, depth: int = 8, hw: int = 64, seed: int = 0
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Build a toy multi-patient CT corpus with a few lesions per volume."""
    rng = np.random.default_rng(seed)
    volumes: Dict[str, np.ndarray] = {}
    masks: Dict[str, np.ndarray] = {}

    for p in range(num_patients):
        # CT-like intensities: roughly centered around 0 HU, stddev 300
        volume = (rng.standard_normal((depth, hw, hw)) * 300.0).astype(np.float32)
        mask = np.zeros((depth, hw, hw), dtype=np.int32)

        num_lesions = rng.integers(low=1, high=4)
        for lesion_id in range(1, num_lesions + 1):
            z = int(rng.integers(0, depth))
            y = int(rng.integers(0, hw - 10))
            x = int(rng.integers(0, hw - 10))
            h = int(rng.integers(3, 10))
            w = int(rng.integers(3, 10))
            mask[z, y : y + h, x : x + w] = lesion_id
            # brighten the lesion region so HU-windowing matters
            volume[z, y : y + h, x : x + w] += 500.0

        volumes[f"patient_{p:03d}"] = volume
        masks[f"patient_{p:03d}"] = mask

    return volumes, masks


def summarize(processed) -> Dict[str, float]:
    total_boxes = sum(s["boxes"].shape[0] for s in processed)
    mean_image = float(np.mean([s["image"].mean() for s in processed])) if processed else 0.0

    if total_boxes == 0:
        return {
            "num_samples": len(processed),
            "total_boxes": 0,
            "mean_area": 0.0,
            "mean_image": mean_image,
        }

    areas = []
    for s in processed:
        for box in s["boxes"]:
            x_min, y_min, x_max, y_max = box
            areas.append((x_max - x_min + 1) * (y_max - y_min + 1))
    return {
        "num_samples": len(processed),
        "total_boxes": total_boxes,
        "mean_area": float(np.mean(areas)),
        "mean_image": mean_image,
    }


def run_ablation() -> None:
    volumes, masks = build_synthetic_corpus()
    task = RetinaUNetDetectionTask()

    configs = [
        {"skip_empty_slices": False, "hu_window": None},
        {"skip_empty_slices": True, "hu_window": None},
        {"skip_empty_slices": True, "hu_window": (-1000.0, 400.0)},   # lung
        {"skip_empty_slices": True, "hu_window": (-160.0, 240.0)},    # abdomen
    ]

    header = (
        f"{'skip_empty':>11} | {'hu_window':>18} | {'samples':>7} | "
        f"{'boxes':>5} | {'mean_area':>9} | {'mean_image':>10}"
    )
    print(header)
    print("-" * len(header))
    for cfg in configs:
        ds = RetinaUNetCTDataset(volumes=volumes, masks=masks, **cfg)
        processed = ds.set_task(task)
        stats = summarize(processed)
        print(
            f"{str(cfg['skip_empty_slices']):>11} | "
            f"{str(cfg['hu_window']):>18} | "
            f"{stats['num_samples']:>7} | "
            f"{stats['total_boxes']:>5} | "
            f"{stats['mean_area']:>9.2f} | "
            f"{stats['mean_image']:>10.4f}"
        )


if __name__ == "__main__":
    run_ablation()
