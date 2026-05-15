"""
Ablation Study: Effect of min_area on detection outputs.

This experiment demonstrates how changing the minimum object area
threshold affects the number of detected bounding boxes.

We use synthetic data for fast execution.

Findings:
- Lower min_area increases sensitivity (more detections)
- Higher min_area filters small objects
"""

import numpy as np
from pyhealth.tasks.retina_unet_detection import RetinaUNetDetectionTask


def generate_sample():
    """Creates a sample with multiple objects of different sizes."""
    image = np.random.rand(128, 128, 3)
    mask = np.zeros((128, 128), dtype=np.int32)

    # Large object
    mask[10:40, 10:40] = 1

    # Medium object
    mask[60:80, 60:80] = 2

    # Small object
    mask[90:95, 90:95] = 3

    return {"image": image, "mask": mask}


def run_ablation():
    sample = generate_sample()

    configs = [1, 20, 100]  # different min_area values

    results = {}

    for min_area in configs:
        task = RetinaUNetDetectionTask(min_area=min_area)
        output = task.process_sample(sample)

        num_boxes = output["boxes"].shape[0]
        results[min_area] = num_boxes

    return results


def main():
    results = run_ablation()

    print("Ablation Results:")
    for min_area, num_boxes in results.items():
        print(f"min_area={min_area} -> detected boxes={num_boxes}")


if __name__ == "__main__":
    main()
