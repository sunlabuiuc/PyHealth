"""Grad-CAM tutorial for CNN-based chest X-ray classification in PyHealth.

Prerequisites:
- A local COVID-19 Radiography Database root passed with ``--root``

Notes:
- For meaningful class-specific visualizations, pass ``--checkpoint`` with a
  trained PyHealth checkpoint. Without a checkpoint, the script still runs as a
  pipeline example, but the classification head is randomly initialized.
- ``--weights DEFAULT`` may trigger a first-run torchvision download. Use
  ``--weights none`` for an offline run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from pyhealth.datasets import COVID19CXRDataset, SampleDataset, get_dataloader
from pyhealth.interpret.methods import GradCAM
from pyhealth.interpret.utils import visualize_image_attr
from pyhealth.models import TorchvisionModel


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the Grad-CAM tutorial.

    Returns:
        argparse.Namespace: Parsed CLI arguments controlling dataset location,
            model initialization, runtime device, and output path.
    """
    parser = argparse.ArgumentParser(
        description="Run Grad-CAM on one chest X-ray sample.",
    )
    parser.add_argument(
        "--root",
        required=True,
        help="Path to the COVID-19 Radiography Database root directory.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional checkpoint to load before inference.",
    )
    parser.add_argument(
        "--output",
        default="gradcam_cxr_overlay.png",
        help="Where to save the Grad-CAM figure.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional device override such as 'cpu' or 'cuda:0'.",
    )
    parser.add_argument(
        "--weights",
        choices=["DEFAULT", "none"],
        default="DEFAULT",
        help="Torchvision backbone weights to use when initializing resnet18.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str | None) -> str:
    """Resolve the device string used for inference.

    Args:
        device_arg: Optional CLI override such as ``"cpu"`` or ``"cuda:0"``.

    Returns:
        str: The resolved device string.
    """
    if device_arg is not None:
        return device_arg
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def load_dataset(root: str) -> SampleDataset:
    """Load the COVID-19 CXR sample dataset for the tutorial.

    Args:
        root: Root directory containing the COVID-19 Radiography Database.

    Returns:
        SampleDataset: Task-applied sample dataset ready for dataloader use.

    Raises:
        SystemExit: If ``openpyxl`` is required but unavailable.
    """
    try:
        dataset = COVID19CXRDataset(root, num_workers=1)
        return dataset.set_task(num_workers=1)
    except ImportError as exc:
        if "openpyxl" in str(exc):
            raise SystemExit(
                "This example needs 'openpyxl' to read the raw metadata sheets. "
                "Install it with: pip install openpyxl"
            ) from exc
        raise


def main() -> None:
    """Run Grad-CAM on a single chest X-ray sample and save a figure."""
    args = parse_args()
    root = Path(args.root).expanduser()
    if not root.exists():
        raise SystemExit(f"Dataset root does not exist: {root}")

    sample_dataset = load_dataset(str(root))
    loader = get_dataloader(sample_dataset, batch_size=1, shuffle=False)
    batch = next(iter(loader))

    weights = None if args.weights == "none" else "DEFAULT"
    model = TorchvisionModel(
        dataset=sample_dataset,
        model_name="resnet18",
        model_config={"weights": weights},
    )
    device = resolve_device(args.device)
    model = model.to(device)
    model.eval()

    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint).expanduser()
        if not checkpoint_path.exists():
            raise SystemExit(f"Checkpoint does not exist: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(
            "Warning: no checkpoint provided. The classifier head is randomly "
            "initialized, so this run is only a pipeline example."
        )

    with torch.no_grad():
        y_prob = model(**batch)["y_prob"][0]

    label_vocab = sample_dataset.output_processors["disease"].label_vocab
    pred_class = int(torch.argmax(y_prob).item())
    id2label = {value: key for key, value in label_vocab.items()}
    pred_label = id2label[pred_class]

    gradcam = GradCAM(
        model,
        target_layer=model.model.layer4[-1].conv2,
        input_key="image",
    )
    cam = gradcam.attribute(class_index=pred_class, **batch)["image"]

    image, heatmap, overlay = visualize_image_attr(
        image=batch["image"][0],
        attribution=cam[0],
    )

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Input")
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title(f"Overlay: {pred_label}")
    axes[2].axis("off")

    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Predicted class: {pred_label}")
    print(f"Saved Grad-CAM visualization to {output_path.resolve()}")


if __name__ == "__main__":
    main()
