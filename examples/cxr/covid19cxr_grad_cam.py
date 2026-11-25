"""
COVID-19 CXR Grad-CAM: where a ResNet looks on a chest X-ray.

Trains (or loads) a ResNet-18 ``TorchvisionModel`` on the COVID-19 Radiography
Database, then explains one test image with ``GradCAM``: one heatmap per class,
showing the regions that raise that class's score.

Data: the COVID-19 Radiography Database,
https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database
(open download, no credentials). Pass ``--root`` as the folder with the four
class folders and their ``*.metadata.xlsx`` files.

Run:
    python examples/cxr/covid19cxr_grad_cam.py --root /path/to/COVID-19_Radiography_Dataset

ResNet-18 starts from ImageNet weights (downloaded by torchvision) and trains
for ``--epochs`` epochs (default 3); a GPU or Apple MPS is recommended. If
``--ckpt`` exists the model is loaded instead of trained. The figure is saved to
``covid19_cxr_grad_cam.png``.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from pyhealth.datasets import COVID19CXRDataset, get_dataloader, split_by_sample
from pyhealth.interpret.methods import GradCAM
from pyhealth.interpret.utils import visualize_image_attr
from pyhealth.models import TorchvisionModel
from pyhealth.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--root", required=True, help="COVID-19_Radiography_Dataset folder")
    parser.add_argument("--cache", default=None, help="PyHealth cache directory")
    parser.add_argument("--ckpt", default="covid19cxr_resnet18.ckpt")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--no-pretrained", action="store_true", help="random init (offline)")
    parser.add_argument("--sample", type=int, default=0, help="test sample to explain")
    parser.add_argument("--device", default=None, help="default: cuda, then mps, then cpu")
    parser.add_argument("--out", default="covid19_cxr_grad_cam.png")
    return parser.parse_args()


def pick_device(requested):
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(42)
    np.random.seed(42)
    device = pick_device(args.device)

    # =========================================================================
    # 1. Data and model
    # =========================================================================
    dataset = COVID19CXRDataset(args.root, cache_dir=args.cache)
    samples = dataset.set_task()
    train_data, val_data, test_data = split_by_sample(samples, [0.7, 0.1, 0.2])

    model = TorchvisionModel(
        dataset=samples,
        model_name="resnet18",
        model_config={"weights": None if args.no_pretrained else "DEFAULT"},
    )
    trainer = Trainer(model=model, device=device, enable_logging=False)
    if os.path.exists(args.ckpt):
        trainer.load_ckpt(args.ckpt)
    else:
        trainer.train(
            train_dataloader=get_dataloader(train_data, args.batch_size, shuffle=True),
            val_dataloader=get_dataloader(val_data, args.batch_size, shuffle=False),
            epochs=args.epochs,
            monitor="accuracy",
        )
        trainer.save_ckpt(args.ckpt)
    test_loader = get_dataloader(test_data, args.batch_size, shuffle=False)
    print(f"Test performance: {trainer.evaluate(test_loader)}")

    # =========================================================================
    # 2. Grad-CAM for every class on one test image
    # =========================================================================
    label_vocab = samples.output_processors["disease"].label_vocab
    id2label = {index: name for name, index in label_vocab.items()}

    if not 0 <= args.sample < len(test_data):
        raise SystemExit(f"--sample must be in [0, {len(test_data)}), got {args.sample}")
    single = get_dataloader(test_data, batch_size=1, shuffle=False)
    for index, batch in enumerate(single):
        if index == args.sample:
            break

    # layer4 is ResNet's last convolutional stage, as in the Grad-CAM paper
    grad_cam = GradCAM(model, target_layers="model.layer4")
    model.eval()
    with torch.no_grad():
        probs = model(**batch)["y_prob"][0].cpu()

    overlays = []
    for class_idx in range(len(id2label)):
        cam = grad_cam.attribute(target_class_idx=class_idx, **batch)["image"]
        _, _, overlay = visualize_image_attr(
            image=batch["image"][0], attribution=cam[0, 0].cpu()
        )
        overlays.append((class_idx, overlay))

    # =========================================================================
    # 3. Figure: the X-ray, then one heatmap per class
    # =========================================================================
    true_name = id2label[batch["disease"].item()]
    fig, axes = plt.subplots(1, len(overlays) + 1, figsize=(4 * (len(overlays) + 1), 4))
    image, _, _ = visualize_image_attr(
        image=batch["image"][0], attribution=torch.zeros_like(batch["image"][0, 0])
    )
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title(f"Ground truth: {true_name}")
    for ax, (class_idx, overlay) in zip(axes[1:], overlays):
        ax.imshow(overlay)
        ax.set_title(f"{id2label[class_idx]} ({probs[class_idx]:.1%})")
    for ax in axes:
        ax.axis("off")
    plt.suptitle("Grad-CAM (ResNet-18, layer4) per class", y=1.02)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved {args.out}")
