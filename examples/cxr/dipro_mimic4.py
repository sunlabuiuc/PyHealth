"""DiPro for disease progression between chest X-rays: smoke test.

SMOKE TEST ONLY. By default this script writes the tiny fabricated MIMIC-IV,
MIMIC-CXR and Chest ImaGenome tree of ``disease_progression_mimic4.py`` to a
temporary directory (random values and images, no patient data), then trains a
tiny DiPro for a few steps. It reproduces nothing, and its numbers mean nothing
clinically.

To run on the real data, you need credentialed PhysioNet access to MIMIC-IV,
MIMIC-CXR-JPG and Chest ImaGenome. Unzip ``silver_dataset/scene_graph.zip``,
then pass the three roots and DiPro's sizes:

    python examples/cxr/dipro_mimic4.py \
        --ehr-root /path/to/mimic-iv/2.2 \
        --cxr-root /path/to/mimic-cxr-jpg/2.0.0 \
        --imagenome-root /path/to/chest-imagenome/1.0.0 \
        --image-size 224 --d-model 256 --backbone resnet50 --pretrained \
        --batch-size 8 --epochs 100

That is still not DiPro's full setup: it uses 17 EHR variables, not 38, and
``Trainer`` has no gradient accumulation or cosine decay.

For reference, Liu et al. (2025) report for DiPro on this task, over three
seeds: precision 0.484 ± 0.008, recall 0.471 ± 0.024, F1 0.466 ± 0.018, AUPRC
0.478 ± 0.018, AUROC 0.664 ± 0.013 and accuracy 0.565 ± 0.013, on 3982/560/1137
train/val/test samples.

Paper:
    Liu et al. "Multimodal Disease Progression Modeling via Spatiotemporal
    Disentanglement and Multiscale Alignment." NeurIPS 2025.
    https://arxiv.org/abs/2510.11112
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from disease_progression_mimic4 import write_ehr, write_imagenome, write_images
from sklearn.metrics import f1_score

from pyhealth.datasets import GoldTestSplit, MIMIC4ImaGenomeDataset, get_dataloader
from pyhealth.models import DiPro
from pyhealth.tasks import DiseaseProgressionMIMIC4
from pyhealth.trainer import Trainer


def add_boxes(root: Path, regions: list[str]) -> None:
    """Adds fabricated boxes to the scene graphs, leaving out the last region."""
    for path in (root / "silver_dataset" / "scene_graph").glob("*.json"):
        graph = json.loads(path.read_text())
        graph["objects"] = [
            {"bbox_name": region, "original_x1": 4 * index, "original_y1": 4,
             "original_x2": 4 * index + 20, "original_y2": 40}
            for index, region in enumerate(regions[:-1])
        ]
        path.write_text(json.dumps(graph))


def macro_f1(y_true: np.ndarray, y_prob: np.ndarray, diseases: int) -> float:
    """Computes DiPro's F1: macro over statuses, then mean over diseases.

    ``Trainer``'s multilabel metrics would count a disease that is not
    mentioned as a negative, so they are not comparable with the paper.
    """
    scores = []
    truth = y_true.reshape(-1, diseases, 3)
    predicted = y_prob.reshape(-1, diseases, 3).argmax(-1)
    for disease in range(diseases):
        mentioned = truth[:, disease].sum(-1) > 0
        if mentioned.any():
            scores.append(
                f1_score(
                    truth[mentioned, disease].argmax(-1),
                    predicted[mentioned, disease],
                    average="macro",
                )
            )
    return float(np.mean(scores)) if scores else float("nan")


def parse_args() -> argparse.Namespace:
    """Parses CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ehr-root", help="MIMIC-IV root; fabricated data if unset")
    parser.add_argument("--cxr-root", help="MIMIC-CXR-JPG root")
    parser.add_argument("--imagenome-root", help="chest-imagenome/1.0.0 root")
    parser.add_argument("--patients", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1.6e-4, help="DiPro's scaled rate")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    """Builds the samples, trains DiPro and evaluates it."""
    args = parse_args()
    torch.manual_seed(args.seed)
    task = DiseaseProgressionMIMIC4(
        image_size=args.image_size, region_crops=True, include_demographics=True
    )
    with tempfile.TemporaryDirectory() as tmp:
        ehr_root, cxr_root = args.ehr_root, args.cxr_root
        imagenome_root = args.imagenome_root
        if ehr_root is None:
            print("SMOKE TEST on fabricated data; results are not meaningful.")
            root, rng = Path(tmp), np.random.default_rng(args.seed)
            ehr_root = root / "mimic-iv"
            cxr_root = root / "mimic-cxr"
            imagenome_root = root / "chest-imagenome"
            write_ehr(ehr_root, args.patients, rng)
            write_images(cxr_root, args.patients, rng)
            write_imagenome(imagenome_root, args.patients, rng)
            add_boxes(imagenome_root, task.REGIONS)
        dataset = MIMIC4ImaGenomeDataset(
            imagenome_root=str(imagenome_root),
            imagenome_tables=[
                "silver_images",
                "silver_objects",
                "silver_comparisons",
                "gold_comparisons",
            ],
            ehr_root=str(ehr_root),
            cxr_root=str(cxr_root),
            ehr_tables=["chartevents", "labevents"],
            cxr_tables=["metadata"],
            cache_dir=str(Path(tmp) / "cache") if args.ehr_root is None else None,
        )
        train, val, test = dataset.set_task(
            task, split=GoldTestSplit(task.gold_subjects(dataset), seed=args.seed)
        )
        print(f"{len(train)}/{len(val)}/{len(test)} train/val/test samples")

        model = DiPro(
            train,
            d_model=args.d_model,
            backbone=args.backbone,
            pretrained=args.pretrained,
        )
        model.set_class_weights(train)
        print(f"{sum(p.numel() for p in model.parameters())} parameters")

        trainer = Trainer(model, metrics=["hamming_loss"], enable_logging=False)
        trainer.train(
            get_dataloader(train, batch_size=args.batch_size, shuffle=True),
            get_dataloader(val, batch_size=args.batch_size),
            epochs=args.epochs,
            optimizer_class=torch.optim.AdamW,
            optimizer_params={"lr": args.lr},
        )
        y_true, y_prob, loss = trainer.inference(
            get_dataloader(test, batch_size=args.batch_size)
        )
        print(f"test loss {loss:.4f}")
        print(f"test F1 {macro_f1(y_true, y_prob, model.num_diseases):.4f}")


if __name__ == "__main__":
    main()
