"""
COVID-19 CXR Tutorial: ViT, Conformal Prediction & Interpretability.

Demonstrates: dataset loading, ViT training, LABEL conformal prediction,
and Chefer attention-based interpretability for uncertain samples.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from pyhealth.calib.predictionset import LABEL
from pyhealth.datasets import (
    COVID19CXRDataset,
    get_dataloader,
    split_by_sample_conformal,
)
from pyhealth.metrics.prediction_set import size, miscoverage_overall_ps
from pyhealth.models import TorchvisionModel
from pyhealth.trainer import Trainer
from pyhealth.interpret.methods import CheferRelevance
from pyhealth.interpret.utils import visualize_image_attr

# Configuration
DATA_ROOT = "/home/johnwu3/projects/PyHealth_Branch_Testing/datasets"
ROOT = f"{DATA_ROOT}/COVID-19_Radiography_Dataset"
CACHE = "/home/johnwu3/projects/covid19cxr_base_cache"
TASK_CACHE = "/home/johnwu3/projects/covid19cxr_task_cache"
CKPT = "/home/johnwu3/projects/covid19cxr_vit_model.ckpt"
SEED = 42

if __name__ == "__main__":
    # Set seeds for reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # =========================================================================
    # Cell 1: Data Loading & Model Training
    # =========================================================================

    # Load dataset and create train/val/calibration/test splits
    dataset = COVID19CXRDataset(ROOT, cache_dir=CACHE, num_workers=8)
    sample_dataset = dataset.set_task(cache_dir=TASK_CACHE, num_workers=8)
    train_data, val_data, cal_data, test_data = split_by_sample_conformal(
        sample_dataset, ratios=[0.6, 0.1, 0.15, 0.15]
    )

    # Create dataloaders
    train_loader = get_dataloader(train_data, batch_size=64, shuffle=True)
    val_loader = get_dataloader(val_data, batch_size=64, shuffle=False)
    test_loader = get_dataloader(test_data, batch_size=64, shuffle=False)

    # Initialize ViT model
    model = TorchvisionModel(
        dataset=sample_dataset,
        model_name="vit_b_16",
        model_config={"weights": "DEFAULT"},
    )
    device = "cuda:4" if torch.cuda.is_available() else "cpu"

    # Train model
    trainer = Trainer(model=model, device=device, enable_logging=False)
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=20,
        monitor="accuracy",
    )
    trainer.save_ckpt(CKPT)

    # Evaluate test performance
    test_metrics = trainer.evaluate(test_loader)
    print(f"Test Performance: {test_metrics}")

    # =========================================================================
    # Cell 2: Conformal Prediction with LABEL
    # =========================================================================

    # Label mapping for visualization
    label_vocab = sample_dataset.output_processors["disease"].label_vocab
    id2label = {v: k for k, v in label_vocab.items()}

    # Calibrate LABEL predictor (90% coverage target)
    label_predictor = LABEL(model=model, alpha=0.01)
    label_predictor.calibrate(cal_dataset=cal_data)

    # Run inference to get prediction sets
    cal_trainer = Trainer(model=label_predictor, device=device)
    results = cal_trainer.inference(
        test_loader, additional_outputs=["y_predset"]
    )
    y_true, y_predset = results[0], results[3]["y_predset"]

    # Compute and print coverage metrics
    coverage = 1 - miscoverage_overall_ps(y_predset, y_true)
    avg_set_size = size(y_predset)
    print(f"Coverage: {coverage:.1%}, Avg set size: {avg_set_size:.2f}")

    # Use sample index 0 (known to be uncertain with SEED=42)
    sample_idx = 0
    single_loader = get_dataloader(test_data, batch_size=1, shuffle=False)
    batch = next(iter(single_loader))

    # Get model prediction and prediction set for this sample
    model.eval()
    with torch.no_grad():
        pred_prob = model(**batch)["y_prob"][0]
    pred_class = pred_prob.argmax().item()
    true_label = batch["disease"].item()
    sample_predset = y_predset[sample_idx]
    predset_class_indices = [i for i, v in enumerate(sample_predset) if v]
    predset_classes = [id2label[i] for i in predset_class_indices]

    # Print sample details
    true_name = id2label[true_label]
    pred_name = id2label[pred_class]
    set_size = len(predset_classes)
    print(f"Sample {sample_idx}: True={true_name}, Pred={pred_name}, "
          f"Set={predset_classes} (size={set_size})")

    # =========================================================================
    # Cell 3: Interpretability (attribution for each class in prediction set)
    # =========================================================================
    # Initialize Chefer/AttentionGrad interpreter
    chefer = CheferRelevance(model)
    n_classes = len(predset_class_indices)

    # Compute attribution for each class in the prediction set
    overlays = []
    for class_idx in predset_class_indices:
        attr_map = chefer.attribute(class_index=class_idx, **batch)["image"]
        _, _, overlay = visualize_image_attr(
            image=batch["image"][0],
            attribution=attr_map[0, 0],
        )
        overlays.append((class_idx, overlay))

    # Create figure: ground truth + attribution for each class
    figsize = (5 * (n_classes + 1), 5)
    fig, axes = plt.subplots(1, n_classes + 1, figsize=figsize)

    # Ground truth image
    img, _, _ = visualize_image_attr(
        image=batch["image"][0],
        attribution=torch.zeros_like(batch["image"][0, 0]),
    )
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title(f"Ground Truth: {true_name}", fontsize=12)
    axes[0].axis('off')

    # Plot attributions
    for i, (class_idx, overlay) in enumerate(overlays):
        axes[i + 1].imshow(overlay)
        prob = pred_prob[class_idx].item()
        class_name = id2label[class_idx]
        axes[i + 1].set_title(f"{class_name} ({prob:.1%})", fontsize=12)
        axes[i + 1].axis('off')

    plt.suptitle("Uncertain Prediction: Multiple Classes", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig("covid19_cxr_interpretability.png", dpi=150)
    print("Saved visualization to covid19_cxr_interpretability.png")
