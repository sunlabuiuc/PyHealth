"""
COVID-19 CXR Tutorial: Conformal Prediction & Interpretability.

This tutorial demonstrates:
1. Training a Vision Transformer (ViT) model on COVID-19 CXR dataset
2. Conformal prediction using LABEL for uncertainty quantification
3. Interpretability visualization using ViT attention-based attribution
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
from pyhealth.models import TorchvisionModel
from pyhealth.trainer import Trainer, get_metrics_fn
from pyhealth.interpret.methods import CheferRelevance
from pyhealth.interpret.utils import visualize_image_attr


def main():
    """Main function to run the COVID-19 CXR tutorial."""
    torch.manual_seed(42)
    np.random.seed(42)

    # ============================================================================
    # STEP 1: Load and prepare dataset
    # ============================================================================
    print("=" * 80)
    print("STEP 1: Loading COVID-19 CXR Dataset")
    print("=" * 80)

    root = "/home/johnwu3/projects/PyHealth_Branch_Testing/datasets/COVID-19_Radiography_Dataset"
    base_cache = "/home/johnwu3/projects/covid19cxr_base_cache"
    task_cache = "/home/johnwu3/projects/covid19cxr_task_cache"
    model_checkpoint = "/home/johnwu3/projects/covid19cxr_vit_model.ckpt"
    
    base_dataset = COVID19CXRDataset(root, cache_dir=base_cache, num_workers=4)
    sample_dataset = base_dataset.set_task(cache_dir=task_cache, num_workers=4)

    print(f"Total samples: {len(sample_dataset)}")
    print(f"Task mode: {sample_dataset.output_schema}")

    label_vocab = sample_dataset.output_processors["disease"].label_vocab
    id2label = {idx: label for label, idx in label_vocab.items()}
    print(f"Classes: {list(id2label.values())}")

    train_data, val_data, cal_data, test_data = split_by_sample_conformal(
        dataset=sample_dataset, ratios=[0.6, 0.1, 0.15, 0.15]
    )

    print(f"Train: {len(train_data)}, Val: {len(val_data)}")
    print(f"Cal: {len(cal_data)}, Test: {len(test_data)}")

    train_loader = get_dataloader(train_data, batch_size=64, shuffle=True)
    val_loader = get_dataloader(val_data, batch_size=64, shuffle=False)
    test_loader = get_dataloader(test_data, batch_size=64, shuffle=False)

    # ============================================================================
    # STEP 2: Train or Load Vision Transformer (ViT) model
    # ============================================================================
    print("\n" + "=" * 80)
    print("STEP 2: Vision Transformer (ViT) Model")
    print("=" * 80)

    model = TorchvisionModel(
        dataset=sample_dataset,
        model_name="vit_b_16",
        model_config={"weights": "DEFAULT"},
    )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    if os.path.exists(model_checkpoint):
        print(f"Loading model from {model_checkpoint}")
        trainer = Trainer(model=model, device=device, checkpoint_path=model_checkpoint)
    else:
        print("Training new model...")
        trainer = Trainer(model=model, device=device)
        trainer.train(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            epochs=50,
            monitor="accuracy",
        )
        trainer.save_ckpt(model_checkpoint)
        print(f"Model saved to {model_checkpoint}")

    print("\nBase model performance on test set:")
    y_true_base, y_prob_base, _ = trainer.inference(test_loader)
    base_metrics = get_metrics_fn("multiclass")(
        y_true_base, y_prob_base, metrics=["accuracy", "f1_weighted"]
    )
    for metric, value in base_metrics.items():
        print(f"  {metric}: {value:.4f}")

    # ============================================================================
    # STEP 3: Conformal Prediction with LABEL
    # ============================================================================
    print("\n" + "=" * 80)
    print("STEP 3: Conformal Prediction with LABEL")
    print("=" * 80)

    alpha = 0.1
    print(f"Target miscoverage rate: {alpha} (90% coverage)")

    label_predictor = LABEL(model=model, alpha=alpha)
    label_predictor.calibrate(cal_dataset=cal_data)

    y_true_label, y_prob_label, _, extra_label = Trainer(model=label_predictor).inference(
        test_loader, additional_outputs=["y_predset"]
    )

    label_metrics = get_metrics_fn("multiclass")(
        y_true_label,
        y_prob_label,
        metrics=["accuracy", "miscoverage_ps"],
        y_predset=extra_label["y_predset"],
    )

    predset = torch.tensor(extra_label["y_predset"])
    avg_set_size = predset.float().sum(dim=1).mean().item()
    miscoverage = float(np.mean(label_metrics["miscoverage_ps"]))
    coverage = 1 - miscoverage

    print(f"\n  Target coverage:    {1-alpha:.0%}")
    print(f"  Empirical coverage: {coverage:.2%}")
    print(f"  Average set size:   {avg_set_size:.2f}")

    # ============================================================================
    # STEP 4: Example Predictions with Prediction Sets
    # ============================================================================
    print("\n" + "=" * 80)
    print("STEP 4: Example Predictions with Prediction Sets")
    print("=" * 80)

    for i in range(min(5, len(y_true_label))):
        true_class = int(y_true_label[i])
        pred_class = int(y_prob_label[i].argmax())
        pred_set = torch.where(predset[i])[0].cpu().numpy()

        print(f"\nExample {i+1}:")
        print(f"  True: {id2label.get(true_class)}, Pred: {id2label.get(pred_class)}")
        print(f"  Prediction set: {[id2label.get(c) for c in pred_set]}")

    # ============================================================================
    # STEP 5: Interpretability Visualization
    # ============================================================================
    print("\n" + "=" * 80)
    print("STEP 5: Interpretability Visualization")
    print("=" * 80)

    single_loader = get_dataloader(test_data, batch_size=1, shuffle=False)
    n_viz = 3
    print(f"\nGenerating Chefer attention attribution for {n_viz} samples...")

    model.eval()
    viz_samples = [batch for i, batch in enumerate(single_loader) if i < n_viz]

    fig, axes = plt.subplots(n_viz, 3, figsize=(15, 5 * n_viz))

    # Initialize Chefer interpreter (auto-detects ViT)
    chefer_gen = CheferRelevance(model)

    for idx, batch in enumerate(viz_samples):
        image = batch["image"]
        true_label = batch["disease"].item()

        with torch.no_grad():
            output = model(**batch)
            pred_prob = output["y_prob"][0]
            pred_class = pred_prob.argmax().item()

        # Get attribution map using attribute()
        # Returns dict keyed by feature key (e.g., {"image": tensor})
        # Input size is inferred automatically from image dimensions
        result = chefer_gen.attribute(
            interpolate=True,
            class_index=pred_class,
            **batch
        )
        attr_map = result["image"]  # Keyed by task schema's feature key
        
        img_display, vit_attr_display, attention_overlay = visualize_image_attr(
            image=image[0],
            attribution=attr_map[0, 0],
            interpolate=True,
        )

        # Plot
        ax1 = axes[idx, 0]
        ax1.imshow(img_display, cmap='gray' if img_display.ndim == 2 else None)
        ax1.set_title(f"Original\nTrue: {id2label.get(true_label)}")
        ax1.axis('off')

        ax2 = axes[idx, 1]
        ax2.imshow(vit_attr_display, cmap='hot')
        ax2.set_title(f"Attribution\nPred: {id2label.get(pred_class)}")
        ax2.axis('off')

        ax3 = axes[idx, 2]
        ax3.imshow(attention_overlay)
        ax3.set_title(f"Overlay\nConf: {pred_prob[pred_class]:.1%}")
        ax3.axis('off')

    plt.tight_layout()
    plt.savefig("covid19_cxr_interpretability.png", dpi=150, bbox_inches='tight')
    print("✓ Saved to: covid19_cxr_interpretability.png")


if __name__ == "__main__":
    main()
