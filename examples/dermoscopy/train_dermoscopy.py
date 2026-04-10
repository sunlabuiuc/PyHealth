# Contributor: [Your Name]
# NetID: [Your NetID]

"""
Baseline Training Pipeline for Melanoma Classification.

Trains various architectures (ResNet50, Swin Transformer, DINOv2) on the combined 
ISIC 2018 and HAM10000 dermoscopy datasets using the PyHealth Trainer.

Paper: "A Study of Artifacts on Melanoma Classification under Diffusion-Based Perturbations" (CHIL 2025)
Models:
    - ResNet: He et al., "Deep Residual Learning for Image Recognition" (CVPR 2016)
    - Swin: Liu et al., "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows" (ICCV 2021)
    - DINOv2: Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision" (2023)
"""

import argparse
import os
import re
import matplotlib.pyplot as plt

# Native PyHealth Core Imports
from pyhealth.datasets import split_by_patient, get_dataloader
from pyhealth.trainer import Trainer

# Native PyHealth Dermoscopy Imports
from pyhealth.datasets import DermoscopyDataset
from pyhealth.tasks import DermoscopyMelanomaClassification
from pyhealth.processors import DermoscopyImageProcessor
from pyhealth.models import TorchvisionModel, DINOv2

def plot_training_progress(output_dir: str, model_name: str, mode: str):
    """
    Parses the PyHealth Trainer's log.txt and exports visual Loss/AUC curves.
    
    Args:
        output_dir (str): The directory containing the generated PyHealth log.txt.
        model_name (str): The name of the architecture used (e.g., 'resnet50').
        mode (str): The frequency ablation mode used (e.g., 'whole').
    """
    log_path = os.path.join(output_dir, "log.txt")
    
    if not os.path.exists(log_path):
        print(f"[!] Warning: Could not find PyHealth log at {log_path} for visualization.")
        return

    epochs, losses, aucs = [], [], []
    
    with open(log_path, 'r') as f:
        for line in f:
            if "loss:" in line and "roc_auc:" in line:
                try:
                    epoch = int(re.search(r"epoch-(\d+)", line).group(1))
                    loss = float(re.search(r"loss:\s*([\d\.]+)", line).group(1))
                    auc = float(re.search(r"roc_auc:\s*([\d\.]+)", line).group(1))
                    epochs.append(epoch)
                    losses.append(loss)
                    aucs.append(auc)
                except AttributeError:
                    continue

    if not epochs:
        return

    plt.figure(figsize=(14, 5))
    
    # Plot Training Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, losses, marker='o', color='red', linestyle='-', linewidth=2)
    plt.title(f'Training Loss ({model_name.upper()} - {mode})', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Plot Validation ROC-AUC
    plt.subplot(1, 2, 2)
    plt.plot(epochs, aucs, marker='s', color='blue', linestyle='-', linewidth=2)
    plt.title(f'Validation ROC-AUC ({model_name.upper()} - {mode})', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('ROC AUC', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plot_file = os.path.join(output_dir, f"{model_name}_{mode}_training_curves.png")
    plt.savefig(plot_file, dpi=300)
    print(f"\n[*] Training progress visualization saved successfully to: {plot_file}")

def main():
    """Executes the training pipeline, from dataset loading to PyHealth Trainer evaluation."""
    parser = argparse.ArgumentParser(description="Train baseline Melanoma Classification models.")
    parser.add_argument('--model', type=str, choices=['resnet50', 'swin', 'dinov2'], default='resnet50', help="Architecture to train.")
    parser.add_argument('--mode', type=str, choices=['whole', 'high_whole', 'low_whole'], default='whole', help="Frequency ablation mode.")
    parser.add_argument('--data_dir', type=str, required=True, help="Root directory containing dermoscopy datasets.")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs.")
    args = parser.parse_args()

    # 1. Dataset Initialization & Task Assignment
    print(f"[*] Initializing Dataset and applying {args.mode} processor...")
    dataset = DermoscopyDataset(root=args.data_dir, dataset_name="isic2018", dev=False)
    processor = DermoscopyImageProcessor(mode=args.mode)
    
    task_dataset = dataset.set_task(
        task=DermoscopyMelanomaClassification,
        input_processors={"image": processor}
    )
    
    # 2. Data Splitting
    train_ds, val_ds, test_ds = split_by_patient(task_dataset, [0.8, 0.1, 0.1])
    train_loader = get_dataloader(train_ds, batch_size=32, shuffle=True)
    val_loader = get_dataloader(val_ds, batch_size=32, shuffle=False)

    # 3. Native Model Initialization
    print(f"[*] Constructing Architecture: {args.model.upper()}")
    
    # 2. Dataset-Aware Model Initialization
    if args.model == "dinov2":
        model = DINOv2(dataset=task_dataset, model_size="vits14")
    else:
        model_name = "swin_t" if args.model == "swin" else "resnet50"

        model = TorchvisionModel(
            dataset=task_dataset, 
            model_name=model_name,
            model_config={"weights": "DEFAULT"}
        )

    # 4. Execute PyHealth Trainer
    # Save to a folder outside the PyHealth repo
    output_path = os.path.expanduser(f"~/dermoscopy_outputs/{args.model}_{args.mode}/")
    
    trainer = Trainer(model=model, metrics=["roc_auc", "accuracy", "pr_auc"], output_path=output_path)
    
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=args.epochs,
        monitor="roc_auc"
    )

    # Use trainer.exp_path to get the exact timestamped folder PyHealth just generated!
    plot_training_progress(trainer.exp_path, args.model, args.mode)

if __name__ == "__main__":
    main()