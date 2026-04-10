# Contributor: [Your Name]
# NetID: [Your NetID]

"""
Epoch Variation Ablation Study.

Automates the training process to periodically halt and evaluate model robustness 
on the PH2 Trap Set at varying epochs. This methodology reveals how heavily models 
rely on spurious clinical artifacts over the course of the training lifecycle.

Paper: "A Study of Artifacts on Melanoma Classification under Diffusion-Based Perturbations" (CHIL 2025)
"""

import argparse
import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from pyhealth.datasets import split_by_patient, get_dataloader
from pyhealth.trainer import Trainer

# Native PyHealth Dermoscopy Imports
from pyhealth.datasets import DermoscopyDataset
from pyhealth.tasks import DermoscopyMelanomaClassification
from pyhealth.processors import DermoscopyImageProcessor
from pyhealth.models import TorchvisionModel, DINOv2

def main():
    """Executes the automated epoch ablation pipeline and plots the final trend line."""
    parser = argparse.ArgumentParser(description="Run the single-pass epoch ablation study.")
    parser.add_argument('--model', type=str, choices=['resnet50', 'swin', 'dinov2'], required=True)
    parser.add_argument('--mode', type=str, choices=['whole', 'high_whole', 'low_whole'], required=True)
    parser.add_argument('--data_dir', type=str, required=True, help="Root directory containing datasets.")
    parser.add_argument('--artifact', type=str, choices=['patches', 'dark_corner', 'ruler', 'ink', 'gel_bubble'], required=True)
    args = parser.parse_args()

    print("="*60)
    print(f"AUTOMATED EPOCH ABLATION: {args.model.upper()} ({args.mode}) on {args.artifact.upper()}")
    print("="*60)

    processor = DermoscopyImageProcessor(mode=args.mode)

    # 1. Load the clean Training Data (ISIC/HAM10000)
    print("[*] Loading Training Data...")
    train_dataset = DermoscopyDataset(root=args.data_dir, dataset_name="isic2018", dev=False)
    train_task_dataset = train_dataset.set_task(task=DermoscopyMelanomaClassification, input_processors={"image": processor})
    # Do not create a test set, to maximize data usage for the study
    train_ds, val_ds, _ = split_by_patient(train_task_dataset, [0.8, 0.2, 0.0]) 
    train_loader = get_dataloader(train_ds, batch_size=32, shuffle=True)

    # 2. Load the out-of-distribution Trap Set for periodic evaluation
    print(f"[*] Loading Trap Set ({args.artifact})...")
    trap_dataset = DermoscopyDataset(root=args.data_dir, dataset_name=f"ph2_with_{args.artifact}", dev=False)
    trap_task_dataset = trap_dataset.set_task(task=DermoscopyMelanomaClassification, input_processors={"image": processor})
    trap_loader = get_dataloader(trap_task_dataset, batch_size=32, shuffle=False)

    # 3. Model Initialization
    if args.model == "dinov2":
        model = DINOv2(dataset=train_task_dataset, model_size="vits14")
    else:
        model_name = "swin_t" if args.model == "swin" else "resnet50"

        model = TorchvisionModel(
            dataset=train_task_dataset, 
            model_name=model_name,
            model_config={"weights": "DEFAULT"}
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # 4. Single-Pass Automation Loop
    epochs_to_test = [3, 5, 10, 20]
    trap_aucs = []

    for epoch in range(1, max(epochs_to_test) + 1):
        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{max(epochs_to_test)}"):
            optimizer.zero_grad()
            # PyHealth models handle the forward pass and loss generation internally
            outputs = model(**batch) 
            loss = outputs["loss"]
            loss.backward()
            optimizer.step()

        # Halt training and evaluate against the Trap Set only at specified epochs
        if epoch in epochs_to_test:
            print(f"\n[*] Target Epoch {epoch} Reached. Evaluating on Trap Set...")
            
            # Utilize native Trainer strictly for evaluation
            eval_trainer = Trainer(model=model, metrics=["roc_auc"])
            results = eval_trainer.evaluate(test_dataloader=trap_loader)
            
            trap_auc = results["roc_auc"]
            trap_aucs.append(trap_auc)
            print(f"[!] Trap Set ROC-AUC at Epoch {epoch}: {trap_auc:.4f}\n")
            
            out_path = f"./output/ablation_{args.model}_{args.mode}"
            os.makedirs(out_path, exist_ok=True)
            torch.save(model.state_dict(), f"{out_path}/epoch_{epoch}.pth")

    # 5. Plot the Final Ablation Trend Graph
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_to_test, trap_aucs, marker='o', color='purple', linewidth=2, markersize=8)
    plt.title(f'Artifact Reliance over Time: {args.model.upper()} ({args.mode}) - {args.artifact.upper()}')
    plt.xlabel('Training Epochs')
    plt.ylabel('Trap Set ROC-AUC (Robustness)')
    plt.xticks(epochs_to_test)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plot_path = f"./output/ablation_curve_{args.model}_{args.mode}_{args.artifact}.png"
    plt.savefig(plot_path, dpi=300)
    print(f"\n[*] Ablation complete! Trend graph saved to {plot_path}")

if __name__ == "__main__":
    main()