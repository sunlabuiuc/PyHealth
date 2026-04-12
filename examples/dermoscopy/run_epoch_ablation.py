# Contributor: [Your Name]
# NetID: [Your NetID]

"""
Learning Dynamics and Epoch Ablation Study.

Automates the training process to periodically halt and evaluate model robustness 
on an out-of-domain Trap Set at varying epochs. 
"""

import argparse
import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from pyhealth.datasets import split_by_sample, get_dataloader
from pyhealth.trainer import Trainer

# Native PyHealth Dermoscopy Imports
from pyhealth.datasets import DermoscopyDataset
from pyhealth.tasks import DermoscopyMelanomaClassification
from pyhealth.processors import DermoscopyImageProcessor
from pyhealth.models import TorchvisionModel, DINOv2

def main():
    parser = argparse.ArgumentParser(description="Run the single-pass epoch ablation study.")
    parser.add_argument('--model', type=str, choices=['resnet50', 'swin', 'dinov2'], required=True)
    parser.add_argument('--mode', type=str, choices=['whole', 'high_whole', 'low_whole'], required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--eval_dataset', type=str, default='ph2', help="Base dataset to evaluate on")
    parser.add_argument('--artifact', type=str, choices=['patches', 'dark_corner', 'ruler', 'ink', 'gel_bubble'], required=True)
    parser.add_argument('--epochs', type=int, default=20)
    args = parser.parse_args()

    print("="*60)
    print(f"AUTOMATED EPOCH ABLATION: {args.model.upper()} ({args.mode}) on {args.eval_dataset.upper()}_{args.artifact.upper()}")
    print("="*60)

    processor = DermoscopyImageProcessor(mode=args.mode)

    # 1. Load the clean Training Data (ISIC2018)
    print("[*] Loading Training Data...")
    train_dataset = DermoscopyDataset(root=args.data_dir, dataset_name="isic2018", dev=False)
    train_task_dataset = train_dataset.set_task(task=DermoscopyMelanomaClassification, input_processors={"image": processor})
    
    train_ds, val_ds, _ = split_by_sample(train_task_dataset, [0.9, 0.1, 0.0]) 
    train_loader = get_dataloader(train_ds, batch_size=32, shuffle=True)

    # 2. Load the out-of-distribution Trap Set dynamically
    dataset_target = f"{args.eval_dataset}_with_{args.artifact}"
    print(f"[*] Loading Trap Set ({dataset_target})...")
    trap_dataset = DermoscopyDataset(root=args.data_dir, dataset_name=dataset_target, dev=False)
    trap_task_dataset = trap_dataset.set_task(task=DermoscopyMelanomaClassification, input_processors={"image": processor})
    trap_loader = get_dataloader(trap_task_dataset, batch_size=32, shuffle=False)

    # 3. Model Initialization
    print(f"[*] Initializing {args.model.upper()} Architecture...")
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
    epochs_to_test = [3, 5, 10, 15, 20]
    trap_aucs = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            optimizer.zero_grad()
            outputs = model(**batch) 
            loss = outputs["loss"]
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        if epoch in epochs_to_test:
            print(f"\n[*] Target Epoch {epoch} Reached. Evaluating on {dataset_target}...")
            eval_trainer = Trainer(model=model, metrics=["roc_auc"])
            results = eval_trainer.evaluate(test_dataloader=trap_loader)
            
            trap_auc = results["roc_auc"]
            trap_aucs.append(trap_auc)
            print(f"[!] Trap Set ROC-AUC at Epoch {epoch}: {trap_auc:.4f}\n")
            
            out_path = os.path.expanduser(f"~/dermoscopy_outputs/ablation_{args.model}_{args.mode}_{args.eval_dataset}_{args.artifact}")
            os.makedirs(out_path, exist_ok=True)
            torch.save(model.state_dict(), f"{out_path}/epoch_{epoch}.pth")

    # 5. Plot the Final Ablation Trend Graph
    if trap_aucs:
        plt.figure(figsize=(10, 6))
        plt.plot(epochs_to_test[:len(trap_aucs)], trap_aucs, marker='o', color='purple', linewidth=2, markersize=8)
        plt.title(f'Artifact Reliance over Time: {args.model.upper()} ({args.mode}) vs {dataset_target.upper()}', fontsize=14)
        plt.xlabel('Training Epochs', fontsize=12)
        plt.ylabel('Trap Set ROC-AUC (Robustness)', fontsize=12)
        plt.xticks(epochs_to_test)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        for i, auc in enumerate(trap_aucs):
            plt.annotate(f"{auc:.3f}", (epochs_to_test[i], trap_aucs[i]), textcoords="offset points", xytext=(0,10), ha='center')

        plot_path = os.path.join(out_path, "epoch_ablation_curve.png")
        plt.savefig(plot_path, dpi=300)
        print(f"\n[*] Ablation complete! Trend graph saved to {plot_path}")

if __name__ == "__main__":
    main()