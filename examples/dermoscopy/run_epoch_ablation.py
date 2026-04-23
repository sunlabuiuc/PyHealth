"""
Learning Dynamics and Epoch Ablation Study.

Automates the training process to periodically halt and evaluate model robustness 
on an out-of-domain Trap Set at varying epochs. Prevents data leakage by utilizing
a custom PyTorch loop, strictly adhering to PyHealth 2.0 object schemas.

Author:
    Mumme, Raymond Paul rmumme2@illinois.edu
"""

import argparse
import os
import torch
from pathlib import Path
import logging
from tqdm import tqdm

from pyhealth.datasets import split_by_sample, get_dataloader
from pyhealth.trainer import Trainer

from pyhealth.datasets import DermoscopyDataset
from pyhealth.tasks import DermoscopyMelanomaClassification
from pyhealth.processors import DermoscopyImageProcessor
from pyhealth.models import DINOv2, TorchvisionModel
from train_dermoscopy import setup_dynamic_logging

def main():
    parser = argparse.ArgumentParser(description="Epoch Ablation")
    parser.add_argument('--model', type=str, choices=['resnet50', 'swin_t', 'dinov2', 'convnext_tiny'], required=True)
    parser.add_argument('--mode', type=str, choices=['whole', 'high_whole', 'low_whole'], required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--log_dir', type=str, default=None, help="Parent log directory to save session output logs")
    parser.add_argument('--train_datasets', nargs='+', default=['isic2018'])
    parser.add_argument('--eval_dataset', type=str, default='ph2')
    parser.add_argument('--artifact', type=str, default=None, help="Optional diffusion artifact to test against.")
    parser.add_argument('--epochs', type=int, default=20)
    args = parser.parse_args()

    # Safely handle the clean baseline to prevent loading "ph2_with_clean"
    if args.artifact is None or args.artifact.lower() in ["none", "clean"]:
        args.artifact = "clean"
        dataset_target = args.eval_dataset
    else:
        dataset_target = f"{args.eval_dataset}_with_{args.artifact}"

    run_details = f"{args.model}_{args.mode}_{dataset_target}"

    # Strip PyHealth's redundant default console handlers so only custom logger is used for the session logs
    logging.getLogger("pyhealth").handlers.clear()
    logfilepath = setup_dynamic_logging(args.log_dir, "ablation_test", run_details)

    print("---------------------------------------------------------------------")
    print(f"[*] Executing: Model({args.model}) | Eval({args.eval_dataset} + {args.artifact})")
    print("---------------------------------------------------------------------")

    processor = DermoscopyImageProcessor(mode=args.mode)

    print("[*] Loading Train Dataset...")
    # Init Train Base Dataset exactly like train_dermoscopy.py to force a cache hit
    train_base_dataset = DermoscopyDataset(
        root=args.data_dir, 
        datasets=args.train_datasets, 
        cache_dir=os.path.join(args.data_dir, ".cache")
    )
    train_task = DermoscopyMelanomaClassification(source_datasets=args.train_datasets)
    train_task_dataset = train_base_dataset.set_task(train_task, input_processors={"image": processor})
    
    # 90/10 Split to mimic the standard cross-validation fold size
    train_ds, val_ds, _ = split_by_sample(train_task_dataset, [0.9, 0.1, 0.0]) 
    train_loader = get_dataloader(train_ds, batch_size=32, shuffle=True)

    print("[*] Loading Trap Dataset...")
    # Init Trap Base Dataset exactly like evaluate_artifact_robustness.py to force a cache hit
    trap_base_dataset = DermoscopyDataset(
        root=args.data_dir, 
        datasets=[dataset_target], 
        cache_dir=os.path.join(args.data_dir, ".cache")
    )
    eval_task = DermoscopyMelanomaClassification(source_datasets=[dataset_target])
    trap_task_dataset = trap_base_dataset.set_task(eval_task, input_processors={"image": processor})
    trap_loader = get_dataloader(trap_task_dataset, batch_size=32, shuffle=False)

    if args.model == "dinov2": 
        model = DINOv2(dataset=train_task_dataset, feature_keys=["image"], label_key="melanoma", mode="binary")
    else:  
        model = TorchvisionModel(dataset=train_task_dataset, model_name=args.model, model_config={"weights": "DEFAULT"})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Initialize Trainer once outside the loop to prevent spamming output folders
    # Point the native output_path away from the repo to keep it clean
    ablation_out_dir = os.path.dirname(logfilepath)
    
    eval_trainer = Trainer(
        model=model, 
        metrics=["roc_auc"], 
        output_path=ablation_out_dir
    )

    epochs_to_test = [3, 5, 10, 15, 20]
    trap_aucs = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            optimizer.zero_grad()
            loss = model(**batch)["loss"]
            loss.backward()
            optimizer.step()

        if epoch in epochs_to_test:
            print(f"\n[*] Evaluating on {dataset_target}...")
            # Because eval_trainer holds a reference to `model`, it will automatically use the updated weights
            trap_auc = eval_trainer.evaluate(dataloader=trap_loader)["roc_auc"]
            trap_aucs.append(trap_auc)
            print(f"[!] Trap Set ROC-AUC at Epoch {epoch}: {trap_auc:.4f}\n")
            
    # Print final summary table
    print("\n" + "="*50)
    print(" EPOCH ABLATION SUMMARY")
    print("="*50)
    print(f"Model: {args.model.upper()} | Mode: {args.mode}")
    print(f"Train: {'+'.join(args.train_datasets)} | Eval: {dataset_target}")
    print("-" * 50)
    for ep, auc in zip(epochs_to_test, trap_aucs):
        print(f"Epoch {ep:<2}: {auc:.4f}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()