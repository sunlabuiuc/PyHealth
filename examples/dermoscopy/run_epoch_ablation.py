# Contributor: [Your Name]
# NetID: [Your NetID]

"""
Learning Dynamics and Epoch Ablation Study.

Automates the training process to periodically halt and evaluate model robustness 
on an out-of-domain Trap Set at varying epochs. Prevents data leakage by utilizing
a custom PyTorch loop, strictly adhering to PyHealth 2.0 object schemas.

Paper: "A Study of Artifacts on Melanoma Classification under Diffusion-Based Perturbations" (CHIL 2025)
"""

import argparse
import os
import torch
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
    parser.add_argument('--model', type=str, choices=['resnet50', 'swin', 'dinov2'], required=True)
    parser.add_argument('--mode', type=str, choices=['whole', 'high_whole', 'low_whole'], required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--log_dir', type=str, default=None, help="Parent log directory to save session output logs (defaults to dermoscopy_logs in home directory)")
    parser.add_argument('--train_datasets', nargs='+', default=['isic2018'])
    parser.add_argument('--eval_dataset', type=str, default='ph2')
    parser.add_argument('--artifact', type=str, default=None, help="Optional diffusion artifact to test against.")
    parser.add_argument('--epochs', type=int, default=20)
    args = parser.parse_args()

    if args.artifact:
        dataset_target = f"{args.eval_dataset}_with_{args.artifact}"
    else:
        dataset_target = args.eval_dataset

    run_details = f"{args.model}_{args.mode}_{dataset_target}"

    # Strip PyHealth's redundant default console handlers so only custom logger is used for the session logs
    logging.getLogger("pyhealth").handlers.clear()
    setup_dynamic_logging(args.log_dir, "ablation_test", run_details)

    processor = DermoscopyImageProcessor(mode=args.mode)

    print("[*] Loading Dataset (Train and Trap Set)...")
    all_datasets = args.train_datasets + [dataset_target]
    dynamic_dataset_name = "_".join(all_datasets)

    dataset = DermoscopyDataset(
        root=args.data_dir, 
        datasets=all_datasets, 
        dataset_name=dynamic_dataset_name,
        cache_dir=os.path.join(args.data_dir, ".cache")
    )

    # Filter for Train
    train_task = DermoscopyMelanomaClassification(source_datasets=args.train_datasets)
    train_task_dataset = dataset.set_task(train_task, input_processors={"image": processor})
    train_ds, val_ds, _ = split_by_sample(train_task_dataset, [0.9, 0.1, 0.0]) 
    train_loader = get_dataloader(train_ds, batch_size=32, shuffle=True)

    # Filter for Eval
    dataset.task = None 
    eval_task = DermoscopyMelanomaClassification(source_datasets=[dataset_target])
    trap_task_dataset = dataset.set_task(eval_task, input_processors={"image": processor})
    trap_loader = get_dataloader(trap_task_dataset, batch_size=32, shuffle=False)

    if args.model == "dinov2": model = DINOv2(dataset=dataset, feature_keys=["image"], label_key="melanoma", mode="binary")
    else:  model = TorchvisionModel(dataset=dataset, model_name=args.model, model_config={"weights": "DEFAULT"})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

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
            eval_trainer = Trainer(model=model, metrics=["roc_auc"])
            trap_auc = eval_trainer.evaluate(test_dataloader=trap_loader)["roc_auc"]
            trap_aucs.append(trap_auc)
            print(f"[!] Trap Set ROC-AUC at Epoch {epoch}: {trap_auc:.4f}\n")
            
if __name__ == "__main__":
    main()