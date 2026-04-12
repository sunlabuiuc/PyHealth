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
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from torch.utils.data import Subset

from pyhealth.datasets import get_dataloader, split_by_patient
from pyhealth.trainer import Trainer
from pyhealth.datasets import DermoscopyDataset
from pyhealth.tasks import DermoscopyMelanomaClassification
from pyhealth.processors import DermoscopyImageProcessor
from pyhealth.models import TorchvisionModel, DINOv2

def main():
    parser = argparse.ArgumentParser(description="Ultimate Melanoma Training Pipeline.")
    parser.add_argument('--model', type=str, choices=['resnet50', 'swin', 'dinov2'], default='resnet50')
    parser.add_argument('--mode', type=str, choices=['whole', 'high_whole', 'low_whole'], default='whole')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--train_datasets', nargs='+', default=['isic2018'], help="Source datasets")
    parser.add_argument('--test_datasets', nargs='+', default=['ham10000', 'ph2'], help="Transfer datasets")
    parser.add_argument('--cv_folds', type=int, default=5)
    args = parser.parse_args()

    processor = DermoscopyImageProcessor(mode=args.mode)

    print(f"[*] Initializing Train Dataset(s): {args.train_datasets}")
    train_ds = DermoscopyDataset(root=args.data_dir, datasets=args.train_datasets, dev=False)
    train_task = train_ds.set_task(task=DermoscopyMelanomaClassification, input_processors={"image": processor})

    # Create a unified parent directory for this specific experiment combination
    train_prefix = "_".join(args.train_datasets)
    base_out_dir = os.path.expanduser(f"~/dermoscopy_outputs/{train_prefix}_{args.model}_{args.mode}")
    os.makedirs(base_out_dir, exist_ok=True)

    # ==========================================
    # PHASE 1: 5-Fold CV (For Paper Tables)
    # ==========================================
    kf = KFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(train_task)))):
        print(f"\n" + "="*50 + f"\n STARTING FOLD {fold + 1}/{args.cv_folds}\n" + "="*50)
        
        train_loader = get_dataloader(Subset(train_task, train_idx), batch_size=32, shuffle=True)
        val_loader = get_dataloader(Subset(train_task, val_idx), batch_size=32, shuffle=False)

        if args.model == "dinov2":
            model = DINOv2(dataset=train_task, model_size="vits14")
        else:
            model_name = "swin_t" if args.model == "swin" else "resnet50"
            model = TorchvisionModel(dataset=train_task, model_name=model_name, model_config={"weights": "DEFAULT"})

        output_path = os.path.join(base_out_dir, f"fold_{fold}")
        trainer = Trainer(model=model, metrics=["roc_auc", "accuracy"], output_path=output_path)
        trainer.train(train_dataloader=train_loader, val_dataloader=val_loader, epochs=args.epochs, monitor="roc_auc")

    # ==========================================
    # PHASE 2: The Master Model (100% Data)
    # ==========================================
    print(f"\n" + "="*50 + "\n TRAINING MASTER MODEL (100% DATA)\n" + "="*50)
    
    # We do a tiny 95/5 split just so the Trainer has a validation set to monitor for early stopping
    master_train, master_val, _ = split_by_patient(train_task, [0.95, 0.05, 0.0])
    master_train_loader = get_dataloader(master_train, batch_size=32, shuffle=True)
    master_val_loader = get_dataloader(master_val, batch_size=32, shuffle=False)

    if args.model == "dinov2":
        master_model = DINOv2(dataset=train_task, model_size="vits14")
    else:
        master_model = TorchvisionModel(dataset=train_task, model_name=model_name, model_config={"weights": "DEFAULT"})

    master_out_path = os.path.join(base_out_dir, "master")
    master_trainer = Trainer(model=master_model, metrics=["roc_auc", "accuracy"], output_path=master_out_path)
    master_trainer.train(train_dataloader=master_train_loader, val_dataloader=master_val_loader, epochs=args.epochs, monitor="roc_auc")
    
    print(f"\n[SUCCESS] Experiment saved to: {base_out_dir}")

if __name__ == "__main__":
    main()