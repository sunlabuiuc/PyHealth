# Contributor: [Your Name]
# NetID: [Your NetID]
# Paper Title: A Study of Artifacts on Melanoma Classification under Diffusion-Based Perturbations

"""
Master Training Script for Melanoma Classification.

Executes 5-Fold Cross-Validation OR Single Master Model training. Handles 
out-of-domain transfer learning evaluation and automatically plots training curves.
"""

import argparse
import os
import re
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from sklearn.model_selection import KFold
from torch.utils.data import Subset
from collections import defaultdict
import matplotlib.pyplot as plt

from pyhealth.datasets import get_dataloader, split_by_sample
from pyhealth.trainer import Trainer
from pyhealth.models.base_model import BaseModel

# Native PyHealth Dermoscopy Imports
from pyhealth.datasets import DermoscopyDataset
from pyhealth.tasks import DermoscopyMelanomaClassification
from pyhealth.processors import DermoscopyImageProcessor
from pyhealth.models import DINOv2

class MelanomaClassifier(BaseModel):
    """Wraps ResNet50 and Swin Transformer into PyHealth BaseModel format."""
    def __init__(self, dataset, feature_keys, label_key, mode, arch="resnet50", **kwargs):
        super().__init__(dataset=dataset)
        self.feature_keys = feature_keys
        self.feature_key = feature_keys[0]
        self.label_key = label_key
        self.mode = mode

        if arch == "resnet50":
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            hidden_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif arch == "swin":
            self.backbone = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
            hidden_dim = self.backbone.head.in_features
            self.backbone.head = nn.Identity()
        else:
            raise ValueError(f"Architecture '{arch}' is not supported here!")

        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, **kwargs):
        x = kwargs[self.feature_key]
        if isinstance(x, (list, tuple)):
            x = torch.stack(x, dim=0)
        x = x.to(self.device)

        features = self.backbone(x)
        logits = self.classifier(features)
        
        y_prob = torch.sigmoid(logits).squeeze(-1)
        res = {"logit": logits, "y_prob": y_prob}
        
        if self.label_key in kwargs:
            y_true = kwargs[self.label_key].to(self.device)
            res["y_true"] = y_true.squeeze(-1)
            
            y_true_float = y_true.float()
            if y_true_float.dim() == 1:
                y_true_float = y_true_float.unsqueeze(1)
            
            loss = nn.BCEWithLogitsLoss()(logits, y_true_float)
            res["loss"] = loss

        return res

def plot_training_curves(base_out_dir, cv_folds):
    """Parses PyHealth logs to visualize Validation ROC-AUC across all folds."""
    plt.figure(figsize=(10, 6))
    
    # Handle Master Model visualization
    folds_to_plot = ["master"] if cv_folds == 1 else [f"fold_{i}" for i in range(cv_folds)]
    
    for fold_name in folds_to_plot:
        log_file = os.path.join(base_out_dir, fold_name, "log.txt")
        if not os.path.exists(log_file):
            continue
            
        epochs, aucs = [], []
        with open(log_file, 'r') as f:
            for line in f:
                if "roc_auc:" in line and "epoch-" in line:
                    epoch_match = re.search(r"epoch-(\d+)", line)
                    auc_match = re.search(r"roc_auc:\s*([\d\.]+)", line)
                    if epoch_match and auc_match:
                        epochs.append(int(epoch_match.group(1)))
                        aucs.append(float(auc_match.group(1)))
        if epochs:
            label = "Master Model" if cv_folds == 1 else f"Fold {fold_name.split('_')[1]}"
            plt.plot(epochs, aucs, marker='o', label=label)
            
    plt.title(f"Validation ROC-AUC over Epochs", fontsize=14)
    plt.xlabel("Training Epoch", fontsize=12)
    plt.ylabel("Validation ROC-AUC", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plot_path = os.path.join(base_out_dir, "training_curves.png")
    plt.savefig(plot_path, dpi=300)
    print(f"\n[*] Training visualization saved to {plot_path}")

def main():
    parser = argparse.ArgumentParser(description="Ultimate Melanoma Training Pipeline.")
    parser.add_argument('--model', type=str, choices=['resnet50', 'swin', 'dinov2'], default='resnet50')
    parser.add_argument('--mode', type=str, default='whole')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--train_datasets', nargs='+', default=['isic2018'], help="Source datasets")
    parser.add_argument('--test_datasets', nargs='*', default=[], help="Out-of-domain datasets for Tables 1 & 2")
    parser.add_argument('--cv_folds', type=int, default=5, help="Number of folds (Set to 1 for Master Model)")
    args = parser.parse_args()

    processor = DermoscopyImageProcessor(mode=args.mode)
    DermoscopyMelanomaClassification.task_name = "melanoma_classification"
    DermoscopyMelanomaClassification.input_schema = {"image": "image"}
    DermoscopyMelanomaClassification.output_schema = {} 

    print(f"[*] Initializing Training Dataset(s): {args.train_datasets} in {args.mode} mode...")
    dataset_name = args.train_datasets[0] if len(args.train_datasets) == 1 else args.train_datasets
    train_dataset = DermoscopyDataset(root=args.data_dir, dataset_name=dataset_name, dev=False)
    task_dataset = train_dataset.set_task(DermoscopyMelanomaClassification, input_processors={"image": processor})
    
    # Prepare Out-of-Domain Test Loaders
    test_loaders = {}
    if args.test_datasets:
        print(f"[*] Initializing Out-of-Domain Test Datasets: {args.test_datasets}")
        for td in args.test_datasets:
            td_ds = DermoscopyDataset(root=args.data_dir, dataset_name=td, dev=False)
            td_task = td_ds.set_task(DermoscopyMelanomaClassification, input_processors={"image": processor})
            test_loaders[td] = get_dataloader(td_task, batch_size=32, shuffle=False)
            
    train_prefix = "_".join(args.train_datasets) if isinstance(args.train_datasets, list) else args.train_datasets
    base_out_dir = os.path.expanduser(f"~/dermoscopy_outputs/{train_prefix}_{args.model}_{args.mode}")
    os.makedirs(base_out_dir, exist_ok=True)

    ood_results = defaultdict(lambda: defaultdict(list))
    
    # ==========================================
    # STRATEGY 1: MASTER MODEL (cv_folds == 1)
    # ==========================================
    if args.cv_folds == 1:
        print(f"\n" + "="*50 + f"\n PHASE 1: MASTER MODEL TRAINING\n" + "="*50)
        
        # 90/10 Split for Master Model
        train_ds, val_ds, _ = split_by_sample(task_dataset, [0.9, 0.1, 0.0])
        train_loader = get_dataloader(train_ds, batch_size=32, shuffle=True)
        val_loader = get_dataloader(val_ds, batch_size=32, shuffle=False)
        
        if args.model == "dinov2":
            model = DINOv2(dataset=train_dataset, feature_keys=["image"], label_key="melanoma", mode="binary")
        else:
            model = MelanomaClassifier(dataset=train_dataset, feature_keys=["image"], label_key="melanoma", mode="binary", arch=args.model)

        output_path = os.path.join(base_out_dir, "master")
        trainer = Trainer(model=model, metrics=["roc_auc", "accuracy", "pr_auc", "f1"], output_path=output_path)
        
        trainer.train(train_dataloader=train_loader, val_dataloader=val_loader, epochs=args.epochs, monitor="roc_auc")
        
        if test_loaders:
            print(f"\n[*] Evaluating Master Model on Out-Of-Domain Data...")
            best_model_path = os.path.join(output_path, "best_model.pth")
            state_dict = torch.load(best_model_path, map_location=trainer.device, weights_only=True)
            model.load_state_dict(state_dict['model'] if 'model' in state_dict else state_dict)
            model.eval()

            eval_trainer = Trainer(model=model, metrics=["roc_auc", "accuracy", "pr_auc", "f1"])
            for test_name, test_loader in test_loaders.items():
                res = eval_trainer.evaluate(test_dataloader=test_loader)
                ood_results[test_name]['roc_auc'].append(res['roc_auc'])
                ood_results[test_name]['accuracy'].append(res['accuracy'])
                print(f"  -> {test_name.upper()} | ROC-AUC: {res['roc_auc']:.4f} | ACC: {res['accuracy']:.4f}")

    # ==========================================
    # STRATEGY 2: 5-FOLD CV (cv_folds > 1)
    # ==========================================
    else:
        print(f"\n" + "="*50 + f"\n PHASE 1: {args.cv_folds}-Fold CV (For Paper Tables)\n" + "="*50)
        kf = KFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(task_dataset)))):
            print(f"\n--- STARTING FOLD {fold} ---")
            train_loader = get_dataloader(Subset(task_dataset, train_idx), batch_size=32, shuffle=True)
            val_loader = get_dataloader(Subset(task_dataset, val_idx), batch_size=32, shuffle=False)

            if args.model == "dinov2":
                model = DINOv2(dataset=train_dataset, feature_keys=["image"], label_key="melanoma", mode="binary")
            else:
                model = MelanomaClassifier(dataset=train_dataset, feature_keys=["image"], label_key="melanoma", mode="binary", arch=args.model)

            output_path = os.path.join(base_out_dir, f"fold_{fold}")
            trainer = Trainer(model=model, metrics=["roc_auc", "accuracy", "pr_auc", "f1"], output_path=output_path)
            
            trainer.train(train_dataloader=train_loader, val_dataloader=val_loader, epochs=args.epochs, monitor="roc_auc")
            
            if test_loaders:
                print(f"\n[*] Evaluating Fold {fold} on Out-Of-Domain Data...")
                best_model_path = os.path.join(output_path, "best_model.pth")
                state_dict = torch.load(best_model_path, map_location=trainer.device, weights_only=True)
                model.load_state_dict(state_dict['model'] if 'model' in state_dict else state_dict)
                model.eval()

                eval_trainer = Trainer(model=model, metrics=["roc_auc", "accuracy", "pr_auc", "f1"])
                for test_name, test_loader in test_loaders.items():
                    res = eval_trainer.evaluate(test_dataloader=test_loader)
                    ood_results[test_name]['roc_auc'].append(res['roc_auc'])
                    ood_results[test_name]['accuracy'].append(res['accuracy'])
                    print(f"  -> {test_name.upper()} | ROC-AUC: {res['roc_auc']:.4f} | ACC: {res['accuracy']:.4f}")

    # Generate Visualization
    plot_training_curves(base_out_dir, args.cv_folds)

    # Print Final Paper Results
    if test_loaders:
        print("\n" + "="*50)
        print(" FINAL OOD RESULTS")
        print("="*50)
        for test_name, metrics in ood_results.items():
            mean_auc = np.mean(metrics['roc_auc'])
            std_auc = np.std(metrics['roc_auc']) if args.cv_folds > 1 else 0.0
            mean_acc = np.mean(metrics['accuracy'])
            print(f" {test_name.upper()}:")
            print(f"   ROC-AUC:  {mean_auc:.4f} ± {std_auc:.4f}")
            print(f"   Accuracy: {mean_acc:.4f}")
        print("="*50)
        
    print(f"[SUCCESS] Training complete! Weights saved to: {base_out_dir}")

if __name__ == "__main__":
    main()