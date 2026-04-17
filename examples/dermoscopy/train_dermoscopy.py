# Contributor: [Your Name]
# NetID: [Your NetID]
# Paper Title: A Study of Artifacts on Melanoma Classification under Diffusion-Based Perturbations

"""
Master Training Script for Melanoma Classification.

Executes 5-Fold Cross-Validation OR Single Master Model training. Handles 
out-of-domain transfer learning evaluation, automatically plots training curves,
and prints formatted Summary Tables.
"""

import argparse
import os
import sys
import logging
import re
import numpy as np
import torch
from pathlib import Path
from sklearn.model_selection import KFold
from torch.utils.data import Subset
from collections import defaultdict
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from pyhealth.datasets import get_dataloader, split_by_sample
from pyhealth.trainer import Trainer

from pyhealth.datasets import DermoscopyDataset
from pyhealth.tasks import DermoscopyMelanomaClassification
from pyhealth.processors import DermoscopyImageProcessor
from pyhealth.models import DINOv2, TorchvisionModel

class PyHealthSubset(Subset):
    """A wrapper for PyTorch's Subset that satisfies PyHealth's DataLoader requirements."""
    def set_shuffle(self, shuffle):
        pass # Dummy method to appease PyHealth

# ==========================================
# LOGGING UTILITIES (Importable by others)
# ==========================================
class DualLogger:
    """Writes standard output to both the terminal and a log file simultaneously."""
    def __init__(self, filepath, stream):
        self.terminal = stream
        self.log = open(filepath, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # This flush method is needed for Python 3 compatibility
        self.terminal.flush()
        self.log.flush()

def setup_dynamic_logging(arg_log_dir, folder_name, run_name):
    """Sets up a catch-all logger for standard prints, errors, tqdm, and PyHealth logs."""
    import datetime

    if arg_log_dir is None:
        log_dir = Path.home() / "dermoscopy_logs"
    else:
        log_dir = Path(arg_log_dir)

    base_log_dir = log_dir / folder_name
    base_log_dir.mkdir(parents=True, exist_ok=True)
    base_log_dir = str(base_log_dir)

    # Build the filename
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filepath = os.path.join(base_log_dir, f"{run_name}_{timestamp}.txt")

    # Hijack Standard Output (print statements)
    sys.stdout = DualLogger(log_filepath, sys.stdout)

    # Hijack Standard Error (tqdm progress bars and crash tracebacks)
    sys.stderr = DualLogger(log_filepath, sys.stderr)

    # Capture internal PyHealth / PyTorch warnings via the logging module
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(log_filepath, encoding="utf-8"),
            logging.StreamHandler(sys.stdout)
        ],
        force=True # This forces PyHealth to abandon its default loggers and use ours
    )

    print(f"[*] Dynamic Logging initialized: {log_filepath}")
    return log_filepath

# ==========================================
# VISUALIZATION UTILITY
# ==========================================
def generate_all_visualizations(base_out_dir, fold_num="Master"):
    """Parses PyHealth logs to automatically generate PNG curves and TensorBoard events."""
    log_path = os.path.join(base_out_dir, "log.txt")
    if not os.path.exists(log_path):
        print(f"[-] No log.txt found in {base_out_dir}")
        return

    train_loss, val_roc = [], []
    
    # 1. Setup TensorBoard Writer
    tb_dir = os.path.join(base_out_dir, f"tb_logs_fold_{fold_num}")
    writer = SummaryWriter(log_dir=tb_dir)

    epoch = 1
    with open(log_path, "r") as f:
        for line in f.readlines():
            # Catch Training Loss
            if "loss:" in line and "roc_auc" not in line:
                match = re.search(r"loss:\s*([0-9\.]+)", line)
                if match: 
                    loss_val = float(match.group(1))
                    train_loss.append(loss_val)
                    writer.add_scalar("Loss/Train", loss_val, epoch)
            
            # Catch Validation Metrics (This marks the end of an epoch)
            if "roc_auc:" in line:
                roc_match = re.search(r"roc_auc:\s*([0-9\.]+)", line)
                pr_match = re.search(r"pr_auc:\s*([0-9\.]+)", line)
                
                if roc_match: 
                    roc_val = float(roc_match.group(1))
                    val_roc.append(roc_val)
                    writer.add_scalar("Metrics/Val_ROC_AUC", roc_val, epoch)
                if pr_match:
                    writer.add_scalar("Metrics/Val_PR_AUC", float(pr_match.group(1)), epoch)
                
                epoch += 1
    
    writer.close()

    # 2. Generate standard PNG Learning Curves
    if not train_loss:
        return
        
    epochs_x = range(1, len(train_loss) + 1)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_x, train_loss, 'b-', label='Train Loss')
    plt.title(f'Loss Curve - Fold {fold_num}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    if val_roc:
        val_epochs_x = range(1, len(val_roc) + 1)
        plt.subplot(1, 2, 2)
        plt.plot(val_epochs_x, val_roc, 'g-', label='Val ROC-AUC')
        plt.title(f'ROC-AUC Curve - Fold {fold_num}')
        plt.xlabel('Epochs')
        plt.ylabel('Score')
        plt.legend()
        
    plt.tight_layout()
    curve_path = os.path.join(base_out_dir, f"learning_curve_fold_{fold_num}.png")
    plt.savefig(curve_path)
    plt.close()
    
    print(f"[*] Visualizations Complete! PNGs and TensorBoard logs saved to {base_out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Dermoscopy Classification Models")
    parser.add_argument('--data_dir', type=str, required=True, help="Absolute path to the dataset root folder")
    parser.add_argument('--out_dir', type=str, default=None, help="Parent output directory to save experiment folders (defaults to dermoscopy_outputs in home directory)")
    parser.add_argument('--log_dir', type=str, default=None, help="Parent log directory to save session output logs (defaults to dermoscopy_logs in home directory)")
    parser.add_argument('--train_datasets', nargs='+', default=["isic2018"])
    parser.add_argument('--test_datasets', nargs='*', default=[])
    parser.add_argument('--model', type=str, choices=['resnet50', 'swin_t', 'dinov2', 'convnext_tiny'], required=True)
    parser.add_argument('--mode', type=str, default="whole")
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--cv_folds', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-4)
    # force_retrain flag defaults to False unless called
    parser.add_argument('--force_retrain', action='store_true', help="Overwrite existing weights and retrain from scratch.")
    args = parser.parse_args()

    ds_str = "_".join(args.train_datasets)
    run_details = f"{ds_str}_{args.model}_{args.mode}"
    
    if args.out_dir is None:
        out_dir = Path.home() / "dermoscopy_outputs"
    else:
        out_dir = Path(args.out_dir)
    
    base_out_dir = out_dir / run_details
    base_out_dir.mkdir(parents=True, exist_ok=True)
    base_out_dir = str(base_out_dir)

    # START DYNAMIC LOGGING
    # Strip PyHealth's redundant default console handlers so only custom logger is used for the session logs
    logging.getLogger("pyhealth").handlers.clear()
    setup_dynamic_logging(args.log_dir, "train", run_details)

    print(f"[*] Initializing Dataset from {args.data_dir}...")

    dataset = DermoscopyDataset(root=args.data_dir, cache_dir=os.path.join(args.data_dir, ".cache"))

    processor = DermoscopyImageProcessor(mode=args.mode)

    # Setup Tasks
    train_task = DermoscopyMelanomaClassification(source_datasets=args.train_datasets)
    task_dataset = dataset.set_task(train_task, input_processors={"image": processor})

    val_loaders = {}
    for td in args.test_datasets:
        test_task = DermoscopyMelanomaClassification(source_datasets=[td])
        td_dataset = dataset.set_task(test_task, input_processors={"image": processor})
        val_loaders[td] = get_dataloader(td_dataset, batch_size=32, shuffle=False)

    metrics_list = ["roc_auc", "pr_auc", "accuracy", "precision", "recall", "f1"]
    source_results = defaultdict(list)
    ood_results = {td: defaultdict(list) for td in args.test_datasets}

    print(f"\n" + "="*50 + f"\n PHASE 1: {args.cv_folds}-Fold CV\n" + "="*50)
    
    if args.cv_folds == 1:
        train_ds, val_ds, _ = split_by_sample(task_dataset, [0.9, 0.1, 0.0])
        train_loader = get_dataloader(train_ds, batch_size=32, shuffle=True)
        val_loader = get_dataloader(val_ds, batch_size=32, shuffle=False)
        
        if args.model == "dinov2": model = DINOv2(dataset=task_dataset, feature_keys=["image"], label_key="melanoma", mode="binary")
        else: model = TorchvisionModel(dataset=task_dataset, model_name=args.model, model_config={"weights": "DEFAULT"})

        weight_path = os.path.join(base_out_dir, "master", "best.ckpt")

        trainer = Trainer(model=model, output_path=base_out_dir, exp_name="master", metrics=metrics_list)
        # Check for weights AND make sure the user didn't flag --force_retrain
        if os.path.exists(weight_path) and not args.force_retrain:
            print(f"[*] Found existing weights at {weight_path}. Skipping training!")
            state_dict = torch.load(weight_path, map_location=trainer.device, weights_only=True)
            
            # Safely extract the weights whether it's nested under 'state_dict', 'model', or unnested
            if 'state_dict' in state_dict:
                model.load_state_dict(state_dict['state_dict'])
            elif 'model' in state_dict:
                model.load_state_dict(state_dict['model'])
            else:
                model.load_state_dict(state_dict)
        else:
            print(f"[*] No existing weights found. Training Master from scratch...")
            trainer.train(
                train_dataloader=train_loader, 
                val_dataloader=val_loader, 
                epochs=args.epochs, 
                monitor="roc_auc",
                optimizer_params={"lr": args.lr}
            )

        output_path = os.path.join(base_out_dir, "master")
        generate_all_visualizations(output_path, fold_num="Master")

    else:
        kf = KFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
        for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(task_dataset)))):
            print(f"\n--- STARTING FOLD {fold+1}/{args.cv_folds} ---")

            # Convert numpy.int64 arrays to pure Python integer lists
            train_idx = train_idx.tolist()
            val_idx = val_idx.tolist()

            # Wrap them in our PyHealth-friendly subset
            train_ds = PyHealthSubset(task_dataset, train_idx)
            val_ds = PyHealthSubset(task_dataset, val_idx)

            # Generate the loaders
            train_loader = get_dataloader(train_ds, batch_size=32, shuffle=True)
            val_loader = get_dataloader(val_ds, batch_size=32, shuffle=False)

            if args.model == "dinov2": model = DINOv2(dataset=task_dataset, feature_keys=["image"], label_key="melanoma", mode="binary")
            else:  model = TorchvisionModel(dataset=task_dataset, model_name=args.model, model_config={"weights": "DEFAULT"})

            weight_path = os.path.join(base_out_dir, f"fold_{fold}", "best.ckpt")
            
            # avoid logging echoes
            logging.getLogger("pyhealth").handlers.clear()
            logging.getLogger().handlers.clear()

            trainer = Trainer(model=model, output_path=base_out_dir, exp_name=f"fold_{fold}", metrics=metrics_list)
            # Check for weights AND make sure the user didn't flag --force_retrain
            if os.path.exists(weight_path) and not args.force_retrain:
                print(f"[*] Found existing weights at {weight_path}. Skipping training!")
                state_dict = torch.load(weight_path, map_location=trainer.device, weights_only=True)
                
                # Safely extract the weights whether it's nested under 'state_dict', 'model', or unnested
                if 'state_dict' in state_dict:
                    model.load_state_dict(state_dict['state_dict'])
                elif 'model' in state_dict:
                    model.load_state_dict(state_dict['model'])
                else:
                    model.load_state_dict(state_dict)
            else:
                print(f"[*] No existing weights found. Training Fold {fold + 1} from scratch...")
                trainer.train(
                    train_dataloader=train_loader, 
                    val_dataloader=val_loader, 
                    epochs=args.epochs, 
                    monitor="roc_auc",
                    optimizer_params={"lr": args.lr}
                )
            
            # Post-training Visualization
            output_path = os.path.join(base_out_dir, f"fold_{fold}")
            generate_all_visualizations(output_path, fold_num=fold)
            
            # Internal Source Evaluation (For Table 1/2 Source Column)
            val_res = trainer.evaluate(dataloader=val_loader)
            for m in metrics_list: source_results[m].append(val_res[m])

            # OOD Evaluation
            if val_loaders:
                best_model_path = os.path.join(output_path, "best.ckpt")
                state_dict = torch.load(best_model_path, map_location=trainer.device, weights_only=True)
                model.load_state_dict(state_dict['model'] if 'model' in state_dict else state_dict)
                model.eval()

                # avoid logging echoes
                logging.getLogger("pyhealth").handlers.clear()
                logging.getLogger().handlers.clear()

                eval_trainer = Trainer(model=model, output_path=base_out_dir, exp_name=f"fold_{fold}", metrics=metrics_list)
                for test_name, val_loader in val_loaders.items():
                    res = eval_trainer.evaluate(dataloader=val_loader)
                    for m in metrics_list: ood_results[test_name][m].append(res[m])
                    print(f"  -> {test_name.upper()} | ROC-AUC: {res['roc_auc']:.4f} | PR-AUC: {res['pr_auc']:.4f}")

    # ==========================================
    # FINAL TABLES 1 & 2 OUTPUT 
    # ==========================================
    if args.cv_folds > 1:
        print("\n" + "="*60)
        print(" FINAL SUMMARY: REPRODUCING PAPER TABLES 1 & 2 ")
        print("="*60)
        print(f"{'Dataset':<15} | {'Metric':<10} | {'Mean ± Std':<15}")
        print("-" * 60)

        def print_stats(name, stats_dict):
            for metric, values in stats_dict.items():
                if not values: continue
                mean = np.mean(values)
                std = np.std(values)
                print(f"{name:<15} | {metric:<10} | {mean:.4f} ± {std:.4f}")

        # 1. Print Source Dataset Results
        print_stats(f"SOURCE ({ds_str})", source_results)
        print("-" * 60)

        # 2. Print Out-of-Domain Results (Target Generalization)
        for target_name, stats in ood_results.items():
            print_stats(target_name.upper(), stats)
            print("-" * 60)