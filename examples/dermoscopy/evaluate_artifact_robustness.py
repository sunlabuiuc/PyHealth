# Contributor: [Your Name]
# NetID: [Your NetID]

"""
Artifact Robustness Evaluation Script

Evaluates trained models against synthetic out-of-distribution artifact datasets (Trap Sets).
Computes full clinical metrics (ROC-AUC, Accuracy, Precision, Recall). Supports 
evaluating single master models, averaging across CV folds, or creating an ensemble.
"""

import argparse
import os
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, average_precision_score, f1_score

from pyhealth.datasets import get_dataloader

from pyhealth.datasets import DermoscopyDataset
from pyhealth.tasks import DermoscopyMelanomaClassification
from pyhealth.processors import DermoscopyImageProcessor
from pyhealth.models import DINOv2
from train_dermoscopy import setup_dynamic_logging, MelanomaClassifier

def load_weights(model, weights_path, device):
    """Safely loads state dict weights into the architecture."""
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict['model'] if 'model' in state_dict else state_dict)
    model.to(device).eval()
    return model

def main():
    parser = argparse.ArgumentParser(description="Smart Artifact Evaluation")
    parser.add_argument('--model', type=str, choices=['resnet50', 'swin', 'dinov2'], default='resnet50')
    parser.add_argument('--mode', type=str, default='whole')
    parser.add_argument('--exp_dir', type=str, required=True, help="Path to the parent experiment directory")
    parser.add_argument('--strategy', type=str, choices=['master', 'ensemble', 'fold_average'], default='fold_average')
    parser.add_argument('--data_dir', type=str, required=True, help="Path to the dataset root folder")
    parser.add_argument('--eval_dataset', type=str, default='ph2', help="The base dataset")
    parser.add_argument('--artifact', type=str, required=True, help="The artifact to evaluate (e.g., ruler, ink)")
    args = parser.parse_args()

    # START DYNAMIC LOGGING
    setup_dynamic_logging("eval_artifacts", f"{args.eval_dataset}_{args.artifact}_{args.model}_{args.mode}")

    dataset_target = f"{args.eval_dataset}_with_{args.artifact}"
    print(f"[*] Loading Dataset from {args.data_dir}...")
    
    # 1. Load Dataset and apply Processor
    dataset = DermoscopyDataset(root=args.data_dir, dataset_name=[dataset_target], cache_dir=os.path.join(args.data_dir, ".cache"))

    processor = DermoscopyImageProcessor(mode=args.mode)
    
    # 2. Filter for the specific Trap Set
    task = DermoscopyMelanomaClassification(source_datasets=[dataset_target])
    task_dataset = dataset.set_task(task=task, input_processors={"image": processor})
    test_loader = get_dataloader(task_dataset, batch_size=32, shuffle=False)

    # 3. Architecture Initialization
    print(f"[*] Initializing Base Architecture: {args.model.upper()}...")
    if args.model == "dinov2":
        base_model = DINOv2(dataset=task_dataset, feature_keys=["image"], label_key="melanoma", mode="binary")
    else:
        base_model = MelanomaClassifier(dataset=task_dataset, feature_keys=["image"], label_key="melanoma", mode="binary", arch=args.model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 4. Resolve Target Weights based on Strategy
    weight_paths = []
    if args.strategy == "master":
        weight_paths.append(os.path.join(args.exp_dir, "master", "best.ckpt"))
    elif args.strategy in ["ensemble", "fold_average"]:
        for i in range(5): 
            weight_paths.append(os.path.join(args.exp_dir, f"fold_{i}", "best.ckpt"))

    print(f"[*] Evaluation Strategy: {args.strategy.upper()} on {dataset_target.upper()}")

    y_true_all = []
    y_prob_ensemble = None
    fold_aucs, fold_accs, fold_precs, fold_recs, fold_pr_aucs, fold_f1s = [], [], [], [], [], []

    # 5. Core Evaluation Loop
    for i, w_path in enumerate(weight_paths):
        if not os.path.exists(w_path):
            print(f"[!] Warning: Could not find weights at {w_path}. Skipping...")
            continue
            
        model = load_weights(base_model, w_path, device)
        y_prob_current_model = []
        current_y_true = []
        
        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                res = model(**batch)
                y_prob_current_model.extend(res['y_prob'].cpu().numpy())
                
                if i == 0: y_true_all.extend(res['y_true'].cpu().numpy())
                current_y_true.extend(res['y_true'].cpu().numpy())
        
        # Calculate full suite of metrics for the current fold
        y_pred_binary = (np.array(y_prob_current_model) >= 0.5).astype(int)
        
        if args.strategy == "fold_average":
            fold_auc = roc_auc_score(current_y_true, y_prob_current_model)
            fold_pr_auc = average_precision_score(current_y_true, y_prob_current_model)
            fold_acc = accuracy_score(current_y_true, y_pred_binary)
            fold_prec = precision_score(current_y_true, y_pred_binary, zero_division=0)
            fold_rec = recall_score(current_y_true, y_pred_binary)
            fold_f1 = f1_score(current_y_true, y_pred_binary)
            
            fold_aucs.append(fold_auc)
            fold_pr_aucs.append(fold_pr_auc)
            fold_accs.append(fold_acc)
            fold_precs.append(fold_prec)
            fold_recs.append(fold_rec)
            fold_f1s.append(fold_f1)
            
            print(f"  -> Fold {i} | AUC: {fold_auc:.4f} | PR-AUC: {fold_pr_auc:.4f} | F1: {fold_f1:.4f} | ACC: {fold_acc:.4f}")
            
        if y_prob_ensemble is None:
            y_prob_ensemble = np.array(y_prob_current_model)
        else:
            y_prob_ensemble += np.array(y_prob_current_model)

    if not weight_paths or (args.strategy == "fold_average" and not fold_aucs):
        print("[!] No successful evaluations completed. Check weight paths.")
        return

    # 6. Final Metric Aggregation
    if args.strategy == "fold_average":
        print(f"\n[!] FINAL PAPER METRIC ({args.artifact}):")
        print(f"    ROC-AUC:   {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
        print(f"    PR-AUC:    {np.mean(fold_pr_aucs):.4f} ± {np.std(fold_pr_aucs):.4f}")
        print(f"    F1-Score:  {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}")
        print(f"    Accuracy:  {np.mean(fold_accs):.4f} ± {np.std(fold_accs):.4f}")
        print(f"    Precision: {np.mean(fold_precs):.4f} ± {np.std(fold_precs):.4f}")
        print(f"    Recall:    {np.mean(fold_recs):.4f} ± {np.std(fold_recs):.4f}")
        
    elif args.strategy in ["ensemble", "master"]:
        num_models = len([w for w in weight_paths if os.path.exists(w)])
        if args.strategy == "ensemble":
            y_prob_final = y_prob_ensemble / num_models
        else:
            y_prob_final = y_prob_ensemble
            
        y_pred_final = (y_prob_final >= 0.5).astype(int)
        
        final_auc = roc_auc_score(y_true_all, y_prob_final)
        final_pr_auc = average_precision_score(y_true_all, y_prob_final)
        final_acc = accuracy_score(y_true_all, y_pred_final)
        final_prec = precision_score(y_true_all, y_pred_final, zero_division=0)
        final_rec = recall_score(y_true_all, y_pred_final)
        final_f1 = f1_score(y_true_all, y_pred_final)
        
        print(f"\n[!] FINAL {args.strategy.upper()} METRICS ({args.artifact}):")
        print(f"    ROC-AUC:   {final_auc:.4f}")
        print(f"    PR-AUC:    {final_pr_auc:.4f}")
        print(f"    F1-Score:  {final_f1:.4f}")
        print(f"    Accuracy:  {final_acc:.4f}")
        print(f"    Precision: {final_prec:.4f}")
        print(f"    Recall:    {final_rec:.4f}")

if __name__ == "__main__":
    main()